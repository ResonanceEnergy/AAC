from __future__ import annotations

"""
tools.autonomous_coder — deterministic repo scanner + safe-fix applier.

Scans the AAC codebase for the recurring drift patterns enumerated in
.github/copilot-instructions.md (forbidden patterns section) and AGENTS.md,
emits a prioritized backlog, and (with --apply) executes the *known-safe*
auto-fixes.

It is NOT an LLM agent. It does not generate novel code. It reports gaps so a
human or coding-agent (Copilot/Claude) can drive them, and applies the
mechanical fixes that are too obvious to need review.

Detectors:
    F01  except Exception: pass / bare-except: pass        (silent swallowing)
    F02  sys.path.insert hack                              (forbidden)
    F03  missing `from __future__ import annotations`      (style rule #1)
    F04  TODO / FIXME / XXX / HACK marker                  (drift indicator)
    F05  NotImplementedError stub                          (unfinished work)
    F06  temp script at repo root (check_/trace_/debug_)   (placement rule)
    F07  hardcoded api_key string                          (security)
    F08  print() in non-script module                      (use _log)
    F09  empty __init__.py with no __all__                 (cosmetic, low)
    F10  test file with zero `def test_` functions         (dead test)

Auto-fixes (only with --apply):
    A01  prepend `from __future__ import annotations` to .py files missing it
         (skips files where the first non-comment line is a docstring; in that
         case inserts after the docstring)
    A02  rewrite `except Exception: pass` to log via structlog (per file)

Output:
    data/autonomous_coder_backlog.json   — machine-readable queue
    stdout                               — colored summary table

Usage:
    python -m tools.autonomous_coder                       # report only
    python -m tools.autonomous_coder --apply               # run safe fixers
    python -m tools.autonomous_coder --paths aac core      # scope to subtrees
    python -m tools.autonomous_coder --severity high       # filter
    python launch.py coder                                 # via launcher
    python launch.py coder --apply
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to skip outright
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "_archive",
    "archive",
    "aac.egg-info",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "logs",
    "reports",
    "data",
    "secrets",
}

# Files at root that are legitimate (not "drift" temp scripts)
ROOT_ALLOWLIST = {
    "launch.py",
    "conftest.py",
    "pipeline_runner.py",
    "health_server.py",
    "setup.py",
}

ROOT_TEMP_PREFIXES = ("check_", "trace_", "debug_", "test_", "_", "scratch_", "tmp_")

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


# ── Models ────────────────────────────────────────────────────────────────


@dataclass
class Finding:
    rule: str
    severity: str
    path: str
    line: int
    snippet: str
    message: str
    auto_fixable: bool = False

    def key(self) -> tuple:
        return (-SEVERITY_RANK.get(self.severity, 0), self.rule, self.path, self.line)


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    fixes_applied: int = 0

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    def by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.rule] = out.get(f.rule, 0) + 1
        return out

    def by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out


# ── File walking ──────────────────────────────────────────────────────────


def _iter_py_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


# ── Detectors ─────────────────────────────────────────────────────────────

_RE_EXCEPT_PASS = re.compile(
    r"^\s*except\s+(?:Exception|BaseException|)\s*(?:as\s+\w+\s*)?:\s*\n\s*pass\s*$",
    re.MULTILINE,
)
_RE_SYS_PATH = re.compile(r"^\s*sys\.path\.insert\s*\(", re.MULTILINE)
_RE_TODO = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b[:\s]*(.{0,80})", re.IGNORECASE)
_RE_NOTIMPL = re.compile(r"\braise\s+NotImplementedError\b")
_RE_HARDCODED_KEY = re.compile(
    r"""(?xi)
    (api[_-]?key|secret|token|password)\s*=\s*
    ["']([A-Za-z0-9_\-]{20,})["']
    """
)
_RE_PRINT = re.compile(r"^\s*print\s*\(", re.MULTILINE)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _has_future_annotations(text: str) -> bool:
    # Look in first ~30 non-blank lines (skip shebang/encoding/docstring)
    for line in text.splitlines()[:30]:
        if "from __future__ import annotations" in line:
            return True
    return False


def _is_script(path: Path) -> bool:
    """True for entry-point/launcher style files where print() is acceptable."""
    name = path.name
    if name in ("launch.py", "pipeline_runner.py", "conftest.py", "health_server.py"):
        return True
    if path.parent.name in ("scripts", "_scratch", "tools"):
        return True
    if path.parts and path.parts[-2:] and path.parts[-2] == "tests":
        return True
    return False


def scan_file(path: Path, result: ScanResult) -> None:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    text = _read(path)
    result.files_scanned += 1

    # F01 — except Exception: pass
    for m in _RE_EXCEPT_PASS.finditer(text):
        result.add(
            Finding(
                rule="F01",
                severity="high",
                path=rel,
                line=_line_of(text, m.start()),
                snippet=m.group(0).strip()[:80],
                message="Silent exception swallowing — narrow the except and log",
                auto_fixable=True,
            )
        )

    # F02 — sys.path.insert
    for m in _RE_SYS_PATH.finditer(text):
        result.add(
            Finding(
                rule="F02",
                severity="medium",
                path=rel,
                line=_line_of(text, m.start()),
                snippet=text.splitlines()[_line_of(text, m.start()) - 1].strip()[:80],
                message="sys.path.insert hack — use proper package imports",
            )
        )

    # F03 — missing future annotations (only for project files with code)
    if text.strip() and not _has_future_annotations(text):
        # Skip __init__.py with no real content
        if not (path.name == "__init__.py" and len(text.strip()) < 50):
            result.add(
                Finding(
                    rule="F03",
                    severity="low",
                    path=rel,
                    line=1,
                    snippet="(missing)",
                    message="Missing `from __future__ import annotations`",
                    auto_fixable=True,
                )
            )

    # F04 — TODO/FIXME markers
    for m in _RE_TODO.finditer(text):
        result.add(
            Finding(
                rule="F04",
                severity="low",
                path=rel,
                line=_line_of(text, m.start()),
                snippet=f"{m.group(1).upper()}: {m.group(2).strip()}"[:80],
                message=f"{m.group(1).upper()} marker — drift indicator",
            )
        )

    # F05 — NotImplementedError
    for m in _RE_NOTIMPL.finditer(text):
        result.add(
            Finding(
                rule="F05",
                severity="medium",
                path=rel,
                line=_line_of(text, m.start()),
                snippet="raise NotImplementedError",
                message="Unfinished stub — implement or delete",
            )
        )

    # F07 — hardcoded credentials
    for m in _RE_HARDCODED_KEY.finditer(text):
        result.add(
            Finding(
                rule="F07",
                severity="critical",
                path=rel,
                line=_line_of(text, m.start()),
                snippet=f"{m.group(1)}=***redacted***",
                message="Possible hardcoded credential — move to .env",
            )
        )

    # F08 — print() in non-script
    if not _is_script(path):
        prints = list(_RE_PRINT.finditer(text))
        if prints:
            result.add(
                Finding(
                    rule="F08",
                    severity="low",
                    path=rel,
                    line=_line_of(text, prints[0].start()),
                    snippet=f"{len(prints)} print() call(s)",
                    message="Use structlog `_log` instead of print() in library modules",
                )
            )

    # F09 — empty __init__.py
    if path.name == "__init__.py" and len(text.strip()) == 0:
        result.add(
            Finding(
                rule="F09",
                severity="low",
                path=rel,
                line=1,
                snippet="(empty)",
                message="Empty __init__.py — consider adding __all__ or package docstring",
            )
        )

    # F10 — test file with no tests
    if path.name.startswith("test_") and path.parent.name == "tests":
        try:
            tree = ast.parse(text)
            has_tests = any(
                isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                for n in ast.walk(tree)
            )
            has_class_tests = any(
                isinstance(n, ast.ClassDef) and n.name.startswith("Test")
                for n in ast.walk(tree)
            )
            if not (has_tests or has_class_tests):
                result.add(
                    Finding(
                        rule="F10",
                        severity="medium",
                        path=rel,
                        line=1,
                        snippet="(no test_* functions)",
                        message="Test file contains no test functions — dead file",
                    )
                )
        except SyntaxError:
            pass


def scan_root_placement(result: ScanResult) -> None:
    """F06 — temp scripts that escaped to project root."""
    for entry in PROJECT_ROOT.iterdir():
        if not entry.is_file() or entry.suffix != ".py":
            continue
        if entry.name in ROOT_ALLOWLIST:
            continue
        if entry.name.startswith(ROOT_TEMP_PREFIXES):
            result.add(
                Finding(
                    rule="F06",
                    severity="medium",
                    path=entry.name,
                    line=0,
                    snippet=entry.name,
                    message="Temp script at repo root — move to _scratch/ or scripts/",
                )
            )


# ── Auto-fixers ───────────────────────────────────────────────────────────


def fix_future_annotations(path: Path) -> bool:
    """A01 — prepend `from __future__ import annotations` if missing."""
    text = _read(path)
    if _has_future_annotations(text):
        return False
    if not text.strip():
        return False

    lines = text.splitlines(keepends=True)
    insert_idx = 0

    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    # Skip encoding declarations
    if insert_idx < len(lines) and re.match(r"^#.*coding[:=]", lines[insert_idx]):
        insert_idx += 1

    # If a module docstring follows, insert AFTER it
    rest = "".join(lines[insert_idx:])
    try:
        tree = ast.parse(rest)
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            doc_end_line = tree.body[0].end_lineno  # 1-based within rest
            if doc_end_line is not None:
                insert_idx += doc_end_line
    except SyntaxError:
        return False

    new_line = "from __future__ import annotations\n"
    # Add a blank line for breathing room if the next line is non-blank
    sep = "" if insert_idx < len(lines) and lines[insert_idx].strip() == "" else "\n"
    lines.insert(insert_idx, new_line + sep)
    path.write_text("".join(lines), encoding="utf-8")
    return True


_RE_EXCEPT_PASS_FIX = re.compile(
    r"^([ \t]*)except\s+(Exception|BaseException)(\s+as\s+\w+)?\s*:\s*\n([ \t]+)pass\s*$",
    re.MULTILINE,
)


def fix_except_pass(path: Path) -> int:
    """A02 — replace `except Exception: pass` with logged narrow form."""
    text = _read(path)
    if not _RE_EXCEPT_PASS_FIX.search(text):
        return 0

    def _sub(m: re.Match) -> str:
        indent = m.group(1)
        body_indent = m.group(4)
        # Preserve `as` clause if present, else add one
        as_clause = m.group(3) or " as exc"
        return (
            f"{indent}except {m.group(2)}{as_clause}:  # noqa: BLE001\n"
            f"{body_indent}_log = __import__('structlog').get_logger() "
            f"if '_log' not in dir() else _log\n"
            f"{body_indent}_log.warning('suppressed_exception', error=str(exc))"
        )

    # Only fix it when the file already imports structlog or logging — otherwise
    # leave a finding so a human handles the import properly.
    if "structlog" not in text:
        return 0

    new_text, n = _RE_EXCEPT_PASS_FIX.subn(_sub, text)
    if n:
        path.write_text(new_text, encoding="utf-8")
    return n


# ── Reporting ─────────────────────────────────────────────────────────────


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def _sev_color(sev: str) -> str:
    return {"critical": "1;31", "high": "31", "medium": "33", "low": "90"}.get(sev, "0")


def render_report(result: ScanResult, limit_per_rule: int = 10) -> str:
    lines: list[str] = []
    bar = "=" * 96
    lines.append(_color(bar, "36"))
    lines.append(_color("  AAC AUTONOMOUS CODER — repo scan", "1;36"))
    lines.append(_color(bar, "36"))
    lines.append(f"  files scanned:  {result.files_scanned}")
    lines.append(f"  findings:       {len(result.findings)}")
    if result.fixes_applied:
        lines.append(_color(f"  fixes applied:  {result.fixes_applied}", "32"))

    lines.append("")
    lines.append(_color("  By severity:", "1"))
    for sev in ("critical", "high", "medium", "low"):
        n = result.by_severity().get(sev, 0)
        if n:
            lines.append(f"    {_color(sev.upper().ljust(8), _sev_color(sev))} {n}")

    lines.append("")
    lines.append(_color("  By rule:", "1"))
    for rule, n in sorted(result.by_rule().items()):
        lines.append(f"    {rule}  {n:>5}")

    # Per-rule sample listing
    grouped: dict[str, list[Finding]] = {}
    for f in sorted(result.findings, key=Finding.key):
        grouped.setdefault(f.rule, []).append(f)

    for rule, items in grouped.items():
        lines.append("")
        sev = items[0].severity
        header = f"  [{rule}] {items[0].message}  ({len(items)} total, severity={sev})"
        lines.append(_color(header, _sev_color(sev)))
        for f in items[:limit_per_rule]:
            tag = _color(" auto", "32") if f.auto_fixable else "     "
            lines.append(f"   {tag}  {f.path}:{f.line}  {f.snippet}")
        if len(items) > limit_per_rule:
            lines.append(_color(f"        ... +{len(items) - limit_per_rule} more", "90"))

    lines.append("")
    lines.append(_color(bar, "36"))
    auto_n = sum(1 for f in result.findings if f.auto_fixable)
    lines.append(
        f"  {auto_n} finding(s) are auto-fixable. Re-run with --apply to fix them."
    )
    lines.append(_color(bar, "36"))
    return "\n".join(lines)


# ── Orchestration ─────────────────────────────────────────────────────────


def run_scan(paths: list[Path]) -> ScanResult:
    result = ScanResult()
    for path in _iter_py_files(paths):
        scan_file(path, result)
    if PROJECT_ROOT in paths or any(p == PROJECT_ROOT for p in paths):
        scan_root_placement(result)
    return result


def apply_fixes(result: ScanResult, *, only_rules: set[str] | None = None) -> int:
    """Run safe auto-fixers against the findings list. Returns count applied."""
    targets_a01 = {f.path for f in result.findings if f.rule == "F03" and f.auto_fixable}
    targets_a02 = {f.path for f in result.findings if f.rule == "F01" and f.auto_fixable}

    applied = 0
    if not only_rules or "A01" in only_rules:
        for rel in sorted(targets_a01):
            p = PROJECT_ROOT / rel
            if p.exists() and fix_future_annotations(p):
                applied += 1
    if not only_rules or "A02" in only_rules:
        for rel in sorted(targets_a02):
            p = PROJECT_ROOT / rel
            n = fix_except_pass(p)
            if n:
                applied += n
    result.fixes_applied = applied
    return applied


def write_backlog(result: ScanResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files_scanned": result.files_scanned,
        "total_findings": len(result.findings),
        "fixes_applied": result.fixes_applied,
        "by_severity": result.by_severity(),
        "by_rule": result.by_rule(),
        "findings": [asdict(f) for f in sorted(result.findings, key=Finding.key)],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="autonomous_coder",
        description="Deterministic AAC repo scanner + safe-fix applier.",
    )
    p.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Subdirs/files to scan (default: whole repo)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Run safe auto-fixers (A01 future-annotations, A02 except-pass)",
    )
    p.add_argument(
        "--severity",
        choices=("critical", "high", "medium", "low"),
        default=None,
        help="Filter report to >= this severity",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "autonomous_coder_backlog.json",
        help="Backlog JSON output path",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout report (still writes JSON)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.paths:
        roots = [PROJECT_ROOT / p for p in args.paths]
        for r in roots:
            if not r.exists():
                print(f"path not found: {r}", file=sys.stderr)
                return 2
    else:
        roots = [PROJECT_ROOT]

    result = run_scan(roots)

    if args.apply:
        apply_fixes(result)
        applied_count = result.fixes_applied
        # Re-scan so the backlog reflects post-fix state
        result = run_scan(roots)
        # Re-scan resets fixes_applied; restore it from the apply pass
        result.fixes_applied = applied_count

    if args.severity:
        threshold = SEVERITY_RANK[args.severity]
        result.findings = [
            f for f in result.findings if SEVERITY_RANK.get(f.severity, 0) >= threshold
        ]

    write_backlog(result, args.out)

    if not args.quiet:
        print(render_report(result))
        print(f"  backlog written: {args.out.relative_to(PROJECT_ROOT)}")

    # Exit code: 0 if clean or report-only; 1 if critical findings remain
    has_critical = any(f.severity == "critical" for f in result.findings)
    return 1 if has_critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
