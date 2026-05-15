"""Forbidden-pattern checker for pre-commit (gap audit 2026-05-15).

Usage: ``python tools/check_forbidden_patterns.py --check <name> [files...]``

Checks supported:
- ``silent-except`` — flags ``except Exception:\n<indent>    pass`` blocks.
- ``syspath`` — flags ``sys.path.insert(`` calls in production code.

Exits non-zero on first violation found across the supplied file list (which
pre-commit passes as positional arguments). When run with no files, scans the
whole tree minus the standard exclusions.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

EXCLUDE_PARTS = ("_archive", "_scratch", ".venv", ".git", "__pycache__", "tests", "archive")

SILENT_EXCEPT = re.compile(
    r"\n([ \t]*)except Exception:\s*\n\1[ \t]{4}pass\b",
    re.MULTILINE,
)
SYSPATH_INSERT = re.compile(r"\bsys\.path\.insert\s*\(")


def _iter_targets(files: list[str]) -> list[pathlib.Path]:
    if files:
        return [pathlib.Path(f) for f in files if f.endswith(".py")]
    root = pathlib.Path(".")
    return [
        p
        for p in root.rglob("*.py")
        if not any(part in EXCLUDE_PARTS for part in p.parts)
    ]


def _check_silent_except(paths: list[pathlib.Path]) -> int:
    violations: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in SILENT_EXCEPT.finditer(text):
            line_no = text[: m.start()].count("\n") + 2
            violations.append(f"{p}:{line_no}: 'except Exception: pass' is forbidden")
    if violations:
        print("\n".join(violations))
        print(f"\n{len(violations)} silent-except violation(s). Log the exception or narrow the type.")
        return 1
    return 0


def _check_syspath(paths: list[pathlib.Path]) -> int:
    violations: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if SYSPATH_INSERT.search(line):
                violations.append(f"{p}:{i}: sys.path.insert is forbidden — use proper package imports")
    if violations:
        print("\n".join(violations))
        print(f"\n{len(violations)} sys.path.insert violation(s). See AGENTS.md.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["silent-except", "syspath"])
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    targets = _iter_targets(args.files)
    if args.check == "silent-except":
        return _check_silent_except(targets)
    return _check_syspath(targets)


if __name__ == "__main__":
    sys.exit(main())
