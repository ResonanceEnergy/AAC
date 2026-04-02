"""Comprehensive gap scanner for AAC codebase."""
import ast
import logging
import os
import re
import sys
from collections import Counter

logger = logging.getLogger(__name__)

ROOT = r"c:\dev\AAC_fresh"
gaps = []
skip_dirs = {
    ".venv", "__pycache__", ".git", "node_modules", ".egg-info",
    "aac.egg-info", "build", "archive", "data", "logs", "reports",
    "version_control",
}
skip_files = {"conftest.py", "_gap_scanner.py", "_full_import_test.py", "_get_pip_bootstrap.py", "_read_helix.py"}


def get_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.endswith(".egg-info")]
        for f in filenames:
            if f.endswith(".py") and f not in skip_files:
                yield os.path.join(dirpath, f)


gap_id = 0

for filepath in get_py_files(ROOT):
    rel = os.path.relpath(filepath, ROOT)
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception:
        continue

    for i, line in enumerate(lines, 1):
        s = line.strip()
        low = s.lower()

        # 1. TODO/FIXME/HACK/XXX/PLACEHOLDER/STUB comments
        if "#" in s:
            idx = s.index("#")
            comment_part = s[idx:]
            cup = comment_part.upper()
            if any(t in cup for t in ["TODO", "FIXME", "HACK ", "XXX:", "PLACEHOLDER", "# STUB", "TEMPORARY", "TEMP FIX"]):
                gap_id += 1
                gaps.append((gap_id, "TODO", rel, i, comment_part[:100]))

        # 2. Swallowed exceptions (except ...: pass)
        if s in ("except:", "except Exception:", "except BaseException:") or s.startswith("except Exception as"):
            if i < len(lines):
                next_s = lines[i].strip() if i < len(lines) else ""
                if next_s in ("pass", "..."):
                    gap_id += 1
                    gaps.append((gap_id, "SWALLOWED_EXCEPTION", rel, i, f"{s} -> {next_s}"))

        # 3. Hardcoded placeholder URLs
        if "http://" in s or "https://" in s:
            if not s.startswith("#") and not s.startswith('"""'):
                if any(x in low for x in ["localhost", "127.0.0.1", "example.com", "placeholder", "your-", "your_"]):
                    gap_id += 1
                    gaps.append((gap_id, "HARDCODED_URL", rel, i, s[:100]))

        # 4. Placeholder credentials
        if any(x in low for x in ["password123", "changeme", "your_api_key", "your-api-key", "replace_with_", "insert_your", "put_your_", "enter_your", "xxx-api-key", "sk-xxx", "api_key_here"]):
            if not s.startswith("#"):
                gap_id += 1
                gaps.append((gap_id, "PLACEHOLDER_CRED", rel, i, s[:100]))

        # 5. print() debug (non-test files)
        if s.startswith("print(") and "test" not in rel.lower():
            gap_id += 1
            gaps.append((gap_id, "PRINT_DEBUG", rel, i, s[:100]))

        # 6. Commented-out code
        if s.startswith("#") and len(s) > 5:
            rest = s[1:].strip()
            if rest.startswith("def ") or rest.startswith("class ") or rest.startswith("import ") or rest.startswith("from ") or rest.startswith("return "):
                gap_id += 1
                gaps.append((gap_id, "COMMENTED_CODE", rel, i, s[:100]))

# AST-based analysis
for filepath in get_py_files(ROOT):
    rel = os.path.relpath(filepath, ROOT)
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("test_") or name.startswith("_test"):
                continue
            body = node.body
            stmts = list(body)
            has_docstring = False
            if stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0].value, "value", None), str):
                has_docstring = True
                stmts = stmts[1:]

            # Public functions missing docstrings
            if not name.startswith("_") and not has_docstring and len(stmts) > 0:
                gap_id += 1
                gaps.append((gap_id, "NO_DOCSTRING", rel, node.lineno, f"def {name}()"))

            # Stub pass
            if len(stmts) == 1 and isinstance(stmts[0], ast.Pass):
                gap_id += 1
                gaps.append((gap_id, "STUB_PASS", rel, node.lineno, f"def {name}()"))

            # Empty body (docstring only, no code)
            if not stmts:
                gap_id += 1
                gaps.append((gap_id, "EMPTY_BODY", rel, node.lineno, f"def {name}()"))

            # NotImplementedError
            if len(stmts) == 1 and isinstance(stmts[0], ast.Raise):
                gap_id += 1
                gaps.append((gap_id, "NOT_IMPL", rel, node.lineno, f"def {name}()"))

            # Return placeholder (return None / return {} / return [])
            if len(stmts) == 1 and isinstance(stmts[0], ast.Return):
                rv = stmts[0].value
                if rv is None:
                    gap_id += 1
                    gaps.append((gap_id, "RETURN_NONE", rel, node.lineno, f"def {name}()"))
                elif isinstance(rv, ast.Dict) and not rv.keys:
                    gap_id += 1
                    gaps.append((gap_id, "RETURN_EMPTY", rel, node.lineno, f"def {name}() -> dict"))
                elif isinstance(rv, ast.List) and not rv.elts:
                    gap_id += 1
                    gaps.append((gap_id, "RETURN_EMPTY", rel, node.lineno, f"def {name}() -> list"))

        # Classes missing docstrings
        elif isinstance(node, ast.ClassDef):
            body = node.body
            has_docstring = False
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str):
                has_docstring = True
            if not has_docstring and not node.name.startswith("_"):
                gap_id += 1
                gaps.append((gap_id, "CLASS_NO_DOC", rel, node.lineno, f"class {node.name}"))

# Write results
with open(os.path.join(ROOT, "gap_scan_full.txt"), "w", encoding="utf-8") as out:
    for g in gaps:
        out.write("|".join(str(x) for x in g) + "\n")

cats = Counter(g[1] for g in gaps)
logger.info(f"Total gaps: {len(gaps)}")
for cat, count in cats.most_common():
    logger.info(f"  {cat}: {count}")

# Show per-file breakdown for top files
file_counts = Counter(g[2] for g in gaps)
logger.info("\nTop 30 files by gap count:")
for f, c in file_counts.most_common(30):
    logger.info(f"  {c:4d}  {f}")
