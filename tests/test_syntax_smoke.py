#!/usr/bin/env python3
"""
Smoke tests — verify every tracked .py file compiles without syntax errors.

This catches broken string literals, invalid identifiers, Unicode corruption,
and other issues that slip past linters but break ``import`` / ``exec``.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to skip (not production code or vendored)
SKIP_DIRS = {".venv", "archive", "node_modules", "build", "__pycache__", ".git"}


def _collect_py_files():
    """Yield every .py file under PROJECT_ROOT that isn't excluded."""
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        # Prune excluded dirs in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)


ALL_PY = list(_collect_py_files())


@pytest.mark.parametrize(
    "filepath",
    ALL_PY,
    ids=[os.path.relpath(f, PROJECT_ROOT) for f in ALL_PY],
)
def test_file_compiles(filepath: str) -> None:
    """Each .py file must parse without a SyntaxError."""
    source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError in {filepath}: {exc}")
