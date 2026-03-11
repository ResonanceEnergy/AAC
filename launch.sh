#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  BARREN WUFFET — Unified Launcher (Unix / macOS)
#  Works on QUSAR and QFORGE.
#  If .venv exists, uses it.  Otherwise uses system Python.
#
#  Usage:  ./launch.sh dashboard
#          ./launch.sh paper
#          ./launch.sh git-sync
#  Setup:  python3 setup_machine.py
# ═══════════════════════════════════════════════════════════════
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer .venv Python if it exists
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python launch.py "$@"
else
    echo "[!] No .venv found. Run: python3 setup_machine.py"
    exec python3 launch.py "$@"
fi
