"""Generate campaign review report from journal data.

Usage:
    python modules/aac_puprime_vce/scripts/run_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.aac_puprime_vce.cli import report

if __name__ == "__main__":
    report()
