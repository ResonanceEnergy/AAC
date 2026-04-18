"""Generate current VCE signals (paper mode).

Usage:
    python modules/aac_puprime_vce/scripts/run_signals.py --all
    python modules/aac_puprime_vce/scripts/run_signals.py --symbols XAUUSD BTCUSD
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.aac_puprime_vce.cli import signals

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VCE: Generate signals")
    parser.add_argument("--all", action="store_true", help="All instruments")
    parser.add_argument("--symbols", nargs="*", default=None, help="Specific symbols")
    args = parser.parse_args()
    symbols = None if args.all else args.symbols
    signals(symbols)
