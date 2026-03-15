"""Run VCE backtest on all (or selected) instruments.

Usage:
    python modules/aac_puprime_vce/scripts/run_backtest.py --all
    python modules/aac_puprime_vce/scripts/run_backtest.py --symbols XAUUSD
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.aac_puprime_vce.cli import backtest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VCE: Run backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all instruments")
    parser.add_argument("--symbols", nargs="*", default=None, help="Specific symbols")
    args = parser.parse_args()
    symbols = None if args.all else args.symbols
    backtest(symbols)
