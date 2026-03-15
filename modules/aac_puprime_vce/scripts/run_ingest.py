"""Ingest MT5 CSV exports and normalize to processed data.

Usage:
    python modules/aac_puprime_vce/scripts/run_ingest.py
    python modules/aac_puprime_vce/scripts/run_ingest.py --symbols XAUUSD EURUSD
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.aac_puprime_vce.cli import ingest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VCE: Ingest MT5 CSV data")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to ingest (default: all)")
    args = parser.parse_args()
    ingest(args.symbols)
