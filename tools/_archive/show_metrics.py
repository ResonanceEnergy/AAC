#!/usr/bin/env python3
"""
Simple Metrics Display for ACC Arbitrage Strategies
Shows current system status and what's missing
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def show_metrics():
    """Show metrics."""
    logger.info("ACC - Accelerated Arbitrage Corp")
    logger.info("METRICS & DEEP DIVE ANALYSIS SYSTEM")
    logger.info("=" * 50)
    logger.info("")

    # Check file counts
    workspace = Path(".")
    py_files = list(workspace.rglob("*.py"))
    csv_files = list(workspace.rglob("*.csv"))
    md_files = list(workspace.rglob("*.md"))

    logger.info("SYSTEM STATUS:")
    logger.info(f"Python files: {len(py_files)}")
    logger.info(f"CSV files: {len(csv_files)}")
    logger.info(f"Documentation files: {len(md_files)}")
    logger.info("")

    # Check strategy files
    strategy_dir = Path("strategies")
    if strategy_dir.exists():
        strategy_files = list(strategy_dir.glob("*.py"))
        logger.info(f"Strategy implementations: {len(strategy_files)}")
    else:
        logger.info("Strategy implementations: 0 (directory missing)")
    logger.info("")

    # Check CSV for strategies
    csv_path = Path("50_arbitrage_strategies.csv")
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            strategy_count = len(lines) - 1  # minus header
        logger.info(f"Strategies defined in CSV: {strategy_count}")
    else:
        logger.info("Strategies CSV: Missing")
    logger.info("")

    # Check key components
    components = [
        "strategy_testing_lab.py",
        "strategy_analysis_engine.py",
        "strategy_metrics_dashboard.py",
        "deep_dive_file_analyzer.py",
        "requirements.txt"
    ]

    logger.info("KEY COMPONENTS:")
    for comp in components:
        if Path(comp).exists():
            logger.info(f"✓ {comp}")
        else:
            logger.info(f"✗ {comp} - MISSING")
    logger.info("")

    # Check dependencies
    try:
        import dash
        import numpy
        import pandas
        import plotly
        logger.info("DEPENDENCIES:")
        logger.info("✓ Dash installed")
        logger.info("✓ Plotly installed")
        logger.info("✓ Pandas installed")
        logger.info("✓ NumPy installed")
    except ImportError as e:
        logger.info(f"DEPENDENCIES: Missing - {e}")
    logger.info("")

    # What's missing
    logger.info("WHAT'S MISSING:")
    missing = []

    if not strategy_dir.exists():
        missing.append("strategies/ directory")

    if not csv_path.exists():
        missing.append("50_arbitrage_strategies.csv")

    for comp in components:
        if not Path(comp).exists():
            missing.append(comp)

    try:
        import dash
    except ImportError:
        missing.append("Dash framework")

    try:
        import plotly
    except ImportError:
        missing.append("Plotly library")

    if len(strategy_files) < 50:
        missing.append(f"{50 - len(strategy_files)} strategy implementations")

    if missing:
        for item in missing:
            logger.info(f"- {item}")
    else:
        logger.info("✓ All core components present")

    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Fix file encoding issues (remove emojis)")
    logger.info("2. Implement remaining strategies")
    logger.info("3. Launch metrics dashboard")
    logger.info("4. Run comprehensive analysis")

if __name__ == "__main__":
    show_metrics()
