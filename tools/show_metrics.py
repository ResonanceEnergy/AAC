#!/usr/bin/env python3
"""
Simple Metrics Display for ACC Arbitrage Strategies
Shows current system status and what's missing
"""

import os
import sys
from pathlib import Path

def show_metrics():
    print("ACC - Accelerated Arbitrage Corp")
    print("METRICS & DEEP DIVE ANALYSIS SYSTEM")
    print("=" * 50)
    print()

    # Check file counts
    workspace = Path(".")
    py_files = list(workspace.rglob("*.py"))
    csv_files = list(workspace.rglob("*.csv"))
    md_files = list(workspace.rglob("*.md"))

    print("SYSTEM STATUS:")
    print(f"Python files: {len(py_files)}")
    print(f"CSV files: {len(csv_files)}")
    print(f"Documentation files: {len(md_files)}")
    print()

    # Check strategy files
    strategy_dir = Path("strategies")
    if strategy_dir.exists():
        strategy_files = list(strategy_dir.glob("*.py"))
        print(f"Strategy implementations: {len(strategy_files)}")
    else:
        print("Strategy implementations: 0 (directory missing)")
    print()

    # Check CSV for strategies
    csv_path = Path("50_arbitrage_strategies.csv")
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            strategy_count = len(lines) - 1  # minus header
        print(f"Strategies defined in CSV: {strategy_count}")
    else:
        print("Strategies CSV: Missing")
    print()

    # Check key components
    components = [
        "strategy_testing_lab.py",
        "strategy_analysis_engine.py",
        "strategy_metrics_dashboard.py",
        "deep_dive_file_analyzer.py",
        "requirements.txt"
    ]

    print("KEY COMPONENTS:")
    for comp in components:
        if Path(comp).exists():
            print(f"✓ {comp}")
        else:
            print(f"✗ {comp} - MISSING")
    print()

    # Check dependencies
    try:
        import dash
        import plotly
        import pandas
        import numpy
        print("DEPENDENCIES:")
        print("✓ Dash installed")
        print("✓ Plotly installed")
        print("✓ Pandas installed")
        print("✓ NumPy installed")
    except ImportError as e:
        print(f"DEPENDENCIES: Missing - {e}")
    print()

    # What's missing
    print("WHAT'S MISSING:")
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
            print(f"- {item}")
    else:
        print("✓ All core components present")

    print()
    print("NEXT STEPS:")
    print("1. Fix file encoding issues (remove emojis)")
    print("2. Implement remaining strategies")
    print("3. Launch metrics dashboard")
    print("4. Run comprehensive analysis")

if __name__ == "__main__":
    show_metrics()