#!/usr/bin/env python3
"""
Quick launcher for PlanktonXD Browser Bot Web Dashboard
=====================================================
Convenient script to launch the web dashboard directly.

Usage: python scripts/launch_planktonxd_web_dashboard.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    """Launch the PlanktonXD Web Dashboard."""
    print("🦠 PlanktonXD Browser Bot Web Dashboard")
    print("======================================")
    print("🚀 Starting web interface...")
    print("🌐 Will open browser at http://localhost:8088")
    print("🎛️ Use clickable buttons to control the bot")
    print("")
    
    # Change to project root
    os.chdir(_ROOT)
    
    # Import and run the dashboard
    try:
        from monitoring.planktonxd_browser_dashboard import run_dashboard
        run_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())