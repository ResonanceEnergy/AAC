#!/usr/bin/env bash
# launch.sh — AAC master launcher (macOS / Linux)
# Usage: ./launch.sh [dashboard|monitor|paper|full]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Load env
if [ -f ".env" ]; then
    export $(grep -v '^#\|^$' .env | xargs)
else
    echo "WARNING: .env not found — copying template. Edit .env with your API keys."
    cp .env.template .env
fi

MODE=${1:-"dashboard"}

case "$MODE" in
    dashboard)
        echo "Starting AAC Master Monitoring Dashboard on http://localhost:8050 ..."
        python monitoring/aac_master_monitoring_dashboard.py
        ;;
    monitor)
        echo "Starting Matrix Monitor ..."
        python shared/monitoring.py
        ;;
    paper)
        echo "Starting Paper Trading Engine ..."
        python shared/paper_trading.py
        ;;
    core)
        echo "Starting AAC Core Launcher ..."
        python core/aac_master_launcher.py
        ;;
    full)
        echo "Starting full AAC system ..."
        python core/orchestrator.py &
        sleep 2
        python monitoring/aac_master_monitoring_dashboard.py
        ;;
    test)
        echo "Running test suite ..."
        python -m pytest tests/ -q --tb=short -m "not live and not exchange"
        ;;
    *)
        echo "Usage: ./launch.sh [dashboard|monitor|paper|core|full|test]"
        echo ""
        echo "  dashboard  — Dash monitoring dashboard (default)"
        echo "  monitor    — Matrix console monitor"
        echo "  paper      — Paper trading engine"
        echo "  core       — AAC master launcher"
        echo "  full       — Orchestrator + dashboard"
        echo "  test       — Run pytest suite"
        exit 1
        ;;
esac
