# monitoring/

AAC Matrix Monitor — Central Command & Control Hub.

The **AAC Matrix Monitor** is a hybrid process / user-interface dashboard,
console display, and command-control center that monitors AND commands
every component of AAC. It is the central go-to hub to access and peek
inside the inner workings of the organisation and make changes as necessary.

## Architecture

- **4 Display Modes**: Terminal (curses/ANSI), Web (Streamlit), Dash (Plotly), API (REST/JSON)
- **20+ Dashboard Panels**: System health, P&L, risk, trading, doctrine, market intel, crisis, strategy, pillar network, registry
- **5 Pillar Endpoints**: NCC_MASTER (8765), NCC (8765), AAC (8080), NCL (8787), BRS (8000)
- **9 Elite Trading Desk Components**: Jonny Bravo, WSB/Reddit, PlanktonXD, Grok AI, OpenClaw, Stock Ticker, NCL Link, Unusual Whales, Matrix Maximizer
- **Doctrine Pack 12**: Matrix Monitor Command & Control — self-monitoring via the Doctrine Engine

## Key Modules

| Module | Purpose |
|--------|---------|
| `aac_master_monitoring_dashboard.py` | **Central C2 Hub** — unified dashboard, all data collectors, 4 display modes, REST API |
| `aac_system_registry.py` | System registry — APIs, exchanges, infra, strategies, departments, orphans |
| `continuous_monitoring.py` | Background health checks (30s intervals), alerting, incident response |
| `security_dashboard.py` | Real-time security status reporting |
| `security_compliance_integration.py` | Security framework + compliance integration |
| `streamlit_dashboard.py` | Streamlit web UI wrapper |
| `dash_dashboard.py` | Plotly Dash analytics wrapper |

## Launch

```bash
# Terminal mode (default)
python launch.py matrix

# Web mode (Streamlit)
python launch.py matrix --display web

# Dash mode (Plotly)
python launch.py matrix --display dash

# CLI entry point
aac-dashboard
```
