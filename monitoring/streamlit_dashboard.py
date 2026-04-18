from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger()
_THIS_FILE = Path(__file__).resolve()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return str(obj)


def _collect_payload() -> dict[str, Any]:
    """Collect dashboard data from mission_control collectors when available."""
    payload: dict[str, Any] = {"ts": dt.datetime.now().isoformat()}

    try:
        from monitoring import mission_control as mc

        collector_map: dict[str, str] = {
            "portfolio": "collect_portfolio",
            "war_room": "collect_war_room",
            "live_feeds": "collect_live_feeds",
            "regime": "collect_regime",
            "doctrine": "collect_doctrine",
            "moon": "collect_moon",
            "health": "collect_health",
            "tasks": "collect_tasks",
            "daily_tasks": "collect_daily_tasks",
            "unusual_whales": "collect_unusual_whales",
            "divisions": "collect_divisions",
            "api_feeds": "collect_api_feeds",
            "polymarket": "collect_polymarket",
            "scenarios": "collect_scenarios",
            "backbone": "collect_backbone",
            "pnl": "collect_pnl",
            "trade_log": "collect_trade_log",
        }

        safe_fn = getattr(mc, "_safe", None)
        for key, fn_name in collector_map.items():
            fn = getattr(mc, fn_name, None)
            if fn is None:
                payload[key] = {"error": f"collector_missing:{fn_name}"}
                continue

            if callable(safe_fn):
                payload[key] = safe_fn(fn, key)
            else:
                try:
                    payload[key] = fn()
                except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
                    payload[key] = {"error": str(exc)}

    except (ImportError, RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
        payload["error"] = f"collector_bootstrap_failed: {exc}"

    return payload


def _render_streamlit(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.set_page_config(page_title="AAC Streamlit Dashboard", layout="wide")
    st.title("AAC Monitoring Dashboard")
    st.caption("Rebuilt streamlit_dashboard module (web mode)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Timestamp", payload.get("ts", "-"))

    portfolio = payload.get("portfolio", {}) if isinstance(payload, dict) else {}
    total_usd = portfolio.get("total_usd") if isinstance(portfolio, dict) else None
    c2.metric("Portfolio USD", f"{total_usd:,.2f}" if isinstance(total_usd, (int, float)) else "-")

    health = payload.get("health", {}) if isinstance(payload, dict) else {}
    health_status = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
    c3.metric("Health", str(health_status))

    st.subheader("War Room")
    st.json(payload.get("war_room", {}), expanded=False)

    st.subheader("Regime")
    st.json(payload.get("regime", {}), expanded=False)

    st.subheader("System Health")
    st.json(health, expanded=False)

    st.subheader("Daily Tasks")
    st.json(payload.get("daily_tasks", {}), expanded=False)

    with st.expander("Raw payload", expanded=False):
        st.code(json.dumps(payload, default=_json_default, indent=2), language="json")


def generate_copilot_response(prompt: str) -> str:
    """Backward-compatible helper referenced by legacy imports."""
    return f"Copilot note: {prompt[:200]}"


def play_audio_response(_text: str) -> None:
    """Backward-compatible no-op placeholder for legacy audio hooks."""
    _log.info("audio_response_skipped")


def run_streamlit_dashboard(port: int = 8501) -> int:
    """Run this module via streamlit CLI in a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_THIS_FILE),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    _log.info("starting_streamlit_dashboard", port=port)
    return subprocess.call(cmd, cwd=str(_THIS_FILE.parent.parent))


class AACStreamlitDashboard:
    """Adapter used by aac_master_monitoring_dashboard web mode."""

    async def run_dashboard(self, port: int = 8501) -> int:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(_THIS_FILE),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
        proc = subprocess.Popen(cmd, cwd=str(_THIS_FILE.parent.parent))
        _log.info("streamlit_dashboard_running", pid=proc.pid, port=port)

        try:
            while proc.poll() is None:
                await asyncio.sleep(1.0)
            return proc.returncode or 0
        except asyncio.CancelledError:
            proc.terminate()
            raise
        except KeyboardInterrupt:
            proc.terminate()
            return 0
        finally:
            if proc.poll() is None:
                proc.terminate()


def main() -> None:
    payload = _collect_payload()
    _render_streamlit(payload)


if __name__ == "__main__":
    main()
