from __future__ import annotations

"""
monitoring.command_dashboard — AAC Command Dashboard (web).

A single-page web dashboard aligned with .context/04_workstreams/GOAL_MANDATE_ROADMAP.md.
Surfaces the FIVE mission goals as live panels, pulling from the Sprint 1-25 subsystems
that actually run today:

    1. FIND TRADES   - SignalJournal, RegimeMonitor, hit rates
    2. EXECUTE       - last fills (PnLTracker.trade_log), ExecutionThrottle, OrderMonitor
    3. MANAGE RISK   - RollManager urgent, StopManager urgent, DrawdownCircuitBreaker,
                       DailyLossGuard, PositionReconciler
    4. TRACK P&L     - PnLTracker.today_report, AccountValueFeed, historical_summary
    5. HEALTH        - HealthMonitor.collect_snapshot (API + system + alerts)

Plus the EOD daily brief (reports/daily_brief.txt).

All collectors fail-open: if a subsystem is missing or errors, the panel shows
"unavailable" with the error rather than crashing the dashboard.

Usage:
    python -m monitoring.command_dashboard            # default port 8400
    python launch.py command-dashboard --port 8400
"""

import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

_log = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Safe collector wrapper ────────────────────────────────────────────────


def _safe(fn: Callable[[], Any], label: str) -> dict[str, Any]:
    """Execute a collector and return {ok, data, error}.  Never raises."""
    try:
        return {"ok": True, "data": fn(), "error": None}
    except Exception as exc:  # noqa: BLE001
        _log.warning("dashboard_collector_failed", label=label, error=str(exc))
        return {"ok": False, "data": None, "error": f"{type(exc).__name__}: {exc}"}


# ── Individual collectors (one per panel) ─────────────────────────────────


def _collect_health() -> dict[str, Any]:
    from monitoring.health_monitor import HealthMonitor  # noqa: PLC0415

    snap = HealthMonitor().collect_snapshot()
    return {
        "checked_at": snap.checked_at.isoformat(),
        "overall_status": snap.overall_status.value,
        "api_health": [
            {
                "name": name,
                "status": h.status.value,
                "note": (h.details.get("note") if h.details else None) or h.error or "",
                "latency_ms": getattr(h, "latency_ms", None),
            }
            for name, h in snap.api_health.items()
        ],
        "system": (
            {
                "cpu_percent": snap.system_resources.details.get("cpu_percent", 0.0),
                "memory_percent": snap.system_resources.details.get("memory_percent", 0.0),
                "disk_percent": snap.system_resources.details.get("disk_percent", 0.0),
            }
            if snap.system_resources
            else None
        ),
        "active_alerts": snap.active_alerts,
    }


def _collect_pnl() -> dict[str, Any]:
    from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415

    tracker = PnLTracker()
    today = tracker.today_report()
    history = tracker.historical_summary(days=14)
    pnl_row = today.get("daily_pnl") or {}
    positions = today.get("positions") or []
    trades = today.get("today_trades") or []
    total_pnl = float(pnl_row.get("total_unrealized_pnl", 0.0)) + float(
        pnl_row.get("total_realized_pnl", 0.0)
    )
    return {
        "date": today.get("date"),
        "total_pnl": total_pnl,
        "unrealized_pnl": float(pnl_row.get("total_unrealized_pnl", 0.0)),
        "realized_pnl": float(pnl_row.get("total_realized_pnl", 0.0)),
        "position_count": len(positions),
        "positions": [
            {
                "symbol": p.get("symbol"),
                "quantity": p.get("quantity"),
                "market_value": p.get("market_value"),
                "unrealized_pnl": p.get("unrealized_pnl"),
            }
            for p in positions[:25]
        ],
        "today_trade_count": len(trades),
        "recent_trades": [
            {
                "logged_at": (t.get("logged_at") or "")[:19],
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "quantity": t.get("quantity"),
                "fill_price": t.get("fill_price"),
                "status": t.get("status"),
                "strategy": t.get("strategy"),
            }
            for t in trades[:15]
        ],
        "history": [
            {
                "date": r.get("snapshot_date"),
                "pnl": float(r.get("total_unrealized_pnl", 0.0))
                + float(r.get("total_realized_pnl", 0.0)),
            }
            for r in history
        ],
    }


def _collect_account_value() -> dict[str, Any]:
    from shared.account_value_feed import AccountValueFeed  # noqa: PLC0415

    feed = AccountValueFeed()
    return {"value": feed.get(), "source": feed.get_source()}


def _collect_drawdown() -> dict[str, Any]:
    from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker  # noqa: PLC0415
    from shared.account_value_feed import AccountValueFeed  # noqa: PLC0415

    state = DrawdownCircuitBreaker().current_state()
    feed = AccountValueFeed()
    feed.get()  # populate cache so get_source() returns the live source
    nav_source = feed.get_source()
    # Synthetic NAV (IBKR offline → default fallback) makes drawdown meaningless.
    # Surface this clearly instead of showing a false TRIPPED state.
    is_synthetic = nav_source not in ("ibkr", "env")
    return {
        "peak_value": state.peak_value,
        "current_value": state.current_value,
        "drawdown_pct": state.drawdown_pct,
        "tripped": state.tripped and not is_synthetic,
        "tripped_at": state.tripped_at,
        "nav_source": nav_source,
        "is_synthetic": is_synthetic,
        "raw_tripped": state.tripped,
    }


def _collect_daily_loss_guard() -> dict[str, Any]:
    from strategies.daily_loss_guard import DailyLossGuard  # noqa: PLC0415
    from shared.account_value_feed import AccountValueFeed  # noqa: PLC0415

    guard = DailyLossGuard()
    account_value = AccountValueFeed().get()
    tripped, reason = guard.is_limit_reached(account_value_usd=account_value)
    return {
        "tripped": tripped,
        "reason": reason or "",
        "max_loss_pct": getattr(guard, "max_loss_pct", None),
        "account_value": account_value,
    }


def _collect_regime() -> dict[str, Any]:
    from strategies.regime_monitor import RegimeMonitor  # noqa: PLC0415

    mon = RegimeMonitor()
    current = mon.current_regime()
    history = mon.get_history(limit=10)
    return {
        "current": current.value if current else None,
        "history": [
            {
                "regime": r.regime,
                "score": round(r.composite_score, 2),
                "at": r.detected_at[:19],
            }
            for r in history
        ],
    }


def _collect_signal_journal() -> dict[str, Any]:
    from strategies.signal_journal import SignalJournal  # noqa: PLC0415

    journal = SignalJournal()
    recent = journal.get_recent(limit=20)
    rates = journal.get_hit_rates()
    return {
        "recent": [
            {
                "id": r.id,
                "ticker": r.ticker,
                "direction": r.direction,
                "confidence": round(r.confidence, 2),
                "strategy": r.strategy_source,
                "outcome": r.outcome or "pending",
                "logged_at": (r.logged_at or "")[:19],
            }
            for r in recent
        ],
        "hit_rates": [hr.to_dict() for hr in rates.values()],
    }


def _collect_throttle() -> dict[str, Any]:
    from core.execution_throttle import ExecutionThrottle  # noqa: PLC0415

    throttle = ExecutionThrottle()
    entries = throttle.all_entries()
    now = datetime.now(tz=timezone.utc).timestamp()
    return {
        "throttle_seconds": getattr(throttle, "_throttle_seconds", None)
        or getattr(throttle, "throttle_seconds", None),
        "entries": [
            {
                "ticker": e.get("ticker"),
                "last_executed_ts": e.get("last_executed_ts"),
                "age_seconds": round(now - float(e.get("last_executed_ts", now)), 1),
            }
            for e in entries
        ],
    }


def _collect_rolls() -> dict[str, Any]:
    from strategies.roll_manager import RollManager  # noqa: PLC0415
    from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415

    positions = PnLTracker().today_report().get("positions") or []
    urgent = RollManager().urgent_only(positions)
    return {
        "urgent_count": len(urgent),
        "urgent": [
            {
                "symbol": getattr(d, "symbol", None),
                "action": getattr(d, "action", None),
                "dte": getattr(d, "days_to_expiry", None),
                "reason": getattr(d, "reason", None),
            }
            for d in urgent[:20]
        ],
    }


def _collect_reconciler() -> dict[str, Any]:
    from core.position_reconciler import PositionReconciler  # noqa: PLC0415

    report = PositionReconciler().reconcile()
    return {
        "has_mismatches": report.has_mismatches,
        "missing_count": report.missing_count,
        "phantom_count": report.phantom_count,
        "size_mismatch_count": report.size_mismatch_count,
        "mismatches": [m.to_dict() for m in (report.mismatches or [])][:20],
    }


def _collect_eod_brief() -> dict[str, Any]:
    path = PROJECT_ROOT / "reports" / "daily_brief.txt"
    if not path.exists():
        return {"exists": False, "text": "(no daily brief yet)", "modified": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"exists": False, "text": f"(read error: {exc})", "modified": None}
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return {"exists": True, "text": text, "modified": mtime}


def _collect_autonomous() -> dict[str, Any]:
    """Read the AutonomousEngine heartbeat file written by _write_heartbeat_state.

    The engine writes data/autonomous_state.json every HEARTBEAT_INTERVAL (5s).
    Stale (>30s) means the engine isn't running.
    """
    import json  # noqa: PLC0415

    path = PROJECT_ROOT / "data" / "autonomous_state.json"
    if not path.exists():
        return {
            "running": False,
            "reason": "no heartbeat file (engine never started in this workspace)",
            "start_command": "python -m core.autonomous_engine",
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "reason": f"state file unreadable: {exc}"}
    hb_at_str = state.get("heartbeat_at")
    age_seconds = None
    stale = True
    if hb_at_str:
        try:
            hb_at = datetime.fromisoformat(hb_at_str)
            age_seconds = (datetime.now(tz=timezone.utc) - hb_at).total_seconds()
            stale = age_seconds > 30.0
        except Exception:  # noqa: BLE001
            pass
    state["age_seconds"] = age_seconds
    state["stale"] = stale
    state["alive"] = bool(state.get("running")) and not stale
    return state


def _collect_openclaw() -> dict[str, Any]:
    """Surface OpenClaw gateway config + reachability without requiring an active session."""
    import socket  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    gateway_url = os.getenv("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789")
    token_present = bool(os.getenv("OPENCLAW_GATEWAY_TOKEN"))
    parsed = urlparse(gateway_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18789

    reachable = False
    error: str | None = None
    try:
        with socket.create_connection((host, port), timeout=1.5):
            reachable = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    # Try to inspect a live bridge if one was created in this process
    bridge_status: dict[str, Any] | None = None
    try:
        from integrations import openclaw_gateway_bridge as _bridge_mod  # noqa: PLC0415

        existing = getattr(_bridge_mod, "_singleton_bridge", None) or getattr(
            _bridge_mod, "_bridge_instance", None
        )
        if existing is not None and hasattr(existing, "get_status"):
            raw = existing.get_status()
            bridge_status = {
                "connected": raw.get("connected"),
                "sessions_active": raw.get("sessions_active"),
                "agents_registered": raw.get("agents_registered"),
                "skills_registered": len(raw.get("skills_registered") or []),
                "cron_jobs": raw.get("cron_jobs"),
            }
    except Exception:  # noqa: BLE001
        bridge_status = None

    return {
        "gateway_url": gateway_url,
        "host": host,
        "port": port,
        "token_present": token_present,
        "reachable": reachable,
        "error": error,
        "bridge": bridge_status,
    }


# ── Snapshot aggregator ───────────────────────────────────────────────────


@dataclass
class DashboardSnapshot:
    generated_at: str
    health: dict[str, Any]
    pnl: dict[str, Any]
    account_value: dict[str, Any]
    drawdown: dict[str, Any]
    loss_guard: dict[str, Any]
    regime: dict[str, Any]
    signals: dict[str, Any]
    throttle: dict[str, Any]
    rolls: dict[str, Any]
    reconciler: dict[str, Any]
    eod: dict[str, Any]
    autonomous: dict[str, Any]
    openclaw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "health": self.health,
            "pnl": self.pnl,
            "account_value": self.account_value,
            "drawdown": self.drawdown,
            "loss_guard": self.loss_guard,
            "regime": self.regime,
            "signals": self.signals,
            "throttle": self.throttle,
            "rolls": self.rolls,
            "reconciler": self.reconciler,
            "eod": self.eod,
            "autonomous": self.autonomous,
            "openclaw": self.openclaw,
        }


def collect_snapshot() -> DashboardSnapshot:
    """Run every collector under _safe() and return a DashboardSnapshot."""
    return DashboardSnapshot(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        health=_safe(_collect_health, "health"),
        pnl=_safe(_collect_pnl, "pnl"),
        account_value=_safe(_collect_account_value, "account_value"),
        drawdown=_safe(_collect_drawdown, "drawdown"),
        loss_guard=_safe(_collect_daily_loss_guard, "loss_guard"),
        regime=_safe(_collect_regime, "regime"),
        signals=_safe(_collect_signal_journal, "signals"),
        throttle=_safe(_collect_throttle, "throttle"),
        rolls=_safe(_collect_rolls, "rolls"),
        reconciler=_safe(_collect_reconciler, "reconciler"),
        eod=_safe(_collect_eod_brief, "eod"),
        autonomous=_safe(_collect_autonomous, "autonomous"),
        openclaw=_safe(_collect_openclaw, "openclaw"),
    )


# ── HTML template ─────────────────────────────────────────────────────────

_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAC Command Dashboard</title>
<style>
  :root {
    --bg: #0b0f17; --panel: #131a26; --border: #1f2a3d; --muted: #6b7a90;
    --text: #d6deeb; --accent: #4fd1c5; --warn: #f6ad55; --bad: #fc8181;
    --good: #68d391; --crit: #f56565;
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--text); margin: 0;
               font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
               font-size: 13px; }
  header { padding: 14px 22px; border-bottom: 1px solid var(--border);
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 16px; letter-spacing: 1.2px; color: var(--accent); }
  header .meta { color: var(--muted); font-size: 12px; }
  main { padding: 18px; display: grid; gap: 14px;
         grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
           padding: 14px; min-height: 140px; }
  .panel h2 { margin: 0 0 10px; font-size: 12px; letter-spacing: 1.4px;
              color: var(--accent); text-transform: uppercase; }
  .panel .goal { color: var(--muted); font-size: 10px; letter-spacing: 1.2px;
                  margin: -8px 0 10px; text-transform: uppercase; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  td, th { padding: 4px 6px; border-bottom: 1px solid var(--border);
           text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 10px;
       text-transform: uppercase; letter-spacing: 1px; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 8px;
          font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
  .pill.ok      { background: rgba(104,211,145,.18); color: var(--good); }
  .pill.warn    { background: rgba(246,173,85,.18); color: var(--warn); }
  .pill.bad     { background: rgba(252,129,129,.18); color: var(--bad); }
  .pill.crit    { background: rgba(245,101,101,.30); color: #fff; }
  .num.pos { color: var(--good); }
  .num.neg { color: var(--bad); }
  .muted { color: var(--muted); }
  .err { color: var(--bad); font-size: 11px; margin-top: 6px; white-space: pre-wrap; }
  pre.brief { background: #08101c; padding: 12px; border-radius: 4px;
              max-height: 320px; overflow: auto; font-size: 11px; line-height: 1.4; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px;
        font-size: 12px; }
  .big { font-size: 24px; font-weight: 600; margin: 6px 0; }
  footer { color: var(--muted); padding: 10px 22px; font-size: 11px;
            border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <h1>BARREN WUFFET &middot; COMMAND DASHBOARD</h1>
  <span class="meta" id="meta">refreshing&hellip;</span>
</header>
<main id="grid"></main>
<footer>
  Aligned with <code>.context/04_workstreams/GOAL_MANDATE_ROADMAP.md</code>.
  Polling every 15s. Source: <code>monitoring/command_dashboard.py</code>.
</footer>
<script>
const $ = id => document.getElementById(id);
const fmtNum = v => (v == null ? '&mdash;' : Number(v).toLocaleString(undefined,
  { maximumFractionDigits: 2 }));
const fmtPct = v => (v == null ? '&mdash;' : (Number(v) * 100).toFixed(2) + '%');
const fmtMoney = v => (v == null ? '&mdash;' : '$' + fmtNum(v));
const cls = (v) => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
const pill = (text, kind) => `<span class="pill ${kind}">${text}</span>`;

function statusPill(s) {
  if (s === 'healthy' || s === 'ok' || s === 'OK') return pill('OK', 'ok');
  if (s === 'degraded' || s === 'warn') return pill('WARN', 'warn');
  if (s === 'unhealthy' || s === 'down') return pill('DOWN', 'bad');
  return pill(s || '?', 'warn');
}

function panel(title, goal, body, error) {
  return `<div class="panel">
    <h2>${title}</h2>
    <div class="goal">${goal}</div>
    ${error ? `<div class="err">${error}</div>` : body}
  </div>`;
}

function renderHealth(r) {
  if (!r.ok) return panel('Health Monitor', 'Goal 5 &mdash; MONITOR HEALTH', '', r.error);
  const d = r.data;
  const rows = d.api_health.map(a =>
    `<tr><td>${statusPill(a.status)}</td><td>${a.name}</td>
       <td class="muted">${(a.note || '').slice(0,48)}</td></tr>`).join('');
  const sys = d.system
    ? `<div class="muted">CPU ${d.system.cpu_percent.toFixed(1)}%
       &middot; Mem ${d.system.memory_percent.toFixed(1)}%
       &middot; Disk ${d.system.disk_percent.toFixed(1)}%</div>` : '';
  const alerts = (d.active_alerts || []).length
    ? `<div class="muted">Alerts: ${d.active_alerts.length}</div>` : '';
  return panel('Health Monitor', 'Goal 5 &mdash; MONITOR HEALTH',
    `<div>Overall: ${statusPill(d.overall_status)}</div>
     <table><tbody>${rows}</tbody></table>
     ${sys}${alerts}`);
}

function renderPnL(r, account) {
  if (!r.ok) return panel('P&amp;L &amp; Positions', 'Goal 4 &mdash; TRACK P&amp;L', '', r.error);
  const d = r.data;
  const av = account && account.ok ? account.data : null;
  const accLine = av
    ? `<div class="muted">Account NAV: ${fmtMoney(av.value)}
        <span class="pill ${av.source==='ibkr'?'ok':'warn'}">${av.source}</span></div>`
    : '';
  const posRows = d.positions.map(p =>
    `<tr><td>${p.symbol}</td><td>${fmtNum(p.quantity)}</td>
      <td>${fmtMoney(p.market_value)}</td>
      <td class="num ${cls(p.unrealized_pnl)}">${fmtMoney(p.unrealized_pnl)}</td></tr>`
  ).join('');
  return panel('P&amp;L &amp; Positions', 'Goal 4 &mdash; TRACK P&amp;L',
    `<div class="big num ${cls(d.total_pnl)}">${fmtMoney(d.total_pnl)}</div>
     <div class="muted">Realized ${fmtMoney(d.realized_pnl)}
        &middot; Unrealized ${fmtMoney(d.unrealized_pnl)}
        &middot; ${d.position_count} open
        &middot; ${d.today_trade_count} fills today</div>
     ${accLine}
     <table><thead><tr><th>Sym</th><th>Qty</th><th>MV</th><th>UPL</th></tr></thead>
     <tbody>${posRows || '<tr><td colspan=4 class="muted">No positions</td></tr>'}</tbody>
     </table>`);
}

function renderRisk(dd, lg, rolls, recon) {
  const ddBody = dd.ok
    ? (() => { const s = dd.data;
        const synthBadge = s.is_synthetic
          ? `${pill('SYNTHETIC NAV', 'warn')} <span class="muted">source: ${s.nav_source}</span>`
          : '';
        const status = s.is_synthetic
          ? pill('STANDBY', 'warn')
          : (s.tripped ? pill('TRIPPED', 'crit') : pill('OK', 'ok'));
        const ddCls = (!s.is_synthetic && s.drawdown_pct>=0.05) ? 'neg' : '';
        const ddVal = s.is_synthetic ? '&mdash;' : fmtPct(s.drawdown_pct);
        return `
        <div>Drawdown: <span class="num ${ddCls}">${ddVal}</span> ${status}</div>
        <div class="muted">Peak ${fmtMoney(s.peak_value)} &middot; Current ${fmtMoney(s.current_value)}</div>
        ${synthBadge ? `<div>${synthBadge}</div>` : ''}`;
      })()
    : `<div class="err">${dd.error}</div>`;
  const lgBody = lg.ok
    ? `<div>Daily Loss: ${lg.data.tripped ? pill('TRIPPED','crit') : pill('OK','ok')}
        <span class="muted">${lg.data.reason || ''}</span></div>`
    : `<div class="err">${lg.error}</div>`;
  const rollBody = rolls.ok
    ? (rolls.data.urgent_count
        ? `<div>Rolls urgent: ${pill(rolls.data.urgent_count, 'warn')}</div>
           <table><tbody>${rolls.data.urgent.map(u =>
             `<tr><td>${u.symbol||''}</td><td>${u.action||''}</td>
               <td class="muted">DTE ${u.dte ?? '?'}</td></tr>`).join('')}</tbody></table>`
        : `<div>Rolls: ${pill('clear', 'ok')}</div>`)
    : `<div class="err">${rolls.error}</div>`;
  const reconBody = recon.ok
    ? (recon.data.has_mismatches
        ? `<div>Position drift: ${pill('MISMATCH', 'bad')}
           <span class="muted">missing ${recon.data.missing_count} &middot;
           phantom ${recon.data.phantom_count} &middot;
           size ${recon.data.size_mismatch_count}</span></div>`
        : `<div>Reconciliation: ${pill('clean', 'ok')}</div>`)
    : `<div class="err">${recon.error}</div>`;
  return panel('Risk Guards', 'Goal 3 &mdash; MANAGE RISK',
    `${ddBody}${lgBody}${rollBody}${reconBody}`);
}

function renderSignals(s, regime) {
  if (!s.ok) return panel('Signals &amp; Calibration',
    'Goal 1 &mdash; FIND TRADES', '', s.error);
  const d = s.data;
  const reg = regime.ok && regime.data.current
    ? pill(regime.data.current, regime.data.current === 'CRISIS' ? 'crit'
          : regime.data.current === 'ELEVATED' ? 'bad'
          : regime.data.current === 'WATCH' ? 'warn' : 'ok')
    : pill('unknown', 'warn');
  const recent = d.recent.map(r =>
    `<tr><td>${r.ticker}</td><td>${r.direction}</td>
      <td>${r.confidence}</td><td class="muted">${r.strategy}</td>
      <td>${r.outcome === 'HIT' ? pill('HIT','ok')
            : r.outcome === 'MISS' ? pill('MISS','bad')
            : pill(r.outcome,'warn')}</td></tr>`).join('');
  const rates = d.hit_rates.map(h =>
    `<tr><td>${h.strategy}</td><td>${(h.rate*100).toFixed(1)}%</td>
      <td class="muted">${h.hits}/${h.hits+h.misses}</td></tr>`).join('');
  return panel('Signals &amp; Regime', 'Goal 1 &mdash; FIND TRADES',
    `<div>Regime: ${reg}</div>
     <table><thead><tr><th>Strategy</th><th>Hit %</th><th>Resolved</th></tr></thead>
     <tbody>${rates || '<tr><td colspan=3 class="muted">No resolved data</td></tr>'}</tbody></table>
     <div class="muted" style="margin-top:8px">Recent signals</div>
     <table><thead><tr><th>Tkr</th><th>Dir</th><th>Conf</th><th>Strat</th><th>Outcome</th></tr></thead>
     <tbody>${recent || '<tr><td colspan=5 class="muted">No signals yet</td></tr>'}</tbody></table>`);
}

function renderExec(pnl, throttle) {
  const trades = pnl.ok ? pnl.data.recent_trades : [];
  const tradeRows = trades.map(t =>
    `<tr><td class="muted">${(t.logged_at||'').slice(11,19)}</td>
      <td>${t.symbol}</td><td>${t.side}</td>
      <td>${fmtNum(t.quantity)}</td>
      <td>${fmtMoney(t.fill_price)}</td>
      <td class="muted">${t.status||''}</td></tr>`).join('');
  const thrBody = throttle.ok
    ? `<div class="muted">Throttle window: ${throttle.data.throttle_seconds}s
        &middot; Active: ${throttle.data.entries.length}</div>
       <table><tbody>${throttle.data.entries.slice(0,8).map(e =>
         `<tr><td>${e.ticker}</td><td class="muted">${Math.round(e.age_seconds)}s ago</td></tr>`
       ).join('') || '<tr><td class="muted">none</td></tr>'}</tbody></table>`
    : `<div class="err">${throttle.error}</div>`;
  return panel('Execution', 'Goal 2 &mdash; EXECUTE TRADES',
    `<div class="muted">Recent fills</div>
     <table><thead><tr><th>Time</th><th>Sym</th><th>Side</th><th>Qty</th><th>Px</th><th>Status</th></tr></thead>
     <tbody>${tradeRows || '<tr><td colspan=6 class="muted">No fills today</td></tr>'}</tbody></table>
     ${thrBody}`);
}

function renderEod(r) {
  if (!r.ok) return panel('Daily Brief', 'EOD Report', '', r.error);
  const d = r.data;
  return panel('Daily Brief', 'EOD Report',
    `<div class="muted">Updated ${(d.modified || '').slice(0,19)}</div>
     <pre class="brief">${(d.text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}</pre>`);
}

function renderAutonomous(r) {
  if (!r.ok) return panel('Autonomous Engine', 'AZ SUPREME core loop', '', r.error);
  const d = r.data;
  if (!d.alive) {
    const why = d.reason || (d.stale ? `heartbeat stale (${Math.round(d.age_seconds||0)}s old)` : 'not running');
    return panel('Autonomous Engine', 'AZ SUPREME core loop',
      `<div>Status: ${pill('OFFLINE', 'bad')}</div>
       <div class="muted">${why}</div>
       <div class="muted" style="margin-top:6px">Start with:
         <code>python -m core.autonomous_engine</code></div>
       ${d.last_report_at ? `<div class="muted">Last report: ${d.last_report_at.slice(0,19)}</div>` : ''}`);
  }
  const stateColor = d.doctrine_state === 'HALT' ? 'crit'
    : d.doctrine_state === 'SAFE_MODE' ? 'bad'
    : d.doctrine_state === 'CAUTION' ? 'warn' : 'ok';
  const upHours = (d.uptime_seconds / 3600).toFixed(1);
  return panel('Autonomous Engine', 'AZ SUPREME core loop',
    `<div>Status: ${pill('LIVE', 'ok')}
      &middot; Doctrine: ${pill(d.doctrine_state, stateColor)}
      &middot; PID ${d.pid}</div>
     <div class="kv" style="margin-top:8px">
       <span class="muted">Uptime</span><span>${upHours}h</span>
       <span class="muted">Cycles</span><span>${d.cycle_count}</span>
       <span class="muted">Errors</span><span class="num ${d.error_count>0?'neg':''}">${d.error_count}</span>
       <span class="muted">Components</span><span>${d.components_healthy}/${d.components_total} healthy</span>
       <span class="muted">Tasks</span><span>${d.tasks_registered} registered</span>
       <span class="muted">Last heartbeat</span><span>${Math.round(d.age_seconds||0)}s ago</span>
       ${d.last_report_at ? `<span class="muted">Last report</span><span>${d.last_report_at.slice(11,19)}</span>` : ''}
     </div>
     ${d.doctrine_reason ? `<div class="muted" style="margin-top:6px">${d.doctrine_reason}</div>` : ''}`);
}

function renderOpenClaw(r) {
  if (!r.ok) return panel('OpenClaw Gateway', 'Cross-platform agent bridge', '', r.error);
  const d = r.data;
  const conn = d.reachable ? pill('REACHABLE', 'ok') : pill('UNREACHABLE', 'bad');
  const tok = d.token_present ? pill('TOKEN', 'ok') : pill('NO TOKEN', 'warn');
  const bridge = d.bridge
    ? `<div class="kv" style="margin-top:8px">
        <span class="muted">Bridge</span>
          <span>${d.bridge.connected ? pill('connected','ok') : pill('disconnected','bad')}</span>
        <span class="muted">Sessions</span><span>${d.bridge.sessions_active||0}</span>
        <span class="muted">Agents</span>
          <span>${(d.bridge.agents_registered||[]).join(', ') || 'none'}</span>
        <span class="muted">Skills</span><span>${d.bridge.skills_registered||0}</span>
        <span class="muted">Cron jobs</span><span>${d.bridge.cron_jobs||0}</span>
       </div>`
    : `<div class="muted" style="margin-top:8px">No active bridge in this process. Start with:
         <code>python launch.py openclaw</code></div>`;
  return panel('OpenClaw Gateway', 'Cross-platform agent bridge',
    `<div>${conn} ${tok}</div>
     <div class="muted">${d.gateway_url}</div>
     ${d.error ? `<div class="err">${d.error}</div>` : ''}
     ${bridge}`);
}

async function refresh() {
  try {
    const r = await fetch('/api/snapshot', {cache: 'no-store'});
    const j = await r.json();
    $('meta').textContent = 'Last update ' + new Date(j.generated_at).toLocaleString();
    $('grid').innerHTML = [
      renderAutonomous(j.autonomous),
      renderOpenClaw(j.openclaw),
      renderHealth(j.health),
      renderSignals(j.signals, j.regime),
      renderExec(j.pnl, j.throttle),
      renderRisk(j.drawdown, j.loss_guard, j.rolls, j.reconciler),
      renderPnL(j.pnl, j.account_value),
      renderEod(j.eod),
    ].join('');
  } catch (e) {
    $('meta').textContent = 'fetch error: ' + e;
  }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


# ── HTTP server (FastAPI if available, falls back to stdlib) ──────────────


def _build_fastapi_app():
    from fastapi import FastAPI  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, JSONResponse  # noqa: PLC0415

    app = FastAPI(title="AAC Command Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:  # noqa: D401
        return _INDEX_HTML

    @app.get("/api/snapshot")
    def api_snapshot() -> JSONResponse:
        return JSONResponse(collect_snapshot().to_dict())

    @app.get("/api/healthz")
    def healthz() -> dict:
        return {"ok": True, "ts": datetime.now(tz=timezone.utc).isoformat()}

    return app


def run_dashboard(port: int = 8400, host: str = "127.0.0.1") -> int:
    """Start the command dashboard.  Blocks until interrupted."""
    try:
        import uvicorn  # noqa: PLC0415

        app = _build_fastapi_app()
        _log.info("command_dashboard_starting", host=host, port=port)
        print(f"  [+] AAC Command Dashboard on http://{host}:{port}", flush=True)
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] command dashboard failed: {exc}", flush=True)
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="AAC Command Dashboard")
    parser.add_argument("--port", type=int, default=int(os.getenv("AAC_DASH_PORT", "8400")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    return run_dashboard(port=args.port, host=args.host)


if __name__ == "__main__":
    sys.exit(main())
