from __future__ import annotations

"""
monitoring.command_console — AAC Command Console (terminal).

Terminal-rendered twin of monitoring.command_dashboard.  Shares the same
collect_snapshot() so the web and console surfaces stay in lock-step.

Renders the FIVE mission goals from GOAL_MANDATE_ROADMAP.md as colored panels:

    1. FIND TRADES   - SignalJournal hit rates + regime
    2. EXECUTE       - throttle + recent fills
    3. MANAGE RISK   - drawdown CB, loss guard, rolls, reconciler
    4. TRACK P&L     - today P&L, account value, positions
    5. HEALTH        - API health + system + alerts

Plus AUTONOMOUS engine + OPENCLAW gateway panels.

Usage:
    python -m monitoring.command_console            # 5s refresh
    python launch.py console                        # via launcher
    python launch.py console --interval 10
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

_log = structlog.get_logger()

# Force UTF-8 stdout on Windows so box-drawing chars don't blow up under cp1252
if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from monitoring.command_dashboard import collect_snapshot  # noqa: E402


# ── ANSI color helpers ────────────────────────────────────────────────────

_ENABLE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_C = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "grey": "\x1b[90m",
    "bg_red": "\x1b[41m",
    "bg_green": "\x1b[42m",
    "bg_yellow": "\x1b[43m",
}


def _c(text: str, *colors: str) -> str:
    if not _ENABLE_COLOR:
        return text
    prefix = "".join(_C.get(c, "") for c in colors)
    return f"{prefix}{text}{_C['reset']}"


# ── Formatting primitives ─────────────────────────────────────────────────

PANEL_WIDTH = 96


def _hr(char: str = "─") -> str:
    return _c(char * PANEL_WIDTH, "grey")


def _panel_header(title: str, subtitle: str = "") -> str:
    bar = "═" * PANEL_WIDTH
    head = _c(bar, "cyan")
    sub = f"  {_c(subtitle, 'grey')}" if subtitle else ""
    return f"{head}\n{_c('  ' + title, 'bold', 'cyan')}{sub}\n{head}"


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def _pill(text: str, kind: str = "ok") -> str:
    palette = {
        "ok": ("bg_green", "white"),
        "warn": ("bg_yellow", "white"),
        "bad": ("bg_red", "white"),
        "crit": ("bg_red", "bold", "white"),
        "muted": ("grey",),
    }
    colors = palette.get(kind, ("grey",))
    return _c(f" {text} ", *colors)


def _err(panel: dict[str, Any]) -> str | None:
    if panel.get("ok"):
        return None
    return _c(f"  unavailable: {panel.get('error', 'unknown')}", "red")


# ── Panel renderers ───────────────────────────────────────────────────────


def _render_header(snapshot: dict[str, Any]) -> str:
    ts = snapshot.get("generated_at", "")
    line1 = _c("  AAC COMMAND CONSOLE", "bold", "cyan")
    line2 = _c(f"  GOAL_MANDATE_ROADMAP — Sprints 1-25 + Autonomous + OpenClaw", "grey")
    line3 = _c(f"  snapshot @ {ts}", "grey")
    bar = _c("=" * PANEL_WIDTH, "cyan")
    return f"{bar}\n{line1}\n{line2}\n{line3}\n{bar}"


def _render_health(panel: dict[str, Any]) -> str:
    body = [_panel_header("5. HEALTH", "API + system + alerts")]
    err = _err(panel)
    if err:
        body.append(err)
        return "\n".join(body)
    d = panel["data"]
    overall = d.get("overall_status", "unknown").upper()
    kind = {"healthy": "ok", "degraded": "warn", "down": "bad"}.get(
        d.get("overall_status", ""), "muted"
    )
    body.append(f"  Overall: {_pill(overall, kind)}")
    sysinfo = d.get("system") or {}
    if sysinfo:
        body.append(
            f"  CPU {sysinfo.get('cpu_percent', 0):.0f}%  "
            f"MEM {sysinfo.get('memory_percent', 0):.0f}%  "
            f"DISK {sysinfo.get('disk_percent', 0):.0f}%"
        )
    body.append("")
    body.append(_c("  API                       Status      Note", "grey"))
    for api in d.get("api_health", []):
        st = api.get("status", "unknown")
        st_color = (
            "green" if st == "healthy" else "yellow" if st == "degraded" else "red"
        )
        name = api.get("name", "?")[:24].ljust(24)
        status = _c(st.ljust(10), st_color)
        note = (api.get("note") or "")[:48]
        body.append(f"  {name}  {status}  {_c(note, 'grey')}")
    alerts = d.get("active_alerts") or []
    if alerts:
        body.append("")
        body.append(_c(f"  Active alerts ({len(alerts)}):", "yellow"))
        for a in alerts[:5]:
            body.append(_c(f"    • {a}", "yellow"))
    return "\n".join(body)


def _render_pnl(panel: dict[str, Any], av: dict[str, Any]) -> str:
    body = [_panel_header("4. TRACK P&L", "today P&L + account value + positions")]
    err = _err(panel)
    if err:
        body.append(err)
        return "\n".join(body)
    d = panel["data"]
    total = d.get("total_pnl", 0.0)
    pnl_color = "green" if total >= 0 else "red"
    body.append(f"  Date:      {d.get('date', '—')}")
    body.append(f"  Total P&L: {_c(_money(total), pnl_color, 'bold')}")
    if av.get("ok"):
        avd = av["data"]
        src = avd.get("source", "—")
        src_pill = _pill(src.upper(), "ok" if src in ("ibkr", "env") else "warn")
        body.append(f"  Account:   {_money(avd.get('value'))}  {src_pill}")
    pos = d.get("positions") or []
    body.append(f"  Positions: {len(pos)}")
    if pos:
        body.append(_c("  Symbol       Qty       Avg Cost     Mkt Value     Unrlz P&L", "grey"))
        for p in pos[:8]:
            sym = (p.get("symbol") or "?")[:12].ljust(12)
            qty = f"{p.get('quantity', 0):>6.0f}"
            avg = f"{p.get('avg_cost', 0):>10.2f}"
            mv = f"{p.get('market_value', 0):>11.2f}"
            up = p.get("unrealized_pnl", 0)
            up_str = _c(f"{up:>+10.2f}", "green" if up >= 0 else "red")
            body.append(f"  {sym} {qty}  {avg}  {mv}  {up_str}")
    trades = d.get("today_trades") or []
    if trades:
        body.append("")
        body.append(_c(f"  Today's fills ({len(trades)}):", "grey"))
        for t in trades[:5]:
            body.append(
                f"    {t.get('symbol', '?'):<10} {t.get('side', '?'):<6} "
                f"{t.get('quantity', 0):>4} @ {t.get('price', 0):>8.2f}"
            )
    return "\n".join(body)


def _render_risk(
    dd: dict[str, Any],
    lg: dict[str, Any],
    rolls: dict[str, Any],
    recon: dict[str, Any],
) -> str:
    body = [_panel_header("3. MANAGE RISK", "drawdown · loss guard · rolls · reconciler")]

    if dd.get("ok"):
        s = dd["data"]
        if s.get("is_synthetic"):
            status = _pill("STANDBY", "warn")
            extra = _c(f"  (synthetic NAV — source: {s.get('nav_source')})", "yellow")
            body.append(f"  Drawdown: {_c('—', 'grey')}  {status}{extra}")
        else:
            tripped = s.get("tripped")
            status = _pill("TRIPPED", "crit") if tripped else _pill("OK", "ok")
            color = "red" if (s.get("drawdown_pct", 0) or 0) >= 0.05 else "white"
            peak = _money(s.get("peak_value"))
            cur = _money(s.get("current_value"))
            body.append(
                f"  Drawdown: {_c(_pct(s.get('drawdown_pct')), color)}  {status}  "
                + _c(f"peak {peak} / cur {cur}", "grey")
            )
    else:
        body.append(_err(dd) or "")

    if lg.get("ok"):
        l = lg["data"]
        status = _pill("TRIPPED", "crit") if l.get("tripped") else _pill("OK", "ok")
        reason = _c(f"  {l.get('reason') or ''}", "grey")
        body.append(f"  Daily Loss: {status}{reason}")
    else:
        body.append(_err(lg) or "")

    if rolls.get("ok"):
        r = rolls["data"]
        n = r.get("urgent_count", 0)
        if n:
            body.append(f"  Rolls urgent: {_pill(str(n), 'warn')}")
            for u in (r.get("urgent") or [])[:5]:
                body.append(
                    f"    {u.get('symbol', '?'):<8} {u.get('action', '?'):<8} "
                    f"DTE {u.get('dte', '?')}"
                )
        else:
            body.append(f"  Rolls: {_pill('clear', 'ok')}")
    else:
        body.append(_err(rolls) or "")

    if recon.get("ok"):
        rd = recon["data"]
        if rd.get("has_mismatches"):
            mc = rd.get("missing_count", 0)
            pc = rd.get("phantom_count", 0)
            sc = rd.get("size_mismatch_count", 0)
            body.append(
                f"  Position drift: {_pill('MISMATCH', 'bad')}  "
                + _c(f"missing {mc} · phantom {pc} · size {sc}", "grey")
            )
        else:
            body.append(f"  Reconciliation: {_pill('clean', 'ok')}")
    else:
        body.append(_err(recon) or "")

    return "\n".join(body)


def _render_signals(sig: dict[str, Any], regime: dict[str, Any]) -> str:
    body = [_panel_header("1. FIND TRADES", "signal journal hit rates + regime")]
    if regime.get("ok"):
        rd = regime["data"]
        cur = rd.get("current", "?")
        score = rd.get("composite_score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
        regime_color = {
            "CALM": "green",
            "WATCH": "yellow",
            "ELEVATED": "yellow",
            "CRISIS": "red",
        }.get(cur, "white")
        body.append(f"  Regime: {_c(cur, regime_color, 'bold')}  composite {score_str}")
    else:
        body.append(_err(regime) or "")

    if sig.get("ok"):
        d = sig["data"]
        body.append(f"  Total resolved: {d.get('total_resolved', 0)}")
        rates = d.get("hit_rates") or {}
        if rates:
            body.append(_c("  Strategy           Hits  Misses   Rate", "grey"))
            for strat, h in rates.items():
                rate = h.get("rate", 0)
                color = "green" if rate >= 0.55 else "yellow" if rate >= 0.45 else "red"
                body.append(
                    f"  {strat:<18} {h.get('hits', 0):>4}  {h.get('misses', 0):>6}   "
                    f"{_c(_pct(rate), color)}"
                )
        recent = d.get("recent") or []
        if recent:
            body.append("")
            body.append(_c(f"  Recent signals ({len(recent)}):", "grey"))
            for r in recent[:5]:
                body.append(
                    f"    {r.get('ticker', '?'):<8} {r.get('direction', '?'):<6} "
                    f"conf {r.get('confidence', 0):.2f}  [{r.get('strategy_source', '?')}]"
                )
    else:
        body.append(_err(sig) or "")
    return "\n".join(body)


def _render_execution(throttle: dict[str, Any], pnl: dict[str, Any]) -> str:
    body = [_panel_header("2. EXECUTE", "throttle + recent fills")]
    if throttle.get("ok"):
        t = throttle["data"]
        active = t.get("active_count", 0)
        body.append(f"  Throttled tickers: {active}  (window {t.get('throttle_seconds', 0)}s)")
        for entry in (t.get("entries") or [])[:5]:
            body.append(
                f"    {entry.get('ticker', '?'):<8} "
                f"remaining {entry.get('remaining_seconds', 0):.0f}s"
            )
    else:
        body.append(_err(throttle) or "")
    if pnl.get("ok"):
        trades = pnl["data"].get("today_trades") or []
        body.append(f"  Today's fills: {len(trades)}")
    return "\n".join(body)


def _render_autonomous(panel: dict[str, Any]) -> str:
    body = [_panel_header("AUTONOMOUS ENGINE", "core/autonomous_engine.py heartbeat")]
    err = _err(panel)
    if err:
        body.append(err)
        return "\n".join(body)
    d = panel["data"]
    if not d.get("running"):
        body.append(f"  {_pill('OFFLINE', 'bad')}  {_c(d.get('reason', ''), 'grey')}")
        body.append(_c("  Start: python launch.py autonomous", "grey"))
        return "\n".join(body)
    alive = d.get("alive")
    stale = d.get("stale")
    if alive and not stale:
        status = _pill("LIVE", "ok")
    elif stale:
        status = _pill("STALE", "warn")
    else:
        status = _pill("UNKNOWN", "muted")
    body.append(f"  {status}  pid {d.get('pid')}  uptime {d.get('uptime_seconds', 0):.0f}s")
    body.append(
        f"  Doctrine: {_c(d.get('doctrine_state', '?'), 'cyan')}  "
        f"({d.get('doctrine_reason', '')})"
    )
    body.append(
        f"  Components healthy: {d.get('components_healthy', 0)}/{d.get('components_total', 0)}  "
        f"tasks: {d.get('tasks_registered', 0)}  cycles: {d.get('cycle_count', 0)}  "
        f"errors: {d.get('error_count', 0)}"
    )
    body.append(
        f"  Heartbeat age: {d.get('age_seconds', 0):.1f}s  "
        f"started {d.get('started_at', '?')}"
    )
    return "\n".join(body)


def _render_openclaw(panel: dict[str, Any]) -> str:
    body = [_panel_header("OPENCLAW BRIDGE", "ws://127.0.0.1:18789 gateway")]
    err = _err(panel)
    if err:
        body.append(err)
        return "\n".join(body)
    d = panel["data"]
    reachable = d.get("reachable")
    token = d.get("token_configured")
    body.append(
        f"  Gateway: {_pill('REACHABLE', 'ok') if reachable else _pill('UNREACHABLE', 'bad')}  "
        f"token: {_pill('CONFIGURED', 'ok') if token else _pill('MISSING', 'warn')}"
    )
    if d.get("bridge_active"):
        body.append(f"  In-process bridge: {_pill('ACTIVE', 'ok')}")
        body.append(f"  Skills registered: {d.get('skills_count', 0)}")
    else:
        body.append(f"  In-process bridge: {_c('not active in this process', 'grey')}")
        body.append(_c("  Start: python launch.py openclaw", "grey"))
    return "\n".join(body)


def _render_eod(panel: dict[str, Any]) -> str:
    body = [_panel_header("DAILY BRIEF", "reports/daily_brief.txt")]
    err = _err(panel)
    if err:
        body.append(err)
        return "\n".join(body)
    d = panel["data"]
    if not d.get("exists"):
        body.append(_c(f"  {d.get('text', '(none)')}", "grey"))
        return "\n".join(body)
    body.append(_c(f"  modified: {d.get('modified', '?')}", "grey"))
    text = d.get("text", "") or ""
    for line in text.splitlines()[:18]:
        body.append(f"  {line}")
    return "\n".join(body)


# ── Main render loop ──────────────────────────────────────────────────────


def _clear_screen() -> None:
    if _ENABLE_COLOR:
        sys.stdout.write("\x1b[2J\x1b[H")
    else:
        os.system("cls" if os.name == "nt" else "clear")


def render_once() -> str:
    snap = collect_snapshot().to_dict()
    sections = [
        _render_header(snap),
        _render_autonomous(snap.get("autonomous", {})),
        _render_signals(snap.get("signals", {}), snap.get("regime", {})),
        _render_execution(snap.get("throttle", {}), snap.get("pnl", {})),
        _render_risk(
            snap.get("drawdown", {}),
            snap.get("loss_guard", {}),
            snap.get("rolls", {}),
            snap.get("reconciler", {}),
        ),
        _render_pnl(snap.get("pnl", {}), snap.get("account_value", {})),
        _render_health(snap.get("health", {})),
        _render_openclaw(snap.get("openclaw", {})),
        _render_eod(snap.get("eod", {})),
        _c("=" * PANEL_WIDTH, "cyan"),
        _c("  Ctrl+C to exit  ·  paired with web dashboard at http://127.0.0.1:8400", "grey"),
    ]
    return "\n".join(sections) + "\n"


def run_console(interval: float = 5.0, once: bool = False) -> int:
    """Render the console; refresh every `interval` seconds (or once and exit)."""
    if once:
        sys.stdout.write(render_once())
        sys.stdout.flush()
        return 0
    try:
        while True:
            _clear_screen()
            sys.stdout.write(render_once())
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write(_c("\n  console stopped\n", "grey"))
        return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="command_console", description="AAC Command Console")
    p.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    p.add_argument("--once", action="store_true", help="render one frame and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_console(interval=args.interval, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
