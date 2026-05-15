"""AAC dashboard v2 — design system primitives.

Reusable formatters, KPI cards, table renderers, and theme helpers used by
``dashboard_v2.py``. Keeps presentation concerns out of the page modules.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from typing import Any

# ── Color palette (semantic) ────────────────────────────────────────────────

GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
BLUE = "#3b82f6"
SLATE = "#94a3b8"
INK = "#e2e8f0"
PANEL = "#0f172a"
BORDER = "#1e293b"

# Columns we never want to show (raw IDs, internal flags)
UUID_LIKE_COLS = {
    "politician_id",
    "transaction_id",
    "filing_id",
    "trade_id",
    "id",
    "uuid",
    "_id",
    "filer_info_id",
    "issuer_id",
    "industry_id",
    "sector_id",
}


# ── Formatters ──────────────────────────────────────────────────────────────


def fmt_usd(v: Any, *, signed: bool = False, decimals: int | None = None) -> str:
    """Format a number as USD with K/M/B units. Returns ``—`` when missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    a = abs(x)
    if a >= 1_000_000_000:
        body = f"${a / 1_000_000_000:.2f}B"
    elif a >= 1_000_000:
        body = f"${a / 1_000_000:.2f}M"
    elif a >= 10_000:
        body = f"${a / 1_000:.1f}K"
    elif decimals is not None:
        body = f"${a:,.{decimals}f}"
    elif a >= 1:
        body = f"${a:,.2f}"
    else:
        body = f"${a:.4f}".rstrip("0").rstrip(".")
    return f"{sign}{body}"


def fmt_num(v: Any, *, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.{decimals}f}"


def fmt_pct(v: Any, *, decimals: int = 1, signed: bool = False) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if (signed and x > 0) else ""
    return f"{sign}{x:.{decimals}f}%"


def fmt_relative_time(value: Any) -> str:
    """Convert ISO timestamp to '2m ago', '3h ago', etc."""
    if not value:
        return "—"
    try:
        if isinstance(value, str):
            ts = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, dt.datetime):
            ts = value
        else:
            return str(value)
    except ValueError:
        return str(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    delta = now - ts
    secs = int(delta.total_seconds())
    if secs < 0:
        return ts.strftime("%H:%M")
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    if secs < 86400 * 7:
        return f"{secs // 86400}d ago"
    return ts.strftime("%Y-%m-%d")


def fmt_date(value: Any) -> str:
    if not value:
        return "—"
    try:
        if isinstance(value, str):
            ts = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, dt.datetime):
            ts = value
        else:
            return str(value)
    except ValueError:
        return str(value)
    return ts.strftime("%Y-%m-%d")


# ── Theme & layout ──────────────────────────────────────────────────────────


def inject_theme() -> None:
    """One-time CSS injection. Pair with Streamlit's dark theme config."""
    import streamlit as st

    css = f"""
    <style>
      .stApp {{ background: #020617; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1600px; }}

      /* KPI cards */
      .kpi-card {{
        background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
      }}
      .kpi-label {{ color: {SLATE}; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.06em; font-weight: 600;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .kpi-value {{ color: {INK}; font-size: 26px; font-weight: 700;
                    font-feature-settings: "tnum"; margin-top: 4px; line-height: 1.1;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .kpi-delta {{ font-size: 12px; margin-top: 4px; font-weight: 600;
                    font-feature-settings: "tnum";
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .kpi-delta.up   {{ color: {GREEN}; }}
      .kpi-delta.down {{ color: {RED};   }}
      .kpi-delta.flat {{ color: {SLATE}; }}
      .kpi-sub {{ color: {SLATE}; font-size: 11px; margin-top: 2px; }}

      /* Status pills */
      .pill {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 10px; border-radius: 999px; font-size: 11px;
        font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid {BORDER};
      }}
      .pill.ok    {{ background: rgba(34,197,94,0.10); color: {GREEN}; border-color: rgba(34,197,94,0.30); }}
      .pill.warn  {{ background: rgba(245,158,11,0.10); color: {AMBER}; border-color: rgba(245,158,11,0.30); }}
      .pill.bad   {{ background: rgba(239,68,68,0.10); color: {RED};   border-color: rgba(239,68,68,0.30); }}
      .pill.idle  {{ background: rgba(148,163,184,0.10); color: {SLATE}; }}

      /* Section headers */
      .section-h {{
        color: {INK}; font-size: 15px; font-weight: 700;
        margin: 18px 0 8px 0; padding-bottom: 6px;
        border-bottom: 1px solid {BORDER};
      }}
      .section-h .sub {{ color: {SLATE}; font-weight: 400; font-size: 12px; margin-left: 8px; }}

      /* Tables */
      [data-testid="stDataFrame"] {{
        font-feature-settings: "tnum";
      }}

      /* Hide streamlit chrome */
      header[data-testid="stHeader"] {{ background: transparent; }}
      footer {{ visibility: hidden; }}
      #MainMenu {{ visibility: hidden; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    import streamlit as st

    extra = f' <span class="sub">· {sub}</span>' if sub else ""
    st.markdown(f'<div class="section-h">{title}{extra}</div>', unsafe_allow_html=True)


def pill(label: str, status: str = "idle") -> str:
    """Return raw HTML string for a status pill. Use inside ``st.markdown``."""
    status = status if status in {"ok", "warn", "bad", "idle"} else "idle"
    glyph = {"ok": "●", "warn": "▲", "bad": "■", "idle": "○"}[status]
    return f'<span class="pill {status}">{glyph} {label}</span>'


def pill_age(label: str, age_seconds: float | None) -> str:
    """Pill coloured by data age. <60s ok, <600s warn, otherwise bad."""
    if age_seconds is None:
        return pill(label, "idle")
    if age_seconds < 60:
        suffix = "now"
        status = "ok"
    elif age_seconds < 600:
        suffix = f"{int(age_seconds//60)}m"
        status = "warn"
    else:
        if age_seconds < 86400:
            suffix = f"{int(age_seconds//3600)}h"
        else:
            suffix = f"{int(age_seconds//86400)}d"
        status = "bad"
    return pill(f"{label} · {suffix}", status)


# ── KPI cards ───────────────────────────────────────────────────────────────


def kpi(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    delta_dir: str = "flat",
    sub: str | None = None,
) -> str:
    """Return raw HTML for a KPI card. Render via ``st.markdown(unsafe_allow_html=True)``."""
    delta_html = ""
    if delta:
        cls = delta_dir if delta_dir in {"up", "down", "flat"} else "flat"
        arrow = {"up": "▲", "down": "▼", "flat": "→"}[cls]
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}{sub_html}"
        "</div>"
    )


def kpi_row(cards: list[str]) -> None:
    """Render a row of KPI cards with equal columns."""
    import streamlit as st

    if not cards:
        return
    cols = st.columns(len(cards), gap="small")
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def delta_dir_from(value: Any, *, inverse: bool = False) -> str:
    """Map a number to up/down/flat. ``inverse`` swaps (e.g. for losses)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "flat"
    if x == 0:
        return "flat"
    up = x > 0
    if inverse:
        up = not up
    return "up" if up else "down"


# ── Table helpers ───────────────────────────────────────────────────────────


def _drop_uuid_cols(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    keys = set().union(*(r.keys() for r in rows if isinstance(r, dict)))
    keep = [k for k in keys if k.lower() not in UUID_LIKE_COLS]
    return [{k: r.get(k) for k in keep} for r in rows if isinstance(r, dict)]


def _coerce_none(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({k: ("—" if v is None else v) for k, v in r.items()})
    return out


def smart_table(
    rows: Iterable[Mapping[str, Any]] | None,
    *,
    column_config: dict[str, Any] | None = None,
    column_order: list[str] | None = None,
    hide_uuids: bool = True,
    height: int | None = None,
    empty_msg: str = "No data.",
) -> None:
    """Render a dataframe with sensible defaults: hide UUIDs, hide index,
    fullscreen + search + download enabled, optional column config."""
    import streamlit as st

    data = list(rows or [])
    if not data:
        st.caption(empty_msg)
        return
    norm = [dict(r) for r in data if isinstance(r, Mapping)]
    if hide_uuids:
        norm = _drop_uuid_cols(norm)
    norm = _coerce_none(norm)

    kwargs: dict[str, Any] = {"width": "stretch", "hide_index": True}
    if column_config:
        kwargs["column_config"] = column_config
    if column_order:
        # Only include columns that actually exist
        present = set().union(*(r.keys() for r in norm))
        kwargs["column_order"] = [c for c in column_order if c in present]
    if height is not None:
        kwargs["height"] = height
    st.dataframe(norm, **kwargs)


# ── Streamlit column_config builders (typed) ────────────────────────────────


def col_usd(label: str, *, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.NumberColumn(label, format="$%.2f", width=width)


def col_usd_compact(label: str, *, width: str = "small") -> Any:
    """Format dollars with K/M/B suffix via Streamlit's `compact` format."""
    import streamlit as st

    return st.column_config.NumberColumn(label, format="compact", width=width)


def col_pct(label: str, *, decimals: int = 1, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.NumberColumn(label, format=f"%.{decimals}f%%", width=width)


def col_int(label: str, *, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.NumberColumn(label, format="%d", width=width)


def col_num(label: str, *, digits: int = 2, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.NumberColumn(label, format=f"%.{digits}f", width=width)


def col_text(label: str, *, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.TextColumn(label, width=width)


def col_date(label: str, *, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.DateColumn(label, format="YYYY-MM-DD", width=width)


def col_progress(label: str, *, min_value: float = 0, max_value: float = 100, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.ProgressColumn(
        label, min_value=min_value, max_value=max_value, format="%.0f%%", width=width
    )


def col_link(label: str, *, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.LinkColumn(label, width=width)


def col_bar(label: str, *, y_min: float | None = None, y_max: float | None = None, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.BarChartColumn(label, y_min=y_min, y_max=y_max, width=width)


def col_line(label: str, *, y_min: float | None = None, y_max: float | None = None, width: str = "small") -> Any:
    import streamlit as st

    return st.column_config.LineChartColumn(label, y_min=y_min, y_max=y_max, width=width)


__all__ = [
    "GREEN", "RED", "AMBER", "BLUE", "SLATE", "INK", "PANEL", "BORDER",
    "fmt_usd", "fmt_num", "fmt_pct", "fmt_relative_time", "fmt_date",
    "inject_theme", "section", "pill", "pill_age",
    "kpi", "kpi_row", "delta_dir_from",
    "smart_table",
    "col_usd", "col_usd_compact", "col_pct", "col_int", "col_num", "col_text",
    "col_date", "col_progress", "col_link", "col_bar", "col_line",
]
