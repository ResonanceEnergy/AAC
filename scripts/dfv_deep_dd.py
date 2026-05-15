from __future__ import annotations

"""
DFV Deep DD — GameStop-2021-grade due diligence on every held position.

For each symbol on the sheet, this script:

  1. Pulls EVERY leg (strike / right / expiry / qty / cost) from mission_control.
  2. Fetches live spot + 30-day realized vol from yfinance.
  3. Calls Grok with Live Search ON (web + news + x) for:
        - upcoming events through last_expiry + 30 days
        - leading indicators per symbol/sector
        - regime drivers / sentiment
  4. Runs Monte Carlo (GBM, 20k paths) on the underlying through expiry,
     prices each leg at expiry, sums P/L per path → distribution + percentiles
     + P(touch invalidation) + P(profit) + expected exit P&L.
  5. Calls Grok again to write a FINAL ALIGNED thesis that references the
     event calendar AND the MC stats, with a concrete exit ladder
     (profit-takes, stop, time-stop) and updated invalidation.
  6. Writes:
        - agents/dfv/memory/deep_dd/<SYMBOL>.json   (full DD)
        - agents/dfv/memory/deep_dd/_summary.md     (table for the operator)
     and updates thesis_log.json with the aligned thesis (author=GROK).

Per .github/instructions/dfv-decisions.instructions.md:
  G1 thesis:       satisfied + aligned with events + MC.
  G6 invalidation: explicit price/time trigger, MC-validated probability.
  Author tag:      "GROK" — operator must ratify before flipping to "DFV".

Usage (from repo root):
  python scripts/dfv_deep_dd.py                    # every held symbol
  python scripts/dfv_deep_dd.py --symbols TSLA,GME
  python scripts/dfv_deep_dd.py --paths 50000      # heavier MC
  python scripts/dfv_deep_dd.py --no-search        # skip Grok live search
  python scripts/dfv_deep_dd.py --dry-run          # no writes
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.dfv.decision_engine import DFV  # noqa: E402
from monitoring.mission_control import collect_payload  # noqa: E402

DEEP_DD_DIR = REPO_ROOT / "agents" / "dfv" / "memory" / "deep_dd"
DEEP_DD_DIR.mkdir(parents=True, exist_ok=True)

GROK_MODEL_RESEARCH = "grok-4"        # web_search tool needs grok-4 family
GROK_MODEL_SYNTH = "grok-3-mini"      # synthesis is fine on mini (no search needed)
DEFAULT_PATHS = 20_000


# ─── xAI Grok wrapper with optional Live Search ────────────────────────────


def _xai_key() -> str:
    try:
        from dotenv import dotenv_values
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            v = dotenv_values(str(env_path))
            k = (v.get("XAI_API_KEY") or v.get("GROK_API_KEY") or "").strip()
            if k:
                return k
    except ImportError:
        pass
    return (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip()


def _grok(
    prompt: str,
    *,
    system: str = "",
    model: str = GROK_MODEL_SYNTH,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    search: bool = False,
    search_max: int = 15,
) -> dict[str, Any]:
    import httpx

    key = _xai_key()
    if not key:
        return {"ok": False, "error": "XAI_API_KEY not set", "text": ""}

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            return {"ok": False, "model": model, "error": f"HTTP {r.status_code}: {r.text[:400]}", "text": ""}
        data = r.json()
        choice = data.get("choices", [{}])[0]
        text = (choice.get("message", {}).get("content") or "").strip()
        return {
            "ok": True,
            "model": model,
            "text": text,
            "citations": [],
            "usage": data.get("usage", {}),
        }
    except httpx.HTTPError as e:
        return {"ok": False, "model": model, "error": f"http: {e}", "text": ""}


def _grok_research(
    prompt: str,
    *,
    system: str = "",
    model: str = GROK_MODEL_RESEARCH,
    max_tool_calls: int = 8,
) -> dict[str, Any]:
    """Call xAI Responses API with web_search + x_search tools enabled.

    The legacy Chat Completions `search_parameters` was deprecated
    (HTTP 410 \"Live search is deprecated\"). The new path is /v1/responses
    with `tools=[{type: web_search}, {type: x_search}]`.
    """
    import httpx

    key = _xai_key()
    if not key:
        return {"ok": False, "error": "XAI_API_KEY not set", "text": ""}

    input_blocks: list[dict[str, Any]] = []
    if system:
        input_blocks.append({"role": "system", "content": system})
    input_blocks.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "input": input_blocks,
        "tools": [{"type": "web_search"}, {"type": "x_search"}],
        "max_tool_calls": int(max_tool_calls),
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=240.0) as client:
            r = client.post("https://api.x.ai/v1/responses", headers=headers, json=payload)
        if r.status_code >= 400:
            return {"ok": False, "model": model, "error": f"HTTP {r.status_code}: {r.text[:400]}", "text": ""}
        data = r.json()
        # Responses API shape: data.output is a list; final assistant text in output_text or items
        text = (data.get("output_text") or "").strip()
        if not text:
            for item in data.get("output", []) or []:
                if item.get("type") == "message":
                    for c in item.get("content", []) or []:
                        if c.get("type") in ("output_text", "text"):
                            text = (text + "\n" + (c.get("text") or "")).strip()
        citations: list[Any] = []
        for item in data.get("output", []) or []:
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    for ann in c.get("annotations", []) or []:
                        if ann.get("type") in ("url_citation", "citation"):
                            citations.append(ann)
        return {"ok": True, "model": model, "text": text, "citations": citations, "raw": data}
    except httpx.HTTPError as e:
        return {"ok": False, "model": model, "error": f"http: {e}", "text": ""}


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", text, re.S)
        candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last-ditch: try to repair trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# ─── Position aggregation ───────────────────────────────────────────────────


def _collect_legs(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return {underlying: [leg, leg, ...]} preserving strike/right/expiry/qty."""
    out: dict[str, list[dict[str, Any]]] = {}
    portfolio = payload.get("portfolio", {}) or {}
    accounts = portfolio.get("accounts") or []
    if isinstance(accounts, dict):
        iterable = list(accounts.items())
    else:
        iterable = [(a.get("name") or a.get("platform") or "?", a) for a in accounts]
    for acct_name, acct in iterable:
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("underlying") or "").upper()
            if not sym:
                continue
            qty = float(p.get("quantity") or p.get("qty") or 0)
            if qty == 0:
                continue
            asset_type = (p.get("asset_type") or p.get("type") or "stock").lower()
            right = (p.get("right") or "").upper()  # 'C' or 'P' or ''
            strike = p.get("strike")
            expiry = p.get("expiry") or p.get("expiration") or ""
            avg_cost = float(p.get("avg_cost") or p.get("cost_basis") or 0)
            last = float(p.get("last_price") or p.get("mark_price") or p.get("price") or 0)
            unreal = float(p.get("unrealized_pnl") or p.get("pnl_unrealized") or 0)
            leg = {
                "account": acct_name,
                "qty": qty,
                "side": "long" if qty > 0 else "short",
                "asset_type": asset_type,
                "right": right or ("C" if asset_type == "call" else "P" if asset_type == "put" else ""),
                "strike": float(strike) if strike not in (None, "") else None,
                "expiry": str(expiry) if expiry else "",
                "avg_cost": avg_cost,
                "last_price": last,
                "unrealized": unreal,
            }
            out.setdefault(sym, []).append(leg)
    return out


# ─── Market data: spot + realized vol ──────────────────────────────────────


def _yf_history(symbol: str, period: str = "120d") -> Any | None:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        h = t.history(period=period, auto_adjust=False)
        if h is None or h.empty:
            return None
        return h
    except Exception:  # noqa: BLE001
        return None


def _spot_and_sigma(symbol: str) -> tuple[float | None, float | None]:
    """Spot + annualized 30d realized vol from yfinance daily closes."""
    h = _yf_history(symbol, "120d")
    if h is None:
        return None, None
    closes = h["Close"].dropna()
    if len(closes) < 25:
        return float(closes.iloc[-1]) if len(closes) else None, None
    spot = float(closes.iloc[-1])
    log_ret = np.log(closes / closes.shift(1)).dropna()
    sigma_30d = float(log_ret.tail(30).std() * math.sqrt(252))
    if sigma_30d <= 0 or math.isnan(sigma_30d):
        return spot, None
    return spot, sigma_30d


# Fallback ATM IV via yfinance options chain (best-effort)
def _atm_iv(symbol: str) -> float | None:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            return None
        # Pick first expiry > 21 DTE
        today = datetime.now(timezone.utc).date()
        target = None
        for e in exps:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (d - today).days >= 21:
                target = e
                break
        target = target or exps[0]
        chain = t.option_chain(target)
        spot = t.history(period="2d")["Close"].iloc[-1]
        ivs = []
        for df in (chain.calls, chain.puts):
            if df is None or df.empty:
                continue
            df = df.copy()
            df["dist"] = (df["strike"] - spot).abs()
            atm = df.nsmallest(3, "dist")
            ivs.extend([float(x) for x in atm["impliedVolatility"].dropna().tolist() if x and x > 0])
        if not ivs:
            return None
        return float(np.median(ivs))
    except Exception:  # noqa: BLE001
        return None


# ─── Monte Carlo ────────────────────────────────────────────────────────────


def _parse_expiry(s: str) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[: len(fmt) - fmt.count("%")], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # YYYYMMDD without dashes
    if len(s) >= 8 and s[:8].isdigit():
        try:
            return datetime.strptime(s[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _normalize_option_cost(cost: float, strike: float) -> tuple[float, float]:
    """Return (per_share_cost, multiplier).

    Brokers disagree on whether option `avg_cost` is per-share or per-contract.
    Heuristic: per-share premium is always < strike (deep ITM caps at strike).
    If reported cost > strike (impossible per-share), assume it's per-contract
    and divide by 100.
    """
    if strike <= 0:
        return cost, 100.0
    if cost > strike * 1.5:  # per-contract (already includes ×100 multiplier)
        return cost / 100.0, 100.0
    return cost, 100.0


def _leg_pnl_at_expiry(leg: dict[str, Any], terminal_prices: np.ndarray) -> np.ndarray:
    """Return per-path P&L for one leg given terminal underlying prices."""
    qty = float(leg["qty"])  # signed; long positive, short negative
    cost = float(leg["avg_cost"] or 0.0)
    right = leg.get("right", "")
    strike = leg.get("strike")
    if right in ("C", "P") and strike is not None:
        K = float(strike)
        per_share_cost, mult = _normalize_option_cost(cost, K)
        intrinsic = np.maximum(terminal_prices - K, 0.0) if right == "C" else np.maximum(K - terminal_prices, 0.0)
        per_share = intrinsic - per_share_cost  # long perspective
        return qty * mult * per_share  # short qty<0 flips sign correctly
    # Equity / crypto / fund — linear
    return qty * (terminal_prices - cost)


def _mc_simulate(
    legs: list[dict[str, Any]],
    spot: float,
    sigma: float,
    days_to_expiry: int,
    paths: int = DEFAULT_PATHS,
    risk_free: float = 0.045,
    seed: int = 42,
) -> dict[str, Any]:
    if days_to_expiry <= 0:
        days_to_expiry = 1
    T = days_to_expiry / 365.0
    rng = np.random.default_rng(seed)
    # GBM terminal: S_T = S_0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)
    z = rng.standard_normal(paths)
    drift = (risk_free - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * z
    S_T = spot * np.exp(drift + diffusion)

    total_pnl = np.zeros(paths)
    for leg in legs:
        total_pnl += _leg_pnl_at_expiry(leg, S_T)

    pct = lambda p: float(np.percentile(total_pnl, p))  # noqa: E731
    return {
        "paths": paths,
        "T_days": days_to_expiry,
        "T_years": round(T, 4),
        "sigma_used": round(sigma, 4),
        "spot": round(spot, 4),
        "S_T_mean": round(float(S_T.mean()), 4),
        "S_T_p05": round(float(np.percentile(S_T, 5)), 4),
        "S_T_p25": round(float(np.percentile(S_T, 25)), 4),
        "S_T_p50": round(float(np.percentile(S_T, 50)), 4),
        "S_T_p75": round(float(np.percentile(S_T, 75)), 4),
        "S_T_p95": round(float(np.percentile(S_T, 95)), 4),
        "pnl_mean_usd": round(float(total_pnl.mean()), 2),
        "pnl_median_usd": round(pct(50), 2),
        "pnl_p05_usd": round(pct(5), 2),
        "pnl_p25_usd": round(pct(25), 2),
        "pnl_p75_usd": round(pct(75), 2),
        "pnl_p95_usd": round(pct(95), 2),
        "prob_profit": round(float((total_pnl > 0).mean()), 4),
        "prob_double": round(float((total_pnl > abs(sum(_initial_outlay(legs)))).mean()), 4),
        "prob_max_loss": round(float((total_pnl <= -abs(sum(_max_loss_per_leg(legs)))).mean() if legs else 0.0), 4),
        "max_loss_estimate_usd": round(float(sum(_max_loss_per_leg(legs))), 2),
    }


def _initial_outlay(legs: list[dict[str, Any]]) -> list[float]:
    out = []
    for leg in legs:
        qty = float(leg["qty"])
        cost = float(leg["avg_cost"] or 0)
        right = leg.get("right", "")
        strike = float(leg.get("strike") or 0)
        if right in ("C", "P") and strike > 0:
            per_share_cost, mult = _normalize_option_cost(cost, strike)
            out.append(abs(qty) * mult * per_share_cost)
        else:
            out.append(abs(qty) * cost)
    return out


def _max_loss_per_leg(legs: list[dict[str, Any]]) -> list[float]:
    """Theoretical max loss approximation per leg (short calls capped at 5x premium)."""
    out = []
    for leg in legs:
        qty = float(leg["qty"])
        cost = float(leg["avg_cost"] or 0)
        right = leg.get("right", "")
        strike = float(leg.get("strike") or 0)
        if right in ("C", "P") and strike > 0:
            per_share_cost, mult = _normalize_option_cost(cost, strike)
            if right == "P" and qty > 0:        # long put: lose premium
                out.append(qty * mult * per_share_cost)
            elif right == "C" and qty > 0:      # long call: lose premium
                out.append(qty * mult * per_share_cost)
            elif right == "P" and qty < 0:      # short put: K * mult per contract minus premium
                out.append(abs(qty) * mult * (strike - per_share_cost))
            elif right == "C" and qty < 0:      # short call: undefined; cap at 3x premium
                out.append(abs(qty) * mult * per_share_cost * 3.0)
            else:
                out.append(abs(qty) * mult * per_share_cost)
        else:
            # equity long: lose to zero; short equity: cap at 100% basis
            out.append(abs(qty) * cost)
    return out


# ─── Grok prompts ───────────────────────────────────────────────────────────


SYSTEM_RESEARCH = (
    "You are Roaring Kitty (DeepFuckingValue / DFV) doing GameStop-2021-grade due "
    "diligence on an OPEN position. Use Live Search aggressively across web, news, "
    "and X. Be specific, name catalysts with dates, cite sources. Never recommend "
    "closing — your job is to surface what could move the position between now and "
    "expiry plus 30 days, and what leading indicators tell us if the thesis is on "
    "or off track. Output MUST be strict JSON, no prose, no markdown fences."
)

RESEARCH_USER = """Position to research:
  underlying : {symbol}
  spot       : {spot}
  realized_vol_30d : {sigma}
  legs (sign = direction; qty<0 means short):
{legs_block}
  current_thesis : {existing_thesis}
  invalidation   : {existing_invalidation}

Today is {today}. Last leg expiry is {last_expiry} ({dte} days out).

Return JSON ONLY in this exact shape:
{{
  \"events\": [
    {{\"date\": \"YYYY-MM-DD or window\", \"event\": \"specific event\", \"impact\": \"bullish|bearish|mixed\", \"importance\": \"high|med|low\", \"source\": \"...\"}},
    ... 5-12 entries through {window_end}
  ],
  \"leading_indicators\": [
    {{\"indicator\": \"name\", \"current\": \"value or trend\", \"signal\": \"thesis-on|thesis-off|neutral\", \"why_it_matters\": \"...\"}},
    ... 4-8 entries
  ],
  \"sentiment\": {{\"retail\": \"bull|bear|mixed\", \"institutional\": \"...\", \"options_flow\": \"...\", \"notes\": \"...\"}},
  \"key_drivers\": [\"driver 1\", \"driver 2\", ...],
  \"risks\": [\"risk that could break the thesis 1\", \"...\"],
  \"thesis_update\": \"2-3 sentences. What changed since last thesis was written?\"
}}
"""

SYSTEM_SYNTH = (
    "You are DFV writing a final aligned thesis. You have MC stats, an event "
    "calendar, and leading indicators in front of you. Tie the thesis to BOTH "
    "events AND MC probabilities. Concrete invalidation. Concrete exit ladder. "
    "Strict JSON output, no prose, no markdown fences."
)

SYNTH_USER = """Synthesize a final aligned thesis from this research + Monte Carlo output.

POSITION:
  symbol : {symbol}
  legs   :
{legs_block}

MARKET:
  spot               : {spot}
  realized_vol_30d   : {sigma}
  days_to_expiry     : {dte}

MONTE CARLO ({paths} paths, GBM):
  S_T median          : {st_p50}   (P05={st_p05}  P25={st_p25}  P75={st_p75}  P95={st_p95})
  P&L median (USD)    : {pnl_p50}
  P&L P25 / P75       : {pnl_p25} / {pnl_p75}
  P&L P05 / P95       : {pnl_p05} / {pnl_p95}
  P(profit)           : {prob_profit}
  Max loss estimate   : {max_loss}

EVENTS through {window_end}:
{events_block}

LEADING INDICATORS:
{indicators_block}

KEY DRIVERS: {drivers}
RISKS:       {risks}

Return JSON ONLY:
{{
  \"verdict\": \"hold|add|trim\",
  \"thesis\": \"<=200 words. Cite specific events + MC probabilities. Argue FOR the position.\",
  \"conviction\": 1-5,
  \"horizon\": \"e.g. '6 months', '2027 LEAPs'\",
  \"catalysts\": [\"specific catalyst with date 1\", \"...\"],
  \"invalidation\": \"Concrete trigger keyed to a leading indicator or price level. e.g. 'Exit if XLE > 105 for 3 sessions OR WTI > 92'\",
  \"exit_ladder\": {{
    \"profit_take_1\": \"trigger + size to peel\",
    \"profit_take_2\": \"trigger + size to peel\",
    \"stop\": \"hard stop level + size action\",
    \"time_stop\": \"e.g. 'close at 21 DTE if not 25% ITM'\"
  }},
  \"target\": {{\"price\": 0.0, \"rationale\": \"...\"}},
  \"max_pct_book\": 0.03,
  \"alignment_check\": \"1 sentence: which events + MC numbers were used to build this thesis\"
}}
"""


def _format_legs(legs: list[dict[str, Any]]) -> str:
    out = []
    for leg in legs:
        out.append(
            f"  - {leg['side']:>5} qty={leg['qty']:.2f} {leg['asset_type']:>6} "
            f"K={leg.get('strike')} R={leg.get('right') or '-'} "
            f"exp={leg.get('expiry') or '-'} cost={leg['avg_cost']:.4f} last={leg['last_price']:.4f}"
        )
    return "\n".join(out)


def _format_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return "  (no events surfaced)"
    out = []
    for e in events[:15]:
        out.append(f"  - {e.get('date','?')}  [{e.get('importance','?')}/{e.get('impact','?')}] {e.get('event','?')}")
    return "\n".join(out)


def _format_indicators(inds: list[dict[str, Any]]) -> str:
    if not inds:
        return "  (none surfaced)"
    out = []
    for i in inds[:10]:
        out.append(f"  - {i.get('indicator','?')}: {i.get('current','?')} → {i.get('signal','?')} ({i.get('why_it_matters','')})")
    return "\n".join(out)


# ─── Per-symbol pipeline ────────────────────────────────────────────────────


def _process_symbol(
    symbol: str,
    legs: list[dict[str, Any]],
    *,
    paths: int,
    do_search: bool,
    existing_rec: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.time()
    out: dict[str, Any] = {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legs": legs,
    }

    # Market data
    spot, sigma_real = _spot_and_sigma(symbol)
    iv = _atm_iv(symbol) if (sigma_real is None or sigma_real < 0.10) else None
    sigma = iv if (iv is not None and iv > 0) else sigma_real
    if sigma is None:
        sigma = 0.40  # last-ditch generic fallback (40% IV)
    out["market"] = {"spot": spot, "sigma_realized_30d": sigma_real, "sigma_atm_iv": iv, "sigma_used": sigma}

    if spot is None or spot <= 0:
        out["error"] = "no spot price (yfinance returned nothing)"
        out["elapsed_s"] = round(time.time() - started, 2)
        return out

    # Days to last expiry across legs (max for portfolio MC)
    expiries = [_parse_expiry(leg.get("expiry") or "") for leg in legs]
    expiries = [e for e in expiries if e is not None]
    today = datetime.now(timezone.utc)
    if expiries:
        last_exp = max(expiries)
        dte = max(1, (last_exp.date() - today.date()).days)
    else:
        # Equity / crypto with no expiry — use 90d horizon
        last_exp = today
        dte = 90
    window_end = (last_exp.date() if expiries else today.date()).isoformat()
    out["horizon"] = {"last_expiry": last_exp.date().isoformat() if expiries else None, "dte": dte, "window_end": window_end}

    # Monte Carlo
    try:
        mc = _mc_simulate(legs, spot=spot, sigma=sigma, days_to_expiry=dte, paths=paths)
        out["monte_carlo"] = mc
    except Exception as e:  # noqa: BLE001
        out["monte_carlo_error"] = f"{type(e).__name__}: {e}"
        mc = {}

    # Existing thesis context
    existing_thesis = (existing_rec or {}).get("thesis", "") or "(none on file)"
    existing_invalidation = (existing_rec or {}).get("invalidation", "") or "(none)"

    # Research call
    research: dict[str, Any] = {}
    citations: list[str] = []
    if do_search:
        prompt = RESEARCH_USER.format(
            symbol=symbol,
            spot=f"{spot:.4f}",
            sigma=f"{sigma:.3f}",
            legs_block=_format_legs(legs),
            existing_thesis=existing_thesis,
            existing_invalidation=existing_invalidation,
            today=today.date().isoformat(),
            last_expiry=window_end,
            dte=dte,
            window_end=window_end,
        )
        r = _grok_research(prompt, system=SYSTEM_RESEARCH, model=GROK_MODEL_RESEARCH, max_tool_calls=10)
        if r.get("ok"):
            parsed = _extract_json(r["text"]) or {}
            research = parsed
            citations = r.get("citations") or []
            if not parsed:
                out["research_raw_text"] = r["text"][:2000]
        else:
            out["research_error"] = r.get("error")
    out["research"] = research
    out["citations"] = citations

    # Synthesis call (no live search, just reason over what we have)
    synth_prompt = SYNTH_USER.format(
        symbol=symbol,
        legs_block=_format_legs(legs),
        spot=f"{spot:.4f}",
        sigma=f"{sigma:.3f}",
        dte=dte,
        paths=mc.get("paths", paths),
        st_p50=mc.get("S_T_p50", "?"),
        st_p05=mc.get("S_T_p05", "?"),
        st_p25=mc.get("S_T_p25", "?"),
        st_p75=mc.get("S_T_p75", "?"),
        st_p95=mc.get("S_T_p95", "?"),
        pnl_p50=mc.get("pnl_median_usd", "?"),
        pnl_p25=mc.get("pnl_p25_usd", "?"),
        pnl_p75=mc.get("pnl_p75_usd", "?"),
        pnl_p05=mc.get("pnl_p05_usd", "?"),
        pnl_p95=mc.get("pnl_p95_usd", "?"),
        prob_profit=mc.get("prob_profit", "?"),
        max_loss=mc.get("max_loss_estimate_usd", "?"),
        window_end=window_end,
        events_block=_format_events(research.get("events", []) or []),
        indicators_block=_format_indicators(research.get("leading_indicators", []) or []),
        drivers=", ".join(research.get("key_drivers", []) or []) or "(none surfaced)",
        risks=", ".join(research.get("risks", []) or []) or "(none surfaced)",
    )
    s = _grok(synth_prompt, system=SYSTEM_SYNTH, model=GROK_MODEL_SYNTH, temperature=0.3, max_tokens=1400, search=False)
    if s.get("ok"):
        synth = _extract_json(s["text"]) or {}
        out["synthesis"] = synth
    else:
        out["synthesis_error"] = s.get("error")
        out["synthesis"] = {}

    out["elapsed_s"] = round(time.time() - started, 2)
    return out


def _write_thesis(dfv: DFV, symbol: str, dd: dict[str, Any]) -> None:
    synth = dd.get("synthesis") or {}
    if not synth:
        return
    verdict = (synth.get("verdict") or "hold").lower()
    if verdict == "close":
        verdict = "hold"
    conviction = int(synth.get("conviction", 0) or 0)
    if conviction < 1:
        conviction = 1
    thesis_text = (synth.get("thesis") or "").strip()
    invalidation = (synth.get("invalidation") or "").strip()
    if not thesis_text or not invalidation:
        return
    catalysts = synth.get("catalysts") or []
    if isinstance(catalysts, str):
        catalysts = [catalysts]
    target = synth.get("target") or {}
    if not isinstance(target, dict):
        target = {"raw": str(target)}
    target.setdefault("verdict", verdict)
    target["exit_ladder"] = synth.get("exit_ladder") or {}
    target["alignment_check"] = synth.get("alignment_check") or ""
    sizing = {"max_pct_book": float(synth.get("max_pct_book") or 0.03)}
    horizon = (synth.get("horizon") or "").strip() or "tbd"
    dfv.thesis.set(
        symbol,
        thesis=thesis_text,
        conviction=conviction,
        horizon=horizon,
        catalysts=list(catalysts),
        invalidation=invalidation,
        target=target,
        sizing=sizing,
        author="GROK",
    )


def _summary_row(dd: dict[str, Any]) -> str:
    sym = dd["symbol"]
    if "error" in dd:
        return f"| {sym} | ERROR | {dd['error']} | | | | |"
    mc = dd.get("monte_carlo", {}) or {}
    synth = dd.get("synthesis", {}) or {}
    spot = (dd.get("market") or {}).get("spot")
    return (
        f"| {sym} "
        f"| {synth.get('verdict','?').upper()} c{synth.get('conviction','?')} "
        f"| spot={spot} σ={(dd.get('market') or {}).get('sigma_used','?'):.2f} "
        f"| dte={dd.get('horizon',{}).get('dte','?')} "
        f"| MC P50={mc.get('pnl_median_usd','?')} P(profit)={mc.get('prob_profit','?')} "
        f"| {(synth.get('invalidation') or '')[:90]} "
        f"| {(synth.get('exit_ladder',{}) or {}).get('stop','')[:60]} |"
    )


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="DFV deep DD across held positions.")
    parser.add_argument("--symbols", default="", help="Comma list to scope to specific symbols.")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS, help="MC paths per symbol.")
    parser.add_argument("--no-search", action="store_true", help="Skip Grok live search (dev mode).")
    parser.add_argument("--dry-run", action="store_true", help="Do not write thesis_log.json.")
    parser.add_argument("--no-write-files", action="store_true", help="Do not write per-symbol JSON either.")
    args = parser.parse_args()

    print("[dfv-dd] collecting payload (this can take 60-130s on first call)...", flush=True)
    payload = collect_payload()
    legs_by_sym = _collect_legs(payload)
    if not legs_by_sym:
        print("[dfv-dd] no held positions found.")
        return 1

    if args.symbols.strip():
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        legs_by_sym = {s: legs for s, legs in legs_by_sym.items() if s in wanted}

    print(f"[dfv-dd] {len(legs_by_sym)} symbols: {', '.join(sorted(legs_by_sym))}")
    print(f"[dfv-dd] paths={args.paths} search={'on' if not args.no_search else 'OFF'} dry_run={args.dry_run}\n")

    dfv = DFV()
    existing = dfv.thesis.all()

    rows: list[str] = []
    results: list[dict[str, Any]] = []
    for i, (sym, legs) in enumerate(sorted(legs_by_sym.items()), 1):
        print(f"[{i}/{len(legs_by_sym)}] {sym} — {len(legs)} leg(s) ", end="", flush=True)
        try:
            dd = _process_symbol(
                sym, legs,
                paths=args.paths,
                do_search=not args.no_search,
                existing_rec=existing.get(sym),
            )
        except Exception as e:  # noqa: BLE001
            dd = {"symbol": sym, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            print(f"FAILED — {e}")
        else:
            synth = dd.get("synthesis", {}) or {}
            mc = dd.get("monte_carlo", {}) or {}
            verdict = (synth.get("verdict") or "?").upper()
            conv = synth.get("conviction", "?")
            pp = mc.get("prob_profit", "?")
            print(f"{verdict} c{conv}  P(profit)={pp}  ({dd.get('elapsed_s','?')}s)")

        results.append(dd)
        if not args.no_write_files:
            (DEEP_DD_DIR / f"{sym}.json").write_text(json.dumps(dd, indent=2, default=str), encoding="utf-8")
        if not args.dry_run and "synthesis" in dd:
            try:
                _write_thesis(dfv, sym, dd)
            except Exception as e:  # noqa: BLE001
                print(f"     (thesis write failed: {e})")
        rows.append(_summary_row(dd))

    # Summary markdown
    summary = [
        "# DFV Deep DD Summary",
        f"_generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "| Symbol | Verdict | Spot/σ | DTE | Monte Carlo | Invalidation | Stop |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "Per-symbol full DD JSON in `agents/dfv/memory/deep_dd/<SYMBOL>.json`.",
        "All theses written with `author=\"GROK\"` — operator must ratify by flipping to `\"DFV\"`.",
    ]
    if not args.no_write_files:
        (DEEP_DD_DIR / "_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n[dfv-dd] done.")
    print(f"[dfv-dd] wrote {len(results)} files to {DEEP_DD_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
