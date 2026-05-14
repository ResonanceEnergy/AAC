from __future__ import annotations
"""strategies/vol_premium_signals.py — Volatility Premium Signal Generator

Second independent signal source for AAC (Sprint 6).

Thesis: Implied volatility of near-term options on broad-market ETFs
systematically exceeds realised historical volatility — the variance risk
premium (VRP).  When IV/HV > threshold the market is pricing in excess
fear; buying puts has a structural edge because panic pricing is elevated.

Algorithm per ticker
────────────────────
1. Fetch 35 days of daily closes (yfinance) → compute 30-day realised HV
2. Fetch front-month ATM put option → extract implied vol
3. If IV/HV > threshold  → emit LONG_PUT TradeSignal
   Else if IV unavailable and HV > 0.25 annualised → emit lower-confidence signal
4. Size is proportional to vol-premium magnitude (3–8 % account)

Output: List[TradeSignal] ordered by confidence descending.
Never raises; returns [] on any failure.
"""

import logging
from dataclasses import dataclass

import numpy as np

from shared.signal import AssetClass, Direction, TradeSignal

_log = logging.getLogger(__name__)

# ── Universe ────────────────────────────────────────────────────────────────
# Loaded from config/watchlist.yaml at call time (cached in-process).
# Fallback if YAML missing: ["SPY", "QQQ", "IWM", "HYG", "JNK"]
def _get_universe() -> list[str]:
    try:
        from shared.watchlist import get_vol_premium_tickers  # noqa: PLC0415
        return get_vol_premium_tickers()
    except Exception:
        return ["SPY", "QQQ", "IWM", "HYG", "JNK"]

# ── Signal thresholds ───────────────────────────────────────────────────────
_IV_HV_THRESHOLD = 1.20   # IV must be ≥20 % above HV
_IV_HV_MAX = 1.80         # ratio above which confidence is saturated
_HV_SOLO_THRESHOLD = 0.25 # use HV-only signal when IV unavailable and HV is this high

# ── Sizing ──────────────────────────────────────────────────────────────────
_BASE_SIZE = 0.03    # 3 % account per signal
_MAX_SIZE = 0.08     # 8 % account hard cap


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VolPremiumReading:
    """IV/HV comparison for a single ticker."""

    ticker: str
    realized_hv: float          # annualised 30-day HV (e.g. 0.18 = 18 %)
    implied_vol: float          # ATM front-month IV; 0.0 if unavailable
    iv_hv_ratio: float          # IV / HV; 0.0 if IV unavailable
    spot: float
    option_available: bool = False


# ── Internal helpers ─────────────────────────────────────────────────────────

def _compute_realized_hv(closes: list[float]) -> float:
    """Annualised 30-day realised HV from a list of daily closes."""
    if len(closes) < 5:
        return 0.0
    arr = np.array(closes, dtype=float)
    log_ret = np.diff(np.log(arr))
    return float(np.std(log_ret, ddof=1) * np.sqrt(252))


def _get_atm_iv(ticker: str, spot: float) -> float:
    """Fetch front-month ATM put IV via yfinance options chain.

    Returns 0.0 if the chain is unavailable (free tier may throttle).
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return 0.0

        chain = t.option_chain(exps[0])
        puts = chain.puts
        if puts.empty:
            return 0.0

        puts = puts.copy()
        puts["_dist"] = abs(puts["strike"] - spot)
        row = puts.loc[puts["_dist"].idxmin()]
        iv = float(row.get("impliedVolatility", 0.0))
        return iv if iv > 0 else 0.0
    except Exception as exc:
        _log.debug("IV fetch failed for %s: %s", ticker, exc)
        return 0.0


def _get_closes_and_spot(ticker: str) -> tuple[list[float], float]:
    """Fetch 35 days of daily closes + latest spot via yfinance."""
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker(ticker).history(period="35d")
        if hist.empty:
            return [], 0.0
        closes = list(hist["Close"].dropna().values)
        spot = float(closes[-1]) if closes else 0.0
        return closes, spot
    except Exception as exc:
        _log.debug("Price fetch failed for %s: %s", ticker, exc)
        return [], 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def get_vol_premium_readings(
    tickers: list[str] | None = None,
    fetch_iv: bool = True,
) -> list[VolPremiumReading]:
    """Compute IV/HV readings for the given universe.

    Args:
        tickers:   Override default universe.
        fetch_iv:  If False, skip the options-chain fetch (fast path for tests).

    Returns:
        List of VolPremiumReading, one per ticker that returned valid data.
    """
    universe = tickers or _get_universe()
    readings: list[VolPremiumReading] = []

    for ticker in universe:
        closes, spot = _get_closes_and_spot(ticker)
        hv = _compute_realized_hv(closes)
        if hv <= 0.0 or spot <= 0.0:
            _log.debug("No valid price data for %s", ticker)
            continue

        iv = _get_atm_iv(ticker, spot) if fetch_iv else 0.0
        ratio = (iv / hv) if (iv > 0 and hv > 0) else 0.0

        readings.append(VolPremiumReading(
            ticker=ticker,
            realized_hv=round(hv, 4),
            implied_vol=round(iv, 4),
            iv_hv_ratio=round(ratio, 3),
            spot=round(spot, 2),
            option_available=iv > 0,
        ))

    return readings


def generate_vol_premium_signals(
    tickers: list[str] | None = None,
    iv_hv_threshold: float = _IV_HV_THRESHOLD,
    fetch_iv: bool = True,
) -> list[TradeSignal]:
    """Second signal source: emit LONG_PUT signals where IV > HV by threshold.

    Independent of the War Room macro-regime model.  Uses only yfinance
    (free tier, no API key required).

    Args:
        tickers:          Override default universe.
        iv_hv_threshold:  Minimum IV/HV ratio to emit a full signal.
        fetch_iv:         Fetch live options IV (False uses HV-only proxy).

    Returns:
        List[TradeSignal] ordered by confidence descending.
        Returns [] on any failure without raising.
    """
    try:
        readings = get_vol_premium_readings(tickers=tickers, fetch_iv=fetch_iv)
    except Exception as exc:
        _log.error("Vol premium readings failed: %s", exc)
        return []

    signals: list[TradeSignal] = []

    for r in readings:
        if not r.option_available:
            # HV-only fallback: emit lower-confidence signal if panic HV detected
            if r.realized_hv < _HV_SOLO_THRESHOLD:
                continue
            confidence = round(min(0.60, 0.30 + r.realized_hv), 3)
            ratio_note = f"HV-only={r.realized_hv:.2%}"
        else:
            if r.iv_hv_ratio < iv_hv_threshold:
                continue
            span = _IV_HV_MAX - iv_hv_threshold
            conf_raw = (r.iv_hv_ratio - iv_hv_threshold) / span if span > 0 else 0.0
            confidence = round(min(0.90, 0.50 + conf_raw * 0.40), 3)
            ratio_note = f"IV/HV={r.iv_hv_ratio:.2f}"

        size = round(_BASE_SIZE + (confidence - 0.50) * 0.10, 3)
        size = max(0.01, min(_MAX_SIZE, size))

        # Stop = underlying rises 25 % above entry (put becomes worthless)
        stop = round(r.spot * 1.25, 2)

        signals.append(TradeSignal(
            ticker=r.ticker,
            direction=Direction.LONG_PUT,
            confidence=confidence,
            entry=r.spot,
            stop=stop,
            target=0.0,
            size=size,
            strategy="vol_premium",
            regime="VOL_PREMIUM",
            asset_class=AssetClass.OPTION,
            notes=f"{ratio_note} | HV={r.realized_hv:.2%} | spot={r.spot}",
        ))

    signals.sort(key=lambda s: s.confidence, reverse=True)
    _log.info(
        "Vol premium signals: %d emitted from %d readings",
        len(signals), len(readings),
    )
    return signals
