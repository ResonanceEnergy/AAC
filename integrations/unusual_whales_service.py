"""Operational Unusual Whales snapshot service with cache and graceful fallback."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from integrations.unusual_whales_client import DarkPoolTrade, OptionsFlow, UnusualWhalesClient

logger = logging.getLogger("UnusualWhalesService")

# Cool-down (in seconds) applied after persistent auth/transport failures so a
# dead/expired API key doesn't generate one log line per endpoint per minute.
_AUTH_FAILURE_COOLDOWN_SECONDS = 3600  # 1h
_TRANSIENT_FAILURE_COOLDOWN_SECONDS = 300  # 5m

# Per-ticker fetches are 2 calls each (IV + GEX); UW free tier is 120 req/min.
# Cap watchlist size to stay well under the limit even with 5-min refresh.
_MAX_WATCHLIST_TICKERS = 5
_DEFAULT_WATCHLIST: List[str] = ["SPY", "QQQ", "IWM"]
_WATCHLIST_PATH = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"


def _load_watchlist() -> List[str]:
    """Load up to ``_MAX_WATCHLIST_TICKERS`` tickers from ``config/watchlist.yaml``.

    Falls back to ``_DEFAULT_WATCHLIST`` if the file is missing, malformed, or
    PyYAML is not installed. Read-only; never raises.
    """
    if not _WATCHLIST_PATH.exists():
        return list(_DEFAULT_WATCHLIST)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return list(_DEFAULT_WATCHLIST)
    try:
        with _WATCHLIST_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return list(_DEFAULT_WATCHLIST)
    raw = data.get("vol_premium") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return list(_DEFAULT_WATCHLIST)
    tickers = [str(t).strip().upper() for t in raw if isinstance(t, str) and t.strip()]
    return tickers[:_MAX_WATCHLIST_TICKERS] if tickers else list(_DEFAULT_WATCHLIST)


@dataclass
class UnusualWhalesSnapshot:
    status: str
    as_of: str
    source: str = "unusual_whales"
    configured: bool = False
    put_call_ratio: float = 0.0
    market_tone: str = "unknown"
    options_flow_signal_count: int = 0
    total_options_premium: float = 0.0
    dark_pool_trade_count: int = 0
    dark_pool_notional: float = 0.0
    congress_trade_count: int = 0
    top_flow_tickers: List[str] = field(default_factory=list)
    market_tide_latest: Optional[Dict[str, Any]] = None
    market_tide_net_call_premium: float = 0.0
    market_tide_net_put_premium: float = 0.0
    iv_ranks: Dict[str, float] = field(default_factory=dict)
    gex_walls: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnusualWhalesSnapshotService:
    """Cached market-intelligence snapshot service for Unusual Whales."""

    def __init__(self, refresh_ttl_seconds: int = 300):
        self.refresh_ttl = timedelta(seconds=refresh_ttl_seconds)
        self._client = UnusualWhalesClient()
        self._watchlist = _load_watchlist()
        self._snapshot = UnusualWhalesSnapshot(
            status="unconfigured" if not self._client.endpoint.enabled else "stale",
            as_of=datetime.now().isoformat(),
            configured=self._client.endpoint.enabled,
        )
        self._last_refresh: Optional[datetime] = None
        self._lock = asyncio.Lock()
        # Circuit breaker: when set, skip API calls until this time. Set on
        # HTTP 401/403 (auth failure) or repeated transport errors so a dead
        # key doesn't spam the log every refresh cycle.
        self._suppressed_until: Optional[datetime] = None
        self._suppression_reason: Optional[str] = None
        self._suppression_logged: bool = False

    @property
    def client(self) -> UnusualWhalesClient:
        return self._client

    def get_cached_snapshot(self) -> Dict[str, Any]:
        return self._snapshot.to_dict()

    async def get_snapshot(self, force_refresh: bool = False) -> Dict[str, Any]:
        async with self._lock:
            if not self._client.endpoint.enabled:
                self._snapshot = UnusualWhalesSnapshot(
                    status="unconfigured",
                    as_of=datetime.now().isoformat(),
                    configured=False,
                    error="UNUSUAL_WHALES_API_KEY is not configured",
                )
                return self._snapshot.to_dict()

            now = datetime.now()

            # Circuit breaker: short-circuit while suppressed (e.g. dead key).
            if self._suppressed_until is not None and now < self._suppressed_until:
                if not self._suppression_logged:
                    logger.warning(
                        "Unusual Whales suppressed until %s (%s); returning cached snapshot",
                        self._suppressed_until.isoformat(),
                        self._suppression_reason or "unknown",
                    )
                    self._suppression_logged = True
                return self._snapshot.to_dict()

            if (
                not force_refresh
                and self._last_refresh is not None
                and now - self._last_refresh < self.refresh_ttl
            ):
                return self._snapshot.to_dict()

            self._snapshot = await self._fetch_snapshot()
            self._last_refresh = now
            self._maybe_trip_circuit_breaker(self._snapshot)
            return self._snapshot.to_dict()

    def _maybe_trip_circuit_breaker(self, snap: "UnusualWhalesSnapshot") -> None:
        """Inspect the latest snapshot and trip the breaker on auth/transport errors."""
        if snap.status != "error" or not snap.error:
            # Healthy refresh — clear any prior suppression.
            if self._suppressed_until is not None:
                logger.info("Unusual Whales recovered; clearing suppression")
            self._suppressed_until = None
            self._suppression_reason = None
            self._suppression_logged = False
            return

        err = snap.error.lower()
        is_auth = (
            "401" in err
            or "403" in err
            or "unauthorized" in err
            or "authentication_required" in err
            or "invalid" in err and "token" in err
        )
        cooldown = (
            _AUTH_FAILURE_COOLDOWN_SECONDS if is_auth else _TRANSIENT_FAILURE_COOLDOWN_SECONDS
        )
        self._suppressed_until = datetime.now() + timedelta(seconds=cooldown)
        self._suppression_reason = (
            "auth failure (check UNUSUAL_WHALES_API_KEY)" if is_auth else f"transport error: {snap.error[:120]}"
        )
        self._suppression_logged = False  # let next get_snapshot log the cool-down once
        logger.error(
            "Unusual Whales tripped circuit breaker for %ds (%s)",
            cooldown,
            self._suppression_reason,
        )

    async def _fetch_snapshot(self) -> UnusualWhalesSnapshot:
        try:
            summary = await self._client.get_market_flow_summary()
            # Bail out fast on auth failure so the remaining ~13 endpoint calls
            # don't each generate their own HTTP 401 log line every cycle.
            if getattr(self._client, "auth_failed", False):
                return UnusualWhalesSnapshot(
                    status="error",
                    as_of=datetime.now().isoformat(),
                    configured=self._client.endpoint.enabled,
                    error=self._client.last_error or "HTTP 401: authentication_required",
                )
            flow = await self._client.get_flow(limit=25, min_premium=0)
            dark_pool = await self._client.get_dark_pool(limit=25)
            congress = await self._client.get_congress_trades(limit=25)

            tide_series = await self._safe_market_tide()
            iv_ranks, gex_walls = await self._safe_per_ticker_fetch(self._watchlist)

            return self._build_snapshot(
                summary,
                flow,
                dark_pool,
                congress,
                tide_series,
                iv_ranks,
                gex_walls,
            )
        except Exception as exc:
            logger.error("Unusual Whales snapshot refresh failed: %s", exc)
            return UnusualWhalesSnapshot(
                status="error",
                as_of=datetime.now().isoformat(),
                configured=self._client.endpoint.enabled,
                error=str(exc),
            )

    async def _safe_market_tide(self) -> List[Dict[str, Any]]:
        """Fetch market-wide tide series; return [] on transport failure."""
        try:
            return await self._client.get_market_tide()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("market_tide fetch failed: %s", exc)
            return []

    async def _safe_per_ticker_fetch(
        self, tickers: List[str]
    ) -> tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
        """Fetch IV rank + top GEX walls per ticker. Each call isolated."""
        iv_ranks: Dict[str, float] = {}
        gex_walls: Dict[str, List[Dict[str, Any]]] = {}
        for ticker in tickers[:_MAX_WATCHLIST_TICKERS]:
            try:
                iv_payload = await self._client.get_interpolated_iv(ticker)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("interpolated_iv(%s) failed: %s", ticker, exc)
                iv_payload = {}
            rank = self._extract_float(
                iv_payload, ["iv_rank", "iv_percentile", "ivRank", "ivPercentile"]
            )
            if rank > 0:
                iv_ranks[ticker] = rank

            try:
                gex_rows = await self._client.get_spot_gex(ticker)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("spot_gex(%s) failed: %s", ticker, exc)
                gex_rows = []
            walls = self._top_gex_walls(gex_rows, top_n=3)
            if walls:
                gex_walls[ticker] = walls
        return iv_ranks, gex_walls

    @staticmethod
    def _top_gex_walls(
        rows: List[Dict[str, Any]], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the top-N strikes by absolute total gamma magnitude."""
        if not rows:
            return []

        def _abs_gamma(row: Dict[str, Any]) -> float:
            # UW spot-exposures/strike returns gamma as ``call_gamma_oi`` /
            # ``put_gamma_oi`` (charm/vanna also _oi-suffixed). Older endpoints
            # used bare ``gamma``; check both schemas.
            for key in (
                "total_gamma_per_one_pct_move_oi",
                "total_gamma",
                "gamma",
                "net_gamma",
            ):
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    return abs(float(value))
                except (TypeError, ValueError):
                    continue
            call_g = (
                row.get("call_gamma_oi")
                or row.get("call_gamma")
                or 0
            )
            put_g = (
                row.get("put_gamma_oi")
                or row.get("put_gamma")
                or 0
            )
            try:
                return abs(float(call_g)) + abs(float(put_g))
            except (TypeError, ValueError):
                return 0.0

        ranked = sorted(rows, key=_abs_gamma, reverse=True)
        return ranked[:top_n]

    def _build_snapshot(
        self,
        summary: Dict[str, Any],
        flow: List[OptionsFlow],
        dark_pool: List[DarkPoolTrade],
        congress: List[Dict[str, Any]],
        tide_series: List[Dict[str, Any]],
        iv_ranks: Dict[str, float],
        gex_walls: Dict[str, List[Dict[str, Any]]],
    ) -> UnusualWhalesSnapshot:
        top_flow_tickers = sorted(
            {item.ticker for item in flow if item.ticker},
            key=lambda ticker: sum(entry.premium for entry in flow if entry.ticker == ticker),
            reverse=True,
        )[:5]

        put_count = sum(1 for item in flow if item.option_type == "put")
        call_count = sum(1 for item in flow if item.option_type == "call")
        put_call_ratio = put_count / call_count if call_count else float(put_count)

        market_tone = "neutral"
        if put_call_ratio > 1.2:
            market_tone = "bearish"
        elif call_count > put_count and call_count > 0:
            market_tone = "bullish"

        summary_ratio = self._extract_float(summary, ["put_call_ratio", "putCallRatio", "pc_ratio"])
        if summary_ratio > 0:
            put_call_ratio = summary_ratio
            if put_call_ratio > 1.2:
                market_tone = "bearish"
            elif put_call_ratio < 0.9:
                market_tone = "bullish"

        tide_latest: Optional[Dict[str, Any]] = tide_series[-1] if tide_series else None
        tide_call_prem = (
            self._extract_float(
                tide_latest, ["net_call_premium", "netCallPremium", "call_premium"]
            )
            if tide_latest
            else 0.0
        )
        tide_put_prem = (
            self._extract_float(
                tide_latest, ["net_put_premium", "netPutPremium", "put_premium"]
            )
            if tide_latest
            else 0.0
        )

        return UnusualWhalesSnapshot(
            status="healthy",
            as_of=datetime.now().isoformat(),
            configured=True,
            put_call_ratio=put_call_ratio,
            market_tone=market_tone,
            options_flow_signal_count=len(flow),
            total_options_premium=sum(item.premium for item in flow),
            dark_pool_trade_count=len(dark_pool),
            dark_pool_notional=sum(item.notional for item in dark_pool),
            congress_trade_count=len(congress),
            top_flow_tickers=top_flow_tickers,
            market_tide_latest=tide_latest,
            market_tide_net_call_premium=tide_call_prem,
            market_tide_net_put_premium=tide_put_prem,
            iv_ranks=iv_ranks,
            gex_walls=gex_walls,
        )

    @staticmethod
    def _extract_float(data: Dict[str, Any], keys: List[str]) -> float:
        for key in keys:
            value = data.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0


_snapshot_service: Optional[UnusualWhalesSnapshotService] = None


def get_unusual_whales_snapshot_service() -> UnusualWhalesSnapshotService:
    """Get or create the shared Unusual Whales snapshot service."""
    global _snapshot_service
    if _snapshot_service is None:
        _snapshot_service = UnusualWhalesSnapshotService()
    return _snapshot_service
