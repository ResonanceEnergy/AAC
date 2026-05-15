#!/usr/bin/env python3
"""
CFTC Commitments of Traders (COT) Client
=========================================
Free weekly large-trader positioning data from cftc.gov.

Two report families used:
  1. Traders in Financial Futures (TFF) — equity index, currencies, rates
     URL: https://www.cftc.gov/dea/newcot/FinFutWk.txt   (current year, fixed-width)
  2. Disaggregated (DFW)               — commodities (gold, oil, ags)
     URL: https://www.cftc.gov/dea/newcot/c_year.txt     (placeholder)

Schedule: Tuesday data, released Friday 3:30pm ET.

Markets we care about (CFTC contract codes):
  ES  S&P 500 e-mini       — code 13874A
  NQ  Nasdaq 100 e-mini    — code 209742
  RTY Russell 2000 e-mini  — code 239742
  YM  Dow e-mini           — code 12460+
  VX  VIX futures          — code 1170E1
  GC  Gold                 — code 088691
  CL  Crude oil WTI        — code 06765A

Trader categories (Financial Futures):
  Dealer Intermediary     — sell-side / banks
  Asset Manager / Inst    — pension, mutual funds, insurance
  Leveraged Funds         — hedge funds (smart money proxy)
  Other Reportables       — corporates
  Non-Reportable          — small specs (retail proxy)

Smart-money signal: Leveraged Funds NET positioning extreme = contrarian setup.

Usage:
    from integrations.cftc_cot_client import CFTCCotClient
    c = CFTCCotClient()
    snap = c.get_latest("ES")              # latest report for S&P 500 e-mini
    signal = c.get_extreme_signal("ES", lookback_weeks=52)
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# Display-name -> regex on the "Market_and_Exchange_Names" column in the CSV.
# CFTC market names use the official exchange-listed contract name.
MARKET_PATTERNS: Dict[str, str] = {
    "ES":  r"E-MINI\s+S&P\s+500.*CHICAGO MERCANTILE EXCHANGE",
    "NQ":  r"NASDAQ-100.*E-MINI.*CHICAGO MERCANTILE EXCHANGE",
    "RTY": r"E-MINI\s+RUSSELL\s+2000.*CHICAGO MERCANTILE EXCHANGE",
    "YM":  r"DJIA\s+x\s+\$5.*CHICAGO BOARD OF TRADE",
    "VX":  r"VIX\s+FUTURES",
}

# Endpoint: CFTC publishes a yearly CSV inside a zip and a fixed-width "txt".
# The CSV inside the zip is the easiest to parse. We use the disaggregated CSV
# for commodities and the TFF CSV for financial futures.
TFF_CSV_URL = "https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip"
TFF_DETAIL_TXT = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

# User-agent: cftc.gov 403s the default Python urllib UA.
USER_AGENT = "AAC-CFTC-Client/1.0 (https://github.com/ResonanceEnergy/AAC)"

DEFAULT_TIMEOUT = 30


@dataclass
class CotReport:
    """A single weekly COT row for one market."""

    market: str
    report_date: str
    dealer_long: int = 0
    dealer_short: int = 0
    asset_mgr_long: int = 0
    asset_mgr_short: int = 0
    leveraged_long: int = 0
    leveraged_short: int = 0
    other_long: int = 0
    other_short: int = 0
    nonreportable_long: int = 0
    nonreportable_short: int = 0
    open_interest: int = 0
    source_url: str = ""

    @property
    def leveraged_net(self) -> int:
        return self.leveraged_long - self.leveraged_short

    @property
    def asset_mgr_net(self) -> int:
        return self.asset_mgr_long - self.asset_mgr_short

    @property
    def dealer_net(self) -> int:
        return self.dealer_long - self.dealer_short

    @property
    def nonreportable_net(self) -> int:
        return self.nonreportable_long - self.nonreportable_short

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["leveraged_net"] = self.leveraged_net
        d["asset_mgr_net"] = self.asset_mgr_net
        d["dealer_net"] = self.dealer_net
        d["nonreportable_net"] = self.nonreportable_net
        return d


@dataclass
class CotExtremeSignal:
    """Z-score-based contrarian signal on Leveraged Funds positioning."""

    market: str
    latest_net: int
    mean_net: float
    std_net: float
    z_score: float
    signal: str   # "extreme_long", "extreme_short", "neutral"
    lookback_weeks: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Column name -> CSV column name in the TFF combined CSV (post-2017 schema).
# Names are case-sensitive and match the CFTC header exactly.
_TFF_COLUMNS = {
    "market": "Market_and_Exchange_Names",
    "report_date": "Report_Date_as_YYYY-MM-DD",
    "open_interest": "Open_Interest_All",
    "dealer_long": "Dealer_Positions_Long_All",
    "dealer_short": "Dealer_Positions_Short_All",
    "asset_mgr_long": "Asset_Mgr_Positions_Long_All",
    "asset_mgr_short": "Asset_Mgr_Positions_Short_All",
    "leveraged_long": "Lev_Money_Positions_Long_All",
    "leveraged_short": "Lev_Money_Positions_Short_All",
    "other_long": "Other_Rept_Positions_Long_All",
    "other_short": "Other_Rept_Positions_Short_All",
    "nonreportable_long": "NonRept_Positions_Long_All",
    "nonreportable_short": "NonRept_Positions_Short_All",
}


def _safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(str(value).replace(",", "")))
    except (ValueError, TypeError):
        return 0


class CFTCCotClient:
    """Free, no-auth client for CFTC weekly COT data.

    Caches the parsed yearly CSV for the lifetime of the instance to avoid
    re-downloading on every call. Use ``refresh()`` to invalidate.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._cache_year: Optional[int] = None
        self._cache_rows: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    def _fetch_zip_csv(self, year: int) -> List[Dict[str, str]]:
        """Download the TFF yearly zip, extract the CSV, return rows as dicts."""
        import zipfile

        url = TFF_CSV_URL.format(year=year)
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - https URL hardcoded
                blob = resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning("CFTC zip fetch failed for %s: %s", year, exc)
            return []

        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                csv_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".txt") or n.lower().endswith(".csv")),
                    None,
                )
                if csv_name is None:
                    logger.warning("CFTC zip for %s contains no csv/txt", year)
                    return []
                with zf.open(csv_name) as fh:
                    text = fh.read().decode("utf-8", errors="replace")
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
            logger.warning("CFTC zip parse failed for %s: %s", year, exc)
            return []

        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    def _ensure_cached(self, year: Optional[int] = None) -> None:
        target = year or datetime.utcnow().year
        if self._cache_year == target and self._cache_rows:
            return
        rows = self._fetch_zip_csv(target)
        if not rows and target == datetime.utcnow().year:
            # Early in the year before any reports — fall back to previous year
            rows = self._fetch_zip_csv(target - 1)
            target = target - 1
        self._cache_year = target
        self._cache_rows = rows

    def refresh(self) -> None:
        self._cache_year = None
        self._cache_rows = []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _row_to_report(self, row: Dict[str, str]) -> Optional[CotReport]:
        market_name = row.get(_TFF_COLUMNS["market"], "")
        if not market_name:
            return None

        # Resolve market display key (ES/NQ/...) from the long market name.
        display = ""
        for key, pattern in MARKET_PATTERNS.items():
            if re.search(pattern, market_name, flags=re.IGNORECASE):
                display = key
                break
        if not display:
            return None

        return CotReport(
            market=display,
            report_date=row.get(_TFF_COLUMNS["report_date"], ""),
            open_interest=_safe_int(row.get(_TFF_COLUMNS["open_interest"])),
            dealer_long=_safe_int(row.get(_TFF_COLUMNS["dealer_long"])),
            dealer_short=_safe_int(row.get(_TFF_COLUMNS["dealer_short"])),
            asset_mgr_long=_safe_int(row.get(_TFF_COLUMNS["asset_mgr_long"])),
            asset_mgr_short=_safe_int(row.get(_TFF_COLUMNS["asset_mgr_short"])),
            leveraged_long=_safe_int(row.get(_TFF_COLUMNS["leveraged_long"])),
            leveraged_short=_safe_int(row.get(_TFF_COLUMNS["leveraged_short"])),
            other_long=_safe_int(row.get(_TFF_COLUMNS["other_long"])),
            other_short=_safe_int(row.get(_TFF_COLUMNS["other_short"])),
            nonreportable_long=_safe_int(row.get(_TFF_COLUMNS["nonreportable_long"])),
            nonreportable_short=_safe_int(row.get(_TFF_COLUMNS["nonreportable_short"])),
            source_url=TFF_CSV_URL.format(year=self._cache_year),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_history(self, market: str) -> List[CotReport]:
        """Return all weekly reports for the cached year, oldest first."""
        if market not in MARKET_PATTERNS:
            raise ValueError(
                f"Unknown market '{market}'; supported: {sorted(MARKET_PATTERNS)}"
            )
        self._ensure_cached()
        out: List[CotReport] = []
        for row in self._cache_rows:
            report = self._row_to_report(row)
            if report and report.market == market:
                out.append(report)
        out.sort(key=lambda r: r.report_date)
        return out

    def get_latest(self, market: str) -> Optional[CotReport]:
        history = self.get_history(market)
        return history[-1] if history else None

    def get_extreme_signal(
        self, market: str, lookback_weeks: int = 52
    ) -> Optional[CotExtremeSignal]:
        """Z-score Leveraged Funds net position vs trailing window.

        |z| > 2 = extreme positioning (contrarian setup).
        """
        history = self.get_history(market)
        if not history:
            return None

        nets = [r.leveraged_net for r in history[-lookback_weeks:]]
        if len(nets) < 4:
            return CotExtremeSignal(
                market=market,
                latest_net=nets[-1] if nets else 0,
                mean_net=0.0,
                std_net=0.0,
                z_score=0.0,
                signal="neutral",
                lookback_weeks=len(nets),
                notes=[f"Insufficient samples ({len(nets)} < 4)"],
            )

        mean = sum(nets) / len(nets)
        variance = sum((x - mean) ** 2 for x in nets) / max(len(nets) - 1, 1)
        std = variance ** 0.5
        latest = nets[-1]
        z = (latest - mean) / std if std > 0 else 0.0

        if z > 2:
            sig = "extreme_long"
        elif z < -2:
            sig = "extreme_short"
        else:
            sig = "neutral"

        return CotExtremeSignal(
            market=market,
            latest_net=latest,
            mean_net=mean,
            std_net=std,
            z_score=z,
            signal=sig,
            lookback_weeks=len(nets),
        )


__all__ = [
    "CFTCCotClient",
    "CotReport",
    "CotExtremeSignal",
    "MARKET_PATTERNS",
]
