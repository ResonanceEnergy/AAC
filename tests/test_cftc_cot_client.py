"""Tests for the CFTC COT client (Financial Futures TFF report)."""
from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

import pytest

from integrations.cftc_cot_client import (
    CFTCCotClient,
    CotExtremeSignal,
    CotReport,
    _safe_int,
)

# Minimal CSV matching the 2017+ TFF schema. Two markets, two weeks each.
_CSV_HEADER = (
    "Market_and_Exchange_Names,Report_Date_as_YYYY-MM-DD,Open_Interest_All,"
    "Dealer_Positions_Long_All,Dealer_Positions_Short_All,"
    "Asset_Mgr_Positions_Long_All,Asset_Mgr_Positions_Short_All,"
    "Lev_Money_Positions_Long_All,Lev_Money_Positions_Short_All,"
    "Other_Rept_Positions_Long_All,Other_Rept_Positions_Short_All,"
    "NonRept_Positions_Long_All,NonRept_Positions_Short_All"
)

_CSV_ROWS = [
    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE,2026-03-04,2500000,250000,300000,800000,500000,400000,420000,100000,90000,50000,40000',
    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE,2026-03-11,2510000,255000,305000,810000,505000,500000,300000,102000,92000,52000,42000',
    'NASDAQ-100 E-MINI - CHICAGO MERCANTILE EXCHANGE,2026-03-04,500000,40000,50000,150000,80000,90000,70000,15000,20000,8000,7000',
]


def _make_zip_blob() -> bytes:
    csv_text = _CSV_HEADER + "\n" + "\n".join(_CSV_ROWS) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FinFutWk.txt", csv_text)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def read(self) -> bytes:
        return self._blob

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class TestSafeInt:
    def test_handles_commas(self) -> None:
        assert _safe_int("1,234,567") == 1234567

    def test_handles_blank(self) -> None:
        assert _safe_int("") == 0
        assert _safe_int(None) == 0

    def test_handles_garbage(self) -> None:
        assert _safe_int("abc") == 0


class TestCFTCClient:
    def _patched_client(self) -> CFTCCotClient:
        client = CFTCCotClient()
        blob = _make_zip_blob()

        def fake_urlopen(_req, timeout=30):  # noqa: ARG001
            return _FakeResponse(blob)

        # Pre-populate cache directly to avoid network
        rows_with_year = client._fetch_zip_csv  # we'll patch the urlopen used inside
        with patch("integrations.cftc_cot_client.urlopen", fake_urlopen):
            client._ensure_cached(year=2026)
        return client

    def test_get_history_filters_by_market(self) -> None:
        client = self._patched_client()
        es_history = client.get_history("ES")
        assert len(es_history) == 2
        assert all(isinstance(r, CotReport) for r in es_history)
        assert all(r.market == "ES" for r in es_history)
        # Sorted oldest first
        assert es_history[0].report_date == "2026-03-04"
        assert es_history[1].report_date == "2026-03-11"

    def test_get_latest_returns_most_recent(self) -> None:
        client = self._patched_client()
        latest = client.get_latest("ES")
        assert latest is not None
        assert latest.report_date == "2026-03-11"
        assert latest.leveraged_long == 500000
        assert latest.leveraged_short == 300000
        assert latest.leveraged_net == 200000

    def test_unknown_market_raises(self) -> None:
        client = self._patched_client()
        with pytest.raises(ValueError):
            client.get_history("XYZ")

    def test_extreme_signal_neutral_on_short_series(self) -> None:
        client = self._patched_client()
        sig = client.get_extreme_signal("ES", lookback_weeks=52)
        assert isinstance(sig, CotExtremeSignal)
        # Only 2 samples → "Insufficient samples" branch
        assert sig.signal == "neutral"
        assert any("Insufficient" in n for n in sig.notes)

    def test_nq_partially_present(self) -> None:
        client = self._patched_client()
        nq = client.get_history("NQ")
        assert len(nq) == 1
        assert nq[0].leveraged_net == 90000 - 70000
