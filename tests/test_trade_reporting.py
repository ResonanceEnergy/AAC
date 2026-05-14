from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from shared import trade_reporting as tr_mod
from shared.trade_reporting import (
    RegulatoryReport,
    TradeRecord,
    TradeReportingSystem,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def system(tmp_path):
    """Fresh TradeReportingSystem with isolated reports_dir + mocked audit logger."""
    sys_ = TradeReportingSystem()
    sys_.reports_dir = tmp_path / "regulatory"
    sys_.reports_dir.mkdir(parents=True, exist_ok=True)
    sys_.trade_records = []
    sys_.pending_reports = []
    # Replace audit logger with AsyncMock so log_event is awaitable
    sys_.audit_logger = type("_A", (), {"log_event": AsyncMock(return_value=None)})()
    return sys_


def _make_record(
    trade_id="t1",
    timestamp=None,
    symbol="AAPL",
    side="buy",
    quantity="10",
    price="100",
    exchange="NYSE",
    flags=None,
):
    ts = timestamp or datetime(2026, 4, 24, 10, 30, 0)
    qty = Decimal(quantity)
    px = Decimal(price)
    return TradeRecord(
        trade_id=trade_id,
        timestamp=ts,
        strategy_id="strat_a",
        symbol=symbol,
        side=side,
        quantity=qty,
        price=px,
        total_value=qty * px,
        commission=Decimal("1"),
        exchange=exchange,
        order_type="market",
        execution_time=ts,
        account_id="acct1",
        regulatory_flags=flags if flags is not None else [],
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestTradeRecord:
    def test_default_factories_independent(self):
        r1 = TradeRecord(
            trade_id="x",
            timestamp=datetime.now(),
            strategy_id="s",
            symbol="X",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("1"),
            total_value=Decimal("1"),
            commission=Decimal("0"),
            exchange="NYSE",
            order_type="market",
            execution_time=datetime.now(),
            account_id="a",
        )
        r2 = TradeRecord(
            trade_id="y",
            timestamp=datetime.now(),
            strategy_id="s",
            symbol="X",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("1"),
            total_value=Decimal("1"),
            commission=Decimal("0"),
            exchange="NYSE",
            order_type="market",
            execution_time=datetime.now(),
            account_id="a",
        )
        r1.regulatory_flags.append("X")
        r1.metadata["k"] = "v"
        assert r2.regulatory_flags == []
        assert r2.metadata == {}


class TestRegulatoryReport:
    def test_default_status_pending(self):
        r = RegulatoryReport(
            report_id="r1",
            report_type="FINRA_TRF",
            reporting_period="2026-04-23",
            generated_at=datetime.now(),
            total_trades=0,
            total_volume=Decimal("0"),
            report_data=[],
        )
        assert r.submission_status == "pending"


# ---------------------------------------------------------------------------
# record_trade
# ---------------------------------------------------------------------------


class TestRecordTrade:
    @pytest.mark.asyncio
    async def test_records_and_returns_uuid(self, system):
        tid = await system.record_trade(
            strategy_id="s1",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("5"),
            price=Decimal("100"),
            exchange="NYSE",
        )
        assert isinstance(tid, str)
        assert len(tid) >= 32  # UUID-ish
        assert len(system.trade_records) == 1
        assert system.trade_records[0].trade_id == tid

    @pytest.mark.asyncio
    async def test_total_value_computed(self, system):
        tid = await system.record_trade(
            "s", "AAPL", "buy", Decimal("4"), Decimal("25"), "NYSE"
        )
        r = system.trade_records[0]
        assert r.total_value == Decimal("100")

    @pytest.mark.asyncio
    async def test_large_trade_flag(self, system):
        await system.record_trade(
            "s", "AAPL", "buy", Decimal("100"), Decimal("200"), "NYSE"
        )
        r = system.trade_records[0]
        assert "large_trade" in r.regulatory_flags
        assert "equity_trade" in r.regulatory_flags

    @pytest.mark.asyncio
    async def test_small_trade_no_large_flag(self, system):
        await system.record_trade(
            "s", "AAPL", "buy", Decimal("1"), Decimal("10"), "NYSE"
        )
        r = system.trade_records[0]
        assert "large_trade" not in r.regulatory_flags
        assert "equity_trade" in r.regulatory_flags

    @pytest.mark.asyncio
    async def test_crypto_flag(self, system):
        await system.record_trade(
            "s", "BTC", "buy", Decimal("1"), Decimal("50000"), "Binance_Crypto"
        )
        r = system.trade_records[0]
        assert "crypto_trade" in r.regulatory_flags

    @pytest.mark.asyncio
    async def test_audit_logger_called(self, system):
        await system.record_trade("s", "AAPL", "buy", Decimal("1"), Decimal("1"), "NYSE")
        system.audit_logger.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_persists_to_json(self, system):
        await system.record_trade("s", "AAPL", "buy", Decimal("1"), Decimal("1"), "NYSE")
        f = system.reports_dir / "trade_records.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_metadata_kept(self, system):
        meta = {"strategy_note": "test"}
        await system.record_trade(
            "s", "AAPL", "buy", Decimal("1"), Decimal("1"), "NYSE", metadata=meta
        )
        assert system.trade_records[0].metadata == meta


# ---------------------------------------------------------------------------
# Load trade records
# ---------------------------------------------------------------------------


class TestLoadTradeRecords:
    def test_loads_existing_records(self, tmp_path):
        sys_ = TradeReportingSystem()
        sys_.reports_dir = tmp_path / "reg"
        sys_.reports_dir.mkdir(parents=True, exist_ok=True)
        sys_.trade_records = []
        # Pre-write file
        f = sys_.reports_dir / "trade_records.json"
        f.write_text(
            json.dumps(
                [
                    {
                        "trade_id": "t1",
                        "timestamp": "2026-04-24T10:00:00",
                        "strategy_id": "s",
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 5.0,
                        "price": 100.0,
                        "total_value": 500.0,
                        "commission": 1.0,
                        "exchange": "NYSE",
                        "order_type": "market",
                        "execution_time": "2026-04-24T10:00:01",
                        "account_id": "a1",
                    }
                ]
            )
        )
        sys_._load_trade_records()
        assert len(sys_.trade_records) == 1
        assert sys_.trade_records[0].trade_id == "t1"

    def test_load_corrupt_does_not_raise(self, tmp_path):
        sys_ = TradeReportingSystem()
        sys_.reports_dir = tmp_path / "reg"
        sys_.reports_dir.mkdir(parents=True, exist_ok=True)
        sys_.trade_records = []
        (sys_.reports_dir / "trade_records.json").write_text("{not json")
        sys_._load_trade_records()  # logs error, does not raise
        assert sys_.trade_records == []

    def test_load_missing_does_nothing(self, tmp_path):
        sys_ = TradeReportingSystem()
        sys_.reports_dir = tmp_path / "absent"
        sys_.reports_dir.mkdir(parents=True, exist_ok=True)
        sys_.trade_records = []
        sys_._load_trade_records()
        assert sys_.trade_records == []


# ---------------------------------------------------------------------------
# FINRA TRF report
# ---------------------------------------------------------------------------


class TestFinraReport:
    @pytest.mark.asyncio
    async def test_includes_equity_trades(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        rec = _make_record(timestamp=ts, flags=["equity_trade"])
        system.trade_records.append(rec)
        report = await system.generate_finra_trf_report(ts)
        assert report.total_trades == 1
        assert report.report_type == "FINRA_TRF"
        assert report in system.pending_reports

    @pytest.mark.asyncio
    async def test_includes_large_trades(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        rec = _make_record(timestamp=ts, quantity="200", price="100", flags=[])
        # total_value = 20000 ≥ 10000
        system.trade_records.append(rec)
        report = await system.generate_finra_trf_report(ts)
        assert report.total_trades == 1
        assert report.total_volume == Decimal("20000")

    @pytest.mark.asyncio
    async def test_excludes_outside_period(self, system):
        target_day = datetime(2026, 4, 24, 10)
        in_window = _make_record(timestamp=target_day, flags=["equity_trade"])
        out_window = _make_record(
            trade_id="t2",
            timestamp=target_day + timedelta(days=2),
            flags=["equity_trade"],
        )
        system.trade_records.extend([in_window, out_window])
        report = await system.generate_finra_trf_report(target_day)
        assert report.total_trades == 1

    @pytest.mark.asyncio
    async def test_excludes_non_equity_small(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        rec = _make_record(timestamp=ts, exchange="LSE", flags=[], quantity="1")
        system.trade_records.append(rec)
        report = await system.generate_finra_trf_report(ts)
        assert report.total_trades == 0

    @pytest.mark.asyncio
    async def test_default_date_is_yesterday(self, system):
        report = await system.generate_finra_trf_report()
        assert "FINRA_TRF_" in report.report_id

    @pytest.mark.asyncio
    async def test_persists_report_file(self, system):
        report = await system.generate_finra_trf_report()
        f = system.reports_dir / f"{report.report_id}.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["report_type"] == "FINRA_TRF"


# ---------------------------------------------------------------------------
# SEC BD report
# ---------------------------------------------------------------------------


class TestSecReport:
    @pytest.mark.asyncio
    async def test_includes_all_trades_in_period(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        for i in range(3):
            system.trade_records.append(_make_record(trade_id=f"t{i}", timestamp=ts))
        report = await system.generate_sec_bd_report(ts)
        assert report.total_trades == 3
        assert report.report_type == "SEC_BD"

    @pytest.mark.asyncio
    async def test_buy_sell_normalized(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        system.trade_records.append(_make_record(timestamp=ts, side="sell"))
        report = await system.generate_sec_bd_report(ts)
        assert report.report_data[0]["buy_sell"] == "S"

    @pytest.mark.asyncio
    async def test_default_date_yesterday(self, system):
        report = await system.generate_sec_bd_report()
        assert "SEC_BD_" in report.report_id

    @pytest.mark.asyncio
    async def test_volume_aggregated(self, system):
        ts = datetime.now() - timedelta(days=1)
        ts = ts.replace(hour=10)
        system.trade_records.append(_make_record(quantity="2", price="50", timestamp=ts))
        system.trade_records.append(
            _make_record(trade_id="t2", quantity="3", price="50", timestamp=ts)
        )
        report = await system.generate_sec_bd_report(ts)
        assert report.total_volume == Decimal("250")


# ---------------------------------------------------------------------------
# Submit report
# ---------------------------------------------------------------------------


class TestSubmitReport:
    @pytest.mark.asyncio
    async def test_submits_pending(self, system):
        report = await system.generate_sec_bd_report()
        ok = await system.submit_regulatory_report(report.report_id)
        assert ok is True
        assert report.submission_status == "submitted"
        system.audit_logger.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_unknown_returns_false(self, system):
        ok = await system.submit_regulatory_report("missing")
        assert ok is False


# ---------------------------------------------------------------------------
# Trade reconstruction
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_returns_dict_for_known(self, system):
        rec = _make_record(trade_id="abc")
        system.trade_records.append(rec)
        out = system.get_trade_reconstruction("abc")
        assert out is not None
        assert out["trade_id"] == "abc"
        assert out["symbol"] == "AAPL"
        assert "audit_trail" in out
        assert len(out["audit_trail"]) == 3

    def test_returns_none_for_unknown(self, system):
        assert system.get_trade_reconstruction("missing") is None

    def test_audit_trail_events(self, system):
        rec = _make_record(trade_id="abc")
        system.trade_records.append(rec)
        out = system.get_trade_reconstruction("abc")
        events = [e["event"] for e in out["audit_trail"]]
        assert events == ["trade_initiated", "order_submitted", "trade_executed"]


# ---------------------------------------------------------------------------
# Daily reporting
# ---------------------------------------------------------------------------


class TestRunDaily:
    @pytest.mark.asyncio
    async def test_generates_and_submits_both(self, system):
        await system.run_daily_reporting()
        types = {r.report_type for r in system.pending_reports}
        assert "FINRA_TRF" in types
        assert "SEC_BD" in types
        assert all(r.submission_status == "submitted" for r in system.pending_reports)


# ---------------------------------------------------------------------------
# Reporting status
# ---------------------------------------------------------------------------


class TestReportingStatus:
    def test_empty_state(self, system):
        st = system.get_reporting_status()
        assert st["total_trades_recorded"] == 0
        assert st["pending_reports"] == 0
        assert st["today_reports_generated"]["FINRA_TRF"] is False
        assert st["today_reports_generated"]["SEC_BD"] is False
        assert st["last_report_date"] is None

    @pytest.mark.asyncio
    async def test_after_recording_and_reports(self, system):
        await system.record_trade("s", "AAPL", "buy", Decimal("1"), Decimal("1"), "NYSE")
        await system.generate_finra_trf_report(datetime.now())
        st = system.get_reporting_status()
        assert st["total_trades_recorded"] == 1
        assert st["pending_reports"] == 1
        # Generated for today
        assert st["today_reports_generated"]["FINRA_TRF"] is True
        assert st["last_report_date"] is not None


# ---------------------------------------------------------------------------
# Module-level singleton + initialize
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_singleton_exists(self):
        assert isinstance(tr_mod.trade_reporting_system, TradeReportingSystem)

    def test_default_schedules(self):
        s = TradeReportingSystem()
        assert "FINRA_TRF" in s.reporting_schedules
        assert "SEC_BD" in s.reporting_schedules
        assert "CFTC_CPR" in s.reporting_schedules

    @pytest.mark.asyncio
    async def test_initialize_runs(self, tmp_path):
        # Patch the global instance to use isolated dirs
        with patch.object(tr_mod, "trade_reporting_system", new=TradeReportingSystem()) as inst:
            inst.reports_dir = tmp_path / "reg"
            inst.reports_dir.mkdir(parents=True, exist_ok=True)
            inst.trade_records = []
            inst.pending_reports = []
            inst.audit_logger = type("_A", (), {"log_event": AsyncMock()})()
            ok = await tr_mod.initialize_trade_reporting()
            assert ok is True
