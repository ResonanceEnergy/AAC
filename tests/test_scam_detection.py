"""Tests for CryptoIntelligence/scam_detection.py — scam pattern detection."""

import pytest

from CryptoIntelligence.scam_detection import (
    PUMP_DUMP_REGEX,
    RiskLevel,
    ScamAlert,
    ScamDetector,
    ScamType,
    TokenLegitimacyReport,
)


@pytest.fixture
def detector():
    """Detector."""
    return ScamDetector()


class TestScamDetectorInit:
    """TestScamDetectorInit class."""
    def test_creates(self, detector):
        assert isinstance(detector, ScamDetector)
        assert detector.alert_history == []


class TestScanToken:
    """TestScanToken class."""
    def test_clean_token_no_alerts(self, detector):
        data = {"address": "0xclean", "chain": "ethereum"}
        alerts = detector.scan_token(data)
        assert alerts == []

    def test_known_scam_address(self, detector):
        detector._known_scam.add("0xbad")
        alerts = detector.scan_token({"address": "0xBAD", "chain": "ethereum"})
        assert len(alerts) == 1
        assert alerts[0].risk_level == RiskLevel.CRITICAL

    def test_known_safe_skipped(self, detector):
        detector._known_safe.add("0xgood")
        alerts = detector.scan_token({"address": "0xgood", "chain": "ethereum"})
        assert alerts == []

    def test_rug_pull_detection(self, detector):
        data = {
            "address": "0xrug",
            "chain": "bsc",
            "contract_code": "liquidity_removed max_supply_modifiable mint_function_unrestricted",
            "deployer_history": {"renounced": True, "reclaimed_after": True},
        }
        alerts = detector.scan_token(data)
        rug_alerts = [a for a in alerts if a.scam_type == ScamType.RUG_PULL]
        assert len(rug_alerts) >= 1
        assert rug_alerts[0].risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_honeypot_detection(self, detector):
        data = {
            "address": "0xhoney",
            "chain": "ethereum",
            "trading_history": {"buy_count": 100, "sell_count": 0},
        }
        alerts = detector.scan_token(data)
        honeypot = [a for a in alerts if a.scam_type == ScamType.HONEYPOT]
        assert len(honeypot) >= 1
        assert honeypot[0].confidence >= 0.9

    def test_pump_dump_social(self, detector):
        data = {
            "address": "0xpump",
            "chain": "ethereum",
            "social_text": "guaranteed 100x gem next shiba presale ending soon",
        }
        alerts = detector.scan_token(data)
        pd = [a for a in alerts if a.scam_type == ScamType.PUMP_AND_DUMP]
        assert len(pd) >= 1

    def test_alerts_archived(self, detector):
        data = {
            "address": "0xhoney",
            "chain": "ethereum",
            "trading_history": {"buy_count": 200, "sell_count": 0},
        }
        detector.scan_token(data)
        assert len(detector.alert_history) > 0


class TestLegitimacyReport:
    """TestLegitimacyReport class."""
    def test_clean_token_high_score(self, detector):
        data = {
            "address": "0xgood",
            "chain": "ethereum",
            "liquidity": {"usd": 1_000_000},
            "holders": {"count": 5000},
        }
        report = detector.get_legitimacy_report("0xgood", "ethereum", data)
        assert isinstance(report, TokenLegitimacyReport)
        assert report.is_safe
        assert report.overall_score >= 70

    def test_scam_token_low_score(self, detector):
        data = {
            "address": "0xscam",
            "chain": "bsc",
            "trading_history": {"buy_count": 100, "sell_count": 0},
            "contract_code": "liquidity_removed mint_function_unrestricted max_supply_modifiable",
            "deployer_history": {"renounced": True, "reclaimed_after": True},
        }
        report = detector.get_legitimacy_report("0xscam", "bsc", data)
        assert not report.is_safe
        assert report.overall_score < 50
        assert len(report.checks_failed) > 0

    def test_skeleton_report_no_data(self, detector):
        report = detector.get_legitimacy_report("0xunknown", "ethereum")
        assert isinstance(report, TokenLegitimacyReport)
        assert report.token_address == "0xunknown"


class TestScamAlertProperties:
    """TestScamAlertProperties class."""
    def test_actionable_high_confidence(self):
        alert = ScamAlert(
            scam_type=ScamType.RUG_PULL,
            risk_level=RiskLevel.CRITICAL,
            description="test",
            confidence=0.9,
        )
        assert alert.is_actionable is True

    def test_not_actionable_low_risk(self):
        alert = ScamAlert(
            scam_type=ScamType.WASH_TRADING,
            risk_level=RiskLevel.LOW,
            description="minor",
            confidence=0.8,
        )
        assert alert.is_actionable is False

    def test_not_actionable_low_confidence(self):
        alert = ScamAlert(
            scam_type=ScamType.HONEYPOT,
            risk_level=RiskLevel.CRITICAL,
            description="test",
            confidence=0.3,
        )
        assert alert.is_actionable is False


class TestPumpDumpRegex:
    """TestPumpDumpRegex class."""

    @pytest.mark.parametrize("text", [
        "guaranteed 100x moon shot",
        "next shiba inu killer",
        "presale ending soon dont miss out",
        "whale bought 500k",
    ])
    def test_matches(self, text):
        assert any(r.search(text) for r in PUMP_DUMP_REGEX)

    @pytest.mark.parametrize("text", [
        "quarterly earnings report",
        "Federal Reserve rate decision",
        "Bitcoin ETF approved",
    ])
    def test_no_match(self, text):
        assert not any(r.search(text) for r in PUMP_DUMP_REGEX)
