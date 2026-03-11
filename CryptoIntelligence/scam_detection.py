"""
Scam Detection Module — CryptoIntelligence Department
=====================================================

Implements FrankenClaw-style pattern matching for crypto scams:
- Rug pull detection (liquidity drain patterns)
- Honeypot contract identification
- Pump-and-dump signal detection
- Social engineering / phishing markers
- Fake audit report detection

Part of BARREN WUFFET's defensive intelligence layer.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class ScamType(Enum):
    """Classification of crypto scam types."""
    RUG_PULL = "rug_pull"
    HONEYPOT = "honeypot"
    PUMP_AND_DUMP = "pump_and_dump"
    PHISHING = "phishing"
    FAKE_AUDIT = "fake_audit"
    PONZI = "ponzi"
    EXIT_SCAM = "exit_scam"
    WASH_TRADING = "wash_trading"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk severity levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScamAlert:
    """Individual scam detection alert."""
    scam_type: ScamType
    risk_level: RiskLevel
    description: str
    confidence: float  # 0.0 to 1.0
    indicators: List[str] = field(default_factory=list)
    token_address: Optional[str] = None
    chain: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_actionable(self) -> bool:
        """Whether this alert requires immediate attention."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and self.confidence >= 0.7


@dataclass
class TokenLegitimacyReport:
    """Full legitimacy assessment for a token/contract."""
    token_address: str
    chain: str
    risk_level: RiskLevel
    overall_score: float  # 0-100, higher = safer
    alerts: List[ScamAlert] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_safe(self) -> bool:
        return self.overall_score >= 70 and self.risk_level in (RiskLevel.SAFE, RiskLevel.LOW)


# ═══════════════════════════════════════════════════════════════════════════
# SCAM PATTERNS — FrankenClaw Heuristics
# ═══════════════════════════════════════════════════════════════════════════

# Red-flag contract function selectors (EVM)
HONEYPOT_SELECTORS = {
    "approve_with_hidden_fee": "0x095ea7b3",  # approve() with hidden tax
    "transfer_blacklist": "0xa9059cbb",       # transfer() with blacklist check
    "swap_max_tax": "0x38ed1739",             # swap with >50% tax
}

# Common rug pull indicators
RUG_PULL_PATTERNS = [
    "liquidity_removed",
    "ownership_renounced_then_reclaimed",
    "mint_function_unrestricted",
    "max_supply_modifiable",
    "proxy_contract_upgradeable",
    "hidden_mint_in_transfer",
    "anti_whale_disabled",
    "trading_cooldown_one_sided",
]

# Pump and dump social signals
PUMP_DUMP_SOCIAL_SIGNALS = [
    r"guaranteed\s+(1000|100|10)x",
    r"to\s+the\s+moon",
    r"100x\s+gem",
    r"next\s+(shiba|doge|pepe)",
    r"presale\s+ending\s+soon",
    r"limited\s+spots",
    r"don.?t\s+miss\s+out",
    r"aping\s+in",
    r"whale\s+bought",
    r"dev\s+is\s+based",
]

# Compiled regex for social signals
PUMP_DUMP_REGEX = [re.compile(p, re.IGNORECASE) for p in PUMP_DUMP_SOCIAL_SIGNALS]


# ═══════════════════════════════════════════════════════════════════════════
# SCAM DETECTOR
# ═══════════════════════════════════════════════════════════════════════════


class ScamDetector:
    """
    Multi-layered scam detection engine.

    Usage:
        detector = ScamDetector()
        alerts = detector.scan_token(token_data)
        report = detector.get_legitimacy_report("0xABC...", "ethereum")
    """

    def __init__(self) -> None:
        self.alert_history: List[ScamAlert] = []
        self._known_safe: set = set()
        self._known_scam: set = set()
        logger.info("ScamDetector initialized — FrankenClaw patterns loaded")

    # ── Core Scanning ──────────────────────────────────────────────────

    def scan_token(self, token_data: Dict[str, Any]) -> List[ScamAlert]:
        """
        Run all scam detection heuristics against token data.

        Args:
            token_data: Dict with keys like 'address', 'chain', 'name',
                        'liquidity', 'holders', 'contract_code', 'social_text',
                        'trading_history', 'deployer_history'

        Returns:
            List of ScamAlerts found (may be empty if clean).
        """
        alerts: List[ScamAlert] = []

        address = token_data.get("address", "").lower()
        chain = token_data.get("chain", "unknown")

        # Skip known-safe tokens
        if address in self._known_safe:
            return alerts

        # Check known-scam blacklist
        if address in self._known_scam:
            alerts.append(ScamAlert(
                scam_type=ScamType.UNKNOWN,
                risk_level=RiskLevel.CRITICAL,
                description=f"Token {address} is on the known scam blacklist",
                confidence=1.0,
                token_address=address,
                chain=chain,
            ))
            return alerts

        # Run all checks
        alerts.extend(self._check_rug_pull(token_data))
        alerts.extend(self._check_honeypot(token_data))
        alerts.extend(self._check_pump_dump(token_data))
        alerts.extend(self._check_liquidity(token_data))
        alerts.extend(self._check_deployer_history(token_data))

        # Archive
        self.alert_history.extend(alerts)

        return alerts

    def get_legitimacy_report(
        self,
        token_address: str,
        chain: str,
        token_data: Optional[Dict[str, Any]] = None,
    ) -> TokenLegitimacyReport:
        """
        Generate a comprehensive legitimacy report for a token.

        Args:
            token_address: Contract address
            chain: Blockchain network
            token_data: Optional pre-fetched data (if None, returns skeleton)

        Returns:
            TokenLegitimacyReport with score, alerts, and pass/fail checks
        """
        data = token_data or {"address": token_address, "chain": chain}
        alerts = self.scan_token(data)

        checks_passed = []
        checks_failed = []

        # Evaluate each dimension
        if not any(a.scam_type == ScamType.RUG_PULL for a in alerts):
            checks_passed.append("rug_pull_check")
        else:
            checks_failed.append("rug_pull_check")

        if not any(a.scam_type == ScamType.HONEYPOT for a in alerts):
            checks_passed.append("honeypot_check")
        else:
            checks_failed.append("honeypot_check")

        if not any(a.scam_type == ScamType.PUMP_AND_DUMP for a in alerts):
            checks_passed.append("pump_dump_social_check")
        else:
            checks_failed.append("pump_dump_social_check")

        liq = data.get("liquidity", {})
        if liq.get("usd", 0) > 50_000:
            checks_passed.append("minimum_liquidity")
        elif liq:
            checks_failed.append("minimum_liquidity")

        holders = data.get("holders", {})
        if holders.get("count", 0) > 100:
            checks_passed.append("holder_distribution")
        elif holders:
            checks_failed.append("holder_distribution")

        # Score: start at 100, deduct per fail
        score = 100.0
        for alert in alerts:
            if alert.risk_level == RiskLevel.CRITICAL:
                score -= 30 * alert.confidence
            elif alert.risk_level == RiskLevel.HIGH:
                score -= 20 * alert.confidence
            elif alert.risk_level == RiskLevel.MEDIUM:
                score -= 10 * alert.confidence
            elif alert.risk_level == RiskLevel.LOW:
                score -= 5 * alert.confidence
        score = max(0.0, score)

        # Determine overall risk
        if score >= 80:
            risk = RiskLevel.SAFE
        elif score >= 60:
            risk = RiskLevel.LOW
        elif score >= 40:
            risk = RiskLevel.MEDIUM
        elif score >= 20:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL

        return TokenLegitimacyReport(
            token_address=token_address,
            chain=chain,
            risk_level=risk,
            overall_score=score,
            alerts=alerts,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            metadata={"raw_alert_count": len(alerts)},
        )

    # ── Individual Checks ──────────────────────────────────────────────

    def _check_rug_pull(self, data: Dict[str, Any]) -> List[ScamAlert]:
        """Detect rug pull indicators."""
        alerts = []
        contract = data.get("contract_code", "")
        address = data.get("address", "")
        chain = data.get("chain", "")

        found_patterns = []
        for pattern in RUG_PULL_PATTERNS:
            if pattern in contract.lower() if isinstance(contract, str) else False:
                found_patterns.append(pattern)

        # Check deployer renounce-then-reclaim
        deployer = data.get("deployer_history", {})
        if deployer.get("renounced") and deployer.get("reclaimed_after"):
            found_patterns.append("ownership_renounced_then_reclaimed")

        # Check unrestricted mint
        if isinstance(contract, str) and "mint" in contract.lower():
            mint_has_auth = any(k in contract.lower() for k in ["onlyowner", "onlyminter", "require(msg.sender"])
            if not mint_has_auth:
                found_patterns.append("mint_function_unrestricted")

        if found_patterns:
            confidence = min(1.0, len(found_patterns) * 0.25)
            alerts.append(ScamAlert(
                scam_type=ScamType.RUG_PULL,
                risk_level=RiskLevel.CRITICAL if len(found_patterns) >= 3 else RiskLevel.HIGH,
                description=f"Rug pull indicators: {', '.join(found_patterns)}",
                confidence=confidence,
                indicators=found_patterns,
                token_address=address,
                chain=chain,
            ))

        return alerts

    def _check_honeypot(self, data: Dict[str, Any]) -> List[ScamAlert]:
        """Detect honeypot contract patterns (can buy, can't sell)."""
        alerts = []
        trading = data.get("trading_history", {})
        address = data.get("address", "")
        chain = data.get("chain", "")

        buy_count = trading.get("buy_count", 0)
        sell_count = trading.get("sell_count", 0)

        if buy_count > 50 and sell_count == 0:
            alerts.append(ScamAlert(
                scam_type=ScamType.HONEYPOT,
                risk_level=RiskLevel.CRITICAL,
                description=f"Honeypot: {buy_count} buys, 0 sells",
                confidence=0.95,
                indicators=["zero_sells_after_many_buys"],
                token_address=address,
                chain=chain,
            ))
        elif buy_count > 20 and sell_count < buy_count * 0.05:
            alerts.append(ScamAlert(
                scam_type=ScamType.HONEYPOT,
                risk_level=RiskLevel.HIGH,
                description=f"Suspected honeypot: {buy_count} buys, {sell_count} sells (<5%)",
                confidence=0.75,
                indicators=["extreme_buy_sell_imbalance"],
                token_address=address,
                chain=chain,
            ))

        # Check sell tax > 50%
        sell_tax = trading.get("sell_tax_pct", 0)
        if sell_tax > 50:
            alerts.append(ScamAlert(
                scam_type=ScamType.HONEYPOT,
                risk_level=RiskLevel.HIGH,
                description=f"Sell tax {sell_tax}% — effectively a honeypot",
                confidence=0.85,
                indicators=["excessive_sell_tax"],
                token_address=address,
                chain=chain,
            ))

        return alerts

    def _check_pump_dump(self, data: Dict[str, Any]) -> List[ScamAlert]:
        """Detect pump and dump social signals."""
        alerts = []
        social_text = data.get("social_text", "")
        address = data.get("address", "")
        chain = data.get("chain", "")

        if not social_text:
            return alerts

        matches = []
        for regex in PUMP_DUMP_REGEX:
            if regex.search(social_text):
                matches.append(regex.pattern)

        if len(matches) >= 3:
            alerts.append(ScamAlert(
                scam_type=ScamType.PUMP_AND_DUMP,
                risk_level=RiskLevel.HIGH,
                description=f"Pump & dump social signals: {len(matches)} red flags",
                confidence=min(1.0, len(matches) * 0.2),
                indicators=matches,
                token_address=address,
                chain=chain,
            ))
        elif len(matches) >= 1:
            alerts.append(ScamAlert(
                scam_type=ScamType.PUMP_AND_DUMP,
                risk_level=RiskLevel.MEDIUM,
                description=f"Possible pump & dump: {len(matches)} social signal(s)",
                confidence=len(matches) * 0.15,
                indicators=matches,
                token_address=address,
                chain=chain,
            ))

        return alerts

    def _check_liquidity(self, data: Dict[str, Any]) -> List[ScamAlert]:
        """Detect dangerously low or single-provider liquidity."""
        alerts = []
        liq = data.get("liquidity", {})
        address = data.get("address", "")
        chain = data.get("chain", "")

        usd_liq = liq.get("usd", 0)
        providers = liq.get("provider_count", 0)

        if 0 < usd_liq < 1_000:
            alerts.append(ScamAlert(
                scam_type=ScamType.RUG_PULL,
                risk_level=RiskLevel.HIGH,
                description=f"Extremely low liquidity: ${usd_liq:,.0f}",
                confidence=0.8,
                indicators=["dust_liquidity"],
                token_address=address,
                chain=chain,
            ))
        elif 0 < usd_liq < 10_000:
            alerts.append(ScamAlert(
                scam_type=ScamType.RUG_PULL,
                risk_level=RiskLevel.MEDIUM,
                description=f"Low liquidity: ${usd_liq:,.0f}",
                confidence=0.5,
                indicators=["low_liquidity"],
                token_address=address,
                chain=chain,
            ))

        if providers == 1 and usd_liq > 0:
            alerts.append(ScamAlert(
                scam_type=ScamType.RUG_PULL,
                risk_level=RiskLevel.MEDIUM,
                description="Single liquidity provider — rug pull risk",
                confidence=0.6,
                indicators=["single_lp"],
                token_address=address,
                chain=chain,
            ))

        return alerts

    def _check_deployer_history(self, data: Dict[str, Any]) -> List[ScamAlert]:
        """Check deployer wallet history for serial scammer patterns."""
        alerts = []
        deployer = data.get("deployer_history", {})
        address = data.get("address", "")
        chain = data.get("chain", "")

        prev_rugs = deployer.get("previous_rug_count", 0)
        if prev_rugs >= 3:
            alerts.append(ScamAlert(
                scam_type=ScamType.EXIT_SCAM,
                risk_level=RiskLevel.CRITICAL,
                description=f"Deployer has {prev_rugs} previous rug pulls",
                confidence=0.95,
                indicators=["serial_scammer"],
                token_address=address,
                chain=chain,
            ))
        elif prev_rugs >= 1:
            alerts.append(ScamAlert(
                scam_type=ScamType.EXIT_SCAM,
                risk_level=RiskLevel.HIGH,
                description=f"Deployer has {prev_rugs} previous rug pull(s)",
                confidence=0.7,
                indicators=["repeat_deployer"],
                token_address=address,
                chain=chain,
            ))

        return alerts

    # ── Utilities ──────────────────────────────────────────────────────

    def add_to_blacklist(self, address: str) -> None:
        """Add a known scam address to the blacklist."""
        self._known_scam.add(address.lower())
        logger.warning(f"Address {address} added to scam blacklist")

    def add_to_safelist(self, address: str) -> None:
        """Mark a verified-safe address."""
        self._known_safe.add(address.lower())
        logger.info(f"Address {address} added to safe list")

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all detected alerts."""
        by_type: Dict[str, int] = {}
        by_risk: Dict[str, int] = {}
        for alert in self.alert_history:
            by_type[alert.scam_type.value] = by_type.get(alert.scam_type.value, 0) + 1
            by_risk[alert.risk_level.value] = by_risk.get(alert.risk_level.value, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "by_type": by_type,
            "by_risk": by_risk,
            "actionable": sum(1 for a in self.alert_history if a.is_actionable),
            "blacklist_size": len(self._known_scam),
            "safelist_size": len(self._known_safe),
        }
