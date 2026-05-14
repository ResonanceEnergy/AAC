"""shared/signal.py — Standard trade signal dataclass used across all strategies.

Every strategy that produces actionable trade ideas MUST emit ``TradeSignal``
objects.  Downstream components (execution, risk, P&L) depend on these fields.

Signal lifecycle
----------------
  Strategy → TradeSignal (this file)
           → risk check  (shared/risk_manager.py, Sprint 3)
           → order       (TradingExecution/trading_engine.py, Sprint 2)
           → fill        (TradingExecution/exchange_connectors/*)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    LONG_PUT = "long_put"    # buy protective / bearish put
    LONG_CALL = "long_call"  # buy bullish call
    FLAT = "flat"            # reduce / exit


class AssetClass(str, Enum):
    EQUITY = "equity"
    OPTION = "option"
    FUTURES = "futures"
    CRYPTO = "crypto"
    ETF = "etf"


@dataclass
class TradeSignal:
    """Standard signal format emitted by every strategy.

    Required fields
    ---------------
    ticker      : Exchange symbol (e.g. "SPY", "^VIX", "BTC-USD")
    direction   : Direction enum — what we want to do
    confidence  : 0.0–1.0 probability / conviction score
    entry       : Suggested entry price (0 = market order)
    stop        : Hard stop-loss price (0 = no stop defined)
    target      : Price target (0 = hold to expiry or undefined)
    size        : Fraction of account to deploy (0.0–1.0) OR contract count
                  if ``size_in_contracts=True``

    Optional fields
    ---------------
    strategy    : Name of the originating strategy module
    regime      : Market regime label at signal time ("CRISIS" / "ELEVATED" / …)
    asset_class : Instrument type
    expiry      : Options expiry date string ("2026-07-18")
    strike      : Options strike price
    notes       : Free-text rationale (not parsed downstream)
    generated_at: UTC timestamp — set automatically at construction
    """

    ticker: str
    direction: Direction
    confidence: float          # 0.0 – 1.0
    entry: float               # 0 = market
    stop: float                # 0 = undefined
    target: float              # 0 = undefined
    size: float                # fraction of account OR contracts

    # Optional context
    strategy: str = "unknown"
    regime: str = "UNKNOWN"
    asset_class: AssetClass = AssetClass.EQUITY
    expiry: Optional[str] = None
    strike: Optional[float] = None
    notes: str = ""
    size_in_contracts: bool = False
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0–1, got {self.confidence}")
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "ticker": self.ticker,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "size": self.size,
            "strategy": self.strategy,
            "regime": self.regime,
            "asset_class": self.asset_class.value,
            "expiry": self.expiry,
            "strike": self.strike,
            "notes": self.notes,
            "size_in_contracts": self.size_in_contracts,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradeSignal":
        return cls(
            ticker=d["ticker"],
            direction=Direction(d["direction"]),
            confidence=float(d["confidence"]),
            entry=float(d["entry"]),
            stop=float(d["stop"]),
            target=float(d["target"]),
            size=float(d["size"]),
            strategy=d.get("strategy", "unknown"),
            regime=d.get("regime", "UNKNOWN"),
            asset_class=AssetClass(d.get("asset_class", "equity")),
            expiry=d.get("expiry"),
            strike=d.get("strike"),
            notes=d.get("notes", ""),
            size_in_contracts=bool(d.get("size_in_contracts", False)),
            generated_at=d.get("generated_at", datetime.utcnow().isoformat()),
        )
