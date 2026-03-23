"""
Feedback Loop — Fill Logging & Parameter Tuning
=================================================
Logs every put fill with full context (flow conviction, AI score,
Greeks, regime), then tracks outcomes (P&L, max drawdown, hold time)
to iteratively tune scoring weights and thresholds.

Storage: JSON-lines in data/options_intelligence/
    fills.jsonl  — one JSON object per fill
    outcomes.jsonl — one JSON object per closed position

Tuning approach:
    - After N fills (default 20), compute win rate by score bucket
    - Adjust conviction threshold up if low-score trades lose
    - Adjust position sizing if risk is too concentrated
    - Emit parameter adjustment recommendations (not auto-apply)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default storage directory
DATA_DIR = Path(os.environ.get(
    "AAC_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "options_intelligence"),
))

FILLS_FILE = "fills.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"
TUNING_FILE = "tuning_report.json"


@dataclass
class FillRecord:
    """Complete context for a single option fill."""
    # Identification
    fill_id: str                     # Unique ID (timestamp + ticker)
    timestamp: str                   # ISO format
    ticker: str
    direction: str                   # "put" or "call"
    strike: float
    expiry: str                      # YYYY-MM-DD
    dte_at_entry: int
    quantity: int

    # Pricing
    fill_price: float                # Per-share premium paid
    total_cost: float                # fill_price * quantity * 100

    # Greeks at entry
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float

    # Decision context
    flow_conviction: float           # 0-1
    ai_score: int                    # 0-100 composite
    skew_value_score: float          # 0-100
    regime: str
    vix_at_entry: float

    # Outcome (filled later when position closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: Optional[str] = None  # "target", "stop", "expiry", "roll", "manual"

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def is_winner(self) -> bool:
        return self.pnl is not None and self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FillRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TuningRecommendation:
    """A recommended parameter adjustment based on fill outcomes."""
    parameter: str
    current_value: float
    suggested_value: float
    reasoning: str
    confidence: str           # "high", "medium", "low"
    sample_size: int


class FeedbackLoop:
    """
    Logs fills, tracks outcomes, and recommends parameter tuning.

    Usage:
        fb = FeedbackLoop()
        fb.log_fill(fill_record)
        fb.update_outcome(fill_id, exit_price=1.50, exit_reason="target")
        recommendations = fb.analyze()
    """

    MIN_FILLS_FOR_TUNING = 20    # Don't tune until this many closed fills
    SCORE_BUCKETS = [(0, 40), (40, 60), (60, 80), (80, 100)]

    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = Path(data_dir) if data_dir else DATA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fills_path = self._dir / FILLS_FILE
        self._outcomes_path = self._dir / OUTCOMES_FILE
        self._tuning_path = self._dir / TUNING_FILE

    def log_fill(self, fill: FillRecord) -> None:
        """Append a fill record to the log."""
        with open(self._fills_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fill.to_dict(), default=str) + "\n")
        logger.info("Logged fill: %s %s %s @ $%.2f (score=%d, conviction=%.0f%%)",
                     fill.ticker, fill.direction, fill.strike,
                     fill.fill_price, fill.ai_score, fill.flow_conviction * 100)

    def update_outcome(
        self,
        fill_id: str,
        exit_price: float,
        exit_reason: str = "manual",
        max_drawdown_pct: float = 0.0,
    ) -> Optional[FillRecord]:
        """
        Update a fill with its outcome (exit price, P&L).
        Returns the updated fill record or None if not found.
        """
        fills = self._load_fills()
        updated = None

        for fill in fills:
            if fill.fill_id == fill_id:
                fill.exit_price = exit_price
                fill.exit_timestamp = datetime.now().isoformat()
                fill.pnl = (exit_price - fill.fill_price) * fill.quantity * 100
                fill.pnl_pct = ((exit_price / fill.fill_price) - 1) if fill.fill_price > 0 else 0
                fill.max_drawdown_pct = max_drawdown_pct
                fill.exit_reason = exit_reason

                if fill.timestamp:
                    entry_dt = datetime.fromisoformat(fill.timestamp)
                    fill.hold_days = (datetime.now() - entry_dt).days

                updated = fill
                break

        if updated:
            # Rewrite fills file with updated record
            self._save_fills(fills)
            # Also append to outcomes file
            with open(self._outcomes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(updated.to_dict(), default=str) + "\n")
            logger.info("Updated outcome for %s: P&L=$%.2f (%s)",
                        fill_id, updated.pnl or 0, exit_reason)

        return updated

    def get_fills(self, closed_only: bool = False) -> List[FillRecord]:
        """Load all fill records."""
        fills = self._load_fills()
        if closed_only:
            fills = [f for f in fills if f.is_closed]
        return fills

    def get_open_positions(self) -> List[FillRecord]:
        """Get fills that haven't been closed yet."""
        return [f for f in self._load_fills() if not f.is_closed]

    def analyze(self) -> List[TuningRecommendation]:
        """
        Analyze closed fills and recommend parameter adjustments.
        Returns empty list if insufficient data.
        """
        closed = [f for f in self._load_fills() if f.is_closed]
        if len(closed) < self.MIN_FILLS_FOR_TUNING:
            logger.info("Only %d closed fills, need %d for tuning",
                        len(closed), self.MIN_FILLS_FOR_TUNING)
            return []

        recommendations = []

        # 1. Win rate by AI score bucket
        recommendations.extend(self._analyze_score_buckets(closed))

        # 2. Conviction threshold effectiveness
        recommendations.extend(self._analyze_conviction(closed))

        # 3. Average hold time vs DTE
        recommendations.extend(self._analyze_timing(closed))

        # 4. Risk sizing
        recommendations.extend(self._analyze_risk(closed))

        # Save tuning report
        self._save_tuning_report(closed, recommendations)

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Quick stats summary."""
        fills = self._load_fills()
        closed = [f for f in fills if f.is_closed]
        winners = [f for f in closed if f.is_winner]
        total_pnl = sum(f.pnl for f in closed if f.pnl is not None)

        return {
            "total_fills": len(fills),
            "open_positions": len(fills) - len(closed),
            "closed_positions": len(closed),
            "winners": len(winners),
            "losers": len(closed) - len(winners),
            "win_rate": len(winners) / len(closed) if closed else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0,
            "avg_ai_score": sum(f.ai_score for f in fills) / len(fills) if fills else 0,
            "avg_conviction": sum(f.flow_conviction for f in fills) / len(fills) if fills else 0,
        }

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _analyze_score_buckets(self, closed: List[FillRecord]) -> List[TuningRecommendation]:
        """Analyze win rates by AI score bucket."""
        recs: List[TuningRecommendation] = []

        for lo, hi in self.SCORE_BUCKETS:
            bucket = [f for f in closed if lo <= f.ai_score < hi]
            if len(bucket) < 3:
                continue
            win_rate = sum(1 for f in bucket if f.is_winner) / len(bucket)
            avg_pnl = sum(f.pnl for f in bucket if f.pnl is not None) / len(bucket)

            if win_rate < 0.40 and lo < 60:
                recs.append(TuningRecommendation(
                    parameter="min_ai_score",
                    current_value=60,
                    suggested_value=max(lo + 10, 60),
                    reasoning=f"Score bucket [{lo}-{hi}) has {win_rate:.0%} win rate "
                              f"(avg P&L ${avg_pnl:.2f}). Raise minimum threshold.",
                    confidence="medium",
                    sample_size=len(bucket),
                ))

        return recs

    def _analyze_conviction(self, closed: List[FillRecord]) -> List[TuningRecommendation]:
        """Analyze conviction threshold effectiveness."""
        recs: List[TuningRecommendation] = []

        high_conv = [f for f in closed if f.flow_conviction >= 0.7]
        low_conv = [f for f in closed if f.flow_conviction < 0.5]

        if len(high_conv) >= 5 and len(low_conv) >= 5:
            high_wr = sum(1 for f in high_conv if f.is_winner) / len(high_conv)
            low_wr = sum(1 for f in low_conv if f.is_winner) / len(low_conv)

            if high_wr > low_wr + 0.15:
                recs.append(TuningRecommendation(
                    parameter="min_flow_conviction",
                    current_value=0.6,
                    suggested_value=0.7,
                    reasoning=f"High conviction trades win at {high_wr:.0%} vs "
                              f"{low_wr:.0%} for low conviction. Raise threshold.",
                    confidence="high" if (len(high_conv) + len(low_conv)) >= 20 else "medium",
                    sample_size=len(high_conv) + len(low_conv),
                ))

        return recs

    def _analyze_timing(self, closed: List[FillRecord]) -> List[TuningRecommendation]:
        """Analyze hold time patterns."""
        recs: List[TuningRecommendation] = []

        with_hold = [f for f in closed if f.hold_days is not None and f.hold_days > 0]
        if len(with_hold) < 10:
            return recs

        winners = [f for f in with_hold if f.is_winner]
        losers = [f for f in with_hold if not f.is_winner]

        if winners and losers:
            avg_winner_hold = sum(f.hold_days for f in winners) / len(winners)
            avg_loser_hold = sum(f.hold_days for f in losers) / len(losers)

            if avg_loser_hold > avg_winner_hold * 1.5:
                recs.append(TuningRecommendation(
                    parameter="max_hold_days",
                    current_value=0,  # No current limit
                    suggested_value=int(avg_winner_hold * 1.3),
                    reasoning=f"Losers held avg {avg_loser_hold:.0f}d vs winners "
                              f"{avg_winner_hold:.0f}d. Consider time stop.",
                    confidence="medium",
                    sample_size=len(with_hold),
                ))

        return recs

    def _analyze_risk(self, closed: List[FillRecord]) -> List[TuningRecommendation]:
        """Analyze risk sizing patterns."""
        recs: List[TuningRecommendation] = []

        big_losses = [f for f in closed if f.pnl is not None and f.pnl < -500]
        if big_losses:
            avg_score = sum(f.ai_score for f in big_losses) / len(big_losses)
            if avg_score < 65:
                recs.append(TuningRecommendation(
                    parameter="max_risk_per_trade",
                    current_value=0.03,
                    suggested_value=0.02,
                    reasoning=f"{len(big_losses)} large losses (>$500), avg score "
                              f"{avg_score:.0f}. Reduce max risk per trade.",
                    confidence="medium",
                    sample_size=len(big_losses),
                ))

        return recs

    # ═══════════════════════════════════════════════════════════════════
    # I/O
    # ═══════════════════════════════════════════════════════════════════

    def _load_fills(self) -> List[FillRecord]:
        """Load all fills from JSONL file."""
        if not self._fills_path.exists():
            return []
        fills = []
        with open(self._fills_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        fills.append(FillRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("Skipping malformed fill record: %s", exc)
        return fills

    def _save_fills(self, fills: List[FillRecord]) -> None:
        """Rewrite the fills file (for updates)."""
        with open(self._fills_path, "w", encoding="utf-8") as f:
            for fill in fills:
                f.write(json.dumps(fill.to_dict(), default=str) + "\n")

    def _save_tuning_report(
        self,
        closed: List[FillRecord],
        recs: List[TuningRecommendation],
    ) -> None:
        """Save tuning analysis report."""
        report = {
            "generated": datetime.now().isoformat(),
            "sample_size": len(closed),
            "stats": {
                "total_pnl": sum(f.pnl for f in closed if f.pnl is not None),
                "win_rate": sum(1 for f in closed if f.is_winner) / len(closed) if closed else 0,
                "avg_ai_score": sum(f.ai_score for f in closed) / len(closed) if closed else 0,
            },
            "recommendations": [
                {
                    "parameter": r.parameter,
                    "current_value": r.current_value,
                    "suggested_value": r.suggested_value,
                    "reasoning": r.reasoning,
                    "confidence": r.confidence,
                    "sample_size": r.sample_size,
                }
                for r in recs
            ],
        }
        with open(self._tuning_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Tuning report saved to %s (%d recommendations)",
                     self._tuning_path, len(recs))
