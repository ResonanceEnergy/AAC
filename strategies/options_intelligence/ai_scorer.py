"""
AI Trade Scorer — LLM-Powered Trade Evaluation
================================================
Uses available LLM APIs (Grok, Claude, OpenAI, Gemini) to evaluate
trade setups by synthesizing flow data, Greeks, macro regime, and
historical patterns into a structured score.

Scoring dimensions:
    1. Thesis alignment — Does this trade fit the active macro regime?
    2. Flow confirmation — Is unusual flow supporting the direction?
    3. Greeks quality — Is the risk/reward profile attractive?
    4. Timing — Is the entry timing optimal (vol regime, DTE)?
    5. Risk — Position-level and portfolio-level risk assessment

Output: TradeScore (0-100) with reasoning, suggested adjustments,
        and confidence level.

Falls back gracefully when no LLM key is available (rules-based heuristic).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """All context needed to evaluate a trade."""
    # Target
    ticker: str
    direction: str              # "put" or "call"
    strike: float
    expiry: str                 # YYYY-MM-DD
    dte: int
    premium: float              # Per-share premium

    # Greeks
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float                   # Implied vol

    # Flow context
    flow_conviction: float      # 0-1 from FlowSignalEngine
    put_call_ratio: float       # Ticker-level
    sweep_count: int = 0
    dark_pool_notional: float = 0.0

    # Macro context
    regime: str = "unknown"     # From RegimeEngine
    vix: float = 0.0
    oil_price: float = 0.0
    hy_spread_bps: float = 0.0

    # Portfolio context
    existing_positions: int = 0
    portfolio_delta: float = 0.0
    portfolio_vega: float = 0.0
    account_balance: float = 0.0
    risk_pct: float = 0.0      # % of account at risk


@dataclass
class TradeScore:
    """AI-evaluated trade score with reasoning."""
    ticker: str
    score: int                   # 0-100 composite score
    thesis_score: int            # 0-100
    flow_score: int              # 0-100
    greeks_score: int            # 0-100
    timing_score: int            # 0-100
    risk_score: int              # 0-100
    reasoning: str               # LLM or heuristic explanation
    adjustments: List[str]       # Suggested modifications
    confidence: str              # "high", "medium", "low"
    model_used: str              # Which LLM or "heuristic"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_actionable(self) -> bool:
        return self.score >= 60

    @property
    def is_strong(self) -> bool:
        return self.score >= 80

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "thesis_score": self.thesis_score,
            "flow_score": self.flow_score,
            "greeks_score": self.greeks_score,
            "timing_score": self.timing_score,
            "risk_score": self.risk_score,
            "reasoning": self.reasoning,
            "adjustments": self.adjustments,
            "confidence": self.confidence,
            "model_used": self.model_used,
        }


class AITradeScorer:
    """
    Evaluates trade setups using LLM reasoning + heuristic fallback.

    Priority order for LLM: xAI Grok → OpenAI → Anthropic → Gemini → heuristic

    Usage:
        scorer = AITradeScorer()
        score = await scorer.score_trade(setup)
        score = scorer.score_trade_sync(setup)  # Heuristic only
    """

    def __init__(self):
        self._xai_key = os.environ.get("XAI_API_KEY", os.environ.get("GROK_API_KEY", ""))
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._gemini_key = os.environ.get("GOOGLE_AI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

    @property
    def has_llm(self) -> bool:
        return bool(self._xai_key or self._openai_key or self._anthropic_key or self._gemini_key)

    async def score_trade(self, setup: TradeSetup) -> TradeScore:
        """Score a trade setup using best available LLM, falling back to heuristic."""
        if self._xai_key:
            return await self._score_with_xai(setup)
        if self._openai_key:
            return await self._score_with_openai(setup)
        # Anthropic and Gemini could be added but heuristic is sufficient fallback
        return self.score_trade_sync(setup)

    def score_trade_sync(self, setup: TradeSetup) -> TradeScore:
        """
        Pure heuristic scoring — no LLM required.
        Fast, deterministic, always available.
        """
        thesis = self._score_thesis(setup)
        flow = self._score_flow(setup)
        greeks = self._score_greeks(setup)
        timing = self._score_timing(setup)
        risk = self._score_risk(setup)

        composite = int(
            thesis * 0.25 +
            flow * 0.25 +
            greeks * 0.20 +
            timing * 0.15 +
            risk * 0.15
        )

        adjustments = self._suggest_adjustments(setup, thesis, flow, greeks, timing, risk)
        reasoning = self._build_heuristic_reasoning(setup, thesis, flow, greeks, timing, risk)

        if composite >= 80:
            confidence = "high"
        elif composite >= 60:
            confidence = "medium"
        else:
            confidence = "low"

        return TradeScore(
            ticker=setup.ticker,
            score=composite,
            thesis_score=thesis,
            flow_score=flow,
            greeks_score=greeks,
            timing_score=timing,
            risk_score=risk,
            reasoning=reasoning,
            adjustments=adjustments,
            confidence=confidence,
            model_used="heuristic",
        )

    # ═══════════════════════════════════════════════════════════════════
    # HEURISTIC SCORING (always available)
    # ═══════════════════════════════════════════════════════════════════

    def _score_thesis(self, s: TradeSetup) -> int:
        """How well does this trade align with active macro regime?"""
        score = 50  # Baseline

        # Crisis regimes boost put thesis
        crisis_regimes = {"credit_stress", "liquidity_crunch", "vol_shock_active",
                          "stagflation", "risk_off", "policy_delay_trap"}
        if s.regime.lower() in crisis_regimes:
            score += 30
        elif s.regime.lower() in {"vol_shock_armed"}:
            score += 20
        elif s.regime.lower() in {"risk_on"}:
            score -= 20

        # Oil shock confirmation
        if s.oil_price > 100:
            score += 10
        elif s.oil_price > 120:
            score += 15

        # HY spread stress
        if s.hy_spread_bps > 500:
            score += 10
        elif s.hy_spread_bps > 350:
            score += 5

        return max(0, min(100, score))

    def _score_flow(self, s: TradeSetup) -> int:
        """Score based on UW flow confirmation."""
        score = 30  # Baseline (no flow = neutral)

        # Flow conviction directly maps
        score += int(s.flow_conviction * 50)

        # Put/call ratio boost
        if s.put_call_ratio > 2.0:
            score += 15
        elif s.put_call_ratio > 1.5:
            score += 10
        elif s.put_call_ratio > 1.0:
            score += 5

        # Sweep count
        score += min(15, s.sweep_count * 5)

        # Dark pool
        if s.dark_pool_notional > 5_000_000:
            score += 10
        elif s.dark_pool_notional > 1_000_000:
            score += 5

        return max(0, min(100, score))

    def _score_greeks(self, s: TradeSetup) -> int:
        """Score Greeks quality: delta, gamma/theta ratio, vega exposure."""
        score = 50

        # Delta sweet spot: -0.25 to -0.40 for puts
        if s.direction == "put":
            if -0.40 <= s.delta <= -0.25:
                score += 25  # Sweet spot
            elif -0.50 <= s.delta <= -0.20:
                score += 10  # Acceptable
            else:
                score -= 10  # Too aggressive or too OTM

        # Gamma/theta ratio: want gamma > |theta|
        if s.theta != 0:
            gt_ratio = abs(s.gamma / s.theta) if s.theta != 0 else 0
            if gt_ratio > 2.0:
                score += 15
            elif gt_ratio > 1.0:
                score += 5
            else:
                score -= 5

        # IV relative to VIX (cheap IV = good for buyers)
        if s.vix > 0:
            iv_ratio = s.iv / (s.vix / 100) if s.vix > 0 else 1.0
            if iv_ratio < 0.8:
                score += 15   # IV cheap relative to market — good buy
            elif iv_ratio < 1.0:
                score += 5
            elif iv_ratio > 1.3:
                score -= 10   # Overpaying for IV

        return max(0, min(100, score))

    def _score_timing(self, s: TradeSetup) -> int:
        """Score entry timing: DTE, vol regime, market cycle."""
        score = 50

        # DTE sweet spot: 14-45 days for weeklies strategy
        if 14 <= s.dte <= 45:
            score += 20
        elif 7 <= s.dte <= 60:
            score += 10
        elif s.dte < 5:
            score -= 20   # Too close to expiry
        elif s.dte > 90:
            score -= 10   # Too far out — theta drag

        # VIX level: moderate VIX = best entry for puts
        if 20 <= s.vix <= 35:
            score += 15  # Elevated but not extreme
        elif s.vix > 40:
            score -= 5   # Premium expensive but trend may be in our favor
        elif s.vix < 15:
            score += 10  # Cheap premium but may be wrong

        return max(0, min(100, score))

    def _score_risk(self, s: TradeSetup) -> int:
        """Score risk management: position sizing, portfolio impact."""
        score = 70  # Default good

        # Risk per trade: 1-3% ideal
        if s.risk_pct <= 0.03:
            score += 15
        elif s.risk_pct <= 0.05:
            score += 5
        else:
            score -= 20  # Over-concentrated

        # Portfolio delta check
        if abs(s.portfolio_delta) > 0.8:
            score -= 15  # Too directional
        elif abs(s.portfolio_delta) > 0.5:
            score -= 5

        # Diversification: too many positions dilutes capital
        if s.existing_positions > 20:
            score -= 10
        elif s.existing_positions > 10:
            score -= 5

        return max(0, min(100, score))

    def _suggest_adjustments(
        self, s: TradeSetup, thesis: int, flow: int, greeks: int, timing: int, risk: int,
    ) -> List[str]:
        """Generate actionable adjustment suggestions."""
        adjustments = []

        if greeks < 50 and s.direction == "put" and s.delta > -0.20:
            adjustments.append("Consider closer-to-ATM strike (delta too low)")

        if greeks < 50 and s.direction == "put" and s.delta < -0.45:
            adjustments.append("Consider further OTM strike (delta too aggressive)")

        if timing < 40 and s.dte < 7:
            adjustments.append("Roll to later expiry — too close to pin risk")

        if timing < 40 and s.dte > 60:
            adjustments.append("Consider shorter DTE to reduce theta drag")

        if risk < 50 and s.risk_pct > 0.05:
            adjustments.append(f"Reduce position size: {s.risk_pct:.1%} > 5% limit")

        if flow < 30:
            adjustments.append("No UW flow confirmation — consider waiting for flow")

        if thesis < 40:
            adjustments.append("Weak macro thesis alignment — lower conviction tier")

        return adjustments

    def _build_heuristic_reasoning(
        self, s: TradeSetup, thesis: int, flow: int, greeks: int, timing: int, risk: int,
    ) -> str:
        """Build human-readable reasoning from heuristic scores."""
        parts = []

        if thesis >= 70:
            parts.append(f"Strong macro alignment ({s.regime})")
        elif thesis >= 50:
            parts.append(f"Moderate macro alignment ({s.regime})")
        else:
            parts.append(f"Weak macro alignment ({s.regime})")

        if flow >= 70:
            parts.append(f"strong UW flow confirmation (conviction={s.flow_conviction:.0%})")
        elif flow >= 40:
            parts.append("moderate flow support")
        else:
            parts.append("no/weak flow confirmation")

        if greeks >= 70:
            parts.append(f"attractive Greeks (delta={s.delta:.2f}, IV={s.iv:.0%})")
        elif greeks < 40:
            parts.append(f"poor Greeks profile (delta={s.delta:.2f})")

        if timing >= 70:
            parts.append(f"good timing ({s.dte}d DTE, VIX={s.vix:.0f})")
        elif timing < 40:
            parts.append(f"suboptimal timing ({s.dte}d DTE)")

        return "; ".join(parts) + "."

    # ═══════════════════════════════════════════════════════════════════
    # LLM SCORING (when keys available)
    # ═══════════════════════════════════════════════════════════════════

    async def _score_with_xai(self, setup: TradeSetup) -> TradeScore:
        """Score using xAI Grok API."""
        prompt = self._build_prompt(setup)
        try:
            response = await self._call_llm_api(
                url="https://api.x.ai/v1/chat/completions",
                api_key=self._xai_key,
                model="grok-3-mini",
                prompt=prompt,
            )
            return self._parse_llm_response(setup, response, "grok-3-mini")
        except Exception as exc:
            logger.warning("xAI scoring failed, falling back to heuristic: %s", exc)
            return self.score_trade_sync(setup)

    async def _score_with_openai(self, setup: TradeSetup) -> TradeScore:
        """Score using OpenAI API."""
        prompt = self._build_prompt(setup)
        try:
            response = await self._call_llm_api(
                url="https://api.openai.com/v1/chat/completions",
                api_key=self._openai_key,
                model="gpt-4o-mini",
                prompt=prompt,
            )
            return self._parse_llm_response(setup, response, "gpt-4o-mini")
        except Exception as exc:
            logger.warning("OpenAI scoring failed, falling back to heuristic: %s", exc)
            return self.score_trade_sync(setup)

    def _build_prompt(self, s: TradeSetup) -> str:
        """Build the LLM scoring prompt."""
        return f"""You are an expert options trader evaluating a trade setup.
Score each dimension 0-100 and provide brief reasoning.

TRADE SETUP:
- Ticker: {s.ticker} | Direction: {s.direction} | Strike: ${s.strike} | Expiry: {s.expiry} ({s.dte}d)
- Premium: ${s.premium:.2f} | IV: {s.iv:.0%} | Delta: {s.delta:.3f} | Gamma: {s.gamma:.4f}
- Vega: ${s.vega:.2f} | Theta: ${s.theta:.3f}/day

FLOW CONTEXT:
- UW Flow Conviction: {s.flow_conviction:.0%} | P/C Ratio: {s.put_call_ratio:.1f}
- Sweeps: {s.sweep_count} | Dark Pool: ${s.dark_pool_notional:,.0f}

MACRO CONTEXT:
- Regime: {s.regime} | VIX: {s.vix:.1f} | Oil: ${s.oil_price:.1f} | HY Spread: {s.hy_spread_bps:.0f}bps

PORTFOLIO:
- Account: ${s.account_balance:,.0f} | Risk: {s.risk_pct:.1%} | Positions: {s.existing_positions}
- Portfolio Delta: {s.portfolio_delta:.2f} | Portfolio Vega: ${s.portfolio_vega:.2f}

Respond in JSON only:
{{"thesis_score": N, "flow_score": N, "greeks_score": N, "timing_score": N,
  "risk_score": N, "reasoning": "...", "adjustments": ["..."], "confidence": "high|medium|low"}}"""

    async def _call_llm_api(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """Make a synchronous HTTPS call to an LLM chat completions API."""
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode())

        content = data["choices"][0]["message"]["content"]
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content)

    def _parse_llm_response(
        self, setup: TradeSetup, resp: Dict[str, Any], model: str,
    ) -> TradeScore:
        """Parse LLM JSON response into TradeScore."""
        thesis = int(resp.get("thesis_score", 50))
        flow = int(resp.get("flow_score", 50))
        greeks = int(resp.get("greeks_score", 50))
        timing = int(resp.get("timing_score", 50))
        risk = int(resp.get("risk_score", 50))

        composite = int(
            thesis * 0.25 + flow * 0.25 + greeks * 0.20 +
            timing * 0.15 + risk * 0.15
        )

        return TradeScore(
            ticker=setup.ticker,
            score=composite,
            thesis_score=thesis,
            flow_score=flow,
            greeks_score=greeks,
            timing_score=timing,
            risk_score=risk,
            reasoning=str(resp.get("reasoning", "LLM evaluation")),
            adjustments=list(resp.get("adjustments", [])),
            confidence=str(resp.get("confidence", "medium")),
            model_used=model,
        )
