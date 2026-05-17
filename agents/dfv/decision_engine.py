from __future__ import annotations

"""DFV decision engine.

The single chokepoint every prompt and every proposed action goes through.
Loads doctrine from config/doctrine/dfv_doctrine.yaml, applies the seven gates,
returns an explicit Verdict the rest of the system must respect.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml

from agents.dfv.memory_store import (
    ConvictionTracker,
    DecisionsLog,
    JournalLog,
    NotificationsLog,
    PostMortemLog,
    ReconciliationLog,
    ThesisLog,
    Watchlist,
)

_log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTRINE_PATH = REPO_ROOT / "config" / "doctrine" / "dfv_doctrine.yaml"

Severity = Literal["hard", "soft"]
GateOutcome = Literal["pass", "fail"]
Verdict = Literal["approved", "approved_with_notes", "returned", "vetoed"]


@dataclass
class GateResult:
    gate_id: str
    name: str
    outcome: GateOutcome
    severity: Severity
    note: str = ""


@dataclass
class Decision:
    verdict: Verdict
    summary: str
    gates: list[GateResult] = field(default_factory=list)
    fixes_required: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    second_opinion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "gates": [g.__dict__ for g in self.gates],
            "fixes_required": self.fixes_required,
            "notes": self.notes,
            "second_opinion": self.second_opinion,
        }


def _load_doctrine() -> dict[str, Any]:
    if not DOCTRINE_PATH.exists():
        raise FileNotFoundError(f"DFV doctrine missing: {DOCTRINE_PATH}")
    return yaml.safe_load(DOCTRINE_PATH.read_text(encoding="utf-8")) or {}


class DFV:
    """Roaring Kitty operator.  All prompts and decisions go through here."""

    def __init__(self) -> None:
        self.doctrine = _load_doctrine()
        mem = self.doctrine.get("memory", {})
        self.thesis = ThesisLog(mem.get("thesis_log", "agents/dfv/memory/thesis_log.json"))
        self.conviction = ConvictionTracker(mem.get("conviction", "agents/dfv/memory/conviction.json"))
        self.watchlist = Watchlist(mem.get("watchlist", "agents/dfv/memory/watchlist.json"))
        self.decisions = DecisionsLog(mem.get("decisions_log", "agents/dfv/memory/decisions.jsonl"))
        self.postmortems = PostMortemLog(mem.get("postmortems", "agents/dfv/memory/postmortems.jsonl"))
        self.journal = JournalLog(mem.get("journal", "agents/dfv/memory/journal.jsonl"))
        self.notifications = NotificationsLog(
            mem.get("notifications", "agents/dfv/memory/notifications.jsonl")
        )
        self.reconciliation = ReconciliationLog(
            mem.get("reconciliation", "agents/dfv/memory/reconciliation.json")
        )

    # ── Public surface ────────────────────────────────────────────
    def evaluate(self, proposal: dict[str, Any]) -> Decision:
        """Run the seven gates against a proposal dict.

        Expected keys (any may be omitted; missing → safe-fail on the gate):
          symbol, action (buy/sell/roll/close), size_pct, side, expected_slippage_pct,
          cash_after_trade, portfolio_value, factor_concentration_after,
          catalyst_within_days, catalyst_acknowledged.
        """
        symbol = (proposal.get("symbol") or "").upper()
        results: list[GateResult] = []

        # G1 — thesis present
        results.append(self._gate_thesis(symbol))
        # G2 — size matches conviction
        results.append(self._gate_conviction_size(symbol, proposal.get("size_pct", 0.0)))
        # G3 — dry powder preserved
        results.append(self._gate_dry_powder(
            float(proposal.get("cash_after_trade", 0.0)),
            float(proposal.get("portfolio_value", 0.0)),
        ))
        # G4 — catalyst window
        results.append(self._gate_catalyst(
            int(proposal.get("catalyst_within_days", 999)),
            bool(proposal.get("catalyst_acknowledged", False)),
        ))
        # G5 — correlation cluster
        results.append(self._gate_correlation(float(proposal.get("factor_concentration_after", 0.0))))
        # G6 — invalidation defined
        results.append(self._gate_invalidation(symbol))
        # G7 — liquidity
        results.append(self._gate_liquidity(float(proposal.get("expected_slippage_pct", 0.0))))
        # G8 — vol regime (long-vega entries only)
        results.append(self._gate_vol_regime(proposal))
        # G9 — autonomy lock: trade execution requires human OK unless
        # doctrine.autonomy.trade_execution == "full". Per operator
        # directive 2026-05-16, this stays "human_in_loop" until
        # explicit DFV-AUTONOMY-CHANGE.
        results.append(self._gate_autonomy(proposal))

        decision = self._render_verdict(symbol, proposal, results)
        decision.second_opinion = self._gemini_second_opinion(symbol, proposal, decision)
        self.decisions.append({
            "symbol": symbol,
            "kind": "proposal",
            "proposal": proposal,
            "decision": decision.to_dict(),
        })
        _log.info(
            "dfv.decision",
            symbol=symbol,
            verdict=decision.verdict,
            failed=[g.gate_id for g in results if g.outcome == "fail"],
            second_opinion=(decision.second_opinion or {}).get("verdict"),
        )
        return decision

    def review_prompt(self, prompt_text: str, *, context: dict[str, Any] | None = None) -> Decision:
        """Apply the DFV lens to a free-text prompt that isn't a structured trade.

        Used by the chatmode / Copilot wrapper before any action is taken.
        """
        ctx = context or {}
        symbol = (ctx.get("symbol") or self._guess_symbol(prompt_text) or "").upper()
        notes: list[str] = []
        gates: list[GateResult] = []

        if symbol:
            gates.append(self._gate_thesis(symbol))
            gates.append(self._gate_invalidation(symbol))
        else:
            gates.append(GateResult("G0", "scope_check", "pass", "soft",
                                    "no specific ticker; treating as research/strategy prompt"))

        # Heuristic flags from prompt language
        lowered = prompt_text.lower()
        if any(kw in lowered for kw in ("yolo", "all in", "bet the farm", "send it")):
            notes.append("Veto trigger: FOMO language detected. Hard rule #5.")
            return Decision(
                verdict="vetoed",
                summary="FOMO language detected. No trade because of FOMO. Ever.",
                gates=gates,
                fixes_required=["Provide written thesis and conviction tier first."],
                notes=notes,
            )
        if any(kw in lowered for kw in ("no thesis", "skip the dd", "just buy")):
            notes.append("Hard rule #1: No position without a written thesis.")
            return Decision(
                verdict="returned",
                summary="Refused: missing thesis.",
                gates=gates,
                fixes_required=["Write a thesis (≤200 words) including invalidation and target."],
                notes=notes,
            )

        decision = self._render_verdict(symbol, {"prompt": prompt_text}, gates, base_notes=notes)
        decision.second_opinion = self._gemini_second_opinion(
            symbol, {"prompt": prompt_text[:600]}, decision
        )
        self.decisions.append({
            "symbol": symbol,
            "kind": "prompt_review",
            "prompt_review": prompt_text[:400],
            "proposal": {"symbol": symbol, "prompt": prompt_text[:400]},
            "decision": decision.to_dict(),
        })
        return decision

    # ── Individual gates ──────────────────────────────────────────
    def _gate_thesis(self, symbol: str) -> GateResult:
        if not symbol:
            return GateResult("G1", "thesis_present", "fail", "hard",
                              "no symbol provided")
        ok = self.thesis.has(symbol)
        return GateResult(
            "G1", "thesis_present",
            "pass" if ok else "fail", "hard",
            "" if ok else f"No thesis on file for {symbol}. Hard rule #1.",
        )

    def _gate_conviction_size(self, symbol: str, size_pct: float) -> GateResult:
        tier = self.conviction.get(symbol) if symbol else 0
        tier_key = str(int(tier)) if tier else "1"
        max_pct = float(
            self.doctrine.get("conviction", {}).get(tier_key, {}).get("max_pct_book", 0.0)
        )
        if size_pct <= max_pct:
            return GateResult("G2", "conviction_size_match", "pass", "hard",
                              f"size {size_pct:.1%} ≤ tier-{tier} cap {max_pct:.1%}")
        return GateResult("G2", "conviction_size_match", "fail", "hard",
                          f"size {size_pct:.1%} exceeds tier-{tier} cap {max_pct:.1%}")

    def _gate_dry_powder(self, cash_after: float, port_value: float) -> GateResult:
        if port_value <= 0:
            return GateResult("G3", "dry_powder_preserved", "pass", "soft", "no portfolio value supplied")
        ratio = cash_after / port_value
        ok = ratio >= 0.10
        return GateResult("G3", "dry_powder_preserved",
                          "pass" if ok else "fail", "soft",
                          f"cash/port = {ratio:.1%} (need ≥10%)")

    def _gate_catalyst(self, days: int, acknowledged: bool) -> GateResult:
        if days > 5:
            return GateResult("G4", "catalyst_window", "pass", "soft", "no catalyst inside 5d")
        if acknowledged:
            return GateResult("G4", "catalyst_window", "pass", "soft",
                              f"catalyst in {days}d, acknowledged")
        return GateResult("G4", "catalyst_window", "fail", "soft",
                          f"catalyst in {days}d, not acknowledged")

    def _gate_correlation(self, conc: float) -> GateResult:
        ok = conc <= 0.40
        return GateResult("G5", "correlation_cluster",
                          "pass" if ok else "fail", "soft",
                          f"factor concentration {conc:.0%} (cap 40%)")

    def _gate_invalidation(self, symbol: str) -> GateResult:
        if not symbol:
            return GateResult("G6", "invalidation_defined", "pass", "hard", "no symbol")
        rec = self.thesis.get(symbol)
        if rec and rec.get("invalidation"):
            return GateResult("G6", "invalidation_defined", "pass", "hard",
                              "invalidation defined")
        return GateResult("G6", "invalidation_defined", "fail", "hard",
                          f"no invalidation level defined for {symbol}")

    def _gate_liquidity(self, slip: float) -> GateResult:
        ok = slip < 0.01
        return GateResult("G7", "liquidity_ok",
                          "pass" if ok else "fail", "soft",
                          f"expected slippage {slip:.2%} (cap 1.0%)")

    def _gate_vol_regime(self, proposal: dict[str, Any]) -> GateResult:
        """G8 — long-vega entries are blocked when VIX < floor AND term structure in contango.

        Skipped (pass) when:
          * doctrine.vol_regime.enabled is false
          * action is not a long-vega entry (sells, closes, rolls)
          * vix/vix3m not supplied (best-effort: pass with note rather than veto)
        """
        cfg = self.doctrine.get("vol_regime", {}) or {}
        if not cfg.get("enabled", True):
            return GateResult("G8", "vol_regime_ok", "pass", "soft", "vol-regime gate disabled")

        long_vega_actions = {a.lower() for a in cfg.get(
            "applies_to_actions", ["buy_put", "buy_call", "long_vol", "buy"],
        )}
        action = str(proposal.get("action") or "").lower()
        side = str(proposal.get("side") or "").lower()
        is_long_vega = bool(proposal.get("long_vega")) or action in long_vega_actions or side in long_vega_actions
        if not is_long_vega:
            return GateResult("G8", "vol_regime_ok", "pass", "soft",
                              "not a long-vega entry; gate not applicable")

        vix = proposal.get("vix")
        vix3m = proposal.get("vix3m")
        if vix is None or vix3m is None:
            return GateResult("G8", "vol_regime_ok", "pass", "soft",
                              "vol regime unknown (vix/vix3m not supplied); flagged-pass")

        floor = float(cfg.get("vix_floor", 15.0))
        contango = float(cfg.get("contango_threshold", 0.0))
        try:
            vix_f = float(vix)
            vix3m_f = float(vix3m)
        except (TypeError, ValueError):
            return GateResult("G8", "vol_regime_ok", "pass", "soft",
                              "vol regime values unparseable; flagged-pass")

        in_contango = (vix3m_f - vix_f) > contango
        low_vix = vix_f < floor
        if low_vix and in_contango:
            return GateResult(
                "G8", "vol_regime_ok", "fail", "soft",
                f"long-vega entry blocked: VIX {vix_f:.1f} < {floor:.1f} AND "
                f"VIX3M-VIX = {vix3m_f - vix_f:+.2f} (contango). Theta beats realized vol here.",
            )
        return GateResult(
            "G8", "vol_regime_ok", "pass", "soft",
            f"VIX {vix_f:.1f}, VIX3M-VIX {vix3m_f - vix_f:+.2f} — regime acceptable",
        )

    # ── Verdict synthesis ─────────────────────────────────────────
    def _gate_autonomy(self, proposal: dict[str, Any]) -> GateResult:
        """G9 — autonomy lock.

        Order placement requires ``doctrine.autonomy.trade_execution ==
        "full"``. Anything else (default ``human_in_loop``) → hard fail
        so the decision renders as ``returned`` and the caller is forced
        to ask the operator before placing the order.

        ``proposal.kind`` of ``"research"`` / ``"screen"`` / ``"alert"``
        / ``"propose"`` bypasses the gate — those are doctrine-blessed
        autonomous activities.
        """
        kind = str(proposal.get("kind") or "").lower()
        if kind in {"research", "screen", "alert", "propose", "memory", "watchlist", "nudge"}:
            return GateResult("G9", "autonomy_lock", "pass", "hard",
                              f"non-execution activity ({kind}); autonomy gate not applicable")
        # Only enforce on explicit order-placement actions.
        action = str(proposal.get("action") or "").lower()
        is_order = bool(proposal.get("place_order")) or action in {
            "buy", "sell", "buy_put", "buy_call", "sell_put", "sell_call",
            "close", "roll", "open", "long_vol", "short_vol", "submit_order",
        }
        if not is_order:
            return GateResult("G9", "autonomy_lock", "pass", "hard",
                              "not an order-placement proposal; autonomy gate not applicable")
        mode = str(
            (self.doctrine.get("autonomy") or {}).get("trade_execution") or "human_in_loop"
        ).lower()
        if mode == "full":
            return GateResult("G9", "autonomy_lock", "pass", "hard",
                              "doctrine.autonomy.trade_execution=full — autonomous execution permitted")
        return GateResult(
            "G9", "autonomy_lock", "fail", "hard",
            f"doctrine.autonomy.trade_execution={mode!r}; order placement requires explicit human OK. "
            "Flip the switch via a DFV-AUTONOMY-CHANGE commit before retrying.",
        )

    def _render_verdict(
        self,
        symbol: str,
        proposal: dict[str, Any],
        gates: list[GateResult],
        base_notes: list[str] | None = None,
    ) -> Decision:
        policy = self.doctrine.get("decision_policy", {})
        hard_cap = int(policy.get("hard_fail_count_to_veto", 1))
        soft_veto = int(policy.get("soft_fail_count_to_veto", 3))
        soft_warn = int(policy.get("soft_fail_count_to_warn", 1))

        hard_fails = [g for g in gates if g.outcome == "fail" and g.severity == "hard"]
        soft_fails = [g for g in gates if g.outcome == "fail" and g.severity == "soft"]

        notes = list(base_notes or [])
        fixes = [g.note for g in hard_fails + soft_fails if g.note]

        if len(hard_fails) >= hard_cap:
            verdict: Verdict = "vetoed"
            summary = f"Vetoed: {len(hard_fails)} hard gate(s) failed."
        elif len(soft_fails) >= soft_veto:
            verdict = "vetoed"
            summary = f"Vetoed: {len(soft_fails)} soft gate(s) failed (cap {soft_veto})."
        elif soft_fails:
            verdict = "returned" if len(soft_fails) > soft_warn else "approved_with_notes"
            summary = f"{len(soft_fails)} soft warning(s)."
        else:
            verdict = "approved"
            summary = f"All {len(gates)} gates passed for {symbol or '(no ticker)'}."

        return Decision(verdict=verdict, summary=summary, gates=gates,
                        fixes_required=fixes, notes=notes)

    # ── Second-opinion (three-LLM jury) ────────────────────────────
    def _gemini_second_opinion(
        self,
        symbol: str,
        proposal: dict[str, Any],
        decision: Decision,
    ) -> dict[str, Any] | None:
        """Compatibility shim — delegates to the three-LLM jury.

        Name kept for historic call sites. Real implementation lives in
        agents.dfv.jury.jury_review. Never overrides the gate verdict.
        """
        from agents.dfv.jury import jury_review  # noqa: PLC0415
        try:
            return jury_review(
                symbol=symbol,
                proposal=proposal,
                decision=decision,
                doctrine=self.doctrine,
            )
        except Exception as exc:  # noqa: BLE001 — never break the engine
            _log.warning("dfv.jury.failed", error=str(exc))
            return None

    # ── Helpers ───────────────────────────────────────────────────
    @staticmethod
    def _guess_symbol(text: str) -> str | None:
        """Best-effort ticker extraction from free text.

        Rules:
          1. $-prefixed cashtag (any length 1-5) wins.
          2. Otherwise, scan 2-5 char all-caps tokens, skip stopwords,
             and prefer the longest non-stopword token (deterministic).
          3. Single-char tokens are NEVER returned (prevents 'I want to add to GME'
             from resolving to 'I' — regression guard).
        """
        import re
        if not text:
            return None
        # 1. Cashtag wins
        m = re.search(r"\$([A-Z]{1,5})\b", text)
        if m:
            return m.group(1)
        # 2. 2-5 char all-caps tokens, skip common false positives
        stopwords = {
            "I", "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN",
            "IS", "IT", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
            "WE", "AM", "ARE", "AND", "BUT", "FOR", "NOT", "THE", "YOU", "OUT",
            "ALL", "ANY", "CAN", "GET", "HAS", "HAD", "HER", "HIM", "HIS", "HOW",
            "ITS", "MAY", "NEW", "NOW", "OUR", "SEE", "SHE", "TWO", "WAS", "WAY",
            "WHO", "WHY", "YES", "YET", "OK", "OKAY", "USD", "CAD", "EUR", "GBP",
            "ETF", "PUT", "CALL", "BUY", "SELL", "ADD", "ROLL", "DD", "DCF",
            "P", "Q", "K", "PE", "PB", "EV", "FCF", "NCAV", "ATH", "ATL",
            "FOMO", "YOLO", "EOD", "EOM", "EOY", "ETA", "FYI", "TBD",
            "AKA", "LOL", "WTF", "DFV", "AAC", "GPT", "AI",
            "DECENT", "SIZE", "NOW", "WANT", "LIKE", "GOOD", "BAD", "DUMP",
            "HOLD", "BULL", "BEAR", "LONG", "SHORT", "OPEN", "CLOSE", "EXIT",
            "TRADE", "STOCK", "OPTION", "STRIKE", "DELTA", "GAMMA", "THETA",
            "VEGA", "VOL", "VIX", "CRYPTO",
        }
        # Hard minimum: 2 chars. Find candidates, then choose longest non-stopword.
        candidates = [t for t in re.findall(r"\b([A-Z]{2,5})\b", text) if t not in stopwords]
        if not candidates:
            return None
        # Deterministic: longest first, then first occurrence.
        candidates.sort(key=lambda t: (-len(t), text.index(t)))
        return candidates[0]


# ── Module-level convenience ──────────────────────────────────────
_singleton: DFV | None = None


def _instance() -> DFV:
    global _singleton
    if _singleton is None:
        _singleton = DFV()
    return _singleton


def decide(proposal: dict[str, Any]) -> Decision:
    """Run a structured trade proposal through DFV's seven gates."""
    return _instance().evaluate(proposal)


def review_prompt(text: str, *, context: dict[str, Any] | None = None) -> Decision:
    """Run a free-text prompt through DFV's lens (used by chatmode wrapper)."""
    return _instance().review_prompt(text, context=context)
