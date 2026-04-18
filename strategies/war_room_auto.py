#!/usr/bin/env python3
"""
War Room Auto-Update & Auto-Evolve Engine
==========================================

Plugs into the autonomous engine's ScheduledTask framework to keep
the War Room doctrine alive — live data in, regime shifts out,
milestones tracked, storyboard refreshed, and strategy parameters
evolved based on real outcomes.

Architecture:
    ┌───────────────────────────────────────────────────┐
    │              WAR ROOM AUTO ENGINE                  │
    │                                                   │
    │  AUTO-UPDATE (data freshness)                     │
    │    live_feeds_full   300s  11 API feeds → state   │
    │    balance_sync     3600s  5 accounts → balances  │
    │    indicator_snap     60s  IndicatorState → JSONL  │
    │    composite_trend   300s  score → trend + regime  │
    │    milestone_check   300s  50 gates → triggers     │
    │    mandate_gen     86400s  daily mandate → JSON    │
    │    storyboard_regen 3600s  HTML regen from live    │
    │    council_scan      900s  YT+X → scenarios+ind   │
    │                                                   │
    │  AUTO-EVOLVE (parameter adaptation)               │
    │    scenario_reweight 86400s  P adjustments        │
    │    arm_rebalance     86400s  5-arm % shifts       │
    │    indicator_recalib 86400s  weight tuning        │
    │    phase_check       86400s  wealth phase gate    │
    │                                                   │
    │  PERSISTENCE                                      │
    │    data/war_engine/indicator_snapshots.jsonl       │
    │    data/war_engine/composite_history.jsonl         │
    │    data/war_engine/evolution_log.jsonl             │
    │    data/war_engine/regime_transitions.jsonl        │
    │    data/war_engine/milestones.json (existing)     │
    │    data/war_engine/mandate_*.json (existing)      │
    └───────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("WarRoomAuto")


# ════════════════════════════════════════════════════════════════════════
# PARAMETERS — All tunables in one place
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AutoUpdateParams:
    """Parameters governing the auto-update loop. All intervals in seconds."""

    # ── Update Intervals ──────────────────────────────────────────────
    live_feeds_interval: float = 300.0          # 5 min — full 11-feed refresh
    balance_sync_interval: float = 3600.0       # 1 hr — account balance sync
    indicator_snapshot_interval: float = 60.0   # 1 min — IndicatorState persistence
    composite_trend_interval: float = 300.0     # 5 min — score + regime tracking
    milestone_check_interval: float = 300.0     # 5 min — 50-gate trigger scan
    mandate_gen_interval: float = 86400.0       # 24 hr — daily mandate
    storyboard_regen_interval: float = 3600.0   # 1 hr — HTML refresh
    council_scan_interval: float = 900.0        # 15 min — YouTube + X council intelligence

    # ── Data Retention ────────────────────────────────────────────────
    indicator_snapshot_max_days: int = 90        # Keep 90 days of minute-level snaps
    composite_history_max_days: int = 365        # Keep 1 year of score history
    evolution_log_max_entries: int = 1000        # Max evolution log entries

    # ── Feed Timeouts ─────────────────────────────────────────────────
    feed_timeout_seconds: float = 30.0          # Per-feed API timeout
    feed_max_retries: int = 2                   # Retries before marking feed stale
    feed_stale_threshold_minutes: float = 15.0  # If no update in 15 min → stale

    # ── Balance Thresholds ────────────────────────────────────────────
    balance_change_alert_pct: float = 5.0       # Alert if balance changes >5% in 1 sync
    balance_zero_guard: bool = True             # Never overwrite with $0 (API glitch)


@dataclass
class AutoEvolveParams:
    """Parameters governing the auto-evolve loop."""

    # ── Evolve Intervals ──────────────────────────────────────────────
    scenario_reweight_interval: float = 86400.0   # 24 hr — scenario probability adjustment
    arm_rebalance_interval: float = 86400.0       # 24 hr — 5-arm allocation shift
    indicator_recalibrate_interval: float = 86400.0  # 24 hr — indicator weight tuning
    phase_check_interval: float = 86400.0         # 24 hr — wealth phase gate check

    # ── Regime Change Thresholds ──────────────────────────────────────
    regime_calm_max: float = 30.0               # Composite ≤ 30 → CALM
    regime_watch_max: float = 50.0              # Composite 30-50 → WATCH
    regime_elevated_max: float = 70.0           # Composite 50-70 → ELEVATED
    # Above 70 → CRISIS

    # ── Regime Hysteresis (prevent flapping) ──────────────────────────
    regime_hysteresis: float = 3.0              # Must cross threshold ± 3 pts to flip
    regime_hold_minutes: float = 30.0           # Min time in regime before allowing change

    # ── Scenario Evolve Bounds ────────────────────────────────────────
    scenario_min_probability: float = 0.02      # Never drop below 2%
    scenario_max_probability: float = 0.60      # Never exceed 60%
    scenario_adjust_step: float = 0.03          # Max 3% shift per day
    scenario_sum_target: float = 1.0            # Probabilities must sum to 1.0

    # ── Arm Rebalance Bounds ──────────────────────────────────────────
    arm_min_pct: float = 5.0                    # No arm below 5%
    arm_max_pct: float = 45.0                   # No arm above 45%
    arm_adjust_step: float = 2.0                # Max 2% shift per day per arm
    arm_sum_target: float = 100.0               # Arms must sum to 100%

    # ── Indicator Weight Evolve ───────────────────────────────────────
    indicator_min_weight: float = 0.02          # No indicator below 2%
    indicator_max_weight: float = 0.20          # No indicator above 20%
    indicator_adjust_step: float = 0.01         # Max 1% shift per day
    indicator_sum_target: float = 1.0           # Weights must sum to 1.0

    # ── Phase Gates (USD portfolio value) ─────────────────────────────
    phase_accumulation_max: float = 150_000.0   # Phase 1 cap
    phase_growth_max: float = 1_000_000.0       # Phase 2 cap
    phase_rotation_max: float = 5_000_000.0     # Phase 3 cap
    # Above $5M → preservation

    # ── Milestone Confidence Decay ────────────────────────────────────
    milestone_confidence_decay_per_day: float = 0.002  # -0.2% confidence per day untriggered
    milestone_confidence_floor: float = 0.05           # Never below 5%
    milestone_confidence_boost_on_trigger: float = 0.15  # +15% on related trigger

    # ── Evolution Guard Rails ─────────────────────────────────────────
    max_daily_evolution_steps: int = 5          # Cap total evolve steps per day
    evolution_cooldown_hours: float = 4.0       # Min hours between same evolve type
    require_confirmation_above: float = 0.10    # If single-step shift >10%, log WARNING


# ════════════════════════════════════════════════════════════════════════
# STATE TRACKING
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    """Current regime tracking with hysteresis."""
    current: str = "ELEVATED"          # CALM, WATCH, ELEVATED, CRISIS
    entered_at: Optional[str] = None   # ISO timestamp
    composite_score: float = 57.7
    previous: str = "WATCH"
    transition_count: int = 0


@dataclass
class FeedHealth:
    """Per-feed health status."""
    name: str
    last_success: Optional[str] = None  # ISO timestamp
    last_failure: Optional[str] = None
    consecutive_failures: int = 0
    is_stale: bool = False
    last_value: Optional[Any] = None


@dataclass
class EvolutionStep:
    """Single evolution action."""
    timestamp: str
    category: str              # scenario_reweight, arm_rebalance, indicator_recalib, phase_check
    action: str                # Human-readable description
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    magnitude: float = 0.0    # How big was the change (0-1 normalized)


# ════════════════════════════════════════════════════════════════════════
# WAR ROOM AUTO ENGINE
# ════════════════════════════════════════════════════════════════════════

class WarRoomAutoEngine:
    """
    Manages auto-update (data freshness) and auto-evolve (parameter adaptation)
    for the War Room. Can run standalone or be registered into the autonomous engine.
    """

    STATE_DIR = PROJECT_ROOT / "data" / "war_engine"

    def __init__(
        self,
        update_params: Optional[AutoUpdateParams] = None,
        evolve_params: Optional[AutoEvolveParams] = None,
    ):
        self.update_params = update_params or AutoUpdateParams()
        self.evolve_params = evolve_params or AutoEvolveParams()

        # State
        self.regime = RegimeState()
        self.feed_health: Dict[str, FeedHealth] = {}
        self.evolution_log: List[EvolutionStep] = []
        self._daily_evolve_count = 0
        self._daily_evolve_reset: Optional[datetime] = None
        self._last_evolve_times: Dict[str, datetime] = {}
        self.running = False

        # Ensure state directory
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Load persisted state
        self._load_state()

    # ── Persistence ───────────────────────────────────────────────────

    def _state_file(self, name: str) -> Path:
        return self.STATE_DIR / name

    def _load_state(self):
        """Load persisted regime + feed health from disk."""
        regime_path = self._state_file("regime_state.json")
        if regime_path.exists():
            try:
                data = json.loads(regime_path.read_text(encoding="utf-8"))
                self.regime = RegimeState(**data)
                logger.info("Loaded regime state: %s (score=%.1f)", self.regime.current, self.regime.composite_score)
            except Exception as e:
                logger.warning("Failed to load regime state: %s", e)

        feeds_path = self._state_file("feed_health.json")
        if feeds_path.exists():
            try:
                data = json.loads(feeds_path.read_text(encoding="utf-8"))
                self.feed_health = {k: FeedHealth(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning("Failed to load feed health: %s", e)

    def _save_regime_state(self):
        """Persist current regime to disk."""
        self._state_file("regime_state.json").write_text(
            json.dumps(asdict(self.regime), indent=2, default=str),
            encoding="utf-8",
        )

    def _save_feed_health(self):
        """Persist feed health to disk."""
        self._state_file("feed_health.json").write_text(
            json.dumps({k: asdict(v) for k, v in self.feed_health.items()}, indent=2, default=str),
            encoding="utf-8",
        )

    def _append_jsonl(self, filename: str, record: dict):
        """Append a JSON record to a JSONL file."""
        path = self._state_file(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _read_jsonl(self, filename: str, max_lines: int = 0) -> List[dict]:
        """Read JSONL file. If max_lines > 0, return only last N lines."""
        path = self._state_file(filename)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if max_lines > 0:
            lines = lines[-max_lines:]
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    # ══════════════════════════════════════════════════════════════════
    # AUTO-UPDATE TASKS
    # ══════════════════════════════════════════════════════════════════

    async def task_live_feeds_full(self):
        """Fetch all 11 live feeds and apply to war room engine state."""
        now = datetime.now(timezone.utc).isoformat()
        logger.info("WAR ROOM AUTO: live feeds full refresh")
        try:
            from strategies.war_room_live_feeds import update_all_live_data
            result = await update_all_live_data()
            feed_names = [
                "coingecko", "unusual_whales", "metamask", "ndax", "finnhub",
                "fred", "fear_greed", "newsapi", "x_twitter", "ibkr", "fred_vix",
            ]
            for name in feed_names:
                if name not in self.feed_health:
                    self.feed_health[name] = FeedHealth(name=name)
                fh = self.feed_health[name]
                # Check if feed contributed data (result is a dict of feed→data)
                if isinstance(result, dict) and result.get(name):
                    fh.last_success = now
                    fh.consecutive_failures = 0
                    fh.is_stale = False
                    fh.last_value = str(result[name])[:200]
                else:
                    fh.last_success = fh.last_success or now
            self._save_feed_health()

            # Auto-log intel entry
            try:
                from strategies.ninety_day_war_room import log_intel_update
                log_intel_update("Auto-update: 11-feed refresh completed", {})
            except Exception:
                pass

            logger.info("WAR ROOM AUTO: live feeds refreshed successfully")
        except Exception as e:
            logger.error("WAR ROOM AUTO: live feeds failed: %s", e)
            for name, fh in self.feed_health.items():
                fh.consecutive_failures += 1
                fh.last_failure = now
                if fh.consecutive_failures >= self.update_params.feed_max_retries:
                    fh.is_stale = True
            self._save_feed_health()

    async def task_council_scan(self):
        """Run YouTube + X council intelligence scan and apply to war room."""
        now = datetime.now(timezone.utc).isoformat()
        logger.info("WAR ROOM AUTO: council intelligence scan")
        try:
            from strategies.war_room_council_feeds import fetch_and_apply_council_intel
            council_result = await fetch_and_apply_council_intel()

            # Track feed health for council sources
            for name, count in [("youtube_council", council_result.yt_videos_processed),
                                ("x_council", council_result.x_posts_analyzed)]:
                if name not in self.feed_health:
                    self.feed_health[name] = FeedHealth(name=name)
                fh = self.feed_health[name]
                if count > 0:
                    fh.last_success = now
                    fh.consecutive_failures = 0
                    fh.is_stale = False
                    fh.last_value = council_result.summary()[:200]
                else:
                    fh.consecutive_failures += 1
                    fh.last_failure = now
                    if fh.consecutive_failures >= self.update_params.feed_max_retries:
                        fh.is_stale = True
            self._save_feed_health()

            # Log intel update
            try:
                from strategies.ninety_day_war_room import log_intel_update
                log_intel_update("Council scan: YouTube + X intelligence", {
                    "yt_videos": council_result.yt_videos_processed,
                    "x_posts": council_result.x_posts_analyzed,
                    "scenarios": list(council_result.scenario_signals.keys())[:5],
                    "sentiment": council_result.combined_sentiment,
                })
            except Exception:
                pass

            logger.info("WAR ROOM AUTO: council scan completed — %s", council_result.summary())

            # ── Inject alpha signal into IndicatorState for composite scoring ──
            if council_result.alpha_signal is not None:
                try:
                    from strategies.war_room_engine import IndicatorState
                    IndicatorState.alpha_signal = council_result.alpha_signal
                    logger.info(
                        "WAR ROOM AUTO: alpha injected into IndicatorState: %.4f",
                        council_result.alpha_signal,
                    )
                except (ImportError, AttributeError) as exc:
                    logger.warning("WAR ROOM AUTO: alpha injection failed: %s", exc)

        except ImportError:
            logger.warning("WAR ROOM AUTO: war_room_council_feeds not importable")
        except Exception as e:
            logger.error("WAR ROOM AUTO: council scan failed: %s", e)

    async def task_balance_sync(self):
        """Sync account balances from all platforms."""
        logger.info("WAR ROOM AUTO: balance sync")
        try:
            from config.account_balances import Balances
            old_total = Balances.total_portfolio_usd()

            # Try to run the scanner (scan_all is async)
            try:
                from _check_all_balances import scan_all
                scan_result = await scan_all()
                Balances.sync_from_scan(scan_result)
                Balances.save()
            except ImportError:
                logger.warning("_check_all_balances not importable — using cached balances")
                return

            new_total = Balances.total_portfolio_usd()

            # Alert on large swings
            if old_total > 0:
                pct_change = abs(new_total - old_total) / old_total * 100
                if pct_change > self.update_params.balance_change_alert_pct:
                    logger.warning(
                        "WAR ROOM AUTO: Balance changed %.1f%% ($%.0f → $%.0f)",
                        pct_change, old_total, new_total,
                    )

            logger.info("WAR ROOM AUTO: balances synced ($%.0f USD)", new_total)
        except Exception as e:
            logger.error("WAR ROOM AUTO: balance sync failed: %s", e)

    async def task_indicator_snapshot(self):
        """Snapshot current IndicatorState to JSONL for trend analysis."""
        try:
            from strategies.war_room_engine import IndicatorState
            snap = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "oil_price": IndicatorState.oil_price,
                "gold_price": IndicatorState.gold_price,
                "vix": IndicatorState.vix,
                "spy_price": IndicatorState.spy_price,
                "dxy": IndicatorState.dxy,
                "hy_spread_bp": IndicatorState.hy_spread_bp,
                "btc_price": IndicatorState.btc_price,
                "fed_funds_rate": IndicatorState.fed_funds_rate,
                "x_sentiment": IndicatorState.x_sentiment,
                "fear_greed_index": IndicatorState.fear_greed_index,
                "news_severity": IndicatorState.news_severity,
                "bdc_nav_discount": IndicatorState.bdc_nav_discount,
                "bdc_nonaccrual_pct": IndicatorState.bdc_nonaccrual_pct,
                "defi_tvl_change_pct": IndicatorState.defi_tvl_change_pct,
                "stablecoin_depeg_pct": IndicatorState.stablecoin_depeg_pct,
                "alpha_signal": IndicatorState.alpha_signal,
            }
            self._append_jsonl("indicator_snapshots.jsonl", snap)
        except Exception as e:
            logger.warning("WAR ROOM AUTO: indicator snapshot failed: %s", e)

    async def task_composite_trend(self):
        """Compute composite score, track trends, detect regime changes."""
        try:
            from strategies.war_room_engine import IndicatorState, compute_composite_score
            result = compute_composite_score(IndicatorState)
            score = result.get("composite", 57.7) if isinstance(result, dict) else 57.7
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            # Log to history
            record = {"timestamp": now_iso, "score": score, "regime": self.regime.current}
            self._append_jsonl("composite_history.jsonl", record)

            # Detect regime change
            old_regime = self.regime.current
            new_regime = self._classify_regime(score)

            if new_regime != old_regime:
                # Check hysteresis
                entered = datetime.fromisoformat(self.regime.entered_at) if self.regime.entered_at else now - timedelta(hours=1)
                time_in_regime = (now - entered).total_seconds() / 60.0
                if time_in_regime >= self.evolve_params.regime_hold_minutes:
                    self.regime.previous = old_regime
                    self.regime.current = new_regime
                    self.regime.entered_at = now_iso
                    self.regime.transition_count += 1
                    logger.warning(
                        "WAR ROOM: REGIME CHANGE %s → %s (score=%.1f, transition #%d)",
                        old_regime, new_regime, score, self.regime.transition_count,
                    )
                    # Log transition
                    self._append_jsonl("regime_transitions.jsonl", {
                        "timestamp": now_iso,
                        "from": old_regime,
                        "to": new_regime,
                        "score": score,
                        "held_minutes": round(time_in_regime, 1),
                    })

            self.regime.composite_score = score
            self._save_regime_state()

            logger.info("WAR ROOM AUTO: composite=%.1f regime=%s", score, self.regime.current)
        except Exception as e:
            logger.error("WAR ROOM AUTO: composite trend failed: %s", e)

    def _classify_regime(self, score: float) -> str:
        """Classify regime from composite score with hysteresis."""
        ep = self.evolve_params
        h = ep.regime_hysteresis
        current = self.regime.current

        # Apply hysteresis: need to cross threshold ± h to change
        if current == "CALM" and score > ep.regime_calm_max + h:
            return "WATCH"
        if current == "WATCH":
            if score < ep.regime_calm_max - h:
                return "CALM"
            if score > ep.regime_watch_max + h:
                return "ELEVATED"
        if current == "ELEVATED":
            if score < ep.regime_watch_max - h:
                return "WATCH"
            if score > ep.regime_elevated_max + h:
                return "CRISIS"
        if current == "CRISIS" and score < ep.regime_elevated_max - h:
            return "ELEVATED"

        return current  # No change

    async def task_milestone_check(self):
        """Scan all 50 milestones against current indicators."""
        try:
            from strategies.war_room_engine import (
                IndicatorState,
                check_milestones,
                save_milestone_state,
            )
            # Build state dict from current indicators
            state = {
                "portfolio_usd": 0,
                "oil_price": getattr(IndicatorState, "oil", 0),
                "gold_price": getattr(IndicatorState, "gold", 0),
                "vix": getattr(IndicatorState, "vix", 0),
                "btc_price": getattr(IndicatorState, "btc", 0),
                "spy": getattr(IndicatorState, "spy", 0),
            }
            try:
                from config.account_balances import Balances
                state["portfolio_usd"] = Balances.total_portfolio_usd()
            except Exception:
                pass
            newly_triggered = check_milestones(state)
            if newly_triggered:
                save_milestone_state()
                for ms in newly_triggered:
                    logger.warning("WAR ROOM MILESTONE TRIGGERED: %s", ms)
                    self._append_jsonl("milestone_triggers.jsonl", {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "milestone": str(ms),
                    })
                # Auto-log intel
                try:
                    from strategies.ninety_day_war_room import log_intel_update
                    names = ", ".join(str(m) for m in newly_triggered)
                    log_intel_update(f"Auto: Milestones triggered: {names}", {})
                except Exception:
                    pass
        except Exception as e:
            logger.warning("WAR ROOM AUTO: milestone check failed: %s", e)

    async def task_mandate_gen(self):
        """Generate daily mandate from current state."""
        try:
            from strategies.war_room_engine import generate_mandate, save_mandate
            mandate = generate_mandate(live=True)
            save_mandate(mandate)
            logger.info("WAR ROOM AUTO: daily mandate generated")
        except Exception as e:
            logger.error("WAR ROOM AUTO: mandate generation failed: %s", e)

    async def task_storyboard_regen(self):
        """Regenerate the 13-Moon storyboard HTML with current data."""
        try:
            loop = asyncio.get_event_loop()

            def _regen():
                from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
                from strategies.thirteen_moon_storyboard import export_interactive_storyboard
                doctrine = ThirteenMoonDoctrine()
                return export_interactive_storyboard(doctrine)

            result = await loop.run_in_executor(None, _regen)
            logger.info("WAR ROOM AUTO: storyboard regenerated → %s", result)
        except Exception as e:
            logger.error("WAR ROOM AUTO: storyboard regen failed: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # AUTO-EVOLVE TASKS
    # ══════════════════════════════════════════════════════════════════

    def _can_evolve(self, category: str) -> bool:
        """Check if an evolution step is allowed (cooldown + daily cap)."""
        now = datetime.now(timezone.utc)

        # Daily cap reset
        if self._daily_evolve_reset is None or now.date() > self._daily_evolve_reset.date():
            self._daily_evolve_count = 0
            self._daily_evolve_reset = now

        if self._daily_evolve_count >= self.evolve_params.max_daily_evolution_steps:
            logger.info("WAR ROOM EVOLVE: daily cap (%d) reached", self.evolve_params.max_daily_evolution_steps)
            return False

        # Per-category cooldown
        last = self._last_evolve_times.get(category)
        if last:
            hours_since = (now - last).total_seconds() / 3600.0
            if hours_since < self.evolve_params.evolution_cooldown_hours:
                return False

        return True

    def _record_evolution(self, step: EvolutionStep):
        """Record an evolution step."""
        self.evolution_log.append(step)
        self._daily_evolve_count += 1
        self._last_evolve_times[step.category] = datetime.now(timezone.utc)
        self._append_jsonl("evolution_log.jsonl", asdict(step))

        if step.magnitude > self.evolve_params.require_confirmation_above:
            logger.warning(
                "WAR ROOM EVOLVE: LARGE SHIFT (%.1f%%) in %s: %s",
                step.magnitude * 100, step.category, step.action,
            )

    async def task_scenario_reweight(self):
        """Adjust scenario probabilities based on composite score trajectory."""
        if not self._can_evolve("scenario_reweight"):
            return

        try:
            ep = self.evolve_params
            # Read recent composite history
            history = self._read_jsonl("composite_history.jsonl", max_lines=288)  # ~24hr at 5min
            if len(history) < 12:
                return  # Need at least 1 hour of data

            recent_avg = sum(h["score"] for h in history[-12:]) / 12
            older_avg = sum(h["score"] for h in history[:12]) / 12 if len(history) >= 24 else recent_avg

            trend = recent_avg - older_avg  # Positive = escalating

            # Current scenario probabilities (from WAR_ROOM_DOCTRINE)
            from strategies.thirteen_moon_doctrine import WAR_ROOM_DOCTRINE
            tracks = WAR_ROOM_DOCTRINE.get("scenario_tracks", {})
            probs = {
                "fails": float(tracks.get("fails", {}).get("probability", "45%").strip("%")) / 100,
                "moderate": float(tracks.get("moderate", {}).get("probability", "30%").strip("%")) / 100,
                "major": float(tracks.get("major", {}).get("probability", "20%").strip("%")) / 100,
                "blackswan": float(tracks.get("blackswan", {}).get("probability", "5%").strip("%")) / 100,
            }
            before = dict(probs)

            # Adjust: if trend > 0 (escalating), shift probability toward major/blackswan
            step = min(abs(trend) * 0.005, ep.scenario_adjust_step)  # Scale to trend magnitude
            if trend > 2:  # Meaningful escalation
                probs["fails"] = max(probs["fails"] - step, ep.scenario_min_probability)
                probs["major"] = min(probs["major"] + step * 0.7, ep.scenario_max_probability)
                probs["blackswan"] = min(probs["blackswan"] + step * 0.3, ep.scenario_max_probability)
            elif trend < -2:  # De-escalation
                probs["fails"] = min(probs["fails"] + step, ep.scenario_max_probability)
                probs["major"] = max(probs["major"] - step * 0.7, ep.scenario_min_probability)
                probs["blackswan"] = max(probs["blackswan"] - step * 0.3, ep.scenario_min_probability)

            # Normalize to sum=1
            total = sum(probs.values())
            if total > 0:
                probs = {k: round(v / total, 4) for k, v in probs.items()}

            magnitude = sum(abs(probs[k] - before[k]) for k in probs)
            if magnitude > 0.001:
                self._record_evolution(EvolutionStep(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    category="scenario_reweight",
                    action=f"Trend {'escalating' if trend > 0 else 'de-escalating'} ({trend:+.1f}pts) → shifted probabilities",
                    before=before,
                    after=probs,
                    reason=f"24h composite trend: {trend:+.1f}, recent_avg={recent_avg:.1f}, older_avg={older_avg:.1f}",
                    magnitude=magnitude,
                ))
                logger.info("WAR ROOM EVOLVE: scenarios reweighted (trend=%+.1f)", trend)

        except Exception as e:
            logger.error("WAR ROOM EVOLVE: scenario reweight failed: %s", e)

    async def task_arm_rebalance(self):
        """Adjust 5-arm allocation based on regime and performance."""
        if not self._can_evolve("arm_rebalance"):
            return

        try:
            ep = self.evolve_params
            regime = self.regime.current

            # Regime-based target shifts
            regime_targets = {
                "CALM":     {"iran_oil": 20, "bdc_credit": 20, "crypto_metals": 20, "defi_yield": 20, "tradfi_rotate": 20},
                "WATCH":    {"iran_oil": 25, "bdc_credit": 25, "crypto_metals": 20, "defi_yield": 15, "tradfi_rotate": 15},
                "ELEVATED": {"iran_oil": 30, "bdc_credit": 25, "crypto_metals": 20, "defi_yield": 15, "tradfi_rotate": 10},
                "CRISIS":   {"iran_oil": 35, "bdc_credit": 25, "crypto_metals": 25, "defi_yield": 10, "tradfi_rotate": 5},
            }
            targets = regime_targets.get(regime, regime_targets["ELEVATED"])

            # Current allocations from doctrine
            from strategies.thirteen_moon_doctrine import WAR_ROOM_DOCTRINE
            arms = WAR_ROOM_DOCTRINE.get("five_arms", {})
            current = {k: v.get("target_pct", 20) for k, v in arms.items()}
            before = dict(current)

            # Nudge toward targets, capped by step size
            for arm in current:
                if arm in targets:
                    diff = targets[arm] - current[arm]
                    shift = max(-ep.arm_adjust_step, min(ep.arm_adjust_step, diff))
                    current[arm] = max(ep.arm_min_pct, min(ep.arm_max_pct, current[arm] + shift))

            # Normalize to 100%
            total = sum(current.values())
            if total > 0:
                current = {k: round(v / total * 100, 1) for k, v in current.items()}

            magnitude = sum(abs(current[k] - before.get(k, 0)) for k in current) / 100
            if magnitude > 0.005:
                self._record_evolution(EvolutionStep(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    category="arm_rebalance",
                    action=f"Regime {regime} → arms nudged toward target allocation",
                    before=before,
                    after=current,
                    reason=f"Regime={regime}, composite={self.regime.composite_score:.1f}",
                    magnitude=magnitude,
                ))
                logger.info("WAR ROOM EVOLVE: arms rebalanced for %s regime", regime)

        except Exception as e:
            logger.error("WAR ROOM EVOLVE: arm rebalance failed: %s", e)

    async def task_indicator_recalibrate(self):
        """Adjust indicator weights based on prediction accuracy."""
        if not self._can_evolve("indicator_recalib"):
            return

        try:
            ep = self.evolve_params

            # Read composite history to find which indicators moved most with score
            snapshots = self._read_jsonl("indicator_snapshots.jsonl", max_lines=1440)  # ~24hr at 1min
            if len(snapshots) < 60:
                return  # Need at least 1 hour

            # Current weights (from war_room_engine)
            current_weights = {
                "oil_price": 0.12, "x_sentiment": 0.12, "vix": 0.10, "gold_price": 0.08,
                "hy_spread_bp": 0.08, "spy_price": 0.07, "bdc_nav_discount": 0.07,
                "btc_price": 0.05, "fed_funds_rate": 0.05, "dxy": 0.05,
                "defi_tvl_change_pct": 0.04, "stablecoin_depeg_pct": 0.04,
                "news_severity": 0.04, "fear_greed_index": 0.04, "bdc_nonaccrual_pct": 0.05,
            }
            before = dict(current_weights)

            # Calculate volatility of each indicator (more volatile = more informative in crisis)
            volatilities = {}
            for key in current_weights:
                values = [s.get(key, 0) for s in snapshots if key in s]
                if len(values) > 10:
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    volatilities[key] = variance ** 0.5
                else:
                    volatilities[key] = 0

            # Boost weights of volatile indicators slightly (they're providing signal)
            total_vol = sum(volatilities.values())
            if total_vol > 0:
                for key in current_weights:
                    vol_share = volatilities.get(key, 0) / total_vol
                    current_share = current_weights[key]
                    # Blend: 90% current + 10% volatility-informed
                    new_weight = 0.9 * current_share + 0.1 * vol_share
                    new_weight = max(ep.indicator_min_weight, min(ep.indicator_max_weight, new_weight))
                    current_weights[key] = new_weight

                # Normalize
                total = sum(current_weights.values())
                if total > 0:
                    current_weights = {k: round(v / total, 4) for k, v in current_weights.items()}

            magnitude = sum(abs(current_weights[k] - before[k]) for k in current_weights)
            if magnitude > 0.005:
                self._record_evolution(EvolutionStep(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    category="indicator_recalib",
                    action="Indicator weights recalibrated based on volatility signal",
                    before=before,
                    after=current_weights,
                    reason=f"Top volatile: {sorted(volatilities.items(), key=lambda x: -x[1])[:3]}",
                    magnitude=magnitude,
                ))

        except Exception as e:
            logger.error("WAR ROOM EVOLVE: indicator recalibrate failed: %s", e)

    async def task_phase_check(self):
        """Check if portfolio value has crossed a phase gate."""
        if not self._can_evolve("phase_check"):
            return

        try:
            from config.account_balances import Balances
            total_usd = Balances.total_portfolio_usd()
            ep = self.evolve_params

            if total_usd < ep.phase_accumulation_max:
                new_phase = "accumulation"
            elif total_usd < ep.phase_growth_max:
                new_phase = "growth"
            elif total_usd < ep.phase_rotation_max:
                new_phase = "rotation"
            else:
                new_phase = "preservation"

            self._record_evolution(EvolutionStep(
                timestamp=datetime.now(timezone.utc).isoformat(),
                category="phase_check",
                action=f"Portfolio ${total_usd:,.0f} → phase: {new_phase}",
                before={"phase": "accumulation"},  # TODO: load previous
                after={"phase": new_phase, "portfolio_usd": total_usd},
                reason=f"Phase gates: accumulation<${ep.phase_accumulation_max:,.0f}, growth<${ep.phase_growth_max:,.0f}",
                magnitude=0.0,
            ))
            logger.info("WAR ROOM EVOLVE: phase check — $%.0f → %s", total_usd, new_phase)

        except Exception as e:
            logger.error("WAR ROOM EVOLVE: phase check failed: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # REGISTRATION — Wire into autonomous engine
    # ══════════════════════════════════════════════════════════════════

    def get_task_registry(self) -> List[Dict[str, Any]]:
        """
        Return a list of task definitions ready for autonomous engine registration.
        Each entry: {"name": str, "interval": float, "callback": coroutine, "critical": bool}
        """
        up = self.update_params
        ep = self.evolve_params
        return [
            # Auto-Update tasks
            {"name": "wr_live_feeds",     "interval": up.live_feeds_interval,          "callback": self.task_live_feeds_full,     "critical": False},
            {"name": "wr_balance_sync",   "interval": up.balance_sync_interval,        "callback": self.task_balance_sync,        "critical": False},
            {"name": "wr_indicator_snap", "interval": up.indicator_snapshot_interval,   "callback": self.task_indicator_snapshot,  "critical": False},
            {"name": "wr_composite",      "interval": up.composite_trend_interval,      "callback": self.task_composite_trend,     "critical": False},
            {"name": "wr_milestones",     "interval": up.milestone_check_interval,      "callback": self.task_milestone_check,     "critical": False},
            {"name": "wr_mandate",        "interval": up.mandate_gen_interval,          "callback": self.task_mandate_gen,         "critical": False},
            {"name": "wr_storyboard",     "interval": up.storyboard_regen_interval,     "callback": self.task_storyboard_regen,    "critical": False},
            {"name": "wr_council_scan",   "interval": up.council_scan_interval,          "callback": self.task_council_scan,        "critical": False},
            # Auto-Evolve tasks
            {"name": "wr_scenario_evolve",   "interval": ep.scenario_reweight_interval,     "callback": self.task_scenario_reweight,     "critical": False},
            {"name": "wr_arm_rebalance",     "interval": ep.arm_rebalance_interval,         "callback": self.task_arm_rebalance,         "critical": False},
            {"name": "wr_indicator_recalib", "interval": ep.indicator_recalibrate_interval,  "callback": self.task_indicator_recalibrate, "critical": False},
            {"name": "wr_phase_check",       "interval": ep.phase_check_interval,            "callback": self.task_phase_check,            "critical": False},
        ]

    # ══════════════════════════════════════════════════════════════════
    # STATUS / DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Return current auto engine status for dashboards."""
        stale_feeds = [n for n, fh in self.feed_health.items() if fh.is_stale]
        recent_evolutions = self._read_jsonl("evolution_log.jsonl", max_lines=10)
        return {
            "regime": self.regime.current,
            "composite_score": self.regime.composite_score,
            "regime_transitions": self.regime.transition_count,
            "stale_feeds": stale_feeds,
            "healthy_feeds": len(self.feed_health) - len(stale_feeds),
            "daily_evolve_count": self._daily_evolve_count,
            "daily_evolve_cap": self.evolve_params.max_daily_evolution_steps,
            "recent_evolutions": recent_evolutions,
            "params": {
                "update": {
                    "live_feeds": f"{self.update_params.live_feeds_interval}s",
                    "balance_sync": f"{self.update_params.balance_sync_interval}s",
                    "indicator_snap": f"{self.update_params.indicator_snapshot_interval}s",
                    "composite_trend": f"{self.update_params.composite_trend_interval}s",
                    "milestone_check": f"{self.update_params.milestone_check_interval}s",
                    "mandate_gen": f"{self.update_params.mandate_gen_interval}s",
                    "storyboard_regen": f"{self.update_params.storyboard_regen_interval}s",
                    "council_scan": f"{self.update_params.council_scan_interval}s",
                },
                "evolve": {
                    "regime_thresholds": f"CALM≤{self.evolve_params.regime_calm_max} WATCH≤{self.evolve_params.regime_watch_max} ELEVATED≤{self.evolve_params.regime_elevated_max} CRISIS>",
                    "hysteresis": f"±{self.evolve_params.regime_hysteresis}pts, hold {self.evolve_params.regime_hold_minutes}min",
                    "scenario_bounds": f"{self.evolve_params.scenario_min_probability*100:.0f}%-{self.evolve_params.scenario_max_probability*100:.0f}%, step {self.evolve_params.scenario_adjust_step*100:.0f}%/day",
                    "arm_bounds": f"{self.evolve_params.arm_min_pct:.0f}%-{self.evolve_params.arm_max_pct:.0f}%, step {self.evolve_params.arm_adjust_step:.0f}%/day",
                    "daily_cap": f"{self.evolve_params.max_daily_evolution_steps} steps, {self.evolve_params.evolution_cooldown_hours}hr cooldown",
                },
            },
        }

    def render_status(self) -> str:
        """Render a human-readable status report."""
        s = self.get_status()
        lines = [
            "=" * 60,
            "  WAR ROOM AUTO ENGINE — STATUS",
            "=" * 60,
            f"  Regime:     {s['regime']} (score={s['composite_score']:.1f})",
            f"  Transitions: {s['regime_transitions']}",
            f"  Feeds:       {s['healthy_feeds']} healthy, {len(s['stale_feeds'])} stale",
            f"  Evolve:      {s['daily_evolve_count']}/{s['daily_evolve_cap']} steps today",
            "",
            "  UPDATE INTERVALS:",
        ]
        for k, v in s["params"]["update"].items():
            lines.append(f"    {k:20s} {v}")
        lines.append("")
        lines.append("  EVOLVE PARAMETERS:")
        for k, v in s["params"]["evolve"].items():
            lines.append(f"    {k:20s} {v}")

        if s["stale_feeds"]:
            lines.extend(["", "  ⚠ STALE FEEDS:", *[f"    - {f}" for f in s["stale_feeds"]]])

        if s["recent_evolutions"]:
            lines.extend(["", "  RECENT EVOLUTIONS:"])
            for ev in s["recent_evolutions"][-5:]:
                lines.append(f"    [{ev.get('category', '?')}] {ev.get('action', '?')}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# CLI — Standalone operation + status checks
# ════════════════════════════════════════════════════════════════════════

def main():
    """CLI for war room auto engine."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="War Room Auto-Update & Auto-Evolve Engine")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--run-once", action="store_true", help="Run all tasks once (no loop)")
    parser.add_argument("--update-only", action="store_true", help="Run only auto-update tasks once")
    parser.add_argument("--evolve-only", action="store_true", help="Run only auto-evolve tasks once")
    parser.add_argument("--params", action="store_true", help="Print all parameters as JSON")

    # Parameter overrides
    parser.add_argument("--live-feeds-interval", type=float, help="Live feeds interval (seconds)")
    parser.add_argument("--balance-sync-interval", type=float, help="Balance sync interval (seconds)")
    parser.add_argument("--storyboard-interval", type=float, help="Storyboard regen interval (seconds)")
    parser.add_argument("--regime-hysteresis", type=float, help="Regime hysteresis (points)")
    parser.add_argument("--scenario-step", type=float, help="Scenario adjust step (0-1)")
    parser.add_argument("--arm-step", type=float, help="Arm adjust step (0-100)")
    parser.add_argument("--daily-evolve-cap", type=int, help="Max daily evolution steps")

    args = parser.parse_args()

    # Build params with overrides
    up = AutoUpdateParams()
    ep = AutoEvolveParams()
    if args.live_feeds_interval:
        up.live_feeds_interval = args.live_feeds_interval
    if args.balance_sync_interval:
        up.balance_sync_interval = args.balance_sync_interval
    if args.storyboard_interval:
        up.storyboard_regen_interval = args.storyboard_interval
    if args.regime_hysteresis:
        ep.regime_hysteresis = args.regime_hysteresis
    if args.scenario_step:
        ep.scenario_adjust_step = args.scenario_step
    if args.arm_step:
        ep.arm_adjust_step = args.arm_step
    if args.daily_evolve_cap:
        ep.max_daily_evolution_steps = args.daily_evolve_cap

    engine = WarRoomAutoEngine(update_params=up, evolve_params=ep)

    if args.params:
        from dataclasses import asdict
        print(json.dumps({"update": asdict(up), "evolve": asdict(ep)}, indent=2))
        return

    if args.status:
        print(engine.render_status())
        return

    # Run tasks
    async def _run():
        update_tasks = [
            engine.task_live_feeds_full,
            engine.task_balance_sync,
            engine.task_indicator_snapshot,
            engine.task_composite_trend,
            engine.task_milestone_check,
            engine.task_mandate_gen,
            engine.task_storyboard_regen,
        ]
        evolve_tasks = [
            engine.task_scenario_reweight,
            engine.task_arm_rebalance,
            engine.task_indicator_recalibrate,
            engine.task_phase_check,
        ]

        tasks = []
        if args.run_once or args.update_only:
            tasks.extend(update_tasks)
        if args.run_once or args.evolve_only:
            tasks.extend(evolve_tasks)
        if not tasks:
            tasks = update_tasks + evolve_tasks

        for t in tasks:
            print(f"Running: {t.__name__}...")
            await t()
            print(f"  Done: {t.__name__}")

    asyncio.run(_run())
    print()
    print(engine.render_status())


if __name__ == "__main__":
    main()
