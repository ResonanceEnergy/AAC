"""Shared doctrine pack registry for AAC.

This module defines the built-in doctrine packs and keeps pack registration
separate from the compliance engine implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .ffd.ffd_engine import FFD_DOCTRINE_PACK


def build_builtin_doctrine_packs(
    Department: Type[Any],
    BarrenWuffetState: Type[Any],
    ActionType: Type[Any],
) -> Dict[int, Dict[str, Any]]:
    """Build the built-in doctrine pack registry.

    The enum types are injected to avoid circular imports with the doctrine engine.
    """

    packs: Dict[int, Dict[str, Any]] = {
        1: {
            "name": "Risk Envelope & Capital Allocation",
            "owner": Department.CENTRAL_ACCOUNTING,
            "key_metrics": [
                "max_drawdown_pct", "daily_loss_pct", "tail_loss_p99",
                "capital_utilization", "margin_buffer", "strategy_correlation_matrix",
                "stressed_var_99", "portfolio_heat",
            ],
            "required_metrics": [
                {"metric": "max_drawdown_pct", "thresholds": {"good": "<5", "warning": "5-10", "critical": ">10"}},
                {"metric": "daily_loss_pct", "thresholds": {"good": "<1", "warning": "1-2", "critical": ">2"}},
                {"metric": "tail_loss_p99", "thresholds": {"good": "<3", "warning": "3-5", "critical": ">5"}},
                {"metric": "capital_utilization", "thresholds": {"good": "<50", "warning": "50-75", "critical": ">75"}},
                {"metric": "margin_buffer", "thresholds": {"good": ">50", "warning": "25-50", "critical": "<25"}},
                {"metric": "strategy_correlation_matrix", "thresholds": {"good": "<0.5", "warning": "0.5-0.7", "critical": ">0.7"}},
                {"metric": "stressed_var_99", "thresholds": {"good": "<3", "warning": "3-5", "critical": ">5"}},
                {"metric": "portfolio_heat", "thresholds": {"good": "<50", "warning": "50-75", "critical": ">75"}},
            ],
            "failure_modes": ["cascading_liquidation", "correlated_drawdown", "leverage_trap"],
            "barren_wuffet_triggers": [
                ("drawdown_exceeds_5pct", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
                ("drawdown_exceeds_10pct", BarrenWuffetState.SAFE_MODE, ActionType.A_STOP_EXECUTION),
                ("daily_loss_exceeds_2pct", BarrenWuffetState.HALT, ActionType.A_STOP_EXECUTION),
            ],
        },
        2: {
            "name": "Security / Secrets / IAM / Key Custody",
            "owner": Department.SHARED_INFRASTRUCTURE,
            "key_metrics": [
                "key_age_days", "failed_auth_rate", "audit_log_completeness",
                "mfa_compliance_rate", "secret_scan_coverage",
            ],
            "required_metrics": [
                {"metric": "key_age_days", "thresholds": {"good": "<30", "warning": "30-60", "critical": ">60"}},
                {"metric": "failed_auth_rate", "thresholds": {"good": "<1", "warning": "1-5", "critical": ">5"}},
                {"metric": "audit_log_completeness", "thresholds": {"good": ">99.9", "warning": "99-99.9", "critical": "<99"}},
                {"metric": "mfa_compliance_rate", "thresholds": {"good": ">95", "warning": "90-95", "critical": "<90"}},
                {"metric": "secret_scan_coverage", "thresholds": {"good": ">95", "warning": "90-95", "critical": "<90"}},
            ],
            "failure_modes": ["key_compromise", "audit_gap"],
            "barren_wuffet_triggers": [
                ("failed_auth_count > 5/min", BarrenWuffetState.SAFE_MODE, ActionType.A_LOCK_KEYS),
                ("key_exposure_detected", BarrenWuffetState.HALT, ActionType.A_LOCK_KEYS),
            ],
        },
        3: {
            "name": "Testing / Simulation / Replay / Chaos",
            "owner": Department.BIGBRAIN_INTELLIGENCE,
            "key_metrics": [
                "backtest_vs_live_correlation", "chaos_test_pass_rate",
                "regression_test_pass_rate", "replay_fidelity_score",
            ],
            "required_metrics": [
                {"metric": "backtest_vs_live_correlation", "thresholds": {"good": ">0.8", "warning": "0.7-0.8", "critical": "<0.7"}},
                {"metric": "chaos_test_pass_rate", "thresholds": {"good": ">95", "warning": "90-95", "critical": "<90"}},
                {"metric": "regression_test_pass_rate", "thresholds": {"good": ">99", "warning": "95-99", "critical": "<95"}},
                {"metric": "replay_fidelity_score", "thresholds": {"good": ">0.95", "warning": "0.9-0.95", "critical": "<0.9"}},
            ],
            "failure_modes": ["test_environment_drift", "flaky_test_syndrome", "backtest_overfitting", "replay_data_corruption"],
            "barren_wuffet_triggers": [
                ("test_coverage_pct < 70", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
                ("regression_test_failures > 0", BarrenWuffetState.CAUTION, ActionType.A_STOP_EXECUTION),
                ("chaos_test_failed", BarrenWuffetState.SAFE_MODE, ActionType.A_ENTER_SAFE_MODE),
                ("backtest_vs_live_drift > 50%", BarrenWuffetState.CAUTION, ActionType.A_FREEZE_STRATEGY),
            ],
        },
        4: {
            "name": "Incident Response + On-Call + Postmortems",
            "owner": Department.SHARED_INFRASTRUCTURE,
            "key_metrics": ["mttd_minutes", "mttr_minutes", "incident_recurrence_rate", "active_sev1_count"],
            "required_metrics": [
                {"metric": "mttd_minutes", "thresholds": {"good": "<5", "warning": "5-15", "critical": ">15"}},
                {"metric": "mttr_minutes", "thresholds": {"good": "<30", "warning": "30-60", "critical": ">60"}},
                {"metric": "incident_recurrence_rate", "thresholds": {"good": "<2", "warning": "2-5", "critical": ">5"}},
                {"metric": "active_sev1_count", "thresholds": {"good": "<1", "warning": "1-2", "critical": ">2"}},
            ],
            "failure_modes": ["alert_fatigue", "escalation_failure", "communication_breakdown", "incomplete_postmortem"],
            "barren_wuffet_triggers": [
                ("active_sev1_count > 0", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
                ("mttd > 10min for sev1", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
                ("mttr > 60min for sev1", BarrenWuffetState.SAFE_MODE, ActionType.A_ENTER_SAFE_MODE),
                ("incident_recurrence within 7 days", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ],
        },
        5: {
            "name": "Liquidity / Market Impact / Partial Fill Logic",
            "owner": Department.TRADING_EXECUTION,
            "key_metrics": [
                "fill_rate", "time_to_fill_p95", "slippage_bps",
                "partial_fill_rate", "adverse_selection_cost", "market_impact_bps",
                "liquidity_available_pct",
            ],
            "required_metrics": [
                {"metric": "fill_rate", "thresholds": {"good": ">95", "warning": "90-95", "critical": "<90"}},
                {"metric": "time_to_fill_p95", "thresholds": {"good": "<300", "warning": "300-600", "critical": ">600"}},
                {"metric": "slippage_bps", "thresholds": {"good": "<5", "warning": "5-10", "critical": ">10"}},
                {"metric": "partial_fill_rate", "thresholds": {"good": "<10", "warning": "10-20", "critical": ">20"}},
                {"metric": "adverse_selection_cost", "thresholds": {"good": "<1", "warning": "1-2", "critical": ">2"}},
                {"metric": "market_impact_bps", "thresholds": {"good": "<5", "warning": "5-10", "critical": ">10"}},
                {"metric": "liquidity_available_pct", "thresholds": {"good": ">80", "warning": "60-80", "critical": "<60"}},
            ],
            "failure_modes": ["liquidity_mirage", "market_impact_underestimation", "partial_fill_cascade", "adverse_selection_trap"],
            "barren_wuffet_triggers": [
                ("slippage_bps_p95 > 10", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
                ("partial_fill_rate > 30%", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
                ("market_impact_bps > 20", BarrenWuffetState.SAFE_MODE, ActionType.A_STOP_EXECUTION),
                ("liquidity_available_pct < 100%", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
            ],
        },
        6: {
            "name": "Counterparty Scoring + Venue Health + Withdrawal Risk",
            "owner": Department.CRYPTO_INTELLIGENCE,
            "key_metrics": [
                "venue_health_score", "withdrawal_success_rate", "counterparty_exposure_pct",
                "settlement_failure_rate", "counterparty_credit_score",
            ],
            "required_metrics": [
                {"metric": "venue_health_score", "thresholds": {"good": ">0.9", "warning": "0.8-0.9", "critical": "<0.8"}},
                {"metric": "withdrawal_success_rate", "thresholds": {"good": ">99", "warning": "97-99", "critical": "<97"}},
                {"metric": "counterparty_exposure_pct", "thresholds": {"good": "<20", "warning": "20-30", "critical": ">30"}},
                {"metric": "settlement_failure_rate", "thresholds": {"good": "<0.1", "warning": "0.1-0.5", "critical": ">0.5"}},
                {"metric": "counterparty_credit_score", "thresholds": {"good": ">80", "warning": "60-80", "critical": "<60"}},
            ],
            "failure_modes": ["venue_insolvency", "withdrawal_freeze"],
            "barren_wuffet_triggers": [
                ("venue_health_score < 0.70", BarrenWuffetState.CAUTION, ActionType.A_ROUTE_FAILOVER),
                ("withdrawal_frozen", BarrenWuffetState.SAFE_MODE, ActionType.A_CREATE_INCIDENT),
                ("settlement_failure_rate > 1%", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
                ("counterparty_credit_score < 50", BarrenWuffetState.SAFE_MODE, ActionType.A_ROUTE_FAILOVER),
            ],
        },
        7: {
            "name": "Research Factory + Experimentation + Strategy Retirement",
            "owner": Department.BIGBRAIN_INTELLIGENCE,
            "key_metrics": [
                "research_pipeline_velocity", "strategy_survival_rate", "feature_reuse_rate",
                "experiment_completion_rate",
            ],
            "required_metrics": [
                {"metric": "research_pipeline_velocity", "thresholds": {"good": ">3", "warning": "2-3", "critical": "<2"}},
                {"metric": "strategy_survival_rate", "thresholds": {"good": ">60", "warning": "40-60", "critical": "<40"}},
                {"metric": "feature_reuse_rate", "thresholds": {"good": ">70", "warning": "50-70", "critical": "<50"}},
                {"metric": "experiment_completion_rate", "thresholds": {"good": ">80", "warning": "60-80", "critical": "<60"}},
            ],
            "failure_modes": ["research_stagnation", "strategy_overfitting", "feature_bloat", "experiment_sprawl"],
            "barren_wuffet_triggers": [
                ("research_pipeline_velocity < 1", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
                ("strategy_survival_rate < 25%", BarrenWuffetState.CAUTION, ActionType.A_FREEZE_STRATEGY),
                ("experiment_completion_rate < 50%", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
                ("model_version_rollback_count > 2 in 7 days", BarrenWuffetState.SAFE_MODE, ActionType.A_FREEZE_STRATEGY),
            ],
        },
        8: {
            "name": "Metric Canon + Truth Arbitration + Retention/Privacy",
            "owner": Department.CENTRAL_ACCOUNTING,
            "key_metrics": ["data_quality_score", "metric_lineage_coverage", "reconciliation_accuracy", "truth_arbitration_latency"],
            "required_metrics": [
                {"metric": "data_quality_score", "thresholds": {"good": ">0.95", "warning": "0.9-0.95", "critical": "<0.9"}},
                {"metric": "metric_lineage_coverage", "thresholds": {"good": ">90", "warning": "80-90", "critical": "<80"}},
                {"metric": "reconciliation_accuracy", "thresholds": {"good": ">99", "warning": "97-99", "critical": "<97"}},
                {"metric": "truth_arbitration_latency", "thresholds": {"good": "<1000", "warning": "1000-5000", "critical": ">5000"}},
            ],
            "failure_modes": ["metric_drift", "truth_conflict_stalemate", "retention_policy_violation", "dashboard_stale_data"],
            "barren_wuffet_triggers": [
                ("data_quality_score < 0.85", BarrenWuffetState.CAUTION, ActionType.A_QUARANTINE_SOURCE),
                ("reconciliation_accuracy < 95%", BarrenWuffetState.CAUTION, ActionType.A_FORCE_RECON),
                ("truth_conflict_unresolved > 30min", BarrenWuffetState.SAFE_MODE, ActionType.A_PAGE_ONCALL),
                ("metric_lineage_coverage < 70%", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ],
        },
        9: {
            "name": "Strategic Warfare",
            "owner": Department.TRADING_EXECUTION,
            "key_metrics": ["terrain_favorability", "force_ratio", "strategic_confidence", "posture_alignment"],
            "required_metrics": [
                {"metric": "terrain_favorability", "thresholds": {"good": ">0.6", "warning": "0.4-0.6", "critical": "<0.4"}},
                {"metric": "force_ratio", "thresholds": {"good": ">1.0", "warning": "0.7-1.0", "critical": "<0.7"}},
                {"metric": "strategic_confidence", "thresholds": {"good": ">0.7", "warning": "0.5-0.7", "critical": "<0.5"}},
                {"metric": "posture_alignment", "thresholds": {"good": ">0.75", "warning": "0.5-0.75", "critical": "<0.5"}},
            ],
            "failure_modes": ["bad_terrain_selection", "force_miscalculation", "timing_disadvantage"],
            "barren_wuffet_triggers": [
                ("force_ratio < 0.5", BarrenWuffetState.CAUTION, ActionType.A_TACTICAL_RETREAT),
                ("terrain_favorability < 0.3", BarrenWuffetState.CAUTION, ActionType.A_TACTICAL_RETREAT),
                ("strategic_confidence < 0.4", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
            ],
        },
        10: {
            "name": "Power Dynamics",
            "owner": Department.TRADING_EXECUTION,
            "key_metrics": ["market_stealth_score", "exchange_reputation", "alpha_uniqueness", "execution_unpredictability"],
            "required_metrics": [
                {"metric": "market_stealth_score", "thresholds": {"good": ">0.7", "warning": "0.5-0.7", "critical": "<0.5"}},
                {"metric": "exchange_reputation", "thresholds": {"good": ">0.8", "warning": "0.6-0.8", "critical": "<0.6"}},
                {"metric": "alpha_uniqueness", "thresholds": {"good": ">0.6", "warning": "0.4-0.6", "critical": "<0.4"}},
                {"metric": "execution_unpredictability", "thresholds": {"good": ">0.65", "warning": "0.45-0.65", "critical": "<0.45"}},
            ],
            "failure_modes": ["telegraphed_execution", "reputation_decay", "alpha_crowding"],
            "barren_wuffet_triggers": [
                ("market_stealth_score < 0.4", BarrenWuffetState.CAUTION, ActionType.A_CONCEAL_POSITION),
                ("exchange_reputation < 0.5", BarrenWuffetState.SAFE_MODE, ActionType.A_TACTICAL_RETREAT),
                ("alpha_uniqueness < 0.3", BarrenWuffetState.CAUTION, ActionType.A_CONCENTRATE_FORCE),
            ],
        },
    }

    packs[11] = {
        "name": FFD_DOCTRINE_PACK["name"],
        "owner": Department.CENTRAL_ACCOUNTING,
        "key_metrics": list(FFD_DOCTRINE_PACK.get("key_metrics", [])),
        "required_metrics": list(FFD_DOCTRINE_PACK.get("required_metrics", [])),
        "failure_modes": list(FFD_DOCTRINE_PACK.get("failure_modes", [])),
        "barren_wuffet_triggers": [
            ("stablecoin_peg_health < 70", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("capital_flight_signal > 50", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("regulatory_shock_score > 60", BarrenWuffetState.SAFE_MODE, ActionType.A_ENTER_SAFE_MODE),
        ],
        "version": "1.0",
        "category": "monetary-transition",
    }

    # ─── Pack 12: Matrix Monitor Command & Control ────────────────────────
    # The AAC Matrix Monitor is a hybrid process/user-interface dashboard,
    # console display, and command-control center that monitors AND commands
    # every component of AAC.  It is the central go-to hub to access and
    # peek inside the inner workings of the organisation and make changes
    # as necessary.
    packs[12] = {
        "name": "Matrix Monitor Command & Control",
        "owner": Department.SHARED_INFRASTRUCTURE,
        "key_metrics": [
            "monitor_uptime_pct",
            "panel_coverage_pct",
            "data_freshness_seconds",
            "pillar_connectivity_pct",
            "api_endpoint_health_pct",
            "elite_desk_components_online",
            "doctrine_compliance_visibility",
            "command_response_latency_ms",
        ],
        "required_metrics": [
            {"metric": "monitor_uptime_pct", "thresholds": {"good": ">99", "warning": "95-99", "critical": "<95"}},
            {"metric": "panel_coverage_pct", "thresholds": {"good": ">95", "warning": "85-95", "critical": "<85"}},
            {"metric": "data_freshness_seconds", "thresholds": {"good": "<10", "warning": "10-30", "critical": ">30"}},
            {"metric": "pillar_connectivity_pct", "thresholds": {"good": ">80", "warning": "60-80", "critical": "<60"}},
            {"metric": "api_endpoint_health_pct", "thresholds": {"good": ">95", "warning": "85-95", "critical": "<85"}},
            {"metric": "elite_desk_components_online", "thresholds": {"good": ">7", "warning": "5-7", "critical": "<5"}},
            {"metric": "doctrine_compliance_visibility", "thresholds": {"good": ">90", "warning": "75-90", "critical": "<75"}},
            {"metric": "command_response_latency_ms", "thresholds": {"good": "<500", "warning": "500-2000", "critical": ">2000"}},
        ],
        "failure_modes": [
            "dashboard_blind_spot",          # Panel data collection fails silently
            "pillar_disconnect",             # One or more pillars unreachable
            "stale_data_display",            # Data refresh stalls but UI keeps showing old data
            "command_channel_failure",       # REST API / directive channel offline
            "elite_desk_desync",             # Trading desk components report stale state
            "doctrine_visibility_gap",       # Compliance data not reaching the monitor
        ],
        "barren_wuffet_triggers": [
            ("monitor_uptime_pct < 90", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("pillar_connectivity_pct < 40", BarrenWuffetState.SAFE_MODE, ActionType.A_PAGE_ONCALL),
            ("data_freshness_seconds > 60", BarrenWuffetState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("elite_desk_components_online < 3", BarrenWuffetState.CAUTION, ActionType.A_THROTTLE_RISK),
        ],
        "version": "1.0",
        "category": "command-control",
        "role": "CENTRAL_C2_HUB",
        "description": (
            "The AAC Matrix Monitor is the supreme command-and-control centre "
            "of the entire AAC organisation.  It is simultaneously a real-time "
            "monitoring dashboard (4 display modes: Terminal, Web, Dash, API), "
            "an operational console that surfaces every metric from every "
            "department, and a command interface through which operators can "
            "inspect and modify the inner workings of the system.  It polls 5 "
            "pillar endpoints (NCC_MASTER, NCC, AAC, NCL, BRS), integrates 9 "
            "elite trading-desk components, tracks 11 doctrine packs, manages "
            "the capital rotation matrix (7 strategies, $10M target), and "
            "exposes a REST API for external orchestration.  When the Matrix "
            "Monitor is down, AAC is blind — making its uptime a top-level "
            "organisational priority."
        ),
        "capabilities": [
            "real_time_system_health_monitoring",
            "doctrine_compliance_display",
            "pnl_and_risk_dashboard",
            "trading_activity_oversight",
            "strategy_metrics_leaderboard",
            "market_intelligence_aggregation",
            "crisis_center_awareness",
            "capital_rotation_tracking",
            "multi_pillar_network_status",
            "elite_trading_desk_integration",
            "rest_api_command_interface",
            "cross_pillar_directive_execution",
            "circuit_breaker_state_reporting",
            "regime_forecaster_display",
            "system_registry_inventory",
        ],
        "display_modes": ["TERMINAL", "WEB", "DASH", "API"],
        "pillar_endpoints": {
            "NCC_MASTER": {"port": 8765, "role": "Supreme Orchestrator"},
            "NCC": {"port": 8765, "role": "Governance & Command"},
            "AAC": {"port": 8080, "role": "Trading & Capital"},
            "NCL": {"port": 8787, "role": "Cognitive Augmentation"},
            "BRS": {"port": 8000, "role": "Digital Labour"},
        },
        "elite_desk_components": [
            "jonny_bravo", "wsb_reddit", "planktonxd", "grok_ai",
            "openclaw", "stock_ticker", "ncl_link", "unusual_whales",
            "matrix_maximizer",
        ],
    }

    return packs


def serialize_doctrine_packs(packs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize doctrine packs to a YAML-safe structure."""
    serialized: Dict[str, Any] = {"doctrine_packs": {}}

    for pack_id, pack in packs.items():
        target = dict(pack)
        owner = target.get("owner")
        if owner is not None and hasattr(owner, "value"):
            target["owner"] = owner.value

        triggers = []
        for trigger in target.get("barren_wuffet_triggers", []):
            if len(trigger) != 3:
                continue
            trigger_str, state, action = trigger
            triggers.append([
                trigger_str,
                state.value if hasattr(state, "value") else state,
                action.value if hasattr(action, "value") else action,
            ])
        target["barren_wuffet_triggers"] = triggers
        serialized["doctrine_packs"][str(pack_id)] = target

    return serialized


def normalize_loaded_doctrine_packs(
    raw_packs: Dict[Any, Dict[str, Any]],
    Department: Type[Any],
    BarrenWuffetState: Type[Any],
    ActionType: Type[Any],
) -> Dict[int, Dict[str, Any]]:
    """Normalize YAML-loaded doctrine packs into runtime objects."""
    normalized: Dict[int, Dict[str, Any]] = {}

    department_by_value = {member.value: member for member in Department}
    state_by_value = {member.value: member for member in BarrenWuffetState}
    action_by_value = {member.value: member for member in ActionType}

    for pack_id, pack in raw_packs.items():
        normalized_pack = dict(pack)
        normalized_id = int(pack_id)

        owner = normalized_pack.get("owner")
        if isinstance(owner, str):
            normalized_pack["owner"] = department_by_value.get(owner, owner)

        normalized_triggers: List[Any] = []
        for trigger in normalized_pack.get("barren_wuffet_triggers", []):
            if not isinstance(trigger, (list, tuple)) or len(trigger) != 3:
                continue
            trigger_str, state, action = trigger
            if isinstance(state, str):
                state = state_by_value.get(state, state)
            if isinstance(action, str):
                action = action_by_value.get(action, action)
            normalized_triggers.append((trigger_str, state, action))
        normalized_pack["barren_wuffet_triggers"] = normalized_triggers

        normalized[normalized_id] = normalized_pack

    return normalized