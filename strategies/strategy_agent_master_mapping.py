#!/usr/bin/env python3
"""
AAC STRATEGY-AGENT MASTER MAPPING FILE
========================================

Comprehensive mapping of all 49 arbitrage strategies to their assigned agents.
Each strategy gets exactly 2 agents:
- 1 Trading Agent (for execution)
- 1 Executive Assistant Agent (for intelligence and oversight)

This file serves as the central registry for all strategy-agent assignments.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Strategy data from CSV
STRATEGIES_DATA = [
    {"id": 1, "name": "ETF–NAV Dislocation Harvesting", "category": "ETF_Arbitrage"},
    {"id": 2, "name": "Index Reconstitution & Closing-Auction Liquidity", "category": "Index_Arbitrage"},
    {"id": 3, "name": "Closing-Auction Imbalance Micro-Alpha", "category": "Microstructure"},
    {"id": 4, "name": "Overnight vs. Intraday Split (News-Guided)", "category": "Time_of_Day"},
    {"id": 5, "name": "FOMC Cycle & Pre-Announcement Drift", "category": "Macro_Event"},
    {"id": 6, "name": "Variance Risk Premium (Cross-Asset)", "category": "Volatility_Premium"},
    {"id": 7, "name": "Session-Split VRP", "category": "Volatility_Premium"},
    {"id": 8, "name": "Active Dispersion (Correlation Risk Premium)", "category": "Dispersion"},
    {"id": 9, "name": "Conditional Correlation Carry", "category": "Correlation"},
    {"id": 10, "name": "Turn-of-the-Month Overlay", "category": "Calendar_Anomaly"},
    {"id": 11, "name": "Overnight Jump Reversion", "category": "Mean_Reversion"},
    {"id": 12, "name": "Post-Earnings/Accruals Subset Alpha", "category": "Event_Driven"},
    {"id": 13, "name": "Earnings IV Run-Up / Crush", "category": "Event_Driven"},
    {"id": 14, "name": "IV–RV Alignment Trades", "category": "Volatility_Arbitrage"},
    {"id": 15, "name": "Index Cash-and-Carry", "category": "Index_Arbitrage"},
    {"id": 16, "name": "Flow-Pressure Contrarian (ETF/Funds)", "category": "Flow_Analysis"},
    {"id": 17, "name": "Be the Patient Counterparty on Rebalance Days", "category": "Index_Arbitrage"},
    {"id": 18, "name": "EU Closing-Auction Imbalance Unlock", "category": "Microstructure"},
    {"id": 19, "name": "ETF Primary-Market Routing", "category": "ETF_Arbitrage"},
    {"id": 20, "name": "Monetary Momentum Window", "category": "Macro_Trend"},
    {"id": 21, "name": "VRP Term/Moneyness Tilt", "category": "Volatility_Premium"},
    {"id": 22, "name": "NLP-Guided Overnight Selector", "category": "Time_of_Day"},
    {"id": 23, "name": "Auction-Aware MM with RL", "category": "Market_Making"},
    {"id": 24, "name": "Flow Pressure & Real-Economy Feedback (Credit-Equity)", "category": "Flow_Analysis"},
    {"id": 25, "name": "Attention-Weighted TOM Overlay", "category": "Calendar_Anomaly"},
    {"id": 26, "name": "Weekly Overnight Seasonality Timing", "category": "Time_of_Day"},
    {"id": 27, "name": "Clientele Split Allocator", "category": "Time_of_Day"},
    {"id": 28, "name": "Pre-FOMC VIX/Equity Pair", "category": "Macro_Event"},
    {"id": 29, "name": "Pre-FOMC Regime Switch Filter", "category": "Macro_Event"},
    {"id": 30, "name": "Index Inclusion Fade", "category": "Index_Arbitrage"},
    {"id": 31, "name": "Reconstitution Close Microstructure", "category": "Index_Arbitrage"},
    {"id": 32, "name": "Euronext Imbalance Capture", "category": "Microstructure"},
    {"id": 33, "name": "ETF Create/Redeem Latency Edge", "category": "ETF_Arbitrage"},
    {"id": 34, "name": "Bubble‑Watch Flow Contrarian (ETFs)", "category": "Flow_Analysis"},
    {"id": 35, "name": "Muni Fund Outflow Liquidity Provision", "category": "Liquidity_Provision"},
    {"id": 36, "name": "Option‑Trading ETF Rollover Signal", "category": "ETF_Arbitrage"},
    {"id": 37, "name": "Cross‑Asset VRP Basket", "category": "Volatility_Premium"},
    {"id": 38, "name": "VRP Term‑Slope Timing", "category": "Volatility_Premium"},
    {"id": 39, "name": "Robust VRP via Synthetic Variance Swaps", "category": "Volatility_Premium"},
    {"id": 40, "name": "Overnight vs Intraday Variance Skew", "category": "Volatility_Premium"},
    {"id": 41, "name": "Conditional Dependence Trades", "category": "Correlation"},
    {"id": 42, "name": "IC–RC Gate for Dispersion", "category": "Dispersion"},
    {"id": 43, "name": "Concentration‑Aware Dispersion", "category": "Dispersion"},
    {"id": 44, "name": "Overnight Jump Fade (Stock-Specific)", "category": "Mean_Reversion"},
    {"id": 45, "name": "Contextual Accruals", "category": "Event_Driven"},
    {"id": 46, "name": "PEAD Disaggregation", "category": "Event_Driven"},
    {"id": 47, "name": "Event Vega Calendars", "category": "Event_Driven"},
    {"id": 48, "name": "Tenor‑Matched IV–RV", "category": "Volatility_Arbitrage"},
    {"id": 49, "name": "TOM Futures‑Only Overlay", "category": "Calendar_Anomaly"}
]

class AACStrategyAgentMapper:
    """
    Master mapper for strategy-agent assignments.
    Ensures each strategy has exactly 2 agents: 1 trading + 1 executive assistant.
    """

    def __init__(self):
        self.strategy_agent_mapping: Dict[str, Dict[str, Any]] = {}
        self.agent_strategy_mapping: Dict[str, str] = {}
        self.executive_assistants: Dict[str, List[str]] = {}
        self._build_mappings()

    def _build_mappings(self):
        """Build complete strategy-agent mappings"""

        # Executive Assistant Agents (Innovation Agents serving as assistants)
        # Each assistant serves approximately 10 strategies
        executive_assistants = {
            "innovation_1": {
                "name": "Alpha Innovation Assistant",
                "specialty": "ETF & Index Arbitrage",
                "strategies_served": 10,
                "capabilities": ["market_intelligence", "risk_assessment", "execution_optimization"]
            },
            "innovation_2": {
                "name": "Beta Innovation Assistant",
                "specialty": "Volatility & Dispersion",
                "strategies_served": 10,
                "capabilities": ["volatility_analysis", "correlation_modeling", "hedge_optimization"]
            },
            "innovation_3": {
                "name": "Gamma Innovation Assistant",
                "specialty": "Event Driven & Macro",
                "strategies_served": 10,
                "capabilities": ["event_prediction", "macro_analysis", "sentiment_processing"]
            },
            "innovation_4": {
                "name": "Delta Innovation Assistant",
                "specialty": "Microstructure & Flow",
                "strategies_served": 10,
                "capabilities": ["order_flow_analysis", "microstructure_modeling", "liquidity_assessment"]
            },
            "innovation_5": {
                "name": "Epsilon Innovation Assistant",
                "specialty": "Calendar Anomalies & Mean Reversion",
                "strategies_served": 9,  # Last one gets 9
                "capabilities": ["seasonality_analysis", "reversion_signals", "calendar_effects"]
            }
        }

        # Assign strategies to agents
        assistant_counter = 0
        strategies_per_assistant = 10

        for i, strategy in enumerate(STRATEGIES_DATA):
            strategy_id = f"s{strategy['id']:02d}"
            strategy_name = strategy['name']
            category = strategy['category']

            # Assign trading agent (1:1 mapping)
            trading_agent_id = f"agent_{i+1:02d}"
            trading_agent_name = f"TradingAgent_{i+1:02d}"

            # Assign executive assistant (grouped)
            assistant_idx = assistant_counter // strategies_per_assistant
            if assistant_idx >= len(executive_assistants):
                assistant_idx = len(executive_assistants) - 1
            executive_agent_id = f"innovation_{assistant_idx + 1}"

            # Build mapping
            self.strategy_agent_mapping[strategy_id] = {
                "strategy_name": strategy_name,
                "strategy_category": category,
                "trading_agent": {
                    "agent_id": trading_agent_id,
                    "agent_name": trading_agent_name,
                    "role": "execution",
                    "specialization": category,
                    "status": "active"
                },
                "executive_assistant": {
                    "agent_id": executive_agent_id,
                    "agent_name": executive_assistants[executive_agent_id]["name"],
                    "role": "intelligence_oversight",
                    "specialty": executive_assistants[executive_agent_id]["specialty"],
                    "capabilities": executive_assistants[executive_agent_id]["capabilities"],
                    "status": "active"
                },
                "assignment_date": datetime.now().isoformat(),
                "integration_status": "complete"
            }

            # Reverse mapping
            self.agent_strategy_mapping[trading_agent_id] = strategy_id
            self.agent_strategy_mapping[executive_agent_id] = strategy_id

            # Track assistant assignments
            if executive_agent_id not in self.executive_assistants:
                self.executive_assistants[executive_agent_id] = []
            self.executive_assistants[executive_agent_id].append(strategy_id)

            assistant_counter += 1

    def get_strategy_agents(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get both agents assigned to a strategy"""
        return self.strategy_agent_mapping.get(strategy_id)

    def get_agent_strategy(self, agent_id: str) -> Optional[str]:
        """Get strategy assigned to an agent"""
        return self.agent_strategy_mapping.get(agent_id)

    def get_executive_assistant_workload(self, assistant_id: str) -> List[str]:
        """Get all strategies served by an executive assistant"""
        return self.executive_assistants.get(assistant_id, [])

    def get_category_strategies(self, category: str) -> List[str]:
        """Get all strategies in a category"""
        return [sid for sid, data in self.strategy_agent_mapping.items()
                if data["strategy_category"] == category]

    def validate_assignments(self) -> Dict[str, Any]:
        """Validate that all assignments are correct"""

        validation_results = {
            "total_strategies": len(STRATEGIES_DATA),
            "total_trading_agents": 0,
            "total_executive_assistants": 0,
            "strategies_with_both_agents": 0,
            "orphaned_agents": [],
            "missing_assignments": [],
            "workload_distribution": {}
        }

        # Count assignments
        trading_agents = set()
        executive_assistants = set()

        for strategy_id, data in self.strategy_agent_mapping.items():
            if data["trading_agent"]["agent_id"]:
                trading_agents.add(data["trading_agent"]["agent_id"])
                validation_results["total_trading_agents"] += 1

            if data["executive_assistant"]["agent_id"]:
                executive_assistants.add(data["executive_assistant"]["agent_id"])
                validation_results["total_executive_assistants"] += 1

            if data["trading_agent"]["agent_id"] and data["executive_assistant"]["agent_id"]:
                validation_results["strategies_with_both_agents"] += 1

        validation_results["unique_trading_agents"] = len(trading_agents)
        validation_results["unique_executive_assistants"] = len(executive_assistants)

        # Check workload distribution
        for assistant_id, strategies in self.executive_assistants.items():
            validation_results["workload_distribution"][assistant_id] = len(strategies)

        return validation_results

    def export_to_json(self, filepath: str):
        """Export mappings to JSON file"""
        export_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_strategies": len(STRATEGIES_DATA),
                "total_agents": len(self.agent_strategy_mapping),
                "validation": self.validate_assignments()
            },
            "strategy_agent_mappings": self.strategy_agent_mapping,
            "agent_strategy_mappings": self.agent_strategy_mapping,
            "executive_assistant_workloads": self.executive_assistants
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def generate_report(self) -> str:
        """Generate comprehensive assignment report"""

        validation = self.validate_assignments()

        report = f"""
# AAC STRATEGY-AGENT ASSIGNMENT REPORT
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- Total Strategies: {validation['total_strategies']}
- Total Trading Agents: {validation['total_trading_agents']}
- Total Executive Assistants: {validation['total_executive_assistants']}
- Strategies with Complete Assignments: {validation['strategies_with_both_agents']}
- Assignment Completion Rate: {(validation['strategies_with_both_agents']/validation['total_strategies']*100):.1f}%

## EXECUTIVE ASSISTANT WORKLOAD DISTRIBUTION
"""

        for assistant_id, count in validation['workload_distribution'].items():
            assistant_data = None
            for strategy_data in self.strategy_agent_mapping.values():
                if strategy_data["executive_assistant"]["agent_id"] == assistant_id:
                    assistant_data = strategy_data["executive_assistant"]
                    break

            if assistant_data:
                report += f"- {assistant_data['agent_name']} ({assistant_id}): {count} strategies\n"
                report += f"  Specialty: {assistant_data['specialty']}\n"

        report += "\n## SAMPLE STRATEGY ASSIGNMENTS\n"

        # Show first 10 strategy assignments
        for i, (strategy_id, data) in enumerate(list(self.strategy_agent_mapping.items())[:10]):
            report += f"\n### {strategy_id}: {data['strategy_name']}\n"
            report += f"- **Trading Agent**: {data['trading_agent']['agent_name']} ({data['trading_agent']['agent_id']})\n"
            report += f"- **Executive Assistant**: {data['executive_assistant']['agent_name']} ({data['executive_assistant']['agent_id']})\n"
            report += f"- **Category**: {data['strategy_category']}\n"

        if len(self.strategy_agent_mapping) > 10:
            report += f"\n... and {len(self.strategy_agent_mapping) - 10} more strategies\n"

        return report

# Global instance
_strategy_agent_mapper: Optional[AACStrategyAgentMapper] = None

def get_strategy_agent_mapper() -> AACStrategyAgentMapper:
    """Get or create the global strategy-agent mapper instance"""
    global _strategy_agent_mapper
    if _strategy_agent_mapper is None:
        _strategy_agent_mapper = AACStrategyAgentMapper()
    return _strategy_agent_mapper

def get_strategy_agents(strategy_id: str) -> Optional[Dict[str, Any]]:
    """Get agents for a specific strategy"""
    mapper = get_strategy_agent_mapper()
    return mapper.get_strategy_agents(strategy_id)

def get_agent_strategy(agent_id: str) -> Optional[str]:
    """Get strategy for a specific agent"""
    mapper = get_strategy_agent_mapper()
    return mapper.get_agent_strategy(agent_id)

def validate_all_assignments() -> Dict[str, Any]:
    """Validate all strategy-agent assignments"""
    mapper = get_strategy_agent_mapper()
    return mapper.validate_assignments()

# CLI interface
if __name__ == "__main__":
    mapper = get_strategy_agent_mapper()

    print("AAC STRATEGY-AGENT MAPPER")
    print("=" * 50)

    # Show validation results
    validation = mapper.validate_assignments()
    print(f"Total Strategies: {validation['total_strategies']}")
    print(f"Strategies with Both Agents: {validation['strategies_with_both_agents']}")
    print(f"Completion Rate: {(validation['strategies_with_both_agents']/validation['total_strategies']*100):.1f}%")

    print("\nExecutive Assistant Workloads:")
    for assistant_id, strategies in mapper.executive_assistants.items():
        assistant_name = None
        for strategy_data in mapper.strategy_agent_mapping.values():
            if strategy_data["executive_assistant"]["agent_id"] == assistant_id:
                assistant_name = strategy_data["executive_assistant"]["agent_name"]
                break
        print(f"- {assistant_name}: {len(strategies)} strategies")

    # Export to JSON
    mapper.export_to_json("strategy_agent_mappings.json")
    print("\nExported mappings to: strategy_agent_mappings.json")

    # Generate and save report
    report = mapper.generate_report()
    with open("STRATEGY_AGENT_ASSIGNMENT_REPORT.md", "w") as f:
        f.write(report)
    print("Generated report: STRATEGY_AGENT_ASSIGNMENT_REPORT.md")