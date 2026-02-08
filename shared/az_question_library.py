#!/usr/bin/env python3
"""
AAC AZ Question Library
=======================

100 Strategic Questions AZ Would Ask Based on Matrix Monitor Data
Deep analysis of AAC system outputs to identify critical insights and
decision points for executive leadership.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import json

class AACAZQuestionLibrary:
    """
    Comprehensive library of 100 strategic questions that AZ (executive leadership)
    would ask when reviewing Matrix Monitor data and system performance.
    """

    def __init__(self):
        self.questions = self._generate_questions()
        self.categories = self._categorize_questions()

    def _generate_questions(self) -> List[Dict[str, Any]]:
        """Generate the 100 strategic questions organized by category"""

        return [
            # Executive Overview (1-10)
            {
                "id": 1,
                "category": "Executive Overview",
                "question": "What is the current system-wide health score and how does it compare to our 99.9% uptime target?",
                "priority": "Critical",
                "data_source": "system_status.health_score",
                "follow_up": "What are the top 3 factors contributing to any health score below 99%?"
            },
            {
                "id": 2,
                "category": "Executive Overview",
                "question": "What is our current quantum advantage ratio and how has it trended over the last 30 days?",
                "priority": "Critical",
                "data_source": "system_status.quantum_advantage_ratio",
                "follow_up": "Are we maintaining our competitive edge in quantum-enhanced arbitrage?"
            },
            {
                "id": 3,
                "category": "Executive Overview",
                "question": "What is the total system capacity utilization across all departments?",
                "priority": "High",
                "data_source": "department_status.capacity_utilization",
                "follow_up": "Which department is the current bottleneck and what are we doing to address it?"
            },
            {
                "id": 4,
                "category": "Executive Overview",
                "question": "What is our current AI accuracy score and confidence level across all decision systems?",
                "priority": "Critical",
                "data_source": "ai_systems.accuracy_score",
                "follow_up": "How does this compare to our 95% accuracy target for production deployment?"
            },
            {
                "id": 5,
                "category": "Executive Overview",
                "question": "What is the end-to-end latency from signal detection to trade execution?",
                "priority": "Critical",
                "data_source": "performance_metrics.end_to_end_latency_us",
                "follow_up": "Are we maintaining sub-millisecond execution as required for high-frequency arbitrage?"
            },

            # Financial Performance (11-25)
            {
                "id": 11,
                "category": "Financial Performance",
                "question": "What is our total realized P&L for today, this week, and this month?",
                "priority": "Critical",
                "data_source": "financial_performance.realized_pnl",
                "follow_up": "What is the breakdown by strategy type and department?"
            },
            {
                "id": 12,
                "category": "Financial Performance",
                "question": "What is our current unrealized P&L and how volatile has it been in the last hour?",
                "priority": "High",
                "data_source": "financial_performance.unrealized_pnl",
                "follow_up": "Are we within our risk limits for unrealized exposure?"
            },
            {
                "id": 13,
                "category": "Financial Performance",
                "question": "What is our Sharpe ratio and maximum drawdown over different time periods?",
                "priority": "High",
                "data_source": "risk_metrics.sharpe_ratio, risk_metrics.max_drawdown",
                "follow_up": "How do these metrics compare to our institutional targets?"
            },
            {
                "id": 14,
                "category": "Financial Performance",
                "question": "What are our total trading costs including commissions, fees, and slippage?",
                "priority": "High",
                "data_source": "trading_costs.total_costs",
                "follow_up": "What percentage of our gross P&L do these costs represent?"
            },
            {
                "id": 15,
                "category": "Financial Performance",
                "question": "What is the capital utilization efficiency across all strategies?",
                "priority": "Medium",
                "data_source": "capital_allocation.utilization_efficiency",
                "follow_up": "Are we optimizing our capital allocation or leaving money on the table?"
            },

            # Risk Management (26-40)
            {
                "id": 26,
                "category": "Risk Management",
                "question": "What is our current Value at Risk (VaR) at 95% and 99% confidence levels?",
                "priority": "Critical",
                "data_source": "risk_metrics.var_95, risk_metrics.var_99",
                "follow_up": "How does this compare to our risk appetite and regulatory limits?"
            },
            {
                "id": 27,
                "category": "Risk Management",
                "question": "What is the current correlation between our different strategy types?",
                "priority": "High",
                "data_source": "risk_metrics.strategy_correlations",
                "follow_up": "Are we properly diversified or do we have concentrated risk exposures?"
            },
            {
                "id": 28,
                "category": "Risk Management",
                "question": "How many circuit breakers have been triggered in the last 24 hours?",
                "priority": "High",
                "data_source": "circuit_breakers.triggered_count_24h",
                "follow_up": "What were the reasons for these triggers and have they been resolved?"
            },
            {
                "id": 29,
                "category": "Risk Management",
                "question": "What is our current liquidity position and funding availability?",
                "priority": "Critical",
                "data_source": "liquidity_metrics.available_funding",
                "follow_up": "Do we have sufficient liquidity to handle a major market event?"
            },
            {
                "id": 30,
                "category": "Risk Management",
                "question": "What is the stress test result for our portfolio under various market scenarios?",
                "priority": "High",
                "data_source": "stress_testing.portfolio_stress_results",
                "follow_up": "Are we prepared for black swan events and market crashes?"
            },

            # Trading Operations (41-55)
            {
                "id": 41,
                "category": "Trading Operations",
                "question": "What is our trade execution success rate and average fill time?",
                "priority": "High",
                "data_source": "trading_operations.fill_rate, trading_operations.avg_fill_time",
                "follow_up": "Are we achieving our target of 98%+ fill rates?"
            },
            {
                "id": 42,
                "category": "Trading Operations",
                "question": "How many arbitrage opportunities are we detecting per minute?",
                "priority": "Medium",
                "data_source": "opportunity_detection.rate_per_minute",
                "follow_up": "Is this rate increasing or decreasing over time?"
            },
            {
                "id": 43,
                "category": "Trading Operations",
                "question": "What is the geographic distribution of our trading activity?",
                "priority": "Medium",
                "data_source": "trading_operations.geographic_distribution",
                "follow_up": "Are we properly diversified across global markets?"
            },
            {
                "id": 44,
                "category": "Trading Operations",
                "question": "What is our market impact when executing large orders?",
                "priority": "Medium",
                "data_source": "trading_operations.market_impact_analysis",
                "follow_up": "Are we minimizing market impact while maximizing execution quality?"
            },
            {
                "id": 45,
                "category": "Trading Operations",
                "question": "How effective is our smart order routing across different venues?",
                "priority": "High",
                "data_source": "trading_operations.routing_efficiency",
                "follow_up": "Are we getting best execution across all our trading venues?"
            },

            # Technology Infrastructure (56-70)
            {
                "id": 56,
                "category": "Technology Infrastructure",
                "question": "What is the current CPU, memory, and network utilization across our infrastructure?",
                "priority": "Medium",
                "data_source": "infrastructure_metrics.cpu_usage, infrastructure_metrics.memory_usage, infrastructure_metrics.network_io",
                "follow_up": "Do we need to scale our infrastructure to handle increased load?"
            },
            {
                "id": 57,
                "category": "Technology Infrastructure",
                "question": "What is the data latency from our various market data sources?",
                "priority": "High",
                "data_source": "data_pipeline.latency_by_source",
                "follow_up": "Are all our data feeds performing within acceptable latency bounds?"
            },
            {
                "id": 58,
                "category": "Technology Infrastructure",
                "question": "How reliable are our API connections to exchanges and data providers?",
                "priority": "High",
                "data_source": "api_connectivity.uptime_by_provider",
                "follow_up": "Do we have sufficient redundancy and failover mechanisms?"
            },
            {
                "id": 59,
                "category": "Technology Infrastructure",
                "question": "What is the performance of our database and caching layers?",
                "priority": "Medium",
                "data_source": "database_metrics.query_performance, cache_metrics.hit_rate",
                "follow_up": "Are we experiencing any database bottlenecks or cache misses?"
            },
            {
                "id": 60,
                "category": "Technology Infrastructure",
                "question": "How secure is our infrastructure against cyber threats?",
                "priority": "Critical",
                "data_source": "security_metrics.threat_detection_rate, security_metrics.incident_response_time",
                "follow_up": "Are we maintaining our security posture in an evolving threat landscape?"
            },

            # AI & Machine Learning (71-80)
            {
                "id": 71,
                "category": "AI & Machine Learning",
                "question": "What is the accuracy and confidence level of our AI trading signals?",
                "priority": "Critical",
                "data_source": "ai_systems.signal_accuracy, ai_systems.signal_confidence",
                "follow_up": "Are our AI models performing better than traditional strategies?"
            },
            {
                "id": 72,
                "category": "AI & Machine Learning",
                "question": "How frequently are we updating our machine learning models?",
                "priority": "High",
                "data_source": "ai_systems.model_update_frequency",
                "follow_up": "Are we keeping pace with market changes and new patterns?"
            },
            {
                "id": 73,
                "category": "AI & Machine Learning",
                "question": "What is the false positive rate of our opportunity detection system?",
                "priority": "High",
                "data_source": "ai_systems.false_positive_rate",
                "follow_up": "Are we wasting resources on low-quality opportunities?"
            },
            {
                "id": 74,
                "category": "AI & Machine Learning",
                "question": "How diverse is our training data across different market conditions?",
                "priority": "Medium",
                "data_source": "ai_systems.training_data_diversity",
                "follow_up": "Are our models robust across various market regimes?"
            },
            {
                "id": 75,
                "category": "AI & Machine Learning",
                "question": "What is the computational cost of our AI inference operations?",
                "priority": "Medium",
                "data_source": "ai_systems.inference_cost",
                "follow_up": "Are we optimizing our AI operations for cost efficiency?"
            },

            # Compliance & Governance (81-90)
            {
                "id": 81,
                "category": "Compliance & Governance",
                "question": "What is our current doctrine compliance score across all 8 packs?",
                "priority": "Critical",
                "data_source": "doctrine_compliance.overall_score, doctrine_compliance.pack_scores",
                "follow_up": "Which packs are below our 95% compliance target?"
            },
            {
                "id": 82,
                "category": "Compliance & Governance",
                "question": "How many regulatory filings have we submitted this quarter?",
                "priority": "High",
                "data_source": "regulatory_compliance.filings_submitted",
                "follow_up": "Are we meeting all our regulatory reporting obligations?"
            },
            {
                "id": 83,
                "category": "Compliance & Governance",
                "question": "What is the status of our audit trail and record keeping?",
                "priority": "High",
                "data_source": "audit_trail.completeness, audit_trail.retention_compliance",
                "follow_up": "Can we reconstruct any trade or decision from our audit logs?"
            },
            {
                "id": 84,
                "category": "Compliance & Governance",
                "question": "How effective is our incident response and post-mortem process?",
                "priority": "Medium",
                "data_source": "incident_response.effectiveness_score, incident_response.post_mortem_completion_rate",
                "follow_up": "Are we learning from incidents and improving our processes?"
            },
            {
                "id": 85,
                "category": "Compliance & Governance",
                "question": "What is our third-party risk assessment for vendors and counterparties?",
                "priority": "Medium",
                "data_source": "third_party_risk.vendor_assessments, third_party_risk.counterparty_scores",
                "follow_up": "Are we doing business with appropriately vetted partners?"
            },

            # Strategic Questions (91-100)
            {
                "id": 91,
                "category": "Strategic Questions",
                "question": "What new arbitrage opportunities have we identified this quarter?",
                "priority": "High",
                "data_source": "strategic_initiatives.new_opportunities_identified",
                "follow_up": "How can we capitalize on these opportunities?"
            },
            {
                "id": 92,
                "category": "Strategic Questions",
                "question": "What is our competitive position in the arbitrage space?",
                "priority": "Critical",
                "data_source": "competitive_analysis.market_position, competitive_analysis.advantage_metrics",
                "follow_up": "Are we maintaining our technological edge?"
            },
            {
                "id": 93,
                "category": "Strategic Questions",
                "question": "What is the ROI on our technology investments this year?",
                "priority": "High",
                "data_source": "strategic_initiatives.technology_roi",
                "follow_up": "Which investments are delivering the highest returns?"
            },
            {
                "id": 94,
                "category": "Strategic Questions",
                "question": "How scalable is our current architecture for future growth?",
                "priority": "High",
                "data_source": "architecture_assessment.scalability_metrics",
                "follow_up": "What architectural changes do we need for 10x growth?"
            },
            {
                "id": 95,
                "category": "Strategic Questions",
                "question": "What is our talent acquisition and retention success rate?",
                "priority": "Medium",
                "data_source": "human_capital.talent_metrics",
                "follow_up": "Do we have the right team to execute our strategic vision?"
            },
            {
                "id": 96,
                "category": "Strategic Questions",
                "question": "What emerging technologies should we be investing in?",
                "priority": "High",
                "data_source": "technology_scanning.emerging_trends",
                "follow_up": "How can we stay ahead of the technological curve?"
            },
            {
                "id": 97,
                "category": "Strategic Questions",
                "question": "What is our carbon footprint and sustainability impact?",
                "priority": "Medium",
                "data_source": "sustainability_metrics.carbon_footprint, sustainability_metrics.energy_efficiency",
                "follow_up": "How can we reduce our environmental impact?"
            },
            {
                "id": 98,
                "category": "Strategic Questions",
                "question": "What geopolitical risks are we monitoring and how exposed are we?",
                "priority": "High",
                "data_source": "geopolitical_risk.exposure_assessment, geopolitical_risk.monitoring_coverage",
                "follow_up": "Do we have adequate hedging strategies for geopolitical events?"
            },
            {
                "id": 99,
                "category": "Strategic Questions",
                "question": "What is our succession planning and leadership development status?",
                "priority": "Medium",
                "data_source": "leadership_development.succession_readiness, leadership_development.pipeline_strength",
                "follow_up": "Are we prepared for leadership transitions?"
            },
            {
                "id": 100,
                "category": "Strategic Questions",
                "question": "What is our long-term vision and are we on track to achieve it?",
                "priority": "Critical",
                "data_source": "strategic_planning.vision_alignment, strategic_planning.progress_metrics",
                "follow_up": "What course corrections do we need to make?"
            }
        ]

    def _categorize_questions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize questions by their category"""
        categories = {}
        for question in self.questions:
            category = question["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(question)
        return categories

    def get_questions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific category"""
        return self.categories.get(category, [])

    def get_questions_by_priority(self, priority: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific priority level"""
        return [q for q in self.questions if q["priority"] == priority]

    def get_question_by_id(self, question_id: int) -> Dict[str, Any]:
        """Get a specific question by ID"""
        for question in self.questions:
            if question["id"] == question_id:
                return question
        return None

    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all 100 questions"""
        return self.questions

    def get_question_count(self) -> int:
        """Get total number of questions"""
        return len(self.questions)

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())

    def export_to_json(self, filename: str = "az_questions.json"):
        """Export all questions to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.questions, f, indent=2, default=str)

    def generate_executive_summary(self) -> str:
        """Generate an executive summary of the question categories"""
        summary = f"""
AAC AZ Executive Question Library
=================================

Total Questions: {self.get_question_count()}

Categories and Priorities:
"""

        for category, questions in self.categories.items():
            critical = len([q for q in questions if q["priority"] == "Critical"])
            high = len([q for q in questions if q["priority"] == "High"])
            medium = len([q for q in questions if q["priority"] == "Medium"])

            summary += f"\n{category} ({len(questions)} questions):"
            summary += f"\n  - Critical: {critical}"
            summary += f"\n  - High: {high}"
            summary += f"\n  - Medium: {medium}"

        summary += "\n\nKey Focus Areas:"
        summary += "\n- Executive Overview: System health and quantum advantage"
        summary += "\n- Financial Performance: P&L analysis and risk-adjusted returns"
        summary += "\n- Risk Management: VaR, correlations, and circuit breakers"
        summary += "\n- Trading Operations: Execution quality and market coverage"
        summary += "\n- Technology Infrastructure: Performance and reliability"
        summary += "\n- AI & Machine Learning: Model accuracy and training"
        summary += "\n- Compliance & Governance: Regulatory and doctrine compliance"
        summary += "\n- Strategic Questions: Competitive position and future growth"

        return summary


# Global instance
_az_question_library = None

def get_az_question_library() -> AACAZQuestionLibrary:
    """Get the global AZ question library instance"""
    global _az_question_library
    if _az_question_library is None:
        _az_question_library = AACAZQuestionLibrary()
    return _az_question_library


if __name__ == "__main__":
    # Test the question library
    library = get_az_question_library()

    print(library.generate_executive_summary())

    print(f"\nTotal questions: {library.get_question_count()}")
    print(f"Categories: {', '.join(library.get_categories())}")

    # Show sample questions
    print("\nSample Critical Priority Questions:")
    critical_questions = library.get_questions_by_priority("Critical")
    for i, q in enumerate(critical_questions[:5]):
        print(f"{i+1}. {q['question']}")

    # Export to JSON
    library.export_to_json("aac_az_questions_100.json")
    print("\nExported questions to aac_az_questions_100.json")