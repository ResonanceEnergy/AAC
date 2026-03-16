#!/usr/bin/env python3
"""
Deep Dive Analysis: What's Missing?
===================================

Comprehensive analysis of the ACC system to identify gaps and create an improvement plan.
"""

import os
import yaml
import json
import asyncio
import sys
import logging

logger = logging.getLogger(__name__)
sys.path.insert(0, '.')

def analyze_missing_components():
    """Analyze what components are missing or incomplete."""
    logger.info('🔍 TARGETED ANALYSIS: What is Really Missing?')
    logger.info('=' * 60)

    missing_items = []
    improvement_areas = []

    # Check actual doctrine packs in config
    try:
        with open('config/doctrine_packs.yaml', 'r', encoding='utf-8') as f:
            doctrine_config = yaml.safe_load(f)

        packs_in_config = list(doctrine_config['doctrine_packs'].keys())
        logger.info(f'📋 Doctrine Packs in Config: {len(packs_in_config)}')
        for pack in packs_in_config:
            logger.info(f'   • {pack}')
    except Exception as e:
        logger.info(f'[CROSS] Cannot read doctrine config: {e}')
        missing_items.append('Doctrine config file')

    # Check actual department directories
    departments = ['TradingExecution', 'BigBrainIntelligence', 'CentralAccounting',
                  'CryptoIntelligence', 'SharedInfrastructure']
    logger.info(f'\n🏢 Department Directories:')
    for dept in departments:
        exists = os.path.exists(dept)
        status = "✅" if exists else "[CROSS]"
        logger.info(f'   {dept}: {status}')
        if not exists:
            missing_items.append(f'Directory: {dept}')

    # Check key files in each department
    logger.info(f'\n📁 Key Files Check:')
    key_files = {
        'TradingExecution': ['execution_engine.py', 'trading_engine.py'],
        'BigBrainIntelligence': ['agents.py'],
        'CentralAccounting': ['financial_analysis_engine.py', 'database.py'],
        'CryptoIntelligence': ['crypto_intelligence_engine.py'],
        'SharedInfrastructure': []  # This directory doesn't exist
    }

    for dept, files in key_files.items():
        if os.path.exists(dept):
            logger.info(f'   {dept}:')
            for file in files:
                file_path = os.path.join(dept, file)
                exists = os.path.exists(file_path)
                status = "✅" if exists else "[CROSS]"
                logger.info(f'     {file}: {status}')
                if not exists:
                    missing_items.append(f'File: {dept}/{file}')
        else:
            logger.info(f'   {dept}: [CROSS] Directory missing')

    # Check config files
    logger.info(f'\n⚙️  Config Files:')
    config_files = ['config/supervisor_config.yaml', 'config/model_risk_caps.json']
    for config in config_files:
        exists = os.path.exists(config)
        status = "✅" if exists else "[CROSS]"
        logger.info(f'   {config}: {status}')
        if not exists:
            missing_items.append(f'Config: {config}')

    # Check bridge issues - these have been implemented in Phase 1
    logger.info(f'\n🌉 Bridge Issues:')
    # Check if bridge components exist (synchronous check)
    bridge_components_exist = True
    try:
        # Test imports
        from shared.crypto_bigbrain_bridge import CryptoBigBrainBridge
        from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
        from BigBrainIntelligence.research_agent import ResearchAgent
        logger.info(f'   Bridge components: ✅ Available')

        # Note: Full async testing would require making this function async
        logger.info(f'   Bridge connections: [WARN]️  Requires async testing')

    except ImportError as e:
        bridge_components_exist = False
        logger.info(f'   Bridge components: [CROSS] Import error: {e}')
        improvement_areas.append('Bridge connections incomplete')

    # Check PowerShell integration
    logger.info(f'\n🔧 PowerShell Integration:')
    ps_files = ['shared/powershell_agent_wrapper.py', 'Doctrine_Implementation/Theater_D.ps1']
    for ps_file in ps_files:
        exists = os.path.exists(ps_file)
        status = "✅" if exists else "[CROSS]"
        logger.info(f'   {ps_file}: {status}')
        if not exists:
            missing_items.append(f'PowerShell: {ps_file}')

    # Check for missing NCC components
    logger.info(f'\n🏛️  NCC Components:')
    ncc_files = ['NCC/NCC-Doctrine/logs/NCC_Agent_Optimization.log']
    for ncc_file in ncc_files:
        exists = os.path.exists(ncc_file)
        status = "✅" if exists else "[CROSS]"
        logger.info(f'   {ncc_file}: {status}')

    logger.info(f'\n📋 MISSING ITEMS ({len(missing_items)}):')
    if missing_items:
        for item in missing_items:
            logger.info(f'   [CROSS] {item}')
    else:
        logger.info('   ✅ None found')

    logger.info(f'\n🔧 IMPROVEMENT AREAS ({len(improvement_areas)}):')
    if improvement_areas:
        for area in improvement_areas:
            logger.info(f'   [WARN]️  {area}')
    else:
        logger.info('   ✅ None identified')

    return missing_items, improvement_areas

async def analyze_system_health():
    """Analyze the overall system health and functionality."""
    logger.info(f'\n🏥 SYSTEM HEALTH ANALYSIS:')
    logger.info('=' * 40)

    try:
        # Test doctrine orchestrator
        from aac.doctrine.doctrine_integration import DoctrineOrchestrator
        orchestrator = DoctrineOrchestrator()
        await orchestrator.initialize()
        logger.info('✅ Doctrine Orchestrator: Working')

        # Test metrics collection
        metrics = await orchestrator.collect_all_metrics()
        logger.info(f'✅ Metrics Collection: {len(metrics)} metrics')

        # Test individual departments
        from CentralAccounting.financial_analysis_engine import get_financial_analysis_engine
        from CryptoIntelligence.crypto_intelligence_engine import get_crypto_intelligence_engine

        fin_engine = await get_financial_analysis_engine()
        crypto_engine = await get_crypto_intelligence_engine()

        fin_metrics = await fin_engine.get_doctrine_metrics()
        crypto_metrics = await crypto_engine.get_doctrine_metrics()

        logger.info(f'✅ Financial Engine: {len(fin_metrics)} metrics')
        logger.info(f'✅ Crypto Engine: {len(crypto_metrics)} metrics')

        # Test research agents
        from BigBrainIntelligence.agents import APIScannerAgent, DataGapFinderAgent
        logger.info('✅ Research Agents: Available')

        # Test gap metrics
        from shared.gap_metrics_collector import GapMetricsCollector
        gap_collector = GapMetricsCollector()
        gap_metrics = await gap_collector.collect_all_gap_metrics()
        logger.info(f'✅ Gap Metrics: {len(gap_metrics)} metrics')

        # Test incident automation
        from shared.incident_postmortem_automation import get_incident_automation
        incident_auto = await get_incident_automation()
        incident_status = await incident_auto.get_monitoring_status()
        status = "Active" if incident_status.get('monitoring_active') else "Inactive"
        logger.info(f'✅ Incident Automation: {status}')

        return True

    except Exception as e:
        logger.info(f'[CROSS] System health check failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def create_improvement_plan(missing_items, improvement_areas):
    """Create a comprehensive improvement plan."""
    logger.info(f'\n📋 COMPREHENSIVE IMPROVEMENT PLAN')
    logger.info('=' * 50)

    plan = {
        "Phase 1: Critical Infrastructure (Week 1)": [
            "Create SharedInfrastructure directory with security and monitoring components",
            "Implement missing config files (supervisor_config.yaml, model_risk_caps.json)",
            "Fix bridge connection issues between departments"
        ],
        "Phase 2: Department Completion (Week 2)": [
            "Complete TradingExecution department with full execution engine",
            "Enhance BigBrainIntelligence with research_agent module",
            "Add missing methods to CryptoIntelligence engine (get_venue_health)",
            "Implement SharedInfrastructure security and incident management"
        ],
        "Phase 3: Integration & Testing (Week 3)": [
            "Complete cross-department bridge connections",
            "Implement department-specific metric attribution",
            "Add comprehensive system health monitoring",
            "Create automated testing suite for all components"
        ],
        "Phase 4: Production Readiness (Week 4)": [
            "Implement production deployment configurations",
            "Add comprehensive logging and alerting",
            "Create operational runbooks and procedures",
            "Perform full system stress testing and validation"
        ]
    }

    for phase, tasks in plan.items():
        logger.info(f'\n🏗️  {phase}')
        for i, task in enumerate(tasks, 1):
            logger.info(f'   {i}. {task}')

    logger.info(f'\n[TARGET] SUCCESS METRICS:')
    logger.info('   • All 5 departments fully implemented with working engines')
    logger.info('   • Cross-department bridges 100% functional')
    logger.info('   • All doctrine packs with proper metric attribution')
    logger.info('   • Zero missing config files or directories')
    logger.info('   • Comprehensive test coverage for all components')
    logger.info('   • Production deployment ready with monitoring')

def main():
    """Main analysis function."""
    missing_items, improvement_areas = analyze_missing_components()

    # Run async system health check
    health_ok = asyncio.run(analyze_system_health())

    create_improvement_plan(missing_items, improvement_areas)

    logger.info(f'\n🏁 FINAL ASSESSMENT:')
    if not missing_items and health_ok:
        logger.info('[CELEBRATION] SYSTEM IS COMPLETE - Ready for production!')
    else:
        logger.info(f'[WARN]️  {len(missing_items)} missing items, {len(improvement_areas)} improvements needed')
        logger.info('📅 Follow the improvement plan to reach production readiness')

if __name__ == "__main__":
    main()