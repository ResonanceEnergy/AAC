#!/usr/bin/env python3
"""
PHASE 1 COMPLETION VERIFICATION
==============================
Final verification that all Phase 1 critical blockers are resolved.
"""

import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from security_dashboard import SecurityDashboard

from shared.audit_trail_integrity import audit_trail_integrity_system
from shared.business_continuity import business_continuity_system
from shared.capital_management import capital_management_system
from shared.compliance_review import compliance_review_system
from shared.risk_disclosure import risk_disclosure_framework
from shared.security_framework import (
    api_security,
    initialize_security_framework,
    rbac,
    security_monitoring,
)


async def initialize_all_systems():
    """Initialize all Phase 1 systems"""
    logger.info("🔧 Initializing All Phase 1 Systems...")
    logger.info("=" * 50)

    # Initialize compliance systems
    logger.info("📋 Initializing Compliance Systems...")

    # Capital Management
    try:
        await capital_management_system.initialize()
        logger.info("✅ Capital Management initialized")
    except Exception as e:
        logger.info(f"❌ Capital Management failed: {e}")

    # Audit Trail Integrity
    try:
        await audit_trail_integrity_system.initialize()
        logger.info("✅ Audit Trail Integrity initialized")
    except Exception as e:
        logger.info(f"❌ Audit Trail Integrity failed: {e}")

    # Risk Disclosure
    try:
        await risk_disclosure_framework.initialize()
        logger.info("✅ Risk Disclosure initialized")
    except Exception as e:
        logger.info(f"❌ Risk Disclosure failed: {e}")

    # Business Continuity
    try:
        await business_continuity_system.initialize()
        logger.info("✅ Business Continuity initialized")
    except Exception as e:
        logger.info(f"❌ Business Continuity failed: {e}")

    # Security Framework
    logger.info("🔒 Initializing Security Framework...")
    try:
        await initialize_security_framework()
        logger.info("✅ Security Framework initialized")
    except Exception as e:
        logger.info(f"❌ Security Framework failed: {e}")

    logger.info("=" * 50)


async def run_phase1_verification():
    """Run complete Phase 1 verification"""
    logger.info("🎯 PHASE 1: CRITICAL BLOCKERS VERIFICATION")
    logger.info("=" * 60)

    # Initialize all systems first
    await initialize_all_systems()

    # Regulatory Compliance Check
    logger.info("📊 REGULATORY COMPLIANCE CHECK:")
    logger.info("-" * 40)

    compliance_report = await compliance_review_system.run_compliance_review()
    compliance_status = compliance_review_system.get_compliance_status()

    overall_compliant = compliance_status.get('overall_compliant', False)
    check_summary = compliance_status.get('check_summary', {})
    passed_checks = check_summary.get('passed_checks', 0)
    total_checks = check_summary.get('total_checks', 0)

    logger.info(f"Overall Compliance: {'✅ PASSED' if overall_compliant else '❌ FAILED'}")
    logger.info(f"Checks Passed: {passed_checks}/{total_checks}")

    # Show failed checks
    check_results = compliance_report.check_results if hasattr(compliance_report, 'check_results') else {}
    failed_checks = []
    for check_id, result in check_results.items():
        if not result.get('passed', False):
            failed_checks.append(f"{check_id}: {result.get('error', 'Failed')}")

    if failed_checks:
        logger.info("Failed Checks:")
        for failure in failed_checks[:3]:  # Show first 3
            logger.info(f"  • {failure}")

    logger.info("")

    # Security Compliance Check
    logger.info("🔒 SECURITY COMPLIANCE CHECK:")
    logger.info("-" * 40)

    dashboard = SecurityDashboard()
    security_report = await dashboard.get_security_status_report()

    security_score = security_report['overall_security_score']
    security_components = len(security_report['security_components'])
    active_alerts = len(security_report['active_alerts'])

    logger.info(f"Security Score: {security_score}/100")
    logger.info(f"Components: {security_components}")
    logger.info(f"Active Alerts: {active_alerts}")

    # Security readiness
    security_ready = security_score >= 70

    logger.info(f"Security Status: {'✅ READY' if security_ready else '❌ NEEDS WORK'}")

    logger.info("")

    # PHASE 1 FINAL STATUS
    logger.info("🎯 PHASE 1 FINAL STATUS:")
    logger.info("=" * 60)

    phase1_complete = overall_compliant and security_ready

    if phase1_complete:
        logger.info("🎉 PHASE 1: COMPLETE!")
        logger.info("✅ All critical blockers resolved")
        logger.info("✅ System production-ready")
        logger.info("")
        logger.info("🚀 READY FOR PHASE 2: FUNCTIONAL COMPLETION")
        logger.info("Next priorities:")
        logger.info("1. ETH_PRIVATE_KEY API integration")
        logger.info("2. BIGBRAIN_AUTH_TOKEN setup")
        logger.info("3. Live exchange connections")
        logger.info("4. Real trading infrastructure")

    else:
        logger.info("⚠️  PHASE 1: INCOMPLETE")
        logger.info("❌ Critical blockers remain")
        logger.info("")
        logger.info("🔧 REMAINING TASKS:")

        if not overall_compliant:
            logger.info("• Fix regulatory compliance issues:")
            for failure in failed_checks:
                logger.info(f"  - {failure}")

        if not security_ready:
            logger.info("• Improve security score:")
            recommendations = security_report['recommendations'][:3]
            for rec in recommendations:
                logger.info(f"  - {rec}")

    logger.info("")
    logger.info("=" * 60)

    return {
        "phase1_complete": phase1_complete,
        "regulatory_compliant": overall_compliant,
        "security_ready": security_ready,
        "compliance_score": f"{passed_checks}/{total_checks}",
        "security_score": security_score
    }


if __name__ == "__main__":
    asyncio.run(run_phase1_verification())
