#!/usr/bin/env python3
"""
PHASE 1 COMPLETION VERIFICATION
==============================
Final verification that all Phase 1 critical blockers are resolved.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.compliance_review import compliance_review_system
from shared.capital_management import capital_management_system
from shared.audit_trail_integrity import audit_trail_integrity_system
from shared.risk_disclosure import risk_disclosure_framework
from shared.business_continuity import business_continuity_system
from shared.security_framework import (
    initialize_security_framework, rbac, api_security, security_monitoring
)
from security_dashboard import SecurityDashboard


async def initialize_all_systems():
    """Initialize all Phase 1 systems"""
    print("🔧 Initializing All Phase 1 Systems...")
    print("=" * 50)

    # Initialize compliance systems
    print("📋 Initializing Compliance Systems...")

    # Capital Management
    try:
        await capital_management_system.initialize()
        print("✅ Capital Management initialized")
    except Exception as e:
        print(f"❌ Capital Management failed: {e}")

    # Audit Trail Integrity
    try:
        await audit_trail_integrity_system.initialize()
        print("✅ Audit Trail Integrity initialized")
    except Exception as e:
        print(f"❌ Audit Trail Integrity failed: {e}")

    # Risk Disclosure
    try:
        await risk_disclosure_framework.initialize()
        print("✅ Risk Disclosure initialized")
    except Exception as e:
        print(f"❌ Risk Disclosure failed: {e}")

    # Business Continuity
    try:
        await business_continuity_system.initialize()
        print("✅ Business Continuity initialized")
    except Exception as e:
        print(f"❌ Business Continuity failed: {e}")

    # Security Framework
    print("🔒 Initializing Security Framework...")
    try:
        await initialize_security_framework()
        print("✅ Security Framework initialized")
    except Exception as e:
        print(f"❌ Security Framework failed: {e}")

    print("=" * 50)


async def run_phase1_verification():
    """Run complete Phase 1 verification"""
    print("🎯 PHASE 1: CRITICAL BLOCKERS VERIFICATION")
    print("=" * 60)

    # Initialize all systems first
    await initialize_all_systems()

    # Regulatory Compliance Check
    print("📊 REGULATORY COMPLIANCE CHECK:")
    print("-" * 40)

    compliance_report = await compliance_review_system.run_compliance_review()
    compliance_status = compliance_review_system.get_compliance_status()

    overall_compliant = compliance_status.get('overall_compliant', False)
    check_summary = compliance_status.get('check_summary', {})
    passed_checks = check_summary.get('passed_checks', 0)
    total_checks = check_summary.get('total_checks', 0)

    print(f"Overall Compliance: {'✅ PASSED' if overall_compliant else '❌ FAILED'}")
    print(f"Checks Passed: {passed_checks}/{total_checks}")

    # Show failed checks
    check_results = compliance_report.check_results if hasattr(compliance_report, 'check_results') else {}
    failed_checks = []
    for check_id, result in check_results.items():
        if not result.get('passed', False):
            failed_checks.append(f"{check_id}: {result.get('error', 'Failed')}")

    if failed_checks:
        print("Failed Checks:")
        for failure in failed_checks[:3]:  # Show first 3
            print(f"  • {failure}")

    print()

    # Security Compliance Check
    print("🔒 SECURITY COMPLIANCE CHECK:")
    print("-" * 40)

    dashboard = SecurityDashboard()
    security_report = await dashboard.get_security_status_report()

    security_score = security_report['overall_security_score']
    security_components = len(security_report['security_components'])
    active_alerts = len(security_report['active_alerts'])

    print(f"Security Score: {security_score}/100")
    print(f"Components: {security_components}")
    print(f"Active Alerts: {active_alerts}")

    # Security readiness
    security_ready = security_score >= 70

    print(f"Security Status: {'✅ READY' if security_ready else '❌ NEEDS WORK'}")

    print()

    # PHASE 1 FINAL STATUS
    print("🎯 PHASE 1 FINAL STATUS:")
    print("=" * 60)

    phase1_complete = overall_compliant and security_ready

    if phase1_complete:
        print("🎉 PHASE 1: COMPLETE!")
        print("✅ All critical blockers resolved")
        print("✅ System production-ready")
        print()
        print("🚀 READY FOR PHASE 2: FUNCTIONAL COMPLETION")
        print("Next priorities:")
        print("1. ETH_PRIVATE_KEY API integration")
        print("2. BIGBRAIN_AUTH_TOKEN setup")
        print("3. Live exchange connections")
        print("4. Real trading infrastructure")

    else:
        print("⚠️  PHASE 1: INCOMPLETE")
        print("❌ Critical blockers remain")
        print()
        print("🔧 REMAINING TASKS:")

        if not overall_compliant:
            print("• Fix regulatory compliance issues:")
            for failure in failed_checks:
                print(f"  - {failure}")

        if not security_ready:
            print("• Improve security score:")
            recommendations = security_report['recommendations'][:3]
            for rec in recommendations:
                print(f"  - {rec}")

    print()
    print("=" * 60)

    return {
        "phase1_complete": phase1_complete,
        "regulatory_compliant": overall_compliant,
        "security_ready": security_ready,
        "compliance_score": f"{passed_checks}/{total_checks}",
        "security_score": security_score
    }


if __name__ == "__main__":
    asyncio.run(run_phase1_verification())