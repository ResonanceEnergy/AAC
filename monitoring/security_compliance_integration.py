#!/usr/bin/env python3
"""
Security Compliance Integration
==============================
Integrates security framework with compliance system for Phase 1 completion.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.compliance_review import compliance_review_system
from shared.security_framework import (
    rbac, api_security, security_monitoring, advanced_encryption
)
from security_dashboard import SecurityDashboard


async def run_security_compliance_check():
    """Run comprehensive security compliance check"""
    print("[SECURITY] Running Security Compliance Check...")
    print("=" * 60)

    # Initialize security dashboard
    dashboard = SecurityDashboard()
    security_report = await dashboard.get_security_status_report()

    print("🔒 Security Framework Status:")
    print(f"  Overall Security Score: {security_report['overall_security_score']}/100")
    print(f"  Components: {len(security_report['security_components'])}")
    print(f"  Active Alerts: {len(security_report['active_alerts'])}")
    print(f"  Recommendations: {len(security_report['recommendations'])}")
    print()

    # Check individual security components
    security_checks = {
        "mfa_implemented": check_mfa_implementation(),
        "encryption_active": check_encryption_system(),
        "rbac_configured": check_rbac_system(),
        "api_security_enabled": check_api_security(),
        "monitoring_active": check_security_monitoring()
    }

    passed_checks = 0
    total_checks = len(security_checks)

    print("🛡️  Security Component Checks:")
    for check_name, check_result in security_checks.items():
        status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
        print(f"  {status} {check_name}: {check_result['message']}")
        if check_result["passed"]:
            passed_checks += 1

    print()
    print(f"Security Checks: {passed_checks}/{total_checks} PASSED")

    # Security compliance score
    security_compliance = (passed_checks / total_checks) * 100
    print(f"Security Compliance Score: {security_compliance:.1f}%")
    # Overall system readiness
    overall_ready = security_compliance >= 80 and security_report['overall_security_score'] >= 70

    print()
    if overall_ready:
        print("🎉 SECURITY COMPLIANCE: PASSED")
        print("✅ System is security-hardened and production-ready")
    else:
        print("⚠️  SECURITY COMPLIANCE: NEEDS IMPROVEMENT")
        print("❌ Address security issues before production deployment")

    print()
    print("📋 Security Recommendations:")
    for rec in security_report['recommendations'][:5]:  # Show top 5
        print(f"  • {rec}")

    print()
    print("=" * 60)

    return {
        "security_compliance_score": security_compliance,
        "security_checks_passed": passed_checks,
        "security_checks_total": total_checks,
        "overall_security_score": security_report['overall_security_score'],
        "system_ready": overall_ready
    }


def check_mfa_implementation():
    """Check MFA implementation status"""
    total_users = len(rbac.users)
    mfa_users = len([u for u in rbac.users.values() if u.mfa_enabled])

    if total_users == 0:
        return {"passed": False, "message": "No users configured"}

    coverage = (mfa_users / total_users) * 100
    passed = coverage >= 50  # At least 50% MFA coverage

    return {
        "passed": passed,
        "message": f"MFA coverage: {mfa_users}/{total_users} users ({coverage:.1f}%)"
    }


def check_encryption_system():
    """Check encryption system status"""
    key_exists = advanced_encryption.master_key is not None

    # Test encryption
    test_passed = False
    try:
        test_data = "encryption_test"
        encrypted = advanced_encryption.encrypt_data(test_data)
        decrypted = advanced_encryption.decrypt_data(encrypted)
        test_passed = decrypted == test_data
    except Exception:
        test_passed = False

    passed = key_exists and test_passed

    status = []
    if key_exists:
        status.append("key configured")
    else:
        status.append("no master key")

    if test_passed:
        status.append("encryption working")
    else:
        status.append("encryption failed")

    return {
        "passed": passed,
        "message": f"Encryption: {', '.join(status)}"
    }


def check_rbac_system():
    """Check RBAC system configuration"""
    users_count = len(rbac.users)
    roles_count = len(rbac.roles)

    # Basic requirements: at least 1 user, 5 roles (our default set)
    passed = users_count >= 1 and roles_count >= 5

    return {
        "passed": passed,
        "message": f"RBAC: {users_count} users, {roles_count} roles"
    }


def check_api_security():
    """Check API security configuration"""
    keys_count = len(api_security.api_keys)
    active_keys = len([k for k in api_security.api_keys.values()
                      if not k.expires_at or k.expires_at > datetime.now()])

    # Require at least 1 active API key
    passed = active_keys >= 1

    return {
        "passed": passed,
        "message": f"API Keys: {active_keys}/{keys_count} active"
    }


def check_security_monitoring():
    """Check security monitoring status"""
    # Check if monitoring system is initialized and working
    status = security_monitoring.get_security_status()

    events_count = status.get("total_events_last_hour", 0)
    alerts_count = status.get("active_alerts", 0)

    # Monitoring is working if we can get status
    passed = isinstance(status, dict) and "total_events_last_hour" in status

    return {
        "passed": passed,
        "message": f"Monitoring: {events_count} events/hour, {alerts_count} alerts"
    }


async def integrate_security_with_compliance():
    """Integrate security checks with main compliance system"""
    print("[INTEGRATION] Integrating Security with Compliance System...")

    # Run security compliance check
    security_results = await run_security_compliance_check()

    # Get current compliance status
    compliance_report = await compliance_review_system.run_compliance_review()
    compliance_status = compliance_review_system.get_compliance_status()

    print("\n📊 Integrated Compliance Status:")
    print("=" * 40)

    # Combine results
    overall_compliance = compliance_status.get('overall_compliant', False)
    security_compliance = security_results['system_ready']

    print(f"Regulatory Compliance: {'✅ PASSED' if overall_compliance else '❌ FAILED'}")
    print(f"Security Compliance: {'✅ PASSED' if security_compliance else '❌ FAILED'}")

    # Phase 1 completion status
    phase1_complete = overall_compliance and security_compliance

    print()
    if phase1_complete:
        print("🎯 PHASE 1: CRITICAL BLOCKERS - COMPLETE!")
        print("✅ System is production-ready for live trading")
        print("✅ All regulatory and security requirements met")
    else:
        print("⚠️  PHASE 1: INCOMPLETE")
        if not overall_compliance:
            print("❌ Regulatory compliance issues remain")
        if not security_compliance:
            print("❌ Security hardening incomplete")

    print()
    print("📈 Next Steps:")
    if phase1_complete:
        print("1. Proceed to Phase 2: API Integration")
        print("2. Add critical trading APIs (ETH_PRIVATE_KEY, etc.)")
        print("3. Implement live exchange connections")
    else:
        print("1. Address remaining compliance issues")
        print("2. Complete security hardening")
        print("3. Re-run compliance checks")

    return phase1_complete


if __name__ == "__main__":
    asyncio.run(integrate_security_with_compliance())