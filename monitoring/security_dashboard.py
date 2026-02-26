#!/usr/bin/env python3
"""
Security Status Dashboard
========================
Real-time security monitoring and status reporting for production deployment.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.security_framework import (
    rbac, api_security, security_monitoring, advanced_encryption
)
from shared.audit_logger import get_audit_logger


class SecurityDashboard:
    """Security status dashboard for monitoring and reporting"""

    def __init__(self):
        self.logger = logging.getLogger("SecurityDashboard")
        self.audit_logger = get_audit_logger()

    async def get_security_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive security status report"""
        report = {
            "timestamp": datetime.now(),
            "overall_security_score": 0,
            "security_components": {},
            "active_alerts": [],
            "recent_events": [],
            "recommendations": []
        }

        # MFA Status
        mfa_status = self._check_mfa_status()
        report["security_components"]["mfa"] = mfa_status

        # Encryption Status
        encryption_status = self._check_encryption_status()
        report["security_components"]["encryption"] = encryption_status

        # RBAC Status
        rbac_status = self._check_rbac_status()
        report["security_components"]["rbac"] = rbac_status

        # API Security Status
        api_status = self._check_api_security_status()
        report["security_components"]["api_security"] = api_status

        # Security Monitoring Status
        monitoring_status = security_monitoring.get_security_status()
        report["security_components"]["monitoring"] = monitoring_status

        # Calculate overall security score
        scores = [
            mfa_status["score"],
            encryption_status["score"],
            rbac_status["score"],
            api_status["score"],
            min(100, monitoring_status.get("total_events_last_hour", 0) * -5 + 100)  # Lower events = higher score
        ]
        report["overall_security_score"] = sum(scores) // len(scores)

        # Get active alerts
        report["active_alerts"] = monitoring_status.get("alerts", [])

        # Get recent security events (last 24 hours)
        recent_events = [
            {
                "event_type": event.event_type,
                "severity": event.severity,
                "timestamp": event.timestamp,
                "user_id": event.user_id
            }
            for event in security_monitoring.security_events
            if (datetime.now() - event.timestamp).total_seconds() < 86400  # 24 hours
        ]
        report["recent_events"] = recent_events[-10:]  # Last 10 events

        # Generate recommendations
        report["recommendations"] = self._generate_security_recommendations(report)

        return report

    def _check_mfa_status(self) -> Dict[str, Any]:
        """Check MFA implementation status"""
        total_users = len(rbac.users)
        mfa_enabled_users = len([u for u in rbac.users.values() if u.mfa_enabled])

        status = {
            "total_users": total_users,
            "mfa_enabled_users": mfa_enabled_users,
            "mfa_coverage": (mfa_enabled_users / total_users * 100) if total_users > 0 else 0,
            "status": "good" if mfa_enabled_users > 0 else "needs_improvement",
            "score": min(100, mfa_enabled_users * 20)  # 5 users with MFA = 100%
        }

        return status

    def _check_encryption_status(self) -> Dict[str, Any]:
        """Check encryption system status"""
        # Check if master key exists and is accessible
        key_exists = advanced_encryption.master_key is not None

        # Test encryption/decryption
        test_success = False
        try:
            test_data = "security_test_data"
            encrypted = advanced_encryption.encrypt_data(test_data)
            decrypted = advanced_encryption.decrypt_data(encrypted)
            test_success = decrypted == test_data
        except Exception:
            test_success = False

        status = {
            "master_key_configured": key_exists,
            "encryption_working": test_success,
            "status": "good" if key_exists and test_success else "critical",
            "score": 100 if key_exists and test_success else 0
        }

        return status

    def _check_rbac_status(self) -> Dict[str, Any]:
        """Check RBAC system status"""
        total_users = len(rbac.users)
        total_roles = len(rbac.roles)
        privileged_users = len([u for u in rbac.users.values() if u.role in ["admin", "super_admin"]])

        status = {
            "total_users": total_users,
            "total_roles": total_roles,
            "privileged_users": privileged_users,
            "status": "good" if total_users > 0 and total_roles >= 5 else "needs_setup",
            "score": min(100, total_users * 10 + total_roles * 10)  # Basic scoring
        }

        return status

    def _check_api_security_status(self) -> Dict[str, Any]:
        """Check API security status"""
        total_keys = len(api_security.api_keys)
        active_keys = len([k for k in api_security.api_keys.values()
                          if not k.expires_at or k.expires_at > datetime.now()])

        # Check rate limiting effectiveness
        rate_limit_hits = sum(1 for limits in api_security.rate_limits.values()
                            if limits.get("blocked_until") and
                            limits["blocked_until"] > datetime.now())

        status = {
            "total_api_keys": total_keys,
            "active_api_keys": active_keys,
            "rate_limit_blocks": rate_limit_hits,
            "status": "good" if active_keys > 0 else "needs_setup",
            "score": min(100, active_keys * 25)  # 4 active keys = 100%
        }

        return status

    def _generate_security_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on current status"""
        recommendations = []

        # MFA recommendations
        mfa_status = report["security_components"]["mfa"]
        if mfa_status["mfa_coverage"] < 50:
            recommendations.append("Enable MFA for all users to improve authentication security")
        if mfa_status["mfa_enabled_users"] == 0:
            recommendations.append("URGENT: No users have MFA enabled - critical security risk")

        # Encryption recommendations
        encryption_status = report["security_components"]["encryption"]
        if not encryption_status["master_key_configured"]:
            recommendations.append("URGENT: Master encryption key not configured")
        if not encryption_status["encryption_working"]:
            recommendations.append("URGENT: Encryption system malfunctioning")

        # RBAC recommendations
        rbac_status = report["security_components"]["rbac"]
        if rbac_status["total_users"] == 0:
            recommendations.append("No users configured in RBAC system")
        if rbac_status["privileged_users"] > rbac_status["total_users"] * 0.2:
            recommendations.append("High ratio of privileged users - review principle of least privilege")

        # API Security recommendations
        api_status = report["security_components"]["api_security"]
        if api_status["total_api_keys"] == 0:
            recommendations.append("No API keys configured - API access unavailable")
        if api_status["rate_limit_blocks"] > 5:
            recommendations.append("High rate limit blocks detected - investigate potential attacks")

        # Overall score recommendations
        if report["overall_security_score"] < 60:
            recommendations.append("URGENT: Overall security score below acceptable threshold")
        elif report["overall_security_score"] < 80:
            recommendations.append("Security score needs improvement - address outstanding issues")

        # Alert-based recommendations
        if report["active_alerts"]:
            recommendations.append(f"Active security alerts: {len(report['active_alerts'])} - investigate immediately")

        return recommendations

    async def export_security_report(self, format: str = "json") -> str:
        """Export security report in specified format"""
        report = await self.get_security_status_report()

        if format.lower() == "json":
            import json
            return json.dumps(report, default=str, indent=2)
        elif format.lower() == "text":
            return self._format_text_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format security report as human-readable text"""
        lines = []
        lines.append("AAC Security Status Report")
        lines.append("=" * 40)
        lines.append(f"Generated: {report['timestamp']}")
        lines.append(f"Overall Security Score: {report['overall_security_score']}/100")
        lines.append("")

        lines.append("Security Components:")
        for component, status in report["security_components"].items():
            lines.append(f"  {component}: {status.get('status', 'unknown')} (Score: {status.get('score', 0)})")

        lines.append("")
        lines.append(f"Active Alerts: {len(report['active_alerts'])}")
        for alert in report["active_alerts"]:
            lines.append(f"  - {alert['event_type']}: {alert.get('count', 0)} occurrences")

        lines.append("")
        lines.append(f"Recent Events (last 24h): {len(report['recent_events'])}")
        for event in report["recent_events"][-5:]:  # Show last 5
            lines.append(f"  - {event['timestamp']}: {event['event_type']} ({event['severity']})")

        lines.append("")
        lines.append("Security Recommendations:")
        for rec in report["recommendations"]:
            lines.append(f"  • {rec}")

        return "\n".join(lines)


async def display_security_dashboard():
    """Display the security dashboard"""
    dashboard = SecurityDashboard()

    print("🔒 AAC Security Dashboard")
    print("=" * 50)

    # Get status report
    report = await dashboard.get_security_status_report()

    print(f"📊 Overall Security Score: {report['overall_security_score']}/100")
    print()

    # Component status
    print("🛡️  Security Components:")
    for component, status in report["security_components"].items():
        score = status.get("score", 0)
        comp_status = status.get("status", "unknown")
        status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        print(f"  {status_icon} {component}: {comp_status} ({score}%)")

    print()

    # Active alerts
    alerts = report["active_alerts"]
    if alerts:
        print(f"🚨 Active Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  • {alert['event_type']}: {alert.get('count', 0)} events")
    else:
        print("✅ No active security alerts")

    print()

    # Recent events
    events = report["recent_events"]
    print(f"📋 Recent Security Events (24h): {len(events)}")
    for event in events[-3:]:  # Show last 3
        print(f"  • {event['timestamp'].strftime('%H:%M')}: {event['event_type']}")

    print()

    # Recommendations
    recommendations = report["recommendations"]
    if recommendations:
        print("💡 Security Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("✅ All security checks passed - no recommendations")

    print()
    print("=" * 50)

    # Export option
    export = input("Export detailed report? (json/text/no): ").lower().strip()
    if export in ["json", "text"]:
        report_content = await dashboard.export_security_report(export)
        filename = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export}"
        with open(filename, 'w') as f:
            f.write(report_content)
        print(f"Report exported to: {filename}")


if __name__ == "__main__":
    asyncio.run(display_security_dashboard())