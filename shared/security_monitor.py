"""
Security Monitor Module
=======================
Re-exports SecurityMonitoring from shared.security_framework as SecurityMonitor
for backward compatibility.
"""
from shared.security_framework import SecurityMonitoring as SecurityMonitor, SecurityEvent

__all__ = ['SecurityMonitor', 'SecurityEvent']
