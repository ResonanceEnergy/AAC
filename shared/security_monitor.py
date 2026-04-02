"""
Security Monitor Module
=======================
Re-exports SecurityMonitoring from shared.security_framework as SecurityMonitor
for backward compatibility.
"""
from shared.security_framework import SecurityEvent
from shared.security_framework import SecurityMonitoring as SecurityMonitor

__all__ = ['SecurityMonitor', 'SecurityEvent']
