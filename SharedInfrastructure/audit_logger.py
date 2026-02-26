"""
Shared Infrastructure - Audit Logger
Comprehensive audit logging system for the ACC platform.
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEntry:
    timestamp: str
    component: str
    action: str
    details: str
    level: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AuditLogger:
    """
    Centralized audit logging system for compliance and security monitoring.
    """

    def __init__(self, log_directory: str = "NCC/NCC-Doctrine/logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("AuditLogger")
        self.logger.setLevel(logging.DEBUG)

        # File handler for audit logs
        audit_log_path = self.log_directory / "audit.log"
        file_handler = logging.FileHandler(audit_log_path)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # In-memory buffer for recent entries
        self.recent_entries: list[AuditEntry] = []
        self.max_buffer_size = 1000

        # Compliance settings
        self.compliance_retention_days = 2555  # 7 years for financial records
        self.log_rotation_enabled = True

    def log_event(self, component: str, action: str, details: str,
                  level: str = "info", user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log an audit event.

        Args:
            component: The system component generating the event
            action: The action being performed
            details: Detailed description of the event
            level: Log level (debug, info, warning, error, critical)
            user_id: User ID if applicable
            session_id: Session ID if applicable
            ip_address: IP address if applicable
            metadata: Additional metadata
        """
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            component=component,
            action=action,
            details=details,
            level=level,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            metadata=metadata or {}
        )

        # Add to buffer
        self.recent_entries.append(entry)
        if len(self.recent_entries) > self.max_buffer_size:
            self.recent_entries.pop(0)

        # Write to file
        self._write_to_file(entry)

        # Log to standard logger
        log_method = getattr(self.logger, level, self.logger.info)
        log_message = f"{component} | {action} | {details}"
        if user_id:
            log_message += f" | user:{user_id}"
        log_method(log_message)

    def _write_to_file(self, entry: AuditEntry):
        """Write audit entry to file."""
        try:
            audit_file = self.log_directory / "audit.log"
            with open(audit_file, 'a', encoding='utf-8') as f:
                json_entry = json.dumps(asdict(entry), ensure_ascii=False)
                f.write(json_entry + '\n')
        except Exception as e:
            # Fallback logging if file write fails
            print(f"Failed to write audit log: {e}")

    async def get_recent_entries(self, limit: int = 100,
                                component: Optional[str] = None,
                                level: Optional[str] = None) -> list[AuditEntry]:
        """
        Get recent audit entries with optional filtering.

        Args:
            limit: Maximum number of entries to return
            component: Filter by component
            level: Filter by log level

        Returns:
            List of audit entries
        """
        entries = self.recent_entries[-limit:]

        if component:
            entries = [e for e in entries if e.component == component]

        if level:
            entries = [e for e in entries if e.level == level]

        return entries

    async def search_entries(self, query: str, limit: int = 50) -> list[AuditEntry]:
        """
        Search audit entries by text query.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum results to return

        Returns:
            Matching audit entries
        """
        query_lower = query.lower()
        matches = []

        # Search recent entries first
        for entry in reversed(self.recent_entries):
            if (query_lower in entry.component.lower() or
                query_lower in entry.action.lower() or
                query_lower in entry.details.lower()):
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    async def get_compliance_report(self, start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for the specified date range.

        Args:
            start_date: Start date for the report
            end_date: End date for the report

        Returns:
            Compliance report data
        """
        # Filter entries by date range
        relevant_entries = [
            entry for entry in self.recent_entries
            if start_date <= datetime.fromisoformat(entry.timestamp) <= end_date
        ]

        # Analyze entries
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_entries": len(relevant_entries),
                "error_count": len([e for e in relevant_entries if e.level == "error"]),
                "critical_count": len([e for e in relevant_entries if e.level == "critical"]),
                "security_events": len([e for e in relevant_entries if "security" in e.component.lower()])
            },
            "components": {},
            "actions": {}
        }

        # Component breakdown
        for entry in relevant_entries:
            report["components"][entry.component] = report["components"].get(entry.component, 0) + 1
            report["actions"][entry.action] = report["actions"].get(entry.action, 0) + 1

        return report

    async def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        try:
            current_time = datetime.now()
            retention_cutoff = current_time.replace(
                year=current_time.year - (self.compliance_retention_days // 365)
            )

            # List all log files
            log_files = list(self.log_directory.glob("*.log"))

            for log_file in log_files:
                if log_file.stat().st_mtime < retention_cutoff.timestamp():
                    log_file.unlink()
                    self.log_event(
                        "audit_logger",
                        "log_cleanup",
                        f"Deleted old log file: {log_file.name}",
                        "info"
                    )

        except Exception as e:
            self.log_event(
                "audit_logger",
                "cleanup_error",
                f"Failed to cleanup old logs: {str(e)}",
                "error"
            )

    def get_log_stats(self) -> Dict[str, Any]:
        """Get current logging statistics."""
        return {
            "buffer_size": len(self.recent_entries),
            "max_buffer_size": self.max_buffer_size,
            "log_directory": str(self.log_directory),
            "retention_days": self.compliance_retention_days,
            "log_files": len(list(self.log_directory.glob("*.log")))
        }

# Global audit logger instance
audit_logger = AuditLogger()

def get_audit_logger():
    """Get the global audit logger instance."""
    return audit_logger