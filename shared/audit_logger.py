#!/usr/bin/env python3
"""
Audit Logger
============
Comprehensive audit logging for API calls, security events, and compliance tracking.
"""

import asyncio
import json
import logging
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
from contextlib import contextmanager


class AuditCategory(str, Enum):
    """Categories for audit events"""
    API_CALL = "api_call"
    ORDER = "order"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    ERROR = "error"
    SYSTEM = "system"
    TRADE = "trade"
    BALANCE = "balance"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    timestamp: datetime
    category: AuditCategory
    severity: AuditSeverity
    action: str
    user: str  # System component or user identifier
    resource: str  # What was accessed/modified
    status: str  # success, failure, pending
    details: Dict[str, Any] = field(default_factory=dict)
    request_data: Optional[Dict] = None  # Sanitized request
    response_summary: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    ip_address: Optional[str] = None
    exchange: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AuditLogger:
    """
    Comprehensive audit logging system with multiple backends.
    
    Features:
    - File-based logging with rotation
    - SQLite database storage for queries
    - Sensitive data redaction
    - Correlation ID tracking
    - Performance metrics
    """
    
    # Fields that should never be logged
    SENSITIVE_FIELDS = {
        'api_key', 'api_secret', 'secret', 'password', 'passphrase',
        'private_key', 'token', 'authorization', 'auth_token',
        'access_token', 'refresh_token', 'session_id', 'secret_key',
    }
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs/audit",
        db_path: Optional[Union[str, Path]] = None,
        enable_file_logging: bool = True,
        enable_db_logging: bool = True,
        max_file_size_mb: int = 100,
        retention_days: int = 90,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path) if db_path else self.log_dir / "audit.db"
        self.enable_file_logging = enable_file_logging
        self.enable_db_logging = enable_db_logging
        self.max_file_size_mb = max_file_size_mb
        self.retention_days = retention_days
        
        self.logger = logging.getLogger("AuditLogger")
        self._event_count = 0
        self._lock = asyncio.Lock()
        
        # Initialize backends
        if enable_db_logging:
            self._init_database()
        if enable_file_logging:
            self._init_file_handler()
    
    def _init_database(self):
        """Initialize SQLite database for audit storage"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user TEXT,
                    resource TEXT,
                    status TEXT,
                    exchange TEXT,
                    correlation_id TEXT,
                    duration_ms REAL,
                    error_message TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON audit_events(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON audit_events(exchange)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlation ON audit_events(correlation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON audit_events(status)")
            conn.commit()
    
    def _init_file_handler(self):
        """Initialize file handler for audit logs"""
        from logging.handlers import RotatingFileHandler
        
        audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m')}.log"
        self._file_handler = RotatingFileHandler(
            audit_file,
            maxBytes=self.max_file_size_mb * 1024 * 1024,
            backupCount=12,  # Keep 12 months
        )
        self._file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.logger.addHandler(self._file_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def close(self):
        """Close all resources - important for cleanup on Windows"""
        if hasattr(self, '_file_handler') and self._file_handler:
            self._file_handler.close()
            self.logger.removeHandler(self._file_handler)
            self._file_handler = None
    
    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self._event_count += 1
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"AUD-{timestamp}-{self._event_count:06d}"
    
    def _sanitize_data(self, data: Any, depth: int = 0) -> Any:
        """
        Recursively sanitize sensitive data from logs.
        
        Redacts API keys, secrets, and other sensitive fields.
        """
        if depth > 10:  # Prevent infinite recursion
            return "[MAX_DEPTH]"
        
        if data is None:
            return None
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                lower_key = key.lower()
                # Check if key contains sensitive patterns
                if any(sensitive in lower_key for sensitive in self.SENSITIVE_FIELDS):
                    if isinstance(value, str) and len(value) > 0:
                        sanitized[key] = f"[REDACTED-{len(value)}chars]"
                    else:
                        sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_data(value, depth + 1)
            return sanitized
        
        elif isinstance(data, list):
            return [self._sanitize_data(item, depth + 1) for item in data]
        
        elif isinstance(data, str):
            # Check for patterns that look like API keys
            if len(data) > 20 and data.isalnum():
                return f"[POSSIBLE_KEY-{len(data)}chars]"
            return data
        
        else:
            return data
    
    async def log_event(
        self,
        category: AuditCategory,
        action: str,
        resource: str,
        status: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        user: str = "system",
        details: Optional[Dict] = None,
        request_data: Optional[Dict] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        exchange: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            category: Event category (API_CALL, ORDER, etc.)
            action: What action was performed
            resource: What resource was accessed/modified
            status: success, failure, pending
            severity: Event severity level
            user: System component or user identifier
            details: Additional event details
            request_data: Sanitized request data
            response_summary: Brief summary of response
            error_message: Error message if applicable
            duration_ms: Operation duration in milliseconds
            exchange: Exchange name if applicable
            correlation_id: ID to correlate related events
        
        Returns:
            Created AuditEvent
        """
        async with self._lock:
            event = AuditEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                category=category,
                severity=severity,
                action=action,
                user=user,
                resource=resource,
                status=status,
                details=self._sanitize_data(details) if details else {},
                request_data=self._sanitize_data(request_data) if request_data else None,
                response_summary=response_summary[:500] if response_summary else None,
                error_message=error_message[:1000] if error_message else None,
                duration_ms=duration_ms,
                exchange=exchange,
                correlation_id=correlation_id,
            )
            
            # Write to backends
            if self.enable_file_logging:
                self._write_to_file(event)
            
            if self.enable_db_logging:
                self._write_to_database(event)
            
            return event
    
    def _write_to_file(self, event: AuditEvent):
        """Write event to log file"""
        log_message = f"{event.category.value} | {event.action} | {event.resource} | {event.status}"
        if event.exchange:
            log_message += f" | {event.exchange}"
        if event.duration_ms:
            log_message += f" | {event.duration_ms:.2f}ms"
        if event.error_message:
            log_message += f" | ERROR: {event.error_message}"
        
        level = getattr(logging, event.severity.value.upper(), logging.INFO)
        self.logger.log(level, log_message)
    
    def _write_to_database(self, event: AuditEvent):
        """Write event to SQLite database"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, category, severity, action,
                        user, resource, status, exchange, correlation_id,
                        duration_ms, error_message, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.category.value,
                    event.severity.value,
                    event.action,
                    event.user,
                    event.resource,
                    event.status,
                    event.exchange,
                    event.correlation_id,
                    event.duration_ms,
                    event.error_message,
                    json.dumps(event.details) if event.details else None,
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to write to database: {e}")
    
    # Convenience methods for common event types
    
    async def log_api_call(
        self,
        exchange: str,
        endpoint: str,
        method: str = "GET",
        status: str = "success",
        duration_ms: Optional[float] = None,
        request_data: Optional[Dict] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an exchange API call"""
        return await self.log_event(
            category=AuditCategory.API_CALL,
            action=f"{method} {endpoint}",
            resource=f"{exchange}:{endpoint}",
            status=status,
            severity=AuditSeverity.ERROR if status == "failure" else AuditSeverity.DEBUG,
            user=f"connector:{exchange}",
            details={"method": method, "endpoint": endpoint},
            request_data=request_data,
            response_summary=response_summary,
            error_message=error_message,
            duration_ms=duration_ms,
            exchange=exchange,
            correlation_id=correlation_id,
        )
    
    async def log_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: str = "created",
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an order event"""
        severity = AuditSeverity.ERROR if "fail" in status.lower() else AuditSeverity.INFO
        return await self.log_event(
            category=AuditCategory.ORDER,
            action=f"{side} {order_type}",
            resource=f"{exchange}:{symbol}",
            status=status,
            severity=severity,
            user="trading_engine",
            details={
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
            },
            error_message=error_message,
            exchange=exchange,
            correlation_id=correlation_id,
        )
    
    async def log_authentication(
        self,
        exchange: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """Log authentication events"""
        severity = AuditSeverity.WARNING if status == "failure" else AuditSeverity.INFO
        return await self.log_event(
            category=AuditCategory.AUTHENTICATION,
            action="authenticate",
            resource=exchange,
            status=status,
            severity=severity,
            user="connector",
            error_message=error_message,
            exchange=exchange,
        )
    
    async def log_security_event(
        self,
        action: str,
        resource: str,
        status: str,
        details: Optional[Dict] = None,
        severity: AuditSeverity = AuditSeverity.WARNING,
    ) -> AuditEvent:
        """Log security-related events"""
        return await self.log_event(
            category=AuditCategory.SECURITY,
            action=action,
            resource=resource,
            status=status,
            severity=severity,
            user="security_monitor",
            details=details,
        )
    
    # Query methods
    
    def query_events(
        self,
        category: Optional[AuditCategory] = None,
        exchange: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Query audit events from database.
        
        Args:
            category: Filter by category
            exchange: Filter by exchange
            status: Filter by status
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results to return
        
        Returns:
            List of event dictionaries
        """
        if not self.enable_db_logging:
            return []
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
        
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return []
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get audit statistics for a time period"""
        if not self.enable_db_logging:
            return {}
        
        time_filter = ""
        params = []
        
        if start_time:
            time_filter += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            time_filter += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        try:
            with self._get_db_connection() as conn:
                # Total events
                total = conn.execute(
                    f"SELECT COUNT(*) FROM audit_events WHERE 1=1 {time_filter}",
                    params
                ).fetchone()[0]
                
                # By category
                by_category = {}
                cursor = conn.execute(
                    f"SELECT category, COUNT(*) FROM audit_events WHERE 1=1 {time_filter} GROUP BY category",
                    params
                )
                for row in cursor:
                    by_category[row[0]] = row[1]
                
                # By status
                by_status = {}
                cursor = conn.execute(
                    f"SELECT status, COUNT(*) FROM audit_events WHERE 1=1 {time_filter} GROUP BY status",
                    params
                )
                for row in cursor:
                    by_status[row[0]] = row[1]
                
                # By exchange
                by_exchange = {}
                cursor = conn.execute(
                    f"SELECT exchange, COUNT(*) FROM audit_events WHERE exchange IS NOT NULL {time_filter} GROUP BY exchange",
                    params
                )
                for row in cursor:
                    by_exchange[row[0]] = row[1]
                
                # Average duration
                avg_duration = conn.execute(
                    f"SELECT AVG(duration_ms) FROM audit_events WHERE duration_ms IS NOT NULL {time_filter}",
                    params
                ).fetchone()[0]
                
                return {
                    "total_events": total,
                    "by_category": by_category,
                    "by_status": by_status,
                    "by_exchange": by_exchange,
                    "average_duration_ms": round(avg_duration, 2) if avg_duration else None,
                }
        except Exception as e:
            self.logger.error(f"Statistics query failed: {e}")
            return {}


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    log_dir: str = "logs/audit",
    enable_db: bool = True,
    enable_file: bool = True,
) -> AuditLogger:
    """Configure the global audit logger"""
    global _audit_logger
    _audit_logger = AuditLogger(
        log_dir=log_dir,
        enable_db_logging=enable_db,
        enable_file_logging=enable_file,
    )
    return _audit_logger
