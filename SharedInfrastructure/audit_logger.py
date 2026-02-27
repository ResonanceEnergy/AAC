"""SharedInfrastructure.audit_logger — re-export from shared.audit_logger (canonical location)."""
from shared.audit_logger import *  # noqa: F401,F403
try:
    from shared.audit_logger import AuditLogger,get_audit_logger,audit_log
except ImportError:
    pass
