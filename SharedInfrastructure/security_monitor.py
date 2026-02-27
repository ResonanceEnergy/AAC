"""SharedInfrastructure.security_monitor — re-export from shared.security_monitor (canonical location)."""
from shared.security_monitor import *  # noqa: F401,F403
try:
    from shared.security_monitor import SecurityMonitor
except ImportError:
    pass
