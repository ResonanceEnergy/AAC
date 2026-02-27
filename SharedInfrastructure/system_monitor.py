"""SharedInfrastructure.system_monitor — re-export from shared.system_monitor (canonical location)."""
from shared.system_monitor import *  # noqa: F401,F403
try:
    from shared.system_monitor import SystemMonitor
except ImportError:
    pass
