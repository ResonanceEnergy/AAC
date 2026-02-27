"""SharedInfrastructure.alert_manager — re-export from shared.alert_manager (canonical location)."""
from shared.alert_manager import *  # noqa: F401,F403
try:
    from shared.alert_manager import AlertManager
except ImportError:
    pass
