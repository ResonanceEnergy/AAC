"""SharedInfrastructure.incident_manager — re-export from shared.incident_manager (canonical location)."""
from shared.incident_manager import *  # noqa: F401,F403
try:
    from shared.incident_manager import IncidentManager
except ImportError:
    pass
