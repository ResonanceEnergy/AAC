"""SharedInfrastructure.health_checker — re-export from shared.health_checker (canonical location)."""
from shared.health_checker import *  # noqa: F401,F403
try:
    from shared.health_checker import HealthChecker
except ImportError:
    pass
