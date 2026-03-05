"""SharedInfrastructure.metrics_collector — re-export from shared.metrics_collector (canonical location)."""
from shared.metrics_collector import *  # noqa: F401,F403
try:
    from shared.metrics_collector import get_metrics_collector  # noqa: F401
except ImportError:
    pass
