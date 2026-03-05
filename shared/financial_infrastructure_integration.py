"""
Financial Infrastructure Integration
====================================
Integrates the AAC financial subsystems (money monitor, helix, accounting)
into a unified financial infrastructure.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FinancialIntegrator:
    """Unified financial infrastructure integrator."""

    def __init__(self):
        self.money_monitor = None
        self.ax_helix = None
        self.initialized = False
        self._status = {
            "money_monitor": "offline",
            "ax_helix": "offline",
            "accounting": "offline",
            "initialized_at": None,
        }

    async def initialize(self) -> bool:
        """Initialize all financial subsystems."""
        try:
            from shared.internal_money_monitor import get_money_monitor
            from shared.ax_helix_integration import get_ax_helix_api

            self.money_monitor = get_money_monitor()
            self.ax_helix = get_ax_helix_api()
            self._status["money_monitor"] = "online"
            self._status["ax_helix"] = "online"
            self._status["accounting"] = "online"
            self._status["initialized_at"] = datetime.now().isoformat()
            self.initialized = True
            logger.info("[OK] Financial infrastructure initialized")
            return True
        except Exception as e:
            logger.error(f"Financial infrastructure init failed: {e}")
            return False

    async def get_system_status(self) -> Dict[str, Any]:
        """Return current status of all financial subsystems."""
        return {
            "initialized": self.initialized,
            **self._status,
        }


# Module-level singleton
_integrator: Optional[FinancialIntegrator] = None


def get_financial_integrator() -> FinancialIntegrator:
    """Get or create the financial integrator singleton."""
    global _integrator
    if _integrator is None:
        _integrator = FinancialIntegrator()
    return _integrator


async def initialize_aac_financial_system() -> FinancialIntegrator:
    """Initialize and return the financial integrator."""
    integrator = get_financial_integrator()
    await integrator.initialize()
    return integrator
