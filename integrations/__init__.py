"""AAC Integrations Package — OpenClaw, Telegram Bot, Exchange Connectors."""

# Lazy imports to avoid heavy dependency loading at package level


def get_barren_wuffet_skills():
    """Get the 35 BARREN WUFFET OpenClaw skill definitions."""
    from .openclaw_barren_wuffet_skills import BARREN_WUFFET_SKILLS
    return BARREN_WUFFET_SKILLS


def get_telegram_bot():
    """Get the BarrenWuffetTelegramBot class."""
    from .barren_wuffet_telegram_bot import BarrenWuffetTelegramBot
    return BarrenWuffetTelegramBot


def get_gateway_bridge():
    """Get the OpenClawGatewayBridge class."""
    from .openclaw_gateway_bridge import OpenClawGatewayBridge
    return OpenClawGatewayBridge


def get_az_supreme_handler():
    """Get the AZ Supreme handler module."""
    from . import openclaw_az_supreme_handler
    return openclaw_az_supreme_handler


__all__ = [
    "get_barren_wuffet_skills",
    "get_telegram_bot",
    "get_gateway_bridge",
    "get_az_supreme_handler",
]
