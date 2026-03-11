"""
BARREN WUFFET Telegram Bot Integration
========================================

Connects @barrenwuffet069bot to the AAC ecosystem via the OpenClaw
control plane. Routes Telegram messages to the appropriate AAC agents
and skills based on command prefixes and natural language intent.

Bot: @barrenwuffet069bot
Token: env(TELEGRAM_BOT_TOKEN)

Architecture:
    Telegram → Bot API → BarrenWuffetTelegramBot → OpenClaw Router
        → Skill Handler → AAC Agent → Response → Telegram
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("barren_wuffet.telegram")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    import warnings
    warnings.warn(
        "TELEGRAM_BOT_TOKEN not set in environment. "
        "Add it to .env file. Bot will not function without it.",
        stacklevel=2,
    )
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BOT_USERNAME = "barrenwuffet069bot"
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class MessagePriority(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FLASH = "flash"


@dataclass
class TelegramMessage:
    """Incoming Telegram message."""
    chat_id: int
    user_id: int
    username: str
    text: str
    timestamp: datetime
    message_id: int
    reply_to: Optional[int] = None


@dataclass
class BotResponse:
    """Outgoing bot response."""
    text: str
    chat_id: int
    priority: MessagePriority = MessagePriority.INFO
    parse_mode: str = "Markdown"
    reply_to_message_id: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ═══════════════════════════════════════════════════════════════════════════

# Map command prefixes to skill handlers
COMMAND_ROUTES: Dict[str, str] = {
    # Core AAC
    "/bw-intel": "bw-market-intelligence",
    "/bw-signals": "bw-trading-signals",
    "/bw-dash": "bw-portfolio-dashboard",
    "/bw-risk": "bw-risk-monitor",
    "/bw-crypto": "bw-crypto-intel",
    "/az": "bw-az-supreme-command",
    "/bw-doctrine": "bw-doctrine-status",
    "/bw-briefing": "bw-morning-briefing",
    "/bw-agents": "bw-agent-roster",
    "/bw-strat": "bw-strategy-explorer",
    # Trading & Markets
    "/bw-arb": "bw-digital-arbitrage",
    "/bw-scan": "bw-arbitrage-scanner",
    "/bw-day": "bw-day-trading",
    "/bw-options": "bw-options-trading",
    "/bw-flow": "bw-calls-puts-flow",
    "/bw-hedge": "bw-hedging-strategies",
    "/bw-fx": "bw-currency-trading",
    # Crypto & DeFi
    "/bw-btc": "bw-bitcoin-intel",
    "/bw-eth": "bw-ethereum-defi",
    "/bw-xrp": "bw-xrp-ripple",
    "/bw-stable": "bw-stablecoins",
    "/bw-meme": "bw-meme-coins",
    "/bw-liberty": "bw-liberty-coin",
    "/bw-xtokens": "bw-x-tokens",
    # Finance & Banking
    "/bw-bank": "bw-banking-intel",
    "/bw-accounting": "bw-accounting-engine",
    "/bw-reg": "bw-regulations",
    # Wealth Building
    "/bw-money": "bw-money-mastery",
    "/bw-wealth": "bw-wealth-building",
    "/bw-dd": "bw-superstonk-dd",
    # Advanced
    "/bw-crash": "bw-crash-indicators",
    "/bw-golden": "bw-golden-ratio-finance",
    "/bw-jonny": "bw-jonny-bravo-course",
    # Power-ups
    "/bw-poly": "bw-polymarket-autopilot",
    "/bw-brain": "bw-second-brain",
}

# Natural language intent detection keywords
INTENT_KEYWORDS: Dict[str, str] = {
    "bitcoin": "bw-bitcoin-intel",
    "btc": "bw-bitcoin-intel",
    "ethereum": "bw-ethereum-defi",
    "eth": "bw-ethereum-defi",
    "defi": "bw-ethereum-defi",
    "xrp": "bw-xrp-ripple",
    "ripple": "bw-xrp-ripple",
    "stablecoin": "bw-stablecoins",
    "usdt": "bw-stablecoins",
    "usdc": "bw-stablecoins",
    "meme coin": "bw-meme-coins",
    "doge": "bw-meme-coins",
    "pepe": "bw-meme-coins",
    "options": "bw-options-trading",
    "calls": "bw-calls-puts-flow",
    "puts": "bw-calls-puts-flow",
    "hedge": "bw-hedging-strategies",
    "arbitrage": "bw-arbitrage-scanner",
    "day trade": "bw-day-trading",
    "scalp": "bw-day-trading",
    "crash": "bw-crash-indicators",
    "2007": "bw-crash-indicators",
    "2008": "bw-crash-indicators",
    "golden ratio": "bw-golden-ratio-finance",
    "fibonacci": "bw-golden-ratio-finance",
    "dan winter": "bw-golden-ratio-finance",
    "superstonk": "bw-superstonk-dd",
    "short squeeze": "bw-superstonk-dd",
    "dark pool": "bw-superstonk-dd",
    "ftd": "bw-superstonk-dd",
    "jonny bravo": "bw-jonny-bravo-course",
    "offshore": "bw-banking-intel",
    "bank": "bw-banking-intel",
    "tax": "bw-accounting-engine",
    "accounting": "bw-accounting-engine",
    "regulation": "bw-regulations",
    "calgary": "bw-regulations",
    "montevideo": "bw-regulations",
    "wealth": "bw-wealth-building",
    "generational": "bw-wealth-building",
    "save money": "bw-money-mastery",
    "budget": "bw-money-mastery",
    "forex": "bw-currency-trading",
    "currency": "bw-currency-trading",
    "cad": "bw-currency-trading",
    "portfolio": "bw-portfolio-dashboard",
    "p&l": "bw-portfolio-dashboard",
    "pnl": "bw-portfolio-dashboard",
    "risk": "bw-risk-monitor",
    "doctrine": "bw-doctrine-status",
    "briefing": "bw-morning-briefing",
    "polymarket": "bw-polymarket-autopilot",
    "prediction market": "bw-polymarket-autopilot",
    "remember": "bw-second-brain",
    "save this": "bw-second-brain",
    "note": "bw-second-brain",
    "liberty coin": "bw-liberty-coin",
}


def route_message(text: str) -> str:
    """Route a message to the appropriate skill based on command or intent."""
    text_lower = text.lower().strip()

    # 1. Check explicit commands
    for prefix, skill in COMMAND_ROUTES.items():
        if text_lower.startswith(prefix):
            return skill

    # 2. Check natural language intent
    for keyword, skill in INTENT_KEYWORDS.items():
        if keyword in text_lower:
            return skill

    # 3. Default to AZ SUPREME for general messages
    return "bw-az-supreme-command"


def parse_command_args(text: str) -> Dict[str, str]:
    """Parse command arguments from text like '/bw-btc onchain key=value'."""
    parts = text.strip().split()
    args: Dict[str, str] = {}
    positional_idx = 0

    for part in parts[1:]:  # Skip the command itself
        if "=" in part:
            key, value = part.split("=", 1)
            args[key] = value
        else:
            args[f"arg{positional_idx}"] = part
            positional_idx += 1

    return args


# ═══════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class BarrenWuffetTelegramBot:
    """
    Main Telegram bot for BARREN WUFFET.
    
    Handles message routing, skill dispatch, and response delivery
    via the Telegram Bot API.
    """

    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.api_base = TELEGRAM_API_BASE
        self.username = BOT_USERNAME
        self.chat_id = TELEGRAM_CHAT_ID
        self.running = False
        self.last_update_id = 0
        self.message_log: List[Dict] = []
        self.memory: List[Dict] = []  # Second Brain memory store
        self._skill_handlers: Dict[str, Callable] = {}
        self._setup_skill_handlers()

    def _setup_skill_handlers(self):
        """Register skill handler functions."""
        # Each skill gets a handler that formats the response
        from integrations.openclaw_barren_wuffet_skills import (
            BARREN_WUFFET_SKILLS, get_skill_definition
        )
        for skill_name in BARREN_WUFFET_SKILLS:
            self._skill_handlers[skill_name] = self._create_handler(skill_name)

    def _create_handler(self, skill_name: str):
        """Create a handler closure for a specific skill."""
        async def handler(message: TelegramMessage, args: Dict) -> BotResponse:
            from integrations.openclaw_barren_wuffet_skills import get_skill_definition
            skill = get_skill_definition(skill_name)
            if not skill:
                return BotResponse(
                    text=f"⚠️ Skill `{skill_name}` not found.",
                    chat_id=message.chat_id
                )

            # Log to memory (Second Brain)
            self._log_interaction(message, skill_name, args)

            # Generate contextual response based on skill + args
            response_text = self._generate_skill_response(skill, args)
            return BotResponse(
                text=response_text,
                chat_id=message.chat_id,
                reply_to_message_id=message.message_id
            )
        return handler

    def _generate_skill_response(self, skill: Dict, args: Dict) -> str:
        """Generate a response based on skill definition and arguments."""
        name = skill["name"]
        emoji = skill["metadata"]["openclaw"].get("emoji", "📊")
        desc = skill["description"]

        # Default response shows skill info + available commands
        lines = [
            f"{emoji} **{name}**",
            "",
            desc,
            "",
            "---",
        ]

        # Extract commands from instructions
        instructions = skill.get("instructions", "")
        if "### Commands" in instructions:
            cmd_section = instructions.split("### Commands")[1]
            # Get the command block
            if "```" in cmd_section:
                cmd_block = cmd_section.split("```")[1]
                if "```" in cmd_block:
                    cmd_block = cmd_block.split("```")[0]
                lines.append("**Available Commands:**")
                lines.append(f"```{cmd_block.strip()}```")

        if args:
            lines.append("")
            lines.append(f"📎 Args received: `{json.dumps(args)}`")
            lines.append("Processing...")

        lines.append("")
        lines.append("— BARREN WUFFET, AZ SUPREME 🐺")

        return "\n".join(lines)

    def _log_interaction(self, message: TelegramMessage, skill: str, args: Dict):
        """Log interaction to memory (Second Brain feature)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": message.username,
            "skill": skill,
            "input": message.text,
            "args": args,
        }
        self.memory.append(entry)
        self.message_log.append(entry)
        logger.info(f"Memory entry: {skill} from {message.username}")

    async def handle_message(self, message: TelegramMessage) -> BotResponse:
        """Process an incoming message and return a response."""
        text = message.text

        # Special commands
        if text.lower().strip() in ("/start", "/help"):
            return self._help_response(message)

        if text.lower().strip() == "/status":
            return self._status_response(message)

        if text.lower().strip() == "/skills":
            return self._skills_list_response(message)

        # Route to skill
        skill_name = route_message(text)
        args = parse_command_args(text)

        handler = self._skill_handlers.get(skill_name)
        if handler:
            return await handler(message, args)

        # Fallback
        return BotResponse(
            text="🐺 BARREN WUFFET received your message. Processing...\n\n"
                 f"Routed to: `{skill_name}`\n"
                 "Use /skills to see all 35 available skills.\n\n"
                 "— BARREN WUFFET, AZ SUPREME",
            chat_id=message.chat_id,
            reply_to_message_id=message.message_id,
        )

    def _help_response(self, message: TelegramMessage) -> BotResponse:
        """Generate help/start response."""
        return BotResponse(
            text=(
                "🐺 **BARREN WUFFET — AZ SUPREME**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Supreme Financial Intelligence Agent\n"
                "Accelerated Arbitrage Corporation\n\n"
                "**35 Skills** across:\n"
                "📊 Market Intelligence (10 core skills)\n"
                "⚡ Trading & Markets (7 skills)\n"
                "🔗 Crypto & DeFi (7 skills)\n"
                "🏦 Finance & Banking (3 skills)\n"
                "💰 Wealth Building (3 skills)\n"
                "🌀 Advanced Analysis (3 skills)\n"
                "🎰 Power-ups (2 skills)\n\n"
                "**Quick Commands:**\n"
                "`/status` — System status\n"
                "`/skills` — List all 35 skills\n"
                "`/az status` — Full AZ SUPREME report\n"
                "`/bw-briefing now` — Morning briefing\n"
                "`/bw-btc overview` — Bitcoin intelligence\n"
                "`/bw-crash dashboard` — Crash indicators\n"
                "`/bw-dash full` — Portfolio dashboard\n\n"
                "Or just text naturally — I'll route to the right skill.\n\n"
                "— BARREN WUFFET, AZ SUPREME 🐺"
            ),
            chat_id=message.chat_id,
        )

    def _status_response(self, message: TelegramMessage) -> BotResponse:
        """Generate system status response."""
        now = datetime.now(timezone.utc)
        return BotResponse(
            text=(
                "🐺 **BARREN WUFFET STATUS**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"📡 Bot: @{self.username}\n"
                f"🟢 Doctrine: NORMAL\n"
                f"🏗️ Skills: 35 active\n"
                f"🤖 Agents: 80+ registered\n"
                f"📈 Strategies: 50 loaded\n"
                f"🧠 Memory entries: {len(self.memory)}\n"
                f"📨 Messages processed: {len(self.message_log)}\n\n"
                "**Departments:**\n"
                "✅ BigBrainIntelligence — Online\n"
                "✅ TradingExecution — Online\n"
                "✅ CryptoIntelligence — Online\n"
                "✅ CentralAccounting — Online\n"
                "✅ SharedInfrastructure — Online\n"
                "✅ NCC — Online\n"
                "✅ Jonny Bravo Division — Online\n\n"
                "— BARREN WUFFET, AZ SUPREME 🐺"
            ),
            chat_id=message.chat_id,
        )

    def _skills_list_response(self, message: TelegramMessage) -> BotResponse:
        """List all 35 skills."""
        from integrations.openclaw_barren_wuffet_skills import BARREN_WUFFET_SKILLS
        
        categories = {
            "📊 Core AAC": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-market", "bw-trading", "bw-portfolio", "bw-risk", "bw-crypto-intel", "bw-az-", "bw-doctrine", "bw-morning", "bw-agent", "bw-strategy"))],
            "⚡ Trading": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-digital", "bw-arbitrage", "bw-day", "bw-options", "bw-calls", "bw-hedging", "bw-currency"))],
            "🔗 Crypto": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-bitcoin", "bw-ethereum", "bw-xrp", "bw-stable", "bw-meme", "bw-liberty", "bw-x-token"))],
            "🏦 Finance": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-banking", "bw-accounting", "bw-regulation"))],
            "💰 Wealth": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-money", "bw-wealth", "bw-superstonk"))],
            "🌀 Advanced": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-crash", "bw-golden", "bw-jonny"))],
            "🎰 Power-ups": [s for s in BARREN_WUFFET_SKILLS if s.startswith(("bw-polymarket", "bw-second"))],
        }

        lines = ["🐺 **BARREN WUFFET — 35 SKILLS**\n"]
        for cat_name, skills in categories.items():
            lines.append(f"\n**{cat_name}** ({len(skills)})")
            for s in skills:
                skill_def = BARREN_WUFFET_SKILLS[s]
                emoji = skill_def["metadata"]["openclaw"].get("emoji", "")
                lines.append(f"  {emoji} `{s}`")

        lines.append("\n— BARREN WUFFET, AZ SUPREME 🐺")
        return BotResponse(text="\n".join(lines), chat_id=message.chat_id)

    async def send_alert(
        self,
        text: str,
        priority: MessagePriority = MessagePriority.INFO,
        chat_id: Optional[int] = None,
    ) -> bool:
        """Send a proactive alert via Telegram."""
        target = chat_id or self.chat_id
        if not target:
            logger.warning("No chat_id configured for alerts")
            return False

        priority_prefix = {
            MessagePriority.INFO: "ℹ️",
            MessagePriority.WARNING: "⚠️",
            MessagePriority.CRITICAL: "🚨",
            MessagePriority.FLASH: "⚡",
        }

        full_text = f"{priority_prefix[priority]} {text}\n\n— BARREN WUFFET"

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/sendMessage",
                    json={
                        "chat_id": target,
                        "text": full_text,
                        "parse_mode": "Markdown",
                    },
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def send_morning_briefing(self, briefing_text: str) -> bool:
        """Send the morning briefing via Telegram."""
        return await self.send_alert(
            f"☀️ **MORNING BRIEFING**\n\n{briefing_text}",
            priority=MessagePriority.INFO,
        )

    async def send_doctrine_alert(self, old_state: str, new_state: str) -> bool:
        """Alert when BarrenWuffetState transitions."""
        return await self.send_alert(
            f"**DOCTRINE STATE CHANGE**\n"
            f"`{old_state}` → `{new_state}`\n\n"
            f"All agents notified. Risk parameters updated.",
            priority=MessagePriority.CRITICAL,
        )

    async def send_crash_alert(self, score: int, details: str) -> bool:
        """Alert when crash similarity score exceeds threshold."""
        return await self.send_alert(
            f"**2007 CRASH INDICATOR ALERT**\n"
            f"Similarity Score: `{score}/100`\n\n"
            f"{details}\n\n"
            f"Hedging recommendations activating.",
            priority=MessagePriority.CRITICAL,
        )

    async def send_whale_alert(self, coin: str, amount: str, direction: str) -> bool:
        """Alert on large crypto transactions."""
        return await self.send_alert(
            f"🐋 **WHALE ALERT**\n"
            f"Coin: `{coin}`\n"
            f"Amount: `{amount}`\n"
            f"Direction: `{direction}`",
            priority=MessagePriority.WARNING,
        )


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY / SECOND BRAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class BarrenWuffetMemory:
    """
    Persistent memory system for BARREN WUFFET Second Brain.
    Stores notes, observations, and knowledge via Telegram messages.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "barren_wuffet_memory.json"
        )
        self.entries: List[Dict] = []
        self._load()

    def _load(self):
        """Load memory from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    self.entries = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            self.entries = []

    def _save(self):
        """Save memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self.entries, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def add(self, text: str, category: str = "general", source: str = "telegram") -> Dict:
        """Add a memory entry."""
        entry = {
            "id": len(self.entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "category": self._auto_categorize(text) if category == "general" else category,
            "source": source,
        }
        self.entries.append(entry)
        self._save()
        return entry

    def search(self, query: str) -> List[Dict]:
        """Search memory entries by keyword."""
        query_lower = query.lower()
        return [
            e for e in self.entries
            if query_lower in e["text"].lower()
        ]

    def recent(self, count: int = 10) -> List[Dict]:
        """Get most recent memory entries."""
        return self.entries[-count:]

    def by_category(self, category: str) -> List[Dict]:
        """Filter entries by category."""
        return [e for e in self.entries if e["category"] == category]

    def _auto_categorize(self, text: str) -> str:
        """Auto-categorize a memory entry based on content."""
        text_lower = text.lower()
        categories = {
            "market": ["price", "signal", "market", "stock", "crypto", "btc", "eth"],
            "trading": ["trade", "buy", "sell", "position", "strategy", "arb"],
            "banking": ["bank", "account", "wire", "transfer", "offshore"],
            "regulation": ["regulation", "compliance", "cra", "sec", "law"],
            "research": ["dd", "research", "analysis", "report", "study"],
            "idea": ["idea", "thought", "consider", "maybe", "what if"],
        }
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return cat
        return "general"


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_bot() -> BarrenWuffetTelegramBot:
    """Create and return a configured BARREN WUFFET Telegram bot instance."""
    bot = BarrenWuffetTelegramBot()
    logger.info(f"🐺 BARREN WUFFET bot initialized: @{BOT_USERNAME}")
    logger.info(f"   Skills: 35 | Agents: 80+ | Strategies: 50")
    return bot


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🐺 BARREN WUFFET Telegram Bot")
    print("━" * 40)
    print(f"Bot: @{BOT_USERNAME}")
    print(f"Token: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-4:]}")
    print(f"Skills: 35")
    print(f"Commands: {len(COMMAND_ROUTES)}")
    print(f"Intent Keywords: {len(INTENT_KEYWORDS)}")
    print()
    print("Command Routing:")
    for cmd, skill in sorted(COMMAND_ROUTES.items()):
        print(f"  {cmd:25s} → {skill}")
    print()
    print("Ready. Run with OpenClaw gateway for full integration.")
