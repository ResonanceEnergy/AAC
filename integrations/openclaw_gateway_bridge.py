"""
OpenClaw Gateway Bridge for AAC
================================

Connects the AAC ecosystem to OpenClaw's WebSocket Gateway control plane.
Enables AZ SUPREME and all AAC agents to be reachable via WhatsApp, Telegram, 
Discord, Slack, Signal, iMessage, and WebChat through OpenClaw's multi-channel
messaging infrastructure.

Architecture:
    AAC Orchestrator ←→ OpenClaw Gateway Bridge ←→ OpenClaw Gateway (ws://127.0.0.1:18789)
                                                         ↕
                                              WhatsApp / Telegram / Discord / Slack / Signal / iMessage

Key Features:
    - Bidirectional WebSocket communication with OpenClaw Gateway
    - Session management (main, per-channel-peer, group isolation)
    - Skill registration and hot-reload for AAC capabilities
    - Cron job scheduling for automated market intelligence
    - Webhook ingestion for real-time market event triggers
    - Multi-agent routing: route OpenClaw messages to AZ SUPREME, AX HELIX, 
      or department-specific agents based on intent classification
    - Heartbeat monitoring for always-on operation
    - Memory persistence via OpenClaw's markdown memory system

Reference: https://docs.openclaw.ai/concepts/architecture
"""

import asyncio
import json
import logging
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Data Models ────────────────────────────────────────────────────────────

class OpenClawChannel(Enum):
    """Supported OpenClaw communication channels"""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    SIGNAL = "signal"
    IMESSAGE = "imessage"
    WEBCHAT = "webchat"
    MSTEAMS = "msteams"
    MATRIX = "matrix"


class MessageIntent(Enum):
    """Classified intent of inbound OpenClaw messages for AAC routing"""
    STRATEGIC_COMMAND = "strategic_command"    # → AZ SUPREME
    OPERATIONAL_QUERY = "operational_query"    # → AX HELIX
    MARKET_DATA = "market_data"               # → BigBrainIntelligence
    TRADING_SIGNAL = "trading_signal"          # → TradingExecution
    RISK_ALERT = "risk_alert"                 # → Risk Management
    PORTFOLIO_STATUS = "portfolio_status"      # → CentralAccounting
    CRYPTO_INTEL = "crypto_intel"             # → CryptoIntelligence
    INFRASTRUCTURE = "infrastructure"          # → SharedInfrastructure
    GENERAL_CHAT = "general_chat"             # → AZ SUPREME (default)
    DOCTRINE_OVERRIDE = "doctrine_override"   # → DoctrineEngine


@dataclass
class OpenClawMessage:
    """Message structure for OpenClaw ↔ AAC communication"""
    message_id: str
    channel: OpenClawChannel
    sender_id: str
    sender_name: str
    content: str
    intent: MessageIntent = MessageIntent.GENERAL_CHAT
    session_key: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    is_group: bool = False
    group_id: Optional[str] = None
    reply_to: Optional[str] = None


@dataclass
class OpenClawSession:
    """Tracks an active OpenClaw session mapped to an AAC agent"""
    session_id: str
    session_key: str
    agent_id: str
    channel: OpenClawChannel
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    context_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenClawCronJob:
    """Represents a scheduled cron job in OpenClaw"""
    job_id: str
    name: str
    schedule: str  # cron expression
    message: str   # message to send to the agent
    session_key: str = "main"
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass
class OpenClawSkill:
    """Represents an AAC skill registered with OpenClaw"""
    name: str
    description: str
    skill_dir: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    user_invocable: bool = True


# ─── Intent Classifier ─────────────────────────────────────────────────────

class AACIntentClassifier:
    """
    Classifies inbound OpenClaw messages to route to the correct AAC agent.
    Uses keyword + pattern matching for fast, zero-latency classification.
    """

    INTENT_PATTERNS = {
        MessageIntent.STRATEGIC_COMMAND: [
            "strategic", "enterprise", "directive", "az supreme", "supreme command",
            "crisis", "pivot", "vision", "roadmap", "executive", "authorize",
            "override", "priority shift", "all departments", "company-wide"
        ],
        MessageIntent.OPERATIONAL_QUERY: [
            "operations", "ax helix", "helix", "integration", "process",
            "efficiency", "optimize", "workflow", "logistics", "talent",
            "gln", "gta", "coordination"
        ],
        MessageIntent.MARKET_DATA: [
            "market", "price", "chart", "analysis", "research", "sentiment",
            "bigbrain", "theater", "narrative", "trend", "signal",
            "correlation", "volatility", "momentum"
        ],
        MessageIntent.TRADING_SIGNAL: [
            "trade", "execute", "buy", "sell", "position", "order",
            "arbitrage", "spread", "entry", "exit", "strategy",
            "execution", "fill", "limit", "stop"
        ],
        MessageIntent.RISK_ALERT: [
            "risk", "drawdown", "exposure", "hedge", "circuit breaker",
            "margin", "liquidation", "var", "stress test", "concentration",
            "stop loss", "max loss"
        ],
        MessageIntent.PORTFOLIO_STATUS: [
            "portfolio", "balance", "pnl", "profit", "loss", "accounting",
            "holdings", "allocation", "performance", "returns", "nav",
            "equity", "capital"
        ],
        MessageIntent.CRYPTO_INTEL: [
            "crypto", "bitcoin", "ethereum", "defi", "nft", "blockchain",
            "on-chain", "whale", "mempool", "gas", "bridge", "dex",
            "yield", "farming", "staking", "airdrop"
        ],
        MessageIntent.INFRASTRUCTURE: [
            "server", "deploy", "health", "uptime", "latency", "monitor",
            "infrastructure", "api", "database", "cache", "queue",
            "kubernetes", "docker", "ci/cd"
        ],
        MessageIntent.DOCTRINE_OVERRIDE: [
            "doctrine", "compliance", "barren wuffet", "safe mode", "halt",
            "caution", "override doctrine", "emergency override",
            "pack", "regulatory"
        ],
    }

    @classmethod
    def classify(cls, message: str) -> MessageIntent:
        """Classify message intent based on keyword matching"""
        lower = message.lower()
        scores: Dict[MessageIntent, int] = {}

        for intent, keywords in cls.INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return MessageIntent.GENERAL_CHAT

        return max(scores, key=scores.get)

    @classmethod
    def get_target_agent(cls, intent: MessageIntent) -> str:
        """Map classified intent to the target AAC agent ID"""
        INTENT_AGENT_MAP = {
            MessageIntent.STRATEGIC_COMMAND: "AZ-SUPREME",
            MessageIntent.OPERATIONAL_QUERY: "AX-HELIX",
            MessageIntent.MARKET_DATA: "BIGBRAIN-INTELLIGENCE",
            MessageIntent.TRADING_SIGNAL: "TRADING-EXECUTION",
            MessageIntent.RISK_ALERT: "RISK-MANAGEMENT",
            MessageIntent.PORTFOLIO_STATUS: "CENTRAL-ACCOUNTING",
            MessageIntent.CRYPTO_INTEL: "CRYPTO-INTELLIGENCE",
            MessageIntent.INFRASTRUCTURE: "SHARED-INFRASTRUCTURE",
            MessageIntent.GENERAL_CHAT: "AZ-SUPREME",
            MessageIntent.DOCTRINE_OVERRIDE: "DOCTRINE-ENGINE",
        }
        return INTENT_AGENT_MAP.get(intent, "AZ-SUPREME")


# ─── OpenClaw Gateway Bridge ───────────────────────────────────────────────

class OpenClawGatewayBridge:
    """
    Main bridge connecting AAC to OpenClaw's WebSocket Gateway.
    
    This is the primary integration point. It:
    1. Connects to the OpenClaw Gateway via WebSocket
    2. Receives messages from any connected channel (WhatsApp, Telegram, etc.)
    3. Classifies intent and routes to the correct AAC agent
    4. Returns responses back through the originating channel
    5. Registers AAC capabilities as OpenClaw skills
    6. Manages cron jobs for automated market intelligence
    7. Handles webhook events for real-time triggers
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789",
        gateway_token: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
        self.workspace_dir = Path(workspace_dir or os.path.expanduser("~/.openclaw/workspace"))

        # Connection state
        self._ws = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

        # Session tracking
        self.sessions: Dict[str, OpenClawSession] = {}

        # Registered AAC agent handlers
        self._agent_handlers: Dict[str, Callable] = {}

        # Cron jobs
        self.cron_jobs: List[OpenClawCronJob] = []

        # Skills
        self.registered_skills: Dict[str, OpenClawSkill] = {}

        # Message history for context
        self._message_log: List[OpenClawMessage] = []

        # Metrics
        self.metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "sessions_created": 0,
            "intents_classified": {},
            "errors": 0,
            "uptime_start": datetime.now(),
            "last_heartbeat": None,
        }

    # ── Connection Management ──

    async def connect(self) -> bool:
        """Establish WebSocket connection to OpenClaw Gateway"""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed. Using mock connection.")
            self._connected = True
            logger.info(f"🦞 OpenClaw Gateway Bridge MOCK connected to {self.gateway_url}")
            return True

        try:
            headers = {}
            if self.gateway_token:
                headers["Authorization"] = f"Bearer {self.gateway_token}"

            self._ws = await websockets.connect(
                self.gateway_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            )
            self._connected = True
            self._reconnect_delay = 1.0
            logger.info(f"🦞 OpenClaw Gateway Bridge connected to {self.gateway_url}")

            # Start message listener
            asyncio.create_task(self._message_listener())
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to OpenClaw Gateway: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from OpenClaw Gateway"""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("🦞 OpenClaw Gateway Bridge disconnected")

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff"""
        while not self._connected:
            logger.info(f"Reconnecting in {self._reconnect_delay}s...")
            await asyncio.sleep(self._reconnect_delay)
            if await self.connect():
                return
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._max_reconnect_delay
            )

    # ── Message Handling ──

    async def _message_listener(self) -> None:
        """Listen for inbound messages from OpenClaw Gateway"""
        while self._connected and self._ws:
            try:
                raw = await self._ws.recv()
                data = json.loads(raw)
                await self._handle_gateway_message(data)
            except Exception as e:
                if self._connected:
                    logger.error(f"Message listener error: {e}")
                    asyncio.create_task(self._reconnect())
                break

    async def _handle_gateway_message(self, data: Dict[str, Any]):
        """Process an inbound message from OpenClaw Gateway"""
        msg_type = data.get("type", "")

        if msg_type == "message.inbound":
            await self._handle_inbound_message(data)
        elif msg_type == "session.update":
            await self._handle_session_update(data)
        elif msg_type == "cron.trigger":
            await self._handle_cron_trigger(data)
        elif msg_type == "webhook.inbound":
            await self._handle_webhook(data)
        elif msg_type == "heartbeat":
            self.metrics["last_heartbeat"] = datetime.now()
        else:
            logger.debug(f"Unhandled gateway message type: {msg_type}")

    async def _handle_inbound_message(self, data: Dict[str, Any]):
        """Handle an inbound chat message from any OpenClaw channel"""
        self.metrics["messages_received"] += 1

        # Parse message
        msg = OpenClawMessage(
            message_id=data.get("id", hashlib.md5(str(data).encode()).hexdigest()[:12]),
            channel=OpenClawChannel(data.get("channel", "webchat")),
            sender_id=data.get("senderId", "unknown"),
            sender_name=data.get("senderName", "User"),
            content=data.get("content", ""),
            session_key=data.get("sessionKey", "main"),
            is_group=data.get("isGroup", False),
            group_id=data.get("groupId"),
            metadata=data.get("metadata", {}),
        )

        # Classify intent
        msg.intent = AACIntentClassifier.classify(msg.content)
        target_agent = AACIntentClassifier.get_target_agent(msg.intent)

        # Track intent metrics
        intent_name = msg.intent.value
        self.metrics["intents_classified"][intent_name] = (
            self.metrics["intents_classified"].get(intent_name, 0) + 1
        )

        # Log message
        self._message_log.append(msg)
        if len(self._message_log) > 1000:
            self._message_log = self._message_log[-500:]

        # Get or create session
        session = self._get_or_create_session(msg, target_agent)

        # Route to AAC agent handler
        response = await self._route_to_agent(target_agent, msg, session)

        # Send response back through OpenClaw
        if response:
            await self.send_response(msg, response)

    async def _route_to_agent(
        self,
        agent_id: str,
        message: OpenClawMessage,
        session: OpenClawSession
    ) -> Optional[str]:
        """Route message to the appropriate AAC agent and get response"""

        handler = self._agent_handlers.get(agent_id)

        if handler:
            try:
                response = await handler(message, session)
                return response
            except Exception as e:
                logger.error(f"Agent {agent_id} handler error: {e}")
                self.metrics["errors"] += 1
                return f"⚠️ Agent {agent_id} encountered an error: {str(e)}"

        # Default response if no handler registered
        return await self._default_handler(agent_id, message)

    async def _default_handler(self, agent_id: str, message: OpenClawMessage) -> str:
        """Default handler when no specific agent handler is registered"""
        return (
            f"🏢 AAC System | Routed to: {agent_id}\n"
            f"📊 Intent: {message.intent.value}\n"
            f"💬 Message received. Agent handler not yet registered.\n"
            f"Available agents: {', '.join(self._agent_handlers.keys()) or 'None registered'}"
        )

    def _get_or_create_session(
        self,
        msg: OpenClawMessage,
        agent_id: str
    ) -> OpenClawSession:
        """Get existing session or create new one"""
        if msg.session_key not in self.sessions:
            session = OpenClawSession(
                session_id=hashlib.md5(
                    f"{msg.channel.value}:{msg.sender_id}:{agent_id}".encode()
                ).hexdigest()[:16],
                session_key=msg.session_key,
                agent_id=agent_id,
                channel=msg.channel,
            )
            self.sessions[msg.session_key] = session
            self.metrics["sessions_created"] += 1
        else:
            session = self.sessions[msg.session_key]
            session.last_activity = datetime.now()
            session.message_count += 1

        return session

    # ── Outbound Messaging ──

    async def send_response(self, original_msg: OpenClawMessage, response: str):
        """Send a response back through the originating OpenClaw channel"""
        self.metrics["messages_sent"] += 1

        payload = {
            "type": "message.send",
            "channel": original_msg.channel.value,
            "sessionKey": original_msg.session_key,
            "content": response,
            "replyTo": original_msg.message_id,
            "metadata": {
                "source": "aac",
                "agent": AACIntentClassifier.get_target_agent(original_msg.intent),
                "timestamp": datetime.now().isoformat(),
            }
        }

        if self._ws and self._connected:
            await self._ws.send(json.dumps(payload))
        else:
            logger.info(f"[MOCK SEND → {original_msg.channel.value}] {response[:200]}")

    async def send_proactive_message(
        self,
        channel: OpenClawChannel,
        session_key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send a proactive message (alert, briefing, etc.) through OpenClaw"""
        self.metrics["messages_sent"] += 1

        payload = {
            "type": "message.send",
            "channel": channel.value,
            "sessionKey": session_key,
            "content": content,
            "metadata": {
                "source": "aac",
                "proactive": True,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            }
        }

        if self._ws and self._connected:
            await self._ws.send(json.dumps(payload))
        else:
            logger.info(f"[PROACTIVE → {channel.value}] {content[:200]}")

    # ── Agent Registration ──

    def register_agent_handler(self, agent_id: str, handler: Callable):
        """
        Register an AAC agent handler for inbound message routing.
        
        The handler must be an async callable with signature:
            async def handler(message: OpenClawMessage, session: OpenClawSession) -> str
        """
        self._agent_handlers[agent_id] = handler
        logger.info(f"Registered agent handler: {agent_id}")

    # ── Cron Job Management ──

    async def register_cron_job(self, job: OpenClawCronJob) -> bool:
        """Register a cron job with OpenClaw Gateway"""
        self.cron_jobs.append(job)

        payload = {
            "type": "cron.register",
            "jobId": job.job_id,
            "name": job.name,
            "schedule": job.schedule,
            "message": job.message,
            "sessionKey": job.session_key,
        }

        if self._ws and self._connected:
            await self._ws.send(json.dumps(payload))

        logger.info(f"📅 Registered cron job: {job.name} ({job.schedule})")
        return True

    async def _handle_cron_trigger(self, data: Dict[str, Any]):
        """Handle a cron job trigger from OpenClaw"""
        job_id = data.get("jobId", "")
        for job in self.cron_jobs:
            if job.job_id == job_id:
                job.last_run = datetime.now()
                logger.info(f"⏰ Cron triggered: {job.name}")
                # Route as if it were a message
                msg = OpenClawMessage(
                    message_id=f"cron-{job_id}-{datetime.now().timestamp()}",
                    channel=OpenClawChannel.WEBCHAT,
                    sender_id="cron-system",
                    sender_name="Cron System",
                    content=job.message,
                    session_key=job.session_key,
                )
                msg.intent = AACIntentClassifier.classify(msg.content)
                target = AACIntentClassifier.get_target_agent(msg.intent)
                response = await self._route_to_agent(target, msg, self._get_or_create_session(msg, target))
                if response:
                    await self.send_proactive_message(
                        OpenClawChannel.TELEGRAM, "main", response
                    )

    # ── Webhook Handling ──

    async def _handle_webhook(self, data: Dict[str, Any]):
        """Handle inbound webhook events (market data sources, exchange alerts, etc.)"""
        webhook_id = data.get("hookId", "")
        payload_data = data.get("payload", {})

        logger.info(f"🔗 Webhook received: {webhook_id}")

        # Convert webhook to internal message for routing
        msg = OpenClawMessage(
            message_id=f"webhook-{webhook_id}-{datetime.now().timestamp()}",
            channel=OpenClawChannel.WEBCHAT,
            sender_id="webhook-system",
            sender_name="Webhook",
            content=json.dumps(payload_data),
            session_key=f"hook:{webhook_id}",
            metadata={"webhook_id": webhook_id, "raw_payload": payload_data},
        )
        msg.intent = AACIntentClassifier.classify(msg.content)
        target = AACIntentClassifier.get_target_agent(msg.intent)
        await self._route_to_agent(target, msg, self._get_or_create_session(msg, target))

    # ── Session Update ──

    async def _handle_session_update(self, data: Dict[str, Any]):
        """Handle session state updates from OpenClaw"""
        session_key = data.get("sessionKey", "")
        if session_key in self.sessions:
            session = self.sessions[session_key]
            session.context_tokens = data.get("contextTokens", session.context_tokens)
            session.metadata.update(data.get("metadata", {}))

    # ── Heartbeat ──

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to OpenClaw Gateway"""
        while self._connected:
            try:
                if self._ws:
                    await self._ws.send(json.dumps({
                        "type": "heartbeat",
                        "source": "aac-bridge",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {
                            "sessions": len(self.sessions),
                            "agents": len(self._agent_handlers),
                            "messages_total": self.metrics["messages_received"],
                        }
                    }))
                    self.metrics["last_heartbeat"] = datetime.now()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(30)

    # ── Memory Persistence ──

    async def save_to_memory(self, key: str, content: str, category: str = "general"):
        """Save data to OpenClaw's markdown memory system"""
        memory_dir = self.workspace_dir / "memory" / "aac"
        memory_dir.mkdir(parents=True, exist_ok=True)

        filepath = memory_dir / f"{category}-{key}.md"
        with open(filepath, "w") as f:
            f.write(f"# {key}\n\n")
            f.write(f"*Updated: {datetime.now().isoformat()}*\n\n")
            f.write(content)

        logger.info(f"💾 Saved to OpenClaw memory: {filepath}")

    async def load_from_memory(self, key: str, category: str = "general") -> Optional[str]:
        """Load data from OpenClaw's markdown memory system"""
        filepath = self.workspace_dir / "memory" / "aac" / f"{category}-{key}.md"
        if filepath.exists():
            return filepath.read_text()
        return None

    # ── Skill Registration ──

    async def register_aac_skills(self):
        """Register AAC capabilities as OpenClaw skills"""
        skills_dir = self.workspace_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        for skill_name, skill in self.registered_skills.items():
            skill_path = skills_dir / skill_name
            skill_path.mkdir(parents=True, exist_ok=True)
            # Skills are written by the separate skills module
            logger.info(f"📦 Registered skill: {skill_name}")

    # ── Status & Metrics ──

    async def shutdown(self):
        """Gracefully shut down the bridge."""
        logger.info("Shutting down OpenClaw bridge")
        self._connected = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
        self.sessions.clear()
        logger.info("OpenClaw bridge shut down complete")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the OpenClaw bridge"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "connected": self._connected,
            "gateway_url": self.gateway_url,
            "uptime_seconds": uptime.total_seconds(),
            "sessions_active": len(self.sessions),
            "agents_registered": list(self._agent_handlers.keys()),
            "skills_registered": list(self.registered_skills.keys()),
            "cron_jobs": len(self.cron_jobs),
            "metrics": self.metrics,
        }


# ─── Singleton Access ──────────────────────────────────────────────────────

_bridge_instance: Optional[OpenClawGatewayBridge] = None


def get_openclaw_bridge(
    gateway_url: str = os.environ.get('OPENCLAW_GATEWAY_URL', 'ws://127.0.0.1:18789'),
    gateway_token: Optional[str] = None,
) -> OpenClawGatewayBridge:
    """Get or create the singleton OpenClaw Gateway Bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        logger.info(f"Creating OpenClaw bridge to {gateway_url}")
        _bridge_instance = OpenClawGatewayBridge(
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )
    return _bridge_instance
