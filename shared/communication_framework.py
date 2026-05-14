"""
Communication Framework
=======================

In-process inter-agent message bus.

Sprint 55 (NO MOCK DATA OR CALLS doctrine): the previous header described this
as a "Mock implementation" and labelled the class / ``_deliver_message`` as
mock.  Those labels were stale -- the implementation IS the production
implementation: an in-memory pub/sub queue with explicit subscribe / send /
broadcast / get.  It is intentionally in-process (single Python interpreter,
no network).  When cross-process messaging is needed the bus must be replaced
with a real transport (Redis pub/sub, Kafka, or the OpenClaw WebSocket bridge
in ``integrations/openclaw_gateway_bridge.py``).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message structure for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    priority: str = "normal"

class CommunicationFramework:
    """In-process inter-agent message bus (single-interpreter pub/sub)."""

    def __init__(self):
        self.messages = []
        self.subscriptions = {}
        self.agents = set()
        self.channels = {}  # Store registered channels

    async def register_agent(self, agent_id: str, agent: Any = None) -> bool:
        """Register an agent in the communication framework"""
        self.agents.add(agent_id)
        self.subscriptions[agent_id] = []
        return True

    async def register_channel(self, channel_name: str, channel_type: str = "general") -> bool:
        """Register a communication channel"""
        self.channels[channel_name] = {
            "type": channel_type,
            "registered_at": asyncio.get_event_loop().time(),
            "active": True
        }
        return True

    async def initialize(self):
        """Initialize the communication framework"""
        self.messages.clear()
        self.subscriptions.clear()
        self.agents.clear()
        self.channels.clear()
        logger.info("CommunicationFramework initialized (base implementation)")

    async def send_message(self, sender: str, recipient: str, message_type: str,
                          payload: Dict[str, Any], priority: str = "normal") -> bool:
        """Send a message to another agent"""
        message = Message(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            timestamp=asyncio.get_event_loop().time(),
            priority=priority
        )

        self.messages.append(message)

        # If recipient is subscribed to this message type, deliver immediately
        if recipient in self.subscriptions and message_type in self.subscriptions[recipient]:
            await self._deliver_message(message)

        return True

    async def broadcast_message(self, sender: str, message_type: str,
                               payload: Dict[str, Any], priority: str = "normal") -> int:
        """Broadcast message to all agents"""
        delivered_count = 0
        for agent_id in self.agents:
            if agent_id != sender:
                success = await self.send_message(sender, agent_id, message_type, payload, priority)
                if success:
                    delivered_count += 1
        return delivered_count

    async def subscribe_to_messages(self, agent_id: str, message_types: List[str]) -> bool:
        """Subscribe agent to specific message types"""
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = []

        self.subscriptions[agent_id].extend(message_types)
        return True

    async def get_messages_for_agent(self, agent_id: str, message_type: Optional[str] = None) -> List[Message]:
        """Get messages for a specific agent"""
        agent_messages = [msg for msg in self.messages if msg.recipient == agent_id]

        if message_type:
            agent_messages = [msg for msg in agent_messages if msg.message_type == message_type]

        return agent_messages

    async def _deliver_message(self, message: Message):
        """Deliver message to recipient.

        For the in-process bus, "delivery" means appending to ``self.messages``
        (already done by ``send_message``) and logging.  Subscribers retrieve
        their messages with ``get_messages_for_agent``.  When this module is
        replaced with a real transport (Redis / Kafka / OpenClaw) this hook is
        where the network publish would live.
        """
        logger.info(
            "Delivered message from %s to %s: %s",
            message.sender,
            message.recipient,
            message.message_type,
        )

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of an agent"""
        return {
            "agent_id": agent_id,
            "registered": agent_id in self.agents,
            "message_count": len([msg for msg in self.messages if msg.recipient == agent_id]),
            "subscriptions": self.subscriptions.get(agent_id, [])
        }

# Global instance
_communication_framework = None

def get_communication_framework() -> CommunicationFramework:
    """Get the global communication framework instance"""
    global _communication_framework
    if _communication_framework is None:
        _communication_framework = CommunicationFramework()
    return _communication_framework
