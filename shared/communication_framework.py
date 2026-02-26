"""
Communication Framework
=======================

Mock implementation for inter-agent communication.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    """Mock communication framework for inter-agent messaging"""

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
        pass

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
        """Deliver message to recipient (mock implementation)"""
        # In a real implementation, this would handle actual delivery
        # For now, just log the delivery
        print(f"📨 Delivered message from {message.sender} to {message.recipient}: {message.message_type}")

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