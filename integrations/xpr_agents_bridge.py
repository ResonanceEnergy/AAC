#!/usr/bin/env python3
"""
XPR Agents ↔ AAC Bridge (Vector 4)
====================================
Bridges the XPR Agents on-chain AI marketplace to the AAC system.

XPR Agents features:
  - On-chain AI agent marketplace on XPR Network
  - 55+ MCP tools available for rent
  - A2A (Agent-to-Agent) protocol for inter-agent communication
  - Escrow-based payments in XPR/XMD
  - Trust scores and capability attestations
  - 0.5-second finality for agent transactions

This bridge allows AAC agents to:
  1. Discover and hire XPR agents for specialized tasks
  2. Publish AAC agent capabilities to the marketplace
  3. Use MCP tools hosted on-chain
  4. Settle agent-to-agent payments via escrow
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class XPRAgent:
    """Represents an agent on the XPR Agents marketplace."""
    agent_id: str
    name: str
    capabilities: List[str]
    trust_score: float  # 0.0 to 1.0
    price_per_call: float  # In XPR
    owner_account: str
    tools: List[str] = field(default_factory=list)
    a2a_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTool:
    """An MCP tool available on XPR Agents."""
    tool_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    price_per_use: float  # In XPR
    provider_agent: str
    category: str = ""


@dataclass
class EscrowPayment:
    """Escrow payment record for agent-to-agent settlement."""
    escrow_id: str
    from_account: str
    to_account: str
    amount: float
    currency: str  # XPR or XMD
    status: str  # pending, released, refunded, disputed
    created_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None


class XPRAgentsBridge:
    """
    Bridges AAC's internal agent system to the XPR Agents on-chain marketplace.

    Usage:
        bridge = XPRAgentsBridge(
            rpc_url="https://xpr-rpc.example.com",
            account_name="aac.agent",
            private_key="5K...",
        )
        await bridge.connect()

        # Discover agents with trading capabilities
        agents = await bridge.discover_agents(capability="trading_analysis")

        # Hire an agent for a task
        result = await bridge.hire_agent(
            agent_id="trader.xpr",
            task={"action": "analyze", "symbol": "BTC/XMD"},
        )

        # List available MCP tools
        tools = await bridge.list_mcp_tools(category="market_data")
    """

    MAINNET_RPC = "https://xpr-rpc.example.com"  # Placeholder — set via config

    def __init__(
        self,
        rpc_url: str = "",
        account_name: str = "",
        private_key: str = "",
    ):
        self.rpc_url = rpc_url or self.MAINNET_RPC
        self.account_name = account_name
        self._private_key = private_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._agent_cache: Dict[str, XPRAgent] = {}
        self._tool_cache: Dict[str, MCPTool] = {}

    async def connect(self) -> bool:
        """Connect to XPR Network RPC."""
        try:
            connector = aiohttp.TCPConnector(
                resolver=aiohttp.resolver.ThreadedResolver()
            )
            self._session = aiohttp.ClientSession(connector=connector)
            self._connected = True
            logger.info(f"XPR Agents Bridge connected to {self.rpc_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect XPR Agents Bridge: {e}")
            return False

    async def disconnect(self):
        """Disconnect from XPR Network."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("XPR Agents Bridge disconnected")

    async def _rpc_call(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make an RPC call to XPR Network."""
        if not self._session:
            raise ConnectionError("XPR Agents Bridge not connected")

        url = f"{self.rpc_url}/v1/{endpoint}"
        try:
            if data:
                async with self._session.post(url, json=data) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            else:
                async with self._session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            logger.error(f"XPR RPC call failed ({endpoint}): {e}")
            raise

    # ==========================================
    # Agent Discovery
    # ==========================================

    async def discover_agents(
        self,
        capability: Optional[str] = None,
        min_trust_score: float = 0.5,
        max_price: Optional[float] = None,
        limit: int = 20,
    ) -> List[XPRAgent]:
        """
        Discover agents on the XPR marketplace.

        Args:
            capability: Filter by capability (e.g., 'trading_analysis', 'data_feed')
            min_trust_score: Minimum trust score (0-1)
            max_price: Maximum price per call in XPR
            limit: Max results
        """
        try:
            params: Dict[str, Any] = {
                "code": "xpragents",
                "scope": "xpragents",
                "table": "agents",
                "limit": limit,
            }

            result = await self._rpc_call("chain/get_table_rows", params)
            rows = result.get("rows", [])

            agents = []
            for row in rows:
                agent = XPRAgent(
                    agent_id=row.get("agent_id", ""),
                    name=row.get("name", ""),
                    capabilities=row.get("capabilities", []),
                    trust_score=float(row.get("trust_score", 0)) / 10000,
                    price_per_call=float(row.get("price", "0").split(" ")[0]),
                    owner_account=row.get("owner", ""),
                    tools=row.get("tools", []),
                    a2a_enabled=row.get("a2a_enabled", False),
                    metadata=row.get("metadata", {}),
                )

                # Apply filters
                if capability and capability not in agent.capabilities:
                    continue
                if agent.trust_score < min_trust_score:
                    continue
                if max_price is not None and agent.price_per_call > max_price:
                    continue

                agents.append(agent)
                self._agent_cache[agent.agent_id] = agent

            return agents

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return []

    # ==========================================
    # MCP Tool Discovery
    # ==========================================

    async def list_mcp_tools(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[MCPTool]:
        """List available MCP tools on the XPR Agents marketplace."""
        try:
            params = {
                "code": "xpragents",
                "scope": "xpragents",
                "table": "mcptools",
                "limit": limit,
            }

            result = await self._rpc_call("chain/get_table_rows", params)
            rows = result.get("rows", [])

            tools = []
            for row in rows:
                tool = MCPTool(
                    tool_id=row.get("tool_id", ""),
                    name=row.get("name", ""),
                    description=row.get("description", ""),
                    input_schema=json.loads(row.get("input_schema", "{}")),
                    price_per_use=float(row.get("price", "0").split(" ")[0]),
                    provider_agent=row.get("provider", ""),
                    category=row.get("category", ""),
                )
                if category and tool.category != category:
                    continue
                tools.append(tool)
                self._tool_cache[tool.tool_id] = tool

            return tools

        except Exception as e:
            logger.error(f"MCP tool listing failed: {e}")
            return []

    # ==========================================
    # Agent Hiring & Task Execution
    # ==========================================

    async def hire_agent(
        self,
        agent_id: str,
        task: Dict[str, Any],
        max_payment: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Hire an XPR agent for a task with escrow payment.

        Args:
            agent_id: The agent to hire
            task: Task specification dict
            max_payment: Max XPR to pay (defaults to agent's price_per_call)
        """
        agent = self._agent_cache.get(agent_id)
        if not agent:
            agents = await self.discover_agents()
            agent = self._agent_cache.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found on XPR marketplace")

        payment = max_payment or agent.price_per_call

        try:
            # Create escrow payment
            escrow_data = {
                "from": self.account_name,
                "to": agent.owner_account,
                "quantity": f"{payment:.4f} XPR",
                "memo": json.dumps({"task": task, "agent": agent_id}),
            }

            result = await self._rpc_call("chain/push_transaction", {
                "actions": [{
                    "account": "xpragents",
                    "name": "hiragent",
                    "authorization": [{"actor": self.account_name, "permission": "active"}],
                    "data": escrow_data,
                }]
            })

            logger.info(f"Hired agent {agent_id} for {payment} XPR — task: {task.get('action', 'unknown')}")

            return {
                "status": "submitted",
                "agent_id": agent_id,
                "escrow_id": result.get("transaction_id", ""),
                "payment_xpr": payment,
                "task": task,
            }

        except Exception as e:
            logger.error(f"Failed to hire agent {agent_id}: {e}")
            raise

    # ==========================================
    # A2A Protocol
    # ==========================================

    async def send_a2a_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send an Agent-to-Agent protocol message.

        Supports: request, response, broadcast, negotiate
        """
        try:
            msg = {
                "from": self.account_name,
                "to": target_agent,
                "msg_type": message_type,
                "payload": json.dumps(payload),
                "timestamp": datetime.now().isoformat(),
            }

            result = await self._rpc_call("chain/push_transaction", {
                "actions": [{
                    "account": "xpragents",
                    "name": "a2amsg",
                    "authorization": [{"actor": self.account_name, "permission": "active"}],
                    "data": msg,
                }]
            })

            return {
                "status": "sent",
                "tx_id": result.get("transaction_id", ""),
                "target": target_agent,
                "type": message_type,
            }

        except Exception as e:
            logger.error(f"A2A message to {target_agent} failed: {e}")
            raise

    # ==========================================
    # Escrow Management
    # ==========================================

    async def get_escrow_status(self, escrow_id: str) -> Optional[EscrowPayment]:
        """Check status of an escrow payment."""
        try:
            result = await self._rpc_call("chain/get_table_rows", {
                "code": "xpragents",
                "scope": "xpragents",
                "table": "escrows",
                "lower_bound": escrow_id,
                "upper_bound": escrow_id,
                "limit": 1,
            })

            rows = result.get("rows", [])
            if not rows:
                return None

            row = rows[0]
            return EscrowPayment(
                escrow_id=row.get("escrow_id", escrow_id),
                from_account=row.get("from", ""),
                to_account=row.get("to", ""),
                amount=float(row.get("amount", "0").split(" ")[0]),
                currency=row.get("amount", "0 XPR").split(" ")[-1],
                status=row.get("status", "unknown"),
            )

        except Exception as e:
            logger.error(f"Escrow status check failed: {e}")
            return None

    # ==========================================
    # AAC Agent Publishing
    # ==========================================

    async def publish_aac_agent(
        self,
        agent_name: str,
        capabilities: List[str],
        price_per_call: float,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Publish an AAC agent to the XPR marketplace.

        This makes AAC's capabilities available for hire by other
        agents on the XPR Network.
        """
        try:
            agent_data = {
                "owner": self.account_name,
                "name": agent_name,
                "capabilities": capabilities,
                "price": f"{price_per_call:.4f} XPR",
                "tools": tools or [],
                "a2a_enabled": True,
                "metadata": json.dumps({
                    "system": "AAC",
                    "version": "2.7.0",
                }),
            }

            result = await self._rpc_call("chain/push_transaction", {
                "actions": [{
                    "account": "xpragents",
                    "name": "regagent",
                    "authorization": [{"actor": self.account_name, "permission": "active"}],
                    "data": agent_data,
                }]
            })

            logger.info(f"Published AAC agent '{agent_name}' to XPR marketplace")

            return {
                "status": "published",
                "agent_name": agent_name,
                "tx_id": result.get("transaction_id", ""),
            }

        except Exception as e:
            logger.error(f"Failed to publish agent: {e}")
            raise
