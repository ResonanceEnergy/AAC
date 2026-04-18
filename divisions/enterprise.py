"""AAC Enterprise -- wire all divisions and run the living arbitrage machine.

This is the top-level entry point that:
1. Instantiates all divisions (Trading arms + Research)
2. Subscribes Research -> all Trading divisions
3. Cross-subscribes War Room <-> other Trading arms
4. Starts all division loops concurrently
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from divisions.division_protocol import DivisionProtocol
from divisions.research.coordinator import ResearchCoordinator
from divisions.trading.agentbravo.daily_strategy import AgentBravoDivision
from divisions.trading.crypto.division import CryptoDivision
from divisions.trading.polymarket.division import PolymarketDivision
from divisions.trading.warroom.division import WarRoomDivision

from councils.crypto.division import CryptoCouncilDivision
from councils.polymarket.division import PolymarketCouncilDivision
from councils.youtube.division import YouTubeCouncilDivision
from councils.xai.division import XaiCouncilDivision

from divisions.trading.crypto_paper.division import CryptoPaperDivision
from divisions.trading.polymarket_paper.division import PolymarketPaperDivision

_log = structlog.get_logger()


class Enterprise:
    """The living arbitrage enterprise.

    Manages all divisions, their interconnections, and lifecycle.
    """

    def __init__(self) -> None:
        # -- Instantiate divisions -------------------------------------------
        self.research = ResearchCoordinator()
        self.warroom = WarRoomDivision()
        self.agentbravo = AgentBravoDivision()
        self.polymarket = PolymarketDivision()
        self.crypto = CryptoDivision()

        # -- Council divisions (intelligence scraping) -----------------------
        self.youtube_council = YouTubeCouncilDivision()
        self.xai_council = XaiCouncilDivision()
        self.polymarket_council = PolymarketCouncilDivision()
        self.crypto_council = CryptoCouncilDivision()

        # -- Paper trading divisions (strategy bots) -------------------------
        self.polymarket_paper = PolymarketPaperDivision()
        self.crypto_paper = CryptoPaperDivision()

        self._divisions: list[DivisionProtocol] = [
            self.research,
            self.warroom,
            self.agentbravo,
            self.polymarket,
            self.crypto,
            self.youtube_council,
            self.xai_council,
            self.polymarket_council,
            self.crypto_council,
            self.polymarket_paper,
            self.crypto_paper,
        ]

        # -- Wire inter-division subscriptions -------------------------------
        # Research publishes to all Trading divisions
        self.research.subscribe(self.warroom)
        self.research.subscribe(self.agentbravo)
        self.research.subscribe(self.polymarket)
        self.research.subscribe(self.crypto)

        # War Room publishes milestones to all other arms
        self.warroom.subscribe(self.agentbravo)
        self.warroom.subscribe(self.polymarket)
        self.warroom.subscribe(self.crypto)

        # Crypto publishes venue health to War Room
        self.crypto.subscribe(self.warroom)

        # Councils publish intel to Research and War Room
        self.youtube_council.subscribe(self.research)
        self.youtube_council.subscribe(self.warroom)
        self.xai_council.subscribe(self.research)
        self.xai_council.subscribe(self.warroom)
        self.polymarket_council.subscribe(self.research)
        self.polymarket_council.subscribe(self.warroom)
        self.crypto_council.subscribe(self.research)
        self.crypto_council.subscribe(self.warroom)

        # Councils feed intel to paper trading divisions
        self.polymarket_council.subscribe(self.polymarket_paper)
        self.crypto_council.subscribe(self.crypto_paper)

        # Paper traders report trade signals to War Room
        self.polymarket_paper.subscribe(self.warroom)
        self.crypto_paper.subscribe(self.warroom)

        _log.info("enterprise.initialized", divisions=len(self._divisions))

    async def start(self) -> None:
        """Start all divisions concurrently."""
        _log.info("enterprise.starting")

        # Scan intervals per division (seconds)
        intervals = {
            "research": 300.0,     # 5 minutes
            "warroom": 60.0,       # 1 minute
            "agentbravo": 3600.0,  # 1 hour (daily strategy)
            "polymarket": 300.0,   # 5 minutes
            "crypto": 600.0,       # 10 minutes (dormant)
            "youtube_council": 1800.0,  # 30 minutes
            "xai_council": 900.0,       # 15 minutes
            "polymarket_council": 900.0,  # 15 minutes
            "crypto_council": 600.0,      # 10 minutes
            "polymarket_paper": 300.0,    # 5 minutes (paper trading)
            "crypto_paper": 600.0,        # 10 minutes (paper trading)
        }

        tasks = []
        for div in self._divisions:
            interval = intervals.get(div.division_name, 60.0)
            tasks.append(asyncio.create_task(
                div.run_loop(interval=interval),
                name=f"division_{div.division_name}",
            ))
            _log.info("enterprise.division_started", division=div.division_name, interval=interval)

        _log.info("enterprise.all_divisions_running", count=len(tasks))
        await asyncio.gather(*tasks)

    async def status(self) -> dict[str, Any]:
        """Get enterprise-wide status."""
        statuses = {}
        for div in self._divisions:
            hb = await div.heartbeat()
            report = await div.report()
            statuses[div.division_name] = {
                "health": hb.health.value,
                "uptime": hb.uptime_seconds,
                "cycles": hb.cycle_count,
                "report": report,
            }
        return {
            "enterprise": "AAC",
            "divisions": statuses,
            "division_count": len(self._divisions),
        }

    async def stop(self) -> None:
        """Stop all divisions."""
        for div in self._divisions:
            await div.stop()
        _log.info("enterprise.stopped")
