"""
AAC Command Center Interface
===========================

Interactive Command Center with Real-Time Dashboard, Avatar Interactions, and Executive Control
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import json
import os
import platform
import random

logger = logging.getLogger(__name__)

# Try to import curses, fallback to text-based interface if not available
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    logger.info("Warning: curses module not available. Using text-based interface.")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from core.command_center import get_command_center, AACCommandCenter
from shared.avatar_system import get_avatar_manager
from shared.config_loader import get_config

class CommandCenterInterface:
    """
    Interactive Command Center Interface
    Real-time dashboard with avatar interactions and executive control
    """

    def __init__(self):
        self.config = get_config()
        self.command_center: Optional[AACCommandCenter] = None
        self.avatars = {}
        self.current_view = "dashboard"
        self.selected_avatar = None
        self.chat_history = []
        self.alerts = []
        self.refresh_rate = 2.0  # seconds

        # Interface state
        self.running = False
        self.last_update = datetime.now()

    async def initialize(self):
        """Initialize the command center interface"""
        logger.info("[DEPLOY] Initializing AAC Command Center Interface...")

        # Initialize command center
        self.command_center = await get_command_center()

        # Initialize avatars
        _mgr = get_avatar_manager()
        self.avatars = {
            "supreme": _mgr.get_avatar("supreme"),
            "helix": _mgr.get_avatar("helix")
        }

        # Start background tasks
        self.running = True

        logger.info("✅ Command Center Interface initialized")

    async def run_interface(self):
        """Run the command center interface"""
        try:
            if not CURSES_AVAILABLE or platform.system() == "Windows":
                await self._run_text_interface()
            else:
                await self._run_curses_interface()
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down Command Center Interface...")
        except Exception as e:
            logger.info(f"[CROSS] Interface error: {e}")
        finally:
            await self._shutdown()

    async def _run_curses_interface(self):
        """Run curses-based interface (Unix/Linux/Mac)"""
        def curses_main(stdscr):
            """Curses main."""
            asyncio.run(self._curses_loop(stdscr))

        curses.wrapper(curses_main)

    async def _curses_loop(self, stdscr):
        """Main curses interface loop"""
        # Setup curses
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)  # 100ms timeout

        # Color setup
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)    # Success
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)      # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Info
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Avatar
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)     # Header

        max_y, max_x = stdscr.getmaxyx()

        while self.running:
            try:
                # Clear screen
                stdscr.clear()

                # Draw interface
                await self._draw_curses_interface(stdscr, max_y, max_x)

                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                elif key == ord('1'):
                    self.current_view = "dashboard"
                elif key == ord('2'):
                    self.current_view = "avatars"
                elif key == ord('3'):
                    self.current_view = "alerts"
                elif key == ord('4'):
                    self.current_view = "executive"
                elif key == ord('r'):
                    # Refresh data
                    pass

                # Update display
                stdscr.refresh()

                # Small delay
                await asyncio.sleep(0.1)

            except Exception as e:
                stdscr.addstr(0, 0, f"Error: {e}")
                stdscr.refresh()
                await asyncio.sleep(1)

    async def _draw_curses_interface(self, stdscr, max_y: int, max_x: int):
        """Draw the curses interface"""
        # Header
        header = "AAC 2100 COMMAND & CONTROL CENTER"
        stdscr.addstr(0, (max_x - len(header)) // 2, header, curses.color_pair(6) | curses.A_BOLD)

        # Status bar
        status = await self._get_status_summary()
        stdscr.addstr(1, 0, f"Status: {status}", curses.color_pair(4))

        # Menu
        menu_y = 3
        stdscr.addstr(menu_y, 0, "1: Dashboard  2: Avatars  3: Alerts  4: Executive  Q: Quit  R: Refresh", curses.A_BOLD)

        # Content area
        content_y = 5

        if self.current_view == "dashboard":
            await self._draw_dashboard(stdscr, content_y, max_y, max_x)
        elif self.current_view == "avatars":
            await self._draw_avatars(stdscr, content_y, max_y, max_x)
        elif self.current_view == "alerts":
            await self._draw_alerts(stdscr, content_y, max_y, max_x)
        elif self.current_view == "executive":
            await self._draw_executive(stdscr, content_y, max_y, max_x)

        # Footer
        footer_y = max_y - 1
        stdscr.addstr(footer_y, 0, f"Last Update: {self.last_update.strftime('%H:%M:%S')}", curses.color_pair(4))

    async def _draw_dashboard(self, stdscr, start_y: int, max_y: int, max_x: int):
        """Draw main dashboard view"""
        y_pos = start_y

        # System Health
        stdscr.addstr(y_pos, 0, "🖥️  SYSTEM HEALTH", curses.A_BOLD)
        y_pos += 1

        status = await self.command_center.get_command_center_status()
        system_health = status.get("real_time_metrics", {}).get("system_health", {})

        stdscr.addstr(y_pos, 0, f"CPU: {system_health.get('cpu_usage', 0):.1f}%")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Memory: {system_health.get('memory_usage', 0):.1f}%")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Network: {system_health.get('network_latency', 0):.1f}ms")
        y_pos += 2

        # Financial Overview
        stdscr.addstr(y_pos, 0, "[MONEY] FINANCIAL OVERVIEW", curses.A_BOLD)
        y_pos += 1

        financial = status.get("real_time_metrics", {}).get("financial", {})
        stdscr.addstr(y_pos, 0, f"Total Equity: ${financial.get('total_equity', 0):,.0f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Daily P&L: ${financial.get('daily_pnl', 0):,.0f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Unrealized P&L: ${financial.get('unrealized_pnl', 0):,.0f}")
        y_pos += 2

        # Integration Status
        stdscr.addstr(y_pos, 0, "🔗 INTEGRATIONS", curses.A_BOLD)
        y_pos += 1

        integrations = status.get("integrations", {})
        gln_status = "✅" if integrations.get("gln_active") else "[CROSS]"
        gta_status = "✅" if integrations.get("gta_active") else "[CROSS]"

        stdscr.addstr(y_pos, 0, f"GLN: {gln_status} GTA: {gta_status}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Critical Hiring: {integrations.get('critical_hiring_needs', 0)}")
        y_pos += 2

        # Avatar Status
        stdscr.addstr(y_pos, 0, "[AI] AVATARS", curses.A_BOLD)
        y_pos += 1

        avatars = status.get("avatar_status", {})
        for avatar_name, avatar_data in avatars.items():
            emotion = avatar_data.get("mood", "unknown")
            stdscr.addstr(y_pos, 0, f"{avatar_data.get('name', avatar_name)}: {emotion}")
            y_pos += 1

    async def _draw_avatars(self, stdscr, start_y: int, max_y: int, max_x: int):
        """Draw avatars interaction view"""
        y_pos = start_y

        stdscr.addstr(y_pos, 0, "[AI] AI AVATAR INTERACTIONS", curses.A_BOLD | curses.color_pair(5))
        y_pos += 2

        # Avatar selection
        stdscr.addstr(y_pos, 0, "Select Avatar:")
        y_pos += 1
        stdscr.addstr(y_pos, 0, "S: AZ SUPREME (Strategic Advisor)")
        y_pos += 1
        stdscr.addstr(y_pos, 0, "H: AX HELIX (Operations Commander)")
        y_pos += 2

        # Current avatar info
        if self.selected_avatar:
            avatar_data = self.avatars.get(self.selected_avatar)
            if avatar_data:
                status = await avatar_data.get_avatar_status()
                stdscr.addstr(y_pos, 0, f"Current: {status['name']}", curses.A_BOLD)
                y_pos += 1
                stdscr.addstr(y_pos, 0, f"Emotion: {status['current_emotion']}")
                y_pos += 1
                stdscr.addstr(y_pos, 0, f"Confidence: {status['confidence_level']:.2f}")
                y_pos += 1
                stdscr.addstr(y_pos, 0, f"Interactions: {status['interaction_count']}")
                y_pos += 2

        # Chat history (last few messages)
        stdscr.addstr(y_pos, 0, "Recent Interactions:", curses.A_BOLD)
        y_pos += 1

        recent_chat = self.chat_history[-5:] if self.chat_history else []
        for i, chat in enumerate(recent_chat):
            if y_pos + i < max_y - 2:
                avatar = chat.get("avatar", "Unknown")
                message = chat.get("message", "")[:50]
                stdscr.addstr(y_pos + i, 0, f"{avatar}: {message}...")

    async def _draw_alerts(self, stdscr, start_y: int, max_y: int, max_x: int):
        """Draw alerts view"""
        y_pos = start_y

        stdscr.addstr(y_pos, 0, "[ALERT] SYSTEM ALERTS", curses.A_BOLD | curses.color_pair(2))
        y_pos += 2

        # Active alerts
        alerts = self.alerts[-10:] if self.alerts else []
        if not alerts:
            stdscr.addstr(y_pos, 0, "No active alerts", curses.color_pair(1))
        else:
            for i, alert in enumerate(alerts):
                if y_pos + i < max_y - 2:
                    severity = alert.get("severity", "unknown")
                    color = curses.color_pair(2) if severity == "critical" else curses.color_pair(3)
                    message = alert.get("message", "Unknown alert")[:60]
                    stdscr.addstr(y_pos + i, 0, f"[{severity.upper()}] {message}", color)

    async def _draw_executive(self, stdscr, start_y: int, max_y: int, max_x: int):
        """Draw executive control view"""
        y_pos = start_y

        stdscr.addstr(y_pos, 0, "👑 EXECUTIVE OVERSIGHT", curses.A_BOLD | curses.color_pair(5))
        y_pos += 2

        status = await self.command_center.get_command_center_status()
        executive = status.get("executive", {})

        # Executive status
        stdscr.addstr(y_pos, 0, "Executive Agents:", curses.A_BOLD)
        y_pos += 1
        supreme_status = "✅ Active" if executive.get("supreme_active") else "[CROSS] Inactive"
        helix_status = "✅ Active" if executive.get("helix_active") else "[CROSS] Inactive"
        stdscr.addstr(y_pos, 0, f"AZ SUPREME: {supreme_status}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"AX HELIX: {helix_status}")
        y_pos += 2

        # Performance metrics
        stdscr.addstr(y_pos, 0, "Performance Metrics:", curses.A_BOLD)
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Strategic Decisions: {executive.get('strategic_decisions', 0)}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Operations Optimized: {executive.get('operations_optimized', 0)}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Integrations Completed: {executive.get('integrations_completed', 0)}")
        y_pos += 2

        # Command options
        stdscr.addstr(y_pos, 0, "Executive Commands:", curses.A_BOLD)
        y_pos += 1
        stdscr.addstr(y_pos, 0, "A: Strategic Analysis  O: Operational Review")
        y_pos += 1
        stdscr.addstr(y_pos, 0, "R: Risk Assessment    P: Performance Report")

    async def _run_text_interface(self):
        """Run text-based interface (Windows/fallback)"""
        logger.info("AAC 2100 Command & Control Center")
        logger.info("=" * 50)

        while self.running:
            try:
                # Display current view
                if self.current_view == "dashboard":
                    await self._display_text_dashboard()
                elif self.current_view == "avatars":
                    await self._display_text_avatars()
                elif self.current_view == "alerts":
                    await self._display_text_alerts()
                elif self.current_view == "executive":
                    await self._display_text_executive()

                # Menu
                logger.info("\nCommands:")
                logger.info("1: Dashboard  2: Avatars  3: Alerts  4: Executive")
                logger.info("Q: Quit  R: Refresh")
                logger.info("-" * 50)

                # Get input
                try:
                    cmd = input("Command: ").strip().lower()
                except KeyboardInterrupt:
                    cmd = 'q'

                if cmd == 'q':
                    self.running = False
                elif cmd == '1':
                    self.current_view = "dashboard"
                elif cmd == '2':
                    self.current_view = "avatars"
                elif cmd == '3':
                    self.current_view = "alerts"
                elif cmd == '4':
                    self.current_view = "executive"
                elif cmd == 'r':
                    # Refresh
                    pass
                elif cmd in ['s', 'h'] and self.current_view == "avatars":
                    self.selected_avatar = "supreme" if cmd == 's' else "helix"
                    await self._handle_avatar_interaction()
                elif self.current_view == "executive" and cmd in ['a', 'o', 'r', 'p']:
                    await self._handle_executive_command(cmd)

                # Clear screen for next iteration (ANSI escape — no shell injection risk)
                logger.info("\033[2J\033[H", end="", flush=True)

            except Exception as e:
                logger.info(f"Interface error: {e}")
                await asyncio.sleep(2)

    async def _display_text_dashboard(self):
        """Display text-based dashboard"""
        status = await self.command_center.get_command_center_status()

        logger.info("🖥️  SYSTEM HEALTH")
        system_health = status.get("real_time_metrics", {}).get("system_health", {})
        logger.info(f"  CPU: {system_health.get('cpu_usage', 0):.1f}%")
        logger.info(f"  Memory: {system_health.get('memory_usage', 0):.1f}%")
        logger.info(f"  Network: {system_health.get('network_latency', 0):.1f}ms")

        logger.info("\n[MONEY] FINANCIAL OVERVIEW")
        financial = status.get("real_time_metrics", {}).get("financial", {})
        logger.info(f"  Total Equity: ${financial.get('total_equity', 0):,.0f}")
        logger.info(f"  Daily P&L: ${financial.get('daily_pnl', 0):,.0f}")
        logger.info(f"  Unrealized P&L: ${financial.get('unrealized_pnl', 0):,.0f}")

        logger.info("\n🔗 INTEGRATIONS")
        integrations = status.get("integrations", {})
        gln_status = "✅ Active" if integrations.get("gln_active") else "[CROSS] Inactive"
        gta_status = "✅ Active" if integrations.get("gta_active") else "[CROSS] Inactive"
        logger.info(f"  GLN: {gln_status}  GTA: {gta_status}")
        logger.info(f"  Critical Hiring Needs: {integrations.get('critical_hiring_needs', 0)}")

        logger.info("\n[AI] AVATARS")
        avatars = status.get("avatar_status", {})
        for avatar_name, avatar_data in avatars.items():
            logger.info(f"  {avatar_data.get('name')}: {avatar_data.get('mood')}")

    async def _display_text_avatars(self):
        """Display text-based avatar interface"""
        logger.info("[AI] AI AVATAR INTERACTIONS")
        logger.info("S: Select AZ SUPREME  H: Select AX HELIX")

        if self.selected_avatar:
            avatar = self.avatars.get(self.selected_avatar)
            if avatar:
                status = await avatar.get_avatar_status()
                logger.info(f"\nCurrent Avatar: {status['name']}")
                logger.info(f"Emotion: {status['current_emotion']}")
                logger.info(f"Confidence: {status['confidence_level']:.2f}")
                logger.info(f"Interactions: {status['interaction_count']}")

                logger.info("\nType your message to the avatar (or 'back' to return):")

    async def _display_text_alerts(self):
        """Display text-based alerts"""
        logger.info("[ALERT] SYSTEM ALERTS")

        alerts = self.alerts[-10:] if self.alerts else []
        if not alerts:
            logger.info("No active alerts")
        else:
            for alert in alerts:
                severity = alert.get("severity", "unknown")
                message = alert.get("message", "Unknown alert")
                logger.info(f"[{severity.upper()}] {message}")

    async def _display_text_executive(self):
        """Display text-based executive interface"""
        logger.info("👑 EXECUTIVE OVERSIGHT")

        status = await self.command_center.get_command_center_status()
        executive = status.get("executive", {})

        logger.info("Executive Agents:")
        supreme_status = "✅ Active" if executive.get("supreme_active") else "[CROSS] Inactive"
        helix_status = "✅ Active" if executive.get("helix_active") else "[CROSS] Inactive"
        logger.info(f"  AZ SUPREME: {supreme_status}")
        logger.info(f"  AX HELIX: {helix_status}")

        logger.info("\nPerformance Metrics:")
        logger.info(f"  Strategic Decisions: {executive.get('strategic_decisions', 0)}")
        logger.info(f"  Operations Optimized: {executive.get('operations_optimized', 0)}")
        logger.info(f"  Integrations Completed: {executive.get('integrations_completed', 0)}")

        logger.info("\nCommands:")
        logger.info("A: Strategic Analysis  O: Operational Review")
        logger.info("R: Risk Assessment     P: Performance Report")

    async def _handle_avatar_interaction(self):
        """Handle avatar interaction in text mode"""
        if not self.selected_avatar:
            return

        avatar = self.avatars.get(self.selected_avatar)
        if not avatar:
            return

        try:
            message = input("You: ").strip()
            if message.lower() == 'back':
                self.selected_avatar = None
                return

            if message:
                # Send message to avatar
                response = await avatar.process_query(message)

                # Add to chat history
                self.chat_history.append({
                    "avatar": avatar.avatar_name,
                    "user_message": message,
                    "avatar_response": response.text,
                    "timestamp": datetime.now().isoformat()
                })

                logger.info(f"\n{avatar.avatar_name}: {response.text}")

                # Trigger speech if available
                if hasattr(response, 'speech_config'):
                    await avatar._synthesize_speech(response.text, response.speech_config)

        except Exception as e:
            logger.info(f"Avatar interaction error: {e}")

    async def _handle_executive_command(self, cmd: str):
        """Handle executive commands"""
        try:
            if cmd == 'a':
                # Strategic analysis
                response = await self.command_center.interact_with_avatar("supreme", "Provide a strategic analysis of current market conditions and system performance.")
                logger.info(f"AZ SUPREME: {response}")
            elif cmd == 'o':
                # Operational review
                response = await self.command_center.interact_with_avatar("helix", "Provide an operational review of system efficiency and integration status.")
                logger.info(f"AX HELIX: {response}")
            elif cmd == 'r':
                # Risk assessment
                response = await self.command_center.interact_with_avatar("supreme", "Conduct a comprehensive risk assessment across all departments.")
                logger.info(f"AZ SUPREME: {response}")
            elif cmd == 'p':
                # Performance report
                response = await self.command_center.interact_with_avatar("helix", "Generate a performance report with key metrics and optimization recommendations.")
                logger.info(f"AX HELIX: {response}")

        except Exception as e:
            logger.info(f"Executive command error: {e}")

    async def _run_text_interface(self):
        """Run enhanced text-based interface"""
        logger.info("\n" + "═"*80)
        logger.info("🎛️  AAC COMMAND CENTER - EXECUTIVE CONTROL INTERFACE")
        logger.info("═"*80)
        logger.info("Accelerated Arbitrage Corp - Real-time Operations & Intelligence Hub")
        logger.info("═"*80)

        while self.running:
            try:
                # Display status header
                status = await self._get_status_summary()
                now = datetime.now()
                logger.info(f"\n📅 {now.strftime('%H:%M:%S')} | {status}")
                logger.info("─" * 80)

                # Main command menu
                logger.info("⚡ COMMAND CENTER OPERATIONS")
                logger.info("─" * 80)
                logger.info("┌─────────────────────────────────────────────────────────────────────────────┐")
                logger.info("│  1. 📊 Executive Dashboard     2. 🤖 AI Avatar Interactions                 │")
                logger.info("│  3. 👑 Executive Oversight     4. 🔧 System Diagnostics                    │")
                logger.info("│  5. 💰 Financial Analytics     6. ⚠️  Risk Management                      │")
                logger.info("│  7. 🎯 Start Agent Contest     8. 🛑 Emergency Stop                        │")
                logger.info("│                                                                             │")
                logger.info("│  0. 🚪 Exit Command Center                                                │")
                logger.info("└─────────────────────────────────────────────────────────────────────────────┘")
                logger.debug("")

                # Get user input
                cmd = input("🎯 Select operation (0-8): ").strip()

                if cmd == '0':
                    logger.info("\n🔄 Shutting down AAC Command Center...")
                    self.running = False
                elif cmd == '1':
                    await self._show_dashboard()
                    input("\n⏎ Press Enter to return to main menu...")
                elif cmd == '2':
                    await self._show_avatars()
                    input("\n⏎ Press Enter to return to main menu...")
                elif cmd == '3':
                    await self._show_executive_commands()
                    input("\n⏎ Press Enter to return to main menu...")
                elif cmd == '4':
                    await self._show_system_status()
                    input("\n⏎ Press Enter to return to main menu...")
                elif cmd == '5':
                    await self._show_financial_insights()
                    input("\n⏎ Press Enter to return to main menu...")
                elif cmd == '6':
                    logger.info("\n⚠️  Risk Management - Feature coming soon...")
                    await asyncio.sleep(1)
                elif cmd == '7':
                    logger.info("\n🎯 Starting Agent Contest...")
                    logger.info("Agent contest feature not yet available")
                    logger.info("\u2139\ufe0f  Agent contest feature not yet available.")
                    await asyncio.sleep(1)
                elif cmd == '8':
                    logger.info("\n\U0001f6d1 EMERGENCY STOP - All operations halted!")
                    try:
                        self.running = False
                        if self.command_center:
                            await self.command_center.shutdown_command_center()
                        logger.warning("Emergency stop executed via command center interface")
                    except Exception as e:
                        logger.error(f"Emergency stop error: {e}")
                    await asyncio.sleep(2)
                else:
                    logger.info("❌ Invalid command. Please select 0-8.")

                # Clear screen effect
                logger.info("\n" * 2)

            except KeyboardInterrupt:
                logger.info("\n\n🛑 Emergency shutdown initiated...")
                self.running = False
            except Exception as e:
                logger.info(f"\n❌ Interface error: {e}")
                await asyncio.sleep(2)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.info(f"Interface error: {e}")
                await asyncio.sleep(1)

    async def _show_dashboard(self):
        """Show enhanced dashboard view with comprehensive metrics"""
        logger.info("\n" + "═"*80)
        logger.info("🏛️  AAC COMMAND CENTER - EXECUTIVE DASHBOARD")
        logger.info("═"*80)

        try:
            status = await self.command_center.get_command_center_status()

            # System timestamp and mode
            now = datetime.now()
            mode = status.get("mode", "unknown").replace("_", " ").title()
            logger.info(f"📅 {now.strftime('%Y-%m-%d %H:%M:%S')} | Mode: {mode}")
            logger.debug("")

            # Executive Summary
            logger.info("📈 EXECUTIVE SUMMARY")
            logger.info("─" * 40)

            # System Health
            system_health = status.get("system_health", {})
            cpu = system_health.get('cpu_usage', 0)
            mem = system_health.get('memory_usage', 0)
            health_score = system_health.get('health_score', 85)  # Default if not available

            health_icon = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
            logger.info(f"System Health: {health_icon} {health_score:.1f}/100")

            # Financial Metrics
            financial = status.get("real_time_metrics", {}).get("financial", {})
            equity = financial.get('total_equity', 0)
            daily_pnl = financial.get('daily_pnl', 0)
            unrealized_pnl = financial.get('unrealized_pnl', 0)

            logger.info(f"Total Equity:   ${equity:,.0f}")
            logger.info(f"Daily P&L:      ${daily_pnl:+,.0f}")
            logger.info(f"Unrealized P&L: ${unrealized_pnl:+,.0f}")
            logger.debug("")

            # Core Metrics Grid
            logger.info("📊 CORE METRICS")
            logger.info("─" * 40)

            # System Resources
            logger.info("System Resources:")
            logger.info(f"  CPU Usage:     {self._progress_bar(cpu)} {cpu:.1f}%")
            logger.info(f"  Memory Usage:  {self._progress_bar(mem)} {mem:.1f}%")
            logger.info(f"  Uptime:        {self._format_uptime(system_health.get('uptime_seconds', 0))}")
            logger.debug("")

            # Trading Activity
            logger.info("Trading Activity:")
            trading = status.get("trading_metrics", {})
            active_trades = trading.get('active_trades', 0)
            today_trades = trading.get('today_trades', 0)
            win_rate = financial.get('win_rate', 0)

            logger.info(f"  Active Trades: {active_trades}")
            logger.info(f"  Today's Trades: {today_trades}")
            logger.info(f"  Win Rate:      {win_rate:.1%}")
            logger.debug("")

            # Integrations Status
            logger.info("🔗 INTEGRATIONS")
            logger.info("─" * 40)

            integrations = status.get("integrations", {})
            gln_status = "✅ Active" if integrations.get("gln_active") else "❌ Inactive"
            gta_status = "✅ Active" if integrations.get("gta_active") else "❌ Inactive"

            logger.info(f"Global Logistics Network: {gln_status}")
            logger.info(f"Global Talent Analytics:   {gta_status}")
            logger.info(f"Critical Hiring Needs:     {integrations.get('critical_hiring_needs', 0)}")
            logger.debug("")

            # AI Avatars Status
            logger.info("[AI] AI AVATARS")
            logger.info("─" * 40)

            avatars = status.get("avatar_status", {})
            if avatars:
                for avatar_name, avatar_data in avatars.items():
                    emotion = avatar_data.get("mood", "unknown")
                    confidence = avatar_data.get("confidence", 0.0)
                    logger.info(f"  {avatar_data.get('name', avatar_name)}: {emotion} (Confidence: {confidence:.2f})")
            else:
                logger.info("  No avatars currently active")
            logger.debug("")

            # Agent Contest Status
            logger.info("🤖 AGENT CONTEST")
            logger.info("─" * 40)

            contest = status.get("agent_contest", {})
            if contest.get("status") == "active":
                logger.info(f"Status:          🟢 ACTIVE")
                logger.info(f"Active Agents:   {contest.get('active_agents', 0)}")
                logger.info(f"Total Agents:    {contest.get('total_agents', 0)}")
                logger.info(f"Rounds:          {contest.get('rounds_completed', 0)}")

                leaderboard = contest.get('leaderboard', [])
                if leaderboard:
                    logger.info("Top Performers:")
                    for i, agent in enumerate(leaderboard[:3], 1):
                        agent_id = agent.get('agent_id', 'Unknown')
                        value = agent.get('portfolio_value', 0)
                        logger.info(f"  {i}. {agent_id}: ${value:,.0f}")
            else:
                logger.info(f"Status:          🔴 {contest.get('status', 'inactive').upper()}")
            logger.debug("")

            # Quick Actions
            logger.info("⚡ QUICK ACTIONS")
            logger.info("─" * 40)
            logger.info("1: View Detailed Metrics    2: Avatar Interactions")
            logger.info("3: Executive Commands       4: System Diagnostics")
            logger.info("5: Financial Insights      6: Risk Management")
            logger.info("7: Start Agent Contest     8: Stop All Operations")
            logger.info("0: Exit Command Center")
            logger.debug("")

        except Exception as e:
            logger.info(f"❌ Error retrieving dashboard data: {e}")
            logger.debug("")

    def _progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable form"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 and not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts) if parts else "0s"

    async def _show_avatars(self):
        """Show avatar interactions"""
        logger.info("\n[AI] AI AVATAR INTERACTIONS")
        logger.info("-" * 25)

        logger.info("Available Avatars:")
        logger.info("S: AZ SUPREME (Strategic Advisor)")
        logger.info("H: AX HELIX (Operations Commander)")

        avatar_choice = input("\nSelect avatar (S/H): ").strip().upper()

        if avatar_choice in ['S', 'H']:
            avatar_name = "supreme" if avatar_choice == 'S' else "helix"
            avatar_display = "AZ SUPREME" if avatar_choice == 'S' else "AX HELIX"

            query = input(f"\n💬 Ask {avatar_display}: ")
            if query.strip():
                try:
                    response = await self.command_center.interact_with_avatar(avatar_name, query)
                    logger.info(f"\n{avatar_display}: {response}")
                except Exception as e:
                    logger.info(f"Error communicating with avatar: {e}")
        else:
            logger.info("Invalid selection.")

    async def _show_executive_commands(self):
        """Show executive command menu"""
        logger.info("\n👑 EXECUTIVE COMMANDS")
        logger.info("-" * 20)

        logger.info("Available Commands:")
        logger.info("A: Strategic Analysis")
        logger.info("O: Operational Review")
        logger.info("R: Risk Assessment")
        logger.info("P: Performance Report")

        cmd = input("\nEnter command: ").strip().lower()

        try:
            if cmd == 'a':
                response = await self.command_center.interact_with_avatar("supreme", "Provide a strategic analysis of current market conditions and system performance.")
                logger.info(f"\nAZ SUPREME: {response}")
            elif cmd == 'o':
                response = await self.command_center.interact_with_avatar("helix", "Provide an operational review of system efficiency and integration status.")
                logger.info(f"\nAX HELIX: {response}")
            elif cmd == 'r':
                response = await self.command_center.interact_with_avatar("supreme", "Conduct a comprehensive risk assessment across all departments.")
                logger.info(f"\nAZ SUPREME: {response}")
            elif cmd == 'p':
                response = await self.command_center.interact_with_avatar("helix", "Generate a performance report with key metrics and optimization recommendations.")
                logger.info(f"\nAX HELIX: {response}")
            else:
                logger.info("Invalid command.")
        except Exception as e:
            logger.info(f"Executive command error: {e}")

    async def _show_system_status(self):
        """Show detailed system status"""
        logger.info("\n🔧 SYSTEM STATUS")
        logger.info("-" * 15)

        try:
            status = await self.command_center.get_command_center_status()

            logger.info(f"Operational Readiness: {'✅ Yes' if status.get('operational_readiness') else '[CROSS] No'}")
            logger.info(f"Mode: {status.get('mode', 'unknown').replace('_', ' ').title()}")

            # Executive Branch
            executive = status.get("executive", {})
            logger.info(f"AZ SUPREME: {'✅ Active' if executive.get('supreme_active') else '[CROSS] Inactive'}")
            logger.info(f"AX HELIX: {'✅ Active' if executive.get('helix_active') else '[CROSS] Inactive'}")

            # Department Status
            departments = status.get("departments", {})
            logger.info("\nDepartment Status:")
            for dept, dept_status in departments.items():
                active = "✅" if dept_status.get("active") else "[CROSS]"
                logger.info(f"  {dept}: {active}")

        except Exception as e:
            logger.info(f"Error retrieving system status: {e}")

    async def _show_financial_insights(self):
        """Show financial insights"""
        logger.info("\n[MONEY] FINANCIAL INSIGHTS")
        logger.info("-" * 20)

        try:
            insights = await self.command_center.get_financial_insights()
            logger.info(f"Total Insights Available: {len(insights)}")

            # Show first 10 insights
            for i, insight in enumerate(insights[:10]):
                logger.info(f"{i+1}. {insight.get('category', 'Unknown')}: {insight.get('insight', 'No details')}")

            if len(insights) > 10:
                logger.info(f"... and {len(insights) - 10} more insights available")

        except Exception as e:
            logger.info(f"Error retrieving financial insights: {e}")

    async def _get_status_summary(self) -> str:
        """Get status summary string"""
        try:
            status = await self.command_center.get_command_center_status()

            operational = "✅ Operational" if status.get("operational_readiness") else "[CROSS] Initializing"
            mode = status.get("mode", "unknown").replace("_", " ").title()

            return f"{operational} | Mode: {mode}"
        except Exception as e:
            logging.getLogger(__name__).debug(f"Status summary error: {e}")
            return "Status: Unknown"

    async def _shutdown(self):
        """Shutdown the interface"""
        self.running = False

        if self.command_center:
            await self.command_center.shutdown_command_center()

        logger.info("Command Center Interface shutdown complete")

async def main():
    """Main entry point"""
    interface = CommandCenterInterface()

    try:
        await interface.initialize()
        await interface.run_interface()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.info(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await interface._shutdown()

if __name__ == "__main__":
    asyncio.run(main())