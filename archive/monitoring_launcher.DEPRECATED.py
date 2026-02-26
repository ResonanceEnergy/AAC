#!/usr/bin/env python3
"""
AAC 2100 Monitoring Launcher
============================
Unified launcher for AAC 2100 monitoring systems.

⚠️  DEPRECATED: This file is deprecated and will be removed.
   Use 'python aac_master_launcher.py --monitoring-only' instead.
   Or use 'python aac_master_launcher.py --dashboard-only' for dashboard only.
   Or use 'python aac_master_launcher.py --service-only' for service only.

New unified launcher:
    python aac_master_launcher.py --monitoring-only   # Full monitoring
    python aac_master_launcher.py --dashboard-only    # Dashboard only
    python aac_master_launcher.py --service-only      # Service only

Launches:
1. Continuous Monitoring Service (background)
2. Real-time Monitoring Dashboard (terminal UI)

Usage:
    python monitoring_launcher.py [options]  # DEPRECATED

Options:
    --dashboard-only    Launch only the monitoring dashboard
    --service-only      Launch only the continuous monitoring service
    --no-dashboard      Launch service without dashboard
    --config FILE       Use specific config file
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import signal
import os

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from continuous_monitoring import ContinuousMonitoringService
from aac_master_monitoring_dashboard import get_master_dashboard, DisplayMode


class MonitoringLauncher:
    """Launcher for AAC 2100 monitoring systems"""

    def __init__(self):
        self.service = None
        self.dashboard = None
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/monitoring_launcher.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('MonitoringLauncher')

    async def launch_full_monitoring(self):
        """Launch both service and dashboard"""
        self.logger.info("[DEPLOY] Launching AAC 2100 Full Monitoring System...")

        try:
            # Initialize service
            self.service = ContinuousMonitoringService()
            await self.service.initialize()

            # Start service in background
            service_task = asyncio.create_task(self.service.start_monitoring())

            # Small delay to let service initialize
            await asyncio.sleep(2)

            # Launch dashboard
            self.dashboard = get_master_dashboard(display_mode=DisplayMode.TERMINAL)
            await self.dashboard.initialize()

            # Start dashboard
            dashboard_task = asyncio.create_task(self.dashboard.run_dashboard())

            self.logger.info("✅ Full monitoring system launched successfully")

            # Wait for both tasks
            await asyncio.gather(service_task, dashboard_task, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"[CROSS] Failed to launch monitoring system: {e}")
            await self.cleanup()

    async def launch_service_only(self):
        """Launch only the continuous monitoring service"""
        self.logger.info("[DEPLOY] Launching AAC 2100 Continuous Monitoring Service...")

        try:
            self.service = ContinuousMonitoringService()
            await self.service.initialize()
            await self.service.start_monitoring()
        except Exception as e:
            self.logger.error(f"[CROSS] Failed to launch monitoring service: {e}")
            await self.cleanup()

    async def launch_dashboard_only(self):
        """Launch only the monitoring dashboard"""
        self.logger.info("[DEPLOY] Launching AAC 2100 Monitoring Dashboard...")

        try:
            self.dashboard = get_master_dashboard(display_mode=DisplayMode.TERMINAL)
            await self.dashboard.initialize()
            await self.dashboard.run_dashboard()
        except Exception as e:
            self.logger.error(f"[CROSS] Failed to launch monitoring dashboard: {e}")
            await self.cleanup()

    async def cleanup(self):
        """Clean up running services"""
        self.logger.info("🧹 Cleaning up monitoring systems...")

        if self.service:
            try:
                await self.service.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping service: {e}")

        if self.dashboard:
            try:
                await self.dashboard.cleanup()
            except Exception as e:
                self.logger.error(f"Error stopping dashboard: {e}")

        self.logger.info("✅ Cleanup completed")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        self.running = False
        asyncio.create_task(self.cleanup())


async def main():
    """Main entry point"""
    print("⚠️  DEPRECATED: monitoring_launcher.py is deprecated!")
    print("   Use: python aac_master_launcher.py --monitoring-only")
    print("   Or:   python aac_master_launcher.py --dashboard-only")
    print("   Or:   python aac_master_launcher.py --service-only")
    print()

    parser = argparse.ArgumentParser(description='AAC 2100 Monitoring Launcher (DEPRECATED)')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Launch only the monitoring dashboard')
    parser.add_argument('--service-only', action='store_true',
                       help='Launch only the continuous monitoring service')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Launch service without dashboard')
    parser.add_argument('--config', type=str,
                       help='Use specific config file')

    args = parser.parse_args()

    launcher = MonitoringLauncher()

    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)

    try:
        if args.dashboard_only:
            await launcher.launch_dashboard_only()
        elif args.service_only:
            await launcher.launch_service_only()
        elif args.no_dashboard:
            await launcher.launch_service_only()
        else:
            await launcher.launch_full_monitoring()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down monitoring systems...")
        await launcher.cleanup()
    except Exception as e:
        print(f"[CROSS] Monitoring launcher failed: {e}")
        await launcher.cleanup()


if __name__ == "__main__":
    asyncio.run(main())