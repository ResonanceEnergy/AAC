#!/usr/bin/env python3
"""
AAC Unified System Runner
=========================

Main entry point for the fully integrated AAC system with:
- 8 Doctrine Packs (1845 lines of operational wisdom)
- 5 Department Adapters (live metric collection)
- AZ Prime State Machine (automated risk response)
- Cross-Department Coordination

Usage:
    python run_integrated_system.py [--mode monitor|check|status]
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aac.doctrine import (
    DoctrineOrchestrator,
    AZPrimeState,
    Department,
    DOCTRINE_PACKS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'integrated_system.log'),
    ]
)
logger = logging.getLogger("AAC-System")


def print_banner():
    """Print system banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗  █████╗  ██████╗    ██████╗  ██████╗  ██████╗████████╗██████╗      ║
║    ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗     ║
║    ███████║███████║██║         ██║  ██║██║   ██║██║        ██║   ██████╔╝     ║
║    ██╔══██║██╔══██║██║         ██║  ██║██║   ██║██║        ██║   ██╔══██╗     ║
║    ██║  ██║██║  ██║╚██████╗    ██████╔╝╚██████╔╝╚██████╗   ██║   ██║  ██║     ║
║    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═════╝  ╚═════╝  ╚═════╝   ╚═╝   ╚═╝  ╚═╝     ║
║                                                                               ║
║              ACCELERATED ARBITRAGE CORP - DOCTRINE SYSTEM                     ║
║                                                                               ║
║    8 Doctrine Packs │ 38 Metrics │ 27 Failure Modes │ 29 AZ Triggers         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


def print_department_map():
    """Print department to doctrine mapping."""
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPARTMENT → DOCTRINE MAP                            │
├─────────────────────┬───────────────────────────────────────────────────────┤
│ TradingExecution    │ Pack 5: Liquidity / Market Impact / Partial Fill     │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ BigBrainIntelligence│ Pack 3: Testing / Simulation / Replay / Chaos        │
│                     │ Pack 7: Research Factory / Experimentation            │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ CentralAccounting   │ Pack 1: Risk Envelope / Capital Allocation           │
│                     │ Pack 8: Metric Canon / Truth Arbitration             │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ CryptoIntelligence  │ Pack 6: Counterparty Scoring / Venue Health          │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ SharedInfrastructure│ Pack 2: Security / Secrets / IAM / Key Custody       │
│                     │ Pack 4: Incident Response / On-Call / Postmortems    │
└─────────────────────┴───────────────────────────────────────────────────────┘
""")


def print_az_prime_states():
    """Print AZ Prime state machine."""
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AZ PRIME STATE MACHINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────┐      trigger      ┌──────────┐      trigger     ┌────────┐ │
│    │  NORMAL  │ ────────────────► │ CAUTION  │ ───────────────► │  SAFE  │ │
│    └──────────┘                   └──────────┘                  │  MODE  │ │
│         │                              │                        └────────┘ │
│         │                              │                             │      │
│         │         ┌────────────────────┘                             │      │
│         │         │                                                  │      │
│         │         ▼                                                  │      │
│         │    ┌──────────┐                                            │      │
│         └───►│   HALT   │◄───────────────────────────────────────────┘      │
│              └──────────┘                                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  NORMAL:    Full operation, all systems active                              │
│  CAUTION:   Risk throttled, increased monitoring                            │
│  SAFE_MODE: Minimal operation, liquidation allowed only                     │
│  HALT:      All execution stopped, on-call paged                            │
└─────────────────────────────────────────────────────────────────────────────┘
""")


async def run_single_check(orchestrator: DoctrineOrchestrator):
    """Run a single compliance check."""
    print("\n🔍 Running Compliance Check...")
    print("─" * 77)
    
    result = await orchestrator.run_compliance_check()
    
    # State indicator
    state_icons = {
        "NORMAL": "🟢",
        "CAUTION": "🟡",
        "SAFE_MODE": "🟠",
        "HALT": "🔴",
    }
    state_icon = state_icons.get(result['az_prime_state'], "⚪")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLIANCE CHECK RESULTS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Timestamp:        {result['timestamp'][:19]}                              │
│  AZ Prime State:   {state_icon} {result['az_prime_state']:<12}                                        │
│  Compliance Score: {result['compliance_score']:>6.2f}%                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ✅ Compliant:     {result['compliant']:>3}                                                         │
│  [WARN]️  Warnings:      {result['warnings']:>3}                                                         │
│  [CROSS] Violations:    {result['violations']:>3}                                                         │
│  [MONITOR] Metrics:       {result['metrics_checked']:>3}                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
""")


async def run_status(orchestrator: DoctrineOrchestrator):
    """Display current system status."""
    status = orchestrator.get_system_status()
    
    print("\n[MONITOR] SYSTEM STATUS")
    print("─" * 77)
    print(f"  AZ Prime State: {status['az_prime_state']}")
    print(f"  Last Check: {status['last_check'] or 'Never'}")
    print(f"  Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
    
    print("\n📋 DEPARTMENT STATUS")
    print("─" * 77)
    
    for dept_name, info in status['departments'].items():
        packs = info.get('doctrine_packs', [])
        metrics = info.get('total_metrics', 0)
        failures = info.get('total_failure_modes', 0)
        
        print(f"\n  🏢 {dept_name}")
        print(f"     Metrics: {metrics} | Failure Modes: {failures}")
        for pack in packs:
            print(f"     • Pack {pack['pack_id']}: {pack['name']}")


async def run_monitor(orchestrator: DoctrineOrchestrator):
    """Run continuous monitoring."""
    print("\n🔄 Starting Continuous Monitoring (Ctrl+C to stop)...")
    print("─" * 77)
    
    try:
        await orchestrator.start_monitoring()
    except KeyboardInterrupt:
        orchestrator.stop_monitoring()
        print("\n⏹️  Monitoring stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AAC Doctrine Integrated System")
    parser.add_argument(
        '--mode',
        choices=['check', 'monitor', 'status'],
        default='check',
        help='Operation mode: check (single), monitor (continuous), status (display)'
    )
    args = parser.parse_args()
    
    # Ensure logs directory exists
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    
    # Print banner
    print_banner()
    
    # Initialize orchestrator
    orchestrator = DoctrineOrchestrator()
    await orchestrator.initialize()
    
    # Print maps
    print_department_map()
    print_az_prime_states()
    
    # Run based on mode
    if args.mode == 'check':
        await run_single_check(orchestrator)
    elif args.mode == 'status':
        await run_status(orchestrator)
    elif args.mode == 'monitor':
        await run_monitor(orchestrator)
    
    print("\n" + "═" * 77)
    print("✅ AAC Doctrine System - Integration Complete")
    print("═" * 77 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
