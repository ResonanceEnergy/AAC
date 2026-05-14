#!/usr/bin/env python3
"""
Activate PlanktonXD Browser Bot — AAC v2.7.0
=============================================

Browser automation version of the PlanktonXD prediction market harvester.
Uses Selenium WebDriver to automate Polymarket interactions while maintaining
all the core PlanktonXD strategies and risk controls.

Usage:
    python scripts/activate_planktonxd_browser_bot.py                    # Single cycle, headless
    python scripts/activate_planktonxd_browser_bot.py --visible          # Single cycle, visible browser  
    python scripts/activate_planktonxd_browser_bot.py --continuous       # Continuous trading
    python scripts/activate_planktonxd_browser_bot.py --test-browser     # Test browser setup
    python scripts/activate_planktonxd_browser_bot.py --status           # Check bot status

Features:
- Browser-based market scanning and trade execution
- All PlanktonXD strategies: deep OTM, spread harvesting, liquidity sniping
- Robust error handling and retry logic
- Screenshot audit trails
- Session persistence
- Risk controls and position sizing
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# UTF-8 fix for Windows Task Scheduler
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("planktonxd.browser_activator")


async def test_browser_setup():
    """Test browser setup and dependencies."""
    print("\n" + "=" * 70)
    print("  PLANKTONXD BROWSER BOT — SETUP TEST")
    print("=" * 70)
    
    checks = {}
    
    # Test Selenium import
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        checks["Selenium"] = "OK Available"
    except ImportError as e:
        checks["Selenium"] = f"FAIL {e}"
        return checks
    
    # Test ChromeDriver
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        driver_path = ChromeDriverManager().install()
        checks["ChromeDriver"] = f"OK {driver_path}"
    except Exception as e:
        checks["ChromeDriver"] = f"FAIL {e}"
    
    # Test browser creation
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        checks["Browser Test"] = f"OK Chrome headless working"
        driver.quit()
    except Exception as e:
        checks["Browser Test"] = f"FAIL {e}"
    
    # Test AAC integration
    try:
        from agents.planktonxd_browser_bot import create_planktonxd_browser_bot
        from shared.communication import CommunicationFramework
        from shared.audit_logger import AuditLogger
        
        comm = CommunicationFramework()
        audit = AuditLogger()
        bot = create_planktonxd_browser_bot(comm, audit, headless=True)
        checks["AAC Integration"] = "OK Factory function working"
    except Exception as e:
        checks["AAC Integration"] = f"FAIL {e}"
    
    # Print results
    for check, status in checks.items():
        print(f"  {check:20}: {status}")
    
    all_passed = all("OK" in status for status in checks.values())
    if all_passed:
        print("\n  ✅ All browser setup checks passed!")
    else:
        print("\n  ❌ Some setup checks failed. See above for details.")
    
    print("=" * 70)
    return checks


async def run_single_cycle(bot, cycle_num: int = 1):
    """Run a single PlanktonXD trading cycle."""
    print(f"\n  🔄 Cycle {cycle_num} — PlanktonXD Browser Bot")
    print(f"     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = await bot.run_planktonxd_cycle()
        
        # Display results
        print(f"  📊 Markets scanned: {results.get('markets_scanned', 0)}")
        print(f"  🎯 Opportunities found: {results.get('opportunities_found', 0)}")
        print(f"  🔄 Trades attempted: {results.get('trades_attempted', 0)}")
        print(f"  ✅ Trades successful: {results.get('trades_successful', 0)}")
        print(f"  💰 Total deployed: ${results.get('total_deployed', 0):,.2f}")
        
        if results.get('errors'):
            print(f"  ⚠️  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"     {error}")
        
        return results
        
    except Exception as e:
        logger.error(f"Cycle {cycle_num} failed: {e}")
        print(f"  ❌ Cycle {cycle_num} failed: {str(e)}")
        return {"error": str(e)}


async def run_continuous_trading(bot, delay_minutes: int = 30, max_cycles: int = 100):
    """Run continuous PlanktonXD trading."""
    print(f"\n  🔄 Continuous Trading Mode")
    print(f"     Delay: {delay_minutes} minutes between cycles")
    print(f"     Max cycles: {max_cycles}")
    
    cycle = 0
    start_time = datetime.now()
    
    try:
        while cycle < max_cycles:
            cycle += 1
            
            # Run cycle
            results = await run_single_cycle(bot, cycle)
            
            # Check for shutdown conditions
            status = await bot.get_portfolio_status()
            
            if status['bankroll'] < 50:  # Below minimum bankroll
                print(f"\n  ⚠️  Bankroll too low (${status['bankroll']}) — stopping")
                break
            
            if cycle < max_cycles:
                next_run = datetime.now() + timedelta(minutes=delay_minutes)
                print(f"  ⏰ Next cycle: {next_run.strftime('%H:%M:%S')}")
                print(f"  💰 Current bankroll: ${status['bankroll']:,.2f}")
                await asyncio.sleep(delay_minutes * 60)
        
        # Final summary
        runtime = datetime.now() - start_time
        final_status = await bot.get_portfolio_status()
        
        print(f"\n  📈 CONTINUOUS TRADING COMPLETE")
        print(f"     Runtime: {runtime}")
        print(f"     Cycles: {cycle}")
        print(f"     Final bankroll: ${final_status['bankroll']:,.2f}")
        print(f"     Total bets: {final_status['total_bets']}")
        
    except KeyboardInterrupt:
        print(f"\n  ⏸️  Continuous trading interrupted by user")
    except Exception as e:
        logger.error(f"Continuous trading error: {e}")
        print(f"  ❌ Continuous trading error: {str(e)}")


async def show_bot_status():
    """Show current bot status without running trades."""
    print("\n" + "=" * 70)
    print("  PLANKTONXD BROWSER BOT — STATUS")
    print("=" * 70)
    
    try:
        # Check session state file
        session_dir = Path("_scratch/planktonxd_session")
        state_file = session_dir / "planktonxd_state.json"
        
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            
            print(f"  💰 Bankroll: ${state.get('bankroll', 0):,.2f}")
            print(f"  📊 Total bets: {state['stats'].get('total_bets', 0)}")
            print(f"  💸 Total invested: ${state['stats'].get('total_invested', 0):,.2f}")
            print(f"  ✅ Winning bets: {state['stats'].get('winning_bets', 0)}")
            print(f"  ❌ Losing bets: {state['stats'].get('losing_bets', 0)}")
            print(f"  📅 Last run: {state.get('last_run', 'Never')}")
            
            # Category exposure
            exposure = state.get('category_exposure', {})
            if exposure:
                print(f"\n  📈 Category Exposure:")
                for cat, amount in exposure.items():
                    if amount > 0:
                        print(f"     {cat}: ${amount:.2f}")
        else:
            print("  ℹ️  No previous session found")
        
        # Check screenshot directory
        screenshots_dir = Path("_scratch/planktonxd_screenshots")
        if screenshots_dir.exists():
            screenshots = list(screenshots_dir.glob("*.png"))
            print(f"\n  📸 Screenshots: {len(screenshots)} available")
            if screenshots:
                latest = max(screenshots, key=lambda p: p.stat().st_mtime)
                print(f"     Latest: {latest.name}")
        
    except Exception as e:
        print(f"  ❌ Error reading status: {e}")
    
    print("=" * 70)


def get_strategy_status():
    """Check PlanktonXD Browser Bot registration status."""
    print("\n" + "=" * 70)
    print("  PLANKTONXD BROWSER BOT — SYSTEM STATUS")
    print("=" * 70)
    
    checks = {}
    
    # Check browser bot import
    try:
        from agents.planktonxd_browser_bot import (
            PlanktonXDBrowserBot,
            create_planktonxd_browser_bot,
        )
        checks["Browser Bot class"] = "OK Loaded"
        checks["Factory function"] = "OK Available"
    except ImportError as e:
        checks["Browser Bot import"] = f"FAIL {e}"
    
    # Check Selenium
    try:
        from selenium import webdriver
        checks["Selenium WebDriver"] = "OK Loaded"
    except ImportError as e:
        checks["Selenium WebDriver"] = f"FAIL {e}"
    
    # Check original PlanktonXD
    try:
        from strategies.planktonxd_prediction_harvester import PlanktonXDPredictionHarvester
        checks["Original PlanktonXD"] = "OK Available (for comparison)"
    except ImportError as e:
        checks["Original PlanktonXD"] = f"WARN {e}"
    
    # Check directories
    try:
        screenshots_dir = Path("_scratch/planktonxd_screenshots")
        session_dir = Path("_scratch/planktonxd_session")
        
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        checks["Directories"] = f"OK Created ({screenshots_dir}, {session_dir})"
    except Exception as e:
        checks["Directories"] = f"FAIL {e}"
    
    for name, status in checks.items():
        print(f"  {name:25}: {status}")
    
    print("=" * 70)
    return checks


async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PlanktonXD Browser Bot Activation")
    parser.add_argument("--visible", action="store_true", help="Run browser in visible mode (not headless)")
    parser.add_argument("--test-browser", action="store_true", help="Test browser setup")
    parser.add_argument("--status", action="store_true", help="Show bot status") 
    parser.add_argument("--continuous", action="store_true", help="Run continuous trading")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (no real trades)")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Starting bankroll")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles (single mode)")
    parser.add_argument("--delay", type=int, default=30, help="Minutes between cycles (continuous mode)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  PLANKTONXD BROWSER BOT — ACTIVATION")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'Visible' if args.visible else 'Headless'} Browser")
    if args.simulate:
        print(f"  🎮 SIMULATION MODE — No real trades will be executed")
    
    # Always show system status first
    get_strategy_status()
    
    if args.test_browser:
        await test_browser_setup()
        return
    
    if args.status:
        await show_bot_status()
        return
    
    # Handle simulation mode
    if args.simulate:
        import os
        # Temporarily set DRY_RUN for this session
        os.environ["DRY_RUN"] = "true"
        os.environ["PAPER_TRADING"] = "true"
        os.environ["LIVE_TRADING_ENABLED"] = "false"
        print(f"  🛡️  Safety: DRY_RUN=true, PAPER_TRADING=true, LIVE_TRADING_ENABLED=false")
        print(f"  📝 All trades will be simulated and logged only")
    
    # Create bot instance
    try:
        from agents.planktonxd_browser_bot import create_planktonxd_browser_bot
        from shared.communication import CommunicationFramework
        from shared.audit_logger import AuditLogger
        
        communication = CommunicationFramework()
        audit_logger = AuditLogger()
        
        bot = create_planktonxd_browser_bot(
            communication=communication,
            audit_logger=audit_logger,
            bankroll=args.bankroll,
            headless=not args.visible  # Invert: --visible means NOT headless
        )
        
        print(f"\n  🤖 PlanktonXD Browser Bot initialized")
        print(f"     Bankroll: ${args.bankroll:,.2f}")
        print(f"     Browser: {'Visible' if args.visible else 'Headless'}")
        if args.simulate:
            print(f"     Mode: 🎮 SIMULATION (no real trades)")
        else:
            print(f"     Mode: ⚡ LIVE TRADING")
        
        try:
            if args.continuous:
                await run_continuous_trading(bot, args.delay, 1000)
            else:
                # Single mode - run specified number of cycles
                for cycle in range(args.cycles):
                    await run_single_cycle(bot, cycle + 1)
                    if cycle < args.cycles - 1:
                        print(f"  ⏰ Waiting 60 seconds...")
                        await asyncio.sleep(60)
                
                # Final status
                final_status = await bot.get_portfolio_status()
                print(f"\n  📊 FINAL STATUS")
                print(f"     Bankroll: ${final_status['bankroll']:,.2f}")
                print(f"     Total Bets: {final_status['total_bets']}")
                print(f"     Open Positions: {final_status['open_positions']}")
                
        finally:
            await bot.shutdown()
            
    except KeyboardInterrupt:
        print("\n  ⏸️  Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot activation failed: {e}")
        print(f"\n  ❌ Bot activation failed: {str(e)}")
        print("  Check that Chrome browser and ChromeDriver are installed.")
        print("  Run with --test-browser to diagnose setup issues.")


if __name__ == "__main__":
    asyncio.run(main())