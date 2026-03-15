#!/usr/bin/env python3
"""
AAC — Full Automation Script
=============================
One command to rule them all.  Runs every pre-launch check,
patches the .env, executes tests, runs the pipeline, and
optionally commits + pushes to git.

Usage
-----
    .venv\\Scripts\\python automate.py                 # full pre-flight (no git push)
    .venv\\Scripts\\python automate.py --commit        # + git commit
    .venv\\Scripts\\python automate.py --commit --push # + git push
    .venv\\Scripts\\python automate.py --go-live       # flip to live trading (DANGER)
    .venv\\Scripts\\python automate.py --schedule      # install Windows Task Scheduler job
    .venv\\Scripts\\python automate.py --pipeline      # run paper pipeline after checks

Phases
------
    1. Environment validation (.env present, Python version, venv)
    2. .env completeness audit (fill missing defaults)
    3. Config.from_env() validation
    4. Core import smoke test
    5. Full pytest suite
    6. Paper pipeline run (--pipeline)
    7. Git commit (--commit)
    8. Git push  (--commit --push)
    9. Windows Task Scheduler setup (--schedule)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ── UTF-8 stdout fix for Windows Task Scheduler ────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── ANSI colours ───────────────────────────────────────────────────────────
_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str: return _c("92", t)
def _red(t: str) -> str: return _c("91", t)
def _yellow(t: str) -> str: return _c("93", t)
def _cyan(t: str) -> str: return _c("96", t)
def _bold(t: str) -> str: return _c("1", t)


def _ok(msg: str) -> None: print(f"  {_green('[OK]')}   {msg}")
def _fail(msg: str) -> None: print(f"  {_red('[FAIL]')} {msg}")
def _warn(msg: str) -> None: print(f"  {_yellow('[WARN]')} {msg}")
def _info(msg: str) -> None: print(f"  {_cyan('[INFO]')} {msg}")


BANNER = r"""
  ================================================================
     AAC  AUTOMATE  —  Full Pre-Flight & Launch Automation
     v3.0-alpha  |  BARREN WUFFET / AZ SUPREME
  ================================================================
"""

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — Environment Validation
# ════════════════════════════════════════════════════════════════════════

def phase1_environment() -> bool:
    """Validate Python version, venv, and basic env."""
    print(_bold("\n[Phase 1] Environment Validation"))
    ok = True

    # Python version
    v = sys.version_info
    if (3, 9) <= v[:2] <= (3, 13):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} — need 3.9-3.13")
        ok = False

    # Venv
    venv_dir = PROJECT_ROOT / ".venv"
    if venv_dir.is_dir():
        _ok(f".venv exists at {venv_dir}")
    else:
        _fail("No .venv directory — run: python setup_machine.py")
        ok = False

    # .env file
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        _ok(".env file present")
    else:
        template = PROJECT_ROOT / ".env.example"
        if template.exists():
            shutil.copy2(template, env_file)
            _warn(".env created from .env.example — fill in your API keys")
        else:
            _fail("No .env or .env.example found")
            ok = False

    # Git repo
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.is_dir():
        _ok("Git repository detected")
    else:
        _warn("Not a git repo — git operations will be skipped")

    return ok


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — .env Completeness (auto-fill safe defaults)
# ════════════════════════════════════════════════════════════════════════

# These are settings that MUST exist in .env with safe defaults.
# They will be appended if missing — API keys are left empty.
ENV_REQUIRED_DEFAULTS: Dict[str, str] = {
    # Trading mode — safe defaults
    "AAC_ENV": "development",
    "PAPER_TRADING": "true",
    "LIVE_TRADING_ENABLED": "false",
    "DRY_RUN": "true",
    # Risk management
    "MAX_POSITION_SIZE_USD": "10000",
    "MAX_DAILY_LOSS_USD": "1000",
    "MAX_OPEN_POSITIONS": "10",
    "STRATEGY_MAX_ALLOCATION_PCT": "25",
    "MAX_DAILY_TRADES": "100",
    "EMERGENCY_STOP_LOSS": "0.10",
    "DEFAULT_RISK_PERCENT": "0.01",
    "MAX_DRAWDOWN_PERCENT": "0.05",
    # FX defaults
    "FX_SPREAD_BPS": "50",
    "FX_POLL_INTERVAL": "60",
    # Database
    "DATABASE_URL": "sqlite:///CentralAccounting/data/accounting.db",
    # IBKR defaults (paper mode)
    "IBKR_HOST": "127.0.0.1",
    "IBKR_PORT": "7497",
    "IBKR_CLIENT_ID": "1",
    "IBKR_PAPER": "true",
    # Misc safe defaults
    "BINANCE_TESTNET": "true",
    "MOOMOO_PAPER": "true",
    "TRADIER_SANDBOX": "true",
    "OPENCLAW_GATEWAY_URL": "ws://127.0.0.1:18789",
    "OPENCLAW_DAILY_SPEND_LIMIT": "10.00",
    "WEBAUTH_APP_ID": "aac_trading",
    "METAL_BLOCKCHAIN_RPC_URL": "https://tahoe.metalblockchain.org/ext/bc/C/rpc",
    "POLYGON_RPC_URL": "https://polygon-rpc.com",
    "ARBITRUM_RPC_URL": "https://arb1.arbitrum.io/rpc",
    "SMTP_HOST": "smtp.gmail.com",
    "SMTP_PORT": "587",
    "SLACK_CHANNEL": "#trading-alerts",
    "REDDIT_USER_AGENT": "AAC-Trading-Bot/1.0",
    "MT5_SERVER": "NoxiRise-Live",
    "MT5_LOGIN": "0",
    "KAFKA_BROKER": "localhost:9092",
    "KAFKA_TOPIC_TRADES": "aac.trades",
    "KAFKA_TOPIC_SIGNALS": "aac.signals",
    "KAFKA_TOPIC_ALERTS": "aac.alerts",
    "KAFKA_GROUP_ID": "aac-consumer-group",
    "REDIS_URL": "redis://localhost:6379/0",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0",
    "DASHBOARD_HOST": "0.0.0.0",
    "DASHBOARD_PORT": "8050",
    "DASHBOARD_URL": "http://localhost:3000",
    "API_PORT": "8000",
    "DEBUG": "false",
}


def _read_env_keys(env_path: Path) -> Dict[str, str]:
    """Parse .env file and return all key=value pairs."""
    data: Dict[str, str] = {}
    if not env_path.exists():
        return data
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                data[key.strip()] = value.strip()
    return data


def phase2_env_completeness() -> int:
    """Ensure all required .env keys exist.  Returns count of keys added."""
    print(_bold("\n[Phase 2] .env Completeness Audit"))
    env_path = PROJECT_ROOT / ".env"
    existing = _read_env_keys(env_path)
    added = 0

    missing_lines: List[str] = []
    for key, default in ENV_REQUIRED_DEFAULTS.items():
        if key not in existing:
            missing_lines.append(f"{key}={default}")
            added += 1

    if missing_lines:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write("\n# --- Auto-added by automate.py ---\n")
            for line in missing_lines:
                f.write(f"{line}\n")
        _warn(f"Added {added} missing defaults to .env")
        for line in missing_lines:
            _info(f"  + {line.split('=')[0]}")
    else:
        _ok("All required .env keys present")

    # Report API keys status
    api_keys_to_check = [
        ("COINGECKO_API_KEY", "CoinGecko (crypto data)", True),
        ("IBKR_ACCOUNT", "IBKR (equities/futures)", True),
        ("FX_API_KEY", "ExchangeRate-API (FX)", False),
        ("FINNHUB_API_KEY", "Finnhub (stocks)", False),
        ("POLYGON_API_KEY", "Polygon.io (stocks)", False),
        ("ALPHAVANTAGE_API_KEY", "Alpha Vantage", False),
        ("FRED_API_KEY", "FRED (economics)", False),
        ("TELEGRAM_BOT_TOKEN", "Telegram alerts", False),
        ("BINANCE_API_KEY", "Binance (crypto)", False),
        ("NDAX_API_KEY", "NDAX (crypto/CA)", False),
    ]
    print()
    configured_count = 0
    for key, name, important in api_keys_to_check:
        val = existing.get(key, "")
        if val:
            _ok(f"{name}")
            configured_count += 1
        elif important:
            _warn(f"{name} — not configured (recommended)")
        else:
            _info(f"{name} — not configured (optional)")

    print(f"\n  API keys configured: {configured_count}/{len(api_keys_to_check)}")
    return added


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — Config Validation
# ════════════════════════════════════════════════════════════════════════

def phase3_config_validation() -> bool:
    """Load Config.from_env() and run validate()."""
    print(_bold("\n[Phase 3] Config Validation"))
    try:
        from shared.config_loader import Config
        config = Config.from_env()
        result = config.validate()

        if result["valid"]:
            _ok("Configuration valid")
        else:
            for issue in result.get("issues", []):
                _fail(f"  {issue}")

        for warning in result.get("warnings", []):
            _warn(f"  {warning}")

        exchanges = result.get("exchanges_configured", [])
        if exchanges:
            _ok(f"Exchanges: {', '.join(exchanges)}")
        else:
            _warn("No exchanges configured")

        _info(f"Environment: {result.get('environment', '?')}")
        _info(f"Dry run: {result.get('dry_run', '?')}")

        return result["valid"]

    except Exception as e:
        _fail(f"Config load failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — Core Import Smoke Test
# ════════════════════════════════════════════════════════════════════════

CORE_MODULES = [
    "shared.config_loader",
    "shared.data_sources",
    "shared.audit_logger",
    "shared.monitoring",
    "core.orchestrator",
    "TradingExecution.execution_engine",
    "CentralAccounting.database",
    "BigBrainIntelligence.agents",
    "CryptoIntelligence.crypto_intelligence_engine",
    "strategies.golden_ratio_finance",
    "strategies.forex_arb_strategy",
    "integrations.knightsbridge_fx_client",
    "shared.forex_data_source",
]


def phase4_imports() -> Tuple[int, int]:
    """Import core modules.  Returns (passed, failed)."""
    print(_bold("\n[Phase 4] Core Import Smoke Test"))
    passed = failed = 0
    for mod_name in CORE_MODULES:
        try:
            importlib.import_module(mod_name)
            _ok(f"import {mod_name}")
            passed += 1
        except Exception as exc:
            _fail(f"import {mod_name}: {exc.__class__.__name__}: {exc}")
            failed += 1
    print(f"\n  Imports: {passed} passed, {failed} failed")
    return passed, failed


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — Pytest Suite
# ════════════════════════════════════════════════════════════════════════

def phase5_tests() -> Tuple[int, bool]:
    """Run the full pytest suite.  Returns (exit_code, passed)."""
    print(_bold("\n[Phase 5] Test Suite"))
    cmd = [
        sys.executable, "-m", "pytest",
        "--timeout=30", "-q", "--tb=short",
        "--ignore=tests/security_integration_test.py",
        "--ignore=tests/test_bridge_integration.py",
        "--ignore=tests/test_ecb_api.py",
        "--ignore=tests/test_market_data_quick.py",
    ]
    _info(f"Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    passed = result.returncode == 0
    if passed:
        _ok("All tests passed")
    else:
        _fail(f"Tests failed (exit code {result.returncode})")
    return result.returncode, passed


# ════════════════════════════════════════════════════════════════════════
# PHASE 6 — Paper Pipeline Run
# ════════════════════════════════════════════════════════════════════════

def phase6_pipeline() -> bool:
    """Run pipeline_runner.py in paper mode."""
    print(_bold("\n[Phase 6] Paper Pipeline Run"))
    cmd = [sys.executable, "pipeline_runner.py"]
    _info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=120)
    passed = result.returncode == 0
    if passed:
        _ok("Pipeline completed successfully")
    else:
        _fail(f"Pipeline failed (exit code {result.returncode})")
    return passed


# ════════════════════════════════════════════════════════════════════════
# PHASE 7 — Git Operations
# ════════════════════════════════════════════════════════════════════════

def phase7_git(commit: bool = False, push: bool = False) -> bool:
    """Git status, optional commit and push."""
    print(_bold("\n[Phase 7] Git Operations"))
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.is_dir():
        _warn("Not a git repo — skipping")
        return True

    # Status
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    changes = result.stdout.strip().splitlines()
    modified = [l for l in changes if l.startswith(" M") or l.startswith("M ")]
    new = [l for l in changes if l.startswith("??") or l.startswith("A ")]
    _info(f"{len(changes)} total changes ({len(modified)} modified, {len(new)} new/untracked)")

    if not changes:
        _ok("Working tree clean")
        return True

    if not commit:
        _info("Use --commit to commit changes")
        return True

    # Stage all
    subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT))

    # Commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"v3.0-alpha: AAC automated commit — {ts}"
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        _ok(f"Committed: {msg}")
    else:
        _warn(f"Commit returned {result.returncode}: {result.stderr.strip()}")
        return False

    if push:
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            _ok("Pushed to origin/main")
        else:
            _fail(f"Push failed: {result.stderr.strip()}")
            return False
    else:
        _info("Use --push to push after commit")

    return True


# ════════════════════════════════════════════════════════════════════════
# PHASE 8 — Windows Task Scheduler
# ════════════════════════════════════════════════════════════════════════

TASK_NAME = "AAC_Automated_Pipeline"
TASK_XML_TEMPLATE = r"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AAC Automated Trading Pipeline — runs every 15 minutes</Description>
    <Author>AAC</Author>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT15M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-01-01T06:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{python_exe}</Command>
      <Arguments>automate.py --pipeline</Arguments>
      <WorkingDirectory>{project_root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""


def phase8_schedule() -> bool:
    """Create a Windows Task Scheduler job."""
    print(_bold("\n[Phase 8] Windows Task Scheduler Setup"))
    if platform.system() != "Windows":
        _warn("Task Scheduler is Windows-only — create a cron job manually")
        return True

    python_exe = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    if not Path(python_exe).exists():
        _fail(f"Python exe not found: {python_exe}")
        return False

    xml_content = TASK_XML_TEMPLATE.format(
        python_exe=python_exe,
        project_root=str(PROJECT_ROOT),
    )

    xml_path = PROJECT_ROOT / "aac_scheduled_task.xml"
    with open(xml_path, "w", encoding="utf-16") as f:
        f.write(xml_content)

    _info(f"Task XML written to {xml_path}")
    _info("To install (run as Administrator):")
    _info(f'  schtasks /create /tn "{TASK_NAME}" /xml "{xml_path}" /f')
    _info(f'To remove: schtasks /delete /tn "{TASK_NAME}" /f')
    _info(f'To run now: schtasks /run /tn "{TASK_NAME}"')

    # Try to install automatically
    result = subprocess.run(
        ["schtasks", "/create", "/tn", TASK_NAME, "/xml", str(xml_path), "/f"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        _ok(f"Task '{TASK_NAME}' installed — runs every 15 minutes")
    else:
        _warn(f"Auto-install failed (may need admin): {result.stderr.strip()}")
        _info("Run the schtasks command above as Administrator")

    return True


# ════════════════════════════════════════════════════════════════════════
# GO LIVE — Flip production switches
# ════════════════════════════════════════════════════════════════════════

def go_live() -> bool:
    """Switch .env from paper to live trading.  DANGEROUS."""
    print(_bold(_red("\n[GO LIVE] Switching to production mode")))
    print(_red("  ╔═══════════════════════════════════════════════╗"))
    print(_red("  ║  WARNING: This enables REAL MONEY trading!   ║"))
    print(_red("  ║  Make sure ALL paper tests pass first!       ║"))
    print(_red("  ╚═══════════════════════════════════════════════╝"))

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        _fail("No .env file")
        return False

    content = env_path.read_text(encoding="utf-8")

    replacements = {
        "DRY_RUN=true": "DRY_RUN=false",
        "PAPER_TRADING=true": "PAPER_TRADING=false",
        "AAC_ENV=development": "AAC_ENV=production",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    # Add live trading confirmation if not present
    if "LIVE_TRADING_CONFIRMATION" not in content:
        content += "\nLIVE_TRADING_CONFIRMATION=YES_I_UNDERSTAND\n"
    else:
        content = content.replace(
            "# LIVE_TRADING_CONFIRMATION=YES_I_UNDERSTAND",
            "LIVE_TRADING_CONFIRMATION=YES_I_UNDERSTAND",
        )

    env_path.write_text(content, encoding="utf-8")
    _warn("DRY_RUN=false, PAPER_TRADING=false, AAC_ENV=production")
    _warn("LIVE_TRADING_CONFIRMATION=YES_I_UNDERSTAND")
    _ok("Production mode enabled — trades WILL execute with real money")
    return True


# ════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════

def print_summary(results: Dict[str, bool | str | int]) -> None:
    """Print a final roll-up of all phases."""
    print(_bold("\n" + "=" * 60))
    print(_bold("  AAC AUTOMATION SUMMARY"))
    print("=" * 60)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Timestamp: {ts}")
    print()

    all_ok = True
    for phase, status in results.items():
        if isinstance(status, bool):
            icon = _green("PASS") if status else _red("FAIL")
            if not status:
                all_ok = False
        else:
            icon = str(status)
        print(f"  {phase:<35} {icon}")

    print()
    if all_ok:
        print(_green("  ALL PHASES PASSED — system is launch-ready"))
    else:
        print(_yellow("  Some phases had warnings/failures — review above"))

    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="automate",
        description="AAC Full Automation — pre-flight checks, tests, pipeline, deploy",
    )
    parser.add_argument("--commit", action="store_true", help="Git add + commit after tests")
    parser.add_argument("--push", action="store_true", help="Git push after commit (requires --commit)")
    parser.add_argument("--pipeline", action="store_true", help="Run paper pipeline after tests")
    parser.add_argument("--schedule", action="store_true", help="Install Windows Task Scheduler job")
    parser.add_argument("--go-live", action="store_true", help="Switch .env to live trading (DANGER)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest (faster iteration)")
    args = parser.parse_args()

    print(BANNER)
    start = time.monotonic()
    results: Dict[str, bool | str | int] = {}

    # Phase 1 — Environment
    results["1. Environment"] = phase1_environment()

    # Phase 2 — .env completeness
    added = phase2_env_completeness()
    results["2. .env completeness"] = True  # non-blocking
    if added:
        results["2. .env keys added"] = added

    # Phase 3 — Config validation
    results["3. Config validation"] = phase3_config_validation()

    # Phase 4 — Core imports
    passed, failed = phase4_imports()
    results["4. Import smoke tests"] = failed == 0

    # Phase 5 — Tests
    if not args.skip_tests:
        exit_code, test_ok = phase5_tests()
        results["5. Pytest suite"] = test_ok
    else:
        _warn("Tests skipped (--skip-tests)")
        results["5. Pytest suite"] = "SKIPPED"

    # Phase 6 — Pipeline
    if args.pipeline:
        results["6. Paper pipeline"] = phase6_pipeline()
    else:
        results["6. Paper pipeline"] = "SKIPPED (use --pipeline)"

    # Phase 7 — Git
    results["7. Git operations"] = phase7_git(commit=args.commit, push=args.push)

    # Phase 8 — Scheduler
    if args.schedule:
        results["8. Task Scheduler"] = phase8_schedule()
    else:
        results["8. Task Scheduler"] = "SKIPPED (use --schedule)"

    # Go Live
    if args.go_live:
        results["9. GO LIVE"] = go_live()

    elapsed = time.monotonic() - start
    results["Total time"] = f"{elapsed:.1f}s"

    print_summary(results)

    # Write results to a log file for auditing
    log_path = PROJECT_ROOT / "logs" / "automate_log.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "results": {k: str(v) for k, v in results.items()},
        "args": vars(args),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    _info(f"Log appended to {log_path}")

    # Exit code: 0 if all bool results are True
    all_pass = all(v for v in results.values() if isinstance(v, bool))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
