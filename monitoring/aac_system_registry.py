"""
AAC System Registry — Unified Status Tracker
==============================================
Single source of truth for the live status of every component in the
AAC system: APIs, exchanges, strategies, services, infrastructure,
and orphan scripts.

Used by ``AACMasterMonitoringDashboard`` to populate the master
Command & Control panel.

Usage::

    from monitoring.aac_system_registry import SystemRegistry
    reg = SystemRegistry()
    snapshot = reg.collect_full_snapshot()
    # snapshot is a dict with keys: apis, exchanges, strategies,
    # infrastructure, services, orphans, summary
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
# Status primitives
# ──────────────────────────────────────────────────────────────

class Health(str, Enum):
    """Traffic-light status."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    GREY = "grey"          # not configured / not applicable


@dataclass
class ComponentStatus:
    """Status of a single system component."""
    name: str
    category: str
    health: Health = Health.GREY
    detail: str = ""
    latency_ms: Optional[float] = None
    checked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["health"] = self.health.value
        return d


# ──────────────────────────────────────────────────────────────
# API REGISTRY (imported from tools/api_registry.py)
# ──────────────────────────────────────────────────────────────

def _load_api_registry() -> List[Dict[str, Any]]:
    """Import the canonical REGISTRY list."""
    try:
        from tools.api_registry import REGISTRY, check_env_var
        out = []
        for api in REGISTRY:
            status_str = check_env_var(api.get("env_var"))
            configured = status_str.startswith("\u2705")      # ✅
            out.append({
                "name": api["name"],
                "env_var": api.get("env_var"),
                "category": api["category"],
                "priority": api.get("priority", "LOW"),
                "configured": configured,
                "status_label": status_str,
            })
        return out
    except Exception as e:
        logger.warning("Could not load API registry: %s", e)
        return []


# ──────────────────────────────────────────────────────────────
# Port / service checks
# ──────────────────────────────────────────────────────────────

def _check_port(host: str, port: int, timeout: float = 1.0) -> tuple[bool, float]:
    """Return (is_open, latency_ms)."""
    t0 = time.monotonic()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        return False, round((time.monotonic() - t0) * 1000, 1)


# ──────────────────────────────────────────────────────────────
# EXCHANGE PROBES
# ──────────────────────────────────────────────────────────────

def _probe_exchanges() -> List[ComponentStatus]:
    """Probe all configured exchange gateways."""
    now = datetime.utcnow().isoformat()
    results: List[ComponentStatus] = []

    # IBKR TWS / Gateway
    ibkr_port = int(os.environ.get("IBKR_PORT", "7496"))  # 7496 = live, 7497 = paper
    ok, lat = _check_port("127.0.0.1", ibkr_port)
    results.append(ComponentStatus(
        name="IBKR TWS/Gateway",
        category="Exchange",
        health=Health.GREEN if ok else Health.RED,
        detail=f"port {ibkr_port} {'open' if ok else 'closed'}",
        latency_ms=lat,
        checked_at=now,
    ))

    # Moomoo OpenD
    moomoo_port = int(os.environ.get("MOOMOO_PORT", "11111"))
    ok, lat = _check_port("127.0.0.1", moomoo_port)
    results.append(ComponentStatus(
        name="Moomoo OpenD",
        category="Exchange",
        health=Health.GREEN if ok else Health.RED,
        detail=f"port {moomoo_port} {'open' if ok else 'closed'}",
        latency_ms=lat,
        checked_at=now,
    ))

    # NDAX (remote REST)
    ok, lat = _check_port("api.ndax.io", 443)
    results.append(ComponentStatus(
        name="NDAX API",
        category="Exchange",
        health=Health.GREEN if ok else Health.RED,
        detail="api.ndax.io:443 " + ("reachable" if ok else "unreachable"),
        latency_ms=lat,
        checked_at=now,
    ))

    return results


# ──────────────────────────────────────────────────────────────
# INFRASTRUCTURE PROBES
# ──────────────────────────────────────────────────────────────

def _probe_infrastructure() -> List[ComponentStatus]:
    """Probe local infrastructure services."""
    now = datetime.utcnow().isoformat()
    results: List[ComponentStatus] = []

    # Health server (port 8080)
    ok, lat = _check_port("127.0.0.1", 8080, timeout=0.5)
    results.append(ComponentStatus(
        name="Health Server",
        category="Infrastructure",
        health=Health.GREEN if ok else Health.YELLOW,
        detail=f"port 8080 {'open' if ok else 'closed'}",
        latency_ms=lat,
        checked_at=now,
    ))

    # Redis (optional)
    ok, lat = _check_port("127.0.0.1", 6379, timeout=0.5)
    results.append(ComponentStatus(
        name="Redis",
        category="Infrastructure",
        health=Health.GREEN if ok else Health.GREY,
        detail="port 6379 " + ("open" if ok else "not running"),
        latency_ms=lat,
        checked_at=now,
    ))

    # Kafka (optional)
    ok, lat = _check_port("127.0.0.1", 9092, timeout=0.5)
    results.append(ComponentStatus(
        name="Kafka",
        category="Infrastructure",
        health=Health.GREEN if ok else Health.GREY,
        detail="port 9092 " + ("open" if ok else "not running"),
        latency_ms=lat,
        checked_at=now,
    ))

    return results


# ──────────────────────────────────────────────────────────────
# MATRIX MAXIMIZER PROBE
# ──────────────────────────────────────────────────────────────

def _probe_matrix_maximizer() -> ComponentStatus:
    """Check Matrix Maximizer availability and last run data."""
    now = datetime.utcnow().isoformat()
    latest_path = PROJECT_ROOT / "data" / "matrix_maximizer_latest.json"
    if latest_path.exists():
        try:
            import json
            data = json.loads(latest_path.read_text(encoding="utf-8"))
            ts = data.get("timestamp", "?")
            mandate = data.get("forecast", {}).get("mandate", "?")
            picks = len(data.get("picks", []))
            return ComponentStatus(
                name="Matrix Maximizer",
                category="Strategy",
                health=Health.GREEN,
                detail=f"last run {ts} — mandate={mandate}, {picks} picks",
                checked_at=now,
            )
        except Exception as e:
            return ComponentStatus(
                name="Matrix Maximizer",
                category="Strategy",
                health=Health.YELLOW,
                detail=f"data file exists but unreadable: {e}",
                checked_at=now,
            )
    else:
        # Check if module importable at all
        try:
            from strategies.matrix_maximizer.runner import MatrixMaximizer  # noqa: F401
            return ComponentStatus(
                name="Matrix Maximizer",
                category="Strategy",
                health=Health.YELLOW,
                detail="importable but no run data yet",
                checked_at=now,
            )
        except Exception:
            return ComponentStatus(
                name="Matrix Maximizer",
                category="Strategy",
                health=Health.RED,
                detail="module not importable",
                checked_at=now,
            )


# ──────────────────────────────────────────────────────────────
# STRATEGY / ENGINE PROBES
# ──────────────────────────────────────────────────────────────

def _probe_strategies() -> List[ComponentStatus]:
    """Check availability of all strategy engines."""
    now = datetime.utcnow().isoformat()
    results: List[ComponentStatus] = []

    results.append(_probe_matrix_maximizer())

    # Regime Engine
    try:
        from strategies.regime_engine import RegimeEngine  # noqa: F401
        results.append(ComponentStatus(
            name="Regime Engine",
            category="Strategy",
            health=Health.GREEN,
            detail="importable",
            checked_at=now,
        ))
    except Exception:
        results.append(ComponentStatus(
            name="Regime Engine",
            category="Strategy",
            health=Health.RED,
            detail="import failed",
            checked_at=now,
        ))

    # Stock Forecaster
    try:
        from strategies.stock_forecaster import StockForecaster  # noqa: F401
        results.append(ComponentStatus(
            name="Stock Forecaster",
            category="Strategy",
            health=Health.GREEN,
            detail="importable",
            checked_at=now,
        ))
    except Exception:
        results.append(ComponentStatus(
            name="Stock Forecaster",
            category="Strategy",
            health=Health.RED,
            detail="import failed",
            checked_at=now,
        ))

    # Crypto Forecaster
    try:
        from strategies.crypto_forecaster import CryptoForecaster  # noqa: F401
        results.append(ComponentStatus(
            name="Crypto Forecaster",
            category="Strategy",
            health=Health.GREEN,
            detail="importable",
            checked_at=now,
        ))
    except Exception:
        results.append(ComponentStatus(
            name="Crypto Forecaster",
            category="Strategy",
            health=Health.RED,
            detail="import failed",
            checked_at=now,
        ))

    # Strategy Testing Lab
    try:
        from strategies.strategy_testing_lab_fixed import strategy_testing_lab  # noqa: F401
        results.append(ComponentStatus(
            name="Strategy Testing Lab",
            category="Strategy",
            health=Health.GREEN,
            detail="importable",
            checked_at=now,
        ))
    except Exception:
        results.append(ComponentStatus(
            name="Strategy Testing Lab",
            category="Strategy",
            health=Health.GREY,
            detail="not available",
            checked_at=now,
        ))

    # Storm Lifeboat Matrix
    results.append(_probe_storm_lifeboat(now))

    return results


def _probe_storm_lifeboat(now: str) -> ComponentStatus:
    """Probe Storm Lifeboat Matrix — importability + last-run data."""
    try:
        from strategies.storm_lifeboat.core import VolRegime  # noqa: F401
        from strategies.storm_lifeboat.monte_carlo import StormMonteCarloEngine  # noqa: F401
        from strategies.storm_lifeboat.scenario_engine import ScenarioEngine  # noqa: F401
    except Exception:
        return ComponentStatus(
            name="Storm Lifeboat Matrix",
            category="Strategy",
            health=Health.RED,
            detail="import failed",
            checked_at=now,
        )

    # Check for latest Helix briefing
    import glob as _glob
    briefing_dir = PROJECT_ROOT / "data" / "storm_lifeboat"
    briefings = sorted(_glob.glob(str(briefing_dir / "helix_briefing_*.json")))
    if briefings:
        try:
            import json as _json
            latest = Path(briefings[-1])
            data = _json.loads(latest.read_text(encoding="utf-8"))
            mandate = data.get("mandate", "?")
            regime = data.get("regime", "?")
            date_str = data.get("date", "?")
            n_active = len(data.get("active_scenarios", []))
            detail = (f"v9.0 — last run {date_str}, mandate={mandate}, "
                      f"regime={regime}, {n_active} active scenarios")
            return ComponentStatus(
                name="Storm Lifeboat Matrix",
                category="Strategy",
                health=Health.GREEN,
                detail=detail,
                checked_at=now,
            )
        except Exception:
            pass

    return ComponentStatus(
        name="Storm Lifeboat Matrix",
        category="Strategy",
        health=Health.YELLOW,
        detail="importable but no run data found",
        checked_at=now,
    )


# ──────────────────────────────────────────────────────────────
# DEPARTMENT PROBES
# ──────────────────────────────────────────────────────────────

def _probe_departments() -> List[ComponentStatus]:
    """Check AAC department engines."""
    now = datetime.utcnow().isoformat()
    results: List[ComponentStatus] = []

    dept_map = {
        "BigBrainIntelligence": "BigBrainIntelligence",
        "CentralAccounting": "CentralAccounting.financial_analysis_engine",
        "CryptoIntelligence": "CryptoIntelligence.crypto_intelligence_engine",
        "TradingExecution": "TradingExecution",
    }

    for dept_name, module_path in dept_map.items():
        try:
            __import__(module_path)
            results.append(ComponentStatus(
                name=dept_name,
                category="Department",
                health=Health.GREEN,
                detail="importable",
                checked_at=now,
            ))
        except Exception:
            results.append(ComponentStatus(
                name=dept_name,
                category="Department",
                health=Health.YELLOW,
                detail="import failed (non-critical)",
                checked_at=now,
            ))

    return results


# ──────────────────────────────────────────────────────────────
# ORPHAN SCRIPT INVENTORY
# ──────────────────────────────────────────────────────────────

def _inventory_orphans() -> List[Dict[str, Any]]:
    """List root _*.py orphan scripts with brief description."""
    results = []
    for p in sorted(PROJECT_ROOT.glob("_*.py")):
        name = p.name
        # Read first docstring line
        desc = ""
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        # Try to get the docstring content
                        desc = stripped.strip("\"' \n")
                        if not desc:
                            desc = next(f, "").strip().strip("\"' \n")
                        break
                    elif stripped.startswith("#"):
                        desc = stripped.lstrip("# ")
                        break
        except Exception:
            pass
        results.append({"script": name, "description": desc[:120]})
    return results


# ══════════════════════════════════════════════════════════════
# SystemRegistry — the master collector
# ══════════════════════════════════════════════════════════════

class SystemRegistry:
    """Aggregates live status from every AAC subsystem.

    The ``collect_full_snapshot()`` method returns a dict structured as::

        {
            "timestamp": "...",
            "apis": [...],           # from tools/api_registry REGISTRY
            "exchanges": [...],      # IBKR, Moomoo, NDAX
            "strategies": [...],     # Matrix Maximizer, Regime Engine, etc.
            "departments": [...],    # BBI, CA, CI, TE
            "infrastructure": [...], # health server, Redis, Kafka
            "orphans": [...],        # _*.py root scripts
            "summary": {
                "total_apis": int,
                "apis_configured": int,
                "apis_missing": int,
                "exchanges_online": int,
                "exchanges_total": int,
                "strategies_ok": int,
                "strategies_total": int,
                "departments_ok": int,
                "departments_total": int,
                "infra_ok": int,
                "infra_total": int,
            },
        }
    """

    def collect_full_snapshot(self) -> Dict[str, Any]:
        """Run all probes and return the unified snapshot."""
        now = datetime.utcnow().isoformat()

        apis = _load_api_registry()
        exchanges = _probe_exchanges()
        strategies = _probe_strategies()
        departments = _probe_departments()
        infrastructure = _probe_infrastructure()
        orphans = _inventory_orphans()

        # Build summary
        apis_configured = sum(1 for a in apis if a["configured"])
        apis_missing = sum(1 for a in apis if not a["configured"] and a["env_var"])

        def _count_health(items: List[ComponentStatus], ok: tuple = (Health.GREEN,)):
            return sum(1 for c in items if c.health in ok)

        summary = {
            "total_apis": len(apis),
            "apis_configured": apis_configured,
            "apis_missing": apis_missing,
            "apis_free": sum(1 for a in apis if a["env_var"] is None),
            "exchanges_online": _count_health(exchanges),
            "exchanges_total": len(exchanges),
            "strategies_ok": _count_health(strategies, (Health.GREEN, Health.YELLOW)),
            "strategies_total": len(strategies),
            "departments_ok": _count_health(departments, (Health.GREEN, Health.YELLOW)),
            "departments_total": len(departments),
            "infra_ok": _count_health(infrastructure, (Health.GREEN,)),
            "infra_total": len(infrastructure),
            "orphan_scripts": len(orphans),
        }

        return {
            "timestamp": now,
            "apis": apis,
            "exchanges": [e.to_dict() for e in exchanges],
            "strategies": [s.to_dict() for s in strategies],
            "departments": [d.to_dict() for d in departments],
            "infrastructure": [i.to_dict() for i in infrastructure],
            "orphans": orphans,
            "summary": summary,
        }
