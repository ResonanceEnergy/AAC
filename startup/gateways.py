"""
startup.gateways — Trading gateway management (IBKR TWS, Moomoo OpenD).

Extracted from launch.py so other modules can reuse gateway startup logic.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Gateway Configurations ──────────────────────────────────────────────────

GATEWAY_CONFIGS: dict[str, dict] = {
    "ibkr": {
        "name": "IBKR TWS",
        "exe": Path(r"C:\Jts\tws.exe"),
        "process_name": "tws",
        "host": "127.0.0.1",
        "port": 7497,
        "wait_secs": 30,
    },
    "moomoo": {
        "name": "Moomoo OpenD",
        "exe": Path(r"C:\FutuOpenD\moomoo_OpenD_10.0.6018_Windows\moomoo_OpenD-GUI_10.0.6018_Windows\moomoo_OpenD-GUI_10.0.6018_Windows.exe"),
        "process_name": "moomoo_OpenD-GUI",
        "host": "127.0.0.1",
        "port": 11111,
        "wait_secs": 15,
    },
}


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if *host*:*port* accepts a TCP connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def is_process_running(name: str) -> bool:
    """Check if a Windows process with *name* is running (tasklist)."""
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}*", "/NH"],
        capture_output=True, text=True,
    )
    return name.lower() in result.stdout.lower()


def start_gateway(key: str) -> bool:
    """Start a single gateway if not already running. Returns True if port is live."""
    cfg = GATEWAY_CONFIGS[key]
    name = cfg["name"]
    host, port = cfg["host"], cfg["port"]

    # Already listening?
    if port_open(host, port):
        logger.info(f"  [+] {name}: already running on {host}:{port}")
        return True

    # Exe exists?
    if not cfg["exe"].exists():
        logger.warning(f"  [!] {name}: not installed ({cfg['exe']})")
        return False

    # Launch
    logger.info(f"  [>] Starting {name} ...")
    try:
        subprocess.Popen(
            [str(cfg["exe"])],
            cwd=str(cfg["exe"].parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        # WinError 740: needs elevation — use shell start
        os.startfile(str(cfg["exe"]))

    # Wait for port
    deadline = time.time() + cfg["wait_secs"]
    while time.time() < deadline:
        if port_open(host, port, timeout=1.0):
            logger.info(f"  [+] {name}: LIVE on {host}:{port}")
            return True
        time.sleep(2)

    # Port not ready — process may need manual login
    if is_process_running(cfg["process_name"]):
        logger.warning(f"  [!] {name}: running but port {port} not ready — log in to the GUI")
    else:
        logger.error(f"  [X] {name}: failed to start")
    return False


def start_all_gateways() -> dict[str, bool]:
    """Start all configured gateways. Returns ``{key: is_live}``."""
    results = {}
    for key in GATEWAY_CONFIGS:
        results[key] = start_gateway(key)
    return results


def gateway_summary(results: dict[str, bool]) -> str:
    """Return a human-readable summary string."""
    live = sum(1 for v in results.values() if v)
    total = len(results)
    lines = [f"  Gateways: {live}/{total} live"]
    for key, ok in results.items():
        cfg = GATEWAY_CONFIGS[key]
        status = "LIVE" if ok else "WAITING"
        lines.append(f"    {cfg['name']:<20s} {cfg['host']}:{cfg['port']}  {status}")
    return "\n".join(lines)
