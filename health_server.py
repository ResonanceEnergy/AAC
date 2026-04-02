#!/usr/bin/env python3
"""
AAC Health Endpoint
===================
Lightweight HTTP health endpoint for external monitors.
Runs on port 8080 by default. No external dependencies (stdlib only).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_start_time = time.monotonic()


class HealthHandler(BaseHTTPRequestHandler):
    """Handle /health and /ready endpoints."""

    def do_GET(self):
        """Do get."""
        if self.path == '/health':
            self._respond_health()
        elif self.path == '/ready':
            self._respond_ready()
        elif self.path == '/platform_status':
            self._respond_platform_status()
        elif self.path == '/enterprise/health':
            self._respond_enterprise_health()
        else:
            self.send_error(404)

    def _respond_health(self):
        """Liveness probe — is the process alive?"""
        uptime = time.monotonic() - _start_time
        body = {
            'status': 'ok',
            'uptime_seconds': round(uptime, 1),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'python': sys.version.split()[0],
        }

        # Check config validation
        try:
            from shared.config_loader import get_config
            config = get_config()
            validation = config.validate()
            body['config_valid'] = validation['valid']
            body['exchanges'] = validation['exchanges_configured']
            body['dry_run'] = validation['dry_run']
            if validation['issues']:
                body['issues'] = validation['issues']
        except Exception as e:
            body['config_error'] = str(e)

        # Options Intelligence pre-market scanner health
        try:
            from startup.phases import get_premarket_scanner_health
            body['options_intelligence'] = get_premarket_scanner_health()
        except Exception:
            body['options_intelligence'] = {'status': 'unavailable'}

        # Watchdog status (if running)
        try:
            from startup.phases import get_watchdog
            dog = get_watchdog()
            if dog is not None:
                body['watchdog'] = dog.status()
        except Exception:
            pass

        # Gateway PIDs
        try:
            from startup.gateways import get_gateway_pids
            pids = get_gateway_pids()
            if pids:
                body['gateway_pids'] = pids
        except Exception:
            pass

        self._json_response(200, body)

    def _respond_ready(self):
        """Readiness probe — can the system accept work?"""
        try:
            from shared.config_loader import get_config
            config = get_config()
            validation = config.validate()
            if validation['valid']:
                self._json_response(200, {'ready': True})
            else:
                self._json_response(503, {'ready': False, 'issues': validation['issues']})
        except Exception as e:
            self._json_response(503, {'ready': False, 'error': str(e)})

    def _respond_platform_status(self):
        """NCC platform status — comprehensive AAC state for NCC Supreme Monitor."""
        try:
            from shared.ncc_integration import get_ncc_bridge
            bridge = get_ncc_bridge()
            body = bridge.platform_status
            self._json_response(200, body)
        except Exception as e:
            # Fallback: basic status even if bridge not initialized
            uptime = time.monotonic() - _start_time
            self._json_response(200, {
                'pillar_id': 'aac',
                'status': 'online',
                'uptime_seconds': round(uptime, 1),
                'ncc_bridge': 'not_initialized',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })

    def _respond_enterprise_health(self):
        """Aggregate health from all four pillars for NCC Supreme Monitor."""
        import urllib.error
        import urllib.request

        pillars = {
            "NCC":  {"url": "http://127.0.0.1:8765/health",  "status": "UNKNOWN"},
            "AAC":  {"url": "http://127.0.0.1:8080/health",  "status": "UNKNOWN"},
            "NCL":  {"url": "http://127.0.0.1:8000/health",  "status": "UNKNOWN"},
            "BRS":  {"url": "http://127.0.0.1:8000/health",  "status": "UNKNOWN"},
            "RELAY": {"url": "http://127.0.0.1:8787/health", "status": "UNKNOWN"},
        }
        online = 0
        for pid, info in pillars.items():
            try:
                req = urllib.request.Request(info["url"], method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                    if resp.status == 200:
                        info["status"] = "GREEN"
                        online += 1
                    else:
                        info["status"] = "RED"
            except (urllib.error.URLError, OSError, ValueError):
                info["status"] = "RED"

        total = len(pillars)
        overall = "GREEN" if online == total else ("YELLOW" if online > 0 else "RED")
        body = {
            "enterprise_status": overall,
            "pillars_online": online,
            "pillars_total": total,
            "pillars": {pid: info["status"] for pid, info in pillars.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._json_response(200, body)

    def _json_response(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """Suppress default access logs."""
        logger.debug("log_message called")


def start_health_server(port: int = 8080, background: bool = True):
    """Start the health endpoint server."""
    port = int(os.environ.get('HEALTH_PORT', port))
    host = os.environ.get('HEALTH_HOST', '127.0.0.1')
    server = HTTPServer((host, port), HealthHandler)

    if background:
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server
    else:
        logger.info(f"Health endpoint listening on http://{host}:{port}/health")
        server.serve_forever()


if __name__ == '__main__':
    start_health_server(background=False)
