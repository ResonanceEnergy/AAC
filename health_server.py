#!/usr/bin/env python3
"""
AAC Health Endpoint
===================
Lightweight HTTP health endpoint for external monitors.
Runs on port 8080 by default. No external dependencies (stdlib only).
"""

import json
import os
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_start_time = time.monotonic()


class HealthHandler(BaseHTTPRequestHandler):
    """Handle /health and /ready endpoints."""

    def do_GET(self):
        if self.path == '/health':
            self._respond_health()
        elif self.path == '/ready':
            self._respond_ready()
        else:
            self.send_error(404)

    def _respond_health(self):
        """Liveness probe — is the process alive?"""
        uptime = time.monotonic() - _start_time
        body = {
            'status': 'ok',
            'uptime_seconds': round(uptime, 1),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
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

    def _json_response(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """Suppress default access logs."""
        pass


def start_health_server(port: int = 8080, background: bool = True):
    """Start the health endpoint server."""
    port = int(os.environ.get('HEALTH_PORT', port))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)

    if background:
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server
    else:
        print(f"Health endpoint listening on http://0.0.0.0:{port}/health")
        server.serve_forever()


if __name__ == '__main__':
    start_health_server(background=False)
