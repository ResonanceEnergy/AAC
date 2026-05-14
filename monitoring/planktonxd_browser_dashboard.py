#!/usr/bin/env python3
"""
PlanktonXD Browser Bot Web Dashboard
=====================================
FastAPI-based web interface for controlling and monitoring the PlanktonXD Browser Bot.
Provides clickable buttons for all bot commands with real-time status updates.

Launch: python monitoring/planktonxd_browser_dashboard.py
        python launch.py planktonxd-web-dashboard
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import structlog

_log = structlog.get_logger()

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlanktonXD Browser Bot Dashboard",
    description="Web interface for controlling and monitoring the PlanktonXD Browser Bot",
    version="1.0.0"
)

# ── Models ───────────────────────────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str
    args: Optional[Dict[str, Any]] = None

class CommandResponse(BaseModel):
    success: bool
    command_id: str
    message: str
    output: Optional[str] = None

class StatusResponse(BaseModel):
    bot_running: bool
    last_activity: Optional[str] = None
    active_commands: List[str] = []
    recent_logs: List[str] = []

# ── Global State ─────────────────────────────────────────────────────────────

class DashboardState:
    """Global state for the dashboard."""
    
    def __init__(self):
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.command_history: List[Dict[str, Any]] = []
        self.recent_logs: List[str] = []
        self.bot_status = {
            "running": False,
            "last_activity": None,
            "active_commands": [],
            "last_error": None
        }
    
    def add_log(self, message: str):
        """Add a log entry with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.recent_logs.append(log_entry)
        # Keep only last 50 entries
        if len(self.recent_logs) > 50:
            self.recent_logs = self.recent_logs[-50:]
        _log.info(message)
    
    def update_bot_status(self, **kwargs):
        """Update bot status information."""
        self.bot_status.update(kwargs)
        if "last_activity" not in kwargs:
            self.bot_status["last_activity"] = datetime.now().isoformat()

dashboard_state = DashboardState()

# ── Command Execution ───────────────────────────────────────────────────────

def execute_command_sync(command: str, args: List[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Execute a command synchronously and return results."""
    if args is None:
        args = []
    
    full_command = [sys.executable] + command.split() + args
    command_str = " ".join(full_command)
    
    dashboard_state.add_log(f"Executing: {command_str}")
    
    try:
        result = subprocess.run(
            full_command,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout + result.stderr if result.stderr else result.stdout
        
        if result.returncode == 0:
            dashboard_state.add_log(f"Command completed successfully: {command_str}")
            return {
                "success": True,
                "output": output,
                "return_code": result.returncode,
                "command": command_str
            }
        else:
            dashboard_state.add_log(f"Command failed with code {result.returncode}: {command_str}")
            return {
                "success": False,
                "output": output,
                "return_code": result.returncode,
                "command": command_str,
                "error": f"Process exited with code {result.returncode}"
            }
    
    except subprocess.TimeoutExpired:
        dashboard_state.add_log(f"Command timed out after {timeout}s: {command_str}")
        return {
            "success": False,
            "output": "",
            "return_code": -1,
            "command": command_str,
            "error": f"Command timed out after {timeout} seconds"
        }
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        dashboard_state.add_log(error_msg)
        return {
            "success": False,
            "output": "",
            "return_code": -1,
            "command": command_str,
            "error": error_msg
        }

def execute_command_async(command: str, args: List[str] = None) -> str:
    """Execute a command asynchronously and return process ID."""
    if args is None:
        args = []
    
    full_command = [sys.executable] + command.split() + args
    command_str = " ".join(full_command)
    
    try:
        process = subprocess.Popen(
            full_command,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        process_id = f"cmd_{int(time.time())}_{process.pid}"
        dashboard_state.active_processes[process_id] = process
        dashboard_state.add_log(f"Started async command: {command_str} (PID: {process.pid})")
        
        # Start thread to monitor process
        def monitor_process():
            try:
                output, _ = process.communicate()
                dashboard_state.add_log(f"Async command completed (PID: {process.pid})")
                if output:
                    for line in output.split('\n'):
                        if line.strip():
                            dashboard_state.add_log(f"OUTPUT: {line}")
            except Exception as e:
                dashboard_state.add_log(f"Error monitoring process {process.pid}: {str(e)}")
            finally:
                if process_id in dashboard_state.active_processes:
                    del dashboard_state.active_processes[process_id]
        
        threading.Thread(target=monitor_process, daemon=True).start()
        
        return process_id
    
    except Exception as e:
        error_msg = f"Error starting async command: {str(e)}"
        dashboard_state.add_log(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlanktonXD Browser Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid rgba(0, 255, 136, 0.2);
        }
        .header h1 {
            font-size: 3rem;
            color: #00ff88;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        .header .subtitle {
            font-size: 1.2rem;
            color: #b0b0b0;
            margin-bottom: 20px;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.3);
            padding: 15px 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid rgba(0, 255, 136, 0.1);
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
        }
        .status-indicator.active {
            background: #00ff88;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        .command-sections {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .command-section {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(0, 255, 136, 0.1);
        }
        .command-section h2 {
            color: #00ff88;
            margin-bottom: 20px;
            font-size: 1.5rem;
            text-align: center;
        }
        .command-grid {
            display: grid;
            gap: 15px;
        }
        .command-btn {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00ff88;
            color: #e0e0e0;
            padding: 15px 20px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1rem;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .command-btn:hover {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6f 100%);
            color: #000;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3);
        }
        .command-btn:active {
            transform: translateY(0);
        }
        .command-btn.running {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
            animation: pulse 2s infinite;
        }
        .command-icon {
            font-size: 1.2rem;
        }
        .logs-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(0, 255, 136, 0.1);
        }
        .logs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .logs-container {
            background: #000;
            padding: 20px;
            border-radius: 10px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .log-entry {
            margin-bottom: 5px;
            color: #b0b0b0;
        }
        .log-entry.success {
            color: #00ff88;
        }
        .log-entry.error {
            color: #ff4444;
        }
        .custom-controls {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            border: 1px solid rgba(0, 255, 136, 0.1);
        }
        .input-group {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            align-items: center;
        }
        .input-group label {
            min-width: 120px;
            color: #00ff88;
        }
        .input-group input {
            flex: 1;
            padding: 10px;
            border: 1px solid #00ff88;
            border-radius: 5px;
            background: rgba(0, 0, 0, 0.3);
            color: #e0e0e0;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
        }
        .refresh-btn {
            background: #16213e;
            border: 1px solid #00ff88;
            color: #00ff88;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        .refresh-btn:hover {
            background: #00ff88;
            color: #000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦠 PlanktonXD Browser Bot</h1>
            <div class="subtitle">Web Dashboard & Control Center</div>
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-indicator" id="botStatus"></div>
                    <span id="botStatusText">Bot Offline</span>
                </div>
                <div class="status-item">
                    <span>Last Activity: <span id="lastActivity">Never</span></span>
                </div>
                <div class="status-item">
                    <span>Active Commands: <span id="activeCommands">0</span></span>
                </div>
                <button class="refresh-btn" onclick="refreshStatus()">🔄 Refresh</button>
            </div>
        </div>

        <div class="command-sections">
            <div class="command-section">
                <h2>🚀 Basic Commands</h2>
                <div class="command-grid">
                    <button class="command-btn" onclick="executeCommand('launch.py planktonxd-browser')">
                        <span>Single Trading Cycle</span>
                        <span class="command-icon">▶️</span>
                    </button>
                    <button class="command-btn" onclick="executeCommand('scripts/activate_planktonxd_browser_bot.py --test-browser')">
                        <span>Test Browser Setup</span>
                        <span class="command-icon">🧪</span>
                    </button>
                    <button class="command-btn" onclick="executeCommand('scripts/activate_planktonxd_browser_bot.py --continuous --visible', true)">
                        <span>Continuous Trading (Visible)</span>
                        <span class="command-icon">🔄</span>
                    </button>
                    <button class="command-btn" onclick="executeCommand('scripts/activate_planktonxd_browser_bot.py --status')">
                        <span>Check Status</span>
                        <span class="command-icon">📊</span>
                    </button>
                </div>
            </div>

            <div class="command-section">
                <h2>⚡ Advanced Usage</h2>
                <div class="command-grid">
                    <button class="command-btn" onclick="executeCommandWithParams('bankroll_cycles')">
                        <span>Custom Bankroll & Cycles</span>
                        <span class="command-icon">💰</span>
                    </button>
                    <button class="command-btn" onclick="executeCommand('scripts/activate_planktonxd_browser_bot.py --simulate')">
                        <span>Simulation Mode</span>
                        <span class="command-icon">🎮</span>
                    </button>
                    <button class="command-btn" onclick="executeCommand('_scratch/test_planktonxd_browser_bot.py')">
                        <span>Run Test Suite</span>
                        <span class="command-icon">🔬</span>
                    </button>
                </div>
            </div>
        </div>

        <div class="custom-controls">
            <h2 style="color: #00ff88; margin-bottom: 20px;">🎛️ Custom Parameters</h2>
            <div class="input-group">
                <label>Bankroll ($):</label>
                <input type="number" id="bankrollInput" value="2000" min="100" max="100000">
            </div>
            <div class="input-group">
                <label>Cycles:</label>
                <input type="number" id="cyclesInput" value="5" min="1" max="50">
            </div>
            <div class="input-group">
                <label>Max Daily Bets:</label>
                <input type="number" id="maxBetsInput" value="200" min="10" max="1000">
            </div>
        </div>

        <div class="logs-section">
            <div class="logs-header">
                <h2 style="color: #00ff88;">📝 Real-time Logs</h2>
                <button class="refresh-btn" onclick="refreshLogs()">🔄 Refresh Logs</button>
            </div>
            <div class="logs-container" id="logsContainer">
                <div class="log-entry">Waiting for commands...</div>
            </div>
        </div>
    </div>

    <script>
        let runningCommands = new Set();

        async function executeCommand(command, isAsync = false) {
            const btn = event.target.closest('.command-btn');
            btn.classList.add('running');
            
            try {
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        command: command,
                        async: isAsync
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    addLog(`✅ Command completed: ${command}`, 'success');
                    if (result.output) {
                        result.output.split('\\n').forEach(line => {
                            if (line.trim()) addLog(`OUTPUT: ${line}`);
                        });
                    }
                } else {
                    addLog(`❌ Command failed: ${result.error || 'Unknown error'}`, 'error');
                }
            } catch (error) {
                addLog(`❌ Error executing command: ${error.message}`, 'error');
            } finally {
                btn.classList.remove('running');
                refreshStatus();
            }
        }

        async function executeCommandWithParams(type) {
            let command = '';
            
            if (type === 'bankroll_cycles') {
                const bankroll = document.getElementById('bankrollInput').value;
                const cycles = document.getElementById('cyclesInput').value;
                command = `scripts/activate_planktonxd_browser_bot.py --bankroll ${bankroll} --cycles ${cycles}`;
            }
            
            if (command) {
                await executeCommand(command);
            }
        }

        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                const indicator = document.getElementById('botStatus');
                const statusText = document.getElementById('botStatusText');
                const lastActivity = document.getElementById('lastActivity');
                const activeCommands = document.getElementById('activeCommands');
                
                if (status.bot_running) {
                    indicator.classList.add('active');
                    statusText.textContent = 'Bot Active';
                } else {
                    indicator.classList.remove('active');
                    statusText.textContent = 'Bot Offline';
                }
                
                lastActivity.textContent = status.last_activity || 'Never';
                activeCommands.textContent = status.active_commands.length;
                
            } catch (error) {
                console.error('Error refreshing status:', error);
            }
        }

        async function refreshLogs() {
            try {
                const response = await fetch('/api/logs');
                const logs = await response.json();
                
                const container = document.getElementById('logsContainer');
                container.innerHTML = '';
                
                logs.recent_logs.forEach(log => {
                    addLogEntry(log);
                });
                
                container.scrollTop = container.scrollHeight;
            } catch (error) {
                console.error('Error refreshing logs:', error);
            }
        }

        function addLog(message, type = '') {
            const container = document.getElementById('logsContainer');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        function addLogEntry(logText) {
            const container = document.getElementById('logsContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = logText;
            container.appendChild(entry);
        }

        // Auto-refresh every 30 seconds
        setInterval(refreshStatus, 30000);
        setInterval(refreshLogs, 10000);

        // Initial load
        refreshStatus();
        refreshLogs();
    </script>
</body>
</html>"""
    
    return HTMLResponse(content=html_content)

@app.post("/api/execute")
async def execute_command_endpoint(request: CommandRequest, background_tasks: BackgroundTasks):
    """Execute a PlanktonXD bot command."""
    command = request.command
    is_async = request.args.get("async", False) if request.args else False
    
    dashboard_state.add_log(f"Received command request: {command}")
    dashboard_state.update_bot_status(running=True)
    
    try:
        if is_async or "continuous" in command:
            # Execute asynchronously for long-running commands
            process_id = execute_command_async(command)
            return CommandResponse(
                success=True,
                command_id=process_id,
                message=f"Command started asynchronously: {command}",
                output=None
            )
        else:
            # Execute synchronously for quick commands
            result = execute_command_sync(command)
            return CommandResponse(
                success=result["success"],
                command_id=f"sync_{int(time.time())}",
                message=f"Command {'completed' if result['success'] else 'failed'}: {command}",
                output=result["output"]
            )
    
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        dashboard_state.add_log(error_msg)
        return CommandResponse(
            success=False,
            command_id="",
            message=error_msg,
            output=None
        )

@app.get("/api/status")
async def get_status():
    """Get current bot status."""
    active_commands = list(dashboard_state.active_processes.keys())
    
    # Check if any processes are still running
    running_processes = []
    for proc_id, process in list(dashboard_state.active_processes.items()):
        if process.poll() is None:  # Process still running
            running_processes.append(proc_id)
        else:
            # Process finished, remove from active list
            del dashboard_state.active_processes[proc_id]
    
    bot_running = len(running_processes) > 0
    dashboard_state.update_bot_status(
        running=bot_running,
        active_commands=running_processes
    )
    
    return StatusResponse(
        bot_running=bot_running,
        last_activity=dashboard_state.bot_status.get("last_activity"),
        active_commands=running_processes,
        recent_logs=dashboard_state.recent_logs[-10:]  # Last 10 logs
    )

@app.get("/api/logs")
async def get_logs():
    """Get recent logs."""
    return {
        "recent_logs": dashboard_state.recent_logs,
        "count": len(dashboard_state.recent_logs)
    }

@app.post("/api/stop")
async def stop_all_commands():
    """Stop all running commands."""
    stopped_count = 0
    
    for proc_id, process in list(dashboard_state.active_processes.items()):
        try:
            process.terminate()
            process.wait(timeout=5)
            stopped_count += 1
            dashboard_state.add_log(f"Stopped command: {proc_id}")
        except subprocess.TimeoutExpired:
            process.kill()
            stopped_count += 1
            dashboard_state.add_log(f"Force-killed command: {proc_id}")
        except Exception as e:
            dashboard_state.add_log(f"Error stopping command {proc_id}: {str(e)}")
    
    dashboard_state.active_processes.clear()
    dashboard_state.update_bot_status(running=False, active_commands=[])
    
    return {"message": f"Stopped {stopped_count} commands"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dashboard_version": "1.0.0",
        "active_processes": len(dashboard_state.active_processes)
    }

# ── Startup/Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize dashboard on startup."""
    dashboard_state.add_log("PlanktonXD Browser Bot Dashboard started")
    dashboard_state.add_log("Ready to execute commands")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    dashboard_state.add_log("Dashboard shutting down...")
    
    # Stop all running processes
    for proc_id, process in dashboard_state.active_processes.items():
        try:
            process.terminate()
        except Exception as e:
            _log.warning(f"Error terminating process {proc_id}: {e}")

# ── CLI Runner ──────────────────────────────────────────────────────────────

def run_dashboard(host: str = "127.0.0.1", port: int = 8088, open_browser: bool = True):
    """Run the PlanktonXD Browser Bot dashboard."""
    import webbrowser
    import uvicorn
    
    dashboard_url = f"http://{host}:{port}"
    
    def open_browser_delayed():
        """Open browser after a delay."""
        time.sleep(2)
        try:
            webbrowser.open(dashboard_url)
            _log.info(f"✅ Browser opened: {dashboard_url}")
        except Exception as e:
            _log.warning(f"Could not open browser: {e}")
            _log.info(f"Please manually open: {dashboard_url}")
    
    if open_browser:
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    _log.info(f"🚀 PlanktonXD Browser Bot Dashboard starting on {dashboard_url}")
    _log.info("🎯 Web interface with command buttons ready")
    
    try:
        uvicorn.run(
            "monitoring.planktonxd_browser_dashboard:app",
            host=host,
            port=port,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        _log.info("Dashboard stopped by user")
    except Exception as e:
        _log.error(f"Error running dashboard: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PlanktonXD Browser Bot Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    run_dashboard(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )