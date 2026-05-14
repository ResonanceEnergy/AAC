# PlanktonXD Browser Bot Web Dashboard Guide

## Overview

The PlanktonXD Browser Bot Web Dashboard provides a modern, browser-based control panel for managing and monitoring the legendary PlanktonXD prediction market strategy. Instead of remembering complex command-line arguments, you can simply click buttons to execute all bot commands.

## Launch Methods

### Method 1: AAC Launcher (Recommended)
```bash
python launch.py planktonxd-web-dashboard
```

### Method 2: Direct Script
```bash
python scripts/launch_planktonxd_web_dashboard.py
```

### Method 3: Module Execution
```bash
python monitoring/planktonxd_browser_dashboard.py
```

## Web Interface Features

### 🚀 Basic Commands Section
- **Single Trading Cycle** — Execute one complete market scanning and trading cycle
- **Test Browser Setup** — Verify Selenium WebDriver and Chrome installation
- **Continuous Trading (Visible)** — Run continuous trading with visible browser for debugging
- **Check Status** — Get current bot status and recent activity

### ⚡ Advanced Usage Section  
- **Custom Bankroll & Cycles** — Use custom parameters from the input fields
- **Simulation Mode** — Run bot in simulation mode (no real trades)
- **Run Test Suite** — Execute comprehensive test suite for validation

### 🎛️ Custom Parameters Panel
- **Bankroll ($)** — Set custom bankroll amount (default: $2000)
- **Cycles** — Number of trading cycles to run (default: 5)  
- **Max Daily Bets** — Maximum bets per day limit (default: 200)

### 📊 Real-time Status Bar
- **Bot Status** — Green dot when active, red when offline
- **Last Activity** — Timestamp of most recent bot activity
- **Active Commands** — Number of currently running commands
- **Refresh Button** — Manual status update

### 📝 Live Logs Panel
- **Real-time output** from all executed commands
- **Color-coded entries** (success in green, errors in red)
- **Auto-scrolling** to show latest entries
- **Manual refresh** button for immediate updates

## Dashboard URL

Once launched, the dashboard is available at:
**http://localhost:8088**

The dashboard automatically opens in your default browser, or you can navigate to the URL manually.

## Command Mappings

The web buttons execute these underlying Python commands:

| Button | Command |
|--------|---------|
| Single Trading Cycle | `python launch.py planktonxd-browser` |
| Test Browser Setup | `python scripts/activate_planktonxd_browser_bot.py --test-browser` |
| Continuous Trading (Visible) | `python scripts/activate_planktonxd_browser_bot.py --continuous --visible` |
| Check Status | `python scripts/activate_planktonxd_browser_bot.py --status` |
| Custom Bankroll & Cycles | `python scripts/activate_planktonxd_browser_bot.py --bankroll X --cycles Y` |
| Simulation Mode | `python scripts/activate_planktonxd_browser_bot.py --simulate` |
| Run Test Suite | `python _scratch/test_planktonxd_browser_bot.py` |

## API Endpoints

The dashboard also exposes REST API endpoints for programmatic access:

- `GET /` — Main dashboard HTML interface
- `POST /api/execute` — Execute bot commands
- `GET /api/status` — Get current bot status
- `GET /api/logs` — Retrieve recent logs
- `POST /api/stop` — Stop all running commands
- `GET /api/health` — Health check endpoint

## Safety Features

### Process Management
- **Process tracking** — All commands are tracked with unique IDs
- **Automatic cleanup** — Finished processes are automatically removed
- **Force termination** — Ability to stop runaway commands

### Error Handling
- **Command validation** — Invalid commands are rejected
- **Timeout protection** — Long-running commands have timeout limits
- **Exception capture** — All errors are logged and displayed

### Visual Feedback
- **Button states** — Running commands show pulsing animation
- **Status indicators** — Color-coded status for quick assessment
- **Progress tracking** — Real-time updates on command execution

## Configuration

### Environment Variables
Set these in your environment to customize dashboard behavior:

```bash
# Dashboard configuration
export PLANKTONXD_DASHBOARD_PORT=8088
export PLANKTONXD_DASHBOARD_HOST=127.0.0.1

# Bot configuration (passed through to commands)
export PLANKTONXD_HEADLESS=false        # Visible browser
export PLANKTONXD_DEBUG=true           # Debug logging
export PLANKTONXD_SIMULATE=true        # Simulate trades
export PLANKTONXD_MAX_DAILY_BETS=50    # Lower bet limit
```

### Custom Port
To run on a different port:

```bash
python monitoring/planktonxd_browser_dashboard.py --port 9000
```

## Integration with AAC

The web dashboard is fully integrated into AAC's architecture:

- **Unified launcher** — Available via `python launch.py planktonxd-web-dashboard`
- **Component registry** — Registered in unified component integrator
- **Logging framework** — Uses AAC's structlog system
- **Project conventions** — Follows AAC file and code patterns

## Troubleshooting

### Dashboard Won't Start
1. Check if port 8088 is already in use
2. Verify all dependencies are installed: `pip install fastapi uvicorn`
3. Check Python version (requires 3.9+)

### Commands Fail to Execute
1. Verify you're in the AAC project root directory
2. Check that the PlanktonXD browser bot files exist
3. Ensure Selenium and Chrome are properly installed

### Browser Won't Open Automatically
1. The dashboard will still be available at http://localhost:8088
2. Check firewall settings that might block local connections
3. Try manually navigating to the URL

### Performance Issues
1. Close unused browser tabs to free memory
2. Stop unnecessary background processes
3. Check system resources (RAM, CPU usage)

## Advanced Usage

### Multiple Instances
Run multiple dashboard instances on different ports:

```bash
python monitoring/planktonxd_browser_dashboard.py --port 8089
python monitoring/planktonxd_browser_dashboard.py --port 8090
```

### Headless Mode
For server deployments without a desktop environment:

```bash
python monitoring/planktonxd_browser_dashboard.py --no-browser
```

### API Integration
Use the REST API for custom integrations:

```bash
# Check status
curl http://localhost:8088/api/status

# Execute command
curl -X POST http://localhost:8088/api/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "scripts/activate_planktonxd_browser_bot.py --status"}'

# Get logs
curl http://localhost:8088/api/logs
```

## Security Considerations

- **Local access only** — Dashboard binds to 127.0.0.1 by default
- **No authentication** — Intended for local development use
- **Command validation** — Only allowed PlanktonXD commands can be executed
- **Process isolation** — Commands run in separate processes

For production deployments, consider:
- Adding authentication middleware
- Using HTTPS with proper certificates
- Implementing role-based access control
- Adding audit logging for all commands

## Support

For issues with the web dashboard:
1. Check the dashboard logs in the web interface
2. Review the console output from the launch command
3. Verify the underlying PlanktonXD browser bot is working via CLI
4. Check AAC's main documentation and troubleshooting guides