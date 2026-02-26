# NCC Dashboard Display

A VS Code extension that displays real-time NCC (NCC Operations) dashboard metrics directly in the editor area above the terminal.

## Features

### 📊 Real-Time Metrics Display
- **Financial Metrics**: FFC Balance ($25M+), AAC Balance ($15M+), Bank Balance, Budget Tracking
- **Operations Metrics**: Active Projects (2), Ops Packets, System Alerts, Active Agents (1920)
- **System Health**: Operational Efficiency (95%+), Security Score (99), AI Performance (97%)
- **LFG Protocol**: 2.0x Performance Multiplier, MAXIMUM Domination Level, 15 Agent Divisions

### 🔄 Status Bar Integration
- FFC Balance: `$(bank) FFC: $25.0M`
- AAC Balance: `$(graph) AAC: $15.0M`
- LFG Status: `$(zap) LFG! ENGAGED`
- Alerts: `$(warning) Alerts: 2`
- Projects: `$(project) Projects: 2/5`

### 🎯 Intelligence & Analytics
- BBIC YouTube Analytics Status
- BBIC Reddit Intelligence (40 subreddits)
- AAC Investment Simulation Results
- Social Media Monitoring Status

### ⚡ LFG! Protocol Integration
- Real-time performance multiplier display
- Domination level monitoring
- Agent team coordination status
- Protocol activation timestamps

## Installation

1. Clone or download this extension
2. Run `npm install` to install dependencies
3. Run `npm run compile` to build the extension
4. Press `F5` to launch extension development host
5. In the new window, open the NCC Doctrine workspace
6. The extension will automatically activate and show status bar items

## Configuration

The extension can be configured through VS Code settings:

```json
{
  "ncc-dashboard.autoRefresh": true,
  "ncc-dashboard.refreshInterval": 15,
  "ncc-dashboard.dataPath": "./data"
}
```

## Commands

- `NCC Dashboard: Show NCC Dashboard` - Display the full dashboard in editor area
- `NCC Dashboard: Refresh NCC Dashboard` - Manually refresh all metrics
- `NCC Dashboard: Hide NCC Dashboard` - Close the dashboard panel

## Data Sources

The extension reads from JSON files in the workspace `data/` directory:
- `settings.json` - System settings and operational metrics
- `projects.json` - Project status and progress
- `budgets.json` - Budget allocations and caps
- `dashboard_status.json` - Dashboard operational status
- `aac_simulation_results.json` - AAC investment simulation data

## Architecture

### Status Bar Items
Compact display of key metrics in VS Code's status bar for quick reference.

### Webview Panel
Full dashboard display in VS Code's editor area with:
- Dark professional theme matching NCC branding
- Real-time data updates
- Responsive grid layout
- LFG! protocol status banner

### Auto-Refresh System
Configurable automatic refresh intervals with manual override capability.

## LFG! Protocol

This extension fully integrates with the NCC LFG! (Let's F***ing Go!) Protocol:
- Performance multiplier tracking
- Real-time domination level monitoring
- Agent coordination status
- Protocol activation confirmations

## Development

### Building
```bash
npm run compile
```

### Testing
```bash
npm run test
```

### Debugging
Press `F5` to launch extension development host with the NCC Doctrine workspace.

## Contributing

This extension is part of the NCC Operations ecosystem. Contributions should align with:
- LFG! Protocol standards
- NCC operational excellence
- Real-time performance requirements
- Security and compliance standards

## License

NCC Internal Use Only - All Rights Reserved