# NCC Master Startup System v2.0.0

## Overview

The NCC Master Startup System is the unified command center for all Natrix Command Corp operations. This system provides a single entry point for launching, monitoring, and managing all NCC systems including ASG (Activate Systems Go), LFG (Let's F***ing Go), daily maintenance, scheduled tasks, and emergency controls.

## Quick Start

### Simple Commands
```batch
# Launch complete system
NCC STARTALL

# Quick system activation
NCC ASG

# Launch cycle operations
NCC LFG

# Check system status
NCC STATUS

# Emergency stop
NCC STOP
```

### PowerShell Commands
```powershell
# Full system activation
.\NCC.MasterStartup.ps1 -StartAll

# Individual components
.\NCC.MasterStartup.ps1 -ASG
.\NCC.MasterStartup.ps1 -LFG
.\NCC.MasterStartup.ps1 -Maintenance
.\NCC.MasterStartup.ps1 -Schedule
.\NCC.MasterStartup.ps1 -Status
.\NCC.MasterStartup.ps1 -Stop
.\NCC.MasterStartup.ps1 -Test
.\NCC.MasterStartup.ps1 -Update
```

## System Components

### 1. NCC.SystemIntegration.ps1
- **Purpose**: Unified system launcher integrating ASG and LFG operations
- **Features**:
  - ASG activation (system initialization)
  - LFG cycle operations (continuous running)
  - Emergency stop mechanisms
  - System status monitoring
  - Cycle mode with configurable intervals

### 2. NCC.DailyMaintenance.ps1
- **Purpose**: Comprehensive daily maintenance and health monitoring
- **Features**:
  - System health checks (disk, memory, processes)
  - Data backup operations
  - Performance optimization (temp cleanup, log rotation)
  - Security scanning (file integrity, permissions)
  - Storage analysis and monitoring

### 3. NCC.ScheduledTasks.ps1
- **Purpose**: Windows Scheduled Tasks management for automation
- **Features**:
  - Automated daily maintenance scheduling
  - System monitoring tasks
  - Backup operation scheduling
  - Task registration and management
  - Status reporting

### 4. NCC.DesktopDisplay.ps1
- **Purpose**: Desktop display integration and system visualization
- **Features**:
  - System status displays
  - Real-time monitoring dashboards
  - Desktop integration
  - Visual system health indicators

### 5. NCC.MasterStartup.ps1
- **Purpose**: Master command center for all NCC operations
- **Features**:
  - Unified command interface
  - System testing and validation
  - Update management
  - Comprehensive logging
  - Emergency controls

## Command Reference

### Batch File Commands (NCC.bat)
| Command | Description |
|---------|-------------|
| `NCC ASG` | Activate Systems Go - Initialize all systems |
| `NCC LFG` | Let's F***ing Go - Launch cycle operations |
| `NCC STATUS` | Show comprehensive system status |
| `NCC STOP` | Emergency system shutdown |
| `NCC MAINTENANCE` | Run daily maintenance routines |
| `NCC SCHEDULE` | Set up automated scheduled tasks |
| `NCC TEST` | Test all system components |
| `NCC UPDATE` | Update all system components |
| `NCC STARTALL` | Launch complete NCC system |
| `NCC HELP` | Show detailed help information |

### PowerShell Parameters
| Parameter | Description |
|-----------|-------------|
| `-StartAll` | Launch complete system (ASG + LFG + Maintenance) |
| `-ASG` | Activate Systems Go |
| `-LFG` | Let's F***ing Go (Cycle operations) |
| `-Maintenance` | Run system maintenance |
| `-Schedule` | Set up scheduled tasks |
| `-Status` | Show system status |
| `-Stop` | Emergency stop |
| `-Test` | Test all systems |
| `-Update` | Update all systems |
| `-Help` | Show help |

## Automated Operations

### Daily Maintenance Schedule
- **Time**: 2:00 AM daily
- **Tasks**:
  - System health checks
  - Data backups
  - Performance optimization
  - Security scanning
  - Log rotation

### Continuous Operations
- **Cycle Mode**: Configurable intervals (default: 2 minutes)
- **Monitoring**: Real-time system health
- **Emergency Controls**: Immediate stop capabilities

### Scheduled Tasks
- `NCC_Daily_Maintenance`: Daily system maintenance
- `NCC_System_Integration_Monitor`: System status monitoring
- `NCC_Backup_Operations`: Automated backups
- `NCC_System_Monitoring`: Continuous health monitoring

## System Architecture

```
NCC Master Startup System
├── NCC.MasterStartup.ps1 (Main Entry Point)
├── NCC.bat (Quick Launcher)
├── System Components
│   ├── NCC.SystemIntegration.ps1 (ASG/LFG Integration)
│   ├── NCC.DailyMaintenance.ps1 (Maintenance System)
│   ├── NCC.ScheduledTasks.ps1 (Task Scheduler)
│   └── NCC.DesktopDisplay.ps1 (Display Integration)
├── Automated Operations
│   ├── Windows Scheduled Tasks
│   ├── Continuous Cycle Operations
│   └── Emergency Control System
└── Monitoring & Logging
    ├── Master Log (NCC_Master_Startup.log)
    ├── Component Logs
    └── System Health Reports
```

## Emergency Procedures

### Emergency Stop
```batch
NCC STOP
```
or
```powershell
.\NCC.MasterStartup.ps1 -Stop
```

### System Recovery
1. Run emergency stop
2. Check system status: `NCC STATUS`
3. Run system test: `NCC TEST`
4. Restart with: `NCC STARTALL`

## Monitoring & Logging

### Log Files
- **Master Log**: `logs\NCC_Master_Startup.log`
- **Component Logs**: Individual component log files
- **System Reports**: Health check and maintenance reports

### Real-time Monitoring
- System status via `NCC STATUS`
- Component health checks
- Scheduled task status
- Running process monitoring

## Configuration

### Environment Setup
- Windows 10/11 with PowerShell 5.1+
- Node.js (for dashboard operations)
- Administrative privileges for scheduled tasks

### Customization
- Edit script parameters for custom intervals
- Modify scheduled task times in NCC.ScheduledTasks.ps1
- Configure maintenance routines in NCC.DailyMaintenance.ps1

## Troubleshooting

### Common Issues
1. **Permission Errors**: Run as Administrator
2. **Missing Scripts**: Ensure all component scripts are present
3. **Scheduled Task Failures**: Check Windows Task Scheduler permissions
4. **Component Test Failures**: Run individual component tests

### Diagnostic Commands
```batch
# Full system test
NCC TEST

# Component status
NCC STATUS

# Update system
NCC UPDATE
```

## Development & Updates

### Version History
- **v2.0.0**: Complete system integration, automated scheduling, emergency controls
- **v1.5.0**: Added daily maintenance automation
- **v1.0.0**: Initial system integration

### Future Enhancements
- Web-based control interface
- Advanced monitoring dashboards
- Multi-system deployment support
- Cloud integration capabilities

## Support

For issues or questions:
1. Check system status: `NCC STATUS`
2. Review logs in `logs\` directory
3. Run system test: `NCC TEST`
4. Contact NCC Command Center

---

**Natrix Command Corp - Master Startup System v2.0.0**
*Built for continuous operation and automated excellence*