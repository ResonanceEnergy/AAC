# AAC Automation System

## 🚀 Overview

The AAC Automation System provides comprehensive automation for the Accelerated Arbitrage Corp Matrix Monitor platform. It handles everything from system health checks to automated git operations and dashboard management.

## 📁 Files Created

### `AAC_AUTO_LAUNCH.bat`
**One-click automation script**
- Automatically syncs changes to git
- Launches dashboard with browser auto-open
- Handles commit and push operations
- Perfect for daily development workflow

### `aac_automation.py`
**Advanced Python automation script**
- Full system health monitoring
- Automated testing and code quality checks
- Git operations with intelligent commit messages
- Dashboard management and browser automation

### `setup_automation.bat`
**Windows scheduler setup**
- Creates automated scheduled tasks
- Daily backups and health checks
- Login-based dashboard startup

## 🎯 Quick Start

### Option 1: One-Click Launch
```bash
# Double-click this file for instant launch + git sync
AAC_AUTO_LAUNCH.bat
```

### Option 2: Python Automation
```bash
# Full automation cycle
python aac_automation.py

# Dashboard only
python aac_automation.py --dashboard-only

# Skip git operations
python aac_automation.py --skip-git

# Skip tests
python aac_automation.py --skip-tests
```

### Option 3: Windows Scheduler
```bash
# Setup automated tasks (run as administrator)
setup_automation.bat
```

## 🔧 What Gets Automated

### ✅ System Health Checks
- Python environment validation
- Required package verification
- AAC component import testing
- Configuration file integrity

### ✅ Git Operations
- Automatic `git add .`
- Intelligent commit messages with timestamps
- Push to configured remotes
- Error handling for missing remotes

### ✅ Code Quality
- Import validation across all modules
- Component dependency checks
- Basic syntax and structure validation

### ✅ Testing
- Automated test suite execution
- Core functionality validation
- Integration testing
- Error reporting and logging

### ✅ Dashboard Management
- Automatic Streamlit server launch
- Browser auto-open functionality
- Port availability checking
- Process management and cleanup

## 📊 Scheduled Tasks (Windows)

The automation system can create the following scheduled tasks:

1. **AAC Daily Backup** (2:00 AM daily)
   - Runs full automation cycle
   - Commits and pushes changes
   - Performs system health checks

2. **AAC Health Check** (Hourly)
   - Validates system components
   - Checks for configuration issues
   - Monitors AAC service health

3. **AAC Dashboard Startup** (At user login)
   - Automatically launches dashboard
   - Opens browser to interface
   - Ensures AAC is always ready

## 🎨 Features

### Intelligent Git Handling
- Only commits when there are actual changes
- Generates descriptive commit messages
- Handles remote push failures gracefully
- Works with or without configured remotes

### Browser Automation
- Automatically opens default browser
- Waits for server startup (3-second delay)
- Handles browser launch failures
- Cross-platform compatibility

### Comprehensive Logging
- Detailed operation logs
- Success/failure tracking
- Error reporting with context
- Timestamped operations

### Error Recovery
- Graceful handling of missing components
- Fallback operations for failed tasks
- Clear error messages and suggestions
- Non-blocking error conditions

## 🚨 Requirements

- Python 3.14+
- Git (for version control features)
- Windows (for .bat files and scheduler)
- Internet connection (for GitHub operations)

## 🔍 Troubleshooting

### Git Operations Fail
```bash
# Check git status
git status

# Configure remote if needed
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Dashboard Won't Start
```bash
# Check port availability
netstat -ano | findstr :8080

# Kill existing processes
taskkill /PID <PID> /F

# Test manual launch
python core/aac_master_launcher.py --dashboard-only --display-mode web
```

### Import Errors
```bash
# Run health check
python -c "from aac_automation import AACAutomation; import asyncio; automation = AACAutomation(); asyncio.run(automation._check_system_health())"
```

## 📈 Advanced Usage

### Custom Automation Scripts
```python
from aac_automation import AACAutomation
import asyncio

async def custom_automation():
    automation = AACAutomation()

    # Run only specific checks
    await automation._check_system_health()
    await automation._handle_git_operations()

    # Custom dashboard launch
    await automation._launch_dashboard_automated()

asyncio.run(custom_automation())
```

### Integration with CI/CD
The automation system can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run AAC Automation
  run: |
    cd repository
    python aac_automation.py --skip-git  # Skip git in CI environment
```

## 🎯 Best Practices

1. **Run Daily**: Use `AAC_AUTO_LAUNCH.bat` for daily development
2. **Monitor Health**: Check logs regularly for system issues
3. **Backup Regularly**: The daily backup task ensures code safety
4. **Test Before Commit**: Run tests before major changes
5. **Keep Updated**: Regularly update automation scripts

## 📞 Support

For issues with the automation system:
1. Check the logs in the terminal output
2. Run individual components manually for debugging
3. Ensure all AAC components are properly installed
4. Verify Python environment and dependencies

---

**AAC Matrix Monitor - Now Fully Automated! 🚀**

The automation system ensures your AAC development environment is always up-to-date, tested, and ready for executive use.