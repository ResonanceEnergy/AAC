#!/usr/bin/env python3
"""
AAC Automation System
=====================

Fully automated AAC Matrix Monitor management system.
Handles git operations, system health checks, and dashboard management.
"""

import asyncio
import subprocess
import sys
import os
import time
import webbrowser
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AAC.Automation")

class AACAutomation:
    """Comprehensive AAC system automation"""

    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.project_root = self.repo_path
        self.python_exe = sys.executable

    async def run_full_automation(self) -> bool:
        """Run complete automation cycle"""
        logger.info("🚀 Starting AAC Full Automation Cycle")

        try:
            # 1. System health check
            if not await self._check_system_health():
                logger.error("❌ System health check failed")
                return False

            # 2. Git operations
            await self._handle_git_operations()

            # 3. Code quality checks
            await self._run_code_quality_checks()

            # 4. Test execution
            await self._run_automated_tests()

            # 5. Launch dashboard
            await self._launch_dashboard_automated()

            logger.info("✅ AAC Automation Cycle Complete")
            return True

        except Exception as e:
            logger.error(f"❌ Automation failed: {e}")
            return False

    async def _check_system_health(self) -> bool:
        """Check overall system health"""
        logger.info("🔍 Checking system health...")

        try:
            # Check Python environment
            result = subprocess.run([self.python_exe, "--version"],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Python check failed")
                return False

            # Check required packages
            required_packages = ['streamlit', 'pandas', 'numpy', 'asyncio']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.warning(f"Missing package: {package}")

            # Check AAC components
            components_to_check = [
                'shared.config_loader',
                'shared.monitoring',
                'aac.doctrine.doctrine_integration',
                'monitoring.aac_master_monitoring_dashboard'
            ]

            for component in components_to_check:
                try:
                    __import__(component)
                    logger.info(f"✅ {component}")
                except ImportError as e:
                    logger.error(f"❌ {component}: {e}")
                    return False

            logger.info("✅ System health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def _handle_git_operations(self) -> None:
        """Handle git add, commit, and push operations"""
        logger.info("📝 Handling git operations...")

        try:
            # Check if we're in a git repo
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  cwd=self.repo_path, capture_output=True)

            if result.returncode != 0:
                logger.warning("Not a git repository")
                return

            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path,
                         capture_output=True)

            # Check for changes
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                  cwd=self.repo_path)

            if result.returncode == 0:
                logger.info("📋 No changes to commit")
                return

            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"Auto-commit: AAC Matrix Monitor updates - {timestamp}"

            # Commit
            result = subprocess.run(['git', 'commit', '-m', commit_msg],
                                  cwd=self.repo_path, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Commit successful")

                # Try to push
                result = subprocess.run(['git', 'push', 'origin', 'main'],
                                      cwd=self.repo_path, capture_output=True)

                if result.returncode == 0:
                    logger.info("✅ Push successful")
                else:
                    logger.warning("⚠️  Push failed - check remote configuration")
            else:
                logger.warning("⚠️  Commit failed")

        except Exception as e:
            logger.error(f"Git operation error: {e}")

    async def _run_code_quality_checks(self) -> None:
        """Run automated code quality checks"""
        logger.info("🔧 Running code quality checks...")

        try:
            # Import checks
            import_check_script = """
import sys
sys.path.insert(0, '.')

modules_to_check = [
    'shared.config_loader',
    'shared.monitoring',
    'aac.doctrine.doctrine_integration',
    'monitoring.aac_master_monitoring_dashboard',
    'core.aac_master_launcher'
]

failed_imports = []
for module in modules_to_check:
    try:
        __import__(module)
        print(f"✅ {module}")
    except ImportError as e:
        failed_imports.append((module, str(e)))
        print(f"❌ {module}: {e}")

if failed_imports:
    print(f"\\n❌ {len(failed_imports)} import failures")
    sys.exit(1)
else:
    print("\\n✅ All imports successful")
"""

            result = subprocess.run([self.python_exe, '-c', import_check_script],
                                  cwd=self.repo_path, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Import checks passed")
            else:
                logger.warning("⚠️  Import check issues found")

        except Exception as e:
            logger.error(f"Code quality check error: {e}")

    async def _run_automated_tests(self) -> None:
        """Run automated test suite"""
        logger.info("🧪 Running automated tests...")

        try:
            # Simple test script
            test_script = """
import asyncio
import sys
sys.path.insert(0, '.')

async def run_basic_tests():
    results = {'passed': 0, 'failed': 0, 'errors': []}

    # Test 1: Import core components
    try:
        from monitoring.aac_master_monitoring_dashboard import get_master_dashboard, DisplayMode
        results['passed'] += 1
        print("✅ Dashboard import test passed")
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Dashboard import: {e}")

    # Test 2: Doctrine integration
    try:
        from aac.doctrine.doctrine_integration import get_doctrine_integration
        doctrine = get_doctrine_integration()
        await doctrine.initialize()
        results['passed'] += 1
        print("✅ Doctrine integration test passed")
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Doctrine integration: {e}")

    # Test 3: Configuration loading
    try:
        from shared.config_loader import get_config
        config = get_config()
        results['passed'] += 1
        print("✅ Configuration test passed")
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Configuration: {e}")

    return results

# Run tests
results = asyncio.run(run_basic_tests())
print(f"\\nTest Results: {results['passed']} passed, {results['failed']} failed")
if results['errors']:
    print("Errors:")
    for error in results['errors']:
        print(f"  - {error}")
"""

            result = subprocess.run([self.python_exe, '-c', test_script],
                                  cwd=self.repo_path, capture_output=True, text=True)

            if 'failed' in result.stdout.lower() or result.returncode != 0:
                logger.warning("⚠️  Some tests failed")
                logger.info(f"Test output: {result.stdout}")
            else:
                logger.info("✅ Automated tests passed")

        except Exception as e:
            logger.error(f"Test execution error: {e}")

    async def _launch_dashboard_automated(self) -> None:
        """Launch dashboard with automation"""
        logger.info("🚀 Launching AAC Matrix Monitor...")

        try:
            # Launch dashboard in background
            dashboard_cmd = [
                self.python_exe,
                "core/aac_master_launcher.py",
                "--dashboard-only",
                "--display-mode", "web"
            ]

            # Start dashboard process
            process = subprocess.Popen(
                dashboard_cmd,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for server to start
            logger.info("⏳ Waiting for dashboard to start...")
            await asyncio.sleep(5)

            # Check if server is running
            check_cmd = ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                        'http://localhost:8080']
            result = subprocess.run(check_cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip() == '200':
                logger.info("✅ Dashboard running on http://localhost:8080")

                # Auto-open browser
                try:
                    webbrowser.open('http://localhost:8080')
                    logger.info("✅ Browser opened automatically")
                except Exception as e:
                    logger.warning(f"Could not open browser: {e}")

                # Keep process running
                logger.info("🎯 AAC Matrix Monitor automation complete!")
                logger.info("📊 Dashboard: http://localhost:8080")
                logger.info("🤖 AZ Executive Assistant ready")

            else:
                logger.error("❌ Dashboard failed to start properly")
                process.terminate()

        except Exception as e:
            logger.error(f"Dashboard launch error: {e}")

async def main():
    """Main automation entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Automation System")
    parser.add_argument("--repo-path", "-p", help="Path to AAC repository")
    parser.add_argument("--skip-git", action="store_true", help="Skip git operations")
    parser.add_argument("--skip-tests", action="store_true", help="Skip automated tests")
    parser.add_argument("--dashboard-only", action="store_true", help="Launch dashboard only")

    args = parser.parse_args()

    automation = AACAutomation(args.repo_path)

    if args.dashboard_only:
        await automation._launch_dashboard_automated()
    else:
        success = await automation.run_full_automation()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())