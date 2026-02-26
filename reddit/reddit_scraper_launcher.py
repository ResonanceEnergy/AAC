#!/usr/bin/env python3
"""
AAC Reddit Scraper Launcher
===========================

Simple launcher script for the AAC Multi-Subreddit Continuous Scraper.
Provides easy start/stop functionality and basic monitoring.
"""

import os
import sys
import time
import signal
import subprocess
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditScraperLauncher:
    """Launcher for the AAC Reddit scraper"""

    def __init__(self):
        self.scraper_script = "aac_multi_subreddit_scraper.py"
        self.process = None
        self.running = False

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import praw
            import apscheduler
            import textblob
            logger.info("✅ All dependencies are installed")
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.error("Install with: pip install praw apscheduler textblob")
            return False

    def check_credentials(self):
        """Check if Reddit credentials are configured"""
        from dotenv import load_dotenv
        load_dotenv()

        required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
        missing = []

        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            logger.error(f"❌ Missing Reddit credentials: {', '.join(missing)}")
            logger.error("Please set them in your .env file")
            return False

        logger.info("✅ Reddit credentials are configured")
        return True

    def start_scraper(self, mode="continuous"):
        """Start the scraper in specified mode"""
        if not self.check_dependencies():
            return False

        if not self.check_credentials():
            return False

        try:
            cmd = [sys.executable, self.scraper_script]
            if mode == "single":
                cmd.append("--single")

            logger.info(f"🚀 Starting Reddit scraper in {mode} mode...")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )

            self.running = True
            logger.info(f"✅ Scraper started with PID: {self.process.pid}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to start scraper: {e}")
            return False

    def stop_scraper(self):
        """Stop the running scraper"""
        if self.process and self.running:
            try:
                logger.info("🛑 Stopping Reddit scraper...")
                self.process.terminate()

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    logger.info("✅ Scraper stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Scraper didn't stop gracefully, forcing termination...")
                    self.process.kill()
                    logger.info("✅ Scraper force-terminated")

                self.running = False
                self.process = None

            except Exception as e:
                logger.error(f"❌ Error stopping scraper: {e}")
        else:
            logger.info("No scraper process running")

    def monitor_scraper(self):
        """Monitor the scraper process"""
        if not self.process:
            return False

        if self.process.poll() is None:
            return True  # Still running
        else:
            # Process has exited
            self.running = False
            return_code = self.process.returncode

            if return_code == 0:
                logger.info("✅ Scraper exited normally")
            else:
                logger.error(f"❌ Scraper exited with code: {return_code}")
                # Log any error output
                if self.process.stderr:
                    error_output = self.process.stderr.read()
                    if error_output:
                        logger.error(f"Scraper stderr: {error_output}")

            return False

    def run_interactive(self):
        """Run in interactive mode with menu"""
        print("AAC Reddit Scraper Launcher")
        print("=" * 40)

        while True:
            print("\nOptions:")
            print("1. Start continuous scraping (15-min intervals)")
            print("2. Run single scrape cycle (test mode)")
            print("3. Stop scraper")
            print("4. Check scraper status")
            print("5. View recent logs")
            print("6. Exit")

            try:
                choice = input("\nSelect option (1-6): ").strip()

                if choice == "1":
                    if self.running:
                        print("❌ Scraper is already running")
                    else:
                        if self.start_scraper("continuous"):
                            print("✅ Continuous scraping started")
                        else:
                            print("❌ Failed to start scraper")

                elif choice == "2":
                    if self.running:
                        print("❌ Scraper is already running")
                    else:
                        print("🧪 Running single scrape cycle...")
                        if self.start_scraper("single"):
                            # Wait for completion
                            while self.monitor_scraper():
                                time.sleep(1)
                            print("✅ Single scrape cycle completed")
                        else:
                            print("❌ Failed to run single cycle")

                elif choice == "3":
                    self.stop_scraper()

                elif choice == "4":
                    if self.monitor_scraper():
                        print(f"✅ Scraper is running (PID: {self.process.pid})")
                    else:
                        print("❌ Scraper is not running")

                elif choice == "5":
                    self.show_recent_logs()

                elif choice == "6":
                    if self.running:
                        confirm = input("Scraper is running. Stop it? (y/N): ").strip().lower()
                        if confirm == "y":
                            self.stop_scraper()
                    print("Goodbye!")
                    break

                else:
                    print("❌ Invalid option")

            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                self.stop_scraper()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_recent_logs(self):
        """Show recent log entries"""
        log_file = "aac_reddit_scraper.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-20:]  # Last 20 lines
                    print("\nRecent Log Entries:")
                    print("-" * 50)
                    for line in lines:
                        print(line.strip())
            except Exception as e:
                print(f"❌ Error reading log file: {e}")
        else:
            print("❌ Log file not found")

def main():
    """Main entry point"""
    launcher = RedditScraperLauncher()

    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "start":
            launcher.start_scraper("continuous")
        elif command == "single":
            launcher.start_scraper("single")
        elif command == "stop":
            launcher.stop_scraper()
        elif command == "status":
            if launcher.monitor_scraper():
                print(f"✅ Scraper is running (PID: {launcher.process.pid})")
            else:
                print("❌ Scraper is not running")
        else:
            print("Usage: python reddit_scraper_launcher.py [start|single|stop|status]")
            print("Or run without arguments for interactive mode")
            sys.exit(1)
    else:
        # Interactive mode
        launcher.run_interactive()

if __name__ == "__main__":
    main()