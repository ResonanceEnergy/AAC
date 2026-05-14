"""
PlanktonXD Browser Bot Configuration — AAC v2.7.0
==================================================

Configuration management for the PlanktonXD Browser Bot.
Handles browser settings, risk parameters, and operational controls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from shared.config_loader import get_config


@dataclass
class BrowserBotConfig:
    """Complete configuration for PlanktonXD Browser Bot."""
    
    # ─── Browser Settings ───────────────────────────────────────────────
    headless: bool = True
    window_width: int = 1920
    window_height: int = 1080
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # ─── Timeouts & Waits ───────────────────────────────────────────────
    page_load_timeout: int = 30
    implicit_wait: int = 10
    explicit_wait: int = 15
    trade_confirmation_timeout: int = 60
    
    # ─── Network & Proxy ─────────────────────────────────────────────────
    proxy: Optional[str] = None
    use_proxy_rotation: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    
    # ─── Paths & Storage ─────────────────────────────────────────────────
    screenshots_dir: str = "_scratch/planktonxd_screenshots"
    session_dir: str = "_scratch/planktonxd_session"
    logs_dir: str = "_scratch/planktonxd_logs"
    
    # ─── Polymarket Integration ─────────────────────────────────────────
    base_url: str = "https://polymarket.com"
    markets_url: str = "https://polymarket.com/markets"
    login_required: bool = False  # Browser bot can work without login for viewing
    wallet_connection_required: bool = True  # Required for placing bets
    
    # ─── PlanktonXD Strategy Parameters ─────────────────────────────────
    # (Same as API version for consistency)
    min_bet_usd: float = 5.0
    max_bet_usd: float = 25.0
    absolute_max_bet_usd: float = 50.0
    max_daily_bets: int = 200
    max_open_positions: int = 500
    
    # Deep OTM thresholds
    deep_otm_max_price: float = 0.03
    deep_otm_min_price: float = 0.001
    tail_bet_max_price: float = 0.01
    
    # Edge thresholds
    min_edge_deep_otm: float = 0.005
    min_edge_spread: float = 0.015
    min_edge_contrarian: float = 0.02
    
    # Risk controls
    max_single_market_exposure_pct: float = 2.0
    max_category_exposure_pct: float = 25.0
    bankroll_protection_floor: float = 0.10
    
    # ─── Market Scanning ─────────────────────────────────────────────────
    markets_per_scan: int = 50
    scan_interval_minutes: int = 30
    categories_to_scan: List[str] = None  # None = all categories
    
    # ─── Rate Limiting ───────────────────────────────────────────────────
    min_delay_between_trades: int = 5    # seconds
    max_delay_between_trades: int = 15   # seconds
    page_navigation_delay: int = 2       # seconds
    
    # ─── Error Handling ──────────────────────────────────────────────────
    max_consecutive_failures: int = 5
    failure_cooldown_minutes: int = 30
    screenshot_on_error: bool = True
    continue_on_error: bool = True
    
    # ─── Development & Debug ─────────────────────────────────────────────
    debug_mode: bool = False
    save_page_source: bool = False
    log_all_elements: bool = False
    simulate_trades: bool = False  # For testing without real trades

    def __post_init__(self):
        """Initialize default categories if not specified."""
        if self.categories_to_scan is None:
            self.categories_to_scan = [
                "crypto_price", "politics", "sports", "weather", 
                "entertainment", "science", "economics"
            ]
        
        # Ensure directories exist
        for dir_path in [self.screenshots_dir, self.session_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "BrowserBotConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Browser settings from env
        config.headless = os.environ.get("PLANKTONXD_HEADLESS", "true").lower() == "true"
        config.proxy = os.environ.get("PLANKTONXD_PROXY")
        config.debug_mode = os.environ.get("PLANKTONXD_DEBUG", "false").lower() == "true"
        config.simulate_trades = os.environ.get("PLANKTONXD_SIMULATE", "false").lower() == "true"
        
        # Risk parameters from env
        if "PLANKTONXD_BANKROLL" in os.environ:
            # Bankroll handled separately in bot creation
            pass
        
        if "PLANKTONXD_MAX_DAILY_BETS" in os.environ:
            config.max_daily_bets = int(os.environ["PLANKTONXD_MAX_DAILY_BETS"])
        
        if "PLANKTONXD_MAX_BET_USD" in os.environ:
            config.max_bet_usd = float(os.environ["PLANKTONXD_MAX_BET_USD"])
        
        return config

    @classmethod
    def for_development(cls) -> "BrowserBotConfig":
        """Create development-friendly config."""
        config = cls()
        config.headless = False  # Visible browser for development
        config.debug_mode = True
        config.simulate_trades = True
        config.screenshot_on_error = True
        config.save_page_source = True
        config.max_daily_bets = 10  # Lower limits for testing
        config.max_bet_usd = 10.0
        return config

    @classmethod  
    def for_production(cls) -> "BrowserBotConfig":
        """Create production-ready config."""
        config = cls()
        config.headless = True
        config.debug_mode = False
        config.simulate_trades = False
        config.continue_on_error = True
        config.screenshot_on_error = True
        
        # Full PlanktonXD parameters
        config.max_daily_bets = 200
        config.max_bet_usd = 25.0
        config.max_open_positions = 500
        
        return config

    def get_chrome_options(self) -> List[str]:
        """Get Chrome options for WebDriver."""
        options = []
        
        if self.headless:
            options.append("--headless=new")
        
        options.extend([
            f"--window-size={self.window_width},{self.window_height}",
            f"--user-agent={self.user_agent}",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage", 
            "--disable-gpu",
            "--disable-extensions",
            "--disable-logging",
            "--disable-dev-tools",
            "--no-first-run",
            "--no-default-browser-check"
        ])
        
        if self.proxy:
            options.append(f"--proxy-server={self.proxy}")
        
        # Session persistence
        session_path = Path(self.session_dir) / "chrome_profile"
        options.append(f"--user-data-dir={session_path}")
        
        return options

    def get_element_selectors(self) -> Dict[str, List[str]]:
        """Get CSS selectors for Polymarket elements."""
        return {
            "market_cards": [
                "[data-testid='market-card']",
                ".market-item",
                ".market-row",
                ".market-container"
            ],
            "market_title": [
                ".market-title",
                "[data-testid='market-title']",
                ".question-text",
                "h3", "h4", "h5"
            ],
            "yes_button": [
                "button:contains('Yes')",
                "[data-testid='yes-button']",
                ".outcome-yes",
                ".bet-yes"
            ],
            "no_button": [
                "button:contains('No')",
                "[data-testid='no-button']", 
                ".outcome-no",
                ".bet-no"
            ],
            "amount_input": [
                "input[placeholder*='amount']",
                "input[placeholder*='Amount']",
                "input[data-testid='bet-amount']",
                "input[type='number']",
                ".amount-input input"
            ],
            "place_bet_button": [
                "button:contains('Place Bet')",
                "button:contains('Confirm')",
                "button:contains('Buy')",
                "[data-testid='place-bet']",
                "[data-testid='confirm-bet']"
            ],
            "price_elements": [
                ".price",
                ".probability", 
                ".odds",
                "[data-testid='yes-price']",
                "[data-testid='no-price']",
                ".outcome-price"
            ],
            "volume_elements": [
                ".volume",
                ".trading-volume",
                "[data-testid='volume']"
            ],
            "success_indicators": [
                ".trade-success",
                ".transaction-success",
                ":contains('Transaction confirmed')",
                ":contains('Bet placed')",
                ".success-message"
            ],
            "error_indicators": [
                ".error-message",
                ".transaction-failed", 
                ":contains('Transaction failed')",
                ":contains('Insufficient')",
                ".error-toast"
            ]
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.min_bet_usd >= self.max_bet_usd:
            issues.append("min_bet_usd must be less than max_bet_usd")
        
        if self.max_bet_usd > self.absolute_max_bet_usd:
            issues.append("max_bet_usd exceeds absolute_max_bet_usd")
        
        if self.deep_otm_min_price >= self.deep_otm_max_price:
            issues.append("deep_otm_min_price must be less than deep_otm_max_price")
        
        if self.max_single_market_exposure_pct > self.max_category_exposure_pct:
            issues.append("max_single_market_exposure_pct should not exceed max_category_exposure_pct")
        
        if self.bankroll_protection_floor <= 0 or self.bankroll_protection_floor >= 1:
            issues.append("bankroll_protection_floor must be between 0 and 1")
        
        # Check required directories can be created
        for dir_path in [self.screenshots_dir, self.session_dir, self.logs_dir]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {dir_path}: {e}")
        
        return issues

    def to_dict(self) -> Dict[str, any]:
        """Convert config to dictionary."""
        return {
            "browser": {
                "headless": self.headless,
                "window_size": [self.window_width, self.window_height],
                "user_agent": self.user_agent,
                "proxy": self.proxy
            },
            "timeouts": {
                "page_load": self.page_load_timeout,
                "implicit_wait": self.implicit_wait,
                "explicit_wait": self.explicit_wait,
                "trade_confirmation": self.trade_confirmation_timeout
            },
            "paths": {
                "screenshots": self.screenshots_dir,
                "session": self.session_dir,
                "logs": self.logs_dir
            },
            "strategy": {
                "min_bet_usd": self.min_bet_usd,
                "max_bet_usd": self.max_bet_usd,
                "max_daily_bets": self.max_daily_bets,
                "deep_otm_max_price": self.deep_otm_max_price,
                "min_edge_deep_otm": self.min_edge_deep_otm
            },
            "scanning": {
                "markets_per_scan": self.markets_per_scan,
                "scan_interval_minutes": self.scan_interval_minutes,
                "categories": self.categories_to_scan
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_browser_bot_config(mode: str = "auto") -> BrowserBotConfig:
    """Get appropriate config based on mode."""
    if mode == "development" or mode == "dev":
        return BrowserBotConfig.for_development()
    elif mode == "production" or mode == "prod":
        return BrowserBotConfig.for_production()
    elif mode == "env":
        return BrowserBotConfig.from_env()
    else:
        # Auto-detect based on environment
        if os.environ.get("AAC_ENV") == "production":
            return BrowserBotConfig.for_production()
        elif os.environ.get("DEBUG", "").lower() == "true":
            return BrowserBotConfig.for_development()
        else:
            return BrowserBotConfig.from_env()


def validate_browser_requirements() -> List[str]:
    """Validate browser automation requirements."""
    issues = []
    
    try:
        from selenium import webdriver
    except ImportError:
        issues.append("Selenium not installed: pip install selenium")
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        issues.append("WebDriver Manager not installed: pip install webdriver-manager")
    
    # Check Chrome availability (basic test)
    try:
        import subprocess
        result = subprocess.run(["google-chrome", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # Try chrome command
            result = subprocess.run(["chrome", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                issues.append("Chrome browser not found in PATH")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("Chrome browser not available or not responding")
    
    return issues