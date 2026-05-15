"""
PlanktonXD Browser Bot — AAC v2.7.0 
=====================================

Browser automation version of the PlanktonXD prediction market harvester.
Emulates the legendary planktonXD trader who turned $1,000 → $106,000 in one year
through high-frequency deep OTM arbitrage on Polymarket.

Uses Selenium WebDriver to automate browser interactions with Polymarket instead
of API calls. Maintains all the core PlanktonXD strategies:

1. DEEP OTM HARVESTING — Buy "impossible" outcomes at 0.1¢–3¢
2. SPREAD MARKET-MAKING — Capture bid-ask spreads in thin markets  
3. LIQUIDITY SNIPING — Pick up panic sells in low-volume markets
4. MULTI-CATEGORY DIVERSIFICATION — Sports, crypto, politics, weather, etc.
5. ANTIFRAGILE POSITION SIZING — $5–$25 per bet, 500+ positions

Key Features:
- Browser-based market scanning and trade execution
- Headless and visible modes for development/debugging
- Proxy/VPN support for resilience  
- Screenshot capture for audit trails
- Session persistence and cookie management
- Robust error handling and retry logic
- Integration with existing AAC framework

Browser Requirements:
- Chrome/Chromium with ChromeDriver
- Metamask extension (for wallet connection)
- Stable internet connection
- JavaScript enabled
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import sys

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

# AAC framework imports
from shared.audit_logger import AuditLogger
from shared.communication import CommunicationFramework
from shared.strategy_framework import BaseArbitrageStrategy, StrategyConfig
from strategies.planktonxd_prediction_harvester import (
    BetType,
    MarketCategory, 
    PlanktonBet,
    PredictionMarket,
    HarvesterStats,
    PositionState
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# BROWSER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BrowserConfig:
    """Browser automation configuration."""
    headless: bool = True
    window_width: int = 1920
    window_height: int = 1080
    page_load_timeout: int = 30
    implicit_wait: int = 10
    explicit_wait: int = 15
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    proxy: Optional[str] = None
    screenshots_dir: str = "_scratch/planktonxd_screenshots"
    session_dir: str = "_scratch/planktonxd_session"
    
    # Polymarket-specific
    base_url: str = "https://polymarket.com"
    markets_url: str = "https://polymarket.com/markets"
    login_required: bool = True
    wallet_connection_timeout: int = 60

@dataclass 
class MarketScanResult:
    """Result from browser-based market scan."""
    market_id: str
    question: str
    url: str
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    category: MarketCategory
    resolution_date: Optional[datetime] = None
    screenshot_path: Optional[str] = None

@dataclass
class TradeExecution:
    """Browser trade execution attempt."""
    bet_id: str
    market_url: str
    outcome: str
    price: float
    size_usd: float
    success: bool
    error_msg: str = ""
    screenshot_path: str = ""
    execution_time: datetime = field(default_factory=datetime.now)
    transaction_hash: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# PLANKTONXD BROWSER BOT
# ═══════════════════════════════════════════════════════════════════════════

class PlanktonXDBrowserBot(BaseArbitrageStrategy):
    """
    Browser automation version of PlanktonXD prediction market harvester.
    
    Uses Selenium WebDriver to automate Polymarket interactions while maintaining
    all the core PlanktonXD strategies and risk controls.
    """
    
    # ─── planktonXD-calibrated constants (same as API version) ─────────
    MIN_BET_USD = 5.0
    MAX_BET_USD = 25.0
    ABSOLUTE_MAX_BET_USD = 50.0
    MAX_DAILY_BETS = 200
    MAX_OPEN_POSITIONS = 500
    
    # Deep OTM thresholds  
    DEEP_OTM_MAX_PRICE = 0.03
    DEEP_OTM_MIN_PRICE = 0.001
    TAIL_BET_MAX_PRICE = 0.01
    
    # Edge thresholds
    MIN_EDGE_DEEP_OTM = 0.005
    MIN_EDGE_SPREAD = 0.015
    MIN_EDGE_CONTRARIAN = 0.02
    
    # Risk controls
    MAX_SINGLE_MARKET_EXPOSURE_PCT = 2.0
    MAX_CATEGORY_EXPOSURE_PCT = 25.0
    BANKROLL_PROTECTION_FLOOR = 0.10

    def __init__(
        self,
        config: StrategyConfig,
        communication: CommunicationFramework,
        audit_logger: AuditLogger,
        browser_config: Optional[BrowserConfig] = None,
        bankroll: float = 1000.0
    ):
        super().__init__(config, communication, audit_logger)
        
        self.browser_config = browser_config or BrowserConfig()
        self.driver: Optional[webdriver.Chrome] = None
        self.session_active = False
        
        # Portfolio state (same as API version)
        self.bankroll = bankroll
        self.peak_bankroll = bankroll
        self.open_bets: Dict[str, PlanktonBet] = {}
        self.closed_bets: List[PlanktonBet] = []
        self.stats = HarvesterStats()
        
        # Browser-specific state
        self.logged_in = False
        self.wallet_connected = False
        self.last_screenshot_path = ""
        self.trade_executions: List[TradeExecution] = []
        self.scan_results: List[MarketScanResult] = []
        
        # Category exposure tracking
        self.category_exposure: Dict[MarketCategory, float] = {
            cat: 0.0 for cat in MarketCategory
        }
        
        # Daily counters
        self._daily_bet_count = 0
        self._daily_invested = 0.0
        self._last_reset_date: Optional[datetime] = None
        
        # Ensure directories exist
        Path(self.browser_config.screenshots_dir).mkdir(parents=True, exist_ok=True)
        Path(self.browser_config.session_dir).mkdir(parents=True, exist_ok=True)

    # ─── Browser Management ─────────────────────────────────────────────

    def start_browser(self) -> bool:
        """Initialize and start the Chrome browser."""
        try:
            # Chrome options
            chrome_options = ChromeOptions()
            
            if self.browser_config.headless:
                chrome_options.add_argument("--headless=new")
            
            chrome_options.add_argument(f"--window-size={self.browser_config.window_width},{self.browser_config.window_height}")
            chrome_options.add_argument(f"--user-agent={self.browser_config.user_agent}")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Proxy support
            if self.browser_config.proxy:
                chrome_options.add_argument(f"--proxy-server={self.browser_config.proxy}")
            
            # Session persistence
            session_path = Path(self.browser_config.session_dir) / "chrome_profile"
            chrome_options.add_argument(f"--user-data-dir={session_path}")
            
            # Initialize WebDriver -- try Chrome, fall back to Edge if Chrome
            # binary is missing (common on machines with only MS Edge).
            try:
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                browser_used = "Chrome"
            except WebDriverException as chrome_err:
                if "cannot find Chrome binary" not in str(chrome_err):
                    raise
                logger.warning(f"Chrome not found ({chrome_err}); falling back to Microsoft Edge")
                from selenium.webdriver.edge.options import Options as EdgeOptions
                from selenium.webdriver.edge.service import Service as EdgeService

                edge_options = EdgeOptions()
                if self.browser_config.headless:
                    edge_options.add_argument("--headless=new")
                edge_options.add_argument(
                    f"--window-size={self.browser_config.window_width},"
                    f"{self.browser_config.window_height}"
                )
                edge_options.add_argument(f"--user-agent={self.browser_config.user_agent}")
                edge_options.add_argument("--disable-blink-features=AutomationControlled")
                edge_options.add_argument("--no-sandbox")
                edge_options.add_argument("--disable-dev-shm-usage")
                edge_options.add_argument("--disable-gpu")
                edge_options.add_argument("--no-first-run")
                edge_options.add_argument("--no-default-browser-check")
                edge_options.add_argument("--remote-debugging-port=0")
                # NOTE: deliberately omit --user-data-dir for Edge -- pointing
                # at the real user profile (or a re-used one) causes Edge to
                # crash with "DevToolsActivePort file doesn't exist".

                # Locate msedgedriver: prefer pre-downloaded binary at
                # %LOCALAPPDATA%\msedgedriver\msedgedriver.exe (Selenium Manager
                # auto-download fails behind Meraki on this host).
                _local_edge = (
                    Path(os.environ.get("LOCALAPPDATA", ""))
                    / "msedgedriver" / "msedgedriver.exe"
                )
                if _local_edge.exists():
                    self.driver = webdriver.Edge(
                        service=EdgeService(executable_path=str(_local_edge)),
                        options=edge_options,
                    )
                else:
                    # Fall back to Selenium Manager (may fail if offline)
                    self.driver = webdriver.Edge(options=edge_options)
                browser_used = "Edge"

            # Configure timeouts
            self.driver.implicitly_wait(self.browser_config.implicit_wait)
            self.driver.set_page_load_timeout(self.browser_config.page_load_timeout)

            # Remove webdriver property (anti-detection)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            self.session_active = True
            logger.info(f"PlanktonXD Browser Bot: {browser_used} driver started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False

    def close_browser(self):
        """Close the browser and cleanup."""
        try:
            if self.driver:
                self.driver.quit()
                self.session_active = False
                logger.info("PlanktonXD Browser Bot: Browser closed successfully")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    def take_screenshot(self, name: str = "") -> str:
        """Take screenshot for audit trail."""
        if not self.driver:
            return ""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not name:
                name = f"planktonxd_bot_{timestamp}"
            else:
                name = f"planktonxd_{name}_{timestamp}"
            
            screenshot_path = Path(self.browser_config.screenshots_dir) / f"{name}.png"
            self.driver.save_screenshot(str(screenshot_path))
            self.last_screenshot_path = str(screenshot_path)
            return str(screenshot_path)
            
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")
            return ""

    def wait_for_element(
        self, 
        by: By, 
        value: str, 
        timeout: Optional[int] = None
    ) -> Optional[Any]:
        """Wait for element to be present and return it."""
        try:
            timeout = timeout or self.browser_config.explicit_wait
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.presence_of_element_located((by, value)))
            return element
        except TimeoutException:
            logger.debug(f"Element not found: {by}={value}")
            return None

    def safe_click(self, element) -> bool:
        """Safely click an element with retry logic."""
        for attempt in range(3):
            try:
                # Wait for element to be clickable
                WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable(element))
                element.click()
                return True
            except (StaleElementReferenceException, WebDriverException) as e:
                logger.debug(f"Click attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        return False

    # ─── Polymarket Navigation ──────────────────────────────────────────

    async def navigate_to_polymarket(self) -> bool:
        """Navigate to Polymarket and handle initial setup."""
        try:
            self.driver.get(self.browser_config.base_url)
            await asyncio.sleep(2)
            
            # Take initial screenshot
            self.take_screenshot("polymarket_landing")
            
            # Check if we need to accept cookies/terms
            await self._handle_initial_popups()
            
            logger.info("Successfully navigated to Polymarket")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate to Polymarket: {e}")
            self.take_screenshot("navigation_error")
            return False

    async def _handle_initial_popups(self):
        """Handle cookie banners, terms acceptance, etc."""
        try:
            # Look for common popup/modal patterns
            popup_selectors = [
                "button[data-testid='accept-cookies']",
                "button:contains('Accept')",
                "button:contains('I understand')",
                ".modal button:contains('Continue')",
                "[data-testid='dismiss-modal']"
            ]
            
            for selector in popup_selectors:
                try:
                    element = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=3)
                    if element and element.is_displayed():
                        self.safe_click(element)
                        await asyncio.sleep(1)
                        logger.debug(f"Dismissed popup: {selector}")
                except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                    continue
                    
        except Exception as e:
            logger.debug(f"Error handling popups: {e}")

    async def scan_markets_page(self, page_url: str = None) -> List[MarketScanResult]:
        """Scan markets page for opportunities."""
        try:
            url = page_url or self.browser_config.markets_url
            self.driver.get(url)
            await asyncio.sleep(3)
            
            self.take_screenshot("markets_scan")
            
            # Find market cards/rows
            market_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[data-testid='market-card'], .market-item, .market-row"
            )
            
            scan_results = []
            
            for idx, element in enumerate(market_elements[:50]):  # Limit to first 50
                try:
                    result = await self._extract_market_data(element, idx)
                    if result and self._is_planktonxd_opportunity(result):
                        scan_results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting market {idx}: {e}")
                    continue
            
            self.scan_results.extend(scan_results)
            logger.info(f"PlanktonXD Browser: Scanned {len(market_elements)} markets, found {len(scan_results)} opportunities")
            return scan_results
            
        except Exception as e:
            logger.error(f"Markets scan failed: {e}")
            self.take_screenshot("scan_error")
            return []

    async def _extract_market_data(self, element, idx: int) -> Optional[MarketScanResult]:
        """Extract market data from a market element."""
        try:
            # Try to extract question/title
            question_selectors = [
                ".market-title",
                "[data-testid='market-title']", 
                ".question-text",
                "h3", "h4", "h5"
            ]
            
            question = ""
            for selector in question_selectors:
                try:
                    q_elem = element.find_element(By.CSS_SELECTOR, selector)
                    if q_elem and q_elem.text.strip():
                        question = q_elem.text.strip()
                        break
                except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                    continue
            
            if not question:
                return None
            
            # Extract prices (Yes/No probabilities)
            yes_price, no_price = await self._extract_prices(element)
            
            # Extract volume/liquidity if available
            volume_24h = await self._extract_volume(element)
            liquidity = volume_24h * 0.1  # Rough estimate
            
            # Get market URL
            market_url = await self._extract_market_url(element)
            
            # Categorize market
            category = self._categorize_market(question)
            
            # Generate market ID
            market_id = f"browser_{idx}_{int(time.time())}"
            
            return MarketScanResult(
                market_id=market_id,
                question=question,
                url=market_url,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=volume_24h,
                liquidity=liquidity,
                category=category
            )
            
        except Exception as e:
            logger.debug(f"Error extracting market data: {e}")
            return None

    async def _extract_prices(self, element) -> Tuple[float, float]:
        """Extract Yes/No prices from market element."""
        try:
            price_selectors = [
                ".price", ".probability", ".odds", 
                "[data-testid='yes-price']", "[data-testid='no-price']",
                ".outcome-price"
            ]
            
            prices = []
            for selector in price_selectors:
                try:
                    price_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for p_elem in price_elements:
                        text = p_elem.text.strip().replace('$', '').replace('%', '')
                        # Look for decimal numbers
                        import re
                        matches = re.findall(r'\d+\.?\d*', text)
                        for match in matches:
                            price = float(match)
                            # Convert percentage to decimal if needed
                            if price > 1:
                                price = price / 100
                            prices.append(price)
                except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                    continue
            
            if len(prices) >= 2:
                return prices[0], prices[1]
            elif len(prices) == 1:
                return prices[0], 1.0 - prices[0]
            else:
                return 0.5, 0.5  # Default if no prices found
                
        except Exception as e:
            logger.debug(f"Error extracting prices: {e}")
            return 0.5, 0.5

    async def _extract_volume(self, element) -> float:
        """Extract 24h volume from market element."""
        try:
            volume_selectors = [
                ".volume", ".trading-volume", "[data-testid='volume']"
            ]
            
            for selector in volume_selectors:
                try:
                    vol_elem = element.find_element(By.CSS_SELECTOR, selector)
                    text = vol_elem.text.strip()
                    # Extract numbers (handle K, M suffixes)
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*([KMkm]?)', text)
                    if match:
                        num = float(match.group(1))
                        suffix = match.group(2).lower()
                        if suffix == 'k':
                            num *= 1000
                        elif suffix == 'm':
                            num *= 1000000
                        return num
                except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                    continue
            
            return 10000  # Default volume estimate
            
        except Exception as e:
            return 10000

    async def _extract_market_url(self, element) -> str:
        """Extract market URL from element."""
        try:
            # Look for links within the element
            link_selectors = ["a", "[href]"]
            
            for selector in link_selectors:
                try:
                    link_elem = element.find_element(By.CSS_SELECTOR, selector)
                    href = link_elem.get_attribute("href")
                    if href and "/market/" in href:
                        return href
                except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                    continue
            
            # Fallback: construct URL from current page
            current_url = self.driver.current_url
            return f"{current_url}/market/unknown"
            
        except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
            return ""

    def _categorize_market(self, question: str) -> MarketCategory:
        """Categorize market based on question text (same logic as API version)."""
        q_lower = question.lower()
        
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "price", "$"]
        politics_keywords = ["election", "president", "trump", "biden", "democrat", "republican"]
        sports_keywords = ["nfl", "nba", "mlb", "nhl", "championship", "win", "season"]
        weather_keywords = ["hurricane", "earthquake", "temperature", "storm"]
        
        if any(kw in q_lower for kw in crypto_keywords):
            return MarketCategory.CRYPTO_PRICE
        elif any(kw in q_lower for kw in politics_keywords):
            return MarketCategory.POLITICS
        elif any(kw in q_lower for kw in sports_keywords):
            return MarketCategory.SPORTS
        elif any(kw in q_lower for kw in weather_keywords):
            return MarketCategory.WEATHER
        else:
            return MarketCategory.ENTERTAINMENT

    def _is_planktonxd_opportunity(self, result: MarketScanResult) -> bool:
        """Check if market meets PlanktonXD criteria."""
        # Deep OTM opportunity: Yes or No price in the sweet spot
        min_price = min(result.yes_price, result.no_price)
        
        # Must be in the deep OTM range
        if not (self.DEEP_OTM_MIN_PRICE <= min_price <= self.DEEP_OTM_MAX_PRICE):
            return False
        
        # Must have reasonable volume
        if result.volume_24h < 1000:
            return False
        
        # Estimate edge using same logic as API version
        implied_prob = min_price
        estimated_true_prob = self._estimate_true_probability(result, implied_prob)
        edge = estimated_true_prob - implied_prob
        
        return edge >= self.MIN_EDGE_DEEP_OTM

    def _estimate_true_probability(self, result: MarketScanResult, implied_prob: float) -> float:
        """Estimate true probability (same logic as API version)."""
        # Default: assume market underprices tail by 2-5x
        if implied_prob < 0.01:
            multiplier = 3.0
        elif implied_prob < 0.03:
            multiplier = 1.5
        else:
            multiplier = 1.1
        
        # Category-specific adjustments
        category_multipliers = {
            MarketCategory.CRYPTO_PRICE: 5.0,
            MarketCategory.ESPORTS: 8.0,
            MarketCategory.SPORTS: 2.5,
            MarketCategory.WEATHER: 4.0,
            MarketCategory.POLITICS: 3.0,
        }
        
        if result.category in category_multipliers and implied_prob < 0.01:
            multiplier = category_multipliers[result.category]
        
        return min(implied_prob * multiplier, 0.8)  # Cap at 80%

    # ─── Trade Execution ────────────────────────────────────────────────

    async def execute_trade(
        self, 
        market_result: MarketScanResult,
        outcome: str,
        bet_size: float
    ) -> TradeExecution:
        """Execute a trade through browser automation."""
        
        bet_id = f"pxd_browser_{self.stats.total_bets + 1:06d}"
        
        execution = TradeExecution(
            bet_id=bet_id,
            market_url=market_result.url,
            outcome=outcome,
            price=market_result.yes_price if outcome.lower() == "yes" else market_result.no_price,
            size_usd=bet_size,
            success=False
        )
        
        try:
            # Navigate to market page
            self.driver.get(market_result.url)
            await asyncio.sleep(3)
            
            self.take_screenshot(f"trade_market_{bet_id}")
            
            # Find the outcome button (Yes/No)
            outcome_button = await self._find_outcome_button(outcome)
            if not outcome_button:
                execution.error_msg = f"Could not find {outcome} button"
                return execution
            
            # Click outcome button
            if not self.safe_click(outcome_button):
                execution.error_msg = f"Failed to click {outcome} button"
                return execution
            
            await asyncio.sleep(2)
            self.take_screenshot(f"trade_clicked_{bet_id}")
            
            # Enter bet amount
            amount_input = await self._find_amount_input()
            if not amount_input:
                execution.error_msg = "Could not find amount input"
                return execution
            
            # Clear and enter amount
            amount_input.clear()
            amount_input.send_keys(str(bet_size))
            await asyncio.sleep(1)
            
            # Find and click place bet button
            place_bet_button = await self._find_place_bet_button()
            if not place_bet_button:
                execution.error_msg = "Could not find place bet button"
                return execution
            
            self.take_screenshot(f"trade_ready_{bet_id}")
            
            if not self.safe_click(place_bet_button):
                execution.error_msg = "Failed to click place bet button"
                return execution
            
            # Wait for transaction confirmation
            success = await self._wait_for_trade_confirmation(bet_id)
            
            if success:
                execution.success = True
                execution.screenshot_path = self.take_screenshot(f"trade_success_{bet_id}")
                logger.info(f"PlanktonXD Browser: Trade executed successfully - {bet_id}")
            else:
                execution.error_msg = "Trade confirmation timeout or failed"
                execution.screenshot_path = self.take_screenshot(f"trade_failed_{bet_id}")
            
            return execution
            
        except Exception as e:
            execution.error_msg = f"Trade execution error: {str(e)}"
            execution.screenshot_path = self.take_screenshot(f"trade_error_{bet_id}")
            logger.error(f"Trade execution failed: {e}")
            return execution

    async def _find_outcome_button(self, outcome: str) -> Optional[Any]:
        """Find the Yes/No outcome button."""
        selectors = [
            f"button:contains('{outcome}')",
            f"[data-testid='{outcome.lower()}-button']",
            f".outcome-{outcome.lower()}",
            f".bet-{outcome.lower()}"
        ]
        
        for selector in selectors:
            try:
                element = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=5)
                if element and outcome.lower() in element.text.lower():
                    return element
            except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                continue
        
        # Fallback: look for buttons with outcome text
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            if button.text and outcome.lower() in button.text.lower():
                return button
        
        return None

    async def _find_amount_input(self) -> Optional[Any]:
        """Find the bet amount input field."""
        selectors = [
            "input[placeholder*='amount']",
            "input[placeholder*='Amount']", 
            "input[data-testid='bet-amount']",
            "input[type='number']",
            ".amount-input input",
            ".bet-amount input"
        ]
        
        for selector in selectors:
            element = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=3)
            if element:
                return element
        
        return None

    async def _find_place_bet_button(self) -> Optional[Any]:
        """Find the place bet/confirm button."""
        selectors = [
            "button:contains('Place Bet')",
            "button:contains('Confirm')",
            "button:contains('Buy')", 
            "[data-testid='place-bet']",
            "[data-testid='confirm-bet']",
            ".place-bet-button",
            ".confirm-button"
        ]
        
        for selector in selectors:
            element = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=3)
            if element:
                return element
        
        return None

    async def _wait_for_trade_confirmation(self, bet_id: str, timeout: int = 60) -> bool:
        """Wait for trade confirmation or error."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Look for success indicators
                success_selectors = [
                    ".trade-success", ".transaction-success",
                    ":contains('Transaction confirmed')",
                    ":contains('Bet placed')", 
                    ".success-message"
                ]
                
                for selector in success_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if element and element.is_displayed():
                            return True
                    except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                        continue
                
                # Look for error indicators 
                error_selectors = [
                    ".error-message", ".transaction-failed",
                    ":contains('Transaction failed')",
                    ":contains('Insufficient')"
                ]
                
                for selector in error_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if element and element.is_displayed():
                            logger.warning(f"Trade error detected: {element.text}")
                            return False
                    except (TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException, AttributeError):
                        continue
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.debug(f"Error waiting for confirmation: {e}")
                await asyncio.sleep(2)
        
        logger.warning(f"Trade confirmation timeout for {bet_id}")
        return False

    # ─── Main Strategy Logic ────────────────────────────────────────────

    async def run_planktonxd_cycle(self) -> Dict[str, Any]:
        """Execute a complete PlanktonXD trading cycle."""
        if not self.session_active:
            if not self.start_browser():
                return {"error": "Failed to start browser"}
        
        cycle_start = datetime.now()
        results = {
            "cycle_start": cycle_start.isoformat(),
            "markets_scanned": 0,
            "opportunities_found": 0,
            "trades_attempted": 0,
            "trades_successful": 0,
            "total_deployed": 0.0,
            "errors": []
        }
        
        try:
            # Navigate to Polymarket
            if not await self.navigate_to_polymarket():
                results["errors"].append("Failed to navigate to Polymarket")
                return results
            
            # Scan markets for opportunities
            opportunities = await self.scan_markets_page()
            results["markets_scanned"] = len(self.scan_results)
            results["opportunities_found"] = len(opportunities)
            
            if not opportunities:
                logger.info("PlanktonXD Browser: No opportunities found this cycle")
                return results
            
            # Execute trades on best opportunities
            executed_trades = 0
            successful_trades = 0
            total_deployed = 0.0
            
            # Sort by edge (lowest price first - biggest potential upside)
            opportunities.sort(key=lambda x: min(x.yes_price, x.no_price))
            
            for opp in opportunities[:10]:  # Top 10 opportunities
                if self._daily_bet_count >= self.MAX_DAILY_BETS:
                    break
                    
                if self.bankroll < self.MIN_BET_USD:
                    break
                
                # Determine best outcome and bet size
                if opp.yes_price < opp.no_price:
                    outcome = "Yes"
                    price = opp.yes_price
                else:
                    outcome = "No"
                    price = opp.no_price
                
                # Calculate bet size using PlanktonXD logic
                edge = self._estimate_true_probability(opp, price) - price
                bet_size = self._calculate_bet_size(edge, opp)
                
                if bet_size < self.MIN_BET_USD:
                    continue
                
                # Execute trade
                execution = await self.execute_trade(opp, outcome, bet_size)
                self.trade_executions.append(execution)
                executed_trades += 1
                
                if execution.success:
                    successful_trades += 1
                    total_deployed += bet_size
                    self._update_portfolio_after_trade(opp, outcome, bet_size, price)
                else:
                    results["errors"].append(f"Trade failed: {execution.error_msg}")
                
                # Rate limiting
                await asyncio.sleep(random.uniform(5, 15))
            
            results["trades_attempted"] = executed_trades
            results["trades_successful"] = successful_trades
            results["total_deployed"] = total_deployed
            
            logger.info(
                f"PlanktonXD Browser cycle complete: "
                f"{executed_trades} trades attempted, {successful_trades} successful, "
                f"${total_deployed:.2f} deployed"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"PlanktonXD cycle error: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            self.take_screenshot("cycle_error")
            return results

    def _calculate_bet_size(self, edge: float, market_result: MarketScanResult) -> float:
        """Calculate bet size using PlanktonXD position sizing rules."""
        base_bet = self.MIN_BET_USD
        
        # Scale with edge
        bet = base_bet + (edge * 200)  # Same as API version
        
        # Apply caps
        bet = min(bet, self.MAX_BET_USD)
        bet = min(bet, self.bankroll * 0.05)  # Max 5% of bankroll
        
        # Category exposure limit
        max_cat = self.bankroll * (self.MAX_CATEGORY_EXPOSURE_PCT / 100)
        cat_exposure = self.category_exposure.get(market_result.category, 0.0)
        bet = min(bet, max(max_cat - cat_exposure, 0))
        
        return max(round(bet, 2), 0.0)

    def _update_portfolio_after_trade(
        self, 
        market_result: MarketScanResult,
        outcome: str, 
        bet_size: float,
        entry_price: float
    ):
        """Update portfolio state after successful trade."""
        self.bankroll -= bet_size
        self._daily_bet_count += 1
        self._daily_invested += bet_size
        self.stats.total_bets += 1
        self.stats.total_invested += bet_size
        
        # Update category exposure
        self.category_exposure[market_result.category] = (
            self.category_exposure.get(market_result.category, 0.0) + bet_size
        )

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            "strategy": "PlanktonXD Browser Bot",
            "bankroll": round(self.bankroll, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "total_bets": self.stats.total_bets,
            "daily_bets": self._daily_bet_count,
            "total_invested": round(self.stats.total_invested, 2),
            "open_positions": len(self.open_bets),
            "browser_active": self.session_active,
            "logged_in": self.logged_in,
            "last_screenshot": self.last_screenshot_path,
            "category_exposure": {
                cat.value: round(exposure, 2) 
                for cat, exposure in self.category_exposure.items()
            }
        }

    # ═══════════════════════════════════════════════════════════════════════
    # BaseArbitrageStrategy Abstract Methods Implementation
    # ═══════════════════════════════════════════════════════════════════════

    async def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        logger.info("Initializing PlanktonXD Browser Bot strategy components...")
        
        # Initialize directories
        screenshots_dir = Path(self.browser_config.screenshots_dir)
        session_dir = Path(self.browser_config.session_dir)
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Load saved state if available
        await self._load_session_state()
        
        logger.info("PlanktonXD Browser Bot strategy initialization complete")

    async def _generate_signals(self) -> List[Any]:
        """Generate trading signals based on browser-scraped market data."""
        signals = []
        
        if not self.driver:
            logger.warning("Browser not initialized, cannot generate signals")
            return signals
        
        try:
            # Navigate to markets page
            self.driver.get(self.browser_config.markets_url)
            await asyncio.sleep(2)
            
            # Scan for opportunities
            opportunities = await self._scan_markets_for_opportunities()
            
            # Convert opportunities to signals
            for opportunity in opportunities:
                if len(signals) >= 10:  # Limit signals per cycle
                    break
                
                # Create a trading signal (simplified structure)
                signal = {
                    'market': opportunity.get('market'),
                    'outcome': opportunity.get('outcome'),
                    'price': opportunity.get('price'),
                    'confidence': opportunity.get('confidence', 0.5),
                    'bet_size': self._calculate_bet_size(opportunity),
                    'timestamp': datetime.now()
                }
                signals.append(signal)
            
            logger.info(f"Generated {len(signals)} trading signals")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals

    def _should_generate_signal(self) -> bool:
        """Determine if conditions are right to generate signals."""
        # Check basic preconditions
        if not self.driver:
            return False
        
        # Check if we've hit daily bet limits
        if self.stats.total_bets >= 200:  # Daily limit
            logger.info("Daily bet limit reached, no new signals")
            return False
        
        # Check bankroll minimum
        if self.bankroll < 50:  # Minimum bankroll threshold
            logger.info("Bankroll too low, no new signals")
            return False
        
        # Check if browser is responsive
        try:
            self.driver.current_url
            return True
        except Exception:
            logger.warning("Browser not responsive, cannot generate signals")
            return False

    async def _load_session_state(self):
        """Load saved session state."""
        try:
            state_path = Path(self.browser_config.session_dir) / "planktonxd_state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                # Restore bankroll and stats
                self.bankroll = state.get('bankroll', self.bankroll)
                stats_data = state.get('stats', {})
                
                self.stats.total_bets = stats_data.get('total_bets', 0)
                self.stats.total_invested = stats_data.get('total_invested', 0.0)
                self.stats.winning_bets = stats_data.get('winning_bets', 0)
                self.stats.losing_bets = stats_data.get('losing_bets', 0)
                
                logger.info("Loaded previous session state")
            
        except Exception as e:
            logger.warning(f"Failed to load session state: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Shutdown and Cleanup
    # ═══════════════════════════════════════════════════════════════════════

    async def shutdown(self):
        """Shutdown the browser bot."""
        logger.info("PlanktonXD Browser Bot shutting down...")
        
        # Close any open positions if needed
        await self._close_open_positions()
        
        # Save state
        await self._save_session_state()
        
        # Close browser
        self.close_browser()
        
        logger.info("PlanktonXD Browser Bot shutdown complete")

    async def _close_open_positions(self):
        """Close open positions before shutdown."""
        # Implementation would go here for position management
        pass

    async def _save_session_state(self):
        """Save session state for persistence."""
        try:
            state = {
                "bankroll": self.bankroll,
                "stats": {
                    "total_bets": self.stats.total_bets,
                    "total_invested": self.stats.total_invested,
                    "winning_bets": self.stats.winning_bets,
                    "losing_bets": self.stats.losing_bets
                },
                "category_exposure": {
                    cat.value: exposure 
                    for cat, exposure in self.category_exposure.items()
                },
                "last_run": datetime.now().isoformat()
            }
            
            state_path = Path(self.browser_config.session_dir) / "planktonxd_state.json"
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_planktonxd_browser_bot(
    communication: CommunicationFramework,
    audit_logger: AuditLogger,
    bankroll: float = 1000.0,
    headless: bool = True
) -> PlanktonXDBrowserBot:
    """Factory function to create PlanktonXD browser bot."""
    
    config = StrategyConfig(
        strategy_id="s51_planktonxd_browser_bot",
        name="PlanktonXD Browser Bot",
        strategy_type="prediction_market_arbitrage_browser",
        edge_source="browser_automated_tail_event_mispricing",
        time_horizon="intraday_to_weekly",
        complexity="high",
        data_requirements=["browser_automation", "polymarket_web_interface"],
        execution_requirements=["selenium_webdriver", "chrome_browser"],
        cross_department_dependencies=[
            "Central Accounting (position tracking)",
            "Risk Management (exposure limits)",
            "Data Sources (market feeds)"
        ],
        risk_envelope={
            "max_single_bet_usd": 25.0,
            "max_daily_bets": 200,
            "max_open_positions": 500,
        }
    )
    
    browser_config = BrowserConfig(headless=headless)
    
    return PlanktonXDBrowserBot(
        config=config,
        communication=communication, 
        audit_logger=audit_logger,
        browser_config=browser_config,
        bankroll=bankroll
    )

# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """CLI interface for PlanktonXD Browser Bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PlanktonXD Browser Bot")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    parser.add_argument("--cycles", type=int, default=1, help="Number of trading cycles")
    parser.add_argument("--delay", type=int, default=300, help="Delay between cycles (seconds)")
    
    args = parser.parse_args()
    
    # Create bot
    communication = CommunicationFramework()
    audit_logger = AuditLogger()
    
    bot = create_planktonxd_browser_bot(
        communication=communication,
        audit_logger=audit_logger,
        bankroll=args.bankroll,
        headless=args.headless
    )
    
    try:
        print(f"\n{'='*70}")
        print("  PLANKTONXD BROWSER BOT — AAC v2.7.0")
        print(f"{'='*70}")
        print(f"  Mode: {'Headless' if args.headless else 'Visible'}")
        print(f"  Bankroll: ${args.bankroll:,.2f}")
        print(f"  Cycles: {args.cycles}")
        
        for cycle in range(args.cycles):
            print(f"\n  Cycle {cycle + 1}/{args.cycles} starting...")
            
            results = await bot.run_planktonxd_cycle()
            
            print(f"  Results: {results['opportunities_found']} opportunities, "
                  f"{results['trades_successful']} successful trades")
            
            if cycle < args.cycles - 1:
                print(f"  Waiting {args.delay} seconds...")
                await asyncio.sleep(args.delay)
        
        # Final status
        status = await bot.get_portfolio_status()
        print(f"\n  Final Status:")
        print(f"    Bankroll: ${status['bankroll']:,.2f}")
        print(f"    Total Bets: {status['total_bets']}")
        print(f"    Total Invested: ${status['total_invested']:,.2f}")
        
    except KeyboardInterrupt:
        print("\n  Bot interrupted by user")
    except Exception as e:
        print(f"\n  Bot error: {e}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())