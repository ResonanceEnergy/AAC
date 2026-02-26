#!/usr/bin/env python3
"""
TradingExecution - Risk Manager
===============================
Pre-trade and portfolio-level risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import sys

# Add shared module to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from .trading_engine import Order, Position, OrderSide


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size_usd: float = 10000.0
    max_daily_loss_usd: float = 1000.0
    max_open_positions: int = 10
    max_order_size_usd: float = 5000.0
    max_leverage: float = 1.0
    max_concentration_pct: float = 0.25  # Max 25% in single asset
    daily_trade_limit: int = 100
    min_liquidity_ratio: float = 0.01  # Order must be < 1% of market liquidity


@dataclass
class RiskState:
    """Current risk state tracking"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    total_exposure_usd: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    violations: List[Dict] = field(default_factory=list)


class RiskManager:
    """
    Portfolio and order-level risk management.
    
    Responsibilities:
    - Pre-trade risk checks
    - Position sizing
    - Daily loss limits
    - Concentration limits
    - Leverage monitoring
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.config = get_config()
        self.logger = logging.getLogger('RiskManager')
        
        # Initialize limits from config or use provided
        if limits is None:
            self.limits = RiskLimits(
                max_position_size_usd=self.config.risk.max_position_size_usd,
                max_daily_loss_usd=self.config.risk.max_daily_loss_usd,
                max_open_positions=self.config.risk.max_open_positions,
            )
        else:
            self.limits = limits
        
        # State tracking
        self.state = RiskState()
        self.positions: Dict[str, Position] = {}
        
        self.logger.info(f"RiskManager initialized with limits: max_position=${self.limits.max_position_size_usd}, max_daily_loss=${self.limits.max_daily_loss_usd}")

    def _reset_daily_state(self):
        """Reset daily tracking if new day"""
        now = datetime.now()
        if now.date() > self.state.last_reset.date():
            self.logger.info("Resetting daily risk state")
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_reset = now

    def check_order(self, order: Order, current_price: float) -> Tuple[bool, List[str]]:
        """
        Validate order against risk limits.
        
        Args:
            order: Order to validate
            current_price: Current market price for the symbol
            
        Returns:
            Tuple of (approved: bool, violations: List[str])
        """
        self._reset_daily_state()
        violations = []
        
        order_value_usd = order.quantity * (order.price or current_price)
        
        # Check 1: Order size limit
        if order_value_usd > self.limits.max_order_size_usd:
            violations.append(f"Order size ${order_value_usd:.2f} exceeds max ${self.limits.max_order_size_usd:.2f}")
        
        # Check 2: Position size limit
        existing_position_value = self._get_position_value(order.symbol)
        new_total = existing_position_value + order_value_usd if order.side == OrderSide.BUY else existing_position_value
        
        if new_total > self.limits.max_position_size_usd:
            violations.append(f"Total position ${new_total:.2f} would exceed max ${self.limits.max_position_size_usd:.2f}")
        
        # Check 3: Daily loss limit
        if self.state.daily_pnl < -self.limits.max_daily_loss_usd:
            violations.append(f"Daily loss limit reached: ${self.state.daily_pnl:.2f}")
        
        # Check 4: Open positions limit
        if len(self.positions) >= self.limits.max_open_positions and order.side == OrderSide.BUY:
            violations.append(f"Max open positions ({self.limits.max_open_positions}) reached")
        
        # Check 5: Daily trade limit
        if self.state.daily_trades >= self.limits.daily_trade_limit:
            violations.append(f"Daily trade limit ({self.limits.daily_trade_limit}) reached")
        
        # Check 6: Concentration limit
        total_exposure = self._calculate_total_exposure()
        if total_exposure > 0:
            concentration = (existing_position_value + order_value_usd) / total_exposure
            if concentration > self.limits.max_concentration_pct:
                violations.append(f"Concentration {concentration:.1%} exceeds max {self.limits.max_concentration_pct:.1%}")
        
        # Log violations
        if violations:
            self.state.violations.append({
                'order_id': order.order_id,
                'timestamp': datetime.now().isoformat(),
                'violations': violations,
            })
            for v in violations:
                self.logger.warning(f"Risk violation: {v}")
        
        return (len(violations) == 0, violations)

    def _get_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol"""
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos.quantity * pos.current_price
        return 0.0

    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(p.quantity * p.current_price for p in self.positions.values())

    def update_position(self, position: Position):
        """Update tracked position"""
        self.positions[position.position_id] = position
        self.state.total_exposure_usd = self._calculate_total_exposure()

    def remove_position(self, position_id: str):
        """Remove a closed position"""
        if position_id in self.positions:
            del self.positions[position_id]
            self.state.total_exposure_usd = self._calculate_total_exposure()

    def record_trade(self, pnl: float):
        """Record a completed trade and its P&L"""
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl
        
        if self.state.daily_pnl < -self.limits.max_daily_loss_usd:
            self.logger.error(f"DAILY LOSS LIMIT BREACHED: ${self.state.daily_pnl:.2f}")

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
    ) -> float:
        """
        Calculate recommended position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            signal_strength: Signal confidence (0-1)
            volatility: Expected volatility
            
        Returns:
            Recommended position size in base currency units
        """
        # Base size from risk limit
        base_usd = min(
            self.limits.max_order_size_usd,
            self.limits.max_position_size_usd - self._get_position_value(symbol),
        )
        
        # Adjust for signal strength
        adjusted_usd = base_usd * signal_strength
        
        # Adjust for volatility (lower size in high vol)
        vol_adjustment = min(1.0, 0.02 / max(volatility, 0.001))
        adjusted_usd *= vol_adjustment
        
        # Convert to units
        position_size = adjusted_usd / current_price
        
        self.logger.debug(f"Position size for {symbol}: {position_size:.6f} units (${adjusted_usd:.2f})")
        
        return position_size

    def get_risk_report(self) -> Dict:
        """Generate risk status report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': self.state.daily_pnl,
            'daily_pnl_limit': -self.limits.max_daily_loss_usd,
            'daily_pnl_remaining': self.limits.max_daily_loss_usd + self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'daily_trade_limit': self.limits.daily_trade_limit,
            'open_positions': len(self.positions),
            'max_positions': self.limits.max_open_positions,
            'total_exposure_usd': self.state.total_exposure_usd,
            'max_position_size_usd': self.limits.max_position_size_usd,
            'recent_violations': self.state.violations[-10:],
            'status': 'OK' if self.state.daily_pnl > -self.limits.max_daily_loss_usd else 'LIMIT_BREACHED',
        }
