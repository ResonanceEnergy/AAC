"""
Structural Arbitrage Division
=============================

Division focused on structural arbitrage opportunities across different markets,
instruments, and geographies. Exploits pricing inefficiencies between related
financial instruments and markets.

Key Components:
- Cross-Market Arbitrage Agent: Exploits price differences across markets
- Convertible Arbitrage Agent: Trades convertible securities vs underlying stocks
- Volatility Arbitrage Agent: Exploits volatility mispricings
- Capital Structure Arbitrage Agent: Trades different securities of same company
- Index Arbitrage Agent: Exploits differences between index and constituent stocks
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from shared.super_agent_framework import SuperAgent
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

class CrossMarketArbitrageAgent(SuperAgent):
    """Agent for cross-market arbitrage opportunities."""

    def __init__(self, agent_id: str, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(agent_id, communication, audit_logger)
        self.market_spreads = {}
        self.arbitrage_opportunities = []

    async def scan_cross_market_spreads(self, market_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Scan for cross-market arbitrage opportunities."""
        opportunities = []

        # Compare prices across markets for same assets
        for asset in market_data:
            prices = market_data[asset]

            if len(prices) < 2:
                continue

            # Calculate spreads between markets
            market_pairs = self._generate_market_pairs(list(prices.keys()))

            for market1, market2 in market_pairs:
                price1 = prices[market1]
                price2 = prices[market2]

                spread = price1 - price2
                spread_pct = abs(spread) / min(price1, price2)

                # Transaction costs and threshold
                transaction_cost = 0.001  # 0.1%
                threshold = transaction_cost * 2  # Need to cover round trip costs

                if spread_pct > threshold:
                    opportunity = {
                        'asset': asset,
                        'markets': (market1, market2),
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'direction': 'BUY_LOW_SELL_HIGH' if spread > 0 else 'BUY_HIGH_SELL_LOW',
                        'expected_profit': spread_pct - (transaction_cost * 2),
                        'timestamp': datetime.now()
                    }

                    opportunities.append(opportunity)

        self.arbitrage_opportunities = opportunities

        return {'opportunities': opportunities}

    def _generate_market_pairs(self, markets: List[str]) -> List[tuple]:
        """Generate all possible market pairs."""
        pairs = []
        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                pairs.append((markets[i], markets[j]))
        return pairs

class ConvertibleArbitrageAgent(SuperAgent):
    """Agent for convertible bond arbitrage strategies."""

    def __init__(self, agent_id: str, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(agent_id, communication, audit_logger)
        self.convertible_positions = {}
        self.arbitrage_signals = []

    async def analyze_convertible_arbitrage(self, convertible_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convertible bonds for arbitrage opportunities."""
        opportunities = []

        for convertible in convertible_data:
            stock_price = convertible.get('stock_price', 0)
            convertible_price = convertible.get('convertible_price', 0)
            conversion_ratio = convertible.get('conversion_ratio', 1)
            bond_floor = convertible.get('bond_floor', 0)

            # Calculate theoretical values
            theoretical_value = max(
                stock_price * conversion_ratio,  # Conversion value
                bond_floor  # Bond floor value
            )

            # Calculate mispricing
            mispricing = convertible_price - theoretical_value
            mispricing_pct = mispricing / theoretical_value if theoretical_value > 0 else 0

            # Check for arbitrage opportunity
            if abs(mispricing_pct) > 0.02:  # 2% threshold
                opportunity = {
                    'convertible': convertible.get('symbol', ''),
                    'stock_price': stock_price,
                    'convertible_price': convertible_price,
                    'theoretical_value': theoretical_value,
                    'mispricing': mispricing,
                    'mispricing_pct': mispricing_pct,
                    'strategy': 'BUY_CONVERTIBLE_SELL_STOCK' if mispricing < 0 else 'BUY_STOCK_SELL_CONVERTIBLE',
                    'expected_return': abs(mispricing_pct),
                    'timestamp': datetime.now()
                }

                opportunities.append(opportunity)

        self.arbitrage_signals = opportunities

        return {'opportunities': opportunities}

class VolatilityArbitrageAgent(SuperAgent):
    """Agent for volatility arbitrage strategies."""

    def __init__(self, agent_id: str, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(agent_id, communication, audit_logger)
        self.volatility_surface = {}
        self.arbitrage_positions = []

    async def analyze_volatility_surface(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility surface for arbitrage opportunities."""
        opportunities = []

        underlying_price = options_data.get('underlying_price', 0)
        options = options_data.get('options', [])

        # Build implied volatility surface
        vol_surface = self._build_volatility_surface(options, underlying_price)

        # Check for arbitrage opportunities
        calendar_arbitrage = self._check_calendar_arbitrage(vol_surface)
        butterfly_arbitrage = self._check_butterfly_arbitrage(vol_surface)
        smile_arbitrage = self._check_volatility_smile_arbitrage(vol_surface)

        opportunities.extend(calendar_arbitrage)
        opportunities.extend(butterfly_arbitrage)
        opportunities.extend(smile_arbitrage)

        self.volatility_surface = vol_surface

        return {
            'volatility_surface': vol_surface,
            'arbitrage_opportunities': opportunities
        }

    def _build_volatility_surface(self, options: List[Dict], underlying_price: float) -> Dict[str, Any]:
        """Build implied volatility surface."""
        surface = {}

        for option in options:
            strike = option.get('strike', 0)
            expiry = option.get('expiry', '')
            volatility = option.get('implied_volatility', 0)
            option_type = option.get('type', '')

            if expiry not in surface:
                surface[expiry] = {}

            moneyness = strike / underlying_price
            surface[expiry][f"{option_type}_{moneyness:.2f}"] = volatility

        return surface

    def _check_calendar_arbitrage(self, vol_surface: Dict[str, Any]) -> List[Dict]:
        """Check for calendar spread arbitrage."""
        opportunities = []

        # Simplified calendar arbitrage check
        expiries = sorted(vol_surface.keys())

        for i in range(len(expiries) - 1):
            near_term = expiries[i]
            far_term = expiries[i + 1]

            near_vols = vol_surface[near_term]
            far_vols = vol_surface[far_term]

            # Check if far-term volatility is unreasonably low
            for strike_key in near_vols:
                if strike_key in far_vols:
                    near_vol = near_vols[strike_key]
                    far_vol = far_vols[strike_key]

                    if far_vol < near_vol * 0.8:  # 20% lower
                        opportunities.append({
                            'type': 'calendar_arbitrage',
                            'near_term': near_term,
                            'far_term': far_term,
                            'strike': strike_key,
                            'near_vol': near_vol,
                            'far_vol': far_vol,
                            'expected_return': (near_vol - far_vol) / near_vol
                        })

        return opportunities

    def _check_butterfly_arbitrage(self, vol_surface: Dict[str, Any]) -> List[Dict]:
        """Check for butterfly arbitrage opportunities."""
        opportunities = []

        # Simplified butterfly check - look for negative butterfly spreads
        for expiry in vol_surface:
            vols = vol_surface[expiry]

            # Get ATM and wing volatilities
            atm_vol = vols.get('call_1.00', 0)  # At-the-money
            wing_vol_low = vols.get('call_0.90', 0)  # 10% OTM
            wing_vol_high = vols.get('call_1.10', 0)  # 10% ITM

            if atm_vol and wing_vol_low and wing_vol_high:
                # Butterfly spread: Buy wings, sell body
                butterfly_cost = wing_vol_low + wing_vol_high - 2 * atm_vol

                if butterfly_cost < 0:  # Arbitrage opportunity
                    opportunities.append({
                        'type': 'butterfly_arbitrage',
                        'expiry': expiry,
                        'atm_vol': atm_vol,
                        'wing_vol_low': wing_vol_low,
                        'wing_vol_high': wing_vol_high,
                        'butterfly_cost': butterfly_cost,
                        'expected_return': abs(butterfly_cost)
                    })

        return opportunities

    def _check_volatility_smile_arbitrage(self, vol_surface: Dict[str, Any]) -> List[Dict]:
        """Check for volatility smile arbitrage."""
        opportunities = []

        for expiry in vol_surface:
            vols = vol_surface[expiry]

            # Check if smile is too steep
            otm_vol = vols.get('call_0.90', 0)
            atm_vol = vols.get('call_1.00', 0)
            itm_vol = vols.get('call_1.10', 0)

            if otm_vol and atm_vol and itm_vol:
                smile_steepness = (otm_vol + itm_vol) / (2 * atm_vol)

                if smile_steepness > 1.2:  # Smile too steep
                    opportunities.append({
                        'type': 'smile_arbitrage',
                        'expiry': expiry,
                        'otm_vol': otm_vol,
                        'atm_vol': atm_vol,
                        'itm_vol': itm_vol,
                        'smile_steepness': smile_steepness,
                        'expected_return': (smile_steepness - 1.0) * 0.1
                    })

        return opportunities

class CapitalStructureArbitrageAgent(SuperAgent):
    """Agent for capital structure arbitrage."""

    def __init__(self, agent_id: str, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(agent_id, communication, audit_logger)
        self.capital_structure_positions = {}
        self.relative_value_signals = []

    async def analyze_capital_structure(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company's capital structure for arbitrage opportunities."""
        opportunities = []

        securities = company_data.get('securities', {})

        # Compare different securities of same company
        if 'common_stock' in securities and 'preferred_stock' in securities:
            stock_price = securities['common_stock'].get('price', 0)
            preferred_price = securities['preferred_stock'].get('price', 0)
            dividend_yield = securities['preferred_stock'].get('dividend_yield', 0)

            # Check for preferred stock arbitrage
            if dividend_yield > 0.08 and preferred_price > stock_price:  # High yield, expensive preferred
                opportunities.append({
                    'type': 'preferred_stock_arbitrage',
                    'stock_price': stock_price,
                    'preferred_price': preferred_price,
                    'dividend_yield': dividend_yield,
                    'strategy': 'SHORT_PREFERRED_LONG_STOCK',
                    'expected_return': dividend_yield * 0.5  # Half the yield as expected return
                })

        # Check CDS vs bonds
        if 'cds_spread' in company_data and 'bond_yield' in company_data:
            cds_spread = company_data['cds_spread']
            bond_yield = company_data['bond_yield']

            # CDS-bond basis trade
            basis = cds_spread - bond_yield

            if abs(basis) > 50:  # 50bps threshold
                opportunities.append({
                    'type': 'cds_bond_basis',
                    'cds_spread': cds_spread,
                    'bond_yield': bond_yield,
                    'basis': basis,
                    'strategy': 'BUY_PROTECTION_SELL_BOND' if basis > 50 else 'SELL_PROTECTION_BUY_BOND',
                    'expected_return': abs(basis) / 10000  # Convert to decimal
                })

        self.relative_value_signals = opportunities

        return {'opportunities': opportunities}

class IndexArbitrageAgent(SuperAgent):
    """Agent for index arbitrage strategies."""

    def __init__(self, agent_id: str, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(agent_id, communication, audit_logger)
        self.index_fair_values = {}
        self.arbitrage_trades = []

    async def calculate_index_fair_value(self, index_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fair value of index vs constituent stocks."""
        index_price = index_data.get('index_price', 0)
        constituents = index_data.get('constituents', [])
        weights = index_data.get('weights', [])

        if not constituents or not weights:
            return {'error': 'Missing constituent data'}

        # Calculate weighted average of constituents
        fair_value = 0
        total_weight = 0

        for i, constituent in enumerate(constituents):
            price = constituent.get('price', 0)
            weight = weights[i] if i < len(weights) else 1/len(constituents)

            fair_value += price * weight
            total_weight += weight

        if total_weight > 0:
            fair_value /= total_weight

        # Calculate mispricing
        mispricing = index_price - fair_value
        mispricing_pct = mispricing / fair_value if fair_value > 0 else 0

        result = {
            'index_price': index_price,
            'fair_value': fair_value,
            'mispricing': mispricing,
            'mispricing_pct': mispricing_pct,
            'constituents_count': len(constituents)
        }

        self.index_fair_values[index_data.get('index_name', 'unknown')] = result

        return result

    async def execute_index_arbitrage(self, mispricing_data: Dict[str, Any],
                                    threshold: float = 0.001) -> Dict[str, Any]:
        """Execute index arbitrage trade if mispricing exceeds threshold."""
        mispricing_pct = mispricing_data.get('mispricing_pct', 0)

        if abs(mispricing_pct) > threshold:
            # Determine trade direction
            if mispricing_pct > threshold:  # Index overvalued
                strategy = 'SELL_INDEX_BUY_CONSTITUENTS'
            else:  # Index undervalued
                strategy = 'BUY_INDEX_SELL_CONSTITUENTS'

            trade = {
                'strategy': strategy,
                'mispricing_pct': mispricing_pct,
                'expected_convergence': abs(mispricing_pct) * 0.8,  # Expected 80% convergence
                'timestamp': datetime.now()
            }

            self.arbitrage_trades.append(trade)

            return trade

        return {'status': 'no_arbitrage', 'mispricing_pct': mispricing_pct}

class StructuralArbitrageDivision:
    """Main division class for Structural Arbitrage operations."""

    def __init__(self, communication: CommunicationFramework, audit_logger: AuditLogger):
        self.communication = communication
        self.audit_logger = audit_logger

        # Initialize specialized agents
        self.cross_market_agent = CrossMarketArbitrageAgent(
            'cross_market_arbitrage_agent',
            communication,
            audit_logger
        )

        self.convertible_agent = ConvertibleArbitrageAgent(
            'convertible_arbitrage_agent',
            communication,
            audit_logger
        )

        self.volatility_agent = VolatilityArbitrageAgent(
            'volatility_arbitrage_agent',
            communication,
            audit_logger
        )

        self.capital_structure_agent = CapitalStructureArbitrageAgent(
            'capital_structure_arbitrage_agent',
            communication,
            audit_logger
        )

        self.index_agent = IndexArbitrageAgent(
            'index_arbitrage_agent',
            communication,
            audit_logger
        )

        self.agents = [
            self.cross_market_agent,
            self.convertible_agent,
            self.volatility_agent,
            self.capital_structure_agent,
            self.index_agent
        ]

    async def initialize_division(self) -> bool:
        """Initialize the Structural Arbitrage Division."""
        try:
            logger.info("Initializing Structural Arbitrage Division...")

            # Initialize all agents
            for agent in self.agents:
                await agent.initialize()

            # Register agents with communication framework
            for agent in self.agents:
                await self.communication.register_agent(agent.agent_id, agent)

            await self.audit_logger.log_event(
                'division_initialization',
                'Structural Arbitrage Division initialized successfully',
                {'agents_count': len(self.agents)}
            )

            logger.info("Structural Arbitrage Division initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Structural Arbitrage Division: {e}")
            await self.audit_logger.log_event(
                'division_initialization_error',
                f'Structural Arbitrage Division initialization failed: {e}',
                {'error': str(e)}
            )
            return False

    async def run_division_operations(self) -> Dict[str, Any]:
        """Run core division operations."""
        results = {}

        try:
            # Run cross-market arbitrage scan
            market_data = {
                'AAPL': {'NYSE': 150.0, 'NASDAQ': 150.5, 'LSE': 149.8},
                'MSFT': {'NYSE': 300.0, 'NASDAQ': 299.5, 'LSE': 301.2}
            }
            cross_market_results = await self.cross_market_agent.scan_cross_market_spreads(market_data)
            results['cross_market_arbitrage'] = cross_market_results

            # Run convertible arbitrage analysis
            convertible_data = [{
                'symbol': 'AAPL_CV',
                'stock_price': 150.0,
                'convertible_price': 155.0,
                'conversion_ratio': 1.0,
                'bond_floor': 140.0
            }]
            convertible_results = await self.convertible_agent.analyze_convertible_arbitrage(convertible_data)
            results['convertible_arbitrage'] = convertible_results

            # Run volatility arbitrage analysis
            options_data = {
                'underlying_price': 150.0,
                'options': [
                    {'strike': 135, 'expiry': '2024-01', 'implied_volatility': 0.25, 'type': 'call'},
                    {'strike': 150, 'expiry': '2024-01', 'implied_volatility': 0.22, 'type': 'call'},
                    {'strike': 165, 'expiry': '2024-01', 'implied_volatility': 0.28, 'type': 'call'}
                ]
            }
            volatility_results = await self.volatility_agent.analyze_volatility_surface(options_data)
            results['volatility_arbitrage'] = volatility_results

            # Run capital structure analysis
            company_data = {
                'securities': {
                    'common_stock': {'price': 150.0},
                    'preferred_stock': {'price': 160.0, 'dividend_yield': 0.09}
                },
                'cds_spread': 200,  # 200bps
                'bond_yield': 180   # 180bps
            }
            capital_results = await self.capital_structure_agent.analyze_capital_structure(company_data)
            results['capital_structure'] = capital_results

            # Run index arbitrage
            index_data = {
                'index_name': 'SPY',
                'index_price': 400.0,
                'constituents': [
                    {'price': 150.0}, {'price': 300.0}, {'price': 250.0}
                ],
                'weights': [0.4, 0.35, 0.25]
            }
            fair_value = await self.index_agent.calculate_index_fair_value(index_data)
            index_arbitrage = await self.index_agent.execute_index_arbitrage(fair_value)
            results['index_arbitrage'] = {'fair_value': fair_value, 'trade': index_arbitrage}

            await self.audit_logger.log_event(
                'division_operations',
                'Structural Arbitrage Division operations completed',
                {'results_count': len(results)}
            )

        except Exception as e:
            logger.error(f"Error in Structural Arbitrage Division operations: {e}")
            results['error'] = str(e)

        return results

    async def shutdown_division(self) -> bool:
        """Shutdown the Structural Arbitrage Division."""
        try:
            logger.info("Shutting down Structural Arbitrage Division...")

            # Shutdown all agents
            for agent in self.agents:
                await agent.shutdown()

            await self.audit_logger.log_event(
                'division_shutdown',
                'Structural Arbitrage Division shut down successfully'
            )

            logger.info("Structural Arbitrage Division shut down successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down Structural Arbitrage Division: {e}")
            return False


async def get_structural_arbitrage_division() -> StructuralArbitrageDivision:
    """Factory function to create and initialize Structural Arbitrage Division."""
    from shared.communication import CommunicationFramework
    from shared.audit_logger import AuditLogger

    communication = CommunicationFramework()
    audit_logger = AuditLogger()

    division = StructuralArbitrageDivision(communication, audit_logger)

    if await division.initialize_division():
        return division
    else:
        raise RuntimeError("Failed to initialize Structural Arbitrage Division")