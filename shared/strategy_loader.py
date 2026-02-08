"""
Strategy Loader and Validator
=============================
Loads and validates arbitrage strategies from CSV configuration.
Provides automated strategy checking and integration with the execution engine.
"""

import csv
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    """Strategy categorization for validation and routing"""
    ETF_ARBITRAGE = "etf_arbitrage"
    INDEX_ARBITRAGE = "index_arbitrage"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    EVENT_DRIVEN = "event_driven"
    SEASONALITY = "seasonality"
    FLOW_BASED = "flow_based"
    MARKET_MAKING = "market_making"
    CORRELATION = "correlation"


class StrategyStatus(Enum):
    """Strategy validation status"""
    VALID = "valid"
    INVALID = "invalid"
    REQUIRES_REVIEW = "requires_review"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class StrategyConfig:
    """Strategy configuration loaded from CSV"""
    id: int
    name: str
    description: str
    sources: List[str]
    category: StrategyCategory
    status: StrategyStatus
    validation_errors: List[str]
    implementation_notes: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == StrategyStatus.VALID


class StrategyLoader:
    """
    Loads and validates arbitrage strategies from CSV configuration.
    Provides automated checking and integration capabilities.
    """

    def __init__(self, csv_path: str = "50_arbitrage_strategies.csv"):
        self.csv_path = Path(__file__).parent.parent / csv_path
        self.strategies: List[StrategyConfig] = []
        self._loaded = False

    async def load_strategies(self) -> List[StrategyConfig]:
        """Load all strategies from CSV and validate them"""
        if self._loaded:
            return self.strategies

        logger.info(f"Loading strategies from {self.csv_path}")

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    strategy = await self._parse_strategy_row(row)
                    if strategy:
                        self.strategies.append(strategy)

            self._loaded = True
            logger.info(f"Loaded {len(self.strategies)} strategies")

        except FileNotFoundError:
            logger.error(f"Strategy CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            raise

        return self.strategies

    async def _parse_strategy_row(self, row: Dict[str, str]) -> Optional[StrategyConfig]:
        """Parse a single strategy row from CSV"""
        try:
            strategy_id = int(row['id'])
            name = row['strategy_name'].strip()
            
            # Sanitize Unicode characters for Windows console compatibility
            # Replace common Unicode punctuation with ASCII equivalents
            name = name.replace('–', '-').replace('—', '-').replace('‑', '-')
            name = name.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
            name = name.replace('…', '...').replace('•', '*')
            
            # Remove any remaining non-ASCII characters
            name = ''.join(c for c in name if ord(c) < 128)
            
            description = row['one_liner'].strip()
            sources = [s.strip() for s in row['sources'].split(';') if s.strip()]

            # Auto-categorize based on keywords
            category = self._categorize_strategy(name, description)

            # Validate strategy
            status, errors = await self._validate_strategy(strategy_id, name, description, sources)

            return StrategyConfig(
                id=strategy_id,
                name=name,
                description=description,
                sources=sources,
                category=category,
                status=status,
                validation_errors=errors
            )

        except Exception as e:
            logger.warning(f"Error parsing strategy row {row.get('id', 'unknown')}: {e}")
            return None

    def _categorize_strategy(self, name: str, description: str) -> StrategyCategory:
        """Auto-categorize strategy based on name and description"""
        text = (name + " " + description).lower()

        if any(keyword in text for keyword in ['etf', 'nav', 'creation', 'redemption']):
            return StrategyCategory.ETF_ARBITRAGE
        elif any(keyword in text for keyword in ['index', 'reconstitution', 'inclusion']):
            return StrategyCategory.INDEX_ARBITRAGE
        elif any(keyword in text for keyword in ['variance', 'volatility', 'vrp', 'iv', 'rv']):
            return StrategyCategory.VOLATILITY_ARBITRAGE
        elif any(keyword in text for keyword in ['earnings', 'fomc', 'event']):
            return StrategyCategory.EVENT_DRIVEN
        elif any(keyword in text for keyword in ['overnight', 'seasonality', 'tom', 'weekly']):
            return StrategyCategory.SEASONALITY
        elif any(keyword in text for keyword in ['flow', 'liquidity', 'pressure']):
            return StrategyCategory.FLOW_BASED
        elif any(keyword in text for keyword in ['auction', 'market making', 'mm']):
            return StrategyCategory.MARKET_MAKING
        elif any(keyword in text for keyword in ['correlation', 'dispersion']):
            return StrategyCategory.CORRELATION
        else:
            return StrategyCategory.EVENT_DRIVEN  # Default

    async def _validate_strategy(self, strategy_id: int, name: str, description: str,
                               sources: List[str]) -> tuple[StrategyStatus, List[str]]:
        """Validate a strategy configuration"""
        errors = []

        # Check required fields
        if not name:
            errors.append("Strategy name is required")
        if not description:
            errors.append("Strategy description is required")
        if not sources:
            errors.append("Strategy sources are required")

        # Check for duplicate IDs (would need to be done at batch level)
        # Check for valid source format
        for source in sources:
            if not source.startswith('turn'):
                errors.append(f"Invalid source format: {source} (should start with 'turn')")

        # Check description quality
        if len(description) < 20:
            errors.append("Strategy description too short (minimum 20 characters)")
        if len(description) > 200:
            errors.append("Strategy description too long (maximum 200 characters)")

        # For now, mark as valid if no errors
        # In production, this would check against implementation status
        if errors:
            return StrategyStatus.INVALID, errors
        else:
            return StrategyStatus.VALID, []

    async def get_strategies_by_category(self, category: StrategyCategory) -> List[StrategyConfig]:
        """Get all strategies in a specific category"""
        await self.load_strategies()
        return [s for s in self.strategies if s.category == category]

    async def get_valid_strategies(self) -> List[StrategyConfig]:
        """Get all valid strategies"""
        await self.load_strategies()
        return [s for s in self.strategies if s.is_valid]

    async def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded strategies"""
        await self.load_strategies()

        total = len(self.strategies)
        valid = len([s for s in self.strategies if s.is_valid])
        invalid = len([s for s in self.strategies if s.status == StrategyStatus.INVALID])

        category_counts = {}
        for strategy in self.strategies:
            category_counts[strategy.category.value] = category_counts.get(strategy.category.value, 0) + 1

        return {
            'total_strategies': total,
            'valid_strategies': valid,
            'invalid_strategies': invalid,
            'categories': category_counts,
            'validation_rate': valid / total if total > 0 else 0
        }

    async def validate_all_strategies(self) -> Dict[str, Any]:
        """Run comprehensive validation on all strategies"""
        await self.load_strategies()

        results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

        for strategy in self.strategies:
            if strategy.is_valid:
                results['passed'].append({
                    'id': strategy.id,
                    'name': strategy.name,
                    'category': strategy.category.value
                })
            else:
                results['failed'].append({
                    'id': strategy.id,
                    'name': strategy.name,
                    'errors': strategy.validation_errors
                })

        return results


# Global instance for easy access
_strategy_loader = None

def get_strategy_loader() -> StrategyLoader:
    """Get the global strategy loader instance"""
    global _strategy_loader
    if _strategy_loader is None:
        _strategy_loader = StrategyLoader()
    return _strategy_loader