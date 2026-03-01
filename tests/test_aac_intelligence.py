import pytest
from unittest.mock import MagicMock
from aac_intelligence import MarketData, FinancialIntelligence
from datetime import datetime, date

# Mock data for testing
mock_balance_sheet = {
    'assets': 100000.0,
    'liabilities': 50000.0,
    'equity': 50000.0
}

mock_income_statement = {
    'revenue': 150000.0,
    'expenses': 80000.0,
    'net_income': 70000.0
}

@pytest.fixture
def mock_accounting_engine():
    """Fixture for a mock accounting engine."""
    engine = MagicMock()
    engine.get_balance_sheet.return_value = mock_balance_sheet
    engine.get_income_statement.return_value = mock_income_statement
    return engine

@pytest.fixture
def financial_intelligence(mock_accounting_engine):
    """Fixture for FinancialIntelligence class."""
    return FinancialIntelligence(accounting_engine=mock_accounting_engine)

def test_marketdata_get_stock_price_happy_path():
    """Test MarketData.get_stock_price() for a known symbol."""
    market_data = MarketData()
    price = market_data.get_stock_price('AAPL')
    assert price == 175.50, "Expected the stock price for 'AAPL' to be 175.50"

def test_marketdata_get_stock_price_unknown_symbol():
    """Test MarketData.get_stock_price() for an unknown symbol."""
    market_data = MarketData()
    price = market_data.get_stock_price('UNKNOWN')
    assert price is None, "Expected the stock price for an unknown symbol to be None"

def test_marketdata_get_economic_indicators():
    """Test MarketData.get_economic_indicators() returning correct keys and types."""
    market_data = MarketData()
    indicators = market_data.get_economic_indicators()
    assert isinstance(indicators, dict), "Expected economic indicators to be a dictionary"
    assert 'inflation_rate' in indicators, "Expected 'inflation_rate' key in indicators"
    assert isinstance(indicators['timestamp'], str), "Expected 'timestamp' to be a string"

@pytest.mark.parametrize("assets, liabilities, expected_current_ratio", [
    (100000.0, 50000.0, 2.0),  # typical case
    (50000.0, 50000.0, 1.0),   # edge case of equal assets and liabilities
    (0.0, 50000.0, 0.0)        # edge case of zero assets
])
def test_calculate_financial_ratios_current_ratio(financial_intelligence, assets, liabilities, expected_current_ratio):
    """Test calculation of current ratio in FinancialIntelligence.calculate_financial_ratios()."""
    balance_sheet = {'assets': assets, 'liabilities': liabilities, 'equity': 50000.0}
    ratios = financial_intelligence.calculate_financial_ratios(balance_sheet, mock_income_statement)
    assert ratios['current_ratio'] == expected_current_ratio, f"Expected current_ratio to be {expected_current_ratio}"

def test_analyze_financial_health_structure(financial_intelligence):
    """Test structure of analyze_financial_health() output."""
    assessment = financial_intelligence.analyze_financial_health()
    assert isinstance(assessment, dict), "Expected assessment to be a dictionary"
    assert 'timestamp' in assessment, "Expected 'timestamp' key in assessment"
    assert 'ratios' in assessment, "Expected 'ratios' key in assessment"
    assert 'health_score' in assessment, "Expected 'health_score' key in assessment"
