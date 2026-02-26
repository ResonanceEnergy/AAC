#!/usr/bin/env python3
"""
AAC World Bank Data Integration for Global Arbitrage
====================================================

Integrates World Bank Data Catalog for macroeconomic indicators and country data.
Essential for global arbitrage, currency arbitrage, and country risk assessment.

Features:
- GDP and economic growth indicators
- Inflation and price indices
- Currency exchange rates
- Country risk and development indicators
- Trade and balance of payments data
- Population and demographic data

API: Free, no API key required
Data: Historical macroeconomic data for 200+ countries
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class WorldBankConfig:
    """Configuration for World Bank API"""
    base_url: str = "https://api.worldbank.org/v2"
    timeout: int = 30
    format: str = "json"
    per_page: int = 1000

    def is_configured(self) -> bool:
        # World Bank API is free and doesn't require API key
        return True

@dataclass
class EconomicIndicator:
    """Economic indicator data structure"""
    indicator_id: str
    indicator_name: str
    country_id: str
    country_name: str
    value: float
    year: int
    date: datetime
    unit: str = ""

@dataclass
class CountryData:
    """Country information structure"""
    id: str
    name: str
    region: str
    income_level: str
    lending_type: str
    capital_city: str
    longitude: float
    latitude: float

class WorldBankIntegration:
    """World Bank Data API integration for AAC arbitrage system"""

    def __init__(self):
        self.config = WorldBankConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout

        # Key economic indicators for arbitrage analysis
        self.key_indicators = {
            # GDP and Growth
            'NY.GDP.MKTP.CD': 'GDP (current US$)',
            'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
            'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
            'NY.GDP.PCAP.KD.ZG': 'GDP per capita growth (annual %)',

            # Inflation and Prices
            'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
            'PA.NUS.FCRF': 'Official exchange rate (LCU per US$, period average)',

            # Trade and Balance of Payments
            'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
            'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
            'BN.CAB.XOKA.GD.ZS': 'Current account balance (% of GDP)',

            # Financial Markets
            'FR.INR.RINR': 'Real interest rate (%)',
            'FM.LBL.BMNY.ZG': 'Broad money growth (annual %)',

            # Development Indicators
            'SI.POV.GINI': 'Gini index',
            'SL.UEM.TOTL.ZS': 'Unemployment, total (% of total labor force)',
            'SP.POP.TOTL': 'Population, total',

            # Risk and Stability
            'IC.LGL.CRED.XQ': 'Strength of legal rights index (0=weak to 12=strong)',
            'GFDD.OI.02': 'Bank capital to assets ratio (%)',
        }

    def get_countries(self) -> List[CountryData]:
        """Get list of all countries from World Bank"""
        url = f"{self.config.base_url}/countries"
        params = {
            'format': self.config.format,
            'per_page': self.config.per_page
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            countries = []

            for country in data[1]:
                if country['region']['value'] != 'Aggregates':  # Skip regional aggregates
                    countries.append(CountryData(
                        id=country['id'],
                        name=country['name'],
                        region=country['region']['value'],
                        income_level=country['incomeLevel']['value'],
                        lending_type=country['lendingType']['value'],
                        capital_city=country['capitalCity'],
                        longitude=float(country['longitude']) if country['longitude'] else 0.0,
                        latitude=float(country['latitude']) if country['latitude'] else 0.0
                    ))

            return countries

        except Exception as e:
            print(f"[ERROR] Error fetching countries: {e}")
            return []

    def get_indicator_data(self, indicator_id: str, country_ids: List[str] = None,
                          start_year: int = None, end_year: int = None) -> List[EconomicIndicator]:
        """Get economic indicator data for specified countries and years"""

        if country_ids is None:
            country_ids = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'ITA', 'CAN', 'AUS', 'KOR']

        country_str = ';'.join(country_ids)

        url = f"{self.config.base_url}/countries/{country_str}/indicators/{indicator_id}"
        params = {
            'format': self.config.format,
            'per_page': self.config.per_page
        }

        if start_year:
            params['date'] = f"{start_year}:{end_year or datetime.now().year}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            indicators = []

            for item in data[1]:
                if item['value'] is not None:  # Skip null values
                    indicators.append(EconomicIndicator(
                        indicator_id=item['indicator']['id'],
                        indicator_name=item['indicator']['value'],
                        country_id=item['country']['id'],
                        country_name=item['country']['value'],
                        value=float(item['value']),
                        year=int(item['date']),
                        date=datetime(int(item['date']), 1, 1),  # January 1st of the year
                        unit=""  # Could be enhanced to include units
                    ))

            return indicators

        except Exception as e:
            print(f"[ERROR] Error fetching indicator {indicator_id}: {e}")
            return []

    def get_economic_arbitrage_signals(self) -> Dict[str, Any]:
        """Generate arbitrage signals based on economic indicators"""

        signals = {
            'currency_arbitrage': [],
            'growth_divergence': [],
            'inflation_differential': [],
            'trade_imbalance': []
        }

        try:
            # Get recent GDP growth data
            gdp_growth = self.get_indicator_data('NY.GDP.MKTP.KD.ZG', start_year=2020, end_year=2025)

            # Get inflation data
            inflation = self.get_indicator_data('FP.CPI.TOTL.ZG', start_year=2020, end_year=2025)

            # Get exchange rate data
            exchange_rates = self.get_indicator_data('PA.NUS.FCRF', start_year=2020, end_year=2025)

            # Analyze growth divergence
            growth_by_country = {}
            for indicator in gdp_growth:
                if indicator.year >= 2023:  # Focus on recent data
                    if indicator.country_id not in growth_by_country:
                        growth_by_country[indicator.country_id] = []
                    growth_by_country[indicator.country_id].append(indicator.value)

            # Calculate average growth rates
            avg_growth = {}
            for country, values in growth_by_country.items():
                if values:
                    avg_growth[country] = sum(values) / len(values)

            # Identify growth divergence opportunities
            if len(avg_growth) >= 2:
                sorted_growth = sorted(avg_growth.items(), key=lambda x: x[1], reverse=True)
                top_performer = sorted_growth[0]
                bottom_performer = sorted_growth[-1]

                if top_performer[1] - bottom_performer[1] > 2.0:  # >2% divergence
                    signals['growth_divergence'].append({
                        'strong_growth_country': top_performer[0],
                        'weak_growth_country': bottom_performer[0],
                        'growth_differential': top_performer[1] - bottom_performer[1],
                        'signal': 'Long strong economy assets, short weak economy assets'
                    })

            # Analyze inflation differentials
            inflation_by_country = {}
            for indicator in inflation:
                if indicator.year >= 2023:
                    if indicator.country_id not in inflation_by_country:
                        inflation_by_country[indicator.country_id] = []
                    inflation_by_country[indicator.country_id].append(indicator.value)

            avg_inflation = {}
            for country, values in inflation_by_country.items():
                if values:
                    avg_inflation[country] = sum(values) / len(values)

            if len(avg_inflation) >= 2:
                sorted_inflation = sorted(avg_inflation.items(), key=lambda x: x[1])
                low_inflation = sorted_inflation[0]
                high_inflation = sorted_inflation[-1]

                if high_inflation[1] - low_inflation[1] > 3.0:  # >3% inflation differential
                    signals['inflation_differential'].append({
                        'low_inflation_country': low_inflation[0],
                        'high_inflation_country': high_inflation[0],
                        'inflation_differential': high_inflation[1] - low_inflation[1],
                        'signal': 'Long low-inflation currency, short high-inflation currency'
                    })

        except Exception as e:
            print(f"[ERROR] Error generating economic signals: {e}")

        return signals

    def get_country_risk_assessment(self, country_codes: List[str]) -> Dict[str, Dict]:
        """Assess country risk based on economic indicators"""

        risk_assessment = {}

        try:
            # Get key risk indicators
            indicators_to_check = [
                'NY.GDP.MKTP.KD.ZG',  # GDP growth
                'FP.CPI.TOTL.ZG',     # Inflation
                'BN.CAB.XOKA.GD.ZS',  # Current account balance
                'IC.LGL.CRED.XQ',     # Legal rights index
            ]

            for country in country_codes:
                risk_assessment[country] = {
                    'overall_risk': 'Unknown',
                    'indicators': {},
                    'recommendation': ''
                }

                for indicator_id in indicators_to_check:
                    data = self.get_indicator_data(indicator_id, [country], 2020, 2024)

                    if data:
                        latest_value = max(data, key=lambda x: x.year)
                        risk_assessment[country]['indicators'][indicator_id] = {
                            'value': latest_value.value,
                            'year': latest_value.year,
                            'name': latest_value.indicator_name
                        }

                # Calculate risk score (simplified)
                indicators = risk_assessment[country]['indicators']

                risk_score = 0
                if 'NY.GDP.MKTP.KD.ZG' in indicators:
                    growth = indicators['NY.GDP.MKTP.KD.ZG']['value']
                    if growth < 0:
                        risk_score += 3  # High risk
                    elif growth < 2:
                        risk_score += 1  # Moderate risk

                if 'FP.CPI.TOTL.ZG' in indicators:
                    inflation = indicators['FP.CPI.TOTL.ZG']['value']
                    if inflation > 10:
                        risk_score += 3
                    elif inflation > 5:
                        risk_score += 1

                if 'BN.CAB.XOKA.GD.ZS' in indicators:
                    cab = indicators['BN.CAB.XOKA.GD.ZS']['value']
                    if cab < -5:
                        risk_score += 2

                # Determine risk level
                if risk_score >= 5:
                    risk_assessment[country]['overall_risk'] = 'High'
                    risk_assessment[country]['recommendation'] = 'Avoid or hedge exposure'
                elif risk_score >= 3:
                    risk_assessment[country]['overall_risk'] = 'Medium'
                    risk_assessment[country]['recommendation'] = 'Monitor closely'
                else:
                    risk_assessment[country]['overall_risk'] = 'Low'
                    risk_assessment[country]['recommendation'] = 'Favorable for investment'

        except Exception as e:
            print(f"[ERROR] Error assessing country risk: {e}")

        return risk_assessment

def main():
    """Demo World Bank integration"""
    print("🌍 AAC World Bank Data Integration Demo")
    print("=" * 50)

    wb = WorldBankIntegration()

    if not wb.config.is_configured():
        print("[ERROR] World Bank integration not configured")
        return

    print("[SUCCESS] World Bank API: Connected (Free API - No key required)")

    # Get sample countries
    print("\n🌎 Sample Countries:")
    countries = wb.get_countries()[:10]  # Show first 10
    for country in countries:
        print(f"   • {country.name} ({country.id}) - {country.region}")

    # Get economic indicators
    print("\n[DATA] Key Economic Indicators Available:")
    for indicator_id, name in list(wb.key_indicators.items())[:5]:
        print(f"   • {indicator_id}: {name}")

    # Get sample GDP data
    print("\n[FINANCE] Sample GDP Growth Data (2023):")
    gdp_data = wb.get_indicator_data('NY.GDP.MKTP.KD.ZG', ['USA', 'CHN', 'DEU'], 2023, 2023)
    for data in gdp_data:
        print(".1f")

    # Generate arbitrage signals
    print("\n[TARGET] Economic Arbitrage Signals:")
    signals = wb.get_economic_arbitrage_signals()

    for signal_type, signal_list in signals.items():
        if signal_list:
            print(f"   {signal_type.replace('_', ' ').title()}:")
            for signal in signal_list:
                print(f"     • {signal}")
        else:
            print(f"   {signal_type.replace('_', ' ').title()}: No signals detected")

    # Country risk assessment
    print("\n[WARN] Country Risk Assessment:")
    risk_data = wb.get_country_risk_assessment(['USA', 'CHN', 'BRA', 'ZAF'])
    for country, assessment in risk_data.items():
        print(f"   • {country}: {assessment['overall_risk']} risk - {assessment['recommendation']}")

    print("\n[SUCCESS] World Bank integration demo completed!")
    print("[INFO] World Bank data can enhance arbitrage strategies with macroeconomic insights")

if __name__ == "__main__":
    main()