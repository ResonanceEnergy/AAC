#!/usr/bin/env python3
"""
PHASE 2: API INTEGRATION GUIDE & VALIDATION
===========================================
Complete guide for configuring and validating all critical APIs for live trading.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api_integration_hub import api_integration_hub

from shared.config_loader import get_config


class APIIntegrationGuide:
    """Guide for Phase 2 API integration"""

    def __init__(self):
        self.config = get_config()

    def get_api_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get all required APIs with setup instructions"""
        return {
            "ETH_PRIVATE_KEY": {
                "priority": "CRITICAL",
                "purpose": "Ethereum DEX trading (Uniswap, SushiSwap, etc.)",
                "setup_steps": [
                    "1. Create Ethereum wallet (MetaMask recommended)",
                    "2. Export private key (NEVER share with anyone)",
                    "3. Fund wallet with ETH for gas fees",
                    "4. Set ETH_RPC_URL to Infura/Alchemy endpoint"
                ],
                "env_vars": ["ETH_PRIVATE_KEY", "ETH_RPC_URL"],
                "configured": bool(self.config.eth_private_key),
                "docs": "https://metamask.io/"
            },
            "BIGBRAIN_AUTH_TOKEN": {
                "priority": "CRITICAL",
                "purpose": "AI-powered trading signals and market analysis",
                "setup_steps": [
                    "1. Register at BigBrain Intelligence platform",
                    "2. Generate API token from dashboard",
                    "3. Set BIGBRAIN_API_URL to service endpoint",
                    "4. Configure BIGBRAIN_AUTH_TOKEN"
                ],
                "env_vars": ["BIGBRAIN_AUTH_TOKEN", "BIGBRAIN_API_URL"],
                "configured": bool(self.config.bigbrain_token),
                "docs": "Contact BigBrain Intelligence team"
            },
            "COINMARKETCAP_API_KEY": {
                "priority": "HIGH",
                "purpose": "Complete cryptocurrency market data",
                "setup_steps": [
                    "1. Sign up at coinmarketcap.com",
                    "2. Navigate to API dashboard",
                    "3. Generate API key (free tier available)",
                    "4. Set COINMARKETCAP_API_KEY"
                ],
                "env_vars": ["COINMARKETCAP_API_KEY"],
                "configured": bool(self.config.coinmarketcap_key),
                "docs": "https://coinmarketcap.com/api/"
            },
            "NEWS_API_KEY": {
                "priority": "HIGH",
                "purpose": "Financial news sentiment analysis",
                "setup_steps": [
                    "1. Register at newsapi.org",
                    "2. Get API key from dashboard",
                    "3. Set NEWS_API_KEY"
                ],
                "env_vars": ["NEWS_API_KEY"],
                "configured": bool(self.config.news_api_key),
                "docs": "https://newsapi.org/"
            },
            "TWITTER_BEARER_TOKEN": {
                "priority": "MEDIUM",
                "purpose": "Social sentiment analysis from Twitter",
                "setup_steps": [
                    "1. Apply for Twitter Developer Account",
                    "2. Create app in Twitter Developer Portal",
                    "3. Generate Bearer Token",
                    "4. Set TWITTER_BEARER_TOKEN"
                ],
                "env_vars": ["TWITTER_BEARER_TOKEN"],
                "configured": bool(self.config.twitter_bearer),
                "docs": "https://developer.twitter.com/"
            },
            "REDDIT_API": {
                "priority": "MEDIUM",
                "purpose": "Social sentiment analysis from Reddit",
                "setup_steps": [
                    "1. Create Reddit app at reddit.com/prefs/apps",
                    "2. Get client_id and client_secret",
                    "3. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET",
                    "4. Set REDDIT_USER_AGENT"
                ],
                "env_vars": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"],
                "configured": bool(self.config.reddit_client_id and self.config.reddit_client_secret),
                "docs": "https://reddit.com/prefs/apps"
            },
            "NCC_AUTH_TOKEN": {
                "priority": "HIGH",
                "purpose": "Neural Coordination Center for agent orchestration",
                "setup_steps": [
                    "1. Deploy NCC Coordinator service",
                    "2. Generate authentication token",
                    "3. Set NCC_COORDINATOR_ENDPOINT and NCC_AUTH_TOKEN"
                ],
                "env_vars": ["NCC_COORDINATOR_ENDPOINT", "NCC_AUTH_TOKEN"],
                "configured": bool(self.config.ncc_token),
                "docs": "Internal NCC documentation"
            },
            "KYC_PROVIDER_API_KEY": {
                "priority": "MEDIUM",
                "purpose": "Identity verification for compliance",
                "setup_steps": [
                    "1. Choose KYC provider (Sumsub, Veriff, etc.)",
                    "2. Register and get API credentials",
                    "3. Set KYC_PROVIDER_API_KEY and KYC_PROVIDER_URL"
                ],
                "env_vars": ["KYC_PROVIDER_API_KEY", "KYC_PROVIDER_URL"],
                "configured": bool(self.config.kyc_provider_key),
                "docs": "Provider-specific documentation"
            }
        }

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get overall API configuration status"""
        requirements = self.get_api_requirements()

        critical_apis = [api for api, info in requirements.items() if info["priority"] == "CRITICAL"]
        high_apis = [api for api, info in requirements.items() if info["priority"] == "HIGH"]
        medium_apis = [api for api, info in requirements.items() if info["priority"] == "MEDIUM"]

        critical_configured = sum(1 for api in critical_apis if requirements[api]["configured"])
        high_configured = sum(1 for api in high_apis if requirements[api]["configured"])
        medium_configured = sum(1 for api in medium_apis if requirements[api]["configured"])

        total_configured = sum(1 for info in requirements.values() if info["configured"])
        total_apis = len(requirements)

        return {
            "total_apis": total_apis,
            "configured_apis": total_configured,
            "configuration_rate": (total_configured / total_apis * 100),
            "critical_apis": {
                "total": len(critical_apis),
                "configured": critical_configured,
                "rate": (critical_configured / len(critical_apis) * 100) if critical_apis else 0
            },
            "high_apis": {
                "total": len(high_apis),
                "configured": high_configured,
                "rate": (high_configured / len(high_apis) * 100) if high_apis else 0
            },
            "medium_apis": {
                "total": len(medium_apis),
                "configured": medium_configured,
                "rate": (medium_configured / len(medium_apis) * 100) if medium_apis else 0
            },
            "api_details": requirements
        }

    async def validate_api_integrations(self) -> Dict[str, Any]:
        """Validate all configured API integrations"""
        logger.info("🔗 Validating API Integrations...")
        logger.info("=" * 50)

        # Get system status from integration hub
        system_status = await api_integration_hub.get_system_status()

        # Get configuration status
        config_status = self.get_configuration_status()

        logger.info("📊 Configuration Status:")
        logger.info(f"  Total APIs: {config_status['total_apis']}")
        logger.info(f"  Configured: {config_status['configured_apis']}")
        logger.info(f"  Configuration Rate: {config_status['configuration_rate']:.1f}%")
        logger.info(f"  Critical APIs: {config_status['critical_apis']['configured']}/{config_status['critical_apis']['total']} ({config_status['critical_apis']['rate']:.1f}%)")
        logger.info(f"  High Priority: {config_status['high_apis']['configured']}/{config_status['high_apis']['total']} ({config_status['high_apis']['rate']:.1f}%)")
        logger.info(f"  Medium Priority: {config_status['medium_apis']['configured']}/{config_status['medium_apis']['total']} ({config_status['medium_apis']['rate']:.1f}%)")
        logger.info("")

        logger.info("🧪 Connection Test Results:")
        logger.info(f"  Tests Run: {system_status['connection_tests_run']}")
        logger.info(f"  Successful: {system_status['successful_connections']}")
        logger.info(f"  Success Rate: {system_status['connection_success_rate']:.1f}%")
        logger.info("")

        # Show detailed results
        if system_status['connection_details']:
            logger.info("🔧 API Connection Details:")
            for api_name, details in system_status['connection_details'].items():
                status_icon = "✅" if details['success'] else "❌"
                config_status = config_status['api_details'].get(api_name.replace('_', '').upper(), {})
                priority = config_status.get('priority', 'UNKNOWN') if config_status else 'UNKNOWN'

                logger.info(f"  {status_icon} {api_name} ({priority}): {details['response_time']:.2f}s")
                if not details['success'] and details.get('error'):
                    logger.info(f"    Error: {details['error']}")

        logger.info("")

        # Phase 2 readiness assessment
        critical_ready = config_status['critical_apis']['rate'] >= 100
        high_ready = config_status['high_apis']['rate'] >= 80
        connection_ready = system_status['connection_success_rate'] >= 80

        phase2_ready = critical_ready and high_ready and connection_ready

        logger.info("🎯 PHASE 2 READINESS ASSESSMENT:")
        logger.info("=" * 50)

        if phase2_ready:
            logger.info("🎉 PHASE 2: API INTEGRATION COMPLETE!")
            logger.info("✅ All critical APIs configured and tested")
            logger.info("✅ Ready for live trading infrastructure")
            logger.info("")
            logger.info("🚀 NEXT: Phase 2 Priority 4 - Live Trading Infrastructure")
        else:
            logger.info("⚠️  PHASE 2: INCOMPLETE")
            logger.info("❌ Missing critical API configurations")
            logger.info("")
            logger.info("🔧 REQUIRED ACTIONS:")

            # Show missing critical APIs
            requirements = config_status['api_details']
            missing_critical = [
                api for api, info in requirements.items()
                if info['priority'] == 'CRITICAL' and not info['configured']
            ]

            if missing_critical:
                logger.info("Critical APIs (REQUIRED):")
                for api in missing_critical:
                    info = requirements[api]
                    logger.info(f"  • {api}: {info['purpose']}")
                    logger.info(f"    Setup: {info['docs']}")

            # Show missing high priority APIs
            missing_high = [
                api for api, info in requirements.items()
                if info['priority'] == 'HIGH' and not info['configured']
            ]

            if missing_high:
                logger.info("High Priority APIs (RECOMMENDED):")
                for api in missing_high:
                    info = requirements[api]
                    logger.info(f"  • {api}: {info['purpose']}")

        logger.info("")
        logger.info("=" * 50)

        return {
            "phase2_ready": phase2_ready,
            "config_status": config_status,
            "connection_status": system_status,
            "critical_ready": critical_ready,
            "high_ready": high_ready,
            "connection_ready": connection_ready
        }


async def show_api_setup_guide():
    """Display comprehensive API setup guide"""
    guide = APIIntegrationGuide()

    logger.info("🔗 PHASE 2: API INTEGRATION SETUP GUIDE")
    logger.info("=" * 60)
    logger.info("Complete guide for configuring all critical APIs for live trading")
    logger.info("")

    requirements = guide.get_api_requirements()

    # Group by priority
    priorities = ["CRITICAL", "HIGH", "MEDIUM"]

    for priority in priorities:
        apis = {name: info for name, info in requirements.items() if info["priority"] == priority}

        if not apis:
            continue

        logger.info(f"🚨 {priority} PRIORITY APIs:")
        logger.info("-" * 40)

        for api_name, info in apis.items():
            configured = "✅ CONFIGURED" if info["configured"] else "❌ NOT CONFIGURED"
            logger.info(f"\n🔧 {api_name} - {configured}")
            logger.info(f"   Purpose: {info['purpose']}")
            logger.info(f"   Documentation: {info['docs']}")
            logger.info("   Setup Steps:")
            for step in info['setup_steps']:
                logger.info(f"     {step}")
            logger.info(f"   Environment Variables: {', '.join(info['env_vars'])}")

        logger.info("")

    logger.info("📝 CONFIGURATION INSTRUCTIONS:")
    logger.info("-" * 40)
    logger.info("1. Copy .env.example to .env")
    logger.info("2. Fill in API keys for required services")
    logger.info("3. Test configuration with: python phase2_api_validation.py")
    logger.info("4. Ensure all CRITICAL APIs are configured before live trading")
    logger.info("")

    # Show current status
    status = guide.get_configuration_status()
    logger.info("📊 CURRENT CONFIGURATION STATUS:")
    logger.info(f"   Critical APIs: {status['critical_apis']['configured']}/{status['critical_apis']['total']} ({status['critical_apis']['rate']:.1f}%)")
    logger.info(f"   High Priority: {status['high_apis']['configured']}/{status['high_apis']['total']}")
    logger.info(f"   Medium Priority: {status['medium_apis']['configured']}/{status['medium_apis']['total']}")

    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "guide":
        asyncio.run(show_api_setup_guide())
    else:
        guide = APIIntegrationGuide()
        asyncio.run(guide.validate_api_integrations())
