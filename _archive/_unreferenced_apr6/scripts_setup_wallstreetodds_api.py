#!/usr/bin/env python3
"""
AAC WallStreetOdds API Setup Script

This script helps you configure the WallStreetOdds API key for the AAC arbitrage system.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_wallstreetodds_api():
    """Setup WallStreetOdds API configuration"""

    logger.info("🔑 AAC WallStreetOdds API Setup")
    logger.info("=" * 40)

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        logger.info("📄 Creating .env file...")
        env_file.touch()

    # Read existing .env content
    existing_content = ""
    if env_file.exists():
        existing_content = env_file.read_text()

    # Check if WallStreetOdds key already exists
    if "WALLSTREETODDS_API_KEY" in existing_content:
        logger.info("⚠️  WallStreetOdds API key already configured!")
        overwrite = input("Do you want to update it? (y/N): ").lower().strip()
        if overwrite != 'y':
            logger.info("✅ Setup cancelled. Existing configuration preserved.")
            return

    # Get API key from user
    logger.info("\n📝 To get your WallStreetOdds API key:")
    logger.info("   1. Visit: https://wallstreetodds.com/register-api/")
    logger.info("   2. Create a free account")
    logger.info("   3. Generate an API key in your dashboard")
    logger.info("   4. Copy the key below")
    logger.info("")

    api_key = input("Enter your WallStreetOdds API key: ").strip()

    if not api_key:
        logger.info("❌ No API key provided. Setup cancelled.")
        return

    # Validate API key format (basic check)
    if len(api_key) < 10:
        logger.info("❌ API key seems too short. Please check and try again.")
        return

    # Update .env file
    lines = existing_content.split('\n') if existing_content else []

    # Remove existing WallStreetOdds key if present
    lines = [line for line in lines if not line.startswith('WALLSTREETODDS_API_KEY=')]

    # Add new key
    lines.append(f'WALLSTREETODDS_API_KEY={api_key}')

    # Remove empty lines at end
    while lines and lines[-1].strip() == '':
        lines.pop()

    # Write back to file
    new_content = '\n'.join(lines) + '\n'
    env_file.write_text(new_content)

    logger.info("✅ WallStreetOdds API key configured successfully!")
    logger.info("\n🧪 Testing configuration...")

    # Test the configuration
    try:
        from aac_wallstreetodds_integration import AACWallStreetOddsIntegration
        wso = AACWallStreetOddsIntegration()

        # Try a simple test
        if hasattr(wso, '_get_api_key') and wso._get_api_key():
            logger.info("✅ Configuration test passed!")
            logger.info("\n🚀 You can now use WallStreetOdds data in AAC arbitrage!")
            logger.info("   Run: python aac_wallstreetodds_integration.py")
        else:
            logger.info("❌ Configuration test failed. Please check your API key.")

    except Exception as e:
        logger.info(f"❌ Configuration test failed: {e}")
        logger.info("   Please verify your API key and try again.")

def main():
    """Main setup function"""
    try:
        setup_wallstreetodds_api()
    except KeyboardInterrupt:
        logger.info("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.info(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
