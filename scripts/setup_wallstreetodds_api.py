#!/usr/bin/env python3
"""
AAC WallStreetOdds API Setup Script

This script helps you configure the WallStreetOdds API key for the AAC arbitrage system.
"""

import os
import sys
from pathlib import Path

def setup_wallstreetodds_api():
    """Setup WallStreetOdds API configuration"""

    print("🔑 AAC WallStreetOdds API Setup")
    print("=" * 40)

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("📄 Creating .env file...")
        env_file.touch()

    # Read existing .env content
    existing_content = ""
    if env_file.exists():
        existing_content = env_file.read_text()

    # Check if WallStreetOdds key already exists
    if "WALLSTREETODDS_API_KEY" in existing_content:
        print("⚠️  WallStreetOdds API key already configured!")
        overwrite = input("Do you want to update it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("✅ Setup cancelled. Existing configuration preserved.")
            return

    # Get API key from user
    print("\n📝 To get your WallStreetOdds API key:")
    print("   1. Visit: https://wallstreetodds.com/register-api/")
    print("   2. Create a free account")
    print("   3. Generate an API key in your dashboard")
    print("   4. Copy the key below")
    print()

    api_key = input("Enter your WallStreetOdds API key: ").strip()

    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return

    # Validate API key format (basic check)
    if len(api_key) < 10:
        print("❌ API key seems too short. Please check and try again.")
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

    print("✅ WallStreetOdds API key configured successfully!")
    print("\n🧪 Testing configuration...")

    # Test the configuration
    try:
        from aac_wallstreetodds_integration import AACWallStreetOddsIntegration
        wso = AACWallStreetOddsIntegration()

        # Try a simple test
        if hasattr(wso, '_get_api_key') and wso._get_api_key():
            print("✅ Configuration test passed!")
            print("\n🚀 You can now use WallStreetOdds data in AAC arbitrage!")
            print("   Run: python aac_wallstreetodds_integration.py")
        else:
            print("❌ Configuration test failed. Please check your API key.")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("   Please verify your API key and try again.")

def main():
    """Main setup function"""
    try:
        setup_wallstreetodds_api()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()