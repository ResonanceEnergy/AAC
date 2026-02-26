#!/usr/bin/env python3
"""
Reddit API Setup Guide for AAC Arbitrage System
===============================================

This script helps you set up Reddit API credentials for PRAW integration.
The PRAW library allows direct access to Reddit's API for sentiment analysis.

Steps to get Reddit API credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the details:
   - Name: AAC Arbitrage Bot
   - App type: script
   - Description: Automated arbitrage trading sentiment analysis
   - About URL: (leave blank)
   - Redirect URI: http://localhost:8080
5. After creating, you'll get:
   - client_id: The string under "name" (e.g., "abc123def456")
   - client_secret: The "secret" field

For authenticated access (recommended):
- Use your Reddit username and password
- This gives higher rate limits (600 requests per 10 minutes vs 60)

For read-only access:
- Leave username/password blank
- Limited to 60 requests per minute

API Documentation: See reddit_api_documentation.py for detailed endpoint specs
"""

import os
from dotenv import load_dotenv

def check_reddit_credentials():
    """Check current Reddit API configuration"""
    load_dotenv()

    credentials = {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': os.getenv('REDDIT_USER_AGENT'),
        'username': os.getenv('REDDIT_USERNAME'),
        'password': os.getenv('REDDIT_PASSWORD')
    }

    print("🔍 Current Reddit API Configuration:")
    print("-" * 40)

    configured = 0
    for key, value in credentials.items():
        status = "✅ Set" if value else "❌ Not set"
        # Mask sensitive values
        if value and key in ['client_secret', 'password']:
            display_value = value[:8] + "..." if len(value) > 8 else value
        elif value:
            display_value = value
        else:
            display_value = "None"

        print(f"   {key}: {status} ({display_value})")
        if value:
            configured += 1

    print(f"\n📊 Configuration Status: {configured}/5 credentials set")

    if configured >= 3:  # client_id, client_secret, user_agent are required
        print("✅ Basic configuration complete - you can use read-only access")
        if configured >= 5:
            print("✅ Full configuration complete - you can use authenticated access")
    else:
        print("❌ Insufficient configuration - please set at least client_id, client_secret, and user_agent")

    return credentials

def generate_reddit_app_guide():
    """Generate step-by-step guide for creating Reddit app"""
    print("\n📋 Reddit API Setup Guide:")
    print("=" * 50)
    print()
    print("1. 🌐 Go to Reddit Apps Page:")
    print("   https://www.reddit.com/prefs/apps")
    print()
    print("2. 🔐 Log in to your Reddit account")
    print()
    print("3. ➕ Create New App:")
    print("   - Click 'Create App' or 'Create Another App'")
    print("   - App type: script")
    print("   - Name: AAC Arbitrage Bot")
    print("   - Description: Automated arbitrage trading sentiment analysis")
    print("   - About URL: (leave blank)")
    print("   - Redirect URI: http://localhost:8080")
    print()
    print("4. 📝 Copy Credentials:")
    print("   - client_id: The string under the app name")
    print("   - client_secret: The 'secret' field")
    print()
    print("5. 🧪 Test Configuration:")
    print("   Run: python praw_reddit_integration.py")
    print()
    print("6. 🔑 Optional - Authenticated Access:")
    print("   For higher rate limits, also set:")
    print("   - username: Your Reddit username")
    print("   - password: Your Reddit password")
    print()
    print("⚠️  Security Note:")
    print("   Never share your Reddit credentials or commit them to version control!")
    print("   The .env file is already in .gitignore for your protection.")

if __name__ == "__main__":
    print("🚀 AAC Reddit API Setup Guide")
    print("=" * 40)

    check_reddit_credentials()
    generate_reddit_app_guide()

    print("\n🛠️  Next Steps:")
    print("1. Follow the guide above to create a Reddit app")
    print("2. Run: python configure_api_keys.py")
    print("3. Set your Reddit API credentials")
    print("4. Test with: python praw_reddit_integration.py")