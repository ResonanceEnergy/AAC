#!/usr/bin/env python3
"""
AAC Alpha Vantage API Test Script
Tests the Alpha Vantage API key configuration
"""

import os
import requests
from dotenv import load_dotenv

def test_alpha_vantage_api():
    """Test Alpha Vantage API connectivity"""

    # Load environment variables
    load_dotenv()

    # Get Alpha Vantage API key
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')

    if not api_key:
        print('❌ Alpha Vantage API key not found in .env file')
        return False

    print('🧪 Testing Alpha Vantage API Key')
    print('=' * 40)
    print(f'🔑 API Key: {api_key[:8]}...{api_key[-4:]}')  # Show partial key for security

    # Test with IBM stock quote
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}'

    try:
        print('📡 Making API request...')
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                symbol = quote.get('01. symbol', 'N/A')
                price = quote.get('05. price', 'N/A')
                change = quote.get('09. change', 'N/A')
                change_pct = quote.get('10. change percent', 'N/A')

                print('✅ Alpha Vantage API: Connected successfully!')
                print(f'📈 Symbol: {symbol}')
                print(f'💰 Price: ${price}')
                print(f'📊 Change: {change} ({change_pct})')
                return True
            else:
                print('⚠️  API connected but no data returned')
                print('Response keys:', list(data.keys()) if data else 'No data')
                if 'Error Message' in data:
                    print(f'Error: {data["Error Message"]}')
                if 'Note' in data:
                    print(f'Note: {data["Note"]}')
                return False
        else:
            print(f'❌ HTTP Error: {response.status_code}')
            print('Response:', response.text[:200])
            return False

    except Exception as e:
        print(f'❌ Connection Error: {e}')
        return False

if __name__ == '__main__':
    success = test_alpha_vantage_api()

    print()
    if success:
        print('🎯 API Key Status: Configured and Ready!')
        print('💡 You can now use Alpha Vantage for worldwide market data in AAC')
    else:
        print('❌ API Key Status: Configuration issue detected')
        print('💡 Check your API key and try again')