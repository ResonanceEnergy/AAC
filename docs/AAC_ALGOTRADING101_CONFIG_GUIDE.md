# AAC AlgoTrading101 API Configuration Guide

## Overview

This guide helps you configure the AAC AlgoTrading101 Integration Hub with the necessary API keys to unlock full functionality. The integration leverages multiple professional trading platforms and data sources.

## Required API Keys

### 1. OpenBB Platform API Key
**Website**: https://openbb.co  
**Purpose**: Professional financial data and analytics platform  
**Cost**: Free tier available, premium plans for advanced features  

**Setup Steps**:
1. Visit https://openbb.co and create an account
2. Navigate to API Keys section in your dashboard
3. Generate a new API key
4. Add to `.env` file: `OPENBB_API_KEY=your_key_here`

### 2. QuiverQuant API Key
**Website**: https://quiverquant.com  
**Purpose**: Alternative data including WallStreetBets sentiment, congressional trading  
**Cost**: Freemium model with paid premium data  

**Setup Steps**:
1. Register at https://quiverquant.com
2. Verify your account to access API
3. Get your API token from the dashboard
4. Add to `.env` file: `QUIVERQUANT_API_KEY=your_token_here`

### 3. QuantConnect API Token
**Website**: https://quantconnect.com  
**Purpose**: Professional algorithmic trading platform with live execution  
**Cost**: Free for backtesting, paid for live trading  

**Setup Steps**:
1. Create account at https://quantconnect.com
2. Go to Account → API Tokens
3. Generate a new token
4. Add to `.env` file: `QUANTCONNECT_TOKEN=your_token_here`

### 4. Google Gemini API Key
**Website**: https://ai.google.dev  
**Purpose**: AI-powered financial analysis and market insights  
**Cost**: Free tier with rate limits  

**Setup Steps**:
1. Visit https://ai.google.dev and sign up
2. Create a new project or select existing
3. Enable Gemini API and generate API key
4. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

## Environment Configuration

### Creating the .env File

Create a file named `.env` in your project root directory:

```env
# OpenBB Platform
OPENBB_API_KEY=your_openbb_api_key_here

# QuiverQuant Alternative Data
QUIVERQUANT_API_KEY=your_quiverquant_api_key_here

# QuantConnect Live Trading
QUANTCONNECT_TOKEN=your_quantconnect_token_here

# Google Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here
```

### Security Best Practices

1. **Never commit .env files** to version control
2. **Use environment-specific keys** (dev/staging/prod)
3. **Rotate keys regularly** for security
4. **Monitor API usage** to avoid unexpected charges
5. **Store keys securely** with proper permissions

## Testing Configuration

After setting up your API keys, run the demo to verify everything works:

```bash
python aac_algotrading101_hub.py
```

You should see "✅ Configured" status for each API you've set up.

## API Usage Limits & Costs

### OpenBB Platform
- **Free Tier**: 1,000 requests/day, basic data
- **Premium**: $49/month for unlimited requests
- **Enterprise**: Custom pricing for high-volume users

### QuiverQuant
- **Free Tier**: Limited historical data, basic endpoints
- **Premium**: $99/month for full dataset access
- **Enterprise**: Custom pricing for hedge funds/institutions

### QuantConnect
- **Free Tier**: Unlimited backtesting, community algorithms
- **Premium**: $39/month for live trading (per algorithm)
- **Enterprise**: Custom pricing for institutions

### Google Gemini
- **Free Tier**: 60 requests/minute, 1,000 requests/day
- **Paid Tier**: Higher limits based on usage
- **Enterprise**: Custom pricing for high-volume AI usage

## Troubleshooting

### Common Issues

**"API key not configured"**
- Check that the key is in your `.env` file
- Verify the key name matches exactly (case-sensitive)
- Ensure no extra spaces or characters

**"Invalid API key"**
- Regenerate the key from the provider's dashboard
- Check for typos in the key value
- Verify the account is active and in good standing

**Rate limit exceeded**
- Wait for the rate limit window to reset
- Upgrade to a paid plan for higher limits
- Implement request throttling in your code

**Connection errors**
- Check your internet connection
- Verify the API endpoints haven't changed
- Ensure your firewall allows outbound HTTPS requests

### Getting Help

1. **Check the provider's documentation** for specific API issues
2. **Review AlgoTrading101 tutorials** for integration examples
3. **Test with the demo script** to isolate problems
4. **Contact AAC support** for integration-specific issues

## Advanced Configuration

### Custom API Endpoints

You can modify the API endpoints in the hub class for custom deployments:

```python
class AACAlgoTrading101Hub:
    def __init__(self):
        # Custom OpenBB endpoint
        self.openbb_base_url = "https://your-custom-openbb-instance.com/v1"
```

### Proxy Configuration

For corporate environments with proxy requirements:

```python
import os
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'
```

### Logging Configuration

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

Once configured, you can:

1. **Run comprehensive backtests** of your arbitrage strategies
2. **Access real-time market data** from OpenBB Platform
3. **Incorporate alternative data** from QuiverQuant
4. **Deploy live trading algorithms** via QuantConnect
5. **Get AI-powered market analysis** from Gemini

## Support Resources

- **AlgoTrading101**: https://algotrading101.com/learn/
- **OpenBB Docs**: https://docs.openbb.co/
- **QuantConnect Docs**: https://www.quantconnect.com/docs/
- **Gemini AI Docs**: https://ai.google.dev/docs

---

**Accelerated Arbitrage Corp** - Professional Algorithmic Trading Integration