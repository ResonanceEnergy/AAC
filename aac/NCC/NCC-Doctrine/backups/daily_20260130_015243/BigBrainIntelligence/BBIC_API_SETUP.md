# 🚀 BBIC API Setup Guide - Get Your APIs Now!

## ✅ APIs Successfully Configured
- Environment loading: ✅ Working
- Libraries installed: ✅ All ready
- Scripts updated: ✅ Environment variables loaded
- Config files: ✅ Ready for real credentials

## 🔑 YouTube Data API v3 Setup

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3:
   - Go to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"

### Step 2: Create API Key
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the API key
4. (Optional) Restrict the key to YouTube Data API only

### Step 3: Update .env file
Replace `your_youtube_api_key_here` in `.env` with your actual API key:
```
YT_API_KEY=AIzaSyD_your_actual_api_key_here
```

## 🔑 Reddit API Setup

### Step 1: Create Reddit App
1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill out the form:
   - **Name:** BBIC Intelligence (or any name)
   - **App type:** script
   - **Description:** BBIC trend analysis for NCC
   - **About URL:** Leave blank
   - **Redirect URI:** http://localhost:8080

### Step 2: Get Credentials
After creating the app, you'll see:
- **client_id:** Under the app name (looks like: abc123def456)
- **client_secret:** The "secret" field

### Step 3: Update .env file
Replace the placeholders in `.env`:
```
REDDIT_CLIENT_ID=your_actual_client_id_here
REDDIT_CLIENT_SECRET=your_actual_client_secret_here
```

## 🧪 Test Your APIs

### Test YouTube API:
```bash
cd "c:\Users\gripa\OneDrive\Desktop\NCC\NCC-Doctrine"
python Scripts/bbic_youtube_ingest.py
```

### Test Reddit API:
```bash
cd "c:\Users\gripa\OneDrive\Desktop\NCC\NCC-Doctrine"
python Scripts/bbic_reddit_ingest.py
```

## 📊 Expected Results
Once APIs are configured, you should see:
- **YouTube:** Videos downloaded and curated trends file created
- **Reddit:** Top posts from 40+ subreddits saved daily
- **Data stored in:** `data/bbic/` directory

## 🎯 Current Configuration
- **YouTube Channels:** MrBeast (example) - update with energy/policy channels
- **YouTube Keywords:** Energy policy, hydropower, grid analysis
- **Reddit Subreddits:** 40+ focused on energy, tech, finance, science, and industry
  - Energy & Environment (9 subs)
  - Technology & Innovation (9 subs)
  - Finance & Business (9 subs)
  - Science & Research (6 subs)
  - Industry Specific (7 subs)

## 🚨 Important Notes
- **YouTube API:** Has daily quota limits (10,000 units/day free tier)
- **Reddit API:** Rate limited but generous for script apps
- **Security:** Keep .env file secure and never commit API keys to git

Ready to activate BBIC trend tracking! 🎉