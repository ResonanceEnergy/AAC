# aac069 — Reddit / DevVit Community Intelligence Module

## Purpose

`aac069` is a **TypeScript/Node.js** sub-project that powers the AAC Reddit community
intelligence layer. It integrates with **Reddit's DevVit platform** (Reddit's app
development framework) to:

- Monitor relevant subreddits (`r/wallstreetbets`, `r/stocks`, `r/SecurityAnalysis`,
  `r/investing`, `r/options`, `r/algotrading`, etc.) for sentiment signals
- Parse and surface high-velocity posts and comment threads in real-time
- Feed aggregated sentiment scores into the Python `shared/paper_trading.py` and
  `BigBrainIntelligence/agents.py` sentiment pipeline via Redis pub/sub

## Stack

| Component | Package | Purpose |
|---|---|---|
| Runtime | `ts-node` / Node.js 20+ | TypeScript execution |
| Bundler | `rollup` | Production bundle for dist/ |
| Reddit API | `@devvit/sdk` + `devvit-cli` | Reddit DevVit app integration |
| Caching | `redis-memory-server` (test) / Redis (prod) | Fast sentiment cache |
| Formatter | `prettier` | Code formatting |

## Setup

Requires Node.js ≥ 20 and npm ≥ 10.

```bash
cd aac/aac069
npm install
npm run build        # compiles TypeScript → dist/
npm run dev          # watch mode
```

For production deployment to Reddit DevVit:
```bash
npm run devvit:upload
npm run devvit:publish
```

## Redis Integration

The module writes sentiment data to Redis at:
- `aac:sentiment:{subreddit}:{ticker}` — rolling 1h sentiment score (-1 to +1)
- `aac:hot:{subreddit}` — top 10 hot posts (JSON array)
- `aac:signal:reddit` — latest aggregated signal (pub/sub channel)

Configure via `REDIS_URL` in the root `.env` file.

## Relationship to Python System

```
aac069 (Node/Reddit)
        │
        ▼ Redis pub/sub
shared/market_data_feeds.py   ← subscribes to aac:signal:reddit
        │
        ▼
BigBrainIntelligence/agents.py  ← consumes sentiment in decision logic
```

## Directory Layout

```
aac/aac069/
├── src/             ← TypeScript source (restore via git checkout if missing)
├── dist/            ← compiled JS output (gitignored — run npm run build)
├── node_modules/    ← npm packages (gitignored — run npm install)
└── package.json     ← project manifest
```

> Note: `src/` and `package.json` were present in the recovery_branch commit
> (`278ba8263`). If missing, restore with:
> ```bash
> git checkout 278ba8263 -- aac/aac069/src aac/aac069/package.json aac/aac069/tsconfig.json
> ```
