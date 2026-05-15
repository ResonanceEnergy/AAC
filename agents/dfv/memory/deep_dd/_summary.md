# DFV Deep DD — Summary Report

**Symbols covered:** 15 | **Source:** xAI Grok-4 (web_search + x_search) + GBM Monte Carlo (20k paths)

## Quick-Look Table

| Sym | Spot | σ30d | Verdict | Conv | P(profit) | MC PnL p50 | Max Loss | Events | Cites | Next Event |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| **SLV** | $69.04 | 53.8% | `hold` | 3 | 39% | $-9,481 | $9,481 | 10 | 52 | 2026-05-13 April 2026 CPI at 3.8% (hotter than expected) |
| **HYG** | $79.46 | 10.2% | `hold` | 2 | 0% | $-8,005 | $8,005 | 8 | 40 | 2026-06-01 ISM Manufacturing PMI |
| **LQD** | $107.86 | 9.2% | `hold` | 2 | 0% | $-6,305 | $6,305 | 9 | 30 | 2026-05-15 Empire State Manufacturing Index May: 19.6 vs 7.0 |
| **XRP** | $16.07 | 42.8% | `hold` | 4 | 100% | $4,808 | $574 | 7 | 44 | 2026-05-14 Senate Banking Committee markup on CLARITY Act (c |
| **EMB** | $94.71 | 15.1% | `trim` | 2 | 0% | $-4,805 | $4,805 | 5 | 26 | 2026-05-01 EMB ex-dividend date for May 2026 distribution $0 |
| **TSLA** | $422.24 | 41.2% | `hold` | 3 | 24% | $-2,042 | $2,042 | 10 | 43 | 2026-07-22 Q2 2026 Earnings Release |
| **OBDC** | $11.21 | 26.4% | `hold` | 3 | 11% | $-1,270 | $1,270 | 9 | 35 | 2026-05-06 Q1 2026 earnings release: EPS $0.31 miss vs $0.35 |
| **XRPT** | $44.07 | 85.6% | `hold` | 3 | 11% | $-1,050 | $1,050 | 10 | 29 | 2026-05-16 CLARITY Act Senate Banking Committee markup sessi |
| **XLE** | $59.44 | 26.5% | `hold` | 3 | 34% | $-806 | $806 | 8 | 43 | 2026-06-07 41st OPEC+ Ministerial Meeting |
| **JNK** | $95.72 | 8.6% | `hold` | 2 | 12% | $-390 | $390 | 7 | 37 | 2026-05-21 Weekly initial jobless claims |
| **ARCC** | $18.90 | 19.4% | `hold` | 3 | 0% | $-340 | $340 | 8 | 29 | 2026-05-11 Ares Capital announces $800M note offering |
| **BKLN** | $20.61 | 31.6% | `hold` | 3 | 30% | $-119 | $119 | 8 | 43 | 2026-06-01 ISM Manufacturing PMI release |
| **KRE** | $66.97 | 20.4% | `hold` | 3 | 0% | $-80 | $80 | 5 | 30 | 2026-03-17 to 2026-03-18 FOMC rate decision hold at 3.5-3.75 |
| **ETH** | $21.10 | 37.5% | `hold` | 4 | 100% | $7 | ? | 10 | 24 | 2026-06 Glamsterdam hard fork upgrade (EIP-7732 ePBS, EIP-79 |
| **OWL** | $9.46 | 57.2% | `hold` | 2 | 12% | $0 | ? | 9 | 51 | 2026-06-03 Blue Owl Capital Annual General Meeting of Shareh |

## Per-Symbol Detail

### SLV  —  verdict: **hold** (conv 3)

- **Spot/σ:** $69.04 / σ=53.8%
- **Legs:** C66.0 x+8 (20260618); C70.0 x+2 (20260618); C75.0 x+2 (20270115); C65.0 x+3 (20270115); C67.5 x+1 (20260918); C70.0 x+1 (20260618)
- **MC:** P(profit)=0.3899, PnL p50=$-9,481, max-loss=$9,481
- **Research:** 10 events, 7 leading indicators, 52 citations
- **Thesis:** Long SLV calls positioned for 2026-12-31 supply deficit confirmation and 2026-Q2/Q3 Silver Institute solar/EV releases to drive physical squeeze. MC 20000 paths show median S_T 64.51 and P&L -9481 but
- **Exit ladder:** PT1=spot >86.87 (MC P75) peel 40% of position | PT2=spot >110 peel remaining 40% | Stop=spot <48.03 close 100% of position | Time-stop=close all at 30 DTE if not 25% ITM
- **Alignment check:** Thesis uses 2026-12-31 deficit event plus MC P95 134.90 tail and 0.3899 P(profit) with COMEX tightness and RV 0.539

### HYG  —  verdict: **hold** (conv 2)

- **Spot/σ:** $79.46 / σ=10.2%
- **Legs:** P77.0 x+1 (20260618)
- **MC:** P(profit)=0.0, PnL p50=$-8,005, max-loss=$8,005
- **Research:** 8 events, 6 leading indicators, 40 citations
- **Thesis:** Long 77-put benefits from June cluster: CPI 6/10, NFP 6/5, FOMC 6/16-17 and PPI 6/11 can force vol spike above current 0.102 RV. MC 20k paths give P05=75.79 (only 4.2% below spot) yet P(profit)=0 and 
- **Exit ladder:** PT1=put >=1.80, peel 50% | PT2=put >=3.20, peel remaining 50% | Stop=HYG >82.50 or premium <0.15, close 100% | Time-stop=close at 7 DTE if not ITM
- **Alignment check:** Thesis built on FOMC/CPI/NFP events plus MC P05=75.79 and P(profit)=0 showing asymmetric cheap tail exposure.

### LQD  —  verdict: **hold** (conv 2)

- **Spot/σ:** $107.86 / σ=9.2%
- **Legs:** P106.0 x+1 (20260515)
- **MC:** P(profit)=0.0, PnL p50=$-6,305, max-loss=$6,305
- **Research:** 9 events, 6 leading indicators, 30 citations
- **Thesis:** Long LQD 106 put exp 20260515. MC 20000 paths GBM shows median S_T 107.8711 (P05 107.02) and P(profit)=0 with uniform -6305 P&L, yet 2026-05-15 high-impact bearish prints (IP +0.7% beat, Empire 19.6 b
- **Exit ladder:** PT1=LQD < 106.50 at 10:00 ET 2026-05-15, close 100% | PT2=none | Stop=LQD > 108.75, close 100% | Time-stop=close at 16:00 ET 2026-05-15 if not ITM
- **Alignment check:** Thesis built from 2026-05-15 IP/Empire beats + MC median S_T 107.8711 and P05 107.02 showing no intrinsic value without event tail

### XRP  —  verdict: **hold** (conv 4)

- **Spot/σ:** $16.07 / σ=42.8%
- **Legs:** C19.0 x+5 (20260618); C17.0 x+15 (20260821); None x+339 ()
- **MC:** P(profit)=1.0, PnL p50=$4,808, max-loss=$574
- **Research:** 7 events, 6 leading indicators, 44 citations
- **Thesis:** P(profit)=1.0 and median P&L 4808 (P05 3166/P95 18186) from 20k GBM paths support holding the 19/17 calls + equity into 2026-08-21 expiry. High/bullish CLARITY Act markup 2026-05-14, FOMC cuts 2026-07
- **Exit ladder:** PT1=spot >18.42 (MC P75) peel 40% of calls | PT2=spot >22.98 (MC P95) peel remaining 60% of calls | Stop=spot <11.02 (MC P05) close entire position | Time-stop=close all at 2026-08-21 expiry if not 25% ITM
- **Alignment check:** Thesis built on MC median P&L 4808/P95 18186 plus high/bullish events CLARITY Act May-14, FOMC July-29 and ETF inflows validating upside to 22.98.

### EMB  —  verdict: **trim** (conv 2)

- **Spot/σ:** $94.71 / σ=15.1%
- **Legs:** P90.0 x+1 (20260515)
- **MC:** P(profit)=0.0, PnL p50=$-4,805, max-loss=$4,805
- **Research:** 5 events, 5 leading indicators, 26 citations
- **Thesis:** MC 20000-path GBM locks 100% loss at median P&L -4805 with P(profit)=0.0 and all quantiles identical, confirming full decay of the 90-put at 1 DTE. 2026-05-11 macro triggers (10Y>4.5%, Fed shift) plus
- **Exit ladder:** PT1=none - position OTM | PT2=none - position OTM | Stop=market sell at 0.05 or lower, close full size | Time-stop=close at expiry open if not ITM
- **Alignment check:** Used 2026-05-11 macro triggers + 2026-05-15 expiry with MC median P&L -4805 and P(profit)=0.0 to justify trim

### TSLA  —  verdict: **hold** (conv 3)

- **Spot/σ:** $422.24 / σ=41.2%
- **Legs:** C500.0 x+1 (20270115)
- **MC:** P(profit)=0.2429, PnL p50=$-2,042, max-loss=$2,042
- **Research:** 10 events, 6 leading indicators, 43 citations
- **Thesis:** Long 500C 20270115 remains aligned: MC shows 24.29% P(profit) with P95 P&L +20183.8 versus median -2041.53 loss, capturing tail from 2026-04-01 Cybercab ramp at Giga Texas, 2026-06-30 Robotaxi expansi
- **Exit ladder:** PT1=S_T>550, peel 40% | PT2=S_T>650, peel remaining 60% | Stop=spot<350 for 3 sessions, close full position | Time-stop=close at 90 DTE if not ITM
- **Alignment check:** Thesis built from 2026-04-01/06-30/01-01 events plus MC P(profit) 0.2429 and P95 P&L +20183.8 for asymmetric upside.

### OBDC  —  verdict: **hold** (conv 3)

- **Spot/σ:** $11.21 / σ=26.4%
- **Legs:** P7.5 x+11 (20260717); P10.0 x+65 (20260417); P7.5 x+10 (20260717)
- **MC:** P(profit)=0.1086, PnL p50=$-1,270, max-loss=$1,270
- **Research:** 9 events, 6 leading indicators, 35 citations
- **Thesis:** Long OBDC puts aligned to 22% NAV discount at 11.21 vs 14.41 post-2026-05-06 EPS miss and dividend cut. MC 20000 paths show P95 P&L +2773.21 despite median -1269.7 and P(profit) 0.1086, capturing tail
- **Exit ladder:** PT1=S_T <10.43 peel 40% at P25 MC level | PT2=S_T <9.38 peel remaining 60% at P05 MC level | Stop=hard stop at -100% premium loss, close all | Time-stop=close at 21 DTE if not 25% ITM
- **Alignment check:** Thesis uses 2026-05-06 earnings miss event plus MC P95 P&L 2773.21 and P(profit) 0.1086 keyed to NAV discount support.

### XRPT  —  verdict: **hold** (conv 3)

- **Spot/σ:** $44.07 / σ=85.6%
- **Legs:** C49.0 x+1 (20260618)
- **MC:** P(profit)=0.1072, PnL p50=$-1,050, max-loss=$1,050
- **Research:** 10 events, 6 leading indicators, 29 citations
- **Thesis:** Hold XRPT 49C into 2026-06-18 expiry. CLARITY Act Senate markup 2026-05-16 and floor vote window 2026-06-05 plus ETF inflow reports 2026-05-20/06-02 provide high-bullish catalysts that align with MC P
- **Exit ladder:** PT1=S_T>58, peel 50% | PT2=S_T>66, peel remaining 50% | Stop=spot<38 close full position | Time-stop=close at 7 DTE if still OTM
- **Alignment check:** Thesis built from 2026-05-16/06-05 CLARITY events plus MC P95 tail (+666 P&L) and 0.856 vol for event-driven upside.

### XLE  —  verdict: **hold** (conv 3)

- **Spot/σ:** $59.44 / σ=26.5%
- **Legs:** C65.0 x+3 (20260618); C65.0 x+3 (20270115); C60.0 x+1 (20260618)
- **MC:** P(profit)=0.34, PnL p50=$-806, max-loss=$806
- **Research:** 8 events, 6 leading indicators, 43 citations
- **Thesis:** Position held for 2027 LEAP gamma. MC shows 0.34 P(profit) with P95 P&L +14376 vs median -806, driven by right skew above spot 59.44. Key support from 2026-07-01 Q2 XOM/CVX earnings (high/bullish), 20
- **Exit ladder:** PT1=XLE 70, peel 1/3 | PT2=XLE 80, peel remaining 1/3 | Stop=XLE 50, close all | Time-stop=close at 30 DTE if not 25% ITM
- **Alignment check:** Q2 2026 earnings + OPEC compliance events plus MC P95 tail of 14376 used to justify hold.

### JNK  —  verdict: **hold** (conv 2)

- **Spot/σ:** $95.72 / σ=8.6%
- **Legs:** P94.0 x+5 (20260417); P92.0 x+2 (20260618)
- **MC:** P(profit)=0.1239, PnL p50=$-390, max-loss=$390
- **Research:** 7 events, 5 leading indicators, 37 citations
- **Thesis:** Long JNK 94/92 put structure positioned for spread widening into 2026-06-17 FOMC decision and 2026-06-05 Employment Situation print. MC 20000-path GBM shows S_T P05 at 92.03 with P&L P95 of +595.94 ve
- **Exit ladder:** PT1=JNK <= 92.50 peel 50% of 94-strike leg | PT2=JNK <= 91.00 peel remaining 50% of 94-strike leg | Stop=hard stop if realized loss exceeds 390 USD close entire position | Time-stop=close all at 21 DTE if not 25% ITM
- **Alignment check:** Thesis uses 2026-06-17 FOMC and 2026-06-05 Employment events plus MC S_T P05=92.03 and 12.39% P(profit) to justify holding the put structure.

### ARCC  —  verdict: **hold** (conv 3)

- **Spot/σ:** $18.90 / σ=19.4%
- **Legs:** P16.0 x+10 (20260417); P15.0 x+5 (20260618)
- **MC:** P(profit)=0.0006, PnL p50=$-340, max-loss=$340
- **Research:** 8 events, 7 leading indicators, 29 citations
- **Thesis:** Long 10x 16P Apr26 + 5x 15P Jun26 at total debit 340 USD functions as cheap tail protection. MC 20k paths give median S_T 18.9441, P05 17.1861 and P(profit) 0.0006, confirming low premium cost while s
- **Exit ladder:** PT1=S_T <=17.20, peel 50% of contracts | PT2=S_T <=16.50, peel remaining 50% | Stop=ARCC >20.50 close 100% | Time-stop=close at 21 DTE if not 25% ITM
- **Alignment check:** Thesis uses MC median S_T 18.9441 / P05 17.1861 plus 15-Jun dividend ex-date and 28-May Fed window to justify holding the OTM put ladder as low-cost downside buffer.

### BKLN  —  verdict: **hold** (conv 3)

- **Spot/σ:** $20.61 / σ=31.6%
- **Legs:** P20.0 x+3 (20260618)
- **MC:** P(profit)=0.3016, PnL p50=$-119, max-loss=$119
- **Research:** 8 events, 6 leading indicators, 43 citations
- **Thesis:** Long 3x 20-strike Jun18 puts positioned for vol spike and downside from 17 Jun FOMC decision plus 5 Jun NFP/10 Jun CPI cluster. MC 20k GBM paths give P(profit)=0.3016 with P95 P&L +607.95 vs median -1
- **Exit ladder:** PT1=P&L +200, peel 1 contract | PT2=P&L +450, peel 1 contract | Stop=BKLN >22.50 or premium <0.05, close all | Time-stop=close at 5 DTE if not 25% ITM
- **Alignment check:** Thesis built on 17 Jun FOMC + 5/10 Jun data releases plus MC P(profit)=0.3016 and P95 P&L +607.95 tail

### KRE  —  verdict: **hold** (conv 3)

- **Spot/σ:** $66.97 / σ=20.4%
- **Legs:** P60.0 x+1 (20260417)
- **MC:** P(profit)=0.0, PnL p50=$-80, max-loss=$80
- **Research:** 5 events, 5 leading indicators, 30 citations
- **Thesis:** Hold KRE 60P 20260417 as tail-risk hedge. MC 20k GBM paths show median S_T 66.97 with P(profit)=0.0 and uniform -80 P&L, confirming current OTM status yet validating static protection value below 68.1
- **Exit ladder:** PT1=KRE <62.00 peel 50% | PT2=KRE <60.50 peel remaining 50% | Stop=KRE >68.18 close full position | Time-stop=close at 30 DTE if not 25% ITM
- **Alignment check:** Thesis built from bearish FOMC 17-18 Mar + Apr earnings events and MC median S_T 66.97 / P(profit)=0.0 confirming hedge not directional bet.

### ETH  —  verdict: **hold** (conv 4)

- **Spot/σ:** $21.10 / σ=37.5%
- **Legs:** None x+0 ()
- **MC:** P(profit)=1.0, PnL p50=$7, max-loss=$0
- **Research:** 10 events, 7 leading indicators, 24 citations
- **Thesis:** Hold long ETH equity into Glamsterdam hard fork (June 2026, EIP-7732/ePBS + 200M gas) and persistent ETF inflows (10+ day streak, BlackRock ETHA $53M daily). MC 20k paths shows P(profit)=1.0, median P
- **Exit ladder:** PT1=ETH > 23.76 (MC P75) peel 40% | PT2=ETH > 28.61 (MC P95) peel remaining 60% | Stop=hard stop ETH < 15.44 close full position | Time-stop=close at 21 DTE if below MC median
- **Alignment check:** Thesis built on Glamsterdam June 2026 event, ETF inflows streak, validator queue flip, and MC 100% P(profit) with S_T P95 28.61

### OWL  —  verdict: **hold** (conv 2)

- **Spot/σ:** $9.46 / σ=57.2%
- **Legs:** P5.0 x+12 (20270115)
- **MC:** P(profit)=0.1168, PnL p50=$0, max-loss=$0
- **Research:** 9 events, 7 leading indicators, 51 citations
- **Thesis:** Long 12 Jan-2027 5.0 puts on OWL at spot 9.46. MC 20k GBM paths give P05 S_T=4.0455 (11.68% P(profit)) with median 8.73, creating asymmetric payoff if redemptions breach 5% caps or SEC FOIA probe leak
- **Exit ladder:** PT1=OWL < 6.00, peel 50% of position | PT2=OWL < 4.50, peel remaining 50% | Stop=OWL > 11.00 close, close entire position | Time-stop=close at 60 DTE if not ITM
- **Alignment check:** Thesis uses MC P05=4.0455 and P(profit)=0.1168 plus 2026-07-30 earnings, 2026-12-31 AUM, and redemption-cap/SEC indicators to justify holding the puts.
