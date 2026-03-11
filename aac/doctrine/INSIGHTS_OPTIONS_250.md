# INSIGHTS BATCH 5 — OPTIONS STRATEGIES DEEP DIVE (v2.7.0)

> **Generated**: 2026-03-01 | **Agent**: BARREN WUFFET (AZ SUPREME)
> **Sources**: OptionAlpha (36-strategy catalog), Fidelity Strategy Guide, ProjectFinance (29 core strategies), Wikipedia Options Strategy taxonomy, Reddit r/options FAQ, Ethereum.org MEV, prior doctrine research
> **Classification**: DOCTRINE MEMORY — PERMANENT
> **Previous**: INSIGHTS_BATCH4.md (v2.6.0, 350 insights)

---

## SECTION I — CORE OPTIONS MECHANICS (Insights 351–390)

351. **Options are contracts, not obligations** — calls grant the right to buy, puts grant the right to sell, at a specified strike price before or at expiration
352. **American-style options can be exercised any time before expiry** — European-style only at expiration; most equity options are American, most index options are European
353. **Intrinsic value = max(0, S-K) for calls, max(0, K-S) for puts** — only ITM options have intrinsic value; OTM options are pure extrinsic
354. **Extrinsic value (time value) decays to zero at expiration** — this is what theta measures; ATM options have the most extrinsic value
355. **Moneyness classification**: ITM (intrinsic > 0), ATM (strike ≈ stock), OTM (intrinsic = 0) — determines risk/reward profile
356. **Options pricing has 6 inputs**: stock price, strike price, time to expiry, volatility, risk-free rate, dividends — Black-Scholes computes fair value
357. **Implied volatility (IV) is the market's forecast of future volatility** — derived by solving Black-Scholes backward from market price
358. **IV Rank measures current IV relative to its 52-week range** — IVR of 80% means IV is near 52-week highs, favorable for selling premium
359. **IV Percentile measures the percentage of days with lower IV** — different from IV Rank; IVP of 90% means IV was lower on 90% of trading days
360. **Historical volatility (HV) looks backward, IV looks forward** — when IV >> HV, options are "expensive"; when IV << HV, options are "cheap"
361. **The volatility risk premium (VRP)** — IV systematically overstates realized volatility by 2-4%, creating edge for premium sellers
362. **Put-call parity: C - P = S - K*e^(-rT)** — fundamental relationship; violations create arbitrage; synthetic positions exploit this
363. **Vertical spreads have same expiration, different strikes** — bull call spread, bear put spread, bull put spread, bear call spread
364. **Horizontal (calendar) spreads have same strike, different expirations** — profit from time decay differential and IV changes
365. **Diagonal spreads combine different strikes AND different expirations** — customizable risk/reward; Poor Man's Covered Call is a diagonal
366. **Credit spreads receive premium upfront** — max profit = credit received; risk-defined; bull put spread and bear call spread
367. **Debit spreads pay premium upfront** — max profit = spread width - debit paid; bull call spread and bear put spread
368. **Ratio spreads use unequal numbers of options** — 1x2, 2x3 ratios; can be zero-cost; introduce unlimited risk on one side
369. **Backspreads are inverted ratio spreads** — more options purchased than sold; limited risk with unlimited profit potential on one side
370. **Box spreads = bull call spread + bear put spread** — risk-free at expiration; used as financing tool; pays difference in strikes
371. **Assignment risk exists for short American-style options** — ITM shorts can be assigned any time; highest risk near ex-dividend dates
372. **Pin risk occurs when stock closes near a short strike at expiration** — uncertain whether assignment will occur; can create unwanted positions overnight
373. **Max pain theory**: stock gravitates toward strike with most open interest at expiration — market makers hedge delta, creating self-fulfilling magnet
374. **OPEX (options expiration) effects**: quarterly OPEX (triple/quad witching) creates $4T+ notional unwind — volatility spikes, then collapses
375. **0DTE (zero days to expiration) options now account for 40%+ of SPX volume** — pure gamma plays; rapid theta decay; extreme leverage
376. **Gamma exposure (GEX) measures dealer hedging flows** — positive GEX = dealers sell rallies/buy dips (suppresses vol); negative GEX = dealers amplify moves
377. **Delta-neutral hedging by market makers** creates the actual price pins, reversals, and explosive moves around key strikes
378. **Charm (delta decay) measures how delta changes as time passes** — OTM options lose delta faster; ITM options gain delta; critical for multi-day positions
379. **Vanna measures how delta changes with IV** — when IV drops, call delta increases and put delta decreases; drives post-earnings reversals
380. **Volga (vomma) measures how vega changes with IV** — convexity of vega; matters for extreme IV changes and tail events
381. **Speed is the third derivative of price** — gamma of gamma; predicts how rapidly delta-hedging costs will escalate
382. **Color is the rate of change of gamma over time** — gamma bleed; at-the-money gamma increases as expiration approaches
383. **Ultima measures sensitivity of vomma to volatility** — fourth-order Greek; only matters for exotic options and extreme scenarios
384. **Options clearing happens through the OCC** — counterparty risk eliminated; $1.8B+ in margin required; 50+ years without default
385. **Regulation T margin requires 50% initial margin for stock** — options strategies have specific margin rules; portfolio margin allows 15% for broad portfolios
386. **CBOE Volatility Index (VIX) measures 30-day expected SPX volatility** — "fear gauge" derived from SPX options prices; mean-reverts from extremes
387. **VIX term structure: contango (normal) vs backwardation (fear)** — when front-month VIX > back-month, market expects near-term turbulence
388. **Skew index measures the steepness of the volatility smile** — high skew = expensive OTM puts relative to calls; tail risk pricing elevated
389. **Volatility surface is 3D**: strike (moneyness) × expiration × IV — provides complete view; spot skew, term structure, and calendar effects simultaneously
390. **Early exercise is optimal ONLY when time value < expected dividend** — deep ITM calls before ex-dividend; almost never optimal for puts on stocks

## SECTION II — BULLISH OPTIONS STRATEGIES (Insights 391–425)

391. **Long call: simplest bullish bet** — unlimited upside, limited risk (premium paid); best when IV is low and strong directional conviction
392. **Long call breakeven = strike + premium paid** — needs stock above breakeven at expiry to profit; time decay works against you
393. **Cash-secured put: bullish yield enhancement** — sell put, hold cash to cover assignment; collect premium while waiting to buy at lower price
394. **Cash-secured put is functionally a limit buy order that pays you to wait** — premium received = immediate income regardless of assignment outcome
395. **Covered call: own 100 shares + sell OTM call** — caps upside at strike; premium reduces cost basis; best in sideways-to-mildly-bullish markets
396. **Covered call yield targets**: 1-3% monthly on blue chips; 3-5% monthly on high-IV names — annualizes to 12-60%; don't chase yield at expense of quality
397. **Bull call spread: buy lower strike call, sell higher strike call** — reduces cost vs naked long call; caps profit at short strike; defined risk
398. **Bull call spread max profit = spread width - debit paid** — occurs when stock is above short strike at expiration
399. **Bull put spread: sell higher strike put, buy lower strike put** — credit spread; profit when stock stays above short put strike
400. **Bull put spread probability of profit typically 60-70%** — higher than long calls; trade-off is capped profit and short premium risk
401. **Collar: own stock + buy OTM put + sell OTM call** — zero-cost or near-zero-cost insurance; protect downside while giving up upside
402. **Zero-cost collar uses call premium to exactly offset put purchase** — often used by executives who can't sell stock but need downside protection
403. **Protective put: own stock + buy put** — insurance policy; unlimited upside preserved; expensive but absolute floor on losses
404. **Married put: buy stock + buy put simultaneously** — functionally identical to protective put; started at same time for tax purposes
405. **Synthetic long stock: buy ATM call + sell ATM put** — replicates stock ownership with less capital; same P/L profile as 100 shares
406. **Synthetic long has no time decay if both options are ATM** — delta ≈ +100; great capital efficiency but assignment risk on short put
407. **Poor Man's Covered Call (PMCC): buy deep ITM LEAPS call + sell short-term OTM call** — diagonal spread; replicates covered call with 1/3 the capital
408. **PMCC max loss = LEAPS debit - short call credit** — long call should have delta > 0.70; expiration 6+ months out for slow decay
409. **Wheel strategy: sell cash-secured put → get assigned → sell covered call → get called away → repeat** — systematic income generation cycle
410. **Wheel optimal on high-quality stocks you'd own anyway** — don't wheel meme stocks; 30-45 DTE, 0.25-0.35 delta puts; collect 2-4% per cycle
411. **Covered strangle: own 100 shares + sell OTM call + sell OTM put** — double premium collection vs covered call alone; more aggressive, higher assignment risk
412. **Long call ladder: buy 1 ATM call + sell 1 OTM call + sell 1 further OTM call** — reduced cost; best in mildly bullish markets; unlimited risk above second short strike
413. **1x2 ratio call spread: buy 1 ITM call, sell 2 OTM calls** — can be entered for zero cost; unlimited risk above upper breakeven
414. **1x2 ratio volatility spread with calls (Fidelity): bullish, unlimited profit, limited risk** — long more calls than short; backspread structure
415. **Bullish split-strike synthetic: buy OTM call + sell OTM put (different strikes)** — less capital than stock; gap between strikes = "dead zone" of no P/L
416. **LEAPS calls as stock replacement** — 80+ delta, 1-2 year expiry; 1/3 capital of stock; time decay minimal at 500+ DTE
417. **Risk reversal: sell OTM put + buy OTM call** — zero-cost bullish exposure; unlimited upside; unlimited downside (like stock but leveraged)
418. **Jade lizard: sell OTM put + sell call spread (bear call)** — bullish; no upside risk if call spread credit > put sold distance; unique risk profile
419. **Broken Wing Butterfly (BWB) bullish: buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call (wider on upside)** — credit or small debit; no risk to downside
420. **Skip-strike butterfly: asymmetric butterfly with gap between wings** — directional bias; cheaper than standard butterfly; targets specific price level
421. **Calendar call spread: sell near-term call, buy longer-term call (same strike)** — profits from time decay differential; best in moderate IV with mild bullish outlook
422. **Double calendar: two calendar spreads at different strikes** — wider profit zone than single calendar; neutral-to-directional; complex management
423. **Covered combination = covered strangle** — Fidelity terminology; collect both call and put premium against stock position
424. **Buy-write = simultaneously buy stock and sell covered call** — entered as single order; better fills than legging; most common options income strategy
425. **Bullish strategies benefit from rising stock, declining IV (for spreads), and time passage (for premium sellers)** — align Greek exposure with market view

## SECTION III — BEARISH OPTIONS STRATEGIES (Insights 426–455)

426. **Long put: simplest bearish bet** — profits as stock falls; limited risk (premium paid); max profit = strike - premium (stock goes to 0)
427. **Long put as portfolio insurance** — buying SPY puts protects against crash; 5% annual portfolio "insurance premium" with 95% protection
428. **Bear put spread: buy higher strike put, sell lower strike put** — debit spread; cheaper than naked put; defined max loss and max profit
429. **Bear put spread max profit = spread width - debit paid** — occurs when stock below lower (short) strike at expiration
430. **Bear call spread: sell lower strike call, buy higher strike call** — credit spread; profit when stock stays below short call strike; defined risk
431. **Bear call spread is the most popular bearish credit strategy** — easier to manage than bear put spread; theta works in your favor
432. **Short call (naked): unlimited risk** — sells call without owning stock; profit = premium if stock stays below strike; margin-intensive
433. **Covered put write: short stock + sell OTM put** — bearish covered call equivalent; doubles premium but doubles downside exposure
434. **Synthetic short stock: sell ATM call + buy ATM put** — replicates -100 shares; less margin than shorting stock; unlimited risk to upside
435. **Put backspread (1x2): sell 1 ITM put, buy 2 OTM puts** — profits from crash; limited/no risk to upside; can be entered for credit
436. **Bear put ladder: buy 1 ATM put + sell 1 OTM put + sell 1 further OTM put** — reduced cost; unlimited risk below lowest strike
437. **Bearish split-strike synthetic (Fidelity): buy OTM put + sell OTM call** — synthetic short with dead zone between strikes
438. **Put debit spread sizing**: risk per trade = debit paid × number of contracts; typically 1-3% of portfolio per trade
439. **When to prefer bear put vs bear call**: use bear put when you want defined timing (debit, time works against); bear call when you want theta on your side
440. **Bearish diagonal: sell near-term put (higher strike), buy longer-term put (lower strike)** — theta positive; directional and time-spread combined
441. **1x2 ratio put spread (Fidelity): buy 1 ATM put, sell 2 OTM puts** — neutral-to-bearish; unlimited risk below lower breakeven
442. **1x2 ratio volatility spread with puts (Fidelity): bearish, substantial profit, limited risk** — backspread structure; more puts bought than sold
443. **Married call: short stock + buy call** — insurance on short position; unlimited upside protection with floor on losses
444. **Protective call on short stock** — same as married call; prevents unlimited loss; premium is insurance cost
445. **Christmas tree spread with puts: buy 1 ATM put, sell 1 OTM put, sell 1 further OTM put** — Fidelity variant; reduced cost; asymmetric payoff; limited profit
446. **Bear call credit spread management**: close at 50% max profit; defend at 2x credit received; roll if challenged
447. **Bearish plays on earnings**: buy put spread into earnings; IV crush hurts long options, so spreads reduce vega exposure
448. **Inverse ETF options as bearish strategy** — SH, SDS, SPXS options allow "bearish calls" with defined risk; avoid leveraged ETF decay
449. **VIX calls as tail hedge** — rise when market falls; mean-reversion makes timing critical; use call spreads to reduce cost
450. **Put spread collar on index positions**: own ETF + buy put spread + sell call — cheaper than protective put; capped upside and partial downside protection
451. **Statistical edge of bearish strategies**: markets fall 3x faster than they rise — puts gain IV premium during drops (VIX spike)
452. **Crash put behavior**: deep OTM puts can go 10-50x during crashes (March 2020 SPY 200 puts gained 4,000%) — portfolio insurance payoffs are convex
453. **Bear market rally traps**: sell credit call spreads into bear market rallies — strong theta + statistical fade setup
454. **Sector rotation bearish plays**: when rates rise, short REITs/utilities via bear call spreads — carry positive theta
455. **Pairs trade via options**: buy puts on weak stock, sell puts on strong peer — sector-neutral bearish conviction with reduced systemic risk

## SECTION IV — NEUTRAL & VOLATILITY STRATEGIES (Insights 456–510)

456. **Iron condor: sell OTM put spread + sell OTM call spread** — defined risk neutral strategy; profit from range-bound, low-volatility environment
457. **Iron condor optimal setup**: 30-45 DTE, 15-25 delta short strikes, close at 50% profit or 21 DTE — statistical edge demonstrated over thousands of occurrences
458. **Iron condor adjustments**: if one side is threatened, roll the untested side closer (inverted iron condor) or close the tested side before max loss
459. **Wide iron condor (10-delta) vs narrow (25-delta)**: wide = higher POP but lower premium; narrow = lower POP but higher premium; both valid approaches
460. **Iron butterfly: sell ATM straddle + buy OTM wings** — maximum theta at center; higher premium than iron condor; requires tighter management
461. **Iron butterfly max profit = net credit** — occurs only if stock expires exactly at ATM strike; uncommon but very high reward-to-risk setup
462. **Short straddle: sell ATM call + sell ATM put** — maximum premium collected; unlimited risk both directions; requires margin and active management
463. **Short straddle breakevens**: upper = ATM + total premium, lower = ATM - total premium — stock must stay within these bounds for profit
464. **Long straddle: buy ATM call + buy ATM put** — profits from big move in either direction; needs stock to move more than total premium paid
465. **Long straddle pre-earnings**: buy 5-7 days before earnings; needs realized move > implied move (IV crush destroys value even if direction is right)
466. **Short strangle: sell OTM call + sell OTM put** — wider profit range than straddle; unlimited risk; preferred by premium sellers for higher POP
467. **Short strangle management**: 45 DTE entry, 21 DTE exit; never let a short option go ITM without adjusting; position size 2-5% of portfolio
468. **Long strangle: buy OTM call + buy OTM put** — cheaper than straddle; needs bigger move to profit; lower breakeven probability
469. **Butterfly spread (long): buy 1 lower, sell 2 middle, buy 1 upper** — maximum profit at center strike; cheap to enter; precise target required
470. **Butterfly spread risk/reward can exceed 10:1** — risk = debit paid; max profit = wing width - debit; perfect for pinning plays
471. **Broken wing butterfly (neutral): buy 1 lower put, sell 2 middle puts, buy 1 far OTM put** — credit entry; no risk to upside; risk below lowest strike
472. **Condor (long): buy outer strikes, sell inner strikes** — wider profit zone than butterfly; lower max profit; more forgiving on price
473. **Calendar spread (neutral): sell near-term, buy longer-term (same strike)** — profits from time decay difference AND IV increase
474. **Calendar spread Greeks: positive theta, positive vega, near-zero delta** — rare combination; profits from time passing AND volatility rising
475. **Double diagonal: two diagonals (different strikes, different expirations)** — wider profit zone than single diagonal; complex Greeks management
476. **Short guts: sell ITM call + sell ITM put** — more margin-intensive than strangle; same P/L profile but harder to manage assignment risk
477. **Long guts: buy ITM call + buy ITM put** — expensive but high delta on both sides; needs strong conviction in volatility expansion
478. **Jelly roll: long calendar at one strike + short calendar at another** — pure interest rate / dividend play; no exposure to stock price
479. **Fence (risk reversal collar variant)**: buy protective put + sell covered call — limits range of outcomes; commonly used by hedgers
480. **Conversion: long stock + long put + short call (same strike)** — arbitrage play; risk-free if options are mispriced; rarely available in liquid markets
481. **Reversal: short stock + short put + long call (same strike)** — reverse of conversion; synthetic arbitrage when put-call parity is violated
482. **Iron condor width optimization**: 5-wide on $50 stocks, 10-wide on $100+ stocks — keep risk:reward between 1:1 and 3:1
483. **Jade lizard on neutral outlook**: sell put + sell call spread — no upside risk if total credit > width of call spread; unique structure
484. **Straddle swap: sell front-month straddle, buy back-month straddle** — time spread at ATM; profits from term structure flattening
485. **Ratio write: own stock + sell 2 covered calls** — one covered, one naked; extra premium but naked call introduces unlimited risk
486. **Volatility arbitrage**: when IV significantly exceeds HV, sell options; when HV significantly exceeds IV, buy options — mean-reversion play
487. **VRP harvesting strategy**: systematically sell 30-delta strangles on SPY monthly — exploits 2-4% average IV overstatement; 60-70% annual return, 15-20% max DD
488. **Dispersion trading**: sell index options (high IV), buy component stock options (lower IV) — profits from correlation risk premium
489. **Variance swap replication**: trade a strip of options across all strikes — captures pure volatility exposure without directional risk
490. **Gamma scalping**: buy straddle, delta-hedge continuously — profit from realized vol exceeding implied vol; pure volatility play
491. **Theta gang philosophy**: systematically sell options premium, let time decay work in your favor — "be the casino, not the gambler"
492. **Expected move calculation**: ATM straddle price × 0.85 ≈ 1 standard deviation expected move — quick estimation for earnings
493. **Iron condor position sizing**: risk no more than 2-5% of account on single iron condor — max loss should never threaten account survival
494. **Rolling strategies**: roll out (further expiry), roll up/down (different strike), roll out-and-up/down — repair losing positions without taking full loss
495. **When to take assignment vs close**: consider tax implications, dividend capture, portfolio needs — assignment is not always bad
496. **Earnings straddle/strangle buy**: profitable only ~30% of time due to IV crush — must be selective; focus on stocks with history of exceeding expected moves
497. **Post-earnings volatility crush** averages 40-60% IV reduction — selling premium into earnings benefits from this crush
498. **FOMC volatility pattern**: IV builds 1-2 weeks before, crushes within minutes of announcement — sell premium day before, buy back after announcement
499. **Quarterly OPEX gamma squeeze dynamics**: dealer delta-hedging can create 2-3% intraday swings on expiration Fridays
500. **Weekly options have accelerated theta decay** — last 5 days of option life see 50%+ of total time decay; weeklies = pure theta plays

## SECTION V — ADVANCED MULTI-LEG & EXOTIC (Insights 501–535)

501. **Christmas tree spread (Fidelity): buy 1 ATM, sell 1 OTM, sell 1 further OTM (same side)** — cheaper than butterfly; asymmetric risk; directional bias
502. **Long Christmas tree with calls: bullish, limited profit, limited risk** — profit zone between short strikes; max loss below ATM or far above top strike
503. **Long Christmas tree with puts: bearish, limited profit, limited risk** — mirror of call version; targets moderate decline
504. **Unbalanced butterfly: different wing widths** — directional bias built in; can be entered for credit; skewed profit zone
505. **Double butterfly: two butterflies at adjacent strikes** — wider profit zone than single butterfly; lower max profit; more tolerance
506. **Condor vs butterfly decision**: condor for wider range; butterfly for pinpoint target — condor = "lazy butterfly" with wider sweet spot
507. **Iron condor to iron butterfly conversion**: move short strikes to same ATM strike — increases premium but narrows profit range
508. **Ladder spread = 3-leg vertical**: buy 1, sell 1, sell 1 higher/lower — extension of vertical spread; introduces unlimited risk beyond 3rd leg
509. **Seagull spread: bull call spread + short OTM put** — zero-cost bullish structure; downside risk below short put
510. **Albatross spread (4-leg): wide condor with very wide wings** — ultra-wide profit zone; minimal premium; very high probability of profit
511. **Zebra (Zero Extrinsic Back Ratio): buy 2 ATM calls, sell 1 ITM call** — synthetic covered call without owning stock; near-zero extrinsic value
512. **Slingshot hedge: own stock + buy ATM put + sell 2 OTM calls** — downside protected; upside capped at first call; free or credit
513. **Super Bull spread: buy call vertical + sell put vertical at same strikes** — amplified directional bet; double premium risk
514. **Reverse iron condor: buy wings, sell body** — debit trade profiting from big move in either direction; opposite of standard IC
515. **Batman spread: iron butterfly with extra wings** — wider break-even range than standard butterfly; resembles bat wings on P/L chart
516. **Twisted sister: unbalanced strangle with different DTE per side** — calendar component on one side; directional time-spread hybrid
517. **Back ratio with calls: sell 1 ITM call, buy 2 OTM calls** — limited/no downside risk; unlimited upside; classic pre-earnings play
518. **Front ratio with puts: buy 1 ATM put, sell 2 OTM puts** — credit entry; bearish; unlimited risk below lower strike; great for IV overstatement
519. **Skip-strike butterfly for earnings**: target expected post-earnings price with center strikes — cheap directional bet with defined risk
520. **Lizard: any strategy where total credit received eliminates risk on one side** — jade lizard, big lizard; unique risk profiles
521. **Big lizard: sell straddle + buy OTM call** — unlimited downside; no upside risk (if call protects above straddle net credit)
522. **Guts strangle vs regular strangle**: guts uses ITM options; higher premium collected but higher assignment risk and margin
523. **Box spread as synthetic loan**: buy box at below risk-free rate → synthetic borrowing/lending — used by institutions
524. **SPX box spread financing**: 3-5 year SPX boxes trade at implied interest rates — cheaper than margin loans; no early exercise risk (European-style)
525. **Combo (synthetic forward)**: buy call + sell put at same strike/expiration — equivalent to forward contract; common in futures options
526. **Call time spread (calendar) vs put time spread**: equivalent P/L due to put-call parity; choose based on skew and liquidity
527. **Portfolio overlay strategies**: sell monthly OTM calls on portfolio to generate 1-2% income — systematically reduce cost basis
528. **Delta-hedged straddle**: buy straddle, adjust hedge as stock moves — pure volatility extraction; no directional bias after hedging
529. **Variance swap via options strip**: requires trading options at many strikes — VRP research shows systematic selling has 2%+ average monthly return
530. **Quanto options**: options with payoff in different currency from underlying — hedge both price and FX risk simultaneously
531. **Barrier options (knock-in/knock-out)**: activate or deactivate at specific price levels — cheaper premium but gap risk at barrier
532. **Digital (binary) options**: all-or-nothing payoff — regulated on NADEX; unregulated versions are SEC-flagged fraud vehicles
533. **Cliquet options**: series of forward-starting ATM options with periodic resets — used in structured products; complex pricing
534. **Lookback options**: payoff based on max/min price during life — most expensive exotic; eliminates timing regret
535. **Asian options**: payoff based on average price — lower premium than vanilla; common in commodity markets; reduces manipulation risk

## SECTION VI — RISK MANAGEMENT & GREEKS MASTERY (Insights 536–570)

536. **Delta: rate of change of option price per $1 change in stock** — positive for calls (0 to +1), negative for puts (-1 to 0); probability proxy
537. **Delta as probability proxy**: 0.30 delta ≈ 30% chance of expiring ITM — not exact but useful approximation for trade selection
538. **Gamma: rate of change of delta per $1 change in stock** — always positive for long options; accelerates gains, accelerates losses; highest for ATM near expiry
539. **Gamma risk peaks at expiration for ATM options** — delta can swing from 0 to 100 with small stock move; "gamma bomb" on short options
540. **Theta: daily time decay of option value** — negative for long options, positive for short; accelerates in final 30 days; steepest last 7 days
541. **Theta decay is not linear** — follows square root of time; option loses more value per day as expiration approaches
542. **Weekend theta**: 3 calendar days but only 1 trading day — debate whether Friday close prices "already" include weekend decay
543. **Vega: sensitivity to 1-point change in IV** — longer-dated options have higher vega; ATM options have highest vega at any expiration
544. **Rho: sensitivity to interest rate changes** — negligible for short-dated options; matters for LEAPS and in high-rate environments
545. **Position Greeks**: sum of individual option Greeks — portfolio delta = sum of all position deltas; enables risk management at portfolio level
546. **Delta-neutral portfolio**: total delta = 0 — immune to small price moves; exposed to gamma, theta, vega — pure volatility play
547. **Gamma-theta trade-off**: long gamma costs theta; short gamma earns theta — fundamental tension in options market
548. **Positive gamma = convexity = "good" curvature** — your profits accelerate as stock moves in your direction; valuable in tail events
549. **Negative gamma = concavity = "bad" curvature** — your losses accelerate as stock moves against you; dangerous near expiration
550. **Vega-theta trade-off in calendars**: earn theta from near-term short, benefit from vega increase on far-term long — dual positive exposure
551. **Greek-based position sizing**: target portfolio theta/delta ratio; cap negative gamma exposure as percentage of account value
552. **Stress testing positions**: recalculate P/L at ±10%, ±20% stock moves AND ±25%, ±50% IV changes — 2D stress matrix
553. **Beta-weighted delta**: convert all positions to SPY-equivalent delta — portfolio has 23 delta across various stocks = 23 shares SPY equivalent
554. **Sector delta concentration**: if all bullish positions are tech, portfolio delta is actually tech-beta-adjusted — diversification matters
555. **Correlation risk in multi-leg**: not all stocks move independently — portfolio of 10 iron condors on correlated stocks ≠ 10x diversification
556. **Tail risk mitigation**: buy far OTM puts at 5% delta — costs 0.5-1% annually but prevents portfolio-ending drawdowns
557. **Systematic hedge ratio**: spend 1-3% of annual returns on protective puts — insurance that enables more aggressive core strategy
558. **Position size by max loss**: never risk more than 2% of account on single options trade — even defined-risk trades can be too large
559. **Account allocation framework**: 50% buying power for positions, 25% reserve for adjustments, 25% cash for new opportunities
560. **Portfolio margin allows 6:1+ leverage vs Reg-T's 2:1** — massively increases capital efficiency but also amplifies risk
561. **Greeks by strategy type**: iron condor = +theta, -gamma, -vega; long straddle = -theta, +gamma, +vega; calendar = +theta, +vega, ~neutral delta
562. **Greek attribution for P/L**: decompose daily P/L into delta P/L + gamma P/L + theta P/L + vega P/L — know exactly why you made/lost money
563. **Skew trading via spreads**: buy options on cheap part of skew, sell on expensive part — vertical spreads naturally exploit skew
564. **Term structure trading via calendars**: sell overpriced near-term IV, buy underpriced far-term IV — calendar spreads exploit term structure
565. **Smile dynamics during earnings**: skew flattens as earnings approaches (IV rises uniformly), then re-establishes after release
566. **Volatility of volatility (vol-of-vol)**: measures how much IV itself moves — high vol-of-vol environment makes all vega trades riskier
567. **Omega (elasticity)**: percentage change in option vs percentage change in stock — leverage measure; OTM options have highest omega
568. **Lambda = omega**: same Greek, different name — common source of confusion in options literature
569. **Fugit: expected time to optimal exercise for American options** — measures "American premium" over European equivalent
570. **Zomma: rate of change of gamma with respect to IV** — when IV spikes, gamma changes; matters for extreme scenarios

## SECTION VII — OPTIONS INCOME & SYSTEMATIC STRATEGIES (Insights 571–600)

571. **The "Wheel" strategy produces 15-30% annualized returns on quality blue chips** — SPY, QQQ, AAPL, MSFT most common underlying
572. **Wheel optimization: sell 0.25 delta puts, 30-45 DTE** — target 2-3% premium per cycle; 82-85% probability of profit per trade
573. **Covered call write index (BXM) has outperformed S&P 500 on risk-adjusted basis since 1986** — lower returns but significantly lower volatility
574. **Systematic monthly strangle selling on SPY: 20-25% annual return, Sharpe ratio ~1.2** — documented VRP harvesting with decades of data
575. **Options income tax treatment**: premium received from short options is short-term capital gain or loss — no long-term advantage regardless of holding
576. **Section 1256 contracts (index options, futures options) get 60/40 tax treatment** — 60% long-term, 40% short-term regardless of holding period
577. **SPX, XSP, RUT, NDX options qualify for 1256 treatment** — significant tax advantage over equity options like SPY, QQQ
578. **Wash sale rules apply to options** — selling a put at loss and selling a "substantially identical" put within 30 days triggers wash sale
579. **Qualified covered calls**: must be ATM or one strike OTM with 30+ days to expiry — non-qualified reduces holding period for underlying
580. **Constructive sale rule**: deep ITM covered calls can trigger taxable event on appreciating stock — IRS treats as selling the stock
581. **Covered call income as retirement strategy**: sell 2-3% monthly premium on ETFs in tax-deferred account — compounds without tax drag
582. **Poor Man's Covered Call generates similar return as covered call with 60% less capital** — but requires active management and roll decisions
583. **LEAPS as stock replacement in IRA**: buy 80-delta, 18-month LEAPS in IRA; sell monthly calls against it — capital-efficient retirement income
584. **Strangle selling position sizing**: each strangle should consume <5% of buying power — 20+ positions minimum for diversification
585. **Sector-diversified premium selling**: sell strangles across 8-10 sectors — correlation breaks down in crashes, making diversification essential
586. **Rolling a covered call**: buy back current month (loss), sell next month at same or higher strike — extends the trade, captures more premium
587. **The "repair" strategy**: double down position by adding bull call spread at cost basis — reduces breakeven without additional risk
588. **Dividend capture with options**: sell puts on high-dividend stocks before ex-date; if assigned, collect dividend; sell covered call after
589. **ETF vs individual stocks for options income**: ETFs have lower IV = less premium BUT better diversification and lower blowup risk
590. **Options income portfolio construction**: 60% ETF premium selling + 30% sector single-name premium + 10% tail hedges = balanced approach
591. **Monthly income target for options traders**: 2-5% return on capital at risk — 24-60% annualized; anything higher typically means excessive risk
592. **Drawdown management**: stop selling premium when portfolio down 10% from peak — reassess positions, reduce size, allow recovery
593. **Win rate vs expectancy**: 80% win rate means nothing if 20% losses are 5x winners — expected value matters more than frequency
594. **1R risk management**: define "1R" as your standard risk unit; winners should average 0.5R (50% of max profit) with 70%+ hit rate
595. **Time-based exits**: close positions at 50% max profit or 21 DTE, whichever comes first — backtested optimal across strategies
596. **Mechanical vs discretionary management**: systematic rules outperform discretionary emotion-driven management in options selling
597. **Black swan protection**: always have 1-2% of portfolio in OTM puts 60+ days out — insurance that lets you sell premium aggressively
598. **Reverse compound annual growth**: $100K account earning 2% monthly options income = $26,824 first year → $34,500 second year with reinvestment
599. **The "10% buffer" rule**: set short strikes 10%+ from current price for blue chips; 15%+ for high-beta names; reduces assignment by 85%+
600. **Options selling in bear markets requires position size reduction by 50%** — VRP still exists but tail risk increases dramatically

---

**Total Insights This Document**: 250 (351–600)
**Research Sources**: OptionAlpha.com, Fidelity Strategy Guide, ProjectFinance (29 strategies), Wikipedia Options Strategy, Reddit r/options FAQ, prior doctrine
**Key Themes**: Core mechanics, bullish strategies, bearish strategies, neutral/volatility strategies, advanced multi-leg, risk management, systematic income
