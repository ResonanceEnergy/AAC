# AAC Deep Audit — Stubs, Placeholders, Bypasses, Workarounds, Silent Fails, Bad Patches, False Data

> **Generated:** 2026-05-15
> **Method:** Six parallel regex sweeps across `aac/`, `core/`, `shared/`, `strategies/`, `TradingExecution/`, `integrations/`, `monitoring/`, `BigBrainIntelligence/`, `CryptoIntelligence/`, `CentralAccounting/`, `agents/`, `divisions/`, `startup/`, `SharedInfrastructure/`, `trading/`, `councils/`, `deployment/`. Excludes `tests/`, `_archive/`, `_scratch/`, `.venv/`.
> **Scope:** 798 production .py files. **~520 distinct hits** triaged into 7 categories × 4 severity bands.
> **Companion to:** `.context/07_gaps/gap-audit-2026-05-15-250.md` (the 250-gap structural audit).

---

## Severity Legend

| Tier | Criterion | Action |
|------|-----------|--------|
| **🔴 SEV-1 / CRITICAL** | False data could reach live trading decisions, risk math, or P&L. | Fix or fence-off **before next live trade**. |
| **🟠 SEV-2 / HIGH** | Silent failure mode could hide an outage; placeholder wired into a dashboard or signal path. | Fix this sprint. |
| **🟡 SEV-3 / MEDIUM** | Abstract stub, dev-only path, or documented future work. | Track; do before dependent feature ships. |
| **🟢 SEV-4 / LOW** | Cosmetic placeholder (UI hint), legitimate disable flag, intentional bypass with clear semantics. | No action. |

---

## 🔴 SEV-1 — FALSE DATA INTO LIVE PATHS (12 sites)

These produce **fabricated numeric values** that flow downstream into signals, dashboards, models, or P&L attribution. Highest priority.

| # | File | Line | What it fakes | Risk |
|---|------|------|---------------|------|
| 1 | [strategies/strategy_execution_engine.py](strategies/strategy_execution_engine.py#L506) | 506, 542, 575, 605, 648, 676, 709, 787, 819, 853, 886 | 11 algorithm methods generate `nav_premium`, `imbalance_ratio`, `vrp_signal`, etc. via `rng.uniform()` with deterministic per-minute seed. Returns full `StrategyTradeSignal` objects. | Live execution path can act on these signals if the engine is wired in. Audited: emits via `signal_callback`. |
| 2 | [shared/ai_strategy_generator.py](shared/ai_strategy_generator.py#L163) | 163-166, 259, 424-427, 536-538 | Trains "ML models" on `random.uniform`-generated synthetic price data, then ranks strategies on those scores. | Strategy bake-off rankings could be pure noise. |
| 3 | [strategies/matrix_maximizer/scanner.py](strategies/matrix_maximizer/scanner.py#L269) | 203, 269, 298, 312, 338, 340 | Generates **synthetic options chain** via Black-Scholes when Polygon key absent; tags `source="synthetic"` but downstream consumers don't always check. | Greek-based signals on fabricated chains. |
| 4 | [strategies/matrix_maximizer/backtester.py](strategies/matrix_maximizer/backtester.py#L248) | 248, 265-279 | Synthetic VIX/oil/SPX scenarios via `random.gauss` for "backtests". | Strategy validation results meaningless. |
| 5 | [monitoring/dashboard_pillars.py](monitoring/dashboard_pillars.py#L224) | 224 | `"call_delta": 0.30  # placeholder — yfinance doesn't expose Greeks` | Dashboard shows fake delta as if real. |
| 6 | [strategies/war_room_engine.py](strategies/war_room_engine.py#L2871) | 2871-2873 | `return 0.5  # fallback on any network error` — fabricated confidence | Council scoring uses 0.5 (max-uncertainty) as if it were a real reading. |
| 7 | [divisions/trading/warroom/engine.py](divisions/trading/warroom/engine.py#L2553) | 2553-2576 | Duplicate of #6 (now a shim, but the canonical war_room_engine.py still has it) | Same as above. |
| 8 | [CentralAccounting/financial_analysis_engine.py](CentralAccounting/financial_analysis_engine.py#L176) | 176 | `strategy_correlation = 0.3  # Placeholder` used in correlation-aware sizing | Position sizing uses fake correlation. |
| 9 | [shared/strategy_parameter_tester.py](shared/strategy_parameter_tester.py#L365) | 279, 365-378 | `random.choice(["SPY","QQQ"])`, `random.uniform(-10,10)` for "parameter testing" | Optimisation results are fake. |
| 10 | [shared/paper_trading.py](shared/paper_trading.py#L418) | 418, 420, 526 | `random.uniform`/`random.gauss` for slippage, ±2% price variation | Acceptable for *paper*, but ensure no live path imports this. |
| 11 | [strategies/planktonxd_prediction_harvester.py](strategies/planktonxd_prediction_harvester.py#L1194) | 1194 | `if random.random() < true_prob:` decides "outcome" of a prediction | Drives Polymarket position selection. |
| 12 | [shared/incident_postmortem_automation.py](shared/incident_postmortem_automation.py#L173) | 173 | `variance = random.uniform(-0.5, 0.5)` injected into postmortem metrics | Reports become unreliable. |

**Recommended fix pattern (fence-off, opt-in):**
```python
def __init__(self, ...):
    self.allow_synthetic = bool(self.config.get("allow_synthetic_signals", False))

async def _xxx_algorithm(self):
    if not self.allow_synthetic:
        return None  # live mode never emits synthetic signals
    ...
```

---

## 🔴 SEV-1 — TEMPLATE-EMBEDDED FAKE EXECUTION (1 file)

| # | File | Line | What |
|---|------|------|------|
| 13 | [core/aac_automation_engine.py](core/aac_automation_engine.py#L428) | 428, 432 | `random.expovariate(...)` for "execution_delay" and `random.random() < fill_probability` to simulate fills. **Confirmed inside a triple-quoted code-generation template** that writes a `paper_order_simulator.py` to disk. Name aside, this is a code-gen template, not an active execution path — but the *generated* file would be a live-path landmine if anyone ran it. |

**Recommended:** delete the template entirely or move it under `paper_trading/templates/` with a banner.

---

## 🟠 SEV-2 — HIGH-RISK SILENT FAILS (28 sites of concern)

Bare `except:` (no exception class), broad `except Exception:` swallowed silently, or `return 0.5/0.0/{}/False` fallbacks on network errors.

### Bare `except:` (no class) — **8 sites, all in one file**
[agents/planktonxd_browser_bot.py](agents/planktonxd_browser_bot.py) lines **400, 460, 521, 557, 577, 749, 816, 832**. Each is `except: continue`. Hides every error class including `KeyboardInterrupt` and `SystemExit`.

### `return X` on broad `except` (silent fallback to fake value)

| File | Line | Code | Problem |
|------|------|------|---------|
| [strategies/war_room_engine.py](strategies/war_room_engine.py#L1584) | 1584 | `except Exception: return {}` | Empty dict masks any failure |
| [strategies/war_room_engine.py](strategies/war_room_engine.py#L2871) | 2871-2894 | `return 0.5` and `return 0.0` on network error | See SEV-1 #6 |
| [divisions/trading/warroom/engine.py](divisions/trading/warroom/engine.py#L1356) | 1356, 2553-2576 | Duplicate of above | Re-export through shim, but file still on disk |
| [integrations/openclaw_gateway_bridge.py](integrations/openclaw_gateway_bridge.py#L833) | 833 | `except Exception: return []` | Hides connector errors |
| [startup/watchdog.py](startup/watchdog.py#L170) | 170 | `except Exception: return False` | Watchdog reports "ok" even when broken |
| [monitoring/aac_system_registry.py](monitoring/aac_system_registry.py#L101) | 101 | `except Exception: return False, latency` | Health check shows latency on failure |
| [shared/cache_manager.py](shared/cache_manager.py#L212) | 212 | `except Exception: return False` | Cache write failures invisible |
| [shared/security_framework.py](shared/security_framework.py#L261) | 261 | `except Exception: return False` | Auth/security failures invisible |
| [CentralAccounting/database.py](CentralAccounting/database.py#L418) | 418 | `except Exception: return False` | DB write failures invisible |

### `except Exception: pass` (still on disk after the gap-audit pass)
The hardened checker confirms **0 silent-pass** in production paths after first remediation. Remaining `pass` cases are scoped (`except KeyboardInterrupt`, `except asyncio.CancelledError`, `except curses.error`) and are acceptable.

### Suspicious composite
[monitoring/aac_master_monitoring_dashboard.py:2789](monitoring/aac_master_monitoring_dashboard.py#L2789) — `except (ImportError, Exception): pass` — the `Exception` parent makes `ImportError` redundant; effectively bare-Exception swallow.

---

## 🟠 SEV-2 — DASHBOARD ENDPOINTS RETURNING `not_implemented` (4 sites)

[monitoring/aac_master_monitoring_dashboard.py](monitoring/aac_master_monitoring_dashboard.py) lines **1031, 1053, 1078, 1099** — four endpoints respond with literal `{"status": "not_implemented"}` to the dashboard UI. Users see a green response when the feature is dead.

---

## 🟠 SEV-2 — FAKE-DATA TRAINING PATH

[shared/strategy_parameter_tester.py:300-301](shared/strategy_parameter_tester.py#L300):
```python
entry_logic="pass",  # Placeholder
exit_logic="pass",   # Placeholder
```
Generated strategy objects have literal `pass` as their entry/exit logic — they will never trade or exit, but the optimiser still scores them.

---

## 🟡 SEV-3 — STUBS / `NotImplementedError` (38 sites)

### Legitimate abstract base classes — leave alone
- [TradingExecution/exchange_connectors/base_connector.py](TradingExecution/exchange_connectors/base_connector.py#L254) — `create_order`, `create_stop_loss_order`, `create_oco_order` (lines 254, 338, 375). Subclasses override.
- [shared/data_sources.py](shared/data_sources.py#L101) — `connect`/`disconnect` (101, 106).
- [shared/websocket_feeds.py](shared/websocket_feeds.py#L227) — `_connect`, `_subscribe`, `_handle_message` (227, 232, 237).
- [shared/market_data_connector.py:232](shared/market_data_connector.py#L232) — `subscribe_symbols`.
- [BigBrainIntelligence/agents.py:131](BigBrainIntelligence/agents.py#L131) — `scan()`.

### Documented Sprint-53/54 doctrine removals — intentional fail-fast
- [agents/agent_based_trading.py:207](agents/agent_based_trading.py#L207) — replaced mock data with NIE per "NO MOCK DATA OR CALLS" doctrine.
- [strategies/strategy_analysis_engine.py:453](strategies/strategy_analysis_engine.py#L453) — same.
- [strategies/strategy_testing_lab_fixed.py:219](strategies/strategy_testing_lab_fixed.py#L219) — same.

### **Real stubs without live callers** — convert to `pass` or delete
- [aac/integration/cross_department_engine.py](aac/integration/cross_department_engine.py#L102) — **8 NIE methods** (102, 106, 110, 114, 124, 128, 132, 136). No concrete subclass. Delete or wire.
- [shared/intelligence_infrastructure_bridge.py](shared/intelligence_infrastructure_bridge.py#L269) — **7 NIE methods** (269, 295, 302, 309, 316, 323, 330). Bridge with no caller.
- [shared/intelligence_accounting_bridge.py](shared/intelligence_accounting_bridge.py#L271) — **5 NIE methods** (271, 293, 300, 307, 314).
- [strategies/walk_forward_backtester.py](strategies/walk_forward_backtester.py#L104) — **2 NIE methods** (104, 108) — `_train_period` and `_test_period`. The class advertises walk-forward but cannot run.

### Connector advertised-but-not-built
- [TradingExecution/exchange_connectors/snaptrade_connector.py:209](TradingExecution/exchange_connectors/snaptrade_connector.py#L209) — "Trading via SnapTrade not implemented." Exposed in connector registry. Either gate it out of the registry or implement.

---

## 🟡 SEV-3 — PLACEHOLDERS & TODOs IN CODE PATHS (15 sites)

### Trading-path TODOs that affect behaviour
| File | Line | Note |
|------|------|------|
| [strategies/polymarket_division/active_scanner.py:525](strategies/polymarket_division/active_scanner.py#L525) | 525 | `# TODO: implement sell order execution via CLOB` — scanner cannot exit positions |
| [strategies/war_room_auto.py:820](strategies/war_room_auto.py#L820) | 820 | `before={"phase": "accumulation"}, # TODO: load previous` — phase transition logged with hardcoded prior |
| [divisions/trading/warroom/auto_update.py:759](divisions/trading/warroom/auto_update.py#L759) | 759 | Duplicate of above |
| [monitoring/aac_master_monitoring_dashboard.py:1650](monitoring/aac_master_monitoring_dashboard.py#L1650) | 1650 | `# TODO: Implement UW paper trading strategy and persist balance` |

### Placeholder modules / classes
| File | Line | Note |
|------|------|------|
| [BigBrainIntelligence/bigbrain_intelligence_advanced_state.py:464](BigBrainIntelligence/bigbrain_intelligence_advanced_state.py#L464) | 464 | `# Placeholder classes for components` |
| [divisions/research/agents/advanced_state.py:464](divisions/research/agents/advanced_state.py#L464) | 464 | Duplicate |
| [integrations/openclaw_skill_runtime_bridge.py:481](integrations/openclaw_skill_runtime_bridge.py#L481) | 481 | `"""Placeholder for skills not yet wired to runtime."""` |
| [integrations/aac_algotrading101_hub.py:299](integrations/aac_algotrading101_hub.py#L299) | 299 | `# Placeholder for actual Bing Chat integration` |
| [shared/market_data_connector.py:1598](shared/market_data_connector.py#L1598) | 1598 | `# For now, return empty list as placeholder` |
| [monitoring/continuous_monitoring.py:250](monitoring/continuous_monitoring.py#L250) | 250 | `# BigBrain Intelligence and Trading Execution - placeholder` |

### Cosmetic banners (low risk)
[integrations/openclaw_barren_wuffet_skills.py](integrations/openclaw_barren_wuffet_skills.py) lines 283-298, 653, 810, 814 and [integrations/openclaw_skills.py](integrations/openclaw_skills.py) lines 196-211, 530-532 — terminal banners with `$XXX,XXX.XX` placeholder strings rendered when no real value available. UI clarity issue, not a data issue.

---

## 🟡 SEV-3 — INTENTIONAL BYPASSES (verify semantics)

These bypass risk filters by design. Verify the bypass is correctly scoped.

| File | Line | Bypass |
|------|------|--------|
| [core/auto_trader.py:149](core/auto_trader.py#L149) | 149, 294, 479, 500 | `ROLL_CLOSE`/`STOP_CLOSE` signals bypass throttle and confidence threshold. **Intentional** — these MUST close. ✅ |
| [strategies/roll_signals.py:10](strategies/roll_signals.py#L10) | 10, 125 | Same — declares bypass | ✅ |
| [strategies/stop_signals.py:10](strategies/stop_signals.py#L10) | 10, 80, 103 | `confidence=1.0  # compulsory — bypass confidence filter` | ✅ |
| [TradingExecution/execution_engine.py:290](TradingExecution/execution_engine.py#L290) | 290 | `# Liquidity validation (can be disabled for paper trading)` — verify the disable path is gated by `paper=True` only. |

---

## 🟢 SEV-4 — NO-OPS (acknowledged, no action)

- All `--disable-*` chrome/edge browser flags in [agents/planktonxd_browser_bot.py](agents/planktonxd_browser_bot.py) and [agents/planktonxd_browser_bot_config.py](agents/planktonxd_browser_bot_config.py) — Selenium/Chromium tuning, not bypasses.
- All `placeholder=` kwargs in [monitoring/streamlit_dashboard.py](monitoring/streamlit_dashboard.py) and [strategies/strategy_metrics_dashboard.py](strategies/strategy_metrics_dashboard.py) — Streamlit input hint text, not data.
- `random.uniform` jitter in retry/backoff logic ([shared/utils.py:58](shared/utils.py#L58), [shared/data_sources.py:453](shared/data_sources.py#L453), [shared/websocket_feeds.py:104](shared/websocket_feeds.py#L104), [integrations/google_trends_client.py:105](integrations/google_trends_client.py#L105), [integrations/openclaw_gateway_bridge.py:339](integrations/openclaw_gateway_bridge.py#L339), [agents/planktonxd_browser_bot.py:921](agents/planktonxd_browser_bot.py#L921)) — legitimate retry-jitter; deterministic alternative would be worse.
- "Bridge hack risk", "synthetic stablecoin", "synthetic variance contract" — domain terminology in [divisions/trading/crypto/defi_yield.py:423](divisions/trading/crypto/defi_yield.py#L423), [aac/doctrine/ffd/ffd_engine.py:609](aac/doctrine/ffd/ffd_engine.py#L609), [strategies/variance_risk_premium.py:154](strategies/variance_risk_premium.py#L154) — not workarounds.

---

## Top-10 Fix List (priority order)

1. **Fence off the 11 RNG signal generators in `strategies/strategy_execution_engine.py`** with a `self.allow_synthetic_signals` guard (init from config; default `False`). Eliminates SEV-1 #1 in one PR.
2. **Delete the paper-order-simulator codegen template in `core/aac_automation_engine.py:428-432`** or move it under `paper_trading/templates/` with banner. Eliminates SEV-1 #13.
3. **Replace [monitoring/dashboard_pillars.py:224](monitoring/dashboard_pillars.py#L224) `call_delta=0.30` placeholder** with real value from `unusual_whales_client.get_greek_flow()` (already wired in Sprint 55) or omit the field with `available=False`.
4. **Decommission `shared/ai_strategy_generator.py` synthetic-training path** — quarantine under `_archive/` or guard with `if not config.get("allow_fake_training"): raise RuntimeError(...)`.
5. **Refactor `strategies/war_room_engine.py:2871,2894` `return 0.5/0.0` fallbacks** — return `None` and let callers handle missing data instead of inventing a confidence.
6. **Convert the bare `except:` in `agents/planktonxd_browser_bot.py`** (8 sites) to narrowed exceptions or `except Exception as e: log.debug(...)`.
7. **Convert the 20 NIE methods in the 3 bridge files** to either `pass`/return-empty stubs (with deprecation warning) or delete entirely if no caller exists.
8. **Implement or remove the 4 dashboard endpoints returning `"not_implemented"`** in [monitoring/aac_master_monitoring_dashboard.py:1031-1099](monitoring/aac_master_monitoring_dashboard.py#L1031).
9. **Replace `strategy_correlation = 0.3` placeholder** in [CentralAccounting/financial_analysis_engine.py:176](CentralAccounting/financial_analysis_engine.py#L176) with a real correlation lookup from `CorrelationTracker.last_snapshot` or refuse to size when unavailable.
10. **Implement Polymarket sell-order execution** at [strategies/polymarket_division/active_scanner.py:525](strategies/polymarket_division/active_scanner.py#L525) — scanner can open but not close positions.

---

## Coverage notes

- The pre-commit hook installed earlier today (`tools/check_forbidden_patterns.py`) blocks **new** introductions of bare `except Exception: pass` and `sys.path.insert`. Extending it to also flag bare `except:` and `random.(uniform|choice|random|gauss|expovariate)\(` outside `tests/` and `_scratch/` would catch **all SEV-1 and most SEV-2 regressions** automatically.
- `tests/` directory was excluded from this scan by design — random/synthetic data is expected there.
- `_archive/` and `_scratch/` excluded — dead and disposable.
- This audit is a snapshot. Re-run by:
  ```
  .venv\Scripts\python.exe tools\check_forbidden_patterns.py --check silent-except
  .venv\Scripts\python.exe tools\check_forbidden_patterns.py --check syspath
  ```
  plus the manual greps documented in the script header to refresh the SEV-1 list.
