# Entropy Controls Playbook — FRACTAL FUTURE Swarm

## Purpose
Define the exact protocol for entropy valve activations, drift detection, and kill-switch behavior across the swarm.

---

## Valve Definitions

### V1 — Entropy Level Rise (+1)
**Trigger:** Entropy moves from E(n) to E(n+1) in any domain
**Action:**
- Reduce scope/exposure by 25% in affected domain
- Markets: reduce position sizes, widen stops
- Human: reduce task count, extend deadlines
- System: reduce parallel cell operations
**Duration:** Until entropy returns to prior level for 2 consecutive readings
**Authority:** Entropy sentinel (automatic)

### V2 — Correlation / Coupling Breach
**Trigger:** Cross-asset correlation > 0.85 (markets) OR cross-cell error correlation > 0.5 (system)
**Action:**
- Cap gross exposure / parallel operations at 50% of normal
- Governance cell alerted immediately
**Duration:** Until correlation drops below threshold for 4 consecutive readings
**Authority:** Entropy sentinel (automatic)

### V3 — Liquidity / Attention Shock
**Trigger:** Gap risk > 2σ (markets) OR human attention entropy > threshold (3+ context switches in 10 min)
**Action:**
- Pause all non-essential cells (research, publishing, experiment)
- Only governance + sentinel remain active
- Markets: pause tight-stop strategies
**Duration:** Until shock subsides (sentinel determines)
**Authority:** Entropy sentinel (automatic, governance confirms)

### V4 — Kill Switch / Drawdown Breach
**Trigger:** Portfolio DD > max threshold OR system cascade detected OR human burnout signal
**Action:**
- ALL cells pause immediately
- All market positions move to close-only
- Mandatory cooldown period (24h minimum)
- Post-mortem required before restart
**Duration:** 24h minimum + governance cell sign-off to resume
**Authority:** Entropy sentinel (automatic), governance cell (to resume)

---

## Drift Detection

### What Is Drift?
Drift = gradual misalignment between doctrine and actual behavior. Examples:
- Strategy cells ignoring entropy regime
- Evidence grades inflated without experiment support
- Publishing E0/E1 material externally
- Skipping governance gates

### Detection Method
- Governance cell compares weekly artifacts against doctrine constraints
- Sentinel monitors for pattern: repeated valve trips without posture change
- Any cell can flag suspected drift (self-report encouraged)

### Response
1. Log DRIFT_INCIDENT report
2. Governance cell investigates
3. If confirmed: mandatory doctrine review + corrective action
4. If recurring: escalate to NCC

---

## Emergency Protocols

### Full Swarm Halt
**Trigger:** E5 in any domain, or sentinel failure
**Action:** All cells stop. No exceptions. Manual restart only after sentinel recovery + governance sign-off.

### Partial Halt
**Trigger:** E4 in one domain
**Action:** Affected-domain cells stop. Other domains continue at reduced scope (V1 applied).

### Sentinel Self-Check
- Sentinel runs self-diagnostic every 6 hours
- If self-check fails: all cells auto-pause
- Sentinel is the only cell that triggers cascade on failure (by design)
