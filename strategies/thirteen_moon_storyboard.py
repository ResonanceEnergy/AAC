"""
13-Moon Interactive Storyboard — HTML/JS Export
=================================================
Generates a self-contained interactive HTML storyboard with:
    - Horizontal timeline with moon cycles as segments
    - Color-coded event markers (astrology, phi, financial, world)
    - Click-to-expand event details with lead-time actions
    - Phi coherence wave overlay
    - Doctrine mandate banners per cycle
    - Current-date indicator
    - Mobile responsive
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/storyboard/thirteen_moon_storyboard.html"


def export_interactive_storyboard(
    doctrine: "ThirteenMoonDoctrine",
    output_path: str = DEFAULT_OUTPUT,
) -> str:
    """Export the full doctrine timeline as an interactive HTML storyboard."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Serialize all data for JS consumption
    timeline_data = _serialize_timeline(doctrine)
    timeline_json = json.dumps(timeline_data, indent=2, ensure_ascii=True)

    html = _TEMPLATE.replace("/*__TIMELINE_DATA__*/", f"const TIMELINE = {timeline_json};")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Interactive storyboard exported to %s", output_path)
    return output_path


def _serialize_timeline(doctrine: "ThirteenMoonDoctrine") -> dict:
    """Convert doctrine to JSON-serializable dict."""
    cycles = []
    for c in doctrine.moon_cycles:
        cycle = {
            "moon": c.moon_number,
            "start": c.start_date.isoformat(),
            "end": c.end_date.isoformat(),
            "name": c.lunar_phase_name,
            "fire_peak": c.fire_peak_date.isoformat() if c.fire_peak_date else None,
            "new_moon": c.new_moon_date.isoformat() if c.new_moon_date else None,
            "doctrine": None,
            "astrology": [],
            "phi": [],
            "financial": [],
            "world": [],
            "aac": [],
        }
        if c.doctrine_action:
            cycle["doctrine"] = {
                "mandate": c.doctrine_action.mandate,
                "desc": c.doctrine_action.description,
                "conviction": c.doctrine_action.conviction,
                "targets": c.doctrine_action.targets,
            }
        for e in c.astrology_events:
            cycle["astrology"].append({
                "date": e.date.isoformat(), "name": e.name,
                "category": e.category, "impact": e.impact,
                "desc": e.description, "vol_mult": e.volatility_mult,
                "sign": getattr(e, "zodiac_sign", ""),
            })
        for p in c.phi_markers:
            cycle["phi"].append({
                "date": p.date.isoformat(), "power": p.phi_power,
                "value": p.phi_value, "days": p.days_from_anchor,
                "label": p.label, "resonance": p.resonance_strength,
            })
        for f in c.financial_events:
            cycle["financial"].append({
                "date": f.date.isoformat(), "name": f.name,
                "category": f.category, "companies": f.companies,
                "impact": f.impact, "desc": f.description,
            })
        for w in c.world_events:
            cycle["world"].append({
                "date": w.date.isoformat(), "name": w.name,
                "category": w.category, "impact": w.impact,
                "desc": w.description,
            })
        for a in c.aac_events:
            cycle["aac"].append({
                "date": a.date.isoformat(), "name": a.name,
                "layer": a.layer, "category": a.category,
                "impact": a.impact, "desc": a.description,
                "assets": a.assets, "conviction": a.conviction,
                "thesis": a.thesis_relevance,
            })
        # Attach moon briefing if available
        from strategies.thirteen_moon_doctrine import MOON_BRIEFINGS
        briefing = MOON_BRIEFINGS.get(c.moon_number)
        if briefing:
            cycle["briefing"] = briefing

        # Attach sacred geometry for this moon
        from strategies.thirteen_moon_doctrine import SACRED_GEOMETRY_OVERLAY
        geo = SACRED_GEOMETRY_OVERLAY.get(c.moon_number)
        if geo:
            cycle["sacred_geometry"] = geo

        cycles.append(cycle)

    # Deep-dive data structures
    from strategies.thirteen_moon_doctrine import (
        AGE_OF_AQUARIUS,
        CRYPTO_DOCTRINE,
        DALIO_BIG_CYCLE,
        LEAPS_PLAYBOOK,
        SATURN_NEPTUNE_DEEPDIVE,
        WAR_ROOM_DOCTRINE,
    )

    return {
        "generated": date.today().isoformat(),
        "anchor": "2026-03-03",
        "phi": 1.6180339887,
        "today": date.today().isoformat(),
        "cycles": cycles,
        "all_events": doctrine.get_all_events_sorted(),
        "saturn_neptune": SATURN_NEPTUNE_DEEPDIVE,
        "age_of_aquarius": AGE_OF_AQUARIUS,
        "dalio_big_cycle": DALIO_BIG_CYCLE,
        "leaps_playbook": LEAPS_PLAYBOOK,
        "crypto_doctrine": CRYPTO_DOCTRINE,
        "war_room_doctrine": WAR_ROOM_DOCTRINE,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE (self-contained, no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>13-Moon Doctrine Timeline | AAC Resonance Energy</title>
<style>
:root {
  --bg: #0a0a14;
  --bg2: #12101f;
  --bg3: #1e1a2e;
  --gold: #c084fc;
  --gold-dim: #7c3aed;
  --silver: #94a3b8;
  --eclipse: #ef4444;
  --phi: #a78bfa;
  --financial: #34d399;
  --world: #f97316;
  --aac: #06b6d4;
  --aac-trade: #ec4899;
  --aac-war: #dc2626;
  --aac-seesaw: #84cc16;
  --aac-scenario: #8b5cf6;
  --aac-strategy: #14b8a6;
  --aac-milestone: #eab308;
  --aac-options: #f43f5e;
  --aac-auto: #6366f1;
  --aac-leaps: #d946ef;
  --geo: #818cf8;
  --aquarius: #38bdf8;
  --crypto: #f59e0b;
  --deploy: #22d3ee;
  --hold: #6b7280;
  --text: #e5e7eb;
  --text-dim: #9ca3af;
  --critical: #ef4444;
  --high: #c084fc;
  --medium: #3b82f6;
  --border: #2e2545;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  min-height: 100vh;
  overflow-x: hidden;
}
.header {
  text-align: center;
  padding: 2rem 1rem;
  background: linear-gradient(135deg, #0a0a14 0%, #1e1040 30%, #2d1b69 50%, #1e1040 70%, #0a0a14 100%);
  border-bottom: 2px solid var(--gold);
}
.header h1 {
  font-size: 1.8rem;
  color: var(--gold);
  letter-spacing: 2px;
  margin-bottom: 0.5rem;
}
.header .subtitle {
  color: var(--phi);
  font-size: 0.9rem;
}
.header .phi-display {
  color: var(--gold-dim);
  font-size: 0.8rem;
  margin-top: 0.3rem;
}
.controls {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  flex-wrap: wrap;
  padding: 1rem;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
}
.controls button {
  padding: 0.4rem 0.8rem;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.75rem;
  border-radius: 4px;
  transition: all 0.2s;
}
.controls button:hover, .controls button.active {
  background: var(--gold);
  color: var(--bg);
  border-color: var(--gold);
}
.today-banner {
  text-align: center;
  padding: 0.8rem;
  background: linear-gradient(90deg, transparent, rgba(192,132,252,0.12), transparent);
  font-size: 0.85rem;
}
.today-banner .moon-info { color: var(--gold); font-weight: bold; }
.today-banner .mandate { color: var(--deploy); }

/* Timeline container */
.timeline-scroll {
  overflow-x: auto;
  overflow-y: visible;
  padding: 1rem 0.5rem 2rem;
}
.timeline {
  display: flex;
  min-width: max-content;
  position: relative;
  padding: 3rem 1rem 1rem;
}
.timeline::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--gold-dim), var(--gold), var(--phi), var(--gold), var(--gold-dim));
  transform: translateY(-50%);
}
.moon-segment {
  position: relative;
  min-width: 220px;
  flex: 1;
  padding: 0 0.3rem;
}
.moon-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem;
  margin-top: 2.5rem;
  cursor: pointer;
  transition: all 0.3s;
  position: relative;
}
.moon-card:hover {
  border-color: var(--gold);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(192,132,252,0.2);
}
.moon-card.current {
  border-color: var(--gold);
  box-shadow: 0 0 15px rgba(192,132,252,0.35);
}
.moon-card .moon-number {
  position: absolute;
  top: -2rem;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg);
  border: 2px solid var(--gold);
  border-radius: 50%;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: bold;
  color: var(--gold);
}
.moon-card.current .moon-number {
  background: var(--gold);
  color: var(--bg);
}
.moon-card .moon-name {
  font-size: 0.7rem;
  color: var(--gold);
  text-align: center;
  margin-bottom: 0.3rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.moon-card .date-range {
  font-size: 0.6rem;
  color: var(--text-dim);
  text-align: center;
  margin-bottom: 0.4rem;
}
.moon-card .mandate-badge {
  display: inline-block;
  padding: 0.15rem 0.4rem;
  border-radius: 3px;
  font-size: 0.6rem;
  font-weight: bold;
  margin-bottom: 0.3rem;
}
.mandate-DEPLOY { background: #064e3b; color: #34d399; }
.mandate-ACCUMULATE { background: #1e3a5f; color: #60a5fa; }
.mandate-HOLD { background: #374151; color: #9ca3af; }
.mandate-REBALANCE { background: #4c1d95; color: #a78bfa; }
.mandate-EXIT_ROTATE { background: #7f1d1d; color: #fca5a5; }
.mandate-REVIEW { background: #713f12; color: #fcd34d; }
.event-dots {
  display: flex;
  gap: 3px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 0.3rem;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  cursor: pointer;
  transition: transform 0.2s;
}
.dot:hover { transform: scale(1.8); }
.dot.astrology { background: var(--eclipse); }
.dot.phi { background: var(--phi); }
.dot.financial { background: var(--financial); }
.dot.world { background: var(--world); }
.dot.aac { background: var(--aac); }
.dot.aac-trade { background: var(--aac-trade); }
.dot.aac-war_room { background: var(--aac-war); }
.dot.aac-seesaw { background: var(--aac-seesaw); }
.dot.aac-scenario { background: var(--aac-scenario); }
.dot.aac-strategy { background: var(--aac-strategy); }
.dot.aac-milestone { background: var(--aac-milestone); }
.dot.aac-options_lifecycle { background: var(--aac-options); }
.dot.aac-automation { background: var(--aac-auto); }
.dot.critical { box-shadow: 0 0 6px currentColor; }
.conviction-bar {
  height: 3px;
  background: var(--bg3);
  border-radius: 2px;
  margin-top: 0.4rem;
  overflow: hidden;
}
.conviction-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--gold-dim), var(--gold));
  transition: width 0.5s;
}

/* LEAPS layer */
.dot.aac-leaps { background: var(--aac-leaps); }
.layer-leaps { background: var(--aac-leaps); color: white; }

/* Crypto layer */
.dot.aac-crypto { background: var(--crypto); }
.layer-crypto { background: var(--crypto); color: #000; }

/* Sacred Geometry panel */
.geo-panel {
  border: 1px solid var(--geo);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(129,140,248,0.06);
  margin-bottom: 1rem;
}
.geo-panel h3 { color: var(--geo); }
.geo-freq {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 3px;
  background: rgba(129,140,248,0.15);
  color: var(--geo);
  font-size: 0.7rem;
  font-weight: bold;
}

/* LEAPS Playbook panel */
.leaps-panel {
  border: 1px solid var(--aac-leaps);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(217,70,239,0.06);
  margin-bottom: 1rem;
}
.leaps-panel h3 { color: var(--aac-leaps); }
.leaps-table { width: 100%; border-collapse: collapse; font-size: 0.65rem; margin-top: 0.5rem; }
.leaps-table th { text-align: left; padding: 0.3rem; border-bottom: 1px solid var(--border); color: var(--aac-leaps); }
.leaps-table td { padding: 0.3rem; border-bottom: 1px solid rgba(46,37,69,0.5); }

/* Aquarius panel */
.aquarius-panel {
  border: 1px solid var(--aquarius);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(56,189,248,0.06);
  margin-bottom: 1rem;
}
.aquarius-panel h3 { color: var(--aquarius); }

/* Dalio Big Cycle panel */
.dalio-panel {
  border: 1px solid var(--world);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(249,115,22,0.06);
  margin-bottom: 1rem;
}
.dalio-panel h3 { color: var(--world); }

/* Crypto Doctrine panel */
.crypto-panel {
  border: 1px solid var(--crypto);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(245,158,11,0.06);
  margin-bottom: 1rem;
}
.crypto-panel h3 { color: var(--crypto); }
.crypto-table { width: 100%; border-collapse: collapse; font-size: 0.65rem; margin-top: 0.5rem; }
.crypto-table th { text-align: left; padding: 0.3rem; border-bottom: 1px solid var(--border); color: var(--crypto); }
.crypto-table td { padding: 0.3rem; border-bottom: 1px solid rgba(46,37,69,0.5); }
.crypto-badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.6rem; font-weight: bold; }
.crypto-badge.active { background: rgba(34,197,94,0.2); color: #22c55e; }
.crypto-badge.paused { background: rgba(234,179,8,0.2); color: #eab308; }
.crypto-badge.liquidated { background: rgba(239,68,68,0.2); color: #ef4444; }
.crypto-badge.monitoring { background: rgba(96,165,250,0.2); color: #60a5fa; }

/* War Room Doctrine panel */
.war-room-panel {
  border: 1px solid var(--aac-war);
  border-radius: 6px;
  padding: 0.8rem;
  background: rgba(220,38,38,0.06);
  margin-bottom: 1rem;
}
.war-room-panel h3 { color: var(--aac-war); }

/* Detail Panel */
.detail-panel {
  position: fixed;
  top: 0;
  right: -500px;
  width: 480px;
  max-width: 90vw;
  height: 100vh;
  background: var(--bg2);
  border-left: 2px solid var(--gold);
  padding: 1.5rem;
  overflow-y: auto;
  transition: right 0.3s ease;
  z-index: 1000;
  scrollbar-color: var(--gold-dim) var(--bg);
}
.detail-panel.open { right: 0; }
.detail-panel .close-btn {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 1.5rem;
  cursor: pointer;
}
.detail-panel h2 {
  color: var(--gold);
  font-size: 1.1rem;
  margin-bottom: 1rem;
  padding-right: 2rem;
}
.detail-section {
  margin-bottom: 1.2rem;
}
.detail-section h3 {
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}
.detail-section h3.astro { color: var(--eclipse); }
.detail-section h3.phi-h { color: var(--phi); }
.detail-section h3.fin { color: var(--financial); }
.detail-section h3.world-h { color: var(--world); }
.detail-section h3.aac-h { color: var(--aac); }
.detail-section h3.doctrine-h { color: var(--deploy); }
.event-item {
  padding: 0.5rem;
  margin-bottom: 0.4rem;
  background: var(--bg3);
  border-radius: 4px;
  font-size: 0.75rem;
  border-left: 3px solid var(--border);
}
.event-item.impact-HIGH { border-left-color: var(--critical); }
.event-item.impact-MEDIUM { border-left-color: var(--medium); }
.event-item .event-date { color: var(--text-dim); font-size: 0.65rem; }
.event-item .event-name { font-weight: bold; margin: 0.2rem 0; }
.event-item .event-desc { color: var(--text-dim); font-size: 0.7rem; }
.event-item .companies { color: var(--financial); font-size: 0.65rem; }
.event-item .assets { color: var(--aac); font-size: 0.65rem; }
.event-item .thesis { color: var(--aac-scenario); font-size: 0.65rem; font-style: italic; }
.layer-tag {
  display: inline-block;
  padding: 0.1rem 0.3rem;
  border-radius: 2px;
  font-size: 0.55rem;
  font-weight: bold;
  margin-right: 0.3rem;
}
.layer-trade { background: var(--aac-trade); color: white; }
.layer-war_room { background: var(--aac-war); color: white; }
.layer-seesaw { background: var(--aac-seesaw); color: var(--bg); }
.layer-scenario { background: var(--aac-scenario); color: white; }
.layer-strategy { background: var(--aac-strategy); color: white; }
.layer-milestone { background: var(--aac-milestone); color: var(--bg); }
.layer-options_lifecycle { background: var(--aac-options); color: white; }
.layer-automation { background: var(--aac-auto); color: white; }
.phi-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.3rem;
}
.phi-bar .resonance-viz {
  flex: 1;
  height: 6px;
  background: var(--bg);
  border-radius: 3px;
  overflow: hidden;
}
.phi-bar .resonance-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--phi), #7c3aed);
}
.targets-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-top: 0.3rem;
}
.target-tag {
  padding: 0.15rem 0.4rem;
  background: var(--bg);
  border: 1px solid var(--gold-dim);
  border-radius: 3px;
  font-size: 0.65rem;
  color: var(--gold);
}

/* Phi wave overlay */
.phi-wave-container {
  padding: 0.5rem 1rem;
  background: var(--bg2);
  border-top: 1px solid var(--border);
}
.phi-wave-container h3 {
  color: var(--phi);
  font-size: 0.75rem;
  margin-bottom: 0.3rem;
  text-align: center;
}
#phi-wave {
  width: 100%;
  height: 60px;
}

/* Legend */
.legend {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  padding: 0.8rem;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  font-size: 0.7rem;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

/* Upcoming alerts */
.alerts-panel {
  padding: 1rem;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  display: none;
}
.alerts-panel.visible { display: block; }
.alerts-panel h3 {
  color: var(--gold);
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
}
.alert-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  margin-bottom: 0.3rem;
  background: var(--bg3);
  border-radius: 4px;
  font-size: 0.7rem;
}
.alert-item .days-badge {
  min-width: 40px;
  text-align: center;
  padding: 0.15rem 0.3rem;
  border-radius: 3px;
  font-weight: bold;
  font-size: 0.65rem;
}
.days-0 { background: var(--critical); color: white; }
.days-soon { background: var(--high); color: var(--bg); }
.days-later { background: var(--bg); color: var(--text-dim); border: 1px solid var(--border); }
.alert-item .alert-type {
  min-width: 60px;
  font-size: 0.6rem;
  color: var(--text-dim);
}
.alert-item .alert-action {
  color: var(--text-dim);
  font-size: 0.65rem;
  flex: 1;
}

.overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 999; }
.overlay.visible { display: block; }

/* Responsive */
@media (max-width: 768px) {
  .moon-segment { min-width: 160px; }
  .detail-panel { width: 100vw; }
  .header h1 { font-size: 1.2rem; }
}

/* Space Weather */
.sw-metric {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.8rem;
  min-width: 120px;
  flex: 1;
}
.sw-label { font-size: 0.65rem; color: var(--text-dim); margin-bottom: 0.2rem; }
.sw-value { font-size: 1.1rem; font-weight: bold; color: var(--gold); }
.sw-sub { font-size: 0.6rem; color: var(--text-dim); margin-top: 0.15rem; }
.sw-scale-badge {
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: bold;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 100px;
}
.sw-scale-badge .scale-label { font-size: 0.55rem; font-weight: normal; opacity: 0.7; }
</style>
</head>
<body>

<div class="header">
  <h1>13-MOON DOCTRINE TIMELINE</h1>
  <div class="subtitle">Resonance Energy -- AAC Compounding Calendar</div>
  <div class="phi-display">Anchor: March 3, 2026 Total Lunar Eclipse | phi = 1.6180339887</div>
</div>

<div class="controls">
  <button onclick="showAll()" class="active" id="btn-all">All Events</button>
  <button onclick="filterType('astrology')" id="btn-astrology">Astrology</button>
  <button onclick="filterType('phi')" id="btn-phi">Phi Coherence</button>
  <button onclick="filterType('financial')" id="btn-financial">Financial</button>
  <button onclick="filterType('world')" id="btn-world">World News</button>
  <button onclick="filterType('aac')" id="btn-aac">AAC Events</button>
  <button onclick="filterType('aac-trade')" id="btn-aac-trade">Trades</button>
  <button onclick="filterType('aac-war_room')" id="btn-aac-war_room">War Room</button>
  <button onclick="filterType('aac-options_lifecycle')" id="btn-aac-options_lifecycle">Options DTE</button>
  <button onclick="filterType('aac-scenario')" id="btn-aac-scenario">Scenarios</button>
  <button onclick="filterType('aac-seesaw')" id="btn-aac-seesaw">Seesaw</button>
  <button onclick="filterType('aac-strategy')" id="btn-aac-strategy">Strategies</button>
  <button onclick="filterType('aac-milestone')" id="btn-aac-milestone">Milestones</button>
  <button onclick="filterType('aac-leaps')" id="btn-aac-leaps">LEAPS</button>
  <button onclick="filterType('aac-crypto')" id="btn-aac-crypto">Crypto</button>
  <button onclick="toggleAlerts()" id="btn-alerts">Upcoming Alerts</button>
  <button onclick="scrollToCurrent()" id="btn-today">Today</button>
</div>

<div class="today-banner" id="today-banner"></div>

<div class="timeline-scroll">
  <div class="timeline" id="timeline"></div>
</div>

<div class="phi-wave-container">
  <h3>PHI COHERENCE WAVE (resonance strength over timeline)</h3>
  <canvas id="phi-wave"></canvas>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--eclipse)"></div> Astrology</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--phi)"></div> Phi Coherence</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--financial)"></div> Financial</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--world)"></div> World News</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--gold)"></div> Fire Peak</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-trade)"></div> Live Trades</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-war)"></div> War Room</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-options)"></div> Options DTE</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-scenario)"></div> Scenarios</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-seesaw)"></div> Seesaw</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-strategy)"></div> Strategies</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-milestone)"></div> Milestones</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-auto)"></div> Automation</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aac-leaps)"></div> LEAPS</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--geo)"></div> Sacred Geometry</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--aquarius)"></div> Aquarius</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--crypto)"></div> Crypto</div>
</div>

<!-- Space Weather Panel -->
<div id="space-weather-panel" style="padding:1rem;background:var(--bg2);border-top:1px solid var(--border)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
    <h3 style="color:#f59e0b;font-size:0.85rem;margin:0">&#9728;&#65039; SPACE WEATHER -- NOAA SWPC Live</h3>
    <span id="sw-updated" style="font-size:0.6rem;color:var(--text-dim)"></span>
  </div>
  <div style="font-size:0.65rem;color:var(--text-dim);margin-bottom:0.8rem">
    Solar cycle 25 peak (2024-2026) amplifies geomagnetic volatility &mdash; correlated with market sentiment shifts.
  </div>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.6rem">
    <div id="sw-kp" class="sw-metric">
      <div class="sw-label">&#129522; Kp Index</div>
      <div class="sw-value">--</div>
      <div class="sw-sub"></div>
    </div>
    <div id="sw-wind" class="sw-metric">
      <div class="sw-label">&#128168; Solar Wind</div>
      <div class="sw-value">--</div>
      <div class="sw-sub"></div>
    </div>
    <div id="sw-flux" class="sw-metric">
      <div class="sw-label">&#9728;&#65039; Solar Flux</div>
      <div class="sw-value">--</div>
      <div class="sw-sub"></div>
    </div>
    <div id="sw-ssn" class="sw-metric">
      <div class="sw-label">&#11088; Sunspot #</div>
      <div class="sw-value">--</div>
      <div class="sw-sub"></div>
    </div>
  </div>
  <div id="sw-scales" style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem"></div>
  <div id="sw-alert" style="display:none;padding:0.5rem 0.8rem;border-radius:4px;font-size:0.7rem;margin-top:0.4rem"></div>
</div>

<div class="alerts-panel" id="alerts-panel"></div>

<div class="overlay" id="overlay" onclick="closeDetail()"></div>
<div class="detail-panel" id="detail-panel">
  <button class="close-btn" onclick="closeDetail()">&times;</button>
  <div id="detail-content"></div>
</div>

<script>
/*__TIMELINE_DATA__*/

// State
let activeFilter = 'all';
let alertsVisible = false;

function parseDate(s) { return new Date(s + 'T00:00:00'); }
function fmtDate(s) {
  const d = parseDate(s);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}
function daysBetween(a, b) {
  return Math.round((parseDate(b) - parseDate(a)) / 86400000);
}

// Build timeline
function buildTimeline() {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  const today = TIMELINE.today;

  TIMELINE.cycles.forEach(c => {
    const seg = document.createElement('div');
    seg.className = 'moon-segment';
    seg.dataset.moon = c.moon;

    const isCurrent = today >= c.start && today <= c.end;
    const card = document.createElement('div');
    card.className = 'moon-card' + (isCurrent ? ' current' : '');
    card.onclick = () => openDetail(c);

    // Moon number circle
    const num = document.createElement('div');
    num.className = 'moon-number';
    num.textContent = c.moon;
    card.appendChild(num);

    // Name
    const name = document.createElement('div');
    name.className = 'moon-name';
    name.textContent = c.name;
    card.appendChild(name);

    // Date range
    const dateRange = document.createElement('div');
    dateRange.className = 'date-range';
    dateRange.textContent = fmtDate(c.start) + ' - ' + fmtDate(c.end);
    card.appendChild(dateRange);

    // Mandate badge
    if (c.doctrine) {
      const badge = document.createElement('div');
      badge.className = 'mandate-badge mandate-' + c.doctrine.mandate;
      badge.textContent = c.doctrine.mandate;
      card.appendChild(badge);
    }

    // Event dots
    const dots = document.createElement('div');
    dots.className = 'event-dots';

    const addDots = (events, type) => {
      events.forEach(e => {
        if (activeFilter !== 'all' && activeFilter !== type) return;
        const dot = document.createElement('div');
        dot.className = 'dot ' + type;
        if (e.impact === 'HIGH') dot.classList.add('critical');
        dot.title = (e.name || e.label) + ' (' + fmtDate(e.date) + ')';
        dots.appendChild(dot);
      });
    };

    addDots(c.astrology, 'astrology');
    addDots(c.phi, 'phi');
    addDots(c.financial, 'financial');
    addDots(c.world, 'world');
    // AAC sub-layer dots
    if (c.aac && c.aac.length) {
      c.aac.forEach(e => {
        const subType = 'aac-' + e.layer;
        if (activeFilter !== 'all' && activeFilter !== 'aac' && activeFilter !== subType) return;
        const dot = document.createElement('div');
        dot.className = 'dot ' + subType;
        if (e.impact === 'HIGH') dot.classList.add('critical');
        dot.title = '[' + e.layer.toUpperCase() + '] ' + e.name + ' (' + fmtDate(e.date) + ')';
        dots.appendChild(dot);
      });
    }
    card.appendChild(dots);

    // Conviction bar
    if (c.doctrine) {
      const bar = document.createElement('div');
      bar.className = 'conviction-bar';
      const fill = document.createElement('div');
      fill.className = 'conviction-fill';
      fill.style.width = (c.doctrine.conviction * 100) + '%';
      bar.appendChild(fill);
      card.appendChild(bar);
    }

    seg.appendChild(card);
    tl.appendChild(seg);
  });

  // Today banner
  updateTodayBanner(today);
}

function updateTodayBanner(today) {
  const banner = document.getElementById('today-banner');
  let currentCycle = null;
  TIMELINE.cycles.forEach(c => {
    if (today >= c.start && today <= c.end) currentCycle = c;
  });
  if (currentCycle) {
    const daysIn = daysBetween(currentCycle.start, today);
    const totalDays = daysBetween(currentCycle.start, currentCycle.end);
    banner.innerHTML = '<span class="moon-info">CURRENT: Moon ' + currentCycle.moon +
      ' (' + currentCycle.name + ') | Day ' + daysIn + '/' + totalDays + '</span>' +
      (currentCycle.doctrine ? ' | <span class="mandate">MANDATE: ' +
      currentCycle.doctrine.mandate + '</span>' : '');
  } else {
    banner.innerHTML = '<span class="moon-info">Outside 13-Moon doctrine window</span>';
  }
}

function openDetail(cycle) {
  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  const overlay = document.getElementById('overlay');

  let html = '<h2>Moon ' + cycle.moon + ' -- ' + cycle.name + '</h2>';
  html += '<div style="font-size:0.75rem;color:var(--text-dim);margin-bottom:1rem">' +
    fmtDate(cycle.start) + ' - ' + fmtDate(cycle.end) + '</div>';

  // Doctrine
  if (cycle.doctrine) {
    html += '<div class="detail-section">';
    html += '<h3 class="doctrine-h">DOCTRINE MANDATE</h3>';
    html += '<div class="event-item"><div class="event-name">' + cycle.doctrine.mandate +
      ' (Conviction: ' + Math.round(cycle.doctrine.conviction * 100) + '%)</div>' +
      '<div class="event-desc">' + cycle.doctrine.desc + '</div>';
    if (cycle.doctrine.targets.length) {
      html += '<div class="targets-list">';
      cycle.doctrine.targets.forEach(t => {
        html += '<span class="target-tag">' + t + '</span>';
      });
      html += '</div>';
    }
    html += '</div></div>';
  }

  // Fire Peak
  if (cycle.fire_peak) {
    html += '<div class="detail-section">';
    html += '<h3 style="color:var(--gold)">FIRE PEAK</h3>';
    html += '<div class="event-item impact-HIGH"><div class="event-date">' +
      fmtDate(cycle.fire_peak) + '</div><div class="event-name">Full Moon Fire Peak</div>' +
      '<div class="event-desc">High-volatility window. Deploy per mandate.</div></div>';
    html += '</div>';
  }

  // Moon Briefing
  if (cycle.briefing) {
    const b = cycle.briefing;
    html += '<div class="detail-section">';
    html += '<h3 style="color:var(--phi)">MOON BRIEFING: ' + (b.theme || '') + '</h3>';
    html += '<div class="event-item">';
    if (b.lunar) html += '<div style="margin-bottom:0.4rem"><strong style="color:var(--gold)">Lunar:</strong> ' + b.lunar + '</div>';
    if (b.astro_highlights) html += '<div style="margin-bottom:0.4rem"><strong style="color:var(--eclipse)">Astro Highlights:</strong> ' + b.astro_highlights + '</div>';
    if (b.market_implication) html += '<div style="margin-bottom:0.4rem"><strong style="color:var(--financial)">Market Implication:</strong> ' + b.market_implication + '</div>';
    if (b.empirical) html += '<div style="font-size:0.65rem;color:var(--text-dim);font-style:italic">' + b.empirical + '</div>';
    html += '</div></div>';
  }

  // Astrology
  if (cycle.astrology.length) {
    html += '<div class="detail-section">';
    html += '<h3 class="astro">ASTROLOGY (' + cycle.astrology.length + ')</h3>';
    cycle.astrology.sort((a,b) => a.date.localeCompare(b.date)).forEach(e => {
      html += '<div class="event-item impact-' + e.impact + '">' +
        '<div class="event-date">' + fmtDate(e.date) + ' | Vol mult: ' + e.vol_mult.toFixed(2) + 'x' +
        (e.sign ? ' | <span style="color:var(--phi)">' + e.sign + '</span>' : '') + '</div>' +
        '<div class="event-name">' + e.name + '</div>' +
        '<div class="event-desc">' + e.desc + '</div></div>';
    });
    html += '</div>';
  }

  // Phi
  if (cycle.phi.length) {
    html += '<div class="detail-section">';
    html += '<h3 class="phi-h">PHI COHERENCE (' + cycle.phi.length + ')</h3>';
    cycle.phi.forEach(p => {
      html += '<div class="event-item"><div class="event-date">' + fmtDate(p.date) +
        ' | +' + p.days.toFixed(1) + ' days from anchor</div>' +
        '<div class="event-name">phi^' + p.power + ' = ' + p.value.toFixed(4) + '</div>' +
        '<div class="phi-bar"><span style="font-size:0.65rem;color:var(--phi)">Resonance</span>' +
        '<div class="resonance-viz"><div class="resonance-fill" style="width:' +
        Math.round(p.resonance * 100) + '%"></div></div>' +
        '<span style="font-size:0.65rem">' + Math.round(p.resonance * 100) + '%</span></div></div>';
    });
    html += '</div>';
  }

  // Financial
  if (cycle.financial.length) {
    html += '<div class="detail-section">';
    html += '<h3 class="fin">FINANCIAL (' + cycle.financial.length + ')</h3>';
    cycle.financial.sort((a,b) => a.date.localeCompare(b.date)).forEach(e => {
      html += '<div class="event-item impact-' + e.impact + '">' +
        '<div class="event-date">' + fmtDate(e.date) + ' | ' + e.category + '</div>' +
        '<div class="event-name">' + e.name + '</div>' +
        (e.companies.length ? '<div class="companies">' + e.companies.join(', ') + '</div>' : '') +
        '<div class="event-desc">' + e.desc + '</div></div>';
    });
    html += '</div>';
  }

  // World
  if (cycle.world.length) {
    html += '<div class="detail-section">';
    html += '<h3 class="world-h">WORLD NEWS (' + cycle.world.length + ')</h3>';
    cycle.world.sort((a,b) => a.date.localeCompare(b.date)).forEach(e => {
      html += '<div class="event-item impact-' + e.impact + '">' +
        '<div class="event-date">' + fmtDate(e.date) + '</div>' +
        '<div class="event-name">' + e.name + '</div>' +
        '<div class="event-desc">' + e.desc + '</div></div>';
    });
    html += '</div>';
  }

  // AAC Events
  if (cycle.aac && cycle.aac.length) {
    html += '<div class="detail-section">';
    html += '<h3 class="aac-h">AAC SYSTEM EVENTS (' + cycle.aac.length + ')</h3>';
    const layerOrder = ['trade','options_lifecycle','war_room','scenario','seesaw','strategy','milestone','automation','leaps','crypto'];
    const layerLabels = {trade:'LIVE TRADES',options_lifecycle:'OPTIONS DTE',war_room:'WAR ROOM',scenario:'SCENARIOS',seesaw:'SEESAW ROTATION',strategy:'STRATEGIES',milestone:'MILESTONES',automation:'AUTOMATION',leaps:'LEAPS PLAYBOOK',crypto:'CRYPTO DOCTRINE'};
    layerOrder.forEach(layer => {
      const items = cycle.aac.filter(e => e.layer === layer);
      if (!items.length) return;
      html += '<div style="margin:0.4rem 0 0.2rem;font-size:0.7rem;font-weight:bold;color:var(--aac-' + (layer === 'options_lifecycle' ? 'options' : layer === 'war_room' ? 'war' : layer === 'automation' ? 'auto' : layer) + ')">' + layerLabels[layer] + '</div>';
      items.sort((a,b) => a.date.localeCompare(b.date)).forEach(e => {
        html += '<div class="event-item impact-' + e.impact + '">';
        html += '<div class="event-date">' + fmtDate(e.date) + ' | ' + e.category + '</div>';
        html += '<span class="layer-tag layer-' + e.layer + '">' + e.layer + '</span>';
        html += '<div class="event-name">' + e.name + '</div>';
        if (e.assets && e.assets.length) html += '<div class="assets">' + e.assets.join(', ') + '</div>';
        html += '<div class="event-desc">' + e.desc + '</div>';
        if (e.thesis) html += '<div class="thesis">Thesis: ' + e.thesis + '</div>';
        if (e.conviction) html += '<div style="font-size:0.6rem;color:var(--text-dim)">Conviction: ' + Math.round(e.conviction * 100) + '%</div>';
        html += '</div>';
      });
    });
    html += '</div>';
  }

  // Sacred Geometry
  if (cycle.sacred_geometry) {
    const g = cycle.sacred_geometry;
    html += '<div class="geo-panel">';
    html += '<h3>SACRED GEOMETRY: ' + g.geometry + '</h3>';
    html += '<div style="margin:0.4rem 0"><span class="geo-freq">' + g.frequency_hz + ' Hz -- ' + g.frequency_name + '</span></div>';
    html += '<div class="event-desc" style="margin-bottom:0.4rem">' + g.description + '</div>';
    if (g.platonic_solid) html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--geo)">Platonic Solid:</strong> ' + g.platonic_solid + '</div>';
    if (g.angle_sum) html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--geo)">Angle Sum:</strong> ' + g.angle_sum + ' degrees</div>';
    if (g.vertices) html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--geo)">Vertices:</strong> ' + g.vertices + '</div>';
    html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--phi)">Phi Link:</strong> ' + g.phi_link + '</div>';
    html += '<div style="font-size:0.7rem;color:var(--gold);font-weight:bold;margin-top:0.5rem;padding-top:0.4rem;border-top:1px solid var(--border)">Correlation: ' + g.correlation + '</div>';
    html += '</div>';
  }

  // Saturn-Neptune Deep Dive (Moon 12 only)
  if (cycle.moon === 12 && TIMELINE.saturn_neptune) {
    const sn = TIMELINE.saturn_neptune;
    html += '<div class="detail-section" style="border:1px solid var(--phi);border-radius:6px;padding:0.8rem;background:rgba(167,139,250,0.08)">';
    html += '<h3 style="color:var(--phi)">SATURN-NEPTUNE CONJUNCTION DEEP DIVE</h3>';
    html += '<div style="font-size:0.7rem;color:var(--gold);margin-bottom:0.5rem">' + sn.date + ' | ' + sn.degree + ' | ' + sn.cycle_years + '-year cycle</div>';
    html += '<div class="event-desc" style="margin-bottom:0.5rem">' + sn.significance + '</div>';
    html += '<div style="margin-bottom:0.5rem"><strong style="color:var(--silver)">Saturn:</strong> ' + sn.saturn_themes + '</div>';
    html += '<div style="margin-bottom:0.5rem"><strong style="color:var(--phi)">Neptune:</strong> ' + sn.neptune_themes + '</div>';
    html += '<div style="margin-bottom:0.5rem"><strong style="color:var(--gold)">Blend:</strong> ' + sn.blend + '</div>';
    if (sn.historical_parallels) {
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--eclipse);margin-bottom:0.3rem">HISTORICAL PARALLELS</div>';
      sn.historical_parallels.forEach(p => {
        html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong>' + p.year + ' (' + p.sign + '):</strong> ' + p.events + '</div>';
      });
      html += '</div>';
    }
    if (sn['2027_predictions']) {
      const pred = sn['2027_predictions'];
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--financial);margin-bottom:0.3rem">2027 PREDICTIONS</div>';
      Object.entries(pred).forEach(([k,v]) => {
        html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong style="text-transform:uppercase">' + k.replace(/_/g,' ') + ':</strong> ' + v + '</div>';
      });
      html += '</div>';
    }
    if (sn.doctrine_action) html += '<div style="margin-top:0.5rem;font-size:0.7rem;color:var(--gold);font-weight:bold">' + sn.doctrine_action + '</div>';
    html += '</div>';
  }

  // Age of Aquarius Deep Dive (Moon 0)
  if (cycle.moon === 0 && TIMELINE.age_of_aquarius) {
    const aq = TIMELINE.age_of_aquarius;
    html += '<div class="aquarius-panel">';
    html += '<h3>AGE OF AQUARIUS DEEP DIVE</h3>';
    html += '<div class="event-desc" style="margin-bottom:0.5rem">' + aq.phenomenon + '</div>';
    html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--aquarius)">Great Year:</strong> ' + aq.great_year + '</div>';
    html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--aquarius)">Transition:</strong> ' + aq.current_transition + '</div>';
    if (aq.sign_qualities) {
      html += '<div style="margin:0.5rem 0;padding:0.4rem;background:rgba(56,189,248,0.08);border-radius:4px;font-size:0.65rem">';
      html += '<strong style="color:var(--aquarius)">Element:</strong> ' + aq.sign_qualities.element + '<br>';
      html += '<strong style="color:var(--aquarius)">Rulers:</strong> ' + aq.sign_qualities.rulers + '<br>';
      html += '<strong style="color:var(--aquarius)">Key Phrase:</strong> <em>' + aq.sign_qualities.key_phrase + '</em><br>';
      html += '<strong style="color:var(--aquarius)">Balance:</strong> ' + aq.sign_qualities.opposite + '</div>';
    }
    if (aq.core_themes) {
      html += '<div style="margin-top:0.4rem;font-size:0.65rem"><strong style="color:var(--aquarius)">Core Themes:</strong><ul style="margin:0.2rem 0 0 1rem">';
      aq.core_themes.forEach(t => { html += '<li>' + t + '</li>'; });
      html += '</ul></div>';
    }
    if (aq.shadow_themes) {
      html += '<div style="margin-top:0.4rem;font-size:0.65rem"><strong style="color:var(--eclipse)">Shadow Themes:</strong><ul style="margin:0.2rem 0 0 1rem">';
      aq.shadow_themes.forEach(t => { html += '<li>' + t + '</li>'; });
      html += '</ul></div>';
    }
    html += '<div style="font-size:0.65rem;margin-top:0.4rem"><strong style="color:var(--phi)">Element Shift:</strong> ' + aq.element_shift + '</div>';
    if (aq.amplifying_transits_2026) {
      html += '<div style="margin-top:0.4rem;font-size:0.65rem"><strong style="color:var(--gold)">2026 Amplifying Transits:</strong><ul style="margin:0.2rem 0 0 1rem">';
      aq.amplifying_transits_2026.forEach(t => { html += '<li>' + t + '</li>'; });
      html += '</ul></div>';
    }
    if (aq.doctrine_alignment) {
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aquarius);margin-bottom:0.3rem">DOCTRINE ALIGNMENT</div>';
      Object.entries(aq.doctrine_alignment).forEach(([k,v]) => {
        html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong style="text-transform:uppercase;color:var(--gold)">' + k.replace(/_/g,' ') + ':</strong> ' + v + '</div>';
      });
      html += '</div>';
    }
    html += '</div>';
  }

  // LEAPS Playbook (Moon 1 entry)
  if (cycle.moon === 1 && TIMELINE.leaps_playbook) {
    const lp = TIMELINE.leaps_playbook;
    html += '<div class="leaps-panel">';
    html += '<h3>LEAPS PLAYBOOK -- $' + (lp.total_capital/1000).toFixed(0) + 'k DEPLOYMENT</h3>';
    html += '<div style="font-size:0.7rem;color:var(--aac-leaps);margin-bottom:0.3rem">' + lp.entry_window + '</div>';
    html += '<div class="event-desc" style="margin-bottom:0.5rem">Strategy: ' + lp.strategy + '</div>';
    html += '<div class="event-desc" style="margin-bottom:0.5rem">Conviction: ' + lp.conviction + '</div>';
    if (lp.positions) {
      html += '<table class="leaps-table"><tr><th>Position</th><th>%</th><th>$</th><th>Strike</th><th>Contracts</th></tr>';
      Object.entries(lp.positions).forEach(([k,p]) => {
        html += '<tr><td style="color:var(--gold)">' + p.ticker + '</td><td>' + p.allocation_pct + '%</td>';
        html += '<td>$' + (p.amount/1000).toFixed(1) + 'k</td><td style="font-size:0.6rem">' + p.strike + '</td>';
        html += '<td>' + p.contracts + '</td></tr>';
      });
      html += '</table>';
  }
    if (lp.exit_rotation_dates) {
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-leaps);margin-bottom:0.3rem">EXIT/ROTATION SCHEDULE</div>';
      Object.entries(lp.exit_rotation_dates).forEach(([k,r]) => {
        html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong style="color:var(--gold)">' + k.replace(/_/g,' ').toUpperCase() + ' (' + r.date + '):</strong> ' + r.action + '</div>';
      });
      html += '</div>';
    }
    if (lp.risk_rules) {
      html += '<div style="margin-top:0.4rem;font-size:0.65rem"><strong style="color:var(--eclipse)">Risk Rules:</strong><ul style="margin:0.2rem 0 0 1rem">';
      lp.risk_rules.forEach(r => { html += '<li>' + r + '</li>'; });
      html += '</ul></div>';
    }
    if (lp.daily_scrape_march_28) {
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--financial);margin-bottom:0.3rem">MARKET SCRAPE (Mar 28, 2026)</div>';
      Object.entries(lp.daily_scrape_march_28).forEach(([k,v]) => {
        html += '<div style="margin-bottom:0.2rem;font-size:0.65rem"><strong style="text-transform:uppercase;color:var(--silver)">' + k.replace(/_/g,' ') + ':</strong> ' + v + '</div>';
      });
      html += '</div>';
    }
    html += '</div>';
  }

  // Crypto Doctrine — per-moon outlook + full overview on Moon 2
  if (TIMELINE.crypto_doctrine) {
    const cr = TIMELINE.crypto_doctrine;
    const mo = cr.moon_phase_outlook && cr.moon_phase_outlook[cycle.moon];
    if (mo) {
      html += '<div class="crypto-panel">';
      html += '<h3>CRYPTO OUTLOOK -- Moon ' + cycle.moon + '</h3>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.4rem">';
      html += '<span class="crypto-badge ' + (mo.risk === 'LOW' ? 'active' : mo.risk === 'CRITICAL' ? 'liquidated' : mo.risk === 'HIGH' ? 'paused' : 'monitoring') + '">' + mo.risk + ' RISK</span>';
      html += '<span style="font-size:0.7rem;color:var(--crypto);font-weight:bold">' + mo.regime + '</span></div>';
      html += '<div class="event-desc" style="margin-bottom:0.3rem"><strong>Action:</strong> ' + mo.action + '</div>';
      html += '<div style="font-size:0.65rem;color:var(--text-dim);font-style:italic">' + mo.note + '</div>';
      html += '</div>';
    }
    // Full crypto overview on Moon 2 (first active moon after NDAX liquidation)
    if (cycle.moon === 2) {
      html += '<div class="crypto-panel">';
      html += '<h3>CRYPTO DOCTRINE -- FULL OVERVIEW</h3>';
      html += '<div class="event-desc" style="margin-bottom:0.5rem">' + cr.thesis + '</div>';
      html += '<div style="font-size:0.65rem;margin-bottom:0.4rem"><strong style="color:var(--crypto)">Current Regime:</strong> ' + cr.regime_current + '</div>';
      // Positions
      if (cr.positions) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--crypto);margin-bottom:0.3rem">POSITIONS</div>';
        Object.entries(cr.positions).forEach(([k,p]) => {
          const badge = p.status === 'LIQUIDATED' ? 'liquidated' : p.status === 'ACTIVE' ? 'active' : 'monitoring';
          html += '<div style="margin-bottom:0.4rem;font-size:0.65rem"><strong style="color:var(--gold)">' + k.toUpperCase() + '</strong> <span class="crypto-badge ' + badge + '">' + p.status + '</span>';
          if (p.proceeds_cad) html += ' <span style="color:var(--financial)">$' + p.proceeds_cad.toLocaleString() + ' CAD</span>';
          if (p.sold) html += '<br>Sold: ' + p.sold.join(', ');
          if (p.strategy) html += '<br>' + p.strategy;
          if (p.reason) html += '<br><em>' + p.reason + '</em>';
          if (p.note) html += '<br><em>' + p.note + '</em>';
          html += '</div>';
        });
        html += '</div>';
      }
      // Watchlist
      if (cr.watchlist) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--crypto);margin-bottom:0.3rem">WATCHLIST</div>';
        html += '<table class="crypto-table"><tr><th>Ticker</th><th>Price</th><th>Moon</th><th>Thesis</th></tr>';
        Object.entries(cr.watchlist).forEach(([k,w]) => {
          html += '<tr><td style="color:var(--crypto);font-weight:bold">' + w.ticker + '</td>';
          html += '<td>' + w.current_price + '</td>';
          html += '<td style="font-size:0.6rem">' + w.moon_affinity + '</td>';
          html += '<td style="font-size:0.6rem">' + w.thesis.substring(0, 120) + '...</td></tr>';
        });
        html += '</table></div>';
      }
      // Active Strategies
      if (cr.strategies_active) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--crypto);margin-bottom:0.3rem">ACTIVE STRATEGIES</div>';
        Object.entries(cr.strategies_active).forEach(([k,s]) => {
          const badge = s.status.includes('ACTIVE') ? 'active' : s.status.includes('PAUSED') ? 'paused' : 'monitoring';
          html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong style="color:var(--gold)">' + s.name + '</strong> <span class="crypto-badge ' + badge + '">' + s.status + '</span></div>';
        });
        html += '</div>';
      }
      // Risk Framework
      if (cr.risk_framework) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--eclipse);margin-bottom:0.3rem">RISK FRAMEWORK</div>';
        html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong>Max Allocation:</strong> ' + cr.risk_framework.max_crypto_allocation + '</div>';
        if (cr.risk_framework.re_entry_triggers) {
          html += '<div style="font-size:0.65rem"><strong style="color:var(--crypto)">Re-Entry Triggers:</strong><ul style="margin:0.2rem 0 0 1rem">';
          cr.risk_framework.re_entry_triggers.forEach(t => { html += '<li>' + t + '</li>'; });
          html += '</ul></div>';
        }
        html += '<div style="font-size:0.65rem;margin-top:0.2rem"><strong style="color:var(--eclipse)">Stop Loss:</strong> ' + cr.risk_framework.stop_loss + '</div>';
        html += '</div>';
      }
      html += '</div>';
    }
  }

  // Dalio Big Cycle (Moon 7 harvest/equinox)
  if (cycle.moon === 7 && TIMELINE.dalio_big_cycle) {
    const dl = TIMELINE.dalio_big_cycle;
    html += '<div class="dalio-panel">';
    html += '<h3>DALIO BIG CYCLE FRAMEWORK</h3>';
    html += '<div class="event-desc" style="margin-bottom:0.5rem">' + dl.framework + '</div>';
    if (dl.five_forces) {
      html += '<div style="font-size:0.65rem;margin-bottom:0.4rem"><strong style="color:var(--world)">Five Forces:</strong><ol style="margin:0.2rem 0 0 1.2rem">';
      dl.five_forces.forEach(f => { html += '<li>' + f + '</li>'; });
      html += '</ol></div>';
    }
    if (dl.stages) {
      html += '<div style="margin:0.4rem 0;padding:0.4rem;background:rgba(249,115,22,0.08);border-radius:4px;font-size:0.65rem">';
      html += '<strong style="color:var(--world)">Stages:</strong><br>';
      Object.entries(dl.stages).forEach(([k,v]) => {
        html += '<div style="margin:0.2rem 0"><strong>' + k.replace(/_/g,' ').toUpperCase() + ':</strong> ' + v + '</div>';
      });
      html += '</div>';
    }
    html += '<div style="font-size:0.7rem;color:var(--eclipse);font-weight:bold;margin:0.4rem 0">Current (2026): ' + dl.current_position_2026 + '</div>';
    if (dl.bridgewater_portfolio_2026) {
      const bw = dl.bridgewater_portfolio_2026;
      html += '<div style="font-size:0.65rem;margin-bottom:0.4rem"><strong style="color:var(--financial)">Bridgewater:</strong> AUM ' + bw.aum + ', ' + bw.holdings + ' holdings, ' + bw.turnover + ' turnover<br>Top: ' + bw.top_positions + '<br><em>' + bw.note + '</em></div>';
    }
    if (dl.doctrine_alignment) {
      html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
      html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--world);margin-bottom:0.3rem">DOCTRINE ALIGNMENT</div>';
      Object.entries(dl.doctrine_alignment).forEach(([k,v]) => {
        html += '<div style="margin-bottom:0.3rem;font-size:0.65rem"><strong style="text-transform:uppercase;color:var(--gold)">' + k.replace(/_/g,' ') + ':</strong> ' + v + '</div>';
      });
      html += '</div>';
    }
    html += '</div>';
  }

  // ── WAR ROOM DOCTRINE ──
  if (TIMELINE.war_room_doctrine) {
    const wr = TIMELINE.war_room_doctrine;
    // Per-moon war room outlook (every moon)
    const wmo = wr.per_moon_outlook && wr.per_moon_outlook[cycle.moon];
    if (wmo) {
      html += '<div class="war-room-panel">';
      html += '<h3>WAR ROOM -- Moon ' + cycle.moon + '</h3>';
      html += '<div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.4rem">';
      const riskColor = wmo.risk === 'RED' || wmo.risk === 'BLACK' ? '#ef4444' : wmo.risk === 'ORANGE' ? '#f97316' : wmo.risk === 'YELLOW' ? '#eab308' : '#22c55e';
      html += '<span style="background:' + riskColor + ';color:#000;padding:2px 8px;border-radius:3px;font-size:0.6rem;font-weight:bold">' + wmo.risk + '</span>';
      html += '<span style="font-size:0.65rem;font-weight:bold;color:var(--aac-war)">' + wmo.regime + '</span>';
      html += '<span style="font-size:0.6rem;color:var(--text-dim)">Pressure: ' + wmo.pressure + '</span>';
      html += '</div>';
      html += '<div class="event-desc">' + wmo.action + '</div>';
      html += '</div>';
    }
    // Full War Room overview on Moon 3 (first full war month, $19.8K injection)
    if (cycle.moon === 3) {
      html += '<div class="war-room-panel">';
      html += '<h3>WAR ROOM DOCTRINE -- FULL OVERVIEW</h3>';
      html += '<div class="event-desc" style="margin-bottom:0.5rem;color:var(--aac-war);font-weight:bold">' + wr.thesis + '</div>';
      html += '<div style="font-size:0.65rem;margin-bottom:0.4rem"><strong style="color:var(--silver)">War Start:</strong> ' + wr.war_start + ' | <strong>Window:</strong> ' + wr.model_window.days + ' days (' + wr.model_window.start + ' to ' + wr.model_window.end + ')</div>';
      // Composite score
      if (wr.composite_score) {
        const cs = wr.composite_score;
        const pct = cs.current;
        const barColor = pct > 70 ? '#ef4444' : pct > 50 ? '#f97316' : pct > 30 ? '#eab308' : '#22c55e';
        html += '<div style="margin:0.5rem 0;padding:0.5rem;background:rgba(220,38,38,0.08);border-radius:6px;border:1px solid rgba(220,38,38,0.2)">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">COMPOSITE SCORE: ' + cs.current + '/100 (' + cs.regime + ')</div>';
        html += '<div style="background:rgba(0,0,0,0.3);border-radius:4px;height:12px;overflow:hidden;margin-bottom:0.4rem">';
        html += '<div style="width:' + pct + '%;height:100%;background:' + barColor + ';border-radius:4px;transition:width 0.5s"></div></div>';
        html += '<div style="font-size:0.6rem;color:var(--text-dim)">' + cs.indicators + ' indicators | Top: ';
        cs.top_weights.forEach((w,i) => { html += (i>0?', ':'') + w.name + ' (' + (w.weight*100) + '%)'; });
        html += '</div>';
        html += '<div style="display:flex;gap:0.25rem;margin-top:0.3rem;flex-wrap:wrap">';
        Object.entries(cs.regimes).forEach(([k,v]) => {
          const active = k === cs.regime;
          html += '<span style="font-size:0.55rem;padding:1px 6px;border-radius:3px;' + (active ? 'background:var(--aac-war);color:#fff;font-weight:bold' : 'background:rgba(255,255,255,0.05);color:var(--text-dim)') + '">' + k + '</span>';
        });
        html += '</div></div>';
      }
      // 5-arm allocation
      if (wr.five_arms) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">5-ARM ALLOCATION</div>';
        Object.entries(wr.five_arms).forEach(([k,a]) => {
          const statusColor = a.status === 'ACTIVE' ? '#22c55e' : a.status === 'WATCHING' ? '#eab308' : '#94a3b8';
          html += '<div style="margin-bottom:0.3rem;font-size:0.65rem">';
          html += '<div style="display:flex;justify-content:space-between;align-items:center">';
          html += '<strong style="color:var(--gold)">' + k.replace(/_/g,' ').toUpperCase() + '</strong>';
          html += '<span style="color:' + statusColor + ';font-size:0.55rem;font-weight:bold">' + a.status + '</span></div>';
          html += '<div style="background:rgba(0,0,0,0.3);border-radius:3px;height:8px;margin:2px 0">';
          html += '<div style="width:' + a.target_pct + '%;height:100%;background:var(--aac-war);border-radius:3px;opacity:0.7"></div></div>';
          html += '<div style="font-size:0.55rem;color:var(--text-dim)">' + a.target_pct + '% target (max ' + a.max_pct + '%) | ' + a.instruments + '</div>';
          html += '</div>';
        });
        html += '</div>';
      }
      // Scenario tracks
      if (wr.scenario_tracks) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">SCENARIO TRACKS</div>';
        html += '<table style="width:100%;font-size:0.6rem;border-collapse:collapse">';
        html += '<tr style="color:var(--gold);font-weight:bold"><th style="text-align:left;padding:2px 4px">Track</th><th>P</th><th>Oil</th><th>Gold</th><th>SPY</th><th style="text-align:left;padding-left:6px">Outcome</th></tr>';
        Object.entries(wr.scenario_tracks).forEach(([k,s]) => {
          const rowBg = k === 'blackswan' ? 'rgba(220,38,38,0.15)' : k === 'major' ? 'rgba(220,38,38,0.08)' : 'transparent';
          html += '<tr style="background:' + rowBg + '"><td style="padding:2px 4px;color:var(--aac-war);font-weight:bold">' + k.toUpperCase() + '</td>';
          html += '<td style="text-align:center">' + s.probability + '</td>';
          html += '<td style="text-align:center">' + s.oil + '</td>';
          html += '<td style="text-align:center">' + s.gold + '</td>';
          html += '<td style="text-align:center">' + s.spy + '</td>';
          html += '<td style="padding-left:6px;color:var(--text-dim)">' + s.outcome + '</td></tr>';
        });
        html += '</table></div>';
      }
      // Capital deployment
      if (wr.capital_deployment) {
        const cd = wr.capital_deployment;
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--financial);margin-bottom:0.3rem">CAPITAL DEPLOYMENT</div>';
        html += '<div style="font-size:0.65rem;margin-bottom:0.3rem"><strong style="color:var(--gold)">Injection:</strong> ' + cd.injection_total + ' | <strong>Schedule:</strong> ' + cd.schedule + '</div>';
        html += '<div style="font-size:0.65rem;margin-bottom:0.2rem"><strong style="color:var(--gold)">' + cd.total_positions + ' positions across ' + cd.accounts.length + ' accounts:</strong></div>';
        cd.accounts.forEach(a => {
          html += '<div style="font-size:0.6rem;margin-left:0.5rem;margin-bottom:0.15rem"><span style="color:var(--aac-war);font-weight:bold">' + a.name + '</span> ' + a.balance + ' -- ' + a.role + ' (' + a.positions + ')</div>';
        });
        html += '</div>';
      }
      // Phases
      if (wr.phases) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">WEALTH PHASES</div>';
        Object.entries(wr.phases).forEach(([k,p]) => {
          html += '<div style="font-size:0.6rem;margin-bottom:0.2rem"><strong style="color:var(--gold)">' + k.toUpperCase() + '</strong> (' + p.range + ') -- ' + p.strategy + '</div>';
        });
        html += '</div>';
      }
      // Correlations
      if (wr.correlations) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">KEY CORRELATIONS</div>';
        wr.correlations.forEach(c => {
          const absVal = Math.abs(c.value);
          const corrColor = c.value > 0 ? '#22c55e' : '#ef4444';
          html += '<div style="font-size:0.6rem;margin-bottom:0.15rem"><strong style="color:' + corrColor + '">' + (c.value > 0 ? '+' : '') + c.value.toFixed(2) + '</strong> ' + c.pair + ' -- <span style="color:var(--text-dim)">' + c.meaning + '</span></div>';
        });
        html += '</div>';
      }
      // 13-week roadmap
      if (wr.roadmap) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--aac-war);margin-bottom:0.3rem">13-WEEK STRATEGIC ROADMAP</div>';
        wr.roadmap.forEach(r => {
          html += '<div style="font-size:0.55rem;margin-bottom:0.2rem;display:flex;gap:0.3rem">';
          html += '<span style="color:var(--aac-war);font-weight:bold;min-width:1.5rem">W' + r.week + '</span>';
          html += '<span style="color:var(--gold);font-weight:bold;min-width:6rem">' + r.label + '</span>';
          html += '<span style="color:var(--text-dim);min-width:5rem">' + r.dates + '</span>';
          html += '<span style="flex:1">' + r.focus + '</span></div>';
        });
        html += '</div>';
      }
      // Risk framework
      if (wr.risk_framework) {
        html += '<div style="margin-top:0.5rem;border-top:1px solid var(--border);padding-top:0.5rem">';
        html += '<div style="font-size:0.7rem;font-weight:bold;color:var(--eclipse);margin-bottom:0.3rem">RISK FRAMEWORK</div>';
        const rfColors = {GREEN:'#22c55e',YELLOW:'#eab308',ORANGE:'#f97316',RED:'#ef4444',BLACK:'#7f1d1d'};
        ['GREEN','YELLOW','ORANGE','RED','BLACK'].forEach(level => {
          if (wr.risk_framework[level]) {
            html += '<div style="font-size:0.6rem;margin-bottom:0.15rem"><span style="color:' + rfColors[level] + ';font-weight:bold">' + level + ':</span> ' + wr.risk_framework[level] + '</div>';
          }
        });
        html += '<div style="font-size:0.6rem;margin-top:0.3rem;color:var(--eclipse)"><strong>Max Drawdown:</strong> ' + wr.risk_framework.max_drawdown + '</div>';
        html += '<div style="font-size:0.6rem;color:var(--financial)"><strong>Profit Taking:</strong> ' + wr.risk_framework.profit_taking + '</div>';
        html += '</div>';
      }
      html += '</div>';
    }
  }

  content.innerHTML = html;
  panel.classList.add('open');
  overlay.classList.add('visible');
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
  document.getElementById('overlay').classList.remove('visible');
}

// Filters
function showAll() {
  activeFilter = 'all';
  updateButtons('btn-all');
  buildTimeline();
}
function filterType(type) {
  activeFilter = type;
  updateButtons('btn-' + type);
  buildTimeline();
}
function updateButtons(activeId) {
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  document.getElementById(activeId).classList.add('active');
}

function scrollToCurrent() {
  const current = document.querySelector('.moon-card.current');
  if (current) current.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}

// Alerts
function toggleAlerts() {
  const panel = document.getElementById('alerts-panel');
  alertsVisible = !alertsVisible;
  if (alertsVisible) {
    buildAlerts();
    panel.classList.add('visible');
    document.getElementById('btn-alerts').classList.add('active');
  } else {
    panel.classList.remove('visible');
    document.getElementById('btn-alerts').classList.remove('active');
  }
}

function buildAlerts() {
  const panel = document.getElementById('alerts-panel');
  const today = TIMELINE.today;
  const horizon = new Date(parseDate(today).getTime() + 30 * 86400000);

  let events = [];
  TIMELINE.all_events.forEach(e => {
    const d = parseDate(e.date);
    if (d >= parseDate(today) && d <= horizon) {
      const days = daysBetween(today, e.date);
      events.push({ ...e, days_until: days });
    }
  });
  events.sort((a, b) => a.days_until - b.days_until);

  let html = '<h3>UPCOMING EVENTS (next 30 days)</h3>';
  if (!events.length) {
    html += '<div style="color:var(--text-dim);font-size:0.75rem">No events in window.</div>';
  } else {
    events.forEach(e => {
      const daysClass = e.days_until === 0 ? 'days-0' : (e.days_until <= 3 ? 'days-soon' : 'days-later');
      html += '<div class="alert-item">' +
        '<div class="days-badge ' + daysClass + '">' + (e.days_until === 0 ? 'TODAY' : e.days_until + 'd') + '</div>' +
        '<div class="alert-type">' + e.type + '</div>' +
        '<div style="flex:1"><strong>' + e.name + '</strong></div>' +
        '<div style="font-size:0.6rem;color:var(--text-dim)">' + fmtDate(e.date) + '</div>' +
        '</div>';
    });
  }
  panel.innerHTML = html;
}

// Phi wave canvas
function drawPhiWave() {
  const canvas = document.getElementById('phi-wave');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 120;
  ctx.scale(2, 1);
  const w = canvas.offsetWidth;
  const h = 60;

  // Background
  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, w, h);

  // Draw resonance markers
  const totalDays = daysBetween(TIMELINE.anchor, TIMELINE.cycles[TIMELINE.cycles.length-1].end);
  const phi = 1.6180339887;

  ctx.strokeStyle = 'rgba(167, 139, 250, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let n = 0; n <= 13; n++) {
    const days = 29.53 * Math.pow(phi, n);
    const x = (days / totalDays) * w;
    if (x <= w) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
  }
  ctx.stroke();

  // Phi coherence wave
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let px = 0; px < w; px++) {
    const t = px / w;
    const dayPos = t * totalDays;
    // Sum of phi harmonics
    let y = 0;
    for (let n = 1; n <= 8; n++) {
      const period = 29.53 * Math.pow(phi, n);
      const amp = Math.exp(-0.15 * n);
      y += amp * Math.sin(2 * Math.PI * dayPos / period);
    }
    const py = h / 2 - y * (h / 4);
    if (px === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Moon cycle boundaries
  ctx.strokeStyle = 'rgba(192, 132, 252, 0.2)';
  ctx.setLineDash([3, 3]);
  TIMELINE.cycles.forEach(c => {
    const d = daysBetween(TIMELINE.anchor, c.start);
    const x = (d / totalDays) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Today marker
  const todayDays = daysBetween(TIMELINE.anchor, TIMELINE.today);
  const todayX = (todayDays / totalDays) * w;
  ctx.strokeStyle = '#c084fc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(todayX, 0);
  ctx.lineTo(todayX, h);
  ctx.stroke();
  ctx.fillStyle = '#c084fc';
  ctx.font = '9px monospace';
  ctx.fillText('NOW', todayX - 10, 10);
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  buildTimeline();
  drawPhiWave();
  scrollToCurrent();
  window.addEventListener('resize', drawPhiWave);
});

// Keyboard
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDetail();
});

// ── Space Weather (live NOAA SWPC) ────────────────────────────────────────
async function fetchSpaceWeather() {
  const timeout = 6000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  const opts = { signal: ctrl.signal };

  try {
    // Kp Index
    try {
      const kpResp = await fetch('https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json', opts);
      const kpData = await kpResp.json();
      if (kpData && kpData.length) {
        const latest = kpData[kpData.length - 1];
        const kp = parseFloat(latest.Kp || latest.kp || 0);
        const label = kp < 4 ? 'Quiet' : kp < 5 ? 'Active' : kp < 6 ? 'Minor Storm' : kp < 7 ? 'Moderate' : 'Strong Storm';
        const color = kp < 4 ? '#22c55e' : kp < 5 ? '#eab308' : kp < 6 ? '#f97316' : '#ef4444';
        document.querySelector('#sw-kp .sw-value').textContent = kp.toFixed(1);
        document.querySelector('#sw-kp .sw-value').style.color = color;
        document.querySelector('#sw-kp .sw-sub').textContent = label;
        // Storm alert
        const alertEl = document.getElementById('sw-alert');
        if (kp >= 5) {
          alertEl.style.display = 'block';
          alertEl.style.background = 'rgba(239,68,68,0.15)';
          alertEl.style.border = '1px solid #ef4444';
          alertEl.style.color = '#fca5a5';
          alertEl.innerHTML = '<strong>&#9888;&#65039; Geomagnetic Storm Active (Kp=' + kp.toFixed(0) + ')</strong> -- Elevated activity correlates with market volatility shifts. Solar cycle 25 peak amplifies fire peak resonance.';
        } else if (kp >= 4) {
          alertEl.style.display = 'block';
          alertEl.style.background = 'rgba(234,179,8,0.1)';
          alertEl.style.border = '1px solid #eab308';
          alertEl.style.color = '#fcd34d';
          alertEl.innerHTML = '&#129522; <strong>Activity Elevated (Kp=' + kp.toFixed(0) + ')</strong> -- Approaching storm threshold. Monitor for CME impacts.';
        }
      }
    } catch(e) {}

    // Solar Wind
    try {
      const windResp = await fetch('https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json', opts);
      const windData = await windResp.json();
      let speed = null;
      if (Array.isArray(windData) && windData.length) speed = windData[0].proton_speed || windData[0].WindSpeed;
      else if (windData && typeof windData === 'object') speed = windData.WindSpeed || windData.proton_speed;
      if (speed) {
        document.querySelector('#sw-wind .sw-value').textContent = speed + ' km/s';
        document.querySelector('#sw-wind .sw-sub').textContent = parseInt(speed) > 600 ? 'Fast stream' : parseInt(speed) > 400 ? 'Normal' : 'Slow';
      }
    } catch(e) {}

    // Solar Flux (10.7cm)
    try {
      const fluxResp = await fetch('https://services.swpc.noaa.gov/products/summary/10cm-flux.json', opts);
      const fluxData = await fluxResp.json();
      let flux = null;
      if (Array.isArray(fluxData) && fluxData.length) flux = fluxData[0].flux || fluxData[0].Flux;
      else if (fluxData && typeof fluxData === 'object') flux = fluxData.Flux || fluxData.flux;
      if (flux) {
        document.querySelector('#sw-flux .sw-value').textContent = flux + ' sfu';
        document.querySelector('#sw-flux .sw-sub').textContent = parseInt(flux) > 150 ? 'Elevated' : parseInt(flux) > 100 ? 'Moderate' : 'Low';
      }
    } catch(e) {}

    // NOAA Scales (G/S/R)
    try {
      const scalesResp = await fetch('https://services.swpc.noaa.gov/products/noaa-scales.json', opts);
      const scalesData = await scalesResp.json();
      const current = scalesData['0'] || {};
      const scalesEl = document.getElementById('sw-scales');
      const scaleColors = {'0':'#22c55e','1':'#eab308','2':'#f97316','3':'#ef4444','4':'#dc2626','5':'#7f1d1d'};
      let html = '';
      [['G','Geomagnetic Storm','geo_storm'],['S','Solar Radiation','solar_rad'],['R','Radio Blackout','radio']].forEach(([key,label]) => {
        const d = current[key] || {};
        const sc = String(d.Scale || '0');
        const bg = scaleColors[sc] || '#22c55e';
        const textColor = parseInt(sc) >= 2 ? '#fff' : '#000';
        html += '<div class="sw-scale-badge" style="background:' + bg + ';color:' + textColor + '">';
        html += '<div>' + key + sc + '</div>';
        html += '<div class="scale-label" style="color:' + textColor + ';opacity:0.8">' + label + '</div>';
        if (d.Text && d.Text !== 'none') html += '<div class="scale-label" style="color:' + textColor + '">' + d.Text + '</div>';
        html += '</div>';
      });
      scalesEl.innerHTML = html;
    } catch(e) {}

    // Sunspot number (predicted)
    try {
      const ssnResp = await fetch('https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json', opts);
      const ssnData = await ssnResp.json();
      const now = new Date().toISOString().slice(0, 7);
      for (const row of ssnData) {
        if ((row['time-tag'] || '').startsWith(now)) {
          document.querySelector('#sw-ssn .sw-value').textContent = Math.round(row.predicted_ssn || 0);
          document.querySelector('#sw-ssn .sw-sub').textContent = 'Cycle 25 (' + (row.low_ssn||'?') + '-' + (row.high_ssn||'?') + ')';
          break;
        }
      }
    } catch(e) {}

    document.getElementById('sw-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } finally {
    clearTimeout(timer);
  }
}

// Fetch space weather on load, refresh every 5 minutes
fetchSpaceWeather();
setInterval(fetchSpaceWeather, 300000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
    d = ThirteenMoonDoctrine()
    path = export_interactive_storyboard(d)
    print(f"Storyboard exported to: {path}")
