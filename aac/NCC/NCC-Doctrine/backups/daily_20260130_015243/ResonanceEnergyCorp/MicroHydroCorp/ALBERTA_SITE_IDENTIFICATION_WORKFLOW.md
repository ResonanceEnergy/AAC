# ALBERTA MICROHYDRO SITE IDENTIFICATION WORKFLOW
## Desktop + Field Method to Identify 50+ Viable Sites

**Date:** January 25, 2026  
**Target:** 50 shortlisted sites in Alberta with 3–15 m head, 50–300+ L/s flow, grid/load proximity  
**Time Estimate:** 1–2 days desktop + 5–10 days field validation  
**Output:** Populated ALBERTA_SITE_SCOUTING_TEMPLATE.csv with priority ranking

---

## PHASE 1: DESKTOP RECONNAISSANCE (8–12 hours)

### **STEP 1: Data Layer Assembly (2 hours)**

**Core Data Sources (Public/Free):**

1. **Hydrometric Data**  
   - Source: Water Survey of Canada (Environment Canada) / Alberta Environment and Parks  
   - URL: https://wateroffice.ec.gc.ca/  
   - What: Active and archived stream gauges; monthly/annual flow statistics  
   - Use: Identify streams with baseflow >50 L/s year-round; note seasonal variation

2. **Digital Elevation Model (DEM)**  
   - Source: Natural Resources Canada, Alberta Geological Survey  
   - URL: https://ftp.maps.canada.ca/pub/elevation/  
   - Resolution: 10 m or better for Alberta  
   - Use: Calculate slope/gradient; identify drops, weirs, rapids

3. **Water Rights & Licenses**  
   - Source: Alberta Environment and Parks Water Act licensing database  
   - URL: https://www.alberta.ca/water-allocation-transfers (check public registry)  
   - Use: Existing licences show active diversions, irrigation canals, municipal intakes

4. **Irrigation District Maps**  
   - Source: Alberta Irrigation Projects Association, individual district websites  
   - Examples: Eastern Irrigation District, Western Irrigation District, St. Mary River Irrigation District  
   - Use: Canal networks, control structures, drops, turnouts — existing infrastructure

5. **Municipal Infrastructure**  
   - Source: Municipal GIS open data portals (Edmonton, Calgary, Lethbridge, Red Deer, etc.)  
   - What: Wastewater treatment plants, water treatment plants, stormwater outfalls  
   - Use: WWTP/WTP sites with consistent discharge and known head

6. **Dam/Weir Inventories**  
   - Source: Canadian Dam Association, Alberta Environment dam safety inventory  
   - URL: https://www.cda.ca/ or provincial records  
   - Use: Low-head dams, weirs, spillways suitable for run-of-river add-ons

7. **Fisheries Sensitive Areas**  
   - Source: Alberta Environment, DFO (Fisheries and Oceans Canada)  
   - URL: https://open.alberta.ca/opendata/fisheries-data  
   - Use: Flag high-sensitivity reaches; prioritize already-modified or non-fish channels

8. **Roads & Grid Infrastructure**  
   - Source: Alberta base maps, utility GIS (ATCO, FortisAlberta public maps)  
   - Use: Proximity to roads (access) and distribution lines (grid-tie or nearby loads)

**Tool Setup:**  
- QGIS (free, open-source GIS): load DEM, hydro layer, infrastructure points/lines  
- Google Earth Pro: quick visual recon, measure distances, save placemarks  
- Spreadsheet: track site attributes in ALBERTA_SITE_SCOUTING_TEMPLATE.csv

---

### **STEP 2: Identify Candidate Sites (4–6 hours)**

**Strategy A: Irrigation Infrastructure (Target: 20–30 sites)**

1. Load irrigation district canal maps into QGIS.
2. Identify control structures, drops, gates, turnouts, spillways (often 1–5 m drops).
3. Cross-reference with flow schedules (irrigation season = high flow, winter = lower but often still present).
4. Mark sites with:
   - Estimated head from DEM or canal elevation profiles.
   - Flow = fraction of canal capacity (often 100s to 1000s L/s during season).
   - Access via canal maintenance roads.
   - Land status = irrigation district or easement.
   - Fish presence = usually LOW (artificial canals, often screened/gated already).
   - Permit complexity = LOW to MEDIUM (existing water rights; add generation clause).

**Strategy B: Municipal Water/Wastewater (Target: 5–10 sites)**

1. Locate WWTPs and WTPs on municipal GIS or Google Earth.
2. Check outfall/discharge points for consistent head (treatment plant elevation to receiving water).
3. Estimate flow from plant capacity (publicly reported for larger municipalities).
4. Note:
   - Head = 2–10 m typical for outfall drops.
   - Flow = consistent year-round (municipal demand fairly stable).
   - Land status = municipal; easements usually in place.
   - Fish presence = LOW at outfall (effluent discharge, no upstream passage).
   - Permit complexity = MEDIUM (environmental discharge compliance, but generation add-on feasible).
   - Grid distance = usually <1 km (plants are grid-connected).

**Strategy C: Existing Weirs/Low-Head Dams (Target: 10–15 sites)**

1. Query dam/weir inventory for structures <15 m head in Alberta.
2. Filter for run-of-river types (not large storage reservoirs).
3. Visual recon on Google Earth: check for spillways, bypass channels, tailrace.
4. Note:
   - Head = structure height (often 2–10 m for low-head).
   - Flow = look up hydrometric gauge if nearby, or estimate from drainage area.
   - Land status = mix of private, Crown, municipal.
   - Fish presence = MEDIUM to HIGH (existing barrier; fish passage upgrade may be required).
   - Permit complexity = HIGH (retrofit to existing structure; environmental review, fish passage).

**Strategy D: Natural Stream Drops + Culverts (Target: 5–10 sites)**

1. Use DEM slope analysis: identify stream segments with >5% gradient over 50–200 m runs.
2. Cross with road crossings (culverts often create localized drops).
3. Filter for perennial streams with gauged or estimated baseflow >50 L/s.
4. Note:
   - Head = elevation change over reach (3–15 m feasible).
   - Flow = from gauge or drainage-area estimate (0.01–0.05 m³/s/km² baseflow in Alberta foothills/mountains).
   - Land status = often Crown (streambed), need water Act authorization.
   - Fish presence = MEDIUM to HIGH (natural streams; require fish-safe design).
   - Permit complexity = HIGH (new diversion; environmental assessment, DFO review).

**Strategy E: Historical Mill Sites (Target: 0–5 sites, bonus)**

1. Search historical records, heritage sites for old water mills, hydro plants.
2. Often have remnant weirs, raceways, tailraces that can be refurbished.
3. Heritage/community interest can support permits.
4. Note:
   - Head/flow = site-specific; often modest (2–5 m).
   - Land status = variable; may have municipal or heritage designation.
   - Permit complexity = MEDIUM (historical precedent helps; still need fish/environment review).

---

### **STEP 3: Initial Site Scoring & Prioritization (2 hours)**

**Scoring Criteria (1–5 scale, higher = better):**

| Criterion | Weight | Notes |
|-----------|--------|-------|
| **Head** | 5 | 3–5 m = 3 pts; 5–10 m = 4 pts; >10 m = 5 pts |
| **Flow** | 5 | 50–100 L/s = 3 pts; 100–200 L/s = 4 pts; >200 L/s = 5 pts |
| **Grid Distance** | 3 | <1 km = 5 pts; 1–3 km = 3 pts; >3 km = 1 pt |
| **Access** | 4 | Paved road = 5 pts; gravel = 4 pts; 4WD track = 2 pts; bushwhack = 1 pt |
| **Permit Complexity** | 4 | LOW (canal/municipal) = 5 pts; MEDIUM (weir retrofit) = 3 pts; HIGH (new stream) = 1 pt |
| **Fish Presence** | 3 | None/artificial = 5 pts; low sensitivity = 3 pts; high sensitivity = 1 pt |
| **Land Status** | 3 | Municipal/utility-owned = 5 pts; private with willing owner = 4 pts; Crown = 2 pts |

**Total Score:** Max = 135 pts

**Priority Tiers:**
- **High (>90 pts):** Pursue immediately; low friction, good economics.
- **Medium (60–90 pts):** Worth field visit; may require negotiation/mitigation.
- **Low (<60 pts):** Hold for later or if low-hanging fruit exhausted.

**Output:** Rank top 50 sites by score; populate ALBERTA_SITE_SCOUTING_TEMPLATE.csv.

---

## PHASE 2: FIELD VALIDATION (5–10 days, 5–10 sites/day)

### **STEP 4: Site Visits (Top 20–30 High-Priority Sites)**

**Field Kit:**
- Clinometer or smartphone app (measure head/slope)
- Tape measure, laser distance meter
- Float + stopwatch (rough flow measurement)
- Camera (photos of infrastructure, fish screens, access)
- Water sample kit (optional: turbidity, temp, basic chemistry)
- Contact list (landowners, irrigation district managers, municipal ops)

**On-Site Checklist:**

1. **Head Verification:**
   - Measure elevation difference: intake to tailrace or top-of-drop to bottom.
   - Use clinometer on penstock route or existing structure.
   - Record: Actual head (m), potential penstock length (m).

2. **Flow Estimation:**
   - **Method 1 (Float):** Mark 10 m channel length; time floating object (surface velocity); multiply by 0.8 and cross-section area.
   - **Method 2 (Channel calc):** Measure width, depth; estimate velocity from roughness/slope.
   - **Method 3 (Existing data):** If gauged or canal with known flow schedule, use records.
   - Record: Estimated flow (L/s), seasonal variation notes.

3. **Power Estimate:**
   - Quick calc: P (kW) ≈ 0.7 × Head (m) × Flow (m³/s) × 9.81 (assume 70% system efficiency).
   - Record: Estimated power (kW).

4. **Infrastructure Assessment:**
   - Existing structures: weir, gate, screen, penstock route, tailrace.
   - Civil needs: intake construction, penstock install, powerhouse footprint, access road.
   - Record: Photos, sketch, cost estimate for civil works (LOW/MEDIUM/HIGH).

5. **Environmental Observations:**
   - Fish presence: observe or ask locals; check for spawning areas, rearing habitat.
   - Bypass feasibility: natural bypass channel, or need to construct?
   - Sediment/debris: turbidity, sediment load (affects turbine wear).
   - Record: Fish species if known, habitat notes, bypass options.

6. **Access & Land:**
   - Road condition: paved, gravel, seasonal, locked gate?
   - Landowner contact: willing to discuss lease/easement?
   - Permits: any known restrictions (parks, protected areas, heritage)?
   - Record: Access rating, landowner contact, permit flags.

7. **Grid/Load Proximity:**
   - Nearest distribution line: measure distance, note voltage class (if visible).
   - Nearby loads: farm, facility, town (for off-grid or net-metering).
   - Record: Grid distance (km), load type/size if known.

**Output:** Update CSV with field-verified data; revise priority scores.

---

### **STEP 5: Stakeholder Engagement (Concurrent with Field Visits)**

**Key Contacts:**

1. **Irrigation Districts:**
   - Managers often open to co-generation if it doesn't impair delivery.
   - Offer: revenue share, O&M cost offset, data sharing.
   - Ask: flow schedules, head data, existing agreements.

2. **Municipalities:**
   - Utilities/public works: interested in WWTP/WTP co-gen for cost offset, green credentials.
   - Offer: turnkey, O&M, revenue/cost share.
   - Ask: discharge data, easements, interconnection feasibility.

3. **Private Landowners:**
   - Farmers, ranchers with on-property water rights or weirs.
   - Offer: EaaS (free/discounted power), lease payments, environmental stewardship branding.
   - Ask: water rights docs, access, willingness to permit.

4. **Regulators (Early Conversation):**
   - Alberta Environment and Parks (Water Act, environmental).
   - DFO (Fisheries Act, if fish-bearing).
   - AUC (Alberta Utilities Commission, if grid-tied >1 MW or commercial).
   - Ask: Process overview, timelines, fish passage standards, pre-application advice.

**Output:** Document interest level, concerns, next steps for top sites.

---

## PHASE 3: SHORTLIST REFINEMENT & NEXT ACTIONS (2–4 hours)

### **STEP 6: Final Ranking & Pilot Selection**

**Criteria for Top 3–5 Pilots:**
- **Diversity:** Mix of infrastructure types (canal, WWTP, weir) for tech/model validation.
- **Willing hosts:** Confirmed interest from landowner/operator; MoU-ready.
- **Permit path:** Clear (or clearest available) regulatory route; no show-stoppers.
- **Data quality:** Good head/flow estimates; realistic power/LCOE projections.
- **Access & logistics:** Feasible install within 3–6 months; contractor availability.

**Output:**
- Top 3 pilots for immediate development (P1, P2, P3 from GIGA_ENTERPRISE_BLUEPRINT).
- Next 10 backup sites for expansion (year 2–3).
- Remaining 37+ sites catalogued for future (geographic spread, tech variations).

---

### **STEP 7: Documentation & Permitting Prep**

For each top pilot:

1. **Site Data Sheet:**
   - Location (lat/lon, land description), infrastructure type.
   - Head (verified), flow (verified + seasonal range), power estimate.
   - Civil works scope, cost estimate.
   - Environmental: fish species, bypass plan, ecological flows.
   - Land/access: ownership, easement status, MoU/lease terms.
   - Grid: distance, voltage, interconnection contact.

2. **Permit Roadmap:**
   - Water Act: application type (new licence, amendment, exemption).
   - Fisheries: screening, passage, monitoring plan.
   - Environmental: baseline study scope, DFO/provincial coordination.
   - Electrical: utility interconnection study request (if grid-tied).
   - Timeline: 6–18 months typical for Alberta (faster for existing infrastructure, slower for new stream diversions).

3. **Financial Pro-Forma:**
   - Capex: civil + hardware + permitting + install.
   - Opex: O&M, insurance, land lease, monitoring.
   - Revenue: kWh × tariff (grid FIT, avoided cost, or EaaS contract).
   - Financing: equity, grants (NRCAN, provincial programs), MFI/DFI for community models.
   - Payback, IRR, NPV.

**Output:** Ready-to-execute pilot packages; permit applications drafted.

---

## FAST-TRACK TIPS

**Maximize Early Wins:**
- Start with **irrigation district canals** and **municipal WWTP/WTP** sites — lowest permit friction, existing infrastructure, cooperative hosts.
- Defer natural stream sites until you have 3–5 canal/municipal pilots operating (build regulatory trust + fish-safe track record).

**Leverage Existing Rights:**
- Retrofit onto existing water licences (irrigation, municipal) where generation is ancillary use — faster than new applications.

**Pre-Application Engagement:**
- Meet Alberta Environment and DFO *before* formal application; show fish-safe design, data transparency commitments.
- Ask: "What would make this approvable?" and design to that standard.

**Community Champions:**
- Partner with local watershed groups, Trout Unlimited chapters, Indigenous communities — turn potential opposition into co-design allies.

**Data Transparency:**
- Commit to public dashboards (flow, fish passage, uptime) from day 1 — builds social license, differentiates from old-school hydro.

---

## TOOLS & RESOURCES SUMMARY

| Tool/Source | Purpose | URL/Access |
|-------------|---------|------------|
| **QGIS** | GIS analysis | https://qgis.org/ |
| **Google Earth Pro** | Visual recon | https://earth.google.com/web/ |
| **Water Survey of Canada** | Hydrometric data | https://wateroffice.ec.gc.ca/ |
| **Alberta Open Data** | GIS layers, permits | https://open.alberta.ca/ |
| **Irrigation District Websites** | Canal maps, flow schedules | (search by district name) |
| **Municipal Open Data** | Infrastructure GIS | (Calgary, Edmonton, etc. open data portals) |
| **DFO Self-Assessment Tool** | Fish passage screening | https://www.dfo-mpo.gc.ca/ |
| **Alberta Environment Water Licensing** | Permit process | https://www.alberta.ca/water-licence |

---

## NEXT STEPS

1. **Desktop (Week 1):** Assemble data layers; identify 50 candidates; populate CSV template; score and rank.
2. **Field (Weeks 2–3):** Visit top 20–30 sites; verify head/flow; engage stakeholders; refine scores.
3. **Shortlist (Week 4):** Select top 3–5 pilots; draft site data sheets and permit roadmaps.
4. **Execute (Months 2–6):** Permits, design, procurement, install for P1–P3.

**Critical Success Factors:**
- Speed: Desktop → field → permits in <90 days for first pilots.
- Relationships: Irrigation districts and municipalities are fastest path.
- Transparency: Fish-safe + data-first to build regulatory and social trust.
- Portfolio: Diversify site types to de-risk tech and business model assumptions.

---

**Template CSV is ready at:** ALBERTA_SITE_SCOUTING_TEMPLATE.csv  
**Fill it as you go; sort by Priority score; commit top sites to detailed diligence.**

