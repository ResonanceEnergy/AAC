# RESEARCH & DEVELOPMENT ARM CHARTER

**Date:** January 25, 2026  
**Mission:** Drive continuous innovation in microhydro technology, environmental performance, data/AI optimization, and business model evolution.  
**Mandate:** Advance 1600-insight foundation; validate field performance; pioneer fish-safe standards; develop next-gen materials, controls, and hybrid systems.

---

## ORGANIZATIONAL STRUCTURE

### **Leadership**
- **Chief Technology Officer (CTO) / Head of R&D:** PhD or equivalent; 15+ years hydro/power systems; publication record; industry + academia credibility.
- **Reporting:** Direct to CEO; dotted line to COO (manufacturing integration).

### **Team (Build to 10–15 FTE by Year 3)**

**1. Fluid Dynamics & Turbomachinery (2–3 FTE)**
- **Responsibilities:** Runner design optimization; CFD modeling; cavitation analysis; efficiency testing; sediment wear mitigation.
- **Profile:** MS/PhD mechanical/aerospace engineering; hydro turbine experience; CFD expertise (ANSYS, OpenFOAM).

**2. Materials & Durability (1–2 FTE)**
- **Responsibilities:** Corrosion-resistant alloys, coatings, composites; fatigue testing; accelerated life testing; cost-down materials.
- **Profile:** MS/PhD materials science; metallurgy or polymer composites; testing/characterization experience.

**3. Power Electronics & Controls (2–3 FTE)**
- **Responsibilities:** MPPT algorithms; power conversion efficiency; SiC/GaN adoption; battery/hybrid integration; grid services (VPP).
- **Profile:** MS/PhD electrical engineering; power electronics design; embedded controls (C/C++, MATLAB/Simulink).

**4. Environmental & Fish Biology (1–2 FTE)**
- **Responsibilities:** Fish passage research; intake velocity optimization; ecological flow modeling; third-party monitoring protocols; regulatory engagement.
- **Profile:** MS/PhD fisheries biology, ecology, or civil/environmental engineering; field research experience; regulatory knowledge.

**5. Data Science & AI (2–3 FTE)**
- **Responsibilities:** Predictive maintenance algorithms; digital twin development; fleet optimization; SCADA analytics; anomaly detection.
- **Profile:** MS/PhD computer science, data science; ML/AI (Python, TensorFlow); time-series analysis; IoT/edge computing.

**6. Systems Integration & Testing (1–2 FTE)**
- **Responsibilities:** Lab test rig design/operation; field instrumentation; prototype build coordination; validation testing; standards compliance.
- **Profile:** BS/MS engineering; hands-on prototyping; instrumentation and data acquisition; safety protocols.

**7. Business Model Innovation (1 FTE)**
- **Responsibilities:** EaaS/paygo model optimization; carbon/biodiversity MRV methodologies; community ownership frameworks; pricing strategy.
- **Profile:** MBA or equivalent; energy access/development finance background; social enterprise experience.

---

## KEY RESEARCH PROGRAMS

### **Program 1: Turbine Efficiency & Range (Ongoing, Priority HIGH)**

**Objective:** Achieve 75%+ field efficiency across 50–125% design flow; minimize cavitation; extend head/flow operating envelope.

**Workstreams:**
1. **CFD Optimization:** Parametric runner blade design; nozzle geometry; draft tube optimization; target 2–3% efficiency gains.
2. **Prototype Testing:** Lab test rig (controlled head/flow); field validation on pilots; compare to model predictions.
3. **Cavitation Mitigation:** NPSH margin verification; surface finish optimization; pressure transient analysis.
4. **Cost-Down Runner Design:** Transition from precision-machined to cast/stamped blades without efficiency loss; DFM iteration.

**Milestones:**
- Month 6: CFD baseline validated against pilot data.
- Month 12: Optimized runner design tested; efficiency >72% field-verified.
- Month 18: DFM runner iteration ready for volume production.

**Partners:** University hydro labs (e.g., TU Munich, EPFL, IIT Roorkee); turbine OEMs for manufacturing insights.

---

### **Program 2: Fish-Safe Intake & Passage (Ongoing, Priority HIGHEST)**

**Objective:** >95% fish survival across species; transparent monitoring; regulatory acceptance; industry standard-setting.

**Workstreams:**
1. **Intake Velocity Optimization:** <0.3 m/s validated across flow conditions; angled/wedge-wire screens; behavioral deterrents (light, sound).
2. **Bypass Design:** Surface and deep bypass channels; flow allocation (5–10% for bypass); passage efficiency testing.
3. **Turbine Passage Safety:** Low RPM (<600); blade tip speed <10 m/s; crossflow vs Archimedes vs Kaplan comparison; strike/injury modeling.
4. **Monitoring Protocols:** PIT tags, acoustic telemetry, video counting; third-party validation; publish data openly.

**Milestones:**
- Month 3: Literature review + initial design standards.
- Month 6: Pilot intake instrumented; baseline fish monitoring.
- Month 12: >90% survival validated (site-specific); adaptive management plan.
- Month 18: Publish peer-reviewed paper; propose industry standard (ISO/IEC TC 4).

**Partners:** DFO/NOAA fisheries scientists; Trout Unlimited; Alden Labs (fish passage testing); universities (UW, OSU, UC Davis).

---

### **Program 3: Advanced Materials (Years 2–5, Priority MEDIUM)**

**Objective:** Cost reduction + performance + sustainability via advanced materials.

**Workstreams:**
1. **Composite Runners:** Carbon fiber or glass fiber reinforced polymers; 30–40% weight reduction; corrosion-free; cost target <stainless @volume.
2. **Ceramic/Nano Coatings:** Wear-resistant coatings for sediment; self-cleaning (lotus effect); longer maintenance intervals.
3. **SiC/GaN Power Electronics:** Wide-bandgap semiconductors; higher efficiency (>98%); higher switching frequency; cost parity by Year 3.
4. **Recyclable/Circular Materials:** Design for disassembly; use abundant, recyclable materials; end-of-life take-back program.

**Milestones:**
- Year 2: Composite runner prototype tested; SiC inverter prototype validated.
- Year 3: Composite runner in pilot; cost <$1.8k/kW for runner/casing subsystem.
- Year 5: 50% of units use composite runners and SiC electronics; circularity metrics published.

**Partners:** Materials suppliers (Toray, Hexcel for composites); SiC/GaN vendors (Wolfspeed, Infineon); university materials labs.

---

### **Program 4: Data/AI Optimization (Years 1–5, Priority HIGH)**

**Objective:** Autonomous operation; predictive maintenance; fleet-wide learning; 2–5% performance uplift via AI.

**Workstreams:**
1. **Predictive Maintenance:** Vibration analysis; bearing health; thermal anomalies; predict failures 30–90 days ahead; reduce truck rolls 30%.
2. **MPPT Enhancement:** Adaptive MPPT via reinforcement learning; handle non-stationary flow; real-time head/flow estimation; 1–2% energy gain.
3. **Digital Twins:** Virtual commissioning; scenario testing; optimize control setpoints remotely; shadow-mode validation.
4. **Fleet Analytics:** Cross-unit learning; anomaly detection; benchmark performance; recommend retrofits; data monetization (anonymized insights for utilities/OEMs).

**Milestones:**
- Year 1: Baseline monitoring data collected from 3–5 pilots; ML model training initiated.
- Year 2: Predictive maintenance deployed; first failure predicted and avoided; 20% truck roll reduction.
- Year 3: Digital twin v1.0; adaptive MPPT live on 50+ units; 1.5% average energy gain measured.
- Year 5: Fleet optimization platform commercialized as SaaS; 500+ units enrolled.

**Partners:** Cloud providers (AWS, Azure); edge AI vendors (NVIDIA Jetson); universities (Stanford AI Lab, MIT CSAIL).

---

### **Program 5: Hybrid & Storage Integration (Years 2–7, Priority MEDIUM)**

**Objective:** Optimize hydro+solar+storage; VPP-ready; maximize renewable penetration; grid services revenue.

**Workstreams:**
1. **Hybrid Control:** Coordinated dispatch; load management; battery charge/discharge optimization; minimize curtailment.
2. **Battery Sizing & Chemistry:** LFP vs NMC tradeoffs; cycle life; cost; second-life batteries; local assembly.
3. **Grid Services:** Frequency regulation; peak shaving; demand response; VPP aggregation (where enabled); revenue stacking.
4. **Microgrid Stability:** Islanded operation; black start; voltage/frequency stability; synthetic inertia.

**Milestones:**
- Year 2: Hybrid pilot (5 kW hydro + 3 kW solar + 10 kWh battery) operational; control algorithms validated.
- Year 3: 10+ hybrid systems deployed; grid services pilot (frequency regulation) in Alberta or pilot country.
- Year 5: VPP aggregation tested; 100+ units providing grid services; revenue model proven.

**Partners:** Solar inverter OEMs (SMA, Fronius); battery vendors (BYD, Pylontech); VPP platforms (AutoGrid, Stem).

---

### **Program 6: Carbon & Biodiversity MRV (Years 1–10, Priority HIGH)**

**Objective:** Rigorous, transparent MRV; Gold Standard/Verra certification; biodiversity co-benefit quantification; premium pricing.

**Workstreams:**
1. **Carbon Baseline & Additionality:** Establish diesel/grid baseline; prove additionality; develop microhydro methodologies (if not existing).
2. **Monitoring Systems:** kWh metered and transmitted; diesel displacement calculated; leakage (rebound effects) assessed.
3. **Biodiversity Co-Benefits:** Fish passage success; water quality (turbidity, DO); riparian health; remote sensing (satellite, drone); species surveys.
4. **Verification & Issuance:** Third-party verifiers (annual); credit issuance; forward sales or spot market; bundled credit portfolios.

**Milestones:**
- Year 1: MRV plan developed; engage Gold Standard/Verra; pilot monitoring instrumented.
- Year 2: First credits issued (5K–10K tCO₂e); forward sales secured; biodiversity monitoring baseline.
- Year 5: 50K+ tCO₂e/year; biodiversity premium pricing achieved; publish methodology openly.

**Partners:** Gold Standard, Verra; verifiers (SCS Global, DNV); conservation NGOs (WWF, TNC).

---

### **Program 7: Manufacturing Innovation (Years 2–10, Priority MEDIUM)**

**Objective:** Cost-down via DFM, automation, lean manufacturing; $1.5–2.5k/kW @scale; quality excellence.

**Workstreams:**
1. **Design for Manufacturing:** Reduce part count; standardize fasteners/connectors; modular assembly; casting/welding optimization.
2. **Automation & Tooling:** Automated winding (generator stators); robotic welding; CNC optimization; reduce touch labor 40%.
3. **Supply Chain Optimization:** Multi-sourcing; near-shoring where feasible; inventory optimization (JIT); quality at source.
4. **Statistical Process Control:** SPC for critical dimensions; process capability studies (Cpk >1.33); zero-defect culture.

**Milestones:**
- Year 2: DFM iteration complete; part count reduced 20%; assembly time reduced 30%.
- Year 3: Automated line pilot (10 units/month); SPC implemented; first-pass yield >95%.
- Year 5: Fully automated line (500 units/year); cost <$1.8k/kW hardware COGS; Cpk >1.67 for critical processes.

**Partners:** Contract manufacturers; automation vendors (FANUC, ABB Robotics); lean consultants (Shingo Institute).

---

## RESEARCH INFRASTRUCTURE

### **Lab Facilities (Build Phase)**

**Year 1 (Minimum Viable):**
- **Location:** Shared university lab space or small leased facility (500–1000 m²).
- **Capabilities:** Benchtop turbine test rig (recirculating flow loop, 0–10 kW, instrumented); generator dynamometer; power electronics test bench; data acquisition; machine shop access.
- **Cost:** $200–500K capex + $100K/yr opex.

**Year 2–3 (Expansion):**
- **Dedicated facility:** 2000–3000 m² lab + office.
- **Capabilities:** Full-scale test rig (20 kW, variable head/flow); environmental chamber (temperature, humidity); vibration/fatigue testing; materials characterization; electronics/EMC testing.
- **Cost:** $1–2M capex + $300K/yr opex.

**Year 5+ (Advanced):**
- **Multi-site:** Labs in Alberta (HQ) + 1–2 developing countries (field R&D centers).
- **Capabilities:** Hybrid system testbed; fish passage flume; pilot-scale manufacturing line; digital twin lab (HILS/SILS).
- **Cost:** $5–10M total capex.

### **Field Test Sites**
- Co-locate with pilots (P1–P3 Alberta, expanding to 10+ international).
- Instrument for high-resolution data (1 Hz+); real-time telemetry; remote control for experiments.
- Community engagement: open houses, student field trips, transparent data sharing.

### **Computing Infrastructure**
- **CFD/FEA:** HPC cluster or cloud (AWS, Azure); ANSYS, COMSOL licenses.
- **ML/AI:** GPU servers (NVIDIA A100 or equivalent); cloud ML platforms (AWS SageMaker, Google Vertex AI).
- **Data Platform:** Time-series database (InfluxDB, TimescaleDB); object storage (S3); visualization (Grafana, Tableau).
- **Cost:** $50–100K/yr (cloud + licenses).

---

## INTELLECTUAL PROPERTY STRATEGY

### **Patent Portfolio (Selective)**
- **Patentable areas:** Novel runner geometries; fish-deterrent systems; AI/control algorithms; hybrid optimization methods.
- **Strategy:** File where defensible competitive advantage; otherwise publish openly (build ecosystem, avoid patent trolls).
- **Budget:** $50–100K/yr (filing, prosecution, maintenance).

### **Open Source / Open Innovation**
- **Publish:** Environmental monitoring protocols; fish passage standards; carbon MRV methodologies.
- **Share:** Basic designs for low-resource settings (creative commons); enable local manufacturing.
- **Benefit:** Build ecosystem; attract partners/collaborators; social license; differentiate on execution, not secrecy.

### **Trade Secrets**
- **Protect:** Manufacturing processes; supplier relationships; cost models; customer data; AI training datasets.
- **Tools:** NDAs, access controls, employee agreements.

---

## PARTNERSHIPS & COLLABORATION

### **Universities (Research Collaborations)**
| Institution | Expertise | Collaboration Type | Status |
|-------------|-----------|-------------------|--------|
| **TU Munich** | Hydro turbine design, CFD | Joint research, PhD supervision | To initiate |
| **EPFL (Switzerland)** | Fluid dynamics, renewable energy | Lab access, co-publication | To initiate |
| **IIT Roorkee (India)** | Small hydro, emerging markets | Field research, local partnerships | To initiate |
| **University of Washington** | Fish passage, ecology | Fish biology research, monitoring | To initiate |
| **Oregon State University** | Hydro engineering, marine energy | Test facilities, validation | To initiate |
| **MIT** | AI/ML, controls, digital twins | Algorithm development, talent | To initiate |
| **Stanford** | AI, impact investing | Business model innovation | To initiate |

**Engagement Model:**
- Sponsored research agreements ($50–200K/project).
- PhD/postdoc co-supervision (industry fellowships).
- Student internships/co-ops (pipeline for hiring).
- Joint publications (credibility, visibility).

### **Industry Consortia & Standards Bodies**
| Organization | Focus | Role | Status |
|--------------|-------|------|--------|
| **IEC TC 4 (Hydraulic Turbines)** | International standards | Propose fish-safe standards, small hydro best practices | To join |
| **ISO TC 301 (Energy Management)** | Energy systems, efficiency | Contribute microhydro guidance | To monitor |
| **Global Off-Grid Lighting Association (GOGLA)** | Off-grid energy quality standards | Adapt standards for microhydro, quality assurance | To engage |
| **Hydraulic Power Association** | Industry advocacy, R&D | Membership, working groups | To join |

### **OEM & Supplier Partnerships**
| Partner Type | Examples | Purpose | Status |
|--------------|----------|---------|--------|
| **Generator OEMs** | ABB, Siemens, Nidec | Joint development, volume pricing | To initiate |
| **Power Electronics** | Infineon, Wolfspeed, SMA | SiC/GaN development, integration | To initiate |
| **Casting/Fabrication** | Local foundries, precision casting shops | DFM collaboration, cost optimization | To initiate |
| **Sensor/IoT Vendors** | Siemens, Schneider, Advantech | Gateway/SCADA integration, co-marketing | To initiate |

---

## PUBLICATION & DISSEMINATION

### **Peer-Reviewed Papers (Target 3–5/year by Year 3)**
- **Journals:** Renewable Energy, Energy, Applied Energy, Ecological Engineering, Water Resources Research.
- **Topics:** Fish passage performance; CFD-validated turbine design; predictive maintenance; carbon MRV; hybrid optimization.
- **Benefit:** Credibility, regulatory acceptance, talent attraction, investor confidence.

### **Conference Presentations (Target 5–10/year)**
- **Conferences:** Hydrovision, ICOLD, ISIE (off-grid energy), ICLR/NeurIPS (AI), AFS (fish passage).
- **Format:** Technical papers, poster sessions, panel discussions, demo booths.

### **Technical Reports & White Papers**
- Publish openly: monitoring data from pilots; fish survival statistics; cost benchmarking; lessons learned.
- Quarterly: R&D updates on website/blog; transparent communication of progress and challenges.

### **Media & Outreach**
- Engage journalists (CleanTechnica, Canary Media, The Guardian environment desk).
- Webinars for stakeholders (investors, regulators, communities).
- Open houses at pilot sites; student field trips.

---

## METRICS & KPIs (R&D ARM)

| Metric | Target (Year 1) | Target (Year 3) | Target (Year 5) |
|--------|-----------------|-----------------|-----------------|
| **System Efficiency (Field)** | 70% | 73% | 75% |
| **Fish Survival Rate** | >90% | >93% | >95% |
| **Cost per kW (Hardware COGS)** | $2.4k | $1.8k | $1.5k |
| **MTBF (Mean Time Between Failures)** | 2000 hrs | 5000 hrs | 8760 hrs (1 yr) |
| **Predictive Maintenance Accuracy** | N/A | 70% | 85% |
| **Patents Filed** | 0–1 | 3–5 | 8–12 |
| **Peer-Reviewed Papers** | 0–1 | 3–5 | 10+ cumulative |
| **University Partnerships** | 1–2 | 5+ | 10+ |
| **PhD/Postdoc Collaborators** | 0 | 2–3 | 5+ |

---

## BUDGET (R&D ARM)

### **Year 1: $500K–800K**
- Personnel (4–5 FTE): $400K
- Lab equipment (shared/leased): $50K
- Materials/prototyping: $50K
- Computing (cloud, licenses): $30K
- Travel/conferences: $30K
- University partnerships: $50K
- Contingency: $40K

### **Year 3: $2–3M**
- Personnel (10–12 FTE): $1.5M
- Lab facility (dedicated): $300K/yr opex
- Equipment/instrumentation: $200K
- Materials/prototyping: $200K
- Computing: $100K
- University partnerships: $200K
- Publication/dissemination: $50K
- Contingency: $150K

### **Year 5: $5–7M**
- Personnel (15–20 FTE): $3M
- Multi-site labs: $1M/yr opex
- Equipment refresh: $500K
- Pilot instrumentation: $300K
- Computing/data: $200K
- University partnerships: $500K
- Standards/IP: $200K
- Dissemination: $100K
- Contingency: $500K

---

## GOVERNANCE

### **R&D Advisory Board**
- **Composition:** 5–7 members; mix of academic (hydro, ecology, AI), industry (OEM, DFI), and independent experts.
- **Mandate:** Review research roadmap; advise on priorities; connect to networks; validate findings.
- **Cadence:** Semi-annual meetings; ad-hoc consultation.

### **Technical Review Process**
- **Design Reviews:** Gate reviews (concept, preliminary, critical, final) for major subsystems; cross-functional (R&D, manufacturing, ops, funding).
- **Field Validation:** All lab results validated in field; statistical rigor; publish data openly (successes and failures).
- **Ethical Review:** Fish studies, community engagement require ethics approval; institutional review board (IRB) for human subjects (surveys, participatory design).

---

## NEXT ACTIONS (IMMEDIATE)

**Week 1–2:**
- Hire/assign interim CTO or Head of R&D (consultant or fractional initially).
- Prioritize Programs 1 (turbine efficiency), 2 (fish-safe), 4 (data/AI) for Year 1.
- Identify university partners for fish passage and turbine optimization; initiate discussions.

**Month 1–3:**
- Secure shared lab space (university or makerspace).
- Procure benchtop test rig or arrange access to existing facilities.
- Instrument Alberta pilots for high-res data collection (Program 4 baseline).

**Month 3–6:**
- Hire 2–3 R&D engineers (fluid dynamics, fish biology, data science).
- Begin CFD optimization studies (Program 1).
- Launch fish monitoring protocol (Program 2); engage third-party validators.

**Month 6–12:**
- Publish interim results from pilots (blog, conference paper).
- Submit first peer-reviewed paper (fish passage or efficiency).
- Apply for research grants (NSF, DOE, provincial innovation funds).
- Expand team to 5–6 FTE.

---

**R&D Arm operational. Innovation pipeline established. Science-driven execution begins.**

