# 25 Validation Requirements

**Purpose:** Define measurable, testable validation criteria across design, engineering, business, and community dimensions. Each requirement traces to our vision, values, and workflow framework.

**Status:** Framework—validation occurs throughout workflow phases.  
**Last Updated:** January 25, 2026

---

## Category A: Vision & Values Alignment (Requirements 1-5)

### Requirement 1: Historic Mill Heritage Integration
**What:** Design must visibly incorporate or reference waterwheel/windmill principles (elegant simplicity, durability, efficiency).  
**Why:** Core to vision ("honor the legacy of mills and waterwheels").  
**How to validate:** 
- Engineering review: Does the final design embody 3+ principles from [HISTORIC_MILLS_VISION_ARCHIVE.md](HISTORIC_MILLS_VISION_ARCHIVE.md) Part 3 (design lessons)?
- Customer feedback: Do customers perceive the design as "honoring heritage" in pilot surveys?
- Visual assessment: Is the mechanism visible, understandable, and celebrated (not hidden)?

**Gate:** Phase 3.4 (design review), Phase 6.3 (prototype validation)

---

### Requirement 2: 30+ Year Durability Validation
**What:** Product must be designed and tested for 30+ year service life per [ETHICS_VALUES_BRAND.md](ETHICS_VALUES_BRAND.md) core value #2.  
**Why:** Durability & Longevity is core value; differentiates from disposable competitors.  
**How to validate:**
- Material testing: Verify material corrosion rates, fatigue limits, and degradation curves under operating conditions
- Simulation: Run 30-year digital model (see [SIMULATION_FRAMEWORK.md](SIMULATION_FRAMEWORK.md) scenarios) and confirm no failure modes below 30 years
- Accelerated testing: Subject prototype components to equivalent 10-year wear in 1-year lab testing; document results
- Bearing/seal replacement: Document planned replacement intervals and costs; verify spares are available and affordable

**Gate:** Phase 4.4 (simulation complete), Phase 6.3 (prototype validation)

---

### Requirement 3: Simplicity & Elegance Metric
**What:** Design must minimize unnecessary complexity while maximizing aesthetic appeal and understandability.  
**Why:** Core value #3 (Simplicity & Elegance); mill designs prove elegance drives long-term success.  
**How to validate:**
- Component count: Final design ≤ [X] moving parts (TBD based on product type)
- Design review: Can an intelligent customer understand the mechanism in <5 minutes of explanation?
- Aesthetic evaluation: Does the design photograph well? Would it appear in an engineering textbook or design magazine?
- Maintenance simplicity: Can any skilled technician perform routine service with basic tools?

**Gate:** Phase 3.4 (design review), Phase 6.3 (prototype), Phase 9.4 (final design)

---

### Requirement 4: Local Resilience & Community Impact
**What:** Product design and manufacturing must support local economies and community scale energy generation.  
**Why:** Core value #5 (Local Resilience & Community); mission commits to "locally-controlled, distributed energy infrastructure."  
**How to validate:**
- Design for local sourcing: ≥80% of components available from suppliers within [X] miles or [X] countries
- Design for local assembly: Can [X]% of assembly be performed by trained local technicians without proprietary tools?
- Community benefit: Document how customer ownership benefits local economy (jobs, knowledge, energy sovereignty)
- Scalability: Is system viable for community scale (10–1000 kW range) without requiring centralized grid?

**Gate:** Phase 7.4 (sourced unit validation), Phase 9.4 (final design optimization)

---

### Requirement 5: Environmental Footprint & Eco-Responsibility
**What:** Product and manufacturing must align with environmental commitments from [ETHICS_VALUES_BRAND.md](ETHICS_VALUES_BRAND.md).  
**Why:** Core value #4 (Connection to Nature & Place); sustainability pledge includes carbon-neutral manufacturing and end-of-life recyclability.  
**How to validate:**
- Material sourcing: Verify all materials are from responsible sources (no conflict minerals, preference for recycled/local)
- Manufacturing carbon: Measure manufacturing footprint per unit; confirm carbon offset plan
- End-of-life design: Material passport created; ≥90% of components recyclable or safely biodegradable
- Water impact: Installation causes minimal ecosystem disruption; environmental impact assessment completed
- Lifecycle assessment (LCA): Third-party LCA shows product carbon payback <[X] years of operation

**Gate:** Phase 3.4 (design review), Phase 9.4 (final design), Phase 10 (company audit)

---

## Category B: Engineering & Performance (Requirements 6-12)

### Requirement 6: Efficiency Target Achievement (75%+)
**What:** System efficiency ≥75% under nominal operating conditions.  
**Why:** Significantly exceeds historic waterwheel efficiency (50–80%); demonstrates modern engineering advantage.  
**How to validate:**
- Simulation: Digital model confirms ≥75% efficiency across baseline, extreme, and degradation scenarios
- Prototype testing: Bench test of physical prototype shows measured efficiency within ±5% of simulation
- Sourced unit testing: COTS version maintains ≥73% efficiency (accounting for component variation)
- Long-term: Field testing of production units confirms efficiency stability over 1+ years

**Gate:** Phase 4.4 (simulation), Phase 6.3 (prototype), Phase 8.3 (dual simulation), Phase 9.4 (final design)

---

### Requirement 7: Multi-Year Simulation Validation
**What:** Complete simulation across 30-day, 6-month, 1-year, 2-year, 5-year, 30-year, and 100-year timeframes (per [SIMULATION_FRAMEWORK.md](SIMULATION_FRAMEWORK.md)).  
**Why:** Validates durability claims; identifies failure modes before field deployment; enables confident 30-year warranty.  
**How to validate:**
- All scenarios complete: 7 simulation runs finished with no convergence errors or unrealistic outputs
- Sensitivity analysis: Tested ±10% variation on high-impact parameters; results show robust performance
- Validation vs. test data: Simulation predictions match prototype physical test data within ±10%
- Failure mode identification: Model identifies top 3 stress points; design mitigations confirmed in prototype

**Gate:** Phase 4.4 (initial simulation), Phase 8.3 (dual simulation), Phase 9.4 (final design validation)

---

### Requirement 8: Prototype vs. Simulation Correlation
**What:** Physical prototype performance within ±10% of digital model predictions.  
**Why:** Confirms simulation framework validity; enables confident scaling to production units.  
**How to validate:**
- Test plan: Identical tests run on prototype and simulated model (flow, pressure, power output, efficiency)
- Data comparison: Plot measured vs. predicted; confirm ≥90% correlation coefficient
- Variance analysis: Document any discrepancies; root cause analysis for variations >10%
- Design updates: Any significant mismatches feed back to CAD and simulation model refinement

**Gate:** Phase 6.3 (prototype validation), Phase 8.3 (dual simulation gate)

---

### Requirement 9: Prototype vs. Sourced Unit Performance (≤15% delta)
**What:** COTS component version maintains ≥85% of prototype performance (tolerance: <15% variance).  
**Why:** Confirms producibility and commercial viability; ensures customers receive consistent value.  
**How to validate:**
- Identical test matrix: Same tests run on prototype and sourced unit under identical conditions
- Performance curves: Plot efficiency, power output, durability indicators for both
- Tolerance analysis: Sourced unit ≥85% of prototype performance across all metrics
- Cost impact: Document cost reduction achieved by sourcing; validate ROI improvement

**Gate:** Phase 7.4 (sourced unit validation), Phase 8.3 (dual simulation approval)

---

### Requirement 10: Bill of Materials (BOM) Completeness & Sourcing
**What:** Final BOM 100% specified with confirmed suppliers, lead times, and unit costs.  
**Why:** Enables cost modeling, production planning, and supply chain risk management.  
**How to validate:**
- BOM documentation: All [X] components listed with part number, supplier, lead time, unit cost, lifecycle cost
- Supplier confirmation: ≥90% of components have signed quotes from at least 1 supplier (preferably 2+)
- Lead time assessment: Longest lead time <[X] weeks; critical components have backup suppliers
- Cost stability: Suppliers confirm pricing stable for [X] months ahead; long-term supply contracts planned

**Gate:** Phase 3.4 (design review), Phase 7.4 (sourced unit), Phase 9.4 (final design), Phase 10.1 (company audit)

---

### Requirement 11: Bearing & Seal Lifecycle Specification
**What:** All bearing and seal components tested and specified with documented replacement intervals and spare parts availability.  
**Why:** Bearings/seals are highest-wear items; serviceability critical to 30-year durability goal.  
**How to validate:**
- Material selection: Bearings and seals selected for wet environment (corrosion-resistant materials confirmed)
- Testing: Accelerated wear testing confirms replacement interval <[X] years
- Spare parts kit: Standard replacement kits documented; pricing made available to customers
- Installation procedure: Service manual includes step-by-step bearing replacement with no specialized tools required

**Gate:** Phase 3.4 (design), Phase 6.3 (prototype), Phase 9.4 (final design)

---

### Requirement 12: Thermal & Structural Stress Validation
**What:** Finite Element Analysis (FEA) confirms no failure modes under nominal, extreme, and worst-case loading.  
**Why:** Prevents field failures; validates material selection and geometry; supports warranty confidence.  
**How to validate:**
- FEA model: All major components modeled with correct material properties and boundary conditions
- Nominal case: Stress <[Y]% of material yield under nominal operating conditions
- Extreme case: Stress <[Y]% of material yield under 1.5x nominal load (safety factor [Z])
- Fatigue analysis: Miner's rule cumulative damage <1.0 over 30-year operating life
- Hot spots identified: FEA locates stress concentrations; geometry refined or material upgraded as needed

**Gate:** Phase 3.4 (design review), Phase 5.3 (redesign approval), Phase 9.4 (final design)

---

## Category C: Design & Manufacturing (Requirements 13-17)

### Requirement 13: CAD Design Completeness & Precision
**What:** Final CAD model includes all components with correct tolerances, assemblies, and export documents.  
**Why:** Enables manufacturing, quality control, and operator maintenance; referenced in [CAD_STATE_AND_VERSION.md](CAD_STATE_AND_VERSION.md).  
**How to validate:**
- Model content: All [X] components modeled; assemblies show correct fit and motion
- Tolerance stack-up: Assembly tolerances analyzed; manufacturing tolerances specified for critical dimensions
- Export completeness: STEP, STL, DXF, and PDF drawings generated from CAD; all match physical prototype
- Version control: CAD design locked with version number, date, and owner before manufacturing release

**Gate:** Phase 3.4 (design review), Phase 9.4 (final design lock)

---

### Requirement 14: Manufacturing Process Definition & Quality Control
**What:** Complete manufacturing procedure documented with quality checkpoints and acceptance criteria.  
**Why:** Ensures consistent, high-quality production; supports [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md) KPIs (first-pass yield ≥[X]%, DPM <[X]).  
**How to validate:**
- Process flowchart: Step-by-step manufacturing sequence documented with time estimates
- Quality checkpoints: Inspection points identified at [X] stages; acceptance criteria clear and measurable
- Tooling & fixtures: Special tooling or fixtures designed and validated; CAD drawings provided
- First Article Inspection (FAI): FAI procedure documented; first production unit passes 100% inspection before series production

**Gate:** Phase 10.1 (company audit), Phase 11 (pre-production gate)

---

### Requirement 15: Local Assembly & Training Capability
**What:** Design and documentation enable trained local technicians to assemble and service units with basic tools.  
**Why:** Supports local resilience value; reduces customer cost and builds community expertise.  
**How to validate:**
- Assembly manual: Step-by-step illustrated guide with estimated assembly time (<[X] hours for fully-trained tech)
- Tool list: All tools required for assembly and service listed; none are specialized/proprietary
- Training program: Curriculum developed and piloted; ≥[X] technicians certified per region
- Feedback: Technician feedback on assembly/service procedures collected; documentation refined

**Gate:** Phase 7.4 (sourced unit), Phase 10 (company audit), Phase 11 (pre-production)

---

### Requirement 16: Design Change Control & Traceability
**What:** All design changes documented with rationale, impact analysis, and approval.  
**Why:** Maintains design integrity; prevents field failures from undocumented changes; supports warranty and liability.  
**How to validate:**
- Change log: All design changes tracked in [CAD_STATE_AND_VERSION.md](CAD_STATE_AND_VERSION.md) with version, date, owner, reason
- Impact analysis: Each change assessed for effects on performance, cost, manufacturing, serviceability
- Approval: Changes require sign-off from engineering lead and product owner before implementation
- Prototype update: Prototype components updated to match final approved CAD; discrepancies resolved

**Gate:** Phase 5.3 (redesign), Phase 9.4 (final design lock), Phase 11 (production release)

---

### Requirement 17: Spare Parts & Field Service Documentation
**What:** Comprehensive spare parts list, pricing, and field service documentation available before production launch.  
**Why:** Enables customers to maintain systems for 30+ years; supports durability value and business sustainability.  
**How to validate:**
- Spare parts kit: Standard kits defined (annual maintenance kit, 5-year major service kit, emergency repair kit)
- Pricing & availability: All spare parts quoted and procurement plan confirmed
- Service manual: Complete field service guide with troubleshooting, maintenance schedules, and diagrams
- Logistics: Spare parts shipping and delivery time <[X] days anywhere in [service region]

**Gate:** Phase 9.4 (final design), Phase 10 (company audit), Phase 11 (pre-production)

---

## Category D: Business, Economics & Operations (Requirements 18-22)

### Requirement 18: ROI Modeling & Customer Economics
**What:** Financial model demonstrates clear ROI within [X] years for typical customer installation.  
**Why:** Economic viability is core value #7; mission commits to "ROI within predictable timeframes."  
**How to validate:**
- Customer profiles: [3-5] typical customer scenarios developed (residential, small community, industrial)
- Financial model: For each scenario, calculate upfront cost, annual energy savings, maintenance costs, ROI timeline
- Sensitivity: Model shows ROI across ±25% variation in energy prices, maintenance costs, component life
- Conservative case: Even in pessimistic scenario (low energy prices, high maintenance), ROI >0 by Year [X]
- Customer feedback: Pilot customers confirm financial model matches their experience

**Gate:** Phase 9.4 (final design optimization), Phase 10 (company audit), Phase 11 (production approval)

---

### Requirement 19: Production Cost Target & Gross Margin
**What:** Manufacturing cost per unit confirmed ≤[X]; gross margin ≥[Y]%.  
**Why:** Enables sustainable business; funds R&D, support, and community reinvestment per values.  
**How to validate:**
- Detailed cost build: BOM cost + labor + overhead + quality + logistics = target unit cost
- Supplier quotes: Locked quotes for all major components; volume discounts negotiated
- Manufacturing labor: Standard work documented; labor rates confirmed with manufacturing partner
- Margin analysis: Target price set to achieve [Y]% gross margin; validated with market research
- Cost reduction roadmap: Year 1-3 cost reduction targets identified (e.g., tooling ROI, supplier scaling)

**Gate:** Phase 7.4 (sourced unit cost validation), Phase 10.1 (company audit), Phase 11 (pricing approval)

---

### Requirement 20: Supply Chain Resilience & Risk Mitigation
**What:** Supply chain stress-tested for disruptions; single-source risks mitigated; alternative suppliers qualified.  
**Why:** Supports [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md) KPI: "On-time supplier delivery ≥95%."  
**How to validate:**
- Supplier assessment: All critical suppliers evaluated for financial health, quality, and geographic diversity
- Dual-source plan: Components with single-source risk have qualified backup supplier confirmed
- Inventory strategy: Safety stock levels calculated for critical long-lead items
- Disruption scenario: Supply chain tested against [3-5] scenarios (supplier failure, shipping delay, geopolitical, pandemic)
- Contingency plan: For each high-impact risk, documented mitigation or alternative sourcing

**Gate:** Phase 10 (company audit), Phase 11 (pre-production), ongoing

---

### Requirement 21: Production Ramp Feasibility & Scaling
**What:** Manufacturing capacity and supply chain can scale from pilot (Month 3-4) to full production (Month 8) per [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md).  
**Why:** Validates business scalability; prevents bottlenecks or quality issues during ramp.  
**How to validate:**
- Capacity analysis: Manufacturing partner confirms ability to ramp [X] units/month (Month 3) → [Y] units/month (Month 8)
- Supplier scalability: Suppliers confirm ability to supply components at ramp schedule with <[X]% price increase
- Quality impact: Manufacturing and quality leads assess risk of quality degradation during ramp; mitigation plans confirmed
- Resource planning: Personnel requirements (manufacturing, quality, logistics) mapped for each ramp phase
- Pilot production: First [X] production units built with intensive inspection; issues identified and resolved before series ramp

**Gate:** Phase 10 (company audit), Phase 11 (pre-production gate)

---

### Requirement 22: Warranty & Customer Support Model Definition
**What:** Warranty terms, support services, and cost implications clearly documented.  
**Why:** Reflects durability confidence; critical to customer trust and long-term ROI.  
**How to validate:**
- Warranty terms: Length, coverage (parts vs. labor), conditions clearly stated and documented
- Support model: Phone/email support availability, field service options, replacement turnaround documented
- Cost modeling: Warranty and support costs estimated as % of revenue; built into pricing model
- Competitive analysis: Warranty terms benchmarked against market leaders; positioning clear
- Customer communication: Warranty terms easy to understand; marketing materials emphasize durability confidence

**Gate:** Phase 9.4 (final design), Phase 10 (company audit), Phase 11 (production approval)

---

## Category E: Community, Sustainability & Brand (Requirements 23-25)

### Requirement 23: Community Engagement & Impact Program
**What:** Documented plan for community education, apprenticeship support, and local reinvestment per [ETHICS_VALUES_BRAND.md](ETHICS_VALUES_BRAND.md).  
**Why:** Core value #5 (Local Resilience & Community); sustainability commitment includes 2% annual revenue to community initiatives.  
**How to validate:**
- Program plan: [3-5] specific community initiatives defined (e.g., apprenticeship scholarship, water conservation workshop, renewable energy education)
- Budget: 2% of projected Year 1 revenue allocated to community programs; responsible organization identified
- Partnerships: Agreements signed with [2-3] local vocational schools, NGOs, or community organizations
- Pilot execution: Pilot community program executed in first 6 months; results documented and refined
- Scaling: Plan for expanding community programs as company grows; impact metrics defined

**Gate:** Phase 10 (company audit), Phase 11 (execution plan), ongoing

---

### Requirement 24: Brand Consistency & Heritage Integration
**What:** All products, marketing, operations, and communications embody mill heritage and brand values per [BRANDING_LOGO_CONCEPT.md](BRANDING_LOGO_CONCEPT.md) and [HISTORIC_MILLS_VISION_ARCHIVE.md](HISTORIC_MILLS_VISION_ARCHIVE.md).  
**Why:** Brand differentiation; customer trust; internal culture alignment.  
**How to validate:**
- Visual consistency: Logo, color palette, typography applied consistently across website, materials, packaging
- Message alignment: Marketing copy and customer communications reflect brand voice (timeless, elegant, grounded, inspiring)
- Heritage presence: Customer-facing materials reference mill heritage and engineering lessons; story well-told
- Internal culture: Team understands and can articulate company values; hiring and promotion decisions aligned
- Customer perception: Customer surveys confirm perceived brand alignment with messaging (timeless, durable, community-focused)

**Gate:** Phase 1.2 (brand approval), Phase 11 (launch), ongoing

---

### Requirement 25: Lifecycle Impact & Eco-Responsibility Verification
**What:** Third-party lifecycle assessment (LCA) confirms environmental benefits and responsible sourcing throughout product life.  
**Why:** Validates sustainability claims; supports eco-responsibility value; differentiates in market.  
**How to validate:**
- LCA scope: Assessment covers material extraction, manufacturing, shipping, installation, 30-year operation, end-of-life
- Carbon payback: LCA shows product carbon payback <[X] years of operation (lower is better; target: <5 years)
- Sourcing audit: Third-party audit confirms responsible material sourcing; no conflict minerals or unethical labor
- End-of-life certification: [X]% of product materials certified recyclable; process for safe disposal documented
- Transparency: LCA results and sourcing audit published on website/marketing; shows commitment to accountability
- Continuous improvement: Year-over-year LCA shows improving environmental footprint (manufacturing carbon, material efficiency)

**Gate:** Phase 9.4 (final design), Phase 10 (company audit), Phase 11 (launch), ongoing

---

## Validation Cadence & Ownership

| Requirement | Primary Gate(s) | Owner | Frequency |
|-------------|-----------------|-------|-----------|
| 1–5 (Vision/Values) | Phase 1.2, 3.4, 6.3, 9.4 | Product lead, brand lead | Design phases + pre-launch |
| 6–12 (Engineering) | Phase 4.4, 6.3, 8.3, 9.4 | Engineering lead, simulation lead | Simulation + prototype phases |
| 13–17 (Design/Mfg) | Phase 3.4, 5.3, 9.4, 11 | CAD lead, manufacturing engineer | Design + pre-production |
| 18–22 (Business/Ops) | Phase 9.4, 10, 11 | Finance, operations, procurement | Company audit + launch |
| 23–25 (Community/Brand) | Phase 1.2, 10, 11 | Brand lead, community manager | Pre-launch + ongoing |

---

## Sign-Off & Tracking

**All 25 requirements must be validated before Phase 11 production execution approval (Gate 10.4).**

- [ ] Requirement 1–5: Leadership review & approval
- [ ] Requirement 6–12: Engineering & simulation validation
- [ ] Requirement 13–17: Design & manufacturing confirmation
- [ ] Requirement 18–22: Finance & operations audit
- [ ] Requirement 23–25: Brand & community sign-off

**Tracking location:** [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md) Quality & Community KPIs section

