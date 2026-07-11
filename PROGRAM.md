# APEX OMNI — THE PROGRAM (v9.4 →)
### A self-governing quantitative research institution on one laptop.
Everything below is committed. Nothing ships as a mockup: each tranche lands
as complete, validated, deployable code; each component carries a research
anchor and a falsifiable acceptance criterion on YOUR vault. Order is
dependency order, not preference.

---
## PILLAR 1 — STRATEGY FACTORY + GLOBAL TRIAL LEDGER
*Anchor: López de Prado AFML ch.1; Bailey–LdP 2014 (DSR); Harvey–Liu–Zhu 2016.*
- **T1 (SHIPPED): `core/trial_registry.py`** — append-only, program-wide
  ledger of every hypothesis ever examined (primary runs, sensitivity cells,
  nightly forge candidates). All deflation math now charges against it;
  forge_history back-filled. p-hacking becomes arithmetically impossible.
- **T3: generic spec runner** — strategies as declarative specs (trigger
  predicate over the feature/regime vocabulary + exit spec + sizing rule);
  ONE harness runs any spec through backtest→paper→cert. First tenants: the
  upside-flip CE variant (its `upside_zone_candidate_s` counter decides
  admission) and the 0-DTE shortvol variant.
  *Accept:* cascade & shortvol re-expressed as specs reproduce their
  certificates byte-for-byte.

## PILLAR 2 — DEALER-FLOW STATE ENGINE
*Anchor: Barbon–Buraschi 2021; Ni–Pearson–Poteshman 2005; Baltussen et al.
2021; Avellaneda–Lipkin 2003 (pinning); Golez–Jackwerth 2012.*
- **T1 (SHIPPED): `core/dealer_flow.py`** — 1 Hz charm/vanna/pin vector per
  index, recovered analytically from the radar's own per-contract profile
  (the gamma-nowcast trick extended; greeks by central differences on the
  Black-76 pricer — derivation-risk zero, tested against the pricer itself).
  Telemetry + report only: certified gate specs are never widened silently.
  *Accept:* at snapshot spot the vector reproduces published net-GEX exactly;
  vanna call≡put numerically; charm matches direct recomputation.
- **T2: archive the vector** (additive columns) → historical charm/vanna for
  harnesses. **T3:** pin-strength gate variant for shortvol and vanna
  early-warning input for cascade — each as a REGISTERED trial with its own
  harness pass, never a knob edit.

## PILLAR 3 — VOL-SURFACE INTELLIGENCE (true VRP, not its shadow)
*Anchor: Corsi 2009 (HAR-RV); Patton 2011 (QLIKE); Carr–Wu 2009;
Gatheral–Jacquier 2014 (SVI); Dubinsky–Johannes (event variance);
Liu–Patton–Sheppard 2015 (RV sampling).*
- **T1 (SHIPPED): `core/rv_forecaster.py` + `tools/rv_skill_report.py`** —
  HAR-RV on the vault's own 1-minute returns + diurnal-profile remaining-day
  projection; nightly walk-forward skill report (QLIKE + log-RMSE vs
  random-walk and MA5 baselines) writing `state/rv_skill_certificate.json`.
  Brain shows live `rv̂` and the measured VRP spread (IV − forecast).
  *Accept (prespecified):* per index, ≥8 eval days, mean QLIKE(HAR) <
  QLIKE(RW) AND ≥60% daily wins. The forecaster touches NO gate until then.
- **T2: RV-Net** — small torch MLP on 1-second RV features (the 4060's one
  scientifically justified daily training job), gated by *beating the HAR
  certificate*; at <30 vault days a net today would be theater, and this
  program does not do theater. **T2:** multi-expiry arbitrage-free SVI +
  event-variance extraction → the event-calendar gate as a *measured*
  quantity. **T3:** shortvol gate v2 = sell only when IV − rv̂ > margin
  (registered trial).

## PILLAR 4 — COUNTERFACTUAL LEDGER (the constitution audits itself)
*Anchor: off-policy evaluation; Precup–Sutton lineage, applied to gates.*
- **T1 (SHIPPED):** the nightly forge shadow-grades every blocked signal on
  the promotion day — meta-veto, bar near-miss, persistence, throttle,
  quotes, governor — through the SAME ask-entry barrier grader, and reports
  **per-gate P&L attribution** (`gate_attribution` in the forge report +
  rolling `state/gate_attribution.jsonl`). Which rules earn their keep,
  in rupees, forever after.
  *Accept:* attribution counts reconcile with the funnel; capped sampling
  documented per gate.
- **T2:** rolling multi-day attribution dashboard; gate-change proposals
  require an attribution case, then a registered trial.

## PILLAR 5 — EPISTEMIC HEALTH (the exam examines itself)
*Anchor: Gneiting–Raftery 2007 (proper scoring); Murphy decomposition;
Politis–Romano 1994 (stationary bootstrap); LdP ch.12 (CPCV).*
- **T2: calibration monitor** — reliability curves + Brier decomposition for
  every probability the system emits (meta P(win), cert win-rates, pop
  priors); nightly, from the ledger. **T2: living certificates** — rolling
  CIs post-certification; self-de-arm on decay. **T3: stress worlds** —
  stationary-bootstrap + regime-conditional stitching of your OWN tick
  segments (recombined reality, never invented prices); CPCV inside the
  forge. Certificates then report robustness across many plausible
  histories.

## PILLAR 6 — EXECUTION SCIENCE
*Anchor: Cont–Kukanov–Stoikov 2014; RCT methodology.*
- **T3:** empirical fill model from your own walkaway/slip/NOFILL records;
  **randomized execution trials in paper** (cross vs post vs wait, randomly
  assigned per paper order) → execution policy certified by an actual A/B
  experiment. At retail costs, execution IS alpha (your −379% GEX-fade
  backtest died on the toll booth).

## PILLAR 7 — CAPITAL-GRADUATION PROTOCOL
*Anchor: MacLean–Thorp–Ziemba; Grossman–Zhou 1993 (drawdown control).*
- **T4:** paper cert → micro-live (1 lot) behind a **live-fill certificate**
  (measured live slippage must reconcile with paper assumptions) → capital
  steps tied to realized deflated-Sharpe and drawdown, fractional-Kelly from
  certificate CIs; graduation AND demotion automatic. Live spread routing
  (basket orders + Kite SPAN margin API) is part of this tranche and not
  before. LIVE_FIRE remains False until this pillar exists whole.

---
## NEGATIVE SPACE (refusals are the identity)
No naked short options, ever. No latency games a laptop can't win. No
scraped alt-data of unknowable provenance. No LLM narratives as signals.
No discretionary override that bypasses a harness.

## TRANCHE MAP
- **T1 — tonight:** Pillars 1/2/3/4 cores (registry, dealer-flow, HAR-RV +
  skill cert, counterfactual attribution) + registry retrofits everywhere.
- **T2 — next:** calibration monitor, living certificates, dealer-flow
  archival, SVI + event-variance, RV-Net (HAR-gated).
- **T3:** generic spec runner + first registered variants, stress worlds +
  CPCV, execution RCT + fill model, gate-v2 trials.
- **T4:** capital graduation + live routing + live-fill certificate.

Every tranche: complete files, validated before delivery, CONFIG_HASH
discipline, certificates fail-closed, forward evidence blended. Same
constitution, larger cathedral.