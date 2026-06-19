# Apex Omni v9 — audited rebuild

Intraday Indian index-options system for Zerodha Kite Connect, rebuilt from
the v8 audit. Runs locally on an RTX 4060 / i7-13650HX laptop. Ships in
**paper mode** and is designed to stay there until the evidence says
otherwise.

## The two numbers you own

`config.py` is the only file you should need to touch.

`TRADING_CAPITAL = 5000.0` — change this one number and every sizer,
affordability check, drawdown line and disaster floor re-scales. The
RiskGovernor walks the strike ladder (ATM → deeper OTM) and buys the first
rung the account can genuinely **hold**: premium×lot inside the half-Kelly
budget, cash covered with a cost buffer, and the worst-case disaster-floor
loss capped at 30% of capital. If no rung qualifies, it logs `BLOCKED` and
does nothing. In live mode it also reads `kite.margins()` and always
believes the smaller number.

`LIVE_FIRE = False` — stays False. Going live ever requires BOTH flipping
this AND exporting `APEX_CONFIRM_LIVE="I-UNDERSTAND-REAL-MONEY"`. One switch
is an accident; two is a decision.

## Setup (first time)

```bash
pip install -r requirements.txt   # simulator alone needs only numpy
# RTX 4060: pip install torch --index-url https://download.pytorch.org/whl/cu121
cp/edit .env                      # fill KITE_API_KEY + KITE_API_SECRET
chmod 600 .env                    # it's .gitignored; keep it private
python get_token.py               # morning login → writes KITE_ACCESS_TOKEN
```

`config.py` auto-loads `.env` at import; exported environment variables
always override it. The access token dies nightly (SEBI daily logout), so
`python get_token.py` is the one manual step of every trading morning.

Before any LIVE order placement (not needed for data or paper): whitelist
your static IP in the Kite developer console — SEBI's algo framework
requires it for order APIs, and exchanges cap algo order rates at 10/s.

## Runbook

**First run (any time, no broker needed):**
`python simulation/run_simulation.py` — proves the whole decision stack on
your machine in seconds; expect 16/16 PASS and `SIM_REPORT.md`.

**Every trading day (~09:10, four terminals):**

1. `python get_token.py` — fresh access token into `.env` (~20 s).
2. `python data_harvester_v9.py` — WebSocket → tick vault + ring buffer;
   writes the `instrument_snapshots` time machine at open and after close.
3. `python macro_gex_v9.py` — Newton IVs, GEX walls, flip line → JSON/3 min.
4. `python apex_main_v9.py` — the brain (paper). Heuristic physics policy
   until the forge promotes a trained pair.
5. Optional: `python apex_scanner_v9.py` (sweep alerts) and
   `python live_trade_analyzer_v9.py watch NIFTY` (live card).

**After close (one command):**
`python run_nightly.py` → verdict (calibration table) → scenario miner →
full regression suite (writes `state/sim_gate.json`) → forge training. The
forge needs ≥2 harvested days and the torch stack; until both exist it says
so and exits cleanly. A model is promoted only if it beats the incumbent on
an unseen day AND the regression gate is green.

**Logs:** every process writes `logs/<component>_<date>.log` alongside its
console output; the brain adds a 60-second heartbeat (feed age, PnL,
deployed, governor state, open positions). `logs/execution_ledger_v9.csv`
remains the fill-truth record the analyzer judges.

**The suite that grows:** `simulation/scenario_miner.py` scans each real
day for stop-runs, air pockets and feed gaps and appends them to
`discovered_scenarios.json`; the nightly regression then replays today's
actual weirdness against the stack forever after. New parameterized
variants are automatic; a genuinely new *kind* of event still needs a human
to author a generator.

## Real-day replay (zero synthetic data)

```bash
python simulation/replay_real_day.py              # latest harvested day, NIFTY
python simulation/replay_real_day.py 2026-06-12 SENSEX
```

Replays a harvested day — actual recorded Kite ticks — through the complete
live stack: the same StateBuilder physics, the same heuristic policy, the
same RiskGovernor / ExecutionEngine / PositionManager / TrapShield. Every
premium, spread, OI delta and feed gap is what the market actually printed;
paper fills resolve against the day's real books with lookahead taken from
the day's real future quotes. Division of labor: the LIVE pipeline and this
replayer are 100% real Kite data; the 16 authored scenarios exist only as
an offline stress exam for disasters reality won't schedule on demand
(flash crashes, reject storms) and never touch trading.

## No hardcodings

Every tunable — feature physics (VPIN buckets, EWMA half-life, Hawkes
decay, OI/OFI windows, dealer-inventory decay), trap-shield weights and
scales, fusion/advisory weights, heuristic policy weights, ladder depth,
entry bars, throttles, harvester/macro cadences, execution micro-knobs,
SAC hyperparameters — lives in `config.py`, nowhere else. The simulator
reads the SAME constants (ladder depth, attempt throttle), so sim == live
by construction. Only pure algorithm internals (Newton iteration counts,
SVI seed parameters) remain in code. The `lot_fallback`/`strike_step`
values in INDICES are cold-start fallbacks only: live values always come
from the Kite instrument dump, which is why the Jan-2026 lot change cost
this system nothing.

## The Trap Shield (anti-"institutional flush")

While in a trade, a stop-loss breach is not obeyed blindly. The shield
scores six fingerprints of a stop-hunt: vertical velocity anomaly, heavy
selling being absorbed at the lows, ΔOI refusing to confirm the break,
option premium dislocating beyond what delta justifies, bid/ask blowing out
(liquidity pulled), and proximity to the GEX put-wall where retail stops
cluster. Score ≥ 0.60 → the stop is suspended for ≤150 s. Non-negotiables:
the disaster floor (1.6× stop distance, capped −45% of premium) ALWAYS
fires; OI confirming the break releases the shield early; a reclaimed flush
re-anchors the stop under the hunt's low; two uses per trade, maximum. Its
evil twin — a genuine breakdown with fresh OI committing — is deliberately
in the test suite, and the shield stands aside for it.

## Simulator

```bash
python simulation/run_simulation.py
```

Sixteen synthetic but microstructure-honest trading days (1-second ticks,
Black-76-repriced premiums, full Zerodha+statutory costs) run against the
REAL RiskGovernor / ExecutionEngine / PositionManager / TrapShield — only
the exchange is fake. Current build: **16/16 PASS**; see `SIM_REPORT.md`
and per-scenario ledgers in `logs/sim_*.csv`. No broker, no GPU, no network
needed.

## File map

`core/` — quant_core (Black-76, Newton IV, per-expiry SVI, real VPIN),
market_state (THE one StateBuilder: 19 features × 30 nodes × 10 frames),
trap_shield, risk_manager, execution_engine (fill truth + cost model),
position_manager (exit ladder), graph_constructor (TGN, torch-guarded),
instruments (date-keyed mapper + as-of snapshots). Top level — harvester,
macro GEX, forge, brain, scanner, analyzer, `apex_ipc_core.py` (seqlock
mmap ring). `simulation/` — engine, 16 scenarios, runner.

## Weapons rack (every Kite Connect capability, wired)

WebSocket order updates (postback-equivalent pushes, covering ALL orders
including GTT triggers) are captured by the harvester and consulted by the
execution engine before any REST poll. India VIX streams as a regime input:
a +4%/5-min VIX spike temporarily raises the entry bar. The macro radar now
also publishes band-wide PCR, the max-pain strike, and an IV-rank built from
its own persisted daily ATM-IV history — the brain consumes all three as
small config-weighted advisories (PCR extremes contrarian, max-pain gravity
on expiry day only, high IV-rank demands more edge before buying premium).
Previous-day high/low/close levels are fetched from real day candles at
startup and confirm breakouts. In live mode the affordability check upgrades
its 2% heuristic to EXACT broker charges via the order-margins API, and an
optional GTT server-side disaster floor (config `LIVE_GTT_FLOOR`; NRML
product, 250-GTT account cap, GTC — read the code comment before enabling
with MIS intraday) survives a dead laptop. `tools/backfill_history.py`
pulls years of real minute/day candles for the indices + VIX (and the
current option chain with OI) into the vault — knowing that Kite keeps no
minute history for expired options, which is why your live tick vault is
the one dataset money cannot re-buy.

## Paper = live (only the order isn't placed)

The single difference between paper and live is that paper places no real
order on your Kite account. Identical in both: every feature and the physics
behind it, the policy, the meta-labeler's win-probability, the trap shield
and its disaster floor, every governor, every macro weapon, the conviction
threshold (0.70 both), the uncalibrated win-probability (0.52 both), and —
because the calls are read-only and place nothing — the EXACT broker charges
from the order-margins API. Paper decisions therefore predict live decisions
exactly. Two documented asterisks, both direct consequences of "no order
placed", not extra divergences: (1) fills are MODELLED, since you cannot know
your queue position without submitting an order — with `PAPER_FILL_REALISM`
(default on) a resting maker order fills only if the market trades strictly
through your limit, a deterministic behind-the-queue proxy, so paper stops
assuming it always gets the touch; (2) the server-side GTT floor is
logged-not-placed in paper (a GTT is a real order) while the in-process floor
runs identically. Live additionally caps cash at the broker balance — fund
your account to ≥ TRADING_CAPITAL and even that vanishes. Set `PAPER_EXPLORE
= True` only if you deliberately want paper to trade more aggressively to
build the certificate faster, accepting it no longer mirrors live.

## Research foundations (why these mechanisms exist)

SEBI's own studies anchor everything: ~93% of individual F&O traders lost
money FY22–FY24 (₹1.8 lakh crore aggregate), only ~1% cleared ₹1 lakh after
costs, the profits went to algorithmic entities — and over 75% of losers
kept trading anyway. That last finding is a psychology bug this system makes
structural: `core/edge_audit.py` bootstraps your real after-cost ledger and
LIVE_FIRE physically cannot arm without a fresh statistical Edge Certificate
(≥100 trades, ≥20 sessions, 95% CI of mean PnL above zero). Sizing follows
López de Prado's meta-labeling: the primary model picks the SIDE, a nightly
triple-barrier meta-model trained on your vault's real prices learns the
SIZE as P(win|features), blended with the empirical calibration table into
the Kelly input. The Kelly budget itself is volatility-managed (scales down
when implied vol runs above target — the volatility-managed-portfolios
result). Late-day conviction gets the Gao–Han–Li–Zhou intraday-momentum
nudge (first half-hour return predicts the last half-hour; stronger on
volatile days). Order-flow imbalance (Cont–Kukanov–Stoikov) was already a
core feature; VPIN is retained with the Andersen–Bondarenko critique in
mind — the meta-model learns its true weight rather than trusting it.
Expiry days run a tighter theta guillotine. Every ledger row and verdict
carries the CONFIG_HASH it traded under — results are attributable to exact
parameters, always.

## Feature-drift monitor (the live regime-shift guard)

A certified edge is proven on past tape; the day the market enters a regime
the model never trained on, its win-probabilities become extrapolation. The
forge writes a per-feature REFERENCE PROFILE (quantile bins + histogram) of
the exact features the meta-model learned. Live, the brain feeds every frame
through the same StateBuilder into a rolling window and each heartbeat
computes, per signal-carrying feature, the Population Stability Index and a
two-sample Kolmogorov–Smirnov D against that reference. The share of features
breaching threshold grades the tape GREEN / WATCH / DRIFTED; the heartbeat
line shows it live (`drift GREEN`). DRIFTED becomes a FOURTH live-arming lock
— even a valid Edge Certificate cannot place a real order while the regime
has left the model's training world — and it clears automatically when the
next nightly forge re-references the model (or the regime reverts). Paper
never stops, so you keep learning through the drift. Inspect anytime with
`python live_trade_analyzer_v9.py drift`. Honest scope: this catches
DISTRIBUTIONAL shift, the failure mode that silently invalidates a trained
model — it cannot catch an edge that decays while the feature distribution
stays put; only live P&L and the expiring certificate catch that.

## Entry: maker vs taker (researched, not a shortcut)

A passive maker buy can only fill when the option ticks DOWN through your
resting price — but a bullish momentum signal fires precisely when the option
is RISING, so pure-maker entries starve on exactly the trades you want (you
will see long runs of `NOFILL ... walked away`). Microstructure research
settles this: momentum/continuation entries should CROSS the spread (the
half-spread is the toll for not missing the move); only mean-reversion
entries rest passively. Apex's heuristic is momentum, so on conviction ≥
`ENTRY_CROSS_CONVICTION` it crosses — taking the ask — under three guards
that stop crossing from becoming blind chasing: (1) a slippage cap of
`ENTRY_SLIP_CAP_PCT` of one strike-step's worth of premium move, measured
from the decision-time micro-price — if the option has already run past it,
the signal is stale and the brain WALKS AWAY rather than buy exhaustion
(critical at 0–2 DTE where chasing buys rich premium into accelerating
decay); (2) crossing only when the spread is inside `ENTRY_CROSS_SPREAD_PCT`,
a band tighter than the hard `MAX_ENTRY_SPREAD_PCT` reject, because immediacy
across a wide spread is a bad trade even on a good signal; (3) a hard ceiling
of `ENTRY_CROSS_CAP_TICKS` past the ask. Marginal signals (below the cross
bar) stay passive and save the spread — which is also why choppy flip-flop
tape costs less than it would if every whipsaw crossed. The filled price
recorded is the TRUE ask paid, so the edge certificate reflects real entry
economics. Paper and live use identical entry logic.

## Honest limits

Paper fills are deterministic and slightly optimistic in dead feeds; the
shield is a probabilistic defense, not immunity (the floor is the price of
that humility); calibration starts empty — let the analyzer build ≥20
trades per conviction bucket in paper before trusting any sizing. Nothing
here is financial advice; it's machinery.
