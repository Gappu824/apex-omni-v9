"""
APEX OMNI v9 — CONFIG / RISK CONSTITUTION
=========================================
Every constant the system reads lives HERE and only here. No getattr()
fallbacks anywhere in v9: if it isn't defined in this file, the code refuses
to start rather than inventing a default (audit finding: phantom ₹40,000
capital fallback in v8).

★ = the knobs you (the human) are expected to touch.
"""

import logging
import os
import sys
import io
import datetime as _dt
from pathlib import Path

# ----------------------------------------------------------------------------
# WINDOWS UTF-8 SAFETY NET (the ₹ symbol and any unicode in logs/ledgers/JSON
# crash under the legacy cp1252 codec Windows still defaults to). Forcing
# UTF-8 here — at the top of the one module every entrypoint imports first —
# makes every read_text/write_text/open/print/CSV row safe platform-wide
# without annotating 30 call sites. No-op on Linux/macOS (already UTF-8).
# ----------------------------------------------------------------------------
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                      # noqa: BLE001
    pass

# ----------------------------------------------------------------------------
# .env SUPPORT — loads BASE-DIR/.env into the environment (existing env vars
# always WIN, so an exported variable overrides the file). The file is
# .gitignored; keep it chmod 600. Convenience vs purity: the tradeoff is
# yours, the default is at least never-in-git.
# ----------------------------------------------------------------------------
def _load_dotenv():
    p = Path(__file__).resolve().parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not v.startswith("..."):
            os.environ.setdefault(k, v)

_load_dotenv()


def setup_logging(component: str, level=logging.INFO):
    """Console + daily file (logs/<component>_<date>.log) for every process.
    Call once at the top of each entrypoint."""
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f"{component}_{_dt.date.today()}.log",
                             encoding="utf-8")
    logging.basicConfig(level=level, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(), fh], force=True)
    logging.getLogger(component).info(
        "=== %s start | %s | capital ₹%.0f | LIVE_FIRE=%s (armed=%s) ===",
        component, VERSION, TRADING_CAPITAL, LIVE_FIRE, live_fire_armed())

# ----------------------------------------------------------------------------
# IDENTITY / SAFETY
# ----------------------------------------------------------------------------
VERSION = "v9.0-audited"

# ★ PAPER MODE. Stays False. To ever go live you must BOTH set this True AND
#   export APEX_CONFIRM_LIVE="I-UNDERSTAND-REAL-MONEY". One switch is an
#   accident; two is a decision.
LIVE_FIRE = False
LIVE_CONFIRM_ENV = "APEX_CONFIRM_LIVE"
LIVE_CONFIRM_PHRASE = "I-UNDERSTAND-REAL-MONEY"

def live_fire_armed() -> bool:
    """FOUR locks, all required: the LIVE_FIRE flag, the confirmation phrase,
    a FRESH Edge Certificate (statistical proof from the paper ledger that
    this account clears the bar SEBI shows ~93% of individuals never clear),
    and NO active feature-drift — the live model may only bet real money on
    a market regime that resembles what it trained on. Fail any one → paper."""
    if not (bool(LIVE_FIRE) and
            os.environ.get(LIVE_CONFIRM_ENV, "") == LIVE_CONFIRM_PHRASE):
        return False
    try:
        import json as _j, time as _t
        c = _j.loads(EDGE_CERT_PATH.read_text())
        fresh = (_t.time() - float(c.get("ts", 0))) < \
            EDGE_CERT_VALID_DAYS * 86400
        same_cfg = c.get("config_hash") == CONFIG_HASH
        if not (bool(c.get("ok")) and fresh and same_cfg):
            return False
    except Exception:                                  # noqa: BLE001
        return False
    # Fourth lock: live regime-drift de-arms even a valid certificate.
    try:
        from core.drift_monitor import drift_blocks_live
        if drift_blocks_live():
            return False
    except Exception:                                  # noqa: BLE001
        pass
    return True

# ----------------------------------------------------------------------------
# PAPER ↔ LIVE PARITY CONTRACT
# ----------------------------------------------------------------------------
# THE ONLY DIFFERENCE between paper and live is that paper places no real order
# on your Kite account. Everything else — features, physics, policy, the
# meta-labeler, the trap shield, the disaster floor, every governor, every
# macro weapon, conviction thresholds, win-probability handling, exact
# broker charges, startup reconcile — is byte-identical. Paper decisions
# therefore exactly predict live decisions.
#
# Two unavoidable, fully-documented asterisks (both are consequences of "no
# real order placed", not extra divergences):
#   • Fills: you cannot know your queue position without submitting an order,
#     so paper MODELS fills (see PAPER_FILL_REALISM) instead of measuring them.
#   • Server-side GTT floor: placing a GTT *is* placing a real order, so in
#     paper it is logged-not-placed; the in-process floor runs identically.
#
# PAPER_EXPLORE breaks parity ON PURPOSE if you want paper to trade more
# aggressively (lower bar, exploratory win-prob) to build the calibration
# table / Edge Certificate faster. Default False = paper mirrors live exactly.
PAPER_EXPLORE = False

# Paper fill realism. The one thing physically impossible to make identical to
# live without submitting a real order. When True (default), a resting maker
# order fills only if the market trades STRICTLY THROUGH your limit — a
# deterministic proxy for sitting behind the queue at your price — so paper
# stops assuming it always gets the touch. Tune toward your real fills later.
PAPER_FILL_REALISM = True

def entry_conviction_bar() -> float:
    """Conviction needed to enter. Identical in paper and live unless you
    explicitly opt into PAPER_EXPLORE."""
    return PAPER_ENTRY_CONVICTION if (PAPER_EXPLORE and not live_fire_armed()) \
        else ENTRY_CONVICTION

def uncalibrated_winprob() -> float:
    """What an unproven conviction is worth before calibration. Identical in
    paper and live unless PAPER_EXPLORE."""
    return PAPER_EXPLORE_WINPROB if (PAPER_EXPLORE and not live_fire_armed()) \
        else UNCALIBRATED_WINPROB

# ----------------------------------------------------------------------------
# ★ CAPITAL  (the single variable you asked for)
# ----------------------------------------------------------------------------
# Change this one number and every sizer, affordability check, drawdown line
# and disaster floor in the system re-scales itself. In live mode the engine
# additionally syncs against kite.margins() and always uses the SMALLER of
# (this number, broker available cash) — the bot may never believe it has
# more money than the broker says.
TRADING_CAPITAL = 60000.0          # ₹
FORGE_BANDIT_BATCH         = 2048
FORGE_BANDIT_WARMUP_EPOCHS = 20     # NEW — no early-stop until the actor has had steps to learn to trade
FORGE_BANDIT_EVAL_ROWS     = 4096   # NEW — rows sampled for the per-epoch proxy (keeps epochs fast)
FORGE_MIN_TRADE_RATE       = 0.001  # NEW — below this TRAIN trade-rate, a model is a non-trader → never promoted
FORGE_BANDIT_MAX_EPOCHS    = 150    # was 60 — give it room; early-stop ends it sooner when held-out plateaus
FORGE_BANDIT_PATIENCE      = 15     # was 6   # stop after this many epochs of no held-out gain
FORGE_BANDIT_REWARD_SCALE  = 100.0 # critic-target conditioning (matches the old /100)

# ----------------------------------------------------------------------------
# CREDENTIALS — environment only. Never a file named _env in the repo again.
# ----------------------------------------------------------------------------
KITE_API_KEY    = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")   # regenerated daily (SEBI daily logout)

# ----------------------------------------------------------------------------
# PATHS (cross-platform; runs from any working directory)
# ----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data";    DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR  = BASE_DIR / "models";  MODEL_DIR.mkdir(exist_ok=True)
STATE_DIR  = BASE_DIR / "state";   STATE_DIR.mkdir(exist_ok=True)
LOG_DIR    = BASE_DIR / "logs";    LOG_DIR.mkdir(exist_ok=True)

DB_PATH            = DATA_DIR / "arjun_tick_vault_v9.db"
RING_BUFFER_PATH   = STATE_DIR / "apex_ring_v9.mmap"
MACRO_STATE_TMPL   = str(STATE_DIR / "macro_state_{idx}.json")
LEDGER_PATH        = LOG_DIR / "execution_ledger_v9.csv"
CALIBRATION_TABLE  = STATE_DIR / "calibration_table.json"
MODEL_MANIFEST     = MODEL_DIR / "current_manifest.json"   # points at the promoted pair

# ----------------------------------------------------------------------------
# UNIVERSE (expiry style verified against the 2026 calendar: NIFTY weekly Tue,
# SENSEX weekly Thu; BANKNIFTY/FINNIFTY/MIDCPNIFTY monthly last-Tue; BANKEX
# monthly Thu. Lot sizes below are FALLBACKS ONLY — the mapper always trusts
# the live instrument dump, which is how v8 absorbed the Jan-2026 lot change.)
# ----------------------------------------------------------------------------
INDICES = {
    "NIFTY":      {"exchange": "NFO", "spot_symbol": "NSE:NIFTY 50",        "weekly": True,  "lot_fallback": 65,  "strike_step": 50},
    "BANKNIFTY":  {"exchange": "NFO", "spot_symbol": "NSE:NIFTY BANK",      "weekly": False, "lot_fallback": 30,  "strike_step": 100},
    "FINNIFTY":   {"exchange": "NFO", "spot_symbol": "NSE:NIFTY FIN SERVICE","weekly": False, "lot_fallback": 60,  "strike_step": 50},
    "MIDCPNIFTY": {"exchange": "NFO", "spot_symbol": "NSE:NIFTY MID SELECT", "weekly": False, "lot_fallback": 140, "strike_step": 25},
    "SENSEX":     {"exchange": "BFO", "spot_symbol": "BSE:SENSEX",          "weekly": True,  "lot_fallback": 20,  "strike_step": 100},
    "BANKEX":     {"exchange": "BFO", "spot_symbol": "BSE:BANKEX",          "weekly": False, "lot_fallback": 30,  "strike_step": 100},
}
INDEX_ORDER = list(INDICES.keys())
# ★ Indices the brain may actually TRADE (others remain context nodes only).
# With ₹5,000 only NIFTY/SENSEX cheap OTM is realistically affordable.
# ★ Indices the brain may actually TRADE (others remain context nodes only).
# NIFTY-only at ₹30k: SENSEX option lots run ~₹4,500+, which never fits the
# Kelly budget at this capital — including it just spams BLOCKED lines. Add
# "SENSEX" back once capital comfortably exceeds ~₹75k, or trade it standalone.
TRADABLE = ["NIFTY", "SENSEX"]

# ----------------------------------------------------------------------------
# MODEL GEOMETRY (v9: 17 → 19 features; adds dte_norm + is_weekly so the net
# can finally tell a 0-DTE weekly from a 25-DTE monthly — audit §5 leap)
# ----------------------------------------------------------------------------
NODES_PER_INDEX   = 5                 # spot, atm_ce, atm_pe, otm_ce, otm_pe
NUM_NODES         = NODES_PER_INDEX * len(INDEX_ORDER)      # 30
FEATURES_PER_NODE = 19
SEQ_LENGTH        = 10
OBS_DIM           = NUM_NODES * FEATURES_PER_NODE * SEQ_LENGTH   # 5700
ACTION_DIM        = 12                # (direction, size) × 6 indices
DEVICE            = "cuda"            # auto-falls back to cpu in code if absent
GCN_HIDDEN        = 128
PROJ_DIM          = 512               # 8 GB VRAM bottleneck — kept from v8

# ----------------------------------------------------------------------------
# RISK CONSTITUTION  (enforced by core/risk_manager.RiskGovernor — nowhere else)
# ----------------------------------------------------------------------------
MAX_DAILY_DRAWDOWN_PCT   = 0.10   # ★ realized, after costs. v8's -50% retired.
MAX_LOSS_PER_TRADE_PCT   = 0.30   # disaster-floor loss may never exceed this × capital
MAX_CONCURRENT_POSITIONS = 1
KELLY_FRACTION           = 0.5
MAX_KELLY_BUDGET_PCT     = 0.80
MIN_TICKS_BEFORE_TRADING = 120    # let physics warm up after open
COOLDOWN_S               = 180
DIRECTION_LOCKOUT_S      = 1800   # after a losing exit, no same-direction re-entry
MAX_ORDER_REJECTS        = 3      # then halt the day (reject-storm kill switch)
DATA_STALE_BLOCK_S       = 5.0    # no NEW entries if feed older than this
DATA_STALE_FLATTEN_S     = 60.0   # emergency-flatten open positions beyond this
MACRO_STALE_S            = 420.0  # GEX json older than this = advisory dead
NO_ENTRY_AFTER           = "14:45"
FORCE_FLATTEN_AT         = "15:15"   # safely before broker MIS auto square-off (~15:20)
SESSION_OPEN             = "09:15"
SESSION_CLOSE            = "15:30"

# Stops / targets (PREMIUM-based; spot context only shapes them)
BASE_SL_PCT          = 0.20   # initial stop: -20% of entry premium
BASE_TP_PCT          = 0.30   # initial target before GEX/DEM shaping
TRAIL_ARM_PCT        = 0.15   # arm trail after +15%
TRAIL_GIVEBACK_PCT   = 0.45   # surrender at most 45% of peak gain

# ---- PROFIT-LOCK FLOOR (constitution — fixed, never learned) ----
# Once the trail arms (position was up past TRAIL_ARM_PCT), a HARD floor turns on
# at breakeven + round-trip costs. It is checked BEFORE the trap shield and
# OVERRIDES it — so the shield may hold a winner through a stop-hunt flush, but
# can NEVER give back below this line. A winner cannot become a loser. This is
# the profit-side twin of the disaster floor; like it, it does not move and is
# not learned. Optionally lock in MORE than breakeven via PROFIT_LOCK_GIVEBACK.
PROFIT_LOCK_ENABLED   = True
PROFIT_LOCK_GIVEBACK  = 1.00   # 1.00 = floor at breakeven (ride the hunt to the
#                                jackpot, but never a loss). Lower it to bank a
#                                fraction of peak gain, e.g. 0.50 = keep ≥50% of
#                                the best unrealized gain even through a hunt-hold.

# ---- MODEL-DRIVEN TARGET EXTENSION (dynamic — dormant until meta trains) ----
# At the target, if a trained meta-model's LIVE P(win) for the held position is
# still above META_HOLD_PAST_TARGET_P, the edge says there's more in the move:
# EXTEND the target by another expected-move increment and keep riding, protected
# by the (now armed) trail and the profit-lock floor. If P(win) has faded, bank
# the target as today. No model → fixed target, unchanged. Re-evaluated each time
# the (extended) target is tagged, up to a cap so it can't run unbounded.
META_HOLD_PAST_TARGET_P = 0.58   # P(win) bar to ride past the target (real edge)
TARGET_EXTEND_MAX       = 4      # max times the target may be extended per trade
DISASTER_FLOOR_MULT  = 1.6    # floor = 1.6 × current stop distance …
ABS_DISASTER_PCT     = 0.45   # … but never worse than -45% of premium. ALWAYS fires.
MAX_HOLD_MINUTES     = 45     # theta guillotine for 0-2 DTE longs
MAX_ENTRY_SPREAD_PCT = 0.03   # refuse entries when (ask-bid)/mid above this

# Conviction → probability. The calibration table (built nightly by the
# analyzer from real/paper outcomes) is the truth; this floor is the fallback.
ENTRY_CONVICTION     = 0.70
PAPER_ENTRY_CONVICTION = 0.55  # used ONLY if PAPER_EXPLORE=True (see below)
MIN_CAL_WINPROB      = 0.55
UNCALIBRATED_WINPROB = 0.52   # what an unproven |conviction|≥0.70 is worth: barely a coin

# ----------------------------------------------------------------------------
# TRAP SHIELD  (★ the anti-"institutional flush" layer — core/trap_shield.py)
# ----------------------------------------------------------------------------
TRAP_SCORE_THRESHOLD = 0.60   # ≥ this → hold through the stop breach (FALLBACK
# when no learned trap model exists). The forge refits this + TRAP_WEIGHTS from
# real stop-outs; the learned value is clamped to [TRAP_THRESHOLD_MIN, MAX].
TRAP_MODEL_PATH      = "state/trap_model.json"
TRAP_THRESHOLD_MIN   = 0.45   # learned threshold can't drop below this (noise
#                               floor) or rise above MAX (shield-disabling) —
#                               these clamps are FIXED, never learned.
TRAP_THRESHOLD_MAX   = 0.80
TRAP_MIN_SAMPLES     = 40     # real stop-breach events (hold+honored) before the
#                              forge trusts a learned trap model. Below this it
#                              writes nothing and the shield uses the fixed guess.
TRAP_MAX_HOLD_S      = 150    # grace window; then the stop is honored
TRAP_VELOCITY_Z      = 3.0    # how abnormal the down-spike must be
TRAP_SPREAD_BLOWOUT  = 2.5    # spread / rolling-avg-spread
TRAP_WALL_PROX_PCT   = 0.0020 # within 0.20% of the GEX put-wall = hunt zone
TRAP_RECLAIM_PCT     = 0.40   # price reclaiming 40% of the spike = trap confirmed
TRAP_MAX_USES_PER_TRADE = 2   # shield is not an excuse machine

# ----------------------------------------------------------------------------
# FEATURE / ESTIMATOR PHYSICS  (every tunable lives HERE; only pure algorithm
# internals — Newton iteration counts, SVI seeds — remain in code)
# ----------------------------------------------------------------------------
EWMA_VOL_HALFLIFE_S   = 20.0
VPIN_BASE_BUCKET      = 5000
VPIN_N_BUCKETS        = 20
HAWKES_DECAY_PER_S    = 2.0
OI_DELTA_WINDOW_S     = 900
OFI_WINDOW_TICKS      = 120
DEALER_INV_DECAY      = 0.995
DEALER_INV_SCALE      = 50.0
DTE_PART_DAY          = 0.3     # intraday remainder added to whole-day DTE

# Strike ladder + entry tempo — SHARED by live brain and simulator (sim==live)
HIERARCHY_DEPTH          = 8
ENTRY_ATTEMPT_THROTTLE_S = 5.0
# ---- SIGNAL PERSISTENCE (a confident trade needs a SUSTAINED read) ----------
# Conviction is read each tick from live OI/flow/momentum. In a choppy tape it
# can flip sign tick-to-tick (the whipsaw the logs showed) — high conviction in
# the moment, but unstable. These require the directional read to have HELD
# before entering: the recent window must agree in sign for ≥ SIGNAL_PERSIST_FRAC
# of samples AND average above SIGNAL_PERSIST_AVG_MULT× the entry bar. This makes
# the system wait for confident, SUSTAINED signals instead of entering on a
# one-tick spike it exits minutes later. Same live Kite data — it just demands
# the signal prove itself over time before committing capital.
SIGNAL_PERSIST_ENABLED  = True
SIGNAL_PERSIST_N        = 4      # ticks in the persistence window; read must hold
SIGNAL_PERSIST_FRAC     = 0.75   # ≥ this fraction of the window must agree in sign
SIGNAL_PERSIST_AVG_MULT = 0.95   # window-average |conv| must exceed this × the bar
# Entry order pricing. A passive maker buy (posted at the bid-side micro-price)
# cannot fill on an option that is RISING — which is exactly when a bullish
# momentum signal fires — so a pure-maker entry starves on trending tape.
# Microstructure research: momentum (continuation) entries should CROSS the
# spread — the half-spread is the toll for not missing the move; only
# mean-reversion entries rest passively. Apex's heuristic is momentum (OFI +
# velocity + dealer-inventory), so it crosses, with three guards so crossing
# never becomes blind chasing:
#   1. ENTRY_CROSS_CONVICTION — only cross when conviction clears this (weaker
#      signals stay passive). Set >1.0 to disable crossing entirely.
#   2. ENTRY_SLIP_CAP_PCT — take the ask ONLY if it sits within this fraction
#      of one strike-step's worth of premium above the decision-time
#      micro-price. If the option has already run past that, the signal is
#      stale and we WALK AWAY rather than buy exhaustion (critical at 0-2 DTE
#      where chasing buys rich premium into accelerating decay).
#   3. ENTRY_CROSS_SPREAD_PCT — crossing is only worth it when the spread is
#      tight; if (ask-bid)/mid exceeds this (a band TIGHTER than the hard
#      MAX_ENTRY_SPREAD_PCT reject), immediacy is too expensive — stay passive.
# Paper and live use the identical logic; crossing is what live would do.
ENTRY_CROSS_CONVICTION = 0.70   # = ENTRY_CONVICTION: momentum entries
# cross (the passive maker path below the entry bar only suits mean-reversion;
# this is a momentum system). The slippage cap still walks away from runners
# that ran past their chase cap, so crossing is filling — not blind chasing.
ENTRY_SLIP_CAP_PCT     = 0.60    # fraction of one strike-step premium move
SLIPCAP_BORDERLINE_FRAC = 0.25   # diagnostics only (changes NO behavior): a
# chase-cap walk-away within this fraction past the cap is "borderline" (the cap
# may be slightly tight on a genuine fill); beyond it is a "runaway" the cap
# correctly refused. Classifies walk-aways for the heartbeat tally / evidence.
ENTRY_CROSS_SPREAD_PCT = 0.015   # DEPRECATED/unused: the separate cross-spread
# band was removed — it starved fills by routing real (1.5–3% spread) NIFTY
# option signals to the passive path. Liquidity is gated by MAX_ENTRY_SPREAD_PCT
# (3%), chasing by ENTRY_SLIP_CAP_PCT. Kept only so external refs don't break.
ENTRY_CROSS_CAP_TICKS  = 2       # hard ceiling: never pay >this many ticks past ask

# Brain: advisory fusion, calibration, cadence
ADVISORY_VPIN_THRESHOLD = 0.6
ADVISORY_SHOCK          = 0.15
FUSION_QUANT_WEIGHT     = 1.0
PAPER_EXPLORE_WINPROB   = 0.60  # what unproven conviction is worth in PAPER
HEARTBEAT_S             = 60.0
TRADE_TRACK_S           = 5.0    # while a position is OPEN, stream a live read
# (PnL, distance to stop/target, OI, trap score, P(win)) every this-many seconds
# so the trade's evolution is visible — independent of the 60s heartbeat.
CAL_RELOAD_S            = 600.0
CAL_BUCKET_WIDTH        = 0.05
CAL_MIN_SAMPLES         = 20
SPREAD_EW_ALPHA         = 0.02
QUOTE_CACHE_FRESH_S     = 1.5
HEURISTIC_W             = (0.45, 0.50, 0.35, 0.40)  # ofi, dealer, vel, momentum

# Execution micro-knobs
PAPER_SLIPPAGE_TICKS  = 1
URGENT_CHASE_TICKS    = 2
LIVE_POLL_INTERVAL_S  = 0.4

# Harvester
PRUNE_STEPS       = 3
DB_BATCH_ROWS     = 1000
RING_WRITE_S      = 1.0
TELEMETRY_S       = 10.0
QUEUE_WARN_DEPTH  = 500
SNAPSHOT_PM_AT    = "15:35"
ICEBERG_VOL_MULT  = 3.0
ICEBERG_QTY_RATIO = 0.8

# Macro GEX
MACRO_LOOP_S      = 180
MACRO_QUOTE_CHUNK = 500
MACRO_STRIKE_BAND = 0.10

# Scanner
SCANNER_ALERT   = 0.85
SCANNER_OFFSETS = (-2, -1, 1, 2)

# Trap-shield internals (weights MUST sum to 1.0)
TRAP_WEIGHTS = {"velocity": 0.22, "absorption": 0.22, "oi": 0.14,
                "dislocation": 0.16, "spread": 0.14, "wall": 0.12}
TRAP_OI_CONFIRM_SCALE = 10.0
TRAP_DISLOCATION_FULL = 0.10
TRAP_VEL_WINDOW_S     = 600

# Forge training knobs
SAC_BUFFER          = 50_000
SAC_BATCH           = 256
SAC_TRAIN_FREQ      = 64
SAC_GRAD_STEPS      = 64
SAC_TIMESTEPS_CAP   = 150_000
FORGE_EVAL_STEP_S   = 60
FORGE_ACT_GATE_TRAIN = 0.3
FORGE_ACT_GATE_EVAL  = 0.5

# ----------------------------------------------------------------------------
# RESEARCH LAYER (each knob traces to published evidence — see README)
# ----------------------------------------------------------------------------
# Edge Certificate — the third lock. SEBI: ~93% of individual F&O traders
# lose; profits accrue to algorithmic entities. This system therefore CANNOT
# arm live until its own paper ledger clears statistical proof of edge.
EDGE_MIN_TRADES      = 100
EDGE_MIN_DAYS        = 20
EDGE_BOOTSTRAP_N     = 10_000
EDGE_CI              = 0.95     # bootstrap CI lower bound of mean PnL must be > 0
EDGE_CERT_PATH       = STATE_DIR / "edge_certificate.json"
EDGE_CERT_VALID_DAYS = 7

# Meta-labeling (López de Prado 2017/2018): primary model picks the SIDE,
# a secondary model learns the SIZE — P(win | features) from triple-barrier
# outcomes on REAL recorded prices, after real costs. Feeds Kelly directly.
META_MODEL_PATH = STATE_DIR / "meta_model.json"
META_MIN_TRAIN  = 300          # labeled signals before the meta model is trusted
META_LR         = 0.05
META_EPOCHS     = 300
META_L2         = 1e-3
META_P_FLOOR    = 0.50
META_P_CAP      = 0.85

# ---- DYNAMIC DECISION (model-driven entry/exit, fixed-threshold fallback) ----
# These activate ONLY when a trained meta-model exists (the forge writes it after
# META_MIN_TRAIN real labeled trades). Until then the brain uses the fixed
# conviction bar + fixed target/stop below — the bootstrap path. In NEITHER mode
# do these touch the risk constitution: the disaster floor, drawdown halt,
# position cap, EOD flatten and the hard stop are unchanged and always win. The
# model may only make decisions TIGHTER inside that envelope, never looser than
# the floors.
META_DECISION_ENABLED = True   # master switch for model-driven entry/exit gating
META_ENTRY_P_BAR      = 0.55   # trained-mode entry: take the trade when the
#                                calibration-blended P(win) clears this (a real
#                                after-cost edge; model floor is 0.50 = coin-flip)
META_ENTRY_CONV_FLOOR = 0.40   # minimal directional sanity floor in trained mode
#                                so the model never acts on near-zero (noise)
#                                conviction. Well below the 0.70 bootstrap bar —
#                                the model, not a hand-set bar, decides above it.
META_EXIT_P_FLOOR     = 0.45   # trained-mode exit: if the model's LIVE P(win)
#                                for the held position decays below this, the edge
#                                is gone — exit early. Only ever cuts SOONER; the
#                                fixed target/stop/floor remain the outer bounds.
META_EXIT_MIN_HOLD_S  = 60     # don't act on the model's exit read in the first
#                                minute (entry-noise guard). The disaster floor,
#                                EOD and hard stop still apply from t=0.

# Volatility-targeted sizing (volatility-managed portfolios literature):
# scale the Kelly budget down when implied vol runs hot. ≤1.0 always.
VOL_TARGET_ANN = 0.14
VOL_SCALE_MIN  = 0.40

# Market intraday momentum (Gao–Han–Li–Zhou, JFE 2018): first half-hour
# return predicts the last half-hour. Late-day advisory only.
ADVISORY_SHOCK_IMOM = 0.08
IMOM_AFTER          = "14:00"

# 0-DTE regime (expiry-day gamma/theta): tighter theta guillotine.
EXPIRY_DTE_LT          = 1.0
MAX_HOLD_MINUTES_0DTE  = 25

# Provenance: every run logs the exact config it traded on.
import hashlib as _hl
# CONFIG_HASH is computed at the END of this module — it must see every constant
# defined below it (e.g. DRIFT_KEY_FEATURES, COSTS). See _compute_config_hash().
# It is a FEATURE/MODEL fingerprint, not a whole-file hash: operational knobs
# (capital, sizing, polling, paths, …) are excluded so editing them no longer
# invalidates a trained reference. live_fire_armed() above reads it at runtime,
# by which point the end-of-file assignment has run.

# ----------------------------------------------------------------------------
# FEATURE-DRIFT MONITOR (the live regime-shift guard — core/drift_monitor.py)
# ----------------------------------------------------------------------------
# A model's win-probabilities are only valid on tape that resembles what it
# trained on. The forge writes a per-feature REFERENCE PROFILE (quantile bin
# edges + histogram) at training time; the live brain accumulates the SAME
# features through the SAME StateBuilder and, each heartbeat, measures
# divergence per feature with two complementary distribution-shift tests:
#   • PSI (Population Stability Index) — industry standard. <0.10 stable,
#     0.10–0.25 moderate, >0.25 significant.
#   • KS (two-sample Kolmogorov–Smirnov D) — max CDF gap, distribution-free.
# Graded GREEN / WATCH / DRIFTED from the share of signal-carrying features
# that breach threshold. DRIFTED de-certifies live automatically: a regime
# the model never trained on is not one it may bet real money in.
DRIFT_PROFILE_PATH     = STATE_DIR / "feature_reference.json"
DRIFT_STATE_PATH       = STATE_DIR / "drift_state.json"
DRIFT_BINS             = 10        # quantile bins per feature for PSI
DRIFT_PSI_MODERATE     = 0.10
DRIFT_PSI_SIGNIFICANT  = 0.25
DRIFT_KS_SIGNIFICANT   = 0.20
DRIFT_MIN_LIVE_SAMPLES = 600       # ~10 min of 1-Hz frames before judging
DRIFT_REF_MAX_SAMPLES  = 50_000    # cap reference rows kept per feature
DRIFT_WATCH_FRAC       = 0.25      # ≥25% of key features moderate → WATCH
DRIFT_DEARM_FRAC       = 0.40      # ≥40% of key features significant → de-arm
DRIFT_KEY_FEATURES     = ["log_ret", "oi_delta_norm", "depth_grad", "vpin",
                          "velocity", "spread_pct", "iv", "skew", "regime_vol",
                          "hawkes", "ofi_z", "delta", "gamma_x1e4",
                          "theta_norm", "dealer_inv"]

# ----------------------------------------------------------------------------
# WEAPONS RACK (all real Kite data; every knob here, defaults conservative)
# ----------------------------------------------------------------------------
VIX_SYMBOL          = "NSE:INDIA VIX"
VIX_SPIKE_5M_PCT    = 0.04    # +4% VIX in 5 min = regime shock →
VIX_BAR_BUMP        = 0.10    #   entry bar temporarily this much higher
IVRANK_HIGH         = 0.80    # ATM IV in top 20% of trailing history →
IVRANK_BAR_BUMP     = 0.05    #   long premium is expensive; demand more edge
IVRANK_MIN_DAYS     = 10      # need this much IV history before ranking

# ---- VOL-SURFACE FORECASTER (predicts near-term ATM-IV change) ----------------
# Reads the REAL persisted IV series (daily history + intraday samples the macro
# loop writes) plus the live surface, and forecasts the ATM-IV move over the next
# VOL_FCAST_HORIZON_MIN minutes. No synthetic series — it predicts from recorded
# state via IV mean-reversion to its own recent level, term-structure slope, and
# the empirical intraday vol-of-vol. Falls back to "no forecast" until enough
# samples exist (VOL_FCAST_MIN_SAMPLES), exactly like every other learned layer.
IV_INTRADAY_PATH      = str(STATE_DIR / "iv_intraday_{idx}.json")
VOL_FCAST_HORIZON_MIN = 20    # forecast ATM-IV this far ahead
VOL_FCAST_MIN_SAMPLES = 120   # intraday IV samples before a forecast is trusted
VOL_FCAST_REVERT_K    = 0.15  # per-step mean-reversion strength toward recent IV
VOL_FCAST_CRUSH_Z     = 1.5   # |z| of IV vs its band beyond which crush/expansion
#                               is flagged (drives the regime layer + exit shaping)
VOL_FCAST_SD_FLOOR_FRAC = 0.05  # floor the IV-band std at this fraction of the IV
#                               level so a dead-calm day can't saturate the z-score
#                               on noise and jam the regime (raise → less sensitive)
VOL_FCAST_MODEL_PATH  = str(STATE_DIR / "vol_forecast_model.json")  # learned
#                               mean-reversion + vol-of-vol params, refit nightly

# ---- REGIME CLASSIFIER (labels market state, scales conviction) ---------------
# Names the regime from state the system already computes and returns a
# conviction MULTIPLIER (never a hard veto, never moves a risk floor). Cut points
# start fixed and are refit nightly to the empirical percentiles of THIS market.
REGIME_TE_TREND     = 0.55    # |trend efficiency| ≥ this → trending (fallback)
REGIME_TE_CHOP      = 0.30    # |trend efficiency| ≤ this → chop (fallback)
REGIME_GEX_SQUEEZE  = -2.0e10 # net GEX ≤ this (deep negative) → squeeze-prone.
#                               Net GEX in THIS market lives at the e10 scale (the
#                               nightly refit's 20th pct logged ≈ -1.6e10); the old
#                               -2e12 was ~100× too extreme and never fired. This
#                               fixed fallback sits just below the typical 20th pct
#                               so it means "deep". SANITY-CHECK against your own
#                               net_gex distribution and adjust if your scale differs.
REGIME_RV_HIGH      = 0.22    # realized vol (annualized) ≥ this → high-vol
REGIME_WALL_PROX_PCT = 0.004  # within this % of a GEX wall counts as "at a wall"
REGIME_MIN_SAMPLES  = 300     # feature rows before percentile refit is trusted
REGIME_MODEL_PATH   = str(STATE_DIR / "regime_model.json")
REGIME_FEATURE_LOG  = str(STATE_DIR / "regime_features.jsonl")
REGIME_LOG_EVERY_S  = 30      # write at most one feature row per index per N s
#                               (keeps the JSONL append off the live hot path;
#                               30 s snapshots are plenty for percentile fitting)
REGIME_FEATURE_LOG_MAX = 60000  # cap retained feature rows (read + on-disk trim);
#                               ~tens of sessions of history, refit needs only 300
REGIME_HYSTERESIS_N = 1       # consecutive ticks on a NEW regime before switching.
#                               1 = OFF (stateless, current behaviour). Raise to
#                               2–3 to kill single-tick label/multiplier flicker
#                               near a cut boundary.
# conviction multipliers per regime (scale the brain's conviction; 1.0 = neutral)
REGIME_MULT_TREND   = 1.15    # clean trend → momentum favored
REGIME_MULT_CHOP    = 0.70    # chop → momentum bleeds, dampen
REGIME_MULT_SQUEEZE = 1.20    # short-gamma at a wall → breakouts run, boost
REGIME_MULT_CRUSH   = 0.65    # IV crush → premium bleed, dampen long premium
REGIME_MULT_HIGHVOL = 0.80    # vol extreme → size down via lower effective conv
PCR_HIGH            = 1.40    # OI put/call ratio extremes (band-wide):
PCR_LOW             = 0.60    #   contrarian advisory nudges
ADVISORY_SHOCK_PCR  = 0.10
ADVISORY_SHOCK_MAXPAIN = 0.08 # expiry-day gravity toward max-pain strike
ADVISORY_SHOCK_LEVELS  = 0.12 # prev-day high/low breakout confirmation
ORDER_UPDATES_PATH  = STATE_DIR / "order_updates.json"   # WS push → engine
LIVE_GTT_FLOOR      = False   # server-side disaster floor (NRML only; GTTs
                              # are GTC and 250/account — read README caveat)
BACKFILL_DAYS       = 120     # tools/backfill_history.py default span

# ----------------------------------------------------------------------------
# COSTS (Zerodha + statutory, NSE index options, 2026) — used by the reward,
# the paper engine and the analyzer so all three agree on the toll booth.
# ----------------------------------------------------------------------------
COSTS = {
    "brokerage_per_order": 20.0,      # flat per executed order
    "stt_sell_pct":        0.001,     # 0.1% on sell premium
    "exch_txn_pct":        0.0003503, # NSE options on premium (both legs)
    "sebi_pct":            0.000001,
    "gst_pct":             0.18,      # on brokerage + txn charges
    "stamp_buy_pct":       0.00003,
}

# ----------------------------------------------------------------------------
# KITE API BUDGETS (verified: orders 10/s & 400/min & 5000/day; quote 1/s;
# historical 3/s; other endpoints 10/s — we run well inside every line)
# ----------------------------------------------------------------------------
RATE = {"order_per_s": 5, "order_per_min": 60, "quote_per_s": 1,
        "hist_per_s": 2, "misc_per_s": 6}
ORDER_POLL_BUDGET_S   = 3.0    # how long buy-side waits for a maker fill
ORDER_REPOST_TICKS    = 1      # one cancel/re-post a tick worse, then walk away
SELL_MARKET_PROTECTION = 5     # % protection if a protected-market exit is ever used

# ----------------------------------------------------------------------------
# FORGE / TRAINING
# ----------------------------------------------------------------------------
FORGE_LOOKBACK_DAYS   = 10     # train on last K days + reservoir of older days
FORGE_RESERVOIR_DAYS  = 5
FORGE_VAL_DAYS        = 1      # walk-forward gate: newest day held out
FORGE_PROMOTE_MARGIN  = 0.95   # promote only if val score ≥ incumbent × this
REWARD_HORIZON_S      = 60
RISK_FREE_RATE        = 0.07

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ============================================================================
# CONFIG_HASH — model/feature fingerprint (computed last, sees all constants)
# ============================================================================
# Stamped into every trained artifact (drift reference, meta-labeler) and the
# Edge Certificate, then checked before any of them is trusted. It fingerprints
# ONLY the constants that change those artifacts: how the 19 features are
# computed, the feature-tensor shape, the surface/greeks math, the triple-
# barrier labels (BASE_TP_PCT / BASE_SL_PCT / MAX_HOLD_MINUTES), broker COSTS
# (labels are net-of-cost), and the regime / trap / vol-forecaster thresholds.
#
# Pure OPERATIONAL knobs are excluded so editing them does NOT invalidate a
# reference the forge already trained: capital & sizing, risk budgets, order
# routing / fills / polling, cooldowns & watchdogs, telemetry & logging cadence,
# filesystem paths, credentials, drift ASSESSMENT thresholds (retune the de-arm
# without a re-forge), edge-audit knobs, and forge training-infra hyper-params.
#
# FAIL-CLOSED: anything not explicitly excluded is fingerprinted, so a newly
# added feature/label constant still invalidates correctly even if nobody
# updates this list. Operational additions you don't want to force a re-forge
# must be named with a path suffix or added to _HASH_EXCLUDE.
#
# NOTE: this changes the hash value itself. After deploying, the existing
# reference reads NO_REF until the nightly forge runs once and restamps it. It
# also means the Edge-Certificate config check no longer trips on these
# operational knobs (including TRADING_CAPITAL) — acceptable while LIVE_FIRE is
# permanently False and the certificate is re-audited from the ledger nightly.
_HASH_EXCLUDE = frozenset({
    # the hash must never fingerprint itself (else recompute is unstable)
    "CONFIG_HASH",
    # identity / run-mode (no effect on features or the trained model)
    "VERSION", "LIVE_FIRE", "LIVE_CONFIRM_ENV", "LIVE_CONFIRM_PHRASE",
    "PAPER_EXPLORE", "PAPER_FILL_REALISM", "PAPER_ENTRY_CONVICTION",
    "PAPER_EXPLORE_WINPROB", "DEVICE",
    # capital / sizing / risk budget (execution economics only)
    "TRADING_CAPITAL", "KELLY_FRACTION", "MAX_KELLY_BUDGET_PCT",
    "MAX_DAILY_DRAWDOWN_PCT", "MAX_LOSS_PER_TRADE_PCT",
    "MAX_CONCURRENT_POSITIONS", "VOL_TARGET_ANN", "VOL_SCALE_MIN",
    # credentials (env-derived, rotate daily)
    "KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN",
    # cooldowns / lockouts / throttles / halts / data-watchdog (gate orders;
    # identical paper↔live; do not change per-second feature/label generation)
    "COOLDOWN_S", "DIRECTION_LOCKOUT_S", "ENTRY_ATTEMPT_THROTTLE_S",
    "MAX_ORDER_REJECTS", "DATA_STALE_BLOCK_S", "DATA_STALE_FLATTEN_S",
    "MACRO_STALE_S",
    # order routing / fills / polling / entry-cross execution
    "PAPER_SLIPPAGE_TICKS", "URGENT_CHASE_TICKS", "LIVE_POLL_INTERVAL_S",
    "ORDER_POLL_BUDGET_S", "ORDER_REPOST_TICKS", "SELL_MARKET_PROTECTION",
    "ENTRY_CROSS_CONVICTION", "ENTRY_SLIP_CAP_PCT", "SLIPCAP_BORDERLINE_FRAC",
    "ENTRY_CROSS_SPREAD_PCT", "ENTRY_CROSS_CAP_TICKS", "LIVE_GTT_FLOOR",
    "MAX_ENTRY_SPREAD_PCT",
    # macro / scanner cadence & coverage  (MACRO_STRIKE_BAND stays IN — it
    # shapes the GEX surface that iv/skew features and the shaped target read)
    "MACRO_LOOP_S", "MACRO_QUOTE_CHUNK", "SCANNER_ALERT", "SCANNER_OFFSETS",
    # telemetry / logging / IO cadence / inference-time calibration build
    "HEARTBEAT_S", "TRADE_TRACK_S", "TELEMETRY_S", "QUEUE_WARN_DEPTH",
    "RING_WRITE_S", "DB_BATCH_ROWS", "PRUNE_STEPS", "SNAPSHOT_PM_AT",
    "LOG_FORMAT", "CAL_RELOAD_S", "QUOTE_CACHE_FRESH_S", "REGIME_LOG_EVERY_S",
    "REGIME_FEATURE_LOG_MAX", "CAL_BUCKET_WIDTH", "CAL_MIN_SAMPLES",
    # edge-certificate audit knobs (the cert is re-audited nightly from ledger)
    "EDGE_MIN_TRADES", "EDGE_MIN_DAYS", "EDGE_BOOTSTRAP_N", "EDGE_CI",
    "EDGE_CERT_VALID_DAYS",
    # forge training-infra hyper-params (change HOW models train, not the
    # data / labels / features / reference distribution)
    "FORGE_BANDIT_BATCH", "FORGE_BANDIT_WARMUP_EPOCHS", "FORGE_BANDIT_EVAL_ROWS",
    "FORGE_MIN_TRADE_RATE", "FORGE_BANDIT_MAX_EPOCHS", "FORGE_BANDIT_PATIENCE",
    "FORGE_BANDIT_REWARD_SCALE", "SAC_BUFFER", "SAC_BATCH", "SAC_TRAIN_FREQ",
    "SAC_GRAD_STEPS", "SAC_TIMESTEPS_CAP", "FORGE_EVAL_STEP_S",
    "FORGE_ACT_GATE_TRAIN", "FORGE_ACT_GATE_EVAL", "FORGE_PROMOTE_MARGIN",
    "FORGE_LOOKBACK_DAYS", "FORGE_RESERVOIR_DAYS", "FORGE_VAL_DAYS",
    # drift ASSESSMENT thresholds (retune de-arm sensitivity with NO re-forge;
    # the reference-CONSTRUCTION knobs DRIFT_BINS / DRIFT_REF_MAX_SAMPLES /
    # DRIFT_KEY_FEATURES are deliberately NOT here — they stay in the hash)
    "DRIFT_PSI_MODERATE", "DRIFT_PSI_SIGNIFICANT", "DRIFT_KS_SIGNIFICANT",
    "DRIFT_WATCH_FRAC", "DRIFT_DEARM_FRAC", "DRIFT_MIN_LIVE_SAMPLES",
    # misc data-source / backfill
    "BACKFILL_DAYS", "VIX_SYMBOL",
})

# names ending in any of these are filesystem locations / log toggles → excluded
_HASH_PATH_SUFFIXES = ("_PATH", "_DIR", "_TMPL", "_TABLE", "_MANIFEST", "_LOG")


def _hash_canon(v):
    """Deterministic, order-stable canonical form for the fingerprint payload."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_hash_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _hash_canon(v[k]) for k in sorted(v, key=str)}
    return repr(v)                       # time/enum/etc. — stable repr


def _compute_config_hash() -> str:
    g = list(globals().items())          # snapshot: don't mutate during iterate
    items = []
    for k, v in g:
        if not k.isupper() or k.startswith("_"):
            continue                     # only public UPPERCASE constants
        if k in _HASH_EXCLUDE or k.endswith(_HASH_PATH_SUFFIXES):
            continue                     # operational knob — excluded
        if isinstance(v, Path):
            continue                     # any stray path constant
        items.append((k, _hash_canon(v)))
    payload = repr(sorted(items)).encode("utf-8")
    return _hl.sha1(payload).hexdigest()[:10]


CONFIG_HASH = _compute_config_hash()