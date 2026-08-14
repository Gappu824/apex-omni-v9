"""
APEX OMNI v9.1 — CONFIG / RISK CONSTITUTION
===========================================
Every constant the system reads lives HERE and only here. No getattr()
fallbacks anywhere in v9 for constitution values: if it isn't defined in this
file, the code refuses to start rather than inventing a default (audit
finding: phantom ₹40,000 capital fallback in v8).

v9.1 (post-audit): dead constants purged so the file describes the running
system again — MIN_CAL_WINPROB (defined, never enforced), the old
multiplicative FORGE_PROMOTE_MARGIN (renamed in code, silently defaulting to
₹0), SIGNAL_PERSIST_N/FRAC/AVG_MULT (replaced by a wall-clock window),
REWARD_HORIZON_S and the four SAC rollout knobs the bandit trainer bypassed,
FORGE_LOOKBACK/RESERVOIR/VAL_DAYS (superseded by FORGE_MAX_TRAIN_DAYS and the
walk-forward harness). Anything deleted here had ZERO live references — the
scan is in the audit report.

★ = the knobs you (the human) are expected to touch.
"""

import logging
import unicodedata as _ud
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


# v9.7.1 ENCODING: the log FILES are valid UTF-8 — verified at byte level
# (`·` = c2 b7, `✓` = e2 9c 93, zero double-encoding). Nothing corrupts them on
# write. But every reader downstream guesses its own codepage, and a Windows
# console, a tail viewer or a clipboard hop that guesses cp1252 renders ₹ as
# "â‚¹" and ✓ as "âœ“". We cannot fix a reader we do not control; the only wire
# format none of them can mangle is ASCII. Source keeps its Unicode (comments
# and docstrings are never emitted) — EMITTED log lines are transliterated.
LOG_ASCII = True          # False keeps the glyphs if your console is UTF-8 clean

_ASCII_MAP = {
    "₹": "Rs ", "✓": "[y]", "✗": "[x]", "⚠": "!", "·": ".", "—": "-",
    "–": "-", "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~",
    "×": "x", "±": "+/-", "σ": "sigma", "Δ": "delta", "μ": "mu", "★": "*",
    # 2026-07-23: these three fell through to "?" in the live logs — the Brier
    # decomposition printed "unc 0.2500 ? res" (U+2212 MINUS, not a hyphen) and
    # the skill line printed "b=1.50 ? break-even" (U+21D2). Mapped explicitly.
    "−": "-", "⇒": "=>", "⇐": "<=", "≠": "!=", "∞": "inf", "√": "sqrt",
    "≡": "==", "≪": "<<", "≫": ">>", "∑": "sum", "∆": "delta",
    # 2026-07-23: the live ENTER line printed "vol-scaled (ATR 5.67, 15m, ?0.50)"
    # — lowercase delta (U+03B4) was unmapped; only capital Delta was.
    "δ": "d", "α": "alpha", "β": "beta", "γ": "gamma", "θ": "theta",
    "λ": "lambda", "ρ": "rho", "τ": "tau", "φ": "phi", "ω": "omega",
    "Σ": "Sum", "Ω": "Omega", "π": "pi",
    "♥": "hb", "…": "...", "─": "-", "━": "-", "│": "|", "•": "*",
    "“": '"', "”": '"', "‘": "'", "’": "'", "📊": "", "°": "deg",
}


class _AsciiSafeFormatter(logging.Formatter):
    """Transliterate emitted log records to ASCII (see LOG_ASCII above)."""

    def format(self, record):
        s = super().format(record)
        if s.isascii():
            return s
        for k, v in _ASCII_MAP.items():
            s = s.replace(k, v)
        if not s.isascii():          # anything unmapped: strip accents, then ?
            s = (_ud.normalize("NFKD", s)
                 .encode("ascii", "replace").decode("ascii"))
        return s


def setup_logging(component: str, level=logging.INFO):
    """Console + daily file (logs/<component>_<date>.log) for every process.
    Call once at the top of each entrypoint."""
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f"{component}_{_dt.date.today()}.log",
                             encoding="utf-8")
    sh = logging.StreamHandler()
    _fmt = (_AsciiSafeFormatter(LOG_FORMAT) if LOG_ASCII
            else logging.Formatter(LOG_FORMAT))
    fh.setFormatter(_fmt)
    sh.setFormatter(_fmt)
    logging.basicConfig(level=level, handlers=[sh, fh], force=True)
    logging.getLogger(component).info(
        "=== %s start | %s | capital ₹%.0f | LIVE_FIRE=%s (armed=%s) ===",
        component, VERSION, TRADING_CAPITAL, LIVE_FIRE, live_fire_armed())

# ----------------------------------------------------------------------------
# IDENTITY / SAFETY
# ----------------------------------------------------------------------------
VERSION = "v9.1-audited"

# ★ PAPER MODE. Stays False. To ever go live you must BOTH set this True AND
#   export APEX_CONFIRM_LIVE="I-UNDERSTAND-REAL-MONEY". One switch is an
#   accident; two is a decision.
LIVE_FIRE = False
LIVE_CONFIRM_ENV = "APEX_CONFIRM_LIVE"
LIVE_CONFIRM_PHRASE = "I-UNDERSTAND-REAL-MONEY"

def live_fire_armed() -> bool:
    """FOUR locks, all required: the LIVE_FIRE flag, the confirmation phrase,
    a FRESH Edge Certificate (statistical proof from the paper ledger that
    this account clears the bar SEBI shows ~91% of individuals never clear),
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
# PAPER_EXPLORE breaks parity ON PURPOSE: paper trades on a lower bar with an
# exploratory win-prob to build the calibration table / Edge Certificate
# ledger. ★ v9.1 DEFAULT: True — the audit showed the mirror-live bootstrap
# (0.70 bar, 0.52 Kelly prior) starves the ledger the certificate needs. The
# EXPECTED scientific outcome of explore mode, given the closed directional
# thesis, is a ledger that statistically CONFIRMS no edge and a certificate
# that correctly refuses to arm — that is the machinery working, not failing.
# Flip back to False the day you want paper to mirror live exactly.
PAPER_EXPLORE = True

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
TRADING_CAPITAL = 60000.0           # ₹ — LIVE knob only (v9.1.1): the forge,
#                                    meta, drift and every cache are sized off
#                                    FORGE_EVAL_CAPITAL below; change this any
#                                    time, nothing re-forges or invalidates.

# Forge bandit-trainer knobs (training INFRA — hash-excluded below)
FORGE_BANDIT_BATCH         = 2048
FORGE_BANDIT_WARMUP_EPOCHS = 20    # no early-stop until the actor has had steps to move off zero
FORGE_BANDIT_EVAL_ROWS     = 4096  # rows sampled for the per-epoch proxy (keeps epochs fast)
FORGE_MIN_TRADE_RATE       = 0.001 # below this TRAIN trade-rate a candidate is recorded as an
#                                    ABSTAINER (a legitimate finding) and not deployed — deploying
#                                    it would freeze the paper ledger the certificate audits.
FORGE_BANDIT_MAX_EPOCHS    = 150
FORGE_BANDIT_PATIENCE      = 15    # epochs of no INNER-day gain before early stop
FORGE_BANDIT_REWARD_SCALE  = 100.0 # critic-target conditioning

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
# APEX_DATA_DIR moves the tick vault to another drive (disk-full remedy):
#   PowerShell:  $env:APEX_DATA_DIR = "D:\apex_data"   (then move the .db there)
DATA_DIR   = Path(os.environ.get("APEX_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
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
# the live instrument dump, which is how the Jan-2026 lot change (NIFTY 75→65,
# BANKNIFTY 35→30, FINNIFTY 65→60, MIDCPNIFTY 140→120; NSE circ. FAOP70616)
# cost this system nothing.)
# ----------------------------------------------------------------------------
INDICES = {
    "NIFTY":      {"exchange": "NFO", "spot_symbol": "NSE:NIFTY 50",        "weekly": True,  "lot_fallback": 65,  "strike_step": 50},
    "BANKNIFTY":  {"exchange": "NFO", "spot_symbol": "NSE:NIFTY BANK",      "weekly": False, "lot_fallback": 30,  "strike_step": 100},
    "FINNIFTY":   {"exchange": "NFO", "spot_symbol": "NSE:NIFTY FIN SERVICE","weekly": False, "lot_fallback": 60,  "strike_step": 50},
    "MIDCPNIFTY": {"exchange": "NFO", "spot_symbol": "NSE:NIFTY MID SELECT", "weekly": False, "lot_fallback": 120, "strike_step": 25},
    "SENSEX":     {"exchange": "BFO", "spot_symbol": "BSE:SENSEX",          "weekly": True,  "lot_fallback": 20,  "strike_step": 100},
    "BANKEX":     {"exchange": "BFO", "spot_symbol": "BSE:BANKEX",          "weekly": False, "lot_fallback": 30,  "strike_step": 100},
}
INDEX_ORDER = list(INDICES.keys())

# ============================================================================
# MCX COMMODITIES (v9.7.1) — DATA HARVEST ONLY (no trading engine yet).
# Declares WHICH commodities to capture; lot size and strike step are read
# from the live Kite instrument dump (authoritative + current), NOT hardcoded,
# because MCX revises specs and mini/regular differ. The commodity "underlying"
# is the FRONT-MONTH FUTURE (there is no spot index) — resolved from the dump
# by name, its token rolling each expiry. `name` is the Kite instrument `name`
# field on MCX; `fut_segment` is where the future/options live.
#
# Commodities differ from equity in ways the FUTURE ENGINE must handle (they
# are irrelevant to raw tick capture): a much longer session (to 23:30 IST),
# news-driven gaps (EIA Wed 20:00 IST, OPEC, weather), monthly option expiries
# that devolve into futures 7 trading days before futures expiry, and far
# higher IV. HARVESTING needs none of that — the WS delivers MCX ticks whenever
# the segment is open and the run-loop is session-agnostic, so capture "just
# works" to 23:30. The trading engine is a SEPARATE, later, evidence-gated build.
COMMODITIES = {
    "CRUDEOIL":   {"exchange": "MCX", "fut_segment": "MCX", "lot_fallback": 100,
                   "strike_step": 50,  "session_close": "23:30"},
    "NATURALGAS": {"exchange": "MCX", "fut_segment": "MCX", "lot_fallback": 1250,
                   "strike_step": 5,   "session_close": "23:30"},
    "GOLD":       {"exchange": "MCX", "fut_segment": "MCX", "lot_fallback": 100,
                   "strike_step": 100, "session_close": "23:30"},
    "SILVER":     {"exchange": "MCX", "fut_segment": "MCX", "lot_fallback": 30,
                   "strike_step": 100, "session_close": "23:55"},
    "COPPER":     {"exchange": "MCX", "fut_segment": "MCX", "lot_fallback": 2500,
                   "strike_step": 5,   "session_close": "23:30"},
}
COMMODITY_ORDER = list(COMMODITIES.keys())
# Commodities the harvester CAPTURES. Trading remains OFF for all of them until
# the vault has enough MCX ticks to calibrate real thresholds (the same
# evidence gate every other subsystem obeys). HARVEST_COMMODITIES drives the
# harvester ONLY; it does NOT add anything to the tradable universe.
HARVEST_COMMODITIES = ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "COPPER"]
COMMODITY_TRADABLE = ["CRUDEOIL"]       # ← stays empty until calibrated. Do not edit.

# --- commodity scheduled-event / news-gap guard (core/event_engine.py) ---
# The honest "news" layer: block/flatten around KNOWN releases (EIA petroleum
# Wed, EIA natgas Thu, plus dated OPEC/FOMC/CPI/NFP in EVENT_OVERRIDES). Pure
# calendar math, DST-correct ET→IST. Advisory (blocks entries only). Harvest-
# only for now; consumed by the future commodity engine + nightly analyst.
EVENT_GUARD_ENABLED     = True
EVENT_BLACKOUT_PRE_MIN  = 20     # block new entries N min before a release
EVENT_SETTLE_POST_MIN   = 30     # volatile settle window N min after
# EVENT_OVERRIDES: list of core.event_engine.MarketEvent for dated one-offs
# (OPEC/FOMC/CPI/NFP) and EIA holiday shifts. Empty ⇒ only weekly EIA rules.
# The nightly analyst (Gemma) can DRAFT additions here for operator review.
EVENT_OVERRIDES         = None

# --- Gemma nightly analyst (tools/gemma_analyst.py) — OFFLINE reasoning ---
# Gemma 4 E4B via Ollama (Q4_K_M, fits 8GB). Runs ONCE nightly in run_evening,
# reasons over the event calendar + the night's reports, writes a digest +
# brief. FAIL-SAFE: if Ollama is down/absent, the analyst is skipped and the
# system runs identically. NEVER in the tick/decision path. All hash-excluded.
GEMMA_ANALYST_ENABLED   = True
# Ollama tag. gemma4:e4b (~9.6GB) has the better reasoning but exceeds 8GB VRAM,
# so on an RTX 4060 Ollama offloads part to CPU — SLOWER but fine for a
# once-nightly analyst that isn't latency-bound. For fully-on-GPU inference on
# 8GB, use "gemma4:e2b" (~7.2GB, fits with headroom). Verify with `ollama list`.
GEMMA_MODEL             = "gemma4:e4b"
OLLAMA_HOST             = "http://127.0.0.1:11434"
GEMMA_NUM_CTX           = 4096            # keep KV cache small on 8GB
GEMMA_TIMEOUT_S         = 120            # generous; nightly, not latency-bound

# --- commodity calibration gates (Track-A daily backfill + Track-B intraday) ---
COMMODITY_CALIB_MIN_DAYS = 250   # Track-A trusts a daily stat only with ≥ this
COMMODITY_CALIB_MIN_TICKS = 30000  # Track-B trusts an intraday stat only w/ this
# commodity heuristic policy weights (same physics as equity HEURISTIC_W; can be
# retuned per commodity microstructure once Track-B data exists). Entry bar can
# sit higher than equity given commodity IV. All hash-excluded (harvest-side).
COMMODITY_HEURISTIC_W = (0.45, 0.25, 0.15, 0.15)   # (ofi, dealer, velocity, mom)
COMMODITY_ENTRY_CONVICTION = 0.72
# commodity forge (nightly_commodity_forge.py) — trains the commodity meta on
# harvested ticks with the SAME purged-CV GBM pipeline as equity. Data-gated.
COMMODITY_META_MIN_TRAIN   = 300   # labeled signals before a model may train
COMMODITY_FORGE_COOLDOWN_S = 180   # sampler cooldown between signals/commodity
# --- evening MCX session capture (operator decision 2026-07-18) ---
# True → the supervisor runs the harvester + commodity brain on the FULL MCX
# window (COMMODITY_SESSION_OPEN → max COMMODITIES session_close); equity procs
# keep the equity window. False → everything reverts to the equity window.
EVENING_CAPTURE_ENABLED = True
COMMODITY_SESSION_OPEN  = "09:00"   # MCX open (before equity 09:15)
RISK_STATE_PERSIST      = True      # day ledger survives crash-restarts (F1)
COMMODITY_CAPITAL_FRAC  = 0.35      # commodity book's capital partition (F5)
# COMMODITY_CAPITAL_RS: absolute rupee capital for the commodity book. When set
# it OVERRIDES the fraction above — operators think in rupees, not fractions,
# and a fraction silently rescales if TRADING_CAPITAL ever changes. None = use
# the fraction. (2026-07-23: operator has Rs 2,000 allocated to commodities.)
COMMODITY_CAPITAL_RS    = 2000.0
COMMODITY_NO_ENTRY_BEFORE_CLOSE_MIN = 25   # per-commodity curfew (F2)
COMMODITY_FLATTEN_BEFORE_CLOSE_MIN  = 30   # per-commodity EOD (S2-F1)
FEED_SILENT_WARN_S      = 30   # harvester shouts if no tick arrives this long
# META_MIN_BSS: refuse to promote a meta whose Brier Skill Score against the
# base-rate climatology falls below this (the discipline rvnet already applies
# to HAR). None = REPORT ONLY — compute and log the skill, promote regardless.
# That is the default, so this knob changes nothing until you arm it; set 0.0
# to require that the model at least beat always-predicting the base rate.
META_MIN_BSS            = None
# META_MIN_POSITIVES: a probability gate cannot be calibrated on a handful of
# winners. Tonight the commodity forge promoted a model fit on ~5 positives out
# of 384 (base rate 1.25%) and reported "98.8% holdout accuracy" — which is just
# the class imbalance, i.e. always predicting LOSS. Below this many positive
# (or negative) outcomes the artifact is refused and the brain stays
# heuristic-only, which is the strictly safer status quo. Equity is unaffected
# (759 signals x 30.3% = ~230 positives).
META_MIN_POSITIVES      = 30
# META_MIN_OOF_SPREAD: refuse a meta whose calibrated OOF predictions barely
# vary (a constant cannot discriminate). None = REPORT ONLY, and that default
# is deliberate: withholding the EQUITY meta re-opens entries the counterfactual
# grades at 0 wins in 400. Diagnose first; act on evidence, not reflex.
# META_MIN_AUC: refuse to promote a meta whose calibrated OOF AUC is below
# this. 0.500 = no ordering ability at all. Unlike the spread and BSS knobs
# this one is ARMED by default: on 2026-07-28 the commodity forge promoted a
# model at AUC 0.4915 that the same run had labelled "NO RANKING SIGNAL", and
# refusing only leaves that brain heuristic-only — strictly safer. Set None to
# disable (report-only).
# META_CROSS_INDEX: append the 3 cross-index peer-context features
# (peer_agree, peer_max_agree, peer_dispersion — see core/cross_index.py) to
# the meta x-vector, taking it from 61 to 64 dims. Measured motivation:
# tools/cross_index_overlap reported co-fire 0.56, outcome agreement 0.85 vs
# 0.62 by chance, rho 0.61 — real structure the models could not see.
# DEFAULT OFF: turning it on changes the feature world, so the forge must
# retrain before serving can use it. The x_dim guard in meta_gbm refuses any
# model/vector width mismatch, so a half-migrated state cannot score silently.
META_CROSS_INDEX        = True

# PARALLEL_DAY_WORKERS: worker processes for independent per-day replays
# (core/parallel_days.py). 0 = auto = min(6, cpu//2). Measured motivation:
# the 2026-07-28 cascade_harness rebuild ran ~57 min/day and was 11 of 30 days
# in after 10.5 hours, on one core, while 13 others sat idle. Each worker holds
# a day of tick arrays so this is memory-bound before CPU-bound — raise it only
# if RAM allows. NOT a GPU workload: the cost is a Python loop over 22,500
# seconds plus SQLite reads, not matrix math.
PARALLEL_DAY_WORKERS    = 0
# v9.9.3 MEMORY-AWARE POOL SIZING (16 GB-laptop safety).
# A day-replay worker holds tick arrays + the prem table; on this vault a
# built day cache is a direct proxy for that working set, so the pool sizer
# MEASURES the median cache size on disk and multiplies by
# PARALLEL_RAM_WORKSET_MULT to estimate live RSS, instead of guessing. When
# no cache exists yet (first-ever prime), PARALLEL_RAM_PER_WORKER_GB is the
# fallback. RESERVE is left for Windows, the browser, and any live process.
PARALLEL_RAM_RESERVE_GB    = 2.5   # v9.9.3: 4.0 zeroed the pool on a busy 16 GB box
PARALLEL_RAM_PER_WORKER_GB = 1.5
PARALLEL_RAM_WORKSET_MULT  = 2.0   # v9.9.3: 3x over-charged cache-READ workers
PARALLEL_RAM_AWARE         = True
# ── v9.9.6 durability & network resilience ───────────────────────────
POSITION_SNAPSHOT_MAX_AGE_S = 900    # refuse a snapshot older than 15 min
NET_GUARD_ENABLED           = True
NET_GUARD_HOSTS             = ["api.kite.trade", "kite.zerodha.com",
                               "ws.kite.trade"]
DNS_CACHE_TTL_S             = 300
DNS_TIMEOUT_S               = 5.0
SOCKET_DEFAULT_TIMEOUT_S    = 15.0
LOOP_STALL_WARN_S           = 90     # 2026-07-29: 36 min of silence, unseen

# ── v9.9.4 discovery / capability ladder ──────────────────────────────
DISCOVERY_ENABLED          = True
LADDER_POWER               = 0.80    # detect at 80% power…
LADDER_ALPHA               = 0.05    # …one-sided 5%
LADDER_SCREEN_MDE          = 0.60    # unlock horizon sweep at this MDE
LADDER_DISCOVER_MDE        = 0.55    # unlock feature screening here
DISCOVERY_FDR_Q            = 0.10    # Benjamini-Hochberg false-discovery rate
HORIZON_CANDIDATES         = [20, 30, 45, 60, 90, 120]
HORIZON_MIN_DAYS           = 20
HORIZON_BOOT               = 1500
HORIZON_AUTO_ADOPT         = False   # recommend by default; operator applies
HORIZON_ADOPT_MIN_NIGHTS   = 3       # a winner must repeat 3 nights running
HORIZON_ADOPT_RANGE        = (20, 120)
FEATURE_MIN_GROUP_TESTS    = 4
PARALLEL_MIN_WORKERS       = 2     # v9.9.3: grant 2 whenever they physically
#                                    fit — 1 worker turned 2026-08-01 into a
#                                    20-hour serial night
# XDIM_REMIND_S: how often to repeat the x-dim mismatch error. It fires on
# every evaluation (several/second/index), so on 2026-07-29 an un-throttled
# ERROR buried the session log. Static condition -> say it once, then remind.
XDIM_REMIND_S           = 900
META_MIN_AUC            = 0.52
META_MIN_OOF_SPREAD     = None
CASCADE_MAX_FLIP_DIST_PCT = 0.05   # reject GEX flips this far from spot
SUPERVISOR_TABS         = True      # one Windows Terminal viewer tab per child
#                                     (tabs tail per-process logs; closing a tab
#                                     kills nothing — supervision stays direct)
# a commodity is TRADE-ELIGIBLE only when BOTH tracks are calibrated AND it is
# explicitly in COMMODITY_TRADABLE (which stays empty until you put it there).

# ★ Indices the brain may actually TRADE (others remain context nodes only).
# At ₹60k the Kelly walker fits NIFTY across most of the week and SENSEX
# (lot 20) on cheaper strikes; both stay in. Trim to ["NIFTY"] if SENSEX
# BLOCKED lines get noisy at lower capital.
TRADABLE = ["NIFTY", "BANKNIFTY", "SENSEX"]   # 2026-07-28: BANKNIFTY added by
# operator. It was already harvested (787k leg-ticks/day, depth rate 1.0) and
# already in INDEX_ORDER, so the frame needs no change — this only makes it
# tradable. NOTE: TRADABLE is hash-INCLUDED, so CONFIG_HASH moves and the
# forge cache/meta/certificates all rebuild against the new world. That is the
# hash doing its job: what the system trades genuinely changed.

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
# v9.9.12: the equity curfew is now DERIVED from the session calendar
# (session close − NO_ENTRY_BEFORE_CLOSE_MIN), not a hand-set clock time,
# so it moved with the 2026-08-03 extension instead of silently shutting
# entries off 35 minutes early. core.session_calendar.entry_curfew_hm()
# is the single source; this constant remains only as the pre-reform
# fallback for anything that reads it directly.
NO_ENTRY_BEFORE_CLOSE_MIN = 5      # last entry this many min before close
NO_ENTRY_AFTER = "15:05"   # 2026-07-28: was 14:45. Five of seven cascade
# triggers on 07-27 fired at/after 14:45 and one was blocked outright — the
# cascade's natural window is the late session. CAUTION: FORCE_FLATTEN_AT is
# 15:15, so an entry at 15:05 has only 10 minutes before a FORCED exit, and
# forced exits fill worse than chosen ones. Watch the EOD_FLATTEN share of
# exits; if late entries mostly die on the flatten, this window is too tight.
FORCE_FLATTEN_AT         = "15:15"   # safely before broker MIS auto square-off (~15:20)
SESSION_OPEN             = "09:15"
# ── 2026-08-03 CLOSING AUCTION SESSION REFORM (NSE circular 2026-05-30,
#    SEBI CAS framework). Equity DERIVATIVES now close 15:40 (was 15:30);
#    the cash market runs a Closing Auction 15:15–15:35 for F&O stocks,
#    whose auction price replaces the VWAP close. Index constituents are
#    all F&O names, so for those twenty minutes the INDEX PRINT is not a
#    continuously-traded price even though the option still is.
#    SESSION_CLOSE stays as the LEGACY value: core.session_calendar is
#    date-aware and returns 15:30 for every session before the reform, so
#    the vault's June/July days replay under the rules that actually
#    governed them.
SESSION_CLOSE            = "15:30"   # pre-reform / fallback
SESSION_CLOSE_CAS        = "15:40"   # NSE derivatives, from 2026-08-03
CAS_START                = "15:15"   # cash auction begins (F&O stocks)
CAS_END                  = "15:35"   # auction price published
CAS_BLACKOUT_ENABLED     = True      # suspend ENTRIES while spot is in auction
HARD_FLAT_MARGIN_MIN     = 5         # be flat this many minutes before close
# BSE has published a CAS scrip-master indicator but its timing circular is
# still awaited. Until it lands, SENSEX/BANKEX keep the 15:30 close: assuming
# an unconfirmed extension would leave positions open in a shut market.
BSE_FOLLOWS_NSE_CAS      = False
# 15:35-15:40 POST-AUCTION WINDOW: cash close known and published, index
# derivatives still open. Ten minutes that did not exist before 2026-08-03
# and about which this vault holds ZERO observations. Entries there stay
# OFF until the harvester has banked enough post-auction sessions for the
# discovery stack to say something; exits and position management run
# normally throughout.
POST_AUCTION_ENTRIES     = False
# Phase 2 of the same SEBI framework (circular 2026-01-16): the morning
# pre-open is restructured from 2026-09-07 — 09:00-09:05 market+limit,
# 09:05-09:10 limit only with random close 09:08-09:10, 09:10-09:12
# matching, 09:12-09:15 transition. Dated here so the system is ready
# rather than surprised; SESSION_OPEN itself is unchanged at 09:15.
PREOPEN_REFORM_DATE      = "2026-09-07"
# ── post-auction regime (15:35-15:40). Opens ITSELF once the vault holds
#    POST_AUCTION_MIN_SESSIONS of the window AND the fitted geometry shows
#    the median move can pay the round-trip spread at 2:1 odds. If it
#    cannot, the window never opens — that is a result, not a failure.
POST_AUCTION_ENABLED        = True   # master switch for the regime
POST_AUCTION_MIN_SESSIONS   = 7      # a week of data before it may trade
POST_AUCTION_HOLD_MIN       = 4.0    # minutes — the window is five long
POST_AUCTION_FLAT_HM        = "15:39"
POST_AUCTION_PREMIUM_LEVERAGE = 12.0  # prior for premium response to spot
POST_AUCTION_MIN_EDGE_MULT  = 1.0    # net edge must exceed 1x the spread
# ── CAS capture (15:15-15:35). The cash constituents are in auction but the
#    INDEX OPTIONS never stop trading, so put-call parity on the ATM pair
#    gives a continuously-traded synthetic underlying — the market's own
#    forecast of what the auction will print. Recorded every second to
#    state/cas_tape/ ; this tape is the training set for the regime and
#    nothing else in the system stores it.
DYN_TP_EXTEND_MULT       = 2.0    # runway may extend the vault target by
#                                   this multiple, never past DYN_TP_MAX
QUOTE_API_DEADLINE_S     = 2.0    # hard bound on any kite.quote() call
MARK_STALE_WARN_S        = 15     # warn when exits run on an older mark
CAS_CAPTURE_ENABLED      = True
CAS_PREPRINT_ENABLED     = True   # may we position BEFORE the 15:35 print?
CAS_MIN_SESSIONS         = 7      # tapes required before that may open
CAS_ALPHA                = 0.05   # exact sign-test level (NOT a hit-rate
#                                   floor: at n=9, 6/9 correct signs is 67%
#                                   and occurs 25% of the time on noise)
CAS_MAX_QUALITY_PENALTY  = 0.10   # drop rows built from >10% wide quotes

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
# v9.9.4 AUTOMATIC HORIZON RESEARCH. tools/horizon_sweep may adopt a new
# label/hold horizon by writing state/horizon_override.json — but ONLY after
# it survives BH-FDR across all candidates AND repeats for
# HORIZON_ADOPT_MIN_NIGHTS consecutive nights AND HORIZON_AUTO_ADOPT is on.
# Reading it HERE (not editing source) means the adopted value flows through
# CONFIG_HASH exactly like a hand edit: caches and artifacts invalidate
# themselves. Delete the file to revert instantly.
def _horizon_override(default: int) -> int:
    try:
        import json as _j
        p = STATE_DIR / "horizon_override.json"
        if p.exists():
            v = int(_j.loads(p.read_text())["max_hold_minutes"])
            if 20 <= v <= 120:
                return v
    except Exception:                                      # noqa: BLE001
        pass
    return default


MAX_HOLD_MINUTES     = _horizon_override(60)   # theta guillotine (v9.9.1:
#                               45→60 by operator request. RIDE_MULT 2.0 ⇒
#                               hard cap 120 min while theta-riding. 0-DTE
#                               keeps its own 25-min knob. FINGERPRINTED:
#                               changing this rotates CONFIG_HASH — labels
#                               regrade at the 60-min first-touch window and
#                               every cache/artifact rebuilds, by design.)
MAX_ENTRY_SPREAD_PCT = 0.03   # refuse entries when (ask-bid)/mid above this

# Conviction → probability. The calibration table (built nightly by the
# analyzer from real/paper outcomes) is the truth; this floor is the fallback.
# (v9.1: MIN_CAL_WINPROB deleted — it was defined and enforced NOWHERE; a
# constant that reads like a safety gate but isn't is worse than none.)
ENTRY_CONVICTION     = 0.70
PAPER_ENTRY_CONVICTION = 0.55  # used ONLY if PAPER_EXPLORE=True (see above)
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
# v9.1: the window is WALL-CLOCK. The old deque of the last 4 brain-loop
# iterations spanned ~0.8 s at the 5 Hz loop; SIGNAL_PERSIST_FRAC / AVG_MULT
# were defined and read nowhere (audit). The tracker in core/decision.py keys
# samples on timestamps and evicts by SIGNAL_PERSIST_WINDOW_S, so "held for N
# seconds" means N seconds at any cadence — brain ~5 Hz, forge replay 1 Hz,
# same window, same test (coherence + Kaufman-ER + tape agreement, unchanged
# in core/signal_persistence.py).
SIGNAL_PERSIST_ENABLED     = True
SIGNAL_PERSIST_WINDOW_S    = 12.0   # the read must have held over this window
SIGNAL_PERSIST_MIN_SAMPLES = 4      # gate is skipped until this many samples

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
#   3. Liquidity is gated by MAX_ENTRY_SPREAD_PCT (3%), chasing by
#      ENTRY_SLIP_CAP_PCT.
# Paper and live use the identical logic; crossing is what live would do.
ENTRY_CROSS_CONVICTION = 0.70   # = ENTRY_CONVICTION: momentum entries cross
ENTRY_SLIP_CAP_PCT     = 0.60    # fraction of one strike-step premium move
SLIPCAP_BORDERLINE_FRAC = 0.25   # diagnostics only (changes NO behavior): a
# chase-cap walk-away within this fraction past the cap is "borderline" (the cap
# may be slightly tight on a genuine fill); beyond it is a "runaway" the cap
# correctly refused. Classifies walk-aways for the heartbeat tally / evidence.
ENTRY_CROSS_SPREAD_PCT = 0.015   # DEPRECATED/unused: the separate cross-spread
# band was removed — it starved fills by routing real (1.5–3% spread) NIFTY
# option signals to the passive path. Kept only so external refs don't break.
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

# Diagnostics (v9.1): every long-running component writes a machine-readable
# daily report to logs/<component>_report_<date>.json — the gate funnel, tick
# coverage, radar health, forge walk-forward table. Purely observational.
DIAG_WRITE_EVERY_S      = 600.0   # brain/harvester/macro report cadence

# Execution micro-knobs
PAPER_SLIPPAGE_TICKS  = 1
URGENT_CHASE_TICKS    = 2
LIVE_POLL_INTERVAL_S  = 0.4

# Harvester
# AUDIT (2026-07-26 weekly): the affordability walker reaches ATM+
# (HIERARCHY_DEPTH-1) = +7 strikes, but this window only subscribed ATM+/-6 —
# so the walker routinely asked for legs the harvester never captured. In the
# cascade spec run that showed up as "unharvested:N" on 11 of 16 skipped
# triggers: 12 REAL triggers (sane flips) never became fills purely because
# their strikes were outside the harvest window. Cascade needs 20 certificate
# events and has 11; those lost fills are the difference. Cost of the fix is
# +48 websocket tokens (298 -> ~346) against Kite's 3000 limit.
# INVARIANT: keep PRUNE_STEPS >= HIERARCHY_DEPTH - 1.
PRUNE_STEPS = 8
#                         capital the first AFFORDABLE rung sits several
#                         strikes OTM (₹5k ⇒ premium ≤ ₹14.3 on NIFTY); the
#                         vault must carry what the account can actually buy,
#                         or live AND replay both die at no_quotes. Hash-
#                         excluded: retune freely, no re-forge.
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

# Forge training knobs (v9.1: the SAC rollout knobs the bandit trainer never
# read — SAC_BUFFER/TRAIN_FREQ/GRAD_STEPS/TIMESTEPS_CAP — are deleted; only
# what the code reads remains)
SAC_BATCH           = 256
FORGE_ACT_GATE_TRAIN = 0.3
FORGE_ACT_GATE_EVAL  = 0.5

# ----------------------------------------------------------------------------
# RESEARCH LAYER (each knob traces to published evidence — see README)
# ----------------------------------------------------------------------------
# Edge Certificate — the third lock. SEBI FY25: 91% of individual F&O traders
# lose (net ₹1.06 lakh crore). This system therefore CANNOT arm live until its
# own paper ledger clears statistical proof of edge.
EDGE_MIN_TRADES      = 100
EDGE_MIN_DAYS        = 20
EDGE_BOOTSTRAP_N     = 10_000
EDGE_CI              = 0.95     # bootstrap CI lower bound of mean PnL must be > 0
EDGE_CERT_PATH       = STATE_DIR / "edge_certificate.json"
EDGE_CERT_VALID_DAYS = 7

# Meta-labeling (López de Prado 2018): primary model picks the SIDE, a
# secondary model learns the SIZE — P(win | features) from triple-barrier
# outcomes on REAL recorded prices, after real costs. Feeds Kelly directly.
# v9.1: samples are UNIQUENESS-WEIGHTED (AFML ch.4 — overlapping 25–45-min
# label windows at 1 Hz are ~99.9% redundant; weighting by 1/concurrency stops
# the fit and its holdout accuracy from being dominated by duplicated paths),
# and the holdout is BY DAY (the last training day), not a row split.
META_MODEL_PATH = STATE_DIR / "meta_model.json"
META_MIN_TRAIN  = 300          # labeled signals before the meta model is trusted
META_LR         = 0.05
META_EPOCHS     = 300
META_L2         = 1e-3
# --- v9.8 META-FORGE v2 (LightGBM + purged CV + isotonic; logistic fallback)
META_ENGINE       = "gbm"      # "gbm" (seconds, calibrated) | "logit" (v9.1)
META_EMBARGO_DAYS = 1          # purge ± this many days around each CV fold
META_GBM_LEAVES   = 15
META_GBM_LR       = 0.05
META_GBM_ROUNDS   = 600        # ceiling; early stopping picks the real one
META_GBM_MINCHILD = 25

# --- v9.9 META-GATE v3 (Venn-Abers intervals + per-trade EV zones) -------
# "ev": three-zone EV gate on VA intervals vs per-trade breakeven p*.
# "bar": legacy fixed META_ENTRY_P_BAR veto (byte-identical fallback).
META_GATE_MODE        = "ev"
META_EV_MARGIN        = 0.02   # static safety margin added to p*
META_ACI_GAMMA        = 0.02   # online margin learning rate (per outcome)
META_ACI_MAX          = 0.10   # |adaptive margin| hard clip
META_VA_MAX_CAL       = 4000   # VA calibration-set cap (stride-thinned)
META_FEAT_WINDOW      = 240    # live x-vectors watched for train/serve skew
META_FEAT_FROZEN_MIN  = 8      # trained-alive features frozen live ⇒ MONITOR
META_PROBE_ENABLED    = True   # ambiguous zone may enter at minimum size
META_PROBE_MAX_PER_DAY   = 3   # probes per book per day (hard)
META_PROBE_MAX_DAY_RISK_PCT = 0.02  # Σ probe worst-case loss ≤ 2% equity/day
COMMODITY_META_GATE   = "size_only"  # "size_only" (today) | "ev" (opt-in)
META_GATE_STATE = STATE_DIR / "meta_gate_aci.json"
META_PROBE_STATE = STATE_DIR / "meta_probe_ledger.json"
# v10.2 SCALE DOCTRINE: evenings must stay O(new days) as the vault grows.
HARNESS_MAX_DAYS    = 60      # certificates graded on the trailing N vault
                              # days (0 = all history). Recency-relevant
                              # evaluation window; cache makes even 0 cheap.
META_TRAIN_MAX_DAYS = 90      # meta/DEE training window (0 = all).
META_USE_PLO      = False      # serve the per-bin Wilson LOWER bound of
                               # P(win) instead of the point estimate — the
                               # conformal-style gate. Flip only as a
                               # registered trial once bins are populated.
META_PLO_MIN_N    = 30
# v9.9 front-month futures for the basis read (update at rollover, e.g.
# {"NIFTY": "NFO:NIFTY25JULFUT", "SENSEX": "BFO:SENSEX25JULFUT"}); empty = off
FUT_SYMBOLS       = {}
VIX_TOKEN         = 264969     # NSE INDIA VIX — harvester archives it so
                               # future harnesses can apply the spike veto
                               # HISTORICALLY (brain already consumes live)
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

# ============================================================================
# FAST LANE (v9.7.1) — conviction-gated quick-profit exit, SEPARATE from the
# normal 45-min path. A high-conviction entry gets a short profit clock: it
# takes a sharp gain fast, but FALLS BACK to the normal exit logic if the move
# doesn't come (so it never cuts a slow winner short — it only ADDS a fast
# take-profit on top). This does NOT change the meta-labeler's 45-min training
# barrier; it's a serving-time exit overlay only. Hash-excluded (exit tempo).
FAST_LANE_ENABLED       = True
FAST_LANE_CONVICTION    = 0.85   # entry conviction to qualify (well above 0.70)
FAST_LANE_MIN_HOLD_S    = 180    # 3 min: don't even check fast-TP before this
FAST_LANE_MAX_HOLD_S    = 600    # 10 min: after this, hand back to normal path
FAST_LANE_TP_PCT        = 0.22   # quick take-profit target (premium fraction)
# 2026-07-24: the fast lane used to bank a hard +22% even while the move was
# still printing new highs, preempting rung 4's target-extension judge. It now
# waits for the spike to STALL (no new high for RUN_GRACE_S). Set
# FAST_LANE_DEFER_WHILE_RISING=False to restore the old hard-bank behaviour.
FAST_LANE_DEFER_WHILE_RISING = True
FAST_LANE_RUN_GRACE_S        = 20.0
FAST_LANE_ARM_PCT       = 0.12   # only arm the fast-TP once this far in profit
# loss-streak circuit breaker — the operator's anti-overtrading guard
LOSS_STREAK_HALT        = 3      # N consecutive losing trades ends the day
LOCKOUT_BYPASS_REQUIRE_RECLAIM = True  # bypass also needs price to have RECLAIMED
#                                        the loss level (confirmed trap, not just
#                                        a stronger signal) — the "strongest bar"

# Provenance: every run logs the exact config it traded on.
import hashlib as _hl
# CONFIG_HASH is computed at the END of this module — it must see every constant
# defined below it (e.g. DRIFT_KEY_FEATURES, COSTS). See _compute_config_hash().
# It is a FEATURE/MODEL fingerprint, not a whole-file hash: operational knobs
# (capital, sizing, polling, paths, …) are excluded so editing them no longer
# invalidates a trained reference. live_fire_armed() above reads it at runtime,
# by which point the end-of-file assignment has run.
# ★ v9.1 NOTE: this hash CHANGES vs v9.0 (label horizon is now 0-DTE-aware,
# entry basis in labels is the ask, dead hashed constants were removed). The
# existing drift reference and meta model read NO_REF / stale until the first
# nightly forge run restamps them. Expected, one-time.

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
# conviction MULTIPLIER. ★ v9.1: the multiplier is applied in LOGIT space
# (core/decision.py) — the audit proved the old linear `conv × mult` made CHOP
# (×0.70) and VOL_CRUSH (×0.65) arithmetically un-enterable under the 0.70 bar
# (tanh < 1), i.e. a hard veto the contract below forbids. Logit scaling keeps
# every regime reachable while still demanding a stronger raw signal in
# dampened tapes. Cut points start fixed and are refit nightly to the
# empirical percentiles of THIS market.
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
# conviction multipliers per regime (logit-space scale on the brain's
# conviction; 1.0 = neutral)
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
# FORGE / TRAINING  (v9.1 — the falsification harness)
# ----------------------------------------------------------------------------
FORGE_MAX_TRAIN_DAYS  = 0      # 0 = train on every harvested day except the
#                                final (promotion) day. Set >0 to cap once the
#                                per-day replay becomes the nightly bottleneck.
FORGE_WF_FOLDS        = 5      # walk-forward out-of-sample folds: each of the
#                                last K pool days is validated by a model
#                                trained ONLY on the days strictly before it.
#                                The promotion day is additionally NEVER seen
#                                by any fold, any inner split, or any
#                                checkpoint selection (audit: the old trainer
#                                selected checkpoints ON the promotion day).
FORGE_PROMOTE_MARGIN_RS = 500.0  # ★ additive ₹ a candidate must clear ABOVE
#                                max(heuristic, incumbent) on the untouched
#                                promotion day. v9.0 read a name that was never
#                                defined and silently ran with ₹0.
# AUDIT (2026-07-29): the auto default was min(4, cpu//2) — a cap chosen when
# a rebuild was rare and small. On a 14-core box facing a full 30-day rebuild
# (which the 2026-07-28 hash move forced) that leaves ~10 cores idle while each
# day costs ~684s. Aligned with core/parallel_days.default_workers() so the
# forge and the harnesses use ONE policy: min(6, cpu//2), memory-bound because
# each worker holds a day of tick arrays. 0 = auto; raise only if RAM allows.
FORGE_PARALLEL_WORKERS = 0
#                                0 = auto (cpu_count//2, i7-13650HX ⇒ ~7),
#                                1 = serial (debug).
FORGE_EVAL_CAPITAL    = 100000.0  # ★ the REFERENCE account every forge label,
#                                reward and grade is sized against (Kelly
#                                budget ≈ ₹16,667 ⇒ ATM NIFTY/SENSEX always
#                                affordable — the exam measures EDGE, never
#                                account size). TRADING_CAPITAL above is the
#                                LIVE knob: hash-EXCLUDED, change it any time
#                                with zero ripple into forge/meta/drift/caches.
#                                This constant IS hashed — editing it changes
#                                labels, so a re-forge is correctly forced.
RISK_FREE_RATE        = 0.07

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ----------------------------------------------------------------------------
# GAMMA-CASCADE MODULE (v9.2 — the structural crash/squeeze engine)
# ----------------------------------------------------------------------------
# Mechanism (published, not invented): below the gamma flip with dealers net
# short gamma, hedging flow trades WITH the move — intraday momentum
# concentrates under negative gamma (Baltussen–Da–Lammers–Martens, JFE 2021),
# option hedging pressure moves the underlying (Ni–Pearson–Poteshman 2005;
# Barbon–Buraschi 2021). Direction comes from the STRUCTURE + impulse sign,
# not from the momentum heuristic the meta-labeler has (correctly) buried.
#
# FALSIFICATION-FIRST, enforced in code: the live detector runs telemetry-only
# until tools/cascade_harness.py has graded every historical trigger on YOUR
# vault (ask entries, real costs, shaped barriers — the exact trade the brain
# would place) and written CASCADE_CERT_PATH with ok=true. The certificate is
# stamped with CONFIG_HASH + a fingerprint of every knob below; touch a knob
# and the cert invalidates until the harness re-passes. Same lock philosophy
# as LIVE_FIRE. All knobs hash-EXCLUDED (they change no feature/label of the
# main system) — the cert fingerprint is their invalidation channel.
CASCADE_LIVE_ENABLED   = True     # master switch; still inert without a cert
CASCADE_PAPER_EXPLORE  = True     # ★ v9.2.1 staging tier: with NO certificate,
#                                   cascade entries are allowed in PAPER ONLY
#                                   (hard-blocked when live-armed) so every
#                                   trigger accrues FORWARD out-of-sample
#                                   evidence through the real execution path.
#                                   The harness blends realized forward fills
#                                   into the certificate; the 2026-07-09 run
#                                   proved backtest-only certification is
#                                   structurally impossible on pre-widening
#                                   vault history (36 triggers, 35 unfillable:
#                                   ATM±3 harvest vs an 8-rung ladder). LIVE
#                                   cascade still requires the certificate
#                                   AND all four live locks.
CASCADE_VEL_WINDOW_S   = 60       # impulse = spot move over this window …
CASCADE_VEL_Z          = 2.0      # … must be ≥ this many σ of its own history
CASCADE_VOL_LOOKBACK_S = 1800     # σ estimated over this rolling window
CASCADE_VOL_MIN_N      = 120      # impulse samples before z is trusted
CASCADE_NET_GEX_MAX    = -1.0e10  # net GEX must be at least this negative
#                                   (deep short-gamma; same e10 scale note as
#                                   REGIME_GEX_SQUEEZE — sanity-check on your
#                                   own net_gex distribution)
CASCADE_HYST_MULT      = 1.0      # zone hysteresis = this × flip_width
#                                   (floor: one strike step) — a full-bracket
#                                   cross, not a tick-flicker
CASCADE_COOLDOWN_S     = 600      # per-index re-arm delay after a fire
CASCADE_MAX_EVENTS_DAY = 3        # per index; a cascade day is one regime,
#                                   not thirty entries
CASCADE_ENTRY_CONV     = 0.85     # signed conviction stamped on cascade
#                                   entries (drives the reversal-exit
#                                   semantics; sizing runs off the cert win
#                                   rate through the normal Kelly governor)
CASCADE_CERT_PATH      = STATE_DIR / "cascade_certificate.json"
CASCADE_CERT_MIN_EVENTS = 20      # harness pass requires ≥ this many events…
CASCADE_CERT_MIN_DAYS   = 5       # …across ≥ this many distinct event-days,
CASCADE_CERT_CI         = 0.90    # …with bootstrap CI lower bound of mean
#                                   after-cost ₹/event > 0 at this level.

# ----------------------------------------------------------------------------
# SHORT-VOL ENGINE (v9.3 — the other side of the trade the system PROVED)
# ----------------------------------------------------------------------------
# Every honest exam of this program produced the same number: long premium at
# these horizons wins ~22% after costs (meta base_rate 0.2185 on 429 real
# labels; 7/7 OOS losers). The counterparty of that condemned trade is the
# variance-risk-premium harvest — the most robustly documented edge in options
# (Carr–Wu RFS 2009; Bakshi–Kapadia 2003; Bollerslev–Tauchen–Zhou 2009;
# Israelov–Nielsen on systematic short premium; Beckmeyer–Branger–Gayda 2023:
# retail 0DTE BUYERS lose systematically). Regime side: premium is sold ONLY
# under positive net gamma (dealers damp moves — Baltussen et al. JFE 2021),
# inside the wall corridor, with rich IV rank — and NEVER while the cascade
# detector's negative-gamma machinery is anywhere near active. Sell the calm,
# own the storm; one book, two regimes, zero overlap by construction.
#
# INSTRUMENT: defined-risk vertical credit spreads (short the tested WALL
# strike, long one step further out). Max loss = width − credit, known at
# entry — the only shape retail capital can responsibly sell. Prespecified
# single spec; no optimization. Staging identical to cascade: vault harness →
# certificate (knob-hash-stamped, fail-closed) → PAPER-EXPLORE forward
# evidence → live ONLY behind cert + the four locks. v9.3.0's ceiling is
# PAPER by construction: live spread ROUTING (basket orders + real SPAN
# margin via Kite's margin API) is deliberately unbuilt until a certificate
# exists to justify it.
SHORTVOL_ENABLED        = True    # master switch; inert without cert/explore
SHORTVOL_PAPER_EXPLORE  = True    # paper-only entries pre-cert (never live)
# ---- MASTER FLY TRADING SWITCH (v9.7.1) ----
# Operator instruction: the butterfly does NOT take positions — not live, not
# paper. Its GATE still evaluates every second and feeds core/fly_intel (the
# fly's read is mined to improve DIRECTIONAL trades); this switch governs only
# whether a fly is ever OPENED. False (default) ⇒ the fly is pure telemetry,
# never occupies the global single-position lock, so directional capital is
# always free and the displacement governor is moot. Set True only to bring
# back the paper-explore fly engine (it still cannot route live — that path is
# unbuilt regardless).
FLY_TRADING_ENABLED     = False
SV_IVRANK_MIN           = 0.60    # premium must be RICH vs its own 60-day range
SV_NET_GEX_MIN          = 1.0e12  # dealers must be LONG gamma (damping regime)
SV_WALL_BUFFER_STEPS    = 1.0     # spot must sit ≥ this many strike-steps
#                                   inside BOTH walls (a real corridor)
SV_CORRIDOR_MIN_STEPS   = 3.0     # and the corridor itself ≥ this wide
SV_DTE_MIN              = 0.8     # v1 skips expiry day (short-gamma-at-pin
#                                   blowup risk; a 0DTE variant needs its OWN
#                                   prespecified harness pass)
SV_DTE_MAX              = 9.0
SV_AFTER_HM             = "10:00" # let the open's gamma flush settle
SV_WIDTH_STEPS          = 1       # long leg = wall ± 1 step (tightest hedge)
SV_MIN_CREDIT_FRAC      = 0.15    # reject if credit < this × width (junk premium)
# --- v9.7 LONG BUTTERFLY (buy-only VRP expression; same gate as shortvol) ---
SV_FLY_WING_STEPS       = 2       # wings this many strike-steps from the body
SV_FLY_MIN_DEBIT_FRAC   = 0.05    # reject fly if debit < this × wing width
SV_FLY_MAX_DEBIT_FRAC   = 0.70    # reject if debit > this × width (no room)
SV_FLY_TP_FRAC          = 0.50    # take profit at this frac of debit→maxvalue
SV_FLY_SL_FRAC          = 0.50    # stop if unwind credit ≤ debit×(1−this)
SV_TP_FRAC              = 0.50    # take profit at 50% of credit captured
SV_SL_CREDIT_MULT       = 1.00    # stop when loss ≥ 1× credit (2× credit to close)
SV_TOUCH_EXIT           = True    # urgent close if spot touches the short strike
SV_CLOSE_HM             = "15:05" # hard flat before the closing auction chop
SV_ATTEMPT_THROTTLE_S   = 300     # one build attempt per 5 min per index
SV_POP_HAIRCUT          = 0.90    # sizing prior = (1 − credit/width) × haircut
SV_RISK_PCT             = 5.0     # % of capital at risk per spread (max loss ×
#                                   lots ≤ this). Fixed-fractional, NOT Kelly:
#                                   Kelly fed the risk-neutral pop is ≈0 by
#                                   construction (the VRP edge IS true-p >
#                                   risk-neutral-p), so fractional risk is the
#                                   literature-standard prespecification.
SHORTVOL_CERT_PATH      = STATE_DIR / "shortvol_certificate.json"
FLY_CERT_PATH           = STATE_DIR / "butterfly_certificate.json"
SV_CERT_MIN_EVENTS      = 25      # cert bar: ≥ events …
SV_CERT_MIN_DAYS        = 6       # … across ≥ event-days …
SV_CERT_CI              = 0.90    # … bootstrap CI lower bound of ₹/event > 0

# ----------------------------------------------------------------------------
# COUNTERFACTUAL LEDGER (v9.4, Pillar 4) — the constitution audits itself.
# The nightly forge shadow-grades blocked signals on the promotion day through
# the SAME ask-entry barrier grader and attributes would-be ₹ to the gate that
# killed each one. Off-policy evaluation of the rulebook; report-only.
# ---- EPISODE RANKER (v9.9.35). A different unit, target and question.
# Four classifiers scored at chance on "will this win?" because 63 rows per
# session are near-duplicates: n=1953 carries n_eff 146-476, and the
# detectable AUC floor is 0.585-0.655 against an observed 0.5070. The same
# code promoted the commodity meta at 0.5979 the same night, so the
# machinery is not the problem.
# Episodes (non-overlapping), R = P&L/risk, and RANKING WITHIN A SESSION —
# because MAX_CONCURRENT_POSITIONS=1 makes this a selection problem, not a
# threshold one. Ranking is a weaker claim than calibration, and weaker
# claims are what low n_eff can support.
# ---- SEQUENCE MODEL (v9.9.36). CNN-GRU over the APPROACH — the last few
# minutes of tape before a signal — rather than the snapshot at the instant
# it fired. That is the one place a neural net beats the trees already here,
# and it is an advantage in REPRESENTATION, not capacity: it does not need
# more sessions, which is exactly what this vault lacks.
# The architecture is not the deliverable. The CONTROLS are: a shuffled-
# label run of the identical pipeline (this vault's overfitting floor) and a
# held-out month never touched by fitting, folds or early stopping.
SEQ_WINDOW_S           = 300    # seconds of approach fed to the encoder
SEQ_HOLDOUT_SESSIONS   = 22     # a trading month, removed before anything
                                # is fitted. Not a CV fold.
SEQ_CONTROL_RUNS       = 5      # shuffled-label repeats; the SD of these is
                                # the noise the real margin must clear
SEQ_MC_PASSES          = 20   # dropout-on forward passes at inference.
                              # Ensemble variance asks "do seeds agree?";
                              # MC-dropout asks "is THIS input in a region
                              # the net knows?" — the distinction the
                              # constant-0.204 meta could not express.
SEQ_N_TRIALS           = 25   # forge_report.trials_for_deflation. Every
                              # study added here raises the real count;
                              # keep this in step or the DSR flatters.
SEQ_MIN_DSR            = 0.95 # P(true Sharpe > 0) after deflation
SEQ_MODEL_ENABLED      = False  # serving switch; the study runs regardless

EPISODE_ALPHA          = 0.05
EPISODE_MIN_STABILITY  = 0.60   # fraction of folds that must pick the same
                                # top feature. An unstable selection is a
                                # different model per fold, and its OOF score
                                # describes none of them.
EPISODE_RANKER_ENABLED = False  # serving switch; measurement runs regardless

CF_NEAR_MISS    = 0.05    # bootstrap-bar near-miss band: shadow |conv| ≥ bar−this
CF_MAX_PER_GATE = 400     # per-gate/day shadow cap (sampling noted in report)

# ----------------------------------------------------------------------------
# SHADOW BOOK (v9.9.13) — every trade we took keeps trading, in parallel,
# until the session closes. The real exit stays the baseline; the shadow
# marks the SAME instrument under alternative exit rules to the bell, so
# "what did the exit leave on the table" becomes a measured number instead
# of an anecdote. Paper-only by construction: the shadow never routes.
#
# EVERY CONSTANT HERE IS IN _HASH_EXCLUDE. Measurement must never be able
# to invalidate the trained model or the raw day caches — a shadow knob
# rotating CONFIG_HASH would trigger the 31-replay cache rebuild that cost
# 12 hours on 2026-07-29. Measurement is operational, not part of the
# feature world.
# ---- META SERVE-PATH INTEGRITY (v9.9.18). meta_gbm measured the spread of
# a single isotonic map; serving uses an IVAP (two isotonic fits per query).
# On matched data the isotonic map carried 925 distinct values and the
# Venn-Abers merge carried 36 — the old check could not see a serve-path
# collapse. 2026-08-10: an artifact that passed it served two values across
# 14 655 evaluations. These gate on the SERVE estimator.
# ---- META TRAINING TARGET (v9.9.19). The shadow book teaches the ENTRY
# model by DE-NOISING the label, not by adding rows: every shadow is a trade
# the gate already took, so it estimates P(outcome | entered) and says
# nothing about blocked signals. "realized" is the incumbent target and the
# default; switching is a MODEL CHANGE and must beat it head-to-head on the
# ladder, paired by day and FDR-corrected. Specs are pre-registered in
# core/shadow_labels.LABEL_SPECS.
#   realized             : did the trade as EXITED make money  (incumbent)
#   best_policy          : would the best pre-registered exit have made money
#   any_policy_positive  : did ANY pre-registered exit come out positive
META_LABEL_SPEC = "realized"

# ---- LABEL CERTIFICATE (v9.9.20). META_LABEL_SPEC alone is NOT authority:
# an operator editing a string must not be able to change what the model is
# fitted to. core/label_certificate.active_label() is what the forge calls,
# and it is fail-closed on the house pattern (core/cascade.py:224).
# A MONTH, not a week: the regime mix rotates on roughly that scale
# (2026-08-07 was CHOP-dominant at 11 504s; 2026-08-10 HIGH_VOL at 12 500s),
# and a target fitted inside one regime is a target fitted to one month of
# weather. 22 sessions is a trading month.
LABEL_CERT_MIN_SESSIONS = 22
LABEL_CERT_MIN_TRADES   = 60
LABEL_CERT_ALPHA        = 0.05
LABEL_CERT_MDE_MULT     = 1.0
LABEL_CERT_HOLDOUT      = 0.30
LABEL_CERT_LADDER_STAGE = "PROMOTE"   # the strictest rung: this is the
                                      # target every other number is measured
                                      # against, not a detector knob
LABEL_CERT_VALID_DAYS   = 45          # longer than EDGE_CERT_VALID_DAYS (7):
                                      # a training target should not thrash
                                      # weekly. Finite all the same.
LABEL_CERT_ADMIT_FRAC   = 0.50   # both models admit the SAME COUNT per
                                      # session, so the test isolates WHICH
                                      # trades are chosen, not how many
LABEL_CERT_PATH         = STATE_DIR / "label_certificate.json"

# The DISCRIMINATION floor, used by BOTH the verdict line and the promotion
# gate in core/meta_gbm.py. They used to disagree (report 0.53, gate 0.52):
# on 2026-08-11 the equity meta scored 0.5210, the log said "NO RANKING
# SIGNAL", and it promoted. 0.53 is the number the report already treated as
# the bar. Refusing leaves the brain heuristic-only, which is strictly safer
# than gating on a coin flip and then SIZING on it through Kelly.
# ---- PAYOFF TARGET (v9.9.24). Predict the MAGNITUDE (R = P&L / initial
# risk), not the sign. Sign prediction on index options is a coin at this
# n_eff: 2026-08-11 gave AUC 0.5210 with a detectability floor of 0.587
# (ICC 0.05) to 0.658 (ICC 0.20). Dispersion is forecastable where direction
# is not, which is why rv_forecaster shows skill and the meta does not.
# core/payoff_target.MEASURES predictability before it fits anything and
# refuses to fit when the answer is no.
# ---- DAY PLAN (v9.9.26). ONE thesis per session. The 2026-08-11 ledger is
# the case: 5 of 6 trades were NIFTY puts, the 24400 strike entered THREE
# times and stopped every time, -Rs1297 for the day. That is one thesis
# re-expressed six times, and MAX_CONCURRENT_POSITIONS=1 cannot prevent it —
# it bounds CONCURRENCY, not REPETITION, and COOLDOWN_S=180 is irrelevant
# against re-entries 65 and 121 minutes apart.
# OFF by default: this changes when and how often the system trades, so it
# is A/B-able against the incumbent rather than a one-way door.
# ---- RANGE REGIME (v9.9.27). The regime label EXISTS and is not a gate:
# the funnel has no `range_bound` key. 2026-08-11 NIFTY trended for 2s of
# 18047 (0.01%) and BANKNIFTY for 7s; the system took five directional
# NIFTY put trades that session and stopped out on every one. Buying
# premium for direction in a market that does not travel is a theta
# donation. Lo-MacKinlay variance ratio at several horizons at once, with
# the heteroskedasticity-robust z — intraday index data is violently
# heteroskedastic and the homoskedastic z manufactures trend out of
# volatility clusters.
# OFF by default: whether refusing these entries PAYS is a question for
# core/entry_counterfactual.py on the real tape, not an assumption.
# ---- NEWS INTEL (v9.9.28). An LLM in the entry path is only defensible if
# it is REPLAYABLE: the model runs once pre-open, the scalar is persisted to
# state/news/news_<day>.json, and the live path and the forge replay read
# the SAME bytes. Same discipline macro_gex already follows — compute live,
# archive, replay the archive.
# THREE separate switches, because recording a score, tilting on it, and
# TRAINING on it are three different decisions with three different risks.
# ---- CAS BOOK (v9.9.31). The closing auction is a different market and
# gets its own container: its own capital slice, its own ledger, its own
# slot. Sharing the day session's PositionManager meant a position open at
# 15:15 made the auction unreachable, a CAS trade consumed the day's one
# thesis, and the two P&Ls were pooled so the auction's edge could never be
# measured — which is exactly what CAS_MIN_SESSIONS exists to answer.
CAS_BOOK_ENABLED          = False
CAS_CAPITAL_FRAC          = 0.25   # carved out, never borrowed back
CAS_MAX_ENTRIES           = 1      # the auction is one event

# ---- OVERNIGHT GAP (v9.9.31). GIFT NIFTY is on NSE IX and this account's
# dump is NFO/BFO/MCX only, so `gift_gap_pct` is ABSENT rather than zero
# until a token resolves. The realised 09:15 gap is observable today and is
# what a 09:50 book can actually use.
GAP_FDR_Q                 = 0.10

NEWS_ENABLED              = False  # run the model at all
NEWS_TILT_COMMIT          = False  # let it tilt the day-plan commit ranking
NEWS_MODEL                = "gemma"
# Feeding news into X adds a COLUMN and genuinely IS a different feature
# world, so it must invalidate the caches. But leaving the name out of
# _HASH_EXCLUDE rotates the hash the moment the constant EXISTS — even at
# False — and that costs a full 12-hour rebuild for a switch nobody turned
# on. So the name is excluded, and core/news_intel.meta_feature() refuses
# to serve a column unless FEATURE_WORLD is ALSO bumped. The invalidation
# is enforced in code rather than by hoping the operator remembers.
NEWS_FEED_META            = False
NEWS_FEED_META_WORLD      = "fw_news_v1"   # the FEATURE_WORLD value that
                                           # must be in force before a news
                                           # column may enter X

RANGE_GATE_ENABLED        = False
RANGE_Z_ALPHA             = 1.96   # two-sided 5% on the robust z
RANGE_MIN_AGREE           = 2
RANGE_ASSESS_EVERY_S      = 300    # re-assess this often. The variance ratio
                                   # over ~18k samples is not cheap and the
                                   # verdict does not flip second to second;
                                   # the A/B caches on this grid.      # horizons that must agree. One window is
                                   # an opinion; several agreeing is evidence.

DAYPLAN_ENABLED           = False
DAYPLAN_ANALYSIS_END_HM   = "09:45"  # observe only until here
DAYPLAN_ENTRY_HM          = "09:50"  # commit window opens
DAYPLAN_COMMIT_END_HM     = "10:20"  # ...and closes. Not a single instant:
                                     # a 1s window misses the day on any
                                     # loop jitter or stale quote.
DAYPLAN_REVIEW_HM         = "12:30"  # re-ask the ENTRY question
DAYPLAN_EXIT_HM           = "15:05"  # hard flat; replaces the theta clock
DAYPLAN_CAS_START_HM      = "15:15"  # CAS is a SEPARATE session/book
DAYPLAN_REVERSAL_CONV     = 0.40     # opposite-sign conv this strong = close
DAYPLAN_DECAY_CONV        = 0.15     # |conv| below this = the edge is gone
DAYPLAN_NEWS_WEIGHT       = 0.15     # bounded TILT on the commit score.
                                     # Never a veto, never a driver: an
                                     # unbounded LLM term would make the
                                     # day's thesis unauditable.

# MAX_HOLD_THETA under the day plan. The operator asked for its removal and
# DAYPLAN_ENABLED does it. Recorded here because the evidence points the
# other way: on 2026-08-11 the guillotine cut the day's WORST trade (SENSEX
# 78300CE, -Rs670.50) which had peaked at 304.80 vs entry 300.35 — up 1.5%,
# once — and sat at 270.20 when the clock fired. Removing it does not save
# that trade, it lets it bleed another hour. What makes the removal
# defensible is that DAYPLAN_EXIT_HM bounds the hold and the mid-session
# review can close a dead thesis long before it. The stop and the disaster
# floor are untouched.
DAYPLAN_DISABLE_THETA     = True

PAYOFF_FDR_Q            = 0.10

META_MIN_AUC            = 0.53

META_MIN_SERVE_SPREAD   = 0.05  # q95-q05 of the merged VA probability
META_MIN_SERVE_DISTINCT = 12    # distinct merged values over the probe

SHADOW_ENABLED          = True    # live parallel book (apex_main + commodity)
SHADOW_MARK_S           = 1.0     # seconds between shadow marks
SHADOW_MAX_STALE_S      = 120     # carry a dead quote this long, then NaN
SHADOW_MIN_COVERAGE     = 0.60    # a trade below this live-mark coverage is
                                  # REPORTED but excluded from any verdict
SHADOW_LEDGER_PATH      = LOG_DIR / "shadow_ledger_v9.csv"
SHADOW_SNAPSHOT_MAX_AGE_S = 900   # restore refuses a snapshot older than this
SHADOW_MAX_OPEN         = 64      # per-process guard on the virtual book

# The policy family. Pre-registered here so no policy can be invented after
# its result exists — the registry check in tools/trade_potential.py reads
# this list, not the tool's own locals.
SHADOW_POLICIES = (
    "as_traded",        # the real exit — the baseline, never a candidate
    "hold_to_close", "hold_2x", "hold_3x",
    "trail_10", "trail_20", "trail_30",
    "target_1R", "target_2R", "target_3R",
    "lock_5pct", "lock_10pct",
    "trail20_hold2x",
)

# ---- promotion gate: what it takes for a measured policy to become law
SHADOW_PROMOTE_ENABLED    = True
SHADOW_PROMOTE_MIN_DAYS   = 20    # distinct sessions of paired evidence
SHADOW_PROMOTE_MIN_TRADES = 60
SHADOW_PROMOTE_FDR_Q      = 0.10  # BH across the policy family
SHADOW_PROMOTE_MDE_MULT   = 1.0   # mean Δ₹/day must exceed this × MDE
SHADOW_PROMOTE_MIN_CI_LO  = 0.0   # ...and the 90% CI lower bound must clear
SHADOW_PROMOTE_HOLDOUT    = 0.30  # last 30% of days held out; sign must agree
# ---- ENTRY BAR SWEEP (v9.9.14). The entry-side twin of the shadow book.
# THE CONSTRAINT THAT FRAMES ALL OF IT: MAX_CONCURRENT_POSITIONS=1 with a
# 60-min guillotine and a 180s cooldown caps the session at ~5 trades. The
# bar cannot buy volume; it only chooses WHICH ~5 slots get filled. Every
# report from core.entry_bar_store leads with that arithmetic.
ENTRY_BAR_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                  0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
ENTRY_BAR_ALPHA          = 0.05   # family-wise alpha on the max statistic
ENTRY_BAR_FDR_Q          = 0.10   # BH across the grid (description only)
ENTRY_BAR_MIN_DAYS       = 30     # the grid is nested; it needs more days
ENTRY_BAR_MDE_MULT       = 1.0
ENTRY_BAR_BOOT           = 20000  # Westfall-Young sign-flip draws
ENTRY_BAR_LADDER_STAGE   = "DISCOVER"
# OFF by default, deliberately. The bar is the knob with the shortest path
# to capital; it moves by hand, after reading the evidence, or not at all.
ENTRY_BAR_PROMOTE_ENABLED = False
ENTRY_BAR_PATH           = STATE_DIR / "entry_bar.json"

SHADOW_PROMOTE_LADDER_STAGE = "DISCOVER"  # capability_ladder COMPARATIVE
                                  # stage required. STAGES = BLIND, SCREEN,
                                  # DISCOVER, PROMOTE. A promotion is a
                                  # decision, so it sits one stage above the
                                  # research tools.
SHADOW_POLICY_PATH        = STATE_DIR / "exit_policy.json"

# ----------------------------------------------------------------------------
# v9.5 - TRANCHES 2-4, delivered whole (PROGRAM.md is the binding scope)
LC_WINDOW           = 15      # living-cert rolling window (fills)
LC_MIN_EVENTS       = 10      # ...minimum before a de-arm can trigger
MACRO_TERM_STRUCTURE = True   # tenor-2 ATM probe -> macro_term_v9 (D-J feed)
TERM_BAND_STEPS     = 2       # probe band: ATM +/- this many steps
STRESS_SPLIT_HM     = "11:30" # regime-stitch splice time
RCT_LIMIT_TIMEOUT_S = 2.0     # LIMIT_FIRST arm: secs before market fallback
RCT_MIN_FIT         = 200     # fill-model refuses below this many labels
GRAD_MICRO_CAPITAL  = 25000.0 # micro-live stage capital (one-lot)
GRAD_KELLY_FRAC     = 0.25    # fractional Kelly at scaling stage
GRAD_DD_MAX_PCT     = 10.0    # Grossman-Zhou drawdown throttle ceiling
SPREAD_LIVE_TOKEN   = STATE_DIR / "ARM_LIVE_SPREADS"  # operator lock #4
# serving conviction engine: "meta" (default) = physics heuristic proposes,
# nightly-retrained GBM meta gates/sizes (the validated learner). "sac" =
# legacy frozen SB3 pair (kept for rollback; it OVERRIDES the heuristic).
POLICY_ENGINE       = "meta"
FORGE_TRAIN_SAC     = False   # v10.2 default: the directional thesis was
                              # FALSIFIED — training it nightly bought hours
                              # and no edge. Exams/counterfactual still run;
                              # meta/heur exam/counterfactual/drift still run


# ----------------------------------------------------------------------------
# v9.7.1 — PEAK-CAPTURE EXITS + DISPLACEMENT (2026-07-15 live-evidence fixes)
# ----------------------------------------------------------------------------
# Three coordinated behaviors (core/exit_engine.py, core/displacement.py):
#   1. Winners are ridden while the tape is efficient and exited off a
#      SMOOTHED, dwell-confirmed chandelier ratchet — not banked at the first
#      touch of a runway-anchored target ("jackpot exited early" fix).
#   2. The fly's TP/SL run on the smoothed unwind credit with dwell — the
#      raw four-leg mark was bouncing ±8% of the debit per minute on a still
#      spot; one wide print can no longer be an exit.
#   3. The v10.1 global lock stays, but a fly can be DISPLACED by a strictly
#      better, sustained directional candidate (or a cascade on either
#      index) through core/displacement.py — every grant/refusal is named.
# ALL knobs below are trade-management/portfolio policy: hash-EXCLUDED (the
# feature/label world is untouched, so no re-forge). The FLY exit knobs are
# instead fingerprinted by core/butterfly.fly_knob_hash — changing them fails
# the butterfly certificate closed until tools/butterfly_harness.py re-passes
# (Bailey–LdP: a new exit rule is a new hypothesis).

# ---- long book: efficiency-gated ride (heuristic mode; meta wins if trained)
RIDE_WINNER_ENABLED     = True    # master switch — False restores v9.7 exits
RIDE_ER_MIN             = 0.30    # Kaufman ER over the ride window to extend.
#                                   Calibration: per-SECOND index ER during a
#                                   genuine trend (drift μ, tick vol σ) is
#                                   ≈ μ/E|N(μ,σ)| — a −450pt/20min SENSEX
#                                   break at σ≈1.5/s reads ~0.30, which is
#                                   exactly the bar the entry persistence
#                                   gate already trusts (SIGNAL_PERSIST_ER_
#                                   MIN=0.30). 0.55 is unreachable at 1 s
#                                   granularity and would silently disable
#                                   the ride.
RIDE_ER_WINDOW_S        = 120     # per-second spots in the signed-ER window
RIDE_OPPOSE_CONV        = 0.35    # conviction this far AGAINST kills the ride
RIDE_CONV               = 0.62

# ----------------------------------------------------------------------------
# FLY INTELLIGENCE (v9.7.1) — the pin engine's read, mined for the DIRECTIONAL
# book instead of expressed as a trade. core/fly_intel.py translates a granted
# butterfly gate (positive-gamma + rich-IV + inside-corridor = a pinning /
# mean-reverting regime) into: (1) a logit-space conviction modulation that
# DAMPENS momentum INTO the near wall and BOOSTS a read trading away/with the
# fade; (2) a regime-conditional shrink of the directional target's wall
# runway; (3) a mean-reversion entry hint when spot is pressed at a corridor
# edge. All ADVISORY (scale-never-veto), neutral whenever the gate isn't
# granting, and hash-EXCLUDED (they change WHEN/HOW a directional entry fires
# and where its target sits — no feature or triple-barrier LABEL — so tuning
# never forces a re-forge). Validate on the vault with tools/fly_intel_report.py.
FLY_INTEL_ENABLED         = True   # master switch (False = the fly read is
#                                    telemetry only; no directional effect)
FLY_INTEL_MODULATE_CONV   = True   # product 1: conv dampen/boost
FLY_INTEL_TARGET_CAP      = True   # product 2: shrink the directional runway
FLY_INTEL_REVERT_HINT     = True   # product 3: surface the fade-to-pin hint
FLY_INTEL_MAX_DAMP        = 0.45   # deepest conv dampening INTO a near wall
#                                    (calibrated: a DEEP pin at the wall can
#                                    push a borderline breakout signal below
#                                    the entry bar; mild pins stay tradeable)
FLY_INTEL_MAX_BOOST       = 0.15   # strongest conv boost trading away/with fade
FLY_INTEL_RUNWAY_MULT_FLOOR = 0.45 # tightest target-runway multiplier at max pin
FLY_INTEL_EDGE_STEPS      = 0.75   # spot within this many steps of a wall = edge
FLY_INTEL_GEX_LOG_SCALE   = 0.5    # log10(netGEX/threshold) scale in pin pressure
# ---- polarity (the vault decides the SIGN; ships at 0 = neutral) ----
FLY_INTEL_USE_POLARITY    = True   # read the vault-measured sign artifact
FLY_INTEL_POLARITY_PATH   = None   # default: LOG_DIR/fly_intel_polarity.json
FLY_INTEL_MIN_EVENT_DAYS  = 8      # min granted-days before a sign is trusted
FLY_INTEL_MIN_SECONDS     = 20000  # min granted-seconds before a sign is trusted
# ---- retest-survival entry filter (the BankNifty trap-killer; polarity-agnostic)
FLY_INTEL_RETEST_FILTER   = True   # require a sustained hold at the wall before
#                                    arming a wall-break directional entry
FLY_INTEL_RETEST_ARM_S    = 20.0   # base arm-delay seconds (scaled by pin)

# ============================================================================
# CASCADE EXIT & SMART LOCKOUT (v9.7.1) — core/cascade_exit.py
# The 2026-07-16 SENSEX jackpot that got away: three short-gamma PE triggers,
# two whipsaw-stopped, the third (strongest) blocked by a blunt post-loss
# lockout. Fix: (A) widen the stop for short-gamma cascade violence so the
# retest wick doesn't pick us off; (B) let a STRONGER, still-aligned cascade
# re-trigger bypass the lockout (trend continuation ≠ revenge). All hash-
# excluded (stop WIDTH + lockout TEMPO for cascade trades; no feature/label).
# ---- Part A: cascade-aware stop width ----
CASCADE_STOP_MULT_BASE  = 1.5    # base widening for any short-gamma cascade entry
CASCADE_STOP_K_DEPTH    = 0.35   # extra per unit of log10(|netGEX|/threshold)
CASCADE_STOP_K_Z        = 0.25   # extra per unit of |z| beyond the trigger floor
CASCADE_STOP_MULT_MAX   = 2.5    # hard ceiling (disaster floor still binds under)
CASCADE_Z_FLOOR         = 2.0    # |z| baseline the K_Z term measures beyond
# ---- Part B: smart post-loss lockout ----
SMART_LOCKOUT_ENABLED     = True   # allow strengthening-trend cascade re-entry
LOCKOUT_BYPASS_MAX_PER_DAY = 3     # cap on lockout bypasses per session
LOCKOUT_BYPASS_COOLDOWN_S  = 60.0  # min seconds between bypasses
LOCKOUT_BYPASS_Z_MARGIN    = 0.30  # re-trigger |z| must beat the losing |z| by this

# ============================================================================
# ORDER-FLOW TOXICITY & TRAP DETECTION (v9.7.1) — core/order_flow.py
# VPIN/OFI-based entry filter (Easley-Lopez de Prado-O'Hara 2012; Cont-Kukanov-
# Stoikov 2014). Blocks chasing INTO adverse informed flow / engineered sweeps.
# ADVISORY (raises the bar only). All thresholds SELF-CALIBRATE nightly from the
# vault (tools/toxicity_report.py writes logs/toxicity_calib.json). Hash-
# excluded (entry TEMPO/quality — no feature/label the forge trains on).
TOXICITY_GATE_ENABLED   = True   # master switch for the entry trap filter
TOX_BUCKET_VOLUME       = 5000.0 # volume-clock bucket size (VPIN); calibrated
TOX_NUM_BUCKETS         = 50     # rolling buckets for the VPIN average
TOX_MIN_BUCKETS         = 10     # min buckets before toxicity is trusted
TOX_HIGH                = 0.40   # "high toxicity" flag threshold (calibrated)
TOX_BLOCK               = 0.55   # block a CHASE against flow above this (calib)
TOX_SWING_LOOKBACK_S    = 180    # window for swing-pivot detection
TOX_PIVOT_HOLD_S        = 30     # a level must stand this long to be a pivot
TOX_PIVOT_SETTLE_S      = 5      # exclude the last N ticks from pivot definition
TOX_SWEEP_BUFFER_FRAC   = 0.10   # pierce depth (× strike step) to count a sweep
TOX_VOL_BASE_S          = 300    # rolling volume baseline for absorption z
TOX_ABSORB_VOL_Z        = 2.0    # volume z-score for absorption (calibrated)
TOX_ABSORB_MOVE_FRAC    = 0.25   # …with price move < this × strike step
TOX_SWEEP_FADE_OK       = True   # allow the post-sweep reversal entry

# ---- dynamic levels + calibration loader (core/calibration.py) ----
USE_CALIBRATION         = True   # read logs/calibration.json (hot-reloaded)
CALIBRATION_PATH        = None   # default: LOG_DIR/calibration.json
CALIB_MIN_TICKS         = 20000  # min per-index ticks before a value is trusted
DYNAMIC_LEVELS_ENABLED  = True   # vol-scaled initial stop/target (else fixed %)
DYN_LEVEL_HORIZON_CAP_MIN = 15.0 # cap the √t horizon scaling
DYN_LEVEL_RR            = 1.6     # target = RR × stop (asymmetric payoff)
DYN_SL_MIN              = 0.12    # rails: stop can't be tighter than this
DYN_SL_MAX              = 0.45    #        …or wider than this
DYN_TP_MIN              = 0.18    #        target rails
DYN_TP_MAX              = 1.20
    # |conv| ≥ this WITH the position rides even
#                                   when the tape ER is choppy-but-real: a
#                                   sustained −0.70 read into a wall IS the
#                                   directional edge (a −450-pt breakdown with
#                                   realistic per-second noise nets ER≈0.25–
#                                   0.35 < RIDE_ER_MIN, yet is the jackpot the
#                                   operator must not exit early). Tape-against
#                                   and hard-oppose vetoes still bind.

# ---- long book: peak-capture trail (core/exit_engine.PeakCaptureTrail)
EXIT_MARK_EMA_HL_S      = 6.0     # mark-smoothing half-life (kills bid bounce)
EXIT_SIGMA_PRIOR_FRAC   = 0.02    # σ̂ warm-start = entry × this
EXIT_K_SIGMA            = 3.0     # giveback ≥ k · σ̂ (tick-noise floor)
EXIT_GIVE_FLOOR_FRAC    = 0.12    # …and ≥ this × entry (pullback floor)
EXIT_GIVE_FRAC_TREND    = 0.25    # …and ≥ this × peak gain in momentum tape
EXIT_GIVE_FRAC_CHOP     = 0.15    # tighter in labeled chop (Kaminski–Lo 2014)
EXIT_CONFIRM_S          = 20.0    # ratchet breach must SUSTAIN this long.
#                                   A W-second flush wick keeps the SMOOTHED
#                                   mark under the ratchet for ≈ W + 1.2×EMA
#                                   half-life (recovery lag), so 20 s covers
#                                   the common 5–15 s hunt-wick class; longer
#                                   sweeps are the TrapShield's case (every
#                                   trail fire is shield-gated in the book).
EXIT_HARD_BREACH_MULT   = 2.5     # breach ≥ this × giveback ⇒ fire now
EXIT_STAGNATION_S       = 420.0   # armed + chop + no new HWM this long ⇒ bank
EXIT_THETA_TIGHTEN_MIN_LEFT = 75.0  # tighten the trail inside this many mins
EXIT_THETA_TIGHTEN_MIN  = 0.40    # …down to this multiplier at the close
REVERSAL_CONFIRM_S      = 20.0    # conviction-reversal exit must sustain (0 = legacy one-tick)
MAX_HOLD_RIDE_MULT      = 2.0     # theta guillotine hard cap while riding

# ---- fly book: smoothed / dwell-confirmed exits (fingerprinted in
#      core/butterfly.fly_knob_hash — the certificate's invalidation channel)
SV_FLY_SL_HARD_FRAC     = 0.65    # smoothed cc ≤ debit×(1−this) ⇒ fire NOW
FLY_STOP_CONFIRM_S      = 45.0    # normal stop breach must sustain this long
FLY_TRAIL_ARM_FRAC      = 0.20    # arm the ratchet once cc ≥ debit×(1+this)
FLY_GIVE_FRAC           = 0.30    # giveback ≤ this × peak gain
FLY_GIVE_FLOOR_FRAC     = 0.20    # …and ≥ this × debit (pullback floor)
FLY_K_SIGMA             = 3.0     # …and ≥ k · σ̂ of the four-leg mark noise
FLY_MARK_EMA_HL_S       = 25.0    # 4-leg conservative mark ⇒ heavier smoothing
FLY_SIGMA_PRIOR_FRAC    = 0.05    # σ̂ warm-start (live 2026-07-15: ~6%/min)
FLY_EXIT_CONFIRM_S      = 45.0    # ratchet breach dwell
FLY_HARD_BREACH_MULT    = 2.5     # waterfall escape multiple
FLY_STAGNATION_S        = 900.0   # armed + no new HWM 15 min ⇒ bank (any regime)
FLY_THETA_TIGHTEN_MIN_LEFT = 60.0 # tighten inside this many mins of SV_CLOSE_HM
FLY_THETA_TIGHTEN_MIN   = 0.40
FLY_RIDE_ENABLED        = True    # False ⇒ bank at SV_FLY_TP_FRAC exactly as v9.7
FLY_PIN_LOSS_WING_FRAC  = 1.00    # spot ≥ this × wing width from the body …
FLY_PIN_LOSS_CONFIRM_S  = 60.0    # … sustained this long ⇒ thesis dead, salvage

# ---- displacement governor (core/displacement.py — portfolio policy)
DISP_ENABLED            = True    # master switch — False restores the blind lock
DISP_MAX_PER_DAY        = 2       # rotation is a scalpel; churn eats 4-leg costs
DISP_COOLDOWN_S         = 900.0   # between displacements
FLY_MIN_HOLD_BEFORE_DISP_S = 600.0  # the pin thesis gets its fair chance
DISP_MIN_MINUTES_LEFT   = 45.0    # freed capital must have session to work
DISP_CONV_MARGIN        = 0.10    # candidate must clear entry bar + this
DISP_CONV_MARGIN_STRONG = 0.20    # …when the fly is pinning AND green
DISP_ER_MIN             = 0.30    # signed tape ER must agree at least this
#                                   hard (same per-second calibration as
#                                   RIDE_ER_MIN / SIGNAL_PERSIST_ER_MIN)
DISP_CAND_SUSTAIN_S     = 20.0    # the qualifying read must hold this long
DISP_FLY_PROGRESS_MAX   = 0.60    # never displace a fly ≥60% of the way to TP

# ============================================================================
# CONFIG_HASH — model/feature fingerprint (computed last, sees all constants)
# ============================================================================
# Stamped into every trained artifact (drift reference, meta-labeler) and the
# Edge Certificate, then checked before any of them is trusted. It fingerprints
# ONLY the constants that change those artifacts: how the 19 features are
# computed, the feature-tensor shape, the surface/greeks math, the triple-
# barrier labels (BASE_TP_PCT / BASE_SL_PCT / MAX_HOLD_MINUTES /
# MAX_HOLD_MINUTES_0DTE / EXPIRY_DTE_LT), broker COSTS (labels are
# net-of-cost), and the regime / trap / vol-forecaster thresholds.
#
# Pure OPERATIONAL knobs are excluded so editing them does NOT invalidate a
# reference the forge already trained: capital & sizing, risk budgets, order
# routing / fills / polling, cooldowns & watchdogs, telemetry & logging cadence,
# filesystem paths, credentials, drift ASSESSMENT thresholds (retune the de-arm
# without a re-forge), edge-audit knobs, forge training-infra hyper-params, and
# the persistence-gate tempo knobs (they gate WHEN an entry fires, identically
# in the brain and the forge grader, but change no feature or label).
#
# FAIL-CLOSED: anything not explicitly excluded is fingerprinted, so a newly
# added feature/label constant still invalidates correctly even if nobody
# updates this list. Operational additions you don't want to force a re-forge
# must be named with a path suffix or added to _HASH_EXCLUDE.
_HASH_EXCLUDE = frozenset({
    # v9.7.1 MCX commodity HARVEST knobs — data capture only, touch NOTHING the
    # forge trains on (the forge sees TRADABLE = equity indices). Adding a
    # commodity or changing its step must NOT invalidate the equity model, so
    # these are operational, not part of the feature-world fingerprint.
    "COMMODITIES", "COMMODITY_ORDER", "HARVEST_COMMODITIES",
    "COMMODITY_TRADABLE", "EVENT_GUARD_ENABLED", "EVENT_BLACKOUT_PRE_MIN",
    "EVENT_SETTLE_POST_MIN", "EVENT_OVERRIDES",
    "GEMMA_ANALYST_ENABLED", "GEMMA_MODEL", "OLLAMA_HOST", "GEMMA_NUM_CTX",
    "GEMMA_TIMEOUT_S", "COMMODITY_CALIB_MIN_DAYS",
    "COMMODITY_CALIB_MIN_TICKS", "COMMODITY_HEURISTIC_W",
    "COMMODITY_ENTRY_CONVICTION", "COMMODITY_META_MIN_TRAIN",
    "COMMODITY_FORGE_COOLDOWN_S", "EVENING_CAPTURE_ENABLED",
    # v9.9.13 SHADOW BOOK — measurement only. These must never rotate
    # CONFIG_HASH: the shadow reads the same tape the engine trades and
    # trains nothing. A rotation here would invalidate every raw day cache
    # and re-run the forge for a knob that cannot change a decision.
    # Promotion gates decide WHICH model serves, never what a feature is,
    # so they must not rotate the hash and rebuild every day cache.
    # Day-plan constants are SCHEDULING, not features — they change when a
    # decision is taken, never what the model sees. Hash-excluded so an A/B
    # does not rebuild every day cache.
    # NEWS_ENABLED / NEWS_TILT_COMMIT / NEWS_MODEL are hash-excluded: they
    # change WHETHER a recorded scalar is consulted, not what a feature is.
    # NEWS_FEED_META is deliberately NOT here — feeding news into X changes
    # the feature world and MUST rotate the hash and rebuild the caches.
    "CAS_BOOK_ENABLED", "CAS_CAPITAL_FRAC", "CAS_MAX_ENTRIES", "GAP_FDR_Q",
    "NEWS_ENABLED", "NEWS_TILT_COMMIT", "NEWS_MODEL",
    "NEWS_FEED_META", "NEWS_FEED_META_WORLD",
    "RANGE_GATE_ENABLED", "RANGE_Z_ALPHA", "RANGE_MIN_AGREE",
    "RANGE_ASSESS_EVERY_S",
    "DAYPLAN_ENABLED", "DAYPLAN_ANALYSIS_END_HM", "DAYPLAN_ENTRY_HM",
    "DAYPLAN_COMMIT_END_HM", "DAYPLAN_REVIEW_HM", "DAYPLAN_EXIT_HM",
    "DAYPLAN_CAS_START_HM", "DAYPLAN_REVERSAL_CONV", "DAYPLAN_DECAY_CONV",
    "DAYPLAN_NEWS_WEIGHT", "DAYPLAN_DISABLE_THETA",
    "PAYOFF_FDR_Q", "META_MIN_AUC", "META_MIN_SERVE_SPREAD", "META_MIN_SERVE_DISTINCT",
    "META_LABEL_SPEC", "LABEL_CERT_MIN_SESSIONS", "LABEL_CERT_MIN_TRADES",
    "LABEL_CERT_ALPHA", "LABEL_CERT_MDE_MULT", "LABEL_CERT_HOLDOUT",
    "LABEL_CERT_LADDER_STAGE", "LABEL_CERT_VALID_DAYS", "LABEL_CERT_PATH",
    "LABEL_CERT_ADMIT_FRAC",
    "SHADOW_ENABLED", "SHADOW_MARK_S", "SHADOW_MAX_STALE_S",
    "SHADOW_MIN_COVERAGE", "SHADOW_LEDGER_PATH", "SHADOW_MAX_OPEN",
    "SHADOW_SNAPSHOT_MAX_AGE_S", "SHADOW_POLICIES",
    "SHADOW_PROMOTE_ENABLED", "SHADOW_PROMOTE_MIN_DAYS",
    "SHADOW_PROMOTE_MIN_TRADES", "SHADOW_PROMOTE_FDR_Q",
    "SHADOW_PROMOTE_MDE_MULT", "SHADOW_PROMOTE_MIN_CI_LO",
    "SHADOW_PROMOTE_HOLDOUT", "SHADOW_POLICY_PATH",
    "SHADOW_PROMOTE_LADDER_STAGE",
    # entry-bar sweep: measurement + a hand-operated gate. Same rule as the
    # shadow constants — studying the bar must never rotate the hash and
    # rebuild every day cache.
    "ENTRY_BAR_GRID", "ENTRY_BAR_ALPHA", "ENTRY_BAR_FDR_Q",
    "ENTRY_BAR_MIN_DAYS", "ENTRY_BAR_MDE_MULT", "ENTRY_BAR_BOOT",
    "ENTRY_BAR_LADDER_STAGE", "ENTRY_BAR_PROMOTE_ENABLED",
    "ENTRY_BAR_PATH",
    "CAS_BLACKOUT_ENABLED", "HARD_FLAT_MARGIN_MIN",
    "POST_AUCTION_ENTRIES", "PREOPEN_REFORM_DATE",
    "POST_AUCTION_ENABLED", "POST_AUCTION_MIN_SESSIONS",
    "POST_AUCTION_HOLD_MIN", "POST_AUCTION_FLAT_HM",
    "POST_AUCTION_PREMIUM_LEVERAGE", "POST_AUCTION_MIN_EDGE_MULT",
    "QUOTE_API_DEADLINE_S", "MARK_STALE_WARN_S", "DYN_TP_EXTEND_MULT",
    "CAS_CAPTURE_ENABLED", "CAS_PREPRINT_ENABLED", "CAS_MIN_SESSIONS",
    "CAS_ALPHA", "CAS_MAX_QUALITY_PENALTY",
    "NO_ENTRY_BEFORE_CLOSE_MIN",
    "COMMODITY_SESSION_OPEN", "SUPERVISOR_TABS", "RISK_STATE_PERSIST",
    "COMMODITY_CAPITAL_FRAC", "COMMODITY_CAPITAL_RS", "COMMODITY_NO_ENTRY_BEFORE_CLOSE_MIN",
    "COMMODITY_FLATTEN_BEFORE_CLOSE_MIN", "FEED_SILENT_WARN_S",
    "LOG_ASCII",
    "META_MIN_BSS", "META_MIN_POSITIVES", "META_MIN_OOF_SPREAD",
    "XDIM_REMIND_S",
    "META_CROSS_INDEX", "PARALLEL_DAY_WORKERS",
    "PARALLEL_RAM_RESERVE_GB", "PARALLEL_RAM_PER_WORKER_GB",
    "PARALLEL_RAM_WORKSET_MULT", "PARALLEL_RAM_AWARE",
    "PARALLEL_MIN_WORKERS", "PROMOTION_MIN_TRADES",
    "POSITION_SNAPSHOT_MAX_AGE_S", "NET_GUARD_ENABLED",
    "NET_GUARD_HOSTS", "DNS_CACHE_TTL_S", "DNS_TIMEOUT_S",
    "SOCKET_DEFAULT_TIMEOUT_S", "LOOP_STALL_WARN_S",
    "DISCOVERY_ENABLED", "LADDER_POWER", "LADDER_ALPHA",
    "LADDER_SCREEN_MDE", "LADDER_DISCOVER_MDE", "DISCOVERY_FDR_Q",
    "HORIZON_CANDIDATES", "HORIZON_MIN_DAYS", "HORIZON_BOOT",
    "HORIZON_AUTO_ADOPT", "HORIZON_ADOPT_MIN_NIGHTS",
    "HORIZON_ADOPT_RANGE", "FEATURE_MIN_GROUP_TESTS",
    "META_MIN_AUC",
    "FAST_LANE_DEFER_WHILE_RISING", "FAST_LANE_RUN_GRACE_S",
    "CASCADE_MAX_FLIP_DIST_PCT",
    # v9.8 meta-forge engine knobs (trainer choice — model files carry their
    # own provenance; these must not fingerprint the feature world)
    # v9.9 meta-gate v3 knobs (gate policy — model files carry provenance;
    # these must not fingerprint the feature world)
    "META_GATE_MODE", "META_EV_MARGIN", "META_ACI_GAMMA", "META_ACI_MAX",
    "META_VA_MAX_CAL", "META_FEAT_WINDOW", "META_FEAT_FROZEN_MIN",
    "META_PROBE_ENABLED", "META_PROBE_MAX_PER_DAY",
    "META_PROBE_MAX_DAY_RISK_PCT", "COMMODITY_META_GATE",
    "META_ENGINE", "META_EMBARGO_DAYS", "META_GBM_LEAVES", "META_GBM_LR",
    "META_USE_PLO", "META_PLO_MIN_N", "FUT_SYMBOLS",
    "VIX_TOKEN",
    "META_GBM_ROUNDS", "META_GBM_MINCHILD",
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
    # cooldowns / lockouts / throttles / halts / data-watchdog / persistence
    # tempo (gate orders; identical paper↔live↔forge-grader; do not change
    # per-second feature/label generation)
    "COOLDOWN_S", "DIRECTION_LOCKOUT_S", "ENTRY_ATTEMPT_THROTTLE_S",
    "MAX_ORDER_REJECTS", "DATA_STALE_BLOCK_S", "DATA_STALE_FLATTEN_S",
    "MACRO_STALE_S", "SIGNAL_PERSIST_ENABLED", "SIGNAL_PERSIST_WINDOW_S",
    "SIGNAL_PERSIST_MIN_SAMPLES",
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
    "DIAG_WRITE_EVERY_S",
    # edge-certificate audit knobs (the cert is re-audited nightly from ledger)
    "EDGE_MIN_TRADES", "EDGE_MIN_DAYS", "EDGE_BOOTSTRAP_N", "EDGE_CI",
    "EDGE_CERT_VALID_DAYS",
    # forge training-infra hyper-params (change HOW models train and are
    # examined, not the data / labels / features / reference distribution)
    "FORGE_BANDIT_BATCH", "FORGE_BANDIT_WARMUP_EPOCHS", "FORGE_BANDIT_EVAL_ROWS",
    "FORGE_MIN_TRADE_RATE", "FORGE_BANDIT_MAX_EPOCHS", "FORGE_BANDIT_PATIENCE",
    "FORGE_BANDIT_REWARD_SCALE", "SAC_BATCH",
    "FORGE_ACT_GATE_TRAIN", "FORGE_ACT_GATE_EVAL", "FORGE_PROMOTE_MARGIN_RS",
    "FORGE_MAX_TRAIN_DAYS", "FORGE_WF_FOLDS", "FORGE_PARALLEL_WORKERS",
    # drift ASSESSMENT thresholds (retune de-arm sensitivity with NO re-forge;
    # the reference-CONSTRUCTION knobs DRIFT_BINS / DRIFT_REF_MAX_SAMPLES /
    # DRIFT_KEY_FEATURES are deliberately NOT here — they stay in the hash)
    "DRIFT_PSI_MODERATE", "DRIFT_PSI_SIGNIFICANT", "DRIFT_KS_SIGNIFICANT",
    "DRIFT_WATCH_FRAC", "DRIFT_DEARM_FRAC", "DRIFT_MIN_LIVE_SAMPLES",
    # misc data-source / backfill
    "BACKFILL_DAYS", "VIX_SYMBOL",
    # gamma-cascade module (trigger/tempo knobs change no feature or label of
    # the main system; the cascade CERTIFICATE fingerprints them instead, so
    # tuning invalidates the cert — never forces a re-forge)
    "CASCADE_LIVE_ENABLED", "CASCADE_PAPER_EXPLORE", "CASCADE_VEL_WINDOW_S",
    "CASCADE_VEL_Z",
    "CASCADE_VOL_LOOKBACK_S", "CASCADE_VOL_MIN_N", "CASCADE_NET_GEX_MAX",
    "CASCADE_HYST_MULT", "CASCADE_COOLDOWN_S", "CASCADE_MAX_EVENTS_DAY",
    "CASCADE_ENTRY_CONV", "CASCADE_CERT_MIN_EVENTS", "CASCADE_CERT_MIN_DAYS",
    "CASCADE_CERT_CI",
    # short-vol engine (v9.3): trigger/exit knobs change no feature or label
    # of the main system; the shortvol CERTIFICATE fingerprints them all —
    # tuning invalidates the cert, never forces a re-forge
    "SHORTVOL_ENABLED", "SHORTVOL_PAPER_EXPLORE", "FLY_TRADING_ENABLED",
    "SV_IVRANK_MIN",
    "SV_NET_GEX_MIN", "SV_WALL_BUFFER_STEPS", "SV_CORRIDOR_MIN_STEPS",
    "SV_DTE_MIN", "SV_DTE_MAX", "SV_AFTER_HM", "SV_WIDTH_STEPS",
    "SV_MIN_CREDIT_FRAC", "SV_TP_FRAC", "SV_SL_CREDIT_MULT", "SV_TOUCH_EXIT",
    "SV_FLY_WING_STEPS", "SV_FLY_MIN_DEBIT_FRAC", "SV_FLY_MAX_DEBIT_FRAC",
    "SV_FLY_TP_FRAC", "SV_FLY_SL_FRAC",
    "SV_CLOSE_HM", "SV_ATTEMPT_THROTTLE_S", "SV_POP_HAIRCUT", "SV_RISK_PCT",
    "SEQ_WINDOW_S", "SEQ_HOLDOUT_SESSIONS", "SEQ_CONTROL_RUNS",
    "SEQ_MC_PASSES", "SEQ_N_TRIALS", "SEQ_MIN_DSR",
    "SEQ_MODEL_ENABLED",
    "EPISODE_ALPHA", "EPISODE_MIN_STABILITY",
    "EPISODE_RANKER_ENABLED",
    "CF_NEAR_MISS", "CF_MAX_PER_GATE",
    "LC_WINDOW", "LC_MIN_EVENTS", "MACRO_TERM_STRUCTURE", "TERM_BAND_STEPS",
    "STRESS_SPLIT_HM", "RCT_LIMIT_TIMEOUT_S", "RCT_MIN_FIT",
    "GRAD_MICRO_CAPITAL", "GRAD_KELLY_FRAC", "GRAD_DD_MAX_PCT",
    "FORGE_TRAIN_SAC", "POLICY_ENGINE",
    "HARNESS_MAX_DAYS", "META_TRAIN_MAX_DAYS",
    "SV_CERT_MIN_EVENTS", "SV_CERT_MIN_DAYS", "SV_CERT_CI",
    # v9.7.1 peak-capture exits + displacement: trade-management / portfolio
    # policy only — no feature or label of the main system changes, so none
    # of these force a re-forge. The FLY exit knobs invalidate through
    # core/butterfly.fly_knob_hash (certificate channel) instead; the
    # displacement knobs are logged verbatim on every DISPLACE ledger row.
    "RIDE_WINNER_ENABLED", "RIDE_ER_MIN", "RIDE_ER_WINDOW_S",
    "RIDE_OPPOSE_CONV", "RIDE_CONV",
    "FLY_INTEL_ENABLED", "FLY_INTEL_MODULATE_CONV", "FLY_INTEL_TARGET_CAP",
    "FLY_INTEL_REVERT_HINT", "FLY_INTEL_MAX_DAMP", "FLY_INTEL_MAX_BOOST",
    "FLY_INTEL_RUNWAY_MULT_FLOOR", "FLY_INTEL_EDGE_STEPS", "FLY_INTEL_GEX_LOG_SCALE",
    "FLY_INTEL_USE_POLARITY", "FLY_INTEL_POLARITY_PATH", "FLY_INTEL_MIN_EVENT_DAYS",
    "FLY_INTEL_MIN_SECONDS", "FLY_INTEL_RETEST_FILTER", "FLY_INTEL_RETEST_ARM_S",
    "CASCADE_STOP_MULT_BASE", "CASCADE_STOP_K_DEPTH", "CASCADE_STOP_K_Z",
    "CASCADE_STOP_MULT_MAX", "CASCADE_Z_FLOOR", "SMART_LOCKOUT_ENABLED",
    "LOCKOUT_BYPASS_MAX_PER_DAY", "LOCKOUT_BYPASS_COOLDOWN_S", "LOCKOUT_BYPASS_Z_MARGIN",
    "TOXICITY_GATE_ENABLED", "TOX_BUCKET_VOLUME", "TOX_NUM_BUCKETS",
    "TOX_MIN_BUCKETS", "TOX_HIGH", "TOX_BLOCK", "TOX_SWING_LOOKBACK_S",
    "TOX_PIVOT_HOLD_S", "TOX_PIVOT_SETTLE_S", "TOX_SWEEP_BUFFER_FRAC",
    "TOX_VOL_BASE_S", "TOX_ABSORB_VOL_Z", "TOX_ABSORB_MOVE_FRAC",
    "TOX_SWEEP_FADE_OK",
    "USE_CALIBRATION", "CALIBRATION_PATH", "CALIB_MIN_TICKS",
    "DYNAMIC_LEVELS_ENABLED", "DYN_LEVEL_HORIZON_CAP_MIN", "DYN_LEVEL_RR",
    "DYN_SL_MIN", "DYN_SL_MAX", "DYN_TP_MIN", "DYN_TP_MAX",
    "EXIT_MARK_EMA_HL_S", "EXIT_SIGMA_PRIOR_FRAC", "EXIT_K_SIGMA",
    "EXIT_GIVE_FLOOR_FRAC", "EXIT_GIVE_FRAC_TREND", "EXIT_GIVE_FRAC_CHOP",
    "EXIT_CONFIRM_S", "EXIT_HARD_BREACH_MULT", "EXIT_STAGNATION_S",
    "EXIT_THETA_TIGHTEN_MIN_LEFT", "EXIT_THETA_TIGHTEN_MIN",
    "REVERSAL_CONFIRM_S", "MAX_HOLD_RIDE_MULT",
    "FAST_LANE_ENABLED", "FAST_LANE_CONVICTION", "FAST_LANE_MIN_HOLD_S",
    "FAST_LANE_MAX_HOLD_S", "FAST_LANE_TP_PCT", "FAST_LANE_ARM_PCT",
    "LOSS_STREAK_HALT",
    "LOCKOUT_BYPASS_REQUIRE_RECLAIM",
    "SV_FLY_SL_HARD_FRAC", "FLY_STOP_CONFIRM_S", "FLY_TRAIL_ARM_FRAC",
    "FLY_GIVE_FRAC", "FLY_GIVE_FLOOR_FRAC", "FLY_K_SIGMA",
    "FLY_MARK_EMA_HL_S", "FLY_SIGMA_PRIOR_FRAC", "FLY_EXIT_CONFIRM_S",
    "FLY_HARD_BREACH_MULT", "FLY_STAGNATION_S",
    "FLY_THETA_TIGHTEN_MIN_LEFT", "FLY_THETA_TIGHTEN_MIN",
    "FLY_RIDE_ENABLED", "FLY_PIN_LOSS_WING_FRAC", "FLY_PIN_LOSS_CONFIRM_S",
    "DISP_ENABLED", "DISP_MAX_PER_DAY", "DISP_COOLDOWN_S",
    "FLY_MIN_HOLD_BEFORE_DISP_S", "DISP_MIN_MINUTES_LEFT",
    "DISP_CONV_MARGIN", "DISP_CONV_MARGIN_STRONG", "DISP_ER_MIN",
    "DISP_CAND_SUSTAIN_S", "DISP_FLY_PROGRESS_MAX",
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