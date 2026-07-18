"""
APEX OMNI v9.1 — LIVE BRAIN (audit §8 rebuilt; v9.1 shared decision path)
=========================================================================
Execution truth, one feature dialect, calibrated conviction, and a watchdog:

  * Startup reconciles against kite.positions() (live) — a restart can never
    hallucinate a flat book.
  * Reads the harvester's ring buffer WITH its age; entries are blocked past
    DATA_STALE_BLOCK_S and open positions are emergency-flattened past
    DATA_STALE_FLATTEN_S (the v8 brain would happily trade a frozen feed).
  * Loads the forge's model + VecNormalize as ONE manifest pair, refuses a
    dim-mismatched pair, hot-swaps only when the manifest changes (both
    halves together — no more new-weights-on-stale-statistics window).
  * Conviction → win probability via the analyzer's calibration table; the
    invented (|a|+1)/2 mapping is gone. Uncalibrated conviction is treated
    as barely-a-coin and sized accordingly.
  * VPIN / gamma-flip overrides are ADVISORY (small logit nudges through
    bayesian_signal_fusion) — nothing force-writes ±1.0 any more.
  * Entries walk the affordability hierarchy; positions are managed by the
    PositionManager + TrapShield every tick. If no torch/SB3 is installed the
    brain runs a transparent physics-only HeuristicPolicy so paper trading
    (and calibration-building) never blocks on the ML stack.

v9.1 (audit fixes — see core/decision.py for the full rationale):
  * EVERY decision stage — advisory shock, fusion, regime scaling, effective
    bar, win-probability blend, persistence, the entry gate — now lives in
    core/decision.py, imported here AND by the nightly forge. The brain no
    longer has a private dialect the trainer can't hear.
  * Regime multiplier applied in LOGIT space: CHOP/VOL_CRUSH dampen instead
    of arithmetically vetoing (the old ×0.70 on a tanh could never clear 0.70).
  * Persistence window is WALL-CLOCK (SIGNAL_PERSIST_WINDOW_S), not "last 4
    loop iterations ≈ 0.8 s".
  * GATE FUNNEL diagnostics: every tick, every tradable index, exactly one
    outcome is counted (which gate blocked, or entered). Heartbeat shows the
    running funnel; logs/brain_report_<date>.json carries the full table plus
    conviction percentiles, regime time-share and P(win) stats — "why is it
    flat" is now a file.

Run:  python apex_main_v9.py

v9.1.2 (2026-07-06 — the last parity gap): DECISIONS run once per RING SECOND.
The harvester writes the ring at 1 Hz; the old ~5 Hz loop re-pushed the same
snapshot ~5×/s, quintuple-counting vol_delta into OFI/VPIN/Hawkes/dealer-inv
(live conv 0.86–0.95 vs the forge replay's ~0.5 ceiling on 2026-07-03) and
inferring on a frame cadence the net never trained on. Now: push/policy/shock/
regime/persistence/gates/entry at exactly the forge grader's cadence — while
the 0.2 s loop keeps exits, fills, trap checks, the stale watchdog and trade
tracking at full tempo (management reads the ≤1 s-old cached decision state).
Bonus fixes for free: drift.observe samples at the reference population's
cadence, and the VIX 5-minute spike window is finally 5 minutes (the 5 Hz
deque had silently shrunk it to ~80 s).
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

import config
from apex_ipc_core import BinaryRingBuffer
from core.market_state import StateBuilder
from core.risk_manager import RiskGovernor
from core.execution_engine import ExecutionEngine
from core.position_manager import PositionManager, LegQuote, TickContext
from core.instruments import LiveMapper
from core.heuristic_policy import HeuristicPolicy
from core.quant_core import black76_greeks
from core import regime_classifier as regime_mod
from core import decision as D
from core.diagnostics import DailyReport, Reservoir
from core.gamma_nowcast import GammaNowcast
from core import cascade as CS
from core import shortvol as SVOL
from core import butterfly as BFLY
from core.displacement import DisplacementGovernor
from core.exit_engine import signed_efficiency
from core import fly_intel as FI
from core import order_flow as OF
from core.calibration import tox_thresholds, bucket_volume
from core.bocpd import BOCPD
from core.dealer_flow import DealerFlow
from core import rv_forecaster as RVF
from core import book as BOOK

log = logging.getLogger("brain")

try:
    from kiteconnect import KiteConnect
    HAVE_KITE = True
except Exception:                                       # noqa: BLE001
    HAVE_KITE = False



# ----------------------------------------------------------------- policy
class PolicyLoader:
    def __init__(self):
        self.kind = "heuristic"
        self.model = None
        self.vec = None
        self.manifest_ts = 0.0
        self._try_load()

    def _try_load(self):
        if not config.MODEL_MANIFEST.exists():
            return
        ts = config.MODEL_MANIFEST.stat().st_mtime
        if ts <= self.manifest_ts:
            return
        try:
            man = json.loads(config.MODEL_MANIFEST.read_text())
            if int(man.get("obs_dim", -1)) != config.OBS_DIM:
                log.error("manifest obs_dim %s ≠ %d — REFUSING pair",
                          man.get("obs_dim"), config.OBS_DIM)
                return
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import VecNormalize
            import gymnasium as gym
            from stable_baselines3.common.vec_env import DummyVecEnv

            class _Dummy(gym.Env):
                observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                   (config.OBS_DIM,), np.float32)
                action_space = gym.spaces.Box(-1, 1, (config.ACTION_DIM,),
                                              np.float32)
                def reset(self, *, seed=None, options=None):
                    return np.zeros(config.OBS_DIM, np.float32), {}
                def step(self, a):
                    return np.zeros(config.OBS_DIM, np.float32), 0, True, False, {}

            vec = VecNormalize.load(str(config.MODEL_DIR / man["norm"]),
                                    DummyVecEnv([_Dummy]))
            vec.training = False           # ★ frozen stats, kept from v8
            vec.norm_reward = False
            model = SAC.load(str(config.MODEL_DIR / man["model"]))
            self.model, self.vec = model, vec
            self.kind = man["version"]
            self.manifest_ts = ts
            log.info("policy pair loaded: %s (val ₹%.2f on %s)",
                     man["version"], man.get("val_score", float("nan")),
                     man.get("val_day"))
        except Exception as e:                           # noqa: BLE001
            log.warning("SB3 pair unavailable (%s) — heuristic stays live", e)

    def conviction(self, obs: np.ndarray | None,
                   frame: np.ndarray) -> np.ndarray:
        self._try_load()                                  # hot-swap check
        if self.model is not None and obs is not None:
            o = self.vec.normalize_obs(obs[None, :])
            a, _ = self.model.predict(o, deterministic=True)
            return a[0]
        return HeuristicPolicy().predict(frame)


# ----------------------------------------------------------------- helpers
_meta_cache = {"ts": 0.0, "m": None}
def load_meta():
    """Meta-labeler (size model). Cached on mtime; numpy arrays prepped."""
    try:
        mt = config.META_MODEL_PATH.stat().st_mtime
    except FileNotFoundError:
        return None
    if mt > _meta_cache["ts"]:
        try:
            j = json.loads(config.META_MODEL_PATH.read_text())
            j["w"] = np.array(j["w"], np.float32)
            j["mu"] = np.array(j["mu"], np.float32)
            j["sd"] = np.array(j["sd"], np.float32)
            _meta_cache.update(ts=mt, m=j)
            logging.getLogger("brain").info(
                "meta size-model loaded: n=%s holdout_acc=%s",
                j.get("n"), j.get("holdout_acc"))
        except Exception:                                  # noqa: BLE001
            return _meta_cache["m"]
    return _meta_cache["m"]


def load_calibration() -> dict:
    if config.CALIBRATION_TABLE.exists():
        try:
            return json.loads(config.CALIBRATION_TABLE.read_text())
        except Exception:                                 # noqa: BLE001
            pass
    return {}

def read_macro(idx: str) -> dict | None:
    p = Path(config.MACRO_STATE_TMPL.format(idx=idx))
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return None
    if time.time() - float(j.get("ts", 0)) > config.MACRO_STALE_S:
        return None                                       # advisory dead, not fatal
    return j


class QuoteCache:
    def __init__(self, kite):
        self.kite = kite
        self.cache: dict[str, tuple[float, dict]] = {}
        self.last_call = 0.0

    def get(self, items: list[tuple[str, str]]) -> dict:
        """items: [(exchange, symbol)] → {symbol: quote}; ≤1 req/s."""
        keys = [f"{e}:{s}" for e, s in items]
        fresh = {k: v for k, (t, v) in self.cache.items()
                 if time.time() - t < config.QUOTE_CACHE_FRESH_S and k in keys}
        missing = [k for k in keys if k not in fresh]
        if missing and self.kite and time.time() - self.last_call >= 1.05:
            try:
                q = self.kite.quote(missing)
                self.last_call = time.time()
                for k, v in q.items():
                    self.cache[k] = (time.time(), v)
                    fresh[k] = v
            except Exception as e:                        # noqa: BLE001
                log.warning("quote(): %s", e)
        return {k.split(":", 1)[1]: v for k, v in fresh.items()}


def _surface_atm_iv(builder, idx, expiry, T, mac):
    """SVI ATM IV once the (idx,expiry) slice is really fit; the trusted macro
    Newton ATM IV as fallback before that. The unfit DEFAULT surface curve is
    a fixed total variance sane only near 30–45 d — for weeklies it explodes
    (~138% at 2 d), which would poison leg-delta selection at the open."""
    if builder.surface.has_fit(idx, expiry):
        return builder.surface.atm_iv(idx, expiry, T)
    nwt = (mac or {}).get("atm_iv")
    return float(nwt) if nwt else builder.surface.atm_iv(idx, expiry, T)


def main():
    config.setup_logging("brain")
    kite = None
    if HAVE_KITE and config.KITE_API_KEY and config.KITE_ACCESS_TOKEN:
        kite = KiteConnect(api_key=config.KITE_API_KEY)
        kite.set_access_token(config.KITE_ACCESS_TOKEN)
    mapper = LiveMapper(kite) if kite else None
    risk = RiskGovernor(kite=kite)
    ring = BinaryRingBuffer()
    builder = StateBuilder()
    engine = ExecutionEngine(kite=kite, quote_fn=lambda tok: {})
    qc = QuoteCache(kite)
    policy = PolicyLoader()
    cal = load_calibration()
    from core.drift_monitor import DriftMonitor
    drift = DriftMonitor()
    drift_grade = "NO_REF"
    pms = {i: PositionManager(i, risk, engine) for i in config.TRADABLE}
    broker_open = engine.reconcile()
    if broker_open:
        risk.kill("broker shows pre-existing open positions — square off "
                  "manually, then restart")
    log.info("brain up | capital ₹%.0f | mode %s | policy %s",
             risk.start_capital,
             "LIVE" if config.live_fire_armed() else "PAPER", policy.kind)

    last_spot: dict[str, float] = {}
    spread_ew: dict[str, float] = {}
    last_try: dict[str, float] = {}
    from collections import deque
    vix_hist: deque = deque(maxlen=400)
    levels: dict[str, dict] = {}
    open_px: dict[str, float] = {}
    p945: dict[str, float] = {}
    regime_now: dict[str, object] = {}                 # last Regime per index
    last_track: dict[str, float] = {}                  # trade-tracking cadence
    persist = {i: D.PersistenceTracker() for i in config.TRADABLE}
    from core.quant_core import EWMAVol
    rvol: dict[str, EWMAVol] = {i: EWMAVol() for i in config.INDICES}
    realized_vol_ann: dict[str, float] = {}
    spot_secs: dict[str, deque] = {i: deque(maxlen=1800)
                                   for i in config.TRADABLE}
    # ---- v9.1.2 DECISION CADENCE state: the harvester writes the ring once
    # per second, so decisions run once per RING SECOND (the forge's exact
    # cadence); the ~5 Hz loop keeps exits/fills/trap/watchdog at full tempo.
    # Cached 1 Hz products the management path reads between decisions:
    last_decision_sec = -1
    obs = None
    frame = None
    actions = np.zeros(config.ACTION_DIM, np.float32)
    last_conv: dict[str, float] = {}       # post-regime conv, ≤1 s old
    last_wp_hold: dict[str, float | None] = {}
    # v9.7.1: signed Kaufman ER of the recent spot path, refreshed 1 Hz — the
    # ride gate's tape evidence (None until enough history) + the
    # DisplacementGovernor that weighs an incoming read against a held fly.
    last_tape_er: dict[str, float | None] = {}
    last_fly_intel: dict[str, object] = {}   # the fly's pinning read, mined
    #                                          into a directional modulation +
    #                                          boundary map (core/fly_intel.py)
    wall_touch_since: dict[str, float | None] = {}   # per-index wall-clock t0
    #                                          since spot first reached the near
    #                                          wall — drives the retest-survival
    #                                          entry gate (BankNifty trap-killer)
    disp = DisplacementGovernor()
    from core.cascade_exit import SmartLockout
    smart_lock = SmartLockout()          # v9.7.1: strengthening-trend cascade
    #                                      re-entry bypasses the blunt post-loss
    #                                      lockout (2026-07-16 jackpot fix)
    cascade_pos_z: dict[str, float | None] = {}   # z of the cascade trigger that
    #                                      opened the CURRENT position (None if
    #                                      the open position isn't a cascade one)
    tox_engine = {i: OF.OrderFlowToxicity(i) for i in config.TRADABLE}
    last_tox: dict[str, object] = {}     # latest TrapVerdict per index, for the
    #                                      entry gate + heartbeat (VPIN/OFI trap)
    last_mac: dict[str, dict | None] = {}
    # ---- v9.1 diagnostics: gate funnel + daily report -----------------------
    # v9.2: resume=True — a mid-session restart merges into the day's report;
    # live counters are RE-SEEDED from it below (reservoir percentiles are
    # restart-local by nature; everything countable survives the bounce).
    funnel = D.GateFunnel(config.TRADABLE)
    report = DailyReport("brain", resume=True)
    conv_res = {i: Reservoir(20_000) for i in config.TRADABLE}   # |conv| stream
    wp_res = {i: Reservoir(20_000) for i in config.TRADABLE}     # gated P(win)
    regime_share: dict[str, dict] = {i: {} for i in config.TRADABLE}
    _prev = report.d
    if _prev.get("gate_funnel"):
        for _i, _sec in _prev["gate_funnel"].items():
            if _i in funnel.counts:
                for _g, _n in (_sec.get("gates") or {}).items():
                    funnel.counts[_i][_g] = int(_n)
                funnel.block_detail[_i] = {
                    k: int(v) for k, v in
                    (_sec.get("top_block_reasons") or {}).items()}
        for _i, _rs in (_prev.get("regime_share_s") or {}).items():
            if _i in regime_share:
                regime_share[_i].update({k: int(v) for k, v in _rs.items()})
        report.d["resume_note"] = ("counters resumed from prior same-day "
                                   "report; conviction/winprob reservoirs are "
                                   "restart-local; block-reason detail resumes "
                                   "top-8 only")
        log.info("brain report RESUMED — funnel/regime counters carried "
                 "across restart (%d restart(s) today)",
                 len(_prev.get("restarts") or []))
    last_report = time.time()

    # ---- v9.2 GAMMA-CASCADE: 1 Hz flip nowcast + structural detector --------
    # Telemetry always; ENTRIES only under a valid harness certificate
    # (core/cascade.load_certificate — fail-closed, knob-hash-stamped). The
    # detector is the SAME bytes tools/cascade_harness.py graded.
    nowcasts = {i: GammaNowcast(i) for i in config.TRADABLE}
    detectors = {i: CS.CascadeDetector(i) for i in config.TRADABLE}
    flip_now: dict[str, object] = {}
    cascade_cert = CS.load_certificate()
    casc_mode = CS.cascade_mode(cascade_cert)      # certified|paper-explore|telemetry
    cascade_fired = {i: 0 for i in config.TRADABLE}
    cascade_entered = {i: 0 for i in config.TRADABLE}
    report.d.setdefault("cascade", {}).setdefault("events", [])
    report.d["cascade"]["mode"] = casc_mode
    if casc_mode == "certified":
        log.info("CASCADE ARMED (CERTIFIED) — cert ok (n=%s events, mean ₹%s, "
                 "CI lo ₹%s, win_lo %.0f%%) knob %s",
                 cascade_cert.get("n_events"), cascade_cert.get("mean_pnl"),
                 cascade_cert.get("ci_lo"),
                 100 * float(cascade_cert.get("win_rate_lo", 0)),
                 cascade_cert.get("knob_hash"))
    elif casc_mode == "paper-explore":
        log.info("CASCADE ARMED (PAPER-EXPLORE) — no certificate yet; entries "
                 "run in PAPER ONLY to accrue forward out-of-sample evidence "
                 "(the harness blends realized paper fills into the cert; "
                 "live cascade stays locked behind cert + the four locks)")
    else:
        log.info("cascade detector: TELEMETRY-ONLY (CASCADE_LIVE_ENABLED "
                 "and/or paper-explore off)")

    # ---- v9.3 SHORT-VOL ENGINE: VRP credit spreads, cascade-vetoed ----------
    # Sells defined-risk verticals at the tested wall ONLY under +gamma
    # pinning with rich IV rank; the cascade machinery is the structural
    # crash-veto (sell the calm, own the storm — zero overlap by
    # construction). Staged like cascade: telemetry → paper-explore forward
    # evidence → certificate. v9.3.0 execution ceiling is PAPER by design:
    # live spread ROUTING stays unbuilt until a certificate justifies it.
    sv_cert = SVOL.load_certificate()
    sv_mode = SVOL.shortvol_mode(sv_cert)
    # v9.7: shortvol's SIGNAL (evaluate_gate) still fires; the traded
    # INSTRUMENT is now the buy-only long butterfly. FlyBook mirrors
    # SpreadBook's surface (try_open/manage/mark/pos/closed_today).
    flybook = BFLY.FlyBook(risk=risk)
    sv_gate_block: dict[str, dict] = {i: {} for i in config.TRADABLE}
    sv_blocked_now: dict[str, bool] = {i: False for i in config.TRADABLE}
    report.d.setdefault("shortvol", {}).setdefault("events", [])
    report.d["shortvol"]["mode"] = sv_mode
    report.d["shortvol"]["fly_trading_enabled"] = bool(
        getattr(config, "FLY_TRADING_ENABLED", False))
    if not getattr(config, "FLY_TRADING_ENABLED", False):
        log.info("FLY: TELEMETRY-ONLY by operator config "
                 "(FLY_TRADING_ENABLED=False) — the butterfly gate evaluates "
                 "every second and feeds core/fly_intel (its read sharpens "
                 "DIRECTIONAL entries: dampen breakouts INTO a wall, boost "
                 "fades toward the pin, cap the target runway), but NO fly is "
                 "ever opened — not paper, not live. The global lock is never "
                 "held by a fly; directional capital stays free. (cert mode "
                 "would be '%s' if trading were on.)", sv_mode)
    elif sv_mode == "paper-explore":
        log.info("SHORTVOL ARMED (PAPER-EXPLORE) — VRP credit spreads: short "
                 "the tested wall, long one step out; +gamma corridor + rich "
                 "IV only; cascade-vetoed; paper fills feed the certificate "
                 "via the forward log")
    elif sv_mode == "certified":
        log.info("SHORTVOL certified (n=%s, mean ₹%s) — NOTE: v9.3.0 live "
                 "ROUTING is unbuilt by design; execution remains paper",
                 sv_cert.get("n_events"), sv_cert.get("mean_pnl"))
    else:
        log.info("shortvol: TELEMETRY-ONLY (gate stats recorded; set "
                 "SHORTVOL_PAPER_EXPLORE=True to accrue forward evidence)")
    _svtok = f"0c {sv_mode}"

    # ---- v9.4 TELEMETRY ORGANS (Pillars 2+3): dealer-flow vector + rv̂ -----
    # Charm/vanna/pin from the radar's own profile; HAR remaining-day vol and
    # the measured VRP spread when a fitted model exists. Telemetry + report
    # ONLY — no certified gate consumes these until their own trials pass
    # (PROGRAM.md; the rv skill certificate is the forecaster's lock).
    dflows = {i: DealerFlow(i) for i in config.TRADABLE}
    last_dflow: dict[str, object] = {}
    rv_models = {i: RVF.load_model(i) for i in config.TRADABLE}
    _rv_acc = {i: {"m": -1, "px": None, "rv": 0.0} for i in config.TRADABLE}
    last_rv: dict[str, dict] = {}
    _bocpd: dict[str, BOCPD] = {}   # v9.9 regime-break sentinel
    _rv_prev: dict[str, float] = {}
    _cp_logged: dict[str, float] = {}
    last_mac_lite: dict[str, dict] = {}
    last_book: dict = {}
    _OPEN_SOD_B = (int(config.SESSION_OPEN.split(":")[0]) * 3600
                   + int(config.SESSION_OPEN.split(":")[1]) * 60)
    _rvm = [i for i, m in rv_models.items() if m]
    if _rvm:
        log.info("rv forecaster loaded for %s — telemetry only until the "
                 "skill certificate passes (tools/rv_skill_report.py)", _rvm)

    def _sv_quotes(spec):
        """Fresh two-leg book via the throttled QuoteCache (≈1 Hz)."""
        qq = qc.get([(spec.exchange, spec.short_symbol),
                     (spec.exchange, spec.long_symbol)])
        out = {}
        for tokn, sym in ((spec.short_token, spec.short_symbol),
                          (spec.long_token, spec.long_symbol)):
            d = (qq.get(sym) or {}).get("depth") or {}
            b0 = (d.get("buy") or [{}])[0]
            s0 = (d.get("sell") or [{}])[0]
            out[tokn] = {"bid": float(b0.get("price") or 0),
                         "ask": float(s0.get("price") or 0)}
        return out

    def _fly_quotes(spec):
        """Fresh three-leg book via the throttled QuoteCache (≈1 Hz)."""
        qq = qc.get([(spec.exchange, spec.wing_in_symbol),
                     (spec.exchange, spec.body_symbol),
                     (spec.exchange, spec.wing_out_symbol)])
        out = {}
        for tokn, sym in ((spec.wing_in_token, spec.wing_in_symbol),
                          (spec.body_token, spec.body_symbol),
                          (spec.wing_out_token, spec.wing_out_symbol)):
            d = (qq.get(sym) or {}).get("depth") or {}
            b0 = (d.get("buy") or [{}])[0]
            s0 = (d.get("sell") or [{}])[0]
            out[tokn] = {"bid": float(b0.get("price") or 0),
                         "ask": float(s0.get("price") or 0)}
        return out

    def _write_report(final: bool = False):
        report.d["mode"] = ("LIVE" if config.live_fire_armed() else
                            ("paper-explore" if config.PAPER_EXPLORE else "paper"))
        report.d["policy"] = policy.kind
        report.d["entry_bar"] = entry_bar
        report.d["drift_grade"] = drift_grade
        report.d["gate_funnel"] = funnel.as_dict()
        report.d["conviction_abs"] = {i: conv_res[i].summary()
                                      for i in config.TRADABLE}
        report.d["winprob_at_gate"] = {i: wp_res[i].summary()
                                       for i in config.TRADABLE}
        report.d["regime_share_s"] = regime_share
        report.d["cascade"]["mode"] = casc_mode
        # ---- v9.6 portfolio book: both engines as ONE set of greeks ----
        try:
            _legs, _lm = [], []
            for _i, _p in pms.items():
                if _p.pos is not None:
                    _legs.append({"index": _i, "strike": _p.pos.strike,
                                  "is_call": _p.pos.direction == "CE",
                                  "qty": _p.pos.qty, "mid": None})
                    _lm.append((f"{_p.pos.exchange}:{_p.pos.symbol}",
                                len(_legs) - 1))
            if flybook.pos is not None:
                _sp, _sc = flybook.pos, flybook.pos.spec
                for _sym, _K, _q in (
                        (_sc.wing_in_symbol, _sc.wing_in_k,
                         _sp.lots * _sc.lot),
                        (_sc.body_symbol, _sc.body_k,
                         -2 * _sp.lots * _sc.lot),
                        (_sc.wing_out_symbol, _sc.wing_out_k,
                         _sp.lots * _sc.lot)):
                    _legs.append({"index": _sc.index, "strike": _K,
                                  "is_call": _sc.side == "CE",
                                  "qty": _q, "mid": None})
                    _lm.append((f"{_sc.exchange}:{_sym}", len(_legs) - 1))
            if _lm:
                try:
                    _qs = kite.quote([k for k, _ in _lm])
                    for _k, _ix in _lm:
                        _q = _qs.get(_k) or {}
                        _d = _q.get("depth") or {}
                        _b = (_d.get("buy") or [{}])[0].get("price") or 0
                        _a = (_d.get("sell") or [{}])[0].get("price") or 0
                        _legs[_ix]["mid"] = ((float(_b) + float(_a)) / 2
                                             if _b and _a else
                                             float(_q.get("last_price") or 0)
                                             or None)
                except Exception:                         # noqa: BLE001
                    pass                       # est_iv fallback carries it
            last_book = (BOOK.compute_book(_legs, last_mac_lite)
                         if _legs else {})
        except Exception as _be:                          # noqa: BLE001
            log.warning("book telemetry: %s", _be)
            last_book = {}
        report.d["book"] = last_book
        report.d["dealer_flow"] = {
            i: ({"net_gex": _v.net_gex,
                 "charm_rs_min": round(_v.charm_flow_rs_min, 1),
                 "vanna_units_volpt": round(_v.vanna_units_volpt, 1),
                 "pin": {str(k): round(pv, 3)
                         for k, pv in _v.pin.items()},
                 "signs_inferred": _v.signs_inferred,
                 "age_s": round(_v.snapshot_age_s, 1)}
                if (_v := last_dflow.get(i)) is not None else None)
            for i in config.TRADABLE}
        report.d["rv"] = dict(last_rv)
        report.d["shortvol"]["open_mark"] = (
            flybook.mark(ts=ts,
                         spot=last_spot.get(flybook.pos.spec.index, 0.0),
                         quotes=_fly_quotes(flybook.pos.spec))
            if flybook.pos is not None else None)
        report.d["shortvol"]["mode"] = sv_mode
        report.d["shortvol"]["gate_blockers"] = {
            i: dict(sorted(b.items(), key=lambda kv: -kv[1])[:6])
            for i, b in sv_gate_block.items()}
        report.d["shortvol"]["closed_today"] = flybook.closed_today
        report.d["shortvol"]["open"] = (None if flybook.pos is None else {
            "id": flybook.pos.fly_id, "index": flybook.pos.spec.index,
            "side": flybook.pos.spec.side,
            "debit": round(flybook.pos.debit, 2), "lots": flybook.pos.lots,
            "max_loss": round(flybook.pos.max_loss, 2),
            "opened": flybook.pos.open_hm})
        report.d["cascade"]["fired"] = dict(cascade_fired)
        report.d["cascade"]["entered"] = dict(cascade_entered)
        report.d["cascade"]["flip_now"] = {
            i: ({"flip": (round(_n.flip, 1) if _n.flip else None),
                 "net_gex": _n.net_gex,
                 "snapshot_age_s": round(_n.snapshot_age_s, 1),
                 "in_band": _n.in_band}
                if (_n := flip_now.get(i)) is not None else None)
            for i in config.TRADABLE}
        report.d["pnl_realized"] = round(risk.realized_pnl, 2)
        report.d["risk_halted"] = bool(risk.halted)
        if final:
            report.d["session_complete"] = True
        report.write()

    if kite:
        try:                                  # prev-day levels (real candles)
            for idxn in config.TRADABLE:
                d = kite.ltp([config.INDICES[idxn]["spot_symbol"]])
                tok = int(list(d.values())[0]["instrument_token"])
                cs = kite.historical_data(
                    tok, dt.date.today() - dt.timedelta(days=7),
                    dt.date.today() - dt.timedelta(days=1), "day")
                if cs:
                    last = cs[-1]
                    levels[idxn] = {"pdh": float(last["high"]),
                                    "pdl": float(last["low"]),
                                    "pdc": float(last["close"])}
            log.info("prev-day levels: %s", levels)
        except Exception as e:                 # noqa: BLE001
            log.warning("levels fetch: %s", e)
    entry_bar = config.entry_conviction_bar()
    _mode = "LIVE" if config.live_fire_armed() else (
        "paper (EXPLORE — not mirroring live)" if config.PAPER_EXPLORE
        else "paper (mirrors live exactly; no real order placed)")
    log.info("entry conviction bar: %.2f | mode: %s", entry_bar, _mode)
    last_cal_load = time.time()
    last_hb = 0.0
    # WHY-FLAT TRACKER: the last gate that stopped a fill, per index, refreshed
    # each tick and surfaced on every heartbeat so the operator sees the CURRENT
    # blocking reason instead of scrolling per-tick skip lines. NOTE: in PAPER,
    # drift de-arm does NOT block (paper continues), so this is the real
    # paper-blocking reason; the live-only drift de-arm stays in the `drift` field.
    skip_reason: dict[str, str] = {i: "warming up" for i in config.TRADABLE}
    stale_logged = False
    ring_quotes: dict[int, dict] = {}
    engine.quote_fn = lambda tok: ring_quotes.get(tok, {})

    while True:
        time.sleep(0.2)
        state, age = ring.read_state()
        if state is None:
            continue
        market = state.get("market", {})
        ts = float(state.get("ts", time.time()))
        risk.on_tick()
        hm = dt.datetime.now().strftime("%H:%M")
        if hm >= config.SESSION_CLOSE:
            if flybook.pos is not None:
                _cr = flybook.manage(ts=ts, hm=hm,
                                    spot=last_spot.get(
                                        flybook.pos.spec.index, 0.0),
                                    quotes=_fly_quotes(flybook.pos.spec),
                                    cascade_event=True)
                if _cr is not None:
                    log.warning("Ⓥ SHORTVOL session-close flat → ₹%+.2f",
                                _cr["pnl"])
                    report.d["shortvol"]["events"].append(_cr)
            log.info("session over — done. PnL ₹%.2f", risk.realized_pnl)
            break

        # ---- STALE-FEED GUARD (the WiFi-drop / harvester-stall case) ----
        # The RiskGovernor already refuses entries when the feed is old, but we
        # must not even PROCESS or attempt against a frozen quote: that wastes
        # cycles, spams the log, and in live mode risks pricing an order against
        # a quote that no longer exists. So when the feed is stale: flatten any
        # open position past the flatten threshold, warn ONCE, and skip the rest
        # of the tick entirely. Trading resumes automatically when ticks return.
        if age > config.DATA_STALE_BLOCK_S:
            if not stale_logged:
                log.warning("⚠ feed STALE (%.0fs) — entries suspended, will "
                            "resume when ticks return (check connection/"
                            "harvester)", age)
                stale_logged = True
            for idx in config.TRADABLE:
                funnel.record(idx, "stale_feed")
            if age > config.DATA_STALE_FLATTEN_S:
                if flybook.pos is not None:
                    _cr = flybook.manage(
                        ts=ts, hm=hm,
                        spot=last_spot.get(flybook.pos.spec.index, 0.0),
                        quotes={}, cascade_event=True)
                    if _cr is not None:
                        log.warning("Ⓥ SHORTVOL stale-feed flat → ₹%+.2f "
                                    "(worst-mark, dead book)", _cr["pnl"])
                        report.d["shortvol"]["events"].append(_cr)
                for idx in config.TRADABLE:
                    pm = pms[idx]
                    if pm.pos is not None:
                        log.warning("flattening %s on stale feed (%.0fs)",
                                    pm.pos.symbol, age)
                        stale_ctx = TickContext(
                            ts=ts, hm=hm,
                            spot=last_spot.get(idx, 0.0),
                            spot_velocity_1s=0.0, data_age_s=age,
                            atm_iv=0.0, minutes_to_close=0.0)
                        pm._exit(stale_ctx,
                                 ring_quotes.get(pm.pos.token, {}),
                                 "STALE_FEED_FLATTEN", urgent=True)
            # heartbeat still beats so you can SEE the staleness climbing
            if time.time() - last_hb >= config.HEARTBEAT_S:
                last_hb = time.time()
                log.info("♥ %s | feed age %.0fs STALE | PnL ₹%+.0f | "
                         "pos %s | conv %s | (waiting for ticks)", hm, age,
                         risk.realized_pnl,
                         {i: (pms[i].pos.symbol if pms[i].pos else "—")
                          for i in config.TRADABLE},
                         {i: (f"{persist[i].latest:+.2f}"
                              if persist[i].latest is not None else "—")
                          for i in config.TRADABLE})
            continue
        if stale_logged:
            log.info("✓ feed recovered (age %.1fs) — entries resumed", age)
            stale_logged = False

        if time.time() - last_cal_load > config.CAL_RELOAD_S:
            cal = load_calibration(); last_cal_load = time.time()
        if time.time() - last_report >= config.DIAG_WRITE_EVERY_S:
            last_report = time.time()
            _write_report()
        if time.time() - last_hb >= config.HEARTBEAT_S:
            last_hb = time.time()
            d = drift.assess()
            drift_grade = d.get("grade", "NO_REF")
            if drift_grade == "DRIFTED":
                log.warning("⚠ REGIME DRIFT: %d/%d key features shifted "
                            "significantly (worst: %s) — LIVE DE-ARMED until "
                            "the forge re-references; paper continues.",
                            d.get("significant"), d.get("features_considered"),
                            ", ".join(list(d.get("worst", {}).keys())[:3]))
            elif drift_grade == "WATCH":
                log.info("drift WATCH: %d features moderately shifted — "
                         "tape is moving, still trading", d.get("moderate"))
            # v9.6.5: mark the open spread ONCE per heartbeat, BEFORE any
            # consumer (pos field, PnL, Ⓥ line). Side-effect-free; reuses the
            # same leg quotes manage() reads this tick, so no extra API load.
            _fly_mark = (flybook.mark(
                ts=ts, spot=last_spot.get(flybook.pos.spec.index, 0.0),
                quotes=_fly_quotes(flybook.pos.spec))
                if flybook.pos is not None else None)
            poss = {i: (pms[i].pos.symbol if pms[i].pos else
                        (f"FLY {_fly_mark['body_k']:.0f} "
                         f"±{_fly_mark['body_k'] - _fly_mark['wing_in_k']:.0f}"
                         f" {_fly_mark['unreal']:+.0f}"
                         if _fly_mark and _fly_mark["index"] == i else "—"))
                    for i in config.TRADABLE}
            _reg = regime_now.get(config.TRADABLE[0])
            _reg_s = f"{_reg.label}×{_reg.conv_mult:.2f}" if _reg else "—"
            # aggregate walk-away diagnostics across tradable indices
            _run = sum(pms[i]._walkaway_tally["runaway"] for i in config.TRADABLE)
            _bord = sum(pms[i]._walkaway_tally["borderline"] for i in config.TRADABLE)
            _wa = f"{_run}R/{_bord}B" if (_run or _bord) else "0"
            # latest signed conviction per tradable index (post-regime-mult —
            # the exact value the entry gate sees). "—" until the first read.
            convs = {i: (f"{persist[i].latest:+.2f}"
                         if persist[i].latest is not None else "—")
                     for i in config.TRADABLE}
            # WHY EACH INDEX IS FLAT right now — the last gate that stopped a fill
            # this tick (a held position shows "in <symbol>"). This is the
            # PAPER-blocking reason; drift de-arm is live-only and is the separate
            # `drift` field, so the two are never conflated.
            no_trade = " · ".join(f"{i}: {skip_reason.get(i, '—')}"
                                  for i in config.TRADABLE)
            _cf_n = sum(cascade_fired.values())
            _ce_n = sum(cascade_entered.values())
            _casc = f"{_cf_n}F/{_ce_n}E {casc_mode}"
            # certificate can appear/refresh mid-session (harness run after a
            # morning close, cron, etc.) — re-resolve the staging mode each
            # heartbeat so arming never needs a brain restart.
            cascade_cert = CS.load_certificate()
            casc_mode = CS.cascade_mode(cascade_cert)
            sv_cert = SVOL.load_certificate()
            sv_mode = SVOL.shortvol_mode(sv_cert)
            _svtok = ((f"{flybook.pos.spec.index}·fly" if flybook.pos
                       else f"{flybook.closed_today}c") + f" {sv_mode}")
            # displacement token is only meaningful when the fly can hold the
            # lock; when fly trading is off, the fly is pure telemetry and
            # there is nothing to displace — say so instead.
            if getattr(config, "FLY_TRADING_ENABLED", False):
                _dtok = (f"{disp.count_today}/"
                         f"{getattr(config, 'DISP_MAX_PER_DAY', 2)}")
                if flybook.pos is not None and disp.last_refusal:
                    _dtok += f" ({disp.last_refusal})"
                _svtok += f" | disp {_dtok}"
            else:
                _svtok = "fly telemetry-only (feeds directional)"
            # v9.7.1: surface the fly's mined directional read (pinning regime,
            # near wall, pin pressure) for whichever tradable index has it live
            _fi_live = next((last_fly_intel.get(i) for i in config.TRADABLE
                             if getattr(last_fly_intel.get(i), "active",
                                        False)), None)
            if _fi_live is not None:
                _pol_s = {1: "MOM", -1: "REV", 0: "undec"}.get(
                    getattr(_fi_live, "polarity", 0), "?")
                _svtok += (f" | fly-intel[{_pol_s}] {_fi_live.near_wall}wall "
                           f"pin{_fi_live.pin_pressure:.2f} "
                           f"runway×{_fi_live.target_runway_mult:.2f}"
                           + (f" arm{_fi_live.retest_arm_delay_s:.0f}s"
                              if _fi_live.retest_arm_delay_s else "")
                           + (f" ride→{_fi_live.revert_hint_side}"
                              if _fi_live.revert_hint_side else ""))
            # v9.7.1: surface the order-flow toxicity / trap read
            _tv_live = next((last_tox.get(i) for i in config.TRADABLE
                             if getattr(last_tox.get(i), "toxicity", 0) > 0),
                            None)
            if _tv_live is not None:
                _svtok += (f" | tox {_tv_live.toxicity:.2f}"
                           f"{'+' if _tv_live.tox_dir > 0 else '-' if _tv_live.tox_dir < 0 else ''}"
                           + (f" SWEEP-{_tv_live.sweep_dir}"
                              if _tv_live.sweep else ""))

            log.info("♥ %s | feed age %.1fs | PnL ₹%+.0f | deployed ₹%.0f | "
                     "halted=%s | pos %s | conv %s | policy %s | VIX %s | "
                     "regime %s | walkaway %s | cascade %s | sv %s | drift %s | "
                     "no-trade: %s",
                     hm, age,
                     risk.realized_pnl + (_fly_mark["unreal"]
                                          if _fly_mark else 0.0),
                     risk.deployed,
                     risk.halted or risk.halt_reason or False, poss, convs,
                     policy.kind,
                     f"{vix_hist[-1][1]:.2f}" if vix_hist else "—",
                     _reg_s, _wa, _casc, _svtok, drift_grade, no_trade)
            # GATE FUNNEL — running per-index tally of what blocked entries so
            # far today (top 5 gates each). The daily JSON has the full table.
            log.info("  funnel %s", " | ".join(funnel.line(i)
                                               for i in config.TRADABLE))
            # KEY LEVELS per tradable index — spot vs prev-day candle (PDC/PDH/PDL)
            # and the live GEX call/put walls + gamma flip from the macro radar
            # (the walls are what the system caps targets at). One line per index;
            # any piece that isn't available yet prints "—".
            for i in config.TRADABLE:
                sp = last_spot.get(i)
                if not sp:
                    log.info("  levels %-6s | spot — (waiting for ticks)", i)
                    continue
                lv = levels.get(i, {})
                mc = read_macro(i) or {}
                pdc = lv.get("pdc")
                pdc_s = (f"PDC {pdc:.0f} ({sp/pdc - 1:+.1%})" if pdc else "PDC —")
                pdh_s = f"PDH {lv['pdh']:.0f}" if lv.get("pdh") else "PDH —"
                pdl_s = f"PDL {lv['pdl']:.0f}" if lv.get("pdl") else "PDL —"
                cw, pw, fl = mc.get("call_wall"), mc.get("put_wall"), mc.get("flip")
                _ncv = flip_now.get(i)
                nc_s = (f"  nflip {_ncv.flip:.0f}({_ncv.snapshot_age_s:.0f}s)"
                        if _ncv is not None and _ncv.flip else "  nflip —")
                _dfv = last_dflow.get(i)
                df_s = ((f"  charm ₹{_dfv.charm_flow_rs_min:+,.0f}/m"
                         f"  vanna {_dfv.vanna_units_volpt:+,.0f}u"
                         + (f"  pin {max(_dfv.pin.values()):.2f}"
                            if _dfv.pin else ""))
                        if _dfv is not None else "")
                _bk = last_book.get(i)
                bk_s = ((f"  Δ₹{_bk['delta_rs']:+,.0f}"
                         f" θ₹{_bk['theta_rs_day']:+,.0f}/d")
                        if _bk else "")
                _rvv = last_rv.get(i)
                rv_s = ((f"  rv̂ {100 * _rvv['rv_hat']:.1f}%"
                         + (f" vrp {100 * _rvv['vrp']:+.1f}%"
                            if _rvv.get("vrp") is not None else ""))
                        if _rvv else "")
                wall_s = (f"Cwall {cw:.0f}" if cw else "Cwall —") + \
                         (f"  Pwall {pw:.0f}" if pw else "  Pwall —") + \
                         (f"  flip {fl:.0f}" if fl else "  flip —") + nc_s \
                         + df_s + rv_s + bk_s
                log.info("  levels %-6s | spot %.0f  %s  %s  %s | %s",
                         i, sp, pdc_s, pdh_s, pdl_s, wall_s)
            if _fly_mark is not None:
                _tt = "live" if _fly_mark["live_quote"] else "stale-mark"
                _tv = _fly_mark.get("trail") or {}
                _tr_s = (f" | smooth ₹{_fly_mark['cc_smooth']:.2f} "
                         f"hwm {_tv.get('hwm', 0):.2f}"
                         + (f" ratchet {_tv['ratchet']:.2f}"
                            if _tv.get("ratchet") is not None else "")
                         + (f" σ {_tv.get('sigma', 0):.2f}" if _tv else "")
                         + (" ★tagged" if _fly_mark.get("tgt_tagged") else "")
                         ) if _tv else ""
                log.info("  Ⓕ %-6s %s | BUY %s + BUY %s + SELL 2× %s | "
                         "debit ₹%.2f → credit ₹%.2f | unreal ₹%+.0f | "
                         "target %+.0f%% floor %+.0f%% | pin dist %g | pop "
                         "%.2f | held %ds | %s [%s]%s",
                         _fly_mark["index"], _fly_mark["fly_id"],
                         _fly_mark["wing_in_symbol"],
                         _fly_mark["wing_out_symbol"], _fly_mark["body_symbol"],
                         _fly_mark["debit"], _fly_mark["close_credit"],
                         _fly_mark["unreal"], _fly_mark["to_target_pct"],
                         _fly_mark["to_floor_pct"], _fly_mark["dist_from_body"],
                         _fly_mark["pop"], _fly_mark["held_s"], _tt,
                         _fly_mark["mode"], _tr_s)

        # refresh ring-backed quotes for the paper engine + surfaces from macro
        ring_quotes.clear()
        for idx, ctx_m in market.items():
            for leg, info in (ctx_m.get("legs") or {}).items():
                if info.get("token"):
                    ring_quotes[info["token"]] = info["snap"]
        # The ring only carries ATM legs. Any OPEN position on a non-ATM strike
        # (common on SENSEX) would otherwise have NO quote here — unfillable AND
        # unmanageable. Refresh each held position's exact strike directly so it
        # can be marked-to-market, tracked, and exited.
        if mapper:
            for _i in config.TRADABLE:
                _p = pms[_i].pos
                if _p is not None and _p.token not in ring_quotes:
                    try:
                        _qq = qc.get([(_p.exchange, _p.symbol)]).get(_p.symbol)
                        if _qq:
                            _d = _qq.get("depth") or {}
                            _b0 = (_d.get("buy") or [{}])[0]
                            _s0 = (_d.get("sell") or [{}])[0]
                            ring_quotes[_p.token] = {
                                "bid": float(_b0.get("price") or 0),
                                "ask": float(_s0.get("price") or 0),
                                "bid_qty": float(_b0.get("quantity") or 0),
                                "ask_qty": float(_s0.get("quantity") or 0),
                                "ltp": float(_qq.get("last_price") or 0)}
                    except Exception:                      # noqa: BLE001
                        pass
        for idx, ctx_m in market.items():
            mac = read_macro(idx)
            if mac and mac.get("strikes"):
                T = float(ctx_m.get("T", 0.01))
                F = float((ctx_m.get("spot") or {}).get("ltp") or 0) * \
                    math.exp(config.RISK_FREE_RATE * T)
                if F > 0:
                    builder.fit_surface(idx, ctx_m.get("expiry", ""),
                                        mac["strikes"], mac["iv"], F, T)

        # ---- v9.1.2 DECISION CADENCE GATE (the last parity gap) -----------
        # The harvester writes the ring ONCE per second (RING_WRITE_S=1.0);
        # this ~5 Hz loop was re-pushing the SAME snapshot ~5×/second, so
        # per-push flow (vol_delta → OFI/VPIN/Hawkes/dealer-inv) was ingested
        # ~5× vs the forge's 1 Hz replay — live conv peaked 0.86–0.95 on
        # 2026-07-03 while the replay's ceiling sat ~0.5, AND the net infers
        # on a frame cadence it never trained on. Decisions now run once per
        # ring second: identical estimator windows, identical persistence
        # sampling, identical funnel counting, live ↔ replay. The 0.2 s loop
        # below keeps exits, fills, trap checks, the stale watchdog and trade
        # tracking at full tempo. drift.observe moves to 1 Hz too, matching
        # the reference population's sampling.
        decide_now = int(ts) != last_decision_sec
        vix = (market.get("_VIX") or {}).get("ltp")
        if vix and decide_now:
            # 1 Hz append: the (ts, vix) deque(400) now spans ~400 s, so the
            # 295 s spike lookback is finally a real 5-minute test (at 5 Hz
            # appends it silently degraded to an ~80 s window).
            vix_hist.append((ts, float(vix)))
        vix_bump = 0.0
        if len(vix_hist) > 5:
            base = next((v for t0_, v in vix_hist if ts - t0_ >= 295),
                        vix_hist[0][1])
            now_v = vix_hist[-1][1]
            if base > 0 and (now_v - base) / base >= config.VIX_SPIKE_5M_PCT:
                vix_bump = config.VIX_BAR_BUMP
                log.debug("VIX spike %.1f→%.1f — entry bar +%.2f",
                          base, now_v, vix_bump)

        if decide_now:
            last_decision_sec = int(ts)
            obs = builder.push(market, ts)
            frame = builder.frames[-1]
            drift.observe(frame)
            actions = policy.conviction(obs, frame)
        if frame is None:
            continue                       # no ring second pushed yet

        for idx in config.TRADABLE:
            ctx_m = market.get(idx)
            if not ctx_m or not ctx_m.get("spot"):
                if decide_now:
                    funnel.record(idx, "no_market")
                continue
            spot = float(ctx_m["spot"].get("ltp") or 0)
            if spot <= 0:
                if decide_now:
                    funnel.record(idx, "no_market")
                continue
            vel = spot - last_spot.get(idx, spot); last_spot[idx] = spot
            i = config.INDEX_ORDER.index(idx)
            pm = pms[idx]
            cascade_ev = None                    # set only on decision seconds

            # ================= 1 Hz DECISION STACK (ring-second cadence —
            # byte-identical to the forge grader's _Replayer) ================
            if decide_now:
                spot_secs[idx].append(spot)
                if spot > 0 and idx in rvol:
                    rvol[idx].update(spot, dt_s=1.0)
                    realized_vol_ann[idx] = rvol[idx].annualized()
                open_px.setdefault(idx, spot)
                if idx not in p945 and hm >= "09:45":
                    p945[idx] = spot
                ai = float(actions[2 * i])
                # ADVISORY nudges only (audit: ±0.99 force-writes retired).
                # The whole stack lives in core/decision.compute_shock — one
                # copy, shared with the forge's replay.
                mac = read_macro(idx)
                last_mac[idx] = mac
                # ---- CASCADE: 1 Hz analytic flip nowcast + detector --------
                # Flip-source hierarchy identical to the harness: fresh
                # analytic nowcast off the radar's per-contract profile, else
                # the radar's own numbers (≤MACRO_STALE_S), else the detector
                # idles. Zero extra API calls.
                nowcasts[idx].update_snapshot(mac)
                _nc = nowcasts[idx].nowcast(spot, ts)
                flip_now[idx] = _nc
                if _nc is not None:
                    _cf, _cw, _cg = _nc.flip, _nc.flip_width, _nc.net_gex
                    _src, _cage = "nowcast", _nc.snapshot_age_s
                elif mac is not None:
                    _cf, _cw, _cg = (mac.get("flip"), mac.get("flip_width"),
                                     mac.get("net_gex"))
                    _src, _cage = "radar", ts - float(mac.get("ts") or ts)
                else:
                    _cf = _cw = _cg = None
                    _src, _cage = "none", 0.0
                cascade_ev = detectors[idx].update(
                    ts=ts, day=report.date, spot=spot, flip=_cf,
                    flip_width=_cw, net_gex=_cg,
                    strike_step=float(config.INDICES[idx]["strike_step"]),
                    flip_source=_src, flip_age_s=float(_cage))
                if cascade_ev is not None:
                    cascade_fired[idx] += 1
                    _row = cascade_ev.as_dict()
                    _row["hm"] = hm
                    _row["mode"] = casc_mode
                    if pm.pos is not None:
                        _row["skip"] = f"in {pm.pos.symbol}"
                    report.d["cascade"]["events"].append(_row)
                    log.warning("⚡ CASCADE %s %s %s z=%+.2f | spot %.0f < "
                                "flip %.0f−hyst | netGEX %.2e [%s %.0fs] | %s",
                                idx, cascade_ev.kind, cascade_ev.direction,
                                cascade_ev.z, spot, cascade_ev.flip,
                                cascade_ev.net_gex, _src, _cage,
                                f"{casc_mode.upper()} — attempting entry"
                                if casc_mode != "telemetry"
                                and pm.pos is None
                                else "telemetry"
                                if casc_mode == "telemetry"
                                else _row.get("skip"))
                # ---- v9.4 telemetry: dealer-flow vector + rv accumulation
                dflows[idx].update_snapshot(mac)
                _dfw = dflows[idx].vector(
                    spot, ts, walls=((mac or {}).get("call_wall"),
                                     (mac or {}).get("put_wall")))
                if _dfw is not None:
                    last_dflow[idx] = _dfw
                last_mac_lite[idx] = {"spot": spot,
                    "dte": (mac or {}).get("dte"),
                    "atm_iv": (mac or {}).get("atm_iv")}
                _acc = _rv_acc[idx]
                _m_now = (int((ts + 19800) % 86400) - _OPEN_SOD_B) // 60
                if 0 <= _m_now < RVF.SESSION_MIN:
                    if _acc["m"] != _m_now:
                        if _acc["px"] and spot > 0 and _acc["m"] >= 0:
                            _r_ = math.log(spot / _acc["px"])
                            _acc["rv"] += _r_ * _r_
                        _acc["m"], _acc["px"] = _m_now, spot
                    _mdl = rv_models.get(idx)
                    if _mdl:
                        _prj = RVF.predict_remaining(_mdl, _m_now, _acc["rv"])
                        if _prj:
                            _ivn = (mac or {}).get("atm_iv")
                            last_rv[idx] = {
                                "rv_hat": round(_prj["day_ann_vol"], 4),
                                "rem_ann": round(_prj["rem_ann_vol"], 4),
                                "vrp": (round(float(_ivn)
                                              - _prj["day_ann_vol"], 4)
                                        if _ivn else None)}
                            # v9.9 BOCPD: did the vol world just BREAK? (telemetry only)
                            try:
                                _rvp = _rv_prev.get(idx)
                                _rvn = last_rv[idx].get("rv")
                                if _rvp and _rvn and _rvp > 0 and _rvn > 0:
                                    _st = _bocpd.setdefault(idx, BOCPD()).update(
                                        math.log(_rvn / _rvp))
                                    last_rv[idx]["cp_prob"] = _st["cp_prob"]
                                    last_rv[idx]["cp_run"] = _st["map_run"]
                                    if _st["cp_prob"] >= 0.80 and ts - _cp_logged.get(
                                            idx, 0) > 120:
                                        _cp_logged[idx] = ts
                                        log.warning("⚡ BOCPD %s: vol regime BREAK "
                                                    "p=%.2f (run was %d)", idx,
                                                    _st["cp_prob"], _st["map_run"])
                                _rv_prev[idx] = _rvn
                            except Exception:                              # noqa: BLE001
                                pass
                # ---- v9.3 SHORT-VOL gate (1 Hz): cascade state is the
                # veto, exactly as the harness applies it --------------------
                _step_sv = float(config.INDICES[idx]["strike_step"])
                _hy_sv = max(config.CASCADE_HYST_MULT * float(_cw or 0.0),
                             _step_sv)
                sv_blocked_now[idx] = bool(
                    cascade_ev is not None
                    or ts < getattr(detectors[idx], "_cooldown_until", -1e18)
                    or (_cf is not None and _cg is not None
                        and spot < _cf - _hy_sv
                        and _cg <= config.CASCADE_NET_GEX_MAX))
                _svg = SVOL.evaluate_gate(
                    hm=hm, spot=spot, mac=mac,
                    net_gex_now=(_cg if _src == "nowcast" else None),
                    dte=ctx_m.get("dte"), strike_step=_step_sv,
                    vix_bump=vix_bump,
                    cascade_blocked=sv_blocked_now[idx])
                # ---- v9.7.1 FLY INTELLIGENCE: mine the gate read for the
                # DIRECTIONAL book (the fly's job is now to make better long
                # trades, not to be the trade). Regime map captured every
                # second the gate grants; the per-candidate conv modulation is
                # applied below once `conv` exists. Ungranted ⇒ neutral.
                last_fly_intel[idx] = (FI.assess(
                    granted=_svg.ok, side=_svg.side, spot=spot,
                    call_wall=(mac or {}).get("call_wall"),
                    put_wall=(mac or {}).get("put_wall"),
                    corridor_steps=_svg.corridor_steps,
                    iv_rank=_svg.iv_rank, net_gex=_svg.net_gex,
                    strike_step=_step_sv, direction=None, conviction=0.0)
                    if getattr(config, "FLY_INTEL_ENABLED", True)
                    else FI.FlyIntel(active=False))
                if not _svg.ok:
                    _bk = sv_gate_block[idx]
                    _k = _svg.reason[:40]
                    if _k in _bk or len(_bk) < 12:
                        _bk[_k] = _bk.get(_k, 0) + 1
                # ---- FLY OPEN PATH — DISABLED BY DEFAULT (v9.7.1) ----------
                # The operator's standing instruction: the butterfly must NOT
                # take positions, not even paper. Its gate still evaluates
                # above (feeding core/fly_intel for the directional book); this
                # branch — the ONLY place a fly is ever opened — is gated off
                # by FLY_TRADING_ENABLED (default False). Off ⇒ no fly ever
                # occupies the global lock, so displacement is moot and the
                # capital is always free for directional trades. Set the knob
                # True only to resurrect the paper-explore fly engine.
                elif (getattr(config, "FLY_TRADING_ENABLED", False)
                      and sv_mode != "telemetry" and flybook.pos is None
                      and all(pms[_j].pos is None
                              for _j in config.TRADABLE)
                      and not risk.halted
                      and mapper is not None):
                    _rungs = mapper.hierarchy(idx, spot, _svg.side)
                    _spec, _why = BFLY.build_fly(
                        idx, _svg.side, _step_sv,
                        (mac or {}).get("call_wall") or 0,
                        (mac or {}).get("put_wall") or 0, _rungs)
                    if _spec is None:
                        sv_gate_block[idx][_why[:40]] = (
                            sv_gate_block[idx].get(_why[:40], 0) + 1)
                        flybook.last_try[idx] = ts
                    else:
                        _r = flybook.try_open(
                            ts=ts, hm=hm, spec=_spec,
                            quotes=_fly_quotes(_spec),
                            capital=config.TRADING_CAPITAL, mode=sv_mode)
                        if "opened" in _r:
                            log.warning(
                                "Ⓕ BUTTERFLY OPEN %s %s body %g wings ±%g "
                                "debit ₹%.2f ×%d (max loss ₹%.0f = debit paid,"
                                " pop~%.2f, ivr %.2f, gex %.1e) [%s] | BUY %s "
                                "+ BUY %s + SELL 2× %s", idx, _svg.side,
                                _spec.body_k, _spec.wing_width, _r["debit"],
                                _r["lots"], _r["max_loss"], _r["pop"],
                                _svg.iv_rank or 0, _svg.net_gex or 0, sv_mode,
                                _spec.wing_in_symbol, _spec.wing_out_symbol,
                                _spec.body_symbol)
                            _r["hm"] = hm
                            report.d["shortvol"]["events"].append(_r)
                        elif _r.get("skip") not in ("throttled",
                                                    "book occupied"):
                            _kk = str(_r.get("skip"))[:40]
                            sv_gate_block[idx][_kk] = (
                                sv_gate_block[idx].get(_kk, 0) + 1)
                node = frame[i * config.NODES_PER_INDEX]
                f30 = ((p945[idx] - open_px[idx]) / open_px[idx]
                       if idx in p945 and open_px.get(idx) else 0.0)
                shock = D.compute_shock(
                    ai=ai, vpin=float(node[4]), dealer_inv=float(node[16]),
                    mac=mac, spot=spot, dte=float(ctx_m.get("dte", 9.0)),
                    levels=levels.get(idx), f30=f30, hm=hm)
                conv = D.fuse(ai, shock)
                sh = spot_secs[idx]
                dsum = sum(abs(b - a) for a, b in zip(sh, list(sh)[1:])) \
                    if len(sh) > 120 else 0.0
                er = (abs(sh[-1] - sh[0]) / dsum) if dsum > 0 else 0.5

                # ---- REGIME: label the tape from state already in hand and
                # scale conviction by it (never a hard veto; risk floors
                # untouched).
                vfc = None
                try:
                    from core.vol_forecaster import forecast as _vol_fcast
                    # iv_now MUST be the SAME series the forecaster learned
                    # from — the macro Newton ATM IV (mac["atm_iv"]), not the
                    # SVI surface value (two estimators inside one z-score
                    # jammed the regime on VOL_CRUSH for three days).
                    _iv_now = (mac or {}).get("atm_iv")
                    if _iv_now is not None:
                        vfc = _vol_fcast(idx, float(_iv_now),
                                         front_iv=(mac or {}).get("atm_iv"),
                                         next_iv=(mac or {}).get("atm_iv_next"),
                                         dte=(mac or {}).get("dte"))
                except Exception:                          # noqa: BLE001
                    vfc = None
                rv = realized_vol_ann.get(idx)
                regime = regime_mod.classify(
                    spot=spot, trend_efficiency=er,
                    net_gex=(mac or {}).get("net_gex"),
                    flip=(mac or {}).get("flip"),
                    call_wall=(mac or {}).get("call_wall"),
                    put_wall=(mac or {}).get("put_wall"),
                    iv_rank=(mac or {}).get("iv_rank"), realized_vol=rv,
                    vol_regime=(vfc.regime if vfc else None),
                    vol_z=(vfc.z if vfc else None),
                    trend_sign=(1 if sh[-1] >= sh[0] else -1)
                    if len(sh) >= 2 else 0,
                    index=idx)
                regime_now[idx] = regime
                # LOGIT-space scaling — dampened regimes demand a stronger raw
                # signal instead of arithmetically vetoing (audit fix).
                conv = D.apply_regime(conv, regime.conv_mult)
                # ---- v9.7.1 FLY-INTEL directional modulation: in a live
                # pinning regime, DAMPEN conviction pointing INTO the near
                # wall (the dealers will fade that breakout — the operator's
                # exact trap) and mildly BOOST a read trading AWAY from it /
                # with the fade. Logit-space, stacked on the regime mult, so
                # it scales-never-vetoes. Neutral whenever the fly gate isn't
                # granting. Toggle: FLY_INTEL_MODULATE_CONV.
                _fi = last_fly_intel.get(idx)
                if (getattr(config, "FLY_INTEL_MODULATE_CONV", True)
                        and _fi is not None and getattr(_fi, "active", False)
                        and conv != 0.0):
                    _fim = FI.assess(
                        granted=True, side=_fi.near_wall, spot=spot,
                        call_wall=(mac or {}).get("call_wall"),
                        put_wall=(mac or {}).get("put_wall"),
                        corridor_steps=_fi.corridor_steps,
                        iv_rank=_fi.iv_rank, net_gex=_fi.net_gex,
                        strike_step=_step_sv,
                        direction=("CE" if conv > 0 else "PE"),
                        conviction=conv)
                    last_fly_intel[idx] = _fim
                    if _fim.conv_mult != 1.0:
                        _conv_pre = conv
                        conv = FI.apply_conv(conv, _fim.conv_mult)
                        log.debug("%s fly-intel %s: conv %+.2f→%+.2f (%s)",
                                  idx, _fim.regime, _conv_pre, conv, _fim.note)
                last_conv[idx] = conv
                last_tape_er[idx] = signed_efficiency(
                    sh, int(getattr(config, "RIDE_ER_WINDOW_S", 120)))
                # wall-clock persistence sampling — 1/second, replay-identical
                persist[idx].push(ts, conv)
                conv_res[idx].add(abs(conv))
                rs = regime_share[idx]
                rs[regime.label] = rs.get(regime.label, 0) + 1
                try:
                    regime_mod.log_features(idx, er, (mac or {}).get("net_gex"))
                except Exception:                          # noqa: BLE001
                    pass

                mins_open = (dt.datetime.strptime(hm, "%H:%M")
                             - dt.datetime.strptime(config.SESSION_OPEN,
                                                    "%H:%M")).seconds / 60.0
                # win-probability path is core/decision — meta logistic +
                # calibration blend, one copy shared with the forge grader.
                wp_meta = D.meta_win_prob(load_meta(), frame, i,
                                          min(mins_open / 375.0, 1.0),
                                          er, f30, 1 if conv > 0 else -1)
                wp = D.blend_winprob(wp_meta, conv, cal)
                log.debug("%s spot %.1f | ai %+.2f shock %+.2f → conv %+.2f "
                          "(wp %.2f)", idx, spot, ai, shock, conv, wp)
                # model's LIVE read of the HELD position (model-shaped exit):
                # refreshed once per second; the ≤1 s-stale cached value feeds
                # the 5 Hz management tctx. Uses the POSITION's direction so a
                # flipping signal can't confuse it.
                if config.META_DECISION_ENABLED and pm.pos is not None:
                    last_wp_hold[idx] = D.meta_win_prob(
                        load_meta(), frame, i, min(mins_open / 375.0, 1.0),
                        er, f30, 1 if pm.pos.direction == "CE" else -1)
                else:
                    last_wp_hold[idx] = None
            else:
                mac = last_mac.get(idx)
                conv = last_conv.get(idx, 0.0)
                regime = regime_now.get(idx)

            # ================= EVERY-TICK MANAGEMENT (~5 Hz: fills, stops,
            # trail, trap shield, floors, tracking — full reaction tempo) ====
            legs_m = ctx_m.get("legs") or {}
            sp_now = 0.0
            atm = legs_m.get("atm_ce", {}).get("snap")
            if atm and atm.get("bid") and atm.get("ask"):
                m = (atm["bid"] + atm["ask"]) / 2
                sp_now = (atm["ask"] - atm["bid"]) / max(m, 0.05)
            spread_ew[idx] = (1 - config.SPREAD_EW_ALPHA) * spread_ew.get(idx, sp_now or 0.01) + \
                config.SPREAD_EW_ALPHA * (sp_now or spread_ew.get(idx, 0.01))
            absorb = any((v.get("snap") or {}).get("iceberg")
                         for v in legs_m.values())
            # REAL option-flow shield inputs: index spot has no volume, so
            # sell-aggression comes from the ATM legs' signed tick-rule flow,
            # and ΔOI from the position-side ATM leg (frame ≤1 s old).
            t_ce = builder.trk.get(f"{idx}:atm_ce")
            t_pe = builder.trk.get(f"{idx}:atm_pe")
            opt_flow = ((t_ce.dealer_inv if t_ce else 0.0)
                        + (t_pe.dealer_inv if t_pe else 0.0))
            sell_ratio = float(np.clip(
                0.5 - 0.5 * math.tanh(opt_flow / config.DEALER_INV_SCALE),
                0, 1))
            oi_node = frame[i * config.NODES_PER_INDEX]
            if pm.pos is not None:
                oi_node = frame[i * config.NODES_PER_INDEX +
                                (1 if pm.pos.direction == "CE" else 2)]
            # v9.7.1 ORDER-FLOW TOXICITY: the index has no volume, so feed the
            # VPIN/OFI estimator from the ATM legs' book + traded volume (the
            # same real flow the shield uses), with SPOT driving the swing-pivot
            # sweep detection. One update per decision second (replay-parity).
            _atm_dir_snap = (legs_m.get("atm_pe" if (pm.pos and
                             pm.pos.direction == "PE") else "atm_ce", {})
                             .get("snap") or {})
            _spot_ltp = float((ctx_m.get("spot") or {}).get("ltp") or 0)
            try:
                last_tox[idx] = tox_engine[idx].update(
                    spot=_spot_ltp,
                    bid=float(_atm_dir_snap.get("bid") or 0),
                    bid_qty=float(_atm_dir_snap.get("bid_qty") or 0),
                    ask=float(_atm_dir_snap.get("ask") or 0),
                    ask_qty=float(_atm_dir_snap.get("ask_qty") or 0),
                    vol_delta=float(_atm_dir_snap.get("vol_delta") or 0))
            except Exception:                              # noqa: BLE001
                pass
            mins_left = max((dt.datetime.strptime(config.SESSION_CLOSE, "%H:%M")
                             - dt.datetime.strptime(hm, "%H:%M")).seconds / 60,
                            1.0)
            tctx = TickContext(
                ts=ts, hm=hm, spot=spot, spot_velocity_1s=vel,
                data_age_s=age,
                atm_iv=_surface_atm_iv(builder, idx,
                                       ctx_m.get("expiry", ""),
                                       float(ctx_m.get("T", 0.01)), mac),
                minutes_to_close=mins_left,
                gex_put_wall=(mac or {}).get("put_wall"),
                gex_call_wall=(mac or {}).get("call_wall"),
                absorption=absorb, aggressive_sell_ratio=sell_ratio,
                oi_delta_since=float(oi_node[2]),
                avg_spread_pct=spread_ew[idx], conviction=conv,
                live_win_prob=last_wp_hold.get(idx),
                regime_label=(regime.label if regime else ""),
                tape_er=last_tape_er.get(idx),
                fly_runway_mult=(getattr(last_fly_intel.get(idx),
                                         "target_runway_mult", 1.0)
                                 if getattr(config, "FLY_INTEL_TARGET_CAP",
                                            True) else 1.0))

            for oid, fill in engine.on_quote(
                    pm.pos.token if pm.pos else -1,
                    ring_quotes.get(pm.pos.token, {}) if pm.pos else {}):
                log.info("resting order %s → %s", oid, fill.status)
            if pm.pos:
                skip_reason[idx] = f"in {pm.pos.symbol}"
                if decide_now:                 # funnel stays 1/sec ↔ replay
                    funnel.record(idx, "in_position")
                if risk.halted:
                    pm._exit(tctx, ring_quotes.get(pm.pos.token, {}),
                             "RISK_HALT", urgent=True)
                else:
                    _pnl_before = pm.risk.realized_pnl
                    _dir_held = pm.pos.direction
                    pm.manage(tctx, ring_quotes.get(pm.pos.token, {}))
                    # v9.7.1: if a CASCADE trade just exited at a loss, tell
                    # SmartLockout the losing z — so a STRONGER re-trigger can
                    # be recognised as trend continuation, not revenge.
                    if (pm.pos is None and cascade_pos_z.get(idx) is not None
                            and pm.risk.realized_pnl < _pnl_before):
                        smart_lock.note_loss(_dir_held, cascade_pos_z[idx],
                                             spot=_spot_ltp)
                    if pm.pos is None:
                        cascade_pos_z[idx] = None
                # continuous trade tracking on its own cadence
                if pm.pos is not None and \
                        time.time() - last_track.get(idx, 0.0) >= config.TRADE_TRACK_S:
                    last_track[idx] = time.time()
                    snap = pm.live_snapshot(tctx,
                                            ring_quotes.get(pm.pos.token, {}))
                    if snap:
                        log.info("%s", snap)
                continue

            # ---- v9.7.1 DISPLACEMENT: while a fly holds the global lock,
            # weigh THIS index's fresh read against the position being held
            # (1 Hz decision cadence; core/displacement.py — every grant and
            # every refusal is named). A grant closes the fly through the
            # normal accounting (close_now → ledger/forward/risk) and falls
            # through: the entry constitution below prices the replacement
            # THIS second through the untouched gates (decision gate,
            # persistence, throttle, RiskGovernor, spread gate, chase cap).
            if (flybook.pos is not None and decide_now and pm.pos is None
                    and getattr(config, "DISP_ENABLED", True)):
                _fm_d = flybook.mark(
                    ts=ts, spot=last_spot.get(flybook.pos.spec.index, 0.0),
                    quotes=_fly_quotes(flybook.pos.spec))
                _bar_d = D.effective_bar(entry_bar, vix_bump,
                                         (mac or {}).get("iv_rank"))
                _pok, _pwhy, _ = persist[idx].check(
                    conv, spot_secs.get(idx, ()),
                    _bar_d + getattr(config, "DISP_CONV_MARGIN", 0.10))
                _v = disp.evaluate(
                    ts=ts, idx=idx, conv=conv, eff_bar=_bar_d,
                    persist_ok=_pok, persist_why=_pwhy,
                    tape_er=last_tape_er.get(idx), cascade_ev=cascade_ev,
                    fly_open_ts=flybook.pos.open_ts,
                    fly_progress_pct=(_fm_d or {}).get("progress_smooth_pct"),
                    fly_unreal=(_fm_d or {}).get("unreal"),
                    fly_pin_frac=(_fm_d or {}).get("pin_frac"),
                    minutes_to_close=mins_left)
                if _v.displace:
                    _crow = flybook.close_now(
                        ts=ts, hm=hm, quotes=_fly_quotes(flybook.pos.spec),
                        reason=f"DISPLACED[{_v.tier}]→{_v.index} {_v.side}")
                    if _crow is not None:
                        disp.register(ts)
                        log.warning(
                            "⇄ DISPLACED fly %s (unreal ₹%+.0f → realized "
                            "₹%+.2f via 4-leg unwind) for %s %s — %s "
                            "[%d/%d today]",
                            _crow["fly_id"],
                            float((_fm_d or {}).get("unreal") or 0.0),
                            _crow["pnl"], _v.index, _v.side, _v.reason,
                            disp.count_today,
                            int(getattr(config, "DISP_MAX_PER_DAY", 2)))
                        report.d["shortvol"]["events"].append(_crow)
                        report.d.setdefault("displacements", []).append({
                            "ts": ts, "hm": hm, "tier": _v.tier,
                            "index": _v.index, "side": _v.side,
                            "fly_id": _crow["fly_id"],
                            "fly_pnl": _crow["pnl"],
                            "reason": _v.reason, "diag": _v.diag})

            # ---- v9.3: spread book management (~5 Hz) + index occupancy ----
            if flybook.pos is not None and flybook.pos.spec.index == idx:
                _crow = flybook.manage(
                    ts=ts, hm=hm, spot=spot,
                    quotes=_fly_quotes(flybook.pos.spec),
                    cascade_event=sv_blocked_now.get(idx, False))
                if _crow is not None:
                    log.warning("Ⓕ BUTTERFLY CLOSE %s %s → ₹%+.2f via %s "
                                "(debit %.2f→credit %.2f, %ds) [%s]", idx,
                                _crow["side"], _crow["pnl"], _crow["why"],
                                _crow["debit"], _crow["close_credit"],
                                _crow["hold_s"], _crow["mode"])
                    report.d["shortvol"]["events"].append(_crow)
                else:
                    skip_reason[idx] = f"in fly {flybook.pos.fly_id}"
                    if decide_now:
                        funnel.record(idx, "in_position", "spread")
                    continue                  # index occupied by the fly

            # v10.1 GLOBAL SINGLE-POSITION LOCK: one live position in the
            # WHOLE system. A fly on either index blocks entries on BOTH.
            if flybook.pos is not None:
                skip_reason[idx] = (f"in fly {flybook.pos.fly_id} "
                                    f"(global lock)")
                if decide_now:
                    funnel.record(idx, "in_position", "fly_global")
                continue

            # ================= 1 Hz ENTRY PATH (gates → attempt) =============
            if not decide_now:
                continue                       # decisions only on ring seconds
            ivr = (mac or {}).get("iv_rank")
            eff_bar = D.effective_bar(entry_bar, vix_bump, ivr)
            if risk.halted:
                skip_reason[idx] = f"risk halted ({risk.halt_reason or 'drawdown'})"
                funnel.record(idx, "risk_halted")
                continue
            # ---- shared attempt tail: ladder → quotes → governor → fill ----
            def _attempt(direction: str, conv_a: float, wp_a: float,
                         gate_desc: str, tag: str) -> None:
                if not mapper:
                    funnel.record(idx, "no_chain", "no kite mapper")
                    return
                hier_rows = mapper.hierarchy(idx, spot, direction)
                if not hier_rows:
                    skip_reason[idx] = "no option chain"
                    funnel.record(idx, "no_chain")
                    return
                quotes = qc.get([(r["exchange"], r["symbol"])
                                 for r in hier_rows])
                T = float(ctx_m.get("T", 0.01))
                F = spot * math.exp(config.RISK_FREE_RATE * T)
                hierarchy: list[LegQuote] = []
                for r in hier_rows:
                    q = quotes.get(r["symbol"])
                    if not q:
                        continue
                    d = q.get("depth") or {}
                    b0 = (d.get("buy") or [{}])[0]
                    s0 = (d.get("sell") or [{}])[0]
                    bid = float(b0.get("price") or 0)
                    ask = float(s0.get("price") or 0)
                    if not (bid and ask):
                        continue
                    mid = (bid + ask) / 2
                    iv = _surface_atm_iv(builder, idx,
                                         ctx_m.get("expiry", ""), T, mac)
                    g = black76_greeks(F, r["strike"], T, max(iv, 0.05),
                                       direction == "CE",
                                       config.RISK_FREE_RATE)
                    hierarchy.append(LegQuote(
                        leg=r["symbol"], symbol=r["symbol"],
                        exchange=r["exchange"], token=r["token"],
                        strike=r["strike"], premium=mid, bid=bid, ask=ask,
                        bid_qty=float(b0.get("quantity") or 0),
                        ask_qty=float(s0.get("quantity") or 0),
                        lot=r["lot"], delta=float(g["delta"]),
                        dte=float(ctx_m.get("dte", 1.0))))
                if not hierarchy:
                    skip_reason[idx] = "no two-sided quotes"
                    funnel.record(idx, "no_quotes")
                    return
                # Seed ring_quotes with the exact freshly-fetched book the
                # decision was priced from, so the paper engine fills against
                # what the brain saw (live fills via Kite, not this dict).
                for _lq in hierarchy:
                    ring_quotes[_lq.token] = {
                        "bid": _lq.bid, "ask": _lq.ask,
                        "bid_qty": _lq.bid_qty, "ask_qty": _lq.ask_qty,
                        "ltp": _lq.premium}
                log.info("%s entry signal — gate: %s | P(win) %.2f | %s",
                         idx, gate_desc, wp_a, tag)
                pm.try_enter(tctx, direction, conv_a, wp_a, hierarchy)
                if pm.pos:
                    skip_reason[idx] = f"in {pm.pos.symbol}"
                    funnel.record(idx, "entered", tag)
                    if tag.startswith("CASCADE"):
                        cascade_entered[idx] += 1
                        report.d["cascade"]["events"][-1]["entered"] = \
                            pm.pos.symbol
                        # remember which trigger opened this position, for the
                        # smart-lockout loss-note on exit
                        cascade_pos_z[idx] = getattr(tctx, "cascade_z", None)
                else:
                    _blk = pm.last_block_reason or "no fill"
                    skip_reason[idx] = _blk
                    funnel.record(idx,
                                  "risk_blocked" if pm.last_block_reason
                                  else "no_fill", f"{tag}: {_blk}")
                    if tag.startswith("CASCADE"):
                        report.d["cascade"]["events"][-1]["skip"] = _blk

            # ---- CASCADE ENTRY (certificate-gated): the structural trigger
            # bypasses the meta gate, persistence and throttle BY DESIGN — the
            # regime state replaces statistical persistence, and the meta was
            # trained on a different (momentum) signal family. Everything
            # downstream is untouched: the ladder, the spread gate, the FULL
            # RiskGovernor (Kelly, floors, curfew, cooldown, halt), the same
            # shaped barriers the harness certified. Direction from structure.
            if cascade_ev is not None and casc_mode != "telemetry":
                last_try[idx] = ts               # normal path stands down 5 s
                # v9.7.1: mark tctx as a cascade entry so the stop widens for
                # short-gamma violence, and ask SmartLockout whether a post-
                # loss lockout should be bypassed (a STRONGER, still-aligned
                # re-trigger is trend continuation, not revenge).
                tctx.from_cascade = True
                tctx.cascade_z = cascade_ev.z
                tctx.net_gex = cascade_ev.net_gex
                _lk = smart_lock.evaluate(
                    ts=ts, direction=cascade_ev.direction, is_cascade=True,
                    cascade_z=cascade_ev.z, spot=spot, flip=cascade_ev.flip,
                    net_gex=cascade_ev.net_gex)
                tctx.lockout_bypass = _lk.bypass
                if _lk.bypass:
                    smart_lock.register_bypass(ts)
                    log.warning("%s cascade lockout BYPASS — %s",
                                idx, _lk.reason)
                # Tier-specific sizing prior: CERTIFIED uses the harness's
                # LOWER-bound win rate; PAPER-EXPLORE (no cert yet) uses the
                # same exploration prior the heuristic ledger was built with —
                # its purpose is forward evidence, and this tier is hard-
                # blocked from ever being live (cascade_mode ∧ four locks).
                _wp_c = (CS.certificate_wp(cascade_cert)
                         if casc_mode == "certified"
                         else config.PAPER_EXPLORE_WINPROB)
                _attempt(cascade_ev.direction,
                         math.copysign(config.CASCADE_ENTRY_CONV,
                                       1 if cascade_ev.direction == "CE"
                                       else -1),
                         _wp_c,
                         f"cascade {cascade_ev.kind} z={cascade_ev.z:+.1f}",
                         f"CASCADE-{casc_mode}-{cascade_ev.kind}")
                if pm.pos is not None:           # forward-evidence join keys
                    CS.log_forward_entry({
                        "symbol": pm.pos.symbol, "entry_ts": pm.pos.entry_ts,
                        "event_ts": cascade_ev.ts, "index": idx,
                        "direction": cascade_ev.direction,
                        "kind": cascade_ev.kind,
                        "z": round(cascade_ev.z, 3), "mode": casc_mode,
                        "flip": round(cascade_ev.flip, 1),
                        "net_gex": cascade_ev.net_gex})
                continue

            # DECISION GATE — model-driven when a trained meta-model exists,
            # else the fixed conviction bar (bootstrap). One shared copy in
            # core/decision.entry_gate — the forge grades with the same bytes.
            gate = D.entry_gate(conv, wp, wp_meta, eff_bar)
            if not gate.ok:
                skip_reason[idx] = gate.reason
                funnel.record(idx, "below_bar", gate.reason)
                continue
            _gate = gate.reason
            wp_res[idx].add(wp)
            # SIGNAL-PERSISTENCE GATE — sustained read, wall-clock window,
            # sampled 1/second exactly like the forge grader.
            _ok, _why, _ = persist[idx].check(conv, spot_secs.get(idx, ()),
                                              gate.floor)
            if not _ok:
                skip_reason[idx] = _why
                funnel.record(idx, "not_persistent", _why)
                log.info("%s signal not persistent — %s", idx, _why)
                continue
            # ---- RETEST-SURVIVAL GATE (fly-intel, BankNifty trap-killer) ----
            # When the fly read says spot is AT a wall and the entry points in
            # the break direction, require a sustained hold since first reaching
            # the wall before arming — so a first-candle breakout that gets
            # retest-wicked can't drag us in at the top. Polarity-agnostic: the
            # retest wick is a trap whether the break rides or fades.
            _fi_e = last_fly_intel.get(idx)
            if (getattr(config, "FLY_INTEL_RETEST_FILTER", True)
                    and _fi_e is not None and getattr(_fi_e, "at_wall", False)
                    and _fi_e.retest_arm_delay_s > 0):
                _cand_side = "CE" if conv > 0 else "PE"
                _break_side = _fi_e.near_wall
                if _cand_side == _break_side:      # entering the break direction
                    _t0 = wall_touch_since.get(idx)
                    if _t0 is None:
                        wall_touch_since[idx] = ts
                        _t0 = ts
                    _held_at_wall = ts - _t0
                    if _held_at_wall < _fi_e.retest_arm_delay_s:
                        _rw = (f"retest guard: held {_held_at_wall:.0f}s < "
                               f"{_fi_e.retest_arm_delay_s:.0f}s at {_break_side} "
                               f"wall")
                        skip_reason[idx] = _rw
                        funnel.record(idx, "retest_guard", _rw)
                        log.info("%s %s — awaiting retest confirmation (%s)",
                                 idx, _cand_side, _rw)
                        continue
            else:
                wall_touch_since[idx] = None       # not at wall ⇒ reset clock
            # ---- ORDER-FLOW TOXICITY TRAP GATE (VPIN/OFI; research-grade) ----
            # Block CHASING into adverse informed flow or an engineered sweep;
            # allow (and flag) a genuine break or a confirmed post-sweep
            # reversal. Thresholds are vault-calibrated per index. Advisory —
            # it can only RAISE the bar here, never lower a floor.
            _tv = last_tox.get(idx)
            if (getattr(config, "TOXICITY_GATE_ENABLED", True)
                    and _tv is not None):
                _thi, _tblk = tox_thresholds(idx)
                _cand_dir = "CE" if conv > 0 else "PE"
                _allow, _twhy = OF.entry_trap_check(
                    _tv, _cand_dir, tox_block=_tblk,
                    sweep_fade_ok=getattr(config, "TOX_SWEEP_FADE_OK", True))
                if not _allow:
                    skip_reason[idx] = _twhy
                    funnel.record(idx, "toxicity_trap", _twhy)
                    log.info("%s %s BLOCKED by trap filter — %s",
                             idx, _cand_dir, _twhy)
                    continue
            if ts - last_try.get(idx, -1e9) < config.ENTRY_ATTEMPT_THROTTLE_S:
                funnel.record(idx, "throttled")
                continue                       # one attempt per 5 s per index
            last_try[idx] = ts
            _attempt("CE" if conv > 0 else "PE", conv, wp, _gate,
                     "MODEL-DRIVEN" if gate.model_driven
                     else "bootstrap (fixed bar)")

    # ---- session end: final diagnostics report ------------------------------
    _write_report(final=True)
    log.info("brain report → %s", report.path)


if __name__ == "__main__":
    main()