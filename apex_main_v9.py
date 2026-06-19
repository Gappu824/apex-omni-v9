"""
APEX OMNI v9 — LIVE BRAIN (audit §8 rebuilt)
============================================
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

Run:  python apex_main_v9.py
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
from core.quant_core import bayesian_signal_fusion
from core.risk_manager import RiskGovernor
from core.execution_engine import ExecutionEngine
from core.position_manager import PositionManager, LegQuote, TickContext
from core.instruments import LiveMapper
from core.heuristic_policy import HeuristicPolicy
from core.quant_core import black76_greeks
from core import regime_classifier as regime_mod

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


def meta_win_prob(frame, i, tod, er, f30, dirn) -> float | None:
    m = load_meta()
    if not m or int(m.get("n", 0)) < config.META_MIN_TRAIN:
        return None
    b0 = i * config.NODES_PER_INDEX
    x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2],
                        [tod, er,
                         math.copysign(min(abs(f30) * 100, 3), f30)
                         if f30 else 0.0,
                         1.0 if dirn > 0 else -1.0]]).astype(np.float32)
    sd = np.asarray(m["sd"], np.float32)
    z = (x - m["mu"]) / np.where(sd > 0.0, sd, 1.0)   # zero-variance dims contribute 0; no /0 → no NaN
    pr = 1.0 / (1.0 + math.exp(-float(z @ m["w"]) - float(m["b"])))
    return float(min(max(pr, config.META_P_FLOOR), config.META_P_CAP))


def load_calibration() -> dict:
    if config.CALIBRATION_TABLE.exists():
        try:
            return json.loads(config.CALIBRATION_TABLE.read_text())
        except Exception:                                 # noqa: BLE001
            pass
    return {}

def win_prob_for(conv: float, table: dict) -> float:
    w = config.CAL_BUCKET_WIDTH
    b = f"{min(abs(conv) // w * w, 1 - w):.2f}"
    if b in table and table[b][1] >= config.CAL_MIN_SAMPLES:
        return float(table[b][0])
    return config.uncalibrated_winprob()

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
    conv_hist: dict[str, deque] = {}                   # rolling conviction window
    from core.quant_core import EWMAVol
    rvol: dict[str, EWMAVol] = {i: EWMAVol() for i in config.INDICES}
    realized_vol_ann: dict[str, float] = {}
    spot_secs: dict[str, deque] = {i: deque(maxlen=1800)
                                   for i in config.TRADABLE}
    last_sec_push: dict[str, int] = {}
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
            if age > config.DATA_STALE_FLATTEN_S:
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
                         "pos %s | (waiting for ticks)", hm, age,
                         risk.realized_pnl,
                         {i: (pms[i].pos.symbol if pms[i].pos else "—")
                          for i in config.TRADABLE})
            continue
        if stale_logged:
            log.info("✓ feed recovered (age %.1fs) — entries resumed", age)
            stale_logged = False

        if time.time() - last_cal_load > config.CAL_RELOAD_S:
            cal = load_calibration(); last_cal_load = time.time()
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
            poss = {i: (pms[i].pos.symbol if pms[i].pos else "—")
                    for i in config.TRADABLE}
            _reg = regime_now.get(config.TRADABLE[0])
            _reg_s = f"{_reg.label}×{_reg.conv_mult:.2f}" if _reg else "—"
            # aggregate walk-away diagnostics across tradable indices
            _run = sum(pms[i]._walkaway_tally["runaway"] for i in config.TRADABLE)
            _bord = sum(pms[i]._walkaway_tally["borderline"] for i in config.TRADABLE)
            _wa = f"{_run}R/{_bord}B" if (_run or _bord) else "0"
            log.info("♥ %s | feed age %.1fs | PnL ₹%+.0f | deployed ₹%.0f | "
                     "halted=%s | pos %s | policy %s | VIX %s | regime %s | "
                     "walkaway %s | drift %s",
                     hm, age, risk.realized_pnl, risk.deployed,
                     risk.halted or risk.halt_reason or False, poss,
                     policy.kind,
                     f"{vix_hist[-1][1]:.2f}" if vix_hist else "—",
                     _reg_s, _wa, drift_grade)

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

        vix = (market.get("_VIX") or {}).get("ltp")
        if vix:
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

        obs = builder.push(market, ts)
        frame = builder.frames[-1]
        drift.observe(frame)
        actions = policy.conviction(obs, frame)

        for idx in config.TRADABLE:
            ctx_m = market.get(idx)
            if not ctx_m or not ctx_m.get("spot"):
                continue
            spot = float(ctx_m["spot"].get("ltp") or 0)
            if spot <= 0:
                continue
            vel = spot - last_spot.get(idx, spot); last_spot[idx] = spot
            if int(ts) != last_sec_push.get(idx):
                last_sec_push[idx] = int(ts)
                spot_secs[idx].append(spot)
                if spot > 0 and idx in rvol:
                    rvol[idx].update(spot, dt_s=1.0)
                    realized_vol_ann[idx] = rvol[idx].annualized()
            open_px.setdefault(idx, spot)
            if idx not in p945 and hm >= "09:45":
                p945[idx] = spot
            i = config.INDEX_ORDER.index(idx)
            ai = float(actions[2 * i])
            # ADVISORY nudges only (audit: ±0.99 force-writes retired)
            mac = read_macro(idx)
            shock = 0.0
            node = frame[i * config.NODES_PER_INDEX]
            if node[4] > config.ADVISORY_VPIN_THRESHOLD:   # vpin
                shock += config.ADVISORY_SHOCK * math.copysign(1.0, node[16] or ai)
            if mac and mac.get("flip"):
                shock += config.ADVISORY_SHOCK * (1.0 if spot > mac["flip"] else -1.0)
            pcr = (mac or {}).get("pcr")
            if pcr is not None:
                if pcr >= config.PCR_HIGH:      # crowded puts → contrarian up
                    shock += config.ADVISORY_SHOCK_PCR
                elif pcr <= config.PCR_LOW:
                    shock -= config.ADVISORY_SHOCK_PCR
            mp = (mac or {}).get("max_pain")
            if mp and float(ctx_m.get("dte", 9.0)) < 1.0:
                shock += config.ADVISORY_SHOCK_MAXPAIN *                     (1.0 if mp > spot else -1.0)   # expiry-day pin gravity
            lv = levels.get(idx)
            if lv:
                if spot > lv["pdh"]:
                    shock += config.ADVISORY_SHOCK_LEVELS
                elif spot < lv["pdl"]:
                    shock -= config.ADVISORY_SHOCK_LEVELS
            f30 = ((p945[idx] - open_px[idx]) / open_px[idx]
                   if idx in p945 and open_px.get(idx) else 0.0)
            if f30 and hm >= config.IMOM_AFTER:   # Gao–Han–Li–Zhou momentum
                shock += config.ADVISORY_SHOCK_IMOM * (1.0 if f30 > 0 else -1.0)
            conv = bayesian_signal_fusion(ai, shock, quant_weight=config.FUSION_QUANT_WEIGHT)
            sh = spot_secs[idx]
            dsum = sum(abs(b - a) for a, b in zip(sh, list(sh)[1:])) \
                if len(sh) > 120 else 0.0
            er = (abs(sh[-1] - sh[0]) / dsum) if dsum > 0 else 0.5

            # ---- REGIME: label the tape from state already in hand and scale
            # conviction by it (never a hard veto; risk floors untouched). This
            # is what dampens momentum signals in a flat, long-gamma tape — the
            # flat-market-bullish-signal problem the live logs showed.
            vfc = None
            try:
                from core.vol_forecaster import forecast as _vol_fcast
                T_y = float(ctx_m.get("T", 0.02))
                vfc = _vol_fcast(idx,
                                 builder.surface.atm_iv(idx, ctx_m.get("expiry", ""), T_y),
                                 front_iv=(mac or {}).get("atm_iv"),
                                 next_iv=(mac or {}).get("atm_iv_next"),
                                 dte=(mac or {}).get("dte"))
            except Exception:                              # noqa: BLE001
                vfc = None
            rv = realized_vol_ann.get(idx)
            regime = regime_mod.classify(
                spot=spot, trend_efficiency=er,
                net_gex=(mac or {}).get("net_gex"), flip=(mac or {}).get("flip"),
                call_wall=(mac or {}).get("call_wall"),
                put_wall=(mac or {}).get("put_wall"),
                iv_rank=(mac or {}).get("iv_rank"), realized_vol=rv,
                vol_regime=(vfc.regime if vfc else None),
                vol_z=(vfc.z if vfc else None))
            regime_now[idx] = regime
            conv = conv * regime.conv_mult                 # scale, never veto
            # signal-PERSISTENCE tracking: a "confident" trade is one where the
            # directional read has HELD, not a single-tick spike. Record the
            # signed conviction each tick; the gate below requires the recent
            # window to agree in direction and average above the bar.
            ch = conv_hist.setdefault(idx, deque(maxlen=config.SIGNAL_PERSIST_N))
            ch.append(conv)
            try:
                regime_mod.log_features(idx, er, (mac or {}).get("net_gex"))
            except Exception:                              # noqa: BLE001
                pass

            mins_open = (dt.datetime.strptime(hm, "%H:%M")
                         - dt.datetime.strptime(config.SESSION_OPEN, "%H:%M")
                         ).seconds / 60.0
            wp_meta = meta_win_prob(frame, i, min(mins_open / 375.0, 1.0),
                                    er, f30, 1 if conv > 0 else -1)
            wp_tab = win_prob_for(conv, cal)
            wp = wp_meta if wp_meta is not None else wp_tab
            w_ = config.CAL_BUCKET_WIDTH
            bkey = f"{min(abs(conv) // w_ * w_, 1 - w_):.2f}"
            if wp_meta is not None and bkey in cal and \
                    cal[bkey][1] >= config.CAL_MIN_SAMPLES:
                wp = 0.5 * (wp_meta + float(cal[bkey][0]))   # blend both judges
            log.debug("%s spot %.1f | ai %+.2f shock %+.2f → conv %+.2f "
                      "(wp %.2f)", idx, spot, ai, shock, conv, wp)

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
            # REAL option-flow shield inputs (audit follow-up): index spot has
            # no volume, so sell-aggression comes from the ATM legs' signed
            # tick-rule flow, and ΔOI from the position-side ATM leg.
            t_ce = builder.trk.get(f"{idx}:atm_ce")
            t_pe = builder.trk.get(f"{idx}:atm_pe")
            opt_flow = ((t_ce.dealer_inv if t_ce else 0.0)
                        + (t_pe.dealer_inv if t_pe else 0.0))
            sell_ratio = float(np.clip(
                0.5 - 0.5 * math.tanh(opt_flow / config.DEALER_INV_SCALE),
                0, 1))
            pm = pms[idx]
            oi_node = node
            if pm.pos is not None:
                oi_node = frame[i * config.NODES_PER_INDEX +
                                (1 if pm.pos.direction == "CE" else 2)]
            mins_left = max((dt.datetime.strptime(config.SESSION_CLOSE, "%H:%M")
                             - dt.datetime.strptime(hm, "%H:%M")).seconds / 60,
                            1.0)
            # model's LIVE read of the HELD position's direction (for the
            # model-shaped exit). Re-evaluated each tick; None unless a trained
            # model exists AND we hold a position. Uses the POSITION's direction,
            # not the current signal's, so a flipping signal can't confuse it.
            wp_hold = None
            if config.META_DECISION_ENABLED and pm.pos is not None:
                wp_hold = meta_win_prob(
                    frame, i, min(mins_open / 375.0, 1.0), er, f30,
                    1 if pm.pos.direction == "CE" else -1)
            tctx = TickContext(
                ts=ts, hm=hm, spot=spot, spot_velocity_1s=vel,
                data_age_s=age,
                atm_iv=builder.surface.atm_iv(idx, ctx_m.get("expiry", ""),
                                              float(ctx_m.get("T", 0.01))),
                minutes_to_close=mins_left,
                gex_put_wall=(mac or {}).get("put_wall"),
                gex_call_wall=(mac or {}).get("call_wall"),
                absorption=absorb, aggressive_sell_ratio=sell_ratio,
                oi_delta_since=float(oi_node[2]),
                avg_spread_pct=spread_ew[idx], conviction=conv,
                live_win_prob=wp_hold,
                regime_label=(regime.label if regime else ""))

            for oid, fill in engine.on_quote(
                    pm.pos.token if pm.pos else -1,
                    ring_quotes.get(pm.pos.token, {}) if pm.pos else {}):
                log.info("resting order %s → %s", oid, fill.status)
            if pm.pos:
                if risk.halted:
                    pm._exit(tctx, ring_quotes.get(pm.pos.token, {}),
                             "RISK_HALT", urgent=True)
                else:
                    pm.manage(tctx, ring_quotes.get(pm.pos.token, {}))
                # continuous trade tracking: stream the live position read
                # (PnL, distance to stop/target, OI, trap, P(win)) on its own
                # cadence while in a trade — so you can SEE the trade evolving.
                if pm.pos is not None and \
                        time.time() - last_track.get(idx, 0.0) >= config.TRADE_TRACK_S:
                    last_track[idx] = time.time()
                    snap = pm.live_snapshot(tctx,
                                            ring_quotes.get(pm.pos.token, {}))
                    if snap:
                        log.info("%s", snap)
                continue
            ivr = (mac or {}).get("iv_rank")
            eff_bar = entry_bar + vix_bump +                 (config.IVRANK_BAR_BUMP
                 if ivr is not None and ivr >= config.IVRANK_HIGH else 0.0)
            if risk.halted:
                continue
            # DECISION GATE — model-driven when a trained meta-model exists,
            # else the fixed conviction bar (bootstrap). The risk envelope
            # (halt above; size/stop/floor downstream) bounds BOTH paths.
            if config.META_DECISION_ENABLED and wp_meta is not None:
                # trained model live → the model's calibration-blended P(win)
                # decides, above a minimal directional floor so it never acts on
                # noise. Threshold-free in the meaningful range; no hand-set bar.
                if abs(conv) < config.META_ENTRY_CONV_FLOOR \
                        or wp < config.META_ENTRY_P_BAR:
                    continue
                _gate = f"meta P(win) {wp:.2f}≥{config.META_ENTRY_P_BAR:.2f}"
            else:
                # bootstrap: no trained model yet → fixed conviction bar
                if abs(conv) < eff_bar:
                    continue
                _gate = f"conv {abs(conv):.2f}≥{eff_bar:.2f}"
            # SIGNAL-PERSISTENCE GATE — the instantaneous conviction cleared the
            # bar, but is the read SUSTAINED or a one-tick spike? Require the
            # recent window to (a) agree in sign and (b) average above the
            # persistence floor. This is the difference between a confident trade
            # and getting whipsawed in a choppy tape. Skipped until the window
            # has filled (warm-up) so it doesn't block the first valid signals.
            if config.SIGNAL_PERSIST_ENABLED:
                ch = conv_hist.get(idx)
                if ch is not None and len(ch) >= config.SIGNAL_PERSIST_N:
                    same_dir = sum(1 for c in ch if (c > 0) == (conv > 0))
                    avg_conv = sum(abs(c) for c in ch) / len(ch)
                    frac_agree = same_dir / len(ch)
                    if frac_agree < config.SIGNAL_PERSIST_FRAC:
                        log.info("%s signal not persistent — direction agreed "
                                 "%.0f%% of last %d ticks (need %.0f%%); skipping "
                                 "whipsaw", idx, frac_agree * 100, len(ch),
                                 config.SIGNAL_PERSIST_FRAC * 100)
                        continue
                    if avg_conv < eff_bar * config.SIGNAL_PERSIST_AVG_MULT:
                        log.info("%s signal not persistent — avg |conv| %.2f over "
                                 "last %d ticks below %.2f; skipping spike",
                                 idx, avg_conv, len(ch),
                                 eff_bar * config.SIGNAL_PERSIST_AVG_MULT)
                        continue
            if ts - last_try.get(idx, -1e9) < config.ENTRY_ATTEMPT_THROTTLE_S:
                continue                       # one attempt per 5 s per index
            last_try[idx] = ts
            direction = "CE" if conv > 0 else "PE"
            if not mapper:
                continue
            hier_rows = mapper.hierarchy(idx, spot, direction)
            if not hier_rows:
                continue
            quotes = qc.get([(r["exchange"], r["symbol"]) for r in hier_rows])
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
                bid, ask = float(b0.get("price") or 0), float(s0.get("price") or 0)
                if not (bid and ask):
                    continue
                mid = (bid + ask) / 2
                iv = builder.surface.atm_iv(idx, ctx_m.get("expiry", ""), T)
                g = black76_greeks(F, r["strike"], T, max(iv, 0.05),
                                   direction == "CE", config.RISK_FREE_RATE)
                hierarchy.append(LegQuote(
                    leg=r["symbol"], symbol=r["symbol"], exchange=r["exchange"],
                    token=r["token"], strike=r["strike"], premium=mid,
                    bid=bid, ask=ask,
                    bid_qty=float(b0.get("quantity") or 0),
                    ask_qty=float(s0.get("quantity") or 0),
                    lot=r["lot"], delta=float(g["delta"]),
                    dte=float(ctx_m.get("dte", 1.0))))
            if hierarchy:
                # The engine's paper quote_fn reads ring_quotes by token. The
                # harvester only streams the ATM legs into the ring, so a chosen
                # strike the ring doesn't track (common on SENSEX) would return an
                # empty quote at fill time → the crossing order can't fill and
                # walks away. Seed ring_quotes with the exact, freshly-fetched
                # quotes the decision was priced from so the engine fills against
                # the same book the brain saw. (Live mode fills via Kite, not
                # this dict, so this only corrects the paper path.)
                for _lq in hierarchy:
                    ring_quotes[_lq.token] = {
                        "bid": _lq.bid, "ask": _lq.ask,
                        "bid_qty": _lq.bid_qty, "ask_qty": _lq.ask_qty,
                        "ltp": _lq.premium}
                log.info("%s entry signal — gate: %s | P(win) %.2f | %s",
                         idx, _gate, wp,
                         "MODEL-DRIVEN" if (config.META_DECISION_ENABLED
                                            and wp_meta is not None)
                         else "bootstrap (fixed bar)")
                pm.try_enter(tctx, direction, conv, wp, hierarchy)


if __name__ == "__main__":
    main()
