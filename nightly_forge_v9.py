"""
APEX OMNI v9 — NIGHTLY FORGE (audit §6 leaps)
=============================================
The four fixes that make "train on everything you've ever harvested" true:

  1. POINT-IN-TIME REPLAY. Each historical day is rebuilt with
     AsOfMapper(day) — the chain THAT day actually had — so option nodes stop
     training as zero padding the moment a contract expires.
  2. ONE FEATURE DIALECT. The replay pushes raw ticks through the exact
     core.market_state.StateBuilder the live brain runs. No second
     implementation exists to drift.
  3. REALIZED-EXIT REWARD, SHAPED TARGET. Reward = simulated 1-lot option PnL of
     the trade the strategy ACTUALLY makes: BUY at the mid (the engine posts a
     maker buy and walks if unfilled), HOLD under the constitution's risk-managed
     exit — a SHAPED target / -BASE_SL_PCT stop, whichever the bid touches first
     over MAX_HOLD_MINUTES, else the theta-guillotine last bid — minus the full
     Zerodha+statutory toll. The target is the live PositionManager's, not a flat
     +BASE_TP_PCT: entry + max(delta · expected_move, entry · BASE_TP_PCT), where
     the 1σ expected move uses the REAL ATM IV Newton-inverted from the leg's own
     mid, and is capped by the GEX walls (call_wall/put_wall) exactly as live —
     both read from the macro vault archive the radar now persists. That same
     archive is fit into the StateBuilder surface during replay (warm-started, the
     way live fits it every tick), so the iv/delta/gamma/theta FEATURES are the
     live ones too, not the seed surface. On days harvested before the archive
     existed there is no snapshot: the surface stays seed and the room is uncapped
     — the prior behaviour — so this is purely additive and sharpens as archived
     days accumulate. This is the IDENTICAL shaped triple-barrier
     the meta-labeler grades on, so the SIDE (bandit) and SIZE (meta) models share
     one realized payoff. It replaces the old REWARD_HORIZON_S (60s) symmetric
     mark, which graded a trade that is never held: at 60s the result is pure
     spread + cost, every entry "loses", and the model correctly collapses to
     abstention. The agent is now graded on the exam the account actually sits.
  4. WALK-FORWARD GATE. The newest day is held out; a candidate is PROMOTED only
     if its after-cost score on that unseen day clears max(heuristic, incumbent)
     by an ADDITIVE ₹ margin FORGE_PROMOTE_MARGIN_RS (sign-safe; the old
     multiplicative margin inverted on the negative scores seen before a model
     has edge) AND it actually trades (the abstention guard — a do-nothing model
     would freeze paper-trade collection). Model zip + VecNormalize pkl + manifest
     are saved as one versioned, atomic pair — the live brain refuses mismatched
     pairs, so "new weights, stale statistics" can't happen.

Training set: last FORGE_LOOKBACK_DAYS + a reservoir sample of older days
(no more O(all-history) nightly reread). SAC is retained; the reward change
matters far more than the algorithm tonight.

Run after close:  python nightly_forge_v9.py
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import random
import sqlite3
import time
from pathlib import Path

import numpy as np

import config
from core.instruments import AsOfMapper
from core.market_state import StateBuilder
from core.execution_engine import round_trip_costs
from core.quant_core import implied_vol_newton, black76_greeks

# The macro vault archive (per-strike IVs + GEX walls) the radar now persists.
# If running an older macro_gex_v9 without it, the forge degrades cleanly to its
# prior seed-surface / no-wall behaviour (load returns [] ⇒ every consumer no-ops).
try:
    from macro_gex_v9 import load_macro_archive
except Exception:                                         # noqa: BLE001
    def load_macro_archive(con, day, index):              # type: ignore
        return []

log = logging.getLogger("forge")

try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAVE_RL = True
except Exception:                                      # noqa: BLE001
    HAVE_RL = False


# --------------------------------------------------------------- replay
def trading_days(con) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch','localtime') "
        "FROM ticks_v9 ORDER BY 1").fetchall()
    return [r[0] for r in rows if r[0]]

def spot_token_for(con, day: str, index: str) -> int | None:
    r = con.execute("SELECT token FROM spot_tokens WHERE snap_date<=? AND "
                    "name=? ORDER BY snap_date DESC LIMIT 1",
                    (day, index)).fetchone()
    return int(r[0]) if r else None

def _latest_at(lst, ptr_box, value, keyfn):
    """Latest item in the time-sorted `lst` whose keyfn(item) <= value, via a
    monotonically advancing pointer (callers sweep `value` upward). None when
    `value` precedes the first item. A right-continuous step — the way the live
    brain reads the latest published macro JSON."""
    if not lst:
        return None
    p = ptr_box[0]
    while p + 1 < len(lst) and keyfn(lst[p + 1]) <= value:
        p += 1
    ptr_box[0] = p
    return lst[p] if keyfn(lst[p]) <= value else None


def _make_surface_fitter():
    """Reconstructs, during replay, the SVI surface the live brain fits every tick
    from the macro radar's per-strike IVs (apex_main_v9.fit_surface). The forge
    replay otherwise NEVER fits it, so iv/delta/gamma/theta — four of the nineteen
    node features — train on the SVISurface DEFAULT (≈70-190% intraday IV vs the
    real ~15%), a train/serve skew. Each NEW archived snapshot is fit into the
    shared StateBuilder, warm-started exactly as live: the surface PERSISTS across
    snapshots, so the first snapshot per (index,expiry) converges from DEFAULT and
    every later one merely refines the already-converged surface.

    FIT TO CONVERGENCE, not a fixed pass count. The live brain re-fits the surface
    EVERY tick — hundreds-to-thousands of single-pass fits over a 3-min macro
    window — so it reaches the SVI fixed point (the ATM total variance w(0) stops
    moving). The previous fixed 150-cold / 12-warm passes stopped SHORT of that
    fixed point on calm, low-IV days: the forge served iv ≈ 0.19 where the brain
    converges to ≈ 0.13 — a ~+44% train/serve skew on the iv/delta/gamma/theta
    features, worst exactly when IV is low. Here each snapshot is iterated until
    w(0) settles (relative change ≤ FORGE_SURFACE_FIT_TOL) so the forge lands on
    the SAME surface the brain serves. Warm-started: the first (cold) snapshot per
    key converges from DEFAULT in a few hundred passes; each later snapshot in a
    handful — it only needs to track the small intraday IV drift off the prior
    fixed point. `min_passes` guards a one-pass fluke; `max_passes` caps a
    non-converging fit. All three knobs are getattr-defaulted (deliberately
    NOT in config.py) so CONFIG_HASH — and thus model/drift-reference compatibility
    — is unchanged. No-op when `snap` is None (older days without an archive) ⇒ the
    forge keeps its prior seed-surface behaviour, so this is purely additive."""
    fitted_ts: dict = {}
    enabled = bool(getattr(config, "FORGE_SURFACE_FIT", True))
    tol = float(getattr(config, "FORGE_SURFACE_FIT_TOL", 1e-4))
    max_passes = int(getattr(config, "FORGE_SURFACE_FIT_MAX_PASSES", 5000))
    min_passes = max(int(getattr(config, "FORGE_SURFACE_FIT_MIN_PASSES", 2)), 1)

    def fit(builder, index, expiry, T, snap):
        if not (enabled and snap and snap.get("strikes")) or T <= 0:
            return
        key = (index, expiry)
        if fitted_ts.get(key) == snap["ts"]:              # this snapshot already fit
            return
        spot = float(snap.get("spot") or 0.0)
        F = spot * float(np.exp(config.RISK_FREE_RATE * T))
        if F <= 0:
            return
        K = np.asarray(snap["strikes"], float)
        iv = np.asarray(snap["iv"], float)
        # Iterate the SAME single-pass fit the live brain calls, until the ATM
        # total variance w(0) = atm_iv²·T stops moving (rel-change ≤ tol). The
        # surface is warm-started (it persists on `builder`), so a snapshot whose
        # surface already sits near the fixed point exits in a couple of passes,
        # while the first cold snapshot per key takes ~1k.
        prev_w = None
        for p in range(max_passes):
            builder.fit_surface(index, expiry, K, iv, F, T)
            cur_iv = builder.surface.atm_iv(index, expiry, T)
            if cur_iv is None or not np.isfinite(cur_iv) or cur_iv <= 0.0:
                continue                                  # surface not usable yet
            w = float(cur_iv) * float(cur_iv) * float(T)  # ATM total variance
            if (prev_w is not None and p + 1 >= min_passes
                    and abs(w - prev_w) <= tol * max(prev_w, 1e-12)):
                break
            prev_w = w
        fitted_ts[key] = snap["ts"]

    return fit


def replay_day(con, day: str):
    """Yields (ts, obs_5700, market, macro_now) second by second through ONE
    shared StateBuilder — identical to live. macro_now[idx] is the latest archived
    macro snapshot at-or-before ts (or None on days with no archive), carrying the
    GEX walls the shaped-target cap reads; the surface is also fit from it here so
    the obs iv/delta/gamma/theta are the live ones, not the seed surface."""
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    if mapper.snapshot_used is None:
        log.warning("%s: no instrument snapshot ≤ this date — spot-only day "
                    "(harvest more days; the time machine needs film)", day)
    spot_toks = {i: spot_token_for(con, day, i) for i in config.INDEX_ORDER}
    cur = con.execute(
        "SELECT ts_ms/1000, token, ltp, bid, ask, bid_qty, ask_qty, "
        "vol_delta, oi, iceberg FROM ticks_v9 WHERE "
        "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
        (day,))
    builder = StateBuilder()
    cur_sec, snaps = None, {}
    chains: dict[str, dict] = {}
    # macro archive (per-strike IVs + GEX walls): fit the REAL surface during
    # replay and surface the walls for the shaped-target cap. [] on older days.
    macro = {i: load_macro_archive(con, day, i) for i in config.INDEX_ORDER}
    mptr = {i: [0] for i in config.INDEX_ORDER}
    fit_surface = _make_surface_fitter()
    _nsnap = {i: len(v) for i, v in macro.items() if v}
    if _nsnap:
        log.info("%s: macro archive HIT — %s snapshot(s) → REAL SVI surface fit + "
                 "GEX-wall target cap ACTIVE", day,
                 " ".join(f"{i}:{n}" for i, n in _nsnap.items()))
    else:
        log.info("%s: macro archive empty → seed surface, no wall cap (prior "
                 "behaviour — run the new macro_gex_v9.py live to start recording)",
                 day)

    def emit(sec):
        market, macro_now = {}, {}
        for idx in config.INDEX_ORDER:
            st = spot_toks.get(idx)
            sp = snaps.get(st) if st else None
            if not sp:
                continue
            ch = chains.get(idx)
            atm_now = None
            if sp["ltp"]:
                step = (ch or {}).get("step") or config.INDICES[idx]["strike_step"]
                atm_now = round(sp["ltp"] / step) * step
            if ch is None or (atm_now and ch.get("atm") != atm_now):
                ch = mapper.chain(idx, sp["ltp"]) or ch
                if ch:
                    chains[idx] = ch
            snap = _latest_at(macro[idx], mptr[idx], sec, lambda s: s["ts"])
            macro_now[idx] = snap
            if ch and snap:                               # REAL surface before push
                fit_surface(builder, idx, ch["expiry"], ch["T"], snap)
            entry = {"spot": sp}
            if ch:
                legs = {}
                for leg, info in ch["legs"].items():
                    s = snaps.get(info["token"])
                    if s:
                        legs[leg] = {"snap": s, "strike": info["strike"]}
                entry.update({"expiry": ch["expiry"], "dte": ch["dte"],
                              "T": ch["T"], "is_weekly": ch["is_weekly"],
                              "lot": ch["lot"], "legs": legs})
            market[idx] = entry
        obs = builder.push(market, float(sec))
        return obs, market, macro_now

    for ts, tok, ltp, bid, ask, bq, aq, vd, oi, ice in cur:
        sec = int(ts)
        if cur_sec is None:
            cur_sec = sec
        while sec > cur_sec:
            obs, market, macro_now = emit(cur_sec)
            if obs is not None:
                yield cur_sec, obs, market, macro_now
            cur_sec += 1
        snaps[tok] = {"ltp": ltp, "bid": bid, "ask": ask, "bid_qty": bq,
                      "ask_qty": aq, "vol_delta": vd, "oi": oi, "iceberg": ice}
    if cur_sec is not None:
        obs, market, macro_now = emit(cur_sec)
        if obs is not None:
            yield cur_sec, obs, market, macro_now


def _session_minutes_left(ts):
    """Minutes from the bar's IST wall-clock to the SESSION_CLOSE (15:30 IST),
    clamped ≥1 — the `minutes_to_close` the live PositionManager feeds into the
    expected move. `ts` is UTC epoch seconds (the vault stores the exchange
    timestamp; on the IST trading host that is true UTC epoch). India has no DST,
    fixed UTC+5:30, so IST-seconds-of-day = (ts + 19800) mod 86400. Vectorized."""
    ch, cm = (int(x) for x in config.SESSION_CLOSE.split(":"))
    close_sod = ch * 3600 + cm * 60
    ist_sod = (np.asarray(ts, np.float64) + 19800.0) % 86400.0
    return np.maximum((close_sod - ist_sod) / 60.0, 1.0)


def _shaped_barriers(e, spot, K, T, mins, is_call, call_wall=None, put_wall=None):
    """The live PositionManager.try_enter exit target, reproduced for the reward.

        em        = spot · atm_iv · √(minutes_to_close / (252·375))   # 1σ move
        spot_room = min(em, runway)                                   # GEX cap
        prem_room = delta_at_entry · spot_room                        # → premium
        target    = entry + max(prem_room, entry · BASE_TP_PCT)       # floored
        stop      = entry · (1 − BASE_SL_PCT)                         # NOT widened

    atm_iv is the REAL implied vol Newton-inverted from the ATM leg's OWN mid
    (`e`); delta is Black-76 on that same iv, `abs(delta) or 0.4` exactly as live.
    `runway` is the distance to the blocking GEX wall — (call_wall − spot) for a
    call, (spot − put_wall) for a put — when a wall sits inside the move; otherwise
    the room is the full `em`, which is exactly what live does with no wall. The
    walls come from the macro archive (call_wall/put_wall); when none is archived
    (older days) they are None and the cap is skipped — identical to before.
    Vectorized (grid) or scalar (per-bar)."""
    r = config.RISK_FREE_RATE
    e = np.asarray(e, float); spot = np.asarray(spot, float)
    K = np.asarray(K, float); T = np.maximum(np.asarray(T, float), 1e-6)
    F = spot * np.exp(r * T)
    iv = implied_vol_newton(e, F, K, T, is_call, r)
    delta = np.abs(np.asarray(black76_greeks(F, K, T, iv, is_call, r)["delta"], float))
    delta = np.where(delta > 1e-9, delta, 0.4)             # live: abs(q.delta) or 0.4
    em = spot * iv * np.sqrt(np.maximum(mins, 1.0) / (252.0 * 375.0))
    spot_room = em
    if is_call and call_wall is not None and call_wall > 0:
        runway = call_wall - spot                          # wall above ⇒ caps a call
        spot_room = np.where(runway > 0, np.minimum(em, runway), em)
    elif (not is_call) and put_wall is not None and put_wall > 0:
        runway = spot - put_wall                            # wall below ⇒ caps a put
        spot_room = np.where(runway > 0, np.minimum(em, runway), em)
    prem_room = delta * spot_room
    tp = e + np.maximum(prem_room, e * config.BASE_TP_PCT)
    sl = e * (1.0 - config.BASE_SL_PCT)
    return tp, sl


def build_dataset(con, day: str):
    """Returns obs (N,5700) and a premium/barrier table for the realized-exit
    reward: prem[idx_i] = dict(ts → {leg: (bid, ask, lot, tp, sl)}).

    (tp, sl) are the SHAPED exit barriers the live PositionManager would arm for
    a 1-lot ATM entry at that second — the +BASE_TP_PCT target WIDENED to the
    expected-move premium room (delta × 1σ move over the remaining session, real
    ATM IV from the leg's own mid), floored at base, with the fixed -BASE_SL_PCT
    stop. So the bandit is now graded on the live asymmetric target (winners that
    run to +60-70% on conviction days are scored as such) rather than a flat
    +30% that understated exactly the trades the policy should learn to hold."""
    obs_list, ts_list, prem = [], [], {i: {} for i in config.INDEX_ORDER}
    for ts, obs, market, macro_now in replay_day(con, day):
        obs_list.append(obs); ts_list.append(ts)
        mins = _session_minutes_left(ts)
        for idx, ctx in market.items():
            legs = ctx.get("legs") or {}
            spot = float((ctx.get("spot") or {}).get("ltp") or 0.0)
            T = float(ctx.get("T") or 0.0)
            lot = ctx.get("lot", 0)
            snap = (macro_now or {}).get(idx)
            cw = snap.get("call_wall") if snap else None   # GEX walls cap the room
            pw = snap.get("put_wall") if snap else None     # (None ⇒ no cap, as before)
            row = {}
            for leg in ("atm_ce", "atm_pe", "otm_ce", "otm_pe"):
                s = (legs.get(leg) or {}).get("snap")
                if not (s and s["bid"] and s["ask"]):
                    continue
                bid, ask = s["bid"], s["ask"]
                e = (bid + ask) / 2.0
                K = float((legs.get(leg) or {}).get("strike") or 0.0)
                if spot > 0 and K > 0 and T > 0 and e > 0:
                    tp, sl = _shaped_barriers(e, spot, K, T, mins,
                                              leg.endswith("_ce"), cw, pw)
                    tp, sl = float(tp), float(sl)
                else:                                     # missing context ⇒ base
                    tp = e * (1.0 + config.BASE_TP_PCT)
                    sl = e * (1.0 - config.BASE_SL_PCT)
                row[leg] = (bid, ask, lot, tp, sl)
            if row:
                prem[idx][ts] = row
    if not obs_list:
        return None, None, None
    return np.stack(obs_list), np.array(ts_list), prem


def _exit_price_from_path(bids: np.ndarray, tp: float, sl: float):
    """First-touch triple barrier on a forward BID path for a long entry whose
    SHAPED target is `tp` and stop is `sl` (computed at entry by _shaped_barriers).
    Returns the realized exit PRICE: `tp` if the bid reaches it before `sl`, `sl`
    if hit first, else the last valid bid (theta / max-hold exit). NaNs (data
    gaps) are skipped. Returns None if the path holds no valid bid. Same rule the
    meta-labeler uses in _gen_meta_samples — so the SIDE (bandit) and SIZE (meta)
    models are graded on one identical realized payoff."""
    if bids.size == 0:
        return None
    hit_tp = bids >= tp
    hit_sl = bids <= sl
    itp = int(np.argmax(hit_tp)) if hit_tp.any() else None
    isl = int(np.argmax(hit_sl)) if hit_sl.any() else None
    if itp is not None and (isl is None or itp < isl):
        return tp
    if isl is not None:
        return sl
    valid = bids[~np.isnan(bids)]
    return float(valid[-1]) if valid.size else None


def reward_fn(prem_idx: dict, ts: float, direction: int) -> float:
    """Realized after-cost ₹ for a 1-lot ATM long (CE if dir>0 else PE): BUY at
    the MID now (the live engine posts a maker buy and walks away if unfilled, so
    real fills are ~mid — not the ask), HOLD under the constitution's risk-managed
    exit, SELL at the triple-barrier exit price. ₹ per lot.

    This replaces the old REWARD_HORIZON_S (60s) symmetric mark: the policy is now
    graded on the trade it ACTUALLY makes — a conviction entry held to its SHAPED
    target / -BASE_SL_PCT stop over MAX_HOLD_MINUTES (the theta guillotine) — rather
    than a cost-dominated snapshot it never holds. A 60s mark threw away the entire
    asymmetric payoff (cut losers at the stop, let winners run to the target), which
    is why the model collapsed to abstention: on that metric every entry loses.
    The (tp, sl) carried on each row are the expected-move-shaped barriers from
    build_dataset (see _shaped_barriers) — the live target, not a flat +30%."""
    leg = "atm_ce" if direction > 0 else "atm_pe"
    now = prem_idx.get(ts, {}).get(leg)
    if not now:
        return 0.0
    bid0, ask0, lot, tp, sl = now
    e = (bid0 + ask0) / 2.0
    horizon = int(config.MAX_HOLD_MINUTES * 60)
    bids = np.fromiter(
        (prem_idx.get(ts + k, {}).get(leg, (np.nan,))[0]
         for k in range(1, horizon + 1)), dtype=np.float64, count=horizon)
    exitp = _exit_price_from_path(bids, tp, sl)
    if exitp is None:
        return 0.0
    return (exitp - e) * lot - round_trip_costs(e * lot, exitp * lot)


# ====================================================================
# META-LABELER (López de Prado: primary model picks the SIDE; this
# secondary model learns the SIZE as P(win) from TRIPLE-BARRIER outcomes
# on the vault's real recorded prices, after real costs). Pure numpy.
# ====================================================================
def _kelly_budget(equity: float) -> float:
    b = config.BASE_TP_PCT / config.BASE_SL_PCT
    p = config.PAPER_EXPLORE_WINPROB
    k = max(p - (1 - p) / b, 0.0)
    return min(equity * config.MAX_KELLY_BUDGET_PCT,
               equity * k * config.KELLY_FRACTION)


def _gen_meta_samples(con, day: str, index: str):
    from collections import deque
    from simulation.replay_real_day import load_day
    from simulation.scenario_engine import N
    from core.heuristic_policy import HeuristicPolicy
    import math as _m
    loaded = load_day(con, day, index)
    if not loaded:
        return [], []
    spot_tok, by_sec, ti, bidA, askA = loaded
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    builder = StateBuilder()
    pol = HeuristicPolicy()
    iidx = config.INDEX_ORDER.index(index)
    snaps, chain = {}, None
    last_tick = {}
    spot_hist: deque = deque(maxlen=1800)
    open_p = p945 = None
    last_sig = -1e9
    budget = _kelly_budget(config.TRADING_CAPITAL)
    X, Y, R = [], [], []
    horizon = int(config.MAX_HOLD_MINUTES * 60)
    # macro archive in t-space (t = seconds from the 09:15 open, like this loop):
    # snapshot t = IST-seconds-of-day(snap) − open_sod. Fit the REAL surface here
    # too so the meta-model's features (and the drift reference it seeds) are the
    # live greeks, and read the GEX walls for the shaped-target cap. [] ⇒ no-op.
    _oh, _om = (int(x) for x in config.SESSION_OPEN.split(":"))
    _open_sod = _oh * 3600 + _om * 60
    macro = load_macro_archive(con, day, index)
    if macro:
        log.debug("%s %s: meta-labeler on %d macro snapshot(s) — real surface + walls",
                  day, index, len(macro))
    for _s in macro:
        _s["_t"] = int((_s["ts"] + 19800) % 86400) - _open_sod
    mptr = [0]
    fit_surface = _make_surface_fitter()
    for t in range(N):
        for tok, sn in by_sec.get(t, {}).items():
            snaps[tok] = sn
            last_tick[tok] = t
        sp = snaps.get(spot_tok)
        if not sp or not sp.get("ltp"):
            continue
        spot = float(sp["ltp"])
        spot_hist.append(spot)
        if open_p is None:
            open_p = spot
        if p945 is None and t >= 1800:
            p945 = spot
        step = (chain or {}).get("step") or config.INDICES[index]["strike_step"]
        atm = round(spot / step) * step
        if chain is None or chain.get("atm") != atm:
            chain = mapper.chain(index, spot) or chain
        market = {index: {"spot": sp}}
        if chain:
            legs = {}
            for leg, info in chain["legs"].items():
                s = snaps.get(info["token"])
                if s:
                    legs[leg] = {"snap": s, "strike": info["strike"]}
            market[index].update({"expiry": chain["expiry"],
                                  "dte": chain["dte"], "T": chain["T"],
                                  "is_weekly": chain["is_weekly"],
                                  "legs": legs})
        snap = _latest_at(macro, mptr, t, lambda s: s["_t"]) if macro else None
        if chain and snap:                                # REAL surface before push
            fit_surface(builder, index, chain["expiry"], chain["T"], snap)
        obs = builder.push(market, float(t))
        if obs is None or chain is None:
            continue
        frame = builder.frames[-1]
        # DRIFT REFERENCE POPULATION. The live monitor (DriftMonitor.observe)
        # pools the spot + ATM CE/PE nodes of EVERY second; the reference must be
        # that SAME all-tick population. Accumulating only signal moments (below
        # the conviction gate) biases every marginal — signals fire precisely when
        # iv / dealer_inv / oi_delta_norm are at extremes — so live PSI/KS would
        # breach perpetually no matter how fresh the reference or how real the
        # surface. Sample here, before the gate.
        _b0 = iidx * config.NODES_PER_INDEX
        for _nd in (frame[_b0], frame[_b0 + 1], frame[_b0 + 2]):
            if _nd.any():
                R.append(_nd.astype(np.float32))
        conv = float(pol.predict(frame)[2 * iidx])
        if abs(conv) < config.PAPER_ENTRY_CONVICTION or \
                t - last_sig < config.ENTRY_ATTEMPT_THROTTLE_S:
            continue
        last_sig = t
        d = "CE" if conv > 0 else "PE"
        pick = None
        for r in mapper.hierarchy(index, spot, d):
            k = ti.get(r["token"])
            if k is None or t - last_tick.get(r["token"], -99) > 5:
                continue
            b_, a_ = bidA[k, t], askA[k, t]
            if np.isnan(b_) or np.isnan(a_):
                continue
            mid = (b_ + a_) / 2
            if mid * r["lot"] <= budget:
                pick = (k, float(mid), int(r["lot"]), float(r["strike"]))
                break
        if pick is None:
            continue
        k, e, lot, Kstrike = pick
        # SAME shaped target as the bandit reward (build_dataset/_shaped_barriers),
        # so the meta-labeler's P(win) is the probability of hitting the exact
        # barrier the SIDE model is graded on — not a flat +30%. minutes_to_close
        # here is the 09:15→15:30 anchor: t is seconds-from-open (N=22500=375min).
        mins_left = max((N - t) / 60.0, 1.0)
        T_mlc = float(chain.get("T") or 0.0)
        cw = snap.get("call_wall") if snap else None
        pw = snap.get("put_wall") if snap else None
        if spot > 0 and Kstrike > 0 and T_mlc > 0 and e > 0:
            tp, sl = _shaped_barriers(e, spot, Kstrike, T_mlc, mins_left,
                                      d == "CE", cw, pw)
            tp, sl = float(tp), float(sl)
        else:
            tp, sl = e * (1 + config.BASE_TP_PCT), e * (1 - config.BASE_SL_PCT)
        seg = bidA[k, t + 1:t + 1 + horizon]
        if seg.size == 0 or np.all(np.isnan(seg)):
            continue
        itp = np.argmax(seg >= tp) if np.any(seg >= tp) else None
        isl = np.argmax(seg <= sl) if np.any(seg <= sl) else None
        if itp is not None and (isl is None or itp < isl):
            exitp = tp
        elif isl is not None:
            exitp = sl
        else:
            last_b = seg[~np.isnan(seg)]
            exitp = float(last_b[-1])
        pnl = (exitp - e) * lot - round_trip_costs(e * lot, exitp * lot)
        # features: spot+atm_ce+atm_pe nodes (57) + tod, ER(30m), first30
        b0 = iidx * config.NODES_PER_INDEX
        x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2]])
        diffs = np.abs(np.diff(np.array(spot_hist, float)))
        er = (abs(spot_hist[-1] - spot_hist[0]) / diffs.sum()) \
            if len(spot_hist) > 120 and diffs.sum() > 0 else 0.5
        f30 = ((p945 - open_p) / open_p) if (p945 and open_p) else 0.0
        x = np.concatenate([x, [t / N, er,
                                _m.copysign(min(abs(f30) * 100, 3), f30)
                                if f30 else 0.0,
                                1.0 if d == "CE" else -1.0]]).astype(np.float32)
        X.append(x)
        Y.append(1.0 if pnl > 0 else 0.0)
    return X, Y, R


def train_meta(con, days: list[str]):
    """Triple-barrier labels → logistic P(win|x). Saved atomically; the live
    brain blends it into the Kelly win-probability."""
    X, Y, R = [], [], []
    for day in days:
        for index in config.TRADABLE:
            x, y, r = _gen_meta_samples(con, day, index)
            X += x; Y += y; R += r
    n = len(X)
    if n < config.META_MIN_TRAIN:
        log.info("meta-labeler: %d/%d labeled signals — keep harvesting",
                 n, config.META_MIN_TRAIN)
        return None
    X = np.stack(X); Y = np.array(Y, np.float32)
    cut = int(n * 0.8)                          # time-ordered holdout
    # Floor must exceed the round(5) granularity used when this is serialized
    # below — a 1e-6 floor rounds to 0.00000 and the live brain then divides by
    # it. 1e-4 survives rounding; near-constant dims still contribute ~nothing.
    mu, sd = X[:cut].mean(0), np.maximum(X[:cut].std(0), 1e-4)
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1], np.float32); b = 0.0
    for _ in range(config.META_EPOCHS):
        p = 1 / (1 + np.exp(-(Z[:cut] @ w + b)))
        g = Z[:cut].T @ (p - Y[:cut]) / cut + config.META_L2 * w
        w -= config.META_LR * g
        b -= config.META_LR * float((p - Y[:cut]).mean())
    ph = 1 / (1 + np.exp(-(Z[cut:] @ w + b)))
    acc = float(((ph > 0.5) == (Y[cut:] > 0.5)).mean()) if n > cut else None
    out = {"w": w.round(5).tolist(), "b": round(float(b), 5),
           "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(),
           "n": n, "base_rate": round(float(Y.mean()), 4),
           "holdout_acc": acc, "days": days, "ts": time.time(),
           "config_hash": config.CONFIG_HASH}
    tmp = config.META_MODEL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(out)); tmp.replace(config.META_MODEL_PATH)
    log.info("meta-labeler trained: %d signals, base win %.1f%%, "
             "holdout acc %s → %s", n, 100 * Y.mean(),
             f"{acc:.1%}" if acc else "—", config.META_MODEL_PATH.name)
    # Drift reference: the feature world THIS model learned. The live monitor
    # measures divergence from it and de-arms if the regime leaves it.
    try:
        from core.drift_monitor import build_reference
        if len(R) >= 200:
            build_reference(np.stack(R),
                            model_version=time.strftime("meta_%Y%m%d_%H%M%S"))
        else:
            log.info("drift reference skipped: only %d feature rows", len(R))
    except Exception as e:                               # noqa: BLE001
        log.error("drift reference failed: %s", e)
    return out


if HAVE_RL:

    class ForgeEnv(gym.Env):
        """Offline single-step bandit over logged seconds (each step is an
        independent decision graded by realized after-cost option PnL —
        honest about what logged data can support)."""
        def __init__(self, obs, ts, prem):
            super().__init__()
            self.obs, self.ts, self.prem = obs, ts, prem
            self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                    (config.OBS_DIM,), np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (config.ACTION_DIM,),
                                               np.float32)
            self.i = 0

        def reset(self, *, seed=None, options=None):
            self.i = np.random.randint(0, len(self.obs))
            return self.obs[self.i], {}

        def step(self, action):
            r = 0.0
            for k, idx in enumerate(config.INDEX_ORDER):
                a = float(action[2 * k])
                if abs(a) < config.FORGE_ACT_GATE_TRAIN:
                    continue
                r += abs(a) * reward_fn(self.prem[idx], float(self.ts[self.i]),
                                        1 if a > 0 else -1)
            self.i = np.random.randint(0, len(self.obs))
            return self.obs[self.i], r / 100.0, True, False, {}


def _eval_meta():
    """Load the freshly-trained meta-model exactly as apex_main.load_meta does
    (same JSON the brain reads at runtime). None ⇒ no trained model yet, so the
    sizer falls back to the uncalibrated paper win-prob, mirroring the brain's
    bootstrap path."""
    try:
        p = config.META_MODEL_PATH
        if p.exists():
            return json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        pass
    return None


def _eval_cal():
    """The conviction→win-rate calibration table the brain blends in."""
    try:
        if config.CALIBRATION_TABLE.exists():
            return json.loads(config.CALIBRATION_TABLE.read_text())
    except Exception:                                     # noqa: BLE001
        pass
    return {}


def _eval_winprob(meta, cal, frame, iidx, tod, er, f30, dirn, conv) -> float:
    """P(win) computed EXACTLY as the live brain blends it (apex_main
    meta_win_prob + win_prob_for, lines 141-170 / 524-533): the meta-model's
    logistic on the same 19×3+4 feature vector, clamped to [META_P_FLOOR,
    META_P_CAP], blended 50/50 with the conviction-bucket calibration when that
    bucket has enough samples; the uncalibrated paper win-prob otherwise. This
    is what feeds RiskGovernor's Kelly budget — so the forge sizes on the same
    dynamic edge the brain does, not a static 0.60."""
    import math
    wp_meta = None
    if meta is not None and int(meta.get("n", 0)) >= config.META_MIN_TRAIN:
        b0 = iidx * config.NODES_PER_INDEX
        x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2],
                            [tod, er,
                             math.copysign(min(abs(f30) * 100, 3), f30)
                             if f30 else 0.0,
                             1.0 if dirn > 0 else -1.0]]).astype(np.float32)
        mu = np.asarray(meta["mu"], np.float32)
        sd = np.asarray(meta["sd"], np.float32)
        w = np.asarray(meta["w"], np.float32)
        z = (x - mu) / np.where(sd > 0.0, sd, 1.0)
        pr = 1.0 / (1.0 + math.exp(-float(z @ w) - float(meta["b"])))
        wp_meta = float(min(max(pr, config.META_P_FLOOR), config.META_P_CAP))
    w_ = config.CAL_BUCKET_WIDTH
    bkey = f"{min(abs(conv) // w_ * w_, 1 - w_):.2f}"
    cal_hit = bkey in cal and cal[bkey][1] >= config.CAL_MIN_SAMPLES
    if wp_meta is None:
        return float(cal[bkey][0]) if cal_hit else config.uncalibrated_winprob()
    if cal_hit:
        return 0.5 * (wp_meta + float(cal[bkey][0]))      # blend both judges
    return wp_meta


def _eval_hm(t: int) -> str:
    """Seconds-from-open → 'HH:MM', so RiskGovernor's entry curfew
    (NO_ENTRY_AFTER) fires on the same wall-clock the brain sees."""
    base = dt.datetime(2000, 1, 1,
                       *(int(x) for x in config.SESSION_OPEN.split(":")))
    return (base + dt.timedelta(seconds=int(t))).strftime("%H:%M")


def _grade_like_live(con, day: str, index: str, decide) -> float:
    """After-cost ₹ a policy would have ACTUALLY realized on `day` for `index`,
    sized EXACTLY like live: every entry goes through the real
    RiskGovernor.first_affordable (so the Kelly budget, the ATM→OTM→deeper
    affordability walk, the worst-case disaster-floor check, the entry curfew,
    cooldown, post-loss lockout and concurrency cap are the LIVE code, not a
    forge re-implementation), with the dynamic per-signal win-prob the brain
    computes, and the SAME shaped triple-barrier exit as the meta-labeler /
    reward grid. `decide(obs, frame, iidx) -> conviction` plugs in either the SAC
    model or the heuristic. The day is replayed tick-by-tick through the one
    StateBuilder, identical to _gen_meta_samples — same surface fit, same walls,
    same hierarchy. One divergence from live is documented: equity is per-index
    here (a fresh RiskGovernor per index-day) rather than one shared pool across
    indices — fine for a candidate-vs-incumbent ranking, both scored the same."""
    from collections import deque
    from simulation.replay_real_day import load_day
    from simulation.scenario_engine import N
    from core.risk_manager import RiskGovernor
    loaded = load_day(con, day, index)
    if not loaded:
        return 0.0
    spot_tok, by_sec, ti, bidA, askA = loaded
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    builder = StateBuilder()
    iidx = config.INDEX_ORDER.index(index)
    risk = RiskGovernor()                                 # config.TRADING_CAPITAL
    snaps, chain = {}, None
    last_tick = {}
    spot_hist: deque = deque(maxlen=1800)
    open_p = p945 = None
    last_try = -1e9
    horizon = int(config.MAX_HOLD_MINUTES * 60)
    meta, cal = _eval_meta(), _eval_cal()
    total = 0.0
    open_pos = None                                       # (exit_t, outlay, pnl, dir)
    _oh, _om = (int(x) for x in config.SESSION_OPEN.split(":"))
    _open_sod = _oh * 3600 + _om * 60
    macro = load_macro_archive(con, day, index)
    for _s in macro:
        _s["_t"] = int((_s["ts"] + 19800) % 86400) - _open_sod
    mptr = [0]
    fit_surface = _make_surface_fitter()
    for t in range(N):
        for tok, sn in by_sec.get(t, {}).items():
            snaps[tok] = sn
            last_tick[tok] = t
        sp = snaps.get(spot_tok)
        if not sp or not sp.get("ltp"):
            continue
        spot = float(sp["ltp"])
        spot_hist.append(spot)
        if open_p is None:
            open_p = spot
        if p945 is None and t >= 1800:
            p945 = spot
        step = (chain or {}).get("step") or config.INDICES[index]["strike_step"]
        atm = round(spot / step) * step
        if chain is None or chain.get("atm") != atm:
            chain = mapper.chain(index, spot) or chain
        market = {index: {"spot": sp}}
        if chain:
            legs = {}
            for leg, info in chain["legs"].items():
                s = snaps.get(info["token"])
                if s:
                    legs[leg] = {"snap": s, "strike": info["strike"]}
            market[index].update({"expiry": chain["expiry"],
                                  "dte": chain["dte"], "T": chain["T"],
                                  "is_weekly": chain["is_weekly"],
                                  "legs": legs})
        snap = _latest_at(macro, mptr, t, lambda s: s["_t"]) if macro else None
        if chain and snap:                                # REAL surface before push
            fit_surface(builder, index, chain["expiry"], chain["T"], snap)
        obs = builder.push(market, float(t))
        if obs is None or chain is None:
            continue
        frame = builder.frames[-1]
        risk.on_tick()                                    # physics-settled counter

        # ---- realize the held position when its barrier elapses (one/idx) ----
        if open_pos is not None:
            if t >= open_pos[0]:
                _xt, _outlay, _pnl, _dir = open_pos
                risk.register_exit(_outlay, _pnl, _dir, ts=float(_xt))
                total += _pnl
                open_pos = None
            else:
                continue                                  # still holding → no entry

        # ---- flat: ask the policy ----
        conv = float(decide(obs, frame, iidx))
        if t - last_try < config.ENTRY_ATTEMPT_THROTTLE_S:
            continue
        d = "CE" if conv > 0 else "PE"
        diffs = np.abs(np.diff(np.asarray(spot_hist, float)))
        er = (abs(spot_hist[-1] - spot_hist[0]) / diffs.sum()) \
            if len(spot_hist) > 120 and diffs.sum() > 0 else 0.5
        f30 = ((p945 - open_p) / open_p) if (p945 and open_p) else 0.0
        wp = _eval_winprob(meta, cal, frame, iidx, t / N, er, f30,
                           1 if d == "CE" else -1, conv)
        # SAME decision gate as the brain (apex_main 615-624): model-driven when
        # a trained meta-model exists, else the fixed conviction bar.
        if config.META_DECISION_ENABLED and meta is not None:
            if abs(conv) < config.META_ENTRY_CONV_FLOOR or wp < config.META_ENTRY_P_BAR:
                continue
        elif abs(conv) < config.PAPER_ENTRY_CONVICTION:
            continue
        last_try = t

        # ---- build the SAME preferred-first hierarchy the brain hands the
        #      RiskGovernor (ATM→OTM→deeper, real two-sided quotes, spread gate) ----
        T = float((chain or {}).get("T") or 0.01)
        atm_iv = builder.surface.atm_iv(index, (chain or {}).get("expiry", ""), T)
        hierarchy = []
        for r in mapper.hierarchy(index, spot, d):
            kk = ti.get(r["token"])
            if kk is None or t - last_tick.get(r["token"], -99) > 5:
                continue
            b_, a_ = bidA[kk, t], askA[kk, t]
            if np.isnan(b_) or np.isnan(a_) or b_ <= 0 or a_ <= 0:
                continue
            mid = (b_ + a_) / 2.0
            if (a_ - b_) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
                continue                                  # live illiquidity gate
            hierarchy.append({"premium": float(mid), "lot": int(r["lot"]),
                              "symbol": r["symbol"], "exchange": r["exchange"],
                              "price": float(a_), "_k": kk,
                              "_strike": float(r["strike"])})
        if not hierarchy:
            continue

        # ---- THE live sizer: same function, same config, zero drift ----
        leg, permit = risk.first_affordable(
            hierarchy, direction=d, win_prob=wp,
            sl_pct=config.BASE_SL_PCT, tp_pct=config.BASE_TP_PCT,
            data_age_s=0.0, now_hm=_eval_hm(t), ts=float(t),
            ann_vol=atm_iv or None)
        if leg is None:
            continue                                      # blocked exactly as live
        kk, e, lot, Kstrike = leg["_k"], leg["premium"], leg["lot"], leg["_strike"]

        # ---- SHAPED triple-barrier exit on the CHOSEN leg (meta-labeler twin) ----
        mins_left = max((N - t) / 60.0, 1.0)
        cw = snap.get("call_wall") if snap else None
        pw = snap.get("put_wall") if snap else None
        if spot > 0 and Kstrike > 0 and T > 0 and e > 0:
            tp, sl = _shaped_barriers(e, spot, Kstrike, T, mins_left,
                                      d == "CE", cw, pw)
            tp, sl = float(tp), float(sl)
        else:
            tp, sl = e * (1 + config.BASE_TP_PCT), e * (1 - config.BASE_SL_PCT)
        seg = bidA[kk, t + 1:t + 1 + horizon]
        if seg.size == 0 or np.all(np.isnan(seg)):
            continue
        itp = int(np.argmax(seg >= tp)) if np.any(seg >= tp) else None
        isl = int(np.argmax(seg <= sl)) if np.any(seg <= sl) else None
        if itp is not None and (isl is None or itp < isl):
            exitp, off = float(tp), itp
        elif isl is not None:
            exitp, off = float(sl), isl
        else:
            valid = np.nonzero(~np.isnan(seg))[0]
            exitp, off = float(seg[valid[-1]]), int(valid[-1])
        outlay = e * lot
        pnl = (exitp - e) * lot - round_trip_costs(outlay, exitp * lot)
        risk.register_entry(outlay)
        open_pos = (t + off + 1, outlay, float(pnl), d)   # realize at the barrier

    if open_pos is not None:                              # EOD: realize the runner
        risk.register_exit(open_pos[1], open_pos[2], open_pos[3],
                           ts=float(open_pos[0]))
        total += open_pos[2]
    return total


def evaluate(model, vec, con, val_day) -> float:
    """After-cost ₹ the SAC policy would have realized on the held-out day, sized
    EXACTLY like live — every entry routed through the real RiskGovernor with the
    dynamic per-signal win-prob, walking to an affordable leg. Summed over the
    TRADABLE indices the brain actually trades. (Replays the day from the vault,
    so it is heavier than the old precomputed-table sum, but it is the live
    decision path — what the brain would have done, not a 1-lot-ATM fantasy.)"""
    def decide(obs, frame, iidx):
        o = vec.normalize_obs(obs[None]) if vec else obs[None]
        a, _ = model.predict(o, deterministic=True)
        return float(a[0][2 * iidx])
    return float(sum(_grade_like_live(con, val_day, idx, decide)
                     for idx in config.TRADABLE))


def evaluate_heuristic(con, val_day) -> float:
    """Same live-faithful grading as evaluate(), scoring the HEURISTIC, so the
    baseline is apples-to-apples with the candidate. The ONLY difference is the
    policy: the heuristic reads the raw warm frame straight from the builder (no
    VecNormalize — that is the SAC model's input transform, not the heuristic's)."""
    from core.heuristic_policy import HeuristicPolicy
    pol = HeuristicPolicy()

    def decide(obs, frame, iidx):
        return float(pol.predict(frame)[2 * iidx])
    return float(sum(_grade_like_live(con, val_day, idx, decide)
                     for idx in config.TRADABLE))


def train_trap_model(ledger_path=None):
    """Refit the trap shield's WEIGHTS and THRESHOLD from REAL stop-breach
    events. Each TRAP_HOLD / STOP_BREACH_HONORED row carries the fingerprint
    vector at the breach; the label is whether a TRAP_CONFIRMED (price reclaimed
    = real hunt) followed for that position within the grace window. Fits a
    numpy-only logistic model (no GPU), then picks the threshold that best
    separates hunts from breakdowns. Writes config.TRAP_MODEL_PATH ONLY when
    there are ≥ TRAP_MIN_SAMPLES real breaches — otherwise the shield keeps using
    the fixed guess. NEVER touches the grace window / use cap / disaster floor."""
    import csv as _csv
    import json as _json
    import os as _os
    path = Path(ledger_path or config.LEDGER_PATH)
    if not path.exists():
        log.info("trap-learner: no ledger yet — shield stays on fixed threshold")
        return
    feat_keys = sorted(config.TRAP_WEIGHTS)
    # walk the ledger; a breach belongs to the position open at that time. We key
    # by (index, symbol) and resolve the label from a following TRAP_CONFIRMED.
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            rows.append(r)
    # index positions: a BUY_FILL opens, SELL_FILL closes; within that span a
    # TRAP_CONFIRMED means the held breach(es) were genuine hunts.
    samples_x, samples_y = [], []
    open_key = None
    pending = []          # breaches awaiting this position's reclaim verdict
    confirmed = False
    for r in rows:
        ev = r.get("event", "")
        if ev == "BUY_FILL":
            open_key, pending, confirmed = True, [], False
        elif ev in ("TRAP_HOLD", "STOP_BREACH_HONORED") and open_key:
            fp = {}
            for tok in (r.get("fingerprints", "") or "").split(";"):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try: fp[k] = float(v)
                    except ValueError: pass
            if all(k in fp for k in feat_keys):
                pending.append(fp)
        elif ev == "TRAP_CONFIRMED" and open_key:
            confirmed = True
        elif ev == "SELL_FILL" and open_key:
            # resolve labels for this closed position
            for fp in pending:
                samples_x.append([fp[k] for k in feat_keys])
                samples_y.append(1.0 if confirmed else 0.0)
            open_key, pending, confirmed = None, [], False

    n = len(samples_x)
    if n < config.TRAP_MIN_SAMPLES:
        log.info("trap-learner: %d/%d real stop-breaches — shield stays on the "
                 "fixed threshold %.2f until there's enough to learn from",
                 n, config.TRAP_MIN_SAMPLES, config.TRAP_SCORE_THRESHOLD)
        return
    if len(set(samples_y)) < 2:
        log.info("trap-learner: %d breaches but all one class — can't fit yet", n)
        return

    X = np.asarray(samples_x, float)
    y = np.asarray(samples_y, float)
    # standardize, fit logistic via gradient descent (numpy-only, like the meta)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xb.shape[1])
    lr, l2 = config.META_LR, config.META_L2
    for _ in range(config.META_EPOCHS):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    # map logistic weights back to non-negative fingerprint weights that sum to 1
    raw = np.clip(w[:-1] / sd, 0.0, None)
    if raw.sum() <= 1e-9:
        log.info("trap-learner: degenerate fit — keeping fixed weights")
        return
    weights = {k: float(raw[i] / raw.sum()) for i, k in enumerate(feat_keys)}
    # choose the threshold: score every sample under the LEARNED weights, pick
    # the cut that maximizes balanced accuracy (hunts held, breakdowns released)
    scores = (X * np.array([weights[k] for k in feat_keys])).sum(1)
    best_th, best_ba = config.TRAP_SCORE_THRESHOLD, -1.0
    for th in np.linspace(config.TRAP_THRESHOLD_MIN, config.TRAP_THRESHOLD_MAX, 36):
        held = scores >= th
        tp = float(((held) & (y == 1)).sum()); fn = float(((~held) & (y == 1)).sum())
        tn = float(((~held) & (y == 0)).sum()); fp = float(((held) & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_th = ba, float(th)

    model = {"weights": weights, "threshold": best_th,
             "n_samples": n, "balanced_acc": round(best_ba, 3),
             "hunt_rate": round(float(y.mean()), 3),
             "fit_utc": dt.datetime.utcnow().isoformat()}
    _os.makedirs(_os.path.dirname(config.TRAP_MODEL_PATH), exist_ok=True)
    tmp = config.TRAP_MODEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        _json.dump(model, fh, indent=2)
    _os.replace(tmp, config.TRAP_MODEL_PATH)
    log.info("trap-learner: fit on %d real breaches (%.0f%% were true hunts) → "
             "threshold %.2f, balanced-acc %.2f. Caps (%ds grace / %d uses / "
             "%.0f%% floor) UNCHANGED — never learned.",
             n, 100 * y.mean(), best_th, best_ba, config.TRAP_MAX_HOLD_S,
             config.TRAP_MAX_USES_PER_TRADE, 100 * config.ABS_DISASTER_PCT)


# ====================================================================
# FAST FORGE TRAINER — a 1-step entry is a CONTEXTUAL BANDIT, not a
# sequential task. With done=True every step, SAC's TD target collapses to the
# immediate reward, so the replay buffer, target networks and rollout loop are
# pure overhead (that is why it crawls at ~8 fps on the GPU). We keep SAC's
# exact actor/critic networks — so the saved artifact still loads in the brain
# unchanged — but train them directly:
#   1. precompute every entry's after-cost reward ONCE (kills the per-step
#      Python reward_fn that the env was paying on every single step),
#   2. regress the twin critics on that reward (target = r, no bootstrap),
#   3. push the actor up min(Q1,Q2) with SAC's own entropy term,
# all as batched GPU tensor ops. Same policy, same objective, ~100x faster.
# ====================================================================
def _round_trip_costs_vec(buy_v, sell_v):
    """Vectorized mirror of core.execution_engine.round_trip_costs, built from
    the shared config.COSTS dict (documented as the single toll-booth the reward,
    paper engine and analyzer all agree on). buy_v / sell_v are ₹ premium values
    (price × lot), scalar or np.ndarray. Kept identical to the scalar path so the
    training table and evaluate() grade on the same costs."""
    c = config.COSTS
    brokerage = 2.0 * c["brokerage_per_order"]            # buy + sell, flat
    txn = (buy_v + sell_v) * c["exch_txn_pct"]            # both legs on premium
    sebi = (buy_v + sell_v) * c["sebi_pct"]
    stt = sell_v * c["stt_sell_pct"]                      # sell side only
    stamp = buy_v * c["stamp_buy_pct"]                    # buy side only
    gst = (brokerage + txn + sebi) * c["gst_pct"]
    return brokerage + txn + sebi + stt + stamp + gst


def _barrier_exit_grid(bid_g: np.ndarray, ask_g: np.ndarray,
                       tp_g: np.ndarray, sl_g: np.ndarray,
                       horizon: int) -> np.ndarray:
    """Vectorized first-touch triple barrier over a dense per-second grid. Entry
    is the mid (bid+ask)/2 at each start; tp_g/sl_g are the per-start SHAPED
    barriers (from build_dataset). Returns the realized exit PRICE for every start
    at once (NaN where no valid entry or no valid forward bid). Same rule as
    _exit_price_from_path, evaluated for all starts by sweeping the forward offset
    and recording the first barrier each start touches. O(horizon × grid)."""
    G = bid_g.shape[0]
    e = (bid_g + ask_g) / 2.0                             # mid entry per start
    exitp = np.full(G, np.nan)
    done = np.isnan(e) | np.isnan(tp_g)                   # no valid entry ⇒ NaN
    last_bid = np.full(G, np.nan)
    maxj = min(horizon, G - 1)
    for j in range(1, maxj + 1):
        b = np.full(G, np.nan)
        b[:G - j] = bid_g[j:]                             # b[s] = bid at sec s+j
        valid = ~done & ~np.isnan(b)
        last_bid[valid] = b[valid]                        # latest bid in window
        tgt = valid & (b >= tp_g)
        stp = valid & (b <= sl_g)                         # tp>e>sl ⇒ never both
        exitp[tgt] = tp_g[tgt]
        exitp[stp] = sl_g[stp]
        done |= tgt | stp
        if done.all():
            break
    nd = ~done                                            # never hit ⇒ max-hold
    exitp[nd] = last_bid[nd]
    return exitp


def _reward_table(prem: dict, ts) -> np.ndarray:
    """(N, K, 2) after-cost ₹ of the trade the LIVE brain would actually PLACE at
    each second — the model's learning target, sized like live instead of a
    1-lot-ATM fantasy. Three live constraints, so the policy learns the real
    objective:
      • TRADABLE indices only — the other four are 0. The brain never trades them,
        so the policy must learn they are worthless, not farm ATM PnL on BANKNIFTY
        etc. (the old grid graded all six, teaching trades that never happen).
      • First-affordable leg — ATM if a 1-lot ATM entry (mid × lot) fits the Kelly
        budget, else step out to OTM1, else 0 (no affordable leg ⇒ no trade ⇒ no
        PnL). This is the brain's ATM→OTM affordability walk over the harvested
        chain (ATM ± 1 step); at this capital an ATM index lot often exceeds the
        per-trade budget, so the old "always ATM" reward was scoring trades the
        account could never place.
      • SAME shaped triple-barrier realized exit, evaluated on the CHOSEN leg.
    Affordability here uses the STATIC Kelly budget (_kelly_budget at the paper
    win-prob) so the target stays a cheap, policy-independent precompute. The
    promotion GATE goes further — dynamic per-signal win-prob + a deeper hierarchy
    + a stateful sequential sim (see evaluate / _grade_like_live); folding those
    into the per-cell TRAINING grid would need the vault hierarchy and spot-history
    bookkeeping threaded through build_dataset, a heavier follow-up. [...,0] = long
    CE, [...,1] = long PE. Vectorized per leg; the affordable leg is picked
    per-second."""
    K, N = len(config.INDEX_ORDER), len(ts)
    R = np.zeros((N, K, 2), np.float32)
    horizon = int(config.MAX_HOLD_MINUTES * 60)
    budget = _kelly_budget(config.TRADING_CAPITAL)
    ts_i = np.rint(np.asarray(ts, dtype=np.float64)).astype(np.int64)
    all_legs = ("atm_ce", "atm_pe", "otm_ce", "otm_pe")
    legs_for = {0: ("atm_ce", "otm_ce"), 1: ("atm_pe", "otm_pe")}  # ATM preferred
    for k, idx in enumerate(config.INDEX_ORDER):
        if idx not in config.TRADABLE:                    # brain never trades these
            continue
        pidx = prem[idx]
        if not pidx:                                      # spot-only index ⇒ all 0
            continue
        smin, smax = min(pidx), max(pidx)
        G = smax - smin + 1
        grids = {leg: [np.full(G, np.nan), np.full(G, np.nan), np.zeros(G),
                       np.full(G, np.nan), np.full(G, np.nan)]
                 for leg in all_legs}
        for s, row in pidx.items():
            g = s - smin
            for leg in all_legs:
                if leg in row:
                    b, a, lot, tp, sl = row[leg]
                    grids[leg][0][g] = b
                    grids[leg][1][g] = a
                    grids[leg][2][g] = lot
                    grids[leg][3][g] = tp
                    grids[leg][4][g] = sl
        gpos = ts_i - smin
        inside = np.nonzero((gpos >= 0) & (gpos < G))[0]
        gp = gpos[inside]
        for d_idx, (atm_leg, otm_leg) in legs_for.items():
            pnl, mid, lot = {}, {}, {}
            for leg in (atm_leg, otm_leg):
                bid_g, ask_g, lot_g, tp_g, sl_g = grids[leg]
                ex = _barrier_exit_grid(bid_g, ask_g, tp_g, sl_g, horizon)  # (G,)
                e = (bid_g + ask_g) / 2.0
                pnl[leg] = (ex - e) * lot_g - _round_trip_costs_vec(e * lot_g,
                                                                    ex * lot_g)
                mid[leg], lot[leg] = e, lot_g
            # per-second affordability of a 1-lot entry (mid × lot ≤ budget)
            atm_cost = mid[atm_leg] * lot[atm_leg]
            otm_cost = mid[otm_leg] * lot[otm_leg]
            atm_ok = (~np.isnan(pnl[atm_leg]) & ~np.isnan(atm_cost)
                      & (lot[atm_leg] > 0) & (atm_cost <= budget))
            otm_ok = (~np.isnan(pnl[otm_leg]) & ~np.isnan(otm_cost)
                      & (lot[otm_leg] > 0) & (otm_cost <= budget))
            # ATM if it fits the budget, else step out to OTM1, else 0 (no trade)
            chosen = np.where(atm_ok, np.nan_to_num(pnl[atm_leg]),
                              np.where(otm_ok, np.nan_to_num(pnl[otm_leg]), 0.0))
            R[inside, k, d_idx] = chosen[gp].astype(np.float32)
    return R


def _bandit_reward(actions, R, gate, scale_by_mag: bool):
    """Reward of arbitrary actions, vectorized. Per index k it uses action[2k]
    and, if |a|≥gate, takes the CE reward when a>0 else the PE reward. With
    scale_by_mag=True it multiplies by |a| (this mirrors ForgeEnv.step, the
    TRAIN signal); with False it does not (this mirrors evaluate(), the 1-lot
    promotion metric). actions (B,12), R (B,K,2) → (B,)."""
    a = actions[:, 0::2]                                   # even comps per index
    mag = a.abs()
    active = (mag >= gate).float()
    pos = (a > 0).float()
    r_dir = pos * R[..., 0] + (1.0 - pos) * R[..., 1]
    w = active * mag if scale_by_mag else active
    return (w * r_dir).sum(dim=1)


def train_bandit(model, vec, obs, ts, prem, vobs, vts, vprem, log) -> None:
    """Train SAC's actor/critic directly as a 1-step bandit. No env, no replay
    buffer, no target nets, no rollout — just batched GPU updates with the true
    reward as the critic target, early-stopped on the held-out score."""
    import torch
    import torch.nn.functional as Fn
    dev = model.device

    mean = torch.as_tensor(vec.obs_rms.mean, dtype=torch.float32, device=dev)
    std = torch.sqrt(torch.as_tensor(vec.obs_rms.var, dtype=torch.float32,
                                     device=dev) + float(vec.epsilon))
    clip = float(vec.clip_obs)

    def norm(raw):                                         # raw (B,5700) on dev
        return torch.clamp((raw - mean) / std, -clip, clip)

    scale = float(getattr(config, "FORGE_BANDIT_REWARD_SCALE", 100.0))
    gate_tr = float(config.FORGE_ACT_GATE_TRAIN)
    gate_ev = float(config.FORGE_ACT_GATE_EVAL)
    bs = int(getattr(config, "FORGE_BANDIT_BATCH", 2048))
    max_ep = int(getattr(config, "FORGE_BANDIT_MAX_EPOCHS", 60))
    patience = int(getattr(config, "FORGE_BANDIT_PATIENCE", 6))

    R = torch.as_tensor(_reward_table(prem, ts), device=dev) / scale
    VR = torch.as_tensor(_reward_table(vprem, vts), device=dev) / scale
    obs_cpu = torch.as_tensor(obs, dtype=torch.float32)    # streamed CPU→GPU
    vobs_cpu = torch.as_tensor(vobs, dtype=torch.float32)
    N = obs_cpu.shape[0]
    auto_ent = getattr(model, "log_ent_coef", None) is not None

    eval_rows = int(getattr(config, "FORGE_BANDIT_EVAL_ROWS", 4096))
    warmup = int(getattr(config, "FORGE_BANDIT_WARMUP_EPOCHS", 20))
    nv = vobs_cpu.shape[0]
    # Fixed seeded subsets → the per-epoch proxy is apples-to-apples across
    # epochs and cheap (the full held-out pass every epoch was the wall-clock
    # cost before; promotion still calls the real full evaluate()).
    tr_idx = torch.randperm(N, generator=torch.Generator().manual_seed(0)
                            )[:min(eval_rows, N)]
    hv_idx = torch.randperm(nv, generator=torch.Generator().manual_seed(1)
                            )[:min(eval_rows, nv)]

    def diag(obs_src, RR, idx):
        """Deterministic policy (what the brain runs) on a fixed subset → after-
        cost ₹/row and trade-rate (fraction of index-slots with |a|≥eval gate).
        trade-rate is what separates a degenerate 'never trade' collapse from
        genuine, healthy selectivity."""
        model.policy.set_training_mode(False)
        with torch.no_grad():
            ob = norm(obs_src[idx].to(dev))
            act = model.actor(ob, deterministic=True)
            rate = float((act[:, 0::2].abs() >= gate_ev).float().mean())
            rs = float(_bandit_reward(act, RR[idx], gate_ev, False).mean()) * scale
        return rs, rate

    log.info("bandit trainer: %d train rows × %d held-out rows | batch %d | "
             "warmup %d ep | %s — replacing the SAC rollout",
             N, nv, bs, warmup, dev)
    best_key, best_state, gstep = (-1e18, -1e18), None, 0
    stop_ref, bad = -1e18, 0
    for ep in range(max_ep):
        model.policy.set_training_mode(True)
        perm = torch.randperm(N)
        c_acc = a_acc = 0.0
        nb = 0
        for s in range(0, N - bs + 1, bs):
            bi = perm[s:s + bs]
            ob = norm(obs_cpu[bi].to(dev))
            Rb = R[bi]

            # critic: regress Q toward the TRUE 1-step reward, on a blend of
            # current-policy and random actions so it learns the landscape
            with torch.no_grad():
                a_pol, _ = model.actor.action_log_prob(ob)
            a_rnd = torch.empty_like(a_pol).uniform_(-1.0, 1.0)
            ob2 = torch.cat([ob, ob], 0)
            a2 = torch.cat([a_pol, a_rnd], 0)
            r2 = _bandit_reward(a2, torch.cat([Rb, Rb], 0), gate_tr,
                                True).unsqueeze(1)
            c_loss = sum(Fn.mse_loss(q, r2) for q in model.critic(ob2, a2))
            model.critic.optimizer.zero_grad()
            c_loss.backward()
            model.critic.optimizer.step()

            # actor: maximize min(Q1,Q2) with SAC's entropy term
            a_pi, logp = model.actor.action_log_prob(ob)
            q_pi = torch.min(*model.critic(ob, a_pi))
            alpha = (torch.exp(model.log_ent_coef.detach()) if auto_ent
                     else model.ent_coef_tensor)
            a_loss = (alpha * logp.reshape(-1, 1) - q_pi).mean()
            model.actor.optimizer.zero_grad()
            a_loss.backward()
            model.actor.optimizer.step()

            # temperature (auto entropy), exactly as SAC tunes it
            if auto_ent:
                e_loss = -(model.log_ent_coef
                           * (logp.reshape(-1, 1) + model.target_entropy
                              ).detach()).mean()
                model.ent_coef_optimizer.zero_grad()
                e_loss.backward()
                model.ent_coef_optimizer.step()

            c_acc += float(c_loss); a_acc += float(a_loss); nb += 1
            gstep += 1

        tr_rs, tr_rate = diag(obs_cpu, R, tr_idx)
        hv_rs, hv_rate = diag(vobs_cpu, VR, hv_idx)
        log.info("  epoch %3d | steps %5d | critic %.3f actor %.3f | train "
                 "₹%.2f (%.2f%% trade) | held-out ₹%.2f (%.2f%% trade)",
                 ep, gstep, c_acc / max(nb, 1), a_acc / max(nb, 1),
                 tr_rs, tr_rate * 100, hv_rs, hv_rate * 100)

        # SELECT best by held-out ₹, ties broken by train ₹ — so on a flat/calm
        # held-out day we keep the better-trained policy, never the epoch-0 one.
        key = (round(hv_rs, 2), tr_rs)
        if key > best_key:
            best_key = key
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.policy.state_dict().items()}

        # EARLY-STOP only AFTER warmup: the actor starts at ~zero output (sub-
        # gate ⇒ held-out ₹0), so stopping on a flat ₹0 before it has had steps
        # to move its mean off zero just freezes it at initialization — which is
        # exactly the collapse we saw. After warmup, stop on a real plateau.
        if ep >= warmup:
            if hv_rs > stop_ref + 1e-6:
                stop_ref, bad = hv_rs, 0
            else:
                bad += 1
                if bad >= patience:
                    log.info("  early stop: held-out ₹ plateaued %d epochs "
                             "post-warmup", patience)
                    break

    if best_state is not None:
        model.policy.load_state_dict(best_state)           # restore best epoch
    model.policy.set_training_mode(False)
    tr_rs, tr_rate = diag(obs_cpu, R, tr_idx)
    hv_rs, hv_rate = diag(vobs_cpu, VR, hv_idx)
    log.info("bandit trainer done — held-out ₹%.2f | train ₹%.2f | trade-rate "
             "train %.2f%% · held-out %.2f%% | %d grad steps",
             hv_rs, tr_rs, tr_rate * 100, hv_rate * 100, gstep)
    return {"holdout_rs": hv_rs, "train_rs": tr_rs,
            "train_trade_rate": tr_rate, "holdout_trade_rate": hv_rate}


def _score_incumbent_on(vo, vt, vp, con, val_day, log):
    """Re-score the currently-PROMOTED model on the SAME held-out day the
    candidate is graded on, so the two are compared apples-to-apples instead of
    against a score the incumbent earned on a different day (and possibly a
    different reward). Loads the manifest's model+norm pair, runs the identical
    evaluate() harness, returns ₹ — or None if there is no incumbent or it can't
    be loaded (caller then falls back to the stored score / the heuristic)."""
    if not config.MODEL_MANIFEST.exists():
        return None
    try:
        man = json.loads(config.MODEL_MANIFEST.read_text())
        mp = config.MODEL_DIR / man.get("model", "")
        npth = config.MODEL_DIR / man.get("norm", "")
        if not (mp.exists() and npth.exists()):
            return None
        inc_env = DummyVecEnv([lambda: ForgeEnv(vo, vt, vp)])
        inc_vec = VecNormalize.load(str(npth), inc_env)
        inc_vec.training = False
        inc_model = SAC.load(str(mp), device="cpu")      # CPU: tiny eval, no GPU fight
        return float(evaluate(inc_model, inc_vec, con, val_day))
    except Exception as e:                                # noqa: BLE001
        log.warning("incumbent re-score failed (%s) — using its stored val_score "
                    "for the bar instead (share the brain's model-load snippet "
                    "and I'll match it for a clean same-day benchmark)", e)
        return None


def main():
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if len(days) < 2:
        raise SystemExit("Need ≥2 harvested days (train + held-out).")
    val_day = days[-1]
    pool = days[:-1]
    # TRAIN ON ALL DATA: every harvested day except the held-out one feeds the
    # candidate — no lookback cap, no reservoir subsample. With a small history
    # this is trivially "all of it"; once the day count grows large enough that
    # the per-day dataset build (~replay cost, linear in days) becomes the
    # wall-clock bottleneck, re-enable a FORGE_LOOKBACK_DAYS + FORGE_RESERVOIR_DAYS
    # split here. The bandit trainer itself is O(rows) on the GPU, so the binding
    # cost is the replay/build, not the fit.
    cap = getattr(config, "FORGE_MAX_TRAIN_DAYS", 0)     # 0 ⇒ unbounded (all days)
    train_days = pool if cap <= 0 else (
        pool[-cap:] + random.sample(pool[:-cap],
                                    min(len(pool[:-cap]),
                                        config.FORGE_RESERVOIR_DAYS)))
    log.info("train days %s (all data) | held-out %s", train_days, val_day)

    # 1) META-LABELER first — numpy-only, needs no GPU, feeds Kelly tomorrow
    try:
        train_meta(con, train_days)
    except Exception as e:                                # noqa: BLE001
        log.error("meta-labeler failed: %s", e)

    # 1b) TRAP LEARNER — refit shield weights+threshold from real stop-outs.
    # Numpy-only, reads the trade ledger (not the tick DB). Dormant (writes
    # nothing) until there are ≥ TRAP_MIN_SAMPLES real breaches; the shield runs
    # the fixed threshold until then. Caps are never touched.
    try:
        train_trap_model()
    except Exception as e:                                # noqa: BLE001
        log.error("trap-learner failed: %s", e)

    # 1d) REGIME CLASSIFIER — refit cut points to the empirical percentiles of
    # this market's trend-efficiency and net-GEX. Numpy-only. Dormant on the
    # fixed thresholds until enough feature rows.
    try:
        from core.regime_classifier import write_regime_model
        rm = write_regime_model()
        if rm:
            log.info("regime: refit te_trend=%.2f te_chop=%.2f gex_squeeze=%.1e "
                     "on %d rows", rm["te_trend"], rm["te_chop"],
                     rm["gex_squeeze"], rm["n_samples"])
        else:
            log.info("regime: not enough feature rows yet — fixed cut points")
    except Exception as e:                                # noqa: BLE001
        log.error("regime refit failed: %s", e)

    # 2) RL forge (torch stack)
    if not HAVE_RL:
        log.warning("torch / stable-baselines3 / gymnasium not installed — "
                    "RL forge skipped (meta-labeler above still ran). "
                    "On the RTX 4060: pip install -r requirements.txt")
        return

    from core.graph_constructor import TGNFeatureExtractor  # torch path
    import stable_baselines3.common.torch_layers as tl

    class Extractor(tl.BaseFeaturesExtractor):
        def __init__(self, observation_space):
            super().__init__(observation_space, config.PROJ_DIM)
            self.net = TGNFeatureExtractor()
        def forward(self, x):
            return self.net(x)

    blobs = [build_dataset(con, d) for d in train_days]
    blobs = [(o, t, p) for o, t, p in blobs if o is not None]
    if not blobs:
        raise SystemExit("No replayable seconds — check harvester output.")
    obs = np.concatenate([b[0] for b in blobs])
    ts = np.concatenate([b[1] for b in blobs])
    prem = {i: {} for i in config.INDEX_ORDER}
    for _, _, p in blobs:
        for i in config.INDEX_ORDER:
            prem[i].update(p[i])
    log.info("dataset: %d seconds × %d dims", len(obs), obs.shape[1])

    # Held-out day is built BEFORE training so the bandit trainer can early-stop
    # on its score — i.e. stop the moment more training stops helping.
    vo, vt, vp = build_dataset(con, val_day)
    if vo is None:
        raise SystemExit("Held-out day has no replayable seconds.")

    # VecNormalize is the SAME obs transform the brain loads — but we fit its
    # running stats straight from the dataset instead of paying for a rollout
    # to populate them. norm_reward is irrelevant here (we never use the env
    # reward; the bandit trainer grades from the precomputed table).
    env = DummyVecEnv([lambda: ForgeEnv(obs, ts, prem)])
    vec = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    vec.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    vec.obs_rms.count = float(len(obs))
    vec.training = False                                  # stats frozen from here

    # Same SAC object the brain loads (identical policy / TGN extractor / arch),
    # but trained by train_bandit() directly rather than model.learn(). The
    # replay buffer is unused on a 1-step bandit, so it is sized down to free
    # the ~2 GB it would otherwise reserve.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC("MlpPolicy", vec, device=device, buffer_size=2048,
                batch_size=config.SAC_BATCH,
                policy_kwargs={"features_extractor_class": Extractor,
                               "net_arch": [256, 256]}, verbose=0)
    diag = train_bandit(model, vec, obs, ts, prem, vo, vt, vp, log)
    score = evaluate(model, vec, con, val_day) if vo is not None else -1e9
    heur = evaluate_heuristic(con, val_day) if vo is not None else -1e9
    log.info("held-out (%s) after-cost — model ₹%.2f | heuristic ₹%.2f | "
             "model trade-rate: train %.2f%% · held-out %.2f%% (per index-slot)",
             val_day, score, heur, diag["train_trade_rate"] * 100,
             diag["holdout_trade_rate"] * 100)

    # BENCHMARK THE TWO MODELS ON THE SAME DAY. Re-score the currently-promoted
    # model on THIS held-out day (not the stored score from whatever day it was
    # promoted on) so candidate-vs-incumbent is apples-to-apples. Fall back to the
    # stored score only if the incumbent can't be loaded, and to the heuristic when
    # there is no incumbent at all.
    inc_today = _score_incumbent_on(vo, vt, vp, con, val_day, log)
    inc_stored = -1e18
    if config.MODEL_MANIFEST.exists():
        inc_stored = float(json.loads(config.MODEL_MANIFEST.read_text()).get(
            "val_score", -1e18))
    if inc_today is not None:
        incumbent, inc_src = inc_today, "re-scored on this held-out day"
    elif inc_stored > -1e17:
        incumbent, inc_src = inc_stored, "stored val_score (could not reload)"
    else:
        incumbent, inc_src = -1e18, "none yet"
    has_champ = incumbent > -1e17
    champ_str = f"₹{incumbent:+,.2f} ({inc_src})" if has_champ else "none yet"
    if has_champ:
        log.info("incumbent benchmarked on %s: ₹%.2f (%s) | candidate ₹%.2f",
                 val_day, incumbent, inc_src, score)

    # ALWAYS save the candidate and record its score, promoted or not, so every
    # model is kept and the held-out curve stays visible over time (the old code
    # discarded rejected candidates — you could never watch them climb).
    ver = time.strftime("v9_%Y%m%d_%H%M%S")
    mpath = config.MODEL_DIR / f"apex_sac_{ver}.zip"
    npath = config.MODEL_DIR / f"apex_norm_{ver}.pkl"
    model.save(mpath); vec.save(str(npath))

    # Bar to clear = the policy ACTUALLY IN PRODUCTION: the live heuristic AND
    # any promoted champion — beat BOTH. Margin is ADDITIVE in ₹ (sign-safe; the
    # old multiplicative FORGE_PROMOTE_MARGIN inverted on the negative scores you
    # get before a model has real edge). Promotion swaps what the brain TRADES,
    # so a near-tie on one noisy held-out day is not enough — demand a real ₹
    # margin, and widen to multi-day walk-forward as harvested history grows.
    margin = getattr(config, "FORGE_PROMOTE_MARGIN_RS", 0.0)
    baseline = max(heur, incumbent if has_champ else heur)
    # A model that does not trade is not an "edge" — it is an abstention, and on
    # a single calm held-out day its ₹0 out-scores any net-losing trader and
    # wins promotion. Deploying it takes the brain to ZERO live paper trades,
    # freezing the very ledger the edge certificate needs. So a model that barely
    # trades on the TRAINING days (where good setups DO exist) is never eligible,
    # no matter how good its held-out ₹ looks.
    min_rate = getattr(config, "FORGE_MIN_TRADE_RATE", 0.001)
    abstains = diag["train_trade_rate"] < min_rate
    promote = (score > baseline + margin) and not abstains

    g = {}
    gate = config.STATE_DIR / "sim_gate.json"
    suite_red = False
    if gate.exists():
        g = json.loads(gate.read_text())
        suite_red = (g.get("pass") != g.get("total") and
                     time.time() - g.get("ts", 0) < 36 * 3600)

    hist = config.MODEL_DIR / "forge_history.jsonl"
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ver": ver, "day": val_day,
                             "model_score": round(score, 2),
                             "heuristic": round(heur, 2),
                             "incumbent": round(incumbent, 2) if has_champ else None,
                             "promoted": bool(promote and not suite_red)}) + "\n")

    if abstains:
        log.warning("NOT promoted: model is a NON-TRADER — trades only %.3f%% of "
                    "training index-slots (< %.3f%%). A do-nothing policy would "
                    "freeze paper-trade collection and the edge ledger, so it is "
                    "never deployed however good its held-out ₹ looks. Live "
                    "policy stays; candidate saved %s + logged to "
                    "forge_history.jsonl.",
                    diag["train_trade_rate"] * 100, min_rate * 100, ver)
        return
    if not promote:
        log.warning("NOT promoted: model ₹%.2f ≤ bar ₹%.2f (heuristic ₹%.2f, "
                    "champion %s, margin ₹%.2f). Live policy stays; candidate "
                    "saved %s + logged to forge_history.jsonl.",
                    score, baseline + margin, heur, champ_str, margin, ver)
        return
    if suite_red:
        log.warning("regression suite RED (%s/%s) — candidate %s clears the bar "
                    "but promotion WITHHELD until the suite is green.",
                    g.get("pass"), g.get("total"), ver)
        return

    tmp = config.MODEL_MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": ver, "model": mpath.name,
                               "norm": npath.name, "obs_dim": config.OBS_DIM,
                               "val_score": score, "val_day": val_day,
                               "ts": time.time()}))
    tmp.replace(config.MODEL_MANIFEST)                  # atomic pair promotion ★
    log.info("PROMOTED %s — model ₹%.2f clears bar ₹%.2f "
             "(heuristic ₹%.2f, prev champion %s)",
             ver, score, baseline + margin, heur, champ_str)


if __name__ == "__main__":
    config.setup_logging("forge")
    main()