"""
APEX OMNI v9.1 — NIGHTLY FORGE (audit Part C: the falsification harness)
========================================================================
v9 closed the FEATURE train/serve gap (one StateBuilder). v9.1 closes the
DECISION and EXAM gaps the audit proved were still open:

  1. ONE DECISION DIALECT. The meta-labeler and the grader now run the brain's
     ACTUAL entry pipeline via core/decision.py — advisory shocks (from the
     macro archive, which now returns flip), logit fusion, LOGIT-space regime
     scaling, the meta/calibration P(win) blend, the wall-clock persistence
     gate, the live entry gate. v9.0 labeled/graded raw heuristic ≥0.55 with
     none of that: the forge examined a system that was never deployed.
  2. HONEST LABELS. Entry basis is the ASK (the live engine CROSSES on
     momentum; mid-basis labels booked the half-spread as free edge), exits on
     bids as before, and the hold horizon is 0-DTE-AWARE (MAX_HOLD_MINUTES_0DTE
     when dte < EXPIRY_DTE_LT) so expiry-day labels stop assuming 45 minutes
     the guillotine never grants. Not modeled (documented, conservative): the
     live +2-tick chase allowance and the slip-cap walk-away.
  3. UNIQUENESS WEIGHTS (AFML ch.4). Overlapping 25–45-min triple-barrier
     windows at 1 Hz are ~99.9% redundant; meta samples are weighted by average
     1/concurrency so the fit and its holdout stop being dominated by
     duplicated paths. Holdout is BY DAY (the last training day).
  4. NO SELECTION LEAKAGE. The bandit trainer early-stops / selects checkpoints
     on an INNER day (the last training day) — the promotion day is never seen
     by any gradient, any selection, or any meta fit. v9.0 selected the best
     checkpoint ON the promotion day (~150 oracle queries against the exam).
  5. ONE RISK GOVERNOR. The grader replays ALL tradable indices through one
     governor in one time loop — the live MAX_CONCURRENT_POSITIONS=1 world —
     instead of a fresh per-index governor that let phantom parallel books
     inflate scores.
  6. WALK-FORWARD + PSR/DSR. Beyond the single promotion day: the last
     FORGE_WF_FOLDS pool days are each validated by a candidate trained ONLY on
     strictly-earlier days (bootstrap path for both policies — per-fold meta
     refits would be the only leak-free alternative and are out of nightly
     budget). Per-fold ₹ feeds Bailey–López de Prado's Probabilistic Sharpe
     Ratio and a deflated variant using the trial count from forge_history —
     with few folds these are honesty labels, not significance claims, and the
     report says so.
  7. LEGIBILITY. logs/forge_report_<date>.json carries the dataset stats,
     uniqueness, meta diagnostics, the WF table, PSR/DSR, the grader's GATE
     FUNNEL per policy (why the exam produced few/no entries), promotion
     reasons, and timing. Cross-day leakage note: labels never span days (paths
     end at session close) and every day gets a fresh StateBuilder, so
     purge/embargo between days is structurally satisfied; the residual
     within-day overlap is what the uniqueness weights address.

Promotion now requires ALL of: final-day ₹ > max(heuristic, incumbent) +
FORGE_PROMOTE_MARGIN_RS (defined at last — v9.0 read an undefined name and ran
with ₹0), walk-forward aggregate ≥ the heuristic's, a sane training trade-rate
(abstention is recorded as a legitimate finding, not deployed), and a green
regression suite.

Run after close:  python nightly_forge_v9.py     (or via run_nightly.py)

v9.1.1 (first-live-day fixes, 2026-07-04):
  • MEMORY: ForgeEnv no longer pins the full training array inside the
    candidate's VecNormalize for the whole run (the ~5 GB that OOM'd the
    walk-forward three times on 2026-07-03); per-day blobs freed post-concat;
    candidate saved + freed BEFORE folds.
  • PARITY: replay aggregates flow WITHIN each second (vol_delta summed,
    iceberg OR'd) instead of last-tick-wins — the discard starved OFI/VPIN/
    Hawkes/dealer-inv vs the ~5 Hz live brain (live conv 0.86–0.95 vs replay
    0.17–0.20 on the same day).
  • CAPITAL-FREE EXAM: every forge label/reward/grade sized against the fixed
    FORGE_EVAL_CAPITAL reference account. TRADING_CAPITAL is live-only —
    change it any time, nothing in the forge/meta/drift/caches moves.
  • CACHES: version-stamped (v2 after the parity fix), plus a decision-knob
    stamp for signal-dependent tiers; per-day META-SAMPLE cache and permanent
    WF-FOLD cache (immutable train sets ⇒ each fold computed once, ever).
    Nightly drops from ~7.8 h to roughly candidate-train + one new fold.
  • FUNNEL: no_quotes carries the ladder's death breakdown (unharvested /
    stale / one-sided / spread) and risk_blocked carries the permit's reason.
"""
from __future__ import annotations
import datetime as dt
import ast
import datetime as _dt
import json
import logging
import math
import os
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np

import config
from core import trial_registry as TR
from core.instruments import AsOfMapper
from core.market_state import StateBuilder
from core.execution_engine import round_trip_costs
from core.quant_core import implied_vol_newton, black76_greeks, EWMAVol
from core.heuristic_policy import HeuristicPolicy
from core import regime_classifier as regime_mod
from core import decision as D
from core.diagnostics import DailyReport

# The macro vault archive (per-strike IVs + GEX walls + flip/PCR/max-pain) the
# radar persists. If running an older macro_gex_v9 without it, the forge
# degrades cleanly to seed-surface / no-wall / no-shock behaviour ([] ⇒ no-op).
try:
    from macro_gex_v9 import load_macro_archive
except Exception:                                         # noqa: BLE001
    def load_macro_archive(con, day, index):              # type: ignore
        return []

log = logging.getLogger("forge")

# v9.7.1: the RL stack (torch/gym/SB3, ~10s import + the gym banner) loads
# ON DEMAND via _load_rl(), never at module import — every evening tool that
# does `from nightly_forge_v9 import trading_days` was paying this cost.
HAVE_RL = False
torch = gym = SAC = DummyVecEnv = VecNormalize = None
ForgeEnv = Extractor = None


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
    moving). Fixed 150/12 passes stopped SHORT of that fixed point on calm,
    low-IV days: the forge served iv ≈ 0.19 where the brain converges to ≈ 0.13.
    Here each snapshot is iterated until w(0) settles (relative change ≤
    FORGE_SURFACE_FIT_TOL). Warm-started; `min_passes` guards a one-pass fluke;
    `max_passes` caps a non-converging fit. All three knobs are getattr-defaulted
    (deliberately NOT in config.py) so CONFIG_HASH is unchanged by tuning them.
    No-op when `snap` is None (older days without an archive)."""
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
    """Yields (ts, obs_5700, frame_30x19, market, macro_now) second by second
    through ONE shared StateBuilder — identical to live. macro_now[idx] is the
    latest archived macro snapshot at-or-before ts (or None on days with no
    archive), carrying walls + flip/PCR/max-pain for the shaped-target cap AND
    the shared advisory-shock stack; the surface is also fit from it here so
    the obs iv/delta/gamma/theta are the live ones, not the seed surface.
    v9.1: the warm FRAME (builder.frames[-1]) is yielded too — the shared
    decision path and the meta features read it, exactly as the brain does."""
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
    macro = {i: load_macro_archive(con, day, i) for i in config.INDEX_ORDER}
    mptr = {i: [0] for i in config.INDEX_ORDER}
    fit_surface = _make_surface_fitter()
    _nsnap = {i: len(v) for i, v in macro.items() if v}
    if _nsnap:
        log.info("%s: macro archive HIT — %s snapshot(s) → REAL SVI surface fit + "
                 "GEX-wall target cap + replay shocks ACTIVE", day,
                 " ".join(f"{i}:{n}" for i, n in _nsnap.items()))
    else:
        log.info("%s: macro archive empty → seed surface, no walls/shocks (prior "
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
        frame = builder.frames[-1] if obs is not None else None
        return obs, frame, market, macro_now

    snap_sec: dict[int, int] = {}          # token → second it last accumulated
    for ts, tok, ltp, bid, ask, bq, aq, vd, oi, ice in cur:
        sec = int(ts)
        if cur_sec is None:
            cur_sec = sec
        while sec > cur_sec:
            obs, frame, market, macro_now = emit(cur_sec)
            if obs is not None:
                yield cur_sec, obs, frame, market, macro_now
            cur_sec += 1
        # v9.1.1 PARITY FIX: aggregate WITHIN the second. The old last-tick-wins
        # snap silently DISCARDED every earlier tick's vol_delta in the second,
        # so OFI/VPIN/Hawkes/dealer-inventory trained and graded on a fraction
        # of the flow the ~5 Hz live brain integrates — live conviction peaked
        # 0.86–0.95 on 2026-07-03 while the replay's mass sat at 0.17–0.20.
        # Flow is SUMMED, iceberg is OR'd, price/book/oi are last-wins (what a
        # 1 Hz reader would see). Tokens silent this second keep their last
        # snap unchanged, exactly like the live ring buffer.
        if snap_sec.get(tok) == sec and tok in snaps:
            s = snaps[tok]
            s["ltp"], s["bid"], s["ask"] = ltp, bid, ask
            s["bid_qty"], s["ask_qty"], s["oi"] = bq, aq, oi
            s["vol_delta"] += vd
            s["iceberg"] = max(s["iceberg"], ice)
        else:
            snaps[tok] = {"ltp": ltp, "bid": bid, "ask": ask, "bid_qty": bq,
                          "ask_qty": aq, "vol_delta": vd, "oi": oi,
                          "iceberg": ice}
            snap_sec[tok] = sec
    if cur_sec is not None:
        obs, frame, market, macro_now = emit(cur_sec)
        if obs is not None:
            yield cur_sec, obs, frame, market, macro_now


def _session_minutes_left(ts):
    """Minutes from the bar's IST wall-clock to the SESSION_CLOSE (15:30 IST),
    clamped ≥1 — the `minutes_to_close` the live PositionManager feeds into the
    expected move. `ts` is UTC epoch seconds (the vault stores the exchange
    timestamp; on the IST trading host that is true UTC epoch). India has no DST,
    fixed UTC+5:30, so IST-seconds-of-day = (ts + 19800) mod 86400. Vectorized."""
    # v9.9.18: date-aware. A June session really did end at 15:30 and an
    # August one ends at 15:40; one constant for both would either
    # fabricate ten minutes of June or truncate ten of August.
    from core import session_calendar as _SCC
    # `ts` is a UTC epoch (scalar or array), not a date — an earlier draft
    # referenced a `day` that was never in this scope and crashed the very
    # first cache build. Derive the session date from the FIRST timestamp
    # of the bar array: every array handed here belongs to one session, so
    # one lookup is both correct and cheap.
    _t0 = float(np.ravel(np.asarray(ts, np.float64))[0])
    _d = _dt.datetime.utcfromtimestamp(_t0 + 19800.0).date()   # IST date
    ch, cm = (int(x) for x in _SCC.session_close_hm(_d).split(":"))
    close_sod = ch * 3600 + cm * 60
    ist_sod = (np.asarray(ts, np.float64) + 19800.0) % 86400.0
    return np.maximum((close_sod - ist_sod) / 60.0, 1.0)


_HOLD_OVERRIDE_S: int | None = None      # set ONLY by tools/horizon_sweep


def set_hold_override(seconds: int | None) -> None:
    """v9.9.4: relabel the vault at a DIFFERENT hold horizon without
    touching config.MAX_HOLD_MINUTES (which is a live exit rule and a
    hash-bearing constant). The sweep sets this, generates samples,
    clears it. Sample caches key on it, so an override can never
    contaminate the production cache — see _meta_cache_path."""
    global _HOLD_OVERRIDE_S
    _HOLD_OVERRIDE_S = int(seconds) if seconds else None


def _hold_seconds(dte: float) -> int:
    """0-DTE-aware theta guillotine: the label horizon the live PositionManager
    actually enforces. v9.0 always used MAX_HOLD_MINUTES (45), grading expiry-day
    entries on a hold the live 25-minute guillotine never permits."""
    if _HOLD_OVERRIDE_S:
        # expiry-day guillotine is a separate physical constraint and is
        # never swept: 0-DTE gamma does not care what the research asks.
        if float(dte or 9.0) < config.EXPIRY_DTE_LT:
            return int(config.MAX_HOLD_MINUTES_0DTE * 60)
        return int(_HOLD_OVERRIDE_S)
    m = (config.MAX_HOLD_MINUTES_0DTE
         if float(dte or 9.0) < config.EXPIRY_DTE_LT
         else config.MAX_HOLD_MINUTES)
    return int(m * 60)


# v9.9: the shaped-barrier physics moved to core/meta_gate.py — ONE copy
# now feeds the label generator (here), the live gate's per-trade breakeven
# and the grader below. The alias keeps every existing call site intact.
from core.meta_gate import shaped_barriers as _shaped_barriers


def build_dataset(con, day: str):
    """Returns obs (N,5700), ts (N,) and a premium/barrier table for the
    realized-exit reward: prem[idx] = dict(ts → {leg: (bid, ask, lot, tp, sl,
    hold_s)}).

    v9.1 row changes: (tp, sl) are shaped from the ASK entry `e = ask` (live
    crosses on momentum; the old mid basis granted the half-spread as free
    edge), and hold_s carries the 0-DTE-aware theta-guillotine horizon for that
    index-day, so the reward grid stops assuming 45 minutes on expiry day."""
    obs_list, ts_list, prem = [], [], {i: {} for i in config.INDEX_ORDER}
    for ts, obs, _frame, market, macro_now in replay_day(con, day):
        obs_list.append(obs); ts_list.append(ts)
        mins = _session_minutes_left(ts)
        for idx, ctx in market.items():
            legs = ctx.get("legs") or {}
            spot = float((ctx.get("spot") or {}).get("ltp") or 0.0)
            T = float(ctx.get("T") or 0.0)
            lot = ctx.get("lot", 0)
            hold_s = _hold_seconds(ctx.get("dte", 9.0))
            snap = (macro_now or {}).get(idx)
            cw = snap.get("call_wall") if snap else None   # GEX walls cap the room
            pw = snap.get("put_wall") if snap else None     # (None ⇒ no cap)
            row = {}
            for leg in ("atm_ce", "atm_pe", "otm_ce", "otm_pe"):
                s = (legs.get(leg) or {}).get("snap")
                if not (s and s["bid"] and s["ask"]):
                    continue
                bid, ask = s["bid"], s["ask"]
                e = ask                                    # ★ live pays the cross
                K = float((legs.get(leg) or {}).get("strike") or 0.0)
                if spot > 0 and K > 0 and T > 0 and e > 0:
                    tp, sl = _shaped_barriers(e, spot, K, T, mins,
                                              leg.endswith("_ce"), cw, pw)
                    tp, sl = float(tp), float(sl)
                else:                                     # missing context ⇒ base
                    tp = e * (1.0 + config.BASE_TP_PCT)
                    sl = e * (1.0 - config.BASE_SL_PCT)
                row[leg] = (bid, ask, lot, tp, sl, hold_s)
            if row:
                prem[idx][ts] = row
    if not obs_list:
        return None, None, None
    return np.stack(obs_list), np.array(ts_list), prem


def _exit_price_from_path(bids: np.ndarray, tp: float, sl: float):
    """First-touch triple barrier on a forward BID path for a long entry whose
    SHAPED target is `tp` and stop is `sl`. Returns the realized exit PRICE:
    `tp` if the bid reaches it before `sl`, `sl` if hit first, else the last
    valid bid (theta / max-hold exit). NaNs (data gaps) are skipped. None if
    the path holds no valid bid. Same rule everywhere — SIDE (bandit) and SIZE
    (meta) models are graded on one identical realized payoff."""
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
    the ASK now (v9.1 — the live engine CROSSES on momentum entries; the old
    mid basis assumed a maker fill live rarely gets on a rising option), HOLD
    under the constitution's risk-managed exit, SELL at the triple-barrier exit
    price on bids, over the 0-DTE-aware guillotine carried on the row. ₹/lot."""
    leg = "atm_ce" if direction > 0 else "atm_pe"
    now = prem_idx.get(ts, {}).get(leg)
    if not now:
        return 0.0
    bid0, ask0, lot, tp, sl, hold_s = now
    e = float(ask0)
    horizon = int(hold_s)
    bids = np.fromiter(
        (prem_idx.get(ts + k, {}).get(leg, (np.nan,))[0]
         for k in range(1, horizon + 1)), dtype=np.float64, count=horizon)
    exitp = _exit_price_from_path(bids, tp, sl)
    if exitp is None:
        return 0.0
    return (exitp - e) * lot - round_trip_costs(e * lot, exitp * lot)


# ====================================================================
# SHARED LIVE-DECISION REPLAY (the v9.1 core). One engine drives BOTH the
# meta-labeler's signal generation and the promotion grader, and it is the
# brain's decision path via core/decision.py — so the exam finally tests the
# deployed system. Emits ("sec", t, ts) every replayed second and
# ("signal", dict) for every entry that clears the SAME gates live applies.
# ====================================================================
class _Replayer:
    def __init__(self, con, day: str, meta, cal, funnel=None, collect_ref=None):
        from collections import deque
        from simulation.replay_real_day import load_day
        # ONE load_day: the vault query carries no token filter, so ti/bidA/
        # askA/by_sec cover EVERY harvested token — all indices' spots + legs.
        loaded = load_day(con, day, config.TRADABLE[0])
        self.ok = loaded is not None
        if not self.ok:
            return
        _stok, self.by_sec, self.ti, self.bidA, self.askA = loaded
        self.con, self.day = con, day
        self.meta, self.cal = meta, cal
        self.funnel = funnel
        self.collect_ref = collect_ref          # list ← drift-reference rows
        self.mapper = AsOfMapper(dt.date.fromisoformat(day))
        self.levels = {i: _prev_levels(con, day, i) for i in config.TRADABLE}
        self.last_tick: dict[int, int] = {}
        self.persist = {i: D.PersistenceTracker() for i in config.TRADABLE}
        self.spot_hist = {i: deque(maxlen=1800) for i in config.TRADABLE}
        self.rvol = {i: EWMAVol() for i in config.TRADABLE}
        self.open_p: dict[str, float] = {}
        self.p945: dict[str, float] = {}
        self.last_try = {i: -1e9 for i in config.TRADABLE}
        self.entry_bar = config.entry_conviction_bar()
        # Known, documented replay degradations vs live (each fails toward
        # FEWER entries or identical behaviour): VIX 5-min spike bump = 0 (VIX
        # isn't in the replay market dict); vol-forecaster regime input = None
        # (its intraday state files are live-only) ⇒ no VOL_CRUSH label;
        # ann_vol for the governor's vol-target scaling = archive ATM IV.

    def run(self, decide, on_block=None, actions_fn=None, on_signal=None):
        """decide(obs, frame, iidx) -> raw policy conviction (pre-shock).
        v9.4: on_block(idx, gate, detail, ctx) — the counterfactual hook —
        fires where the replayer itself kills a signal (meta-veto near-miss,
        persistence, throttle); the grader wires it on the promotion day.

        v9.9.14: on_signal(idx, ctx) — the COMPLETE conviction stream, fired
        unconditionally the moment conviction is final and BEFORE any gate
        sees it. on_block alone is not a signal stream: it fires only where a
        signal was REJECTED, so it silently omits exactly the handful the
        live bar ACCEPTED — which are the highest-conviction seconds of the
        day. A bar sweep fed from on_block would therefore evaluate every
        candidate bar below the incumbent on a sample with its best members
        deleted, and would conclude those bars are worse than they are. This
        hook is read-only and changes no grading decision."""
        from simulation.scenario_engine import N
        _oh, _om = (int(x) for x in config.SESSION_OPEN.split(":"))
        open_sod = _oh * 3600 + _om * 60
        for ts, obs, frame, market, macro_now in replay_day(self.con, self.day):
            t = int((ts + 19800) % 86400) - open_sod
            if not (0 <= t < N):
                continue
            for tok in self.by_sec.get(t, {}):
                self.last_tick[tok] = t
            hm = _eval_hm(t)
            yield ("sec", t, ts)
            # v9.9.1 CROSS-INDEX PARITY: live builds _conv_all from ONE
            # policy action vector per frame; the sample generator does the
            # same (fresh deterministic predict). The grader used to serve
            # peer ZEROS here — scoring the model off the manifold it was
            # trained on (the exact skew decision._meta_x warns about).
            # Now: one full-vector predict per tick, same extraction
            # function, zero intra-tick staleness. Skipped when no meta is
            # loaded (nothing scores) or the flag is off.
            _cv_all = None
            if (actions_fn is not None and self.meta is not None
                    and bool(getattr(config, "META_CROSS_INDEX", False))):
                try:
                    from core.cross_index import convictions_from_actions
                    _cv_all = convictions_from_actions(
                        actions_fn(obs, frame), len(config.INDEX_ORDER))
                except Exception as _e:                    # noqa: BLE001
                    _cv_all = None
            for idx in config.TRADABLE:
                # v9.9.11: from 2026-08-03 the live brain suspends entries
                # during the cash Closing Auction (index constituents in
                # auction ⇒ spot is not a traded price). Grading an entry
                # there would train the meta on decisions serving can never
                # make — the same train/serve skew class as the peer-context
                # bug. Pre-reform days are untouched.
                from core import session_calendar as _SCG
                if not _SCG.entries_allowed(ts, day=self.day,
                                            index=idx)[0]:
                    if self.funnel is not None:
                        self.funnel.record(idx, "cas_auction",
                                           "session phase")
                    continue
                ctx = market.get(idx)
                spot = float(((ctx or {}).get("spot") or {}).get("ltp") or 0)
                if not ctx or spot <= 0:
                    if self.funnel:
                        self.funnel.record(idx, "no_market")
                    continue
                sh = self.spot_hist[idx]
                sh.append(spot)
                self.rvol[idx].update(spot, dt_s=1.0)
                self.open_p.setdefault(idx, spot)
                if idx not in self.p945 and t >= 1800:
                    self.p945[idx] = spot
                iidx = config.INDEX_ORDER.index(idx)
                node = frame[iidx * config.NODES_PER_INDEX]
                if self.collect_ref is not None:
                    # DRIFT REFERENCE POPULATION — all-tick, pre-gate (the live
                    # monitor pools every second; a signals-only reference
                    # biases every marginal toward extremes).
                    b0 = iidx * config.NODES_PER_INDEX
                    for _nd in (frame[b0], frame[b0 + 1], frame[b0 + 2]):
                        if _nd.any():
                            self.collect_ref.append(_nd.astype(np.float32))
                ai = float(decide(obs, frame, iidx))
                f30 = ((self.p945[idx] - self.open_p[idx]) / self.open_p[idx]
                       if idx in self.p945 and self.open_p.get(idx) else 0.0)
                mac = (macro_now or {}).get(idx)
                shock = D.compute_shock(
                    ai=ai, vpin=float(node[4]), dealer_inv=float(node[16]),
                    mac=mac, spot=spot, dte=float(ctx.get("dte", 9.0)),
                    levels=self.levels.get(idx), f30=f30, hm=hm)
                conv = D.fuse(ai, shock)
                dsum = sum(abs(b - a) for a, b in zip(sh, list(sh)[1:])) \
                    if len(sh) > 120 else 0.0
                er = (abs(sh[-1] - sh[0]) / dsum) if dsum > 0 else 0.5
                regime = regime_mod.classify(
                    spot=spot, trend_efficiency=er,
                    net_gex=(mac or {}).get("net_gex"),
                    flip=(mac or {}).get("flip"),
                    call_wall=(mac or {}).get("call_wall"),
                    put_wall=(mac or {}).get("put_wall"),
                    iv_rank=(mac or {}).get("iv_rank"),
                    realized_vol=self.rvol[idx].annualized(),
                    vol_regime=None, vol_z=None,
                    trend_sign=(1 if sh[-1] >= sh[0] else -1)
                    if len(sh) >= 2 else 0,
                    index=idx)
                conv = D.apply_regime(conv, regime.conv_mult)
                self.persist[idx].push(float(ts), conv)
                wp_meta = D.meta_win_prob(self.meta, frame, iidx,
                                          min(t / N, 1.0), er, f30,
                                          1 if conv > 0 else -1,
                                          conv_by_index=_cv_all)
                wp = D.blend_winprob(wp_meta, conv, self.cal)
                eff_bar = D.effective_bar(self.entry_bar, 0.0,
                                          (mac or {}).get("iv_rank"))
                # ---- v9.9: grade with the SAME v3 gate the brain runs.
                # Interval from the same artifact; p* from the same shaped
                # barriers on the ATM ask at t. ACI margin is 0 here — the
                # margin is a LIVE-serving adaptation; the exam measures
                # the model+EV core. Missing pieces ⇒ legacy bytes.
                _ivl = _econ = None
                if getattr(config, "META_GATE_MODE", "bar") == "ev":
                    _ivl = D.meta_win_interval(self.meta, frame, iidx,
                                               min(t / N, 1.0), er, f30,
                                               1 if conv > 0 else -1,
                                               conv_by_index=_cv_all)
                    if _ivl is not None:
                        try:
                            from core.meta_gate import candidate_economics
                            _d_g = "CE" if conv > 0 else "PE"
                            for _r in self.mapper.hierarchy(idx, spot, _d_g):
                                _k = self.ti.get(_r["token"])
                                if (_k is None or t - self.last_tick.get(
                                        _r["token"], -99) > 5):
                                    continue
                                _a_g = self.askA[_k, t]
                                if np.isnan(_a_g) or _a_g <= 0:
                                    continue
                                _econ = candidate_economics(
                                    float(_a_g), spot, float(_r["strike"]),
                                    float(ctx.get("T") or 0.0),
                                    max((N - t) / 60.0, 1.0),
                                    _d_g == "CE", int(_r["lot"]),
                                    (mac or {}).get("call_wall"),
                                    (mac or {}).get("put_wall"))
                                break
                        except Exception:                  # noqa: BLE001
                            _econ = None
                if on_signal is not None:
                    on_signal(idx, {"t": t, "ts": ts, "conv": conv, "wp": wp,
                                    "direction": "CE" if conv > 0 else "PE",
                                    "spot": spot, "mac": mac, "hm": hm,
                                    "eff_bar": eff_bar,
                                    "dte": (ctx or {}).get("dte", 9.0),
                                    "T": (ctx or {}).get("T") or 0.01})
                gate = D.entry_gate_v3(conv, wp, wp_meta, eff_bar, _ivl,
                                       _econ[0] if _econ else None)
                if not gate.ok:
                    if self.funnel:
                        self.funnel.record(idx, "below_bar", gate.reason)
                    if on_block is not None and abs(conv) >= (
                            getattr(gate, "floor", self.entry_bar)
                            - config.CF_NEAR_MISS):
                        on_block(idx, "below_bar", gate.reason,
                                 {"t": t, "ts": ts, "conv": conv, "wp": wp,
                                  "direction": "CE" if conv > 0 else "PE",
                                  "spot": spot, "mac": mac, "hm": hm,
                                  "dte": (ctx or {}).get("dte", 9.0),
                                  "T": (ctx or {}).get("T") or 0.01})
                    continue
                pok, pwhy, _ = self.persist[idx].check(conv, sh, gate.floor)
                if not pok:
                    if self.funnel:
                        self.funnel.record(idx, "not_persistent", pwhy)
                    if on_block is not None:
                        on_block(idx, "not_persistent", pwhy,
                                 {"t": t, "ts": ts, "conv": conv, "wp": wp,
                                  "direction": "CE" if conv > 0 else "PE",
                                  "spot": spot, "mac": mac, "hm": hm,
                                  "dte": (ctx or {}).get("dte", 9.0),
                                  "T": (ctx or {}).get("T") or 0.01})
                    continue
                if t - self.last_try[idx] < config.ENTRY_ATTEMPT_THROTTLE_S:
                    if self.funnel:
                        self.funnel.record(idx, "throttled")
                    if on_block is not None:
                        on_block(idx, "throttled", "",
                                 {"t": t, "ts": ts, "conv": conv, "wp": wp,
                                  "direction": "CE" if conv > 0 else "PE",
                                  "spot": spot, "mac": mac, "hm": hm,
                                  "dte": (ctx or {}).get("dte", 9.0),
                                  "T": (ctx or {}).get("T") or 0.01})
                    continue
                self.last_try[idx] = t
                yield ("signal", {
                    "t": t, "ts": ts, "idx": idx, "iidx": iidx, "conv": conv,
                    "wp": wp, "wp_meta": wp_meta,
                    "direction": "CE" if conv > 0 else "PE",
                    "frame": frame, "er": er, "f30": f30, "spot": spot,
                    "mac": mac, "ctx": ctx, "hm": hm, "gate": gate.reason})


def _prev_levels(con, day: str, index: str) -> dict | None:
    """Prev-day PDH/PDL/PDC for the shared level-break shock — REAL vault spot
    ticks from the most recent harvested day before `day` (the brain fetches
    the same candle from Kite historical). None when no earlier day exists;
    the shock term then degrades to zero, exactly like live without candles."""
    tok = spot_token_for(con, day, index)
    if not tok:
        return None
    r = con.execute(
        "SELECT MAX(date(ts_local_ms/1000,'unixepoch','localtime')) "
        "FROM ticks_v9 WHERE token=? AND "
        "date(ts_local_ms/1000,'unixepoch','localtime')<?", (tok, day)).fetchone()
    prev = r[0] if r else None
    if not prev:
        return None
    hl = con.execute(
        "SELECT MAX(ltp), MIN(ltp) FROM ticks_v9 WHERE token=? AND ltp>0 AND "
        "date(ts_local_ms/1000,'unixepoch','localtime')=?", (tok, prev)).fetchone()
    c = con.execute(
        "SELECT ltp FROM ticks_v9 WHERE token=? AND ltp>0 AND "
        "date(ts_local_ms/1000,'unixepoch','localtime')=? "
        "ORDER BY ts_ms DESC LIMIT 1", (tok, prev)).fetchone()
    if not hl or not hl[0]:
        return None
    return {"pdh": float(hl[0]), "pdl": float(hl[1]),
            "pdc": float(c[0]) if c else None}


# ====================================================================
# META-LABELER (López de Prado: primary model picks the SIDE; this
# secondary model learns the SIZE as P(win) from TRIPLE-BARRIER outcomes
# on the vault's real recorded prices, after real costs). Pure numpy.
# v9.1: signals come from the SHARED live decision path; entries are ASK-
# based; horizon is 0-DTE-aware; samples carry AFML uniqueness weights.
# ====================================================================
def _kelly_budget(equity: float) -> float:
    b = config.BASE_TP_PCT / config.BASE_SL_PCT
    p = config.PAPER_EXPLORE_WINPROB
    k = max(p - (1 - p) / b, 0.0)
    return min(equity * config.MAX_KELLY_BUDGET_PCT,
               equity * k * config.KELLY_FRACTION)


def _gen_meta_samples(con, day: str):
    """One day, ALL tradable indices, brain-identical signal generation.
    Returns (X, Y, W, R): features, win labels, uniqueness weights, drift rows.
    Affordability uses the static Kelly budget on the ASK we would pay (the
    label-time sizer, as before; the GRADER uses the dynamic governor)."""
    RET = []   # v10: barrier P&L per sample
    ECON = []  # v9.9: (entry_ask, tp, sl, lot) per sample — the payoff
    #            geometry the label was graded on; meta_gate_replay prices
    #            each sample's OWN breakeven p* from exactly this.
    from simulation.scenario_engine import N
    import math as _m
    R: list = []
    rep = _Replayer(con, day, meta=None, cal={}, funnel=None, collect_ref=R)
    if not rep.ok:
        return [], [], [], [], [], []
    pol = HeuristicPolicy()

    def decide(obs, frame, iidx):
        return float(pol.predict(frame)[2 * iidx])

    # v9.1.1: affordability at the FIXED reference account — the exam measures
    # edge, never live account size. TRADING_CAPITAL is a live-only knob now.
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)
    X, Y, spans = [], [], []                    # spans: (idx, t_in, t_out)
    for ev in rep.run(decide,
                      actions_fn=lambda _o, _f: pol.predict(_f)):
        if ev[0] != "signal":
            continue
        s = ev[1]
        idx, t, d = s["idx"], s["t"], s["direction"]
        pick = None
        for r in rep.mapper.hierarchy(idx, s["spot"], d):
            k = rep.ti.get(r["token"])
            if k is None or t - rep.last_tick.get(r["token"], -99) > 5:
                continue
            b_, a_ = rep.bidA[k, t], rep.askA[k, t]
            if np.isnan(b_) or np.isnan(a_) or a_ <= 0:
                continue
            if a_ * r["lot"] <= budget:          # afford the ASK we actually pay
                pick = (k, float(a_), int(r["lot"]), float(r["strike"]))
                break
        if pick is None:
            continue
        k, e, lot, Kstrike = pick                # e = ASK ★
        dte = float(s["ctx"].get("dte", 9.0))
        horizon = _hold_seconds(dte)
        mins_left = max((N - t) / 60.0, 1.0)
        T_ = float(s["ctx"].get("T") or 0.0)
        cw = (s["mac"] or {}).get("call_wall")
        pw = (s["mac"] or {}).get("put_wall")
        if s["spot"] > 0 and Kstrike > 0 and T_ > 0 and e > 0:
            tp, sl = _shaped_barriers(e, s["spot"], Kstrike, T_, mins_left,
                                      d == "CE", cw, pw)
            tp, sl = float(tp), float(sl)
        else:
            tp, sl = e * (1 + config.BASE_TP_PCT), e * (1 - config.BASE_SL_PCT)
        seg = rep.bidA[k, t + 1:t + 1 + horizon]
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
        pnl = (exitp - e) * lot - round_trip_costs(e * lot, exitp * lot)
        b0 = s["iidx"] * config.NODES_PER_INDEX
        frame = s["frame"]
        # CROSS-INDEX PEER CONTEXT — must match core/decision.meta_win_prob
        # exactly. Both extract conviction through
        # cross_index.convictions_from_actions and append the same 3 features
        # in the same order, AFTER the original 61.
        _peer = []
        if bool(getattr(config, "META_CROSS_INDEX", False)):
            from core.cross_index import (peer_features,
                                          convictions_from_actions)
            _cv = convictions_from_actions(pol.predict(frame),
                                           len(config.INDEX_ORDER))
            _peer = peer_features(_cv, s["iidx"], d)
        x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2],
                            [t / N, s["er"],
                             _m.copysign(min(abs(s["f30"]) * 100, 3), s["f30"])
                             if s["f30"] else 0.0,
                             1.0 if d == "CE" else -1.0],
                            _peer]).astype(np.float32)
        X.append(x)
        Y.append(1.0 if pnl > 0 else 0.0)
        RET.append(float(pnl))
        ECON.append((float(e), float(tp), float(sl), int(lot)))
        spans.append((idx, t, t + off + 1))
    # ---- AFML uniqueness: w_i = mean over the label span of 1/concurrency ----
    W = np.ones(len(X), np.float32)
    for idx in config.TRADABLE:
        rows = [j for j, (i2, _, _) in enumerate(spans) if i2 == idx]
        if not rows:
            continue
        c = np.zeros(N + 1, np.int32)
        for j in rows:
            _, a, b = spans[j]
            c[a:min(b, N)] += 1
        for j in rows:
            _, a, b = spans[j]
            seg = c[a:min(b, N)]
            if seg.size:
                W[j] = float(np.mean(1.0 / np.maximum(seg, 1)))
    return X, Y, list(W), R, RET, ECON


def train_meta(con, days: list[str]):
    """Triple-barrier labels → UNIQUENESS-WEIGHTED logistic P(win|x), holdout
    BY DAY (the last training day). Saved atomically; the live brain blends it
    into the Kelly win-probability. Returns (model_dict|None, diag_dict)."""
    if getattr(config, "META_TRAIN_MAX_DAYS", 0) > 0:
        days = days[-config.META_TRAIN_MAX_DAYS:]   # v10.2 bounded window
    perday = []
    dee_rows = []
    R_all: list = []
    # v9.9.2: fill stale sample caches IN PARALLEL first (atomic writes,
    # own sqlite per worker), then the serial loop below is pure cache
    # reads. Same bytes, same order, fraction of the wall time.
    try:
        from core.parallel_days import map_days as _mapd
        _stale = [d for d in days if not _meta_cache_fresh(d)]
        if len(_stale) > 1:
            _mapd(_prime_meta_samples_worker, _stale,
                  desc="meta sample prime")
    except Exception as _e:                                # noqa: BLE001
        log.warning("sample prime skipped (%s) — serial path continues", _e)
    for day in days:
        x, y, w, r, ret, _ec = _gen_meta_samples_cached(con, day)
        perday.append((day, x, y, w))
        dee_rows += [(day, xi, ri, wi) for xi, ri, wi in zip(x, ret, w)]
        R_all += r
        log.info("meta samples %s: %d signals (mean uniq %.3f)", day, len(x),
                 float(np.mean(w)) if w else float("nan"))
    n = sum(len(x) for _, x, _, _ in perday)
    diag = {"n": n, "per_day": {d: len(x) for d, x, _, _ in perday}}

    # ---- v9.9.25: publish the training matrix for the PAYOFF target.
    # core/payoff_target.py predicts R = P&L / initial risk instead of
    # P(win). The equity meta scored AUC 0.5210 on 2026-08-11 against a
    # detectability floor of 0.587-0.658 at this n_eff — sign prediction on
    # index options is not resolvable here, while dispersion is (the same
    # reason rv_forecaster shows skill and the meta does not).
    #
    # Everything that target needs is ALREADY computed above and then
    # discarded: `ret` is the per-sample barrier P&L and ECON carries
    # (entry_ask, tp, sl, lot) — the payoff geometry the label was graded
    # on. Risk is (entry_ask - sl) * lot, so R = ret / risk needs no new
    # replay and no join to the shadow ledger. Writing it costs one file.
    try:
        import numpy as _np
        _Xs, _rets, _risks, _days, _ws = [], [], [], [], []
        for _d in days:
            _x, _y, _w, _r, _ret, _ec = _gen_meta_samples_cached(con, _d)
            for _i in range(len(_x)):
                try:
                    _ea, _tp, _sl, _lot = _ec[_i]
                    _risk = abs(float(_ea) - float(_sl)) * float(_lot)
                except (IndexError, TypeError, ValueError):
                    continue
                if _risk <= 0:
                    continue
                _Xs.append(_x[_i]); _rets.append(float(_ret[_i]))
                _risks.append(_risk); _days.append(_d); _ws.append(float(_w[_i]))
        if _Xs:
            _mp = config.STATE_DIR / "meta_train_matrix.npz"
            _mp.parent.mkdir(parents=True, exist_ok=True)
            _tmp = _mp.with_suffix(".tmp.npz")
            # np.savez appends .npz to a name that lacks it — the 2026-08-05
            # WinError 5 that broke every atomic cache publication. The name
            # already ends in .npz, so os.replace sees the file it expects.
            _np.savez(_tmp, X=_np.asarray(_Xs, _np.float32),
                      ret=_np.asarray(_rets, _np.float64),
                      risk=_np.asarray(_risks, _np.float64),
                      day=_np.asarray(_days), w=_np.asarray(_ws, _np.float64),
                      config_hash=_np.asarray([config.CONFIG_HASH]))
            import os as _os
            _os.replace(_tmp, _mp)
            log.info("payoff matrix published: %d row(s) x %d feature(s) "
                     "over %d day(s) -> %s", len(_Xs),
                     _np.asarray(_Xs).shape[1], len(set(_days)), _mp)
    except Exception as _e:                                # noqa: BLE001
        log.warning("payoff matrix not published (%s) — the meta path is "
                    "unaffected; only the R-target study is skipped", _e)
    # DRIFT REFERENCE — the ALL-TICK feature world of the training pool. Built
    # BEFORE the sample-count gate below (v9.1.2 fix): the reference has NO
    # dependency on the meta model — it is the population the live monitor
    # compares against — and gating it behind META_MIN_TRAIN left the brain on
    # NO_REF through the entire bootstrap phase (2026-07-03/04: two runs with
    # ~1.6M perfectly good feature rows wrote nothing).
    try:
        from core.drift_monitor import build_reference
        if len(R_all) >= 200:
            build_reference(np.stack(R_all),
                            model_version=time.strftime("meta_%Y%m%d_%H%M%S"))
        else:
            log.info("drift reference skipped: only %d feature rows", len(R_all))
    except Exception as e:                               # noqa: BLE001
        log.error("drift reference failed: %s", e)
    if n < config.META_MIN_TRAIN:
        log.info("meta-labeler: %d/%d labeled signals — keep harvesting",
                 n, config.META_MIN_TRAIN)
        return None, diag
    # ---- META-FORGE v2: purged-CV LightGBM + isotonic (v9.8). Falls back
    # to the proven logistic below if lightgbm is absent or CV is too thin —
    # a missing package can never cost a nightly.
    if getattr(config, "META_ENGINE", "logit") == "gbm":
        _nsamp = sum(len(x) for _, x, _, _ in perday)
        try:
            import lightgbm as _lgb            # explicit: is the package here?
            _have_lgb = True
        except Exception:                                 # noqa: BLE001
            _have_lgb = False
        try:
            from core import meta_gbm as MG
            out = MG.fit_gbm(perday, config.META_MIN_TRAIN)
        except Exception as e:                            # noqa: BLE001
            log.error("meta-gbm failed (%s) — falling back to logistic", e)
            out = None
        if out is not None:
            tmp = config.META_MODEL_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(out))
            tmp.replace(config.META_MODEL_PATH)
            log.info("META-FORGE v2: %d signals | OOF Brier %.4f→%.4f "
                     "(raw→isotonic) | acc %.1f%% | %s | fit %.1fs → %s",
                     out["n"], out["oof_brier_raw"], out["oof_brier_cal"],
                     100 * out["holdout_acc"], out["holdout"],
                     out["fit_seconds"], config.META_MODEL_PATH.name)
            diag.update({"base_rate": out["base_rate"],
                         "holdout_acc": out["holdout_acc"],
                         "engine": "gbm",
                         "oof_brier_cal": out["oof_brier_cal"],
                         # AUDIT (2026-07-24): the skill verdict was computed
                         # and logged but never PERSISTED, so nightly skill
                         # could not be tracked across runs — the one number
                         # that says whether the gate model beats always
                         # predicting the base rate. It belongs in the report.
                         "brier_climatology": out.get("brier_climatology"),
                         "bss_cal": out.get("bss_cal"),
                         "bss_raw": out.get("bss_raw"),
                         "auc_raw": out.get("auc_raw"),
                         "auc_cal": out.get("auc_cal"),
                         "oof_spread_p05_p95":
                             out.get("oof_spread_p05_p95"),
                         "breakeven_p": out.get("breakeven_p")})
            try:                       # v10 DISTRIBUTIONAL EDGE (fail-open)
                from core import dist_edge as DE
                dee = DE.fit_dee(dee_rows, config.META_MIN_TRAIN)
                if dee:
                    (config.STATE_DIR / "dist_edge.json").write_text(
                        json.dumps(dee))
                    log.info("DIST-EDGE: n=%d cover[q10,q90]=%.2f "
                             "pinball(q50)=%.4f fit %.1fs", dee["n"],
                             dee["oof_cover_q10_q90"], dee["pinball_q50"],
                             dee["fit_seconds"])
                    diag["dee_cover"] = dee["oof_cover_q10_q90"]
            except Exception as e_:                       # noqa: BLE001
                log.warning("dist-edge skipped: %s", e_)
            return out, diag
        if _have_lgb:
            # v9.9.3: lightgbm is INSTALLED and fit_gbm still returned None —
            # the guard refused (AUC/positives/CV): THESE SAMPLES CARRY NO
            # SIGNAL. On 2026-08-01 this fell through and trained the pre-v2
            # logistic on the very samples the guard had just rejected,
            # wrote it stamped with the CURRENT hash, and the promotion-day
            # grader served its floored 0.50 for 18k index-seconds — the
            # exact disease Meta-Forge v2 was built to end. A refusal now
            # means NO artifact of ANY engine: conviction bar governs, the
            # v3 gate stays dormant until real ordering ability exists.
            log.warning("META refusal honored: lightgbm present, guard said "
                        "no signal — logistic fallback SKIPPED, no artifact "
                        "written, conviction bar governs.")
            return None, diag
        if not _have_lgb:
            log.warning("META-FORGE: engine=gbm requested but lightgbm is NOT "
                        "importable — using logistic. Fix: pip install "
                        "lightgbm (in the interpreter that runs the forge).")
        elif _nsamp < config.META_MIN_TRAIN:
            log.warning("META-FORGE: engine=gbm, lightgbm present, but only "
                        "%d labeled equity signals < gate %d — logistic this "
                        "run. This resolves as the vault deepens (same gate "
                        "the commodity forge reports).", _nsamp,
                        config.META_MIN_TRAIN)
        else:
            log.warning("META-FORGE: gbm present with %d≥%d samples yet "
                        "fit_gbm returned None (CV too thin on the day folds) "
                        "— logistic this run.", _nsamp, config.META_MIN_TRAIN)
    # day-based holdout: last day with samples; fall back to time-ordered 20%
    # only if that day is too thin to judge on.
    ho_day = next((d for d, x, _, _ in reversed(perday) if x), None)
    ho_n = len(next(x for d, x, _, _ in reversed(perday) if d == ho_day))
    Xtr, Ytr, Wtr, Xho, Yho, Who = [], [], [], [], [], []
    if len(days) >= 2 and ho_n >= 20 and ho_n < n:
        for d, x, y, w in perday:
            tgt = (Xho, Yho, Who) if d == ho_day else (Xtr, Ytr, Wtr)
            tgt[0].extend(x); tgt[1].extend(y); tgt[2].extend(w)
        split = f"day:{ho_day}"
    else:
        allX = [x for _, xs, _, _ in perday for x in xs]
        allY = [y for _, _, ys, _ in perday for y in ys]
        allW = [w for _, _, _, ws in perday for w in ws]
        cut = int(len(allX) * 0.8)
        Xtr, Ytr, Wtr = allX[:cut], allY[:cut], allW[:cut]
        Xho, Yho, Who = allX[cut:], allY[cut:], allW[cut:]
        split = "row:80/20 (single-day fallback)"
        ho_day = None
    Xtr = np.stack(Xtr); Ytr = np.asarray(Ytr, np.float32)
    Wn = np.asarray(Wtr, np.float32)
    Wn = Wn / max(float(Wn.mean()), 1e-9)        # mean-1 so LR stays comparable
    # sd floor must survive the round(5) serialization the brain divides by.
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-4)
    Ztr = (Xtr - mu) / sd
    w = np.zeros(Ztr.shape[1], np.float32); b = 0.0
    ntr = len(Ztr)
    for _ in range(config.META_EPOCHS):
        p = 1 / (1 + np.exp(-(Ztr @ w + b)))
        g = Ztr.T @ ((p - Ytr) * Wn) / ntr + config.META_L2 * w
        w -= config.META_LR * g
        b -= config.META_LR * float(((p - Ytr) * Wn).mean())
    acc = acc_w = None
    if Xho:
        Zho = (np.stack(Xho) - mu) / sd
        Yh = np.asarray(Yho, np.float32)
        Wh = np.asarray(Who, np.float32)
        ph = 1 / (1 + np.exp(-(Zho @ w + b)))
        hit = ((ph > 0.5) == (Yh > 0.5)).astype(np.float32)
        acc = float(hit.mean())
        acc_w = float((hit * Wh).sum() / max(Wh.sum(), 1e-9))
    uniq = float(np.mean([u for _, _, _, ws in perday for u in ws])) if n else None
    out = {"w": w.round(5).tolist(), "b": round(float(b), 5),
           "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(),
           "n": n, "base_rate": round(float(np.mean(Ytr)), 4),
           "holdout_acc": acc, "holdout_acc_w": acc_w,
           "holdout": split, "uniqueness_mean": round(uniq, 4) if uniq else None,
           "days": days, "ts": time.time(), "config_hash": config.CONFIG_HASH}
    tmp = config.META_MODEL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(out)); tmp.replace(config.META_MODEL_PATH)
    log.info("meta-labeler trained: %d signals (mean uniq %.3f), base win "
             "%.1f%%, holdout[%s] acc %s (uniq-weighted %s) → %s",
             n, uniq or 0.0, 100 * float(np.mean(Ytr)), split,
             f"{acc:.1%}" if acc is not None else "—",
             f"{acc_w:.1%}" if acc_w is not None else "—",
             config.META_MODEL_PATH.name)
    diag.update({"base_rate": out["base_rate"], "holdout_acc": acc,
                 "holdout_acc_w": acc_w, "holdout": split,
                 "uniqueness_mean": out["uniqueness_mean"]})
    return out, diag


def _load_rl() -> bool:
    """Import torch/gym/SB3 and define the SAC-side classes, once, on demand.
    Returns False (with a log) if the stack is absent — callers skip SAC."""
    global torch, gym, SAC, DummyVecEnv, VecNormalize, HAVE_RL, ForgeEnv, Extractor
    if HAVE_RL:
        return True
    try:
        import torch as _torch
        import gymnasium as _gym
        from stable_baselines3 import SAC as _SAC
        from stable_baselines3.common.vec_env import (DummyVecEnv as _DVE,
                                                      VecNormalize as _VN)
    except Exception as e:                              # noqa: BLE001
        log.info("RL stack unavailable (%s) — SAC paths skipped", e)
        return False
    torch, gym, SAC, DummyVecEnv, VecNormalize = _torch, _gym, _SAC, _DVE, _VN

    class ForgeEnv(gym.Env):
        """Offline single-step bandit over logged seconds (each step is an
        independent decision graded by realized after-cost option PnL —
        honest about what logged data can support). In v9.1 it exists to host
        VecNormalize; training runs in train_bandit."""
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

    from core.graph_constructor import TGNFeatureExtractor
    import stable_baselines3.common.torch_layers as tl

    class Extractor(tl.BaseFeaturesExtractor):
        def __init__(self, observation_space):
            super().__init__(observation_space, config.PROJ_DIM)
            self.net = TGNFeatureExtractor()

        def forward(self, x):
            return self.net(x)



    HAVE_RL = True
    return True

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


def _eval_hm(t: int) -> str:
    """Seconds-from-open → 'HH:MM', so RiskGovernor's entry curfew
    (NO_ENTRY_AFTER) fires on the same wall-clock the brain sees."""
    base = dt.datetime(2000, 1, 1,
                       *(int(x) for x in config.SESSION_OPEN.split(":")))
    return (base + dt.timedelta(seconds=int(t))).strftime("%H:%M")


def _shadow_trade(rep, s, t):
    """v9.4 counterfactual grade of a BLOCKED signal: the trade the gate
    refused, priced and exited exactly as the grader would — first fresh
    two-sided rung within the spread cap (NO governor, NO affordability: the
    question is the signal's worth, not the account's size), ASK entry,
    shaped barriers on the snapshot walls, 0-DTE-aware hold, real costs.
    Returns after-cost ₹, or None when history cannot fill it."""
    from simulation.scenario_engine import N
    d = s["direction"]
    try:
        rows = rep.mapper.hierarchy(s["idx"], s["spot"], d)
    except Exception:                                     # noqa: BLE001
        return None
    pick = None
    for r in rows:
        kk = rep.ti.get(r["token"])
        if kk is None or t - rep.last_tick.get(r["token"], -99) > 5:
            continue
        b_, a_ = rep.bidA[kk, t], rep.askA[kk, t]
        if np.isnan(b_) or np.isnan(a_) or b_ <= 0 or a_ <= 0:
            continue
        mid = (b_ + a_) / 2.0
        if (a_ - b_) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
            continue
        pick = (kk, float(a_), int(r["lot"]), float(r["strike"]))
        break
    if pick is None:
        return None
    kk, e, lot, K = pick
    horizon = _hold_seconds(float(s.get("dte") or 9.0))
    mins_left = max((N - t) / 60.0, 1.0)
    T_ = float(s.get("T") or 0.01)
    cw = (s.get("mac") or {}).get("call_wall")
    pw = (s.get("mac") or {}).get("put_wall")
    if s["spot"] > 0 and K > 0 and T_ > 0 and e > 0:
        tp, sl = _shaped_barriers(e, s["spot"], K, T_, mins_left,
                                  d == "CE", cw, pw)
        tp, sl = float(tp), float(sl)
    else:
        tp, sl = e * (1 + config.BASE_TP_PCT), e * (1 - config.BASE_SL_PCT)
    seg = rep.bidA[kk, t + 1:t + 1 + horizon]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None
    itp = int(np.argmax(seg >= tp)) if np.any(seg >= tp) else None
    isl = int(np.argmax(seg <= sl)) if np.any(seg <= sl) else None
    if itp is not None and (isl is None or itp < isl):
        exitp = float(tp)
    elif isl is not None:
        exitp = float(sl)
    else:
        v = np.nonzero(~np.isnan(seg))[0]
        exitp = float(seg[v[-1]])
    return float((exitp - e) * lot
                 - round_trip_costs(e * lot, exitp * lot))


def _grade_day(con, day: str, decide, meta, cal, funnel=None,
               attribution=None, on_entry=None, actions_fn=None):
    """After-cost ₹ a policy would have ACTUALLY realized on `day`, sized
    EXACTLY like live: ONE RiskGovernor across ALL tradable indices (the live
    MAX_CONCURRENT_POSITIONS=1 world — v9.0's per-index governors let phantom
    parallel books inflate scores), every entry through the real
    first_affordable (Kelly budget, ATM→OTM walk, disaster-floor check, entry
    curfew, cooldown, lockout), the SHARED decision path for signals, ASK
    entry, dte-aware barriers, bid exits. Returns (₹, stats).

    v9.7.1: `on_entry(info)` is an OPTIONAL, purely-observational callback fired
    at the instant a trade is entered, receiving the entry premium, the forward
    bid segment, conviction, and the normal first-touch outcome. It CANNOT
    change grading — read-only telemetry for tools/fast_lane_report.py to
    compute the fast-lane counterfactual on the identical entry+segment the
    forge graded. When None (training, exams, every existing caller), this
    function is byte-identical to before."""
    from core.risk_manager import RiskGovernor
    from simulation.scenario_engine import N
    rep = _Replayer(con, day, meta, cal, funnel)
    if not rep.ok:
        return 0.0, {"trades": 0, "wins": 0, "no_data": True}
    # v9.1.1: the exam's governor runs the FULL constitution (drawdown halt,
    # cooldown, lockout, curfew, one position, Kelly walk) at the fixed
    # REFERENCE account — invariant to the live TRADING_CAPITAL knob, so the
    # exam grades edge, not account size. (2026-07-03: a live 60k→5k edit
    # between session and forge run silently changed what was being examined.)
    risk = RiskGovernor(capital=config.FORGE_EVAL_CAPITAL)
    total = 0.0
    trades = wins = 0
    open_pos = None                    # (exit_t, outlay, pnl, dir, idx)
    # AUDIT (2026-07-24): the graded sample used to be the FIRST
    # CF_MAX_PER_GATE blocks in replay order — i.e. the opening minutes of the
    # session — and the remaining ~17k were merely counted. Any verdict drawn
    # from it ("0 wins in 400") therefore described the open, not the day, and
    # the open is its own regime (gap resolution, wide spreads, high vol).
    # Reservoir sampling (Algorithm R) keeps the same cost and the same 400
    # gradings, but makes them a UNIFORM RANDOM sample of every block in the
    # window, so the counterfactual finally describes what it claims to.
    _cf_pool: dict = {}
    _cf_rng = np.random.default_rng(20260724)

    def _cf(gate_, s_=None, t_=None, count_only=False):
        """Counterfactual accumulator: n counts every block; a uniform random
        sample of up to CF_MAX_PER_GATE per gate is graded after the replay."""
        if attribution is None:
            return
        a_ = attribution.setdefault(gate_, {"n": 0, "graded": 0, "sum": 0.0,
                                            "wins": 0, "capped": 0})
        a_["n"] += 1
        if count_only or s_ is None:
            return
        pool = _cf_pool.setdefault(gate_, [])
        k = int(config.CF_MAX_PER_GATE)
        seen = a_["n"]
        if len(pool) < k:
            pool.append((dict(s_), t_))
        else:                              # replace with probability k/seen
            j = int(_cf_rng.integers(0, seen))
            if j < k:
                pool[j] = (dict(s_), t_)

    def _cf_grade_pool():
        """Grade the sampled blocks once the replay has finished."""
        for gate_, pool in _cf_pool.items():
            a_ = attribution.setdefault(gate_, {"n": 0, "graded": 0,
                                                "sum": 0.0, "wins": 0,
                                                "capped": 0})
            for s_, t_ in pool:
                try:
                    pnl_ = _shadow_trade(rep, s_, t_)
                except Exception as e_:                   # noqa: BLE001
                    log.warning("counterfactual shadow failed (%s): %s — "
                                "telemetry degrades, the exam continues",
                                gate_, e_)
                    continue
                if pnl_ is None:
                    continue
                a_["graded"] += 1
                a_["sum"] += pnl_
                a_["wins"] += int(pnl_ > 0)
            a_["capped"] = max(a_["n"] - a_["graded"], 0)
            a_["sampling"] = "uniform random (reservoir)"

    _hook = (lambda i_, g_, d_, c_: _cf(g_, {"idx": i_, **c_}, c_["t"])) \
        if attribution is not None else None
    for ev in rep.run(decide, on_block=_hook, actions_fn=actions_fn):
        if ev[0] == "sec":
            t = ev[1]
            if open_pos is not None and t >= open_pos[0]:
                risk.register_exit(open_pos[1], open_pos[2], open_pos[3],
                                   ts=float(open_pos[0]))
                total += open_pos[2]
                trades += 1
                wins += int(open_pos[2] > 0)
                open_pos = None
            risk.on_tick()
            continue
        s = ev[1]
        idx, t = s["idx"], s["t"]
        if open_pos is not None:                          # global 1-position cap
            if funnel:
                funnel.record(idx, "in_position")
            continue
        if risk.halted:
            if funnel:
                funnel.record(idx, "risk_halted")
            continue
        d = s["direction"]
        hierarchy = []
        skip = {"unharvested": 0, "stale>5s": 0, "one-sided": 0, "spread>cap": 0}
        for r in rep.mapper.hierarchy(idx, s["spot"], d):
            kk = rep.ti.get(r["token"])
            if kk is None:
                skip["unharvested"] += 1
                continue
            if t - rep.last_tick.get(r["token"], -99) > 5:
                skip["stale>5s"] += 1
                continue
            b_, a_ = rep.bidA[kk, t], rep.askA[kk, t]
            if np.isnan(b_) or np.isnan(a_) or b_ <= 0 or a_ <= 0:
                skip["one-sided"] += 1
                continue
            mid = (b_ + a_) / 2.0
            if (a_ - b_) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
                skip["spread>cap"] += 1                   # live illiquidity gate
                continue
            hierarchy.append({"premium": float(mid), "lot": int(r["lot"]),
                              "symbol": r["symbol"], "exchange": r["exchange"],
                              "price": float(a_), "_k": kk,
                              "_strike": float(r["strike"])})
        if not hierarchy:
            if funnel:                       # WHY the whole ladder died (v9.1.1)
                why = " ".join(f"{k}:{n}" for k, n in skip.items() if n)
                funnel.record(idx, "no_quotes", f"ladder dead — {why}")
            _cf("no_quotes", count_only=True)
            continue
        leg, permit = risk.first_affordable(
            hierarchy, direction=d, win_prob=s["wp"],
            sl_pct=config.BASE_SL_PCT, tp_pct=config.BASE_TP_PCT,
            data_age_s=0.0, now_hm=s["hm"], ts=float(t),
            ann_vol=(s["mac"] or {}).get("atm_iv") or None)
        if leg is None:
            if funnel:                       # the permit's own words (v9.1.1)
                funnel.record(idx, "risk_blocked",
                              str(permit.reason)[:80] if permit else "blocked")
            _cf("risk_blocked", {"idx": idx, **{k: s[k] for k in ("t", "ts", "conv", "wp", "direction", "spot", "mac", "hm")}, "dte": s.get("dte") or (s.get("ctx") or {}).get("dte", 9.0), "T": s.get("T") or (s.get("ctx") or {}).get("T") or 0.01}, t)
            continue
        kk, lot, Kstrike = leg["_k"], leg["lot"], leg["_strike"]
        e = float(leg["price"])                           # ASK — live crosses ★
        horizon = _hold_seconds(s["ctx"].get("dte", 9.0))
        mins_left = max((N - t) / 60.0, 1.0)
        T_ = float(s["ctx"].get("T") or 0.01)
        cw = (s["mac"] or {}).get("call_wall")
        pw = (s["mac"] or {}).get("put_wall")
        if s["spot"] > 0 and Kstrike > 0 and T_ > 0 and e > 0:
            tp, sl = _shaped_barriers(e, s["spot"], Kstrike, T_, mins_left,
                                      d == "CE", cw, pw)
            tp, sl = float(tp), float(sl)
        else:
            tp, sl = e * (1 + config.BASE_TP_PCT), e * (1 - config.BASE_SL_PCT)
        seg = rep.bidA[kk, t + 1:t + 1 + horizon]
        if seg.size == 0 or np.all(np.isnan(seg)):
            if funnel:
                funnel.record(idx, "no_fill", "no forward bids")
            _cf("no_fill", count_only=True)
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
        open_pos = (t + off + 1, outlay, float(pnl), d, idx)
        if on_entry is not None:
            # read-only fast-lane telemetry: the identical entry premium (ASK),
            # the forward BID path the grader just used, conviction, lot, and
            # the normal first-touch outcome (exit premium, offset, ₹). The
            # report re-runs ONLY the exit rule on this same segment.
            try:
                on_entry({"idx": idx, "t": int(t), "e": float(e),
                          "lot": int(lot), "seg": seg, "conv": float(s["conv"]),
                          "wp": float(s["wp"]), "direction": d,
                          "norm_exitp": float(exitp), "norm_off": int(off),
                          "norm_pnl": float(pnl), "outlay": float(outlay)})
            except Exception as e_:                           # noqa: BLE001
                log.warning("on_entry telemetry failed: %s — grading "
                            "unaffected", e_)
        if funnel:
            funnel.record(idx, "entered")
    if open_pos is not None:                              # EOD: realize the runner
        risk.register_exit(open_pos[1], open_pos[2], open_pos[3],
                           ts=float(open_pos[0]))
        total += open_pos[2]
        trades += 1
        wins += int(open_pos[2] > 0)
    if attribution is not None:      # grade the uniform sample, post-replay
        _cf_grade_pool()
    return float(total), {"trades": trades, "wins": wins}


def evaluate(model, vec, con, day, meta, cal, funnel=None, attribution=None):
    """SAC candidate on `day`, live-faithful (see _grade_day)."""
    def decide(obs, frame, iidx):
        o = vec.normalize_obs(obs[None]) if vec else obs[None]
        a, _ = model.predict(o, deterministic=True)
        return float(a[0][2 * iidx])

    def _acts(obs, frame):        # v9.9.1: same transform, full vector
        o = vec.normalize_obs(obs[None]) if vec else obs[None]
        a, _ = model.predict(o, deterministic=True)
        return a[0]
    return _grade_day(con, day, decide, meta, cal, funnel,
                      attribution=attribution, actions_fn=_acts)


def evaluate_heuristic(con, day, meta, cal, funnel=None,
                       attribution=None, on_entry=None):
    """Heuristic on `day`, identical grading (raw warm frame, no VecNormalize —
    that is the SAC model's input transform, not the heuristic's)."""
    pol = HeuristicPolicy()

    def decide(obs, frame, iidx):
        return float(pol.predict(frame)[2 * iidx])
    return _grade_day(con, day, decide, meta, cal, funnel,
                      attribution=attribution, on_entry=on_entry,
                      actions_fn=lambda _o, _f: pol.predict(_f))


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
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            rows.append(r)
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
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xb.shape[1])
    lr, l2 = config.META_LR, config.META_L2
    for _ in range(config.META_EPOCHS):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    raw = np.clip(w[:-1] / sd, 0.0, None)
    if raw.sum() <= 1e-9:
        log.info("trap-learner: degenerate fit — keeping fixed weights")
        return
    weights = {k: float(raw[i] / raw.sum()) for i, k in enumerate(feat_keys)}
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
# sequential task. SAC's exact actor/critic networks are kept (so the saved
# artifact still loads in the brain unchanged) but trained directly:
# precomputed rewards → critic regression → actor ascent on min(Q1,Q2) with
# SAC's entropy term, as batched GPU tensor ops. v9.1: epoch selection and
# early-stop run on an INNER training day — the promotion day is untouchable.
# ====================================================================
def _round_trip_costs_vec(buy_v, sell_v):
    """Vectorized mirror of core.execution_engine.round_trip_costs, built from
    the shared config.COSTS dict. buy_v / sell_v are ₹ premium values."""
    c = config.COSTS
    brokerage = 2.0 * c["brokerage_per_order"]            # buy + sell, flat
    txn = (buy_v + sell_v) * c["exch_txn_pct"]            # both legs on premium
    sebi = (buy_v + sell_v) * c["sebi_pct"]
    stt = sell_v * c["stt_sell_pct"]                      # sell side only
    stamp = buy_v * c["stamp_buy_pct"]                    # buy side only
    gst = (brokerage + txn) * c["gst_pct"]                # scalar's GST base —
    # NOTE it excludes SEBI charges (Zerodha's sheet includes them; ~₹0.002/
    # trip). Mirroring the scalar is the contract; fixing both is your call.
    return brokerage + txn + sebi + stt + stamp + gst


def _barrier_exit_grid(bid_g: np.ndarray, ask_g: np.ndarray,
                       tp_g: np.ndarray, sl_g: np.ndarray,
                       hor_g: np.ndarray) -> np.ndarray:
    """Vectorized first-touch triple barrier over a dense per-second grid.
    v9.1: entry is the ASK (live crosses); `hor_g` is the PER-START hold
    horizon in seconds (0-DTE-aware), so expiry-day starts stop being graded
    on 45 minutes the live guillotine never grants. Returns the realized exit
    PRICE for every start (NaN where no valid entry or no valid forward bid).
    O(max-horizon × grid)."""
    G = bid_g.shape[0]
    e = ask_g                                             # ★ ask entry
    hor = np.where(np.isfinite(hor_g), hor_g, 0.0)
    exitp = np.full(G, np.nan)
    done = np.isnan(e) | np.isnan(tp_g) | (hor <= 0)
    last_bid = np.full(G, np.nan)
    maxj = int(min(np.max(hor, initial=0.0), G - 1))
    for j in range(1, maxj + 1):
        b = np.full(G, np.nan)
        b[:G - j] = bid_g[j:]                             # b[s] = bid at sec s+j
        within = j <= hor
        valid = ~done & ~np.isnan(b) & within
        last_bid[valid] = b[valid]                        # latest bid in window
        tgt = valid & (b >= tp_g)
        stp = valid & (b <= sl_g)                         # tp>e>sl ⇒ never both
        exitp[tgt] = tp_g[tgt]
        exitp[stp] = sl_g[stp]
        done |= tgt | stp
        expire = ~done & within & (hor <= j)              # own guillotine now
        exitp[expire] = last_bid[expire]
        done |= expire
        if done.all():
            break
    nd = ~done                                            # data ends first
    exitp[nd] = last_bid[nd]
    return exitp


def _reward_table(prem: dict, ts) -> np.ndarray:
    """(N, K, 2) after-cost ₹ of the trade the LIVE brain would actually PLACE
    at each second — the model's learning target. Live constraints baked in:
      • TRADABLE indices only — the others are 0 (the brain never trades them).
      • First-affordable leg (ATM else OTM1 else no trade) at the STATIC Kelly
        budget on the MID×lot the governor prices affordability from.
      • ASK entry / bid exits / SAME shaped triple barrier / per-second 0-DTE-
        aware horizon (v9.1).
    The promotion GATE goes further (dynamic win-prob, deeper hierarchy, one
    stateful governor) — see _grade_day. [...,0]=long CE, [...,1]=long PE."""
    K, N = len(config.INDEX_ORDER), len(ts)
    R = np.zeros((N, K, 2), np.float32)
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)   # reference account —
    # the learning target must not change when the live capital knob does
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
                       np.full(G, np.nan), np.full(G, np.nan), np.zeros(G)]
                 for leg in all_legs}
        for s, row in pidx.items():
            g = s - smin
            for leg in all_legs:
                if leg in row:
                    b, a, lot, tp, sl, hold = row[leg]
                    grids[leg][0][g] = b
                    grids[leg][1][g] = a
                    grids[leg][2][g] = lot
                    grids[leg][3][g] = tp
                    grids[leg][4][g] = sl
                    grids[leg][5][g] = hold
        gpos = ts_i - smin
        inside = np.nonzero((gpos >= 0) & (gpos < G))[0]
        gp = gpos[inside]
        for d_idx, (atm_leg, otm_leg) in legs_for.items():
            pnl, mid, lot = {}, {}, {}
            for leg in (atm_leg, otm_leg):
                bid_g, ask_g, lot_g, tp_g, sl_g, hor_g = grids[leg]
                ex = _barrier_exit_grid(bid_g, ask_g, tp_g, sl_g, hor_g)  # (G,)
                e = ask_g                                 # ★ ask entry
                pnl[leg] = (ex - e) * lot_g - _round_trip_costs_vec(e * lot_g,
                                                                    ex * lot_g)
                mid[leg] = (bid_g + ask_g) / 2.0          # governor's afford basis
                lot[leg] = lot_g
            atm_cost = mid[atm_leg] * lot[atm_leg]
            otm_cost = mid[otm_leg] * lot[otm_leg]
            atm_ok = (~np.isnan(pnl[atm_leg]) & ~np.isnan(atm_cost)
                      & (lot[atm_leg] > 0) & (atm_cost <= budget))
            otm_ok = (~np.isnan(pnl[otm_leg]) & ~np.isnan(otm_cost)
                      & (lot[otm_leg] > 0) & (otm_cost <= budget))
            chosen = np.where(atm_ok, np.nan_to_num(pnl[atm_leg]),
                              np.where(otm_ok, np.nan_to_num(pnl[otm_leg]), 0.0))
            R[inside, k, d_idx] = chosen[gp].astype(np.float32)
    return R


def _bandit_reward(actions, R, gate, scale_by_mag: bool):
    """Reward of arbitrary actions, vectorized. Per index k it uses action[2k]
    and, if |a|≥gate, takes the CE reward when a>0 else the PE reward. With
    scale_by_mag=True it multiplies by |a| (the TRAIN signal); with False it
    does not (the 1-lot diagnostic). actions (B,12), R (B,K,2) → (B,)."""
    a = actions[:, 0::2]                                   # even comps per index
    mag = a.abs()
    active = (mag >= gate).float()
    pos = (a > 0).float()
    r_dir = pos * R[..., 0] + (1.0 - pos) * R[..., 1]
    w = active * mag if scale_by_mag else active
    return (w * r_dir).sum(dim=1)


def train_bandit(model, vec, obs, ts, prem, in_obs, in_ts, in_prem, log) -> dict:
    """Train SAC's actor/critic directly as a 1-step bandit. No env, no replay
    buffer, no target nets, no rollout — batched GPU updates with the true
    reward as the critic target. v9.1: checkpoint selection and early-stop run
    on the INNER day (the last TRAINING day) — never the promotion day."""
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
    IR = torch.as_tensor(_reward_table(in_prem, in_ts), device=dev) / scale
    obs_cpu = torch.as_tensor(obs, dtype=torch.float32)    # streamed CPU→GPU
    in_cpu = torch.as_tensor(in_obs, dtype=torch.float32)
    N = obs_cpu.shape[0]
    auto_ent = getattr(model, "log_ent_coef", None) is not None

    eval_rows = int(getattr(config, "FORGE_BANDIT_EVAL_ROWS", 4096))
    warmup = int(getattr(config, "FORGE_BANDIT_WARMUP_EPOCHS", 20))
    ni = in_cpu.shape[0]
    tr_idx = torch.randperm(N, generator=torch.Generator().manual_seed(0)
                            )[:min(eval_rows, N)]
    in_idx = torch.randperm(ni, generator=torch.Generator().manual_seed(1)
                            )[:min(eval_rows, ni)]

    def diag(obs_src, RR, idx):
        model.policy.set_training_mode(False)
        with torch.no_grad():
            ob = norm(obs_src[idx].to(dev))
            act = model.actor(ob, deterministic=True)
            rate = float((act[:, 0::2].abs() >= gate_ev).float().mean())
            rs = float(_bandit_reward(act, RR[idx], gate_ev, False).mean()) * scale
        return rs, rate

    log.info("bandit trainer: %d train rows × %d inner-day rows | batch %d | "
             "warmup %d ep | %s — selection on the INNER day only",
             N, ni, bs, warmup, dev)
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

            a_pi, logp = model.actor.action_log_prob(ob)
            q_pi = torch.min(*model.critic(ob, a_pi))
            alpha = (torch.exp(model.log_ent_coef.detach()) if auto_ent
                     else model.ent_coef_tensor)
            a_loss = (alpha * logp.reshape(-1, 1) - q_pi).mean()
            model.actor.optimizer.zero_grad()
            a_loss.backward()
            model.actor.optimizer.step()

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
        in_rs, in_rate = diag(in_cpu, IR, in_idx)
        log.info("  epoch %3d | steps %5d | critic %.3f actor %.3f | train "
                 "₹%.2f (%.2f%% trade) | inner ₹%.2f (%.2f%% trade)",
                 ep, gstep, c_acc / max(nb, 1), a_acc / max(nb, 1),
                 tr_rs, tr_rate * 100, in_rs, in_rate * 100)

        # SELECT best by INNER-day ₹, ties broken by train ₹.
        key = (round(in_rs, 2), tr_rs)
        if key > best_key:
            best_key = key
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.policy.state_dict().items()}

        # EARLY-STOP only AFTER warmup: the actor starts at ~zero output (sub-
        # gate ⇒ ₹0), so stopping on a flat ₹0 before it has had steps to move
        # off zero just freezes it at initialization.
        if ep >= warmup:
            if in_rs > stop_ref + 1e-6:
                stop_ref, bad = in_rs, 0
            else:
                bad += 1
                if bad >= patience:
                    log.info("  early stop: inner-day ₹ plateaued %d epochs "
                             "post-warmup", patience)
                    break

    if best_state is not None:
        model.policy.load_state_dict(best_state)           # restore best epoch
    model.policy.set_training_mode(False)
    tr_rs, tr_rate = diag(obs_cpu, R, tr_idx)
    in_rs, in_rate = diag(in_cpu, IR, in_idx)
    log.info("bandit trainer done — inner ₹%.2f | train ₹%.2f | trade-rate "
             "train %.2f%% · inner %.2f%% | %d grad steps",
             in_rs, tr_rs, tr_rate * 100, in_rate * 100, gstep)
    return {"inner_rs": in_rs, "train_rs": tr_rs,
            "train_trade_rate": tr_rate, "inner_trade_rate": in_rate,
            "grad_steps": gstep}


# ====================================================================
# STATISTICS — Bailey & López de Prado PSR / deflated PSR (stdlib NormalDist;
# no scipy). With ≤5 walk-forward folds these are HONESTY LABELS on the daily
# ₹ series (skew/kurtosis-adjusted P(true SR > 0)), not significance claims —
# the report says so explicitly. The deflated variant raises the benchmark SR
# by the expected maximum over `trials` independent candidates (every nightly
# candidate logged in forge_history is one trial against this exam family).
# ====================================================================
def _psr(rets, sr0: float = 0.0):
    n = len(rets)
    if n < 3:
        return None
    r = np.asarray(rets, float)
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return None
    sr = float(r.mean()) / sd
    z = (r - r.mean()) / sd
    g3 = float((z ** 3).mean())
    g4 = float((z ** 4).mean())
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return None
    from statistics import NormalDist
    stat = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return {"psr": float(NormalDist().cdf(stat)), "sr": sr,
            "skew": g3, "kurt": g4, "n": n}


def _deflated_psr(rets, trials: int):
    """(dsr, psr_dict). dsr = PSR against the expected-max SR of `trials`
    independent zero-skill candidates (Bailey–LdP 2014)."""
    base = _psr(rets)
    if base is None:
        return None, None
    if trials < 2:
        return base["psr"], base
    from statistics import NormalDist
    n, sr, g3, g4 = base["n"], base["sr"], base["skew"], base["kurt"]
    var_sr = max((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / max(n - 1, 1),
                 1e-12)
    nd = NormalDist()
    gamma = 0.5772156649015329
    e_max = ((1 - gamma) * nd.inv_cdf(1 - 1.0 / trials)
             + gamma * nd.inv_cdf(1 - 1.0 / (trials * math.e)))
    sr0 = math.sqrt(var_sr) * e_max
    d = _psr(rets, sr0=sr0)
    return (d["psr"] if d else None), base


# ====================================================================
# CACHES — three tiers under data/forge_cache/, each with the invalidation
# stamp its content actually depends on (2026-07-03 lesson: hash-only stamps
# would have silently served stale caches after the replay-aggregation fix):
#   • DATASET (obs/ts/prem)   ← CONFIG_HASH + _CACHE_VER. _CACHE_VER bumps
#     whenever replay/build CODE changes output for the same config (e.g. the
#     v9.1.1 intra-second flow aggregation).
#   • META SAMPLES (X/Y/W/R)  ← the above + _decision_stamp(): the hash-
#     EXCLUDED knobs that gate WHICH signals fire (persistence tempo, paper
#     bar, throttle, spread cap, eval capital…). Excluded from CONFIG_HASH on
#     purpose (no re-forge on tuning) — so they must stamp THIS cache.
#   • WF FOLD RESULTS         ← the above + the fold's train-day list. A
#     fold's train set is IMMUTABLE once its days are past, so each fold is
#     computed ONCE, ever. One SAC training draw per fold, cached — retraining
#     nightly would give a different random draw at 5× the cost, not more
#     truth. Delete data/forge_cache/wf_*.json to force redraws.
# ====================================================================
_CACHE_DIR = config.DATA_DIR / "forge_cache"
_CACHE_VER = 2          # v2: intra-second vol_delta aggregation in replay_day


def _decision_stamp() -> str:
    """Fingerprint of the hash-EXCLUDED knobs that change which signals fire
    and how attempts are judged — anything here shifts meta samples and fold
    grades without touching CONFIG_HASH, so the sample/fold caches carry it."""
    import hashlib as _h
    knobs = (config.entry_conviction_bar(), config.PAPER_EXPLORE,
             config.SIGNAL_PERSIST_ENABLED, config.SIGNAL_PERSIST_WINDOW_S,
             config.SIGNAL_PERSIST_MIN_SAMPLES,
             getattr(config, "SIGNAL_PERSIST_ER_MIN", 0.30),
             config.ENTRY_ATTEMPT_THROTTLE_S, config.MAX_ENTRY_SPREAD_PCT,
             config.META_DECISION_ENABLED, config.META_ENTRY_P_BAR,
             config.META_ENTRY_CONV_FLOOR, config.FORGE_EVAL_CAPITAL,
             config.uncalibrated_winprob(),
             # AUDIT (2026-07-28): META_CROSS_INDEX changes the WIDTH of every
             # meta sample (61 -> 64). It is hash-excluded, so without this the
             # cache would stay "valid" after flipping the flag and the forge
             # would retrain on STALE 61-dim samples while serving builds 64 —
             # the x_dim guard would then refuse every score and the brain would
             # silently fall back to the conviction bar forever. Any knob that
             # changes the feature WORLD must invalidate the sample cache.
             bool(getattr(config, "META_CROSS_INDEX", False)))
    return _h.sha1(repr(knobs).encode()).hexdigest()[:8]


def _replace_retry(src, dst, tries: int = 4) -> None:
    """os.replace with brief retries: Windows denies replacement while ANY
    process (an AV scan, a straggling reader) holds the destination open.
    The handle leak is fixed at the source (v9.9.3); this absorbs external
    transients only."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(0.25 * (k + 1))


def _build_path_config_names() -> list[str]:
    """v9.9.2: the set of config.* names the DAY-CACHE BUILD PATH actually
    reads, extracted by AST from the functions in that closure. The cache
    fingerprint hashes exactly these values — nothing else. Consequences,
    both by construction: (a) tuning any decision knob the build never
    touches (gate bars, Kelly, meta/probe/cascade/toxicity knobs, DYN
    rails …) can NEVER again invalidate 31 days of raw replay — the
    2026-07-29 evening spent ~12 h rebuilding tick arrays because a HOLD
    knob rotated the global CONFIG_HASH; (b) any constant the build DOES
    read stays load-bearing: change MAX_HOLD (baked into the prem table's
    grading windows) and the stamp rotates, correctly. Adding a new
    config reference to the build path changes the name set itself ⇒ one
    rebuild ⇒ still fail-closed."""
    import inspect
    from core import meta_gate as _MGT
    from core import session_calendar as _SC
    fns = [build_dataset, replay_day, _hold_seconds, _session_minutes_left,
           _MGT.shaped_barriers, _SC.session_minutes, _SC.session_close_hm,
           _SC.cas_window, _SC.in_cas_blackout, _SC.cas_phase,
           _SC.index_price_quality, _SC.in_post_auction,
           _SC.entries_allowed]
    names: set[str] = set()
    for fn in fns:
        try:
            tree = ast.parse(inspect.getsource(fn))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "config"):
                names.add(node.attr)
            if (isinstance(node, ast.Call) and len(node.args) >= 2
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "config"
                    and isinstance(node.args[1], ast.Constant)):
                names.add(str(node.args[1].value))
    return sorted(names)


def _data_stamp() -> str:
    import hashlib
    parts = []
    for n in _build_path_config_names():
        v = getattr(config, n, None)
        if callable(v):
            continue
        parts.append(f"{n}={v!r}")
    fp = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return f"{fp}:v{_CACHE_VER}"


def _cache_paths(day: str):
    return (_CACHE_DIR / f"{day}.npz", _CACHE_DIR / f"{day}.prem.pkl",
            _CACHE_DIR / f"{day}.meta.json", _CACHE_DIR / f"{day}.empty")


def _build_and_cache(dbpath: str, day: str) -> tuple[str, bool, float]:
    """Top-level (Windows-spawn-safe) worker: replay one day, persist to cache.
    Own sqlite connection; numpy-only (no torch in workers)."""
    t0 = time.time()
    npz, pkl, metaf, emptyf = _cache_paths(day)
    _CACHE_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(dbpath)
    try:
        # v9.9.19: one bad day must not abort a 37-day rebuild hours in.
        # Report the failure, return the empty marker map_days already
        # understands, and let every remaining day continue.
        try:
            o, t, p = build_dataset(con, day)
        except Exception as _e:                            # noqa: BLE001
            log.error("  %s: build failed (%s) — skipped; the remaining "
                      "days continue", day, _e)
            return day, False, 0.0
    finally:
        con.close()
    if o is None:
        emptyf.write_text(_data_stamp())
        return day, False, time.time() - t0
    # v9.9.2: atomic publication — tmp → os.replace per artifact, stamp file
    # LAST (it is the freshness gate), so no reader or racing builder can
    # observe a torn cache.
    # BUGFIX 2026-07-30: np.savez APPENDS ".npz" unless the path already
    # ends in it, so a ".npz.tmp" temp silently became ".npz.tmp.npz" and
    # every os.replace raised WinError/FileNotFoundError — 32/32 days
    # "skipped" on the first parallel prime. Temp names now END in .npz
    # (savez leaves them alone) and carry the pid so two concurrent runs
    # can never collide. try/finally guarantees no debris is left behind.
    _tn = npz.with_name(f"{npz.stem}.{os.getpid()}.tmp.npz")
    _tp = pkl.with_name(f"{pkl.stem}.{os.getpid()}.tmp.pkl")
    _tm = metaf.with_name(f"{metaf.stem}.{os.getpid()}.tmp.json")
    try:
        np.savez(_tn, obs=o.astype(np.float32), ts=t)
        _replace_retry(_tn, npz)
        with open(_tp, "wb") as f:
            pickle.dump(p, f)
        _replace_retry(_tp, pkl)
        _tm.write_text(json.dumps({"stamp": _data_stamp(),
                                   "rows": int(len(o)), "ts": time.time()}))
        _replace_retry(_tm, metaf)
    finally:
        for _junk in (_tn, _tp, _tm):
            try:
                _junk.unlink(missing_ok=True)
            except OSError:
                pass
    return day, True, time.time() - t0


def _cache_fresh(day: str) -> bool:
    npz, pkl, metaf, emptyf = _cache_paths(day)
    if emptyf.exists():
        return emptyf.read_text().strip() == _data_stamp()
    if not (npz.exists() and pkl.exists() and metaf.exists()):
        return False
    try:
        return json.loads(metaf.read_text()).get("stamp") == _data_stamp()
    except Exception:                                     # noqa: BLE001
        return False


def _load_cached(day: str):
    npz, pkl, metaf, emptyf = _cache_paths(day)
    if emptyf.exists():
        return None
    # v9.9.3: np.load keeps its handle OPEN until closed; on Windows any open
    # handle on a cache file makes a later os.replace fail with WinError 5.
    # On 2026-08-01 handles held by this module denied all 32 parallel
    # sample-cache publishes and the vault was replayed THREE times (prime,
    # train, verdict — ~12 h of redundant compute). Materialise, close.
    with np.load(npz) as z:
        obs, ts = z["obs"], z["ts"]
    with open(pkl, "rb") as f:
        p = pickle.load(f)
    return obs, ts, p


# ---- META-SAMPLE cache: one live-path replay per day, EVER (was: every
#      pool day re-replayed nightly — 41 min and +3.5 min/day, forever) ------
def _meta_cache_path(day: str):
    if _HOLD_OVERRIDE_S:
        return _CACHE_DIR / (f"{day}.meta_samples.h"
                             f"{int(_HOLD_OVERRIDE_S)}.npz")
    return _CACHE_DIR / f"{day}.meta_samples.npz"


def _meta_samples_stamp() -> str:
    h = f"|h{int(_HOLD_OVERRIDE_S)}" if _HOLD_OVERRIDE_S else ""
    return (f"{_data_stamp()}:{_decision_stamp()}") + "|ret1" + h


def _meta_cache_fresh(day: str) -> bool:
    p = _meta_cache_path(day)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return (str(z["stamp"]) == _meta_samples_stamp()
                    and "E" in z.files)
    except Exception:                                     # noqa: BLE001
        return False


def _prime_meta_samples_worker(day: str) -> tuple[str, int]:
    """Windows-spawn-safe: own sqlite, returns a count, never the arrays."""
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        x, *_ = _gen_meta_samples_cached(con, day)
        return day, len(x)
    finally:
        con.close()


def _gen_meta_samples_cached(con, day: str):
    p = _meta_cache_path(day)
    if p.exists():
        try:
            with np.load(p, allow_pickle=False) as z:
                if (str(z["stamp"]) == _meta_samples_stamp()
                        and "E" in z.files):
                    return (list(z["X"]), list(z["Y"]), list(z["W"]),
                            list(z["R"]), list(z["RET"]),
                            [tuple(e) for e in z["E"]])
        except Exception:                                 # noqa: BLE001
            pass                                          # torn/old ⇒ rebuild
    X, Y, W, R, RET, ECON = _gen_meta_samples(con, day)
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        dim = 3 * config.FEATURES_PER_NODE + 4
        _ts_p = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npz")   # see
        #        BUGFIX 2026-07-30 in _build_and_cache: must end in .npz
        np.savez(_ts_p, stamp=np.str_(_meta_samples_stamp()),
                 X=(np.stack(X) if X else np.zeros((0, dim), np.float32)),
                 Y=np.asarray(Y, np.float32),
                 W=np.asarray(W, np.float32),
                 R=(np.stack(R) if R else
                    np.zeros((0, config.FEATURES_PER_NODE), np.float32)),
                 RET=np.asarray(RET, np.float32),
                 E=(np.asarray(ECON, np.float32) if ECON
                    else np.zeros((0, 4), np.float32)))
        _replace_retry(_ts_p, p)
    except Exception as e:                                # noqa: BLE001
        # a cache write is an optimisation, never a correctness
        # dependency — warn, drop any debris, return the real arrays.
        log.warning("meta-sample cache write failed for %s: %s", day, e)
        try:
            _ts_p.unlink(missing_ok=True)
        except (OSError, NameError):
            pass
    return X, Y, W, R, RET, ECON


def _prepare_cache(days: list[str], report: DailyReport):
    todo = [d for d in days if not _cache_fresh(d)]
    stale = [d for d in days if d not in todo]
    log.info("dataset cache: %d fresh, %d to build %s", len(stale), len(todo),
             todo or "")
    if not todo:
        return
    nw = int(config.FORGE_PARALLEL_WORKERS)
    if nw == 0:
        # single source of truth with the harnesses (core/parallel_days), so
        # one knob governs every day-parallel path instead of two disagreeing
        # caps (this was min(4, cpu//2) while the harnesses used min(6, ...)).
        try:
            from core.parallel_days import default_workers as _dw
            nw = _dw()
        except Exception:                                 # noqa: BLE001
            nw = max(1, min(6, (os.cpu_count() or 2) // 2))
    built = {}
    if nw > 1 and len(todo) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(nw, len(todo))) as ex:
                for day, ok, secs in ex.map(_build_and_cache,
                                            [str(config.DB_PATH)] * len(todo),
                                            todo):
                    built[day] = round(secs, 1)
                    log.info("  built %s in %.0fs (%s)", day, secs,
                             "ok" if ok else "empty")
        except Exception as e:                            # noqa: BLE001
            log.warning("PARALLEL BUILD FAILED (%s) — falling back to SERIAL. "
                        "At ~684s/day this turns a %d-day rebuild into ~%.1f "
                        "hours. Worth fixing rather than waiting.", e,
                        len(todo), 684.0 * len(todo) / 3600.0)
            built = {}
    if not built:                                         # serial (or fallback)
        for day in todo:
            day, ok, secs = _build_and_cache(str(config.DB_PATH), day)
            built[day] = round(secs, 1)
            log.info("  built %s in %.0fs (%s)", day, secs,
                     "ok" if ok else "empty")
    report.d.setdefault("cache", {})["built_s"] = built


def _score_incumbent_on(con, day, meta, cal, log):
    """Re-score the currently-PROMOTED model on the SAME promotion day the
    candidate is graded on (apples-to-apples). Returns ₹ or None."""
    if not (config.MODEL_MANIFEST.exists() and _load_rl()):
        return None
    try:
        man = json.loads(config.MODEL_MANIFEST.read_text())
        mp = config.MODEL_DIR / man.get("model", "")
        npth = config.MODEL_DIR / man.get("norm", "")
        if not (mp.exists() and npth.exists()):
            return None

        class _Dummy(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               (config.OBS_DIM,), np.float32)
            action_space = gym.spaces.Box(-1, 1, (config.ACTION_DIM,),
                                          np.float32)
            def reset(self, *, seed=None, options=None):
                return np.zeros(config.OBS_DIM, np.float32), {}
            def step(self, a):
                return np.zeros(config.OBS_DIM, np.float32), 0, True, False, {}

        inc_vec = VecNormalize.load(str(npth), DummyVecEnv([_Dummy]))
        inc_vec.training = False
        inc_model = SAC.load(str(mp), device="cpu")   # tiny eval, no GPU fight
        rs, _ = evaluate(inc_model, inc_vec, con, day, meta, cal)
        return float(rs)
    except Exception as e:                                # noqa: BLE001
        log.warning("incumbent re-score failed (%s) — using its stored "
                    "val_score for the bar instead", e)
        return None


def main():
    t_start = time.time()
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if len(days) < 2:
        raise SystemExit("Need ≥2 harvested days (train + promotion day).")
    final_day = days[-1]
    pool = days[:-1]
    cap = int(getattr(config, "FORGE_MAX_TRAIN_DAYS", 0))
    if cap > 0:
        pool = pool[-cap:]
    report = DailyReport("forge")
    report.d["days"] = {"all": days, "pool": pool, "promotion_day": final_day}
    log.info("pool %s | promotion day %s (untouched by any fit/selection)",
             pool, final_day)

    # 1) META-LABELER — numpy-only, POOL DAYS ONLY (the promotion day never
    #    feeds any fit), shared decision path, uniqueness weights, day holdout.
    meta_diag = {}
    try:
        _, meta_diag = train_meta(con, pool)
    except Exception as e:                                # noqa: BLE001
        log.error("meta-labeler failed: %s", e)
        meta_diag = {"error": str(e)}
    report.d["meta"] = meta_diag

    # 1b) TRAP LEARNER — refit shield weights+threshold from real stop-outs.
    try:
        train_trap_model()
    except Exception as e:                                # noqa: BLE001
        log.error("trap-learner failed: %s", e)

    # 1d) REGIME CLASSIFIER — refit cut points to this market's percentiles.
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
    if config.FORGE_TRAIN_SAC and not _load_rl():
        log.warning("torch / stable-baselines3 / gymnasium not installed — "
                    "RL forge skipped (meta-labeler above still ran). "
                    "On the RTX 4060: pip install -r requirements.txt")
        report.d["rl"] = "skipped: no torch stack"
        report.write()
        return

    _prepare_cache(pool, report)   # exams replay the vault directly; only
    #                                training (fit/inner/WF) reads the cache

    def make_and_train(fit_days: list[str], inner_day: str, tag: str):
        """Fresh SAC pair trained on `fit_days`, selected on `inner_day`."""
        blobs = [b for b in (_load_cached(d) for d in fit_days)
                 if b is not None]
        inner = _load_cached(inner_day)
        if not blobs or inner is None:
            log.warning("%s: no replayable rows (fit %s / inner %s)", tag,
                        [d for d in fit_days], inner_day)
            return None, None, None
        obs = np.concatenate([b[0] for b in blobs])
        ts = np.concatenate([b[1] for b in blobs])
        prem = {i: {} for i in config.INDEX_ORDER}
        for _, _, p in blobs:
            for i in config.INDEX_ORDER:
                prem[i].update(p[i])
        del blobs                      # MEMORY FIX: the per-day arrays were
        #                                double-held alongside the concat copy
        io, it, ip = inner
        log.info("%s: %d train rows (%s) | inner %s (%d rows)", tag, len(obs),
                 ",".join(fit_days), inner_day, len(io))
        assert _load_rl(), "FORGE_TRAIN_SAC=True but RL stack absent"
        env = DummyVecEnv([lambda: ForgeEnv(obs, ts, prem)])
        vec = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
        vec.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
        vec.obs_rms.count = float(len(obs))
        vec.training = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAC("MlpPolicy", vec, device=device, buffer_size=2048,
                    batch_size=config.SAC_BATCH,
                    policy_kwargs={"features_extractor_class": Extractor,
                                   "net_arch": [256, 256]}, verbose=0)
        d = train_bandit(model, vec, obs, ts, prem, io, it, ip, log)
        # MEMORY FIX (the 2026-07-03 triple-OOM): ForgeEnv pinned the FULL
        # training array (~5 GB at 11 days) inside `vec` for the rest of the
        # run — evaluation only ever touches vec.obs_rms, never the env. Null
        # the env's references so the arrays free the moment locals drop.
        _e = vec.venv.envs[0]
        _e.obs = _e.ts = _e.prem = None
        return model, vec, d

    # ---- deployment CANDIDATE: fit on pool[:-1], select on pool[-1] ---------
    if len(pool) >= 2:
        fit_days, inner_day = pool[:-1], pool[-1]
    else:
        fit_days, inner_day = pool, pool[-1]
        log.warning("only one pool day — inner selection runs ON the train "
                    "day (unavoidable until a 3rd day is harvested)")
    if config.FORGE_TRAIN_SAC:
        model, vec, diag = make_and_train(fit_days, inner_day, "candidate")
    else:
        model = vec = None
        diag = {"frozen": True}
        log.info("SAC FROZEN (FORGE_TRAIN_SAC=False) — the directional "
                 "gravestone stands: no candidate tonight; meta retrain, "
                 "heuristic+meta exam, counterfactual, drift, regime and "
                 "caches all run in full.")
    if config.FORGE_TRAIN_SAC and model is None:
        raise SystemExit("No replayable seconds — check harvester output.")
    report.d["bandit"] = {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in diag.items()}
    report.d["bandit"]["fit_days"] = fit_days
    report.d["bandit"]["inner_day"] = inner_day

    # ---- PROMOTION-DAY EXAM (deployed mode: fresh meta + calibration) -------
    meta, cal = _eval_meta(), _eval_cal()
    fun_sac = D.GateFunnel(config.TRADABLE)
    fun_heur = D.GateFunnel(config.TRADABLE)
    score, st_s = (
        evaluate(model, vec, con, final_day, meta, cal, fun_sac)
        if model is not None else (0.0, {"trades": 0, "wins": 0}))
    gate_attr: dict = {}
    heur, st_h = evaluate_heuristic(con, final_day, meta, cal, fun_heur,
                                    attribution=gate_attr)
    if gate_attr:
        log.info("counterfactual gate attribution (promotion day — the ₹ each"
                 " rule refused; graded ≤%d/gate):", config.CF_MAX_PER_GATE)
        for g_, a_ in sorted(gate_attr.items()):
            log.info("  %-14s n=%-6d graded=%-4d Σ₹%+11.2f mean ₹%+9.2f "
                     "wins %d%s", g_, a_["n"], a_["graded"], a_["sum"],
                     a_["sum"] / a_["graded"] if a_["graded"] else 0.0,
                     a_["wins"],
                     f" (capped +{a_['capped']})" if a_["capped"] else "")
        try:
            with (config.STATE_DIR / "gate_attribution.jsonl").open(
                    "a", encoding="utf-8") as f_:
                f_.write(json.dumps({"date": final_day,
                                     "attribution": gate_attr}) + "\n")
        except Exception:                                 # noqa: BLE001
            pass
    heur_boot, st_b = evaluate_heuristic(con, final_day, None, {}, None)
    log.info("promotion day (%s) after-cost — SAC ₹%.2f (%d trades) | "
             "heuristic+meta ₹%.2f (%d) | heuristic-bootstrap ₹%.2f (%d)",
             final_day, score, st_s["trades"], heur, st_h["trades"],
             heur_boot, st_b["trades"])
    for i in config.TRADABLE:
        log.info("  exam funnel SAC  %s", fun_sac.line(i))
        log.info("  exam funnel HEUR %s", fun_heur.line(i))

    inc_today = _score_incumbent_on(con, final_day, meta, cal, log)
    inc_stored = -1e18
    if config.MODEL_MANIFEST.exists():
        inc_stored = float(json.loads(config.MODEL_MANIFEST.read_text()).get(
            "val_score", -1e18))
    if inc_today is not None:
        incumbent, inc_src = inc_today, "re-scored on this promotion day"
    elif inc_stored > -1e17:
        incumbent, inc_src = inc_stored, "stored val_score (could not reload)"
    else:
        incumbent, inc_src = -1e18, "none yet"
    has_champ = incumbent > -1e17
    champ_str = f"₹{incumbent:+,.2f} ({inc_src})" if has_champ else "none yet"
    if has_champ:
        log.info("incumbent on %s: ₹%.2f (%s) | candidate ₹%.2f",
                 final_day, incumbent, inc_src, score)

    # ---- SAVE + FREE the candidate BEFORE walk-forward (memory fix part 2:
    # the candidate's tensors have no business coexisting with fold training;
    # the promotion gate below only needs its SCORE and the saved paths) ------
    trained = model is not None          # survives the del below (v9.6.4)
    ver = time.strftime("v91_%Y%m%d_%H%M%S")
    mpath = config.MODEL_DIR / f"apex_sac_{ver}.zip"
    npath = config.MODEL_DIR / f"apex_norm_{ver}.pkl"
    if trained:
        model.save(mpath); vec.save(str(npath))
        del model, vec                    # memory fix: free before WF folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- WALK-FORWARD (bootstrap path both policies; promotion day excluded).
    # Each fold trains a FRESH candidate only on strictly-earlier days, inner-
    # selected on its own last training day. Meta is None in folds — the pool-
    # trained meta has seen every fold day, so using it here would leak.
    # v9.1.1: a fold's train set is IMMUTABLE, so each fold is computed ONCE
    # and cached (one training draw per fold — see the cache-block rationale).
    def _wf_cache_path(fd: str):
        return _CACHE_DIR / f"wf_{fd}.json"

    def _wf_stamp(train_days: list[str]) -> str:
        import hashlib as _h
        return (f"{_meta_samples_stamp()}:"
                f"{_h.sha1(repr(train_days).encode()).hexdigest()[:8]}")

    # v9.7.1 crash fix (flow-sensitive del/unbound class — the same family
    # /tmp/del_sweep.py was built to catch): wf_hits / wf_new were initialized
    # ONLY inside `if trained:`, but the cache-report dict below reads them
    # UNCONDITIONALLY. With FORGE_TRAIN_SAC=False the SAC candidate is None →
    # trained=False → the else branch set wf_rows/sac_wf/heur_wf but NOT the
    # two counters, so `report.d["cache"]` raised UnboundLocalError and took
    # the whole nightly down AFTER all real work (meta, exams, counterfactual)
    # had already succeeded. Bind every name the merge point consumes here,
    # before the split, so neither branch can leave a hole.
    wf_rows: list = []
    wf_hits = wf_new = 0
    sac_wf: list = []
    heur_wf: list = []
    wf_funnel = D.GateFunnel(config.TRADABLE)   # AUDIT: MUST bind before the
    # `if trained:` split — line 2211 reads it unconditionally at the merge
    # point. The 2026-07-19 hoist moved the other four names but left this one
    # inside the guard, so FORGE_TRAIN_SAC=False (trained=False) still raised
    # UnboundLocalError here. Now genuinely bound on every path.
    if trained:
        K = min(int(config.FORGE_WF_FOLDS), max(len(pool) - 2, 0))
        for fd in pool[len(pool) - K:] if K > 0 else []:
            j = pool.index(fd)
            tr = pool[:j]
            cpath = _wf_cache_path(fd)
            if cpath.exists():
                try:
                    row = json.loads(cpath.read_text())
                    if row.get("stamp") == _wf_stamp(tr):
                        row["cached"] = True
                        wf_rows.append(row)
                        wf_hits += 1
                        log.info("  WF %s | cached | SAC ₹%+.2f (%d tr) | heur "
                                 "₹%+.2f (%d tr)", fd, row["sac"],
                                 row["sac_trades"], row["heuristic"],
                                 row["heur_trades"])
                        continue
                except Exception:                             # noqa: BLE001
                    pass                                      # torn/old ⇒ recompute
            wf_model, wf_vec, _wd = make_and_train(tr[:-1], tr[-1], f"wf:{fd}")
            if wf_model is None:
                continue
            s_rs, s_st = evaluate(wf_model, wf_vec, con, fd, None, {}, wf_funnel)
            h_rs, h_st = evaluate_heuristic(con, fd, None, {})
            row = {"day": fd, "train_days": len(tr), "stamp": _wf_stamp(tr),
                   "sac": round(s_rs, 2), "heuristic": round(h_rs, 2),
                   "sac_trades": s_st["trades"], "heur_trades": h_st["trades"],
                   "eval_capital": config.FORGE_EVAL_CAPITAL,
                   "ts": time.time()}
            try:
                cpath.write_text(json.dumps(row))
            except Exception as e:                            # noqa: BLE001
                log.warning("WF fold cache write failed for %s: %s", fd, e)
            wf_rows.append(row)
            wf_new += 1
            log.info("  WF %s | %d train d | SAC ₹%+.2f (%d tr) | heur ₹%+.2f "
                     "(%d tr)", fd, len(tr), s_rs, s_st["trades"], h_rs,
                     h_st["trades"])
            del wf_model, wf_vec
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if K == 0:
            log.info("walk-forward skipped: needs ≥4 harvested days "
                     "(have %d) — grows automatically", len(days))

        sac_wf = [r["sac"] for r in wf_rows]
        heur_wf = [r["heuristic"] for r in wf_rows]
        # v9.4: deflation charges the GLOBAL trial registry (Pillar 1) — the
        # pre-registry forge_history era is back-filled once, idempotently.
        TR.ensure_forge_backfill()
    # else: SAC frozen / untrained — wf_rows, wf_hits, wf_new, sac_wf, heur_wf
    # keep their pre-bound empty/zero values from above (no re-assignment
    # needed; the merge point below is now hole-free on both branches).
    if trained:
        TR.register("forge", ver, "candidate", day=final_day)

    hist_path = config.MODEL_DIR / "forge_history.jsonl"
    trials = TR.trials_for_deflation("forge")
    dsr, psr = _deflated_psr(sac_wf, trials) if len(sac_wf) >= 3 else (None, None)
    if psr:
        log.info("walk-forward SAC: sum ₹%.2f over %d folds | PSR(SR>0)=%.2f "
                 "| deflated (%d trials)=%s — %d folds is an honesty label, "
                 "not significance", sum(sac_wf), len(sac_wf), psr["psr"],
                 trials, f"{dsr:.2f}" if dsr is not None else "—", len(sac_wf))

    # ---- PROMOTION GATE (candidate already saved + freed before WF) ---------
    report.d["eval_capital"] = config.FORGE_EVAL_CAPITAL
    report.d.setdefault("cache", {}).update(
        {"wf_folds_cached": wf_hits, "wf_folds_computed": wf_new,
         "cache_ver": _CACHE_VER, "decision_stamp": _decision_stamp()})
    margin = float(config.FORGE_PROMOTE_MARGIN_RS)
    baseline = max(heur, incumbent if has_champ else heur)
    min_rate = float(config.FORGE_MIN_TRADE_RATE)
    # v9.7.1 crash fix (same flow-sensitive family as wf_hits): when SAC is
    # FROZEN (FORGE_TRAIN_SAC=False), the else-branch above sets diag =
    # {"frozen": True} with NO "train_trade_rate" key, but this line read it
    # unconditionally → KeyError that took the whole nightly down AFTER meta
    # retrain / exams / counterfactual / drift had all succeeded. There is no
    # SAC candidate when frozen, so the abstainer test is vacuous: default the
    # rate high (well above min_rate) so `abstains` is False and promotion is
    # governed by the real gates below (beats_final / wf_ok / suite).
    _train_rate = diag.get("train_trade_rate")
    abstains = (_train_rate is not None) and (_train_rate < min_rate)
    wf_ok = (not wf_rows) or (sum(sac_wf) >= sum(heur_wf))
    beats_final = score > baseline + margin
    promote = beats_final and wf_ok and not abstains
    # v9.9.3: on 2026-08-01 a promotion fired with FORGE_TRAIN_SAC=False and
    # ZERO candidate trades — "did nothing" out-scored a losing baseline and
    # a ghost champion landed in the manifest. A candidate must exist and
    # must have actually traded on the promotion day.
    _min_tr = int(getattr(config, "PROMOTION_MIN_TRADES", 5))
    _no_cand = not bool(getattr(config, "FORGE_TRAIN_SAC", False))
    _thin = int(st_s.get("trades", 0)) < _min_tr
    promote = promote and not _no_cand and not _thin

    g = {}
    gate = config.STATE_DIR / "sim_gate.json"
    suite_red = False
    if gate.exists():
        g = json.loads(gate.read_text())
        suite_red = (g.get("pass") != g.get("total") and
                     time.time() - g.get("ts", 0) < 36 * 3600)

    reasons = []
    if _no_cand:
        reasons.append("SAC frozen (FORGE_TRAIN_SAC=False) — no candidate "
                       "exists; nothing to promote")
    if _thin:
        reasons.append(f"candidate traded {int(st_s.get('trades', 0))} < "
                       f"{_min_tr} on promotion day — a do-nothing policy "
                       f"cannot be promoted")
    if abstains:
        reasons.append(f"abstainer: train trade-rate "
                       f"{diag['train_trade_rate']:.4%} < {min_rate:.4%}")
    if not beats_final:
        reasons.append(f"promotion-day ₹{score:.2f} ≤ bar ₹{baseline + margin:.2f} "
                       f"(heur+meta ₹{heur:.2f}, champion {champ_str}, "
                       f"margin ₹{margin:.0f})")
    if wf_rows and not wf_ok:
        reasons.append(f"walk-forward: SAC ₹{sum(sac_wf):.2f} < heuristic "
                       f"₹{sum(heur_wf):.2f} over {len(wf_rows)} folds")
    if suite_red:
        reasons.append(f"regression suite RED ({g.get('pass')}/{g.get('total')})")

    with hist_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ver": ver, "day": final_day, "model_score": round(score, 2),
            "heuristic": round(heur, 2), "heuristic_boot": round(heur_boot, 2),
            "incumbent": round(incumbent, 2) if has_champ else None,
            "inner_rs": (round(diag["inner_rs"], 2)
                         if diag.get("inner_rs") is not None else None),
            "wf_sac": sac_wf, "wf_heur": heur_wf,
            "psr": round(psr["psr"], 3) if psr else None,
            "dsr": round(dsr, 3) if dsr is not None else None,
            "meta_n": meta_diag.get("n"),
            "promoted": bool(promote and not suite_red)}) + "\n")

    report.d["promotion_day"] = {
        "day": final_day, "sac": round(score, 2),
        "sac_trades": st_s["trades"], "sac_wins": st_s["wins"],
        "heuristic_meta": round(heur, 2), "heur_trades": st_h["trades"],
        "heuristic_bootstrap": round(heur_boot, 2),
        "incumbent": round(incumbent, 2) if has_champ else None,
        "incumbent_src": inc_src,
        "funnel_sac": fun_sac.as_dict(),
        "funnel_heuristic": fun_heur.as_dict(),
        "gate_attribution": gate_attr}
    report.d["walk_forward"] = {
        "folds": wf_rows, "sac_sum": round(sum(sac_wf), 2) if sac_wf else None,
        "heur_sum": round(sum(heur_wf), 2) if heur_wf else None,
        "psr": psr, "dsr": dsr, "trials_for_deflation": trials,
        "funnel": wf_funnel.as_dict(),
        "note": "bootstrap path both policies; ≤5 folds ⇒ honesty label, "
                "not significance"}
    report.d["promotion"] = {"promoted": bool(promote and not suite_red),
                             "candidate": ver, "margin_rs": margin,
                             "baseline": round(baseline, 2),
                             "blocked_by": reasons or None}
    report.d["runtime_s"] = round(time.time() - t_start, 1)
    report.write()
    log.info("forge report → %s", report.path)

    if not (promote and not suite_red):
        log.warning("NOT promoted (%s). Live policy stays; candidate saved %s "
                    "+ logged to forge_history.jsonl.",
                    " | ".join(reasons), ver)
        return

    tmp = config.MODEL_MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": ver, "model": mpath.name,
                               "norm": npath.name, "obs_dim": config.OBS_DIM,
                               "val_score": score, "val_day": final_day,
                               "ts": time.time()}))
    tmp.replace(config.MODEL_MANIFEST)                  # atomic pair promotion ★
    log.info("PROMOTED %s — ₹%.2f clears bar ₹%.2f (heur+meta ₹%.2f, prev "
             "champion %s) AND walk-forward ₹%.2f ≥ heuristic ₹%.2f",
             ver, score, baseline + margin, heur, champ_str,
             sum(sac_wf) if sac_wf else 0.0, sum(heur_wf) if heur_wf else 0.0)


if __name__ == "__main__":
    config.setup_logging("forge")
    main()