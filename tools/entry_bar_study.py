"""
ENTRY BAR STUDY — where should the conviction bar sit, and is that answerable?
==============================================================================
    python tools/entry_bar_study.py [--days N] [--json out.json] [--dry-run]

The question this exists to answer, from the 2026-08-07 session: NIFTY logged
12 311 conviction observations, the bar is 0.55, conviction p90 is 0.529, and
three trades were taken all day. The obvious reading is "the bar is too high."

The obvious reading is wrong, and the first thing this tool prints is why.

    MAX_CONCURRENT_POSITIONS = 1 (across ALL indices)
    MAX_HOLD_MINUTES = 60  →  3600s guillotine
    COOLDOWN_S = 180
    entry window 09:15→15:05 = 21 000s
    ⇒ one slot costs 3 780s ⇒ AT MOST 5 TRADES PER SESSION

The bar does not control how many trades happen. It controls WHICH of ~5
slots get filled, and how early. Dropping the bar to 0.20 raises the number
of eligible candidates by ~100× and the number of trades by approximately
zero — it just fills the same slots sooner, with weaker signals. That makes
this an optimal-stopping problem (take this 0.56, or hold the slot for an
0.80 that may not come?), not a threshold-calibration problem.

HOW THE SWEEP IS RUN
--------------------
core.entry_counterfactual replays each session K times in ONE pass, once per
pre-registered bar in config.ENTRY_BAR_GRID, each with its own book under the
real constraints — one position, throttle, cooldown, curfew, affordability
against TRADING_CAPITAL — and ONE shared exit rule, so a bar difference can
never be confounded with an exit difference. Because every bar sees the
identical signal stream in the identical order, the comparison is exactly
paired at the day level and each book's trades are non-overlapping by
construction. That is what earns the day-clustered statistics; it is not
assumed.

Two reference policies run alongside and are printed BEFORE any bar result:
ORACLE_TOPK (perfect-hindsight slot filling — the ceiling) and RANDOM_SLOT
(fill slots at random — the floor). If the incumbent does not beat RANDOM,
the conviction score is not selecting and a bar change is premature. If the
incumbent is within 10% of ORACLE, there is no headroom in the bar at all
and the binding constraint is elsewhere.

WHY THE STATISTICS ARE NOT BH
-----------------------------
Fourteen bars form a nested, strongly correlated ladder and the reported
quantity is a MAXIMUM over it. The maximum of correlated statistics has its
own null. core.entry_bar_store runs a Westfall–Young max-statistic
permutation — whole DAYS sign-flipped, bar axis intact — which calibrates at
6.0% against a nominal 5% where a naive t-test on the best bar runs at 16.8%.
Marginal and BH columns are printed as description. They are not the
criterion.

Promotion is OFF by default (ENTRY_BAR_PROMOTE_ENABLED=False). The bar has
the shortest path to capital of any knob in this system; it moves by hand,
after someone reads the evidence, or it does not move.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("entry_bar_study")
import logging                                             # noqa: E402
log = logging.getLogger("entry_bar_study")

from core import entry_bar_store as EBS                    # noqa: E402
from core.entry_counterfactual import (BarSweep, Signal,    # noqa: E402
                                       bar_grid, capacity_note,
                                       ORACLE, RANDOM)


def _register(bars) -> None:
    try:
        from core.trial_registry import register
        for b in bars:
            register(family="entry_bar", spec_id=f"bar_{b:.2f}",
                     kind="pre_registered", n_trades=0)
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)


def make_adapters(rep):
    """Bind BarSweep's two closures to a nightly_forge Replayer.

    The replayer already holds the dense per-second arrays (rep.bidA,
    rep.askA, rep.ti) and the as-of chain walk (rep.mapper.hierarchy). Using
    them directly means the sweep prices its counterfactual entries off the
    IDENTICAL tape the grader prices real ones off — no second data path to
    drift.
    """
    def quote_fn(token, t):
        k = rep.ti.get(int(token))
        if k is None or t >= rep.bidA.shape[1] or t < 0:
            return (0.0, 0.0, False)
        b, a = rep.bidA[k, t], rep.askA[k, t]
        stale = (t - rep.last_tick.get(int(token), -10 ** 9)) > int(
            getattr(config, "SHADOW_MAX_STALE_S", 120))
        ok = bool(np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0
                  and not stale)
        return (float(b) if ok else 0.0, float(a) if ok else 0.0, ok)

    def chain_fn(index, spot, direction, t):
        try:
            rows = rep.mapper.hierarchy(index, spot, direction)
        except Exception:                                  # noqa: BLE001
            return []
        out = []
        for r in rows:
            b, a, fresh = quote_fn(r["token"], t)
            out.append({"token": int(r["token"]),
                        "symbol": r.get("symbol") or str(r["token"]),
                        "lot": int(r.get("lot") or 0),
                        "strike": float(r.get("strike") or 0),
                        "bid": b, "ask": a, "fresh": fresh})
        return out

    return chain_fn, quote_fn


def _sweep_day_worker(args):
    """ONE session, every bar, in its own process. Module-level and
    picklable by design.

    v9.9.16: this was a serial loop in main(). Measured on the 2026-08-09
    run it cost ~15 MINUTES PER DAY — 9.8 hours across 38 sessions, the
    single most expensive step in the whole evening chain, larger than
    cascade and shortvol combined.

    Profiling showed the BarSweep itself is NOT the cost: 3.4s per day for
    3 indices x 14 bars (336k quote marks, 320k policy steps). The 15
    minutes is the uncached _Replayer pass — `decide` and `actions_fn` each
    run HeuristicPolicy().predict per second per index, and unlike the
    forge's own path this replay is not served from DC. So the fix is not
    to micro-optimise the sweep; it is to stop running 38 independent
    replays down a single core while nineteen threads idle — exactly the
    defect this tool's own harness siblings had.

    SQLite handles and the policy object cannot cross a process boundary,
    so the worker builds its own, the same way _cascade_day_worker does.
    """
    day, bars, N = args
    bars = list(bars)          # back to a list for the sweep
    import sqlite3 as _sq
    from nightly_forge_v9 import _Replayer, _eval_meta, _eval_cal
    from core.heuristic_policy import HeuristicPolicy

    con = _sq.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        meta, cal = _eval_meta(), _eval_cal()
        _pol = HeuristicPolicy()

        def decide(obs, frame, iidx):
            return float(_pol.predict(frame)[2 * iidx])

        def actions_fn(_o, _f):
            return _pol.predict(_f)

        return sweep_day(con, day, decide, meta, cal, bars,
                         actions_fn=actions_fn)
    except Exception as e:                                 # noqa: BLE001
        log.warning("  %s: sweep failed (%s)", day, e)
        return None
    finally:
        try:
            con.close()
        except Exception:                                  # noqa: BLE001
            pass


def sweep_day(con, day, decide, meta, cal, bars, actions_fn=None) -> dict:
    """One session, every bar, one pass. Returns {policy: realised ₹}."""
    from nightly_forge_v9 import _Replayer
    from simulation.scenario_engine import N

    from core import signal_stream as SS
    stream, rep = SS.build(con, day, decide, meta, cal, actions_fn)
    if not getattr(rep, "ok", False):
        raise RuntimeError("no day cache — run the forge's _prepare_cache "
                           "first")
    if stream is None:
        raise RuntimeError("no signal stream")
    chain_fn, quote_fn = make_adapters(rep)
    curfew = _curfew_t()
    sw = BarSweep(bars, chain_fn, quote_fn, session_n=N, curfew_t=curfew,
                  seed=abs(hash(day)) % (2 ** 31))

    # v9.9.14: drive the sweep from on_signal — the COMPLETE conviction
    # stream — not on_block. on_block fires only where a signal was
    # REJECTED, so it omits precisely the seconds the live bar ACCEPTED,
    # which are the strongest of the day. Feeding a bar sweep from on_block
    # deletes the best members of the sample from every candidate bar below
    # the incumbent and makes them look worse than they are.
    def _hook(idx, ctx):
        sw.offer(Signal(t=int(ctx["t"]), index=idx,
                        conv=float(ctx["conv"]), wp=float(ctx.get("wp") or 0),
                        spot=float(ctx.get("spot") or 0),
                        ts=float(ctx.get("ts") or 0),
                        dte=float(ctx.get("dte") or 9.0)))

    # Fed from the SHARED stream: the decision loop ran once for this
    # session, in whichever study reached it first.
    last_t, si, n_sig = 0, 0, len(stream)
    end = min(int(stream.n_sec), N - 1)
    for sec in range(end + 1):
        while si < n_sig and int(stream.t[si]) == sec:
            _hook(str(stream.idx[si]),
                  {"t": sec, "conv": float(stream.conv[si]),
                   "wp": float(stream.wp[si]),
                   "spot": float(stream.spot[si])})
            si += 1
        sw.mark(sec)
        last_t = sec
    sw.finish(min(last_t, N - 1))
    s = sw.summary(curfew)
    log.info("  %s: %s", day, " | ".join(
        f"{k.replace('bar_', '')}={v['n_trades']}t/₹{v['pnl']:,.0f}"
        for k, v in sorted(s.items())))
    return {k: v["pnl"] for k, v in s.items()}, s


def _curfew_t() -> int:
    hm = str(getattr(config, "ENTRY_CURFEW", "15:05"))
    h, m = (int(x) for x in hm.split(":")[:2])
    return h * 3600 + m * 60 - (9 * 3600 + 15 * 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cap = capacity_note(_curfew_t())
    log.info("=" * 72)
    log.info("BOOK CAPACITY — read this before any bar number:")
    log.info("  MAX_CONCURRENT_POSITIONS = %d across ALL indices",
             cap["max_concurrent"])
    log.info("  %ds hold + %ds cooldown = %ds per slot, over a %ds entry "
             "window", cap["hold_s"], cap["cooldown_s"], cap["slot_s"],
             cap["entry_window_s"])
    log.info("  ⇒ AT MOST %d TRADE(S) PER SESSION, at ANY bar.",
             cap["max_trades_per_session"])
    log.info("  Lowering the bar buys eligibility, not trades. It spends the "
             "same slots earlier, on weaker signals.")
    log.info("=" * 72)

    bars = bar_grid()
    _register(bars)
    log.info("pre-registered grid (config.ENTRY_BAR_GRID): %s",
             ", ".join(f"{b:.2f}" for b in bars))

    # The real forge accessors. `decide` and `actions_fn` are built exactly
    # as evaluate_heuristic() builds them, so the sweep scores conviction
    # through the identical policy path the grader uses — a second decision
    # path here would make every bar comparison describe a different system.
    try:
        from nightly_forge_v9 import _eval_meta, _eval_cal
        from core.heuristic_policy import HeuristicPolicy
        meta, cal = _eval_meta(), _eval_cal()
        _pol = HeuristicPolicy()

        def decide(obs, frame, iidx):
            return float(_pol.predict(frame)[2 * iidx])

        def actions_fn(_o, _f):
            return _pol.predict(_f)
    except Exception as e:                                 # noqa: BLE001
        log.error("cannot build the replay decision path (%s). This tool "
                  "must run inside the evening chain, after the day caches "
                  "exist.", e)
        return 1

    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    per_day, detail = {}, {}
    try:
        days = sorted({r[0] for r in con.execute(
            "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch','localtime') "
            "FROM ticks_v9")})
    finally:
        con.close()
    if a.days > 0:
        days = days[-a.days:]
    log.info("sweeping %d session(s) across a pool — every bar sees the "
             "identical signal stream within a day, so the pairing is "
             "unaffected by which worker ran it", len(days))

    from core.parallel_days import map_days
    from simulation.scenario_engine import N as _N
    # TUPLE, not list. core/parallel_days.map_days stores results as
    # out[unit], so the work unit must be HASHABLE. On 2026-08-14 this was
    # (day, [0.2, 0.25, ...], 22500) — a tuple containing a list — and every
    # one of the 40 days completed its full replay and then raised
    # "unhashable type: 'list'" on the assignment. 59 494 seconds of correct
    # computation, discarded at the last line, reported only as "day
    # skipped". The most expensive class of bug there is: it does the work,
    # loses it, and looks like a data problem.
    _res = map_days(_sweep_day_worker,
                    [(d, tuple(bars), _N) for d in days],
                    desc="entry bar sweep", log_every=2)
    for _d, _o in zip(days, _res):
        if _o is None:
            log.warning("  %s: no result — session omitted from EVERY bar "
                        "(dropping it from one bar only would break the "
                        "pairing)", _d)
            continue
        pnl, s = _o
        per_day[_d] = pnl
        detail[_d] = s

    if len(per_day) < 3:
        log.info("only %d usable session(s) — nothing to conclude",
                 len(per_day))
        return 0

    # Same guard as gate_ab: a sweep in which no bar took a trade has
    # measured nothing. Reporting it as a flat grid would be a verdict with
    # no evidence behind it.
    _tt = sum(int(v["n_trades"]) for d in per_day for v in per_day[d].values()
              if isinstance(v, dict) and "n_trades" in v)
    if per_day and _tt == 0:
        log.error("EVERY bar took ZERO trades over %d session(s) — a "
                  "plumbing failure, not a flat grid. Check that the signal "
                  "stream restored last_tick onto the replayer. NO verdict.",
                  len(per_day))
        return 1

    v = EBS.evaluate(per_day, n_boot=int(getattr(config, "ENTRY_BAR_BOOT",
                                                 20000)))
    EBS.report(v, log)
    EBS.promote(v, dry_run=a.dry_run)

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "capacity": cap, "grid": bars, "per_day": per_day,
           "detail": detail, "verdict": v}
    p = Path(a.json) if a.json else (config.LOG_DIR /
                                     f"entry_bar_study_"
                                     f"{time.strftime('%Y-%m-%d')}.json")
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())