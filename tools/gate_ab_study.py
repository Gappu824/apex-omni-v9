"""
GATE A/B STUDY — what would the day plan and the range gate have DONE?
=======================================================================
Two gates were built off the 2026-08-11 tape and both are OFF by default:

  DAY PLAN (core/day_plan.py) — one thesis per session, committed in a
    window after the open, reviewed mid-session, flat at 15:05. Motivated
    by a ledger in which five of six trades were NIFTY puts and the 24400
    strike was entered three times, stopping out every time, for
    -Rs1297.44. MAX_CONCURRENT_POSITIONS=1 bounds concurrency, not
    repetition, and COOLDOWN_S=180 is irrelevant against re-entries 65 and
    121 minutes apart.

  RANGE GATE (core/range_regime.py) — refuse directional premium buying
    when a Lo-MacKinlay variance ratio says the tape is mean-reverting on
    several horizons at once. Motivated by NIFTY trending for 2 seconds of
    18047 on the day those five puts were bought.

Both diagnoses are supported by the ledger. NEITHER is evidence that the
GATE PAYS. "This trade lost and my gate would have blocked it" is the
oldest mistake in systematic trading: the gate that blocks a loser also
blocks winners, and on 2026-08-11 the ONLY profitable trade was the 09:20
entry — which the day plan's 09:45 observation window would have refused.
That single fact is why this study exists and why the switches ship off.

THE DESIGN: 2x2 FACTORIAL, PAIRED BY SESSION
---------------------------------------------
Four arms replay the SAME tape, in the same order, with the same shared
exit engine:

    A  incumbent          (neither gate)
    B  +day plan
    C  +range gate
    D  +both

A 2x2 rather than three separate comparisons, because the gates OVERLAP:
the day plan already refuses most entries, so the range gate may add
nothing on top of it. A factorial estimates the two MAIN EFFECTS and the
INTERACTION, and the interaction is the number that says whether the
second gate is redundant. Running B and C separately and adding them would
double-count whatever they both block.

Every arm sees the identical signal stream, so the pairing is exact at the
day level and no arm can benefit from a different sample. Contrasts are
per-session means, tested with the day-clustered bootstrap in
core/capability_ladder.py, and Benjamini-Hochberg is applied across the
three contrasts — testing three and reporting the best one uncorrected is
how a coin flip gets promoted.

WHAT THIS STUDY CANNOT SETTLE
------------------------------
It conditions on the signals the incumbent policy produced. A gate that
changes WHICH trades happen also changes what the book learns, and this
replay cannot capture that feedback. It answers a narrower question
honestly: holding the signal stream fixed, does the gate improve the
realised outcome? That is the right question to ask before arming
something, and it is not the same as "is the gate correct".
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

config.setup_logging("gate_ab_study")
import logging                                             # noqa: E402
log = logging.getLogger("gate_ab_study")

from core import range_regime as RR                        # noqa: E402
from core.entry_counterfactual import (BarSweep, Signal,    # noqa: E402
                                       capacity_note)

ARMS = ("incumbent", "day_plan", "range_gate", "both")


class _PlanSim:
    """The day plan, driven by session-seconds instead of the wall clock.

    core/day_plan.py reads the clock because it lives in the live loop.
    Re-implementing its rules here would be a second implementation that
    could drift from the one that trades — the exact skew that made
    meta_gbm serve a constant. So this holds the SAME thresholds, read
    from the same config keys, and the audit asserts the two agree.
    """

    def __init__(self, t0_sod: int = 9 * 3600 + 15 * 60):
        self.t0 = t0_sod
        self.committed = False
        self.entry_conv = 0.0
        self.reviewed = False
        self.closed = False

    @staticmethod
    def _sod(hm_key: str, default: str) -> int:
        hm = str(getattr(config, hm_key, default))
        h, m = (int(x) for x in hm.split(":")[:2])
        return h * 3600 + m * 60

    def _t(self, key, default):
        return self._sod(key, default) - self.t0

    def may_enter_t(self, t: int):
        if self.committed:
            return False, "day plan: already committed — one thesis"
        if t < self._t("DAYPLAN_ENTRY_HM", "09:50"):
            return False, "day plan: observing/analysing"
        if t >= self._t("DAYPLAN_COMMIT_END_HM", "10:20"):
            return False, "day plan: commit window closed"
        return True, ""

    def commit_t(self, t: int, conv: float, index: str) -> None:
        self.committed = True
        self.entry_conv = float(conv)

    def tick_t(self, t: int, live_conv: float):
        if self.closed:
            return False, ""
        if t >= self._t("DAYPLAN_EXIT_HM", "15:05"):
            self.closed = True
            return True, "DAYPLAN_SESSION_EXIT"
        if (not self.reviewed
                and t >= self._t("DAYPLAN_REVIEW_HM", "12:30")):
            self.reviewed = True
            rev = float(getattr(config, "DAYPLAN_REVERSAL_CONV", 0.40))
            dec = float(getattr(config, "DAYPLAN_DECAY_CONV", 0.15))
            es = 1.0 if self.entry_conv > 0 else -1.0
            ls = 1.0 if live_conv > 0 else -1.0
            if ls != es and abs(live_conv) >= rev:
                self.closed = True
                return True, "DAYPLAN_THESIS_REVERSED"
            if abs(live_conv) < dec:
                self.closed = True
                return True, "DAYPLAN_THESIS_DECAYED"
        return False, ""


def _range_fn(spot_by_t, cache):
    """Assess the range regime on the spot series SO FAR — never the whole
    day. Using the full session would leak the afternoon into a morning
    decision, which is the single easiest way to make a gate look good."""
    step = int(getattr(config, "RANGE_ASSESS_EVERY_S", 300))

    def fn(t: int, index: str):
        k = (index, t // step)
        if k not in cache:
            hist = spot_by_t.get(index)
            # enabled=True unconditionally: this ARM exists to measure the
            # gate, so it must not inherit the live default.
            cache[k] = (True, "") if hist is None else \
                RR.may_trade_directional(RR.assess(hist[:t + 1]),
                                         enabled=True)
        return cache[k]
    return fn


def study_day(con, day, decide, meta, cal, actions_fn=None) -> dict | None:
    from nightly_forge_v9 import _Replayer
    from simulation.scenario_engine import N

    rep = _Replayer(con, day, meta, cal)
    if not getattr(rep, "ok", False):
        return None

    from tools.entry_bar_study import make_adapters
    chain_fn, quote_fn = make_adapters(rep)

    curfew = _PlanSim()._t("DAYPLAN_EXIT_HM", "15:05")
    bar = float(config.entry_conviction_bar())
    spot_by_t: dict[str, np.ndarray] = {}
    for idx in config.TRADABLE:
        s = getattr(rep, "spot", None)
        if isinstance(s, dict):
            s = s.get(idx)
        if s is not None:
            spot_by_t[idx] = np.asarray(s, float)
    rcache: dict = {}
    rfn = _range_fn(spot_by_t, rcache)

    sw = BarSweep([bar], chain_fn, quote_fn, session_n=N,
                  curfew_t=N - 1, seed=abs(hash(day)) % (2 ** 31),
                  include_reference=False)
    sw.clear_bar_grid()
    sw.add_arm("incumbent", bar)
    sw.add_arm("day_plan", bar, day_plan=_PlanSim())
    sw.add_arm("range_gate", bar, range_fn=rfn)
    sw.add_arm("both", bar, day_plan=_PlanSim(), range_fn=rfn)

    def _hook(idx, ctx):
        sw.offer(Signal(t=int(ctx["t"]), index=idx, conv=float(ctx["conv"]),
                        wp=float(ctx.get("wp") or 0),
                        spot=float(ctx.get("spot") or 0)))

    last = 0
    for ev in rep.run(decide, on_signal=_hook, actions_fn=actions_fn):
        if ev and ev[0] == "sec":
            last = int(ev[1])
            sw.mark(last)
    sw.finish(min(last, N - 1))
    s = sw.summary(curfew)
    log.info("  %s | %s", day, " | ".join(
        f"{a}: {s[a]['n_trades']}t Rs{s[a]['pnl']:,.0f}" for a in ARMS
        if a in s))
    return s


def analyse(per_day: dict) -> dict:
    from core import capability_ladder as CL
    days = sorted(per_day)
    if len(days) < 3:
        return {"ok": False, "reason": f"{len(days)} session(s)"}

    def col(a):
        return np.array([per_day[d][a]["pnl"] for d in days], float)

    A, B, C, D = (col(x) for x in ARMS)
    # 2x2 main effects and interaction, per session
    contrasts = {
        "day_plan_main": ((B + D) - (A + C)) / 2.0,
        "range_main": ((C + D) - (A + B)) / 2.0,
        "interaction": (D - C) - (B - A),
    }
    rows = []
    for name, v in contrasts.items():
        st = CL.paired_test({days[i]: float(v[i]) for i in range(len(days))})
        rows.append({"contrast": name, "mean": st["mean"],
                     "ci90": st.get("ci90"), "p": st.get("p", 1.0),
                     "mde": st.get("mde", float("nan")),
                     "total": round(float(np.sum(v)), 2)})
    rej, adj = CL.benjamini_hochberg([r["p"] for r in rows], 0.10)
    for r, ok_, q_ in zip(rows, rej, adj):
        r["p_adj_bh"] = round(float(q_), 4)
        r["significant"] = bool(ok_)

    arms = {a: {"total": round(float(col(a).sum()), 2),
                "mean_per_day": round(float(col(a).mean()), 2),
                "trades": int(sum(per_day[d][a]["n_trades"] for d in days))}
            for a in ARMS}
    return {"ok": True, "n_days": len(days), "arms": arms,
            "contrasts": rows, "config_hash": config.CONFIG_HASH}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    cap = capacity_note()
    log.info("=" * 70)
    log.info("GATE A/B — 2x2 factorial, paired by session, same tape")
    log.info("  arms: %s", ", ".join(ARMS))
    log.info("  book: %d concurrent, %ds slot => at most %d trade(s)/session",
             cap["max_concurrent"], cap["slot_s"],
             cap["max_trades_per_session"])
    log.info("  NOTE: on 2026-08-11 the only PROFITABLE trade was the 09:20 "
             "entry, which the day plan's observation window refuses. A gate "
             "that blocks losers blocks winners too — that is what this "
             "measures.")
    log.info("=" * 70)

    try:
        from nightly_forge_v9 import _eval_meta, _eval_cal
        from core.heuristic_policy import HeuristicPolicy
        meta, cal = _eval_meta(), _eval_cal()
        pol = HeuristicPolicy()

        def decide(obs, frame, iidx):
            return float(pol.predict(frame)[2 * iidx])

        def actions_fn(_o, _f):
            return pol.predict(_f)
    except Exception as e:                                 # noqa: BLE001
        log.error("cannot build the replay decision path (%s)", e)
        return 1

    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    per_day = {}
    try:
        days = sorted({r[0] for r in con.execute(
            "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch','localtime') "
            "FROM ticks_v9")})
        if a.days > 0:
            days = days[-a.days:]
        for d in days:
            try:
                s = study_day(con, d, decide, meta, cal, actions_fn)
                if s and all(x in s for x in ARMS):
                    per_day[d] = s
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s skipped (%s)", d, e)
    finally:
        con.close()

    v = analyse(per_day)
    if not v.get("ok"):
        log.info("not enough sessions (%s)", v.get("reason"))
        return 0

    log.info("─" * 70)
    log.info("%-12s %12s %12s %8s", "arm", "total Rs", "Rs/session", "trades")
    for k, x in v["arms"].items():
        log.info("%-12s %12s %12s %8d", k, f"{x['total']:,.0f}",
                 f"{x['mean_per_day']:,.0f}", x["trades"])
    log.info("─" * 70)
    log.info("%-16s %12s %10s %10s  verdict", "contrast", "Rs/session",
             "p(BH)", "MDE")
    for r in v["contrasts"]:
        verdict = ("HELPS" if r["significant"] and r["mean"] > 0 else
                   "HURTS" if r["significant"] else
                   f"indistinguishable (could resolve Rs{r['mde']:,.0f})")
        log.info("%-16s %12s %10.3f %10s  %s", r["contrast"],
                 f"{r['mean']:,.0f}", r.get("p_adj_bh", 1.0),
                 f"{r['mde']:,.0f}", verdict)
    log.info("─" * 70)
    log.info("The INTERACTION is the redundancy test: the gates overlap, so "
             "a large negative interaction means the second gate blocks "
             "what the first already blocked and adds nothing.")
    log.info("Nothing is armed by this study. DAYPLAN_ENABLED and "
             "RANGE_GATE_ENABLED remain operator decisions.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "capacity": cap, "per_day": per_day, "verdict": v}
    p = Path(a.json) if a.json else (
        config.LOG_DIR / f"gate_ab_{time.strftime('%Y-%m-%d')}.json")
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())