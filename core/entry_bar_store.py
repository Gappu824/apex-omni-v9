"""
ENTRY BAR STORE — choosing a threshold you swept, without fooling yourself
==========================================================================
Sweeping fourteen bars and reporting the best one is a multiple-comparisons
trap with a specific, well-understood shape, and the standard corrections do
NOT fix it. BH controls the false discovery rate across a family of
independent-ish hypotheses; here the hypotheses are a nested, strongly
correlated ladder (bar 0.50 and bar 0.55 share most of their trades), and the
quantity actually being reported is a MAXIMUM over that ladder. The maximum
of correlated statistics has its own null distribution, and it is not the
marginal one.

So the primary test here is the MAX-STATISTIC PERMUTATION (Westfall & Young):

    1. For every bar, compute the per-day paired Δ₹ against the incumbent.
    2. Observed statistic  M = max over bars of mean(Δ₹).
    3. Under the null "no bar differs from the incumbent", the sign of each
       DAY's Δ vector is exchangeable — flip whole days, not trades, because
       trades within a day share the tape and are not exchangeable.
    4. Recompute M for each of B sign-flip draws, KEEPING THE BAR AXIS
       INTACT so the correlation between bars is preserved in the null.
    5. p_max = P(M_null ≥ M_observed).

That p is family-wise valid for the statement anyone actually wants to make:
"the best bar in the grid beats the incumbent." Marginal per-bar p-values and
BH are still reported, but as description, never as the promotion criterion.

TWO REFERENCE POLICIES DECIDE WHETHER THE QUESTION IS EVEN LIVE
---------------------------------------------------------------
core.entry_counterfactual simulates ORACLE_TOPK (perfect-hindsight slot
filling) and RANDOM_SLOT (fill slots at random) alongside every bar. They
bracket the achievable range and answer two questions no p-value can:

  * incumbent ≥ ORACLE − ε   → there is no headroom. Moving the bar cannot
    help, and the constraint is elsewhere (the book, the exits, the signal).
  * incumbent ≤ RANDOM       → selection is not working at all. A bar change
    is premature; the conviction score itself is the problem.

Both are logged before any bar verdict, because a promotion that ignores them
is optimising a knob on a system whose binding constraint is somewhere else.

AND THE CONSTRAINT THAT REFRAMES EVERYTHING
-------------------------------------------
MAX_CONCURRENT_POSITIONS=1, a 60-minute guillotine and a 180s cooldown cap
the session at ~5 trades. Lowering the bar does not buy trades; it spends the
same slots earlier on weaker signals. Every report from this module leads
with that arithmetic so a bar result is never read as a volume story.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("entry_bar_store")

ORACLE = "ORACLE_TOPK"
RANDOM = "RANDOM_SLOT"


def _path() -> Path:
    return Path(getattr(config, "ENTRY_BAR_PATH",
                        config.STATE_DIR / "entry_bar.json"))


def max_stat_permutation(delta_by_day: dict[str, np.ndarray],
                         n_boot: int = 20000, seed: int = 0
                         ) -> tuple[float, float, np.ndarray]:
    """Westfall–Young max-T over a correlated grid.

    delta_by_day: day -> vector of Δ₹ (one entry per bar, same order).
    Whole DAYS are sign-flipped; the bar axis is never permuted, so the
    correlation structure across bars is preserved under the null. That is
    what makes the max distribution the right one.
    """
    days = sorted(delta_by_day)
    M = np.vstack([np.asarray(delta_by_day[d], float) for d in days])
    if M.shape[0] < 3:
        return float("nan"), 1.0, np.zeros(M.shape[1])
    obs_means = np.nanmean(M, axis=0)
    obs_max = float(np.nanmax(obs_means))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_boot)):
        s = rng.choice((-1.0, 1.0), size=(M.shape[0], 1))
        if float(np.nanmax(np.nanmean(M * s, axis=0))) >= obs_max:
            hits += 1
    p = (hits + 1) / (int(n_boot) + 1)
    return obs_max, float(p), obs_means


def evaluate(per_day: dict, incumbent: float | None = None,
             n_boot: int = 20000) -> dict:
    """Judge a bar sweep.

    per_day: {day: {policy_name: realised ₹}} where policy_name is
    'bar_0.55', ORACLE or RANDOM. Every day must carry every policy —
    a ragged panel would make the pairing false.
    """
    from core import capability_ladder as CL

    inc = float(incumbent if incumbent is not None
                else config.entry_conviction_bar())
    inc_key = f"bar_{inc:.2f}"
    days = sorted(per_day)
    if len(days) < 3:
        return {"ok": False, "reason": f"only {len(days)} session(s)",
                "n_days": len(days)}

    bars = sorted({k for d in days for k in per_day[d]
                   if k.startswith("bar_")},
                  key=lambda k: float(k.split("_")[1]))
    if inc_key not in bars:
        return {"ok": False,
                "reason": f"incumbent {inc_key} is not in the sweep — the "
                          f"comparison has no baseline", "n_days": len(days)}
    cand = [b for b in bars if b != inc_key]

    # ---- reference bracket: is the question even live?
    def _mean(k):
        v = [float(per_day[d][k]) for d in days if k in per_day[d]]
        return float(np.mean(v)) if v else float("nan")

    ref = {"incumbent_mean": _mean(inc_key), "oracle_mean": _mean(ORACLE),
           "random_mean": _mean(RANDOM)}
    span = ref["oracle_mean"] - ref["random_mean"]
    # ORACLE_TOPK is perfect hindsight ON CONVICTION, not on outcome — it is
    # the ceiling of what ANY conviction threshold could reach. So span <= 0
    # is not a numerical edge case, it is a verdict: the highest-conviction
    # signals of the day do WORSE than randomly chosen ones. When that holds,
    # conviction is anti-selective over this sample and NO bar anywhere on
    # the grid can help — the score is the problem, not the threshold.
    ref["anti_selective"] = bool(np.isfinite(span) and span <= 0)
    ref["headroom_frac"] = (float((ref["oracle_mean"] - ref["incumbent_mean"])
                                  / span) if np.isfinite(span) and span > 0
                            else float("nan"))
    ref["beats_random"] = bool(ref["incumbent_mean"] > ref["random_mean"])
    ref["near_oracle"] = bool(np.isfinite(ref["headroom_frac"])
                              and ref["headroom_frac"] < 0.10)

    # ---- paired deltas, one vector per day across the candidate grid
    delta_by_day = {d: np.array([per_day[d][c] - per_day[d][inc_key]
                                 for c in cand], float)
                    for d in days if inc_key in per_day[d]}
    obs_max, p_max, obs_means = max_stat_permutation(delta_by_day, n_boot)

    rows = []
    for j, c in enumerate(cand):
        col = np.array([delta_by_day[d][j] for d in sorted(delta_by_day)],
                       float)
        st = CL.paired_test({d: float(delta_by_day[d][j])
                             for d in sorted(delta_by_day)})
        rows.append({"policy": c, "bar": float(c.split("_")[1]),
                     "mean": st["mean"], "ci90": st.get("ci90"),
                     "p_marginal": st.get("p", 1.0),
                     "mde": st.get("mde", float("nan")),
                     "total_rs": round(float(np.nansum(col)), 2),
                     "n_days": int(np.isfinite(col).sum())})
    if rows:
        rej, adj = CL.benjamini_hochberg(
            [r["p_marginal"] for r in rows],
            float(getattr(config, "ENTRY_BAR_FDR_Q", 0.10)))
        for r, ok_, q_ in zip(rows, rej, adj):
            r["p_adj_bh"] = round(float(q_), 4)
            r["significant_marginal"] = bool(ok_)

    best = max(rows, key=lambda r: r["mean"]) if rows else None

    # ---- ladder
    y = np.array([1.0 if float(per_day[d][inc_key]) > 0 else 0.0
                  for d in days])
    dd = np.array(days)
    cap = CL.assess(y, np.ones(len(days)), dd)
    stage_req = str(getattr(config, "ENTRY_BAR_LADDER_STAGE", "DISCOVER"))

    gates = {
        f"ladder_comparative_{stage_req}": bool(
            cap.allows_comparative(stage_req)),
        "enough_days": len(days) >= int(getattr(config,
                                                "ENTRY_BAR_MIN_DAYS", 30)),
        "selection_beats_random": ref["beats_random"],
        "headroom_exists": not ref["near_oracle"],
        "conviction_is_selective": not ref["anti_selective"],
        "max_stat_significant": bool(
            p_max <= float(getattr(config, "ENTRY_BAR_ALPHA", 0.05))),
    }
    checks = {}
    if best is not None:
        checks = {
            "best_mean_positive": best["mean"] > 0,
            "above_mde": (best["mean"] >
                          float(getattr(config, "ENTRY_BAR_MDE_MULT", 1.0))
                          * float(best.get("mde", float("inf")))),
            "ci_lower_clears": (best.get("ci90") is not None
                                and best["ci90"][0] is not None
                                and float(best["ci90"][0]) > 0.0),
        }

    ok = bool(best is not None and all(gates.values()) and all(checks.values()))
    return {"ok": ok, "winner": (best["policy"] if ok else None),
            "winner_bar": (best["bar"] if ok else None),
            "incumbent": inc, "incumbent_key": inc_key,
            "p_max_westfall_young": round(float(p_max), 5),
            "observed_max_mean": (round(float(obs_max), 2)
                                  if np.isfinite(obs_max) else None),
            "n_boot": int(n_boot), "reference": ref, "gates": gates,
            "checks": checks, "bars": rows, "n_days": len(days),
            "capacity": _capacity(),
            "ladder": cap.as_dict() if hasattr(cap, "as_dict") else {},
            "config_hash": config.CONFIG_HASH, "ts": time.time()}


def _capacity() -> dict:
    from core.entry_counterfactual import capacity_note
    return capacity_note()


def report(v: dict, logger: logging.Logger | None = None) -> None:
    lg = logger or log
    cap = v.get("capacity", {})
    lg.info("BOOK CAPACITY: %d concurrent, %ds hold + %ds cooldown = %ds per "
            "slot over a %ds entry window → AT MOST %d trade(s)/session. "
            "The bar cannot change this number.",
            cap.get("max_concurrent", 1), cap.get("hold_s", 0),
            cap.get("cooldown_s", 0), cap.get("slot_s", 0),
            cap.get("entry_window_s", 0),
            cap.get("max_trades_per_session", 0))
    r = v.get("reference", {})
    lg.info("REFERENCE BRACKET: random ₹%s | incumbent ₹%s | oracle ₹%s "
            "→ headroom %s of the achievable span "
            "(oracle = perfect hindsight ON CONVICTION, the ceiling any bar "
            "could reach)",
            f"{r.get('random_mean', float('nan')):,.0f}",
            f"{r.get('incumbent_mean', float('nan')):,.0f}",
            f"{r.get('oracle_mean', float('nan')):,.0f}",
            f"{100 * r.get('headroom_frac', float('nan')):.0f}%")
    if not r.get("beats_random", True):
        lg.warning("the incumbent bar does NOT beat random slot-filling — "
                   "the conviction score is not selecting. A bar change is "
                   "premature; this is a signal problem, not a threshold "
                   "problem.")
    if r.get("anti_selective"):
        lg.warning("ANTI-SELECTIVE: perfect-hindsight selection ON "
                   "CONVICTION (₹%s) came in BELOW random slot-filling "
                   "(₹%s). Over this sample the highest-conviction signals "
                   "are worse than arbitrary ones, so no bar on the grid can "
                   "help — the conviction score is the problem, not the "
                   "threshold. Moving the bar here would be tuning a knob "
                   "that is wired to nothing.",
                   f"{r.get('oracle_mean', float('nan')):,.0f}",
                   f"{r.get('random_mean', float('nan')):,.0f}")
    if r.get("near_oracle"):
        lg.warning("the incumbent is within 10%% of perfect-hindsight slot "
                   "filling — there is no headroom in the bar. The binding "
                   "constraint is elsewhere (book size, exits, or signal).")
    lg.info("─" * 72)
    lg.info("%-10s %12s %12s %10s %10s", "bar", "Σ Δ₹", "mean/day",
            "p(marg)", "p(BH)")
    for b in sorted(v.get("bars", []), key=lambda z: z["bar"]):
        lg.info("%-10.2f %12s %12s %10.3f %10.3f", b["bar"],
                f"{b['total_rs']:,.0f}", f"{b['mean']:,.0f}",
                b.get("p_marginal", 1.0), b.get("p_adj_bh", 1.0))
    lg.info("─" * 72)
    lg.info("MAX-STATISTIC (Westfall–Young, %d sign-flip draws over whole "
            "days, bar axis intact): observed max mean Δ₹ %s/day, "
            "p_max = %.4f", v.get("n_boot", 0),
            f"{v.get('observed_max_mean', float('nan')):,.0f}",
            v.get("p_max_westfall_young", 1.0))
    lg.info("This p is the family-wise one. The per-bar columns above are "
            "description, not the criterion — the grid is a nested ladder "
            "and its maximum has its own null.")
    for k, okk in v.get("gates", {}).items():
        lg.info("gate %-32s %s", k, "PASS" if okk else "FAIL")
    for k, okk in v.get("checks", {}).items():
        lg.info("check %-31s %s", k, "PASS" if okk else "FAIL")


def promote(v: dict, dry_run: bool = False) -> bool:
    if not bool(getattr(config, "ENTRY_BAR_PROMOTE_ENABLED", False)):
        log.info("ENTRY_BAR_PROMOTE_ENABLED is False — verdict recorded, "
                 "the bar is unchanged. Entry-side promotion is OFF by "
                 "default: the bar is the single knob with the most direct "
                 "path to capital, and it moves by hand or not at all.")
        return False
    if not v.get("ok"):
        bad = [k for k, x in {**v.get("gates", {}),
                              **v.get("checks", {})}.items() if not x]
        log.info("no bar cleared the gate over %d session(s) — %s stands. "
                 "blocked by: %s", v.get("n_days", 0),
                 v.get("incumbent_key"), ", ".join(bad) or "n/a")
        return False
    if dry_run:
        log.info("DRY RUN — bar %.2f would replace %.2f", v["winner_bar"],
                 v["incumbent"])
        return False
    body = {"bar": v["winner_bar"], "previous": v["incumbent"],
            "promoted_ts": time.time(),
            "promoted_day": dt.date.today().isoformat(),
            "config_hash": config.CONFIG_HASH,
            "evidence": {k: v.get(k) for k in
                         ("n_days", "p_max_westfall_young", "reference",
                          "gates", "checks", "bars", "capacity", "ladder")}}
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(body, indent=1, default=float),
                       encoding="utf-8")
        os.replace(tmp, p)
        log.info("PROMOTED entry bar %.2f → %.2f (p_max=%.4f over %d "
                 "session(s))", v["incumbent"], v["winner_bar"],
                 v["p_max_westfall_young"], v["n_days"])
        return True
    except Exception as e:                                 # noqa: BLE001
        log.warning("bar promotion write failed (%s) — bar unchanged", e)
        return False


def active_bar() -> float:
    """The promoted bar for THIS config world, else the config default."""
    try:
        p = _path()
        if not p.exists():
            return float(config.entry_conviction_bar())
        b = json.loads(p.read_text(encoding="utf-8"))
        if b.get("config_hash") != config.CONFIG_HASH:
            log.info("promoted bar %.2f was measured under CONFIG_HASH %s "
                     "≠ %s — ignored", float(b.get("bar", 0)),
                     b.get("config_hash"), config.CONFIG_HASH)
            return float(config.entry_conviction_bar())
        return float(b.get("bar") or config.entry_conviction_bar())
    except Exception as e:                                 # noqa: BLE001
        log.debug("active bar read failed (%s)", e)
        return float(config.entry_conviction_bar())