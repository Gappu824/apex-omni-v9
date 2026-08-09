"""
EXIT POLICY STORE — measuring is not permission
================================================
tools/trade_potential.py has been in the evening chain since v9.9.10 and
NOTHING reads its output. It writes logs/trade_potential_<date>.json and
that file is never opened again — not by the forge, not by the ladder,
not by the exit stack, not even by gemma_analyst's brief. The loop has
been open the whole time: the system measures what its exits leave on
the table and then discards the answer.

This module closes it, and closes it CONSERVATIVELY. An exit rule is the
part of the stack with the most direct path to the P&L, so promoting one
on a flattering sample is the most expensive mistake available here. The
gate below is deliberately harder to pass than the one that promotes a
model:

  1. LADDER. core.capability_ladder must allow comparative claims at all.
     Below that stage the vault cannot resolve a policy difference and
     the question is not yet askable.
  2. SAMPLE. ≥ SHADOW_PROMOTE_MIN_DAYS distinct sessions and
     ≥ SHADOW_PROMOTE_MIN_TRADES trades. Days are the exchangeable unit;
     trades alone can be one lucky morning.
  3. PAIRED, CLUSTERED, CORRECTED. Per-day mean Δ₹ vs as_traded through
     capability_ladder.paired_test (day-cluster bootstrap + exact sign-
     flip permutation), then Benjamini-Hochberg across the whole
     pre-registered family. Testing thirteen policies and reporting the
     best one uncorrected is how a random walk gets promoted.
  4. MDE FLOOR. mean Δ₹/day must exceed SHADOW_PROMOTE_MDE_MULT × the
     minimum detectable effect. "Significant" on a thin sample can still
     be an effect too small for the sample to have resolved honestly.
  5. CI SIGN. The 90% CI lower bound must clear SHADOW_PROMOTE_MIN_CI_LO.
  6. HOLDOUT AGREEMENT. The last SHADOW_PROMOTE_HOLDOUT of days is held
     out of the decision and must agree in SIGN. A policy that only wins
     on the oldest data is a regime artifact, and the regimes this system
     trades rotate.
  7. STALE EXCLUSION. Any trade whose live-mark coverage was below
     SHADOW_MIN_COVERAGE is excluded from the verdict and COUNTED in the
     report. A policy cannot be promoted on paths that were forward-
     filled corpses — that is the flat-line defect being laundered into
     a decision.

A promotion writes state/exit_policy.json with the winning policy, the
full evidence that justified it, and the CONFIG_HASH it was measured
under. Reading code MUST check that hash: a policy measured under a
different feature world is not evidence about this one.

Nothing here arms anything by itself. `active_policy()` returns the
promoted name; whether the live exit stack honours it is a separate
decision made in the exit stack, behind its own switch.
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

log = logging.getLogger("exit_policy_store")


def _path() -> Path:
    return Path(getattr(config, "SHADOW_POLICY_PATH",
                        config.STATE_DIR / "exit_policy.json"))


# --------------------------------------------------------------- verdict
def evaluate(per_trade: list[dict]) -> dict:
    """Judge the policy family against as_traded.

    per_trade rows need: day, policy_pnl {name: ₹}, as_traded ₹, coverage.
    Returns the full verdict — including every reason a policy did NOT
    pass, because a gate that only reports its winners teaches nothing.
    """
    from core import capability_ladder as CL

    min_cov = float(getattr(config, "SHADOW_MIN_COVERAGE", 0.60))
    usable = [r for r in per_trade if float(r.get("coverage", 0.0)) >= min_cov]
    excluded = len(per_trade) - len(usable)
    if not usable:
        return {"ok": False, "reason": "no trade cleared the coverage floor",
                "n_trades": 0, "n_excluded_stale": excluded}

    days = sorted({str(r["day"]) for r in usable})
    names = [n for n in getattr(config, "SHADOW_POLICIES", ()) 
             if n != "as_traded"]

    # per-policy, per-day mean Δ₹ vs the real exit
    deltas: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    for r in usable:
        base = float(r.get("as_traded", 0.0))
        for n in names:
            v = r.get("policy_pnl", {}).get(n)
            if v is None:
                continue
            deltas[n].setdefault(str(r["day"]), []).append(float(v) - base)

    hold_frac = float(getattr(config, "SHADOW_PROMOTE_HOLDOUT", 0.30))
    n_hold = max(int(round(len(days) * hold_frac)), 1) if len(days) >= 4 else 0
    train_days = set(days[:len(days) - n_hold]) if n_hold else set(days)
    hold_days = set(days[len(days) - n_hold:]) if n_hold else set()

    rows = []
    for n in names:
        per_day = {d: float(np.mean(v)) for d, v in deltas[n].items() if v}
        if len(per_day) < 3:
            rows.append({"policy": n, "n_days": len(per_day),
                         "verdict": "insufficient days", "significant": False,
                         "mean": float("nan"), "p": 1.0})
            continue
        tr = {d: v for d, v in per_day.items() if d in train_days}
        st = CL.paired_test(tr if len(tr) >= 3 else per_day)
        ho_mean = (float(np.mean([per_day[d] for d in hold_days
                                  if d in per_day]))
                   if hold_days and any(d in per_day for d in hold_days)
                   else float("nan"))
        rows.append({"policy": n, "n_days": len(per_day),
                     "n_train_days": len(tr), "mean": st["mean"],
                     "ci90": st.get("ci90"), "p": st.get("p", 1.0),
                     "mde": st.get("mde", float("nan")),
                     "holdout_mean": ho_mean,
                     "total_rs": round(sum(sum(v) for v in
                                           deltas[n].values()), 2)})

    live = [r for r in rows if np.isfinite(r.get("mean", float("nan")))]
    if live:
        rej, adj = CL.benjamini_hochberg(
            [r["p"] for r in live],
            float(getattr(config, "SHADOW_PROMOTE_FDR_Q", 0.10)))
        for r, ok_, q_ in zip(live, rej, adj):
            r["p_adj_bh"] = round(float(q_), 4)
            r["significant"] = bool(ok_)

    # ladder: is a comparative claim admissible at all?
    y = np.array([1.0 if float(r.get("as_traded", 0.0)) > 0 else 0.0
                  for r in usable])
    d = np.array([str(r["day"]) for r in usable])
    cap = CL.assess(y, np.ones(len(usable)), d)

    min_days = int(getattr(config, "SHADOW_PROMOTE_MIN_DAYS", 20))
    min_trades = int(getattr(config, "SHADOW_PROMOTE_MIN_TRADES", 60))
    mde_mult = float(getattr(config, "SHADOW_PROMOTE_MDE_MULT", 1.0))
    ci_lo_min = float(getattr(config, "SHADOW_PROMOTE_MIN_CI_LO", 0.0))

    # capability_ladder.STAGES = (BLIND, SCREEN, DISCOVER, PROMOTE).
    # A policy promotion is a DECISION, not research, so it is gated on
    # the COMPARATIVE ladder at DISCOVER — one stage above the bar the
    # research tools (horizon_sweep, feature_discovery) run at. The
    # comparative ladder is the right one because every claim here is
    # within-day paired against as_traded, so between-day variance is
    # cancelled by construction rather than paid for.
    stage_req = str(getattr(config, "SHADOW_PROMOTE_LADDER_STAGE",
                            "DISCOVER"))
    gates = {
        f"ladder_comparative_{stage_req}": bool(
            cap.allows_comparative(stage_req)),
        "enough_days": len(days) >= min_days,
        "enough_trades": len(usable) >= min_trades,
    }

    winner, why = None, []
    for r in sorted(rows, key=lambda z: -(z.get("mean") or -1e18)):
        name = r["policy"]
        checks = {
            "significant_bh": bool(r.get("significant")),
            "mean_positive": float(r.get("mean", 0) or 0) > 0,
            "above_mde": (float(r.get("mean", 0) or 0) >
                          mde_mult * float(r.get("mde", float("inf")))),
            "ci_lower_clears": (r.get("ci90") and r["ci90"][0] is not None
                                and float(r["ci90"][0]) > ci_lo_min),
            "holdout_agrees": (not hold_days or
                               (np.isfinite(r.get("holdout_mean",
                                                  float("nan")))
                                and float(r["holdout_mean"]) > 0)),
        }
        r["checks"] = checks
        if all(checks.values()) and all(gates.values()):
            winner = name
            break
        why.append({name: [k for k, v in checks.items() if not v]})

    return {"ok": winner is not None, "winner": winner,
            "gates": gates, "policies": rows, "failed_checks": why,
            "n_trades": len(usable), "n_days": len(days),
            "n_excluded_stale": excluded,
            "holdout_days": sorted(hold_days),
            "ladder": cap.as_dict() if hasattr(cap, "as_dict") else {},
            "config_hash": config.CONFIG_HASH, "ts": time.time()}


# ------------------------------------------------------------- promotion
def promote(verdict: dict, dry_run: bool = False) -> bool:
    """Write the promotion if and only if the verdict earned it."""
    if not bool(getattr(config, "SHADOW_PROMOTE_ENABLED", True)):
        log.info("SHADOW_PROMOTE_ENABLED is False — verdict recorded, "
                 "nothing promoted")
        return False
    if not verdict.get("ok"):
        log.info("no policy cleared the gate (%d trade(s), %d day(s)) — "
                 "as_traded stands", verdict.get("n_trades", 0),
                 verdict.get("n_days", 0))
        bad_gates = [k for k, v in verdict.get("gates", {}).items() if not v]
        if bad_gates:
            # A SAMPLE gate failure is a different statement from a policy
            # failure: it says the question is not yet answerable, not that
            # the answer is no. Conflating them is how "not enough data"
            # gets read as "nothing works".
            log.info("   blocked by sample/ladder gate(s): %s — this is "
                     "'cannot yet tell', NOT 'no policy is better'",
                     ", ".join(bad_gates))
        for item in verdict.get("failed_checks", [])[:5]:
            for name, failed in item.items():
                log.info("   %-15s %s", name,
                         ("failed: " + ", ".join(failed)) if failed
                         else "cleared every per-policy check; held back by "
                              "the sample/ladder gate above")
        return False
    if dry_run:
        log.info("DRY RUN — %s would be promoted", verdict["winner"])
        return False
    body = {"policy": verdict["winner"],
            "promoted_ts": time.time(),
            "promoted_day": dt.date.today().isoformat(),
            "config_hash": config.CONFIG_HASH,
            "evidence": {k: verdict.get(k) for k in
                         ("n_trades", "n_days", "gates", "policies",
                          "holdout_days", "ladder", "n_excluded_stale")}}
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if p.exists():
            try:
                prev = json.loads(p.read_text(encoding="utf-8"))
                history = list(prev.get("history") or [])
                history.append({k: prev.get(k) for k in
                                ("policy", "promoted_ts", "config_hash")})
            except (OSError, ValueError):
                history = []
        body["history"] = history[-20:]
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(body, indent=1, default=float),
                       encoding="utf-8")
        os.replace(tmp, p)
        log.info("PROMOTED exit policy '%s' — %d trade(s) over %d session(s)",
                 verdict["winner"], verdict["n_trades"], verdict["n_days"])
        return True
    except Exception as e:                                 # noqa: BLE001
        log.warning("promotion write failed (%s) — as_traded stands", e)
        return False


def active_policy() -> tuple[str, dict]:
    """The promoted policy for THIS config world, or ('as_traded', {}).

    A policy measured under a different CONFIG_HASH is not evidence about
    this one — the feature world it was measured in no longer exists.
    """
    try:
        p = _path()
        if not p.exists():
            return "as_traded", {}
        body = json.loads(p.read_text(encoding="utf-8"))
        if body.get("config_hash") != config.CONFIG_HASH:
            log.info("promoted policy '%s' was measured under CONFIG_HASH "
                     "%s ≠ %s — ignored until re-measured",
                     body.get("policy"), body.get("config_hash"),
                     config.CONFIG_HASH)
            return "as_traded", body
        return str(body.get("policy") or "as_traded"), body
    except Exception as e:                                 # noqa: BLE001
        log.debug("active policy read failed (%s)", e)
        return "as_traded", {}