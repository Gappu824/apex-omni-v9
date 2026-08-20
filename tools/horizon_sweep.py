"""
HORIZON SWEEP — is 60 minutes the right question to ask the model?
===================================================================
    python tools/horizon_sweep.py [--days N] [--adopt]

The meta labels every sample by first touch inside ONE hold window. That
window is a modelling choice, not a fact: at 20 minutes the labels are
mostly microstructure noise; at 120 they are mostly drift. Three nights
of AUC ≈ 0.49 tell us the features do not rank winners AT 60 MINUTES.
They say nothing about 30, or 90.

This tool asks all of them — honestly, which is the entire difficulty.
Trying k horizons and keeping the best is a guaranteed way to find a
0.55 in pure noise. So:

  1. THE LADDER GATES THE SEARCH. core.capability_ladder computes the
     smallest AUC this vault can distinguish from chance. If that number
     is worse than the horizons could plausibly differ by, the sweep
     refuses to run and reports how many more trading days are needed.
     Searching an underpowered vault manufactures discoveries.

  2. EVERY HORIZON IS PRE-REGISTERED. Each candidate is written to the
     trial registry BEFORE its result is known, so the deflated-Sharpe
     machinery already in the repo counts these attempts against every
     later claim. No free rolls.

  3. SIGNIFICANCE IS DAY-CLUSTERED AND FDR-CONTROLLED. Per horizon:
     purged day-fold OOF scores, one-sided AUC>0.5 p-value by day-
     cluster bootstrap, then Benjamini-Hochberg across the k horizons.
     A horizon "wins" only if it survives that correction.

  4. A WINNER MUST REPEAT. Adoption additionally requires the same
     horizon to win on HORIZON_ADOPT_MIN_NIGHTS consecutive nights.
     One night's winner is a coin flip with a p-value attached.

  5. ADOPTION IS AN OVERRIDE FILE, NOT A SOURCE EDIT. The winner is
     written to state/horizon_override.json; config reads it at import,
     so MAX_HOLD_MINUTES changes, CONFIG_HASH rotates, and every cache
     and artifact invalidates through the machinery that already exists.
     Default HORIZON_AUTO_ADOPT=False: the system recommends, the
     operator disposes, until you decide otherwise.

The expected result today is NO WINNER. That is a finding, recorded with
its statistics, and it becomes a null baseline the vault grows against.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("horizon_sweep")
import logging                                             # noqa: E402
log = logging.getLogger("horizon_sweep")

from core import capability_ladder as CL                   # noqa: E402
from core import meta_gbm as MG                            # noqa: E402

STATE = config.STATE_DIR / "horizon_sweep_history.jsonl"
OVERRIDE = config.STATE_DIR / "horizon_override.json"
REPORT = config.LOG_DIR / "horizon_sweep_{d}.json"


def _samples_worker(args):
    """One (day, hold) unit, in its own process.

    v9.9.40: `_samples_at` looped over days SERIALLY, and the loop is run
    once PER HORIZON — the sample cache is keyed on |h{hold}, so every
    horizon regenerates all 40 sessions from scratch. On the 2026-08-16
    chain this step was still walking the vault 26 hours in, at 11-12
    minutes per session, while gate_ab_study and entry_bar_study had
    finished in under two hours each by reading the shared stream.

    The hold override is a MODULE-LEVEL global in nightly_forge_v9, so it
    must be set inside the worker — setting it in the parent does not
    cross a spawn boundary, and a worker that inherited the default would
    silently label at 60 minutes while the caller believed it was
    measuring 20. That is the same class of failure as the day-plan knob
    in cascade_harness: a global that looks set and is not.
    """
    day, hold_min = args
    import sqlite3 as _sq
    import nightly_forge_v9 as F
    con = _sq.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    F.set_hold_override(int(hold_min) * 60)
    try:
        X, y_, w_, _r, _ret, _e = F._gen_meta_samples_cached(con, day)
        if not X:
            return None
        return (day, list(X), list(y_), list(w_))
    except Exception as e:                                 # noqa: BLE001
        log.warning("    %s @ %dm: sample generation failed (%s)", day,
                    hold_min, e)
        return None
    finally:
        # ALWAYS restore, even on error: a leaked override would poison
        # every later call in this process.
        F.set_hold_override(None)
        try:
            con.close()
        except Exception:                                  # noqa: BLE001
            pass


def _samples_at(con, days, hold_min: int):
    """Regenerate the vault's labels at `hold_min` minutes, across the pool.

    Caches are keyed on the override, so this neither reads nor writes the
    production sample caches.
    """
    from core.parallel_days import map_days
    res = map_days(_samples_worker, [(d, int(hold_min)) for d in days],
                   desc=f"horizon {hold_min}m", log_every=5)
    perday, Y, W, SD = [], [], [], []
    for r in res:
        if not r:
            continue
        d, X, y_, w_ = r
        perday.append((d, X, y_, w_))
        Y += list(y_)
        W += list(w_)
        SD += [d] * len(X)
    return perday, np.asarray(Y, float), np.asarray(W, float), \
        np.asarray(SD)


def _evaluate(hold_min: int, perday, Y, W, SD) -> dict | None:
    """Purged day-fold OOF fit at this horizon → AUC, day-clustered p."""
    if len(perday) < 3 or Y.size < config.META_MIN_TRAIN:
        return {"hold_min": hold_min, "n": int(Y.size),
                "days": len(perday), "skipped": "too few samples"}
    with tempfile.TemporaryDirectory() as td:
        oof: dict = {}
        MG.fit_gbm(perday, min_train=min(config.META_MIN_TRAIN, Y.size),
                   oof_out=oof, model_path=Path(td) / "h.txt")
    if "mask" not in oof:
        return {"hold_min": hold_min, "n": int(Y.size),
                "days": len(perday), "skipped": "no OOF produced"}
    m = oof["mask"]
    s, y = np.asarray(oof["oof_raw"], float), np.asarray(oof["y"], float)
    d = SD[m]
    auc, ci_lo, p = CL.cluster_bootstrap_auc_p(
        s, y, d, n_boot=int(getattr(config, "HORIZON_BOOT", 1500)))
    # v9.9.7: PER-DAY AUC as well, so horizons can be compared PAIRED — each
    # day is its own control. The absolute figures stay for context.
    per_day = {}
    for dd in np.unique(d):
        sel = d == dd
        yy = y[sel]
        if yy.size < 4 or yy.min() == yy.max():
            continue
        per_day[str(dd)] = float(_auc_np(s[sel], yy))
    return {"hold_min": hold_min, "n": int(y.size), "days": int(len(perday)),
            "base_rate": round(float(y.mean()), 4),
            "auc": round(float(auc), 4), "auc_ci90_lo": round(float(ci_lo), 4),
            "p_one_sided": round(float(p), 5), "_per_day": per_day}


def _auc_np(scores, y) -> float:
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def _history() -> list[dict]:
    if not STATE.exists():
        return []
    out = []
    for line in STATE.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:                                  # noqa: BLE001
            pass
    return out


def _adopt(hold_min: int) -> None:
    OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"max_hold_minutes": int(hold_min),
                               "adopted_utc": time.time(),
                               "by": "horizon_sweep"}))
    import os
    os.replace(tmp, OVERRIDE)
    log.warning("HORIZON ADOPTED: MAX_HOLD_MINUTES → %d min via %s. "
                "CONFIG_HASH will rotate on next import; every cache and "
                "artifact re-derives, by design. Revert by deleting that "
                "file.", hold_min, OVERRIDE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--adopt", action="store_true",
                    help="permit adoption this run (overrides the config "
                         "default for a manual run)")
    a = ap.parse_args()
    if not bool(getattr(config, "DISCOVERY_ENABLED", True)):
        log.info("discovery disabled by config — nothing to do")
        return 0

    from nightly_forge_v9 import trading_days
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        days = trading_days(con)
        if a.days > 0:
            days = days[-a.days:]
        if len(days) < int(getattr(config, "HORIZON_MIN_DAYS", 20)):
            log.info("vault has %d day(s); the sweep needs %d before its "
                     "answers mean anything. Collect, don't search.",
                     len(days), int(getattr(config, "HORIZON_MIN_DAYS", 20)))
            return 0

        # ---- gate 1: can this vault detect anything at all?
        base_perday, Y0, W0, SD0 = _samples_at(
            con, days, int(config.MAX_HOLD_MINUTES))
        cap = CL.assess(Y0, W0, SD0)
        log.info("LADDER | %s", cap.reason)
        log.info("LADDER | stage %s | %d more trading day(s) to reach "
                 "promotion-grade power at the current sample rate "
                 "(%.0f/day)", cap.stage, cap.days_to_promote_power,
                 cap.samples_per_day)
        # v9.9.7: the horizon question is COMPARATIVE — "is 90 better than
        # 60?" — so it is gated on the stratified resolution, not the
        # pooled one. On 2026-08-04 the pooled gate (0.729) refused work
        # that ab_ablation was resolving to 0.03 the same night.
        if not cap.allows_comparative("SCREEN"):
            log.warning("COMPARATIVE STAGE %s — even a paired, day-matched "
                        "comparison can only resolve %.3f here. Refusing; "
                        "a sweep now would report its own noise floor.",
                        cap.stage_comparative, cap.mde_auc_within)
            _write(days, cap, [], None, "refused: underpowered (comparative "
                   "STAGE " + cap.stage_comparative + ")")
            return 0
        log.info("GATE | comparative stage %s (stratified resolution %.3f) "
                 "— pooled stage is %s (%.3f), which governs ABSOLUTE "
                 "claims only", cap.stage_comparative, cap.mde_auc_within,
                 cap.stage, cap.mde_auc)

        cands = [int(h) for h in getattr(
            config, "HORIZON_CANDIDATES", [20, 30, 45, 60, 90, 120])]
        log.info("sweeping %d horizon(s): %s min | incumbent %d",
                 len(cands), cands, config.MAX_HOLD_MINUTES)

        # ---- gate 2: PRE-REGISTER every attempt, before any result
        try:
            from core.trial_registry import register
            for h in cands:
                register(family="horizon_sweep", spec_id=f"hold_{h}m",
                         kind="pre_registered", n_days=len(days))
        except Exception as e:                             # noqa: BLE001
            log.debug("registry unavailable (%s)", e)

        rows = []
        for h in cands:
            t0 = time.time()
            if h == int(config.MAX_HOLD_MINUTES):
                perday, Y, W, SD = base_perday, Y0, W0, SD0
            else:
                perday, Y, W, SD = _samples_at(con, days, h)
            r = _evaluate(h, perday, Y, W, SD)
            r["secs"] = round(time.time() - t0, 1)
            rows.append(r)
            if r.get("skipped"):
                log.info("  %3d min: %s (n=%d)", h, r["skipped"], r["n"])
            else:
                log.info("  %3d min: n=%-5d base %.3f | AUC %.4f "
                         "(CI90 lo %.4f) | p=%.4f | MDE %.3f | %.0fs",
                         # .get, not []. A per-horizon row that took an
                         # early-return path carries no "mde_auc", and on
                         # 2026-08-14 the KeyError killed the tool after
                         # 33 944s — so the stale a5b65c350f artifact kept
                         # being served into gemma's digest as if current.
                         # A REPORTING line must never be able to destroy
                         # the result it is reporting.
                         h, r.get("n", 0), r.get("base_rate", float("nan")),
                         r.get("auc", float("nan")),
                         r.get("auc_ci90_lo", float("nan")),
                         r.get("p_one_sided", float("nan")),
                         r.get("mde_auc", float("nan")),
                         r.get("secs", 0.0))
    finally:
        con.close()

    # ---- gate 3: FDR across the horizons actually tested
    live = [r for r in rows if not r.get("skipped")]
    winner = None
    inc = next((r for r in live
                if r["hold_min"] == int(config.MAX_HOLD_MINUTES)), None)
    if live and inc:
        # v9.9.7 PAIRED SELECTION: per-day ΔAUC against the INCUMBENT
        # horizon, day-cluster bootstrap + sign-flip permutation, then BH
        # across the challengers. A challenger must beat 60 minutes on the
        # same days — not merely beat chance on its own.
        chall = [r for r in live if r is not inc]
        for r in chall:
            d = {k: r["_per_day"][k] - inc["_per_day"][k]
                 for k in r["_per_day"] if k in inc["_per_day"]}
            st = CL.paired_test(d, n_boot=int(getattr(config,
                                                      "HORIZON_BOOT", 1500)))
            r["delta_vs_incumbent"] = {k: (round(v, 5)
                                           if isinstance(v, float) else v)
                                       for k, v in st.items()
                                       if k != "ci90"}
            r["delta_vs_incumbent"]["ci90"] = st["ci90"]
            log.info("  %3d vs %3d min: ΔAUC %+.4f (CI90 %+.4f..%+.4f, "
                     "p=%.3f, resolvable %.4f over %d day(s))",
                     r["hold_min"], inc["hold_min"], st["mean"],
                     st["ci90"][0], st["ci90"][1], st["p"],
                     st.get("mde", float("nan")), st["n_days"])
        rej, adj = CL.benjamini_hochberg(
            [r["delta_vs_incumbent"]["p"] for r in chall],
            float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
        for r, ok_, q_ in zip(chall, rej, adj):
            r["p_adj_bh"] = round(float(q_), 5)
            r["survives_fdr"] = bool(ok_)
        inc["survives_fdr"] = False
        passing = [r for r in chall if r["survives_fdr"]
                   and r["delta_vs_incumbent"]["mean"] > 0]
        if passing:
            winner = max(passing,
                         key=lambda r: r["delta_vs_incumbent"]["mean"])
            log.info("FDR survivor(s): %s | best %d min AUC %.4f "
                     "(q=%.4f)", [r["hold_min"] for r in passing],
                     winner["hold_min"], winner["auc"], winner["p_adj_bh"])
        else:
            log.info("NO challenger beats the incumbent after BH-FDR "
                     "q=%.2f (bar %.2f). The %d-minute null stands — a "
                     "recorded result, not a failure.",
                     float(getattr(config, "DISCOVERY_FDR_Q", 0.10)),
                     config.META_MIN_AUC, int(config.MAX_HOLD_MINUTES))

    # ---- gate 4: a winner must REPEAT before it may change live behaviour
    verdict = "no winner"
    if winner:
        need = int(getattr(config, "HORIZON_ADOPT_MIN_NIGHTS", 3))
        hist = _history()[-(need - 1):] if need > 1 else []
        concordant = 1 + sum(1 for h in hist
                             if h.get("winner_hold_min") == winner["hold_min"])
        verdict = (f"winner {winner['hold_min']}m, concordant "
                   f"{concordant}/{need} night(s)")
        log.info("STABILITY | %s", verdict)
        allow = (bool(getattr(config, "HORIZON_AUTO_ADOPT", False))
                 or a.adopt)
        lo, hi = getattr(config, "HORIZON_ADOPT_RANGE", (20, 120))
        if concordant >= need and allow and lo <= winner["hold_min"] <= hi:
            _adopt(winner["hold_min"])
            verdict += " → ADOPTED"
        elif concordant >= need:
            log.warning("RECOMMENDATION: adopt MAX_HOLD_MINUTES=%d "
                        "(AUC %.4f, q=%.4f, %d concordant nights). "
                        "Auto-adopt is OFF — set HORIZON_AUTO_ADOPT=True "
                        "or re-run with --adopt to apply.",
                        winner["hold_min"], winner["auc"],
                        winner["p_adj_bh"], concordant)
            verdict += " → recommended (auto-adopt off)"
    _write(days, cap, rows, winner, verdict)
    return 0


def _write(days, cap, rows, winner, verdict) -> None:
    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "days": len(days), "incumbent_hold_min": config.MAX_HOLD_MINUTES,
           "ladder": cap.as_dict() if cap else None,
           "fdr_q": float(getattr(config, "DISCOVERY_FDR_Q", 0.10)),
           "horizons": rows,
           "winner_hold_min": (winner or {}).get("hold_min"),
           "verdict": verdict}
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with STATE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        p = Path(str(REPORT).format(d=time.strftime("%Y-%m-%d")))
        p.write_text(json.dumps(out, indent=1))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)


if __name__ == "__main__":
    raise SystemExit(main())