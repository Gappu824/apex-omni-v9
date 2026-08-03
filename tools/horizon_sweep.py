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


def _samples_at(con, days, hold_min: int):
    """Regenerate the vault's labels at `hold_min` minutes. Caches are
    keyed on the override, so this neither reads nor writes production
    sample caches."""
    import nightly_forge_v9 as F
    F.set_hold_override(int(hold_min) * 60)
    try:
        perday, Y, W, SD = [], [], [], []
        for d in days:
            try:
                X, y_, w_, _r, _ret, _e = F._gen_meta_samples_cached(con, d)
            except Exception as e:                         # noqa: BLE001
                log.warning("    %s: sample generation failed (%s)", d, e)
                continue
            if not X:
                continue
            perday.append((d, X, y_, w_))
            Y += list(y_); W += list(w_); SD += [d] * len(X)
        return perday, np.asarray(Y, float), np.asarray(W, float), \
            np.asarray(SD)
    finally:
        F.set_hold_override(None)          # ALWAYS restore, even on error


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
    cap = CL.assess(y, W[m], d)
    return {"hold_min": hold_min, "n": int(y.size), "days": int(len(perday)),
            "base_rate": round(float(y.mean()), 4),
            "auc": round(float(auc), 4), "auc_ci90_lo": round(float(ci_lo), 4),
            "p_one_sided": round(float(p), 5),
            "mde_auc": round(cap.mde_auc, 4), "stage": cap.stage}


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
        if not cap.allows("SCREEN"):
            log.warning("STAGE %s — the smallest AUC this vault can "
                        "separate from chance is %.3f. Sweeping horizons "
                        "now would report its own noise floor. Refusing; "
                        "re-run when the vault is ~%d day(s) deeper.",
                        cap.stage, cap.mde_auc, cap.days_to_promote_power)
            _write(days, cap, [], None, "refused: underpowered (STAGE "
                   + cap.stage + ")")
            return 0

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
                         h, r["n"], r["base_rate"], r["auc"],
                         r["auc_ci90_lo"], r["p_one_sided"], r["mde_auc"],
                         r["secs"])
    finally:
        con.close()

    # ---- gate 3: FDR across the horizons actually tested
    live = [r for r in rows if not r.get("skipped")]
    winner = None
    if live:
        rej, adj = CL.benjamini_hochberg(
            [r["p_one_sided"] for r in live],
            float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
        for r, ok_, q_ in zip(live, rej, adj):
            r["p_adj_bh"] = round(float(q_), 5)
            r["survives_fdr"] = bool(ok_)
        passing = [r for r in live if r["survives_fdr"]
                   and r["auc"] >= config.META_MIN_AUC]
        if passing:
            winner = max(passing, key=lambda r: r["auc"])
            log.info("FDR survivor(s): %s | best %d min AUC %.4f "
                     "(q=%.4f)", [r["hold_min"] for r in passing],
                     winner["hold_min"], winner["auc"], winner["p_adj_bh"])
        else:
            log.info("NO horizon clears BH-FDR q=%.2f AND the promotion "
                     "bar %.2f. The 60-minute null stands — this is a "
                     "recorded result, not a failure.",
                     float(getattr(config, "DISCOVERY_FDR_Q", 0.10)),
                     config.META_MIN_AUC)

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