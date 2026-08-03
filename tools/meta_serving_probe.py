"""
APEX OMNI — META SERVING PROBE (why does live P(win) never move?)
=================================================================
CONFIRMED SYMPTOM (2026-07-24 session, from your own artifacts):

    brain_report winprob_at_gate : NIFTY p50=p90=p95=p99=max = 0.1384
                                   SENSEX  ... = 0.1384   (n = 5,954 each)
    brain log P(win) prints      : 1,853 of 1,853 read exactly 0.50
                                   (0.50 = META_P_FLOOR, i.e. the TRUE value
                                    was below the floor every single time)
    forge_report same model      : oof_spread_p05_p95 = 0.5058
                                   auc_cal            = 0.5907  (z ~ 2.8)

The forge says this model varies and ranks. Production says it emits ONE
number, identical on two different indices whose x-vectors read different
frame nodes. Both cannot be true of the same code path, so one of two things
is wrong — and they demand opposite fixes:

    (A) THE MODEL/SERVING MATH  -> the artifact cannot vary at serving
    (B) THE LIVE INPUT          -> the model is fine; live x-vectors are
                                   near-identical because live frame features
                                   are dead (the 2026-07-20 drift table showed
                                   11 of 15 features with live_std = 0.0000)

Three candidate mechanisms for (A) were tested and FALSIFIED — isotonic
domain clamping, a degenerate final model on no-signal data, and AFML
uniqueness weights (~0.2) starving LightGBM splits. All three still produced
100+ distinct served values. That points hard at (B), but pointing is not
proving, so this tool settles it with YOUR model and YOUR vault.

WHAT IT DOES
------------
Scores the PROMOTED artifact through `meta_gbm.score_vec` — the exact function
apex_main serves with — on x-vectors built by the forge's own
`_gen_meta_samples` from replayed vault ticks.

    varied output  => the model and serving math are healthy. The constant is
                      caused by LIVE FRAME FEATURES being dead. Fix the
                      feature path; the meta is not the problem.
    constant output=> the artifact genuinely cannot vary at serving, and
                      oof_spread (measured on FOLD models) is not describing
                      the model that actually gets served. Fix the forge.

It also prints the per-feature standard deviation of the training x-vectors,
so you can see which of the 61 inputs carry variance at all — the same list
the live path must reproduce.

Read-only: loads a model, scores vectors, prints. No writes, no config change.

  python tools/meta_serving_probe.py --days 5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402
from nightly_forge_v9 import (_gen_meta_samples_cached,      # noqa: E402
                              trading_days)
from core import meta_gbm as MG                            # noqa: E402

config.setup_logging("serving_probe")
import logging                                             # noqa: E402
log = logging.getLogger("serving_probe")

_NAMES = ([f"spot[{i}]" for i in range(config.FEATURES_PER_NODE)]
          + [f"ce[{i}]" for i in range(config.FEATURES_PER_NODE)]
          + [f"pe[{i}]" for i in range(config.FEATURES_PER_NODE)]
          + ["t_frac", "kaufman_er", "mom30_capped", "direction"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    a = ap.parse_args()

    path = config.META_MODEL_PATH
    if not path.exists():
        log.error("no promoted meta at %s — nothing to probe", path)
        return
    meta = json.loads(path.read_text())
    log.info("probing artifact: engine=%s n=%s auc_cal=%s oof_spread=%s",
             meta.get("engine"), meta.get("n"), meta.get("auc_cal"),
             meta.get("oof_spread_p05_p95"))
    if meta.get("engine") != "gbm":
        log.warning("artifact is '%s', not gbm — this probe targets the gbm "
                    "serving path", meta.get("engine"))

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-a.days:]
    log.info("replaying %d day(s) %s->%s for forge-built x-vectors",
             len(days), days[0] if days else "-", days[-1] if days else "-")
    # _gen_meta_samples takes ONE day and returns (X, Y, W, R, RET); the cached
    # wrapper reuses the forge's own npz cache, so this is fast and is exactly
    # the sample set the nightly promotion trained on.
    X = []
    for d in days:
        try:
            xs, _ys, _ws, _r, _ret, _e = _gen_meta_samples_cached(con, d)
        except Exception as e:                             # noqa: BLE001
            log.warning("  %s: sample generation failed (%s) — skipped", d, e)
            continue
        X += list(xs)
        log.info("  %s: %d sample(s)", d, len(xs))
    if len(X) < 50:
        log.error("only %d samples — widen --days", len(X))
        return

    MG._BOOSTERS.clear()          # force a fresh load of the promoted booster
    served = [MG.score_vec(meta, x, clamp=False) for x in X]
    served = [s for s in served if s is not None]
    if not served:
        log.error("score_vec returned None for every vector — the booster file "
                  "referenced by the artifact is missing or unreadable")
        return
    arr = np.asarray(served, float)
    distinct = len(set(np.round(arr, 6)))
    spread = float(np.quantile(arr, 0.95) - np.quantile(arr, 0.05))

    log.info("")
    log.info("SERVED on %d forge-built vectors: min %.6f  max %.6f",
             arr.size, arr.min(), arr.max())
    log.info("  distinct values %d | p05-p95 spread %.6f", distinct, spread)
    log.info("  clears META_ENTRY_P_BAR (%.2f): %d of %d (%.2f%%)",
             config.META_ENTRY_P_BAR,
             int((arr >= config.META_ENTRY_P_BAR).sum()), arr.size,
             100.0 * float((arr >= config.META_ENTRY_P_BAR).mean()))
    log.info("")

    # CEILING CHECK — added 2026-07-25 after the first real run. Reporting the
    # spread alone let a bigger fact hide: if the model's MAXIMUM output on
    # healthy inputs sits below META_ENTRY_P_BAR, the gate can never open no
    # matter how good the features are, and fixing any feature skew buys
    # exactly zero trades. That has to be stated, not left to be inferred.
    _bar = float(config.META_ENTRY_P_BAR)
    _floor = float(config.META_P_FLOOR)
    _be = 1.0 / (1.0 + config.BASE_TP_PCT / max(config.BASE_SL_PCT, 1e-9))
    if arr.max() < _bar:
        log.warning("GATE UNREACHABLE: the model's MAXIMUM on real inputs is "
                    "%.4f, below META_ENTRY_P_BAR %.2f. The gate cannot open "
                    "for ANY signal — it is decorative, not selective. Note "
                    "also max %.4f < META_P_FLOOR %.2f, so every live P(win) "
                    "print clamps to %.2f even with perfectly healthy "
                    "features: a clamped log line is NOT evidence of feature "
                    "skew.", arr.max(), _bar, arr.max(), _floor, _floor)
        _above_be = int((arr >= _be).sum())
        if _above_be:
            log.warning("  BUT %d of %d (%.1f%%) score at or above the "
                        "BREAK-EVEN probability %.2f (payoff b=%.2f). Those "
                        "signals are rated positive-expectancy by the model "
                        "yet blocked by the bar. tools/meta_lift.py measures "
                        "whether that slice actually pays — that is the "
                        "decisive test, not the bar's value.",
                        _above_be, arr.size, 100.0 * _above_be / arr.size,
                        _be, config.BASE_TP_PCT / config.BASE_SL_PCT)
        else:
            log.warning("  and none reach break-even %.2f either — the model "
                        "rates every signal as negative-expectancy.", _be)

    if distinct <= 2:
        log.warning("VERDICT (A): the artifact is CONSTANT at serving even on "
                    "forge-built inputs. oof_spread=%s is describing the "
                    "cross-validation FOLD models, not the final model that "
                    "gets served — the forge is reporting a variability the "
                    "served model does not have. Fix belongs in the forge, "
                    "not the feature path.", meta.get("oof_spread_p05_p95"))
    else:
        log.warning("VERDICT (B): the artifact VARIES on forge-built inputs "
                    "(%d distinct, spread %.4f) — the model and serving math "
                    "are healthy. Production emitting a single constant "
                    "therefore means the LIVE x-vectors are near-identical, "
                    "i.e. live frame features are dead where replay features "
                    "are alive. That is a train/serve FEATURE skew: the forge "
                    "learns on inputs the live brain never reproduces.",
                    distinct, spread)

    Xa = np.asarray(X, float)
    sd = Xa.std(axis=0)
    dead = int((sd < 1e-9).sum())
    log.info("")
    log.info("x-vector variance in TRAINING (61 features): %d carry no "
             "variance at all", dead)
    order = np.argsort(-sd)
    log.info("  most variable inputs the LIVE path must also reproduce:")
    for j in order[:10]:
        log.info("    %-14s sd %.6f", _NAMES[j], sd[j])
    if dead:
        log.info("  zero-variance in training: %s",
                 ", ".join(_NAMES[j] for j in np.nonzero(sd < 1e-9)[0][:12]))
    log.info("")
    log.info("NEXT: if VERDICT (B), compare this list against the live drift "
             "table (brain log, 'drift detail') — any feature with live_std "
             "0.0000 but training sd above is a dead input feeding the gate.")


if __name__ == "__main__":
    main()