"""
LABEL CERTIFICATE — evidence report and issuance
=================================================
    python tools/label_cert_report.py [--issue] [--dry-run]

The shadow book can teach the ENTRY model a better target: instead of
"did the trade as EXITED make money", ask "was there anything here to
collect". 2026-08-10 trade 3 is the case in point — NIFTY 24900CE was
labelled a LOSS at -Rs126.99 because MAX_HOLD_THETA fired at 60 minutes,
while lock_5pct on the identical path came out at -Rs0.07. The incumbent
target charges exit noise to the entry.

Knowing that is not permission to switch. This step runs AFTER the forge,
on purpose:

    forge (run_nightly)  trains under the CURRENTLY certified target and
                         writes its feature matrix
    label_cert (here)    judges whether a different target would have
                         produced a better model, on that same matrix
    next run's forge     reads core.label_certificate.active_label()

The one-run lag is deliberate and removes a circularity: a certificate
issued from historical evidence governs the next training, never the one
that produced it.

WHAT IT PRINTS WHEN THERE IS NOT YET ENOUGH EVIDENCE
----------------------------------------------------
The month countdown, plainly. That is the point of running it every
night from the first night: an operator should be able to see how far
away the question is from being answerable, instead of discovering in
four weeks that half the sessions had unusable coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("label_cert_report")
import logging                                             # noqa: E402
log = logging.getLogger("label_cert_report")

from core import label_certificate as LC                   # noqa: E402
from core.shadow_labels import (read_shadow_ledger, report,  # noqa: E402
                                active_spec, LABEL_SPECS)


def _matrix_path() -> Path:
    """Where the forge parks the matrix it just trained on, if it does."""
    return config.STATE_DIR / "meta_train_matrix.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", action="store_true",
                    help="write the certificate if the evidence earns it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--candidate", default="best_policy",
                    choices=[s for s in LABEL_SPECS if s != "realized"])
    a = ap.parse_args()

    spec, spec_hash = active_spec()
    cur = LC.active_label()
    log.info("training target in force: %r (spec hash %s)", cur, spec_hash)
    if cur != spec:
        log.warning("config.META_LABEL_SPEC says %r but the CERTIFICATE "
                    "says %r — the certificate wins. Editing the config "
                    "string is not authority to change what the model is "
                    "fitted to.", spec, cur)

    trades, stats = read_shadow_ledger()
    report(trades, stats, log)

    # ---- the month countdown, every night, from the first night
    need_s = int(getattr(config, "LABEL_CERT_MIN_SESSIONS", 22))
    need_t = int(getattr(config, "LABEL_CERT_MIN_TRADES", 60))
    days = sorted({t.day for t in trades if t.day})
    log.info("─" * 68)
    log.info("EVIDENCE TOWARD THE LABEL CERTIFICATE")
    log.info("  sessions with usable shadows : %3d / %d   %s",
             len(days), need_s,
             "READY" if len(days) >= need_s else
             f"{need_s - len(days)} more session(s)")
    log.info("  labelled trades              : %3d / %d   %s",
             len(trades), need_t,
             "READY" if len(trades) >= need_t else
             f"{need_t - len(trades)} more trade(s)")
    if stats.get("thin_coverage"):
        log.warning("  %d shadow(s) were BELOW the %.0f%% coverage floor and "
                    "do not count. A shadow whose feed died has a fictional "
                    "MFE; labelling from it would launder the flat-line "
                    "error into the training target.",
                    stats["thin_coverage"],
                    100 * float(getattr(config, "SHADOW_MIN_COVERAGE", 0.6)))
    if stats.get("abandoned"):
        log.warning("  %d shadow(s) were ABANDONED (snapshot went stale "
                    "across a restart) — recorded, not counted.",
                    stats["abandoned"])
    log.info("─" * 68)

    # ---- the evaluation itself
    # ---- THE FEATURE MATRIX FOR A LABEL COMPARISON.
    # An earlier version joined state/meta_train_matrix.npz on shadow_id and
    # failed with "shadow_id is not a file in the archive". That was not a
    # missing key — it was a category error. The payoff matrix holds REPLAYED
    # SIGNALS from _gen_meta_samples (1953 rows over 31 sessions); the
    # labelled trades are LIVE FILLS from the shadow ledger. Different
    # populations, no correspondence, and no shadow_id could ever exist on
    # the replay side. Forcing a join would have silently compared a target
    # on rows that were never traded.
    #
    # The label question only needs features for the FILLS, and the ledger
    # already records what the gate saw when it decided: conviction, the
    # served win probability, the hold, and the hour. A small, honest
    # feature set on the right population beats a large one on the wrong
    # population.
    X = None
    if trades:
        import numpy as np
        import datetime as _dt
        rows = []
        for t_ in trades:
            ets = float(getattr(t_, "entry_ts", 0.0) or 0.0)
            lt = _dt.datetime.fromtimestamp(ets) if ets else None
            rows.append([
                float(getattr(t_, "conviction", 0.0) or 0.0),
                abs(float(getattr(t_, "conviction", 0.0) or 0.0)),
                float(getattr(t_, "win_prob", 0.0) or 0.0),
                float(getattr(t_, "coverage", 0.0) or 0.0),
                float((lt.hour * 60 + lt.minute) if lt else 0.0),
            ])
        X = np.asarray(rows, float)
        log.info("feature matrix for the label comparison: %d fill(s) x %d "
                 "column(s) [conv, |conv|, win_prob, coverage, minute] — "
                 "built from the LEDGER, which is the population the labels "
                 "describe. state/meta_train_matrix.npz is replayed signals "
                 "and is deliberately NOT joined.", X.shape[0], X.shape[1])

    v = LC.evaluate(trades, candidate=a.candidate, X=X)
    LC.report(v, log)
    if a.issue or a.dry_run:
        LC.issue(v, dry_run=a.dry_run)

    try:
        out = config.LOG_DIR / "label_cert_report.json"
        out.write_text(json.dumps(
            {"active_label": cur, "spec_hash": spec_hash,
             "evidence": {"sessions": len(days), "trades": len(trades),
                          "need_sessions": need_s, "need_trades": need_t},
             "ledger_stats": stats, "verdict": v},
            indent=1, default=float), encoding="utf-8")
        log.info("report → %s", out)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())