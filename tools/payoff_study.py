"""
PAYOFF STUDY — is R predictable? Runs itself once the data exists.
==================================================================
    python tools/payoff_study.py [--json out.json]

Wired into run_training.py, after the forge. It needs no operator decision
and no threshold to be flipped by hand: nightly_forge publishes
state/meta_train_matrix.npz on every run, this reads it, and
core/payoff_target.measure() decides whether there is enough evidence to
say anything. Below the bar it prints the countdown and stops. Above it,
it measures, and only if the measurement clears BH + MDE does it fit.

WHY THIS IS NOT WAITING ON THE SHADOW LEDGER
---------------------------------------------
An earlier design had this joined to shadow trades by shadow_id, which
would have meant waiting ~20 sessions for live shadows to accumulate. That
was unnecessary. The forge already grades every replayed signal with a
barrier P&L (`ret`) and the payoff geometry it was graded on (`ECON`:
entry_ask, tp, sl, lot), so R = ret / risk exists for every one of the
~1900 rows across 38 sessions the meta already trains on. Same population,
same features, different target. The matrix costs one file to publish.

The shadow ledger still matters — it is the LIVE cross-check on the same
quantity. A target measured only in replay and never observed forward is
how a backtest becomes a belief. But it is not a prerequisite for asking
the question.

WHAT THIS CAN CONCLUDE
----------------------
"Not predictable" is a real, useful answer and the most likely one. The
equity meta's failure at AUC 0.5210 was not a modelling failure, it was
the sample telling us it cannot resolve a sign at this n_eff. R may be no
different. If it isn't predictable, this prints so, fits nothing, and the
brain keeps running heuristic-only — which is the correct outcome and
costs nothing but the reading.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("payoff_study")
import logging                                             # noqa: E402
log = logging.getLogger("payoff_study")

from core import payoff_target as PT                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    rows, X, names = PT.load_forge_matrix()
    if rows is None:
        log.info("no payoff matrix at %s yet. nightly_forge publishes it on "
                 "every run; this step self-arms the moment it appears — no "
                 "switch to flip.", config.STATE_DIR / "meta_train_matrix.npz")
        return 0

    days = sorted({r.day for r in rows})
    log.info("payoff matrix: %d row(s) x %d feature(s) over %d session(s)",
             len(rows), X.shape[1], len(days))
    log.info("target R = P&L / initial risk. Predicting the MAGNITUDE, not "
             "the sign: at this n_eff the AUC detectability floor is "
             "0.587-0.658 and the meta scored 0.5210, so a sign is not "
             "resolvable here. Dispersion may be.")

    if len(days) < PT.MIN_SESSIONS or len(rows) < PT.MIN_TRADES:
        log.info("EVIDENCE: %d/%d session(s), %d/%d row(s) — %s",
                 len(days), PT.MIN_SESSIONS, len(rows), PT.MIN_TRADES,
                 "not yet measurable" if len(days) < PT.MIN_SESSIONS
                 else "waiting on rows")
        return 0

    m = PT.measure(rows, X, names, target="r_real")
    PT.report(m, log)

    fit = None
    if m.get("predictable"):
        fit = PT.fit_quantiles(rows, X, names, m, target="r_real")
        if fit:
            log.info("─" * 66)
            for q, v in sorted(fit["quantiles"].items()):
                log.info("  q=%s  pinball %.5f vs unconditional %.5f  "
                         "skill %+.4f  %s", q, v["pinball_model"],
                         v["pinball_unconditional"], v["skill"] or 0.0,
                         "BEATS the flat baseline"
                         if v["beats_unconditional"] else
                         "LOSES to predicting one number")
            log.info("Beating the unconditional quantile is the bar a "
                     "conditional model has to clear to be worth serving. "
                     "Nothing is promoted from this — it is a measurement, "
                     "and the EV gate still runs on the incumbent path.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "n_rows": len(rows), "n_days": len(days),
           "measurement": m, "fit": fit}
    p = Path(a.json) if a.json else (
        config.LOG_DIR / f"payoff_study_{time.strftime('%Y-%m-%d')}.json")
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())