"""
EPISODE STUDY — can a within-session ranker beat picking at random?
====================================================================
    python tools/episode_study.py [--json out.json]

Reads the payoff matrix nightly_forge publishes, collapses it to
NON-OVERLAPPING episodes, and asks the only question the book can act on:
given today's candidates and ONE slot, can the model order them?

This is not a fifth classifier. It changes the unit (episodes, not
seconds), the target (R = P&L/risk, not win/lose) and the question
(rank within a session, not calibrate across them) — each forced by a
measured finding, all three documented in core/episode_ranker.py.

Self-arming: it runs whenever the matrix exists and states the countdown
when the sample is short. Nothing it produces is promoted automatically;
EPISODE_RANKER_ENABLED is a separate, operator-owned switch.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
config.setup_logging("episode_study")
import logging
log = logging.getLogger("episode_study")
from core import episode_ranker as ER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    mp = config.STATE_DIR / "meta_train_matrix.npz"
    if not mp.exists():
        log.info("no payoff matrix at %s — nightly_forge publishes it on "
                 "every run; this self-arms when it appears", mp)
        return 0
    try:
        z = np.load(mp, allow_pickle=False)
        ch = str(z["config_hash"][0]) if "config_hash" in z.files else ""
        if ch and ch != config.CONFIG_HASH:
            log.warning("matrix built under CONFIG_HASH %s != %s — a matrix "
                        "from another feature world is not evidence about "
                        "this one. Stopping.", ch, config.CONFIG_HASH)
            return 0
        X = np.asarray(z["X"], float)
        ret = np.asarray(z["ret"], float)
        risk = np.asarray(z["risk"], float)
        days = np.array([str(d) for d in z["day"]])
        ts = (np.asarray(z["t"], float) if "t" in z.files
              else np.arange(len(ret), dtype=float))
        if "t" not in z.files:
            log.warning("the matrix carries no per-row session-second, so "
                        "episode boundaries fall back to row order. That is "
                        "weaker than a real clock: add `t` to the forge's "
                        "publication for exact non-overlap.")
    except Exception as e:                                 # noqa: BLE001
        log.error("could not read the payoff matrix (%s)", e)
        return 0

    keep = risk > 0
    r = np.where(keep, ret / np.where(risk > 0, risk, 1.0), np.nan)
    eps, st = ER.build_episodes(days[keep], ts[keep], r[keep], X[keep])

    log.info("=" * 68)
    log.info("EPISODE RANKER — unit: episode | target: R = P&L/risk | "
             "question: rank WITHIN the session")
    log.info("  the book has ONE slot and ~5 per day, so it needs to know "
             "WHICH candidate to spend it on, not E[R] in the abstract")
    log.info("=" * 68)

    n_days = len({e.day for e in eps})
    if n_days < ER.MIN_SESSIONS or len(eps) < ER.MIN_EPISODES:
        log.info("EVIDENCE: %d/%d session(s), %d/%d episode(s) — not yet "
                 "measurable. Episodes accumulate ~4-6 per session, so this "
                 "is a matter of weeks, not of a better model.",
                 n_days, ER.MIN_SESSIONS, len(eps), ER.MIN_EPISODES)
        return 0

    cv = ER.cross_validate(eps)
    v = ER.evaluate(cv)
    ER.report(v, log)
    if v.get("ok"):
        log.info("CLEARS EVERY GATE. Not armed: EPISODE_RANKER_ENABLED is "
                 "an operator decision, and a first clearance on one vault "
                 "is exactly when a holdout month is worth more than a "
                 "deployment.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "episodes": len(eps), "sessions": n_days,
           "build_stats": st, "verdict": v}
    p = Path(a.json) if a.json else (
        config.LOG_DIR / f"episode_study_{time.strftime('%Y-%m-%d')}.json")
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())