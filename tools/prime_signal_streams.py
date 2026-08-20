"""
PRIME SIGNAL STREAMS — pay for the replay once, in the pool, up front
======================================================================
core/signal_stream caches each session's per-second decision loop so the
studies that follow do not each pay for it. But "cached" means "whoever
gets there first builds it", and that is an ordering accident: on
2026-08-14 entry_bar_study would have built all 40 streams inside its own
16.5-hour wall while gate_ab_study ran beside it, re-deriving the same
answer because the cache was not populated yet.

This step makes the cost explicit and parallel. It runs in the `prime`
group — before `discovery` — so every study downstream is a cache hit,
and the build happens across the worker pool instead of inside one
study's serial section.

Idempotent: a session whose stream matches the current stamp is skipped,
so a re-run after a partial evening costs seconds. The stamp binds
CONFIG_HASH, the data stamp, the decision stamp and the stream schema —
see core/signal_stream for why each is necessary.
"""
from __future__ import annotations
import argparse, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
config.setup_logging("prime_signal_streams")
import logging
log = logging.getLogger("prime_signal_streams")
from core import signal_stream as SS


def _worker(args):
    day, = args
    import sqlite3 as _sq
    from nightly_forge_v9 import _eval_meta, _eval_cal
    from core.heuristic_policy import HeuristicPolicy
    con = _sq.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        meta, cal = _eval_meta(), _eval_cal()
        pol = HeuristicPolicy()

        def decide(obs, frame, iidx):
            return float(pol.predict(frame)[2 * iidx])

        def actions_fn(_o, _f):
            return pol.predict(_f)

        s, _rep = SS.build(con, day, decide, meta, cal, actions_fn)
        return int(len(s)) if s is not None else 0
    except Exception as e:                                 # noqa: BLE001
        log.warning("  %s failed (%s)", day, e)
        return None
    finally:
        try:
            con.close()
        except Exception:                                  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()

    SS.purge_stale()
    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        days = sorted({r[0] for r in con.execute(
            "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch','localtime') "
            "FROM ticks_v9")})
    finally:
        con.close()
    if a.days > 0:
        days = days[-a.days:]

    todo = [d for d in days if SS.load(d) is None]
    log.info("signal streams | %d session(s) in vault | %d already cached | "
             "%d to build", len(days), len(days) - len(todo), len(todo))
    if not todo:
        log.info("nothing to build — every discovery step will cache-hit "
                 "the decision loop")
        return 0

    t0 = time.time()
    from core.parallel_days import map_days
    res = map_days(_worker, [(d,) for d in todo],
                   desc="signal stream", log_every=2)
    ok = [r for r in res if r]
    log.info("built %d/%d stream(s) in %.1f min | %d signal(s) total. Every "
             "study in discovery now reads these instead of re-running the "
             "per-second decision loop.", len(ok), len(todo),
             (time.time() - t0) / 60.0, sum(ok))
    if len(ok) < len(todo):
        log.warning("%d session(s) produced no stream — those will fall back "
                    "to a full replay in whichever study reaches them",
                    len(todo) - len(ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())