"""
APEX OMNI — DAY-PARALLEL EXECUTION
==================================
MEASURED PROBLEM (2026-07-28 cascade_harness):

    21:03 start, 30 days queued
    07:33 next morning, still on 2026-06-30 (day 11)
    => ~57 minutes PER DAY, 10.5 hours and unfinished

Two facts make this fixable, and one makes it less alarming than it looks.

  * It is a ONE-TIME rebuild. cascade_knob_hash() begins with
    config.CONFIG_HASH, so moving the hash (TRADABLE gained BANKNIFTY)
    invalidated every cached day at once. Steady-state nights serve from
    core/day_cache and are fast. This is the price of a correct hash move, not
    a permanent regression.
  * The days are INDEPENDENT. cascade_harness threads `primary_rows` through
    _run_day, which looks like a sequential dependency — it is not. Line 252
    uses it only to decide whether to LOG the event. Nothing about day N's
    result depends on day N-1.

WHY NOT THE GPU
---------------
The RTX 4060 cannot help this workload and saying otherwise would be theatre.
The cost is a Python loop over ~22,500 seconds x N indices, updating a
sequential StateBuilder, on top of SQLite reads. It is control-flow and I/O
bound, not matrix-math bound. The one genuinely GPU-shaped step — fitting
LightGBM on ~1000 x 64 — already takes milliseconds. Moving this to a GPU
would cost more in transfer and rewrite risk than it could ever return.

The i7-13650HX, on the other hand, has 14 cores / 20 threads sitting idle
while one core grinds through 30 sequential days. THAT is the free lunch.

WHAT THIS PROVIDES
------------------
`map_days(fn, days, workers)` runs an independent per-day function across
processes and returns results IN DAY ORDER, so downstream aggregation is
byte-identical to the sequential path. Each worker opens its own SQLite
connection (handles are not shareable across processes), so `fn` must take a
day string and do its own connecting.

Falls back to a plain sequential loop when workers <= 1, when the platform
cannot fork/spawn cleanly, or when anything at all goes wrong — a parallel
speed-up must never become a correctness risk.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

import logging

log = logging.getLogger("parallel")


def default_workers() -> int:
    """Conservative worker count.

    Each worker holds one day of tick arrays (hundreds of MB), so this is
    memory-bound before it is CPU-bound. Half the logical cores, capped at 6,
    keeps headroom for the OS and for the live processes if this ever runs
    during a session.
    """
    n = int(getattr(config, "PARALLEL_DAY_WORKERS", 0) or 0)
    if n > 0:
        return n
    cpu = os.cpu_count() or 2
    return max(1, min(6, cpu // 2))


def map_days(fn, days: list, workers: int | None = None,
             desc: str = "days") -> list:
    """Run fn(day) for each day, in parallel, returning results in DAY ORDER.

    fn MUST be a module-level function (picklable) that opens whatever
    resources it needs — SQLite connections cannot cross a process boundary.
    """
    days = list(days)
    if not days:
        return []
    w = int(workers if workers is not None else default_workers())
    w = max(1, min(w, len(days)))
    if w <= 1:
        return [fn(d) for d in days]

    out: dict = {}
    try:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(fn, d): d for d in days}
            done = 0
            for fu in as_completed(futs):
                d = futs[fu]
                try:
                    out[d] = fu.result()
                except Exception as e:                     # noqa: BLE001
                    log.warning("  %s: worker failed (%s) — day skipped", d, e)
                    out[d] = None
                done += 1
                if done % 5 == 0 or done == len(days):
                    log.info("  %s: %d/%d %s complete", desc, done, len(days),
                             desc)
    except Exception as e:                                 # noqa: BLE001
        # spawn/fork trouble, pickling trouble, anything — correctness first
        log.warning("parallel execution unavailable (%s) — running "
                    "sequentially", e)
        return [fn(d) for d in days]
    return [out.get(d) for d in days]