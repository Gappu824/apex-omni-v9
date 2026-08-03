"""
PRIME DAY CACHES — pay the replay cost ONCE, in parallel, up front
==================================================================
    python tools/prime_day_caches.py [--days N] [--workers W]

Every evening tool sits on the same raw day caches (obs arrays + prem
table). Before v9.9.2 the first tool to touch a stale day paid a full
sqlite replay inline — serially, mid-analysis, sometimes more than once
across tools racing each other. This tool makes cache-building an
explicit stage: enumerate stale days, rebuild them across cores with
atomic publication, report the bill. Every tool after it cache-hits.

Zero new build logic — it calls the forge's own `_build_and_cache`
(Windows-spawn-safe, own sqlite per worker) through the repo's
`map_days` pool. Correctness is the forge's; this is pure scheduling.
"""
from __future__ import annotations

import argparse
import functools
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

# Windows spawn re-imports this module in every worker, which would reprint
# the start banner once per core. Workers still get full logging (their
# forge progress lines are the useful ones) — only the banner is suppressed.
import multiprocessing as _mp                              # noqa: E402
config.setup_logging("prime_day_caches"
                     if _mp.current_process().name == "MainProcess"
                     else "prime_worker")
import logging                                             # noqa: E402
log = logging.getLogger("prime_day_caches")


def _sweep_temp_debris() -> int:
    """Remove half-written cache temps. Two patterns matter: the current
    `*.<pid>.tmp.*` scheme (only ever left behind by a killed process) and
    the 2026-07-30 defect's `*.npz.tmp.npz` orphans, which np.savez wrote
    under a name os.replace then could not find. Neither is ever read —
    freshness is decided by the stamp file — so removal is always safe."""
    from nightly_forge_v9 import _CACHE_DIR
    n = 0
    for pat in ("*.tmp.npz", "*.tmp.pkl", "*.tmp.json", "*.npz.tmp.npz",
                "*.npz.tmp", "*.pkl.tmp", "*.json.tmp"):
        for f in _CACHE_DIR.glob(pat):
            try:
                sz = f.stat().st_size
                f.unlink()
                n += 1
                log.info("  swept %s (%.1f MB)", f.name, sz / 1e6)
            except OSError as e:
                log.warning("  could not remove %s (%s)", f.name, e)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0,
                    help="limit to the most recent N days (0 = all)")
    ap.add_argument("--workers", type=int, default=0,
                    help="pool size (0 = parallel_days default)")
    a = ap.parse_args()
    from nightly_forge_v9 import (_build_and_cache, _cache_fresh,
                                  _data_stamp, trading_days)
    from core.parallel_days import map_days
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        days = trading_days(con)
    finally:
        con.close()
    if a.days > 0:
        days = days[-a.days:]
    swept = _sweep_temp_debris()
    if swept:
        log.info("swept %d orphaned cache temp file(s) before building", swept)
    stale = [d for d in days if not _cache_fresh(d)]
    log.info("day caches | %d day(s) in vault | %d stale | stamp %s",
             len(days), len(stale), _data_stamp())
    if not stale:
        log.info("nothing to build — every downstream tool will cache-hit")
        return 0
    log.info("stale days: %s%s", ", ".join(stale[:8]),
             f" … +{len(stale) - 8} more" if len(stale) > 8 else "")
    log.info("building now — each day is a full tick replay (~minutes); "
             "one progress line per completed day with elapsed/ETA follows")
    t0 = time.time()
    fn = functools.partial(_build_and_cache, str(config.DB_PATH))
    res = map_days(fn, stale, workers=(a.workers or None),
                   desc="prime day cache", log_every=1)
    built = sum(1 for r in res if r and r[1])
    empty = sum(1 for r in res if r and not r[1])
    slowest = sorted((r for r in res if r), key=lambda r: -r[2])[:3]
    log.info("primed %d built / %d empty in %.0fs | slowest: %s",
             built, empty, time.time() - t0,
             ", ".join(f"{d} {t:.0f}s" for d, _, t in slowest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())