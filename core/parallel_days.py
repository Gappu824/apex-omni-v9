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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

import logging

log = logging.getLogger("parallel")


def _available_ram_gb() -> float | None:
    """Physically available RAM, or None if it cannot be measured.
    psutil if present; otherwise the OS's own API (Win32
    GlobalMemoryStatusEx / Linux MemAvailable). Never raises."""
    try:
        import psutil                                       # noqa: PLC0415
        return psutil.virtual_memory().available / 1e9
    except Exception:                                       # noqa: BLE001
        pass
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullAvailPhys / 1e9
        else:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:                                       # noqa: BLE001
        pass
    return None


def _measured_workset_gb() -> float | None:
    """Estimate one worker's RSS from THIS vault's own day caches: the
    median built cache is a direct proxy for the arrays a replay holds,
    scaled by PARALLEL_RAM_WORKSET_MULT for the build-time working set.
    None until at least three days are cached (first-ever prime)."""
    try:
        d = config.DATA_DIR / "forge_cache"
        sizes = sorted(f.stat().st_size for f in d.glob("*.npz")
                       if ".tmp." not in f.name
                       and "meta_samples" not in f.name)
        if len(sizes) < 3:
            return None
        med = sizes[len(sizes) // 2] / 1e9
        return max(0.2, med * float(getattr(
            config, "PARALLEL_RAM_WORKSET_MULT", 3.0)))
    except Exception:                                       # noqa: BLE001
        return None


_EXPLAINED = False


def default_workers() -> int:
    """Conservative worker count.

    Each worker holds one day of tick arrays (hundreds of MB), so this is
    memory-bound before it is CPU-bound. Half the logical cores, capped at 6,
    keeps headroom for the OS and for the live processes if this ever runs
    during a session.
    """
    global _EXPLAINED
    n = int(getattr(config, "PARALLEL_DAY_WORKERS", 0) or 0)
    if n > 0:
        if not _EXPLAINED:
            log.info("  pool sizing: %d worker(s) — pinned by "
                     "config.PARALLEL_DAY_WORKERS", n)
            _EXPLAINED = True
        return n
    cpu = os.cpu_count() or 2
    base = max(1, min(6, cpu // 2))
    why = f"{cpu} logical core(s) → {base}"
    # ---- v9.9.3: RAM is the binding constraint on a 16 GB laptop, not CPU.
    if bool(getattr(config, "PARALLEL_RAM_AWARE", True)):
        avail = _available_ram_gb()
        if avail is not None:
            per = (_measured_workset_gb()
                   or float(getattr(config,
                                    "PARALLEL_RAM_PER_WORKER_GB", 1.5)))
            src = ("measured from day caches" if _measured_workset_gb()
                   else "config estimate")
            usable = avail - float(getattr(config,
                                           "PARALLEL_RAM_RESERVE_GB", 4.0))
            ram_cap = max(1, int(usable / max(per, 0.1)))
            # v9.9.3: grant the configured minimum whenever it PHYSICALLY
            # fits in available-minus-nothing (reserve already too tight ⇒
            # min workers only when truly affordable). 1 worker = a serial
            # night; 2 is the floor of usefulness.
            _minw = int(getattr(config, "PARALLEL_MIN_WORKERS", 2))
            if (ram_cap < _minw
                    and avail >= _minw * per + 1.0):
                ram_cap = _minw
            if ram_cap < base:
                why = (f"{avail:.1f} GB free − "
                       f"{getattr(config, 'PARALLEL_RAM_RESERVE_GB', 4.0)}"
                       f" GB reserve ÷ {per:.2f} GB/worker ({src}) → "
                       f"{ram_cap} (RAM-bound; {cpu} cores would allow "
                       f"{base})")
            base = min(base, ram_cap)
    # v9.9.3 NESTED-PARALLELISM CAP. run_evening runs several tools at once,
    # and each tool used to open its OWN full-size day pool — 3 tools x 6
    # workers = 18 processes, each holding a day of tick arrays. On a
    # laptop that is RAM thrash, not throughput. The evening scheduler now
    # publishes each child's share in APEX_INNER_WORKERS, so the TOTAL
    # number of heavy day-workers alive at once stays at `base` no matter
    # how many tools are in flight. Unset (tool run by hand, or a solo
    # stage like prime/forge) ⇒ the full budget, exactly as before.
    share = os.environ.get("APEX_INNER_WORKERS")
    out = base
    if share:
        try:
            out = max(1, min(base, int(share)))
            why += f"; scheduler share {share} → {out}"
        except ValueError:
            pass
    if not _EXPLAINED:
        log.info("  pool sizing: %d worker(s) | %s", out, why)
        _EXPLAINED = True
    return out


def map_days(fn, days: list, workers: int | None = None,
             desc: str = "days", log_every: int = 5) -> list:
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
    _t0 = time.time()
    log.info("  %s: %d day(s) across %d worker(s) — first completions "
             "arrive after roughly one full day's build time", desc,
             len(days), w)
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
                if (done % max(log_every, 1) == 0 or done == len(days)):
                    _el = time.time() - _t0
                    _eta = _el / done * (len(days) - done)
                    log.info("  %s: %d/%d done | %s | %.1f min elapsed | "
                             "~%.1f min left", desc, done, len(days), d,
                             _el / 60, _eta / 60)
    except Exception as e:                                 # noqa: BLE001
        # spawn/fork trouble, pickling trouble, anything — correctness first
        log.warning("parallel execution unavailable (%s) — running "
                    "sequentially", e)
        return [fn(d) for d in days]
    return [out.get(d) for d in days]