"""
APEX OMNI v10.2 — INCREMENTAL DAY CACHE (evenings stay O(new days) forever)
===========================================================================
A harness grading of one vault day is DETERMINISTIC given (engine family,
knob hash, day): same code, same knobs, same archived ticks → same event
rows. So grade each day ONCE, persist the rows, and let every future evening
assemble certificates from cache + only the genuinely new days. With months
or years of vault history the nightly cost stays constant — O(days added
since yesterday) — instead of growing linearly for the rest of the system's
life.

Invalidation is automatic and honest: the cache key IS the knob hash (which
already fingerprints every certifiable parameter) plus a CACHE_VER bump for
any grading-code change. Change a knob or the doctrine → that family's cache
misses → full regrade, exactly as correctness demands. Atomic writes;
corrupt/partial files read as misses, never as data.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

CACHE_VER = "dc1"
_ROOT = config.STATE_DIR / "harness_cache"


def _path(family: str, stamp: str, day: str) -> Path:
    d = _ROOT / family / f"{stamp}_{CACHE_VER}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day}.json"


def get(family: str, stamp: str, day: str):
    """Cached (closes, skips, blockers) for the day, or None on miss."""
    p = _path(family, stamp, day)
    if not p.exists():
        return None
    try:
        z = json.loads(p.read_text(encoding="utf-8"))
        return z["closes"], z["skips"], z["blockers"]
    except Exception:                                     # noqa: BLE001
        return None


def put(family: str, stamp: str, day: str, closes, skips, blockers) -> None:
    p = _path(family, stamp, day)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"closes": closes, "skips": skips,
                               "blockers": blockers}), encoding="utf-8")
    tmp.replace(p)


def run_cached(family: str, stamp: str, day: str, fn):
    """fn() → (closes, skips, blockers); served from cache when present."""
    hit = get(family, stamp, day)
    if hit is not None:
        return hit
    closes, skips, blockers = fn()
    put(family, stamp, day, closes, skips, blockers)
    return closes, skips, blockers