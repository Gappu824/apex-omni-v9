"""
APEX OMNI v9.1 — DIAGNOSTIC REPORTS (audit follow-up: make the system legible)
==============================================================================
Every long-running component (harvester, macro radar, brain, forge) now writes
a machine-readable daily report to logs/<component>_report_<date>.json in
addition to its human log. The reports are the raw material for the next audit
round: gate funnels, coverage rates, conviction distributions, walk-forward
tables. Atomic temp→rename writes so a crash mid-write never leaves a torn
file; every report is stamped with VERSION + CONFIG_HASH so numbers are always
attributable to the exact configuration that produced them.

Pure stdlib + numpy-free. Importable everywhere, costs nothing on hot paths
(writers rate-limit themselves; this module only serializes).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from collections import deque

import config


def _atomic_write_json(path, payload: dict) -> None:
    d = os.path.dirname(str(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=_json_default)
    os.replace(tmp, str(path))


def _json_default(o):
    try:
        return float(o)                       # numpy scalars, Decimals
    except Exception:                         # noqa: BLE001
        return repr(o)


class Reservoir:
    """Bounded sample of a stream for cheap percentile reporting (keeps the
    most recent `maxlen` values — recency-weighted on purpose: today's tape is
    what tomorrow's decisions face)."""

    __slots__ = ("buf",)

    def __init__(self, maxlen: int = 20_000):
        self.buf: deque = deque(maxlen=maxlen)

    def add(self, x: float) -> None:
        self.buf.append(float(x))

    def summary(self) -> dict:
        n = len(self.buf)
        if n == 0:
            return {"n": 0}
        s = sorted(self.buf)

        def pct(p):
            return round(s[min(int(p * (n - 1)), n - 1)], 4)

        return {"n": n, "p50": pct(0.50), "p90": pct(0.90),
                "p95": pct(0.95), "p99": pct(0.99), "max": round(s[-1], 4)}


class DailyReport:
    """A dict with a date-stamped path and an atomic writer. Components mutate
    `self.d` freely and call write() on their own cadence; write() also stamps
    freshness so a stalled process is visible from the file alone."""

    def __init__(self, component: str):
        self.component = component
        self.date = str(dt.date.today())
        self.path = config.LOG_DIR / f"{component}_report_{self.date}.json"
        self.d: dict = {"component": component, "date": self.date,
                        "version": config.VERSION,
                        "config_hash": config.CONFIG_HASH,
                        "started_utc": dt.datetime.utcnow().isoformat(
                            timespec="seconds") + "Z"}
        self._last_write = 0.0

    def write(self, min_interval_s: float = 0.0) -> None:
        now = time.time()
        if min_interval_s and now - self._last_write < min_interval_s:
            return
        self._last_write = now
        self.d["updated_utc"] = dt.datetime.utcnow().isoformat(
            timespec="seconds") + "Z"
        try:
            _atomic_write_json(self.path, self.d)
        except Exception:                                 # noqa: BLE001
            pass                                          # never hurt the host