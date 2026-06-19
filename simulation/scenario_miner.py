"""
APEX OMNI v9 — SCENARIO MINER (the suite that grows)
====================================================
The 16 authored scenarios are the constitution's exam. This module makes the
exam GROW from reality: after each harvested day, `mine()` scans the spot
tick series for three signatures —

  * stop-runs   : ≥25-pt flush inside ≤45 s that reclaims ≥60% within 5 min
                  (your "institutional trap", as it actually printed),
  * air pockets : ≥0.8% repricing inside ≤20 s (flash legs),
  * feed gaps   : ≥30 s with zero spot ticks during session hours,

parameterizes each one (depth, duration, reclaim, time of day) and appends
it to simulation/discovered_scenarios.json. `load_discovered()` turns those
records into runnable Scenario objects with invariant-only pass criteria
(no floor overruns, no phantom fills, no post-halt buys, flat at close), and
run_simulation.py executes them after the core 16 — so tomorrow's regression
gate already contains today's weirdness.

Honest scope: the miner adds new PARAMETERIZED VARIANTS of known event
kinds automatically. A genuinely new KIND of event (something with a shape
none of the generators can express) still needs a human to author a
generator — it will, however, show up loudly in the harvester/brain logs.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

import config
from simulation.scenario_engine import SimDay, Scenario, Signal, T0_SEC, N

log = logging.getLogger("miner")
STORE = Path(__file__).resolve().parent / "discovered_scenarios.json"
MAX_KEEP = 30


# ----------------------------------------------------------------- mining
def _spot_series(con, day: str, index: str = "NIFTY"):
    r = con.execute("SELECT token FROM spot_tokens WHERE snap_date<=? AND "
                    "name=? ORDER BY snap_date DESC LIMIT 1",
                    (day, index)).fetchone()
    if not r:
        return None, None
    tok = int(r[0])
    rows = con.execute(
        "SELECT ts_local_ms/1000, ltp FROM ticks_v9 WHERE token=? AND "
        "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY 1",
        (tok, day)).fetchall()
    if len(rows) < 600:
        return None, None
    px = np.full(N, np.nan)
    seen = np.zeros(N, bool)
    for s, p in rows:
        t = int(s) % 86400 - T0_SEC
        if 0 <= t < N and p:
            px[t] = p
            seen[t] = True
    # forward-fill for analysis; `seen` keeps the truth about feed gaps
    last = np.nan
    for t in range(N):
        if np.isnan(px[t]):
            px[t] = last
        else:
            last = px[t]
    return px, seen


def _hm(t: int) -> str:
    s = T0_SEC + t
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def _find_stop_runs(px):
    out, t = [], 660                       # skip the first 11 minutes
    while t < N - 360:
        if np.isnan(px[t]):
            t += 1; continue
        w0 = max(t - 60, 0)
        hi_i = w0 + int(np.nanargmax(px[w0:t + 1]))
        hi = px[hi_i]
        drop = hi - px[t]
        if drop >= 25 and (t - hi_i) <= 45:
            lo_i, lo = t, px[t]
            for u in range(t, min(t + 60, N)):
                if px[u] < lo:
                    lo, lo_i = px[u], u
            depth = hi - lo
            rec_i, rec = None, 0.0
            for u in range(lo_i, min(lo_i + 300, N)):
                rec = max(rec, px[u] - lo)
                if rec >= 0.6 * depth:
                    rec_i = u; break
            if rec_i is not None and depth >= 25:
                out.append({"kind": "stop_run", "hm": _hm(hi_i),
                            "depth_pts": round(float(depth), 1),
                            "down_s": int(max(lo_i - hi_i, 8)),
                            "recover_pts": round(float(rec), 1),
                            "recover_s": int(max(rec_i - lo_i, 30)),
                            "open": round(float(px[0]), 1)})
                t = rec_i + 600            # ≥10 min between events
                continue
        t += 1
    return out[:3]


def _find_gaps(px):
    out = []
    for t in range(680, N, 20):
        a, b = px[t - 20], px[t]
        if np.isnan(a) or np.isnan(b):
            continue
        if abs(b - a) / a >= 0.008:
            out.append({"kind": "gap", "hm": _hm(t - 20),
                        "pts": round(float(b - a), 1),
                        "open": round(float(px[0]), 1)})
    dedup, last = [], -10_000
    for e in out:
        ti = int(e["hm"][:2]) * 3600 + int(e["hm"][3:]) * 60 - T0_SEC
        if ti - last > 600:
            dedup.append(e); last = ti
    return dedup[:2]


def _find_stale(seen):
    out, run, start = [], 0, None
    for t in range(N):
        if not seen[t]:
            if start is None:
                start = t
            run += 1
        else:
            if run >= 30 and start and start > 300:
                out.append({"kind": "stale", "hm": _hm(start),
                            "dur_s": int(run)})
            run, start = 0, None
    return out[:2]


def mine(day: str | None = None, index: str = "NIFTY") -> int:
    """Scan the latest (or given) harvested day; append discoveries. Returns
    the number of NEW scenarios stored."""
    if not Path(config.DB_PATH).exists():
        log.info("no tick vault yet — nothing to mine")
        return 0
    con = sqlite3.connect(config.DB_PATH)
    if day is None:
        r = con.execute("SELECT MAX(date(ts_local_ms/1000,'unixepoch',"
                        "'localtime')) FROM ticks_v9").fetchone()
        day = r[0] if r else None
    if not day:
        return 0
    px, seen = _spot_series(con, day, index)
    con.close()
    if px is None:
        log.info("%s: not enough spot ticks to mine", day)
        return 0
    found = _find_stop_runs(px) + _find_gaps(px) + _find_stale(seen)
    if not found:
        log.info("%s: clean day — nothing mined", day)
        return 0
    store = {"scenarios": []}
    if STORE.exists():
        try:
            store = json.loads(STORE.read_text())
        except Exception:                                  # noqa: BLE001
            pass
    known = {s["name"] for s in store["scenarios"]}
    added = 0
    for e in found:
        e["date"], e["index"] = day, index
        e["name"] = f"mined_{e['kind']}_{day}_{e['hm'].replace(':', '')}"
        if e["name"] in known:
            continue
        store["scenarios"].append(e)
        added += 1
        log.info("⛏ mined %s: %s", e["name"],
                 {k: v for k, v in e.items() if k not in ("name", "date")})
    store["scenarios"] = store["scenarios"][-MAX_KEEP:]
    STORE.write_text(json.dumps(store, indent=1))
    return added


# ----------------------------------------------------------------- replay
def _generic_ok(r):
    if r.violations:
        return False, "; ".join(r.violations)
    return True, (f"invariants held — {len(r.trades)} trade(s), "
                  f"₹{r.pnl:+.0f}, {r.trap_holds} trap hold(s)"
                  f"{', halted' if r.halted else ''}")


def _build(spec: dict) -> Scenario | None:
    seed = int(hashlib.md5(spec["name"].encode()).hexdigest()[:6], 16)
    open_ = float(spec.get("open", 24500.0))
    kind = spec["kind"]
    if kind == "stop_run":
        day = (SimDay(open_spot=open_, seed=seed, noise_pts_s=0.35)
               .trend("09:40", spec["hm"], 0.4)
               .stop_run(spec["hm"], depth_pts=spec["depth_pts"],
                         down_s=max(int(spec["down_s"]), 8), hold_s=20,
                         recover_pts=spec["recover_pts"],
                         recover_s=max(int(spec["recover_s"]), 30))
               .trend(spec["hm"], "15:00", 0.8))
        atm = round(open_ / day.step) * day.step
        day.walls(put=atm - spec["depth_pts"] - 8, call=atm + 180)
        h, m = map(int, spec["hm"].split(":"))
        sig_hm = f"{(h * 60 + m - 10) // 60:02d}:{(h * 60 + m - 10) % 60:02d}"
        sigs = [Signal(max(sig_hm, "09:48"), +0.85, window_s=480)]
    elif kind == "gap":
        day = SimDay(open_spot=open_, seed=seed, noise_pts_s=0.45) \
            .gap(spec["hm"], spec["pts"], over_s=20)
        sigs = [Signal("09:50", +0.84 if spec["pts"] < 0 else -0.84,
                       window_s=600)]
    elif kind == "stale":
        day = (SimDay(open_spot=open_, seed=seed, noise_pts_s=0.40)
               .trend("09:45", "13:00", 0.8)
               .feed_stale(spec["hm"], int(spec["dur_s"])))
        h, m = map(int, spec["hm"].split(":"))
        sig_hm = f"{(h * 60 + m - 8) // 60:02d}:{(h * 60 + m - 8) % 60:02d}"
        sigs = [Signal(max(sig_hm, "09:48"), +0.84, window_s=900)]
    else:
        return None
    return Scenario(spec["name"],
                    f"mined from {spec.get('date', '?')} live data "
                    f"({kind} @ {spec['hm']})", day, sigs, _generic_ok)


def load_discovered() -> list[Scenario]:
    if not STORE.exists():
        return []
    try:
        specs = json.loads(STORE.read_text()).get("scenarios", [])
    except Exception:                                      # noqa: BLE001
        return []
    out = []
    for s in specs:
        try:
            sc = _build(s)
            if sc:
                out.append(sc)
        except Exception as e:                             # noqa: BLE001
            log.warning("skipping malformed mined scenario %s: %s",
                        s.get("name"), e)
    return out


if __name__ == "__main__":
    config.setup_logging("miner")
    n = mine()
    print(f"{n} new scenario(s) mined → {STORE.name}")
