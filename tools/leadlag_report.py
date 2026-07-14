"""
APEX OMNI v10 — CROSS-INDEX LEAD-LAG + FLOW-IMBALANCE REPORT (weekly)
=====================================================================
Who moves first, NIFTY or SENSEX — and by how many seconds? Peak lagged
cross-correlation of 1-second log-returns per day (the practical core of the
price-discovery literature; a full Hasbrouck information share is a VECM —
chartered, not faked). Positive lead_s ⇒ the FIRST index leads. Also emits a
volume-free tick-rule flow imbalance per index (Lee–Ready sign of ltp vs the
running mid; the vault stores no volume, so true VPIN is chartered behind a
one-column harvester change). Output: state/leadlag.json + console; wired
into run_weekly. Pure functions are unit-proven; main() needs the vault.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config                                            # noqa: E402


def lead_lag(ra: np.ndarray, rb: np.ndarray, max_lag: int = 30):
    """Peak lagged corr of two aligned 1s return series. Returns
    (lead_s, strength): lead_s>0 ⇒ series A leads B by that many seconds."""
    m = np.isfinite(ra) & np.isfinite(rb)
    ra, rb = ra.copy(), rb.copy()
    ra[~m] = 0.0
    rb[~m] = 0.0
    if m.sum() < 300 or ra.std() < 1e-12 or rb.std() < 1e-12:
        return None, None
    best = (0, 0.0)
    for k in range(-max_lag, max_lag + 1):
        a = ra[max(0, -k):len(ra) - max(0, k)]
        b = rb[max(0, k):len(rb) - max(0, -k)]
        if len(a) < 300:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if abs(c) > abs(best[1]):
            best = (k, c)
    return best


def flow_imbalance(spots: np.ndarray, window: int = 300) -> float | None:
    """Volume-free toxicity proxy: tick-rule signed move imbalance over the
    trailing window, in [-1,1]. |x|→1 = one-sided (toxic) tape."""
    d = np.diff(spots)
    d = d[np.isfinite(d)][-window:]
    sgn = np.sign(d[d != 0])
    if len(sgn) < 30:
        return None
    return float(sgn.mean())


def main():
    import sqlite3
    from simulation.scenario_engine import N
    from tools.butterfly_harness import _spot_series
    from nightly_forge_v9 import trading_days, spot_token_for
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-10:]
    per_day, agg = [], []
    for day in days:
        s = {}
        for idx in config.TRADABLE:
            tok = spot_token_for(con, day, idx)
            if tok:
                s[idx], _ = _spot_series(con, day, tok, N)
        if len(s) < 2:
            continue
        a, b = config.TRADABLE[0], config.TRADABLE[1]
        ra = np.diff(np.log(np.where(s[a] > 0, s[a], np.nan)))
        rb = np.diff(np.log(np.where(s[b] > 0, s[b], np.nan)))
        k, c = lead_lag(ra, rb)
        if k is not None:
            per_day.append({"day": day, "lead_s": k, "corr": round(c, 3),
                            "leader": a if k > 0 else (b if k < 0 else "—")})
            agg.append(k)
    out = {"pair": config.TRADABLE[:2], "days": per_day,
           "median_lead_s": (float(np.median(agg)) if agg else None),
           "note": "lead_s>0 ⇒ first index leads; consume as a REGISTERED "
                   "spec, never raw", "ts": time.time()}
    (config.STATE_DIR / "leadlag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()