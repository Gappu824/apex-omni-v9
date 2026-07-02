#!/usr/bin/env python3
"""
macro_direction_test.py — does the RAW GEX signal predict index DIRECTION,
significantly, with nothing else in the way?

The options backtest realised 55 % directional hit (a coin flip) — but it BLENDED
features, FILTERED on max-pain (dropping ~123 of ~230 signals), and used variable
max-pain/stop/time EXITS, every one of which dilutes a raw signal. This removes
all of it. For each macro feature ALONE, oriented so "+ = expect spot UP", at the
fixed IC horizons: sign(feature) vs sign(forward index return).

For each feature × horizon it reports:
  • hit rate, with an across-DAY t-test (each day = one observation, so
    autocorrelation doesn't inflate significance — the honest test)
  • the hit rate the feature's OWN Spearman IC IMPLIES, 0.5 + arcsin(ρ)/π, so you
    can see whether the sign-test agrees with the rank IC or whether the backtest's
    55 % was construction leakage
  • the average forward return conditioned on signal-up vs signal-down (does the
    signal actually separate up-moves from down-moves, and by how many bp)

This settles the open question cheaply: if a raw feature clears t≥2 with hit >52 %
AND separates up/down returns, the edge is REAL and the options backtest merely
expressed it badly — a better expression is worth building. If every raw feature
is a coin flip even here, the IC was a pooled-rank artefact that does not survive
as a tradable directional call, and the GEX-channel thesis closes.

Offline, no torch, no replay — macro_snapshots_v9 + spot ticks only.
    python macro_direction_test.py [--db PATH] [--horizons 300,900,1800]
"""
from __future__ import annotations
import argparse
import sqlite3
import math
import numpy as np

import config

TRADABLE = config.TRADABLE
DATE = "date(ts_local_ms/1000,'unixepoch','localtime')"

# each feature oriented so POSITIVE = expect spot UP (per the measured IC signs)
def _feats(s, cw, pw, mp, flip):
    f = {}
    f["maxpain_pull"] = (mp - s) / s                       # +IC
    f["channel_mid"] = (0.5 * (cw + pw) - s) / s if cw > pw else float("nan")
    f["putwall(-)"] = -((s - pw) / s)                      # −IC → flip
    f["callwall"] = (cw - s) / s                           # +IC
    f["flip(-)"] = -((s - flip) / s) if np.isfinite(flip) else float("nan")  # −IC → flip
    return f
FEAT_NAMES = ["maxpain_pull", "channel_mid", "putwall(-)", "callwall", "flip(-)"]


# ------------------------------------------------------------- spearman (no scipy)
def _rank(a):
    a = np.asarray(a, float)
    o = a.argsort(kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    sa = a[o]; i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / d) if d > 0 else float("nan")


def implied_hit(ic):
    return 0.5 + math.asin(max(-1.0, min(1.0, ic))) / math.pi if np.isfinite(ic) else float("nan")


# ------------------------------------------------------------- data
def load_macro(con, day, idx):
    try:
        rows = con.execute(
            "SELECT ts_ms/1000.0, spot, flip, call_wall, put_wall, net_gex, max_pain "
            "FROM macro_snapshots_v9 WHERE index_name=? AND "
            f"date(ts_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (idx, day)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(float(ts), s, flip, cw, pw, ng, mp)
            for ts, s, flip, cw, pw, ng, mp in rows]


def spot_path(con, day, tok):
    rows = con.execute(
        f"SELECT ts_ms/1000, ltp FROM ticks_v9 WHERE token=? AND ltp>0 AND "
        f"{DATE}=? ORDER BY ts_ms", (tok, day)).fetchall()
    if not rows:
        return np.array([]), np.array([])
    return (np.array([int(r[0]) for r in rows]),
            np.array([float(r[1]) for r in rows]))


def fwd_ret(secs, px, t, h):
    if len(secs) == 0:
        return float("nan")
    j0 = np.searchsorted(secs, t, side="right") - 1
    if j0 < 0:
        return float("nan")
    j1 = np.searchsorted(secs, t + h, side="left")
    if j1 >= len(secs) or secs[j1] - (t + h) > 90:
        return float("nan")
    s0, s1 = px[j0], px[j1]
    return (s1 - s0) / s0 if s0 > 0 else float("nan")


def run(db, horizons):
    con = sqlite3.connect(db)
    try:
        days = [r[0] for r in con.execute(
            f"SELECT DISTINCT {DATE} FROM ticks_v9 ORDER BY 1").fetchall()]
    except sqlite3.OperationalError as e:
        print(f"cannot read ticks_v9: {e}")
        return
    import nightly_forge_v9 as forge
    print(f"vault: {db}\ndays: {len(days)} ({days[0]} … {days[-1]})  "
          f"horizons(s): {horizons}\n")

    # per index/feature/horizon: daily hit rates, pooled (feat,fwd) pairs,
    # and conditional up/down forward-return pools
    daily = {i: {f: {h: [] for h in horizons} for f in FEAT_NAMES} for i in TRADABLE}
    pool = {i: {f: {h: [[], []] for h in horizons} for f in FEAT_NAMES} for i in TRADABLE}
    updown = {i: {f: {h: [[], []] for h in horizons} for f in FEAT_NAMES} for i in TRADABLE}

    for day in days:
        for idx in TRADABLE:
            macro = load_macro(con, day, idx)
            tok = forge.spot_token_for(con, day, idx)
            if not macro or not tok:
                continue
            secs, px = spot_path(con, day, tok)
            if len(secs) == 0:
                continue
            for h in horizons:
                per_feat_hits = {f: [] for f in FEAT_NAMES}
                for ts, s, flip, cw, pw, ng, mp in macro:
                    if s is None or s <= 0:
                        continue
                    r = fwd_ret(secs, px, ts, h)
                    if not np.isfinite(r):
                        continue
                    fv = _feats(s, cw or float("nan"), pw or float("nan"),
                                mp or float("nan"), flip if flip is not None else float("nan"))
                    for f in FEAT_NAMES:
                        v = fv[f]
                        if not np.isfinite(v) or v == 0:
                            continue
                        per_feat_hits[f].append(1.0 if (v > 0) == (r > 0) else 0.0)
                        pool[idx][f][h][0].append(v)
                        pool[idx][f][h][1].append(r)
                        (updown[idx][f][h][0] if v > 0 else updown[idx][f][h][1]).append(r)
                for f in FEAT_NAMES:
                    if len(per_feat_hits[f]) >= 10:
                        daily[idx][f][h].append(float(np.mean(per_feat_hits[f])))

    for idx in TRADABLE:
        print("=" * 84)
        print(f" {idx} — raw directional hit rate (sign of feature vs sign of forward return)")
        print("=" * 84)
        for h in horizons:
            print(f"\n  horizon {h//60} min")
            print(f"    {'feature':14s} {'hit%':>6s} {'t':>6s}  {'IC':>6s} "
                  f"{'IC→hit%':>7s}  {'up_bp':>7s} {'dn_bp':>7s}  read")
            for f in FEAT_NAMES:
                ics = daily[idx][f][h]
                px_, py_ = pool[idx][f][h]
                if len(ics) < 2 or len(px_) < 20:
                    print(f"    {f:14s} {'—':>6s}")
                    continue
                m = float(np.mean(ics)); sd = float(np.std(ics, ddof=1))
                t = (m - 0.5) / (sd / math.sqrt(len(ics))) if sd > 0 else 0.0
                ic = spearman(np.array(px_), np.array(py_))
                ih = implied_hit(ic)
                up = np.array(updown[idx][f][h][0]); dn = np.array(updown[idx][f][h][1])
                up_bp = up.mean() * 1e4 if len(up) else float("nan")
                dn_bp = dn.mean() * 1e4 if len(dn) else float("nan")
                star = "*" if abs(t) >= 2 else " "
                sep = "sep" if (np.isfinite(up_bp) and np.isfinite(dn_bp)
                               and up_bp > dn_bp) else "—"
                print(f"    {f:14s} {m*100:5.1f}%{star}{t:+6.1f}  {ic:+6.3f} "
                      f"{ih*100:6.1f}%  {up_bp:+7.1f} {dn_bp:+7.1f}  {sep}")
        print()
    con.close()

    # global verdict
    best_t = 0.0; best = None
    for idx in TRADABLE:
        for f in FEAT_NAMES:
            for h in horizons:
                ics = daily[idx][f][h]
                if len(ics) >= 2:
                    sd = np.std(ics, ddof=1)
                    if sd > 0:
                        t = (np.mean(ics) - 0.5) / (sd / math.sqrt(len(ics)))
                        if abs(t) > abs(best_t):
                            best_t, best = t, (idx, f, h, float(np.mean(ics)))
    print("=" * 84)
    print(" VERDICT")
    print("=" * 84)
    if best and abs(best_t) >= 2 and best[3] > 0.52:
        print(f"  A raw feature DOES beat a coin flip: {best[1]} on {best[0]} @{best[2]//60}m "
              f"= {best[3]*100:.1f}% hit (t={best_t:+.1f}).")
        print("  The directional edge is REAL — the 55% in the options backtest was")
        print("  construction leakage (blend + max-pain filter + variable exits), not the")
        print("  signal. A cleaner expression (single feature, fixed horizon) is worth")
        print("  building — and the options-vs-index / long-vs-short question reopens.")
    else:
        bt = f"{best[1]} on {best[0]} @{best[2]//60}m, t={best_t:+.1f}" if best else "none"
        print(f"  NO raw feature beats a coin flip significantly (best: {bt}).")
        print("  The channel IC was a pooled-rank association that does NOT survive as a")
        print("  tradable directional call at any horizon, even stripped of blend/filter/")
        print("  exits. The GEX-channel DIRECTIONAL thesis closes — stop engineering it.")
        print("  (The pinning/mean-reversion may still be a VOLATILITY structure, but that")
        print("   is a different, non-directional strategy, not this policy.)")
    print("  (in-sample, 1 positive-gamma regime, 10 days — necessary, not sufficient.)")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--horizons", default="300,900,1800")
    a = ap.parse_args()
    run(a.db, [int(x) for x in a.horizons.split(",")])


if __name__ == "__main__":
    main()