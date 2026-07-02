#!/usr/bin/env python3
"""
feature_ic_analyzer.py — does the directional SIGNAL exist on the vault, and is
it in the obs or only in the macro state?

The policy probe showed the policy does not USE direction. It can't show whether
direction is THERE to be used. This measures, per feature, the Information
Coefficient (rank correlation with forward index return) on the real vault, for
two groups:

  IN-OBS directional features (faithfully reconstructed by re-running the forge's
  OWN replay_day — the identical StateBuilder the policy trains on — and reading
  them straight out of the 5700-vector at the current frame):
    • log_ret      — the spot node's return (directional price)
    • ofi_skew     — ATM(CE − PE) order-flow imbalance  (call vs put buying)
    • dealer_skew  — ATM(CE − PE) decayed signed aggressive flow
    • oi_skew      — ATM(CE − PE) OI change             (positioning skew)

  MACRO / GEX features (the edge currently only bolted on as advisory nudges,
  read from macro_snapshots_v9 — NOT in the policy obs):
    • flip_dist    — (spot − flip)/spot         (above/below the GEX flip line)
    • maxpain_pull — (max_pain − spot)/spot      (pull toward max-pain)
    • net_gex      — signed net gamma            (sign = dealer long/short gamma)
    • pcr          — put/call ratio
    • callwall_dist, putwall_dist, wall_pos — distance to / position in the GEX
                     channel (level-break geometry)

Decision this settles
---------------------
  • IN-OBS features show real IC  → the signal is already in the obs; the policy
    just never learned to use it. FIX = the LABEL (clean signed target), no obs
    change, no OBS_DIM bump, cheaper retrain.
  • IN-OBS flat but MACRO shows IC → the edge is real but OUTSIDE the obs.
    Folding macro into the observation is justified AND now validated before you
    spend the hours.
  • Neither shows IC at your horizons → direction is not predictable from these
    at this timescale. The most important thing to learn before building anything.

Method: per DAY, Spearman IC of each feature vs the h-second-forward spot return,
then aggregate across days as mean ± std with t = mean/(std/√n_days) — the IC
information-ratio. Intraday samples are heavily autocorrelated, so the pooled-N
significance is inflated; the across-DAY t-stat (each day = one observation) is
the honest test and is what to read. Horizons default to 5 / 15 / 30 min.

Offline, no torch (the forge's torch import is optional and unused here). Run on
the machine where the vault lives:
    python feature_ic_analyzer.py [--db PATH] [--sample-sec 60] [--horizons 300,900,1800]
"""
from __future__ import annotations
import argparse
import sqlite3
import json
import numpy as np

import config
import nightly_forge_v9 as forge

NPI = config.NODES_PER_INDEX
F = config.FEATURES_PER_NODE
NUM_NODES = config.NUM_NODES
SEQ = config.SEQ_LENGTH
CUR = (SEQ - 1) * NUM_NODES * F          # offset of the current (last) frame
TRADABLE = config.TRADABLE

INOBS = ["log_ret", "ofi_skew", "dealer_skew", "oi_skew"]
MACRO = ["flip_dist", "maxpain_pull", "net_gex", "pcr",
         "callwall_dist", "putwall_dist", "wall_pos"]


# ----------------------------------------------------------- IC math (no scipy)
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (Spearman-correct)."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average tied ranks
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math_sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / d) if d > 0 else float("nan")


def math_sqrt(v):
    return float(np.sqrt(v))


# ----------------------------------------------- feature extraction from obs
def _node_off(i: int, leg: int) -> int:
    # leg: 0 spot, 1 atm_ce, 2 atm_pe, 3 otm_ce, 4 otm_pe  (LEG_ORDER)
    return CUR + (i * NPI + leg) * F


def inobs_features(obs: np.ndarray, i: int) -> dict | None:
    spot, ce, pe = _node_off(i, 0), _node_off(i, 1), _node_off(i, 2)
    def g(off, feat):
        return float(obs[off + feat])
    # leg nodes are zero when that day has no chain → skews undefined there
    ce_present = any(obs[ce:ce + F])
    pe_present = any(obs[pe:pe + F])
    legs = ce_present and pe_present
    return {
        "log_ret":     g(spot, 0),
        "ofi_skew":    (g(ce, 12) - g(pe, 12)) if legs else float("nan"),
        "dealer_skew": (g(ce, 16) - g(pe, 16)) if legs else float("nan"),
        "oi_skew":     (g(ce, 2) - g(pe, 2)) if legs else float("nan"),
    }


def macro_features(row: dict, spot: float) -> dict:
    def sig(x):
        return x if (x is not None and np.isfinite(x)) else float("nan")
    flip = sig(row.get("flip")); mp = sig(row.get("max_pain"))
    cw = sig(row.get("call_wall")); pw = sig(row.get("put_wall"))
    ng = sig(row.get("net_gex")); pcr = sig(row.get("pcr"))
    s = spot if spot else float("nan")
    chan = (cw - pw) if (np.isfinite(cw) and np.isfinite(pw) and cw > pw) else float("nan")
    return {
        "flip_dist":     (s - flip) / s if np.isfinite(flip) else float("nan"),
        "maxpain_pull":  (mp - s) / s if np.isfinite(mp) else float("nan"),
        "net_gex":       ng,
        "pcr":           pcr,
        "callwall_dist": (cw - s) / s if np.isfinite(cw) else float("nan"),
        "putwall_dist":  (s - pw) / s if np.isfinite(pw) else float("nan"),
        "wall_pos":      ((s - pw) / chan - 0.5) if np.isfinite(chan) else float("nan"),
    }


def load_macro_rows(con, day: str, index: str) -> list[tuple[float, dict]]:
    try:
        rows = con.execute(
            "SELECT ts_ms/1000.0, spot, flip, call_wall, put_wall, net_gex, "
            "pcr, max_pain FROM macro_snapshots_v9 WHERE index_name=? AND "
            "date(ts_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (index, day)).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for ts, spot, flip, cw, pw, ng, pcr, mp in rows:
        out.append((float(ts), {"spot": spot, "flip": flip, "call_wall": cw,
                                 "put_wall": pw, "net_gex": ng, "pcr": pcr,
                                 "max_pain": mp}))
    return out


# ----------------------------------------------------------- per-day collect
def collect_day(con, day: str, sample_sec: int):
    """Return {index: {"t": [...], "spot_path": {sec: spot}, "feat": {name:[...]}}}.
    feat rows are sampled on the grid; spot_path is every second (for fwd returns)."""
    macro = {idx: load_macro_rows(con, day, idx) for idx in TRADABLE}
    mptr = {idx: 0 for idx in TRADABLE}
    data = {idx: {"t": [], "spot_path": {}, "feat": {k: [] for k in INOBS + MACRO}}
            for idx in TRADABLE}
    last_sample = {idx: -10 ** 9 for idx in TRADABLE}
    for ts, obs, market, _macro_now in forge.replay_day(con, day):
        if obs is None:
            continue
        sec = int(ts)
        for idx in TRADABLE:
            ctx = market.get(idx)
            if not ctx:
                continue
            spot = float((ctx.get("spot") or {}).get("ltp") or 0.0)
            if spot <= 0:
                continue
            data[idx]["spot_path"][sec] = spot
            if sec - last_sample[idx] < sample_sec:
                continue
            last_sample[idx] = sec
            i = config.INDEX_ORDER.index(idx)
            fo = inobs_features(obs, i)
            # advance macro pointer to latest row at-or-before sec
            rows = macro[idx]
            p = mptr[idx]
            while p + 1 < len(rows) and rows[p + 1][0] <= ts:
                p += 1
            mptr[idx] = p
            mrow = rows[p][1] if rows else {}
            mf = macro_features(mrow, spot) if rows else {k: float("nan") for k in MACRO}
            data[idx]["t"].append(sec)
            for k in INOBS:
                data[idx]["feat"][k].append(fo[k])
            for k in MACRO:
                data[idx]["feat"][k].append(mf[k])
    return data


def _fwd_return(spot_path: dict, secs: list[int], h: int):
    """For each sample sec, forward return over h seconds using the next spot at
    or after sec+h (within a 90s tolerance). Returns array aligned to secs (nan
    where no forward point)."""
    keys = np.array(sorted(spot_path.keys()))
    out = np.full(len(secs), np.nan)
    for n, s in enumerate(secs):
        target = s + h
        j = np.searchsorted(keys, target, side="left")
        if j >= len(keys):
            continue
        if keys[j] - target > 90:       # gap too large
            continue
        s0 = spot_path[int(s)]
        s1 = spot_path[int(keys[j])]
        if s0 > 0:
            out[n] = (s1 - s0) / s0
    return out


# ----------------------------------------------------------- IC aggregation
def analyze(db: str, sample_sec: int, horizons: list[int]):
    con = sqlite3.connect(db)
    try:
        days = forge.trading_days(con)
    except sqlite3.OperationalError as e:
        print(f"cannot read ticks_v9 from {db}: {e}")
        return
    if not days:
        print(f"no trading days in {db}")
        return
    print(f"vault: {db}\ntrading days: {len(days)} ({days[0]} … {days[-1]})")
    print(f"grid: {sample_sec}s | horizons(s): {horizons} | indices: {TRADABLE}\n")

    # per (index, feature, horizon): list of daily ICs, and pooled pairs
    daily = {idx: {f: {h: [] for h in horizons} for f in INOBS + MACRO} for idx in TRADABLE}
    pooled = {idx: {f: {h: [[], []] for h in horizons} for f in INOBS + MACRO} for idx in TRADABLE}

    for d_i, day in enumerate(days, 1):
        try:
            data = collect_day(con, day, sample_sec)
        except Exception as e:
            print(f"  [{d_i}/{len(days)}] {day}: replay failed ({e}); skipping")
            continue
        for idx in TRADABLE:
            secs = data[idx]["t"]
            if len(secs) < 5:
                continue
            for h in horizons:
                fwd = _fwd_return(data[idx]["spot_path"], secs, h)
                for f in INOBS + MACRO:
                    fv = np.asarray(data[idx]["feat"][f], float)
                    m = np.isfinite(fv) & np.isfinite(fwd)
                    if m.sum() < 20:
                        continue
                    x, y = fv[m], fwd[m]
                    if np.ptp(x) == 0:
                        continue
                    ic = spearman(x, y)
                    if np.isfinite(ic):
                        daily[idx][f][h].append(ic)
                        pooled[idx][f][h][0].extend(x.tolist())
                        pooled[idx][f][h][1].extend(y.tolist())
        print(f"  [{d_i}/{len(days)}] {day}: done")

    for idx in TRADABLE:
        print("\n" + "=" * 78)
        print(f" {idx} — Information Coefficient (Spearman vs forward return)")
        print("=" * 78)
        for group, feats in (("IN-OBS (already seen by policy)", INOBS),
                             ("MACRO / GEX (not in obs — the candidate edge)", MACRO)):
            print(f"\n  {group}")
            hdr = "    {:14s}".format("feature")
            for h in horizons:
                hdr += f" | {h//60}m: {'IC':>6s} {'t':>5s}"
            print(hdr + "   pooled(N)")
            for f in feats:
                line = "    {:14s}".format(f)
                pooled_txt = ""
                for h in horizons:
                    ics = daily[idx][f][h]
                    if len(ics) >= 2:
                        m, sd = float(np.mean(ics)), float(np.std(ics, ddof=1))
                        t = m / (sd / np.sqrt(len(ics))) if sd > 0 else 0.0
                        star = "*" if abs(t) >= 2 else " "
                        line += f" | {m:+6.3f}{star}{t:+5.1f}"
                    elif len(ics) == 1:
                        line += f" | {ics[0]:+6.3f} {'1d':>5s}"
                    else:
                        line += f" | {'—':>6s} {'—':>5s}"
                    px, py = pooled[idx][f][h]
                    if px and h == horizons[-1]:
                        pic = spearman(np.array(px), np.array(py))
                        pooled_txt = f"  {pic:+.3f} (N={len(px)})"
                print(line + pooled_txt)
        print("\n  (* = |across-day t| ≥ 2. Read the t-stat, not pooled-N — pooled is")
        print("   autocorrelation-inflated. |IC|≳0.03 with t≥2 is a usable intraday edge.)")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--sample-sec", type=int, default=60)
    ap.add_argument("--horizons", default="300,900,1800")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    analyze(args.db, args.sample_sec, horizons)


if __name__ == "__main__":
    main()