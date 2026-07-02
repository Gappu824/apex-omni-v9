#!/usr/bin/env python3
"""
forge_direction_balance.py — is the training window up-day-heavy?

The probe showed the policy is directionally inert with a positive tilt. Two
things can produce a positive tilt: (1) the policy structurally leaning long, and
(2) the days it trained on simply being mostly up-days, so "calls paid" more
often. This reads the SAME vault the forge trains from (config.DB_PATH, table
ticks_v9) and reports, per index, how many up vs down days the window contains —
both the full vault and the heavily-weighted last FORGE_LOOKBACK_DAYS, which is
the slice that most shapes the policy.

A day's direction = sign(close − open) of the index SPOT, computed exactly the
way the forge resolves a day: grouped by date(ts_local_ms/1000,'localtime'), spot
token resolved as-of the day from spot_tokens. Read-only.

    python forge_direction_balance.py [--db PATH]
"""
from __future__ import annotations
import argparse
import sqlite3

import config

REF_INDICES = ["NIFTY", "SENSEX"]
DATE_EXPR = "date(ts_local_ms/1000,'unixepoch','localtime')"


def trading_days(con) -> list[str]:
    return [r[0] for r in con.execute(
        f"SELECT DISTINCT {DATE_EXPR} FROM ticks_v9 ORDER BY 1").fetchall()]


def spot_token_for(con, day: str, index: str) -> int | None:
    r = con.execute("SELECT token FROM spot_tokens WHERE snap_date<=? AND "
                    "name=? ORDER BY snap_date DESC LIMIT 1",
                    (day, index)).fetchone()
    return int(r[0]) if r else None


def _edge_px(con, token: int, day: str, newest: bool) -> float | None:
    order = "DESC" if newest else "ASC"
    r = con.execute(
        f"SELECT ltp FROM ticks_v9 WHERE token=? AND {DATE_EXPR}=? "
        f"AND ltp>0 ORDER BY ts_ms {order} LIMIT 1", (token, day)).fetchone()
    return float(r[0]) if r and r[0] else None


def day_direction(con, day: str, index: str):
    """Return (pct_move, open, close) or None if the day can't be resolved."""
    tok = spot_token_for(con, day, index)
    if tok is None:
        return None
    o = _edge_px(con, tok, day, newest=False)
    c = _edge_px(con, tok, day, newest=True)
    if not o or not c:
        return None
    return ((c - o) / o * 100.0, o, c)


def _tally(rows: list[float], tol: float = 0.05):
    up = sum(1 for x in rows if x > tol)
    dn = sum(1 for x in rows if x < -tol)
    fl = len(rows) - up - dn
    net = sum(rows)
    return up, dn, fl, net


def _report_block(title: str, per_index: dict[str, list[float]]):
    print(f"\n{title}")
    print(f"  {'index':7s} | {'days':>4s} {'up':>4s} {'down':>4s} {'flat':>4s} "
          f"| {'net drift%':>10s} {'mean%':>7s}  balance")
    for idx in REF_INDICES:
        rows = per_index.get(idx, [])
        if not rows:
            print(f"  {idx:7s} |  (no resolvable days)")
            continue
        up, dn, fl, net = _tally(rows)
        mean = net / len(rows)
        n = len(rows)
        skew = (up - dn) / n
        tag = ("UP-heavy" if skew > 0.25 else
               "DOWN-heavy" if skew < -0.25 else "balanced")
        print(f"  {idx:7s} | {n:4d} {up:4d} {dn:4d} {fl:4d} "
              f"| {net:+10.2f} {mean:+7.2f}  {tag} ({up}↑/{dn}↓)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    try:
        days = trading_days(con)
    except sqlite3.OperationalError as e:
        print(f"could not read ticks_v9 from {args.db}: {e}")
        return
    if not days:
        print(f"no trading days in {args.db}")
        return

    lookback = int(getattr(config, "FORGE_LOOKBACK_DAYS", 10))
    reservoir = int(getattr(config, "FORGE_RESERVOIR_DAYS", 5))
    print(f"vault: {args.db}")
    print(f"trading days: {len(days)}  ({days[0]} … {days[-1]})")
    print(f"forge window: last {lookback} days + reservoir of {reservoir} older "
          f"days (the last {lookback} carry the most weight)")

    all_rows = {idx: [] for idx in REF_INDICES}
    look_rows = {idx: [] for idx in REF_INDICES}
    look_set = set(days[-lookback:])
    detail = []
    for day in days:
        line = {"day": day}
        for idx in REF_INDICES:
            d = day_direction(con, day, idx)
            if d is None:
                line[idx] = None
                continue
            pct, o, c = d
            all_rows[idx].append(pct)
            if day in look_set:
                look_rows[idx].append(pct)
            line[idx] = pct
        detail.append(line)

    _report_block("FULL VAULT", all_rows)
    _report_block(f"FORGE LOOKBACK (last {lookback} days — most-weighted)", look_rows)

    print(f"\nper-day detail (NIFTY / SENSEX open→close %):")
    for ln in detail:
        mark = "  «lookback" if ln["day"] in look_set else ""
        def fmt(v):
            return f"{v:+6.2f}" if isinstance(v, float) else "   —  "
        print(f"  {ln['day']}  N {fmt(ln.get('NIFTY'))}  S {fmt(ln.get('SENSEX'))}{mark}")

    print("\nread: if the lookback block is UP-heavy, part of the positive tilt is the\n"
          "window, not just the policy — but the probe's INERTNESS is independent of\n"
          "this. Rebalancing days (or de-meaning the directional label) addresses the\n"
          "tilt; folding macro direction into the obs addresses the inertness.")
    con.close()


if __name__ == "__main__":
    main()