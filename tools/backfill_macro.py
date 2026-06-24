"""
APEX OMNI v9 — MACRO ARCHIVE BACKFILL
=====================================
    python tools/backfill_macro.py                 # every pre-archive day in the vault
    python tools/backfill_macro.py 2026-06-16 2026-06-17 ...
    python tools/backfill_macro.py --days 5        # last 5 days lacking snapshots
    python tools/backfill_macro.py --source ticks  # ticks only (book mid)
    python tools/backfill_macro.py --source candles # 1-min candles only (close+OI)
    python tools/backfill_macro.py --source auto    # DEFAULT: ticks per token, candle fallback
    python tools/backfill_macro.py --force          # re-do days that already have rows
    python tools/backfill_macro.py --dry-run        # count only, write nothing

WHY THIS EXISTS
---------------
macro_snapshots_v9 only records from the first moment you ran the new
macro_gex_v9.py live. Days harvested BEFORE that have data but no macro
snapshots, so nightly_forge_v9 replays them on the SVI seed surface (≈70-190%
IV) with no wall cap — the train/serve skew the archive exists to kill.

You cannot recover a LIVE macro state for a past second, but the surface +
walls AS OF each second can be RECONSTRUCTED from recorded option data, replayed
through the SAME assemble_snapshot() the live radar uses (imported, never
reimplemented — zero drift). Two recorded sources, tried in this order per token
in `auto`:

  1. ticks_v9      — the live harvester's tick vault: real book bid/ask -> mid,
                     plus oi. Highest fidelity. Only exists for days/strikes the
                     harvester was actually subscribed to.
  2. candles_1m    — 1-minute candles WITH open interest pulled by
                     `backfill_history.py --with-options`: close as the price
                     proxy, oi as recorded. Covers days/strikes the live
                     harvester missed (Kite keeps minute OI for the *current*
                     expiry back to its listing date).

TOKEN -> STRIKE MAPPING (the part candles cannot supply on their own)
--------------------------------------------------------------------
candles_1m carries price+OI but NOT strike/expiry/lot. Those come from
instrument_snapshots, resolved leniently:

  * Tier 1: AsOfMapper(day)            — the snapshot as it stood on/before day.
  * Tier 2: UNION of every snapshot    — any snap_date, kept per token, filtered
            (used when Tier 1 is empty)  to expiry >= day. Since snapshots
                                         accumulate daily, a day with no snapshot
                                         of its own is still mapped EXACTLY by a
                                         later one from the same expiry week.

If neither tier knows a token (no instrument_snapshots anywhere for it), that
token is skipped — we do NOT guess strike/expiry by parsing the trading symbol,
because a single mis-decode would silently corrupt the very surface this fixes.
Run the live harvester once (it writes a snapshot), or `backfill_history.py
--with-options` (now writes one too), and re-run.

HONEST LIMITATION
-----------------
The harvester subscribes ATM +/- config.PRUNE_STEPS strikes and --with-options
pulls ATM +/- 4 — both NARROWER than the live radar's +/- MACRO_STRIKE_BAND. The
reconstructed surface covers a tighter grid: the ATM IV level and near-spot
walls are real (the point); deep wings never recorded cannot be invented.
Idempotent: INSERT OR REPLACE on (ts_ms, index_name).
"""
from __future__ import annotations
import datetime as dt
import logging
import sqlite3
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core.instruments import AsOfMapper                  # noqa: E402
from macro_gex_v9 import assemble_snapshot, MacroArchive, dealer_sign  # noqa: E402

log = logging.getLogger("backfill_macro")

_TDATE = "date(ts_ms/1000,'unixepoch','localtime')"      # ticks_v9.ts_ms is MS
_CDATE = "date(ts,'unixepoch','localtime')"              # candles_1m.ts is SECONDS

Series = tuple[list[int], list[float], list[float]]       # ts_ms, price, oi


# ----------------------------------------------------------------- day discovery
def _distinct_days(con: sqlite3.Connection, table: str, expr: str) -> list[str]:
    try:
        return [r[0] for r in con.execute(
            f"SELECT DISTINCT {expr} AS d FROM {table} WHERE d IS NOT NULL")]
    except sqlite3.OperationalError:                      # table absent
        return []


def _days_with_macro(con: sqlite3.Connection) -> set[str]:
    try:
        return {r[0] for r in con.execute(
            f"SELECT DISTINCT {_TDATE} FROM macro_snapshots_v9")}
    except sqlite3.OperationalError:
        return set()


# ------------------------------------------------------------- instrument maps
def _union_by_index(con: sqlite3.Connection, day: str) -> dict[str, list[dict]]:
    """Tier 2: every instrument_snapshots row, deduped per token to the latest
    snap_date, filtered to contracts still alive on `day` (expiry >= day). Mirrors
    the row dicts AsOfMapper yields, so callers treat both tiers identically."""
    try:
        rows = con.execute(
            "SELECT token, symbol, name, expiry, strike, itype, lot, step, "
            "exchange, snap_date FROM instrument_snapshots WHERE expiry >= ? "
            "ORDER BY snap_date", (day,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    by_tok: dict[int, dict] = {}                          # latest snap_date wins
    for (tok, sym, name, expiry, strike, itype, lot, step, exch, _sd) in rows:
        by_tok[int(tok)] = {"token": int(tok), "symbol": sym, "name": name,
                            "expiry": dt.date.fromisoformat(expiry),
                            "strike": strike, "itype": itype, "lot": lot,
                            "step": step, "exchange": exch}
    out: dict[str, list[dict]] = {}
    for r in by_tok.values():
        out.setdefault(r["name"], []).append(r)
    return out


def _resolve_by_index(con: sqlite3.Connection, day: str) -> tuple[dict, str]:
    as_of = dt.date.fromisoformat(day)
    try:
        primary = AsOfMapper(as_of, config.DB_PATH).by_index or {}
        snap = AsOfMapper(as_of, config.DB_PATH).snapshot_used
    except Exception:                                     # noqa: BLE001
        primary, snap = {}, None
    union = _union_by_index(con, day)
    merged = dict(union)                                  # base = union (Tier 2)
    merged.update({k: v for k, v in primary.items() if v})  # prefer Tier 1 where present
    tier = ("asof" if snap else "—") + ("+union" if union else "")
    return merged, (f"{snap or 'none<=day'} [{tier}]")


def _spot_token(con: sqlite3.Connection, index: str, day: str) -> int | None:
    r = con.execute(
        "SELECT token FROM spot_tokens WHERE name=? AND snap_date<=? "
        "ORDER BY snap_date DESC LIMIT 1", (index, day)).fetchone()
    if r:
        return int(r[0])
    try:                                                  # candle fallback: spot saved by name
        r = con.execute("SELECT token FROM candle_tokens WHERE name=? AND "
                        "kind='minute' LIMIT 1", (index,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    return int(r[0]) if r else None


# --------------------------------------------------------------- price/oi series
def _tick_series(con: sqlite3.Connection, tokens: list[int], day: str,
                 spot: bool = False) -> dict[int, Series]:
    if not tokens:
        return {}
    qm = ",".join("?" * len(tokens))
    out = {t: ([], [], []) for t in tokens}
    for tok, ts_ms, ltp, bid, ask, oi in con.execute(
            f"SELECT token, ts_ms, ltp, bid, ask, oi FROM ticks_v9 "
            f"WHERE token IN ({qm}) AND {_TDATE}=? ORDER BY ts_ms",
            (*tokens, day)):
        if spot:
            px = ltp or 0.0
        else:                                             # options: require a real
            if not (bid and ask and bid > 0 and ask > 0):  # two-sided book — no stale
                continue                                   # last-price on thin BSE strikes
            px = (bid + ask) / 2.0
        ts_l, px_l, oi_l = out[tok]
        ts_l.append(int(ts_ms)); px_l.append(float(px)); oi_l.append(float(oi or 0.0))
    return out


def _candle_series(con: sqlite3.Connection, tokens: list[int], day: str
                   ) -> dict[int, Series]:
    if not tokens:
        return {}
    qm = ",".join("?" * len(tokens))
    out = {t: ([], [], []) for t in tokens}
    try:
        cur = con.execute(
            f"SELECT token, ts, c, oi FROM candles_1m "
            f"WHERE token IN ({qm}) AND {_CDATE}=? ORDER BY ts", (*tokens, day))
    except sqlite3.OperationalError:
        return out
    for tok, ts_s, c, oi in cur:
        ts_l, px_l, oi_l = out[tok]
        ts_l.append(int(ts_s) * 1000)                     # seconds -> ms (macro clock)
        px_l.append(float(c or 0.0)); oi_l.append(float(oi or 0.0))
    return out


def _merge(tick: dict[int, Series], cand: dict[int, Series], tokens: list[int],
           source: str) -> tuple[dict[int, Series], int, int]:
    """Per token: ticks if present (auto/ticks), else candles (auto/candles).
    Returns (series, n_from_ticks, n_from_candles)."""
    out: dict[int, Series] = {}
    nt = nc = 0
    for t in tokens:
        ts = tick.get(t, ([], [], []))
        cs = cand.get(t, ([], [], []))
        if source != "candles" and ts[0]:
            out[t] = ts; nt += 1
        elif source != "ticks" and cs[0]:
            out[t] = cs; nc += 1
        else:
            out[t] = ([], [], [])
    return out, nt, nc


def _latest(series: Series, ts_ms: int):
    ts_l, px_l, oi_l = series
    if not ts_l:
        return None
    i = bisect_right(ts_l, ts_ms) - 1
    return (px_l[i], oi_l[i]) if i >= 0 else None


# ------------------------------------------------------------------- per day
def backfill_day(rcon, archive, day, s_call, s_put, step_s, source, dry,
                 only=None) -> int:
    as_of = dt.date.fromisoformat(day)
    by_index, prov = _resolve_by_index(rcon, day)
    if not by_index:
        log.warning("%s: no instrument snapshot maps this day (run the harvester "
                    "or backfill_history --with-options once) — skipped", day)
        return 0
    log.info("%s: instrument map %s", day, prov)

    day_total = 0
    for index in config.INDEX_ORDER:
        if only and index not in only:
            continue
        rows = by_index.get(index, [])
        if not rows:
            continue
        exps = sorted({r["expiry"] for r in rows if r["expiry"] >= as_of})
        if not exps:
            continue
        exp = exps[0]
        chain = [r for r in rows if r["expiry"] == exp and r["itype"] in ("CE", "PE")]
        if len(chain) < 8:
            continue
        lot = chain[0].get("lot") or config.INDICES[index]["lot_fallback"]
        tok_meta = {int(r["token"]): (float(r["strike"]), r["itype"] == "CE")
                    for r in chain}

        sp_tok = _spot_token(rcon, index, day)
        if sp_tok is None:
            continue
        opt_toks = list(tok_meta)

        # build merged series per option token + spot (ticks first, candle fallback)
        opt = _merge(_tick_series(rcon, opt_toks, day),
                     _candle_series(rcon, opt_toks, day), opt_toks, source)[0]
        sp_t = _tick_series(rcon, [sp_tok], day, spot=True).get(sp_tok, ([], [], []))
        sp_c = _candle_series(rcon, [sp_tok], day).get(sp_tok, ([], [], []))
        sp_ser = sp_t if (source != "candles" and sp_t[0]) else (
            sp_c if source != "ticks" else ([], [], []))
        if not sp_ser[0]:
            continue

        opt_ts = [t for s in opt.values() for t in s[0]]
        if not opt_ts:
            continue
        start = max(min(opt_ts), sp_ser[0][0])
        end = min(max(opt_ts), sp_ser[0][-1])
        if end <= start:
            continue

        dte = max((exp - as_of).days, 0) + config.DTE_PART_DAY
        nt_total = sum(1 for t in opt_toks if opt.get(t, ([],))[0])
        n_idx = 0
        ts_ms = start
        while ts_ms <= end:
            sp = _latest(sp_ser, ts_ms)
            if sp and sp[0] > 0:
                K, mid, oi, is_call = [], [], [], []
                for tok, (strike, is_c) in tok_meta.items():
                    v = _latest(opt.get(tok, ([], [], [])), ts_ms)
                    if v and v[0] > 0:
                        K.append(strike); mid.append(v[0]); oi.append(v[1]); is_call.append(is_c)
                if len(K) >= 8:
                    res = assemble_snapshot(
                        ts=ts_ms / 1000.0, index=index, spot=sp[0], exp=exp,
                        dte=dte, K=K, mid=mid, oi=oi, is_call=is_call,
                        lot=lot, s_call=s_call, s_put=s_put)
                    if res is not None:                   # None ⇒ too many strikes pinned
                        if not dry:
                            archive.write(res[0])
                        n_idx += 1
            ts_ms += step_s * 1000
        if n_idx:
            log.info("%s %-10s exp %s: %d snapshots from %d tokens (%s)",
                     day, index, exp, n_idx, nt_total, "dry-run" if dry else "written")
        day_total += n_idx
    return day_total


def main():
    config.setup_logging("backfill_macro")
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    force = "--force" in argv
    step_s = config.MACRO_LOOP_S
    source = "auto"
    n_days = None
    only = None
    consumed: set[int] = set()                            # indices of flag VALUES
    for i, a in enumerate(argv):
        if a == "--source" and i + 1 < len(argv):
            if argv[i + 1] in ("auto", "ticks", "candles"):
                source = argv[i + 1]
            consumed.add(i + 1)
        elif a == "--days" and i + 1 < len(argv):
            if argv[i + 1].isdigit():
                n_days = int(argv[i + 1])
            consumed.add(i + 1)
        elif a == "--only" and i + 1 < len(argv):
            only = {s.strip().upper() for s in argv[i + 1].split(",") if s.strip()}
            consumed.add(i + 1)
    explicit = [a for i, a in enumerate(argv)
                if not a.startswith("--") and i not in consumed]

    rcon = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
    rcon.execute("PRAGMA busy_timeout=5000;")

    if explicit:
        days = sorted(explicit)
    else:
        all_days = sorted(set(_distinct_days(rcon, "ticks_v9", _TDATE))
                          | set(_distinct_days(rcon, "candles_1m", _CDATE)))
        have = set() if force else _days_with_macro(rcon)
        days = [d for d in all_days if d not in have]
        if n_days is not None:
            days = days[-n_days:]

    if not days:
        log.info("nothing to backfill (every recorded day already has macro "
                 "snapshots — use --force to rebuild)")
        return

    log.info("backfilling %d day(s) [source=%s%s]: %s%s", len(days), source,
             f", only={','.join(sorted(only))}" if only else "",
             ", ".join(days[:8]), " ..." if len(days) > 8 else "")
    s_call, s_put = dealer_sign()
    archive = MacroArchive()
    grand = 0
    for day in days:
        try:
            grand += backfill_day(rcon, archive, day, s_call, s_put, step_s,
                                  source, dry, only)
        except Exception:                                 # noqa: BLE001
            log.exception("%s: backfill failed", day)
    log.info("DONE: %d snapshot rows %s across %d day(s) -> %s", grand,
             "computed (dry-run)" if dry else "written", len(days),
             config.DB_PATH.name)


if __name__ == "__main__":
    main()