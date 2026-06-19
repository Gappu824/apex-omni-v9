"""
APEX OMNI v9 — HISTORICAL BACKFILL (real Kite candles, the data multiplier)
===========================================================================
    python tools/backfill_history.py [days] [--with-options]

Pulls REAL history through the Kite historical API (included in the ₹500
plan) into `candles_1m` / `candles_day` inside the vault:

  * 1-minute candles for all six index spots + INDIA VIX (minute data goes
    back ~3 years; the API caps minute requests at 60 days each, so this
    loops in chunks at ≤2 req/s, well under the 3 req/s historical limit).
  * Day candles much further back (up to ~2000 days per request) for
    regime/levels context.
  * --with-options: minute candles WITH OPEN INTEREST (oi=1) for the
    CURRENT chain around ATM. Know the hard truth this tool works around:
    Kite keeps NO minute data for expired option contracts — which is
    exactly why the live harvester's tick vault is irreplaceable. This flag
    simply back-fills the current expiry from its listing date so tonight's
    forge isn't blind to the part of this week you weren't recording.

Resumable: each (token, interval) continues from its last stored candle.
"""
from __future__ import annotations
import datetime as dt
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                      # noqa: E402
from core.instruments import LiveMapper            # noqa: E402

log = logging.getLogger("backfill")

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS candles_1m (
    token INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL,
    vol REAL, oi REAL, PRIMARY KEY (token, ts));
CREATE TABLE IF NOT EXISTS candles_day (
    token INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL,
    vol REAL, oi REAL, PRIMARY KEY (token, ts));
CREATE TABLE IF NOT EXISTS candle_tokens (
    token INTEGER PRIMARY KEY, name TEXT, kind TEXT);
"""
MIN_CHUNK_DAYS = 55          # API hard cap is 60/request for minute candles
RATE_SLEEP = 0.55            # ≤2 req/s (historical limit is 3 req/s)


def last_ts(con, table, token):
    r = con.execute(f"SELECT MAX(ts) FROM {table} WHERE token=?",
                    (token,)).fetchone()
    return r[0] if r and r[0] else None


def pull(kite, con, token, name, interval, frm, to, oi=False):
    table = "candles_1m" if interval == "minute" else "candles_day"
    chunk = dt.timedelta(days=MIN_CHUNK_DAYS if interval == "minute" else 1900)
    got = 0
    cur = frm
    while cur < to:
        end = min(cur + chunk, to)
        try:
            rows = kite.historical_data(token, cur, end, interval, oi=oi)
        except Exception as e:                          # noqa: BLE001
            log.warning("%s %s %s→%s: %s", name, interval, cur, end, e)
            rows = []
        if rows:
            def _cms(d):
                try:
                    return int(d.timestamp())
                except (OSError, OverflowError, ValueError):
                    return 0
            con.executemany(
                f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?,?,?,?)",
                [(token, _cms(r["date"]), r["open"], r["high"],
                  r["low"], r["close"], r.get("volume", 0),
                  r.get("oi", 0)) for r in rows])
            con.commit()
            got += len(rows)
        cur = end + dt.timedelta(days=1)
        time.sleep(RATE_SLEEP)
    if got:
        con.execute("INSERT OR REPLACE INTO candle_tokens VALUES (?,?,?)",
                    (token, name, interval))
        con.commit()
    log.info("%-22s %-6s +%d candles", name, interval, got)
    return got


def main():
    config.setup_logging("backfill")
    if KiteConnect is None:
        sys.exit("pip install kiteconnect first")
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
        else config.BACKFILL_DAYS
    with_opts = "--with-options" in sys.argv
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    kite.set_access_token(config.KITE_ACCESS_TOKEN)
    con = sqlite3.connect(config.DB_PATH)
    con.executescript(SCHEMA)
    today = dt.date.today()

    syms = {n: v["spot_symbol"] for n, v in config.INDICES.items()}
    syms["INDIA VIX"] = config.VIX_SYMBOL
    data = kite.ltp(list(syms.values()))
    total = 0
    for name, sym in syms.items():
        d = data.get(sym)
        if not d:
            continue
        tok = int(d["instrument_token"])
        lt = last_ts(con, "candles_1m", tok)
        frm = (dt.date.fromtimestamp(lt) if lt
               else today - dt.timedelta(days=days))
        total += pull(kite, con, tok, name, "minute", frm, today)
        ld = last_ts(con, "candles_day", tok)
        dfrm = (dt.date.fromtimestamp(ld) if ld
                else today - dt.timedelta(days=min(days * 8, 1900)))
        total += pull(kite, con, tok, name, "day", dfrm, today)

    if with_opts:
        mapper = LiveMapper(kite)
        for idx in config.TRADABLE:
            spot = float(data[syms[idx]]["last_price"])
            ch = mapper.chain(idx, spot)
            if not ch:
                continue
            rows = [r for r in mapper.by_index[idx]
                    if str(r["expiry"]) == ch["expiry"]
                    and abs(r["strike"] - ch["atm"]) <= 4 * ch["step"]]
            frm = today - dt.timedelta(days=min(days, 30))
            for r in rows:
                lt = last_ts(con, "candles_1m", r["token"])
                f2 = dt.date.fromtimestamp(lt) if lt else frm
                total += pull(kite, con, r["token"],
                              r["symbol"], "minute", f2, today, oi=True)
    print(f"\nbackfill complete: {total} candles added → {config.DB_PATH}")
    print("note: expired option contracts have no minute history at Kite — "
          "your live tick vault is the only place that data will ever exist.")


if __name__ == "__main__":
    main()
