"""
APEX OMNI v9 — INSTRUMENTS (audit §4/§6 leap: the point-in-time vault key)
==========================================================================
Two jobs:
  1. LiveMapper — today's chain from the Kite instrument dump, cached BY DATE
     (v8 invalidated by hour-of-day and could serve yesterday's strikes all
     morning). Lot size, strike step and expiry are read from the dump, never
     hardcoded — that's how the Jan-2026 lot change was absorbed for free.
  2. Snapshots — every run persists today's option metadata into
     `instrument_snapshots`. AsOfMapper(date) then rebuilds the chain THAT
     DAY actually had, which is the single fix that lets the nightly forge
     reopen the older volumes of your archive instead of training on zeros.
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import pickle
import sqlite3
from pathlib import Path

import config

log = logging.getLogger("instruments")

SNAP_SCHEMA = """CREATE TABLE IF NOT EXISTS instrument_snapshots (
    snap_date TEXT, token INTEGER, symbol TEXT, name TEXT, expiry TEXT,
    strike REAL, itype TEXT, lot INTEGER, step REAL, exchange TEXT,
    PRIMARY KEY (snap_date, token));
CREATE INDEX IF NOT EXISTS idx_snap_name ON instrument_snapshots
    (snap_date, name, expiry);"""


def _pick_expiry(expiries: list[dt.date], today: dt.date) -> dt.date | None:
    fut = sorted(e for e in expiries if e >= today)
    return fut[0] if fut else None


class BaseMapper:
    """rows: list of dicts with name, expiry(date), strike, itype, token,
    symbol, lot, step, exchange — source differs (dump vs snapshot)."""
    def __init__(self, rows: list[dict], today: dt.date):
        self.today = today
        self.by_index: dict[str, list[dict]] = {}
        for r in rows:
            self.by_index.setdefault(r["name"], []).append(r)

    def chain(self, index: str, spot: float) -> dict | None:
        rows = self.by_index.get(index)
        if not rows or spot <= 0:
            return None
        exp = _pick_expiry(sorted({r["expiry"] for r in rows}), self.today)
        if exp is None:
            return None
        rows = [r for r in rows if r["expiry"] == exp]
        step = rows[0]["step"] or config.INDICES[index]["strike_step"]
        lot = rows[0]["lot"] or config.INDICES[index]["lot_fallback"]
        atm = round(spot / step) * step
        legs = {}
        want = {"atm_ce": (atm, "CE"), "atm_pe": (atm, "PE"),
                "otm_ce": (atm + step, "CE"), "otm_pe": (atm - step, "PE")}
        idx = {(r["strike"], r["itype"]): r for r in rows}
        for leg, key in want.items():
            r = idx.get(key)
            if r:
                legs[leg] = {"token": r["token"], "symbol": r["symbol"],
                             "strike": key[0], "itype": key[1]}
        dte = max((exp - self.today).days, 0) + config.DTE_PART_DAY
        return {"expiry": str(exp), "dte": dte,
                "T": max(dte, 0.3) / 365.0, "lot": int(lot),
                "step": float(step),
                "is_weekly": config.INDICES[index]["weekly"],
                "atm": atm, "legs": legs}

    def hierarchy(self, index: str, spot: float, direction: str,
                  depth: int | None = None) -> list[dict]:
        """Preferred-first strike ladder for the affordability walker:
        ATM → 1·step OTM → … (long CE walks up, long PE walks down)."""
        depth = depth or config.HIERARCHY_DEPTH
        rows = self.by_index.get(index)
        if not rows:
            return []
        exp = _pick_expiry(sorted({r["expiry"] for r in rows}), self.today)
        rows = [r for r in rows if r["expiry"] == exp and r["itype"] == direction]
        step = rows[0]["step"] or config.INDICES[index]["strike_step"]
        lot = rows[0]["lot"] or config.INDICES[index]["lot_fallback"]
        atm = round(spot / step) * step
        idx = {r["strike"]: r for r in rows}
        out = []
        for i in range(depth):
            k = atm + i * step if direction == "CE" else atm - i * step
            r = idx.get(k)
            if r:
                out.append({"token": r["token"], "symbol": r["symbol"],
                            "strike": k, "lot": int(lot),
                            "exchange": r["exchange"]})
        return out


class LiveMapper(BaseMapper):
    def __init__(self, kite):
        today = dt.date.today()
        cache = config.STATE_DIR / f"instruments_{today}.pkl"   # date-keyed ★
        if cache.exists():
            rows = pickle.loads(cache.read_bytes())
        else:
            rows = []
            for exch in ("NFO", "BFO"):
                for ins in kite.instruments(exch):
                    if ins.get("name") in config.INDICES and \
                       ins.get("instrument_type") in ("CE", "PE"):
                        rows.append({"name": ins["name"],
                                     "expiry": ins["expiry"]
                                     if isinstance(ins["expiry"], dt.date)
                                     else dt.date.fromisoformat(str(ins["expiry"])[:10]),
                                     "strike": float(ins["strike"]),
                                     "itype": ins["instrument_type"],
                                     "token": int(ins["instrument_token"]),
                                     "symbol": ins["tradingsymbol"],
                                     "lot": int(ins["lot_size"]),
                                     "step": None, "exchange": exch})
            # infer strike step per (name, expiry) from the dump itself
            from collections import defaultdict
            strikes = defaultdict(set)
            for r in rows:
                strikes[(r["name"], r["expiry"])].add(r["strike"])
            for r in rows:
                ss = sorted(strikes[(r["name"], r["expiry"])])
                diffs = [b - a for a, b in zip(ss, ss[1:]) if b > a]
                r["step"] = min(diffs) if diffs else \
                    config.INDICES[r["name"]]["strike_step"]
            cache.write_bytes(pickle.dumps(rows, protocol=5))
            log.info("Instrument dump cached for %s (%d option rows)",
                     today, len(rows))
        super().__init__(rows, today)
        self._rows = rows

    def write_snapshot(self, db_path: Path | None = None):
        """Persist TODAY's chain metadata — the forge's time machine."""
        con = sqlite3.connect(db_path or config.DB_PATH)
        con.executescript(SNAP_SCHEMA)
        con.executemany(
            "INSERT OR REPLACE INTO instrument_snapshots VALUES "
            "(?,?,?,?,?,?,?,?,?,?)",
            [(str(self.today), r["token"], r["symbol"], r["name"],
              str(r["expiry"]), r["strike"], r["itype"], r["lot"],
              r["step"], r["exchange"]) for r in self._rows])
        con.commit(); con.close()
        log.info("instrument_snapshots written for %s", self.today)


class AsOfMapper(BaseMapper):
    """Rebuilds the chain as it existed on `as_of` — historical replay truth."""
    def __init__(self, as_of: dt.date, db_path: Path | None = None):
        con = sqlite3.connect(db_path or config.DB_PATH)
        con.executescript(SNAP_SCHEMA)
        snap = con.execute("SELECT MAX(snap_date) FROM instrument_snapshots "
                           "WHERE snap_date <= ?", (str(as_of),)).fetchone()[0]
        rows = []
        if snap:
            for (token, symbol, name, expiry, strike, itype, lot, step,
                 exch) in con.execute(
                    "SELECT token,symbol,name,expiry,strike,itype,lot,step,"
                    "exchange FROM instrument_snapshots WHERE snap_date=?",
                    (snap,)):
                rows.append({"token": token, "symbol": symbol, "name": name,
                             "expiry": dt.date.fromisoformat(expiry),
                             "strike": strike, "itype": itype, "lot": lot,
                             "step": step, "exchange": exch})
        con.close()
        self.snapshot_used = snap
        super().__init__(rows, as_of)
