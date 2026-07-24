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


def _sel_today(name: str, today: dt.date) -> dt.date:
    """Effective 'today' for expiry selection. v9.7.1 AUDIT S5-F2: MCX option
    chains rolled only when the expiry PASSED — trading/harvesting the dying
    contract through its own expiry day (ITM MCX options DEVOLVE into futures
    at expiry; the microstructure of that day is a different animal). Equity
    0DTE is the game and stays on `e >= today`; commodities roll
    COMMODITY_OPT_ROLL_DAYS early."""
    if name in getattr(config, "COMMODITIES", {}):
        return today + dt.timedelta(
            days=int(getattr(config, "COMMODITY_OPT_ROLL_DAYS", 2)))
    return today


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
        exp = _pick_expiry(sorted({r["expiry"] for r in rows}),
                           _sel_today(index, self.today))
        if exp is None:
            return None
        rows = [r for r in rows if r["expiry"] == exp]
        _spec = _spec_for(index) or {}
        step = rows[0]["step"] or _spec.get("strike_step", 1.0)
        lot = rows[0]["lot"] or _spec.get("lot_fallback", 1)
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
                "is_weekly": bool(_spec.get("weekly", False)),
                "atm": atm, "legs": legs}

    def hierarchy(self, index: str, spot: float, direction: str,
                  depth: int | None = None) -> list[dict]:
        """Preferred-first strike ladder for the affordability walker:
        ATM → 1·step OTM → … (long CE walks up, long PE walks down)."""
        depth = depth or config.HIERARCHY_DEPTH
        rows = self.by_index.get(index)
        if not rows:
            return []
        exp = _pick_expiry(sorted({r["expiry"] for r in rows}),
                           _sel_today(index, self.today))
        rows = [r for r in rows if r["expiry"] == exp and r["itype"] == direction]
        if not rows:      # AUDIT S5-F1: a one-sided dump slice used to
            return []     # IndexError on rows[0] straight into the live loop
        _spec = _spec_for(index) or {}
        step = rows[0]["step"] or _spec.get("strike_step", 1.0)
        lot = rows[0]["lot"] or _spec.get("lot_fallback", 1)
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


_LOT_WARNED: set = set()


def _check_lot(name: str, dump_lot, spec: dict):
    """AUDIT (2026-07-23): `lot = rows[0]["lot"] or lot_fallback` means a dump
    value of 1 (TRUTHY) shadows the config's known contract size — MCX crude
    booked at lot=1 costs Rs 647 where a 100-barrel lot costs Rs 64,775. That
    is a 100x sizing error in whichever direction the dump is wrong, and it
    decides whether a book can afford a contract at all. We do NOT silently
    override (if the dump is right, forcing the fallback would block every
    trade); we make the disagreement impossible to miss, once per contract."""
    try:
        fb = spec.get("lot_fallback")
        if not fb or not dump_lot or int(dump_lot) == int(fb):
            return
        if name in _LOT_WARNED:
            return
        _LOT_WARNED.add(name)
        log.warning("%s CONTRACT SIZE DISAGREEMENT: instrument dump says "
                    "lot=%s, config lot_fallback says %s. Sizing uses the DUMP "
                    "(%s). If the dump is wrong every %s position is %.0fx "
                    "mis-sized — verify the contract size on Kite before this "
                    "book ever trades real money.", name, dump_lot, fb,
                    dump_lot, name, int(fb) / max(int(dump_lot), 1))
    except Exception:                                          # noqa: BLE001
        pass


def _spec_for(name: str) -> dict | None:
    """Contract spec (strike_step, lot_fallback) for an index OR a commodity —
    the harvest universe is INDICES ∪ COMMODITIES. Returns None if unknown."""
    if name in config.INDICES:
        return config.INDICES[name]
    return getattr(config, "COMMODITIES", {}).get(name)


def _harvest_universe() -> set:
    """Every underlying we capture options for: all indices plus the commodities
    flagged for harvest. Trading universe is separate and unaffected."""
    return set(config.INDICES) | set(getattr(config, "HARVEST_COMMODITIES", []))


class LiveMapper(BaseMapper):
    def __init__(self, kite):
        today = dt.date.today()
        cache = config.STATE_DIR / f"instruments_{today}.pkl"   # date-keyed ★
        if cache.exists():
            rows = pickle.loads(cache.read_bytes())
        else:
            rows = []
            universe = _harvest_universe()
            # equity option exchanges + MCX for commodities (only scanned if any
            # commodity is flagged for harvest — zero cost otherwise).
            exchanges = ["NFO", "BFO"]
            if getattr(config, "HARVEST_COMMODITIES", []):
                exchanges.append("MCX")
            for exch in exchanges:
                for ins in kite.instruments(exch):
                    if ins.get("name") in universe and \
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
                _spec = _spec_for(r["name"]) or {}
                r["step"] = min(diffs) if diffs else \
                    _spec.get("strike_step", 1.0)
            cache.write_bytes(pickle.dumps(rows, protocol=5))
            log.info("Instrument dump cached for %s (%d option rows across %s)",
                     today, len(rows), "/".join(exchanges))
        super().__init__(rows, today)
        self._rows = rows
        # v9.7.1: resolve each harvested commodity's FRONT-MONTH FUTURE (its
        # "spot" — there is no commodity index). Scanned from the same MCX dump;
        # the nearest non-expired future per name. Empty for equity-only setups.
        self.commodity_futures: dict[str, dict] = {}
        if getattr(config, "HARVEST_COMMODITIES", []):
            try:
                self._resolve_commodity_futures(kite, today)
            except Exception as e:                            # noqa: BLE001
                log.warning("commodity futures resolve failed: %s "
                            "(options still harvested; spot-track degraded)", e)
            try:
                self._reconcile_commodity_lots()
            except Exception as e:                            # noqa: BLE001
                log.warning("commodity lot reconcile failed: %s", e)

    def _reconcile_commodity_lots(self):
        """DYNAMIC contract multiplier, straight from the Kite dump.

        An MCX option is an option ON A FUTURES CONTRACT, and its premium is
        quoted in the futures' price unit — so the economic multiplier of an
        option position is the UNDERLYING FUTURE's lot_size, not the option
        row's own lot_size (which Kite reports as 1 for MCX). Booking crude at
        lot=1 valued a position at Rs 647 where the contract is 100 barrels
        (Rs 64,775) — a 100x sizing error that decides affordability outright.

        We already resolve each commodity's front-month future from the same
        live dump, so the multiplier is derived from the API, not hardcoded:
            option lot  <-  front-month FUTURES lot_size
        Config's lot_fallback is used only if the dump yields nothing, and any
        disagreement between the two is logged. Because the corrected lot is
        written into self._rows BEFORE write_snapshot(), the forge's AsOfMapper
        replay inherits identical sizing with no extra code."""
        names = set(getattr(config, "HARVEST_COMMODITIES", []))
        if not names or not self._rows:
            return
        applied: dict[str, tuple] = {}
        for name in names:
            fut = (self.commodity_futures or {}).get(name) or {}
            fut_lot = int(fut.get("lot") or 0)
            spec = (getattr(config, "COMMODITIES", {}) or {}).get(name, {})
            cfg_lot = int(spec.get("lot_fallback") or 0)
            src, mult = ("futures dump", fut_lot) if fut_lot > 1 else \
                        ("config fallback", cfg_lot) if cfg_lot > 1 else ("", 0)
            if mult <= 1:
                continue
            if fut_lot > 1 and cfg_lot > 1 and fut_lot != cfg_lot:
                log.warning("%s: LIVE futures lot_size=%d disagrees with "
                            "config lot_fallback=%d — using the LIVE value "
                            "(the exchange is authoritative); update config.",
                            name, fut_lot, cfg_lot)
            n = 0
            for r in self._rows:
                if r.get("name") == name and r.get("itype") in ("CE", "PE") \
                        and int(r.get("lot") or 1) < mult:
                    r["lot"] = mult
                    n += 1
            if n:
                applied[name] = (mult, src, n)
        for name, (mult, src, n) in applied.items():
            log.warning("%s CONTRACT MULTIPLIER RESOLVED: option rows carried "
                        "lot=1; using %d from the %s (%d option rows "
                        "corrected). A position now costs premium x %d — this "
                        "is what the exchange actually charges.",
                        name, mult, src, n, mult)

    def _resolve_commodity_futures(self, kite, today):
        names = set(getattr(config, "HARVEST_COMMODITIES", []))
        if not names:
            return
        fcache = config.STATE_DIR / f"commodity_futs_{today}.pkl"
        if fcache.exists():
            self.commodity_futures = pickle.loads(fcache.read_bytes())
            return
        futs: dict[str, list] = {}
        for ins in kite.instruments("MCX"):
            if ins.get("name") in names and ins.get("instrument_type") == "FUT":
                exp = ins["expiry"] if isinstance(ins["expiry"], dt.date) \
                    else dt.date.fromisoformat(str(ins["expiry"])[:10])
                if exp >= today + dt.timedelta(days=int(getattr(
                        config, "COMMODITY_FUT_ROLL_DAYS", 2))):
                    # S5-F2: the front-month "spot" must not be a contract in
                    # its tender window — liquidity has already migrated.
                    futs.setdefault(ins["name"], []).append(
                        {"token": int(ins["instrument_token"]),
                         "symbol": ins["tradingsymbol"], "expiry": exp,
                         "lot": int(ins["lot_size"])})
        for name, lst in futs.items():
            lst.sort(key=lambda r: r["expiry"])       # front month first
            self.commodity_futures[name] = lst[0]
        fcache.write_bytes(pickle.dumps(self.commodity_futures, protocol=5))
        log.info("Commodity front-month futures: %s",
                 {k: v["symbol"] for k, v in self.commodity_futures.items()})

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