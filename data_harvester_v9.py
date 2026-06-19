"""
APEX OMNI v9 — DATA HARVESTER (audit §4 leaps)
==============================================
What changed vs v6:
  * POINT-IN-TIME VAULT. Raw per-tick DELTAS only (ltp, book, vol_delta, oi,
    iceberg flag) — derived physics are NOT stored; the forge replays raw
    ticks through the SAME StateBuilder the live brain uses, so features can
    never diverge between training and serving again.
  * Timestamps: exchange timestamp when present, else local — both stored as
    INTEGER epoch-milliseconds. WAL mode + (token, ts_ms) index: the nightly
    full-table ORDER BY sort is gone.
  * instrument_snapshots written at startup and again after the close — the
    one table that lets AsOfMapper reopen your archive's older volumes.
  * spot_tokens meta table so the forge knows which token was which index's
    spot on any given day.
  * Subscriptions FOLLOW the ATM and now also PRUNE: legs further than
    ±PRUNE_STEPS from the current ATM are unsubscribed (v8 only ever grew).
  * Queue-depth telemetry — if the writer falls behind the WebSocket you'll
    see it in the log before it becomes a stale-feed halt.

Run:  python data_harvester_v9.py
"""
from __future__ import annotations
import datetime as dt
import logging
import json
import queue
import sqlite3
import threading
import time

import config
from apex_ipc_core import BinaryRingBuffer
from core.instruments import LiveMapper

log = logging.getLogger("harvester")

try:
    from kiteconnect import KiteConnect, KiteTicker
    HAVE_KITE = True
except Exception:                                     # noqa: BLE001
    HAVE_KITE = False


TICKS_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS ticks_v9 (
    ts_ms INTEGER, ts_local_ms INTEGER, token INTEGER, ltp REAL,
    bid REAL, ask REAL, bid_qty REAL, ask_qty REAL,
    vol_delta REAL, oi REAL, iceberg INTEGER);
CREATE INDEX IF NOT EXISTS idx_ticks_token_ts ON ticks_v9 (token, ts_ms);
CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks_v9 (ts_ms);
CREATE TABLE IF NOT EXISTS spot_tokens (
    snap_date TEXT, name TEXT, token INTEGER,
    PRIMARY KEY (snap_date, name));
"""


class VaultKeeper:
    def __init__(self):
        self.con = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.con.executescript(TICKS_SCHEMA)
        self.rows: list[tuple] = []
        self.lock = threading.Lock()

    def add(self, row: tuple):
        with self.lock:
            self.rows.append(row)
            if len(self.rows) >= config.DB_BATCH_ROWS:
                self._flush()

    def _flush(self):
        if not self.rows:
            return
        self.con.executemany(
            "INSERT INTO ticks_v9 VALUES (?,?,?,?,?,?,?,?,?,?,?)", self.rows)
        self.con.commit()
        self.rows.clear()

    def flush(self):
        with self.lock:
            self._flush()

    def record_spot_tokens(self, mapping: dict[str, int]):
        today = str(dt.date.today())
        self.con.executemany(
            "INSERT OR REPLACE INTO spot_tokens VALUES (?,?,?)",
            [(today, n, t) for n, t in mapping.items()])
        self.con.commit()


class Harvester:
    def __init__(self):
        if not HAVE_KITE:
            raise SystemExit("kiteconnect not installed — pip install kiteconnect")
        if not (config.KITE_API_KEY and config.KITE_ACCESS_TOKEN):
            raise SystemExit("Set KITE_API_KEY / KITE_ACCESS_TOKEN env vars "
                             "(token regenerates daily under the SEBI logout).")
        self.kite = KiteConnect(api_key=config.KITE_API_KEY)
        self.kite.set_access_token(config.KITE_ACCESS_TOKEN)
        self.mapper = LiveMapper(self.kite)
        self.mapper.write_snapshot()                      # time-machine entry ★
        self.vault = VaultKeeper()
        self.ring = BinaryRingBuffer(writer=True)
        self.q: queue.Queue = queue.Queue()
        self.last_cumvol: dict[int, float] = {}
        self.last_bid: dict[int, tuple] = {}              # (price, qty)
        self.snap: dict[int, dict] = {}                   # latest per token
        self.spot_tokens: dict[str, int] = {}
        self.token_role: dict[int, tuple] = {}            # token → (idx, leg, strike)
        self.subscribed: set[int] = set()
        self.chains: dict[str, dict] = {}
        self._resolve_spot_tokens()
        self.vault.record_spot_tokens(self.spot_tokens)
        self.kws = KiteTicker(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN)
        self.kws.on_ticks = lambda ws, ticks: self.q.put(ticks)
        self.kws.on_order_update = lambda ws, d: self._order_update(d)
        self.kws.on_connect = self._on_connect
        self.kws.on_close = lambda ws, code, reason: log.warning(
            "WS closed: %s %s (auto-reconnect handles retry)", code, reason)
        self.snapshot_written_pm = False

    def _resolve_spot_tokens(self):
        syms = [v["spot_symbol"] for v in config.INDICES.values()] + \
            [config.VIX_SYMBOL]
        data = self.kite.ltp(syms)
        for name, cfgv in config.INDICES.items():
            d = data.get(cfgv["spot_symbol"])
            if d:
                tok = int(d["instrument_token"])
                self.spot_tokens[name] = tok
                self.token_role[tok] = (name, "spot", 0.0)
        v = data.get(config.VIX_SYMBOL)
        self.vix_token = int(v["instrument_token"]) if v else None
        if self.vix_token:
            self.token_role[self.vix_token] = ("VIX", "spot", 0.0)
        log.info("Spot tokens: %s | VIX token %s", self.spot_tokens,
                 self.vix_token)

    # ------------------------------------------------------------ ws wiring
    def _order_update(self, d: dict):
        """WS order pushes (same payload as postbacks; covers ALL orders,
        GTT-triggered ones included) → tiny json the engine reads instead
        of burning REST polls."""
        try:
            book = {}
            if config.ORDER_UPDATES_PATH.exists():
                book = json.loads(config.ORDER_UPDATES_PATH.read_text())
            oid = str(d.get("order_id"))
            book[oid] = {"status": d.get("status"),
                         "avg": d.get("average_price"),
                         "filled": d.get("filled_quantity"),
                         "msg": d.get("status_message"),
                         "ts": time.time()}
            if len(book) > 200:
                book = dict(sorted(book.items(),
                                   key=lambda kv: kv[1]["ts"])[-200:])
            tmp = config.ORDER_UPDATES_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(book))
            tmp.replace(config.ORDER_UPDATES_PATH)
        except Exception as e:                         # noqa: BLE001
            log.warning("order_update write: %s", e)

    def _on_connect(self, ws, response):
        toks = list(self.spot_tokens.values())
        if self.vix_token:
            toks.append(self.vix_token)
        ws.subscribe(toks)
        ws.set_mode(ws.MODE_FULL, toks)
        self.subscribed.update(toks)
        log.info("WS connected — %d spot tokens subscribed", len(toks))

    def _resubscribe(self, index: str, spot: float):
        ch = self.mapper.chain(index, spot)
        if not ch:
            return
        self.chains[index] = ch
        want: set[int] = set()
        step, atm = ch["step"], ch["atm"]
        rows = self.mapper.by_index.get(index, [])
        exp_rows = {(r["strike"], r["itype"]): r for r in rows
                    if str(r["expiry"]) == ch["expiry"]}
        for k_off in range(-config.PRUNE_STEPS, config.PRUNE_STEPS + 1):
            for it in ("CE", "PE"):
                r = exp_rows.get((atm + k_off * step, it))
                if r:
                    want.add(r["token"])
                    self.token_role[r["token"]] = (index, it, r["strike"])
        for leg, info in ch["legs"].items():
            self.token_role[info["token"]] = (index, leg, info["strike"])
            want.add(info["token"])
        stale = {t for t, role in self.token_role.items()
                 if role[0] == index and role[1] != "spot"} - want
        new = want - self.subscribed
        if new:
            self.kws.subscribe(list(new))
            self.kws.set_mode(self.kws.MODE_FULL, list(new))
            self.subscribed |= new
        if stale:
            self.kws.unsubscribe(list(stale))            # ★ pruning
            self.subscribed -= stale
            for t in stale:
                self.token_role.pop(t, None)
        if new or stale:
            log.info("%s ATM %s: +%d / -%d tokens (now %d total)",
                     index, atm, len(new), len(stale), len(self.subscribed))

    # ------------------------------------------------------------ processing
    @staticmethod
    def _safe_ms(ex, now_ms: int) -> int:
        """Kite occasionally sends a zero/sentinel exchange_timestamp (often
        at connect). datetime.timestamp() raises OSError [Errno 22] on
        Windows for any datetime at/before the epoch, so guard it: a bad or
        pre-2001 timestamp falls back to local wall-clock time."""
        if not isinstance(ex, dt.datetime):
            return now_ms
        try:
            ms = int(ex.timestamp() * 1000)
        except (OSError, OverflowError, ValueError):
            return now_ms
        # sanity floor: anything before 2001 is a sentinel, not a real tick time
        return ms if ms > 978_307_200_000 else now_ms

    def _process(self, tick: dict):
        tok = int(tick["instrument_token"])
        now_ms = int(time.time() * 1000)
        ex = tick.get("exchange_timestamp") or tick.get("last_trade_time")
        ts_ms = self._safe_ms(ex, now_ms)
        depth = tick.get("depth") or {}
        buy0 = (depth.get("buy") or [{}])[0]
        sell0 = (depth.get("sell") or [{}])[0]
        bid, bq = float(buy0.get("price") or 0), float(buy0.get("quantity") or 0)
        ask, aq = float(sell0.get("price") or 0), float(sell0.get("quantity") or 0)
        cum = float(tick.get("volume_traded") or 0)
        vol_d = max(cum - self.last_cumvol.get(tok, cum), 0.0)
        self.last_cumvol[tok] = cum
        # iceberg: heavy prints while best bid price+qty refuse to move
        last = self.last_bid.get(tok)
        iceberg = int(bool(last and bid == last[0] and bq >= last[1] * config.ICEBERG_QTY_RATIO
                           and vol_d > config.ICEBERG_VOL_MULT * max(bq, 1)))
        self.last_bid[tok] = (bid, bq)
        ltp = float(tick.get("last_price") or 0)
        oi = float(tick.get("oi") or 0)
        self.snap[tok] = {"ltp": ltp, "bid": bid, "ask": ask, "bid_qty": bq,
                          "ask_qty": aq, "vol_delta": vol_d, "oi": oi,
                          "iceberg": iceberg}
        self.vault.add((ts_ms, now_ms, tok, ltp, bid, ask, bq, aq,
                        vol_d, oi, iceberg))

    def _assemble_market(self) -> dict:
        market = {}
        if getattr(self, "vix_token", None):
            vs = self.snap.get(self.vix_token)
            if vs:
                market["_VIX"] = {"ltp": vs["ltp"]}
        for idx, spot_tok in self.spot_tokens.items():
            sp = self.snap.get(spot_tok)
            if not sp:
                continue
            ch = self.chains.get(idx)
            entry = {"spot": sp}
            if ch:
                legs = {}
                for leg, info in ch["legs"].items():
                    s = self.snap.get(info["token"])
                    if s:
                        legs[leg] = {"snap": s, "strike": info["strike"],
                                     "token": info["token"],
                                     "symbol": info["symbol"]}
                entry.update({"expiry": ch["expiry"], "dte": ch["dte"],
                              "T": ch["T"], "is_weekly": ch["is_weekly"],
                              "lot": ch["lot"], "step": ch["step"],
                              "atm": ch["atm"], "legs": legs})
            market[idx] = entry
        return market

    # ------------------------------------------------------------ main loop
    def run(self):
        self.kws.connect(threaded=True)   # KiteTicker's own thread — installs
        # no signal handlers, so the Windows/Twisted ValueError is gone.
        last_ring = 0.0
        last_tel = time.time()
        last_atm: dict[str, float] = {}
        while True:
            try:
                ticks = self.q.get(timeout=1.0)
                for t in ticks:
                    self._process(t)
            except queue.Empty:
                pass
            now = time.time()
            if now - last_ring >= config.RING_WRITE_S:
                last_ring = now
                for idx, tok in self.spot_tokens.items():
                    sp = self.snap.get(tok)
                    if not sp or not sp["ltp"]:
                        continue
                    ch = self.chains.get(idx)
                    step = (ch or {}).get("step") or \
                        config.INDICES[idx]["strike_step"]
                    atm = round(sp["ltp"] / step) * step
                    if last_atm.get(idx) != atm:
                        self._resubscribe(idx, sp["ltp"])
                        last_atm[idx] = atm
                self.ring.write_state({"market": self._assemble_market(),
                                       "ts": now}, ts=now)
            if now - last_tel >= config.TELEMETRY_S:
                log.info("telemetry: queue=%d snaps=%d subs=%d",
                         self.q.qsize(), len(self.snap), len(self.subscribed))
                if self.q.qsize() > config.QUEUE_WARN_DEPTH:
                    log.warning("⚠ writer lagging the WebSocket — "
                                "queue depth %d", self.q.qsize())
                self.vault.flush()
                last_tel = now
            hm = dt.datetime.now().strftime("%H:%M")
            if hm >= config.SNAPSHOT_PM_AT and not self.snapshot_written_pm:
                self.mapper.write_snapshot()              # post-close snapshot ★
                self.snapshot_written_pm = True
                self.vault.flush()


if __name__ == "__main__":
    config.setup_logging("harvester")
    Harvester().run()
