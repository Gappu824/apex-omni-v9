"""
APEX OMNI v9.1 — DATA HARVESTER (audit §4 leaps + v9.1 diagnostics)
===================================================================
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

v9.1 (audit follow-up — make coverage measurable):
  * Every tick is classified by token_role and counted: per-index spot ticks,
    option-leg ticks, OI-present rate, two-sided-depth rate.
  * Spot FEED GAPS (>5 s between consecutive spot ticks per index) counted
    with the worst gap — the number the forge's replay quality depends on.
  * WebSocket reconnects, writer-queue high-water mark, rows flushed.
  * All of it lands in logs/harvester_report_<date>.json (atomic, refreshed
    every ~60 s and at the post-close snapshot), so "did today's harvest
    actually cover the session?" is answered by one file, not a log scroll.

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
from core.diagnostics import DailyReport

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

_GAP_S = 5.0            # a spot silence longer than this counts as a feed gap
_REPORT_EVERY_S = 60.0  # diagnostics JSON refresh cadence


class HarvestDiag:
    """Pure counters — zero I/O on the tick path; serialized by the report."""

    def __init__(self, seed: dict | None = None):
        # equity indices + harvested commodities — both get coverage counters
        _names = list(config.INDEX_ORDER) + \
            list(getattr(config, "HARVEST_COMMODITIES", []))
        self.per_idx: dict[str, dict] = {
            i: {"spot_ticks": 0, "leg_ticks": 0, "leg_oi_nonzero": 0,
                "leg_depth_2sided": 0, "gap_count": 0, "max_gap_s": 0.0}
            for i in _names}
        # AUDIT (2026-07-27): these counters RESET on every restart while
        # DailyReport(resume=True) resumed only the envelope. On 2026-07-24 the
        # harvester restarted at 19:32 IST — four hours after the equity close —
        # so the day's report showed every equity index with leg_ticks = 26
        # (exactly the subscribed-leg count: the initial snapshot callbacks) and
        # SENSEX leg_depth_2sided = 0. That number was read as "BFO never
        # delivers depth" and chased for days; tools/feed_probe then measured
        # BFO at 100% two-sided, and 2026-07-27 recorded 630,101 SENSEX leg
        # ticks at 99.9% depth. It was an artifact of measuring a closed market.
        # Seeding from the prior same-day report makes coverage a TRUE daily
        # total again.
        if seed:
            for name, prev in (seed or {}).items():
                cur = self.per_idx.get(name)
                if not cur or not isinstance(prev, dict):
                    continue
                for k in ("spot_ticks", "leg_ticks", "leg_oi_nonzero",
                          "leg_depth_2sided", "gap_count"):
                    try:
                        cur[k] += int(prev.get(k) or 0)
                    except (TypeError, ValueError):
                        pass
                try:
                    cur["max_gap_s"] = max(float(cur["max_gap_s"]),
                                           float(prev.get("max_gap_s") or 0.0))
                except (TypeError, ValueError):
                    pass
        self.vix_ticks = 0
        self.ws_connects = 0
        self.ws_closes = 0
        self.queue_max = 0
        self.rows_flushed = 0
        self.flushes = 0
        self._last_spot_ts: dict[str, float] = {}

    def tick(self, role: tuple | None, now: float, oi: float,
             bid: float, ask: float) -> None:
        if not role:
            return
        idx, leg, _ = role
        if idx == "VIX":
            self.vix_ticks += 1
            return
        d = self.per_idx.get(idx)
        if d is None:
            return
        if leg == "spot":
            d["spot_ticks"] += 1
            last = self._last_spot_ts.get(idx)
            if last is not None:
                gap = now - last
                if gap > _GAP_S:
                    d["gap_count"] += 1
                    if gap > d["max_gap_s"]:
                        d["max_gap_s"] = round(gap, 1)
            self._last_spot_ts[idx] = now
        else:
            d["leg_ticks"] += 1
            if oi > 0:
                d["leg_oi_nonzero"] += 1
            if bid > 0 and ask > 0:
                d["leg_depth_2sided"] += 1

    def snapshot(self, q_depth: int, n_subs: int, n_snaps: int,
                 chains: dict) -> dict:
        self.queue_max = max(self.queue_max, q_depth)
        per = {}
        for idx, d in self.per_idx.items():
            legs = max(d["leg_ticks"], 1)
            per[idx] = dict(d)
            per[idx]["leg_oi_rate"] = round(d["leg_oi_nonzero"] / legs, 3)
            per[idx]["leg_depth_rate"] = round(d["leg_depth_2sided"] / legs, 3)
            ch = chains.get(idx) or {}
            if ch:
                # AUDIT (2026-07-28): the report named the expiry and strike
                # but never the CONTRACT, so answering "which instrument was
                # this actually trading?" meant cross-referencing the ledger.
                # These names are not constructed — they are the live Kite
                # dump's `tradingsymbol`, carried through unchanged, so the
                # format tracks whatever the exchange publishes (weekly
                # NIFTY2672124200PE vs monthly NIFTY26JUL23950CE) with no
                # convention hardcoded anywhere.
                _legs = ch.get("legs") or {}
                _names = {k: (v or {}).get("symbol")
                          for k, v in _legs.items() if (v or {}).get("symbol")}
                per[idx]["chain"] = {"expiry": str(ch.get("expiry")),
                                     "dte": ch.get("dte"),
                                     "atm": ch.get("atm"),
                                     "lot": ch.get("lot"),
                                     "contracts": _names or None}
        return {"per_index": per, "vix_ticks": self.vix_ticks,
                "ws": {"connects": self.ws_connects, "closes": self.ws_closes},
                "writer": {"rows_flushed": self.rows_flushed,
                           "flushes": self.flushes,
                           "queue_now": q_depth,
                           "queue_max": self.queue_max},
                "subscribed_tokens": n_subs, "tokens_with_snap": n_snaps}


class VaultKeeper:
    def __init__(self, diag: HarvestDiag | None = None):
        self.con = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.con.executescript(TICKS_SCHEMA)
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.diag = diag

    def add(self, row: tuple):
        with self.lock:
            self.rows.append(row)
            if len(self.rows) >= config.DB_BATCH_ROWS:
                self._flush()

    def _flush(self):
        if not self.rows:
            return
        n = len(self.rows)
        self.con.executemany(
            "INSERT INTO ticks_v9 VALUES (?,?,?,?,?,?,?,?,?,?,?)", self.rows)
        self.con.commit()
        self.rows.clear()
        if self.diag:
            self.diag.rows_flushed += n
            self.diag.flushes += 1

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
        # v9.6.1: restart-safe forensics (parity with brain). The report MUST
        # be constructed BEFORE the diag — it carries the prior segment's
        # coverage, which seeds the counters (see HarvestDiag.__init__).
        self.report = DailyReport("harvester", resume=True)
        _seed = ((self.report.d.get("coverage") or {}).get("per_index")
                 if isinstance(getattr(self.report, "d", None), dict)
                 else None)
        self.diag = HarvestDiag(seed=_seed)
        self.vault = VaultKeeper(self.diag)
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
        # v10.2.2: record "_VIX" in the vault's spot_tokens META TABLE so
        # replay tools can discover the stream — WITHOUT putting it in
        # self.spot_tokens, which downstream loops treat as index-keyed
        # (config.INDICES[idx] — "_VIX" there crashed startup on 07-15).
        # Subscription needs nothing: the harvester already subscribes VIX
        # via self.vix_token / token_role.
        _rec = dict(self.spot_tokens)
        if int(self.vix_token or getattr(config, "VIX_TOKEN", 0)):
            _rec["_VIX"] = int(self.vix_token or config.VIX_TOKEN)
        self.vault.record_spot_tokens(_rec)
        self.kws = KiteTicker(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN)
        self.kws.on_ticks = lambda ws, ticks: self.q.put(ticks)
        self.kws.on_order_update = lambda ws, d: self._order_update(d)
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
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
        # v9.7.1: commodity "spot" = front-month FUTURE (from the mapper). These
        # feed the same ATM-tracking + option-leg subscription path as indices.
        for name, fut in getattr(self.mapper, "commodity_futures", {}).items():
            tok = fut["token"]
            self.spot_tokens[name] = tok
            self.token_role[tok] = (name, "spot", 0.0)
        if getattr(self.mapper, "commodity_futures", {}):
            log.info("Commodity spot(=fut) tokens: %s",
                     {k: self.spot_tokens.get(k)
                      for k in self.mapper.commodity_futures})
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
        self.diag.ws_connects += 1
        toks = list(self.spot_tokens.values())
        if self.vix_token:
            toks.append(self.vix_token)
        ws.subscribe(toks)
        ws.set_mode(ws.MODE_FULL, toks)
        self.subscribed.update(toks)
        # After a RECONNECT the ticker only knows the tokens re-subscribed on
        # this connection — re-arm every option leg we were carrying too.
        legs = [t for t, role in self.token_role.items()
                if role[1] != "spot" and t not in toks]
        if legs:
            ws.subscribe(legs)
            ws.set_mode(ws.MODE_FULL, legs)
            self.subscribed.update(legs)
        log.info("WS connected (#%d) — %d spot + %d leg tokens subscribed",
                 self.diag.ws_connects, len(toks), len(legs))

    def _on_close(self, ws, code, reason):
        self.diag.ws_closes += 1
        log.warning("WS closed: %s %s (auto-reconnect handles retry; "
                    "close #%d today)", code, reason, self.diag.ws_closes)

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
        now = time.time()
        self.last_tick_ts = now      # tick-flow watchdog (see telemetry)
        now_ms = int(now * 1000)
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
        self.diag.tick(self.token_role.get(tok), now, oi, bid, ask)

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

    def _write_report(self):
        self.report.d["coverage"] = self.diag.snapshot(
            self.q.qsize(), len(self.subscribed), len(self.snap), self.chains)
        self.report.write()

    # ------------------------------------------------------------ main loop
    def run(self):
        self.kws.connect(threaded=True)   # KiteTicker's own thread — installs
        # no signal handlers, so the Windows/Twisted ValueError is gone.
        last_ring = 0.0
        last_tel = time.time()
        last_rep = time.time()
        last_atm: dict[str, float] = {}
        while True:
            try:
                ticks = self.q.get(timeout=1.0)
                for t in ticks:
                    try:
                        self._process(t)
                    except Exception as _pe:              # noqa: BLE001
                        # v9.7.1 AUDIT S4-F4: a single malformed tick used to
                        # propagate out of run(), kill the process, and — six
                        # restarts later — take the HARVESTER down for the day
                        # (the worst possible failure: total data loss).
                        # Poison ticks are counted, throttled-logged, dropped.
                        self._bad_ticks = getattr(self, "_bad_ticks", 0) + 1
                        if self._bad_ticks <= 5 or self._bad_ticks % 500 == 0:
                            log.warning("malformed tick dropped (#%d): %s %r",
                                        self._bad_ticks, _pe, str(t)[:120])
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
                    _spec = (config.INDICES.get(idx)
                             or getattr(config, "COMMODITIES", {}).get(idx)
                             or {})
                    step = (ch or {}).get("step") or \
                        _spec.get("strike_step", 1.0)
                    atm = round(sp["ltp"] / step) * step
                    if last_atm.get(idx) != atm:
                        self._resubscribe(idx, sp["ltp"])
                        last_atm[idx] = atm
                self.ring.write_state({"market": self._assemble_market(),
                                       "ts": now}, ts=now)
            if now - last_tel >= config.TELEMETRY_S:
                self.diag.queue_max = max(self.diag.queue_max, self.q.qsize())
                # AUDIT (2026-07-22 20:20:56): the Kite WebSocket dropped
                # (1006, uncleanly) and this line kept printing an identical
                # healthy-looking telemetry for 58s with ZERO ticks arriving —
                # a silent data hole right through the EIA release. queue=0
                # looks the same whether the feed is quiet or DEAD, so the age
                # of the last tick is now part of the heartbeat, and a silent
                # feed says so in words.
                _last = getattr(self, "last_tick_ts", 0.0)
                _age = (now - _last) if _last else float("inf")
                log.info("telemetry: queue=%d snaps=%d subs=%d | last tick "
                         "%s ago", self.q.qsize(), len(self.snap),
                         len(self.subscribed),
                         f"{_age:.0f}s" if _last else "never")
                if _age > float(getattr(config, "FEED_SILENT_WARN_S", 30)):
                    log.warning("⚠ NO TICKS for %.0fs — the WebSocket is up "
                                "but silent, or the peer dropped it. The ring "
                                "is going stale; both brains will hit their "
                                "stale-feed guards. Check the connection.",
                                _age)
                if self.q.qsize() > config.QUEUE_WARN_DEPTH:
                    log.warning("⚠ writer lagging the WebSocket — "
                                "queue depth %d", self.q.qsize())
                self.vault.flush()
                last_tel = now
            if now - last_rep >= _REPORT_EVERY_S:
                self._write_report()
                last_rep = now
            hm = dt.datetime.now().strftime("%H:%M")
            if hm >= config.SNAPSHOT_PM_AT and not self.snapshot_written_pm:
                self.mapper.write_snapshot()              # post-close snapshot ★
                self.snapshot_written_pm = True
                self.vault.flush()
                self._write_report()                      # end-of-day coverage
                log.info("post-close snapshot + final coverage report written "
                         "→ %s", self.report.path)


if __name__ == "__main__":
    config.setup_logging("harvester")
    Harvester().run()