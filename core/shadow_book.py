"""
SHADOW BOOK — every trade we took keeps trading until the bell
==============================================================
When a real position goes flat, position_manager sets `self.pos = None`
and PS.clear(). The instrument ceases to exist for the system at that
instant. Whatever it did for the rest of the session — the 40% it went
on to add after a MAX_HOLD_THETA exit, the collapse a "premature" stop
actually dodged — was never observed, so the exit stack has been tuned
against anecdotes from log-reading rather than a measured distribution.

The shadow book is the parallel position. It opens the moment the real
trade opens, marks the SAME instrument off the SAME tick ring every
second, and does not stop when the real trade exits. It runs the whole
config.SHADOW_POLICIES family concurrently — each with its own peak,
its own lock, its own clock — through core.exit_policies, the same
functions the nightly study replays. At the bell every shadow closes
and the session's counterfactuals are written to the shadow ledger.

FOUR PROPERTIES THAT MAKE THIS SAFE TO RUN IN THE LIVE LOOP
------------------------------------------------------------
1. IT CANNOT TRADE. There is no engine handle in this module, no order
   path, no risk-governor call. It reads quotes and writes files. The
   worst failure available to it is a wrong number in a report.
2. IT CANNOT RAISE INTO THE TRADING PATH. Every public method is total:
   it catches, logs and returns. A shadow that throws while the brain is
   managing a live stop would be strictly worse than no shadow at all.
3. IT CANNOT COST THE HOT LOOP. Marks are O(open shadows × policies) of
   pure arithmetic, throttled to SHADOW_MARK_S, on quotes the loop
   already has in hand. No I/O per mark — persistence is snapshot-on-
   change with the same tmp→os.replace discipline as position_store.
4. IT SURVIVES A RESTART. The 2026-07-29 lesson applies with full force:
   a shadow that resets on restart silently restarts every theta clock
   and every peak, which would bias the study toward whichever policy
   benefits from a fresh start. Snapshots are fenced by day, CONFIG_HASH
   and age, exactly like core.position_store.

WHAT IT DOES NOT DO
-------------------
It does not decide anything. Promotion of a measured policy into the
live exit stack is a separate, gated act (core.exit_policy_store), and
it requires paired-by-day evidence, FDR correction, an MDE floor and a
holdout that agrees. Measuring is not permission.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from core.exit_policies import (PolicySpec, PolicyState, TradeCtx, step,
                                pnl_of)

log = logging.getLogger("shadow_book")

SHADOW_FIELDS = ["ts", "event", "shadow_id", "index", "symbol", "token",
                 "kind", "side", "qty", "entry_px", "entry_ts",
                 "real_exit_px", "real_exit_ts", "real_pnl", "real_reason",
                 "session_close_hm", "coverage", "mfe_px", "mae_px",
                 "peak_t", "policies", "config_hash"]


def _now_sod(ts: float | None = None) -> int:
    lt = dt.datetime.fromtimestamp(ts if ts is not None else time.time())
    return lt.hour * 3600 + lt.minute * 60 + lt.second


def _hm_to_sod(hm: str) -> int:
    h, m = str(hm).split(":")[:2]
    return int(h) * 3600 + int(m) * 60


def session_end_t(entry_ts: float, index: str) -> int:
    """Seconds from entry to the bell, from the date-aware calendar."""
    try:
        from core import session_calendar as SC
        day = dt.datetime.fromtimestamp(entry_ts).date()
        close_hm = SC.session_close_hm(day, index)
    except Exception:                                      # noqa: BLE001
        comms = getattr(config, "COMMODITIES", {}) or {}
        close_hm = (str(comms[index].get("session_close", "23:55"))
                    if index in comms else "15:30")
    return max(_hm_to_sod(close_hm) - _now_sod(entry_ts), 1)


@dataclass
class ShadowTrade:
    shadow_id: str
    index: str
    symbol: str
    token: int
    kind: str
    ctx: TradeCtx
    entry_ts: float
    states: dict[str, PolicyState] = field(default_factory=dict)
    real_exit_px: float = 0.0
    real_exit_ts: float = 0.0
    real_pnl: float = 0.0
    real_reason: str = ""
    real_closed: bool = False
    marks: int = 0
    fresh_marks: int = 0
    mfe_px: float = 0.0
    mae_px: float = 0.0
    peak_t: int = 0
    finished: bool = False

    @property
    def coverage(self) -> float:
        return (self.fresh_marks / self.marks) if self.marks else 0.0

    def all_closed(self) -> bool:
        return all(s.closed for s in self.states.values())

    def to_json(self) -> dict:
        d = {"shadow_id": self.shadow_id, "index": self.index,
             "symbol": self.symbol, "token": self.token, "kind": self.kind,
             "entry_ts": self.entry_ts,
             "ctx": {"entry": self.ctx.entry, "qty": self.ctx.qty,
                     "side": self.ctx.side, "stop_pct": self.ctx.stop_pct,
                     "hold_budget_s": self.ctx.hold_budget_s,
                     "session_end_t": self.ctx.session_end_t,
                     "floor_pct": self.ctx.floor_pct},
             "states": {k: v.as_dict() for k, v in self.states.items()},
             "real_exit_px": self.real_exit_px,
             "real_exit_ts": self.real_exit_ts, "real_pnl": self.real_pnl,
             "real_reason": self.real_reason,
             "real_closed": self.real_closed, "marks": self.marks,
             "fresh_marks": self.fresh_marks, "mfe_px": self.mfe_px,
             "mae_px": self.mae_px, "peak_t": self.peak_t}
        return d

    @staticmethod
    def from_json(d: dict) -> "ShadowTrade":
        ctx = TradeCtx(**d["ctx"])
        st = ShadowTrade(shadow_id=d["shadow_id"], index=d["index"],
                         symbol=d["symbol"], token=int(d["token"]),
                         kind=d.get("kind", "SINGLE"), ctx=ctx,
                         entry_ts=float(d["entry_ts"]))
        st.states = {k: PolicyState(**v) for k, v in d["states"].items()}
        st.real_exit_px = float(d.get("real_exit_px") or 0.0)
        st.real_exit_ts = float(d.get("real_exit_ts") or 0.0)
        st.real_pnl = float(d.get("real_pnl") or 0.0)
        st.real_reason = str(d.get("real_reason") or "")
        st.real_closed = bool(d.get("real_closed"))
        st.marks = int(d.get("marks") or 0)
        st.fresh_marks = int(d.get("fresh_marks") or 0)
        st.mfe_px = float(d.get("mfe_px") or ctx.entry)
        st.mae_px = float(d.get("mae_px") or ctx.entry)
        st.peak_t = int(d.get("peak_t") or 0)
        return st


class ShadowBook:
    """One per process. Holds every shadow open today."""

    def __init__(self, index: str, costs_fn=None,
                 ledger_path: Path | None = None):
        self.index = index
        self.open: dict[str, ShadowTrade] = {}
        self.done: list[ShadowTrade] = []
        self._last_mark = 0.0
        self._dirty = False
        self._specs = PolicySpec.family()
        self._ledger = Path(ledger_path or getattr(
            config, "SHADOW_LEDGER_PATH",
            config.LOG_DIR / "shadow_ledger_v9.csv"))
        if costs_fn is None:
            try:
                from core.execution_engine import round_trip_costs
                costs_fn = round_trip_costs
            except Exception:                              # noqa: BLE001
                costs_fn = None
        self._costs = costs_fn
        self._ensure_ledger()

    # ------------------------------------------------------------ ledger
    def _ensure_ledger(self) -> None:
        try:
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            if not self._ledger.exists():
                with self._ledger.open("w", newline="",
                                       encoding="utf-8") as f:
                    csv.DictWriter(f, SHADOW_FIELDS).writeheader()
        except OSError as e:                               # noqa: BLE001
            log.warning("shadow ledger unavailable (%s) — the book will "
                        "still mark, but nothing will be recorded", e)

    def _row(self, **row) -> None:
        try:
            body = {k: row.get(k, "") for k in SHADOW_FIELDS}
            with self._ledger.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, SHADOW_FIELDS).writerow(body)
        except OSError as e:                               # noqa: BLE001
            log.warning("shadow ledger write failed (%s)", e)

    # ------------------------------------------------------------- state
    def _path(self) -> Path:
        return config.STATE_DIR / f"shadow_book_{self.index}.json"

    def snapshot(self, force: bool = False) -> None:
        """Atomic publication. Never raises."""
        if not (self._dirty or force):
            return
        try:
            body = {"_day": dt.date.today().isoformat(),
                    "_config_hash": config.CONFIG_HASH,
                    "_saved_ts": time.time(),
                    "index": self.index,
                    "open": [t.to_json() for t in self.open.values()]}
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
            tmp.write_text(json.dumps(body), encoding="utf-8")
            os.replace(tmp, p)
            self._dirty = False
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow snapshot failed (%s) — a restart would lose "
                        "%d open shadow(s)", e, len(self.open))

    def restore(self) -> int:
        """Rebuild today's open shadows. Fenced by day, hash and age."""
        try:
            p = self._path()
            if not p.exists():
                return 0
            body = json.loads(p.read_text(encoding="utf-8"))
            if body.get("_day") != dt.date.today().isoformat():
                log.info("shadow snapshot is from %s — discarded",
                         body.get("_day"))
                p.unlink(missing_ok=True)
                return 0
            if body.get("_config_hash") != config.CONFIG_HASH:
                log.warning("shadow snapshot CONFIG_HASH %s ≠ %s — discarded",
                            body.get("_config_hash"), config.CONFIG_HASH)
                p.unlink(missing_ok=True)
                return 0
            age = time.time() - float(body.get("_saved_ts") or 0)
            max_age = float(getattr(config, "SHADOW_SNAPSHOT_MAX_AGE_S", 900))
            if age > max_age:
                log.warning("shadow snapshot is %.0fs old (> %.0fs) — "
                            "discarded; the tape moved unobserved",
                            age, max_age)
                p.unlink(missing_ok=True)
                return 0
            for d in body.get("open", []):
                t = ShadowTrade.from_json(d)
                self.open[t.shadow_id] = t
            log.info("shadow book restored: %d open shadow(s) for %s",
                     len(self.open), self.index)
            return len(self.open)
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow restore failed (%s) — starting empty", e)
            return 0

    # -------------------------------------------------------------- open
    def open_shadow(self, symbol: str, token: int, entry_px: float,
                    qty: int, side: int = +1, kind: str = "SINGLE",
                    entry_ts: float | None = None,
                    hold_budget_s: int | None = None,
                    stop_pct: float | None = None) -> str:
        """Start marking a trade the engine just entered. Returns an id
        (empty string if the book declined) and never raises."""
        try:
            if not bool(getattr(config, "SHADOW_ENABLED", True)):
                return ""
            if len(self.open) >= int(getattr(config, "SHADOW_MAX_OPEN", 64)):
                log.warning("shadow book full (%d) — not tracking %s",
                            len(self.open), symbol)
                return ""
            ts = float(entry_ts if entry_ts is not None else time.time())
            if hold_budget_s is None:
                hold_budget_s = int(float(
                    getattr(config, "MAX_HOLD_MINUTES", 60)) * 60)
            ctx = TradeCtx(entry=float(entry_px), qty=int(qty),
                           side=int(side), stop_pct=stop_pct,
                           hold_budget_s=int(hold_budget_s),
                           session_end_t=session_end_t(ts, self.index))
            sid = f"{self.index}:{symbol}:{int(ts)}"
            t = ShadowTrade(shadow_id=sid, index=self.index, symbol=symbol,
                            token=int(token), kind=kind, ctx=ctx,
                            entry_ts=ts, mfe_px=float(entry_px),
                            mae_px=float(entry_px))
            t.states = {s.name: PolicyState.start(ctx) for s in self._specs}
            self.open[sid] = t
            self._dirty = True
            self._row(ts=f"{ts:.3f}", event="SHADOW_OPEN", shadow_id=sid,
                      index=self.index, symbol=symbol, token=token,
                      kind=kind, side=side, qty=qty,
                      entry_px=f"{entry_px:.2f}", entry_ts=f"{ts:.3f}",
                      session_close_hm=ctx.session_end_t,
                      config_hash=config.CONFIG_HASH)
            log.info("SHADOW OPEN %s %s ×%d @ %.2f — %d polic(ies) to the "
                     "bell (%ds)", self.index, symbol, qty, entry_px,
                     len(t.states), ctx.session_end_t)
            return sid
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow open failed for %s (%s) — trading is "
                        "unaffected", symbol, e)
            return ""

    # -------------------------------------------------------------- mark
    def mark(self, quotes: dict, now: float | None = None,
             force: bool = False) -> int:
        """Advance every open shadow one step. `quotes` is token → quote
        dict (the tick ring the main loop already holds). Returns the
        number of shadows that finished on this mark. Never raises."""
        try:
            now = float(now if now is not None else time.time())
            cadence = float(getattr(config, "SHADOW_MARK_S", 1.0))
            if not force and (now - self._last_mark) < cadence:
                return 0
            self._last_mark = now
            finished = 0
            for sid in list(self.open):
                t = self.open[sid]
                px = self._bid_of(quotes, t.token, t.ctx.side)
                tt = int(now - t.entry_ts)
                t.marks += 1
                if px is not None:
                    t.fresh_marks += 1
                    if (px - t.ctx.entry) * t.ctx.side > \
                            (t.mfe_px - t.ctx.entry) * t.ctx.side:
                        t.mfe_px = px
                        t.peak_t = tt      # when the trade was at its BEST
                    if (px - t.ctx.entry) * t.ctx.side < \
                            (t.mae_px - t.ctx.entry) * t.ctx.side:
                        t.mae_px = px
                mark_px = px if px is not None else float("nan")
                for spec in self._specs:
                    st = t.states.get(spec.name)
                    if st is None or st.closed:
                        continue
                    step(st, spec, t.ctx, mark_px, tt, self._costs)
                if t.all_closed():
                    self._finish(t, now)
                    finished += 1
            if finished or (self.open and int(now) % 30 == 0):
                self._dirty = True
                self.snapshot()
            return finished
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow mark failed (%s) — trading is unaffected", e)
            return 0

    @staticmethod
    def _bid_of(quotes: dict, token: int, side: int) -> float | None:
        """Mark at the side you would actually transact against: a long
        premium position exits into the BID, a short covers at the ASK."""
        q = (quotes or {}).get(int(token)) or {}
        key = "bid" if side > 0 else "ask"
        v = q.get(key)
        if v is None:
            v = q.get("last_price") or q.get("ltp")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    # -------------------------------------------------------- real exit
    def note_real_exit(self, shadow_id: str, exit_px: float, pnl: float,
                       reason: str, ts: float | None = None) -> None:
        """The engine's own exit — the baseline every policy is measured
        against. The shadow does NOT stop here; that is the point."""
        try:
            t = self.open.get(shadow_id)
            if t is None:
                return
            t.real_exit_px = float(exit_px)
            t.real_exit_ts = float(ts if ts is not None else time.time())
            t.real_pnl = float(pnl)
            t.real_reason = str(reason)
            t.real_closed = True
            self._dirty = True
            log.info("SHADOW baseline %s: real exit @%.2f (%s) ₹%.2f — "
                     "shadow continues to the bell", t.symbol, exit_px,
                     reason, pnl)
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow baseline note failed (%s)", e)

    def id_for(self, symbol: str) -> str:
        """Newest open shadow for a symbol — the one an exit refers to."""
        cands = [t for t in self.open.values() if t.symbol == symbol]
        if not cands:
            return ""
        return max(cands, key=lambda t: t.entry_ts).shadow_id

    # ------------------------------------------------------------ finish
    def _finish(self, t: ShadowTrade, now: float) -> None:
        t.finished = True
        pol = {}
        for name, st in t.states.items():
            pol[name] = {"exit_px": round(st.exit_px, 2),
                         "exit_t": st.exit_t, "reason": st.exit_reason,
                         "stale": bool(st.stale_exit),
                         "pnl": round(pnl_of(t.ctx, st.exit_px, self._costs),
                                      2)}
        pol["as_traded"] = {"exit_px": round(t.real_exit_px, 2),
                            "exit_t": int(t.real_exit_ts - t.entry_ts),
                            "reason": t.real_reason, "stale": False,
                            "pnl": round(t.real_pnl, 2)}
        self._row(ts=f"{now:.3f}", event="SHADOW_CLOSE",
                  shadow_id=t.shadow_id, index=t.index, symbol=t.symbol,
                  token=t.token, kind=t.kind, side=t.ctx.side,
                  qty=t.ctx.qty, entry_px=f"{t.ctx.entry:.2f}",
                  entry_ts=f"{t.entry_ts:.3f}",
                  real_exit_px=f"{t.real_exit_px:.2f}",
                  real_exit_ts=f"{t.real_exit_ts:.3f}",
                  real_pnl=f"{t.real_pnl:.2f}", real_reason=t.real_reason,
                  coverage=f"{t.coverage:.3f}", mfe_px=f"{t.mfe_px:.2f}",
                  mae_px=f"{t.mae_px:.2f}", peak_t=t.peak_t,
                  policies=json.dumps(pol, separators=(",", ":")),
                  config_hash=config.CONFIG_HASH)
        best = max(pol.items(), key=lambda kv: kv[1]["pnl"])
        log.info("SHADOW CLOSE %s | as_traded ₹%.2f | best %s ₹%.2f | "
                 "left on table ₹%.2f | coverage %.0f%%",
                 t.symbol, t.real_pnl, best[0], best[1]["pnl"],
                 best[1]["pnl"] - t.real_pnl, 100.0 * t.coverage)
        self.done.append(t)
        self.open.pop(t.shadow_id, None)

    def close_session(self, now: float | None = None) -> int:
        """The bell. Force every remaining shadow closed at its last mark."""
        try:
            now = float(now if now is not None else time.time())
            n = 0
            for sid in list(self.open):
                t = self.open[sid]
                tt = int(now - t.entry_ts)
                for spec in self._specs:
                    st = t.states.get(spec.name)
                    if st is None or st.closed:
                        continue
                    st.closed = True
                    st.exit_px = st.last_fresh_px or t.ctx.entry
                    st.exit_t = tt
                    st.exit_reason = "SESSION_END"
                    st.stale_exit = (st.last_fresh_t < tt - 5)
                self._finish(t, now)
                n += 1
            self.snapshot(force=True)
            if n:
                log.info("shadow book flat: %d shadow(s) closed at the bell",
                         n)
            return n
        except Exception as e:                             # noqa: BLE001
            log.warning("shadow session close failed (%s)", e)
            return 0

    # ------------------------------------------------------------- pins
    def pinned_tokens(self) -> set[int]:
        """Tokens the harvester must NOT prune: a shadow marking a leg
        that has been unsubscribed is measuring a corpse."""
        return {int(t.token) for t in self.open.values() if t.token}