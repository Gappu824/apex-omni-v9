"""
POSITION STORE — an open trade must survive a restart
======================================================
2026-07-29: the brain restarted three times during a live paper session.
Each restart vaporised the open position. The daily risk ledger DID
survive (it was made durable in an earlier audit), so the books stayed
consistent — but the trade itself was gone: no stop, no target, no
theta clock, no trail. Within a minute the brain opened a fresh position
in the same index, and the orphan's outcome never reached the P&L. The
day's realised figure was wrong by three trades.

WHAT MUST SURVIVE, AND WHY EACH ITEM
------------------------------------
* entry_ts — THE critical one. Restore a position with a fresh timestamp
  and the theta guillotine restarts from zero: a trade 55 minutes into a
  60-minute budget silently gets another full hour. The clock must be
  absolute, not relative to process start.
* entry / qty / symbol / token — identity and cost basis.
* stop / target / floor / profit_lock / breakeven_px — the exit stack.
  Recomputing them from a restored mark would plant them at the WRONG
  reference (post-move prices), which is how a restart turns a stop into
  a take-profit.
* peak / peak_ts / trail_armed — the ratchet. Losing the peak un-arms a
  trail that had already locked profit; the trade would be re-risked
  from scratch after the move that earned the lock.
* pnl_realized / extends_used / theta_rides / meta_gate_zone — accounting
  and provenance across partial exits.

DESIGN RULES (crash consistency, not cleverness)
------------------------------------------------
1. THE BROKER IS THE AUTHORITY. The snapshot is a HINT about a position
   the broker is believed to hold. On restart the engine reconciles
   first; a snapshot is honoured only if reconciliation agrees that the
   position exists (live), or if the run is paper — where the paper
   ledger is the only ledger there is. A snapshot must never be able to
   conjure a position that does not exist.
2. ATOMIC PUBLICATION. tmp → os.replace, one file, whole-object writes.
   A torn snapshot is worse than none, and a half-written stop is a
   trade with no floor.
3. FENCED BY DAY, CONFIG AND EXPIRY. A snapshot from another trading
   day, another CONFIG_HASH, or a contract past expiry is discarded
   loudly. Yesterday's stop level is not information about today.
4. RESTORE DOES NOT RE-CHARGE RISK. The daily ledger already booked this
   trade's risk at entry and that ledger is itself durable. Restoration
   rebuilds the Position object only — it never calls request_entry.
   Double-charging would silently halve the day's remaining capacity.
5. STALENESS IS BOUNDED. A snapshot older than POSITION_SNAPSHOT_MAX_AGE_S
   is refused: a long gap means the market has moved without supervision
   and a human should look before the machine resumes managing.

Everything here is pure I/O over a plain dict. No trading logic lives in
this module — the PositionManager owns that, and rebuilds its own
objects from the fields returned.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import config

log = logging.getLogger("pos_store")

# fields that define the trade and its exit stack; everything else is
# rebuilt by the PositionManager (shield, trail object, transient marks).
DURABLE = ("index", "direction", "symbol", "exchange", "token", "strike",
           "qty", "entry", "entry_ts", "delta_at_entry", "conviction",
           "win_prob", "order_id", "n_buy_orders", "gtt_id", "dte",
           "stop", "target", "floor", "profit_lock", "breakeven_px",
           "extends_used", "peak", "peak_ts", "trail_armed",
           "spike_ref_spot", "spike_ref_oi", "reversal_since",
           "theta_rides", "fast_lane", "entry_conviction",
           "_dyn_tp_pct", "_dyn_src", "meta_gate_zone", "pnl_realized",
           # v9.9.33: the link to the live shadow. Without it a restored
           # position falls back to shadow.id_for(symbol), which picks the
           # NEWEST shadow for that symbol — and on 2026-08-11 the 24400
           # strike was entered three times, so "newest" is not "mine".
           # The exit would then be recorded as the baseline for the wrong
           # trade and both shadows would be wrong.
           "shadow_id")


def _path(index: str) -> Path:
    return config.STATE_DIR / f"position_{index}.json"


def _today() -> str:
    return dt.date.today().isoformat()


def save(index: str, pos) -> None:
    """Snapshot the open position. Called on every state change that a
    restart would otherwise lose. Never raises into the trading path —
    a failed snapshot degrades durability, it must not stop a trade."""
    try:
        body = {k: getattr(pos, k, None) for k in DURABLE}
        body["_day"] = _today()
        body["_config_hash"] = config.CONFIG_HASH
        body["_saved_ts"] = time.time()
        body["_live_fire"] = bool(getattr(config, "LIVE_FIRE", False))
        p = _path(index)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("position snapshot failed for %s (%s) — a restart "
                    "would lose this trade", index, e)


def clear(index: str) -> None:
    """Position is flat: the snapshot is now a lie. Remove it."""
    try:
        _path(index).unlink(missing_ok=True)
    except OSError as e:
        log.warning("could not clear position snapshot for %s (%s)", index, e)


def load(index: str, broker_symbols: set[str] | None = None
         ) -> dict | None:
    """Return a validated snapshot, or None with the reason logged.

    broker_symbols: what reconciliation says the broker actually holds.
    In LIVE_FIRE the snapshot must appear there. In paper mode pass None
    — there is no broker to ask, and the paper ledger is authoritative.
    """
    p = _path(index)
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                 # noqa: BLE001
        log.error("position snapshot for %s is unreadable (%s) — "
                  "discarding; reconcile manually", index, e)
        clear(index)
        return None

    def _reject(why: str):
        log.error("POSITION SNAPSHOT REFUSED (%s): %s. Not restoring; the "
                  "brain starts flat. If the broker holds this position, "
                  "square it manually.", index, why)
        clear(index)
        return None

    if j.get("_day") != _today():
        return _reject(f"snapshot is from {j.get('_day')}, today is "
                       f"{_today()}")
    if j.get("_config_hash") != config.CONFIG_HASH:
        return _reject(f"snapshot was written under config "
                       f"{j.get('_config_hash')}, running "
                       f"{config.CONFIG_HASH} — exit constants may differ")
    if bool(j.get("_live_fire")) != bool(getattr(config, "LIVE_FIRE", False)):
        return _reject("snapshot crosses the paper/live boundary")
    age = time.time() - float(j.get("_saved_ts") or 0)
    max_age = float(getattr(config, "POSITION_SNAPSHOT_MAX_AGE_S", 900))
    if age > max_age:
        return _reject(f"snapshot is {age / 60:.1f} min old (limit "
                       f"{max_age / 60:.0f} min) — the market moved "
                       f"unsupervised")
    if not j.get("symbol") or not j.get("qty"):
        return _reject("snapshot is missing symbol/qty")
    if broker_symbols is not None and j["symbol"] not in broker_symbols:
        return _reject(f"broker does not report {j['symbol']} — the "
                       f"position was closed or never filled")
    held_min = (time.time() - float(j.get("entry_ts") or 0)) / 60.0
    log.warning("POSITION RESTORED (%s): %s qty %s entered %.1f min ago at "
                "%.2f | stop %.2f target %.2f peak %.2f trail_armed=%s "
                "realised ₹%.2f. The THETA CLOCK CONTINUES from the "
                "original entry — this trade has %.1f min of its budget "
                "left, not a fresh one.", index, j["symbol"], j["qty"],
                held_min, j.get("entry", 0.0), j.get("stop", 0.0),
                j.get("target", 0.0), j.get("peak", 0.0),
                j.get("trail_armed"), j.get("pnl_realized", 0.0),
                max(config.MAX_HOLD_MINUTES - held_min, 0.0))
    return j


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                 # pragma: no cover
    import sys
    import tempfile
    from dataclasses import dataclass, field as _f

    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    config.STATE_DIR = Path(tempfile.mkdtemp())

    @dataclass
    class P:
        index: str = "NIFTY"
        direction: str = "CE"
        symbol: str = "NIFTY26AUG24400CE"
        exchange: str = "NFO"
        token: int = 12345
        strike: float = 24400.0
        qty: int = 75
        entry: float = 51.95
        entry_ts: float = 0.0
        delta_at_entry: float = 0.45
        conviction: float = 0.71
        win_prob: float = 0.58
        order_id: str = "o1"
        n_buy_orders: int = 1
        gtt_id: str = None
        dte: float = 2.0
        stop: float = 41.56
        target: float = 67.53
        floor: float = 36.0
        profit_lock: float = 53.2
        breakeven_px: float = 52.4
        extends_used: int = 1
        peak: float = 61.28
        peak_ts: float = 0.0
        trail_armed: bool = True
        spike_ref_spot: float = 24288.0
        spike_ref_oi: float = 1e6
        reversal_since: float = 0.0
        theta_rides: int = 0
        fast_lane: bool = False
        entry_conviction: float = 0.71
        _dyn_tp_pct: float = 0.30
        _dyn_src: str = "vault"
        meta_gate_zone: str = "FULL"
        pnl_realized: float = 552.12
        shield: object = None
        trail: object = None

    now = time.time()
    p = P(entry_ts=now - 25 * 60, peak_ts=now - 300)
    save("NIFTY", p)
    r = load("NIFTY")
    chk("round-trips", r is not None)
    chk("entry_ts preserved to the second (theta clock survives)",
        r and abs(r["entry_ts"] - p.entry_ts) < 1e-6)
    chk("exit stack preserved",
        r and r["stop"] == p.stop and r["target"] == p.target
        and r["floor"] == p.floor and r["profit_lock"] == p.profit_lock)
    chk("ratchet preserved",
        r and r["peak"] == p.peak and r["trail_armed"] is True)
    chk("accounting preserved",
        r and r["pnl_realized"] == 552.12 and r["extends_used"] == 1
        and r["meta_gate_zone"] == "FULL")
    chk("transient objects are NOT persisted",
        r is not None and "shield" not in r and "trail" not in r)

    # broker authority
    save("NIFTY", p)
    chk("live: broker confirming ⇒ restore",
        load("NIFTY", {"NIFTY26AUG24400CE"}) is not None)
    save("NIFTY", p)
    chk("live: broker silent ⇒ refuse", load("NIFTY", set()) is None)
    chk("refusal clears the snapshot", not _path("NIFTY").exists())

    # fences
    save("NIFTY", p)
    j = json.loads(_path("NIFTY").read_text()); j["_day"] = "2020-01-01"
    _path("NIFTY").write_text(json.dumps(j))
    chk("stale day refused", load("NIFTY") is None)

    save("NIFTY", p)
    j = json.loads(_path("NIFTY").read_text()); j["_config_hash"] = "deadbeef"
    _path("NIFTY").write_text(json.dumps(j))
    chk("cross-config refused", load("NIFTY") is None)

    save("NIFTY", p)
    j = json.loads(_path("NIFTY").read_text())
    j["_saved_ts"] = time.time() - 4000
    _path("NIFTY").write_text(json.dumps(j))
    chk("stale snapshot refused", load("NIFTY") is None)

    save("NIFTY", p)
    j = json.loads(_path("NIFTY").read_text()); j["_live_fire"] = True
    _path("NIFTY").write_text(json.dumps(j))
    chk("paper/live crossing refused", load("NIFTY") is None)

    _path("NIFTY").write_text("{not json")
    chk("corrupt snapshot refused, not crashed", load("NIFTY") is None)

    save("NIFTY", p); clear("NIFTY")
    chk("clear removes it", load("NIFTY") is None)
    chk("absent snapshot is simply None", load("BANKNIFTY") is None)

    # atomicity: no debris after a save
    save("SENSEX", p)
    debris = [f.name for f in config.STATE_DIR.glob("*.tmp.json")]
    chk("atomic write leaves no temp debris", not debris)

    total = 17
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)