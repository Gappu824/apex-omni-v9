"""
TOKEN PINS — the harvester must not unsubscribe what we are still marking
=========================================================================
data_harvester_v9._resubscribe prunes every option leg further than
PRUNE_STEPS (8) from the running ATM. That is correct behaviour for a
chain harvester: without it the subscription set only grows and the WS
budget is finite (v8 never pruned and eventually starved).

But the harvester has no idea what the brain is holding. They are
separate processes. So on a day where spot runs, a leg we are IN gets
unsubscribed, its ticks stop, and:

  * the live shadow book marks a token that no longer quotes — every
    policy after that second is arithmetic on a dead price;
  * the vault has no ticks for it, so the nightly replay reconstructs a
    forward-filled flat line to the close and reports MFE, capture and
    hold-to-close numbers computed over a path that stopped existing.

This module is the shared manifest that fixes it. The trading processes
publish the tokens they are tracking; the harvester reads the manifest
each time it re-subscribes and REMOVES those tokens from the prune set.

DESIGN
------
* FILE, NOT SOCKET. The processes already communicate through
  config.STATE_DIR and ORDER_UPDATES_PATH; a JSON manifest with the same
  tmp→os.replace discipline needs no new transport and no new failure
  mode.
* PINS EXPIRE. A crashed brain must not pin tokens forever and slowly
  starve the subscription budget. Every pin carries a timestamp and is
  ignored once older than PIN_TTL_S. A live process re-publishes on
  every change, so a healthy pin never expires.
* FENCED BY DAY. Yesterday's strikes are not information about today.
* THE HARVESTER STILL WINS ON BUDGET. read_pins() is advisory: if the
  pin set ever exceeded a sane bound the harvester logs and truncates
  (oldest first) rather than blowing its WS limit. Measurement never
  outranks capture.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import config

log = logging.getLogger("token_pins")

PIN_TTL_S = 900.0          # a pin from a dead process expires in 15 min
MAX_PINS = 96              # hard ceiling on what pinning may cost the WS


def _path() -> Path:
    return config.STATE_DIR / "token_pins.json"


def publish(owner: str, tokens: set[int] | list[int]) -> None:
    """Declare the tokens `owner` is currently tracking. Total: never
    raises into a trading loop."""
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {}
        if p.exists():
            try:
                body = json.loads(p.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError):
                body = {}
        if body.get("_day") != dt.date.today().isoformat():
            body = {"_day": dt.date.today().isoformat()}
        toks = sorted({int(t) for t in tokens if t})
        if not toks:
            body.pop(owner, None)
        else:
            body[owner] = {"ts": time.time(), "tokens": toks}
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:                                 # noqa: BLE001
        log.debug("pin publish failed for %s (%s)", owner, e)


def read_pins() -> set[int]:
    """Tokens no chain harvester may prune right now."""
    try:
        p = _path()
        if not p.exists():
            return set()
        body = json.loads(p.read_text(encoding="utf-8")) or {}
        if body.get("_day") != dt.date.today().isoformat():
            return set()
        now, out, stale = time.time(), set(), []
        for owner, rec in body.items():
            if owner.startswith("_") or not isinstance(rec, dict):
                continue
            age = now - float(rec.get("ts") or 0)
            if age > PIN_TTL_S:
                stale.append((owner, age))
                continue
            out.update(int(t) for t in rec.get("tokens") or [])
        if stale:
            log.info("ignoring %d expired pin owner(s): %s", len(stale),
                     [f"{o} ({a:.0f}s)" for o, a in stale])
        if len(out) > MAX_PINS:
            log.warning("pin set is %d tokens (> %d) — truncating. Capture "
                        "budget outranks measurement.", len(out), MAX_PINS)
            out = set(sorted(out)[:MAX_PINS])
        return out
    except Exception as e:                                 # noqa: BLE001
        log.debug("pin read failed (%s)", e)
        return set()


def clear(owner: str) -> None:
    publish(owner, set())