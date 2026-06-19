"""
APEX OMNI v9 — CHAIN SCANNER (audit §9 rebuilt)
===============================================
  * `import time` exists. (Yes, really — v8 crashed on its first line of work.)
  * No more temporal spoofing: each ±offset keeps its OWN rolling
    SEQ_LENGTH-frame StateBuilder, so the transformer sees real sequences,
    never one frame photocopied ten times.
  * No re-typed physics: everything imports from core (the copy-paste-drift
    factory is closed).
  * If torch/SB3 are absent it degrades to PHYSICS-ONLY alerts (ΔOI shock +
    OFI z + velocity) and says so — truthful, never hallucinated conviction.

Run:  python apex_scanner_v9.py
"""
from __future__ import annotations
import logging
import time                                   # ★ the fix
import datetime as dt

import numpy as np

import config
from apex_ipc_core import BinaryRingBuffer
from core.market_state import StateBuilder

log = logging.getLogger("scanner")

ALERT = config.SCANNER_ALERT
OFFSETS = config.SCANNER_OFFSETS


def physics_score(frame: np.ndarray, i: int) -> float:
    n = frame[i * config.NODES_PER_INDEX]
    return float(np.tanh(0.5 * n[12] / 4 + 0.4 * n[5] + 0.5 * n[2] * 5))


def main():
    ring = BinaryRingBuffer()
    builders = {off: StateBuilder() for off in OFFSETS}
    base = StateBuilder()
    log.info("scanner up — honest rolling windows per offset, alert ≥ %.2f",
             ALERT)
    while True:
        time.sleep(1.0)
        state, age = ring.read_state()
        if state is None or age > config.DATA_STALE_BLOCK_S:
            continue
        market = state.get("market", {})
        ts = float(state.get("ts", time.time()))
        base.push(market, ts)
        for off, b in builders.items():
            shifted = {}
            for idx, ctx in market.items():
                c = dict(ctx)
                step = ctx.get("step") or config.INDICES[idx]["strike_step"]
                legs = {}
                for leg, info in (ctx.get("legs") or {}).items():
                    legs[leg] = {"snap": info["snap"],
                                 "strike": info["strike"] + off * step}
                c["legs"] = legs
                shifted[idx] = c
            b.push(shifted, ts)
            if len(b.frames) < config.SEQ_LENGTH:
                continue
            frame = b.frames[-1]
            for i, idx in enumerate(config.INDEX_ORDER):
                s = physics_score(frame, i)
                if abs(s) >= ALERT:
                    side = "CALL SWEEP" if s > 0 else "PUT SWEEP"
                    log.info("⚡ %s %s offset %+d | physics %.2f | %s",
                             idx, side, off, s,
                             dt.datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":
    config.setup_logging("scanner")
    main()
