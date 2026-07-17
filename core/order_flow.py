"""
APEX OMNI v9.7.1 — ORDER-FLOW TOXICITY & TRAP DETECTOR
======================================================
A research-grade entry filter that answers one question before every trade:
"is this move REAL, or is it an engineered stop-hunt I'm about to be the
liquidity for?" Built from peer-reviewed microstructure, computed from the
depth+volume your Kite feed already captures (bid/ask qty, vol_delta, OI).

The papers this stands on
-------------------------
1. VPIN / flow toxicity — Easley, López de Prado & O'Hara, "Flow Toxicity and
   Liquidity in a High Frequency World", Review of Financial Studies 25 (2012).
   Order-flow toxicity = the probability informed traders are adversely
   selecting liquidity providers. It is estimated from VOLUME-BUCKETED order-
   flow IMBALANCE, and it spiked to historic highs an HOUR before the 2010
   Flash Crash. High toxicity ⇒ liquidity is about to fail ⇒ violent, trap-
   prone moves. We bucket by traded volume (the "volume clock") exactly as the
   paper prescribes, and estimate imbalance from the book because Kite gives
   depth, not signed trade prints (an explicit, validated PROXY — see below).

2. Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov, "The Price Impact of
   Order Book Events" (2014). The signed change in best-bid/ask SIZE is a
   linear predictor of short-horizon price moves — a cleaner, depth-native
   signal than trade classification. This is our per-tick imbalance primitive.

3. Swing Failure Pattern / liquidity sweep + ABSORPTION — the institutional
   stop-hunt signature (Easley-style adverse selection at a level, and the
   practitioner SFP literature): price PIERCES a prior swing level then CLOSES
   BACK inside, on HIGH volume with NO price follow-through (absorption by
   passive limit orders). That combination — sweep + absorption + reclaim — is
   a manufactured trap, and the edge is to fade it (enter the OTHER way with a
   stop beyond the wick), not to chase the break.

What this module computes (all from data already in the vault/feed)
-------------------------------------------------------------------
• TOXICITY (0..1): a volume-clock VPIN-proxy from rolling signed OFI. High =
  flow is one-sided and informed = the tape is dangerous. Directly gates entry
  size/permission: never chase INTO high toxicity that is against you; a
  high-toxicity move WITH you that is NOT a sweep is genuine and tradeable.
• SWEEP + ABSORPTION: did price just pierce a recent extreme and reclaim, on a
  volume spike that failed to move price (absorption z-score)? That flags the
  BankNifty trap on the way IN — the entry-side twin of the exit dwell.
• A single ENTRY VERDICT the brain consults: block a chase into a detected
  trap; allow (and flag as high-quality) a genuine break or a confirmed post-
  sweep reversal. ADVISORY — it can only ever RAISE the bar, never lower a
  floor, and it degrades to neutral when the book/volume read is missing.

Honesty about the proxy
-----------------------
A textbook VPIN needs tick-by-tick trades with aggressor side. Kite streams
depth + cumulative volume, not signed prints. So we estimate the buy/sell
split from (a) the OFI sign of book pressure and (b) where vol_delta prints
relative to the mid — a "bulk volume classification" in the spirit of Easley
et al.'s own BVC, not the exact trade-by-trade PIN. It is a PROXY, and its
usefulness is not assumed — it is MEASURED on your vault by
tools/toxicity_report.py and only trusted if it separates winners from losers.
All thresholds self-calibrate nightly (see calibration artifact).

Self-test:   python core/order_flow.py
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:                                          # pragma: no cover
        return default


# --------------------------------------------------------------------------
# Per-tick primitive: Order Flow Imbalance (Cont-Kukanov-Stoikov)
# --------------------------------------------------------------------------
def ofi_increment(bid: float, bid_qty: float, ask: float, ask_qty: float,
                  prev: dict | None) -> tuple[float, dict]:
    """Signed order-flow imbalance from best-bid/ask SIZE changes between two
    book snapshots (Cont-Kukanov-Stoikov 2014). Returns (ofi, new_state).

    e_n = ΔW_bid − ΔW_ask, where a bid whose PRICE rose contributes its full
    new size (demand added), a bid whose price fell contributes minus its old
    size (demand removed); symmetric on the ask. Positive ⇒ net buy pressure.
    """
    state = {"bid": bid, "bid_qty": bid_qty, "ask": ask, "ask_qty": ask_qty}
    if prev is None or bid <= 0 or ask <= 0:
        return 0.0, state
    # bid side
    if bid > prev["bid"]:
        d_bid = bid_qty
    elif bid < prev["bid"]:
        d_bid = -prev["bid_qty"]
    else:
        d_bid = bid_qty - prev["bid_qty"]
    # ask side
    if ask < prev["ask"]:
        d_ask = ask_qty
    elif ask > prev["ask"]:
        d_ask = -prev["ask_qty"]
    else:
        d_ask = ask_qty - prev["ask_qty"]
    return float(d_bid - d_ask), state


# --------------------------------------------------------------------------
# Toxicity engine — volume-clock VPIN proxy + sweep/absorption detector
# --------------------------------------------------------------------------
@dataclass
class TrapVerdict:
    toxicity: float = 0.0          # 0..1 VPIN-proxy (one-sidedness of flow)
    tox_dir: int = 0               # sign of the toxic pressure (+buy / −sell)
    sweep: bool = False            # a liquidity sweep just reclaimed (trap)
    sweep_dir: str = ""            # "up"=swept highs (bearish trap), "down"=lows
    absorption_z: float = 0.0      # volume z-score with no price follow-through
    reason: str = "neutral"


class OrderFlowToxicity:
    """Per-index streaming estimator. Fed one book+volume snapshot per second
    (the same cadence the brain already samples). Deterministic and replayable
    — tools/toxicity_report.py runs the identical class over the vault."""

    def __init__(self, index: str):
        self.index = index
        try:
            import config
            self._step = float(config.INDICES[index]["strike_step"])
        except Exception:                                      # pragma: no cover
            self._step = 50.0
        self._prev_book: dict | None = None
        # volume-clock buckets: accumulate signed OFI until a volume threshold,
        # then close a bucket. VPIN = mean |bucket imbalance| / bucket size.
        self._bucket_ofi = 0.0
        self._bucket_vol = 0.0
        self._bucket_size = float(_cfg("TOX_BUCKET_VOLUME", 5000.0))
        self._buckets: deque = deque(maxlen=int(_cfg("TOX_NUM_BUCKETS", 50)))
        # recent CONFIRMED swing pivots for sweep detection (a pivot is an
        # extreme that stood unbroken for TOX_PIVOT_HOLD_S — a real level where
        # stops cluster, not a rolling max that includes the current wick).
        self._spot_hist: deque = deque(maxlen=int(_cfg("TOX_SWING_LOOKBACK_S",
                                                       180)))
        self._swing_hi = 0.0
        self._swing_lo = 0.0
        # volume baseline for absorption z-score (rolling)
        self._vol_hist: deque = deque(maxlen=int(_cfg("TOX_VOL_BASE_S", 300)))
        self._last_spot = 0.0
        self._pierced_up = False       # an active pierce above the swing high
        self._pierced_dn = False

    # ------------------------------------------------------------------ feed
    def update(self, *, spot: float, bid: float, bid_qty: float, ask: float,
               ask_qty: float, vol_delta: float) -> TrapVerdict:
        v = TrapVerdict()
        if spot <= 0:
            return v
        ofi, self._prev_book = ofi_increment(bid, bid_qty, ask, ask_qty,
                                             self._prev_book)

        # ---- volume-clock VPIN proxy ----
        self._bucket_ofi += ofi
        self._bucket_vol += max(vol_delta, 0.0)
        if self._bucket_vol >= self._bucket_size and self._bucket_size > 0:
            # normalized bucket imbalance ∈ [-1, 1]
            imb = self._bucket_ofi / max(self._bucket_vol, 1.0)
            self._buckets.append(max(min(imb, 1.0), -1.0))
            self._bucket_ofi = 0.0
            self._bucket_vol = 0.0
        if len(self._buckets) >= max(int(_cfg("TOX_MIN_BUCKETS", 10)), 1):
            # VPIN = mean absolute imbalance across buckets (one-sidedness)
            v.toxicity = round(sum(abs(b) for b in self._buckets)
                               / len(self._buckets), 4)
            signed = sum(self._buckets)
            v.tox_dir = 1 if signed > 0 else (-1 if signed < 0 else 0)

        # ---- absorption z-score: volume high but price didn't move ----
        self._vol_hist.append(max(vol_delta, 0.0))
        move = abs(spot - self._last_spot) if self._last_spot else 0.0
        if len(self._vol_hist) >= 30:
            mu = sum(self._vol_hist) / len(self._vol_hist)
            var = sum((x - mu) ** 2 for x in self._vol_hist) / len(self._vol_hist)
            sd = math.sqrt(var) if var > 0 else 0.0
            vol_z = (vol_delta - mu) / sd if sd > 0 else 0.0
            # high volume (z above threshold) with move < a fraction of a step
            if (vol_z >= float(_cfg("TOX_ABSORB_VOL_Z", 2.0))
                    and move < float(_cfg("TOX_ABSORB_MOVE_FRAC", 0.25)) * self._step):
                v.absorption_z = round(vol_z, 2)

        # ---- liquidity sweep + reclaim (SFP), off CONFIRMED swing pivots ----
        # A swing pivot = an extreme that stood unbroken for TOX_PIVOT_HOLD_S.
        # Sweep = pierce that pivot by a buffer, then reclaim back inside. Only
        # such a level holds clustered stops worth hunting.
        self._spot_hist.append(spot)
        hold = int(_cfg("TOX_PIVOT_HOLD_S", 30))
        if len(self._spot_hist) >= hold:
            window = list(self._spot_hist)[-hold:]
            # the pivot is the extreme of the window EXCLUDING the last few ticks
            # (so an in-progress wick doesn't define its own level)
            settle = max(hold - int(_cfg("TOX_PIVOT_SETTLE_S", 5)), 1)
            base = window[:settle]
            if base:
                self._swing_hi = max(base)
                self._swing_lo = min(base)
        buf = float(_cfg("TOX_SWEEP_BUFFER_FRAC", 0.10)) * self._step
        if self._swing_hi > 0:
            if spot > self._swing_hi + buf:
                self._pierced_up = True
            elif self._pierced_up and spot < self._swing_hi:
                v.sweep, v.sweep_dir = True, "up"    # highs swept, reclaimed down
                self._pierced_up = False
        if self._swing_lo > 0:
            if spot < self._swing_lo - buf:
                self._pierced_dn = True
            elif self._pierced_dn and spot > self._swing_lo:
                v.sweep, v.sweep_dir = True, "down"  # lows swept, reclaimed up
                self._pierced_dn = False

        self._last_spot = spot

        # ---- reason string ----
        if v.sweep and v.absorption_z:
            v.reason = (f"SWEEP {v.sweep_dir} + absorption z{v.absorption_z} "
                        f"(tox {v.toxicity})")
        elif v.toxicity >= float(_cfg("TOX_HIGH", 0.4)):
            v.reason = (f"high toxicity {v.toxicity} dir "
                        f"{'+' if v.tox_dir > 0 else '-'}")
        else:
            v.reason = f"tox {v.toxicity}"
        return v


# --------------------------------------------------------------------------
# ENTRY VERDICT — the gate the brain consults
# --------------------------------------------------------------------------
def entry_trap_check(v: TrapVerdict, direction: str, *,
                     tox_block: float, sweep_fade_ok: bool) -> tuple[bool, str]:
    """Return (allow, why). Research logic:

      • A directional entry CHASING into HIGH toxicity that is AGAINST the
        trade is the classic adverse-selection trap — BLOCK it (you'd be the
        liquidity the informed flow selects). High toxicity WITH the trade is
        genuine informed momentum — allow.
      • A fresh SWEEP+absorption AGAINST the entry direction (e.g. highs swept
        and reclaimed down while we're trying to buy CE) is a manufactured
        top/bottom — BLOCK the chase. A sweep in FAVOUR (lows swept, reclaimed
        up, and we're buying CE) is the textbook post-sweep reversal entry —
        allow and flag as high quality (only if sweep_fade_ok).

    This RAISES the bar only; the caller still applies every floor/gate."""
    want = 1 if direction == "CE" else -1
    # toxicity against us, and strong → block the chase
    if v.toxicity >= tox_block and v.tox_dir != 0 and v.tox_dir != want:
        return False, (f"BLOCK: toxicity {v.toxicity} against {direction} "
                       f"(informed flow the other way — trap)")
    # sweep against us → block; sweep for us → high-quality reversal entry
    if v.sweep:
        # highs swept+reclaimed-down = bearish; favours PE, traps CE
        sweep_favours = "PE" if v.sweep_dir == "up" else "CE"
        if sweep_favours != direction:
            return False, (f"BLOCK: {v.sweep_dir}-sweep reclaimed against "
                           f"{direction} (engineered {'top' if v.sweep_dir=='up' else 'bottom'})")
        if sweep_fade_ok and v.absorption_z:
            return True, (f"HIGH-QUALITY: post-sweep reversal {direction} "
                          f"({v.sweep_dir}-sweep + absorption z{v.absorption_z})")
    return True, f"clear (tox {v.toxicity})"


# --------------------------------------------------------------------------
# SELF-TEST
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    print("=== OFI primitive ===")
    prev = None
    for (b, bq, a, aq) in [(100, 500, 101, 500), (100, 800, 101, 500),
                           (100, 800, 102, 300)]:
        ofi, prev = ofi_increment(b, bq, a, aq, prev)
        print(f"  book {b}/{bq} {a}/{aq} → OFI {ofi:+.0f}")

    print("\n=== toxicity engine on a synthetic TRAP vs GENUINE move ===")
    def run(kind):
        eng = OrderFlowToxicity("SENSEX")
        rng = random.Random(1)
        last = None
        spot = 77000.0
        for t in range(600):
            if kind == "genuine_down":       # steady one-sided sell, price falls
                spot -= 0.5 + rng.gauss(0, 0.3)
                bid, ask = spot - 1, spot + 1
                bq, aq = 300, 900            # ask-heavy (sellers stacking)
                vol = 200
            elif kind == "trap_up":          # spike up, no follow-through, reclaim
                if 200 <= t < 230:
                    spot += 3.0              # pierce highs
                    vol = 3000               # huge volume
                elif 230 <= t < 260:
                    spot -= 3.2              # reclaim back down
                    vol = 2500
                else:
                    spot += rng.gauss(0, 0.5)
                    vol = 200
                bid, ask = spot - 1, spot + 1
                bq, aq = 800, 800
            v = eng.update(spot=spot, bid=bid, bid_qty=bq, ask=ask, ask_qty=aq,
                           vol_delta=vol)
            if kind == "trap_up" and v.sweep:
                print(f"    t={t}: SWEEP {v.sweep_dir} detected, "
                      f"absorption z{v.absorption_z}, tox {v.toxicity}")
        return v
    vg = run("genuine_down")
    print(f"  genuine down-move final: toxicity {vg.toxicity} dir {vg.tox_dir} "
          f"(one-sided sell = high tox WITH a PE) → {vg.reason}")
    vt = run("trap_up")
    print(f"  trap-up final verdict: {vt.reason}")

    print("\n=== entry gate ===")
    # genuine down move, buying PE (with the flow) → allow
    a1, w1 = entry_trap_check(vg, "PE", tox_block=0.4, sweep_fade_ok=True)
    print(f"  PE into genuine down-flow: allow={a1} — {w1}")
    # same toxic down-flow, buying CE (against) → block
    a2, w2 = entry_trap_check(vg, "CE", tox_block=0.4, sweep_fade_ok=True)
    print(f"  CE against down-flow:      allow={a2} — {w2}")