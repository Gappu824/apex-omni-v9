"""
signal_persistence.py — is the instantaneous read a SUSTAINED directional
signal, or a one-tick spike / a whipsaw?

Why this exists
---------------
The legacy gate asked two tick-by-tick questions: (a) does the SIGN of the last
N convictions agree ≥75 % of the time, and (b) is the window-average |conv|
above a floor. Both are fragile to the case that matters most intraday: a tape
that ticks up-down-up-down but NETS to a real move. Sign-agreement counts every
counter-tick as evidence against the trend, so a genuine noisy down-move
(say 4 down ticks, 2 up blips) reads as "67 % agreement → whipsaw → skip", and
the absolute-value average throws away the very directionality we are testing.

This module replaces both with NET-DISPLACEMENT / SIGNAL-TO-NOISE measures, which
are the textbook way to separate trend from noise on a short window.

Research basis
--------------
• Kaufman Efficiency Ratio — Perry J. Kaufman, *Trading Systems and Methods*
  (the ratio that drives KAMA, Kaufman's Adaptive Moving Average):
        ER = |p_t − p_{t−n}| / Σ_{i}|p_i − p_{i−1}|        ∈ [0, 1]
  numerator = net displacement, denominator = total path length. ER ≈ 1 is a
  clean directional move; ER ≈ 0 is chop. This is exactly "net displacement over
  a window" and is robust to oscillation that nets to a trend — the case
  tick-agreement gets wrong. We apply it to the per-second SPOT path.
• Lo & MacKinlay (1988), "Stock Market Prices Do Not Follow Random Walks":
  the variance-ratio test, VR(q) = Var(q-period return)/(q·Var(1-period return));
  VR>1 ⇒ persistent/trending, <1 ⇒ mean-reverting. ER is preferred here because
  it is stable on the very short windows an entry gate runs on, where a variance
  estimate is itself too noisy to trust.

The conviction series is a LEVEL signal (not increments), so its directional
coherence is measured by the magnitude-weighted ratio
        CCR = |mean(c)| / mean(|c|)                          ∈ [0, 1]
the continuous, size-weighted generalisation of "fraction of ticks agreeing in
sign": a couple of small opposite blips no longer veto a strong consistent read,
while a symmetric flip-flop (mean → 0) is correctly rejected. The SIGNED mean
also fixes the direction itself — the trade side is sign(mean(c)), taken from the
net of the window, not from the latest tick.

Decision (a feature is a sustained signal when ALL hold)
--------------------------------------------------------
  1. |mean(conv)|            ≥ conv_floor      actionable net strength (signed,
                                               so oscillation nets out)
  2. CCR = |mean|/mean|·|    ≥ ccr_min         directional coherence (not whipsaw)
  3. sign(conv_now)          == sign(mean)     latest tick not against the net read
  4. price ER (Kaufman)      ≥ er_min          the TAPE itself is trending, not
                                               chopping  (skipped if too little
                                               spot history yet)
  5. [optional] sign(net spot move) == sign(mean(conv))     the read agrees with
                                               the tape — a momentum confirmation
                                               that also guards a mis-signed
                                               policy until it is retrained.

All thresholds are getattr-defaults so they can be tuned from config WITHOUT
touching the model fingerprint (no re-forge). Promote them into config.py's
hash-EXCLUDE block to pin them.

`conv > 0` ⇒ CE (bullish, expects spot UP); `conv < 0` ⇒ PE (bearish). Rule 5
uses that convention to compare the read against the realised spot move.
"""
from __future__ import annotations

import math
from typing import Sequence

try:
    import config
except Exception:                                              # pragma: no cover
    config = None


def _cfg(name: str, default):
    return getattr(config, name, default) if config is not None else default


# ---- tunables (config overrides win; defaults keep it self-contained) --------
def _params() -> dict:
    return {
        # rule 1: actionable net strength is handed in per-call (the live floor:
        # META_ENTRY_CONV_FLOOR in model mode, else the bootstrap bar).
        "ccr_min":     float(_cfg("SIGNAL_PERSIST_CCR_MIN", 0.55)),   # rule 2
        "er_window":   int(_cfg("SIGNAL_PERSIST_ER_WINDOW", 20)),     # rule 4 (spot samples ≈ seconds)
        "er_min":      float(_cfg("SIGNAL_PERSIST_ER_MIN", 0.30)),    # rule 4
        "price_agree": bool(_cfg("SIGNAL_PERSIST_PRICE_AGREE", True)),# rule 5 on/off
        "price_tol_bp": float(_cfg("SIGNAL_PERSIST_PRICE_TOL_BP", 1.0)),  # net move < this ⇒ tape directionless
    }


def efficiency_ratio(path: Sequence[float]) -> float:
    """Kaufman Efficiency Ratio of a price path ∈ [0,1]; 0 if undefined."""
    if path is None or len(path) < 2:
        return 0.0
    net = abs(float(path[-1]) - float(path[0]))
    total = 0.0
    prev = float(path[0])
    for p in path[1:]:
        p = float(p)
        total += abs(p - prev)
        prev = p
    return (net / total) if total > 1e-12 else 0.0


def assess_persistence(conv: float,
                       conv_window: Sequence[float],
                       spot_window: Sequence[float],
                       *,
                       conv_floor: float,
                       params: dict | None = None) -> tuple[bool, str, dict]:
    """Return (ok, reason, diag).

    conv         : current signed conviction (post fusion / regime mult)
    conv_window  : recent signed convictions (the rolling persistence window)
    spot_window  : recent per-second spot prices for this index (for Kaufman ER)
    conv_floor   : actionable magnitude floor for the ACTIVE decision path
    """
    p = params or _params()
    n = len(conv_window)
    if n == 0:
        return True, "warming up", {"n": 0}

    mu = sum(conv_window) / n                      # signed net conviction
    mad = sum(abs(c) for c in conv_window) / n     # mean magnitude
    ccr = (abs(mu) / mad) if mad > 1e-12 else 0.0  # directional coherence
    direction = 1 if mu > 0 else (-1 if mu < 0 else 0)

    diag = {"n": n, "mean_conv": round(mu, 4), "ccr": round(ccr, 3),
            "direction": direction, "conv_floor": round(conv_floor, 3)}

    # 1) actionable net strength (signed mean → symmetric oscillation nets to ~0)
    if abs(mu) < conv_floor:
        return False, (f"weak net read |mean conv| {abs(mu):.2f}<"
                       f"{conv_floor:.2f}"), diag
    # 2) directional coherence (replaces tick-by-tick sign agreement)
    if ccr < p["ccr_min"]:
        return False, f"whipsaw: coherence {ccr:.2f}<{p['ccr_min']:.2f}", diag
    # 3) the latest tick must not point against the window's net direction
    if direction != 0 and (1 if conv > 0 else -1) != direction:
        return False, "spike against net direction", diag

    # 4) Kaufman ER on the tape — only once enough spot history has accrued, so
    #    warm-up never blocks the first valid signals (matches legacy behaviour).
    sw = list(spot_window or ())
    if len(sw) >= max(p["er_window"], 3):
        seg = sw[-p["er_window"]:]
        er = efficiency_ratio(seg)
        diag["price_er"] = round(er, 3)
        if er < p["er_min"]:
            return False, f"chop: price ER {er:.2f}<{p['er_min']:.2f}", diag
        # 5) optional momentum confirmation: the read must agree with the tape.
        if p["price_agree"]:
            net = seg[-1] - seg[0]
            ref = seg[0] if abs(seg[0]) > 1e-9 else 1.0
            net_bp = (net / ref) * 1e4
            price_dir = 1 if net_bp > p["price_tol_bp"] else (
                -1 if net_bp < -p["price_tol_bp"] else 0)
            diag["net_bp"] = round(net_bp, 2)
            if price_dir != 0 and price_dir != direction:
                side = "CE" if direction > 0 else "PE"
                tape = "up" if price_dir > 0 else "down"
                return False, (f"read {side} vs tape {tape} "
                               f"({net_bp:+.0f}bp) — disagree"), diag

    return True, "persistent", diag


# ----------------------------------------------------------------- self-test
def _legacy(conv, ch, floor, frac=0.75, mult=0.95):
    """The OLD gate, for side-by-side comparison in the self-test."""
    if len(ch) == 0:
        return True, "warmup"
    same = sum(1 for c in ch if (c > 0) == (conv > 0))
    avg = sum(abs(c) for c in ch) / len(ch)
    if same / len(ch) < frac:
        return False, f"whipsaw dir {same/len(ch)*100:.0f}%<{frac*100:.0f}%"
    if avg < floor * mult:
        return False, f"avg|conv| {avg:.2f}<{floor*mult:.2f}"
    return True, "persistent"


if __name__ == "__main__":
    FLOOR = 0.40                                   # META_ENTRY_CONV_FLOOR
    # a gently falling tape (net down ~ -0.4 %) with realistic chop, and a
    # symmetric chop tape that nets nowhere.
    down_tape = [23900 - 8 * i + (12 if i % 2 else -12) for i in range(25)]
    chop_tape = [23900 + (25 if i % 2 else -25) for i in range(25)]

    cases = [
        # name, conv_now, conv_window, spot_window
        ("strong noisy DOWN-trend (the case you raised)",
         -0.82, [-0.9, +0.2, -0.8, +0.1, -0.9, -0.7], down_tape),
        ("symmetric whipsaw (true chop)",
         +0.85, [+0.9, -0.9, +0.8, -0.85, +0.9, -0.88], chop_tape),
        ("clean strong DOWN",
         -0.80, [-0.78, -0.82, -0.80, -0.79], down_tape),
        ("weak net read (below floor)",
         -0.35, [-0.33, -0.40, -0.30, -0.38], down_tape),
        ("MIS-SIGNED: conv says CE while tape fell (today's bias)",
         +0.78, [+0.74, +0.80, +0.76, +0.79], down_tape),
    ]
    print(f"{'scenario':52s} | {'OLD':26s} | NEW")
    print("-" * 110)
    for name, c, win, sw in cases:
        lo_ok, lo_why = _legacy(c, win, FLOOR)
        nw_ok, nw_why, _ = assess_persistence(c, win, sw, conv_floor=FLOOR)
        print(f"{name:52s} | {('PASS ' if lo_ok else 'SKIP ')+lo_why:26s} "
              f"| {('PASS ' if nw_ok else 'SKIP ')+nw_why}")