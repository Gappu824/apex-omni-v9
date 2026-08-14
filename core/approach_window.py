"""
APPROACH WINDOW — the tape leading into a signal, as a tensor
==============================================================
WHY THIS FILE EXISTS
--------------------
Every model this system has fitted reads a hand-summarised SNAPSHOT: ~64
numbers describing the instant a signal fired. core/seq_model.py can read
the APPROACH instead — the shape of the minutes before that instant — but
only if something hands it the tape. Without this extractor the CNN-GRU
receives a one-step window, degenerates to an MLP, and its entire reason
to exist is unavailable.

Two signals with identical snapshots can arrive on completely different
paths: a slow grind into the level, or a spike that already failed once
and is retesting. Spot, IV rank and net GEX may read the same at t=0 and
mean opposite things. That difference is what a sequence encoder is for,
and it is thrown away by construction in a snapshot.

WHAT IS EXTRACTED, AND WHY EACH CHANNEL
----------------------------------------
W seconds ending at the signal second, C channels, all DERIVED from data
the replayer already holds — no new query, no second data path that could
drift from the one the grader uses:

  spot_ret      log return per second        direction and pace of travel
  spot_absret   |ret|                        realised vol shape; a burst
                                             and a drift differ here even
                                             when net travel matches
  spot_z        (spot - mean)/sd of window   position within the range
  leg_mid_ret   the chosen leg's own return  premium behaviour, which is
                                             not a deterministic function
                                             of spot once IV moves
  leg_spread    (ask-bid)/mid                liquidity through the
                                             approach; a widening book
                                             before entry is the trap
                                             signature TrapShield hunts
  leg_fresh     1 if a real tick, else 0     the honesty channel. Carried
                                             quotes are NOT ticks, and a
                                             model that cannot tell them
                                             apart will read a dead feed
                                             as a calm market — the exact
                                             flat-line error that made
                                             hold_to_close look free.

THE RULE THAT MATTERS: STRICTLY CAUSAL
---------------------------------------
The window ends at t-1, never t. Not caution — necessity. Including the
signal second leaks the bar that triggered the entry into the features
that are supposed to predict it, and the model would learn to detect its
own trigger. That is the most common way a sequence model on financial
data produces a spectacular backtest and nothing else.

Standardisation is per-window and causal too: each channel is centred and
scaled using ONLY its own window. A global scaler fitted across the vault
would carry future sessions' variance into a past window — a subtle leak
that survives cross-validation because the folds never see it.

COST
----
float16, W=300, C=6 is 3.6 KB per episode; a 40-session vault is a few MB.
The arrays already exist in the replayer, so this is a slice and a
subtraction per sample, not a re-read.
"""
from __future__ import annotations

import logging

import numpy as np

import config

log = logging.getLogger("approach_window")

CHANNELS = ("spot_ret", "spot_absret", "spot_z", "leg_mid_ret",
            "leg_spread", "leg_fresh")
N_CH = len(CHANNELS)


def window_len() -> int:
    return int(getattr(config, "SEQ_WINDOW_S", 300))


def extract(rep, t: int, token: int | None, idx: str,
            w: int | None = None) -> np.ndarray | None:
    """The (W, C) approach ending at t-1, or None if the tape is too thin.

    `rep` is a nightly_forge _Replayer: it already carries spot_hist (a
    1800s deque per index), the dense bidA/askA arrays and the ti map, so
    nothing here re-reads the vault. Using the replayer's OWN arrays is
    deliberate — a second data path could drift from the one the grader
    prices labels on, and then the features would describe a different
    world than the target.
    """
    w = int(w or window_len())
    if t <= w:
        return None                     # not enough session before it

    # ---- spot channels
    try:
        sh = rep.spot_hist.get(idx)
        spot = np.asarray(sh, dtype=float) if sh is not None else None
    except Exception:                                      # noqa: BLE001
        spot = None
    if spot is None or spot.size < w + 2:
        return None
    s = spot[-(w + 1):]                 # ends at t-1: STRICTLY CAUSAL
    if s.size < w + 1 or not np.isfinite(s).all() or (s <= 0).any():
        return None
    ret = np.diff(np.log(s))
    absret = np.abs(ret)
    sd = float(np.std(s[:-1])) or 1.0
    z = (s[:-1] - float(np.mean(s[:-1]))) / sd

    # ---- leg channels
    mid_ret = np.zeros(w)
    spread = np.zeros(w)
    fresh = np.zeros(w)
    k = rep.ti.get(int(token)) if token else None
    if k is not None:
        lo = max(t - w, 0)
        b = np.asarray(rep.bidA[k, lo:t], dtype=float)
        a = np.asarray(rep.askA[k, lo:t], dtype=float)
        if b.size == w and a.size == w:
            good = np.isfinite(b) & np.isfinite(a) & (b > 0) & (a > 0)
            mid = np.where(good, (a + b) / 2.0, np.nan)
            spread = np.where(good & (mid > 0), (a - b) / np.maximum(mid,
                                                                    1e-9), 0.0)
            lm = np.log(np.where(np.isfinite(mid) & (mid > 0), mid, np.nan))
            d = np.diff(lm, prepend=lm[0])
            mid_ret = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            # a CARRIED quote is not a tick. The model is told which is
            # which rather than being left to read a dead feed as calm.
            last = rep.last_tick.get(int(token), -10 ** 9)
            fresh = good.astype(float)
            if last < lo:
                fresh[:] = 0.0

    X = np.stack([ret, absret, z, mid_ret, spread, fresh], axis=1)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # PER-WINDOW standardisation, causal by construction. A scaler fitted
    # across the vault would carry later sessions' variance into an earlier
    # window — a leak that survives cross-validation because no fold ever
    # sees it. `leg_fresh` is left alone: it is a flag, not a magnitude.
    for c in range(X.shape[1]):
        if CHANNELS[c] == "leg_fresh":
            continue
        mu = float(np.mean(X[:, c]))
        sg = float(np.std(X[:, c]))
        X[:, c] = (X[:, c] - mu) / (sg if sg > 1e-12 else 1.0)
    return X.astype(np.float16)


def empty(w: int | None = None) -> np.ndarray:
    return np.zeros((int(w or window_len()), N_CH), dtype=np.float16)


def summarise(n_ok: int, n_total: int, logger=None) -> None:
    lg = logger or log
    if not n_total:
        return
    frac = n_ok / n_total
    lg.info("approach windows: %d/%d sample(s) (%.0f%%) have a full %ds "
            "causal window across %d channel(s)", n_ok, n_total,
            100 * frac, window_len(), N_CH)
    if frac < 0.5:
        lg.warning("fewer than half the samples carry a usable window — "
                   "most signals fire in the first %d seconds of the "
                   "session, before enough tape exists. The sequence model "
                   "will be fitted on the later-session subset, which is "
                   "NOT the same population the snapshot model sees.",
                   window_len())