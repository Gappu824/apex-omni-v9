"""
APEX OMNI — CROSS-INDEX PEER CONTEXT
====================================
MEASURED FACT THAT MOTIVATES THIS (tools/cross_index_overlap.py, 12 days,
739 signals, hash eb09a42fbc):

    co-fire rate            0.5602   (same direction, another index, +/-5s)
    outcome agreement       0.8486   vs 0.6154 expected by chance
    intra-cluster rho       0.6063
    design effect           1.2775   -> n_effective 578 of 739

So 56% of signals have a twin on a correlated index, and when they pair their
outcomes agree far above chance. The models could not see any of it: the meta's
x-vector is frame[b0], frame[b0+1], frame[b0+2] — three nodes of ONE index —
and HeuristicPolicy scores each index from its own slice. NIFTY's vector has
never contained one bit about BANKNIFTY or SENSEX.

WHAT THIS ADDS, AND WHY IT IS THREE FEATURES AND NOT FORTY
----------------------------------------------------------
The naive move is to concatenate the peers' nodes: +38 dims onto a 61-dim model
whose effective sample is ~785 and whose AUC is 0.60. That is how you
manufacture overfitting — more parameters, same information. Instead we add the
smallest set that carries the ECONOMIC content of the correlation:

  peer_agree      correlation-free mean of peer convictions signed toward THIS
                  index's direction. High => the whole complex is moving my way.
  peer_max_agree  the single strongest agreeing peer. Separates "one peer is
                  screaming" from "everyone is mildly nodding".
  peer_dispersion spread of peer convictions. THIS IS THE NOISE TERM: a move
                  the complex agrees on (low dispersion) is a market event; a
                  move one index makes alone while its peers disagree (high
                  dispersion) is far more likely idiosyncratic noise.

No correlation matrix is imposed. The weights are LEARNED by the GBM from
outcomes, which is both more honest and more adaptive than a hardcoded rho —
if BANKNIFTY stops tracking NIFTY, the model discovers it from the labels
rather than from a constant someone forgot to update.

SEPARATE BUT CONNECTED
----------------------
Each index keeps its own decision, its own position, its own risk. The peer
signal enters only as CONTEXT on the gate — exactly the "separate but
connected" shape: no shared parameters, no merged action space, no coupling of
the books. One index's noise cannot move another index's trade; it can only
inform how much to trust that trade.

FAIL-SAFE AND FALSIFIABLE
-------------------------
Returns zeros when peers are unavailable, so an absent frame degrades to the
old 61-dim behaviour rather than crashing. And it is falsifiable by the
machinery already in place: if these features carry nothing, the forge's AUC
and BSS guards will show it and the model will not promote. The system can
decide for itself whether this earns its place.
"""
from __future__ import annotations

import numpy as np

N_PEER_FEATURES = 3


def peer_features(conv_by_index, iidx: int, direction: str,
                  peers: list[int] | None = None) -> list[float]:
    """Cross-index context for index `iidx` taking `direction`.

    conv_by_index : per-INDEX_ORDER signed conviction, +ve = bullish (CE).
    iidx          : this index's position in INDEX_ORDER.
    direction     : "CE" or "PE" — the side actually being taken.
    peers         : INDEX_ORDER positions to treat as peers. None = every other
                    index present in conv_by_index.

    Returns [peer_agree, peer_max_agree, peer_dispersion], each clipped to
    [-1, 1] / [0, 1] so a bad frame cannot inject an outlier into the model.
    """
    try:
        conv = np.asarray(conv_by_index, dtype=float).ravel()
    except Exception:                                          # noqa: BLE001
        return [0.0] * N_PEER_FEATURES
    if conv.size == 0 or not (0 <= iidx < conv.size):
        return [0.0] * N_PEER_FEATURES
    idxs = [j for j in (peers if peers is not None else range(conv.size))
            if j != iidx and 0 <= j < conv.size]
    if not idxs:
        return [0.0] * N_PEER_FEATURES
    p = conv[idxs]
    p = p[np.isfinite(p)]
    if p.size == 0:
        return [0.0] * N_PEER_FEATURES
    # sign the peers toward the side WE are taking: a PE trade wants peers
    # bearish, so a negative peer conviction is agreement.
    s = 1.0 if str(direction).upper() == "CE" else -1.0
    signed = np.clip(p * s, -1.0, 1.0)
    agree = float(signed.mean())
    max_agree = float(signed.max())
    # dispersion on the RAW peer convictions — disagreement among peers is
    # direction-independent, and it is the noise term.
    disp = float(np.std(np.clip(p, -1.0, 1.0))) if p.size > 1 else 0.0
    return [round(agree, 6), round(max_agree, 6), round(min(disp, 1.0), 6)]


def convictions_from_actions(actions, n_indices: int) -> list[float]:
    """Per-index conviction from a policy action vector.

    PARITY: the forge reads `pol.predict(frame)[2 * iidx]` and apex_main reads
    `policy.conviction(obs, frame)[2 * i]` — and with POLICY_ENGINE="meta"
    PolicyLoader.conviction() returns HeuristicPolicy().predict(frame), so both
    are the SAME vector with the same stride-2 layout. Extracting through one
    function keeps them that way: if the layout ever changes, it changes for
    both paths at once and cannot silently skew training against serving.
    """
    try:
        a = np.asarray(actions, dtype=float).ravel()
    except Exception:                                          # noqa: BLE001
        return [0.0] * int(n_indices)
    out = []
    for j in range(int(n_indices)):
        k = 2 * j
        v = float(a[k]) if k < a.size and np.isfinite(a[k]) else 0.0
        out.append(v)
    return out


def peer_feature_names() -> list[str]:
    return ["peer_agree", "peer_max_agree", "peer_dispersion"]