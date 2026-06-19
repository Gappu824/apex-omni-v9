"""
APEX OMNI v9 — BOOTSTRAP HEURISTIC POLICY (shared: live brain + replay)
=======================================================================
Transparent physics fallback used until the forge promotes a trained pair.
Index SPOT ticks carry no volume/book, so the signal lives in the ATM option
nodes: CE-vs-PE differentials of order-flow imbalance, dealer inventory and
trade velocity, plus spot momentum. All weights in config.HEURISTIC_W.
"""
from __future__ import annotations
import math
import numpy as np

import config


class HeuristicPolicy:
    def predict(self, frame_30x19: np.ndarray) -> np.ndarray:
        a = np.zeros(config.ACTION_DIM, np.float32)
        for i, idx in enumerate(config.INDEX_ORDER):
            b = i * config.NODES_PER_INDEX
            spot, ce, pe = frame_30x19[b], frame_30x19[b + 1], frame_30x19[b + 2]
            if not ce.any() and not pe.any():
                continue                      # no option legs streaming yet
            w_ofi, w_dlr, w_vel, w_mom = config.HEURISTIC_W
            flow = (w_ofi * (ce[12] - pe[12]) / 4.0     # OFI z differential
                    + w_dlr * (ce[16] - pe[16])         # dealer inventory
                    + w_vel * (ce[5] - pe[5]))          # velocity differential
            mom = w_mom * float(spot[0])                # spot log-ret (×100)
            a[2 * i] = math.tanh(flow + mom)
        return a
