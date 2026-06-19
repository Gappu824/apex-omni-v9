"""
APEX OMNI v9 — GRAPH / TGN (audit §5 leaps: DTE-aware features arrive via
market_state; this module adds dead-node masking and per-node embeddings so
zero-padded monthly-only indices stop eating model capacity as fake signal)
Torch is imported lazily so the harvester/simulator never need it.
"""
from __future__ import annotations
import numpy as np
import config

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:                                   # noqa: BLE001
    HAVE_TORCH = False


def build_edge_index() -> np.ndarray:
    """Static intra-index star + ATM↔OTM links + spot↔spot ring across
    indices (lets cross-index info flow both ways — v8 was one-directional)."""
    edges = []
    n = config.NODES_PER_INDEX
    for i in range(len(config.INDEX_ORDER)):
        b = i * n                       # spot, atm_ce, atm_pe, otm_ce, otm_pe
        for j in range(1, n):
            edges += [(b, b + j), (b + j, b)]
        edges += [(b + 1, b + 3), (b + 3, b + 1),       # ce: atm↔otm
                  (b + 2, b + 4), (b + 4, b + 2),       # pe: atm↔otm
                  (b + 1, b + 2), (b + 2, b + 1)]       # atm ce↔pe
    spots = [i * n for i in range(len(config.INDEX_ORDER))]
    for a in spots:
        for c in spots:
            if a != c:
                edges.append((a, c))
    return np.array(edges, dtype=np.int64).T            # (2, E)


def adjacency_dense() -> np.ndarray:
    """Row-normalized dense adjacency with self-loops (v8's good 8 GB-VRAM
    instinct, kept)."""
    A = np.eye(config.NUM_NODES, dtype=np.float32)
    ei = build_edge_index()
    A[ei[0], ei[1]] = 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A


if HAVE_TORCH:

    class DenseGCNLayer(nn.Module):
        def __init__(self, fin, fout):
            super().__init__()
            self.lin = nn.Linear(fin, fout)
            self.register_buffer("A", torch.tensor(adjacency_dense()))

        def forward(self, x):                       # x: (B, N, F)
            return torch.relu(self.lin(self.A @ x))

    class TGNFeatureExtractor(nn.Module):
        """obs (B, 5700) → (B, 512). Node mask zeroes the message-passing of
        legs that were empty in the LAST frame (illiquid monthlies, audit §5)."""
        def __init__(self, obs_dim=config.OBS_DIM, out_dim=config.PROJ_DIM):
            super().__init__()
            F, N, S = (config.FEATURES_PER_NODE, config.NUM_NODES,
                       config.SEQ_LENGTH)
            self.F, self.N, self.S = F, N, S
            self.gcn1 = DenseGCNLayer(F, config.GCN_HIDDEN)
            self.gcn2 = DenseGCNLayer(config.GCN_HIDDEN, config.GCN_HIDDEN)
            self.node_emb = nn.Parameter(torch.randn(N, 8) * 0.02)
            self.proj = nn.Linear(N * (config.GCN_HIDDEN + 8), out_dim)
            enc = nn.TransformerEncoderLayer(out_dim, nhead=4,
                                             dim_feedforward=out_dim * 2,
                                             batch_first=True)
            self.tx = nn.TransformerEncoder(enc, num_layers=2)
            self.pos = nn.Parameter(torch.randn(1, S, out_dim) * 0.02)

        def forward(self, obs):                     # (B, S*N*F)
            B = obs.shape[0]
            x = obs.view(B, self.S, self.N, self.F)
            mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-9).float()
            x = x.view(B * self.S, self.N, self.F)
            h = self.gcn2(self.gcn1(x))
            h = h.view(B, self.S, self.N, -1) * mask
            emb = self.node_emb.expand(B, self.S, -1, -1)
            h = torch.cat([h, emb * mask], dim=-1).flatten(2)
            z = self.proj(h) + self.pos
            return self.tx(z)[:, -1]                # last timestep → (B, 512)
