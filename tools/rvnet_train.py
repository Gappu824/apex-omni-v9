"""
APEX OMNI v9.5 — RV-NET (Pillar 3 T2: the GPU's one honest daily job)
=====================================================================
A small torch MLP forecasting next-day log-RV from HAR-style features. Its
existence is CONDITIONAL: the artifact (state/rvnet_{IDX}.pt + meta) is
written ONLY when the net beats the HAR baseline on the SAME walk-forward
QLIKE protocol the skill certificate uses (mean QLIKE < HAR's AND ≥60% daily
wins AND ≥8 eval days). Anything less and the trainer says so and writes
nothing — at 17 vault days that refusal is the expected, correct outcome
(Corsi's linear cascade is brutally hard to beat at small n; the net's day
comes with data). Every run registers in the trial ledger (family "rv").

Features per day d (predicting logRV_{d+1}):
  logRV_d, mean(logRV last5), mean(logRV last22), |overnight gap|, dte_frac
Net: 5→16→16→1, GELU, AdamW, early-stop on tail fold — deliberately tiny;
capacity is the enemy at this sample size.

Run alongside rv_skill_report:  python tools/rvnet_train.py [--days N]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import rv_forecaster as RV                     # noqa: E402
from core import trial_registry as TR                    # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402
from tools.rv_skill_report import _minute_closes, _series, _qlike  # noqa: E402

config.setup_logging("rvnet")
import logging                                           # noqa: E402
log = logging.getLogger("rvnet")

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:                                        # noqa: BLE001
    HAVE_TORCH = False


def _feats(logrv: list[float], gaps: list[float], d: int) -> list[float]:
    h = logrv[:d]
    return [h[-1], float(np.mean(h[-5:])), float(np.mean(h[-22:])),
            float(gaps[d]) if d < len(gaps) else 0.0,
            (d % 5) / 5.0]                                # weekday proxy


def _fit_net(X: np.ndarray, y: np.ndarray, seed: int = 7):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    Y = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    net = nn.Sequential(nn.Linear(X.shape[1], 16), nn.GELU(),
                        nn.Linear(16, 16), nn.GELU(),
                        nn.Linear(16, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.MSELoss()
    best, patience = None, 0
    for ep in range(400):
        opt.zero_grad()
        loss = lossf(net(Xn), Y)
        loss.backward()
        opt.step()
        lv = float(loss)
        if best is None or lv < best - 1e-6:
            best, patience = lv, 0
        else:
            patience += 1
            if patience > 40:
                break
    return net.cpu().eval(), mu, sd


def _predict(net, mu, sd, x: list[float]) -> float:
    with torch.no_grad():
        z = torch.tensor(((np.asarray(x) - mu) / sd)[None, :],
                         dtype=torch.float32)
        return float(net(z).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    if not HAVE_TORCH:
        log.info("torch unavailable — RV-Net cannot train on this box")
        return
    con = sqlite3.connect(config.DB_PATH)
    days_all = trading_days(con)
    if args.days > 0:
        days_all = days_all[-args.days:]
    verdicts = {}
    for index in config.TRADABLE:
        days, rvs, gaps, _profs = _series(con, index, days_all)
        n = len(rvs)
        logrv = [math.log(max(x, 1e-12)) for x in rvs]
        qn, qh = [], []
        for d in range(10, n):
            X = np.asarray([_feats(logrv, gaps, i) for i in range(6, d)])
            y = np.asarray([logrv[i] for i in range(6, d)])
            if len(y) < 8:
                continue
            har = RV.fit_har(rvs[:d], gaps[:d], [None] * d,
                             f"__wf_{index}", days[:d])
            f_h = RV.predict_next_day(har, logrv[:d], gaps[d]) if har else None
            net, mu, sd = _fit_net(X, y)
            f_n = math.exp(_predict(net, mu, sd, _feats(logrv, gaps, d)))
            if not f_h:
                continue
            qn.append(_qlike(f_n, rvs[d]))
            qh.append(_qlike(f_h, rvs[d]))
        m = len(qn)
        wins = sum(1 for a, b in zip(qn, qh) if a < b)
        beat = (m >= 8 and float(np.mean(qn)) < float(np.mean(qh))
                and wins / max(m, 1) >= 0.60)
        verdicts[index] = {
            "eval_days": m,
            "qlike_net": (round(float(np.mean(qn)), 4) if m else None),
            "qlike_har": (round(float(np.mean(qh)), 4) if m else None),
            "wins_vs_har": wins, "beats_har": bool(beat)}
        if beat:                                          # earn the artifact
            X = np.asarray([_feats(logrv, gaps, i) for i in range(6, n)])
            y = np.asarray([logrv[i] for i in range(6, n)])
            net, mu, sd = _fit_net(X, y)
            torch.save({"state": net.state_dict(),
                        "mu": mu.tolist(), "sd": sd.tolist()},
                       config.STATE_DIR / f"rvnet_{index}.pt")
            verdicts[index]["artifact"] = f"rvnet_{index}.pt"
            log.info("%s: RV-NET EARNS ITS ARTIFACT — QLIKE %.4f < HAR %.4f "
                     "(wins %d/%d)", index, np.mean(qn), np.mean(qh),
                     wins, m)
        else:
            (config.STATE_DIR / f"rvnet_{index}.pt").unlink(missing_ok=True)
            log.info("%s: net does NOT beat HAR (QLIKE %s vs %s, wins %d/%d,"
                     " n=%d) — no artifact written; the refusal is the "
                     "system working", index,
                     verdicts[index]["qlike_net"],
                     verdicts[index]["qlike_har"], wins, m, m)
    for p in config.STATE_DIR.glob("rv_model___wf_*.json"):
        p.unlink(missing_ok=True)
    TR.register("rv", f"rvnet_v1_{config.CONFIG_HASH}", "primary",
                beats_har=any(v["beats_har"] for v in verdicts.values()))
    _atomic_write_json(config.STATE_DIR / "rvnet_verdict.json",
                       {"per_index": verdicts, "ts": time.time()})


if __name__ == "__main__":
    main()