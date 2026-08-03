"""
META-GATE v3 REPLAY — the ₹-denominated exam for retiring the fixed bar
=======================================================================
Question this tool answers, on YOUR vault, with the forge's OWN samples:

    "If yesterday's decisions had run through the v3 EV gate instead of
     the legacy 0.55 bar, what changes — in rupees?"

Method (no re-derivation, no look-ahead):
  1. Samples come from `_gen_meta_samples_cached` — the forge's label
     pipeline: brain-identical signals, shaped-barrier first-touch
     labels, realized net P&L (RET), and (v9.9) each sample's OWN
     payoff geometry (entry, tp, sl, lot).
  2. One honest model fit via `MG.fit_gbm(..., oof_out=..., model_path=
     <tmp>)` — the audit-approved research redirect; the production
     booster is never touched. Every probability used below is OUT-OF-
     FOLD under the purged day split.
  3. LEGACY gate per sample: isotonic-calibrated OOF p (calibrator fit
     on the OTHER folds' OOF — no self-calibration), blended and
     clamped exactly as `core.decision` serves it, vs META_ENTRY_P_BAR.
  4. V3 gate per sample: cross Venn-Abers interval (calibration set =
     OTHER folds' OOF) vs the sample's OWN breakeven p* from its
     (entry, tp, sl, lot) through `round_trip_costs`. ACI margin = 0
     (live-serving adaptation; the exam measures the model+EV core).
  5. Score both gates against RET. PROBE-zone P&L is reported both raw
     and at probe weight (1 lot ≈ the raw sample; probes ARE 1 lot).

Output: per-zone counts and ₹, the two headline numbers —
     UNBLOCKED ₹  (legacy vetoed, v3 takes — the trades you're losing)
     SAVED ₹      (legacy took, v3 vetoes — the losers it still stops)
— and a JSON certificate for the trial registry.

`--synthetic` runs the identical machinery on generated data (CI smoke
for machines without the vault).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("meta_gate_replay")
import logging                                             # noqa: E402
log = logging.getLogger("meta_gate_replay")

from core import meta_gbm as MG                            # noqa: E402
from core.meta_gate import (VennAbers, breakeven_p)        # noqa: E402


# --------------------------------------------------------------------------
def _cross_fold_scores(days, oof_raw, y, w, sample_day):
    """For every sample: legacy calibrated p AND the VA interval, both
    computed with calibrators fit ONLY on other purged folds' OOF."""
    n = len(oof_raw)
    p_leg = np.full(n, np.nan)
    p0 = np.full(n, np.nan)
    p1 = np.full(n, np.nan)
    from core.meta_gbm import _isotonic
    for test_days, train_days in MG._purged_day_folds(
            list(days), config.META_EMBARGO_DAYS):
        tr = np.isin(sample_day, train_days)
        te = np.isin(sample_day, test_days)
        if tr.sum() < 50 or te.sum() < 1:
            continue
        ix, iy = _isotonic(oof_raw[tr], y[tr], w[tr])
        p_leg[te] = np.interp(oof_raw[te], ix, iy)
        va = VennAbers(oof_raw[tr], y[tr], w[tr])
        for j in np.nonzero(te)[0]:
            p0[j], p1[j] = va.interval(float(oof_raw[j]))
    return p_leg, p0, p1


def _legacy_gate(p_cal: float) -> bool:
    """Exactly `core.decision`: clamp to [floor,cap]; no calibration-table
    hit in this offline context ⇒ wp == clamped meta p; pass iff ≥ bar."""
    wp = min(max(p_cal, config.META_P_FLOOR), config.META_P_CAP)
    return wp >= config.META_ENTRY_P_BAR


def _run(days, perday_x, y, w, ret, econ, sample_day, label):
    n = len(y)
    with tempfile.TemporaryDirectory() as td:
        oof: dict = {}
        art = MG.fit_gbm(perday_x, min_train=min(config.META_MIN_TRAIN, n),
                         oof_out=oof,
                         model_path=Path(td) / "replay_gbm.txt")
    if art is None or "mask" not in oof:
        # v9.9.3: this is a VERDICT, not a malfunction. On 2026-08-02 the
        # guard refused (AUC < floor) exactly as designed, this tool logged
        # it at ERROR and exited 1 — and the evening summary stamped the
        # only <<< FAILED of the night on the one step that behaved
        # perfectly. A refusal certificate is a successful examination
        # with a negative result: exit 0, write it, register it.
        log.warning("VERDICT (%s): fit_gbm refused — no ranking signal in "
                    "this window. Neither gate has a model to serve; the "
                    "brain correctly runs the conviction bar. This is the "
                    "answer, not an error.", label)
        out = {"label": label, "refused": True,
               "reason": "guard refused fit (AUC/positives below floor)",
               "ts": time.time(), "config_hash": config.CONFIG_HASH}
        try:
            from core.trial_registry import register
            register(family="meta_gate_v3", spec_id=f"replay:{label}",
                     kind="replay_refused")
        except Exception as e:                             # noqa: BLE001
            log.debug("trial registry unavailable (%s)", e)
        return out
    m = oof["mask"]
    s_raw = oof["oof_raw"]
    yy = oof["y"]
    ww = np.asarray(w, np.float32)[m]
    rr = np.asarray(ret, np.float32)[m]
    ee = [econ[i] for i in np.nonzero(m)[0]]
    dd = np.asarray(sample_day)[m]
    p_leg, p0, p1 = _cross_fold_scores(sorted(set(dd)), s_raw, yy, ww, dd)
    ok = ~np.isnan(p_leg) & ~np.isnan(p0)
    zones = {"FULL": [], "PROBE": [], "VETO": []}
    legacy_pass = []
    ev_m = float(getattr(config, "META_EV_MARGIN", 0.02))
    for j in np.nonzero(ok)[0]:
        e_, tp_, sl_, lot_ = ee[j]
        ps = breakeven_p(float(e_), float(tp_), float(sl_), int(lot_))
        bar = ps + ev_m
        if p0[j] >= bar:
            z = "FULL"
        elif p1[j] < bar:
            z = "VETO"
        else:
            z = "PROBE"
        zones[z].append(j)
        if _legacy_gate(float(p_leg[j])):
            legacy_pass.append(j)
    legacy_pass = set(legacy_pass)
    out = {"label": label, "n_scored": int(ok.sum()),
           "days": len(set(dd)), "ts": time.time(),
           "config_hash": config.CONFIG_HASH,
           "bar_legacy": config.META_ENTRY_P_BAR,
           "ev_margin": ev_m}
    tot_pass_v3 = 0
    print(f"\n══ {label}: {int(ok.sum())} OOF-scored samples over "
          f"{len(set(dd))} day(s) ══")
    for z, idxs in zones.items():
        r = rr[idxs]
        wins = int((r > 0).sum())
        out[z] = {"n": len(idxs), "wins": wins,
                  "pnl": round(float(r.sum()), 2),
                  "pnl_mean": round(float(r.mean()), 2) if len(idxs) else 0.0}
        if z != "VETO":
            tot_pass_v3 += len(idxs)
        print(f"  v3 {z:6s}: {len(idxs):5d} trades | {wins:4d} wins | "
              f"₹{r.sum():+12,.0f} | mean ₹{(r.mean() if len(idxs) else 0):+9,.2f}")
    lp = sorted(legacy_pass)
    r_leg = rr[lp]
    print(f"  legacy PASS: {len(lp):5d} trades | {(r_leg > 0).sum():4d} wins"
          f" | ₹{r_leg.sum():+12,.0f}   (bar {config.META_ENTRY_P_BAR:.2f})")
    out["legacy_pass"] = {"n": len(lp), "pnl": round(float(r_leg.sum()), 2)}
    # the two headline numbers
    v3_take = set(zones["FULL"]) | set(zones["PROBE"])
    unblocked = sorted(v3_take - legacy_pass)
    saved = sorted(legacy_pass - v3_take)
    ru, rs = rr[unblocked], rr[saved]
    print(f"\n  UNBLOCKED (legacy vetoed → v3 takes): {len(unblocked):5d} "
          f"| ₹{ru.sum():+12,.0f}  ← the trades the old bar was costing")
    print(f"  SAVED     (legacy took → v3 vetoes) : {len(saved):5d} "
          f"| ₹{-rs.sum():+12,.0f}  ← losers the EV veto still stops")
    out["unblocked"] = {"n": len(unblocked), "pnl": round(float(ru.sum()), 2)}
    out["saved"] = {"n": len(saved), "pnl_avoided": round(float(-rs.sum()), 2)}
    net = float(ru.sum() - rs.sum())
    out["net_improvement"] = round(net, 2)
    print(f"  NET Δ vs legacy gate: ₹{net:+,.0f}")
    if len(unblocked) + len(saved) < 30:
        print("  ⚠ small-sample verdict — treat as direction, not size; "
              "re-run as the vault deepens.")
        out["small_sample"] = True
    # trial registry — selection bias stays arithmetic, not honor
    try:
        from core.trial_registry import register
        register(family="meta_gate_v3", spec_id=f"replay:{label}",
                 kind="replay_exam",
                 n_scored=out["n_scored"],
                 net_improvement=out["net_improvement"],
                 unblocked_pnl=out["unblocked"]["pnl"],
                 saved_pnl=out["saved"]["pnl_avoided"])
        print("  trial registered.")
    except Exception as e:                                 # noqa: BLE001
        log.debug("trial registry unavailable (%s)", e)
    return out


# --------------------------------------------------------------------------
def _synthetic(n_days=8, per_day=120, seed=11):
    """CI smoke: planted edge with heterogeneous payoff geometry, so the
    EV gate has something real to find. entry≈100; b ranges 0.6–2.5."""
    rng = np.random.default_rng(seed)
    days = [f"2026-07-{k:02d}" for k in range(1, n_days + 1)]
    perday_x, Y, W, RET, ECON, SD = [], [], [], [], [], []
    for d in days:
        xs, ys, ws = [], [], []
        for _ in range(per_day):
            x = rng.normal(size=16).astype(np.float32)
            drive = float(x[0] + 0.6 * x[1])
            p_true = 1 / (1 + np.exp(-1.3 * drive))
            e = 100.0
            b = float(np.clip(rng.lognormal(0.1, 0.45), 0.6, 2.5))
            sl = e * (1 - config.BASE_SL_PCT)
            tp = e + b * (e - sl)
            lot = 75
            won = rng.random() < p_true
            from core.execution_engine import round_trip_costs
            pnl = ((tp - e) * lot - round_trip_costs(e * lot, tp * lot)
                   if won else
                   -(e - sl) * lot - round_trip_costs(e * lot, sl * lot))
            xs.append(x); ys.append(1.0 if won else 0.0); ws.append(1.0)
            Y.append(1.0 if won else 0.0); W.append(1.0)
            RET.append(float(pnl)); ECON.append((e, tp, sl, lot))
            SD.append(d)
        perday_x.append((d, xs, ys, ws))
    return days, perday_x, Y, W, RET, ECON, SD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--synthetic", action="store_true",
                    help="planted-edge smoke run (no vault needed)")
    a = ap.parse_args()
    if a.synthetic:
        out = _run(*_synthetic(), label="SYNTHETIC")
    else:
        from nightly_forge_v9 import (_gen_meta_samples_cached,
                                      trading_days)
        con = sqlite3.connect(config.DB_PATH)
        days = trading_days(con)[-a.days:]
        if getattr(config, "META_TRAIN_MAX_DAYS", 0) > 0:
            days = days[-config.META_TRAIN_MAX_DAYS:]
        perday_x, Y, W, RET, ECON, SD = [], [], [], [], [], []
        for d in days:
            try:
                X, y_, w_, _r, ret_, e_ = _gen_meta_samples_cached(con, d)
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: sample generation failed (%s)", d, e)
                continue
            if not X:
                continue
            if len(e_) != len(X):
                log.error("  %s: cache predates the v9.9 economics column "
                          "— it will rebuild on the next forge run; "
                          "skipping today", d)
                continue
            perday_x.append((d, X, y_, w_))
            Y += list(y_); W += list(w_); RET += list(ret_)
            ECON += [tuple(t) for t in e_]; SD += [d] * len(X)
        if not perday_x:
            log.error("no usable days — run the forge once to build caches")
            return 1
        out = _run(days, perday_x, Y, W, RET, ECON, SD, label="VAULT")
    if out and a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"  → {a.json}")
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())