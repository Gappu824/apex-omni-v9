#!/usr/bin/env python3
"""
policy_direction_probe.py — does the live policy track DIRECTION, and if not,
what does it key off instead?

History of this probe:
  • Probe A (synthetic tape, order-flow held NEUTRAL) → conviction near-constant
    positive. First read "inert" — but A only varies PRICE, and the policy barely
    reads price, so A under-tests it.
  • Probe C (single-feature ±σ sweep) → big leverage on 'velocity', but velocity =
    tanh(vol_delta/ewma − 1) is volume INTENSITY, not a signed direction; and the
    genuinely-signed features (ofi_z, dealer_inv) only showed their common-mode
    (uniform) response. So C didn't isolate direction either.

This version separates the two cleanly:
  PROBE C — per feature, reports ANTISYMMETRIC leverage (spot+CE +, PE − : the
      physically-directional config — the real "reads direction" axis for a SIGNED
      feature) SEPARATELY from UNIFORM leverage (all legs together : a common-mode
      / activity axis). Each feature is tagged dir / amb / act so the numbers are
      read correctly:
        dir = signed directional (log_ret, ofi_z, dealer_inv) → ANTISYM is the test
        amb = signed but direction-ambiguous (oi_delta_norm: position building)
        act = activity/intensity (velocity) → only UNIFORM is meaningful, NOT direction
  PROBE D — the decisive test: perturb the SIGNED directional features TOGETHER
      (log_ret + ofi_z + dealer_inv) as one coherent state, from full-bearish
      (k=−3: price down, sell-side flow, dealers short) to full-bullish (k=+3),
      and read conviction. A policy that reads direction tracks this monotonically
      through zero. A policy flat or wrong-signed here does NOT read direction,
      whatever single features do in isolation.

Read-only inference. Needs torch / SB3 + the promoted pair; run on the machine:
    python policy_direction_probe.py
"""
from __future__ import annotations
import math
import numpy as np

import config
from core.market_state import StateBuilder, LEG_ORDER
from apex_main_v9 import PolicyLoader

F           = config.FEATURES_PER_NODE
NPI         = config.NODES_PER_INDEX
NODE_STRIDE = config.NUM_NODES * F
SEQ         = config.SEQ_LENGTH
STEP_BP     = {"NIFTY": 50.0, "SENSEX": 100.0}
BASE_SPOT   = {"NIFTY": 23900.0, "SENSEX": 76600.0}
DTE, IV     = 2.0, 0.15
T_YEARS     = DTE / 365.0

# feature index → (name, type). type drives how leverage is interpreted.
FEATURES = {0: ("log_ret", "dir"), 2: ("oi_delta_norm", "amb"),
            5: ("velocity", "act"), 12: ("ofi_z", "dir"), 16: ("dealer_inv", "dir")}
DIR_FEATURES = [f for f, (_, t) in FEATURES.items() if t == "dir"]   # 0,12,16


# ----------------------------------------------------------- synthetic market
def _scenario_path(s0, kind, n):
    out, s = [], s0
    drift = {"down": -1.0, "up": +1.0, "chop": 0.0, "flat": 0.0}[kind]
    for i in range(n):
        trend = drift * 0.0004 * s0
        chop = (s0 * 0.0006) * (1 if i % 2 else -1)
        if kind == "flat":
            chop *= 0.1
        s = s + trend + chop
        out.append(s)
    return out


def _leg_snap(ltp, tick):
    return {"ltp": ltp, "bid": ltp - tick / 2, "ask": ltp + tick / 2,
            "bid_qty": 500.0, "ask_qty": 500.0, "vol_delta": 100.0, "oi": 10000.0}


def _market_at(spots):
    m = {}
    for idx, S in spots.items():
        step = STEP_BP.get(idx, 50.0)
        atm = round(S / step) * step
        tv = 0.4 * S * IV * math.sqrt(T_YEARS)
        tick = 0.05
        legs = {
            "atm_ce": {"strike": atm,        "snap": _leg_snap(tv + max(S - atm, 0.0), tick)},
            "atm_pe": {"strike": atm,        "snap": _leg_snap(tv + max(atm - S, 0.0), tick)},
            "otm_ce": {"strike": atm + step, "snap": _leg_snap(tv + max(S - (atm + step), 0.0), tick)},
            "otm_pe": {"strike": atm - step, "snap": _leg_snap(tv + max((atm - step) - S, 0.0), tick)},
        }
        m[idx] = {"spot": {"ltp": S}, "expiry": "PROBE", "dte": DTE,
                  "is_weekly": True, "T": T_YEARS, "legs": legs}
    return m


def _fit_flat_surface(builder):
    for idx in config.INDEX_ORDER:
        S = BASE_SPOT.get(idx, 20000.0)
        step = STEP_BP.get(idx, 50.0)
        Fwd = S * math.exp(config.RISK_FREE_RATE * T_YEARS)
        ks = [S + k * step for k in range(-3, 4)]
        builder.fit_surface(idx, "PROBE", ks, [IV] * len(ks), Fwd, T_YEARS)


def probe_a(pol, n_steps=30, settle=12):
    out = {}
    for kind in ("down", "up", "chop", "flat"):
        builder = StateBuilder()
        _fit_flat_surface(builder)
        paths = {idx: _scenario_path(BASE_SPOT[idx], kind, n_steps) for idx in config.TRADABLE}
        flat_other = {idx: BASE_SPOT.get(idx, 20000.0)
                      for idx in config.INDEX_ORDER if idx not in config.TRADABLE}
        acc = {idx: [] for idx in config.TRADABLE}
        for t in range(n_steps):
            spots = {idx: paths[idx][t] for idx in config.TRADABLE}
            spots.update(flat_other)
            obs = builder.push(_market_at(spots), ts=float(t))
            if obs is None:
                continue
            actions = pol.conviction(obs, builder.frames[-1])
            if t >= settle:
                for idx in config.TRADABLE:
                    acc[idx].append(float(actions[2 * config.INDEX_ORDER.index(idx)]))
        out[kind] = {idx: (float(np.mean(v)) if v else float("nan")) for idx, v in acc.items()}
    return out


def _node_signs(i):
    b = i * NPI
    return {b: +1, b + 1: +1, b + 2: -1, b + 3: +1, b + 4: -1}   # spot,CE,PE,CE,PE


def _stats(pol):
    mean = np.asarray(pol.vec.obs_rms.mean, np.float32)
    std = np.sqrt(np.asarray(pol.vec.obs_rms.var, np.float32))
    return mean, std, np.zeros((config.NUM_NODES, F), np.float32)


def probe_c(pol, ks=np.linspace(-3, 3, 7)):
    """Per-feature antisym (directional) and uniform (common-mode) leverage."""
    if pol.model is None or pol.vec is None:
        return None
    mean, std, dummy = _stats(pol)
    out = {}
    for idx in config.TRADABLE:
        i = config.INDEX_ORDER.index(idx)
        signs = _node_signs(i)
        per = {}
        for f, (name, typ) in FEATURES.items():
            def sweep(uniform):
                a0 = a1 = None
                for k in (ks[0], ks[-1]):
                    o = mean.copy()
                    for s in range(SEQ):
                        for node, sg in signs.items():
                            p = s * NODE_STRIDE + node * F + f
                            o[p] = mean[p] + (1 if uniform else sg) * k * std[p]
                    ai = float(pol.conviction(o.astype(np.float32), dummy)[2 * i])
                    if a0 is None:
                        a0 = ai
                    else:
                        a1 = ai
                return a1 - a0
            per[name] = {"type": typ, "antisym": sweep(False), "uniform": sweep(True)}
        out[idx] = per
    return out


def probe_d(pol, ks=np.linspace(-3, 3, 9)):
    """Coherent directional bundle: log_ret+ofi_z+dealer_inv moved together,
    bearish(k<0) → bullish(k>0). Returns {index: [(k, ai)]}."""
    if pol.model is None or pol.vec is None:
        return None
    mean, std, dummy = _stats(pol)
    out = {}
    for idx in config.TRADABLE:
        i = config.INDEX_ORDER.index(idx)
        signs = _node_signs(i)
        curve = []
        for k in ks:
            o = mean.copy()
            for s in range(SEQ):
                for node, sg in signs.items():
                    for f in DIR_FEATURES:
                        p = s * NODE_STRIDE + node * F + f
                        o[p] = mean[p] + sg * k * std[p]
            curve.append((float(k), float(pol.conviction(o.astype(np.float32), dummy)[2 * i])))
        out[idx] = curve
    return out


def _tier(x):
    a = abs(x)
    return "INERT" if a < 0.05 else ("weak" if a < 0.20 else "RESPONSIVE")


def _verdict(a, c, d):
    print("\n" + "=" * 72)
    print(" VERDICT")
    print("=" * 72)
    for idx in config.TRADABLE:
        print(f"  {idx}:")
        # directional response = antisym leverage on the SIGNED dir features
        dir_lev = {n: v["antisym"] for n, v in c[idx].items() if v["type"] == "dir"} if c else {}
        mx = max(dir_lev.values(), key=abs) if dir_lev else 0.0
        if d and idx in d:
            curve = d[idx]
            lev = curve[-1][1] - curve[0][1]                      # bearish→bullish
            crosses = any(p[1] <= 0 for p in curve) and any(p[1] >= 0 for p in curve)
            mono = all(curve[j][1] <= curve[j + 1][1] + 1e-6 for j in range(len(curve) - 1))
            print(f"    Probe-D bundle leverage (bear→bull) = {lev:+.3f}  "
                  f"crosses 0: {'yes' if crosses else 'NO'}  monotonic↑: {'yes' if mono else 'no'}")
        else:
            lev, crosses = 0.0, False
        print(f"    directional (antisym) leverage on signed features: "
              + ", ".join(f"{n}={v:+.3f}" for n, v in dir_lev.items()))
        if lev >= 0.20 and crosses:
            print("    → READS DIRECTION: conviction tracks a coherent directional state.")
            print("      The live positive-while-falling is then NOT an obs gap but either")
            print("      (a) a positive BASELINE offset at neutral flow, and/or (b) live")
            print("      order-flow that is itself contrarian (retail buying calls into")
            print("      weakness — cf. the macro log's 'retail heavy options BUYER').")
            print("      FIX: de-mean the directional label / penalise the long baseline;")
            print("      a reward change, not an observation change.")
        elif abs(mx) >= 0.20 and not (lev >= 0.20 and crosses):
            print("    → MIXED: responds to individual signed features but does NOT track")
            print("      the coherent bundle (sign-inconsistent across features). The")
            print("      directional features fight each other. FIX: the LABEL is the lever")
            print("      — retrain on a clean signed forward-return target so the features")
            print("      align to one direction.")
        else:
            print("    → DOES NOT READ DIRECTION (any responsiveness is to velocity =")
            print("      ACTIVITY, not direction). The signed directional features have")
            print("      little pull. FIX: get directional signal into the OBSERVATION —")
            print("      fold macro/GEX direction (flip, PCR, max-pain, levels) into obs.")
    print("=" * 72)


def main():
    pol = PolicyLoader()
    print(f"policy kind: {pol.kind}  | OBS_DIM={config.OBS_DIM} "
          f"ACTION_DIM={config.ACTION_DIM} | tradable={config.TRADABLE}")
    if pol.model is None:
        print("\n⚠ SB3 pair did NOT load (heuristic fallback). Probes C/D need the pair.")
    a = probe_a(pol)
    print("\nPROBE A — mean raw conviction by scenario (order-flow held NEUTRAL)")
    print(f"  {'index':7s} | {'down':>8s} {'up':>8s} {'chop':>8s} {'flat':>8s}")
    for idx in config.TRADABLE:
        print(f"  {idx:7s} | " + " ".join(f"{a[k][idx]:+8.3f}" for k in ('down', 'up', 'chop', 'flat')))
    c = probe_c(pol)
    if c:
        print("\nPROBE C — leverage per feature: ANTISYM (directional) vs UNIFORM (common-mode)")
        for idx in config.TRADABLE:
            print(f"  {idx}:  {'feature':14s} {'type':4s} {'antisym':>8s} {'uniform':>8s}   read")
            for f, (name, typ) in FEATURES.items():
                v = c[idx][name]
                axis = v["antisym"] if typ == "dir" else (v["uniform"] if typ == "act" else v["antisym"])
                print(f"     {'':5s}{name:14s} {typ:4s} {v['antisym']:+8.3f} {v['uniform']:+8.3f}   "
                      f"{_tier(axis)}{' (activity, not direction)' if typ=='act' else ''}")
    d = probe_d(pol)
    if d:
        print("\nPROBE D — coherent directional bundle  ai(k): bear(−3) → bull(+3)")
        for idx in config.TRADABLE:
            print(f"  {idx}:")
            for k, ai in d[idx]:
                bar = "#" * int(abs(ai) * 20)
                print(f"     k={k:+5.2f}  ai={ai:+.3f}  {'+' if ai>=0 else '-'}{bar}")
    _verdict(a, c, d)


if __name__ == "__main__":
    main()