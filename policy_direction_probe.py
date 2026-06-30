#!/usr/bin/env python3
"""
policy_direction_probe.py — does the live policy have a directional (long-call)
bias?  Two days running, NIFTY conviction stayed POSITIVE (= CE / bullish) while
the index fell. Conviction sign is upstream of every entry filter, so before
touching any gate we test the policy's directional response directly.

This loads the SAME pair the brain loads (PolicyLoader) and drives the SAME
feature dialect (StateBuilder), then asks one question two independent ways:

  PROBE A — synthetic-market replay (end-to-end, faithful):
      feed the real StateBuilder a clean DOWN / UP / CHOP / FLAT tape, push it
      through to a real obs, read the policy's raw per-index conviction
      ai = actions[2*i]. If a clean down-tape yields ai > 0, the policy is
      structurally long.

  PROBE B — obs-space transfer function (controlled, pipeline-independent):
      start from the policy's OWN normalisation mean (vec.obs_rms.mean → the
      "average" training input, which normalises to ~0), then sweep a directional
      perturbation δ on the log_ret feature (feature 0) of the index's nodes
      across the whole 10-step sequence. Plot ai(δ). A healthy policy crosses
      zero (ai<0 for δ<0, ai>0 for δ>0); a biased one stays positive throughout.

Nothing here trades or writes — it is read-only inference. Needs torch / SB3 and
the promoted model pair, so run it on the machine, not in the sandbox:

    python policy_direction_probe.py
"""
from __future__ import annotations
import math
import numpy as np

import config
from core.market_state import StateBuilder, LEG_ORDER
from apex_main_v9 import PolicyLoader

F          = config.FEATURES_PER_NODE          # 19
NPI        = config.NODES_PER_INDEX            # 5
NODE_STRIDE = config.NUM_NODES * F             # 570  (per sequence step)
SEQ        = config.SEQ_LENGTH                 # 10
STEP_BP    = {"NIFTY": 50.0, "SENSEX": 100.0}  # strike spacing
BASE_SPOT  = {"NIFTY": 23900.0, "SENSEX": 76600.0}
DTE, IV    = 2.0, 0.15
T_YEARS    = DTE / 365.0


# ----------------------------------------------------------- synthetic market
def _scenario_path(s0: float, kind: str, n: int) -> list[float]:
    """A per-step spot path. 'down'/'up' net ~±0.5 % with chop; 'chop' nets ~0;
    'flat' barely moves. The chop is deliberately large so direction-agreement
    would mislabel it — the whole point."""
    out, s = [], s0
    drift = {"down": -1.0, "up": +1.0, "chop": 0.0, "flat": 0.0}[kind]
    for i in range(n):
        trend = drift * 0.0004 * s0                       # ~4 bp/step net
        chop = (s0 * 0.0006) * (1 if i % 2 else -1)        # ±6 bp oscillation
        if kind == "flat":
            chop *= 0.1
        s = s + trend + chop
        out.append(s)
    return out


def _leg_snap(ltp: float, tick: float) -> dict:
    """Neutral microstructure so only the directional log_ret moves."""
    return {"ltp": ltp, "bid": ltp - tick / 2, "ask": ltp + tick / 2,
            "bid_qty": 500.0, "ask_qty": 500.0, "vol_delta": 100.0, "oi": 10000.0}


def _market_at(spots: dict[str, float]) -> dict:
    """Build one market-dict snapshot from a {index: spot} map. ATM time value
    is a crude flat-vol proxy; only the intrinsic part (which moves with spot)
    needs to carry the directional signal."""
    m = {}
    for idx, S in spots.items():
        step = STEP_BP.get(idx, 50.0)
        atm = round(S / step) * step
        tv = 0.4 * S * IV * math.sqrt(T_YEARS)             # ~ATM time value
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


def _fit_flat_surface(builder: StateBuilder):
    for idx in config.INDEX_ORDER:
        S = BASE_SPOT.get(idx, 20000.0)
        step = STEP_BP.get(idx, 50.0)
        Fwd = S * math.exp(config.RISK_FREE_RATE * T_YEARS)
        ks = [S + k * step for k in range(-3, 4)]
        builder.fit_surface(idx, "PROBE", ks, [IV] * len(ks), Fwd, T_YEARS)


def probe_a(pol: PolicyLoader, n_steps: int = 30, settle: int = 12) -> dict:
    """Drive each scenario; return {scenario: {index: mean_ai}}."""
    out = {}
    for kind in ("down", "up", "chop", "flat"):
        builder = StateBuilder()
        _fit_flat_surface(builder)
        paths = {idx: _scenario_path(BASE_SPOT[idx], kind, n_steps)
                 for idx in config.TRADABLE}
        # non-tradable indices held flat at base so the policy sees a full frame
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
        out[kind] = {idx: (float(np.mean(v)) if v else float("nan"))
                     for idx, v in acc.items()}
    return out


def probe_b(pol: PolicyLoader, deltas=np.linspace(-5, 5, 11)) -> dict | None:
    """Sweep log_ret perturbation on each tradable index; return {index:[(δ,ai)]}.
    Returns None if the SB3 pair / its VecNormalize stats are unavailable."""
    if pol.model is None or pol.vec is None:
        return None
    base = np.asarray(pol.vec.obs_rms.mean, np.float32).copy()   # training mean
    dummy_frame = np.zeros((config.NUM_NODES, F), np.float32)
    res = {}
    for idx in config.TRADABLE:
        i = config.INDEX_ORDER.index(idx)
        nodes = [i * NPI + j for j in range(NPI)]                # spot + 4 legs
        # CE-side legs (spot, atm_ce, otm_ce) move WITH δ; PE-side legs opposite.
        sign = {nodes[0]: +1, nodes[1]: +1, nodes[2]: -1, nodes[3]: +1, nodes[4]: -1}
        curve = []
        for d in deltas:
            o = base.copy()
            for s in range(SEQ):
                for node, sg in sign.items():
                    o[s * NODE_STRIDE + node * F + 0] = sg * d   # feature 0 = log_ret
            a = pol.conviction(o.astype(np.float32), dummy_frame)
            curve.append((float(d), float(a[2 * i])))
        res[idx] = curve
    return res


def _verdict(a: dict, b: dict | None):
    print("\n" + "=" * 68)
    print(" VERDICT")
    print("=" * 68)
    flagged = []
    for idx in config.TRADABLE:
        down = a["down"][idx]
        up = a["up"][idx]
        tag = ""
        if not (math.isnan(down) or math.isnan(up)):
            if down > 0.05 and up > 0.05:
                tag = "→ LONG-BIASED (positive on BOTH up and down tapes)"
                flagged.append(idx)
            elif down >= up:
                tag = "→ SUSPECT (down-tape conviction ≥ up-tape)"
                flagged.append(idx)
            else:
                tag = "→ direction-tracking (down<up, as expected)"
        print(f"  {idx:7s}  Probe-A  down={down:+.3f}  up={up:+.3f}   {tag}")
        if b and idx in b:
            lo = b[idx][0][1]; hi = b[idx][-1][1]
            crosses = any(p[1] <= 0 for p in b[idx]) and any(p[1] >= 0 for p in b[idx])
            print(f"            Probe-B  ai(δ=-5)={lo:+.3f}  ai(δ=+5)={hi:+.3f}  "
                  f"crosses 0: {'yes' if crosses else 'NO → sign-locked'}")
    print("-" * 68)
    if flagged:
        print(f"  Directional bias indicated on: {', '.join(flagged)}.")
        print("  This is a TRAINING-side fix (the conviction sign itself is wrong),")
        print("  not a gate tweak: rebalance the directional labelling / widen the")
        print("  forge window to include down-trend regimes / penalise the long tilt.")
    else:
        print("  No clear long bias in the probe — conviction tracks direction.")
        print("  The flat/weak behaviour is then magnitude (meta P~0.50), not sign.")
    print("=" * 68)


def main():
    pol = PolicyLoader()
    print(f"policy kind: {pol.kind}  | OBS_DIM={config.OBS_DIM} "
          f"ACTION_DIM={config.ACTION_DIM} | tradable={config.TRADABLE}")
    if pol.model is None:
        print("\n⚠ SB3 pair did NOT load (heuristic fallback live). Probe B needs the\n"
              "  trained pair; run where models/current_manifest.json resolves. Probe A\n"
              "  will reflect the HEURISTIC, not the RL policy.")
    a = probe_a(pol)
    print("\nPROBE A — mean raw conviction ai=actions[2*i] by scenario")
    print(f"  {'index':7s} | {'down':>8s} {'up':>8s} {'chop':>8s} {'flat':>8s}")
    for idx in config.TRADABLE:
        print(f"  {idx:7s} | " + " ".join(
            f"{a[k][idx]:+8.3f}" for k in ('down', 'up', 'chop', 'flat')))
    b = probe_b(pol)
    if b:
        print("\nPROBE B — directional transfer function ai(δ on log_ret)")
        for idx in config.TRADABLE:
            print(f"  {idx}:")
            for d, ai in b[idx]:
                bar = "#" * int(abs(ai) * 20)
                print(f"     δ={d:+5.1f}  ai={ai:+.3f}  {'+' if ai>=0 else '-'}{bar}")
    _verdict(a, b)


if __name__ == "__main__":
    main()