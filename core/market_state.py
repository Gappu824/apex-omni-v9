"""
APEX OMNI v9 — SHARED MARKET STATE (audit §8 leap: ONE build_market_state)
==========================================================================
v8 computed regime_vol, velocity, depth_gradient, vpin and dealer inventory
with DIFFERENT formulas in the forge vs the live brain (and never set
net_dealer_inventory live at all). v9 has exactly one StateBuilder. The
harvester, the nightly forge, the live brain, the scanner and the simulator
all instantiate THIS class and feed it ticks; identical input stream ⇒
byte-identical features. The skew is structurally impossible now.

19 features per node (audit §5 leap: DTE + weekly flag added):
  0 log_ret      1 moneyness     2 oi_delta_norm  3 depth_grad   4 vpin
  5 velocity     6 spread_pct    7 iv             8 skew         9 regime_vol
 10 hawkes      11 iceberg      12 ofi_z         13 delta       14 gamma_x1e4
 15 theta_norm  16 dealer_inv   17 dte_norm      18 is_weekly
Node order per index: [spot, atm_ce, atm_pe, otm_ce, otm_pe].
"""
from __future__ import annotations
import math
from collections import deque, defaultdict
import numpy as np

import config
from core.quant_core import (EWMAVol, RealVPIN, HawkesExcitation,
                             depth_gradient, SVISurface, black76_greeks)

FEATURE_NAMES = ["log_ret", "moneyness", "oi_delta_norm", "depth_grad", "vpin",
                 "velocity", "spread_pct", "iv", "skew", "regime_vol",
                 "hawkes", "iceberg", "ofi_z", "delta", "gamma_x1e4",
                 "theta_norm", "dealer_inv", "dte_norm", "is_weekly"]
assert len(FEATURE_NAMES) == config.FEATURES_PER_NODE

LEG_ORDER = ["spot", "atm_ce", "atm_pe", "otm_ce", "otm_pe"]

# Wall-time guards / horizons for the cadence-invariant estimators below. These
# are deliberately NOT config.py constants: they are integration/aggregation
# guards, not trained-artifact parameters, so tuning them must not change
# CONFIG_HASH and invalidate every drift reference.
#   _MAX_UPDATE_DT_S clamps the per-update dt so a gap in the tape (a session
#     boundary, or a second with no ticks during forge replay) can't nuke or
#     spike a wall-time accumulator/EWMA.
#   _RET_HORIZON_S is the return reference horizon — the forge's 1 Hz replay
#     step. At dt == _RET_HORIZON_S every estimator here is byte-identical to its
#     old per-tick form, so the live brain (~5 Hz) converges to the SAME value
#     the 1 Hz-built reference encodes (no re-forge).
_MAX_UPDATE_DT_S = 5.0
_RET_HORIZON_S = 1.0


class _Trackers:
    """Per-leg stateful physics. Same code path everywhere."""
    __slots__ = ("vol", "vpin", "hawkes", "last_ltp", "ew_voldelta",
                 "ofi_hist", "dealer_inv", "last_bid", "last_ask",
                 "last_bq", "last_aq", "oi_hist", "last_ts", "last_side",
                 "ltp_hist")

    def __init__(self):
        self.vol = EWMAVol(config.EWMA_VOL_HALFLIFE_S)
        self.vpin = RealVPIN(config.VPIN_BASE_BUCKET, config.VPIN_N_BUCKETS)
        self.hawkes = HawkesExcitation(config.HAWKES_DECAY_PER_S)
        self.last_ltp = None
        self.ew_voldelta = 1.0          # self-normalizing expected volume/s
        # buffers below are evicted by WALL-TIME in leg_features (not by element
        # count) so their windows span the same seconds at any tick cadence.
        self.ofi_hist = deque()         # (ts, ofi) — OFI z window (seconds)
        self.dealer_inv = 0.0           # decayed signed aggressive flow
        self.last_ts = None             # wall-clock of last update (dt for decay)
        self.last_side = 0.0            # tick-test: prevailing sign carried on flats
        self.last_bid = self.last_ask = 0.0
        self.last_bq = self.last_aq = 0.0
        self.oi_hist = deque()          # (ts, oi) — ΔOI window (seconds)
        self.ltp_hist = deque()         # (ts, ltp) — log-return horizon (seconds)


class StateBuilder:
    def __init__(self):
        self.surface = SVISurface()
        self.trk: dict[str, _Trackers] = defaultdict(_Trackers)
        self.frames: deque = deque(maxlen=config.SEQ_LENGTH)

    # ----------------------------------------------------------------- leg
    def leg_features(self, key: str, snap: dict, *, index: str, expiry: str,
                     strike: float, opt_type: str, spot: float, T_years: float,
                     dte: float, is_weekly: bool, ts: float) -> np.ndarray:
        """snap: ltp,bid,ask,bid_qty,ask_qty,vol_delta,oi,iceberg(optional).
        opt_type: 'CE' | 'PE' | 'SPOT'."""
        t = self.trk[key]
        ltp = float(snap.get("ltp") or 0.0)
        bid, ask = float(snap.get("bid") or 0), float(snap.get("ask") or 0)
        bq, aq = float(snap.get("bid_qty") or 0), float(snap.get("ask_qty") or 0)
        vol_d = max(float(snap.get("vol_delta") or 0.0), 0.0)
        oi = float(snap.get("oi") or 0.0)

        # ---- one wall-time dt for every cadence-anchored estimator below ----
        # At the forge's 1 Hz dt == 1.0 and every line here reduces to its old
        # per-tick form (byte-identical → no re-forge); at the brain's ~5 Hz the
        # SAME wall-clock value is produced. dt is clamped so a tape gap can't
        # nuke or spike an accumulator.
        dt = 1.0 if t.last_ts is None else \
            min(max(ts - t.last_ts, 0.0), _MAX_UPDATE_DT_S)
        t.last_ts = ts

        # log return over the ~1 s reference horizon. A per-tick return shrinks
        # with the sampling interval (a 0.2 s return at 5 Hz vs a 1 s return in
        # the 1 Hz reference); anchoring to _RET_HORIZON_S fixes that. At 1 Hz the
        # oldest price in the horizon IS the previous tick → log(ltp/last_ltp);
        # if a data gap collapses the window to this tick we fall back to the last
        # good price, so the value is byte-identical at 1 Hz even across gaps.
        log_ret = 0.0
        if ltp > 0:
            t.ltp_hist.append((ts, ltp))
            while len(t.ltp_hist) > 1 and t.ltp_hist[0][0] < ts - _RET_HORIZON_S:
                t.ltp_hist.popleft()
            p_ref = t.ltp_hist[0][1] if len(t.ltp_hist) > 1 else (t.last_ltp or ltp)
            if p_ref > 0:
                log_ret = math.log(ltp / p_ref)
        t.last_ltp = ltp if ltp > 0 else t.last_ltp

        # regime vol / hawkes — one formula, period. EWMAVol now sees the real dt
        # (return ÷√dt, half-life anchored to seconds) so it no longer reads hot
        # and fast at 5 Hz. hawkes reads the PRE-update ew_voldelta (kept).
        regime_vol = t.vol.update(ltp if ltp > 0 else (t.last_ltp or 1.0), dt_s=dt)
        hawkes = t.hawkes.update(ts, min(vol_d / max(t.ew_voldelta, 1.0), 5.0))

        # ew_voldelta: expected volume PER SECOND. The EWMA weight is anchored to a
        # 1 s step (1 - 0.999**dt == 0.001 at dt == 1) so its time-constant is
        # wall-clock, not tick-count — which also tightens velocity, dealer_inv and
        # the hawkes magnitude that read it.
        if t.ew_voldelta:
            _a = 1.0 - 0.999 ** dt
            t.ew_voldelta = (1.0 - _a) * t.ew_voldelta + _a * max(vol_d, 0.0)
        else:
            t.ew_voldelta = max(vol_d, 1.0)
        if vol_d <= 0 and t.ew_voldelta <= 1.0:
            velocity = 0.0          # volume-less feed (index spot): no signal,
        else:                       # not a constant −0.76 artifact
            velocity = math.tanh(vol_d / max(t.ew_voldelta, 1.0) - 1.0)

        vpin = t.vpin.update(ltp if ltp > 0 else 1.0, vol_d, regime_vol)
        dg = depth_gradient(bq, aq)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ltp
        spread_pct = (ask - bid) / mid if mid > 0 and ask >= bid > 0 else 0.0

        # OFI (Cont–Kukanov, top of book) → z over a TIME window. Was a tick-count
        # window (maxlen OFI_WINDOW_TICKS): 120 ticks is 120 s at the forge's 1 Hz
        # but only ~24 s at 5 Hz, so the z-score normalized over a different span
        # live vs in the reference. Now evicted by seconds; at 1 Hz it holds the
        # same 120 ticks → identical mean/std.
        ofi = 0.0
        if t.last_bid:
            if bid >= t.last_bid: ofi += bq
            if bid <= t.last_bid: ofi -= t.last_bq
            if ask <= t.last_ask: ofi -= aq
            if ask >= t.last_ask: ofi += t.last_aq
        t.ofi_hist.append((ts, ofi))
        _ofi_w = config.OFI_WINDOW_TICKS          # ticks → seconds at the 1 Hz ref
        while len(t.ofi_hist) > 1 and t.ofi_hist[0][0] <= ts - _ofi_w:
            t.ofi_hist.popleft()
        _ofis = np.fromiter((o for _, o in t.ofi_hist), dtype=float,
                            count=len(t.ofi_hist))
        mu = float(_ofis.mean()); sd = float(_ofis.std()) or 1.0
        ofi_z = float(np.clip((ofi - mu) / sd, -4, 4))
        t.last_bid, t.last_ask, t.last_bq, t.last_aq = bid, ask, bq, aq

        # dealer inventory: tick-test direction + wall-time-integrated signed flow.
        #   • TICK TEST for direction. The old `>= 0 ⇒ +1` rule voted every
        #     unchanged-price tick as a BUY, so the feature's mean scaled with the
        #     RATE of flat ticks — and a fine feeder (the live brain, ~0.2 s loop)
        #     sees far more flat ticks than a coarse one (the forge replay, 1 s).
        #     A flat tick now carries the PREVAILING side (Lee–Ready tick test),
        #     which has no directional bias yet keeps the flat-tick VOLUME.
        #   • decay and the flow increment are integrated over the shared wall-time
        #     dt: byte-identical at the forge's 1 s cadence (dt = 1) yet
        #     rate-invariant at any other cadence.
        if log_ret > 0:
            t.last_side = 1.0
        elif log_ret < 0:
            t.last_side = -1.0
        side = t.last_side
        t.dealer_inv = (config.DEALER_INV_DECAY ** dt) * t.dealer_inv \
            + side * (vol_d / max(t.ew_voldelta, 1.0)) * dt
        dealer_inv = math.tanh(t.dealer_inv / config.DEALER_INV_SCALE)

        # 15-min ΔOI, normalized. The lookup is unchanged; the buffer is now
        # evicted by TIME (was maxlen = OI_DELTA_WINDOW_S ELEMENTS → only ~180 s of
        # history at 5 Hz, truncating the intended 900 s window). At 1 Hz it holds
        # the same elements the maxlen buffer did → identical reference price.
        t.oi_hist.append((ts, oi))
        while len(t.oi_hist) > 1 and t.oi_hist[0][0] <= ts - config.OI_DELTA_WINDOW_S:
            t.oi_hist.popleft()
        old = next((o for s, o in t.oi_hist if ts - s <= config.OI_DELTA_WINDOW_S), t.oi_hist[0][1])
        oi_delta_norm = (oi - old) / max(oi, 1.0)

        if opt_type == "SPOT":
            iv = self.surface.atm_iv(index, expiry, T_years)
            k = 0.0; delta = 1.0; gamma = 0.0; theta_n = 0.0
        else:
            F = spot * math.exp(config.RISK_FREE_RATE * T_years)
            k = math.log(max(strike, 1e-6) / max(F, 1e-6))
            w = float(self.surface.total_variance(index, expiry, k))
            iv = math.sqrt(max(w, 1e-8) / max(T_years, 1e-6))
            g = black76_greeks(F, strike, T_years, iv, opt_type == "CE",
                               config.RISK_FREE_RATE)
            delta = float(g["delta"]); gamma = float(g["gamma"])
            theta_n = float(g["theta"]) / max(ltp, 0.5)

        return np.array([
            np.clip(log_ret * 100, -5, 5), np.clip(k, -0.5, 0.5),
            np.clip(oi_delta_norm, -1, 1), dg, vpin, velocity,
            np.clip(spread_pct, 0, 0.5), np.clip(iv, 0, 3),
            self.surface.skew(index, expiry), np.clip(regime_vol, 0, 3),
            np.clip(hawkes, 0, 10), float(bool(snap.get("iceberg"))),
            ofi_z, delta, np.clip(gamma * 1e4, 0, 50),
            np.clip(theta_n, -1, 0), dealer_inv,
            min(dte, 30.0) / 30.0, 1.0 if is_weekly else 0.0,
        ], dtype=np.float32)

    # ----------------------------------------------------------------- frame
    def build_frame(self, market: dict, ts: float) -> np.ndarray:
        """market: {index: {'spot':snap, 'expiry':str, 'dte':float,
        'is_weekly':bool, 'T':float, 'legs': {atm_ce|atm_pe|otm_ce|otm_pe:
        {'snap':…, 'strike':float}}}}. Missing legs → zero node (masked
        honestly, not silently)."""
        F = config.FEATURES_PER_NODE
        out = np.zeros((config.NUM_NODES, F), dtype=np.float32)
        for i, idx in enumerate(config.INDEX_ORDER):
            ctx = market.get(idx)
            if not ctx:
                continue
            spot_snap = ctx.get("spot") or {}
            spot = float(spot_snap.get("ltp") or 0.0)
            exp, dte = ctx.get("expiry", ""), float(ctx.get("dte", 1.0))
            T = float(ctx.get("T", max(dte, 0.05) / 365.0))
            wk = bool(ctx.get("is_weekly"))
            base = i * config.NODES_PER_INDEX
            out[base] = self.leg_features(f"{idx}:SPOT", spot_snap, index=idx,
                expiry=exp, strike=0.0, opt_type="SPOT", spot=spot,
                T_years=T, dte=dte, is_weekly=wk, ts=ts)
            for j, leg in enumerate(LEG_ORDER[1:], start=1):
                info = (ctx.get("legs") or {}).get(leg)
                if not info or not info.get("snap"):
                    continue
                out[base + j] = self.leg_features(
                    f"{idx}:{leg}", info["snap"], index=idx, expiry=exp,
                    strike=float(info.get("strike") or 0.0),
                    opt_type="CE" if leg.endswith("ce") else "PE",
                    spot=spot, T_years=T, dte=dte, is_weekly=wk, ts=ts)
        return out

    def push(self, market: dict, ts: float) -> np.ndarray | None:
        self.frames.append(self.build_frame(market, ts))
        if len(self.frames) < config.SEQ_LENGTH:
            return None
        return np.stack(self.frames).reshape(-1).astype(np.float32)  # 5700

    def fit_surface(self, index, expiry, strikes, ivs, F, T):
        """Feed REAL Newton IVs in; store total variance per (index, expiry)."""
        k = np.log(np.asarray(strikes, float) / max(F, 1e-6))
        w = np.square(np.asarray(ivs, float)) * max(T, 1e-6)
        self.surface.fit(index, expiry, k, w)