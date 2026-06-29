"""
APEX OMNI v9 — QUANT CORE (audit §3 fixes)
==========================================
Fixes vs v8:
  * ONE definition of sigma everywhere: annualized IV. SVI stores total
    variance w(k); consumers call atm_iv() which returns sqrt(w/T). The v8
    sqrt(a)≈IV·√T unit bug (tiny targets, huge stops) is dead.
  * Per-(index, expiry) SVI parameter sets — no more six indices fighting
    over one shared vector.
  * Real implied vol: vectorized Newton with analytic vega on the Black-76
    forward, replacing the ATM-only Brenner–Subrahmanyam shortcut.
  * Gatheral butterfly (g(k) ≥ 0) sanity check before a fit is accepted.
  * Put charm: spurious +r·e^(−rT) term removed (charm_put == charm_call, q=0).
  * VPIN: true volume buckets, tick-rule signing, adaptive bucket size that
    actually receives volatility, and a rolling value at EVERY tick (the v8
    column of zeros is dead).
  * depth_gradient: instantaneous book imbalance — no cumulative-volume
    denominator, so 2 p.m. looks like 10 a.m.
Pure numpy. No torch import here (forge/sim/analyzer all reuse this).
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)

# ----------------------------------------------------------------- Black-76
def _norm_cdf(x): return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))
def _norm_pdf(x): return np.exp(-0.5 * np.square(x)) / SQRT_2PI

def black76_price(F, K, T, sigma, is_call, r=0.07):
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    T = np.maximum(T, 1e-6); sigma = np.maximum(sigma, 1e-6)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    df = np.exp(-r * T)
    call = df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    put  = df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
    return np.where(is_call, call, put)

def black76_greeks(F, K, T, sigma, is_call, r=0.07):
    """Returns dict of arrays: delta, gamma, vega, theta(/day), charm(/day), vanna."""
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    T = np.maximum(T, 1e-6); sigma = np.maximum(sigma, 1e-6)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    df = np.exp(-r * T)
    pdf = _norm_pdf(d1)
    delta = np.where(is_call, df * _norm_cdf(d1), -df * _norm_cdf(-d1))
    gamma = df * pdf / (F * sT)
    vega  = df * F * pdf * np.sqrt(T) / 100.0          # per 1 vol-pt
    theta = (-(F * pdf * sigma * df) / (2.0 * np.sqrt(T))
             + np.where(is_call, -1, 1) * r * K * df * _norm_cdf(np.where(is_call, d2, -d2))
             + r * np.where(is_call, 1, -1) * F * df * _norm_cdf(np.where(is_call, d1, -d1))) / 365.0
    # Charm dDelta/dt. With q=0 the call/put charm are IDENTICAL (audit fix:
    # v8 added a phantom +r·e^(−rT) on the put side).
    charm = -df * pdf * (2 * r * T - d2 * sT) / (2 * T * sT) / 365.0
    charm = np.where(is_call, charm, charm)            # explicit: same formula
    vanna = -df * pdf * d2 / sigma
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "charm": charm, "vanna": vanna}

def implied_vol_newton(price, F, K, T, is_call, r=0.07, iters=8, lo=0.01, hi=4.0):
    """Vectorized Newton–Raphson with analytic vega. Microseconds for a chain."""
    price, F, K, T = (np.asarray(a, float) for a in (price, F, K, T))
    T = np.maximum(T, 1e-6)
    sigma = np.full_like(F, 0.20)
    # honest seed: Brenner ATM guess where it is actually valid
    atmish = np.abs(np.log(F / K)) < 0.01
    with np.errstate(all="ignore"):
        sigma = np.where(atmish, np.clip(price / (0.4 * F * np.sqrt(T) * np.exp(-r * T)), lo, hi), sigma)
    for _ in range(iters):
        sT = np.maximum(sigma, 1e-6) * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
        vega = np.exp(-r * T) * F * _norm_pdf(d1) * np.sqrt(T)
        diff = black76_price(F, K, T, sigma, is_call, r) - price
        sigma = np.clip(sigma - diff / np.maximum(vega, 1e-8), lo, hi)
    return sigma

# ----------------------------------------------------------------- SVI (per key)
class SVISurface:
    """
    w(k) = a + b·(ρ(k−m) + sqrt((k−m)² + σ²)) — Gatheral raw SVI, fitted
    SEPARATELY per (index, expiry). k = ln(K/F). Stores TOTAL VARIANCE w;
    every consumer converts to annualized IV via iv = sqrt(w / T).
    """
    DEFAULT = np.array([0.0004, 0.10, -0.30, 0.0, 0.10])   # a,b,rho,m,sig

    def __init__(self):
        self.params: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _w(p, k):
        a, b, rho, m, s = p
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))

    @staticmethod
    def _butterfly_ok(p, k_grid):
        """Gatheral g(k) ≥ 0 (numeric) — rejects arbitrageable fits."""
        w = SVISurface._w(p, k_grid)
        dw = np.gradient(w, k_grid); d2w = np.gradient(dw, k_grid)
        g = ((1 - k_grid * dw / (2 * w)) ** 2
             - 0.25 * dw * dw * (1.0 / w + 0.25) + 0.5 * d2w)
        return bool(np.all(g > -1e-6))

    def fit(self, index, expiry, k, w_obs, steps=60, lr=0.05):
        """Warm-started gradient fit (kept from v8 — it was a good idea) but
        per-key, on REAL IV-derived variances, with a butterfly gate."""
        k = np.asarray(k, float); w_obs = np.asarray(w_obs, float)
        mask = np.isfinite(w_obs) & (w_obs > 0)
        if mask.sum() < 3:
            return self.params.get((index, expiry), self.DEFAULT.copy())
        k, w_obs = k[mask], w_obs[mask]
        p = self.params.get((index, expiry), self.DEFAULT.copy()).astype(float)
        for _ in range(steps):
            eps = 1e-5; grad = np.zeros(5)
            base = float(np.mean((self._w(p, k) - w_obs) ** 2))
            for i in range(5):
                q = p.copy(); q[i] += eps
                grad[i] = (np.mean((self._w(q, k) - w_obs) ** 2) - base) / eps
            p -= lr * grad
            p[0] = max(p[0], 1e-6); p[1] = max(p[1], 1e-4)
            p[2] = float(np.clip(p[2], -0.999, 0.999)); p[4] = max(p[4], 1e-3)
        if self._butterfly_ok(p, np.linspace(-0.15, 0.15, 41)):
            self.params[(index, expiry)] = p
        return self.params.get((index, expiry), self.DEFAULT.copy())

    def total_variance(self, index, expiry, k):
        return self._w(self.params.get((index, expiry), self.DEFAULT), np.asarray(k, float))

    def atm_iv(self, index, expiry, T_years):
        """THE sigma. Annualized ATM implied vol — the one unit everyone uses."""
        w = float(self.total_variance(index, expiry, 0.0))
        return math.sqrt(max(w, 1e-8) / max(T_years, 1e-6))

    def skew(self, index, expiry):
        return float(self.params.get((index, expiry), self.DEFAULT)[2])

def expected_move(spot, atm_iv_annual, minutes_remaining):
    """1-sigma expected move over the REMAINING intraday horizon. Single time
    discount (audit: v8 discounted twice and produced ~20-pt NIFTY targets)."""
    frac_year = max(minutes_remaining, 1.0) / (252.0 * 375.0)
    return float(spot) * float(atm_iv_annual) * math.sqrt(frac_year)

# ----------------------------------------------------------------- microstructure
def depth_gradient(bid_qty, ask_qty):
    """Instantaneous top-of-book imbalance in [-1, 1]. No cumulative-volume
    denominator — the v8 'fades to zero by 2 p.m.' bug is gone."""
    b, a = float(bid_qty), float(ask_qty)
    tot = b + a
    return (b - a) / tot if tot > 0 else 0.0

class RealVPIN:
    """Volume-synchronized probability of informed trading, done honestly:
    tick-rule signed volume, fixed-volume buckets whose size ADAPTS to
    realized volatility, rolling mean over the last N buckets, and a smooth
    intra-bucket estimate so the value is defined at every tick."""
    def __init__(self, base_bucket=5000, n_buckets=20):
        self.base = base_bucket
        self.bucket_target = base_bucket
        self.buy = 0.0; self.sell = 0.0
        self.done = deque(maxlen=n_buckets)
        self.last_price = None

    def update(self, price, volume, volatility=None):
        if volatility is not None and volatility > 0:
            # higher vol → smaller buckets → faster clock (Easley/O'Hara spirit)
            self.bucket_target = float(np.clip(self.base / (1.0 + 50.0 * volatility),
                                               self.base * 0.2, self.base * 3.0))
        if self.last_price is None:
            self.last_price = price
        side_buy = price >= self.last_price          # tick rule
        self.last_price = price
        (self.__dict__.__setitem__("buy", self.buy + volume) if side_buy
         else self.__dict__.__setitem__("sell", self.sell + volume))
        while self.buy + self.sell >= self.bucket_target:
            tot = self.buy + self.sell
            imb = abs(self.buy - self.sell) / tot if tot else 0.0
            self.done.append(imb)
            scale = self.bucket_target / tot
            self.buy *= (1 - scale); self.sell *= (1 - scale)
            if tot <= self.bucket_target: break
        hist = list(self.done)
        cur_tot = self.buy + self.sell
        if cur_tot > 0:
            hist = hist + [abs(self.buy - self.sell) / cur_tot]
        return float(np.mean(hist)) if hist else 0.0

class EWMAVol:
    """Per-second EWMA of |log returns| → crude annualized vol. SAME class is
    used by harvester, forge, live brain and analyzer (train/serve unity)."""
    def __init__(self, half_life_s=20.0):
        self.alpha = 1 - math.exp(math.log(0.5) / half_life_s)
        self.ewma = 0.0; self.last = None
    def update(self, price, dt_s=1.0):
        # dt_s-aware: the return is normalized by sqrt(dt) and the EWMA weight is
        # anchored to a 1 s step (a == self.alpha at dt_s == 1.0), so the half-life
        # is wall-clock seconds rather than a fixed tick count. Called with the
        # real dt the value is cadence-invariant; called with the dt_s=1.0 default
        # (e.g. the 1 Hz-gated rvol in the brain) it is byte-identical to before.
        if self.last and self.last > 0 and price > 0:
            dt = max(float(dt_s), 1e-6)
            r = abs(math.log(price / self.last)) / max(math.sqrt(dt), 1e-3)
            a = 1.0 - (1.0 - self.alpha) ** dt
            self.ewma = (1 - a) * self.ewma + a * r
        self.last = price
        return self.annualized()
    def annualized(self):
        return self.ewma * math.sqrt(252.0 * 375.0 * 60.0)

class HawkesExcitation:
    """Self-exciting intensity with an exponential (time-based) kernel.

    CADENCE-INVARIANT and reference-preserving. The old form added the
    excitation once per call (``+= intensity``), so the steady-state value
    scaled with the *update rate*: at the brain's ~5 Hz it sat ~2.6x above the
    forge's 1 Hz reference — a train/serve skew identical in shape to the
    dealer-inventory one. Here the per-step excitation is weighted by
    ``(1-e^{-decay·dt}) / (1-e^{-decay·dt_ref})`` with ``dt_ref`` = the forge's
    1 Hz replay step. Two properties fall out:
      • at dt = dt_ref the weight is 1.0  → ``+= intensity`` exactly, so the
        existing drift reference is unchanged (NO re-forge);
      • the steady-state value is ``intensity / (1-e^{-decay·dt_ref})`` for ANY
        dt, so the brain at 5 Hz converges to the SAME value the 1 Hz reference
        encodes (a plain ``×dt`` does not — it overshoots to ~0.5x).
    dt is clamped to guard feed gaps / out-of-order timestamps.
    """
    _MAX_DT_S = 5.0
    _REF_DT_S = 1.0          # forge replays at 1 Hz; reference is built at this step

    def __init__(self, decay_per_s=2.0):
        self.decay = decay_per_s; self.val = 0.0; self.last_t = None
        self._ref_w = 1.0 - math.exp(-decay_per_s * self._REF_DT_S)
    def update(self, t, intensity):
        if self.last_t is None:
            dt = self._REF_DT_S                  # first tick: matches old (+= intensity)
        else:
            dt = min(max(t - self.last_t, 0.0), self._MAX_DT_S)
            self.val *= math.exp(-self.decay * dt)
        self.last_t = t
        w = (1.0 - math.exp(-self.decay * dt)) / self._ref_w
        self.val += float(intensity) * w
        return self.val

def micro_price(bid, ask, bid_qty, ask_qty):
    tot = bid_qty + ask_qty
    if tot <= 0 or bid <= 0 or ask <= 0:
        return (bid + ask) / 2.0 if (bid and ask) else max(bid, ask)
    return (bid * ask_qty + ask * bid_qty) / tot      # Stoikov

def bayesian_signal_fusion(ai_conviction, quant_shock, quant_weight=2.0):
    """Logit-space fusion (single shared copy — v8 had three diverging ones)."""
    ai = float(np.clip(ai_conviction, -0.999, 0.999))
    logit = math.atanh(ai) + quant_weight * float(np.clip(quant_shock, -1, 1))
    return math.tanh(logit)