"""
APEX OMNI v9 — MACRO GEX RADAR (audit §7 leaps)
===============================================
  * REAL implied vols: vectorized Newton with analytic vega on the futures-
    style forward — the ATM-only Brenner shortcut across ±10% of strikes is
    gone, so the wings stop distorting gamma.
  * The flip line is the ZERO-CROSSING of a smoothed net-GEX profile, with a
    confidence width (distance between the bracketing strikes), instead of
    the argmin of a noisy discrete array that teleported tick to tick.
  * Publishes FULL per-strike arrays (strike, iv, gex, ΔOI-15m) so downstream
    consumers can judge conviction instead of trusting three numbers.
  * Dealer-sign calibration hook: drop a participant_oi.json (built from
    NSE's daily participant-wise OI file) next to this script and the sign
    convention stops being imported US folklore. Defaults to the classic
    dealers-long-calls / short-puts book with a clear log line saying so.
  * Atomic temp→rename JSON (kept from v8 — it was right).
GPU optional: numpy is plenty for ~200 strikes; torch is used if present.
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

import config
from core.instruments import LiveMapper
from core.quant_core import implied_vol_newton, black76_greeks

log = logging.getLogger("macro")

try:
    from kiteconnect import KiteConnect
    HAVE_KITE = True
except Exception:                                     # noqa: BLE001
    HAVE_KITE = False


_oi_hist: dict[str, deque] = {}


def dealer_sign():
    p = Path(__file__).parent / "participant_oi.json"
    if p.exists():
        try:
            j = json.loads(p.read_text())
            s_call, s_put = float(j.get("call_sign", 1)), float(j.get("put_sign", -1))
            log.info("dealer sign calibrated from participant_oi.json: "
                     "call %+.1f put %+.1f", s_call, s_put)
            return s_call, s_put
        except Exception as e:                         # noqa: BLE001
            log.warning("participant_oi.json unreadable (%s) — using default", e)
    log.info("dealer sign: DEFAULT (long calls / short puts) — calibrate with "
             "NSE participant-wise OI when you can; Indian retail is a heavy "
             "net options BUYER and may flip this.")
    return 1.0, -1.0


def atomic_write(path: str, payload: dict):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def compute_index(kite, mapper: LiveMapper, index: str, s_call, s_put):
    spot_sym = config.INDICES[index]["spot_symbol"]
    spot = float(kite.ltp([spot_sym])[spot_sym]["last_price"])
    rows = mapper.by_index.get(index, [])
    if not rows or spot <= 0:
        return
    exps = sorted({r["expiry"] for r in rows if r["expiry"] >= dt.date.today()})
    if not exps:
        return
    exp = exps[0]
    band = [r for r in rows if r["expiry"] == exp
            and abs(r["strike"] - spot) / spot <= config.MACRO_STRIKE_BAND]
    if len(band) < 8:
        return
    lot = band[0]["lot"]
    dte = max((exp - dt.date.today()).days, 0) + config.DTE_PART_DAY
    T = dte / 365.0
    F = spot * math.exp(config.RISK_FREE_RATE * T)

    keys = [f'{r["exchange"]}:{r["symbol"]}' for r in band]
    quotes = {}
    for i in range(0, len(keys), config.MACRO_QUOTE_CHUNK):
        quotes.update(kite.quote(keys[i:i + config.MACRO_QUOTE_CHUNK]))
        time.sleep(1.05)                                # quote API: 1 req/s
    K, prem, oi, is_call = [], [], [], []
    for r, key in zip(band, keys):
        q = quotes.get(key)
        if not q:
            continue
        d = q.get("depth") or {}
        b = (d.get("buy") or [{}])[0].get("price") or 0
        a = (d.get("sell") or [{}])[0].get("price") or 0
        mid = (b + a) / 2 if b and a else q.get("last_price") or 0
        if mid <= 0:
            continue
        K.append(r["strike"]); prem.append(mid)
        oi.append(float(q.get("oi") or 0)); is_call.append(r["itype"] == "CE")
    if len(K) < 8:
        return
    K = np.array(K); prem = np.array(prem)
    oi = np.array(oi); is_call = np.array(is_call)

    iv = implied_vol_newton(prem, F, K, T, is_call, config.RISK_FREE_RATE)
    g = black76_greeks(F, K, T, iv, is_call, config.RISK_FREE_RATE)
    sign = np.where(is_call, s_call, s_put)
    gex = sign * g["gamma"] * oi * lot * spot * spot * 0.01
    dex = sign * g["delta"] * oi * lot * spot

    # net per strike, smoothed, true zero crossing
    uniq = np.unique(K)
    net = np.array([gex[K == k].sum() for k in uniq])
    if len(uniq) >= 5:
        kern = np.array([1, 2, 3, 2, 1], float); kern /= kern.sum()
        sm = np.convolve(net, kern, mode="same")
    else:
        sm = net
    flip = flip_w = None
    sgn = np.sign(sm)
    for i in range(len(sm) - 1):
        if sgn[i] != 0 and sgn[i + 1] != 0 and sgn[i] != sgn[i + 1]:
            x0, x1, y0, y1 = uniq[i], uniq[i + 1], sm[i], sm[i + 1]
            flip = float(x0 - y0 * (x1 - x0) / (y1 - y0))
            flip_w = float(x1 - x0)
            break
    cg = np.where(is_call, gex, 0); pg = np.where(~is_call, np.abs(gex), 0)
    call_wall = float(K[np.argmax(cg)]) if cg.any() else None
    put_wall = float(K[np.argmax(pg)]) if pg.any() else None

    h = _oi_hist.setdefault(index, deque(maxlen=12))
    h.append((time.time(), {float(k): float(o) for k, o in zip(K, oi)}))
    doi15 = []
    if len(h) >= 2:
        old = next((m for t, m in h if time.time() - t <= 930), h[0][1])
        doi15 = [float(o - old.get(float(k), o)) for k, o in zip(K, oi)]

    # ---- weapons: PCR, max pain, IV rank (all from the same real chain) ----
    put_oi = float(oi[~is_call].sum()); call_oi = float(oi[is_call].sum())
    pcr = put_oi / call_oi if call_oi > 0 else None
    uK = np.unique(K)
    pain = [float((np.where(is_call, oi, 0) * np.maximum(s - K, 0)).sum()
                  + (np.where(~is_call, oi, 0) * np.maximum(K - s, 0)).sum())
            for s in uK]
    max_pain = float(uK[int(np.argmin(pain))]) if len(uK) else None
    atm_iv = float(iv[np.argmin(np.abs(K - spot))])
    hist_p = config.STATE_DIR / f"iv_history_{index}.json"
    today = str(dt.date.today())
    ivh = {}
    if hist_p.exists():
        try:
            ivh = json.loads(hist_p.read_text())
        except Exception:                              # noqa: BLE001
            ivh = {}
    ivh[today] = atm_iv
    ivh = dict(sorted(ivh.items())[-60:])
    tmpf = hist_p.with_suffix(".tmp")
    tmpf.write_text(json.dumps(ivh)); tmpf.replace(hist_p)
    past = [v for d_, v in ivh.items() if d_ != today]
    iv_rank = (float(np.mean([v <= atm_iv for v in past]))
               if len(past) >= config.IVRANK_MIN_DAYS else None)

    # intraday ATM-IV sample for the vol-surface forecaster (daily history is
    # too coarse to forecast crush). Real recorded series, no fabrication.
    try:
        from core.vol_forecaster import append_intraday_sample
        append_intraday_sample(index, atm_iv)
    except Exception:                                  # noqa: BLE001
        pass

    payload = {"ts": time.time(), "index": index, "spot": spot,
               "expiry": str(exp), "flip": flip, "flip_width": flip_w,
               "call_wall": call_wall, "put_wall": put_wall,
               "net_gex": float(gex.sum()), "net_dex": float(dex.sum()),
               "pcr": pcr, "max_pain": max_pain, "atm_iv": atm_iv,
               "iv_rank": iv_rank, "dte": dte,
               "strikes": K.tolist(), "iv": iv.round(4).tolist(),
               "gex": gex.round(2).tolist(), "doi15": doi15}
    atomic_write(config.MACRO_STATE_TMPL.format(idx=index), payload)
    log.info("%s spot %.1f flip %s±%s walls %s/%s PCR %s maxpain %s "
             "IVrank %s netGEX %.2e",
             index, spot, f"{flip:.0f}" if flip else "—",
             f"{flip_w:.0f}" if flip_w else "—", put_wall, call_wall,
             f"{pcr:.2f}" if pcr else "—",
             f"{max_pain:.0f}" if max_pain else "—",
             f"{iv_rank:.2f}" if iv_rank is not None else "—", gex.sum())


def main():
    if not HAVE_KITE:
        raise SystemExit("kiteconnect not installed")
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    kite.set_access_token(config.KITE_ACCESS_TOKEN)
    mapper = LiveMapper(kite)
    s_call, s_put = dealer_sign()
    while True:
        t0 = time.time()
        for idx in config.INDEX_ORDER:
            try:
                compute_index(kite, mapper, idx, s_call, s_put)
            except Exception as e:                     # noqa: BLE001
                log.error("%s: %s", idx, e)
        time.sleep(max(config.MACRO_LOOP_S - (time.time() - t0), 5))


if __name__ == "__main__":
    config.setup_logging("macro")
    main()
