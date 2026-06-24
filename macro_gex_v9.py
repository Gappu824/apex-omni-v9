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

v9.1: the per-snapshot surface/GEX math is factored into assemble_snapshot()
so the historical backfill (tools/backfill_macro.py) reconstructs past days by
calling the EXACT same function the live radar uses — one implementation, zero
train/serve drift. The live path (compute_index) is unchanged in behaviour.
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import math
import os
import sqlite3
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


# --------------------------------------------------------------- vault archive
# WHY THIS EXISTS. The live brain fits its per-expiry SVI surface from the macro
# radar's full per-strike IVs (apex_main_v9.fit_surface ← mac["strikes"]/["iv"]),
# and shapes every exit target with the GEX walls (call_wall / put_wall). The
# nightly forge replays raw ticks but has NO macro JSON for a past second — it is
# overwritten every loop — so its StateBuilder surface is never fit: iv/delta/
# gamma/theta train on the SVISurface DEFAULT (≈70-190% intraday IV, not the real
# ~15%), and the reward's shaped target falls back to the no-wall expected move.
# Persisting each snapshot to the SAME tick vault, timestamped, lets the forge
# reconstruct the real surface and the real walls AS OF each second — closing a
# train/serve skew across the four greeks features and the target's wall cap.
# Future tape only (you can't archive a past macro state); start now.
#
# Best-effort by construction: the vault write is wrapped so a locked/again-busy
# DB can NEVER delay or break the live JSON the brain reads. WAL + busy_timeout
# let it coexist with the harvester writing ticks to the same file.
MACRO_ARCHIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_snapshots_v9 (
    ts_ms       INTEGER, index_name TEXT, spot REAL, expiry TEXT, dte REAL,
    flip        REAL, flip_width REAL, call_wall REAL, put_wall REAL,
    net_gex     REAL, net_dex REAL, pcr REAL, max_pain REAL,
    atm_iv      REAL, iv_rank REAL, strikes_json TEXT, iv_json TEXT,
    PRIMARY KEY (ts_ms, index_name));
CREATE INDEX IF NOT EXISTS idx_macro_idx_ts
    ON macro_snapshots_v9 (index_name, ts_ms);
"""
_MACRO_COLS = ("ts_ms", "index_name", "spot", "expiry", "dte", "flip",
               "flip_width", "call_wall", "put_wall", "net_gex", "net_dex",
               "pcr", "max_pain", "atm_iv", "iv_rank", "strikes_json", "iv_json")


class MacroArchive:
    """Persists each radar snapshot to the tick vault (config.DB_PATH). One lazy
    connection, reused; ts stored as ts_ms to match ticks_v9 so the forge can
    line snapshots up with ticks on the same clock."""

    def __init__(self):
        self.con: sqlite3.Connection | None = None

    def _ensure(self) -> sqlite3.Connection:
        if self.con is None:
            con = sqlite3.connect(str(config.DB_PATH), timeout=5.0,
                                  check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA busy_timeout=5000;")     # wait, don't error, on lock
            con.executescript(MACRO_ARCHIVE_SCHEMA)
            self.con = con
        return self.con

    def write(self, p: dict) -> None:
        try:
            con = self._ensure()
            row = (int(float(p["ts"]) * 1000), p["index"], p.get("spot"),
                   p.get("expiry"), p.get("dte"), p.get("flip"),
                   p.get("flip_width"), p.get("call_wall"), p.get("put_wall"),
                   p.get("net_gex"), p.get("net_dex"), p.get("pcr"),
                   p.get("max_pain"), p.get("atm_iv"), p.get("iv_rank"),
                   json.dumps(p.get("strikes") or []),
                   json.dumps(p.get("iv") or []))
            con.execute(
                f"INSERT OR REPLACE INTO macro_snapshots_v9 VALUES "
                f"({','.join('?' * len(_MACRO_COLS))})", row)
            con.commit()
        except Exception as e:                            # noqa: BLE001
            log.warning("macro vault archive skipped (live JSON unaffected): %s", e)


_ARCHIVE = MacroArchive()


def load_macro_archive(con: sqlite3.Connection, day: str, index: str) -> list[dict]:
    """Forge-side reader. Every archived snapshot for (day, index), oldest first,
    strikes/iv parsed back to lists. The replay uses the latest snapshot
    at-or-before each second — a right-continuous step, exactly how the brain
    reads the latest published JSON — to fit the surface and read the walls.
    Returns [] when nothing was archived (older days): the forge then keeps its
    current seed-surface / no-wall behaviour, so this is purely additive."""
    try:
        rows = con.execute(
            "SELECT ts_ms, spot, expiry, dte, call_wall, put_wall, atm_iv, "
            "iv_rank, pcr, max_pain, net_gex, strikes_json, iv_json "
            "FROM macro_snapshots_v9 WHERE index_name=? AND "
            "date(ts_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (index, day)).fetchall()
    except sqlite3.OperationalError:                      # table absent ⇒ no archive
        return []
    out = []
    for (ts_ms, spot, expiry, dte, cw, pw, aiv, ivr, pcr, mp, ng,
         sj, ij) in rows:
        out.append({"ts": ts_ms / 1000.0, "spot": spot, "expiry": expiry,
                    "dte": dte, "call_wall": cw, "put_wall": pw, "atm_iv": aiv,
                    "iv_rank": ivr, "pcr": pcr, "max_pain": mp, "net_gex": ng,
                    "strikes": json.loads(sj or "[]"),
                    "iv": json.loads(ij or "[]")})
    return out


# IV-pin rejection. implied_vol_newton clamps sigma to [_IV_FLOOR, _IV_CEIL]; a
# vol that comes back pinned at (or beside) a clamp means the input price was
# outside the no-arbitrage range — a stale or one-sided quote, not a real vol.
# Its gamma (∝ 1/sigma) would explode and poison net_gex and the wall picks, which
# is exactly what the thin BSE chains (SENSEX/BANKEX) do when a missing book falls
# back to a stale last price. assemble_snapshot drops any strike outside the
# trusted band before any greek is taken. Tied to the solver clamp so they cannot
# drift apart.
_IV_FLOOR = 0.01
_IV_CEIL = 4.0
_IV_MIN_VALID = 0.02       # 2% — below this the solve has pinned to the floor
_IV_MAX_VALID = 3.0        # 300% — above this it has pinned to the ceiling
_MIN_VALID_CONTRACTS = 6   # too few real strikes survive ⇒ skip the snapshot


def assemble_snapshot(*, ts, index, spot, exp, dte, K, mid, oi, is_call, lot,
                      s_call, s_put):
    """PURE surface/GEX math — no I/O, no module globals. Shared verbatim by the
    live radar (compute_index) and the historical backfill so replayed snapshots
    are numerically identical to what a live run would have published.

    Inputs are per-contract arrays already collapsed to a single expiry:
        K, mid, oi, is_call  — strike / option mid price / open interest / CE?
        lot                  — contract lot size for that index
        dte                  — calendar DTE (+ intraday remainder)
        s_call, s_put        — dealer sign convention

    Returns (payload, K_arr, iv_arr, gex_arr). `payload` carries every column the
    vault archive stores; iv_rank is left None (the live path fills it from the
    daily IV-history file; the forge does not read iv_rank from the vault). The
    raw arrays are handed back so the live path can attach its JSON-only extras
    (per-strike gex, ΔOI-15m)."""
    K = np.asarray(K, float); prem = np.asarray(mid, float)
    oi = np.asarray(oi, float); is_call = np.asarray(is_call, bool)
    T = max(dte, 0.0) / 365.0
    F = spot * math.exp(config.RISK_FREE_RATE * T)

    iv = implied_vol_newton(prem, F, K, T, is_call, config.RISK_FREE_RATE,
                            lo=_IV_FLOOR, hi=_IV_CEIL)
    # Drop strikes whose vol pinned to the clamp (a stale/one-sided quote): their
    # gamma blows up (∝ 1/sigma). A strike is only trustworthy when BOTH legs
    # priced cleanly — drop the whole strike if either leg pins, else the surviving
    # leg breaks the call/put gamma cancellation that net_gex and the flip rest on.
    # If too few real strikes remain, skip the snapshot: the forge's
    # right-continuous step holds the last good one, which beats a poisoned surface.
    ok = (iv >= _IV_MIN_VALID) & (iv <= _IV_MAX_VALID) & np.isfinite(iv)
    bad_strikes = set(K[~ok].tolist())
    keep = np.array([k not in bad_strikes for k in K], bool)
    if int(keep.sum()) < _MIN_VALID_CONTRACTS or len(np.unique(K[keep])) < 3:
        return None
    K, oi, is_call, iv = K[keep], oi[keep], is_call[keep], iv[keep]

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

    # ---- weapons: PCR, max pain, ATM IV (all from the same real chain) ----
    put_oi = float(oi[~is_call].sum()); call_oi = float(oi[is_call].sum())
    pcr = put_oi / call_oi if call_oi > 0 else None
    uK = np.unique(K)
    pain = [float((np.where(is_call, oi, 0) * np.maximum(s - K, 0)).sum()
                  + (np.where(~is_call, oi, 0) * np.maximum(K - s, 0)).sum())
            for s in uK]
    max_pain = float(uK[int(np.argmin(pain))]) if len(uK) else None
    atm_iv = float(iv[np.argmin(np.abs(K - spot))])

    payload = {"ts": ts, "index": index, "spot": spot,
               "expiry": str(exp), "flip": flip, "flip_width": flip_w,
               "call_wall": call_wall, "put_wall": put_wall,
               "net_gex": float(gex.sum()), "net_dex": float(dex.sum()),
               "pcr": pcr, "max_pain": max_pain, "atm_iv": atm_iv,
               "iv_rank": None, "dte": dte,
               "strikes": K.tolist(), "iv": iv.round(4).tolist()}
    return payload, K, iv, gex


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
        mid = (b + a) / 2 if (b and a) else 0            # require a real two-sided book
        if mid <= 0:
            continue
        K.append(r["strike"]); prem.append(mid)
        oi.append(float(q.get("oi") or 0)); is_call.append(r["itype"] == "CE")
    if len(K) < 8:
        return

    res = assemble_snapshot(
        ts=time.time(), index=index, spot=spot, exp=exp, dte=dte, K=K,
        mid=prem, oi=oi, is_call=is_call, lot=lot, s_call=s_call, s_put=s_put)
    if res is None:                                       # whole chain too thin/stale
        return
    payload, K, iv, gex = res

    # ---- live-only enrichments (the forge reads NONE of these from the vault) ---
    h = _oi_hist.setdefault(index, deque(maxlen=12))
    h.append((time.time(), {float(k): float(o) for k, o in zip(K, oi)}))
    doi15 = []
    if len(h) >= 2:
        old = next((m for t, m in h if time.time() - t <= 930), h[0][1])
        doi15 = [float(o - old.get(float(k), o)) for k, o in zip(K, oi)]

    atm_iv = payload["atm_iv"]
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
    payload["iv_rank"] = iv_rank

    # intraday ATM-IV sample for the vol-surface forecaster (daily history is
    # too coarse to forecast crush). Real recorded series, no fabrication.
    try:
        from core.vol_forecaster import append_intraday_sample
        append_intraday_sample(index, atm_iv)
    except Exception:                                  # noqa: BLE001
        pass

    payload["gex"] = gex.round(2).tolist()
    payload["doi15"] = doi15
    atomic_write(config.MACRO_STATE_TMPL.format(idx=index), payload)
    _ARCHIVE.write(payload)                               # vault: forge surface + walls
    log.info("%s spot %.1f flip %s±%s walls %s/%s PCR %s maxpain %s "
             "IVrank %s netGEX %.2e",
             index, spot, f'{payload["flip"]:.0f}' if payload["flip"] else "—",
             f'{payload["flip_width"]:.0f}' if payload["flip_width"] else "—",
             payload["put_wall"], payload["call_wall"],
             f'{payload["pcr"]:.2f}' if payload["pcr"] else "—",
             f'{payload["max_pain"]:.0f}' if payload["max_pain"] else "—",
             f"{iv_rank:.2f}" if iv_rank is not None else "—", payload["net_gex"])


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