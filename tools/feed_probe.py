"""
APEX OMNI — FEED PROBE (does BFO deliver WebSocket depth?)
==========================================================
THE EVIDENCE THAT PROMPTED THIS (all from your own artifacts, 28 days):

  harvester   SENSEX / BANKEX leg_depth_2sided = 0 on EVERY day, while every
              NFO index reports leg_depth_rate = 1.0
  cascade     9 of 12 real triggers lost to unharvested/stale legs are SENSEX
  shortvol    11 of 11 "short leg one-sided" skips are SENSEX
  macro REST  NFO quotes 190/190, 226/226, 204/204 — BFO quotes 0 of 304

The harvester's extraction is textbook-correct Kite FULL structure
(`depth["buy"][0]["price"]`), so this is not a parsing bug. Either BFO ticks
arrive without a depth book, or they arrive one-sided. Those have different
consequences and this tool tells them apart with YOUR subscription.

WHY IT MATTERS
--------------
Fills are priced from the ring, which is fed by the WebSocket — not by REST.
If BFO never delivers 2-sided depth then SENSEX is STRUCTURALLY UNFILLABLE:
it is half of TRADABLE, it is starving the cascade certificate (11 of 20
events), and every SENSEX signal the system generates is unactionable. That is
worth knowing before another month of harvesting.

WHAT IT DOES
------------
Subscribes a handful of NFO and BFO ATM option tokens in MODE_FULL, listens,
and reports per exchange: ticks received, how many carried a depth book, how
many were 2-sided, and the raw field names on a sample tick. Read-only — it
opens its own socket, touches no ring, no vault, no model, and places nothing.

  python tools/feed_probe.py --seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402
from core.instruments import LiveMapper                    # noqa: E402

config.setup_logging("feed_probe")
import logging                                             # noqa: E402
log = logging.getLogger("feed_probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--per-index", type=int, default=4)
    a = ap.parse_args()

    try:
        from kiteconnect import KiteConnect, KiteTicker
    except Exception as e:                                 # noqa: BLE001
        log.error("kiteconnect not importable (%s)", e)
        return
    api_key = getattr(config, "KITE_API_KEY", "") or ""
    token = getattr(config, "KITE_ACCESS_TOKEN", "") or ""
    if not api_key or not token:
        log.error("no Kite creds in the environment (KITE_API_KEY / "
                  "KITE_ACCESS_TOKEN) — the same vars apex_main uses. Set "
                  "them and re-run DURING market hours.")
        return

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    mapper = LiveMapper(kite)

    # pick ATM option legs for one NFO and one BFO index
    picks: dict = {}
    for idx in ("NIFTY", "SENSEX"):
        try:
            # the key is spot_symbol (already "NSE:NIFTY 50" / "BSE:SENSEX"
            # form), matching data_harvester_v9:224 and macro_gex_v9:432.
            sym = config.INDICES[idx]["spot_symbol"]
            q = kite.ltp([sym])
            if sym not in q:
                raise KeyError(f"no LTP for {sym}")
            spot = float(q[sym]["last_price"])
            ch = mapper.chain(idx, spot)
            toks = []
            for leg, info in (ch.get("legs") or {}).items():
                if info and info.get("token"):
                    toks.append((int(info["token"]), info.get("symbol", leg)))
            picks[idx] = {"exchange": config.INDICES[idx].get("exchange", "?"),
                          "spot": spot, "tokens": toks[:a.per_index]}
            log.info("%s [%s] spot %.2f — probing %d leg(s)", idx,
                     picks[idx]["exchange"], spot, len(picks[idx]["tokens"]))
        except Exception as e:                             # noqa: BLE001
            log.warning("%s: could not resolve a chain (%s) — skipped", idx, e)

    all_toks = [t for p in picks.values() for t, _ in p["tokens"]]
    if not all_toks:
        log.error("no tokens resolved. If the chain lookups above failed with "
                  "a KeyError the fault is this tool, not the market; if they "
                  "timed out, re-run during market hours.")
        return
    owner = {t: idx for idx, p in picks.items() for t, _ in p["tokens"]}

    stats: dict = defaultdict(lambda: {"ticks": 0, "has_depth": 0,
                                       "two_sided": 0, "bid_only": 0,
                                       "ask_only": 0, "neither": 0,
                                       "sample": None})
    kws = KiteTicker(api_key, token)

    def on_ticks(ws, ticks):
        for tk in ticks:
            idx = owner.get(int(tk.get("instrument_token", 0)))
            if idx is None:
                continue
            s = stats[idx]
            s["ticks"] += 1
            if s["sample"] is None:
                s["sample"] = sorted(tk.keys())
            d = tk.get("depth") or {}
            if d:
                s["has_depth"] += 1
            b = (d.get("buy") or [{}])[0].get("price") or 0
            k = (d.get("sell") or [{}])[0].get("price") or 0
            if b > 0 and k > 0:
                s["two_sided"] += 1
            elif b > 0:
                s["bid_only"] += 1
            elif k > 0:
                s["ask_only"] += 1
            else:
                s["neither"] += 1

    def on_connect(ws, response):
        ws.subscribe(all_toks)
        ws.set_mode(ws.MODE_FULL, all_toks)
        log.info("subscribed %d token(s) in MODE_FULL — listening %ds...",
                 len(all_toks), a.seconds)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.connect(threaded=True)
    time.sleep(a.seconds)
    try:
        kws.close()
    except Exception:                                      # noqa: BLE001
        pass

    log.info("")
    log.info("%-9s %-5s %7s %10s %10s %9s %9s", "index", "exch", "ticks",
             "has depth", "2-sided", "bid only", "ask only")
    log.info("%s", "-" * 66)
    verdict = {}
    for idx, p in picks.items():
        s = stats[idx]
        n = max(s["ticks"], 1)
        log.info("%-9s %-5s %7d %9.0f%% %9.0f%% %8.0f%% %8.0f%%", idx,
                 p["exchange"], s["ticks"], 100 * s["has_depth"] / n,
                 100 * s["two_sided"] / n, 100 * s["bid_only"] / n,
                 100 * s["ask_only"] / n)
        verdict[idx] = (s["ticks"], s["two_sided"])
    log.info("")
    for idx, p in picks.items():
        s = stats[idx]
        if s["sample"]:
            log.info("%s sample tick fields: %s", idx, ", ".join(s["sample"]))

    nfo = [v for i, v in verdict.items() if i == "NIFTY"]
    bfo = [v for i, v in verdict.items() if i == "SENSEX"]
    log.info("")
    if nfo and bfo and nfo[0][0] and bfo[0][0]:
        n_ok = nfo[0][1] / max(nfo[0][0], 1)
        b_ok = bfo[0][1] / max(bfo[0][0], 1)
        if n_ok > 0.5 and b_ok < 0.05:
            log.warning("VERDICT: NFO delivers 2-sided depth (%.0f%% of "
                        "ticks); BFO delivers essentially none (%.0f%%). "
                        "SENSEX is therefore STRUCTURALLY UNFILLABLE on this "
                        "feed — fills price from the ring, and the ring has "
                        "no book for it. Options: (a) raise it with Zerodha "
                        "as a BFO depth entitlement question, (b) drop SENSEX "
                        "from TRADABLE so the cascade sample stops being "
                        "diluted by signals that can never fill, or (c) keep "
                        "harvesting SENSEX for research only. Do NOT "
                        "synthesise fills from LTP — that manufactures an "
                        "edge that does not exist.", 100 * n_ok, 100 * b_ok)
        elif b_ok >= 0.05:
            log.info("VERDICT: BFO DOES deliver 2-sided depth (%.0f%% of "
                     "ticks). The historical zeros are then NOT a feed "
                     "limitation — look at subscription timing or strike "
                     "coverage instead (PRUNE_STEPS now 8).", 100 * b_ok)
        else:
            log.info("VERDICT: inconclusive — too few ticks. Re-run during "
                     "active market hours with --seconds 120.")
    else:
        log.info("VERDICT: inconclusive — one side received no ticks at all. "
                 "Re-run during market hours.")


if __name__ == "__main__":
    main()