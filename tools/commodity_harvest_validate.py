"""
APEX OMNI v9.7.1 — COMMODITY HARVEST WIRING VALIDATION
Proves the MCX commodity harvest path without a live Kite: a mock instrument
dump exercises the mapper (chain, hierarchy, front-month future) and the config
universe. Equity behaviour must be unchanged; commodity behaviour must be
correct. Trading must remain OFF for all commodities.

  python tools/commodity_harvest_validate.py
"""
import sys, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config, os
import core.instruments as I

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"; _f=0
def check(n,c,d=""):
    global _f
    print(f"  [{PASS if c else FAIL}] {n}"+(f" — {d}" if d else ""))
    if not c: _f+=1

today=dt.date.today(); e_eq=today+dt.timedelta(days=3); e_co=today+dt.timedelta(days=20)
class MockKite:
    def instruments(self,exch):
        if exch=="NFO":
            return [{"name":"NIFTY","expiry":e_eq,"strike":k,"instrument_type":it,
                     "instrument_token":1000+k+(1 if it=="CE" else 2),
                     "tradingsymbol":f"NIFTY{k}{it}","lot_size":65}
                    for k in (24000,24050,24100,24150) for it in ("CE","PE")]
        if exch=="MCX":
            out=[{"name":"CRUDEOIL","expiry":e_co,"strike":k,"instrument_type":it,
                  "instrument_token":5000+k+(1 if it=="CE" else 2),
                  "tradingsymbol":f"CRUDEOIL{k}{it}","lot_size":100}
                 for k in (5800,5850,5900,5950,6000) for it in ("CE","PE")]
            out+=[{"name":"CRUDEOIL","expiry":e_co,"strike":0,"instrument_type":"FUT",
                   "instrument_token":9001,"tradingsymbol":"CRUDEOILFUT1","lot_size":100},
                  {"name":"CRUDEOIL","expiry":e_co+dt.timedelta(days=30),"strike":0,
                   "instrument_type":"FUT","instrument_token":9002,
                   "tradingsymbol":"CRUDEOILFUT2","lot_size":100}]
            return out
        return []
    def ltp(self,syms): return {}

for f in config.STATE_DIR.glob("instruments_*.pkl"): os.remove(f)
for f in config.STATE_DIR.glob("commodity_futs_*.pkl"): os.remove(f)
m=I.LiveMapper(MockKite())

print("\n=== config universe ===")
check("commodities declared", set(config.COMMODITIES)>= {"CRUDEOIL","NATURALGAS","GOLD","SILVER","COPPER"})
check("COMMODITY_TRADABLE is EMPTY (no trading)", config.COMMODITY_TRADABLE==[])
check("equity TRADABLE unchanged", config.TRADABLE==["NIFTY","SENSEX"])
check("CONFIG_HASH unchanged", config.CONFIG_HASH=="47689d19a5", config.CONFIG_HASH)

print("\n=== equity path unchanged ===")
ch=m.chain("NIFTY",24075)
check("NIFTY chain builds", ch is not None and len(ch["legs"])==4)
check("NIFTY step 50 / weekly True", ch["step"]==50.0 and ch["is_weekly"] is True)

print("\n=== commodity path correct ===")
ch=m.chain("CRUDEOIL",5920)
check("CRUDEOIL chain builds", ch is not None and len(ch["legs"])==4)
check("CRUDEOIL step auto-inferred (50)", ch["step"]==50.0, f"step={ch['step']}")
check("CRUDEOIL monthly (weekly False)", ch["is_weekly"] is False)
check("CRUDEOIL lot from dump (100)", ch["lot"]==100)
fut=m.commodity_futures.get("CRUDEOIL")
check("front-month future resolved (nearer expiry)", fut and fut["token"]==9001, str(fut))
h=m.hierarchy("CRUDEOIL",5920,"CE",depth=3)
check("CE ladder walks up from ATM", [r["strike"] for r in h]==[5900.0,5950.0,6000.0])

print("\n"+"="*54)
if _f: print(f"  {FAIL}: {_f} check(s) failed"); sys.exit(1)
print(f"  {PASS}: commodity harvest wiring validated (trading OFF)")