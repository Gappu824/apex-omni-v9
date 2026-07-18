"""
APEX OMNI v9.7.1 — EVENING MCX CAPTURE VALIDATION
Proves the full evening chain: per-process supervisor windows; the windowed
vault loader (equity path untouched, blind to evening; commodity path sees it);
the forge labels EVENING signals; serving t_frac matches the forge window.
  python tools/evening_capture_validate.py
"""
import sys, os, sqlite3, random, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"; _f=0
def check(n,c,d=""):
    global _f
    print(f"  [{PASS if c else FAIL}] {n}"+(f" — {d}" if d else ""))
    if not c: _f+=1

# ---- 1. supervisor per-process windows ----
import supervisor as S
def sod(h,m): return h*3600+m*60
eq, mcx = S._equity_window(), S._mcx_window()
w = {p: S._window_for(p) for p in S.PROCS}
check("harvester+commodity on MCX window; equity procs on equity window",
      w["data_harvester_v9.py"]==mcx==w["apex_commodity_main.py"]
      and w["apex_main_v9.py"]==eq==w["macro_gex_v9.py"], f"mcx={mcx}")
check("16:30 evening: only harvester+commodity active",
      S._active(sod(16,30),mcx) and not S._active(sod(16,30),eq))
_prev = config.EVENING_CAPTURE_ENABLED
config.EVENING_CAPTURE_ENABLED = False
check("master switch OFF reverts all to equity window",
      all(S._window_for(p)==eq for p in S.PROCS))
config.EVENING_CAPTURE_ENABLED = _prev

# ---- 2. synthetic vault: MORNING (10:00) + EVENING (20:00) tick blocks ----
db = Path(config.DB_PATH)
if db.exists(): os.remove(db)
con = sqlite3.connect(db); cu = con.cursor()
cu.execute("CREATE TABLE ticks_v9 (token INT, ts_ms INT, ts_local_ms INT, ltp REAL,"
           " bid REAL, ask REAL, bid_qty INT, ask_qty INT, vol_delta REAL, oi REAL, iceberg INT)")
cu.execute("CREATE TABLE spot_tokens (snap_date TEXT, name TEXT, token INT, PRIMARY KEY (snap_date,name))")
from core.instruments import SNAP_SCHEMA
cu.executescript(SNAP_SCHEMA)
day = dt.date.today(); ds = day.isoformat()
FUT=900001; TOK={"atm_ce":910001,"atm_pe":910002,"otm_ce":910003,"otm_pe":910004}
cu.execute("INSERT INTO spot_tokens VALUES (?,?,?)",(ds,"CRUDEOIL",FUT))
exp = day + dt.timedelta(days=20)
cu.executemany("INSERT OR REPLACE INTO instrument_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)",
 [(ds,FUT,"CRUDEFUT","CRUDEOIL",str(exp),0.0,"FUT",100,50.0,"MCX"),
  (ds,TOK["atm_ce"],"C6000CE","CRUDEOIL",str(exp),6000.0,"CE",100,50.0,"MCX"),
  (ds,TOK["atm_pe"],"C6000PE","CRUDEOIL",str(exp),6000.0,"PE",100,50.0,"MCX"),
  (ds,TOK["otm_ce"],"C6050CE","CRUDEOIL",str(exp),6050.0,"CE",100,50.0,"MCX"),
  (ds,TOK["otm_pe"],"C5950PE","CRUDEOIL",str(exp),5950.0,"PE",100,50.0,"MCX")])
random.seed(5)
def block(start_hm, secs, rows):
    base = int(dt.datetime.combine(day, dt.time(*start_hm)).timestamp())
    spot, ce, pe = 6000.0, 4.0, 3.8
    for i in range(secs):
        drift = 0.35 if (i//500)%2==0 else -0.35
        spot += drift + random.gauss(0,0.6)
        ce = max(ce + drift*0.05 + random.gauss(0,0.05), 0.5)
        pe = max(pe - drift*0.05 + random.gauss(0,0.05), 0.5)
        ts=(base+i)*1000; up=drift>0
        def r(tok,px,bq,aq,oi): rows.append((tok,ts,ts,px,px-0.05,px+0.05,bq,aq,random.uniform(0,20),oi,0))
        r(FUT,spot,30,30,0); r(TOK["atm_ce"],ce,90 if up else 15,15 if up else 90,5000)
        r(TOK["atm_pe"],pe,15 if up else 90,90 if up else 15,5000)
        r(TOK["otm_ce"],ce*0.6,40,40,3000); r(TOK["otm_pe"],pe*0.6,40,40,3000)
rows=[]; block((10,0),2500,rows); block((20,0),2500,rows)   # morning + EIA-hour
cu.executemany("INSERT INTO ticks_v9 VALUES (?,?,?,?,?,?,?,?,?,?,?)",rows)
con.commit()
print(f"vault: {len(rows)} ticks in two blocks (10:00 and 20:00)")

# ---- 3. loaders: equity blind to evening; commodity sees both ----
from simulation.replay_real_day import load_day, load_day_window
from simulation.scenario_engine import N as N_EQ
eq_loaded = load_day(con, ds, "CRUDEOIL")
import nightly_commodity_forge as CF
co_loaded = load_day_window(con, ds, "CRUDEOIL", CF._C_T0, CF.N)
def _count(loaded):
    if not loaded: return 0,0
    _s, by_sec, ti, bidA, _a = loaded
    k = ti.get(FUT); 
    if k is None: return 0,0
    v = ~np.isnan(bidA[k]) & (np.arange(bidA.shape[1])>=0)
    raw = sorted(by_sec)
    return len(raw), (raw[-1] if raw else 0)
n_eq, last_eq = _count(eq_loaded); n_co, last_co = _count(co_loaded)
check("equity load_day sees ONLY the morning (unchanged path)",
      n_eq>0 and last_eq < N_EQ, f"secs={n_eq}, last_t={last_eq} < N_eq={N_EQ}")
check("commodity loader sees morning AND evening",
      n_co > n_eq and last_co > N_EQ, f"secs={n_co}, last_t={last_co}")

# ---- 4. forge labels EVENING signals ----
config.COMMODITY_ENTRY_CONVICTION = 0.05
config.COMMODITY_FORGE_COOLDOWN_S = 60
X,Y,W = CF.gen_samples(con, ds)
ev = [x for x in X if x[3*config.FEATURES_PER_NODE] > (sod(15,30)-CF._C_T0)/CF.N]
check("gen_samples labels signals in the EVENING session", len(X)>=8 and len(ev)>=3,
      f"total={len(X)}, evening={len(ev)}")
if ev:
    check("evening t_frac inside (0,1] of the commodity window",
          all(0 < x[3*config.FEATURES_PER_NODE] <= 1.0 for x in ev))

# ---- 5. serving t_frac parity at 20:00 ----
from core.commodity_brain import CommodityBrain
b = CommodityBrain(); b.meta = {"engine":"gbm"}   # force the t_frac path
b._spot_hist["CRUDEOIL"].extend([6000+i*0.2 for i in range(60)])
import core.meta_gbm as MG
_expect = (sod(20,0)-CF._C_T0)/CF.N
_got = {}
_orig = MG.score_vec
MG.score_vec = lambda m,x: _got.update(t=float(x[3*config.FEATURES_PER_NODE])) or 0.6
wp = b._meta_wp("CRUDEOIL", np.zeros((5,config.FEATURES_PER_NODE),np.float32),
                "CE", dt.datetime(2026,7,20,20,0))
MG.score_vec = _orig
check("serving t_frac at 20:00 equals the forge's window fraction",
      wp==0.6 and abs(_got.get("t",-1)-_expect)<1e-6,
      f"serving={_got.get('t'):.4f} forge={_expect:.4f}")

os.remove(db)
print("\n"+"="*58)
if _f: print(f"  {FAIL}: {_f} check(s) failed"); sys.exit(1)
print(f"  {PASS}: evening MCX capture validated end-to-end")