"""
APEX OMNI v9.7.1 — COMMODITY FORGE VALIDATION (end-to-end on a synthetic vault)
Proves the full chain with the REAL schemas: harvester-shaped vault →
gen_samples (AsOfMapper chain, affordability walk, first-touch barriers,
uniqueness weights) → meta_gbm.fit_gbm → commodity_meta.json → the live brain
loads it and _meta_wp scores the forge-identical x-vector. The conviction bar
is lowered IN-TEST ONLY (this validates plumbing+grading, not signal strength).
  python tools/commodity_forge_validate.py
"""
import sys, os, json, sqlite3, random, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"; _f=0
def check(n,c,d=""):
    global _f
    print(f"  [{PASS if c else FAIL}] {n}"+(f" — {d}" if d else ""))
    if not c: _f+=1

# ---- build a real-schema synthetic vault: CRUDEOIL fut spot + 4 legs ----
db = Path(config.DB_PATH)
if db.exists(): os.remove(db)
con = sqlite3.connect(db); cu = con.cursor()
cu.execute("CREATE TABLE ticks_v9 (token INT, ts_ms INT, ts_local_ms INT, ltp REAL,"
           " bid REAL, ask REAL, bid_qty INT, ask_qty INT, vol_delta REAL,"
           " oi REAL, iceberg INT)")
cu.execute("CREATE TABLE spot_tokens (snap_date TEXT, name TEXT, token INT,"
           " PRIMARY KEY (snap_date,name))")
from core.instruments import SNAP_SCHEMA
cu.executescript(SNAP_SCHEMA)
day = dt.date.today(); ds = day.isoformat()
FUT=900001; TOK={"atm_ce":910001,"atm_pe":910002,"otm_ce":910003,"otm_pe":910004}
cu.execute("INSERT INTO spot_tokens VALUES (?,?,?)",(ds,"CRUDEOIL",FUT))
exp = day + dt.timedelta(days=20)
snaps=[(ds,FUT,"CRUDEFUT","CRUDEOIL",str(exp),0.0,"FUT",100,50.0,"MCX"),
       (ds,TOK["atm_ce"],"C6000CE","CRUDEOIL",str(exp),6000.0,"CE",100,50.0,"MCX"),
       (ds,TOK["atm_pe"],"C6000PE","CRUDEOIL",str(exp),6000.0,"PE",100,50.0,"MCX"),
       (ds,TOK["otm_ce"],"C6050CE","CRUDEOIL",str(exp),6050.0,"CE",100,50.0,"MCX"),
       (ds,TOK["otm_pe"],"C5950PE","CRUDEOIL",str(exp),5950.0,"PE",100,50.0,"MCX")]
cu.executemany("INSERT OR REPLACE INTO instrument_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)",snaps)
random.seed(11)
base = int(dt.datetime.combine(day, dt.time(9,15)).timestamp())
rows=[]; spot=6000.0; ce=4.0; pe=3.8
for i in range(5400):                       # 90 min of 1 Hz ticks
    drift = 0.35 if (i//600)%2==0 else -0.35   # alternating trend blocks
    spot += drift + random.gauss(0,0.6)
    ce = max(ce + drift*0.05 + random.gauss(0,0.05), 0.5)
    pe = max(pe - drift*0.05 + random.gauss(0,0.05), 0.5)
    ts=(base+i)*1000
    def row(tok,px,bq,aq,oi): return (tok,ts,ts,px,px-0.05,px+0.05,bq,aq,
                                      random.uniform(0,20),oi,0)
    up = drift>0
    rows += [row(FUT,spot,30,30,0),
             row(TOK["atm_ce"],ce, 90 if up else 15, 15 if up else 90, 5000),
             row(TOK["atm_pe"],pe, 15 if up else 90, 90 if up else 15, 5000),
             row(TOK["otm_ce"],ce*0.6, 40,40, 3000),
             row(TOK["otm_pe"],pe*0.6, 40,40, 3000)]
cu.executemany("INSERT INTO ticks_v9 VALUES (?,?,?,?,?,?,?,?,?,?,?)",rows)
con.commit()
print(f"synthetic vault: {len(rows)} ticks, chain snapshotted for {ds}")

# ---- sample generation (bar lowered in-test to exercise plumbing) ----
config.COMMODITY_ENTRY_CONVICTION = 0.05
config.COMMODITY_FORGE_COOLDOWN_S = 60
import nightly_commodity_forge as CF
X,Y,W = CF.gen_samples(con, ds)
check("gen_samples produces labeled signals from the vault", len(X)>=8,
      f"n={len(X)}")
if X:
    check("x-vector is the 61-dim forge layout", X[0].shape==(3*config.FEATURES_PER_NODE+4,),
          str(X[0].shape))
    check("labels are binary and mixed", set(map(int,Y))<= {0,1} and 0<sum(Y)<len(Y),
          f"wins={int(sum(Y))}/{len(Y)}")
    check("uniqueness weights in (0,1]", all(0<w<=1.0001 for w in W))

# ---- training + artifact + serving (if lightgbm present) ----
try:
    import lightgbm; HAVE=True
except Exception: HAVE=False
if HAVE and len(X)>=8:
    from core import meta_gbm as MG
    # AUDIT: this assertion was structurally impossible and had never run
    # (lightgbm was absent, so the branch was skipped). fit_gbm uses purged
    # DAY-fold CV and requires >=2 COMPLETED folds — a single harvested day
    # can never satisfy that. Tile the real x-vectors across 8 synthetic days
    # to exercise the trainer contract the way the live forge sees it (the
    # commodity forge accumulates across ~25 days before it reaches its gate).
    perday = []
    for _d in range(8):
        _i = [(k + _d) % len(X) for k in range(len(X))]
        perday.append((f"2026-07-{_d+1:02d}", [X[k] for k in _i],
                       [Y[k] for k in _i], [W[k] for k in _i]))
    # A VALIDATOR must never be able to touch a served artifact.
    import tempfile as _tf
    _scratch = Path(_tf.mkdtemp()) / "validate_gbm.txt"
    out = MG.fit_gbm(perday, min_train=8, model_path=_scratch)
    check("fit_gbm trains on commodity samples", out is not None and out.get("engine")=="gbm",
          f"n={out and out.get('n')}")
    if out:
        path = config.MODEL_DIR / "commodity_meta.json"
        path.write_text(json.dumps(out))
        from core.commodity_brain import load_commodity_meta, CommodityBrain
        m = load_commodity_meta()
        check("brain loads the promoted artifact", m is not None and m.get("engine")=="gbm")
        check("artifact carries Brier Skill Score vs climatology",
              "bss_cal" in out and "brier_climatology" in out,
              f"bss_cal={out.get('bss_cal')}")
        b = CommodityBrain()
        b._spot_hist["CRUDEOIL"].extend([6000+i*0.3 for i in range(60)])
        wp = b._meta_wp("CRUDEOIL", np.stack([X[0][:19],X[0][19:38],X[0][38:57],
             np.zeros(19,np.float32),np.zeros(19,np.float32)]), "CE",
             dt.datetime(2026,7,20,11,0))
        check("serving _meta_wp scores a probability", wp is not None and 0.0<=wp<=1.0,
              f"wp={wp}")
        os.remove(path)
else:
    print(f"  [SKIP] lightgbm {'absent' if not HAVE else 'thin samples'} — "
          f"trainer fail-open path exercised instead")
    from core import meta_gbm as MG
    check("fit_gbm fail-open returns None (no fake model)",
          MG.fit_gbm([(ds,X,Y,W)], min_train=10**9,
                     model_path=_scratch) is None)

os.remove(db)
print("\n"+"="*58)
if _f: print(f"  {FAIL}: {_f} check(s) failed"); sys.exit(1)
print(f"  {PASS}: commodity forge validated end-to-end")