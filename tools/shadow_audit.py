"""
SHADOW AUDIT — demonstrate-broken, then demonstrate-correct
============================================================
    python tools/shadow_audit.py          # exit code = failures

The v9.9.13 shadow work fixed six defects that each silently DELETED or
FABRICATED evidence rather than raising. Silent defects come back. This
harness is the standing guard: every check below reproduces the exact
condition that was wrong and asserts the current code gets it right.

Part 1  REGRESSION GUARDS — the six defects, one check each.
Part 2  DRIVER EQUIVALENCE — the live shadow book and the nightly replay
        must produce bit-identical exits. This is the check that matters
        most: the moment they diverge, every nightly verdict describes a
        policy that never ran, which is precisely the train/serve skew
        that made meta_gbm.py emit a constant in production while
        scoring well offline.

Everything here exercises the REAL modules. Only the tape is synthetic —
a ledger is data, a tick table is data. Nothing under test is mocked.
"""
from __future__ import annotations

import csv
import datetime as dt
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config                                              # noqa: E402

FAILURES: list[str] = []


def inconclusive(name: str, detail: str = "") -> None:
    """A check that could not run here. NOT a pass. Printed loudly so a
    green run never implies coverage the host could not provide."""
    print(f"    SKIP  {name}\n          INCONCLUSIVE ON THIS HOST: {detail}")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"    {'PASS' if ok else 'FAIL'}  {name}"
          + (f"\n          {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _ledger(rows, cols):
    p = Path(tempfile.mkdtemp()) / "execution_ledger_v9.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


# ====================================================== PART 1: GUARDS
def guard_f_token_column():
    print("\n[F] the ledger must identify the instrument by TOKEN")
    from core.position_manager import LEDGER_FIELDS
    check("LEDGER_FIELDS carries 'token'", "token" in LEDGER_FIELDS,
          "the tick vault is keyed by token; a symbol cannot look up a "
          "path. Without this column every study reads ti.get(0) and "
          "silently skips every trade.")
    check("LEDGER_FIELDS carries 'shadow_id'", "shadow_id" in LEDGER_FIELDS)

    # A column that exists but is never written reproduces defect F exactly:
    # every study reads "" -> 0 -> ti.get(0) misses -> every trade skipped,
    # silently. Schema presence is necessary and nowhere near sufficient.
    import ast as _ast
    import inspect as _inspect
    import core.position_manager as _pm
    src = Path(_inspect.getfile(_pm)).read_text(encoding="utf-8")
    tree = _ast.parse(src)
    emitters = {}
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "_log"):
            continue
        kw = {k.arg for k in node.keywords if k.arg}
        ev = None
        for k in node.keywords:
            if k.arg == "event" and isinstance(k.value, _ast.Constant):
                ev = k.value.value
        if ev in ("BUY_FILL", "SELL_FILL"):
            emitters[ev] = kw
    for ev in ("BUY_FILL", "SELL_FILL"):
        check(f"{ev} actually EMITS token=", "token" in emitters.get(ev, ()),
              f"_log(event={ev!r}) kwargs = "
              f"{sorted(emitters.get(ev, ())) or 'CALL NOT FOUND'}")

    # ...and prove the writer does not silently drop it on the way out.
    import csv as _csv
    import tempfile as _tf
    from types import SimpleNamespace as _NS
    lp = Path(_tf.mkdtemp()) / "l.csv"
    with lp.open("w", newline="", encoding="utf-8") as fh:
        _csv.DictWriter(fh, LEDGER_FIELDS).writeheader()
    fake = _NS(ledger=lp, index="NIFTY", events=[])
    try:
        _pm.PositionManager._log(fake, ts="1", event="BUY_FILL",
                                 symbol="X", token=123456, qty=65)
        row = next(_csv.DictReader(lp.open(encoding="utf-8")))
        got = str(row.get("token") or "")
    except Exception as e:                                 # noqa: BLE001
        got = f"<{type(e).__name__}: {e}>"
    check("_log() writes token through to the ledger row", got == "123456",
          f"token column read back as {got!r}")


def guard_a_session_window():
    print("\n[A] the session window must be DATE-AWARE, not hard-coded")
    from simulation.session_paths import window_for
    pre = window_for("2026-07-15", "NIFTY")
    post = window_for("2026-08-05", "NIFTY")
    check("pre-reform day still closes 15:30", pre.close_hm == "15:30",
          f"got {pre.close_hm} — replaying history under a 15:40 close "
          f"would fabricate ten minutes that did not exist")
    check("post-reform day closes 15:40", post.close_hm == "15:40",
          f"got {post.close_hm}")
    t_1536 = post.sod_to_t(15 * 3600 + 36 * 60)
    check("a 15:36 post-auction entry is INSIDE the window",
          post.contains(t_1536),
          f"t0={t_1536}, window n={post.n}")
    check("the same entry was OUTSIDE the old 22500-column array",
          t_1536 >= 22500,
          f"t0={t_1536} vs legacy N=22500 — this is the defect, preserved "
          f"as evidence")


def guard_b_short_legs():
    print("\n[B] short legs and butterflies must reconstruct")
    from core.position_manager import LEDGER_FIELDS
    from core.trade_reconstruct import reconstruct
    p = _ledger([
        dict(ts=1, event="BUY_FILL", index="N", symbol="A", qty=75,
             price=100, token=111),
        dict(ts=2, event="SELL_FILL", index="N", symbol="A", qty=75,
             price=120, pnl=1500, token=111),
        dict(ts=3, event="SELL_FILL", index="N", symbol="B", qty=150,
             price=90, token=222),
        dict(ts=4, event="BUY_FILL", index="N", symbol="B", qty=150,
             price=70, pnl=3000, token=222),
    ], LEDGER_FIELDS)
    r = reconstruct(p)
    s = r.summary()
    check("both round trips reconstruct", s["trades"] == 2,
          f"got {s['trades']} (the old pair-by-symbol shape returned 1)")
    check("one is recognised as SHORT", s["short"] == 1,
          f"long={s['long']} short={s['short']}")
    check("no fill was orphaned", s["orphans"] == 0)

    fly = _ledger([
        dict(ts=1, event="FLY_OPEN", index="N",
             symbol="N24000CE+N24100CEx2+N24200CE", qty=75, price=12.5),
        dict(ts=2, event="FLY_CLOSE", index="N",
             symbol="N24000CE+N24100CEx2+N24200CE", qty=75, price=18.0,
             pnl=412.5),
    ], LEDGER_FIELDS)
    rf = reconstruct(fly)
    ok = (len(rf.trades) == 1 and rf.trades[0].kind == "FLY"
          and len(rf.trades[0].legs) == 3)
    sides = [l.side for l in rf.trades[0].legs] if rf.trades else []
    check("a butterfly is ONE trade with three signed legs", ok,
          f"legs={[(l.symbol, l.side, l.mult) for l in rf.trades[0].legs]}"
          if rf.trades else "no trade")
    check("the doubled body is the SHORT leg", sides == [1, -1, 1],
          f"sides={sides}")


def guard_c_fifo():
    print("\n[C] re-entries must stack and partials must split")
    from core.position_manager import LEDGER_FIELDS
    from core.trade_reconstruct import reconstruct
    p = _ledger([
        dict(ts=1, event="BUY_FILL", index="N", symbol="X", qty=15,
             price=200, token=1),
        dict(ts=2, event="BUY_FILL", index="N", symbol="X", qty=15,
             price=260, token=1),
        dict(ts=3, event="SELL_FILL", index="N", symbol="X", qty=15,
             price=240, pnl=600, token=1),
        dict(ts=4, event="SELL_FILL", index="N", symbol="X", qty=15,
             price=250, pnl=-150, token=1),
    ], LEDGER_FIELDS)
    tr = reconstruct(p).trades
    entries = sorted(t.entry_px for t in tr)
    check("both entries survive (the @200 is no longer overwritten)",
          entries == [200.0, 260.0], f"entries={entries}")
    check("FIFO pairs oldest-first",
          any(t.entry_px == 200.0 and t.exit_px == 240.0 for t in tr))

    q = _ledger([
        dict(ts=1, event="BUY_FILL", index="N", symbol="Y", qty=3,
             price=100, token=2),
        dict(ts=2, event="SELL_FILL", index="N", symbol="Y", qty=1,
             price=110, pnl=10, token=2),
        dict(ts=3, event="SELL_FILL", index="N", symbol="Y", qty=2,
             price=90, pnl=-20, token=2),
    ], LEDGER_FIELDS)
    tq = reconstruct(q).trades
    check("a 3-lot entry closed 1+2 yields two apportioned trades",
          sorted(t.qty for t in tq) == [1, 2],
          f"qtys={[t.qty for t in tq]}, pnls={[t.realized_pnl for t in tq]}")
    check("realised P&L is apportioned, never duplicated",
          abs(sum(t.realized_pnl for t in tq) - (-10.0)) < 1e-6,
          f"sum={sum(t.realized_pnl for t in tq)} (ledger total was -10)")


def guard_e_staleness():
    print("\n[E] a dead feed must read as DEAD, not as a flat price")
    from simulation.session_paths import load_session_paths
    day = "2026-07-15"
    TOK, SPOT = 999001, 26000
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE ticks_v9 (ts_ms INTEGER, ts_local_ms INTEGER, "
        "token INTEGER, ltp REAL, bid REAL, ask REAL, bid_qty REAL, "
        "ask_qty REAL, vol_delta REAL, oi REAL, iceberg INTEGER);"
        "CREATE TABLE spot_tokens (snap_date TEXT, name TEXT, token INT);")
    con.execute("INSERT INTO spot_tokens VALUES (?,?,?)", (day, "NIFTY",
                                                           SPOT))
    midnight = dt.datetime.combine(dt.date.fromisoformat(day),
                                   dt.time(0, 0)).timestamp()
    rows = []
    for i in range(2400):                       # 09:30 -> 10:10 only
        sod = 9 * 3600 + 30 * 60 + i
        px = 100.0 + 8.0 * np.sin(i / 400.0)
        ms = int((midnight + sod) * 1000)
        rows.append((ms, ms, TOK, px, px, px + 1.0, 1, 1, 0, 0, 0))
    con.executemany("INSERT INTO ticks_v9 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    rows)
    ps = load_session_paths(con, day, "NIFTY", tokens={TOK})
    path = ps.path(TOK)
    fresh = ps.fresh_mask(TOK)
    last_real = ps.window.sod_to_t(9 * 3600 + 30 * 60 + 2399)
    carry = int(getattr(config, "SHADOW_MAX_STALE_S", 120))
    tail = path[last_real + carry + 5:]
    check("the path goes NaN after the bounded carry window",
          bool(np.all(np.isnan(tail))),
          f"{int(np.isfinite(tail).sum())} finite sample(s) survived past "
          f"{carry}s of carry — a flat line would show {tail.size}")
    check("the freshness mask marks the dead region",
          not bool(fresh[last_real + carry + 5:].any()))
    check("coverage reports the truth", ps.coverage(TOK) < 0.25,
          f"coverage={ps.coverage(TOK):.3f} — the old loader reported a "
          f"perfectly finite path for the whole session")
    check("died_at names when the feed stopped",
          ps.died_at(TOK) is not None and
          abs(ps.died_at(TOK) - last_real) <= 1,
          f"died_at={ps.window.t_to_hm(ps.died_at(TOK))}")


def guard_tz_independence():
    """The host runs IST; CI and containers usually run UTC. A loader that
    derives seconds-of-day with `epoch % 86400` is silently correct on UTC
    and silently returns None on every session under IST — which is how a
    study reports 'no ticks in the vault' all night without raising."""
    print("\n[TZ] the path loader must not assume the host is UTC")
    import os
    import subprocess
    import sys as _sys
    probe = (
        "import sys,sqlite3,datetime as dt;sys.path.insert(0,%r);"
        "import config;"
        "from simulation.session_paths import load_session_paths;"
        "day='2026-07-15';TOK=999001;con=sqlite3.connect(':memory:');"
        "con.executescript('CREATE TABLE ticks_v9 (ts_ms INTEGER,"
        "ts_local_ms INTEGER,token INTEGER,ltp REAL,bid REAL,ask REAL,"
        "bid_qty REAL,ask_qty REAL,vol_delta REAL,oi REAL,iceberg INTEGER);');"
        "mid=dt.datetime.combine(dt.date.fromisoformat(day),"
        "dt.time(0,0)).timestamp();"
        "rows=[(int((mid+9*3600+30*60+i)*1000),)*2+(TOK,100.,100.,101.,"
        "1,1,0,0,0) for i in range(2400)];"
        "con.executemany('INSERT INTO ticks_v9 VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?)',rows);"
        "ps=load_session_paths(con,day,'NIFTY',tokens={TOK});"
        "off=dt.datetime.now().astimezone().utcoffset();"
        "print(('NONE' if ps is None else "
        "ps.window.t_to_hm(ps.died_at(TOK)))+'@'+str(off))"
    ) % str(Path(__file__).resolve().parents[1])
    got = {}
    for tz in ("UTC", "Asia/Kolkata", "America/New_York"):
        env = dict(os.environ)
        env["TZ"] = tz
        try:
            r = subprocess.run([_sys.executable, "-c", probe], env=env,
                               capture_output=True, text=True, timeout=120)
            got[tz] = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
        except Exception as e:                             # noqa: BLE001
            got[tz] = f"<{e}>"
    # SELF-VALIDATION. Python on Windows does not honour a POSIX zone name
    # in TZ (there is no tzset, and the CRT understands only a limited
    # "EST5EDT" form), so forking with TZ="Asia/Kolkata" can leave all three
    # children in the SAME zone. This check would then compare three
    # identical runs, pass, and report timezone coverage it never had — on
    # the exact host where the UTC-seconds-of-day bug lived. So the probe
    # reports its own UTC offset and we refuse to call it a pass unless the
    # offsets actually differed.
    offsets = {tz: v.split("@")[-1] for tz, v in got.items()}
    answers = {tz: v.split("@")[0] for tz, v in got.items()}
    varied = len(set(offsets.values())) > 1
    if not varied:
        inconclusive(
            "the loader returns a path in every timezone",
            f"TZ had no effect — all three children ran at offset "
            f"{next(iter(offsets.values()))} ({offsets}). This host cannot "
            f"vary the zone from the environment, so this check proves "
            f"NOTHING here. The underlying property is still covered by [E] "
            f"under the host's own zone: that fixture builds its ticks from "
            f"local midnight and the loader recovers them, which is exactly "
            f"the anchoring the fix introduced. For real cross-zone "
            f"coverage run this under WSL, Linux CI, or a container.")
        return
    check("the loader returns a path in every timezone",
          all(v == "10:09" for v in answers.values()),
          f"answers={answers} at offsets={offsets}")


def guard_d_loop_closed():
    print("\n[D] the measurement loop must be CLOSED")
    # A grep for the module NAME is not evidence of consumption: it matches
    # the module's own logger string, docstrings and comments. The earlier
    # form of this check reported four "consumers" of which exactly one was
    # real — and would have passed with that one deleted. Only an actual
    # import statement, parsed, counts.
    import ast as _ast
    root = Path(__file__).resolve().parents[1]
    importers, mentions = [], []
    for f in root.rglob("*.py"):
        if f.name == "shadow_audit.py" or f.name == "exit_policy_store.py":
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "exit_policy_store" not in src:
            continue
        rel = str(f.relative_to(root))
        real = False
        try:
            for node in _ast.walk(_ast.parse(src)):
                if isinstance(node, _ast.ImportFrom) and node.module and \
                        "exit_policy_store" in node.module:
                    real = True
                elif isinstance(node, _ast.ImportFrom) and \
                        any("exit_policy_store" in (a.name or "")
                            for a in node.names):
                    real = True
                elif isinstance(node, _ast.Import) and \
                        any("exit_policy_store" in a.name for a in node.names):
                    real = True
        except SyntaxError:
            pass
        (importers if real else mentions).append(rel)
    check("the verdict has a REAL importer (not a docstring mention)",
          len(importers) >= 1,
          f"importers={sorted(importers)} | name-only mentions "
          f"(NOT consumption)={sorted(mentions)}")
    from core import exit_policy_store as EPS
    name, body = EPS.active_policy()
    check("active_policy() answers without a promotion on disk",
          name == "as_traded", f"got '{name}'")
    check("promotion is gated, not automatic",
          EPS.promote({"ok": False, "n_trades": 0, "n_days": 0}) is False)


def guard_hash_stability():
    print("\n[HASH] measurement must never invalidate the feature world")
    excl = getattr(config, "_HASH_EXCLUDE", frozenset())
    shadow = [k for k in dir(config) if k.startswith("SHADOW_")]
    missing = [k for k in shadow if k not in excl]
    check("every SHADOW_* constant is in _HASH_EXCLUDE", not missing,
          f"missing: {missing} — a rotation here rebuilds every raw day "
          f"cache and re-runs the forge for a knob that cannot change a "
          f"decision (the 2026-07-29 12-hour evening)")


# ============================================== PART 2: DRIVER EQUIVALENCE
def equivalence():
    print("\n[EQ] the LIVE book and the NIGHTLY replay must agree exactly")
    import tempfile as _tf
    from core.exit_policies import (PolicySpec, TradeCtx, replay)
    from core.execution_engine import round_trip_costs

    config.SHADOW_LEDGER_PATH = Path(_tf.mkdtemp()) / "sl.csv"
    config.STATE_DIR = Path(_tf.mkdtemp())
    from core.shadow_book import ShadowBook

    rng = np.random.default_rng(3)
    n = 5400
    for trial, side in enumerate((+1, -1, +1)):
        # a path with a real peak, a real collapse, and a dead patch
        base = np.concatenate([
            np.linspace(100, 100 + 34 * side, 1500),
            np.linspace(100 + 34 * side, 100 - 22 * side, 2000),
            np.full(1900, 100 - 22 * side)]) + rng.normal(0, 0.2, n)
        base = np.maximum(base, 1.0)
        dead = np.zeros(n, bool)
        dead[3000:3400] = True          # the harvester pruned the leg here
        entry = float(base[0])

        # ---- offline driver
        path = base.copy()
        fresh = ~dead
        ctx = TradeCtx(entry=entry, qty=75, side=side, hold_budget_s=3600,
                       session_end_t=n - 1)
        off = {s.name: replay(s, ctx, path, round_trip_costs, fresh)
               for s in PolicySpec.family()}

        # ---- live driver: the same tape, one mark per second
        sb = ShadowBook("NIFTY")
        t0 = 1_800_000_000.0
        sid = sb.open_shadow("SYM", token=7, entry_px=entry, qty=75,
                             side=side, entry_ts=t0, hold_budget_s=3600)
        sb.open[sid].ctx.session_end_t = n - 1      # same bell as offline
        key = "bid" if side > 0 else "ask"
        for t in range(n):
            q = {} if dead[t] else {7: {key: float(base[t])}}
            sb.mark(q, now=t0 + t, force=True)
        live_states = (sb.open[sid].states if sid in sb.open
                       else sb.done[-1].states)

        diffs = []
        for name, o in off.items():
            st = live_states[name]
            if (abs(st.exit_px - o.exit_px) > 1e-9 or st.exit_t != o.exit_t
                    or st.exit_reason != o.reason):
                diffs.append(f"{name}: live=({st.exit_px:.4f},{st.exit_t},"
                             f"{st.exit_reason}) offline=({o.exit_px:.4f},"
                             f"{o.exit_t},{o.reason})")
        check(f"trial {trial + 1} (side={side:+d}): all "
              f"{len(off)} policies identical", not diffs,
              "\n          ".join(diffs[:4]))


def dead_feed_semantics():
    print("\n[DF] a policy must not be able to trigger on a price "
          "that does not exist")
    from core.exit_policies import PolicySpec, TradeCtx, replay
    from core.execution_engine import round_trip_costs
    n = 3600
    path = np.concatenate([np.linspace(100, 140, 600),
                           np.full(n - 600, np.nan)])
    fresh = np.zeros(n, bool)
    fresh[:600] = True
    ctx = TradeCtx(entry=100.0, qty=75, side=+1, hold_budget_s=3600,
                   session_end_t=n - 1)
    outs = {s.name: replay(s, ctx, path, round_trip_costs, fresh)
            for s in PolicySpec.family()}
    htc = outs["hold_to_close"]
    check("hold_to_close exits at the LAST REAL mark, not a carried one",
          abs(htc.exit_px - 140.0) < 0.5, f"exit_px={htc.exit_px:.2f}")
    check("...and says the feed died", "DEAD_FEED" in htc.reason,
          f"reason={htc.reason}")
    check("...and flags the exit as stale", htc.stale_exit)
    tr = outs["trail_20"]
    check("a trail cannot fire on the dead stretch",
          tr.exit_t < 600 or "DEAD_FEED" in tr.reason,
          f"trail_20 exited t={tr.exit_t} reason={tr.reason}")
    check("coverage is reported honestly", htc.coverage < 0.25,
          f"coverage={htc.coverage:.3f}")


# ======================================== PART 3: ENTRY-SIDE (v9.9.14)
def guard_capacity_is_the_constraint():
    print("\n[CAP] the BAR must not be able to change the trade COUNT")
    from core.entry_counterfactual import (BarSweep, Signal, bar_grid,
                                           capacity_note)
    import numpy as _np
    cap = capacity_note(21000)
    check("capacity arithmetic is exposed to every report",
          cap["max_trades_per_session"] <= 6,
          f"{cap['max_concurrent']} concurrent, {cap['slot_s']}s per slot "
          f"over {cap['entry_window_s']}s => at most "
          f"{cap['max_trades_per_session']} trade(s)/session")
    rng = _np.random.default_rng(4)
    N, TOK = 21000, 4242
    path = _np.maximum(100 + _np.cumsum(rng.normal(0, 0.05, N)), 5.0)
    q = lambda tok, t: ((float(path[t]), float(path[t]) + .5, True)
                        if t < N else (0., 0., False))
    c = lambda i, s, d, t: [{"token": TOK, "symbol": "S", "lot": 65,
                             "strike": 24000, "bid": float(path[t]),
                             "ask": float(path[t]) + .5, "fresh": True}]
    sw = BarSweep(bar_grid(), c, q, session_n=N, curfew_t=N, seed=1)
    for t in range(N):
        sw.offer(Signal(t=t, index="NIFTY",
                        conv=float(_np.clip(rng.normal(0, .30), -1, 1)),
                        spot=24000.))
        sw.mark(t)
    sw.finish(N - 1)
    s = sw.summary(N)
    bars = {k: v for k, v in s.items() if k.startswith("bar_")}
    counts = {v["n_trades"] for v in bars.values()}
    offers = [v["offered"] for v in bars.values()]
    check("trade COUNT is flat across the whole grid", len(counts) <= 2,
          f"counts={sorted(counts)} across bars "
          f"{min(bars)}..{max(bars)}")
    check("...while ELIGIBILITY spans orders of magnitude",
          max(offers) / max(min(offers), 1) > 20,
          f"offered {min(offers)} -> {max(offers)} "
          f"({max(offers)/max(min(offers),1):.0f}x). The bar selects WHICH, "
          f"never HOW MANY.")
    convs = [(v["bar"], v["mean_conv"]) for v in bars.values()
             if v["mean_conv"] is not None]
    convs.sort()
    # Rank correlation, NOT strict pairwise monotonicity: each bar fills
    # only ~6 slots, so adjacent bars invert on sampling noise. Asserting
    # strict monotonicity on a 6-sample mean is a test bug, not a standard.
    xs = _np.array([c[0] for c in convs], float)
    ys = _np.array([c[1] for c in convs], float)
    rx = _np.argsort(_np.argsort(xs)); ry = _np.argsort(_np.argsort(ys))
    rho = float(_np.corrcoef(rx, ry)[0, 1])
    check("mean taken conviction rises WITH the bar (rank corr)", rho > 0.90,
          f"Spearman rho={rho:.3f}: {ys[0]:.2f} at bar {xs[0]:.2f} -> "
          f"{ys[-1]:.2f} at bar {xs[-1]:.2f}")


def guard_max_statistic_calibration():
    print("\n[WY] the swept-grid test must control false positives")
    import numpy as _np
    from math import erf, sqrt
    from core.entry_bar_store import max_stat_permutation
    from core import capability_ladder as CL
    rng = _np.random.default_rng(0)
    NB, ND, TR = 14, 40, 150
    fp_max = fp_naive = 0
    for tr in range(TR):
        M = rng.normal(0, 900, (ND, 1)) + rng.normal(0, 380, (ND, NB))
        _o, p_max, means = max_stat_permutation(
            {f"d{i}": M[i] for i in range(ND)}, n_boot=800, seed=tr)
        fp_max += int(p_max <= 0.05)
        col = M[:, int(_np.argmax(means))]
        tt = col.mean() / (col.std(ddof=1) / _np.sqrt(ND))
        fp_naive += int((1 - .5 * (1 + erf(tt / sqrt(2)))) <= 0.05)
    r_max, r_naive = fp_max / TR, fp_naive / TR
    check("max-statistic holds ~nominal 5% under the null", r_max <= 0.10,
          f"{100*r_max:.1f}% vs naive best-bar t-test at {100*r_naive:.1f}% "
          f"(nominal 5%)")
    check("...and the naive test is demonstrably inflated",
          r_naive > r_max, f"naive {100*r_naive:.1f}% > WY {100*r_max:.1f}%")
    hits = 0
    for tr in range(80):
        M = rng.normal(0, 900, (ND, 1)) + rng.normal(0, 380, (ND, NB))
        M[:, 9] += 520.0
        _o, p, _m = max_stat_permutation(
            {f"d{i}": M[i] for i in range(ND)}, n_boot=800, seed=tr)
        hits += int(p <= 0.05)
    check("...while retaining power against a real effect", hits / 80 > 0.60,
          f"{100*hits/80:.0f}% detection at +Rs520/day")


def guard_entry_gate_closed():
    print("\n[EG] entry-side promotion must be gated and OFF by default")
    from core import entry_bar_store as EBS
    check("ENTRY_BAR_PROMOTE_ENABLED defaults False",
          not bool(getattr(config, "ENTRY_BAR_PROMOTE_ENABLED", True)),
          "the bar has the shortest path to capital of any knob here")
    check("promote() refuses a failing verdict",
          EBS.promote({"ok": False, "n_days": 0}) is False)
    check("active_bar() falls back to config",
          abs(EBS.active_bar() - config.entry_conviction_bar()) < 1e-9)
    excl = getattr(config, "_HASH_EXCLUDE", frozenset())
    miss = [k for k in dir(config) if k.startswith("ENTRY_BAR_")
            and k not in excl]
    check("every ENTRY_BAR_* constant is hash-excluded", not miss,
          f"missing: {miss}")


def guard_live_wiring():
    """STRUCTURAL, not executed. Everything above tests modules in isolation;
    NOTHING above proves a real fill ever reaches the shadow book. That path
    runs through apex_main_v9's loop and PositionManager's fill handlers, and
    it cannot be exercised without a broker session and a live tape. So this
    checks the wiring is PRESENT at the call sites — which catches the
    realistic regression (an edit to apex_main quietly drops the mark call)
    while making no claim to have run it."""
    print("\n[WIRE] the live path must be wired (source-level, NOT executed)")
    import ast as _ast
    root = Path(__file__).resolve().parents[1]

    def _src(rel):
        return (root / rel).read_text(encoding="utf-8", errors="ignore")

    pm = _src("core/position_manager.py")
    calls = set()
    for node in _ast.walk(_ast.parse(pm)):
        if isinstance(node, _ast.Call) and isinstance(node.func,
                                                      _ast.Attribute):
            calls.add(node.func.attr)
    check("PositionManager calls _shadow_open on entry",
          "_shadow_open" in calls)
    check("PositionManager calls _shadow_real_exit on exit",
          "_shadow_real_exit" in calls)
    check("PositionManager publishes token pins",
          "_pin_shadow_tokens" in calls)

    am = _src("apex_main_v9.py")
    check("apex_main constructs a ShadowBook", "ShadowBook(" in am)
    check("apex_main restores it (a restart must not reset peaks/clocks)",
          ".restore()" in am)
    check("apex_main marks it every tick", ".mark(ring_quotes" in am)
    check("apex_main closes it at the bell", ".close_session(" in am)

    hv = _src("data_harvester_v9.py")
    check("the harvester consults the pin manifest before pruning",
          "token_pins" in hv and "read_pins" in hv)

    ev = _src("run_evening.py")
    check("the evening chain runs the exit study with --promote",
          "trade_potential.py --promote" in ev)
    check("the evening chain runs the entry sweep",
          "entry_bar_study.py" in ev)
    check("the evening chain refuses an incomplete tape",
          "_day_is_complete" in ev)

    fg = _src("nightly_forge_v9.py")
    check("the forge exposes the COMPLETE signal stream (on_signal)",
          "on_signal" in fg,
          "on_block alone omits every signal the live bar ACCEPTED — the "
          "strongest seconds of the day — and starves every bar above the "
          "incumbent")


if __name__ == "__main__":
    print("=" * 74)
    print(f"APEX OMNI — shadow subsystem audit | CONFIG_HASH "
          f"{config.CONFIG_HASH}")
    print("=" * 74)
    print("\nPART 1 — REGRESSION GUARDS (each defect, one check)")
    guard_f_token_column()
    guard_a_session_window()
    guard_b_short_legs()
    guard_c_fifo()
    guard_e_staleness()
    guard_tz_independence()
    guard_d_loop_closed()
    guard_hash_stability()
    print("\nPART 2 — DRIVER EQUIVALENCE")
    equivalence()
    dead_feed_semantics()
    print("\nPART 3 — ENTRY SIDE")
    guard_capacity_is_the_constraint()
    guard_max_statistic_calibration()
    guard_entry_gate_closed()
    print("\nPART 4 — LIVE WIRING")
    guard_live_wiring()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    else:
        print("0 failures.")
        print()
        print("WHAT THIS DOES AND DOES NOT COVER")
        print("  Covered, by execution: reconstruction, the session window,")
        print("    staleness, the policy engine, live/offline equivalence,")
        print("    the promotion gates, the swept-grid statistics.")
        print("  Covered, source-level only: the live wiring in PART 4. No")
        print("    check here has ever seen a real fill reach the shadow")
        print("    book — that needs a broker session and a live tape.")
        print("  NOT covered: mark-cadence jitter (equivalence is proven at")
        print("    a forced 1s mark; the live loop is not exactly 1 Hz),")
        print("    real post-auction fills (0/7 harvested so far), and")
        print("    multi-index concurrency in the live book.")
    print("=" * 74)
    raise SystemExit(len(FAILURES))