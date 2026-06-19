"""
APEX OMNI v9 — SCENARIO CATALOG
===============================
Sixteen intraday situations a real Indian options day can throw at the
system, each with explicit PASS criteria. The marquee one is #5 — the exact
"institutions flush retail, then the contract soars" pattern you described —
paired with #6, its evil twin (a GENUINE breakdown wearing the same opening
costume), because a shield that can't tell them apart is just denial.
"""
from __future__ import annotations
import config
from simulation.scenario_engine import SimDay, Scenario, Signal


def _no_viol(r):
    return not r.violations


def s01():
    day = (SimDay(seed=11).trend("09:45", "12:30", 1.4))
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        if len(r.trades) != 1: return False, f"{len(r.trades)} trades (want 1)"
        if r.pnl <= 0: return False, f"PnL ₹{r.pnl:.0f} not positive"
        return True, (f"walked ladder to {r.trades[0]['sym']}, "
                      f"exit {r.trades[0]['reason']}, ₹{r.pnl:+.0f}")
    return Scenario("trend_up_clean", "Steady up-trend; long CE rides trail "
                    "to target", day, [Signal("09:50", +0.84)], ok)


def s02():
    day = (SimDay(seed=12).trend("09:45", "12:30", -1.4))
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        if len(r.trades) != 1 or r.pnl <= 0:
            return False, f"trades={len(r.trades)} PnL ₹{r.pnl:.0f}"
        return True, f"PE side symmetric, ₹{r.pnl:+.0f}"
    return Scenario("trend_down_clean", "Steady down-trend; long PE",
                    day, [Signal("09:50", -0.84)], ok)


def s03():
    day = SimDay(seed=13, noise_pts_s=0.8)
    sigs = [Signal(hm, c, 0.66, 200) for hm, c in
            [("10:00", .78), ("10:20", -.78), ("10:40", .78), ("11:00", -.78),
             ("11:20", .78), ("11:40", -.78), ("12:00", .78), ("12:20", -.78)]]
    def ok(r):
        if len(r.trades) > 6:
            return False, f"{len(r.trades)} trades — cooldowns failed to cap churn"
        if r.pnl < -0.13 * config.TRADING_CAPITAL:
            return False, f"chop bled ₹{r.pnl:.0f} past every governor"
        capper = "drawdown halt" if r.halted else "cooldown+lockout"
        return True, (f"{len(r.trades)} trades from 8 flip-flops, "
                      f"₹{r.pnl:+.0f} — {capper} capped the day")
    # Anti-churn is tested on the passive path (cross bar raised) so the test
    # isolates cooldown/lockout/drawdown behaviour from crossing-spread cost,
    # which the trend scenarios cover. Default live config crosses these.
    return Scenario("range_chop", "Sideways chop with 8 flip-flopping signals "
                    "— anti-churn must cap the bleed", day, sigs, ok,
                    cfg={"ENTRY_CROSS_CONVICTION": 0.99})


def s04():
    day = (SimDay(open_spot=24700, seed=14).trend("09:30", "11:30", -1.2)
           .walls(put=24480, call=24760))
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        if len(r.trades) != 1 or r.pnl <= 0:
            return False, f"trades={len(r.trades)} PnL ₹{r.pnl:.0f}"
        return True, f"gap faded with PE, ₹{r.pnl:+.0f}"
    return Scenario("gap_up_fade", "Open +200 gap, fades all morning",
                    day, [Signal("09:40", -0.84)], ok)


def s05():
    day = (SimDay(seed=15, noise_pts_s=0.30)
           .trend("09:40", "09:58", 0.5)
           .stop_run("10:00", depth_pts=38, down_s=12, hold_s=30,
                     recover_pts=50, recover_s=60, iv_bump=0.002)
           .trend("10:04", "12:00", 1.9)
           .walls(put=24462, call=24700))
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        if r.trap_holds < 1:
            return False, "shield never engaged on the flush"
        if any(t["reason"].startswith("DISASTER") for t in r.trades):
            return False, "flushed out at the floor — shield failed"
        if len(r.trades) != 1 or r.pnl <= 0:
            return False, f"trades={len(r.trades)} PnL ₹{r.pnl:.0f}"
        return True, (f"held {r.trap_holds} breach tick(s) through the hunt"
                      f"{' (trap confirmed)' if r.trap_confirmed else ''}, "
                      f"then ₹{r.pnl:+.0f}")
    return Scenario("stop_hunt_trap_long", "★ THE one you asked for: long CE, "
                    "institutions flush −26pts on pulled quotes + absorbed "
                    "panic selling at the put wall, ΔOI flat — then the rip",
                    day, [Signal("09:48", +0.85, window_s=420)], ok)


def s06():
    day = (SimDay(seed=16)
           .trend("09:40", "10:18", 0.4)
           .breakdown("10:20", depth_pts=90, over_s=160)
           .trend("10:23", "13:00", -0.9))
    def ok(r):
        if len(r.trades) != 1: return False, f"{len(r.trades)} trades"
        t = r.trades[0]
        if t["pnl"] >= 0: return False, "profited from a breakdown?!"
        if not t["reason"].startswith(("STOP", "DISASTER")):
            return False, f"exit was {t['reason']}"
        held = t["t_out"] - day._idx("10:20")
        if held > config.TRAP_MAX_HOLD_S + 60:
            return False, f"shield overstayed a REAL break ({held:.0f}s)"
        if any(v for v in r.violations):
            return False, "; ".join(r.violations)
        return True, (f"exited {t['reason'].split(' ')[0]} {held:.0f}s into "
                      f"the break, loss contained ₹{t['pnl']:+.0f}")
    return Scenario("real_breakdown", "Evil twin of #5: sustained sell with "
                    "FRESH OI committing — the shield must stand aside",
                    day, [Signal("09:50", +0.85)], ok)


def s07():
    day = SimDay(seed=17, dte_days=0.30, base_iv=0.125, noise_pts_s=0.5)
    def ok(r):
        if not r.trades: return False, "never entered"
        t = r.trades[0]
        dur_min = (t["t_out"] - t["t_in"]) / 60
        if dur_min > config.MAX_HOLD_MINUTES + 2:
            return False, f"held {dur_min:.0f}m on expiry day"
        if not t["reason"].startswith(("MAX_HOLD", "STOP")):
            return False, f"exit {t['reason']}"
        if not _no_viol(r): return False, "; ".join(r.violations)
        return True, (f"theta discipline: out in {dur_min:.0f}m via "
                      f"{t['reason'].split(' ')[0]}, ₹{r.pnl:+.0f}")
    return Scenario("expiry_pin_theta", "0-DTE pin day, price glued to the "
                    "strike — the guillotine must cut before decay does",
                    day, [Signal("10:00", +0.82)], ok)


def s08():
    day = (SimDay(seed=18, base_iv=0.14)
           .trend("10:00", "10:30", 0.8)
           .iv_step("10:30", 0.095))
    def ok(r):
        if len(r.trades) != 1: return False, f"{len(r.trades)} trades"
        t = r.trades[0]
        crush = day._idx("10:30")
        if t["t_out"] < crush:
            return False, "stopped out before the crush even hit"
        if t["pnl"] >= 0: return False, "IV crush should hurt a long"
        if t["t_out"] - crush > 90:
            return False, f"lingered {t['t_out']-crush:.0f}s after the crush"
        held_after = [e for e in r.events if e["event"] == "TRAP_HOLD"
                      and float(e["ts"]) >= crush]
        if held_after:
            return False, ("shield held an IV crush — spot never moved, "
                           "that's genuine repricing, not a hunt")
        return True, (f"vega hit taken honestly via "
                      f"{t['reason'].split(' ')[0]} {t['t_out']-crush:.0f}s "
                      f"after the event, ₹{t['pnl']:+.0f}")
    return Scenario("iv_crush_event", "13:30 IV 14→9.5 with spot flat — the "
                    "shield must NOT mistake vega for a trap", day,
                    [Signal("10:00", +0.82)], ok)


def s09():
    day = (SimDay(seed=19)
           .gap("10:30", -240, over_s=15)        # −1% air pocket in 15 s
           .gap("11:20", +260, over_s=15))       # violent V back up
    sigs = [Signal("09:50", +0.85), Signal("11:05", -0.85),
            Signal("12:30", +0.85, window_s=600)]
    def ok(r):
        if not r.halted or "drawdown" not in r.halt_reason:
            return False, f"governor never tripped ({r.halt_reason!r})"
        if len(r.trades) != 2:
            return False, f"{len(r.trades)} trades (want 2 then halt)"
        if any("BUY after risk halt" in v for v in r.violations):
            return False, "traded after the halt"
        if r.pnl < -0.25 * config.TRADING_CAPITAL:
            return False, f"loss ₹{r.pnl:.0f} blew past the floors"
        return True, (f"2 floored exits, day halted at ₹{r.pnl:+.0f} "
                      f"({-r.pnl/config.TRADING_CAPITAL:.0%}); 3rd signal met "
                      f"a dead switch — zero post-halt orders")
    return Scenario("flash_crash_dd_halt", "−1% air pocket then a violent V: "
                    "both directions floored, daily-drawdown kill switch must "
                    "end the day", day, sigs, ok,
                    # Pinned to the reference capital this fixture was authored
                    # for: it asserts the EXACT trade count that accumulates the
                    # 10% daily-drawdown halt, which is capital-dependent (larger
                    # capital → each loss is a smaller % → more trades to reach
                    # 10%). The halt itself is capital-relative and fires
                    # correctly at any capital; this isolates the MECHANIC.
                    cfg={"TRADING_CAPITAL": 5000.0})


def s10():
    day = (SimDay(seed=20, noise_pts_s=0.40)
           .feed_stale("09:50", 240)
           .trend("09:55", "10:40", 1.5)
           .feed_stale("10:12", 80))
    sigs = [Signal("09:51", +0.85, window_s=1200)]
    def ok(r):
        stale_blocks = [e for e in r.events if e["event"] == "BLOCKED"
                        and "stale" in e["reason"]]
        if not stale_blocks:
            return False, "entry was never blocked on the dead feed"
        if len(r.trades) != 1:
            return False, f"{len(r.trades)} trades"
        if not r.trades[0]["reason"].startswith("STALE_FEED"):
            return False, f"exit {r.trades[0]['reason']}"
        return True, (f"{len(stale_blocks)} stale-blocked attempts, entered on "
                      f"recovery, second outage flattened the position")
    return Scenario("data_stale_watchdog", "Feed dies twice: entries blocked "
                    "while blind, open position emergency-flattened past 60s",
                    day, sigs, ok)


def s11():
    day = SimDay(seed=21)
    def ok(r):
        if r.trades or any(e["event"] == "BUY_FILL" for e in r.events):
            return False, "a phantom position appeared from rejected orders"
        if r.rejects < config.MAX_ORDER_REJECTS:
            return False, f"only {r.rejects} rejects recorded"
        if not r.halted or "reject" not in r.halt_reason:
            return False, "reject storm never tripped the kill switch"
        return True, f"{r.rejects} rejects → halted, zero phantom inventory"
    return Scenario("reject_storm", "Broker rejects everything: kill switch "
                    "after 3, and — the v8 sin — NO position may be registered",
                    day, [Signal("09:50", +0.85, window_s=600)], ok,
                    inject_rejects=3)


def s12():
    day = (SimDay(seed=22)
           .trend("09:50", "09:51", +150)         # ask runs away from maker
           .trend("10:00", "12:00", 0.8))
    # Tests the PASSIVE maker walk-away path. The default live config crosses
    # (cross bar = entry bar), so this scenario raises the cross bar to force
    # the passive path and verify it walks away without phantom inventory.
    sigs = [Signal("09:50", +0.85, win_prob=0.66, window_s=900)]
    def ok(r):
        if r.nofills < 1:
            return False, "maker never missed — can't test walk-away"
        if not _no_viol(r): return False, "; ".join(r.violations)
        return True, (f"{r.nofills} maker miss(es) walked away clean"
                      f"{', then a real fill' if r.trades else ''} — "
                      f"position equals fill, ₹{r.pnl:+.0f}")
    return Scenario("maker_nofill_repost", "Price sprints off the maker limit: "
                    "cancel/walk-away, retry later — position must equal FILL, "
                    "never intent", day, sigs, ok,
                    cfg={"ENTRY_CROSS_CONVICTION": 0.99})


def s13():
    day = (SimDay(seed=23).widen_spread("14:00", "15:30", 4.0))
    def ok(r):
        if r.trades:
            return False, "entered through a 5% spread"
        skips = [e for e in r.events if e["event"] == "SKIP"
                 and "spread" in e["reason"]]
        if not skips:
            return False, "spread gate never spoke"
        return True, "afternoon swamp refused — every leg failed the spread gate"
    return Scenario("afternoon_illiquidity", "Post-14:00 spreads blow out 4×: "
                    "MAX_ENTRY_SPREAD_PCT must refuse the swamp", day,
                    [Signal("14:10", -0.85)], ok)


def s14():
    day = SimDay(seed=24, dte_days=6.0, base_iv=0.20)
    def ok(r):
        if r.trades:
            return False, "bought something the capital cannot hold"
        blocks = [e for e in r.events if e["event"] == "BLOCKED"]
        if not blocks:
            return False, "never even consulted the governor"
        why = blocks[0]["reason"]
        if not any(k in why for k in ("exceeds", "cannot HOLD", "Kelly")):
            return False, f"blocked for the wrong reason: {why}"
        return True, f"every rung too rich for ₹{config.TRADING_CAPITAL:.0f}: " \
                     f"\"{why[:60]}…\""
    return Scenario("capital_affordability", "Far-dated fat premiums: the "
                    "whole ladder exceeds what ₹5,000 can HOLD — zero orders "
                    "allowed", day, [Signal("10:00", +0.85)], ok,
                    # Pinned: this fixture's premiums are sized to exceed ₹5,000
                    # of holding capacity. At larger capital those same premiums
                    # ARE affordable, so the "refuse everything" property only
                    # holds at the authored capital. The affordability walker
                    # works correctly at any capital — this isolates the refusal.
                    cfg={"TRADING_CAPITAL": 5000.0})


def s15():
    day = SimDay(seed=25, noise_pts_s=0.35)
    def ok(r):
        if len(r.trades) != 1: return False, f"{len(r.trades)} trades"
        t = r.trades[0]
        if not t["reason"].startswith("EOD"):
            return False, f"exit {t['reason']}"
        out_hm = t["t_out"]
        if out_hm > day._idx("15:16"):
            return False, "flattened too late for the 15:20 MIS square-off"
        if not _no_viol(r): return False, "; ".join(r.violations)
        return True, "flat at 15:15 sharp — minutes ahead of the broker's hand"
    return Scenario("eod_flatten", "Quiet drift into the close holding a "
                    "position: 15:15 force-flatten beats the MIS auto "
                    "square-off", day, [Signal("14:40", +0.82)], ok)


def s16():
    day = (SimDay(seed=26, noise_pts_s=0.30)
           .trend("09:40", "10:08", 0.6)
           .stop_run("10:10", depth_pts=38, down_s=12, hold_s=30,
                     recover_pts=50, recover_s=60, iv_bump=0.002)
           .trend("10:14", "10:38", 0.7)
           .stop_run("10:40", depth_pts=38, down_s=12, hold_s=30,
                     recover_pts=50, recover_s=60, iv_bump=0.002)
           .trend("10:44", "12:30", 1.5)
           .walls(put=24470, call=24680))
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        windows = {e["ts"] for e in r.events if e["event"] == "TRAP_HOLD"}
        if r.trap_holds < 2:
            return False, f"only {r.trap_holds} shield engagements"
        if any(t["reason"].startswith("DISASTER") for t in r.trades):
            return False, "a whipsaw reached the floor"
        if r.pnl <= 0:
            return False, f"₹{r.pnl:.0f} after surviving both traps"
        return True, (f"two hunts survived ({len(windows)} hold ticks, "
                      f"uses capped at {config.TRAP_MAX_USES_PER_TRADE}), "
                      f"₹{r.pnl:+.0f}")
    return Scenario("whipsaw_double_trap", "Two flushes in one position 30 "
                    "min apart — shield's per-trade use cap and re-anchoring "
                    "both under test", day,
                    [Signal("09:48", +0.85, window_s=420)], ok,
                    # Pinned: the fixed-depth stop-runs in this fixture are sized
                    # to breach the stop of the cheap OTM option that ₹5,000
                    # affords (high gamma → moves fast). At larger capital the
                    # walker buys a deeper, lower-gamma option the same point-
                    # depth runs don't breach, so the shield never engages. The
                    # shield logic is unchanged; this isolates the trap MECHANIC
                    # on the strike profile it was authored against.
                    cfg={"TRADING_CAPITAL": 5000.0})


def s17():
    # Trained model is LIVE. Position opens healthy, then the model's P(win)
    # decays below the exit floor while price is still well ABOVE the fixed stop
    # and disaster floor. The model-shaped exit must cut the trade EARLY via
    # META_EDGE_GONE — proving the dynamic exit shapes timing inside the envelope
    # (it pulls the exit IN; it never extends past, nor suppresses, the floors).
    day = (SimDay(seed=31, noise_pts_s=0.25)
           .trend("09:45", "10:05", +0.5)      # gentle drift up — no stop, no target
           .trend("10:05", "11:30", +0.1))
    # win_prob starts healthy (0.62), decays to 0.40 (below 0.45 floor) at 10:02
    sigs = [Signal("09:50", +0.80, win_prob=0.62, window_s=300),
            Signal("10:02", +0.80, win_prob=0.40, window_s=600)]
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        reasons = [t["reason"] for t in r.trades]
        if not any(x.startswith("META_EDGE_GONE") for x in reasons):
            return False, f"model-shaped exit never fired (exits: {reasons})"
        if any(x.startswith("DISASTER") or x.startswith("STOP") for x in reasons):
            return False, "a risk floor fired — edge-decay should exit first"
        return True, (f"model cut the trade on P(win) decay via META_EDGE_GONE "
                      f"before any fixed stop/floor — ₹{r.pnl:+.0f}")
    return Scenario("dynamic_edge_decay_exit", "Trained model live: P(win) "
                    "decays mid-hold → model-shaped early exit fires inside the "
                    "fixed risk envelope", day, sigs, ok, meta_live=True)


def s18():
    # Profit-lock guarantee: a winner that runs up, then suffers a PLAIN
    # (non-hunt) reversal must exit at/above breakeven — never round-trip to a
    # loss. No trap signature, so the shield does not hold; the profit-lock floor
    # is the backstop. (The hunt case — where the shield DOES hold through to a
    # reclaim — is s16/whipsaw_double_trap.)
    day = (SimDay(seed=37, noise_pts_s=0.20)
           .trend("09:45", "10:10", +1.4)       # strong run up → arms trail+lock
           .trend("10:10", "11:30", -1.1))      # slow grind back down (NOT a flush)
    def ok(r):
        if not _no_viol(r): return False, "; ".join(r.violations)
        if not r.trades:
            return False, "no trade taken"
        worst = min(t["pnl"] for t in r.trades)
        if worst < -1.0:                          # never a loss (±costs rounding)
            return False, f"a winner round-tripped to a loss ₹{worst:.0f}"
        reasons = {t["reason"] for t in r.trades}
        return True, (f"plain reversal exited at/above breakeven via "
                      f"{'/'.join(sorted(reasons))} — winner never became a "
                      f"loser (worst ₹{worst:+.0f})")
    return Scenario("profit_lock_no_roundtrip", "Winner runs, then plain (non-"
                    "hunt) reversal — profit-lock floor guarantees it can't "
                    "become a loss", day,
                    [Signal("09:50", +0.85, win_prob=0.66, window_s=420)], ok,
                    # Pinned: the +1.4% trend in this fixture is sized to move the
                    # cheap OTM option ₹5,000 affords past the +15% TRAIL_ARM_PCT
                    # that arms the profit-lock. At larger capital the walker buys
                    # a deeper, lower-gamma option that the same % spot move does
                    # NOT lift +15%, so the trail never arms and the theta stop
                    # fires first. Profit-lock logic is unchanged; it only
                    # guarantees breakeven ONCE ARMED, and this fixture is built
                    # to reach that arm. Isolates the profit-lock MECHANIC.
                    cfg={"TRADING_CAPITAL": 5000.0})


ALL = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13,
       s14, s15, s16, s17, s18]