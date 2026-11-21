"""
backtest/system_backtest.py
System backtest: a walk-forward simulation of the "scanner scoring"
(scanner.score_candidate) on historical candles — i.e. exactly the decision
the tool makes today, re-run on hundreds of past points and its result measured.

Anti-lookahead rules:
- At each point t, only the slice df[:t+1] is given to build_report. build_report
  computes all its components (including the internal pattern backtest that is
  the basis of the pattern component) only on that same input df and reads no
  global/network data (verified in analyzer.py); so the pattern rates also only
  see the past of t.
- Entry is at the close price of candle t and the simulation starts from candle t+1.

Trade simulation rules:
- If score >= threshold and both stop loss and target exist, a trade is opened.
- If in one candle both stop loss and target are touched, conservatively the
  "stop loss" is counted first (exactly the rule of backtest/engine.py).
- After SYSBT_MAX_HOLD_CANDLES candles with no result, exit at the close price
  of that candle (timeout). A trade whose full future window is not in the
  data is not opened at all, so the statistics do not suffer end-truncation bias.
- Costs: round-trip fee + two-sided slippage (config constants), deducted from
  profit, and the result is reported in risk units (R = |entry − stop loss|).

Trailing stop (additive, off by default — the trailing parameter in
run_system_backtest; with None, exactly the previous behavior):
- When the trade's floating profit reaches activation_r R units (favorable
  price = entry ± activation_r×R), trailing activates and the floating stop
  locks at a distance of trail_r R units behind the best price seen.
- The floating stop only moves in the trade's favor (ratchet) and never gets
  worse than the initial stop loss.
- Anti-lookahead: the trailing update takes effect from the "next candle" — a
  stop moved by the high/low of candle t is not applied to candle t itself
  (candle t is only judged against a stop that was known from its previous
  candles).
- Simultaneous touch of the stop (fixed or floating) and the target in one
  candle = stop first (the same existing conservative rule). Exit by the
  floating stop is reported with the result 'trailing'.
- ⚠️ Deviation from the semantics of paper_trading.py: there trailing_pct is a
  percentage of the "price" and rides the best price from the moment of entry
  (with no activation threshold); here the parameters are in R with an
  activation threshold, because R is this engine's common unit of measurement
  and activation from entry would have made the stop tighter than the
  structural stop from the very first candle. Details and the A/B test result
  are in TRAILING_AB_FA.md.

The output for each threshold is split into three parts: all / in_sample
(first 60%) / out_sample (last 40%); the reliable number is out_sample.

⚠️ No existing computational file has changed for this module; only the
existing build_report and score_candidate are used.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import scanner
from analysis import analyzer


# ---------------------------------------------------------------------------
# Simulating one trade
# ---------------------------------------------------------------------------
def _simulate_trade(df, entry_pos: int, direction: str, entry: float,
                    stop: float, target: float, max_hold: int,
                    trailing: dict = None):
    """
    Moves forward from the candle after entry up to max_hold candles.
    Output: (exit price, result) — result: 'target' | 'stop' | 'timeout'
    and with trailing active, 'trailing' is also possible.

    trailing (additive, default None = exactly the previous behavior):
    {"activation_r": A, "trail_r": D} in units of R = |entry − initial stop loss|.
    The rules (conservative and lookahead-free) are in the module docstring
    above: activation when floating profit reaches A×R; floating stop D×R
    behind the best price; movement only in the trade's favor; effect from the
    next candle; the stop (of either kind) wins on a simultaneous touch with
    the target.
    """
    n = len(df)
    end_pos = min(entry_pos + max_hold, n - 1)

    use_trail = False
    risk = 0.0
    activation_price = None
    eff_stop = stop
    if trailing:
        risk = abs(entry - stop)
        act = float(trailing["activation_r"])
        if direction == "long":
            activation_price = entry + act * risk
        else:
            activation_price = entry - act * risk
        use_trail = risk > 0

    for pos in range(entry_pos + 1, end_pos + 1):
        high = float(df["high"].iloc[pos])
        low = float(df["low"].iloc[pos])
        # 1) Checking the exit against the effective stop loss known from previous candles (not this candle)
        if direction == "long":
            hit_target = high >= target
            hit_stop = low <= eff_stop
        else:
            hit_target = low <= target
            hit_stop = high >= eff_stop
        if hit_target and hit_stop:
            # The intra-candle order is unknown — conservative: stop loss first
            return eff_stop, ("trailing" if eff_stop != stop else "stop")
        if hit_target:
            return target, "target"
        if hit_stop:
            return eff_stop, ("trailing" if eff_stop != stop else "stop")
        # 2) Updating the trailing from this candle — its effect is from the next candle
        if use_trail:
            if direction == "long":
                fav = high >= activation_price
                if fav:
                    cand_stop = high - float(trailing["trail_r"]) * risk
                    if cand_stop > eff_stop:
                        eff_stop = cand_stop
            else:
                fav = low <= activation_price
                if fav:
                    cand_stop = low + float(trailing["trail_r"]) * risk
                    if cand_stop < eff_stop:
                        eff_stop = cand_stop
    return float(df["close"].iloc[end_pos]), "timeout"


def _trade_r(direction: str, entry: float, exit_price: float, risk: float) -> float:
    """Trade return in R after deducting fee and slippage."""
    gross = (exit_price - entry) if direction == "long" else (entry - exit_price)
    cost = entry * (config.SYSBT_FEE_PCT / 100.0
                    + 2.0 * config.SYSBT_SLIPPAGE_PCT / 100.0)
    return (gross - cost) / risk


def _metrics(trades: list) -> dict:
    """Summary metrics from the trade list (in time order)."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate_pct": None, "avg_r": None,
                "expectancy_r": None, "total_r": 0.0, "max_drawdown_r": 0.0,
                "longest_losing_streak": 0,
                "outcomes": {"target": 0, "stop": 0, "timeout": 0, "trailing": 0}}
    rs = [t["r"] for t in trades]
    wins = sum(1 for r in rs if r > 0)
    equity, peak, max_dd = 0.0, 0.0, 0.0
    streak, longest = 0, 0
    outcomes = {"target": 0, "stop": 0, "timeout": 0, "trailing": 0}
    for t in trades:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
        equity += t["r"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if t["r"] <= 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return {
        "trades": n,
        "win_rate_pct": round(wins / n * 100, 1),
        "avg_r": round(sum(rs) / n, 3),
        "expectancy_r": round(sum(rs) / n, 3),   # Expected value = mean R
        "total_r": round(sum(rs), 2),
        "max_drawdown_r": round(max_dd, 2),
        "longest_losing_streak": longest,
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# Parameter sensitivity (additive — the param_grid idea from an external pack; only inside this module)
# ---------------------------------------------------------------------------
_SCORE_MAXES = {
    "trend": config.SCORE_TREND_MAX,
    "momentum": config.SCORE_MOMENTUM_MAX,
    "pattern": config.SCORE_PATTERN_MAX,
    "risk_reward": config.SCORE_RR_MAX,
    "volume": config.SCORE_VOLUME_MAX,
}


def _variant_score(components: dict, weights: dict) -> float:
    """
    Score of one weight variant: each component's share is scaled in proportion
    to the ratio "new cap/base cap". Keys missing in weights mean the base cap.
    Direction/entry/stop loss/target do not depend on the total score, so only
    the "trade selection" changes.
    """
    total = 0.0
    for name, base_max in _SCORE_MAXES.items():
        if not base_max:
            continue
        new_max = float(weights.get(name, base_max))
        total += (components.get(name) or 0.0) * (new_max / base_max)
    return round(total, 1)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run_system_backtest(symbol: str = "BTC", timeframe: str = "4h",
                        thresholds=(55, 60, 65), step: int = None,
                        max_hold: int = None, df=None,
                        min_history: int = None, progress: bool = False,
                        param_grid: list = None,
                        trailing: dict = None,
                        include_trailing_grid: bool = False) -> dict:
    """
    System backtest of one symbol/timeframe. If df is not given, it is fetched from Binance.
    Output: a dict including metadata and, for each threshold: all/in/out metrics + the trade log.

    param_grid (additive, default None = exactly the previous behavior): a list of
    dicts overriding component caps, e.g. {"pattern": 10.0} or
    {"trend": 35.0, "volume": 5.0}. For each variant, the same simulated trades
    are re-selected with the variant score and the in/out metrics are computed
    and returned under the key "param_grid" (together with the beats_base_out
    flag for each threshold). The implementation is entirely inside this module;
    the scanner is untouched.

    trailing (additive, default None = exactly the previous behavior): a dict
    {"activation_r": A, "trail_r": D} passed to the simulation of all trades
    of this run (rules in the module docstring). With trailing, each trade's
    record, in addition to the previous fields, has the key "exit_stop" and the
    flag "trailed", and the result is recorded in the "trailing" key at the top
    of the output.

    include_trailing_grid (additive, default False): if True, in addition to
    the main run (without trailing), the small grid config.SYSBT_TRAILING_GRID
    is also run on the same evaluations and returned under the key
    "trailing_grid" (for each variant and each threshold: in/out/all metrics,
    with no trade log, to keep the output light). Variant selection in the A/B
    test is done with in-sample only — the selection tool is in TRAILING_AB_FA.md.
    """
    if step is None:
        step = config.SYSBT_STEP
    if max_hold is None:
        # Improvement program (additive): if step 3 has recorded an accepted value
        # (IMPROVE_SYSBT_MAX_HOLD), that is the default;
        # otherwise exactly the previous value SYSBT_MAX_HOLD_CANDLES. Passing
        # max_hold explicitly takes precedence over both.
        max_hold = (getattr(config, "IMPROVE_SYSBT_MAX_HOLD", None)
                    or config.SYSBT_MAX_HOLD_CANDLES)
    if min_history is None:
        min_history = config.SYSBT_MIN_HISTORY
    if df is None:
        df = analyzer.fetch_candles_binance(symbol, timeframe)

    n = len(df)
    split_pos = int(n * config.SYSBT_INSAMPLE_FRAC)
    last_entry_pos = n - 1 - max_hold  # After this point the trade's full window is not in the data

    per_threshold = {th: [] for th in thresholds}
    variant_trades = [{th: [] for th in thresholds} for _ in (param_grid or [])]
    grid = list(config.SYSBT_TRAILING_GRID) if include_trailing_grid else []
    grid_trades = [{th: [] for th in thresholds} for _ in grid]
    evaluations = 0
    eval_errors = 0

    for pos in range(min_history - 1, last_entry_pos + 1, step):
        window = df.iloc[:pos + 1]
        try:
            report = analyzer.build_report(window, symbol, timeframe)
            cand = scanner.score_candidate(window, report, symbol, timeframe)
        except Exception:
            eval_errors += 1
            continue
        evaluations += 1
        if progress:
            print(f"  ارزیابی {pos + 1}/{n} — امتیاز {cand['score']}", flush=True)

        stop, target, entry = cand.get("stop"), cand.get("target"), cand.get("entry")
        if stop is None or target is None or entry is None:
            continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        direction = cand["direction"]
        # The direction must be consistent with the stop loss position, otherwise the trade is meaningless
        if direction == "long" and not (stop < entry < target):
            continue
        if direction == "short" and not (target < entry < stop):
            continue

        exit_price, outcome = _simulate_trade(
            df, pos, direction, entry, stop, target, max_hold, trailing=trailing)
        r_value = _trade_r(direction, entry, exit_price, risk)
        segment = "in" if pos < split_pos else "out"
        record = {
            "entry_time": str(df.index[pos]),
            "segment": segment,
            "direction": direction,
            "score": cand["score"],
            "components": dict(cand["components"]),
            "pattern": (cand.get("pattern") or {}).get("name"),
            "entry": entry, "stop": stop, "target": target,
            "exit_price": exit_price, "outcome": outcome,
            "r": round(r_value, 3),
        }
        if trailing:
            # Additive fields only when trailing is active (the default path untouched)
            record["exit_stop"] = exit_price if outcome in ("stop", "trailing") else None
            record["trailed"] = outcome == "trailing"
        for th in thresholds:
            if cand["score"] >= th:
                per_threshold[th].append(record)
        # Weight variants: the same trade is selected with a rescaled score
        for vi, weights in enumerate(param_grid or []):
            v_score = _variant_score(cand["components"], weights)
            v_record = dict(record, score=v_score)
            for th in thresholds:
                if v_score >= th:
                    variant_trades[vi][th].append(v_record)
        # Trailing grid: the same entries, re-simulated with each variant
        for gi, tparams in enumerate(grid):
            g_exit, g_outcome = _simulate_trade(
                df, pos, direction, entry, stop, target, max_hold, trailing=tparams)
            g_record = dict(record)
            g_record["exit_price"] = g_exit
            g_record["outcome"] = g_outcome
            g_record["r"] = round(_trade_r(direction, entry, g_exit, risk), 3)
            g_record["exit_stop"] = g_exit if g_outcome in ("stop", "trailing") else None
            g_record["trailed"] = g_outcome == "trailing"
            for th in thresholds:
                if cand["score"] >= th:
                    grid_trades[gi][th].append(g_record)

    result = {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "candles": n, "evaluations": evaluations, "eval_errors": eval_errors,
        "step": step, "max_hold": max_hold,
        "split_at": str(df.index[split_pos]) if 0 < split_pos < n else None,
        "thresholds": {},
    }
    for th in thresholds:
        trades = per_threshold[th]
        result["thresholds"][str(th)] = {
            "all": _metrics(trades),
            "in_sample": _metrics([t for t in trades if t["segment"] == "in"]),
            "out_sample": _metrics([t for t in trades if t["segment"] == "out"]),
            "trades": trades,
        }

    if param_grid:
        result["param_grid"] = []
        for vi, weights in enumerate(param_grid):
            entry_v = {"weights": dict(weights), "thresholds": {}}
            for th in thresholds:
                trades = variant_trades[vi][th]
                out_m = _metrics([t for t in trades if t["segment"] == "out"])
                base_out = result["thresholds"][str(th)]["out_sample"]["expectancy_r"]
                beats = (out_m["expectancy_r"] is not None
                         and base_out is not None
                         and out_m["expectancy_r"] > base_out)
                entry_v["thresholds"][str(th)] = {
                    "all": _metrics(trades),
                    "in_sample": _metrics([t for t in trades if t["segment"] == "in"]),
                    "out_sample": out_m,
                    "beats_base_out": beats,
                }
            result["param_grid"].append(entry_v)

    if trailing:
        result["trailing"] = dict(trailing)
    if include_trailing_grid:
        result["trailing_grid"] = []
        for gi, tparams in enumerate(grid):
            entry_g = {"params": dict(tparams), "thresholds": {}}
            for th in thresholds:
                trades = grid_trades[gi][th]
                entry_g["thresholds"][str(th)] = {
                    "all": _metrics(trades),
                    "in_sample": _metrics([t for t in trades if t["segment"] == "in"]),
                    "out_sample": _metrics([t for t in trades if t["segment"] == "out"]),
                }
            result["trailing_grid"].append(entry_g)
    return result


def format_summary_fa(result: dict) -> str:
    """Persian text summary of the system backtest result."""
    lines = [
        f"بک‌تست سیستمی {result['symbol']}/USDT — {result['timeframe']} "
        f"({result['candles']} کندل، {result['evaluations']} ارزیابی)",
        f"تقسیم نمونه: قبل از {result['split_at']} = داخل نمونه، بعد از آن = خارج نمونه",
    ]
    for th, data in result["thresholds"].items():
        for seg, fa in (("in_sample", "داخل نمونه"), ("out_sample", "خارج نمونه"), ("all", "کل")):
            m = data[seg]
            if m["trades"] == 0:
                lines.append(f"آستانه {th} | {fa}: معامله‌ای نیست")
            else:
                lines.append(
                    f"آستانه {th} | {fa}: {m['trades']} معامله | برد {m['win_rate_pct']}٪ | "
                    f"امید ریاضی {m['expectancy_r']}R | مجموع {m['total_r']}R | "
                    f"حداکثر افت {m['max_drawdown_r']}R | طولانی‌ترین باخت پیاپی {m['longest_losing_streak']}")
    if result.get("param_grid"):
        lines.append("— حساسیت پارامتر (واریانت‌های وزن) —")
        for entry_v in result["param_grid"]:
            w_txt = ", ".join(f"{k}={v:g}" for k, v in entry_v["weights"].items())
            lines.append(f"واریانت {{{w_txt}}}:")
            for th, data in entry_v["thresholds"].items():
                mi, mo = data["in_sample"], data["out_sample"]
                flag = ""
                if data["beats_base_out"]:
                    flag = (" ⚠️ خارج نمونه از پایه بهتر شد — این می‌تواند اورفیت "
                            "باشد؛ بدون تأیید روی دوره/نماد دیگر وزن را عوض نکنید.")
                lines.append(
                    f"  آستانه {th} | داخل نمونه: {mi['trades']} معامله، "
                    f"امید ریاضی {mi['expectancy_r']}R | خارج نمونه: "
                    f"{mo['trades']} معامله، امید ریاضی {mo['expectancy_r']}R{flag}")
    if result.get("trailing"):
        t = result["trailing"]
        lines.append(
            f"تریلینگ‌استاپ فعال در این اجرا: فعال‌سازی {t['activation_r']}R، "
            f"فاصله {t['trail_r']}R (خروج‌های trailing در شمارش نتایج جداست).")
    if result.get("trailing_grid"):
        lines.append("— گرید تریلینگ‌استاپ (A/B) —")
        for entry_g in result["trailing_grid"]:
            p = entry_g["params"]
            lines.append(
                f"واریانت {{فعال‌سازی {p['activation_r']}R، فاصله {p['trail_r']}R}}:")
            for th, data in entry_g["thresholds"].items():
                mi, mo = data["in_sample"], data["out_sample"]
                lines.append(
                    f"  آستانه {th} | داخل نمونه: {mi['trades']} معامله، "
                    f"امید ریاضی {mi['expectancy_r']}R | خارج نمونه: "
                    f"{mo['trades']} معامله، امید ریاضی {mo['expectancy_r']}R")
        lines.append("⚠️ انتخاب واریانت فقط با داخل نمونه مجاز است؛ خارج نمونه "
                     "فقط برای گزارش و تصمیم نهایی است (ضد اورفیت).")
    lines.append("⚠️ عدد قابل اتکا «خارج نمونه» است؛ گذشته تضمین آینده نیست.")
    return "\n".join(lines)


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = run_system_backtest("BTC", "4h")
    print(format_summary_fa(res))
    print(f"(زمان اجرا: {time.time() - t0:.0f} ثانیه)")
