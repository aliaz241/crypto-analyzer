"""
backtest/system_backtest.py
بک‌تست سیستمی: شبیه‌سازی walk-forward از «امتیازدهی اسکنر»
(scanner.score_candidate) روی کندل‌های تاریخی — یعنی دقیقاً همان تصمیمی که
ابزار امروز می‌گیرد، روی صدها نقطه‌ی گذشته بازاجرا و نتیجه‌اش سنجیده می‌شود.

قواعد ضد نشت آینده (lookahead):
- در هر نقطه‌ی t فقط برش df[:t+1] به build_report داده می‌شود. build_report
  همه‌ی اجزایش (از جمله بک‌تست داخلی الگوها که مبنای مؤلفه‌ی الگوست) را فقط
  روی همان df ورودی حساب می‌کند و هیچ داده‌ی سراسری/شبکه‌ای نمی‌خواند
  (در analyzer.py بررسی شد)؛ پس نرخ‌های الگو هم فقط گذشته‌ی t را می‌بینند.
- ورود با قیمت بسته‌شدن کندل t است و شبیه‌سازی از کندل t+1 شروع می‌شود.

قواعد شبیه‌سازی معامله:
- اگر امتیاز >= آستانه و حد ضرر و هدف هر دو موجود باشند، معامله باز می‌شود.
- اگر در یک کندل هم حد ضرر و هم هدف لمس شوند، محافظه‌کارانه «حد ضرر» اول
  حساب می‌شود (دقیقاً قاعده‌ی backtest/engine.py).
- بعد از SYSBT_MAX_HOLD_CANDLES کندل بدون نتیجه، خروج با قیمت بسته‌شدن
  همان کندل (timeout). معامله‌ای که پنجره‌ی کامل آینده‌اش در داده نیست،
  اصلاً باز نمی‌شود تا آمار دچار سوگیری برش انتهایی نشود.
- هزینه‌ها: کارمزد رفت‌وبرگشت + اسلیپیج دو طرف (ثابت‌های config)، از سود
  کسر و نتیجه بر حسب واحد ریسک (R = |ورود − حد ضرر|) گزارش می‌شود.

تریلینگ‌استاپ (افزودنی، پیش‌فرض خاموش — پارامتر trailing در
run_system_backtest؛ با None دقیقاً رفتار قبلی):
- وقتی سود شناور معامله به activation_r واحد R برسد (قیمت مطلوب =
  ورود ± activation_r×R)، تریلینگ فعال می‌شود و حد ضرر شناور روی فاصله‌ی
  trail_r واحد R پشت بهترین قیمتِ دیده‌شده قفل می‌شود.
- حد ضرر شناور فقط به نفع معامله حرکت می‌کند (رچت) و هیچ‌وقت از حد ضرر
  اولیه بدتر نمی‌شود.
- ضد نشت آینده: به‌روزرسانی تریلینگ از «کندل بعدی» اثر می‌گذارد — حد
  ضرری که با سقف/کف کندل t جابه‌جا شده، روی خود کندل t اعمال نمی‌شود
  (کندل t فقط با حد ضرری سنجیده می‌شود که از کندل‌های قبلش معلوم بوده).
- لمس هم‌زمان حد ضرر (ثابت یا شناور) و هدف در یک کندل = حد ضرر اول
  (همان قاعده‌ی محافظه‌کارانه‌ی موجود). خروج با حد ضرر شناور با نتیجه‌ی
  'trailing' گزارش می‌شود.
- ⚠️ انحراف از سمانتیک paper_trading.py: آنجا trailing_pct درصدی از
  «قیمت» است و از لحظه‌ی ورود (بدون آستانه‌ی فعال‌سازی) روی بهترین قیمت
  سوار می‌شود؛ اینجا پارامترها بر حسب R و با آستانه‌ی فعال‌سازی‌اند،
  چون R واحد مشترک سنجش این موتور است و فعال‌سازی از ورود، حد ضرر را
  از همان کندل اول تنگ‌تر از استاپ ساختاری می‌کرد. جزئیات و نتیجه‌ی
  تست A/B در TRAILING_AB_FA.md.

خروجی برای هر آستانه به سه بخش all / in_sample (۶۰٪ اول) / out_sample
(۴۰٪ آخر) تقسیم می‌شود؛ عدد قابل اتکا out_sample است.

⚠️ هیچ فایل محاسباتی موجودی برای این ماژول تغییر نکرده؛ فقط از
build_report و score_candidate موجود استفاده می‌شود.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import scanner
from analysis import analyzer


# ---------------------------------------------------------------------------
# شبیه‌سازی یک معامله
# ---------------------------------------------------------------------------
def _simulate_trade(df, entry_pos: int, direction: str, entry: float,
                    stop: float, target: float, max_hold: int,
                    trailing: dict = None):
    """
    از کندل بعد از ورود تا حداکثر max_hold کندل جلو می‌رود.
    خروجی: (قیمت خروج, نتیجه) — نتیجه: 'target' | 'stop' | 'timeout'
    و با تریلینگ فعال: 'trailing' هم ممکن است.

    trailing (افزودنی، پیش‌فرض None = دقیقاً رفتار قبلی):
    {"activation_r": A, "trail_r": D} بر حسب R = |ورود − حد ضرر اولیه|.
    قواعد (محافظه‌کارانه و بدون نشت آینده) در docstring بالای ماژول آمده:
    فعال‌سازی با رسیدن سود شناور به A×R؛ حد ضرر شناور D×R پشت بهترین
    قیمت؛ فقط حرکت به نفع معامله؛ اثر از کندل بعدی؛ حد ضرر (هر نوع) در
    لمس هم‌زمان با هدف برنده است.
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
        # ۱) سنجش خروج با حد ضرر مؤثرِ معلوم از کندل‌های قبل (نه این کندل)
        if direction == "long":
            hit_target = high >= target
            hit_stop = low <= eff_stop
        else:
            hit_target = low <= target
            hit_stop = high >= eff_stop
        if hit_target and hit_stop:
            # ترتیب داخل کندل نامعلوم است — محافظه‌کارانه: اول حد ضرر
            return eff_stop, ("trailing" if eff_stop != stop else "stop")
        if hit_target:
            return target, "target"
        if hit_stop:
            return eff_stop, ("trailing" if eff_stop != stop else "stop")
        # ۲) به‌روزرسانی تریلینگ از روی همین کندل — اثرش از کندل بعدی است
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
    """بازده معامله بر حسب R بعد از کسر کارمزد و اسلیپیج."""
    gross = (exit_price - entry) if direction == "long" else (entry - exit_price)
    cost = entry * (config.SYSBT_FEE_PCT / 100.0
                    + 2.0 * config.SYSBT_SLIPPAGE_PCT / 100.0)
    return (gross - cost) / risk


def _metrics(trades: list) -> dict:
    """معیارهای خلاصه از لیست معامله‌ها (به ترتیب زمانی)."""
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
        "expectancy_r": round(sum(rs) / n, 3),   # امید ریاضی = میانگین R
        "total_r": round(sum(rs), 2),
        "max_drawdown_r": round(max_dd, 2),
        "longest_losing_streak": longest,
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# حساسیت پارامتر (افزودنی — ایده‌ی param_grid از پک خارجی؛ فقط داخل همین ماژول)
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
    امتیاز یک واریانت وزن: سهم هر مؤلفه متناسب با نسبت «سقف جدید/سقف پایه»
    مقیاس می‌شود. کلیدهای غایب در weights یعنی سقف پایه. جهت/ورود/حد ضرر/
    هدف به امتیاز کل وابسته نیستند، پس فقط «انتخاب معامله‌ها» عوض می‌شود.
    """
    total = 0.0
    for name, base_max in _SCORE_MAXES.items():
        if not base_max:
            continue
        new_max = float(weights.get(name, base_max))
        total += (components.get(name) or 0.0) * (new_max / base_max)
    return round(total, 1)


# ---------------------------------------------------------------------------
# اجرای اصلی
# ---------------------------------------------------------------------------
def run_system_backtest(symbol: str = "BTC", timeframe: str = "4h",
                        thresholds=(55, 60, 65), step: int = None,
                        max_hold: int = None, df=None,
                        min_history: int = None, progress: bool = False,
                        param_grid: list = None,
                        trailing: dict = None,
                        include_trailing_grid: bool = False) -> dict:
    """
    بک‌تست سیستمی یک ارز/تایم‌فریم. اگر df داده نشود، از بایننس گرفته می‌شود.
    خروجی: دیکشنری شامل متادیتا و برای هر آستانه: معیارهای all/in/out + لاگ معامله‌ها.

    param_grid (افزودنی، پیش‌فرض None = رفتار قبلی دقیقاً): لیستی از
    دیکشنری‌های بازنویسی سقف مؤلفه‌ها، مثل {"pattern": 10.0} یا
    {"trend": 35.0, "volume": 5.0}. برای هر واریانت، همان معامله‌های
    شبیه‌سازی‌شده با امتیاز واریانت دوباره انتخاب و معیارهای in/out حساب
    می‌شود و زیر کلید "param_grid" برمی‌گردد (همراه پرچم beats_base_out
    برای هر آستانه). پیاده‌سازی کاملاً داخل همین ماژول است؛ scanner دست نخورده.

    trailing (افزودنی، پیش‌فرض None = دقیقاً رفتار قبلی): دیکشنری
    {"activation_r": A, "trail_r": D} که به شبیه‌سازی همه‌ی معامله‌های
    همین اجرا داده می‌شود (قواعد در docstring ماژول). با trailing، رکورد
    هر معامله علاوه بر فیلدهای قبلی کلید "exit_stop" و پرچم "trailed"
    دارد و نتیجه در کلید "trailing" بالای خروجی ثبت می‌شود.

    include_trailing_grid (افزودنی، پیش‌فرض False): اگر True باشد، علاوه
    بر اجرای اصلی (بدون تریلینگ)، گرید کوچک config.SYSBT_TRAILING_GRID
    هم روی همان ارزیابی‌ها اجرا و زیر کلید "trailing_grid" برگردانده
    می‌شود (برای هر واریانت و هر آستانه: معیارهای in/out/all، بدون لاگ
    معامله برای سبک‌ماندن خروجی). انتخاب واریانت در تست A/B فقط با
    داخل‌نمونه انجام می‌شود — ابزار انتخاب در TRAILING_AB_FA.md است.
    """
    if step is None:
        step = config.SYSBT_STEP
    if max_hold is None:
        # برنامه‌ی بهبود (افزودنی): اگر مرحله‌ی ۳ مقدار پذیرفته‌شده‌ای
        # ثبت کرده باشد (IMPROVE_SYSBT_MAX_HOLD)، پیش‌فرض همان است؛
        # وگرنه دقیقاً مقدار قبلی SYSBT_MAX_HOLD_CANDLES. پاس‌دادن صریح
        # max_hold همیشه بر هر دو مقدم است.
        max_hold = (getattr(config, "IMPROVE_SYSBT_MAX_HOLD", None)
                    or config.SYSBT_MAX_HOLD_CANDLES)
    if min_history is None:
        min_history = config.SYSBT_MIN_HISTORY
    if df is None:
        df = analyzer.fetch_candles_binance(symbol, timeframe)

    n = len(df)
    split_pos = int(n * config.SYSBT_INSAMPLE_FRAC)
    last_entry_pos = n - 1 - max_hold  # بعد از این نقطه پنجره‌ی کامل معامله در داده نیست

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
        # جهت باید با جای حد ضرر سازگار باشد، وگرنه معامله بی‌معناست
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
            # فیلدهای افزودنی فقط وقتی تریلینگ فعال است (مسیر پیش‌فرض دست نخورده)
            record["exit_stop"] = exit_price if outcome in ("stop", "trailing") else None
            record["trailed"] = outcome == "trailing"
        for th in thresholds:
            if cand["score"] >= th:
                per_threshold[th].append(record)
        # واریانت‌های وزن: همان معامله، با امتیاز بازمقیاس‌شده انتخاب می‌شود
        for vi, weights in enumerate(param_grid or []):
            v_score = _variant_score(cand["components"], weights)
            v_record = dict(record, score=v_score)
            for th in thresholds:
                if v_score >= th:
                    variant_trades[vi][th].append(v_record)
        # گرید تریلینگ: همان ورودها، شبیه‌سازی دوباره با هر واریانت
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
    """خلاصه‌ی متنی فارسی از نتیجه‌ی بک‌تست سیستمی."""
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
