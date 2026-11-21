"""
backtest/forex_backtest.py
مدل هزینه‌ی فارکس و اجرای بک‌تست سیستمی فارکس با همان پشته‌ی نهایی
کریپتو (تریلینگ پذیرفته‌شده + مجموعه‌امتیاز both/regime + دروازه‌ی EV).

چرا wrapper و نه پارامتر داخل run_system_backtest؟
موتور سیستمی کریپتو هزینه را داخل _trade_r با کارمزد درصدی حساب می‌کند
و دست‌کاری آن رفتار/اعداد کریپتو را به خطر می‌انداخت. اینجا شبیه‌سازی
خود موتور پشته (improvement_runs/stack_engine — همان موتوری که
matrix_final را ساخت) دوباره استفاده می‌شود، با این تفاوت که R هر
معامله از (ورود، خروج، ریسک) خام با هزینه‌ی اسپردی فارکس بازحساب
می‌شود:

    cost_price = spread_pips × pip_size  +  slippage_pips × pip_size
    R = (سود ناخالص قیمتی − cost_price) / |ورود − حد ضرر|

یعنی اسپرد رفت‌وبرگشت + اسلیپیج، در واحد قیمت، عیناً مثل کارمزد در
فرمول _trade_r از سود کم می‌شود. اسپرد هر نماد: config.FOREX_SPREAD_PIPS
(برای XAUUSD مقدار 2.5 «پیپ» 0.1 ـی = 0.25 واحد قیمت = ۲۵ سنت).
سواپ/رول‌اور شبانه عمداً حساب نمی‌شود (نگهداری تا ۴۰ کندل؛ در
FOREX_FA.md به‌عنوان قید ثبت است).

اعداد کریپتو دست نخورده‌اند: این ماژول هیچ تابع موجودی را تغییر
نمی‌دهد و اثباتش در تست‌ها + رگرسیون (بازتولید matrix_final) آمده.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backtest.system_backtest import _metrics

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))            # پوشه‌ی crypto_analyzer_v2/
_IMPROVEMENT = os.path.join(_ROOT, "improvement_runs")


# ---------------------------------------------------------------------------
# مدل هزینه
# ---------------------------------------------------------------------------
def forex_cost_price(symbol: str) -> float:
    """هزینه‌ی رفت‌وبرگشت یک معامله‌ی فارکس در واحد قیمت (اسپرد+اسلیپیج)."""
    sym = str(symbol).upper()
    pips = (config.FOREX_SPREAD_PIPS[sym]
            + getattr(config, "FOREX_SLIPPAGE_PIPS", 0.0))
    return pips * config.FOREX_PIP_SIZE[sym]


def forex_trade_r(direction: str, entry: float, exit_price: float,
                  risk: float, symbol: str) -> float:
    """بازده معامله بر حسب R با هزینه‌ی اسپردی فارکس (معادل _trade_r)."""
    if risk <= 0:
        raise ValueError("risk باید مثبت باشد.")
    gross = (exit_price - entry) if direction == "long" else (entry - exit_price)
    return (gross - forex_cost_price(symbol)) / risk


def reprice_records(records: list, symbol: str) -> list:
    """
    رکوردهای فشرده‌ی موتور پشته ({p, s, mh:{hold:{r, o, entry, exit,
    risk, dir}}}) را با هزینه‌ی فارکس دوباره قیمت‌گذاری می‌کند: فقط
    کلید r عوض می‌شود؛ بقیه عین رکورد ورودی است.
    """
    out = []
    for rec in records:
        new = dict(rec)
        mh = {}
        for hold, sim in rec["mh"].items():
            r = forex_trade_r(sim["dir"], sim["entry"], sim["exit"],
                              sim["risk"], symbol)
            mh[hold] = {"r": round(r, 3), "o": sim["o"]}
        new["mh"] = mh
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# دروازه‌ی EV (همان سمانتیک stage1_gate / stage4_matrix)
# ---------------------------------------------------------------------------
def _kept_bars(base_records: list, threshold: int, max_hold: int,
               bar_candles: int = 100) -> set:
    """بارهایی که دروازه‌ی on نگه می‌دارد (آمار هر بار فقط از بارهای قبلی)."""
    def sim_of(rec):
        return rec["mh"].get(str(max_hold)) or rec["mh"].get(max_hold)
    by_bar = {}
    for rec in base_records:
        if rec["s"] >= threshold:
            by_bar.setdefault(rec["p"] // bar_candles, []).append(sim_of(rec)["r"])
    kept, prior = set(), []
    for b in sorted(by_bar):
        n = len(prior)
        exp = (sum(prior) / n) if n else None
        if n < config.IMPROVE_GATE_N_MIN or (exp is not None and exp >= 0):
            kept.add(b)
        prior.extend(by_bar[b])
    return kept


def build_forex_result(symbol: str, timeframe: str, sets: dict,
                       meta: dict, df, thresholds=(55, 60, 65),
                       max_hold: int = 40) -> dict:
    """
    از مجموعه‌امتیازهای قیمت‌گذاری‌شده با هزینه‌ی فارکس، نتیجه‌ی هم‌قالبِ
    matrix_final می‌سازد: برای هر آستانه، معیارهای all/in/out + لاگ
    معامله با segment. مجموعه‌امتیاز: 4h → "both"، 1d → "regime"
    (دقیقاً قرارداد stage4_matrix) و دروازه روی سایه‌ی "base".
    """
    symbol = symbol.upper()
    sets = {name: reprice_records(recs, symbol) for name, recs in sets.items()}
    score_set = "both" if timeframe == "4h" else "regime"
    split_pos = meta["split_pos"]
    out_thresholds = {}
    for th in thresholds:
        kb = _kept_bars(sets["base"], th, max_hold)
        trades = []
        for rec in sets[score_set]:
            if rec["s"] < th or (rec["p"] // 100) not in kb:
                continue
            sim = rec["mh"].get(str(max_hold)) or rec["mh"].get(max_hold)
            trades.append({
                "entry_index": rec["p"],
                "entry_time": str(df.index[rec["p"]]),
                "segment": "in" if rec["p"] < split_pos else "out",
                "score": rec["s"],
                "outcome": sim["o"],
                "r": sim["r"],
            })
        stripped = [{"r": t["r"], "outcome": t["outcome"]} for t in trades]
        out_thresholds[str(th)] = {
            "all": _metrics(stripped),
            "in_sample": _metrics([{"r": t["r"], "outcome": t["outcome"]}
                                   for t in trades if t["segment"] == "in"]),
            "out_sample": _metrics([{"r": t["r"], "outcome": t["outcome"]}
                                    for t in trades if t["segment"] == "out"]),
            "trades": trades,
        }
    return {
        "symbol": symbol, "timeframe": timeframe, "candles": len(df),
        "evaluations": meta["evaluations"], "eval_errors": meta["eval_errors"],
        "step": config.SYSBT_STEP, "max_hold": max_hold,
        "split_at": str(df.index[split_pos]),
        "stack": {"trailing": config.SYSBT_TRAILING_ADOPTED,
                  "gate": {"mode": "on", "n_min": config.IMPROVE_GATE_N_MIN,
                           "bar_candles": 100,
                           "bookkeeping": "base pipeline shadow trades, prior bars only"},
                  "score_set": score_set,
                  "cost_model": "forex spread (config.FOREX_SPREAD_PIPS) + "
                                "slippage, no swap",
                  "note": "final stack after improvement program, forex cost model"},
        "thresholds": out_thresholds,
    }


def evaluate_forex_coin(symbol: str, df_main, df_daily=None):
    """
    ارزیابی یک ترکیب فارکس با موتور پشته (improvement_runs/stack_engine)
    و غنی‌سازی رکوردها با فیلدهای خام (entry/exit/risk/dir) برای
    قیمت‌گذاری فارکس در build_forex_result.

    روش: _simulate_trade موتور پشته در طول ارزیابی با یک wrapper جایگزین
    می‌شود که امضای اصلی را دقیق نگه می‌دارد و فقط (قیمت خروج، ورود،
    ریسک، جهت) هر شبیه‌سازی را بر اساس (اندیس ورود، max_hold) ضبط
    می‌کند؛ بعد از ارزیابی، خام‌ها داخل رکوردها ادغام می‌شوند. محاسبه‌ی
    خود موتور (از جمله R کریپتوییِ موقت) دست نمی‌خورد و در مرحله‌ی
    reprice با هزینه‌ی فارکس جایگزین می‌شود.
    موتور پشته در مخزن توسعه (improvement_runs) است، نه داخل پکیج؛ اگر
    در دسترس نبود خطای روشن می‌دهد (مسیرهای زنده به این تابع وابسته
    نیستند).
    """
    if _IMPROVEMENT not in sys.path:
        sys.path.insert(0, _IMPROVEMENT)
    try:
        import stack_engine
    except ImportError as exc:
        raise RuntimeError(
            "موتور پشته (improvement_runs/stack_engine) در دسترس نیست؛ "
            "بک‌تست فارکس به آن نیاز دارد.") from exc

    raw = {}
    orig_sim = stack_engine._simulate_trade

    def recording_sim(df, entry_pos, direction, entry, stop, target,
                      max_hold, trailing=None):
        xp, oc = orig_sim(df, entry_pos, direction, entry, stop, target,
                          max_hold, trailing=trailing)
        raw[(entry_pos, max_hold)] = {"entry": entry, "exit": xp,
                                      "risk": abs(entry - stop),
                                      "dir": direction}
        return xp, oc

    stack_engine._simulate_trade = recording_sim
    try:
        sets, meta = stack_engine.evaluate_coin(symbol, df_main, df_daily)
    finally:
        stack_engine._simulate_trade = orig_sim

    missing = 0
    for recs in sets.values():
        for rec in recs:
            for hold, sim in rec["mh"].items():
                info = raw.get((rec["p"], int(hold)))
                if info is None:
                    missing += 1
                    continue
                sim.update(info)
    if missing:
        raise RuntimeError(
            f"{missing} شبیه‌سازی بدون فیلد خام ماند؛ قیمت‌گذاری فارکس "
            f"ناقص می‌شود و ادامه امن نیست.")
    return sets, meta
