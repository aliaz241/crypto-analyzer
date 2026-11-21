"""
scanner.py
اسکنر چندارزی بازار: برای هر جفت (ارز × تایم‌فریم) از لیست config.SCAN_SYMBOLS
داده را از بایننس می‌گیرد، همان گزارش کامل analyzer را می‌سازد و یک امتیاز
شفاف ۰ تا ۱۰۰ حساب می‌کند تا بشود نامزدها را رتبه‌بندی کرد و بهترین را
به‌عنوان «پیشنهاد امروز» نشان داد.

مؤلفه‌های امتیاز (همه‌ی سقف‌ها و ثابت‌ها در config.py هستند):
- روند (۲۵): هم‌ترازی قیمت با EMA50 و EMA200 + جهت EMA50 نسبت به EMA200؛
  با قدرت روند (ADX) مقیاس می‌شود. علامت این مؤلفه جهت معامله را تعیین می‌کند.
- مومنتوم (۲۰): هم‌جهتی هیستوگرام MACD با جهت + قرارگیری RSI در ناحیه‌ی سالم
  همان جهت؛ مقادیر افراطی RSI (خرید/فروش اشباع) جریمه می‌شوند.
- الگو (۲۵): بهترین الگوی تأییدشده‌ی اخیر (کندلی یا نموداری) که هم‌جهت است،
  بر اساس نرخ موفقیت بک‌تست همان الگو روی همین ارز/تایم‌فریم نمره می‌گیرد.
- سودبه‌ریسک (۲۰): فاصله تا اولین سطح مقابل (هدف) نسبت به فاصله تا حد ضرر
  پیشنهادی.
- حجم (۱۰): هم‌جهتی روند OBV با جهت معامله.

⚠️ امتیاز بالا «تضمین سود» نیست؛ فقط یعنی در این لحظه، از بین گزینه‌های اسکن‌شده
این ترکیب ارز/تایم‌فریم/جهت از نظر همین قواعد مکانیکی وضعیت بهتری دارد.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
from analysis import analyzer
from analysis import ev as ev_mod
from analysis import llm_client as llm_client_mod
from analysis import llm_engine as llm_engine_mod
from indicators import trend as trend_ind
from indicators import volatility as vol_ind


# کش آخرین اسکن: (symbol, timeframe) -> (df, report)
# برای این‌که main بتواند نمودار پیشنهاد اول را بدون گرفتن دوباره‌ی داده رسم کند.
_SCAN_CACHE = {}


def get_scan_cache(symbol: str, timeframe: str):
    """(df, report) ذخیره‌شده‌ی یک نامزد از آخرین اسکن را برمی‌گرداند یا None."""
    return _SCAN_CACHE.get((symbol.upper(), timeframe))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# مؤلفه‌ها
# ---------------------------------------------------------------------------
def _trend_signed_score(df, ind) -> float:
    """
    امتیاز روند با علامت: مثبت = تمایل لانگ، منفی = تمایل شورت، بازه‌ی [-۲۵, +۲۵].
    سه شرط هم‌ترازی (هرکدام ±۱): قیمت بالای EMA50، قیمت بالای EMA200،
    EMA50 بالای EMA200. بعد با قدرت روند (ADX) مقیاس می‌شود: ADX>=25 مقیاس
    کامل، ADX پایین‌تر مقیاس خطی کمتر (کف ۰٫۵)، ADX ناموجود مقیاس ۰٫۸.
    """
    price = ind["price"]
    ema50 = trend_ind.ema(df, 50).iloc[-1]
    ema200 = trend_ind.ema(df, 200).iloc[-1]
    if price is None or ema50 != ema50 or ema200 != ema200:  # NaN check
        return 0.0

    align = 0.0
    align += 1 if price > ema50 else (-1 if price < ema50 else 0)
    align += 1 if price > ema200 else (-1 if price < ema200 else 0)
    align += 1 if ema50 > ema200 else (-1 if ema50 < ema200 else 0)
    raw = (align / 3.0) * config.SCORE_TREND_MAX

    adx = ind.get("adx")
    if adx is None:
        factor = 0.8
    elif adx >= 25:
        factor = 1.0
    else:
        factor = 0.5 + 0.5 * (adx / 25.0)
    return raw * factor


def _momentum_score(ind, direction: str) -> float:
    """امتیاز مومنتوم ۰ تا ۲۰ برای جهت داده‌شده (MACD هم‌جهت ۱۲ + ناحیه‌ی RSI تا ۸)."""
    score = 0.0
    hist = ind["macd"].get("histogram")
    if hist is None:
        score += config.SCORE_MOMENTUM_MAX * 0.3  # نامشخص: نصف بخش MACD
    elif (direction == "long" and hist > 0) or (direction == "short" and hist < 0):
        score += config.SCORE_MOMENTUM_MAX * 0.6

    rsi = ind.get("rsi")
    if rsi is None:
        score += 4.0
    else:
        # برای شورت، RSI را آینه می‌کنیم تا یک جدول واحد برای هر دو جهت کافی باشد
        eff = rsi if direction == "long" else 100.0 - rsi
        if 45 <= eff <= 65:
            score += 8.0    # ناحیه‌ی سالم روند
        elif 40 <= eff < 45 or 65 < eff <= 70:
            score += 5.0
        elif 35 <= eff < 40 or 70 < eff <= 75:
            score += 3.0
        elif eff > 75:
            score += 0.0    # اشباع افراطی علیه ورود تازه در این جهت
        else:  # eff < 35
            score += 2.0
    return min(score, config.SCORE_MOMENTUM_MAX)


def _pattern_score(report, direction: str):
    """
    بهترین الگوی اخیر هم‌جهت را پیدا می‌کند و بر اساس بک‌تست خودش روی همین
    ارز/تایم‌فریم نمره می‌دهد. خروجی: (امتیاز, اطلاعات الگو یا None)
    نگاشت نرخ موفقیت: ۳۰٪ -> ۰ و ۷۰٪ -> ۲۵ (خطی)، ضرب در min(1, رخداد/۱۰).
    """
    want_bias = "bullish" if direction == "long" else "bearish"
    best = None

    def _consider(pattern, stats, kind):
        nonlocal best
        if not stats or not stats.get("total_occurrences"):
            return
        rate = stats["success_rate_pct"]
        occ = stats["total_occurrences"]
        span = config.SCORE_PATTERN_RATE_CEIL - config.SCORE_PATTERN_RATE_FLOOR
        pts = config.SCORE_PATTERN_MAX * _clamp(
            (rate - config.SCORE_PATTERN_RATE_FLOOR) / span, 0.0, 1.0)
        pts *= min(1.0, occ / 10.0)
        if best is None or pts > best[0]:
            best = (pts, {
                "name": pattern["name"], "fa": pattern["fa"], "kind": kind,
                "bias": pattern["bias"],
                "success_rate_pct": rate,
                "total_occurrences": occ,
                "reliable_sample": stats.get("reliable_sample", False),
            })

    bt = report.get("backtest", {})
    for p in report.get("latest_chart_patterns", []):
        if p["bias"] == want_bias:
            _consider(p, bt.get("chart", {}).get(p["name"]), "chart")
    for p in report.get("latest_candlestick_patterns", []):
        if p["bias"] == want_bias:
            _consider(p, bt.get("candlestick", {}).get(p["name"]), "candlestick")

    if best is None:
        return 0.0, None
    return best[0], best[1]


def _rr_score(report, direction: str, price: float):
    """
    امتیاز سودبه‌ریسک ۰ تا ۲۰ + (نسبت محاسبه‌شده, سطح هدف).
    هدف = اولین سطح مقابلِ قیمت (مقاومت برای لانگ، حمایت برای شورت)؛
    ریسک = فاصله تا حد ضرر پیشنهادی همان جهت. اگر هرکدام موجود نباشد
    امتیاز ثابت config.SCORE_RR_MISSING داده می‌شود.
    """
    sr = report.get("support_resistance", {})
    if direction == "long":
        levels = sr.get("resistance") or []
        stop_data = (report.get("stop_loss_long") or {}).get("recommended") or {}
    else:
        levels = sr.get("support") or []
        stop_data = (report.get("stop_loss_short") or {}).get("recommended") or {}

    target = levels[0]["level"] if levels else None
    stop = stop_data.get("stop_loss")

    if target is None or stop is None:
        return config.SCORE_RR_MISSING, None, target

    if direction == "long":
        reward, risk = target - price, price - stop
    else:
        reward, risk = price - target, stop - price
    if reward <= 0 or risk <= 0:
        return config.SCORE_RR_MISSING, None, target

    rr = reward / risk
    pts = config.SCORE_RR_MAX * _clamp((rr - 0.5) / 2.0, 0.0, 1.0)
    return pts, rr, target


def _volume_score(ind, direction: str) -> float:
    """روند OBV هم‌جهت: ۱۰ | نامشخص: ۵ | مخالف: ۰."""
    obv = ind.get("obv_trend")
    if obv is None:
        return config.SCORE_VOLUME_MAX / 2.0
    if (direction == "long" and obv == "up") or (direction == "short" and obv == "down"):
        return config.SCORE_VOLUME_MAX
    return 0.0


# ---------------------------------------------------------------------------
# گام ۳ — طبقه‌بندی رژیم بازار (تابع جدید؛ فقط با apply_regime=True استفاده می‌شود)
# ---------------------------------------------------------------------------
def classify_regime(df, report) -> str:
    """
    رژیم بازار را برمی‌گرداند: "trend" | "range" | "volatile".

    قواعد (همه‌ی ثابت‌ها در config):
    - volatile: صدک پهنای باند بولینگر فعلی در پنجره‌ی REGIME_BBW_WINDOW کندل
      اخیر حداقل REGIME_BBW_VOLATILE_PERCENTILE باشد. اول بررسی می‌شود، چون
      در نوسان افراطی حتی روندها هم قابل اتکا نیستند.
    - trend: ADX حداقل REGIME_ADX_TREND_MIN و شیب EMA200 روی
      REGIME_EMA200_SLOPE_LOOKBACK کندل حداقل REGIME_EMA200_SLOPE_MIN_PCT
      درصد قدرمطلق داشته باشد. اگر EMA200 هنوز داده‌ی کافی نداشته باشد،
      فقط ADX ملاک است.
    - range: بقیه‌ی حالت‌ها.
    """
    ind = (report or {}).get("indicators") or {}

    try:
        bb = vol_ind.bollinger_bands(df)
        width = ((bb["upper"] - bb["lower"]) / bb["middle"]).dropna()
        if len(width) >= 20:
            window = width.tail(config.REGIME_BBW_WINDOW)
            pct = float((window <= width.iloc[-1]).mean() * 100.0)
            if pct >= config.REGIME_BBW_VOLATILE_PERCENTILE:
                return "volatile"
    except Exception:
        pass

    adx = ind.get("adx")
    if adx is not None and adx >= config.REGIME_ADX_TREND_MIN:
        try:
            ema200 = trend_ind.ema(df, 200).dropna()
            lb = config.REGIME_EMA200_SLOPE_LOOKBACK
            if len(ema200) > lb and ema200.iloc[-1 - lb] > 0:
                slope_pct = (ema200.iloc[-1] / ema200.iloc[-1 - lb] - 1.0) * 100.0
                if abs(slope_pct) < config.REGIME_EMA200_SLOPE_MIN_PCT:
                    return "range"
        except Exception:
            pass
        return "trend"
    return "range"


# ---------------------------------------------------------------------------
# گام ۴ — امتیاز مؤلفه‌ی ساختار (فقط وقتی extras به score_candidate داده شود)
# ---------------------------------------------------------------------------
def _structure_score(extras: dict, direction: str) -> float:
    """
    از خروجی analysis.structure.analyze_structure یک امتیاز در بازه‌ی
    [-STRUCTURE_MAX, +STRUCTURE_MAX] می‌سازد: هم‌جهتی هر سیگنال ساختاری با
    جهت نامزد امتیاز مثبت و مخالفتش امتیاز منفی می‌دهد (واگرایی RSI: ±۴،
    واگرایی MACD: ±۳، آخرین رویداد BOS/CHoCH: ±۳ — ثابت‌ها در config).
    """
    def _aligned(bias) -> float:
        if bias is None:
            return 0.0
        want = "bullish" if direction == "long" else "bearish"
        return 1.0 if bias == want else -1.0

    pts = 0.0
    pts += _aligned(extras.get("rsi_divergence")) * config.STRUCTURE_RSI_DIV_POINTS
    pts += _aligned(extras.get("macd_divergence")) * config.STRUCTURE_MACD_DIV_POINTS
    event = extras.get("last_event") or {}
    pts += _aligned(event.get("direction")) * config.STRUCTURE_EVENT_POINTS
    return max(-config.STRUCTURE_MAX, min(config.STRUCTURE_MAX, pts))


# ---------------------------------------------------------------------------
# گام ۶ — حجم پوزیشن (فقط محاسبه‌ی افزودنی روی خروجی‌های موجود)
# ---------------------------------------------------------------------------
def _position_sizing(report, entry, stop):
    """
    حجم پیشنهادی با ریسک ثابت: مبلغ ریسک = RISK_PER_TRADE_PCT درصد از
    ACCOUNT_SIZE_USDT (placeholder — نه موجودی واقعی کاربر) و حجم =
    مبلغ ریسک / |ورود − حد ضرر|. ATR گزارش هم برای اطلاع کنارش می‌آید.
    اگر ورود/حد ضرر نباشد None برمی‌گردد.
    """
    if entry is None or stop is None:
        return None
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return None
    risk_amount = config.ACCOUNT_SIZE_USDT * config.RISK_PER_TRADE_PCT / 100.0
    size = risk_amount / risk_per_unit
    return {
        "account_size_usdt": config.ACCOUNT_SIZE_USDT,
        "risk_pct": config.RISK_PER_TRADE_PCT,
        "risk_amount_usdt": round(risk_amount, 2),
        "size_units": round(size, 6),
        "notional_usdt": round(size * entry, 2),
        "atr": (report.get("indicators") or {}).get("atr"),
    }


# ---------------------------------------------------------------------------
# امتیازدهی یک نامزد
# ---------------------------------------------------------------------------
def score_candidate(df, report, symbol: str, timeframe: str,
                    higher_tf_direction: str = None,
                    apply_regime: bool = False,
                    extras: dict = None) -> dict:
    """
    پارامترهای افزودنی (همه پیش‌فرضِ «خاموش» دارند و مسیر پیش‌فرض دقیقاً
    رفتار قبلی است):
    - higher_tf_direction (گام ۲): جهت تایم‌فریم بالاتر ("long"/"short")؛
      اگر با جهت نامزد مخالف باشد، امتیاز کل در MTF_CONFLICT_FACTOR ضرب و
      کلید "mtf" به خروجی اضافه می‌شود.
    - apply_regime (گام ۳): اگر True، مؤلفه‌ها بر اساس classify_regime
      مقیاس می‌شوند و کلید "regime" اضافه می‌شود.
    - extras (گام ۴): خروجی analysis.structure.analyze_structure؛ اگر داده
      شود مؤلفه‌ی "structure" (تا ±۱۰) به امتیاز اضافه می‌شود — در این
      حالت مقیاس امتیاز /۱۱۰ است، نه /۱۰۰. مسیر پیش‌فرض /۱۰۰ می‌ماند.
    """
    ind = report["indicators"]
    price = ind["price"]

    trend_signed = _trend_signed_score(df, ind)
    if abs(trend_signed) >= config.SCORE_TREND_WASH_THRESHOLD:
        direction = "long" if trend_signed > 0 else "short"
    else:
        # روند خنثی است؛ جهت از علامت هیستوگرام MACD گرفته می‌شود
        hist = ind["macd"].get("histogram")
        if hist is not None and hist != 0:
            direction = "long" if hist > 0 else "short"
        else:
            ema50 = trend_ind.ema(df, 50).iloc[-1]
            direction = "long" if price >= ema50 else "short"

    dir_sign = 1.0 if direction == "long" else -1.0
    trend_pts = max(0.0, trend_signed * dir_sign)
    momentum_pts = _momentum_score(ind, direction)
    pattern_pts, pattern_info = _pattern_score(report, direction)
    rr_pts, rr_value, target = _rr_score(report, direction, price)
    volume_pts = _volume_score(ind, direction)

    stop_data = (report.get("stop_loss_long" if direction == "long"
                            else "stop_loss_short") or {}).get("recommended") or {}
    stop = stop_data.get("stop_loss")

    # --- گام ۳: مقیاس رژیم (فقط با apply_regime=True؛ پیش‌فرض خاموش) ---
    regime_name = None
    if apply_regime:
        regime_name = classify_regime(df, report)
        if regime_name == "range":
            trend_pts = trend_pts * config.REGIME_RANGE_TREND_SCALE
            momentum_pts = min(momentum_pts * config.REGIME_RANGE_MOMENTUM_SCALE,
                               config.SCORE_MOMENTUM_MAX)
        elif regime_name == "volatile":
            pattern_pts = pattern_pts * config.REGIME_VOLATILE_PATTERN_SCALE

    components = {
        "trend": round(trend_pts, 1),
        "momentum": round(momentum_pts, 1),
        "pattern": round(pattern_pts, 1),
        "risk_reward": round(rr_pts, 1),
        "volume": round(volume_pts, 1),
    }
    total = round(sum(components.values()), 1)

    # دلیل یک‌خطی: دو مؤلفه‌ای که بیشترین سهم نسبی از سقف خودشان را گرفته‌اند
    fragments = {
        "trend": "روند هم‌سو با EMA50/200" + (f" (ADX {ind['adx']})" if ind.get("adx") is not None else ""),
        "momentum": "مومنتوم هم‌جهت (MACD و RSI)",
        "pattern": (f"الگوی {pattern_info['fa']} با موفقیت تاریخی "
                    f"{pattern_info['success_rate_pct']}٪ روی همین تایم‌فریم") if pattern_info else "",
        "risk_reward": (f"سودبه‌ریسک {rr_value:.1f} تا سطح بعدی") if rr_value else "سطح مقابل نزدیک",
        "volume": "تأیید حجم (OBV هم‌جهت)",
    }
    maxes = {"trend": config.SCORE_TREND_MAX, "momentum": config.SCORE_MOMENTUM_MAX,
             "pattern": config.SCORE_PATTERN_MAX, "risk_reward": config.SCORE_RR_MAX,
             "volume": config.SCORE_VOLUME_MAX}
    ranked = sorted(components.items(),
                    key=lambda kv: kv[1] / maxes[kv[0]] if maxes[kv[0]] else 0,
                    reverse=True)
    top_fragments = [fragments[k] for k, v in ranked[:2] if v > 0 and fragments[k]]
    reason = "دلیل: " + (" + ".join(top_fragments) if top_fragments else "سیگنال مشخصی ندارد")

    # --- گام ۴: مؤلفه‌ی ساختار (فقط وقتی extras داده شود؛ مقیاس /۱۱۰) ---
    # بعد از محاسبه‌ی «دلیل» اضافه می‌شود تا منطق رتبه‌بندی دلیل (که فقط پنج
    # مؤلفه‌ی اصلی را می‌شناسد) دست نخورد.
    if extras is not None:
        components["structure"] = round(_structure_score(extras, direction), 1)
        total = round(total + components["structure"], 1)

    # --- گام ۲: دروازه‌ی MTF (فقط وقتی جهت تایم‌فریم بالاتر داده شود) ---
    mtf_info = None
    if higher_tf_direction is not None:
        conflict = higher_tf_direction != direction
        if conflict:
            total = round(total * config.MTF_CONFLICT_FACTOR, 1)
        mtf_info = {"higher_tf_direction": higher_tf_direction,
                    "conflict": conflict,
                    "factor": config.MTF_CONFLICT_FACTOR if conflict else 1.0}

    result = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "direction": direction,
        "direction_fa": "لانگ (خرید)" if direction == "long" else "شورت (فروش)",
        "score": total,
        "components": components,
        "price": price,
        "entry": price,
        "stop": stop,
        "target": target,
        "risk_reward": round(rr_value, 2) if rr_value else None,
        "pattern": pattern_info,
        "reason": reason,
    }
    if mtf_info is not None:
        result["mtf"] = mtf_info
    if regime_name is not None:
        result["regime"] = regime_name
    if extras is not None:
        result["structure"] = extras
    # گام ۶: حجم پوزیشن — کلید افزودنی، محاسبه از همان ورود/حد ضرر موجود
    result["sizing"] = _position_sizing(report, result["entry"], result["stop"])
    return result


# ---------------------------------------------------------------------------
# همبستگی آلت‌کوین با BTC (افزودنی — پورت از پک خارجی؛ فقط نمایشی/تحلیلی)
# ---------------------------------------------------------------------------
def _btc_corr_from_frames(alt_df, btc_df, window: int = 30):
    """
    همبستگی پیرسونِ «بازده‌ها» (نه قیمت خام) بین بسته‌شدن‌های آلت و BTC روی
    آخرین `window` بازده‌ی هم‌تراز. همبستگی روی قیمت خام به‌خاطر روند مشترک
    گمراه‌کننده است. خروجی گرد‌شده به ۳ رقم، یا None اگر داده کافی نیست.
    نزدیک ۱ یعنی آلت عملاً بتای BTC است؛ نزدیک ۰/منفی یعنی رفتار مستقل‌تر.
    """
    try:
        joined = pd.DataFrame({"alt": alt_df["close"].astype(float),
                               "btc": btc_df["close"].astype(float)}).dropna()
        if len(joined) < window + 1:
            return None
        rets = joined.pct_change().dropna().tail(window)
        if len(rets) < 10:
            return None
        if rets["alt"].std() == 0 or rets["btc"].std() == 0:
            return None
        return round(float(rets["alt"].corr(rets["btc"])), 3)
    except Exception:
        return None


def alt_btc_correlation(symbol: str, timeframe: str, window: int = 30):
    """
    آخرین همبستگی بازده‌های `symbol` با BTC در همان تایم‌فریم (پنجره‌ی ۳۰).
    برای خود BTC مقدار 1.0 برمی‌گردد. روی هر خطا (شبکه/داده‌ی ناکافی) None.
    """
    try:
        if symbol.upper() == "BTC":
            return 1.0
        need = window + 60   # حاشیه برای هم‌ترازی زمانی و pct_change
        alt_df = analyzer.fetch_candles_binance(symbol, timeframe,
                                                total_candles=need)
        btc_df = analyzer.fetch_candles_binance("BTC", timeframe,
                                                total_candles=need)
        return _btc_corr_from_frames(alt_df, btc_df, window)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# لایه‌ی تصمیم EV (افزودنی — analysis/ev.py)
# ---------------------------------------------------------------------------
def _ev_gate_annotation(cand: dict):
    """
    حاشیه‌ی افزودنی دروازه‌ی EV واک‌فوروارد (برنامه‌ی بهبود، مرحله‌ی ۱ —
    پذیرفته‌شده با حالت config.IMPROVE_GATE_MODE) روی کلید "ev_gate".

    فرم زنده‌ی دروازه: آمار «تحقق‌یافته‌ی دنباله‌ای» یک ترکیب همان
    EV خارج‌نمونه‌ی کش است (از ماتریس نهایی پشته)؛ دروازه وقتی سرکوب
    می‌کند که n ≥ IMPROVE_GATE_N_MIN و امید تحقق‌یافته زیر لبه باشد
    (حالت on: ‎<۰‎؛ smooth: ‎<−۰٫۲R‎). این حاشیه اطلاعاتی است و تصمیم
    واقعی را decide_candidate می‌گیرد — قواعد اجباری آن (نمونه‌ی ناکافی
    ‎n<۱۰‎ → NO_TRADE و امید ‎≤۰‎ → NO_TRADE) در هر نقطه‌ای که دروازه
    سرکوب می‌کند، از قبل همان NO_TRADE را می‌دهند (لایه‌ی زنده در بُعد
    حجم نمونه سخت‌گیرتر از دروازه است: دروازه از ‎n≥۸‎ فعال می‌شود ولی
    تصمیم خرید در لایه‌ی EV به ‎n≥۱۰‎ برای هر تصمیم و ‎n≥۳۰‎ برای
    BUY/STRONG نیاز دارد). اختلاف این دو فرم در
    IMPROVEMENT_PROGRAM_FA.md ثبت شده است.
    """
    mode = getattr(config, "IMPROVE_GATE_MODE", None)
    if not mode:
        return
    ev = cand.get("ev") or {}
    n = ev.get("n_trades") or 0
    exp = ev.get("expectancy_r")
    edge = 0.0 if mode == "on" else config.IMPROVE_GATE_SMOOTH_EDGE_R
    suppressed = bool(exp is not None and n >= config.IMPROVE_GATE_N_MIN
                      and exp < edge)
    cand["ev_gate"] = {"mode": mode, "n_min": config.IMPROVE_GATE_N_MIN,
                       "trailing_n": n, "trailing_expectancy_r": exp,
                       "suppressed": suppressed}


def attach_ev_decision(cand: dict) -> dict:
    """
    کلیدهای افزودنی "ev" و "decision"/"decision_fa"/"decision_reason" را
    به یک نامزد اضافه می‌کند. EV فقط از کش/فایل‌های ذخیره‌شده خوانده
    می‌شود (allow_compute=False) — این تابع هرگز بک‌تست اجرا نمی‌کند تا
    اسکن و مسیرهای زنده کند نشوند. اگر نامزد از قبل هر دو کلید را داشته
    باشد (مثلاً stub تست یا صدا‌زننده‌ی بالادستی)، دست نمی‌خورد.
    روی هر خطا: ev=None و تصمیم NO_TRADE با دلیل صریح (fail-safe).
    کلید افزودنی "ev_gate" (حاشیه‌ی دروازه‌ی مرحله‌ی ۱ برنامه‌ی بهبود)
    در هر دو مسیر اضافه می‌شود؛ وقتی IMPROVE_GATE_MODE خالی باشد،
    کلیدی اضافه نمی‌شود و رفتار دقیقاً قبلی است.
    """
    _ev_gate_annotation(cand)
    if "ev" in cand and "decision" in cand:
        return cand
    try:
        ev = ev_mod.ev_for_candidate(cand["symbol"], cand["timeframe"],
                                     direction=cand.get("direction"),
                                     allow_compute=False)
    except Exception:
        ev = None
    cand["ev"] = ev
    try:
        dec = ev_mod.decide_candidate(cand, ev)
    except Exception:
        dec = {"decision": "NO_TRADE",
               "decision_fa": ev_mod.DECISION_FA["NO_TRADE"],
               "decision_reason": "رد: خطا در لایه‌ی تصمیم EV."}
    cand.update(dec)
    _ev_gate_annotation(cand)
    return cand


# ---------------------------------------------------------------------------
# لایه‌ی بازبینی LLM (افزودنی — analysis/llm_engine.py؛ LLM_LAYER_FA.md)
# ---------------------------------------------------------------------------
def llm_layer_active() -> bool:
    """True یعنی لایه‌ی LLM برای یک اسکن قابل استفاده است: پیش‌فرض کانفیگ
    (LLM_ENABLED_DEFAULT) روشن باشد یا صدا‌زننده صریحاً روشنش کرده باشد،
    و یک کلید NVIDIA resolve شود. بدون کلید همیشه False است."""
    if not getattr(config, "LLM_ENABLED_DEFAULT", False):
        return False
    try:
        return bool(llm_client_mod.resolve_api_key())
    except Exception:
        return False


def attach_llm_analysis(cand: dict, report_data_builder=None,
                        client=None) -> dict:
    """
    بازبینی LLM را به‌صورت کلید افزودنی "llm" روی یک نامزد می‌نشاند و
    همان نامزد را برمی‌گرداند (الگوی attach_ev_decision).

    گاردها (همه لازمه‌ی صدا شدن موتور):
    - نامزد حتماً تصمیم EV دارد (attach_ev_decision قبلاً رویش رفته) و
      تصمیمش NO_TRADE نیست — کف سخت: NO_TRADE یعنی صفر فراخوانی LLM
      (موتور هم خودش همین کف را اعمال می‌کند؛ این‌جا گارد دوم است).
    - اگر کلیدی resolve نشود یا موتور None بدهد (لایه غیرفعال)، هیچ
      کلیدی اضافه نمی‌شود و نامزد دقیقاً مثل قبل برمی‌گردد.
    - report_data_builder (اختیاری): تابعی که با خود نامزد صدا زده
      می‌شود و «گزارش محاسباتی» اضافه برای پرامپت مدل می‌سازد؛ اگر
      داده نشود، خود نامزد کل ورودی مدل است. خطای سازنده بی‌صدا رد
      می‌شود (گزارش None) تا اسکن هرگز به‌خاطر آن نخوابد.
    - روی هر خطای دیگر هم نامزد دست‌نخورده برمی‌گردد (fail-safe).
    """
    if cand.get("decision") is None or cand.get("decision") == "NO_TRADE":
        return cand
    try:
        report_data = None
        if report_data_builder is not None:
            try:
                report_data = report_data_builder(cand)
            except Exception:
                report_data = None
        result = llm_engine_mod.analyze_candidate(cand, report_data,
                                                  client=client)
    except Exception:
        return cand
    if result is not None:
        cand["llm"] = result
    return cand


def _attach_llm_to_scan(out: dict, enabled: bool, client=None) -> dict:
    """
    گام LLM برای خروجی اسکن (داخلی؛ scan_market و scan_forex_market هر
    دو از همین‌جا رد می‌شوند تا رفتار دقیقاً یکی باشد).

    - enabled=False (پیش‌فرض همه‌ی مسیرها) → خروجی دست‌نخورده، بدون حتی
      یک کلید اضافه.
    - enabled=True ولی کلیدی resolve نمی‌شود → کلید "llm_summary" با
      active=False و دلیل «کلید تنظیم نشده» اضافه می‌شود تا صدا‌زننده
      (CLI) بتواند یک خط واضح چاپ کند؛ هیچ نامزدی کلید "llm" نمی‌گیرد.
    - وگرنه حداکثر config.LLM_SCAN_TOP_K نامزدِ اولِ برشِ نمایشی که
      تصمیمشان NO_TRADE نیست بازبینی می‌شوند و "llm_summary" شمارش
      بازبینی‌شده‌ها و توزیع تصمیم‌های LLM را می‌آورد. هر خطای LLM فقط
      همان نامزد را رد می‌کند؛ اسکن و بقیه‌ی نامزدها دقیقاً مثل قبل‌اند.
    """
    if not enabled:
        return out
    try:
        has_key = bool(llm_client_mod.resolve_api_key()) \
            or client is not None
    except Exception:
        has_key = client is not None
    if not has_key:
        out["llm_summary"] = {"active": False, "reason": "no_api_key",
                              "analyzed": 0, "decisions": {}}
        return out
    top_k = getattr(config, "LLM_SCAN_TOP_K", 3)
    analyzed = 0
    decisions = {}
    for cand in out.get("candidates", []):
        if analyzed >= top_k:
            break
        if cand.get("decision") in (None, "NO_TRADE"):
            continue
        before = "llm" in cand
        try:
            attach_llm_analysis(cand, client=client)
        except Exception:
            continue
        if "llm" in cand and not before:
            analyzed += 1
            d = (cand["llm"] or {}).get("llm_decision")
            if d:
                decisions[d] = decisions.get(d, 0) + 1
    out["llm_summary"] = {"active": True, "analyzed": analyzed,
                          "decisions": decisions}
    return out


# ---------------------------------------------------------------------------
# اسکن بازار
# ---------------------------------------------------------------------------
def scan_market(symbols: list = None, timeframes: list = None,
                limit: int = None, progress: bool = True,
                use_mtf: bool = None,
                include_btc_corr: bool = False,
                only_actionable: bool = False,
                apply_regime: bool = None,
                llm: bool = False,
                llm_client=None) -> dict:
    """
    همه‌ی ترکیب‌های ارز × تایم‌فریم را اسکن و رتبه‌بندی می‌کند.
    اگر یک ترکیب خطا بدهد (مثلاً نماد ناموجود)، خطا ثبت و اسکن ادامه پیدا می‌کند.
    خروجی: {"candidates": [...مرتب‌شده نزولی، حداکثر limit تا],
            "errors": [{"symbol","timeframe","error"}...], "scanned": تعداد موفق}

    use_mtf (گام ۲): اگر True باشد، برای نامزدهای
    4h جهت تایم‌فریم روزانه با همان امتیازدهی حساب و به‌عنوان دروازه‌ی MTF
    به score_candidate داده می‌شود (جهت روزانه برای هر ارز یک بار گرفته و
    در طول اسکن کش می‌شود). خطای گرفتن روزانه، نامزد را بدون دروازه نگه
    می‌دارد و در errors با برچسب "1d (mtf)" ثبت می‌شود.
    پیش‌فرض None = از config.IMPROVE_MTF_DEFAULT خوانده می‌شود (برنامه‌ی
    بهبود، مرحله‌ی ۲: پذیرفته و True شد)؛ False صریح همیشه بر کانفیگ
    غلبه می‌کند و رفتار قبلی را برمی‌گرداند.

    apply_regime (برنامه‌ی بهبود، مرحله‌ی ۲ — افزودنی): اگر True باشد،
    مقیاس رژیم (گام ۳ score_candidate) روی نامزدها اعمال می‌شود.
    پیش‌فرض None = از config.IMPROVE_REGIME_DEFAULT (پذیرفته‌شده: True)؛
    False صریح = خاموش. جهت روزانه‌ی MTF عمداً «بدون» رژیم حساب می‌شود
    — همان سمانتیک موتور ارزیابی مراحل (جهت پایپ‌لاین ساده‌ی روزانه) و
    رفتار قبلی همین مسیر.

    include_btc_corr (افزودنی، پیش‌فرض False = رفتار قبلی): اگر True باشد،
    هر نامزد کلید "btc_correlation" می‌گیرد (همبستگی بازده با BTC؛ برای
    خود BTC برابر 1.0 و روی خطا None). دیتافریم BTC برای هر تایم‌فریم یک
    بار گرفته و در طول اسکن کش می‌شود.

    لایه‌ی EV (افزودنی، همیشه فعال ولی فقط‌خواندنی از کش): هر نامزد
    کلیدهای "ev" و "decision"/"decision_fa"/"decision_reason" می‌گیرد و
    خروجی کلید "decision_summary" (شمارش هر وضعیت روی همه‌ی ترکیب‌های
    اسکن‌شده، قبل از برش limit) دارد. نامزدهای NO_TRADE هرگز بی‌صدا حذف
    نمی‌شوند — با دلیل ردشان در لیست می‌مانند.

    only_actionable (افزودنی، پیش‌فرض False): اگر True باشد، خروجی یک
    کلید جدا به نام "actionable" هم دارد: فقط نامزدهایی که تصمیمشان
    BUY/SELL/STRONG_* است (از کل اسکن، نه فقط برش limit). لیست اصلی
    "candidates" در هر دو حالت کامل و بدون فیلتر پنهان است.

    llm (افزودنی، پیش‌فرض False = خروجی دقیقاً مثل قبل): اگر True باشد و
    کلید NVIDIA resolve شود، حداکثر config.LLM_SCAN_TOP_K نامزد اولِ
    غیر NO_TRADE بازبینی LLM می‌گیرند (کلید "llm" روی نامزد + کلید
    "llm_summary" روی خروجی — _attach_llm_to_scan). llm_client فقط برای
    تست است (کلاینت جعلی)؛ در مسیر واقعی None می‌ماند.
    """
    if symbols is None:
        symbols = config.SCAN_SYMBOLS
    if timeframes is None:
        timeframes = config.SCAN_TIMEFRAMES
    if limit is None:
        limit = config.SCAN_TOP_N
    if use_mtf is None:
        use_mtf = config.IMPROVE_MTF_DEFAULT
    if apply_regime is None:
        apply_regime = config.IMPROVE_REGIME_DEFAULT

    results = []
    errors = []
    daily_dir_cache = {}
    btc_df_cache = {}
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال اسکن {symbol} {timeframe} ...", flush=True)
            try:
                df = analyzer.fetch_candles_binance(symbol, timeframe)
                report = analyzer.build_report(df, symbol, timeframe)
                _SCAN_CACHE[(symbol.upper(), timeframe)] = (df, report)
                cand = score_candidate(df, report, symbol, timeframe,
                                       apply_regime=apply_regime)
                if use_mtf and timeframe == "4h":
                    try:
                        key = symbol.upper()
                        if key not in daily_dir_cache:
                            ddf = analyzer.fetch_candles_binance(symbol, "1d")
                            drep = analyzer.build_report(ddf, symbol, "1d")
                            daily_dir_cache[key] = score_candidate(
                                ddf, drep, symbol, "1d")["direction"]
                        cand = score_candidate(
                            df, report, symbol, timeframe,
                            higher_tf_direction=daily_dir_cache[key],
                            apply_regime=apply_regime)
                    except Exception as exc:
                        errors.append({"symbol": symbol, "timeframe": "1d (mtf)",
                                       "error": str(exc)})
                if include_btc_corr:
                    if symbol.upper() == "BTC":
                        cand["btc_correlation"] = 1.0
                    else:
                        try:
                            if timeframe not in btc_df_cache:
                                btc_df_cache[timeframe] = \
                                    analyzer.fetch_candles_binance("BTC", timeframe)
                            cand["btc_correlation"] = _btc_corr_from_frames(
                                df, btc_df_cache[timeframe])
                        except Exception:
                            cand["btc_correlation"] = None
                # لایه‌ی EV: فقط از کش خوانده می‌شود، بک‌تست اجرا نمی‌شود
                attach_ev_decision(cand)
                results.append(cand)
            except Exception as exc:
                errors.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
                if progress:
                    print(f"  ⚠️ خطا در اسکن {symbol} {timeframe}: {exc}", flush=True)

    results.sort(key=lambda c: c["score"], reverse=True)
    # خلاصه‌ی تصمیم روی همه‌ی ترکیب‌های اسکن‌شده (قبل از برش limit)
    decision_summary = {}
    for c in results:
        d = c.get("decision", "NO_TRADE")
        decision_summary[d] = decision_summary.get(d, 0) + 1
    out = {
        "market": "crypto",
        "candidates": results[:limit],
        "errors": errors,
        "scanned": len(results),
        "decision_summary": decision_summary,
    }
    if only_actionable:
        out["actionable"] = [c for c in results
                             if c.get("decision") in ev_mod.ACTIONABLE_DECISIONS]
    _attach_llm_to_scan(out, llm, client=llm_client)
    return out


# ---------------------------------------------------------------------------
# اسکن فارکس (افزودنی — FOREX_FA.md؛ scan_market کریپتو دست نخورده است)
# ---------------------------------------------------------------------------
def scan_forex_market(symbols: list = None, timeframes: list = None,
                      limit: int = None, progress: bool = True,
                      use_mtf: bool = None,
                      only_actionable: bool = False,
                      apply_regime: bool = None,
                      llm: bool = False,
                      llm_client=None) -> dict:
    """
    اسکن جفت‌های فارکس × [4h, 1d] با دقیقاً همان score_candidate و پشته‌ی
    نهایی اسکن کریپتو (رژیم/MTF از همان پیش‌فرض‌های config، لایه‌ی EV
    فقط‌خواندنی از کش). تفاوت‌ها فقط در منبع داده است
    (analyzer.fetch_market با market="forex") و نبود همبستگی BTC.
    ترکیب‌هایی که در کش EV ورودی ندارند (قبل از ماتریس فارکس) تصمیم
    NO_TRADE با دلیل «نمونه ناکافی» می‌گیرند — رفتار درستِ پیش از ماتریس.
    خروجی هم‌قالب scan_market است، به‌علاوه‌ی کلید "market": "forex".
    XAUUSD در این اسکن نماینده‌ی کالا (طلا) است، نه جفت‌ارز.

    llm (افزودنی، پیش‌فرض False): دقیقاً همان سمانتیک scan_market —
    بازبینی LLM فقط با llm=True و کلید resolveشده (به _attach_llm_to_scan
    نگاه کنید). اولویت محصولی فعلی فارکس است، پس این مسیر مصرف‌کننده‌ی
    اصلی لایه‌ی LLM است.
    """
    if symbols is None:
        symbols = config.FOREX_SYMBOLS
    if timeframes is None:
        timeframes = config.FOREX_TIMEFRAMES
    if limit is None:
        limit = config.SCAN_TOP_N
    if use_mtf is None:
        use_mtf = config.IMPROVE_MTF_DEFAULT
    if apply_regime is None:
        apply_regime = config.IMPROVE_REGIME_DEFAULT

    results = []
    errors = []
    daily_dir_cache = {}
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال اسکن فارکس {symbol} {timeframe} ...", flush=True)
            try:
                df = analyzer.fetch_market(symbol, timeframe, market="forex")
                report = analyzer.build_report(df, symbol, timeframe)
                _SCAN_CACHE[(symbol.upper(), timeframe)] = (df, report)
                cand = score_candidate(df, report, symbol, timeframe,
                                       apply_regime=apply_regime)
                cand["market"] = "forex"
                if use_mtf and timeframe == "4h":
                    try:
                        key = symbol.upper()
                        if key not in daily_dir_cache:
                            ddf = analyzer.fetch_market(symbol, "1d",
                                                        market="forex")
                            drep = analyzer.build_report(ddf, symbol, "1d")
                            daily_dir_cache[key] = score_candidate(
                                ddf, drep, symbol, "1d")["direction"]
                        cand = score_candidate(
                            df, report, symbol, timeframe,
                            higher_tf_direction=daily_dir_cache[key],
                            apply_regime=apply_regime)
                        cand["market"] = "forex"
                    except Exception as exc:
                        errors.append({"symbol": symbol,
                                       "timeframe": "1d (mtf)",
                                       "error": str(exc)})
                attach_ev_decision(cand)
                results.append(cand)
            except Exception as exc:
                errors.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
                if progress:
                    print(f"  ⚠️ خطا در اسکن فارکس {symbol} {timeframe}: {exc}",
                          flush=True)

    results.sort(key=lambda c: c["score"], reverse=True)
    decision_summary = {}
    for c in results:
        d = c.get("decision", "NO_TRADE")
        decision_summary[d] = decision_summary.get(d, 0) + 1
    out = {
        "market": "forex",
        "candidates": results[:limit],
        "errors": errors,
        "scanned": len(results),
        "decision_summary": decision_summary,
    }
    if only_actionable:
        out["actionable"] = [c for c in results
                             if c.get("decision") in ev_mod.ACTIONABLE_DECISIONS]
    _attach_llm_to_scan(out, llm, client=llm_client)
    return out
