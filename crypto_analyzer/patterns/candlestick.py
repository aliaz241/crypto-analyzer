"""
patterns/candlestick.py
تشخیص الگوهای کندل‌استیک (کندل‌شناسی).

هر تابع یک pandas.Series بولی هم‌تراز با ایندکس df برمی‌گرداند:
True در اندیسی که الگو در آن کندل (یا کندل پایانی الگوهای چندکندلی) تکمیل شده.

این پیاده‌سازی مستقل است (وابسته به TA-Lib نیست) تا در محیط‌هایی مثل Termux
یا ویندوز ۳۲بیتی بدون نیاز به کامپایل کتابخانه‌ی C کار کند.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def _candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    is_bullish = df["close"] > df["open"]
    is_bearish = df["close"] < df["open"]
    body_pct = (body / candle_range).fillna(0)
    return pd.DataFrame({
        "body": body,
        "range": candle_range,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "is_bullish": is_bullish,
        "is_bearish": is_bearish,
        "body_pct": body_pct,
    })


def _prior_trend(df: pd.DataFrame, lookback: int = 5, threshold_pct: float = 1.0,
                  shift_from: int = 1) -> pd.Series:
    """
    روند را قبل از شروع الگو تخمین می‌زند: close در 'shift_from' کندل قبل از
    کندل جاری را با close در 'shift_from + lookback' کندل قبل مقایسه می‌کند.
    برای الگوهای تک‌کندلی shift_from=1 (پیش‌فرض) درست است؛ برای الگوهای
    چندکندلی (مثل Tweezer که دو کندل را در بر می‌گیرد) باید shift_from را
    به تعداد کندل‌های الگو افزایش داد تا روند قبل از *کل* الگو سنجیده شود،
    نه قبل از آخرین کندلش.
    خروجی: سری رشته‌ای با مقادیر 'up', 'down', 'flat'
    """
    reference = df["close"].shift(shift_from)
    past = df["close"].shift(shift_from + lookback)
    pct_change = (reference - past) / past * 100

    trend = pd.Series("flat", index=df.index)
    trend[pct_change > threshold_pct] = "up"
    trend[pct_change < -threshold_pct] = "down"
    return trend


# ---------------------------------------------------------------------------
# Single-candle patterns
# ---------------------------------------------------------------------------
def doji(df: pd.DataFrame) -> pd.Series:
    """بدنه‌ی بسیار کوچک نسبت به کل رنج — نشانه‌ی تردید بازار."""
    m = _candle_metrics(df)
    return m["body_pct"] <= 0.07


def hammer(df: pd.DataFrame) -> pd.Series:
    """بدنه‌ی کوچک بالای رنج + سایه‌ی پایین بلند، بعد از روند نزولی = برگشت صعودی."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["lower_shadow"] >= 2 * m["body"])
        & (m["upper_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "down")


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """همان شکل چکش ولی بعد از روند صعودی = هشدار برگشت نزولی."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["lower_shadow"] >= 2 * m["body"])
        & (m["upper_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "up")


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """بدنه‌ی کوچک پایین رنج + سایه‌ی بالا بلند، بعد از نزول = احتمال برگشت صعودی."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["upper_shadow"] >= 2 * m["body"])
        & (m["lower_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "down")


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """همان شکل بالا ولی بعد از روند صعودی = هشدار برگشت نزولی."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["upper_shadow"] >= 2 * m["body"])
        & (m["lower_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "up")


def bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    """کندل صعودی توپر بدون سایه — فشار خرید قوی و یکطرفه."""
    m = _candle_metrics(df)
    return m["is_bullish"] & (m["body_pct"] >= 0.95)


def bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    """کندل نزولی توپر بدون سایه — فشار فروش قوی و یکطرفه."""
    m = _candle_metrics(df)
    return m["is_bearish"] & (m["body_pct"] >= 0.95)


def spinning_top(df: pd.DataFrame) -> pd.Series:
    """بدنه‌ی کوچک با سایه‌های بالا و پایین تقریباً مساوی — سردرگمی بازار."""
    m = _candle_metrics(df)
    shadow_balance = (m["upper_shadow"] - m["lower_shadow"]).abs() <= (m["range"] * 0.15)
    return (m["body_pct"] <= 0.3) & (m["body_pct"] > 0.07) & shadow_balance


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------
def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """کندل صعودی بدنه‌ی کندل نزولی قبلی را کامل می‌بلعد — برگشت صعودی قوی."""
    m = _candle_metrics(df)
    prev_bearish = m["is_bearish"].shift(1)
    curr_bullish = m["is_bullish"]
    engulf = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bearish & curr_bullish & engulf


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """کندل نزولی بدنه‌ی کندل صعودی قبلی را کامل می‌بلعد — برگشت نزولی قوی."""
    m = _candle_metrics(df)
    prev_bullish = m["is_bullish"].shift(1)
    curr_bearish = m["is_bearish"]
    engulf = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bullish & curr_bearish & engulf


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    """کندل نزولی بزرگ و بعد یک کندل صعودی کوچک داخل بدنه‌ی آن — کاهش فشار فروش."""
    m = _candle_metrics(df)
    prev_bearish = m["is_bearish"].shift(1)
    curr_bullish = m["is_bullish"]
    inside = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bearish & curr_bullish & inside


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    """کندل صعودی بزرگ و بعد یک کندل نزولی کوچک داخل بدنه‌ی آن — کاهش فشار خرید."""
    m = _candle_metrics(df)
    prev_bullish = m["is_bullish"].shift(1)
    curr_bearish = m["is_bearish"]
    inside = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bullish & curr_bearish & inside


def piercing_line(df: pd.DataFrame) -> pd.Series:
    """
    کندل صعودی که بیش از نیمی از بدنه‌ی نزولی قبلی را پوشش می‌دهد — برگشت صعودی.
    تعریف کلاسیک این الگو gap-down واقعی بین دو کندل می‌خواهد، ولی چون کریپتو
    ۲۴/۷ معامله می‌شود و gap واقعی نادر است، شرط به «باز شدن در سطح یا پایین‌تر
    از close قبلی» نرم شده — ویژگی اصلی الگو (نفوذ عمیق به بدنه‌ی قبلی) حفظ شده.
    """
    m = _candle_metrics(df)
    prev_bearish = m["is_bearish"].shift(1)
    curr_bullish = m["is_bullish"]
    prev_midpoint = (df["open"].shift(1) + df["close"].shift(1)) / 2
    opens_at_or_below_prev_close = df["open"] <= df["close"].shift(1)
    closes_above_mid = df["close"] > prev_midpoint
    closes_below_prev_open = df["close"] < df["open"].shift(1)
    return prev_bearish & curr_bullish & opens_at_or_below_prev_close & closes_above_mid & closes_below_prev_open


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """
    کندل نزولی که بیش از نیمی از بدنه‌ی صعودی قبلی را می‌پوشاند — برگشت نزولی.
    مشابه piercing_line، شرط gap به «باز شدن در سطح یا بالاتر از close قبلی»
    نرم شده تا در بازار بدون gap هم قابل تشخیص باشد.
    """
    m = _candle_metrics(df)
    prev_bullish = m["is_bullish"].shift(1)
    curr_bearish = m["is_bearish"]
    prev_midpoint = (df["open"].shift(1) + df["close"].shift(1)) / 2
    opens_at_or_above_prev_close = df["open"] >= df["close"].shift(1)
    closes_below_mid = df["close"] < prev_midpoint
    closes_above_prev_open = df["close"] > df["open"].shift(1)
    return prev_bullish & curr_bearish & opens_at_or_above_prev_close & closes_below_mid & closes_above_prev_open


def tweezer_top(df: pd.DataFrame) -> pd.Series:
    """دو کندل با سقف تقریباً برابر بعد از روند صعودی — مقاومت دوبار تست شده."""
    m = _candle_metrics(df)
    # shift_from=2 چون این الگو دو کندل دارد؛ روند باید قبل از هر دو سنجیده شود
    trend = _prior_trend(df, lookback=6, shift_from=2)
    high_diff_pct = (df["high"] - df["high"].shift(1)).abs() / df["high"].shift(1) * 100
    return (high_diff_pct <= 0.3) & (trend == "up") & m["is_bearish"]


def tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """دو کندل با کف تقریباً برابر بعد از روند نزولی — حمایت دوبار تست شده."""
    m = _candle_metrics(df)
    trend = _prior_trend(df, lookback=6, shift_from=2)
    low_diff_pct = (df["low"] - df["low"].shift(1)).abs() / df["low"].shift(1) * 100
    return (low_diff_pct <= 0.3) & (trend == "down") & m["is_bullish"]


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------
def morning_star(df: pd.DataFrame) -> pd.Series:
    """
    نزولی بزرگ + کندل کوچک (تردید) + صعودی بزرگ — برگشت صعودی سه‌کندلی.
    به‌جای gap واقعی بین کندل اول و دوم (نادر در بازار ۲۴/۷)، شرط «بدنه‌ی
    کندل دوم پایین‌تر از نیمه‌ی بدنه‌ی کندل اول باشد» استفاده شده — جوهر
    اصلی الگو (مکث/تردید نزدیک کف حرکت نزولی) را بدون نیاز به gap می‌گیرد.
    """
    m = _candle_metrics(df)
    first_bearish = m["is_bearish"].shift(2)
    first_big = m["body_pct"].shift(2) >= 0.6
    second_small = m["body_pct"].shift(1) <= 0.3
    first_midpoint = (df["open"].shift(2) + df["close"].shift(2)) / 2
    second_body_top = df[["open", "close"]].shift(1).max(axis=1)
    second_below_first_mid = second_body_top < first_midpoint
    third_bullish = m["is_bullish"]
    third_closes_into_first = df["close"] > first_midpoint
    return (first_bearish & first_big & second_small & second_below_first_mid
            & third_bullish & third_closes_into_first)


def evening_star(df: pd.DataFrame) -> pd.Series:
    """
    صعودی بزرگ + کندل کوچک (تردید) + نزولی بزرگ — برگشت نزولی سه‌کندلی.
    مشابه morning_star، به‌جای gap واقعی، «بدنه‌ی کندل دوم بالاتر از نیمه‌ی
    بدنه‌ی کندل اول» به عنوان معیار استفاده شده.
    """
    m = _candle_metrics(df)
    first_bullish = m["is_bullish"].shift(2)
    first_big = m["body_pct"].shift(2) >= 0.6
    second_small = m["body_pct"].shift(1) <= 0.3
    first_midpoint = (df["open"].shift(2) + df["close"].shift(2)) / 2
    second_body_bottom = df[["open", "close"]].shift(1).min(axis=1)
    second_above_first_mid = second_body_bottom > first_midpoint
    third_bearish = m["is_bearish"]
    third_closes_into_first = df["close"] < first_midpoint
    return (first_bullish & first_big & second_small & second_above_first_mid
            & third_bearish & third_closes_into_first)


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """سه کندل صعودی پیاپی، هرکدام با close بالاتر از قبلی — روند صعودی قوی."""
    m = _candle_metrics(df)
    all_bullish = m["is_bullish"] & m["is_bullish"].shift(1) & m["is_bullish"].shift(2)
    progressive_close = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    decent_bodies = (m["body_pct"] >= 0.5) & (m["body_pct"].shift(1) >= 0.5) & (m["body_pct"].shift(2) >= 0.5)
    return all_bullish & progressive_close & decent_bodies


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    """سه کندل نزولی پیاپی، هرکدام با close پایین‌تر از قبلی — روند نزولی قوی."""
    m = _candle_metrics(df)
    all_bearish = m["is_bearish"] & m["is_bearish"].shift(1) & m["is_bearish"].shift(2)
    progressive_close = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
    decent_bodies = (m["body_pct"] >= 0.5) & (m["body_pct"].shift(1) >= 0.5) & (m["body_pct"].shift(2) >= 0.5)
    return all_bearish & progressive_close & decent_bodies


def three_inside_up(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Harami + یک کندل صعودی تأییدی که بالاتر از open کندل اول (نه فقط
    close آن) بسته شود — این آستانه‌ی استاندارد تأیید الگو است؛ مقایسه با
    close کندل اول (که برای کندل نزولی پایین‌تر از open است) آستانه را
    بیش از حد ساده می‌کرد.
    """
    har = bullish_harami(df).shift(1).fillna(False)
    m = _candle_metrics(df)
    confirmation = m["is_bullish"] & (df["close"] > df["open"].shift(2))
    return har.astype(bool) & confirmation


def three_inside_down(df: pd.DataFrame) -> pd.Series:
    """Bearish Harami + یک کندل نزولی تأییدی که پایین‌تر از open کندل اول بسته شود."""
    har = bearish_harami(df).shift(1).fillna(False)
    m = _candle_metrics(df)
    confirmation = m["is_bearish"] & (df["close"] < df["open"].shift(2))
    return har.astype(bool) & confirmation


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PATTERN_REGISTRY = {
    "doji":               {"fn": doji,               "fa": "دوجی",              "bias": "neutral"},
    "hammer":              {"fn": hammer,              "fa": "چکش",               "bias": "bullish"},
    "hanging_man":         {"fn": hanging_man,         "fa": "مرد آویزان",         "bias": "bearish"},
    "inverted_hammer":     {"fn": inverted_hammer,     "fa": "چکش معکوس",          "bias": "bullish"},
    "shooting_star":       {"fn": shooting_star,       "fa": "ستاره‌ی دنباله‌دار",  "bias": "bearish"},
    "bullish_marubozu":    {"fn": bullish_marubozu,    "fa": "ماروبوزو صعودی",      "bias": "bullish"},
    "bearish_marubozu":    {"fn": bearish_marubozu,    "fa": "ماروبوزو نزولی",      "bias": "bearish"},
    "spinning_top":        {"fn": spinning_top,        "fa": "فرفره",             "bias": "neutral"},
    "bullish_engulfing":   {"fn": bullish_engulfing,   "fa": "پوشای صعودی",         "bias": "bullish"},
    "bearish_engulfing":   {"fn": bearish_engulfing,   "fa": "پوشای نزولی",         "bias": "bearish"},
    "bullish_harami":      {"fn": bullish_harami,      "fa": "هارامی صعودی",       "bias": "bullish"},
    "bearish_harami":      {"fn": bearish_harami,      "fa": "هارامی نزولی",       "bias": "bearish"},
    "piercing_line":       {"fn": piercing_line,       "fa": "خط نافذ",           "bias": "bullish"},
    "dark_cloud_cover":    {"fn": dark_cloud_cover,    "fa": "ابر تیره",           "bias": "bearish"},
    "tweezer_top":         {"fn": tweezer_top,         "fa": "موچین سقف",          "bias": "bearish"},
    "tweezer_bottom":      {"fn": tweezer_bottom,      "fa": "موچین کف",           "bias": "bullish"},
    "morning_star":        {"fn": morning_star,        "fa": "ستاره‌ی صبحگاهی",     "bias": "bullish"},
    "evening_star":        {"fn": evening_star,        "fa": "ستاره‌ی عصرگاهی",     "bias": "bearish"},
    "three_white_soldiers":{"fn": three_white_soldiers,"fa": "سه سرباز سفید",       "bias": "bullish"},
    "three_black_crows":   {"fn": three_black_crows,   "fa": "سه کلاغ سیاه",        "bias": "bearish"},
    "three_inside_up":     {"fn": three_inside_up,     "fa": "سه‌تایی داخلی صعودی", "bias": "bullish"},
    "three_inside_down":   {"fn": three_inside_down,   "fa": "سه‌تایی داخلی نزولی", "bias": "bearish"},
}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    همه‌ی الگوهای کندلی را روی کل تاریخچه اجرا می‌کند.
    خروجی: DataFrame بولی هم‌اندازه‌ی df، یک ستون به ازای هر الگو.
    """
    results = {}
    for name, info in PATTERN_REGISTRY.items():
        try:
            results[name] = info["fn"](df)
        except Exception:
            results[name] = pd.Series(False, index=df.index)
    return pd.DataFrame(results, index=df.index)


def detect_latest(df: pd.DataFrame, n_candles: int = 1) -> list:
    """
    الگوهایی که در 'n_candles' کندل آخر فعال شده‌اند را برمی‌گرداند.
    خروجی: لیستی از دیکشنری {"name": ..., "fa": ..., "bias": ..., "index": ...}
    """
    all_patterns = detect_all(df)
    recent = all_patterns.tail(n_candles)
    found = []
    for name in recent.columns:
        hits = recent.index[recent[name]]
        for idx in hits:
            info = PATTERN_REGISTRY[name]
            found.append({
                "name": name,
                "fa": info["fa"],
                "bias": info["bias"],
                "index": idx,
            })
    return found
