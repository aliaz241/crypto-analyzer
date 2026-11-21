"""
patterns/candlestick.py
Detecting candlestick patterns (candle reading).

Each function returns a boolean pandas.Series aligned with the df index:
True at the index where the pattern completed in that candle (or the final
candle of multi-candle patterns).

This implementation is independent (not dependent on TA-Lib) so it works in
environments like Termux or 32-bit Windows without needing to compile a C library.
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
    Estimates the trend before the pattern starts: compares the close at
    'shift_from' candles before the current candle with the close at
    'shift_from + lookback' candles before. For single-candle patterns
    shift_from=1 (default) is right; for multi-candle patterns (like Tweezer,
    which spans two candles) shift_from must be increased by the number of
    the pattern's candles so the trend is measured before the *whole* pattern,
    not before its last candle.
    Output: a string series with values 'up', 'down', 'flat'
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
    """A very small body relative to the whole range — a sign of market indecision."""
    m = _candle_metrics(df)
    return m["body_pct"] <= 0.07


def hammer(df: pd.DataFrame) -> pd.Series:
    """A small body at the top of the range + a long lower shadow, after a downtrend = bullish reversal."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["lower_shadow"] >= 2 * m["body"])
        & (m["upper_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "down")


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """The same hammer shape but after an uptrend = bearish reversal warning."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["lower_shadow"] >= 2 * m["body"])
        & (m["upper_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "up")


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """A small body at the bottom of the range + a long upper shadow, after a decline = probable bullish reversal."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["upper_shadow"] >= 2 * m["body"])
        & (m["lower_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "down")


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """The same shape as above but after an uptrend = bearish reversal warning."""
    m = _candle_metrics(df)
    trend = _prior_trend(df)
    shape = (
        (m["body_pct"] <= 0.35)
        & (m["upper_shadow"] >= 2 * m["body"])
        & (m["lower_shadow"] <= m["body"] * 0.5)
    )
    return shape & (trend == "up")


def bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    """A solid bullish candle with no shadow — strong, one-sided buying pressure."""
    m = _candle_metrics(df)
    return m["is_bullish"] & (m["body_pct"] >= 0.95)


def bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    """A solid bearish candle with no shadow — strong, one-sided selling pressure."""
    m = _candle_metrics(df)
    return m["is_bearish"] & (m["body_pct"] >= 0.95)


def spinning_top(df: pd.DataFrame) -> pd.Series:
    """A small body with roughly equal upper and lower shadows — market confusion."""
    m = _candle_metrics(df)
    shadow_balance = (m["upper_shadow"] - m["lower_shadow"]).abs() <= (m["range"] * 0.15)
    return (m["body_pct"] <= 0.3) & (m["body_pct"] > 0.07) & shadow_balance


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------
def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """A bullish candle completely engulfs the previous bearish candle's body — strong bullish reversal."""
    m = _candle_metrics(df)
    prev_bearish = m["is_bearish"].shift(1)
    curr_bullish = m["is_bullish"]
    engulf = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bearish & curr_bullish & engulf


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """A bearish candle completely engulfs the previous bullish candle's body — strong bearish reversal."""
    m = _candle_metrics(df)
    prev_bullish = m["is_bullish"].shift(1)
    curr_bearish = m["is_bearish"]
    engulf = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bullish & curr_bearish & engulf


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    """A large bearish candle then a small bullish candle inside its body — decreasing selling pressure."""
    m = _candle_metrics(df)
    prev_bearish = m["is_bearish"].shift(1)
    curr_bullish = m["is_bullish"]
    inside = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bearish & curr_bullish & inside


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    """A large bullish candle then a small bearish candle inside its body — decreasing buying pressure."""
    m = _candle_metrics(df)
    prev_bullish = m["is_bullish"].shift(1)
    curr_bearish = m["is_bearish"]
    inside = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bullish & curr_bearish & inside


def piercing_line(df: pd.DataFrame) -> pd.Series:
    """
    A bullish candle that covers more than half of the previous bearish body — bullish reversal.
    The classic definition of this pattern wants a real gap-down between the
    two candles, but since crypto trades 24/7 and a real gap is rare, the
    condition is softened to "opening at or below the previous close" — the
    pattern's main feature (deep penetration into the previous body) is preserved.
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
    A bearish candle that covers more than half of the previous bullish body — bearish reversal.
    Like piercing_line, the gap condition is softened to "opening at or above
    the previous close" so it is detectable in a gapless market too.
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
    """Two candles with roughly equal highs after an uptrend — resistance tested twice."""
    m = _candle_metrics(df)
    # shift_from=2 because this pattern has two candles; the trend must be measured before both
    trend = _prior_trend(df, lookback=6, shift_from=2)
    high_diff_pct = (df["high"] - df["high"].shift(1)).abs() / df["high"].shift(1) * 100
    return (high_diff_pct <= 0.3) & (trend == "up") & m["is_bearish"]


def tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """Two candles with roughly equal lows after a downtrend — support tested twice."""
    m = _candle_metrics(df)
    trend = _prior_trend(df, lookback=6, shift_from=2)
    low_diff_pct = (df["low"] - df["low"].shift(1)).abs() / df["low"].shift(1) * 100
    return (low_diff_pct <= 0.3) & (trend == "down") & m["is_bullish"]


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------
def morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Large bearish + a small candle (indecision) + large bullish — a three-candle bullish reversal.
    Instead of a real gap between the first and second candles (rare in a 24/7
    market), the condition "the second candle's body is below the midpoint of
    the first candle's body" is used — it captures the pattern's main essence
    (a pause/indecision near the low of the bearish move) without needing a gap.
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
    Large bullish + a small candle (indecision) + large bearish — a three-candle bearish reversal.
    Like morning_star, instead of a real gap, "the second candle's body is
    above the midpoint of the first candle's body" is used as the criterion.
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
    """Three consecutive bullish candles, each closing higher than the previous — a strong uptrend."""
    m = _candle_metrics(df)
    all_bullish = m["is_bullish"] & m["is_bullish"].shift(1) & m["is_bullish"].shift(2)
    progressive_close = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    decent_bodies = (m["body_pct"] >= 0.5) & (m["body_pct"].shift(1) >= 0.5) & (m["body_pct"].shift(2) >= 0.5)
    return all_bullish & progressive_close & decent_bodies


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three consecutive bearish candles, each closing lower than the previous — a strong downtrend."""
    m = _candle_metrics(df)
    all_bearish = m["is_bearish"] & m["is_bearish"].shift(1) & m["is_bearish"].shift(2)
    progressive_close = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
    decent_bodies = (m["body_pct"] >= 0.5) & (m["body_pct"].shift(1) >= 0.5) & (m["body_pct"].shift(2) >= 0.5)
    return all_bearish & progressive_close & decent_bodies


def three_inside_up(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Harami + a confirming bullish candle that closes above the first
    candle's open (not just its close) — this is the pattern's standard
    confirmation threshold; comparing with the first candle's close (which for
    a bearish candle is below its open) made the threshold too easy.
    """
    har = bullish_harami(df).shift(1).fillna(False)
    m = _candle_metrics(df)
    confirmation = m["is_bullish"] & (df["close"] > df["open"].shift(2))
    return har.astype(bool) & confirmation


def three_inside_down(df: pd.DataFrame) -> pd.Series:
    """Bearish Harami + a confirming bearish candle that closes below the first candle's open."""
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
    Runs all candlestick patterns over the whole history.
    Output: a boolean DataFrame the same size as df, one column per pattern.
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
    Returns the patterns that activated in the last 'n_candles' candles.
    Output: a list of dicts {"name": ..., "fa": ..., "bias": ..., "index": ...}
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
