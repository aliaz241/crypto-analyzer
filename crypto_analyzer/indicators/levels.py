"""
indicators/levels.py
سطوح قیمتی: Fibonacci Retracement, Pivot Points, Support/Resistance (بر پایه‌ی swing)
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Fibonacci Retracement
# ---------------------------------------------------------------------------
FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSION_RATIOS = [1.272, 1.414, 1.618, 2.0, 2.618]


def fibonacci_retracement(swing_high: float, swing_low: float) -> dict:
    """
    سطوح اصلاحی فیبوناچی بین یک قله و یک دره.
    اگر روند صعودی است (دره قبل از قله)، سطوح از بالا به پایین اصلاح می‌شوند.
    """
    diff = swing_high - swing_low
    levels = {}
    for ratio in FIB_RATIOS:
        levels[f"{ratio:.3f}"] = swing_high - (diff * ratio)
    return levels


def fibonacci_extension(swing_high: float, swing_low: float) -> dict:
    """سطوح گسترشی فیبوناچی — برای تخمین اهداف قیمتی فراتر از swing اصلی."""
    diff = swing_high - swing_low
    levels = {}
    for ratio in FIB_EXTENSION_RATIOS:
        levels[f"{ratio:.3f}"] = swing_high + (diff * (ratio - 1))
    return levels


# ---------------------------------------------------------------------------
# Pivot Points (فرمول کلاسیک روزانه)
# ---------------------------------------------------------------------------
def classic_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """
    پیوت پوینت کلاسیک بر اساس کندل/دوره‌ی قبلی.
    خروجی: PP, R1-R3, S1-S3
    """
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


# ---------------------------------------------------------------------------
# Swing-based Support / Resistance
# ---------------------------------------------------------------------------
def find_swing_points(df: pd.DataFrame, lookback: int = None) -> pd.DataFrame:
    """
    قله‌ها (swing high) و دره‌های (swing low) محلی را پیدا می‌کند.
    یک کندل swing high است اگر high آن از 'lookback' کندل قبل و بعدش بیشتر باشد.
    خروجی: ستون‌های is_swing_high, is_swing_low به df اضافه می‌شود.
    """
    if lookback is None:
        lookback = config.SWING_LOOKBACK_CANDLES

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback: i + lookback + 1]
        window_low = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_high.max() and np.argmax(window_high) == lookback:
            is_swing_high[i] = True
        if lows[i] == window_low.min() and np.argmin(window_low) == lookback:
            is_swing_low[i] = True

    result = df.copy()
    result["is_swing_high"] = is_swing_high
    result["is_swing_low"] = is_swing_low
    return result


def cluster_levels(prices: list, tolerance_pct: float = None) -> list:
    """
    چند سطح قیمتی نزدیک به هم را در یک سطح واحد (میانگین) ادغام می‌کند
    و بر اساس تعداد برخورد (قدرت سطح) مرتب شده برمی‌گرداند.
    خروجی: لیستی از دیکشنری {"level": قیمت, "touches": تعداد برخورد}
    """
    if tolerance_pct is None:
        tolerance_pct = config.LEVEL_EQUALITY_TOLERANCE_PCT
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters = [[sorted_prices[0]]]

    for price in sorted_prices[1:]:
        last_cluster_mean = np.mean(clusters[-1])
        pct_diff = abs(price - last_cluster_mean) / last_cluster_mean * 100
        if pct_diff <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    result = [{"level": float(np.mean(c)), "touches": len(c)} for c in clusters]
    result.sort(key=lambda x: x["touches"], reverse=True)
    return result


def support_resistance_levels(df: pd.DataFrame, lookback: int = None) -> dict:
    """
    سطوح حمایت و مقاومت را از روی swing high/low های اخیر استخراج می‌کند.
    خروجی: {"resistance": [...], "support": [...]}
    هر سطح شامل level (قیمت) و touches (تعداد برخورد، یعنی قدرت سطح) است.

    پنجره‌ی تازگی (رفع مشکل ممیزی شماره‌ی ۴): فقط swing های
    config.RECENT_SR_CANDLES کندل اخیر (پیش‌فرض ۲۵۰) در خوشه‌بندی شرکت می‌کنند.
    روی تاریخچه‌های بلند بایننس (۱۰۰۰+ کندل) سطوح کهنه و دور از رفتار فعلی قیمت
    نتیجه را خراب می‌کردند و در روندهای نزولی عملاً هیچ حمایتی پیدا نمی‌شد.
    اگر کل داده از این پنجره کوتاه‌تر باشد، همه‌ی داده استفاده می‌شود.
    """
    swing_df = find_swing_points(df, lookback)
    current_price = df["close"].iloc[-1]

    recent_window = getattr(config, "RECENT_SR_CANDLES", 250)
    if recent_window and len(swing_df) > recent_window:
        swing_df = swing_df.iloc[-recent_window:]

    swing_highs = swing_df.loc[swing_df["is_swing_high"], "high"].tolist()
    swing_lows = swing_df.loc[swing_df["is_swing_low"], "low"].tolist()

    resistance_clusters = cluster_levels(swing_highs)
    support_clusters = cluster_levels(swing_lows)

    # فقط سطوحی که واقعاً بالای/پایین قیمت فعلی‌اند به عنوان مقاومت/حمایت معنادارند
    resistance = [r for r in resistance_clusters if r["level"] > current_price]
    support = [s for s in support_clusters if s["level"] < current_price]

    resistance.sort(key=lambda x: x["level"])
    support.sort(key=lambda x: x["level"], reverse=True)

    return {"resistance": resistance, "support": support}
