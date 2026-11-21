"""
indicators/levels.py
Price levels: Fibonacci Retracement, Pivot Points, Support/Resistance (swing-based)
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
    Fibonacci retracement levels between a peak and a trough.
    If the trend is up (trough before peak), the levels retrace from top to bottom.
    """
    diff = swing_high - swing_low
    levels = {}
    for ratio in FIB_RATIOS:
        levels[f"{ratio:.3f}"] = swing_high - (diff * ratio)
    return levels


def fibonacci_extension(swing_high: float, swing_low: float) -> dict:
    """Fibonacci extension levels — for estimating price targets beyond the main swing."""
    diff = swing_high - swing_low
    levels = {}
    for ratio in FIB_EXTENSION_RATIOS:
        levels[f"{ratio:.3f}"] = swing_high + (diff * (ratio - 1))
    return levels


# ---------------------------------------------------------------------------
# Pivot Points (classic daily formula)
# ---------------------------------------------------------------------------
def classic_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """
    Classic pivot points based on the previous candle/period.
    Output: PP, R1-R3, S1-S3
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
    Finds local peaks (swing highs) and troughs (swing lows).
    A candle is a swing high if its high is greater than 'lookback' candles before and after it.
    Output: the columns is_swing_high, is_swing_low are added to df.
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
    Merges several nearby price levels into one single level (the mean)
    and returns them sorted by the number of touches (level strength).
    Output: a list of dicts {"level": price, "touches": number of touches}
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
    Extracts support and resistance levels from recent swing highs/lows.
    Output: {"resistance": [...], "support": [...]}
    Each level includes level (price) and touches (number of touches, i.e. level strength).

    Recency window (fix for audit issue no. 4): only the swings of the last
    config.RECENT_SR_CANDLES candles (default 250) take part in the clustering.
    On long Binance histories (1000+ candles), stale levels far from the current
    price behavior were ruining the result and in downtrends practically no
    support was found. If the whole data is shorter than this window, all of
    the data is used.
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

    # Only levels that are really above/below the current price are meaningful as resistance/support
    resistance = [r for r in resistance_clusters if r["level"] > current_price]
    support = [s for s in support_clusters if s["level"] < current_price]

    resistance.sort(key=lambda x: x["level"])
    support.sort(key=lambda x: x["level"], reverse=True)

    return {"resistance": resistance, "support": support}
