"""
patterns/chart_patterns.py
Detecting chart (geometric) patterns based on swing high/low.

Unlike candlestick patterns, which are single-candle events, these patterns
span several candles. Each detected pattern is returned as a dict with the
keys below, so it can be used both for the current report and for the backtest:
    name, fa, bias, start_index, completion_index, neckline, target, pattern_price

⚠️ Version 2 — neckline confirmation (fix for audit issue no. 2): a formation
(double top/bottom or head and shoulders) counts as "completed" only when,
after the second swing point, the price breaks the neckline in the direction
of the pattern's bias with a candle close. completion_index is the candle of
the confirmation (break), not the swing candle. An unconfirmed formation
appears neither in the recent-patterns report nor is it counted in backtest
samples.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.levels import find_swing_points


# ---------------------------------------------------------------------------
# Swing sequence helper
# ---------------------------------------------------------------------------
def get_alternating_swings(df: pd.DataFrame, lookback: int = None) -> list:
    """
    Returns a zigzag list of swing points that alternate between high and low.
    If two consecutive swings are of the same type, only the most extreme (highest/lowest) is kept.
    Output: a list of dicts {"pos": numeric position in df, "index": df index, "price": price, "type": "high"/"low"}
    """
    swing_df = find_swing_points(df, lookback)
    points = []
    for pos in range(len(swing_df)):
        if swing_df["is_swing_high"].iloc[pos]:
            points.append({"pos": pos, "index": swing_df.index[pos],
                            "price": swing_df["high"].iloc[pos], "type": "high"})
        elif swing_df["is_swing_low"].iloc[pos]:
            points.append({"pos": pos, "index": swing_df.index[pos],
                            "price": swing_df["low"].iloc[pos], "type": "low"})

    if not points:
        return []

    alternating = [points[0]]
    for p in points[1:]:
        last = alternating[-1]
        if p["type"] == last["type"]:
            # Two consecutive same-type swings — keep only the most extreme
            if p["type"] == "high" and p["price"] > last["price"]:
                alternating[-1] = p
            elif p["type"] == "low" and p["price"] < last["price"]:
                alternating[-1] = p
        else:
            alternating.append(p)
    return alternating


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / b * 100


def _find_neckline_confirmation(df: pd.DataFrame, from_pos: int, neckline: float,
                                invalidation: float, direction: str):
    """
    Neckline-break confirmation (fix for audit issue no. 2).
    Moves forward from the candle after the second swing point (from_pos):
    - Direction "above" (bullish patterns): the first candle whose close closes above the neckline.
    - Direction "below" (bearish patterns): the first candle whose close closes below the neckline.
    If, before the neckline break, the price crosses the invalidation level
    (a close on the other side of the invalidation level), the formation is
    dead and None is returned; if no break happens by the end of the data,
    None is returned again — i.e. the pattern is not yet "completed".
    Output: the numeric position of the confirmation candle (pos) or None.
    """
    closes = df["close"].values
    for pos in range(from_pos + 1, len(df)):
        c = closes[pos]
        if direction == "above":
            if c < invalidation:
                return None  # The pattern was invalidated before confirmation
            if c > neckline:
                return pos
        else:  # below
            if c > invalidation:
                return None  # The pattern was invalidated before confirmation
            if c < neckline:
                return pos
    return None


# ---------------------------------------------------------------------------
# Double Top / Double Bottom
# ---------------------------------------------------------------------------
def detect_double_top(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 2):
        a, neck, b = swings[i], swings[i + 1], swings[i + 2]
        if a["type"] == "high" and neck["type"] == "low" and b["type"] == "high":
            if _pct_diff(a["price"], b["price"]) <= tol:
                height = a["price"] - neck["price"]
                if height <= 0:
                    continue
                invalidation = max(a["price"], b["price"])
                # The pattern completes only with a downward break of the neckline after the second peak
                conf_pos = _find_neckline_confirmation(
                    df, b["pos"], neck["price"], invalidation, "below")
                if conf_pos is None:
                    continue
                found.append({
                    "name": "double_top", "fa": "سقف دوقلو", "bias": "bearish",
                    "start_index": a["index"], "completion_index": df.index[conf_pos],
                    "formation_index": b["index"],
                    "neckline": neck["price"], "pattern_price": a["price"],
                    "invalidation": invalidation,
                    "target": neck["price"] - height,
                })
    return found


def detect_double_bottom(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 2):
        a, neck, b = swings[i], swings[i + 1], swings[i + 2]
        if a["type"] == "low" and neck["type"] == "high" and b["type"] == "low":
            if _pct_diff(a["price"], b["price"]) <= tol:
                height = neck["price"] - a["price"]
                if height <= 0:
                    continue
                invalidation = min(a["price"], b["price"])
                # The pattern completes only with an upward break of the neckline after the second trough
                conf_pos = _find_neckline_confirmation(
                    df, b["pos"], neck["price"], invalidation, "above")
                if conf_pos is None:
                    continue
                found.append({
                    "name": "double_bottom", "fa": "کف دوقلو", "bias": "bullish",
                    "start_index": a["index"], "completion_index": df.index[conf_pos],
                    "formation_index": b["index"],
                    "neckline": neck["price"], "pattern_price": a["price"],
                    "invalidation": invalidation,
                    "target": neck["price"] + height,
                })
    return found


# ---------------------------------------------------------------------------
# Head & Shoulders / Inverse Head & Shoulders
# ---------------------------------------------------------------------------
def detect_head_and_shoulders(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 4):
        ls, n1, head, n2, rs = swings[i:i + 5]
        types = [ls["type"], n1["type"], head["type"], n2["type"], rs["type"]]
        if types != ["high", "low", "high", "low", "high"]:
            continue
        if head["price"] <= ls["price"] or head["price"] <= rs["price"]:
            continue
        if _pct_diff(ls["price"], rs["price"]) > tol * 2:
            continue
        neckline = (n1["price"] + n2["price"]) / 2
        height = head["price"] - neckline
        if height <= 0:
            continue
        # The pattern completes only with a downward break of the neckline after the right shoulder
        conf_pos = _find_neckline_confirmation(
            df, rs["pos"], neckline, head["price"], "below")
        if conf_pos is None:
            continue
        found.append({
            "name": "head_and_shoulders", "fa": "سر و شانه", "bias": "bearish",
            "start_index": ls["index"], "completion_index": df.index[conf_pos],
            "formation_index": rs["index"],
            "neckline": neckline, "pattern_price": head["price"],
            "invalidation": head["price"],
            "target": neckline - height,
        })
    return found


def detect_inverse_head_and_shoulders(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 4):
        ls, n1, head, n2, rs = swings[i:i + 5]
        types = [ls["type"], n1["type"], head["type"], n2["type"], rs["type"]]
        if types != ["low", "high", "low", "high", "low"]:
            continue
        if head["price"] >= ls["price"] or head["price"] >= rs["price"]:
            continue
        if _pct_diff(ls["price"], rs["price"]) > tol * 2:
            continue
        neckline = (n1["price"] + n2["price"]) / 2
        height = neckline - head["price"]
        if height <= 0:
            continue
        # The pattern completes only with an upward break of the neckline after the right shoulder
        conf_pos = _find_neckline_confirmation(
            df, rs["pos"], neckline, head["price"], "above")
        if conf_pos is None:
            continue
        found.append({
            "name": "inverse_head_and_shoulders", "fa": "سر و شانه‌ی معکوس", "bias": "bullish",
            "start_index": ls["index"], "completion_index": df.index[conf_pos],
            "formation_index": rs["index"],
            "neckline": neckline, "pattern_price": head["price"],
            "invalidation": head["price"],
            "target": neckline + height,
        })
    return found


# ---------------------------------------------------------------------------
# Triangles & Wedges (based on linear regression on swings in a sliding window)
# ---------------------------------------------------------------------------
FLAT_SLOPE_THRESHOLD_PCT = 0.05   # A slope smaller than this (percent of price per candle) = "almost flat"


def _fit_line(points: list):
    """points: a list of (pos, price). Output: (slope, intercept) or None if not enough."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


def _scan_triangle_wedge_windows(df: pd.DataFrame, window: int = 40, step: int = 10) -> list:
    found = []
    swing_df = find_swing_points(df)
    n = len(df)

    starts = list(range(0, n - window, step))
    # Make sure the last window reaches exactly the end of the data so recent candles
    # are not dropped because of the fixed step
    if n - window >= 0 and (not starts or starts[-1] != n - window):
        starts.append(n - window)

    for start in starts:
        end = start + window
        segment = swing_df.iloc[start:end]
        avg_price = segment["close"].mean()
        if avg_price == 0 or np.isnan(avg_price):
            continue

        highs = [(i, segment["high"].iloc[i - start]) for i in range(start, end)
                  if segment["is_swing_high"].iloc[i - start]]
        lows = [(i, segment["low"].iloc[i - start]) for i in range(start, end)
                 if segment["is_swing_low"].iloc[i - start]]

        if len(highs) < 2 or len(lows) < 2:
            continue

        upper_fit = _fit_line(highs)
        lower_fit = _fit_line(lows)
        if upper_fit is None or lower_fit is None:
            continue

        upper_slope_pct = upper_fit[0] / avg_price * 100
        lower_slope_pct = lower_fit[0] / avg_price * 100

        upper_flat = abs(upper_slope_pct) <= FLAT_SLOPE_THRESHOLD_PCT
        lower_flat = abs(lower_slope_pct) <= FLAT_SLOPE_THRESHOLD_PCT

        completion_index = df.index[end - 1]
        pattern_high = max(p[1] for p in highs)
        pattern_low = min(p[1] for p in lows)
        height = pattern_high - pattern_low
        if height <= 0:
            continue

        last_close = df["close"].iloc[end - 1]

        # invalidation = the level at which, if price returns to it, the pattern hypothesis is invalidated
        pattern = None
        if upper_flat and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT:
            pattern = {
                "name": "ascending_triangle", "fa": "مثلث صعودی", "bias": "bullish",
                "target": pattern_high + height, "invalidation": pattern_low,
            }
        elif lower_flat and upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT:
            pattern = {
                "name": "descending_triangle", "fa": "مثلث نزولی", "bias": "bearish",
                "target": pattern_low - height, "invalidation": pattern_high,
            }
        elif upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT:
            # The symmetrical triangle is inherently directionless; the current price's position relative to the triangle's middle
            # is taken as an operational "bias", not classical certainty.
            leaning_bullish = last_close >= (pattern_high + pattern_low) / 2
            pattern = {
                "name": "symmetrical_triangle",
                "fa": "مثلث متقارن (تمایل صعودی)" if leaning_bullish else "مثلث متقارن (تمایل نزولی)",
                "bias": "bullish" if leaning_bullish else "bearish",
                "target": last_close + height if leaning_bullish else last_close - height,
                "invalidation": pattern_low if leaning_bullish else pattern_high,
            }
        elif upper_slope_pct > FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT \
                and lower_slope_pct > upper_slope_pct:
            pattern = {
                "name": "rising_wedge", "fa": "گوه‌ی صعودی", "bias": "bearish",
                "target": pattern_low - height, "invalidation": pattern_high,
            }
        elif upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT \
                and upper_slope_pct > lower_slope_pct:
            pattern = {
                "name": "falling_wedge", "fa": "گوه‌ی نزولی", "bias": "bullish",
                "target": pattern_high + height, "invalidation": pattern_low,
            }

        if pattern:
            # Break-of-boundary confirmation (fix for audit issue no. 2 for triangles and wedges):
            # The windowed scanner builds the boundary lines by regression on the swings, and the
            # "completion" point is inherently the end of that same window; so a separate multi-candle
            # break, like the swing patterns, cannot be tracked cleanly. As a minimum condition, the close
            # of the window's last candle must have crossed the nearer boundary line in the direction of the pattern's bias
            # (bullish: above the upper line / bearish: below the lower line).
            # Without this crossing, the formation is only a price congestion and does not count as a
            # "completed pattern" — it appears neither in the report nor is it counted in the backtest.
            end_pos = end - 1
            upper_at_end = upper_fit[0] * end_pos + upper_fit[1]
            lower_at_end = lower_fit[0] * end_pos + lower_fit[1]
            if pattern["bias"] == "bullish" and not (last_close > upper_at_end):
                pattern = None
            elif pattern["bias"] == "bearish" and not (last_close < lower_at_end):
                pattern = None

        if pattern:
            pattern.update({
                "start_index": df.index[start],
                "completion_index": completion_index,
                "neckline": pattern_high if pattern["bias"] == "bearish" else pattern_low,
                "pattern_price": pattern["invalidation"],
            })
            found.append(pattern)

    # Removing overlapping detections: when consecutive windows show the same pattern type
    # (because the windows advance with a step smaller than their width), we keep only the newest.
    deduped = []
    for p in found:
        if deduped and deduped[-1]["name"] == p["name"] and \
                (df.index.get_loc(p["completion_index"]) - df.index.get_loc(deduped[-1]["completion_index"])) < window:
            deduped[-1] = p
        else:
            deduped.append(p)

    return deduped


# ---------------------------------------------------------------------------
# Master scan
# ---------------------------------------------------------------------------
def scan_all(df: pd.DataFrame) -> list:
    """
    Scans all chart patterns over the whole history.
    Output: a list of pattern dicts (usable for the current report and for the backtest).
    """
    swings = get_alternating_swings(df)
    results = []
    results += detect_double_top(df, swings)
    results += detect_double_bottom(df, swings)
    results += detect_head_and_shoulders(df, swings)
    results += detect_inverse_head_and_shoulders(df, swings)
    results += _scan_triangle_wedge_windows(df)

    results.sort(key=lambda p: df.index.get_loc(p["completion_index"]))
    return results


def latest_patterns(df: pd.DataFrame, recent_candles: int = 15) -> list:
    """Only the patterns whose completion_index is in the last 'recent_candles' candles."""
    all_patterns = scan_all(df)
    cutoff_pos = len(df) - recent_candles
    return [p for p in all_patterns if df.index.get_loc(p["completion_index"]) >= cutoff_pos]
