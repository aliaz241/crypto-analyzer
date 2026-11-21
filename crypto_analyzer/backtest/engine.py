"""
backtest/engine.py
Backtest of patterns on historical data.

For each historical occurrence of a pattern, it is checked whether the price
reached the "target" first (success / win), touched the "pattern invalidation"
level first (failure / loss), or neither happened within the specified time
window (timeout). Final success rate = wins / total.

⚠️ This is a historical statistic of the pattern's past performance on this
asset/timeframe, not a guarantee for the future. Market behavior can change.
"""

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.volatility import atr as calc_atr
from patterns import candlestick, chart_patterns


def _simulate_forward(df: pd.DataFrame, start_pos: int, target: float,
                       invalidation: float, direction: str, max_candles: int) -> str:
    """
    Moves forward from the candle after start_pos up to max_candles candles.
    Output: 'win' | 'loss' | 'timeout'
    """
    n = len(df)
    end_pos = min(start_pos + max_candles, n - 1)
    for pos in range(start_pos + 1, end_pos + 1):
        high = df["high"].iloc[pos]
        low = df["low"].iloc[pos]
        if direction == "bullish":
            hit_target = high >= target
            hit_invalid = low <= invalidation
        else:
            hit_target = low <= target
            hit_invalid = high >= invalidation

        if hit_target and hit_invalid:
            # Both happened in one candle — since the intra-candle order is unknown,
            # conservatively we assume the invalidation happened first (loss)
            return "loss"
        if hit_target:
            return "win"
        if hit_invalid:
            return "loss"
    return "timeout"


def _summarize(outcomes: dict, instances: list, fa: str, bias: str) -> dict:
    total = sum(outcomes.values())
    return {
        "fa": fa,
        "bias": bias,
        "total_occurrences": total,
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "timeouts": outcomes["timeout"],
        "success_rate_pct": round(outcomes["win"] / total * 100, 1) if total else 0.0,
        "reliable_sample": total >= config.MIN_RELIABLE_SAMPLE_SIZE,
        "instances": instances,
    }


def backtest_candlestick_patterns(df: pd.DataFrame, forward_window: int = None) -> dict:
    """Historical success rate of each directional candlestick pattern (except doji/spinning top)."""
    if forward_window is None:
        forward_window = config.BACKTEST_FORWARD_WINDOW_CANDLES

    atr_series = calc_atr(df, config.ATR_PERIOD)
    all_patterns = candlestick.detect_all(df)

    results = {}
    for name in all_patterns.columns:
        bias = candlestick.PATTERN_REGISTRY[name]["bias"]
        if bias == "neutral":
            continue

        occurrences = all_patterns.index[all_patterns[name]]
        outcomes = {"win": 0, "loss": 0, "timeout": 0}
        instances = []

        for idx in occurrences:
            pos = df.index.get_loc(idx)
            if pos >= len(df) - 1:
                continue
            entry_atr = atr_series.iloc[pos]
            if pd.isna(entry_atr) or entry_atr <= 0:
                continue
            entry_price = df["close"].iloc[pos]

            if bias == "bullish":
                target = entry_price + entry_atr * config.BACKTEST_TARGET_ATR_MULTIPLIER
                invalidation = entry_price - entry_atr * config.ATR_STOP_LOSS_MULTIPLIER
            else:
                target = entry_price - entry_atr * config.BACKTEST_TARGET_ATR_MULTIPLIER
                invalidation = entry_price + entry_atr * config.ATR_STOP_LOSS_MULTIPLIER

            outcome = _simulate_forward(df, pos, target, invalidation, bias, forward_window)
            outcomes[outcome] += 1
            instances.append({"index": str(idx), "outcome": outcome})

        if sum(outcomes.values()) == 0:
            continue
        results[name] = _summarize(outcomes, instances, candlestick.PATTERN_REGISTRY[name]["fa"], bias)

    return results


def backtest_chart_patterns(df: pd.DataFrame, forward_window: int = None) -> dict:
    """
    Historical success rate of each chart pattern (a longer time window, since these patterns respond more slowly).

    Important note about backtest validity: each pattern's completion_index is
    based on a swing point that itself needed SWING_LOOKBACK_CANDLES following
    candles to be "confirmed" (because swing high/low is detected by looking at
    both sides). If we start the simulation exactly from completion_index, its
    first few candles are actually data that was "already seen" to confirm the
    pattern itself — a kind of future-information leak. To prevent this, we move
    the real simulation start point forward by SWING_LOOKBACK_CANDLES
    (actionable_pos); this is exactly the moment a real trader could first have
    seen this pattern as completed.
    """
    if forward_window is None:
        forward_window = config.BACKTEST_FORWARD_WINDOW_CANDLES * 2

    all_patterns = chart_patterns.scan_all(df)
    grouped = {}

    for p in all_patterns:
        raw_pos = df.index.get_loc(p["completion_index"])
        pos = raw_pos + config.SWING_LOOKBACK_CANDLES  # The truly actionable point
        if pos >= len(df) - 1:
            continue

        outcome = _simulate_forward(df, pos, p["target"], p["invalidation"], p["bias"], forward_window)

        g = grouped.setdefault(p["name"], {"fa": p["fa"], "bias": p["bias"],
                                            "outcomes": {"win": 0, "loss": 0, "timeout": 0},
                                            "instances": []})
        g["outcomes"][outcome] += 1
        g["instances"].append({"completion_index": str(p["completion_index"]), "outcome": outcome})

    results = {}
    for name, data in grouped.items():
        if sum(data["outcomes"].values()) == 0:
            continue
        results[name] = _summarize(data["outcomes"], data["instances"], data["fa"], data["bias"])

    return results


def backtest_all(df: pd.DataFrame) -> dict:
    """Backtests both categories (candlestick + chart) and returns them together."""
    return {
        "candlestick": backtest_candlestick_patterns(df),
        "chart": backtest_chart_patterns(df),
    }
