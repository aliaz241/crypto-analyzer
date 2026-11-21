"""
risk/stop_loss.py
Computing the stop loss by two methods: based on ATR and based on price structure (the last high/low).
"""

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.volatility import atr as calc_atr
from indicators.levels import find_swing_points


def atr_based_stop_loss(df: pd.DataFrame, direction: str,
                         atr_period: int = None, multiplier: float = None) -> dict:
    """
    Stop loss based on a multiple of ATR from the current price — proportional to the market's real volatility.
    direction: 'long' or 'short'
    """
    if atr_period is None:
        atr_period = config.ATR_PERIOD
    if multiplier is None:
        multiplier = config.ATR_STOP_LOSS_MULTIPLIER

    atr_series = calc_atr(df, atr_period)
    current_atr = atr_series.iloc[-1]
    current_price = df["close"].iloc[-1]

    if pd.isna(current_atr):
        return {"stop_loss": None, "reason": "داده‌ی کافی برای محاسبه‌ی ATR وجود ندارد"}

    distance = current_atr * multiplier
    if direction == "long":
        stop = current_price - distance
    elif direction == "short":
        stop = current_price + distance
    else:
        raise ValueError("direction باید 'long' یا 'short' باشد")

    return {
        "stop_loss": float(stop),
        "method": "atr",
        "atr_value": float(current_atr),
        "distance": float(distance),
        "distance_pct": float(distance / current_price * 100),
    }


def structure_based_stop_loss(df: pd.DataFrame, direction: str, lookback_swings: int = 1) -> dict:
    """
    Stop loss based on the last structural high/low (swing) — more logical from a
    price-action standpoint, because it picks the point which, if broken, means
    the analysis hypothesis was wrong.
    """
    swing_df = find_swing_points(df)
    current_price = df["close"].iloc[-1]

    if direction == "long":
        recent_lows = swing_df.loc[swing_df["is_swing_low"], "low"]
        if recent_lows.empty:
            return {"stop_loss": None, "reason": "swing low‌ای برای تعیین حد ضرر پیدا نشد"}
        stop = recent_lows.iloc[-lookback_swings:].min()
    elif direction == "short":
        recent_highs = swing_df.loc[swing_df["is_swing_high"], "high"]
        if recent_highs.empty:
            return {"stop_loss": None, "reason": "swing high‌ای برای تعیین حد ضرر پیدا نشد"}
        stop = recent_highs.iloc[-lookback_swings:].max()
    else:
        raise ValueError("direction باید 'long' یا 'short' باشد")

    distance = abs(current_price - stop)
    return {
        "stop_loss": float(stop),
        "method": "structure",
        "distance": float(distance),
        "distance_pct": float(distance / current_price * 100),
    }


def combined_stop_loss(df: pd.DataFrame, direction: str) -> dict:
    """
    Computes both methods and returns the more conservative stop loss (farther
    from the current price, i.e. less risk of an early stop hit) as the main
    suggestion.
    """
    atr_sl = atr_based_stop_loss(df, direction)
    structure_sl = structure_based_stop_loss(df, direction)

    candidates = [sl for sl in (atr_sl, structure_sl) if sl.get("stop_loss") is not None]
    if not candidates:
        return {"stop_loss": None, "reason": "هیچ‌کدام از دو روش قابل محاسبه نبود"}

    # More conservative = farther from the current price
    recommended = max(candidates, key=lambda x: x["distance"])

    return {
        "recommended": recommended,
        "atr_based": atr_sl,
        "structure_based": structure_sl,
    }
