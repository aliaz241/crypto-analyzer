"""
indicators/trend.py
اندیکاتورهای روند: SMA, EMA, MACD, ADX
همه توابع یک pandas.DataFrame با ستون‌های open/high/low/close/volume می‌گیرند
و یک pandas.Series (یا DataFrame چندستونی) هم‌تراز با ایندکس ورودی برمی‌گردانند.
"""

import numpy as np
import pandas as pd

from .volatility import true_range as _true_range


def sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """میانگین متحرک ساده."""
    return df[column].rolling(window=period, min_periods=period).mean()


def ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """میانگین متحرک نمایی."""
    return df[column].ewm(span=period, adjust=False, min_periods=period).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
          column: str = "close") -> pd.DataFrame:
    """
    MACD کلاسیک.
    خروجی: DataFrame با ستون‌های macd, signal, histogram
    """
    ema_fast = df[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index — قدرت روند (بدون جهت).
    خروجی: DataFrame با ستون‌های adx, plus_di, minus_di
    adx بالای 25 یعنی روند قوی؛ زیر 20 یعنی بازار رنج/بدون روند مشخص.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx_line,
        "plus_di": plus_di,
        "minus_di": minus_di,
    })
