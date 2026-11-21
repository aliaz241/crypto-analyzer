"""
indicators/momentum.py
اندیکاتورهای مومنتوم: RSI, Stochastic, CCI, Williams %R
"""

import numpy as np
import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """
    Relative Strength Index.
    بالای 70 = اشباع خرید (overbought) | زیر 30 = اشباع فروش (oversold)
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))
    # وقتی avg_loss صفر است یعنی RSI باید 100 باشد
    rsi_value = rsi_value.where(avg_loss != 0, 100.0)
    return rsi_value


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                smooth_k: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    خروجی: DataFrame با ستون‌های k, d
    بالای 80 = اشباع خرید | زیر 20 = اشباع فروش
    """
    lowest_low = df["low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df["high"].rolling(window=k_period, min_periods=k_period).max()

    raw_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"k": k, "d": d})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index.
    بالای +100 = روند صعودی قوی / احتمال اشباع خرید
    زیر -100 = روند نزولی قوی / احتمال اشباع فروش
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
    mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R.
    بالای -20 = اشباع خرید | زیر -80 = اشباع فروش
    """
    highest_high = df["high"].rolling(window=period, min_periods=period).max()
    lowest_low = df["low"].rolling(window=period, min_periods=period).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
