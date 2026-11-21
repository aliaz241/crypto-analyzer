"""
indicators/volatility.py
اندیکاتورهای نوسان: Bollinger Bands, ATR, Keltner Channel, OBV
"""

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — معیار اصلی نوسان، پایه‌ی محاسبه‌ی حد ضرر."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                     column: str = "close") -> pd.DataFrame:
    """
    باند بولینگر.
    خروجی: DataFrame با ستون‌های middle, upper, lower, bandwidth
    """
    middle = df[column].rolling(window=period, min_periods=period).mean()
    std = df[column].rolling(window=period, min_periods=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    bandwidth = (upper - lower) / middle
    return pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
    })


def keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                     atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    کانال کلتنر — مشابه بولینگر ولی بر پایه‌ی ATR به جای انحراف معیار.
    """
    middle = df["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    atr_value = atr(df, atr_period)
    upper = middle + (atr_value * atr_multiplier)
    lower = middle - (atr_value * atr_multiplier)
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — حجم تجمعی بر اساس جهت حرکت قیمت.
    واگرایی بین OBV و قیمت می‌تواند نشانه‌ی ضعف روند باشد.
    """
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()
