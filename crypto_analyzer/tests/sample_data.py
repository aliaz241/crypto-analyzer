"""
tests/sample_data.py
Generating synthetic candle data for testing the whole pipeline without needing the internet/CoinGecko.
This data is not real — it is only useful for making sure the code does not crash
and the patterns are detected correctly. Do not interpret the output values as real prices.
"""

import numpy as np
import pandas as pd


def generate_synthetic_candles(n: int = 300, start_price: float = 50000.0,
                                seed: int = 42, freq: str = "1D") -> pd.DataFrame:
    """
    Generates a random price series but with a realistic trend and volatility
    (random walk + drift) so that indicators and pattern detection are tested
    on data with market-like behavior.
    """
    rng = np.random.default_rng(seed)

    # A combination of several regimes (bullish/bearish/ranging) so that different patterns actually occur
    n_segments = 6
    segment_len = n // n_segments
    drifts = rng.choice([0.0015, -0.0012, 0.0002, -0.0005, 0.002, -0.0018], size=n_segments)

    returns = []
    for seg in range(n_segments):
        length = segment_len if seg < n_segments - 1 else n - segment_len * (n_segments - 1)
        seg_returns = rng.normal(loc=drifts[seg], scale=0.018, size=length)
        returns.extend(seg_returns)

    returns = np.array(returns)
    close = start_price * np.exp(np.cumsum(returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    intrabar_range = np.abs(rng.normal(loc=0.01, scale=0.006, size=n)) * close
    high = np.maximum(open_, close) + intrabar_range * rng.uniform(0.2, 1.0, size=n)
    low = np.minimum(open_, close) - intrabar_range * rng.uniform(0.2, 1.0, size=n)
    volume = np.abs(rng.normal(loc=1_000_000, scale=300_000, size=n))

    index = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=n, freq=freq)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=index)
    return df
