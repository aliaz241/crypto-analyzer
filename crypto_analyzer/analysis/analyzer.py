"""
analysis/analyzer.py
Coordinator of the whole process: fetch data -> compute indicators -> detect patterns
-> support/resistance levels -> stop loss -> backtest -> final report.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.coingecko_client import CoinGeckoClient, CoinGeckoError
from data.binance_client import BinanceClient, BinanceError
from indicators import trend, momentum, volatility, levels
from patterns import candlestick, chart_patterns
from risk import stop_loss
from backtest import engine as backtest_engine


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_dataframe(candles: list) -> pd.DataFrame:
    """Converts a list of candle dicts into a DataFrame indexed by datetime."""
    df = pd.DataFrame(candles)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()
    return df


RESAMPLE_RULE_MAP = {
    "1h": "1h",     # raw 30-minute -> up to 1-hour
    "4h": None,     # raw is already native 4-hour — no need to resample again
    "1d": "1D",     # raw 4-hour -> up to daily
    "1w": "1W",     # raw 4-day -> up to weekly
}


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples raw candles into a larger timeframe if needed."""
    rule = RESAMPLE_RULE_MAP.get(timeframe)
    if rule is None:
        return df
    resampled = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return resampled.dropna()


def fetch_candles(symbol_query: str, vs_currency: str = None, timeframe: str = "1d") -> pd.DataFrame:
    """Searches for the symbol, fetches candles from CoinGecko and converts them into a ready DataFrame."""
    if vs_currency is None:
        vs_currency = config.DEFAULT_VS_CURRENCY
    if timeframe not in config.TIMEFRAME_DAYS_MAP:
        raise AnalysisError(f"تایم‌فریم '{timeframe}' پشتیبانی نمی‌شود. گزینه‌ها: {list(config.TIMEFRAME_DAYS_MAP)}")

    client = CoinGeckoClient()
    coin_id = client.search_coin_id(symbol_query)
    days = config.TIMEFRAME_DAYS_MAP[timeframe]
    candles = client.get_candles_with_volume(coin_id, vs_currency, days)

    df = build_dataframe(candles)
    df = resample_to_timeframe(df, timeframe)

    if len(df) < config.MIN_CANDLES_REQUIRED:
        raise AnalysisError(
            f"فقط {len(df)} کندل در دسترس بود؛ حداقل {config.MIN_CANDLES_REQUIRED} کندل "
            f"برای تحلیل و بک‌تست معنادار لازم است."
        )
    return df


# ---------------------------------------------------------------------------
# Binance data (primary source since v2 — native candles, no resample, real volume)
# ---------------------------------------------------------------------------
BINANCE_INTERVAL_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


def fetch_candles_binance(symbol: str, timeframe: str = "4h",
                          total_candles: int = None) -> pd.DataFrame:
    """
    Fetches candles directly from Binance (USDT pair, no API key needed).
    The candles are native to that timeframe, so no resampling is done and the
    trading volume is the exchange's real volume (fixes audit issues 1 and 7).
    The default candle count is read from config.BINANCE_CANDLES_PER_TIMEFRAME.
    """
    if timeframe not in BINANCE_INTERVAL_MAP:
        raise AnalysisError(
            f"تایم‌فریم '{timeframe}' پشتیبانی نمی‌شود. گزینه‌ها: {list(BINANCE_INTERVAL_MAP.keys())}")
    if total_candles is None:
        total_candles = config.BINANCE_CANDLES_PER_TIMEFRAME[timeframe]

    client = BinanceClient()
    try:
        candles = client.get_candles(symbol, BINANCE_INTERVAL_MAP[timeframe], total_candles)
    except BinanceError as exc:
        raise AnalysisError(str(exc)) from exc

    df = build_dataframe(candles)

    if len(df) < config.MIN_CANDLES_REQUIRED:
        raise AnalysisError(
            f"فقط {len(df)} کندل در دسترس بود؛ حداقل {config.MIN_CANDLES_REQUIRED} کندل "
            f"برای تحلیل و بک‌تست معنادار لازم است."
        )
    return df


# ---------------------------------------------------------------------------
# Indicator snapshot (latest values)
# ---------------------------------------------------------------------------
def _safe_round(value, decimals=6):
    return round(float(value), decimals) if pd.notna(value) and np.isfinite(value) else None


def compute_indicator_snapshot(df: pd.DataFrame) -> dict:
    rsi_series = momentum.rsi(df)
    macd_df = trend.macd(df)
    bb_df = volatility.bollinger_bands(df)
    stoch_df = momentum.stochastic(df)
    adx_df = trend.adx(df)
    atr_series = volatility.atr(df)
    cci_series = momentum.cci(df)
    williams_series = momentum.williams_r(df)
    keltner_df = volatility.keltner_channel(df)
    obv_series = volatility.obv(df)

    # OBV direction over the last few candles (to help detect volume/price divergence)
    obv_trend = None
    if len(obv_series.dropna()) >= 5:
        obv_recent = obv_series.iloc[-5:]
        obv_trend = "up" if obv_recent.iloc[-1] > obv_recent.iloc[0] else "down"

    return {
        "price": float(df["close"].iloc[-1]),
        "rsi": _safe_round(rsi_series.iloc[-1], 1),
        "macd": {
            "macd": _safe_round(macd_df["macd"].iloc[-1]),
            "signal": _safe_round(macd_df["signal"].iloc[-1]),
            "histogram": _safe_round(macd_df["histogram"].iloc[-1]),
        },
        "bollinger": {
            "upper": _safe_round(bb_df["upper"].iloc[-1]),
            "middle": _safe_round(bb_df["middle"].iloc[-1]),
            "lower": _safe_round(bb_df["lower"].iloc[-1]),
        },
        "keltner": {
            "upper": _safe_round(keltner_df["upper"].iloc[-1]),
            "middle": _safe_round(keltner_df["middle"].iloc[-1]),
            "lower": _safe_round(keltner_df["lower"].iloc[-1]),
        },
        "stochastic": {
            "k": _safe_round(stoch_df["k"].iloc[-1], 1),
            "d": _safe_round(stoch_df["d"].iloc[-1], 1),
        },
        "adx": _safe_round(adx_df["adx"].iloc[-1], 1),
        "atr": _safe_round(atr_series.iloc[-1]),
        "cci": _safe_round(cci_series.iloc[-1], 1),
        "williams_r": _safe_round(williams_series.iloc[-1], 1),
        "obv_trend": obv_trend,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def fetch_candles_forex(symbol: str, timeframe: str = "4h",
                        total_candles: int = None) -> pd.DataFrame:
    """
    Forex candles from the Dukascopy feed (forex add-on module — FOREX_FA.md).
    The output shape is exactly like fetch_candles_binance; the real depth depends
    on the feed, and if it is below MIN_CANDLES_REQUIRED it raises a clear error.
    """
    from data import forex_client
    try:
        df = forex_client.get_candles(symbol, timeframe, total_candles)
    except forex_client.ForexError as exc:
        raise AnalysisError(str(exc)) from exc
    if len(df) < config.MIN_CANDLES_REQUIRED:
        raise AnalysisError(
            f"فقط {len(df)} کندل فارکس در دسترس بود؛ حداقل "
            f"{config.MIN_CANDLES_REQUIRED} کندل لازم است.")
    return df


def fetch_market(symbol: str, timeframe: str = "4h",
                 market: str = None) -> pd.DataFrame:
    """
    Market routing (add-on): an explicit market ("forex"/"crypto") takes precedence;
    without it, symbols in the forex set (config.FOREX_SYMBOLS) go to the forex
    feed and everything else goes to Binance exactly as before.
    """
    from data import forex_client
    if market is None:
        market = "forex" if forex_client.is_forex_symbol(symbol) else "crypto"
    if market == "forex":
        return fetch_candles_forex(symbol, timeframe)
    if market == "crypto":
        return fetch_candles_binance(symbol, timeframe)
    raise AnalysisError(
        f"بازار '{market}' ناشناخته است. گزینه‌ها: crypto, forex")


def run_full_analysis(symbol_query: str, vs_currency: str = None, timeframe: str = "1d",
                      data_source: str = "binance") -> dict:
    """
    Full analysis of one pair/timeframe: indicators, current patterns, S/R, stop loss, backtest.
    data_source: "binance" (default in v2 — native candles and long history) or
    "coingecko" (the old v1 behavior, kept as a fallback path)
    or "forex" (Dukascopy feed — forex add-on) or "market" (automatic
    routing by symbol: forex or crypto).
    """
    if data_source == "binance":
        df = fetch_candles_binance(symbol_query, timeframe)
    elif data_source == "coingecko":
        df = fetch_candles(symbol_query, vs_currency, timeframe)
    elif data_source == "forex":
        df = fetch_candles_forex(symbol_query, timeframe)
    elif data_source == "market":
        df = fetch_market(symbol_query, timeframe)
    else:
        raise AnalysisError(
            f"منبع داده‌ی '{data_source}' ناشناخته است. گزینه‌ها: binance, coingecko, forex, market")
    return build_report(df, symbol_query, timeframe)


def build_report(df: pd.DataFrame, symbol_label: str, timeframe: str) -> dict:
    """The same full analysis, but on a ready DataFrame (also usable for tests with synthetic data)."""
    indicator_snapshot = compute_indicator_snapshot(df)
    sr_levels = levels.support_resistance_levels(df)

    latest_candle_patterns = candlestick.detect_latest(df, n_candles=3)
    latest_chart_patterns = chart_patterns.latest_patterns(df, recent_candles=15)

    stop_long = stop_loss.combined_stop_loss(df, "long")
    stop_short = stop_loss.combined_stop_loss(df, "short")

    backtest_results = backtest_engine.backtest_all(df)

    return {
        "symbol": symbol_label,
        "timeframe": timeframe,
        "candle_count": len(df),
        "last_updated": str(df.index[-1]),
        "indicators": indicator_snapshot,
        "support_resistance": sr_levels,
        "latest_candlestick_patterns": latest_candle_patterns,
        "latest_chart_patterns": latest_chart_patterns,
        "stop_loss_long": stop_long,
        "stop_loss_short": stop_short,
        "backtest": backtest_results,
    }
