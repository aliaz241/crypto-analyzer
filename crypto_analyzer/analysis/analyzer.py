"""
analysis/analyzer.py
هماهنگ‌کننده‌ی کل فرآیند: گرفتن داده -> محاسبه‌ی اندیکاتورها -> تشخیص الگوها
-> سطوح حمایت/مقاومت -> حد ضرر -> بک‌تست -> گزارش نهایی.
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
    """لیست دیکشنری کندل‌ها را به DataFrame ایندکس‌شده با datetime تبدیل می‌کند."""
    df = pd.DataFrame(candles)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()
    return df


RESAMPLE_RULE_MAP = {
    "1h": "1h",     # خام 30 دقیقه‌ای -> بالا به 1 ساعته
    "4h": None,     # خام از قبل 4 ساعته‌ی طبیعی است — resample دوباره لازم نیست
    "1d": "1D",     # خام 4 ساعته -> بالا به روزانه
    "1w": "1W",     # خام 4 روزه -> بالا به هفتگی
}


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """کندل‌های خام را در صورت نیاز به تایم‌فریم بزرگ‌تر resample می‌کند."""
    rule = RESAMPLE_RULE_MAP.get(timeframe)
    if rule is None:
        return df
    resampled = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return resampled.dropna()


def fetch_candles(symbol_query: str, vs_currency: str = None, timeframe: str = "1d") -> pd.DataFrame:
    """نماد را جستجو، کندل‌ها را از CoinGecko می‌گیرد و به DataFrame آماده تبدیل می‌کند."""
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
# Binance data (منبع اصلی از نسخه‌ی ۲ — کندل بومی، بدون resample، با حجم واقعی)
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
    کندل‌ها را مستقیم از بایننس می‌گیرد (جفت USDT، بدون نیاز به API key).
    کندل‌ها بومیِ همان تایم‌فریم‌اند، پس هیچ resample ای انجام نمی‌شود و حجم
    معاملات هم حجم واقعی صرافی است (رفع مشکلات ممیزی ۱ و ۷).
    تعداد کندل پیش‌فرض از config.BINANCE_CANDLES_PER_TIMEFRAME خوانده می‌شود.
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
# Indicator snapshot (آخرین مقادیر)
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

    # جهت OBV در چند کندل اخیر (برای کمک به تشخیص واگرایی حجم/قیمت)
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
    کندل‌های فارکس از فید Dukascopy (ماژول افزودنی فارکس — FOREX_FA.md).
    شکل خروجی دقیقاً مثل fetch_candles_binance است؛ عمق واقعی به فید
    بستگی دارد و اگر از MIN_CANDLES_REQUIRED کمتر باشد خطای روشن می‌دهد.
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
    مسیریابی بازار (افزودنی): market صریح ("forex"/"crypto") مقدم است؛
    بدون آن، نمادهای مجموعه‌ی فارکس (config.FOREX_SYMBOLS) به فید
    فارکس و بقیه دقیقاً مثل قبل به بایننس می‌روند.
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
    تحلیل کامل یک جفت‌ارز/تایم‌فریم: اندیکاتورها، الگوهای فعلی، S/R، حد ضرر، بک‌تست.
    data_source: "binance" (پیش‌فرض نسخه‌ی ۲ — کندل بومی و تاریخچه‌ی بلند) یا
    "coingecko" (رفتار قدیمی نسخه‌ی ۱، به عنوان مسیر جایگزین نگه داشته شده)
    یا "forex" (فید Dukascopy — افزودنی فارکس) یا "market" (مسیریابی
    خودکار بر اساس نماد: فارکس یا کریپتو).
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
    """همان تحلیل کامل، ولی روی یک DataFrame آماده (برای تست با داده‌ی مصنوعی هم قابل استفاده)."""
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
