"""
backtest/engine.py
بک‌تست الگوها روی داده‌ی تاریخی.

برای هر رخداد تاریخی یک الگو، بررسی می‌شود که قیمت ابتدا به «هدف» رسیده
(موفق / win) یا ابتدا سطح «بطلان الگو» را لمس کرده (ناموفق / loss) یا در بازه‌ی
زمانی مشخص هیچ‌کدام رخ نداده (timeout). نرخ موفقیت نهایی = wins / total.

⚠️ این یک آمار تاریخی از عملکرد گذشته‌ی الگو روی همین دارایی/تایم‌فریم است،
نه تضمینی برای آینده. رفتار بازار می‌تواند تغییر کند.
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
    از کندل بعد از start_pos تا حداکثر max_candles کندل جلو می‌رود.
    خروجی: 'win' | 'loss' | 'timeout'
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
            # هر دو در یک کندل رخ داده — چون ترتیب داخل کندل مشخص نیست،
            # محافظه‌کارانه فرض می‌کنیم ابتدا بطلان رخ داده (loss)
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
    """نرخ موفقیت تاریخی هر الگوی کندلی جهت‌دار (غیر از دوجی/فرفره)."""
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
    نرخ موفقیت تاریخی هر الگوی نموداری (بازه‌ی زمانی بلندتر چون این الگوها کندتر جواب می‌دهند).

    نکته‌ی مهم درباره‌ی اعتبار بک‌تست: completion_index هر الگو بر پایه‌ی یک
    swing point تعیین می‌شود که خودش برای «تأیید شدن» به SWING_LOOKBACK_CANDLES
    کندل بعدی نیاز داشته (چون swing high/low با نگاه‌کردن به هر دو طرف تشخیص
    داده می‌شود). اگر شبیه‌سازی را دقیقاً از completion_index شروع کنیم، چند
    کندل اول آن در واقع داده‌ای است که برای تأیید خود الگو «از قبل دیده شده»
    — یعنی یک‌جور نشت اطلاعات آینده. برای جلوگیری از این، نقطه‌ی واقعی شروع
    شبیه‌سازی را به‌اندازه‌ی SWING_LOOKBACK_CANDLES جلو می‌بریم (actionable_pos)؛
    این دقیقاً همان لحظه‌ای است که یک معامله‌گر واقعی می‌توانست اولین‌بار این
    الگو را تکمیل‌شده ببیند.
    """
    if forward_window is None:
        forward_window = config.BACKTEST_FORWARD_WINDOW_CANDLES * 2

    all_patterns = chart_patterns.scan_all(df)
    grouped = {}

    for p in all_patterns:
        raw_pos = df.index.get_loc(p["completion_index"])
        pos = raw_pos + config.SWING_LOOKBACK_CANDLES  # نقطه‌ی واقعاً قابل‌اقدام
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
    """هر دو دسته (کندلی + نموداری) را بک‌تست می‌کند و یکجا برمی‌گرداند."""
    return {
        "candlestick": backtest_candlestick_patterns(df),
        "chart": backtest_chart_patterns(df),
    }
