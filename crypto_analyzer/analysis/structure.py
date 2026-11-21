"""
analysis/structure.py
واگرایی‌ها و ساختار بازار — ماژول افزودنی (گام ۴).

این ماژول هیچ‌کدام از محاسبات موجود را تغییر نمی‌دهد و build_report هم از
آن استفاده نمی‌کند. خروجی‌اش (analyze_structure) فقط وقتی اثر دارد که
صراحتاً به‌عنوان extras به scanner.score_candidate داده شود؛ در آن حالت
مؤلفه‌ی «structure» (تا ±۱۰ امتیاز) به امتیاز اضافه می‌شود و مقیاس /۱۱۰
خواهد بود. مسیر پیش‌فرض اسکنر بدون این ماژول و با مقیاس /۱۰۰ می‌ماند.

تعریف‌ها:
- swing high/low: نقطه‌ای که از SWING_LOOKBACK_CANDLES همسایه‌ی دو طرفش
  بزرگ‌تر/کوچک‌تر است (همان ثابت الگوهای نموداری پروژه).
- واگرایی معمولی RSI/MACD: بین دو swing هم‌نوع آخر قیمت؛ سقف بالاترِ قیمت
  با سقف پایین‌ترِ اسیلاتور = bearish؛ کف پایین‌ترِ قیمت با کف بالاترِ
  اسیلاتور = bullish. swing دوم باید در STRUCTURE_RECENT_CANDLES کندل اخیر
  باشد، وگرنه واگرایی کهنه حساب نمی‌شود.
- ساختار بازار: مقایسه‌ی دو swing high و دو swing low آخر (HH+HL =
  uptrend، LH+LL = downtrend، وگرنه sideways).
- رویداد BOS/CHoCH (ساده‌سازی مستند): بعد از تأیید آخرین swing، اولین
  بسته‌شدن فراتر از آن سطح یک شکست است؛ هم‌جهت با ساختار = BOS و خلاف
  ساختار = CHoCH. (ردیابی کامل رویدادهای میانی انجام نمی‌شود؛ فقط آخرین
  سطح و اولین شکست بعد از آن.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from indicators import momentum as momentum_ind
from indicators import trend as trend_ind


def _swing_indices(values, lookback: int, kind: str) -> list:
    """ایندکس‌های swing روی یک آرایه‌ی عددی (مقایسه‌ی اکید با همسایه‌ها)."""
    out = []
    v = np.asarray(values, dtype=float)
    for i in range(lookback, len(v) - lookback):
        if np.isnan(v[i]):
            continue
        seg = v[i - lookback:i + lookback + 1]
        if kind == "high" and np.all(v[i] > np.delete(seg, lookback)):
            out.append(i)
        elif kind == "low" and np.all(v[i] < np.delete(seg, lookback)):
            out.append(i)
    return out


def _last_divergence(price, osc, lookback: int, recent: int):
    """
    آخرین واگرایی معمولی بین قیمت و یک اسیلاتور.
    خروجی: "bullish" | "bearish" | None
    """
    n = len(price)
    highs = _swing_indices(price, lookback, "high")
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if b >= n - recent and not (np.isnan(osc[a]) or np.isnan(osc[b])):
            if price[b] > price[a] and osc[b] < osc[a]:
                return "bearish"
    lows = _swing_indices(price, lookback, "low")
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if b >= n - recent and not (np.isnan(osc[a]) or np.isnan(osc[b])):
            if price[b] < price[a] and osc[b] > osc[a]:
                return "bullish"
    return None


def _market_structure(df, lookback: int):
    """(ساختار, آخرین رویداد) — تعریف و ساده‌سازی در ابتدای فایل آمده است."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    sh = _swing_indices(highs, lookback, "high")
    sl = _swing_indices(lows, lookback, "low")

    structure = "sideways"
    if len(sh) >= 2 and len(sl) >= 2:
        if highs[sh[-1]] > highs[sh[-2]] and lows[sl[-1]] > lows[sl[-2]]:
            structure = "uptrend"
        elif highs[sh[-1]] < highs[sh[-2]] and lows[sl[-1]] < lows[sl[-2]]:
            structure = "downtrend"

    breaks = []
    if sh:
        level = highs[sh[-1]]
        for pos in range(sh[-1] + lookback, len(closes)):
            if closes[pos] > level:
                breaks.append((pos, "bullish", level))
                break
    if sl:
        level = lows[sl[-1]]
        for pos in range(sl[-1] + lookback, len(closes)):
            if closes[pos] < level:
                breaks.append((pos, "bearish", level))
                break

    event = None
    if breaks:
        pos, bias, level = max(breaks, key=lambda b: b[0])
        if structure == "uptrend":
            etype = "BOS" if bias == "bullish" else "CHoCH"
        elif structure == "downtrend":
            etype = "BOS" if bias == "bearish" else "CHoCH"
        else:
            etype = "BOS"
        event = {"type": etype, "direction": bias,
                 "index": str(df.index[pos]), "level": float(level)}
    return structure, event


def analyze_structure(df) -> dict:
    """
    خروجی آماده‌ی پاس‌دادن به score_candidate(..., extras=...):
    {"rsi_divergence", "macd_divergence", "structure", "last_event"}
    """
    lookback = config.SWING_LOOKBACK_CANDLES
    recent = config.STRUCTURE_RECENT_CANDLES
    close = df["close"].to_numpy(dtype=float)
    rsi = momentum_ind.rsi(df).to_numpy(dtype=float)
    macd_line = trend_ind.macd(df)["macd"].to_numpy(dtype=float)
    structure, event = _market_structure(df, lookback)
    return {
        "rsi_divergence": _last_divergence(close, rsi, lookback, recent),
        "macd_divergence": _last_divergence(close, macd_line, lookback, recent),
        "structure": structure,
        "last_event": event,
    }
