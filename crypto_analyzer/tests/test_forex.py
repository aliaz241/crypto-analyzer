#!/usr/bin/env python3
"""
tests/test_forex.py
تست آفلاین ماژول فارکس — بدون اینترنت:

    python tests/test_forex.py

پوشش:
۱) تجزیه‌ی .bi5 مصنوعی → تیک معلوم → کندل ساعتی معلوم (bid، حجم تیک،
   مرتب‌سازی بر اساس ms، نقطه‌ی JPY).
۲) resample به 4h/1d بدون ساخت کندل آخر هفته (شکاف‌ها می‌مانند).
۳) مدل هزینه‌ی اسپردی و forex_trade_r با جواب دستی (EURUSD و XAUUSD).
۴) مسیریابی: نماد فارکس → مسیر فارکس، نماد کریپتو → بایننس (mock)،
   و scan_forex_market روی داده‌ی مصنوعی کش‌شده (تصمیم EV بدون ورودی
   کش = NO_TRADE).
۵) قیمت paper برای فارکس از latest_price می‌آید و پوزیشن فارکس فقط
   دستی باز می‌شود.
۶) تجزیه‌ی پاسخ Yahoo با mock (حذف ردیف None، حجم صفر، مسیریابی
   get_candles به منبع اصلی).
"""

import lzma
import os
import struct
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from data import forex_client as fx

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}"
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _tick(ms, ask_i, bid_i, av, bv):
    return struct.pack(">iii", ms, ask_i, bid_i) + struct.pack(">ff", av, bv)


# ---------------------------------------------------------------------------
# ۱) تجزیه و کندل ساعتی
# ---------------------------------------------------------------------------
def test_parse_and_candle():
    # عمداً نامرتب: ms دوم بزرگ‌تر از سومی نیست — مرتب‌سازی باید open/close را درست کند
    raw = (_tick(120000, 110030, 110020, 0.5, 0.5)
           + _tick(0, 110010, 110000, 1.0, 2.0)
           + _tick(60000, 109990, 109980, 1.5, 1.0))
    ticks = fx.parse_bi5_ticks(lzma.compress(raw))
    check("تعداد تیک", len(ticks) == 3, str(len(ticks)))
    c = fx.ticks_to_hour_candle(ticks, 100000)
    check("کندل EURUSD: open از زودترین تیک", approx(c[0], 1.10000), str(c))
    check("کندل EURUSD: high", approx(c[1], 1.10020), str(c))
    check("کندل EURUSD: low", approx(c[2], 1.09980), str(c))
    check("کندل EURUSD: close از دیرترین تیک", approx(c[3], 1.10020), str(c))
    check("کندل EURUSD: حجم = جمع bid+ask", approx(c[4], 6.5), str(c))

    raw_jpy = _tick(0, 150123, 150120, 1.0, 1.0)
    cj = fx.ticks_to_hour_candle(fx.parse_bi5_ticks(lzma.compress(raw_jpy)), 1000)
    check("کندل USDJPY: نقطه 1e3", approx(cj[0], 150.120), str(cj))
    check("تیک خالی → None", fx.ticks_to_hour_candle(
        np.empty(0, dtype=fx._TICK_DTYPE), 100000) is None)


# ---------------------------------------------------------------------------
# ۲) resample و شکاف آخر هفته
# ---------------------------------------------------------------------------
def test_resample_gaps():
    idx = pd.date_range("2026-11-06 00:00", periods=5, freq="1h", tz="UTC")
    idx = idx.append(pd.date_range("2026-11-09 00:00", periods=3,
                                   freq="1h", tz="UTC"))
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 10.0}, index=idx)
    df.index.name = "datetime"
    d1 = fx._resample(df, "1d")
    check("روزانه: جمعه و دوشنبه جدا، شنبه/یکشنبه ساخته نشد",
          len(d1) == 2 and d1.index[0].day == 6 and d1.index[1].day == 9,
          str(list(d1.index)))
    h4 = fx._resample(df, "4h")
    check("4h: حجم جمعه (۵ ساعت → دو کندل 4h + پس‌مانده در resample)",
          len(h4) == 3, str(len(h4)))


# ---------------------------------------------------------------------------
# ۳) هزینه و R
# ---------------------------------------------------------------------------
def test_cost_model():
    from backtest.forex_backtest import forex_cost_price, forex_trade_r
    # EURUSD: اسپرد 1.0 + اسلیپیج 0.2 = 1.2 پیپ = 0.00012
    check("هزینه EURUSD", approx(forex_cost_price("EURUSD"), 0.00012),
          str(forex_cost_price("EURUSD")))
    # XAUUSD: (2.5 + 0.2) × 0.1 = 0.27
    check("هزینه XAUUSD", approx(forex_cost_price("XAUUSD"), 0.27),
          str(forex_cost_price("XAUUSD")))
    # USDJPY: (1.2 + 0.2) × 0.01 = 0.014
    check("هزینه USDJPY", approx(forex_cost_price("USDJPY"), 0.014),
          str(forex_cost_price("USDJPY")))
    # لانگ: ورود 1.1000، خروج 1.1100، ریسک 0.0050 → (0.0100 − 0.00012)/0.005 = 1.976
    r = forex_trade_r("long", 1.1000, 1.1100, 0.0050, "EURUSD")
    check("R لانگ با کسر اسپرد", approx(r, 1.976, 1e-9), str(r))
    # شورت: ورود 1.1000، خروج 1.0950، ریسک 0.0050 → (0.005 − 0.00012)/0.005 = 0.976
    r2 = forex_trade_r("short", 1.1000, 1.0950, 0.0050, "EURUSD")
    check("R شورت با کسر اسپرد", approx(r2, 0.976, 1e-9), str(r2))


# ---------------------------------------------------------------------------
# ۴) مسیریابی + اسکن روی کش مصنوعی
# ---------------------------------------------------------------------------
def _synthetic_1h(symbol, hours=1600, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01 00:00", periods=hours, freq="1h", tz="UTC")
    base = {"EURUSD": 1.10, "XAUUSD": 2300.0}.get(symbol, 1.20)
    steps = rng.normal(0, 0.0006, hours).cumsum()
    close = base * (1 + steps)
    df = pd.DataFrame({
        "open": close, "high": close * 1.0008, "low": close * 0.9992,
        "close": close, "volume": 1000.0}, index=idx)
    df.index.name = "datetime"
    return df


def test_routing_and_scan():
    import scanner
    from analysis import analyzer

    # سقف‌های کوچک تا کش مصنوعی کافی باشد و هیچ واکشی شبکه‌ای رخ ندهد
    config.FOREX_CANDLES_PER_TIMEFRAME = {"4h": 120, "1d": 30, "1h": 500}
    fx._save_cached_1h("EURUSD", _synthetic_1h("EURUSD"))
    df = analyzer.fetch_market("EURUSD", "4h", market=None)
    check("مسیریابی خودکار: EURUSD از فارکس",
          list(df.columns) == ["open", "high", "low", "close", "volume"]
          and len(df) == 120,
          str(df.shape))

    # کریپتو: مسیر بایننس با mock (نباید شبکه صدا زده شود)
    orig = analyzer.fetch_candles_binance
    try:
        analyzer.fetch_candles_binance = lambda s, t, total_candles=None: \
            _synthetic_1h("BTC", 100)
        df2 = analyzer.fetch_market("BTC", "4h", market=None)
        check("مسیریابی خودکار: BTC از بایننس (mock)", len(df2) == 100)
    finally:
        analyzer.fetch_candles_binance = orig

    res = scanner.scan_forex_market(symbols=["EURUSD"], timeframes=["4h"],
                                    progress=False)
    check("اسکن فارکس: market=forex", res.get("market") == "forex")
    cands = res["candidates"]
    check("اسکن فارکس: یک نامزد", len(cands) == 1 and res["scanned"] == 1,
          str(res["scanned"]))
    if cands:
        c = cands[0]
        check("نامزد فارکس: کلید market", c.get("market") == "forex")
        check("نامزد فارکس بدون کش EV → NO_TRADE",
              c.get("decision") == "NO_TRADE", str(c.get("decision")))


# ---------------------------------------------------------------------------
# ۵) paper فارکس
# ---------------------------------------------------------------------------
def test_paper_forex():
    import paper_trading as paper
    fx._save_cached_1h("EURUSD", _synthetic_1h("EURUSD", hours=100))
    # تازه‌سازی سبک latest_price در تست خاموش است تا شبکه صدا زده نشود
    # (کش مصنوعی قدیمی است و وگرنه واکشی ساعت‌های اخیر تلاش می‌شود).
    import data.forex_client as _fxc
    _saved_lp = _fxc.latest_price
    _fxc.latest_price = lambda s: (float(
        _fxc._load_cached_1h(s)["close"].iloc[-1]),
        _fxc._load_cached_1h(s).index[-1])
    try:
        price = paper.get_price("EURUSD")
    finally:
        _fxc.latest_price = _saved_lp
    cached = fx._load_cached_1h("EURUSD")
    check("قیمت paper فارکس = آخرین بستن کش",
          approx(price, float(cached["close"].iloc[-1])), str(price))

    import tempfile
    orig_state = paper.STATE_FILE
    try:
        paper.STATE_FILE = os.path.join(tempfile.mkdtemp(), "state.json")
        pos = paper.open_position(999001, "EURUSD", "long", 100)
        check("پوزیشن فارکس دستی باز شد", pos["symbol"] == "EURUSD"
              and pos.get("market") == "forex", str(pos.get("market")))
        try:
            paper.open_position(999001, "EURUSD", "long", 100, source="auto")
            check("پوزیشن فارکس خودکار ممنوع", False)
        except Exception:
            check("پوزیشن فارکس خودکار ممنوع", True)
        text = paper.positions_text(999001)
        check("نمایش پوزیشن: برچسب تأخیر فارکس", "با تأخیر" in text)
    finally:
        paper.STATE_FILE = orig_state


def test_yahoo_parse():
    """۶) تجزیه‌ی پاسخ Yahoo (mock requests.get — بدون شبکه)."""
    import json as _json

    import data.forex_client as fxc

    ts0 = 1762473600  # 2025-11-07 00:00 UTC
    payload = {"chart": {"result": [{
        "timestamp": [ts0, ts0 + 3600, ts0 + 7200],
        "indicators": {"quote": [{
            "open": [1.10, 1.11, None], "high": [1.12, 1.13, None],
            "low": [1.09, 1.10, None], "close": [1.11, 1.12, None],
            "volume": [0, None, 0]}]}}], "error": None}}

    class _Resp:
        status_code = 200

        def json(self):
            return payload

    orig_get = fxc.requests.get
    orig_primary = config.FOREX_PRIMARY
    try:
        fxc.requests.get = lambda url, timeout=None, headers=None: _Resp()
        config.FOREX_PRIMARY = "yahoo"
        df = fxc.fetch_yahoo_candles("EURUSD", "1h", 10)
        check("Yahoo: ردیف None حذف می‌شود", len(df) == 2, str(len(df)))
        check("Yahoo: حجم None/صفر → 0.0",
              float(df["volume"].sum()) == 0.0)
        check("Yahoo: close معلوم", approx(df["close"].iloc[-1], 1.12),
              str(df["close"].iloc[-1]))
        df2 = fxc.get_candles("EURUSD", "1h", 10)
        check("get_candles: منبع اصلی Yahoo", len(df2) == 2)
    finally:
        fxc.requests.get = orig_get
        config.FOREX_PRIMARY = orig_primary


def main():
    test_parse_and_candle()
    test_resample_gaps()
    test_cost_model()
    test_routing_and_scan()
    test_paper_forex()
    test_yahoo_parse()
    print(f"\n{'❌ ناموفق: ' + ', '.join(fails) if fails else '✅ همه‌ی تست‌های فارکس پاس شدند'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
