"""
data/coingecko_client.py
کلاینت گرفتن داده‌ی قیمت/کندل از CoinGecko.

نکته‌ی مهم: CoinGecko در پلن رایگان، OHLC خام را فقط برای بازه‌های محدودی
می‌دهد (1/7/14/30/90/180/365 روز) و خروجی آن [timestamp, open, high, low, close]
است — بدون حجم معاملات. برای حجم از اندپوینت market_chart استفاده و بر اساس
timestamp با کندل‌ها merge می‌کنیم.
"""

import time
from datetime import datetime, timezone

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CoinGeckoError(Exception):
    """خطای مربوط به ارتباط یا پاسخ CoinGecko."""
    pass


class CoinGeckoClient:
    def __init__(self):
        self.base_url = config.COINGECKO_BASE_URL
        self.session = requests.Session()
        if config.COINGECKO_API_KEY:
            self.session.headers.update({"x-cg-demo-api-key": config.COINGECKO_API_KEY})
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _throttle(self):
        """جلوگیری از برخورد با محدودیت نرخ درخواست."""
        elapsed = time.time() - self._last_request_time
        wait = config.MIN_SECONDS_BETWEEN_REQUESTS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        last_error = None
        for attempt in range(1, config.REQUEST_RETRY_COUNT + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS
                )
                if response.status_code == 429:
                    # محدودیت نرخ — کمی بیشتر صبر کن و دوباره تلاش کن
                    time.sleep(config.REQUEST_RETRY_DELAY_SECONDS * attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(config.REQUEST_RETRY_DELAY_SECONDS)
        raise CoinGeckoError(f"درخواست به CoinGecko شکست خورد: {last_error}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search_coin_id(self, query: str) -> str:
        """
        تبدیل نماد یا نام ارز (مثل 'btc' یا 'bitcoin') به coin_id که
        CoinGecko برای سایر endpoint ها لازم دارد.
        """
        data = self._get("/search", {"query": query})
        coins = data.get("coins", [])
        if not coins:
            raise CoinGeckoError(f"ارزی با نام/نماد '{query}' پیدا نشد.")
        # اولین نتیجه معمولاً معتبرترین/پرحجم‌ترین تطابق است
        return coins[0]["id"]

    def get_ohlc(self, coin_id: str, vs_currency: str, days: int) -> list:
        """
        خروجی: لیستی از [timestamp_ms, open, high, low, close]
        """
        params = {"vs_currency": vs_currency, "days": days}
        data = self._get(f"/coins/{coin_id}/ohlc", params)
        if not data:
            raise CoinGeckoError("داده‌ی کندلی خالی برگشت — نماد یا بازه را بررسی کن.")
        return data

    def get_market_chart(self, coin_id: str, vs_currency: str, days: int) -> dict:
        """
        خروجی شامل prices, market_caps, total_volumes — هرکدام لیستی
        از [timestamp_ms, value]. از این برای استخراج حجم معاملات استفاده می‌شود.
        """
        params = {"vs_currency": vs_currency, "days": days}
        return self._get(f"/coins/{coin_id}/market_chart", params)

    def get_candles_with_volume(self, coin_id: str, vs_currency: str, days: int) -> list:
        """
        کندل‌های OHLC را با حجم معاملات (از market_chart) ترکیب می‌کند.

        نکته: دانه‌بندی market_chart معمولاً ریزتر از OHLC است (مثلاً حجم
        ساعتی در برابر کندل 4ساعته)، پس به‌جای برداشتن «نزدیک‌ترین» نقطه،
        همه‌ی نقاط حجمی که در بازه‌ی زمانی هر کندل (از پایان کندل قبلی تا
        پایان کندل فعلی) قرار می‌گیرند را جمع می‌زنیم — دقیق‌تر از matching
        تک‌نقطه‌ای.
        خروجی: لیستی از دیکشنری {timestamp, open, high, low, close, volume}
        """
        ohlc = self.get_ohlc(coin_id, vs_currency, days)
        chart = self.get_market_chart(coin_id, vs_currency, days)
        volumes = chart.get("total_volumes", [])

        ohlc_timestamps = [row[0] for row in ohlc]

        def _summed_volume(idx: int) -> float:
            if not volumes:
                return 0.0
            window_end = ohlc_timestamps[idx]
            window_start = ohlc_timestamps[idx - 1] if idx > 0 else float("-inf")
            return sum(v for ts, v in volumes if window_start < ts <= window_end)

        candles = []
        for idx, row in enumerate(ohlc):
            ts, o, h, l, c = row
            candles.append({
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": _summed_volume(idx),
            })
        return candles
