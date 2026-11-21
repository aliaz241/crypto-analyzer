"""
data/coingecko_client.py
Client for fetching price/candle data from CoinGecko.

Important note: on the free plan, CoinGecko gives raw OHLC only for limited
ranges (1/7/14/30/90/180/365 days) and its output is [timestamp, open, high,
low, close] — without trading volume. For volume we use the market_chart
endpoint and merge it with the candles by timestamp.
"""

import time
from datetime import datetime, timezone

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CoinGeckoError(Exception):
    """Error related to the connection or CoinGecko's response."""
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
        """Avoiding hitting the request rate limit."""
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
                    # Rate limit — wait a bit more and try again
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
        Converts a currency symbol or name (e.g. 'btc' or 'bitcoin') to the coin_id
        that CoinGecko requires for its other endpoints.
        """
        data = self._get("/search", {"query": query})
        coins = data.get("coins", [])
        if not coins:
            raise CoinGeckoError(f"ارزی با نام/نماد '{query}' پیدا نشد.")
        # The first result is usually the most valid/highest-volume match
        return coins[0]["id"]

    def get_ohlc(self, coin_id: str, vs_currency: str, days: int) -> list:
        """
        Output: a list of [timestamp_ms, open, high, low, close]
        """
        params = {"vs_currency": vs_currency, "days": days}
        data = self._get(f"/coins/{coin_id}/ohlc", params)
        if not data:
            raise CoinGeckoError("داده‌ی کندلی خالی برگشت — نماد یا بازه را بررسی کن.")
        return data

    def get_market_chart(self, coin_id: str, vs_currency: str, days: int) -> dict:
        """
        The output includes prices, market_caps, total_volumes — each a list
        of [timestamp_ms, value]. This is used to extract trading volume.
        """
        params = {"vs_currency": vs_currency, "days": days}
        return self._get(f"/coins/{coin_id}/market_chart", params)

    def get_candles_with_volume(self, coin_id: str, vs_currency: str, days: int) -> list:
        """
        Combines the OHLC candles with trading volume (from market_chart).

        Note: market_chart granularity is usually finer than OHLC (e.g. hourly
        volume against a 4-hour candle), so instead of taking the "nearest" point,
        we sum all volume points that fall within each candle's time range (from
        the end of the previous candle to the end of the current candle) — more
        accurate than single-point matching.
        Output: a list of dicts {timestamp, open, high, low, close, volume}
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
