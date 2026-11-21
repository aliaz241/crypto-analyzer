"""
data/binance_client.py
Client for fetching candle data from Binance's public market endpoint (Binance Public Market Data).

Important points:
- No API key is needed for public candle data, and none is requested anywhere.
- The main domain is data-api.binance.vision (dedicated to public data); api.binance.com
  returns error 451 in some regions, so it is only a fallback.
- Pagination is done backwards with endTime: each page at most 1000 candles.
- On error 400 (e.g. a nonexistent symbol) there is no retry because the error is
  definitive; Binance's real message is read and returned in Persian (fix for audit
  issue no. 5).
"""

import time
from datetime import datetime, timezone

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BinanceError(Exception):
    """Error related to the connection or Binance's response."""
    pass


class BinanceClient:
    def __init__(self):
        self.base_urls = list(getattr(config, "BINANCE_BASE_URLS",
                                      ["https://data-api.binance.vision"]))
        self.timeout = getattr(config, "BINANCE_REQUEST_TIMEOUT_SECONDS", 15)
        self.max_retries = getattr(config, "BINANCE_MAX_RETRIES", 3)
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Symbol normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_symbol(symbol_usdt: str) -> str:
        """
        Converts various inputs to the standard Binance symbol:
        "BTC" -> "BTCUSDT" | "btc" -> "BTCUSDT" | "BTCUSDT" -> "BTCUSDT"
        "BTC/USDT" -> "BTCUSDT" | "BTC/USDT" with spaces/lowercase also works.
        """
        s = str(symbol_usdt).strip().upper().replace("/", "").replace("-", "").replace(" ", "")
        if not s:
            raise BinanceError("نماد خالی است.")
        if not s.endswith("USDT"):
            s = s + "USDT"
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _raise_for_binance_error(self, response, symbol: str):
        """Reads Binance's real error message and builds a clear Persian error (without retry)."""
        code = None
        msg = ""
        try:
            payload = response.json()
            code = payload.get("code")
            msg = payload.get("msg", "")
        except ValueError:
            msg = response.text[:200]
        if code == -1121:
            raise BinanceError(
                f"نماد '{symbol}' در بایننس یافت نشد (Invalid symbol). "
                f"نماد را بررسی کن — این ابزار فقط جفت‌های USDT را پشتیبانی می‌کند."
            )
        if code == -1120:
            raise BinanceError(f"تایم‌فریم نامعتبر است. پیام بایننس: {msg}")
        raise BinanceError(f"خطای بایننس (HTTP {response.status_code}): {msg or code}")

    def _get_klines_page(self, symbol: str, interval: str, limit: int,
                         end_time: int = None) -> list:
        """Fetches one page of candles. Retries on 429/5xx and network errors, not on 400."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            for base in self.base_urls:
                try:
                    response = self.session.get(
                        f"{base}/api/v3/klines", params=params, timeout=self.timeout
                    )
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    continue
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 400:
                    # Definitive error (invalid symbol/interval) — retrying is useless
                    self._raise_for_binance_error(response, symbol)
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = BinanceError(
                        f"HTTP {response.status_code} از {base}")
                    continue
                # Other codes (e.g. 451 regional restriction) — try the next domain
                last_error = BinanceError(f"HTTP {response.status_code} از {base}")
            if attempt < self.max_retries:
                time.sleep(attempt)  # Simple backoff: 1 second, 2 seconds
        raise BinanceError(f"اتصال به بایننس ناموفق بود: {last_error}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_candles(self, symbol_usdt: str, interval: str, total_candles: int) -> list:
        """
        Collects a total of total_candles of the symbol's last candles, paging backwards.
        Output: an ascending list (old to new) of dicts
        {timestamp(ms), datetime(tz UTC), open, high, low, close, volume}
        """
        symbol = self.normalize_symbol(symbol_usdt)
        page_limit = getattr(config, "BINANCE_PAGE_LIMIT", 1000)

        collected = []
        end_time = None
        while len(collected) < total_candles:
            limit = min(page_limit, total_candles - len(collected))
            page = self._get_klines_page(symbol, interval, limit, end_time)
            if not page:
                break
            collected = page + collected  # Older pages come to the beginning of the list
            end_time = int(page[0][0]) - 1  # One millisecond before the oldest candle of this page
            if len(page) < limit:
                break  # We have reached the beginning of the symbol's history

        # Sorting and removing possible duplicates by timestamp, keeping the last ones
        by_ts = {int(row[0]): row for row in collected}
        rows = [by_ts[ts] for ts in sorted(by_ts)][-total_candles:]

        if not rows:
            raise BinanceError(f"برای نماد '{symbol}' هیچ کندلی برنگشت.")

        candles = []
        for row in rows:
            ts = int(row[0])
            candles.append({
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return candles
