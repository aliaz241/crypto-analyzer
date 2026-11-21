"""
data/binance_client.py
کلاینت گرفتن داده‌ی کندل از اندپوینت عمومی بازار بایننس (Binance Public Market Data).

نکته‌های مهم:
- برای دیتای عمومی کندل هیچ API key لازم نیست و هیچ جا هم درخواست نمی‌شود.
- دامنه‌ی اصلی data-api.binance.vision است (مخصوص دیتای عمومی)؛ api.binance.com
  در بعضی مناطق خطای 451 می‌دهد، پس فقط fallback است.
- صفحه‌بندی (pagination) با endTime به عقب انجام می‌شود: هر صفحه حداکثر ۱۰۰۰ کندل.
- روی خطای 400 (مثل نماد ناموجود) تلاش مجدد نمی‌شود چون خطا قطعی است؛ پیام واقعی
  بایننس خوانده و به فارسی برگردانده می‌شود (رفع مشکل ممیزی شماره‌ی ۵).
"""

import time
from datetime import datetime, timezone

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BinanceError(Exception):
    """خطای مربوط به ارتباط یا پاسخ بایننس."""
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
        ورودی‌های مختلف را به نماد استاندارد بایننس تبدیل می‌کند:
        "BTC" -> "BTCUSDT" | "btc" -> "BTCUSDT" | "BTCUSDT" -> "BTCUSDT"
        "BTC/USDT" -> "BTCUSDT" | "BTC/USDT" با فاصله/حروف کوچک هم درست می‌شود.
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
        """پیام خطای واقعی بایننس را می‌خواند و خطای فارسی واضح می‌سازد (بدون retry)."""
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
        """یک صفحه کندل می‌گیرد. روی 429/5xx و خطای شبکه retry می‌کند، روی 400 نه."""
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
                    # خطای قطعی (نماد/بازه‌ی نامعتبر) — تلاش مجدد بی‌فایده است
                    self._raise_for_binance_error(response, symbol)
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = BinanceError(
                        f"HTTP {response.status_code} از {base}")
                    continue
                # بقیه‌ی کدها (مثل 451 محدودیت منطقه‌ای) — دامنه‌ی بعدی را امتحان کن
                last_error = BinanceError(f"HTTP {response.status_code} از {base}")
            if attempt < self.max_retries:
                time.sleep(attempt)  # backoff ساده: ۱ ثانیه، ۲ ثانیه
        raise BinanceError(f"اتصال به بایننس ناموفق بود: {last_error}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_candles(self, symbol_usdt: str, interval: str, total_candles: int) -> list:
        """
        تعداد total_candles کندل آخر نماد را با صفحه‌بندی به عقب جمع می‌کند.
        خروجی: لیست صعودیِ (قدیمی به جدید) دیکشنری‌های
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
            collected = page + collected  # صفحات قدیمی‌تر به ابتدای لیست می‌آیند
            end_time = int(page[0][0]) - 1  # یک میلی‌ثانیه قبل از قدیمی‌ترین کندل این صفحه
            if len(page) < limit:
                break  # به ابتدای تاریخچه‌ی نماد رسیدیم

        # مرتب‌سازی و حذف تکراری‌های احتمالی بر اساس timestamp، نگه داشتن آخرین‌ها
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
