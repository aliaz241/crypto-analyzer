"""
data/futures_client.py
زمینه‌ی بازار مشتقه (فیوچرز) از اندپوینت‌های عمومی بایننس — ماژول افزودنی
(گام ۵). فقط برای نمایش/اطلاع است و در هیچ امتیازدهی‌ای شرکت نمی‌کند.

داده‌ها:
- نرخ فاندینگ فعلی (premiumIndex: lastFundingRate)
- بهره‌ی باز (open interest) فعلی و تغییرش نسبت به ۲۴ ساعت قبل
  (openInterestHist با period=1h و limit=25)

⚠️ اندپوینت‌های fapi از بعضی IP/منطقه‌ها مسدودند (451/403)؛ تابع اصلی
get_derivatives_context روی «هر» خطایی None برمی‌گرداند تا نبودنش هیچ
جریان دیگری را خراب نکند. بین دامنه‌ها fallback هست:
fapi.binance.com ← fapi1..4.binance.com
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import config
from data.binance_client import BinanceClient

FAPI_BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]


class FuturesClient:
    """کلاینت سبک فیوچرز؛ مثل BinanceClient بدون API key و با retry محدود."""

    def __init__(self):
        self.base_urls = list(FAPI_BASE_URLS)
        self.timeout = getattr(config, "BINANCE_REQUEST_TIMEOUT_SECONDS", 15)
        self.session = requests.Session()

    def _get_json(self, path: str, params: dict):
        """اولین پاسخ 200 از بین دامنه‌ها؛ اگر هیچ‌کدام نشد، استثنا."""
        last_error = None
        for base in self.base_urls:
            try:
                resp = self.session.get(f"{base}{path}", params=params,
                                        timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                continue
            if resp.status_code == 200:
                return resp.json()
            last_error = RuntimeError(f"HTTP {resp.status_code} از {base}")
        raise RuntimeError(f"دسترسی به فیوچرز بایننس ناموفق بود: {last_error}")

    def get_premium_index(self, symbol_usdt: str) -> dict:
        """نرخ فاندینگ فعلی نماد (lastFundingRate به صورت کسر، مثلاً 0.0001)."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/fapi/v1/premiumIndex", {"symbol": symbol})

    def get_open_interest_hist(self, symbol_usdt: str, period: str = "1h",
                               limit: int = 25) -> list:
        """تاریخچه‌ی بهره‌ی باز؛ هر عضو sumOpenInterest دارد."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/futures/data/openInterestHist",
                              {"symbol": symbol, "period": period, "limit": limit})

    def get_ticker_24h(self, symbol_usdt: str) -> dict:
        """آمار ۲۴ ساعته؛ priceChangePercent درصد تغییر قیمت ۲۴ ساعت است."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/fapi/v1/ticker/24hr", {"symbol": symbol})


# ---------------------------------------------------------------------------
# لایه‌ی تفسیر مشتقه (افزودنی — پورت از پک خارجی؛ توابع خالص، بدون شبکه)
# ---------------------------------------------------------------------------
FUNDING_EXTREME_THRESHOLD_PCT = 0.05   # فاندینگ هر ۸ ساعت (%)؛ بالای این یعنی ازدحام


def interpret_funding_rate(funding_rate_pct: float, threshold_pct: float = None) -> dict:
    """
    تفسیر نرخ فاندینگ (ورودی به «درصد»، مثلاً 0.05 یعنی 0.05٪ در ۸ ساعت):
    - مثبتِ بالا: لانگ‌ها شلوغ‌اند و به شورت‌ها پول می‌دهند — ریسک تسویه‌ی
      آبشاری لانگ‌ها در اصلاح بالا می‌رود (احتیاطی/contrarian، نه نزولی قطعی).
    - منفیِ پایین: شورت‌ها شلوغ‌اند — ریسک short-squeeze بالا می‌رود.
    خروجی: {"label", "fa", "note"}
    """
    if threshold_pct is None:
        threshold_pct = FUNDING_EXTREME_THRESHOLD_PCT
    if funding_rate_pct >= threshold_pct:
        return {"label": "crowded_long", "fa": "ازدحام پوزیشن لانگ",
                "note": "فاندینگ مثبت بالا — لانگ‌ها شلوغ‌اند، ریسک تسویه‌ی آبشاری لانگ در اصلاح بالاست."}
    if funding_rate_pct <= -threshold_pct:
        return {"label": "crowded_short", "fa": "ازدحام پوزیشن شورت",
                "note": "فاندینگ منفی بالا — شورت‌ها شلوغ‌اند، ریسک short-squeeze بالاست."}
    return {"label": "neutral", "fa": "خنثی", "note": "فاندینگ در بازه‌ی عادی."}


def interpret_oi_price_relationship(price_change_pct: float, oi_change_pct: float) -> dict:
    """
    ترکیب تغییر قیمت و تغییر بهره‌ی باز در یک بازه — سنجش «سلامت» حرکت:
      قیمت ↑ + OI ↑ -> پول جدید وارد شده؛ روند قوی‌تر/معتبرتر
      قیمت ↑ + OI ↓ -> short covering (بسته‌شدن شورت‌ها)؛ کم‌دوام‌تر
      قیمت ↓ + OI ↑ -> پوزیشن شورت جدید؛ روند نزولی قوی‌تر
      قیمت ↓ + OI ↓ -> خروج/لیکویید لانگ‌ها؛ کم‌دوام‌تر
    خروجی: {"label", "fa"}
    """
    price_up = price_change_pct > 0
    oi_up = oi_change_pct > 0
    if price_up and oi_up:
        return {"label": "healthy_uptrend", "fa": "روند صعودی سالم (ورود پول جدید)"}
    if price_up and not oi_up:
        return {"label": "short_covering", "fa": "احتمال short covering (کم‌دوام‌تر)"}
    if not price_up and oi_up:
        return {"label": "healthy_downtrend", "fa": "روند نزولی سالم (پوزیشن شورت جدید)"}
    return {"label": "long_liquidation_or_exit", "fa": "احتمال خروج/لیکویید لانگ (کم‌دوام‌تر)"}


def get_derivatives_context(symbol_usdt: str):
    """
    خلاصه‌ی زمینه‌ی مشتقه برای نمایش:
    {"funding_rate_pct": درصد نرخ فاندینگ فعلی,
     "funding_bias_fa": تفسیر کوتاه فارسی,
     "funding_interpretation": خروجی interpret_funding_rate (افزودنی),
     "open_interest": بهره‌ی باز فعلی (به واحد خود ارز),
     "oi_change_24h_pct": درصد تغییر بهره‌ی باز در ۲۴ ساعت اخیر یا None,
     "price_change_24h_pct": درصد تغییر قیمت در ۲۴ ساعت اخیر یا None (افزودنی),
     "oi_interpretation": خروجی interpret_oi_price_relationship یا None (افزودنی)}
    روی هر خطایی (شبکه، مسدودی منطقه‌ای، نماد نامعتبر…) None برمی‌گرداند.
    """
    try:
        client = FuturesClient()
        premium = client.get_premium_index(symbol_usdt)
        funding = float(premium["lastFundingRate"])
        funding_pct = round(funding * 100.0, 4)
        if funding > 0.0005:
            bias = "فاندینگ مثبت بالا — لانگ‌ها شلوغ‌اند"
        elif funding < -0.0005:
            bias = "فاندینگ منفی — شورت‌ها شلوغ‌اند"
        else:
            bias = "فاندینگ خنثی"

        oi_now, oi_change = None, None
        try:
            hist = client.get_open_interest_hist(symbol_usdt)
            if hist:
                oi_now = float(hist[-1]["sumOpenInterest"])
                if len(hist) >= 2 and float(hist[0]["sumOpenInterest"]) > 0:
                    oi_change = round(
                        (oi_now / float(hist[0]["sumOpenInterest"]) - 1.0) * 100.0, 2)
        except Exception:
            pass  # بهره‌ی باز اختیاری است؛ فاندینگ به‌تنهایی هم مفید است

        # تغییر قیمت ۲۴ ساعته (اختیاری) — برای تفسیر رابطه‌ی قیمت×بهره‌ی باز
        price_change_24h = None
        try:
            t24 = client.get_ticker_24h(symbol_usdt)
            price_change_24h = round(float(t24["priceChangePercent"]), 2)
        except Exception:
            pass

        oi_interp = None
        if oi_change is not None and price_change_24h is not None:
            oi_interp = interpret_oi_price_relationship(price_change_24h, oi_change)

        return {
            "funding_rate_pct": funding_pct,
            "funding_bias_fa": bias,
            "funding_interpretation": interpret_funding_rate(funding_pct),
            "open_interest": oi_now,
            "oi_change_24h_pct": oi_change,
            "price_change_24h_pct": price_change_24h,
            "oi_interpretation": oi_interp,
        }
    except Exception:
        return None
