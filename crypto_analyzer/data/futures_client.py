"""
data/futures_client.py
Derivatives (futures) market context from Binance's public endpoints — add-on
module (step 5). For display/information only; it takes part in no scoring.

Data:
- Current funding rate (premiumIndex: lastFundingRate)
- Current open interest and its change versus 24 hours ago
  (openInterestHist with period=1h and limit=25)

⚠️ The fapi endpoints are blocked from some IPs/regions (451/403); the main
function get_derivatives_context returns None on "any" error so that its
absence breaks no other flow. There is fallback across domains:
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
    """Light futures client; like BinanceClient, with no API key and limited retry."""

    def __init__(self):
        self.base_urls = list(FAPI_BASE_URLS)
        self.timeout = getattr(config, "BINANCE_REQUEST_TIMEOUT_SECONDS", 15)
        self.session = requests.Session()

    def _get_json(self, path: str, params: dict):
        """The first 200 response among the domains; if none works, an exception."""
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
        """The symbol's current funding rate (lastFundingRate as a fraction, e.g. 0.0001)."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/fapi/v1/premiumIndex", {"symbol": symbol})

    def get_open_interest_hist(self, symbol_usdt: str, period: str = "1h",
                               limit: int = 25) -> list:
        """Open interest history; each member has sumOpenInterest."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/futures/data/openInterestHist",
                              {"symbol": symbol, "period": period, "limit": limit})

    def get_ticker_24h(self, symbol_usdt: str) -> dict:
        """24-hour statistics; priceChangePercent is the 24-hour price change percent."""
        symbol = BinanceClient.normalize_symbol(symbol_usdt)
        return self._get_json("/fapi/v1/ticker/24hr", {"symbol": symbol})


# ---------------------------------------------------------------------------
# Derivatives interpretation layer (add-on — ported from an external pack; pure functions, no network)
# ---------------------------------------------------------------------------
FUNDING_EXTREME_THRESHOLD_PCT = 0.05   # Funding every 8 hours (%); above this means crowding


def interpret_funding_rate(funding_rate_pct: float, threshold_pct: float = None) -> dict:
    """
    Interpretation of the funding rate (input in "percent", e.g. 0.05 means 0.05% per 8 hours):
    - High positive: longs are crowded and pay the shorts — the risk of a
      cascading liquidation of longs in a correction rises (cautionary/contrarian,
      not definitively bearish).
    - Low negative: shorts are crowded — the risk of a short squeeze rises.
    Output: {"label", "fa", "note"}
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
    Combining price change and open-interest change over a window — measuring the "health" of the move:
      price ↑ + OI ↑ -> new money has entered; a stronger/more valid trend
      price ↑ + OI ↓ -> short covering (shorts closing); less durable
      price ↓ + OI ↑ -> new short positions; a stronger bearish trend
      price ↓ + OI ↓ -> longs exiting/liquidating; less durable
    Output: {"label", "fa"}
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
    Summary of the derivatives context for display:
    {"funding_rate_pct": percent of the current funding rate,
     "funding_bias_fa": short Persian interpretation,
     "funding_interpretation": output of interpret_funding_rate (add-on),
     "open_interest": current open interest (in the currency's own unit),
     "oi_change_24h_pct": percent change of open interest in the last 24 hours or None,
     "price_change_24h_pct": percent change of price in the last 24 hours or None (add-on),
     "oi_interpretation": output of interpret_oi_price_relationship or None (add-on)}
    On any error (network, regional block, invalid symbol…) returns None.
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
            pass  # Open interest is optional; funding alone is also useful

        # 24-hour price change (optional) — for interpreting the price×open-interest relationship
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
