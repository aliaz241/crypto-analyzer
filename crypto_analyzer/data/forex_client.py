"""
data/forex_client.py
Forex data client from the free historical Dukascopy feed (no API key).

Data source and format (verified from this environment, 2026-11-07):
- One tick file per UTC hour:
  {FOREX_BASE_URL}/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
  where MM in the URL is 0-indexed (January = 00).
- The file is LZMA; each tick is 20 bytes big-endian:
  int32 millisecond offset from the start of the hour, int32 ask price
  (integer × point), int32 bid price, float32 ask volume, float32 bid volume.
- Each symbol's point is in config.FOREX_POINT (most pairs 1e5, JPY and gold 1e3).

Honest feed constraints (also in FOREX_FA.md):
- The feed sometimes returns 503 / connection drops / timeouts (it is
  sensitive to the request rate); fetching is done with retries and backoff
  and everything is cached on disk so the fetch happens once.
- The data is not a "live price": the last complete available hour is the
  basis of the price (variable delay, from a few minutes to more) — paper
  prices are on this same basis and are labeled "delayed".
- Volume: the sum of bid+ask tick volumes; for FX pairs there is no real
  spot-market volume, so the volume/OBV component here measures "tick
  activity", not real traded volume.

Candles are built from the bid price (the common forex chart convention).
The base candle is 1h; 4h and 1d are built by resampling on UTC boundaries
(the forex day here is the UTC day — a documented simplification). Hours
with no file (weekends and holidays) simply have no candle; no fake candle
is made.
"""

import io
import lzma
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_TICK_DTYPE = np.dtype([("ms", ">i4"), ("ask", ">i4"), ("bid", ">i4"),
                        ("ask_vol", ">f4"), ("bid_vol", ">f4")])

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "forex_cache")


class ForexError(Exception):
    """Error related to the forex feed or a symbol not present in the forex set."""
    pass


def normalize_forex_symbol(symbol: str) -> str:
    """'eur/usd' / 'EURUSD' / 'eurusd' → 'EURUSD'; only the config set."""
    s = str(symbol).strip().upper().replace("/", "").replace("-", "") \
        .replace(" ", "")
    if s not in config.FOREX_SYMBOLS:
        raise ForexError(
            f"نماد '{symbol}' در مجموعه‌ی فارکس نیست. گزینه‌ها: "
            f"{config.FOREX_SYMBOLS}")
    return s


def is_forex_symbol(symbol: str) -> bool:
    try:
        normalize_forex_symbol(symbol)
        return True
    except ForexError:
        return False


# ---------------------------------------------------------------------------
# Parsing the tick file (.bi5)
# ---------------------------------------------------------------------------
def parse_bi5_ticks(raw: bytes) -> np.ndarray:
    """Converts the compressed bytes of one hourly file into an array of ticks."""
    if not raw:
        return np.empty(0, dtype=_TICK_DTYPE)
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        # Some responses may already be decompressed (raw content); if its length
        # is a multiple of 20, we count it as raw ticks.
        if len(raw) % 20 == 0:
            data = raw
        else:
            raise ForexError("فایل تیک قابل بازگشایی نیست (LZMA).")
    if len(data) % 20 != 0:
        raise ForexError(f"طول داده‌ی تیک نامعتبر است: {len(data)}")
    return np.frombuffer(data, dtype=_TICK_DTYPE)


def ticks_to_hour_candle(ticks: np.ndarray, point: int):
    """
    Converts one hour's ticks into one 1h candle (bid price).
    Output: (open, high, low, close, volume) or None if there are no ticks.
    volume = the sum of (bid_vol + ask_vol) over the ticks (tick activity index).
    """
    if ticks is None or len(ticks) == 0:
        return None
    order = np.argsort(ticks["ms"], kind="stable")
    ticks = ticks[order]
    bid = ticks["bid"].astype(np.float64) / float(point)
    vol = (ticks["bid_vol"].astype(np.float64)
           + ticks["ask_vol"].astype(np.float64))
    return (float(bid[0]), float(bid.max()), float(bid.min()),
            float(bid[-1]), float(vol.sum()))


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------
def _hour_url(symbol: str, hour: datetime) -> str:
    return (f"{config.FOREX_BASE_URL}/{symbol}/{hour.year}/"
            f"{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5")


class ForexClient:
    def __init__(self):
        self.session = requests.Session()
        self.timeout = getattr(config, "FOREX_FETCH_TIMEOUT", 30)
        self.retries = getattr(config, "FOREX_FETCH_RETRIES", 6)
        self.pause = getattr(config, "FOREX_FETCH_PAUSE", 0.35)

    def fetch_hour_ticks(self, symbol: str, hour: datetime):
        """
        Fetches one hour's ticks. Output: a tick array or None when that
        hour has no data (404 or an empty file — e.g. a weekend).
        On transient errors (503/drop/timeout) it retries with backoff and,
        after the attempt cap, raises ForexError so the caller can decide.
        """
        url = _hour_url(symbol, hour)
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(
                    url, timeout=self.timeout,
                    headers={"User-Agent": "Mozilla/5.0"})
            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(min(5 * attempt, 30))
                continue
            if resp.status_code == 200:
                if not resp.content:
                    return None
                try:
                    return parse_bi5_ticks(resp.content)
                except ForexError:
                    # A 200 body but corrupted/incomplete (sometimes comes via a proxy):
                    # counted as a transient error so the next attempt has a chance.
                    last_error = ForexError("بدنه‌ی پاسخ 200 نامعتبر بود")
                    time.sleep(min(5 * attempt, 30))
                    continue
            if resp.status_code == 404:
                return None
            last_error = ForexError(f"HTTP {resp.status_code}")
            time.sleep(min(5 * attempt, 30))
        raise ForexError(f"واکشی {url} ناموفق بود: {last_error}")


# ---------------------------------------------------------------------------
# Disk cache (base 1h candle for each symbol)
# ---------------------------------------------------------------------------
def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_1h.pkl")


def _load_cached_1h(symbol: str):
    path = _cache_path(symbol)
    if os.path.exists(path):
        try:
            df = pd.read_pickle(path)
            if isinstance(df, pd.DataFrame) and len(df):
                return df
        except Exception:
            pass
    return None


def _save_cached_1h(symbol: str, df: pd.DataFrame):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol)
    tmp = path + ".tmp"
    df.to_pickle(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Second source: Yahoo Finance (add-on — for matrix depth)
# ---------------------------------------------------------------------------
def fetch_yahoo_candles(symbol: str, timeframe: str = "4h",
                        total_candles: int = None) -> pd.DataFrame:
    """
    Forex candles from the Yahoo Finance chart API (no key) — the matrix's
    main source because it gives real depth from this environment (verified
    2026-11-07): 1h up to 1 year (≈6200 hours) and 1d up to 5 years (≈1300 days).

    Honest constraints:
    - Volume for FX pairs is zero/None → the volume column stays zero and
      obv_trend becomes neutral (volume score = half of neutral); recorded
      in FOREX_FA.md. (For Dukascopy, volume was tick activity.)
    - Daily candles return with midnight-ET timestamps; they are normalized
      to midnight UTC of the same calendar date so they match the project's
      UTC convention (the trading day's hour does not change, only the hour
      label).
    - Ranges with no price (None) are removed; no weekend candle is made
      (except the few Sunday-night reopening hours the feed itself gives).
    """
    symbol = normalize_forex_symbol(symbol)
    if timeframe not in ("1h", "4h", "1d"):
        raise ForexError(
            f"تایم‌فریم '{timeframe}' برای فارکس پشتیبانی نمی‌شود. "
            f"گزینه‌ها: 1h, 4h, 1d")
    if total_candles is None:
        total_candles = config.FOREX_CANDLES_PER_TIMEFRAME.get(
            timeframe, 1500)
    ysym = config.FOREX_YAHOO_SYMBOLS[symbol]
    base = config.FOREX_YAHOO_URL
    timeout = getattr(config, "FOREX_YAHOO_TIMEOUT", 25)
    headers = {"User-Agent": "Mozilla/5.0"}

    def _fetch(interval: str, rng: str) -> pd.DataFrame:
        url = f"{base}/{ysym}?range={rng}&interval={interval}"
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=timeout, headers=headers)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code != 200:
                last_error = ForexError(f"Yahoo HTTP {resp.status_code}")
                time.sleep(2 * (attempt + 1))
                continue
            try:
                result = resp.json()["chart"]["result"][0]
                ts = result["timestamp"]
                quote = result["indicators"]["quote"][0]
            except Exception as exc:
                last_error = ForexError(f"پاسخ Yahoo نامعتبر: {exc}")
                time.sleep(2 * (attempt + 1))
                continue
            df = pd.DataFrame({
                "open": quote.get("open"), "high": quote.get("high"),
                "low": quote.get("low"), "close": quote.get("close"),
                "volume": quote.get("volume")}, index=pd.to_datetime(
                ts, unit="s", utc=True))
            df = df.dropna(subset=["close"])
            df["volume"] = df["volume"].fillna(0.0)
            df = df.astype(float)
            df.index.name = "datetime"
            return df[~df.index.duplicated(keep="last")].sort_index()
        raise ForexError(f"واکشی Yahoo برای {symbol} ناموفق بود: {last_error}")

    if timeframe == "1d":
        df = _fetch("1d", "5y")
        # Yahoo's raw label for the daily candle is midnight ET; over the 12 days
        # shared with Dukascopy bid candles (2026-11-07) the mean absolute
        # OHLC difference for this same label was 0.0011 — less than the ±1-day
        # shift (0.0027 and 0.0035) — so the label = the calendar date of that same
        # label in UTC. The remaining difference is of the kind of trading-day
        # convention difference (17:00 New York close vs the UTC day) and a
        # different feed, not a one-day error; recorded in FOREX_FA.md.
        df.index = pd.DatetimeIndex(df.index.date).tz_localize("UTC")
        df.index.name = "datetime"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # Yahoo's "Sunday" candles are artificial: open=high=low=close equal to
        # Friday's close (the market is closed and no price traded; in the hourly
        # series too, only Sunday's 23:00 reopening hour is real). Removing them means
        # there is no weekend candle in the daily frame; the Mon–Fri sequence stays
        # clean and the engine evaluates on real days.
        df = df[df.index.weekday < 5]
        return df.iloc[-int(total_candles):]
    df1h = _fetch("1h", "1y")
    if timeframe == "1h":
        return df1h.iloc[-int(total_candles):]
    return _resample(df1h, "4h").iloc[-int(total_candles):]


# ---------------------------------------------------------------------------
# Building the candles
# ---------------------------------------------------------------------------
def _resample(df1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1h":
        return df1h
    rule = {"4h": "4h", "1d": "1D"}[timeframe]
    out = df1h.resample(rule).agg({"open": "first", "high": "max",
                                   "low": "min", "close": "last",
                                   "volume": "sum"})
    return out.dropna()


def build_candles(symbol: str, timeframe: str = "4h",
                  total_candles: int = None,
                  client: ForexClient = None,
                  progress: bool = False) -> pd.DataFrame:
    """
    Returns the forex candles: exactly the same output shape as
    analyzer.fetch_candles_binance (a UTC datetime index and the
    open/high/low/close/volume columns).
    First the disk cache is read; if the cache depth is not enough for
    total_candles, older hours are fetched and added to the cache (up to
    FOREX_MAX_FETCH_HOURS attempts per call). Whatever was actually obtained
    is returned — a depth below the target is not faked.
    """
    symbol = normalize_forex_symbol(symbol)
    if timeframe not in ("1h", "4h", "1d"):
        raise ForexError(
            f"تایم‌فریم '{timeframe}' برای فارکس پشتیبانی نمی‌شود. "
            f"گزینه‌ها: 1h, 4h, 1d")
    if total_candles is None:
        total_candles = config.FOREX_CANDLES_PER_TIMEFRAME.get(
            timeframe, 1500)
    hours_needed = int(total_candles) * {"1h": 1, "4h": 4, "1d": 24}[timeframe]

    df1h = _load_cached_1h(symbol)
    if df1h is None or len(df1h) < hours_needed:
        if client is None:
            client = ForexClient()
        point = config.FOREX_POINT[symbol]
        have = set()
        if df1h is not None:
            have = set(df1h.index)
        # We go back from the last complete UTC hour and fill the missing hours;
        # hours already in the cache are not fetched again.
        # The end of the window is fixed (config.FOREX_WINDOW_END — the last
        # complete day for which the feed had verified data) so that building
        # candles does not change as "today" passes with time and the matrix
        # stays reproducible; latest_price, for a fresher price, is deliberately separate from this rule.
        now = datetime.fromisoformat(
            getattr(config, "FOREX_WINDOW_END", "2026-11-07T00:00:00+00:00"))
        # The fetch is done over one specified continuous range (from the end of
        # the window backwards, exactly hours_needed calendar hours) so that scattered,
        # far-apart hours do not create a fake "deep" frame; empty hours
        # (weekends) are natural within this range. Hours outside the range that
        # are already in the cache are kept but are not the basis of depth.
        window_start = now - timedelta(hours=hours_needed)
        in_window = {h for h in have if window_start <= h < now}
        cursor = now - timedelta(hours=1)
        rows = {}
        fetched = 0
        attempts = 0
        max_attempts = int(getattr(config, "FOREX_MAX_FETCH_HOURS", 30000))
        while len(in_window) + len(rows) < hours_needed and attempts < max_attempts:
            hour = cursor - timedelta(hours=attempts)
            attempts += 1
            if hour in have:
                continue
            try:
                ticks = client.fetch_hour_ticks(symbol, hour)
            except ForexError:
                # The feed did not answer for this hour even after the attempts; we skip it
                # so the whole build does not stall for one hour (the gap is recorded).
                continue
            fetched += 1
            candle = ticks_to_hour_candle(ticks, point) if ticks is not None else None
            if candle is not None:
                rows[hour] = candle
            if progress and fetched % 200 == 0:
                print(f"  {symbol}: {len(in_window) + len(rows)} ساعت کندل "
                      f"(تلاش {attempts})", flush=True)
            time.sleep(client.pause)
        if rows:
            new = pd.DataFrame.from_dict(
                rows, orient="index",
                columns=["open", "high", "low", "close", "volume"])
            new.index = pd.DatetimeIndex(new.index)
            if df1h is not None:
                df1h = pd.concat([df1h, new])
            else:
                df1h = new
            df1h = df1h[~df1h.index.duplicated(keep="last")].sort_index()
            df1h.index.name = "datetime"
            _save_cached_1h(symbol, df1h)
    if df1h is None or len(df1h) == 0:
        raise ForexError(f"برای نماد '{symbol}' هیچ کندلی از فید برنگشت.")
    df = _resample(df1h, timeframe)
    return df.iloc[-int(total_candles):]


def get_candles(symbol: str, timeframe: str = "4h",
                total_candles: int = None, progress: bool = False):
    """
    Same-named API as the project's other clients: a forex candle DataFrame.

    Main source = Yahoo (config.FOREX_PRIMARY) because it gives real depth
    from this environment; if Yahoo errors, Dukascopy (build_candles,
    cache-centric) is tried as a backup, and if that also fails, Yahoo's
    error surfaces.
    """
    primary = getattr(config, "FOREX_PRIMARY", "yahoo")
    if primary == "dukascopy":
        return build_candles(symbol, timeframe, total_candles,
                             progress=progress)
    try:
        return fetch_yahoo_candles(symbol, timeframe, total_candles)
    except ForexError:
        try:
            return build_candles(symbol, timeframe, total_candles,
                                 progress=progress)
        except ForexError:
            raise


def latest_price(symbol: str):
    """
    Forex "current price" for paper: the close of the last complete 1h candle
    from the feed (delayed — FOREX_FA.md). If the cache is empty it does a
    minimal fetch. Output: (price, candle time) — the time is needed for the
    delay label.

    Light refresh: if the cache's last candle is more than ~2 hours behind
    "now", first only the last few hours (outside the matrix's fixed window)
    are fetched and added to the cache; on a feed error, the same cached
    price is returned.
    """
    symbol = normalize_forex_symbol(symbol)
    # Main price path: Yahoo (reliable from this environment); error → the cache path below.
    try:
        dfy = fetch_yahoo_candles(symbol, "1h", 48)
        if len(dfy):
            return float(dfy["close"].iloc[-1]), dfy.index[-1]
    except ForexError:
        pass
    df1h = _load_cached_1h(symbol)
    if df1h is None or len(df1h) == 0:
        df1h = build_candles(symbol, "1h", 48)
    try:
        now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                 microsecond=0)
        if df1h.index[-1] < now - timedelta(hours=2):
            client = ForexClient()
            point = config.FOREX_POINT[symbol]
            rows = {}
            for back in range(1, 7):
                hour = now - timedelta(hours=back)
                if hour in df1h.index:
                    continue
                try:
                    ticks = client.fetch_hour_ticks(symbol, hour)
                except ForexError:
                    continue
                candle = (ticks_to_hour_candle(ticks, point)
                          if ticks is not None else None)
                if candle is not None:
                    rows[hour] = candle
            if rows:
                import pandas as _pd
                new = _pd.DataFrame.from_dict(
                    rows, orient="index",
                    columns=["open", "high", "low", "close", "volume"])
                new.index = _pd.DatetimeIndex(new.index)
                df1h = _pd.concat([df1h, new])
                df1h = df1h[~df1h.index.duplicated(keep="last")].sort_index()
                _save_cached_1h(symbol, df1h)
    except Exception:
        pass
    last_ts = df1h.index[-1]
    return float(df1h["close"].iloc[-1]), last_ts
