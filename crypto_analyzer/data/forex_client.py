"""
data/forex_client.py
کلاینت داده‌ی فارکس از فید تاریخی رایگان Dukascopy (بدون API key).

منبع و قالب داده (راستی‌آزمایی‌شده از همین محیط، 2026-11-07):
- برای هر ساعت UTC یک فایل تیک: 
  {FOREX_BASE_URL}/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
  که MM در URL صفرمبنا (0-indexed) است (ژانویه = 00).
- فایل LZMA است؛ هر تیک ۲۰ بایت big-endian:
  int32 آفست میلی‌ثانیه از شروع ساعت، int32 قیمت ask (صحیح × پوینت)،
  int32 قیمت bid، float32 حجم ask، float32 حجم bid.
- پوینت هر نماد در config.FOREX_POINT است (بیشتر جفت‌ها 1e5، JPY و طلا 1e3).

قیدهای صادقانه‌ی فید (در FOREX_FA.md هم آمده):
- فید گاهی 503 / قطع اتصال / timeout می‌دهد (به نرخ درخواست حساس است)؛
  واکشی با تلاش مجدد و backoff انجام و همه‌چیز روی دیسک کش می‌شود تا
  واکشی یک‌باره باشد.
- داده «قیمت لحظه‌ای» نیست: آخرین ساعت کاملِ در دسترس مبنای قیمت است
  (تأخیر متغیر، از چند دقیقه تا بیشتر) — قیمت‌های paper روی همین مبناست
  و «با تأخیر» برچسب می‌خورند.
- حجم: مجموع حجم تیک‌های bid+ask است؛ برای جفت‌های FX حجم واقعی بازار
  اسپات وجود ندارد، پس مؤلفه‌ی حجم/OBV اینجا «فعالیت تیک» را می‌سنجد،
  نه حجم معامله‌شده‌ی واقعی.

کندل‌ها از قیمت bid ساخته می‌شوند (قرارداد رایج چارت فارکس). کندل پایه
1h است؛ 4h و 1d با resample روی مرزهای UTC ساخته می‌شوند (روز فارکس
اینجا روز UTC است — ساده‌سازی مستندشده). ساعت‌های بدون فایل (آخر هفته و
تعطیلات) به‌سادگی کندل ندارند؛ هیچ کندل جعلی ساخته نمی‌شود.
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
    """خطای مربوط به فید فارکس یا نماد ناموجود در مجموعه‌ی فارکس."""
    pass


def normalize_forex_symbol(symbol: str) -> str:
    """'eur/usd' / 'EURUSD' / 'eurusd' → 'EURUSD'؛ فقط مجموعه‌ی config."""
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
# تجزیه‌ی فایل تیک (.bi5)
# ---------------------------------------------------------------------------
def parse_bi5_ticks(raw: bytes) -> np.ndarray:
    """بایت‌های فشرده‌ی یک فایل ساعتی را به آرایه‌ی تیک‌ها تبدیل می‌کند."""
    if not raw:
        return np.empty(0, dtype=_TICK_DTYPE)
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        # بعضی پاسخ‌ها ممکن است از قبل باز شده باشند (محتوای خام)؛ اگر طولش
        # مضربی از ۲۰ بود همان را تیک خام حساب می‌کنیم.
        if len(raw) % 20 == 0:
            data = raw
        else:
            raise ForexError("فایل تیک قابل بازگشایی نیست (LZMA).")
    if len(data) % 20 != 0:
        raise ForexError(f"طول داده‌ی تیک نامعتبر است: {len(data)}")
    return np.frombuffer(data, dtype=_TICK_DTYPE)


def ticks_to_hour_candle(ticks: np.ndarray, point: int):
    """
    تیک‌های یک ساعت را به یک کندل 1h تبدیل می‌کند (قیمت bid).
    خروجی: (open, high, low, close, volume) یا None اگر تیکی نیست.
    volume = مجموع (bid_vol + ask_vol) روی تیک‌ها (شاخص فعالیت تیک).
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
# واکشی شبکه
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
        تیک‌های یک ساعت را می‌گیرد. خروجی: آرایه‌ی تیک یا None وقتی آن
        ساعت داده‌ای ندارد (404 یا فایل خالی — مثلاً آخر هفته).
        روی خطاهای گذرا (503/قطع/timeout) با backoff تلاش مجدد می‌کند و
        بعد از سقف تلاش‌ها ForexError می‌دهد تا صدا‌زننده تصمیم بگیرد.
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
                    # بدنه‌ی 200 ولی خراب/ناقص (گاهی از پراکسی می‌آید):
                    # خطای گذرا حساب می‌شود تا تلاش بعدی شانس داشته باشد.
                    last_error = ForexError("بدنه‌ی پاسخ 200 نامعتبر بود")
                    time.sleep(min(5 * attempt, 30))
                    continue
            if resp.status_code == 404:
                return None
            last_error = ForexError(f"HTTP {resp.status_code}")
            time.sleep(min(5 * attempt, 30))
        raise ForexError(f"واکشی {url} ناموفق بود: {last_error}")


# ---------------------------------------------------------------------------
# کش دیسک (کندل پایه‌ی 1h برای هر نماد)
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
# منبع دوم: Yahoo Finance (افزودنی — برای عمق ماتریس)
# ---------------------------------------------------------------------------
def fetch_yahoo_candles(symbol: str, timeframe: str = "4h",
                        total_candles: int = None) -> pd.DataFrame:
    """
    کندل‌های فارکس از Yahoo Finance chart API (بدون key) — منبع اصلی
    ماتریس چون از این محیط عمق واقعی می‌دهد (راستی‌آزمایی 2026-11-07):
    1h تا ۱ سال (≈۶۲۰۰ ساعت) و 1d تا ۵ سال (≈۱۳۰۰ روز).

    قیدهای صادقانه:
    - حجم برای جفت‌های FX صفر/None است → ستون volume صفر می‌ماند و
      obv_trend خنثی می‌شود (امتیاز حجم = نصفِ خنثی)؛ در FOREX_FA.md
      ثبت است. (برای Dukascopy حجم = فعالیت تیک بود.)
    - کندل‌های روزانه با تایم‌استمپ نیمه‌شب ET برمی‌گردند؛ به نیمه‌شب
      UTC همان تاریخ تقویمی نرمال می‌شوند تا با قرارداد UTC پروژه
      یکی شوند (ساعتِ روز معاملاتی عوض نمی‌شود، فقط برچسب ساعت).
    - بازه‌های بدون قیمت (None) حذف می‌شوند؛ آخر هفته کندلی ساخته
      نمی‌شود (به‌جز چند ساعت بازگشایی یکشنبه شب که خود فید می‌دهد).
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
        # برچسب خام Yahoo برای کندل روزانه نیمه‌شب ET است؛ روی ۱۲ روز
        # مشترک با کندل‌های bid دوکاسکوپی (2026-11-07) میانگین اختلاف
        # قدرمطلق OHLC برای همین برچسب ۰٫۰۰۱۱ بود — کمتر از شیفت ±۱
        # روز (۰٫۰۰۲۷ و ۰٫۰۰۳۵) — پس برچسب = تاریخ تقویمیِ همان برچسب
        # در UTC. اختلاف باقی‌مانده از جنس تفاوت قرارداد روز معاملاتی
        # (بسته‌شدن ۱۷:۰۰ نیویورک در برابر روز UTC) و فید متفاوت است،
        # نه خطای یک‌روزه؛ در FOREX_FA.md ثبت است.
        df.index = pd.DatetimeIndex(df.index.date).tz_localize("UTC")
        df.index.name = "datetime"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # کندل‌های «یکشنبه» یاهو مصنوعی‌اند: open=high=low=close برابر
        # close جمعه (بازار بسته است و قیمتی معامله نشده؛ در سری ساعتی
        # هم فقط ساعت بازگشایی ۲۳:۰۰ یکشنبه واقعی است). حذفشان یعنی
        # هیچ کندل آخر هفته‌ای در قاب روزانه نیست؛ توالی Mon–Fri تمیز
        # می‌ماند و انجین روی روزهای واقعی ارزیابی می‌کند.
        df = df[df.index.weekday < 5]
        return df.iloc[-int(total_candles):]
    df1h = _fetch("1h", "1y")
    if timeframe == "1h":
        return df1h.iloc[-int(total_candles):]
    return _resample(df1h, "4h").iloc[-int(total_candles):]


# ---------------------------------------------------------------------------
# ساخت کندل‌ها
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
    کندل‌های فارکس را برمی‌گرداند: دقیقاً همان شکل خروجی
    analyzer.fetch_candles_binance (ایندکس datetime ‏UTC و ستون‌های
    open/high/low/close/volume).
    اول کش دیسک خوانده می‌شود؛ اگر عمق کش برای total_candles کافی نبود،
    ساعت‌های قدیمی‌تر واکشی و به کش اضافه می‌شوند (تا سقف
    FOREX_MAX_FETCH_HOURS تلاش در هر صدا). هرچه واقعاً به دست آمد همان
    برمی‌گردد — عمق کمتر از هدف جعل نمی‌شود.
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
        # از آخرین ساعت کاملِ UTC به عقب می‌رویم و ساعت‌های گمشده را پر
        # می‌کنیم؛ ساعت‌های موجود در کش دوباره گرفته نمی‌شوند.
        # انتهای پنجره ثابت است (config.FOREX_WINDOW_END — آخرین روز
        # کاملی که فید برایش داده‌ی تأییدشده داشت) تا ساختن کندل‌ها با
        # گذر زمان «امروز» عوض نشود و ماتریس reproducible بماند؛
        # latest_price برای قیمت تازه‌تر عمداً از این قاعده جداست.
        now = datetime.fromisoformat(
            getattr(config, "FOREX_WINDOW_END", "2026-11-07T00:00:00+00:00"))
        # واکشی روی یک بازه‌ی پیوسته‌ی مشخص انجام می‌شود (از انتهای
        # پنجره به عقب، دقیقاً hours_needed ساعت تقویمی) تا ساعت‌های
        # پراکنده و دور از هم یک قاب «عمیق» جعلی نسازند؛ ساعت‌های خالی
        # (آخر هفته) در همین بازه طبیعی‌اند. ساعت‌های خارج از بازه که
        # از قبل در کش‌اند نگه داشته می‌شوند ولی مبنای عمق نیستند.
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
                # فید برای این ساعت هم بعد از تلاش‌ها جواب نداد؛ رد می‌شویم
                # تا کل ساخت به‌خاطر یک ساعت نایستد (شکاف ثبت می‌شود).
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
    API همنامِ کلاینت‌های دیگر پروژه: DataFrame کندل فارکس.

    منبع اصلی = Yahoo (config.FOREX_PRIMARY) چون از این محیط عمق واقعی
    می‌دهد؛ اگر Yahoo خطا داد، Dukascopy (build_candles، کش‌محور) به‌
    عنوان پشتیبان تلاش می‌شود و اگر آن هم نشد خطای Yahoo بالا می‌آید.
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
    «قیمت فعلی» فارکس برای paper: بسته‌شدن آخرین کندل 1h کامل از فید
    (با تأخیر — FOREX_FA.md). اگر کش خالی باشد یک واکشی حداقلی می‌کند.
    خروجی: (قیمت, زمان کندل) — زمان برای برچسب تأخیر لازم است.

    تازه‌سازی سبک: اگر آخرین کندل کش بیش از ~۲ ساعت از «اکنون» عقب
    باشد، اول فقط چند ساعت اخیر (خارج از پنجره‌ی ثابت ماتریس) واکشی و
    به کش اضافه می‌شود؛ روی خطای فید، همان قیمت کش برمی‌گردد.
    """
    symbol = normalize_forex_symbol(symbol)
    # مسیر اصلی قیمت: Yahoo (قابل اتکا از این محیط)؛ خطا → مسیر کش زیر.
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
