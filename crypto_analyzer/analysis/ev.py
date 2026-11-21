"""
analysis/ev.py — لایه‌ی تصمیم بر پایه‌ی امید ریاضی (EV Decision Layer)

کاملاً افزودنی است: هیچ محاسبه‌ی موجودی (امتیاز اسکنر، بک‌تست، الگوها)
تغییر نمی‌کند. این ماژول فقط «مصرف‌کننده»‌ی خروجی بک‌تست سیستمی است و
برای هر ترکیب ارز/تایم‌فریم یک EV خالص می‌سازد و از روی آن + امتیاز فعلی
اسکنر، وضعیت تصمیم (decision) تعیین می‌کند.

تعریف EV در این سیستم
---------------------
EV هر ترکیب = امید ریاضی (میانگین R) معاملات «خارج‌ازنمونه»
(out-of-sample) بک‌تست سیستمیِ همان ترکیب، در آستانه‌ی امتیاز
config.EV_BACKTEST_THRESHOLD (پیش‌فرض ۶۰). R ها در خود بک‌تست سیستمی
بعد از کسر کارمزد رفت‌وبرگشت و اسلیپیج دو طرف حساب شده‌اند، پس این EV
«خالص» است و هزینه‌ی جداگانه‌ای نباید از آن کم شود.

⚠️ فقط out-of-sample: آمار داخل‌نمونه (in-sample) همان داده‌ای است که
قواعد روی آن شکل گرفته‌اند و برای تصمیم زنده بیش‌برازش است؛ این ماژول
عمداً هرگز in-sample را نمی‌خواند — نه در مسیر فایل‌های ذخیره‌شده و نه
در مسیر محاسبه‌ی تازه.

⚠️ محدودیت جهت (مهم و صادقانه): معاملات بک‌تست را خود پایپ‌لاین تولید
کرده و جهت هر معامله را همان موقع انتخاب کرده است؛ پس آمار EV «عملکرد
کلی پایپ‌لاین روی این ترکیب» است، نه آمار جدا برای لانگ و شورت. ما آمار
به‌تفکیک جهت نداریم و وانمود نمی‌کنیم که داریم؛ جهت فعلی نامزد فقط از
اسکنر می‌آید و EV به هر دو جهت یکسان اعمال می‌شود.

منابع داده (به ترتیب اولویت در ev_for_candidate):
۱) کش حافظه، ۲) کش دیسک (ev_cache.json کنار پکیج)،
۳) فایل‌های نتیجه‌ی بک‌تست ذخیره‌شده (پوشه‌ی matrix/ کنار پکیج یا
   btc4h_backtest_result.json) — از لیست معاملاتشان فقط segment="out"
   خوانده می‌شود و همه‌ی معیارها (نرخ برد، میانگین برد/باخت، پروفیت
   فاکتور، نرخ برد سربه‌سر) از همان R های واقعی دوباره حساب می‌شوند،
۴) فقط وقتی allow_compute=True و هیچ‌کدام نبود: اجرای تازه‌ی
   run_system_backtest و ذخیره در کش. مسیر اسکن (scan_market) همیشه با
   allow_compute=False صدا می‌زند تا اسکن ۴۰ ترکیبی به ۴۰ بک‌تست تبدیل
   نشود؛ بازنشانی کش صریح است: ev_for_candidate(..., refresh=True) یا
   refresh_ev_cache(...) یا پرچم CLI ‏--refresh-ev در main.py.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(_PACKAGE_DIR, config.EV_CACHE_FILENAME)

# کش حافظه: (SYMBOL, timeframe) -> ورودی کش (شامل r_values_oos)
_MEM_CACHE = {}

_RELIABILITY_FA = {
    "insufficient": "نمونه ناکافی",
    "low": "اطمینان کم (نمونه کم)",
    "ok": "نمونه قابل اتکا",
}

DECISION_FA = {
    "STRONG_BUY": "خرید قوی",
    "STRONG_SELL": "فروش قوی",
    "BUY": "خرید",
    "SELL": "فروش",
    "WAIT": "انتظار — فعلاً ورود نکن",
    "NO_TRADE": "بدون معامله (NO TRADE)",
}

ACTIONABLE_DECISIONS = ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL")


# ---------------------------------------------------------------------------
# محاسبه‌ی EV از آمار/معاملات
# ---------------------------------------------------------------------------
def _reliability(n: int) -> str:
    if n < config.EV_SAMPLE_INSUFFICIENT_MAX:
        return "insufficient"
    if n < config.EV_SAMPLE_OK_MIN:
        return "low"
    return "ok"


def _extract_r_values(stats) -> list:
    """از ورودی‌های پشتیبانی‌شده لیست R استخراج می‌کند (بدون فیلتر segment؛
    فیلتر خارج‌ازنمونه وظیفه‌ی صدا‌زننده است)."""
    if stats is None:
        return []
    if isinstance(stats, (list, tuple)):
        items = list(stats)
    elif isinstance(stats, dict):
        if "r_values" in stats:
            items = list(stats["r_values"])
        elif "rs" in stats:
            items = list(stats["rs"])
        elif "trades" in stats:
            items = list(stats["trades"])
        else:
            return []
    else:
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            r = it.get("r")
        else:
            r = it
        if r is None:
            continue
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(r):
            out.append(r)
    return out


def compute_ev(stats) -> dict:
    """
    معیارهای EV را از یکی از این ورودی‌ها حساب می‌کند:
    - لیست R ها (عدد یا دیکشنری معامله با کلید "r")
    - دیکشنری با کلید "r_values" / "rs" / "trades"
    - دیکشنری تجمیعی با کلیدهای win_rate (کسر ۰..۱) و avg_win_r و
      avg_loss_r (قدرمطلق میانگین باخت) و n_trades

    خروجی:
      win_rate (کسر) + win_rate_pct، avg_win_r، avg_loss_r (قدرمطلق مثبت)،
      expectancy_r (امید ریاضی خالص، همان میانگین R)، profit_factor
      (سود ناخالص/|زیان ناخالص|؛ اگر باختی نباشد و سود باشد inf، اگر
      سودی نباشد 0.0، اگر معامله‌ای نباشد None)،
      break_even_win_rate = avg_loss / (avg_win + avg_loss) (کسر) + _pct،
      n_trades، reliability و reliability_fa.
    """
    # مسیر تجمیعی آماده
    if isinstance(stats, dict) and "avg_win_r" in stats and "avg_loss_r" in stats \
            and "win_rate" in stats and "n_trades" in stats:
        n = int(stats["n_trades"])
        wr = float(stats["win_rate"])
        aw = float(stats["avg_win_r"])
        al = float(stats["avg_loss_r"])
        expectancy = wr * aw - (1.0 - wr) * al
        gross_profit = wr * n * aw
        gross_loss = (1.0 - wr) * n * al
        if n == 0:
            pf = None
        elif gross_loss <= 0:
            pf = float("inf") if gross_profit > 0 else None
        else:
            pf = gross_profit / gross_loss
        be = (al / (aw + al)) if (aw + al) > 0 else None
        rel = _reliability(n)
        return {
            "win_rate": round(wr, 4), "win_rate_pct": round(wr * 100.0, 1),
            "avg_win_r": round(aw, 4), "avg_loss_r": round(al, 4),
            "expectancy_r": round(expectancy, 4),
            "profit_factor": (round(pf, 4) if pf is not None and math.isfinite(pf) else pf),
            "break_even_win_rate": (round(be, 4) if be is not None else None),
            "break_even_win_rate_pct": (round(be * 100.0, 1) if be is not None else None),
            "gross_profit_r": round(gross_profit, 3),
            "gross_loss_r": round(gross_loss, 3),
            "n_trades": n, "reliability": rel,
            "reliability_fa": _RELIABILITY_FA[rel],
        }

    rs = _extract_r_values(stats)
    n = len(rs)
    rel = _reliability(n)
    if n == 0:
        return {
            "win_rate": None, "win_rate_pct": None,
            "avg_win_r": None, "avg_loss_r": None,
            "expectancy_r": None, "profit_factor": None,
            "break_even_win_rate": None, "break_even_win_rate_pct": None,
            "gross_profit_r": 0.0, "gross_loss_r": 0.0,
            "n_trades": 0, "reliability": rel,
            "reliability_fa": _RELIABILITY_FA[rel],
        }
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    expectancy = sum(rs) / n
    if gross_loss <= 0:
        pf = float("inf") if gross_profit > 0 else None
    else:
        pf = gross_profit / gross_loss
    be = (avg_loss / (avg_win + avg_loss)) if (avg_win + avg_loss) > 0 else None
    return {
        "win_rate": round(win_rate, 4), "win_rate_pct": round(win_rate * 100.0, 1),
        "avg_win_r": round(avg_win, 4), "avg_loss_r": round(avg_loss, 4),
        "expectancy_r": round(expectancy, 4),
        "profit_factor": (round(pf, 4) if pf is not None and math.isfinite(pf) else pf),
        "break_even_win_rate": (round(be, 4) if be is not None else None),
        "break_even_win_rate_pct": (round(be * 100.0, 1) if be is not None else None),
        "gross_profit_r": round(gross_profit, 3),
        "gross_loss_r": round(gross_loss, 3),
        "n_trades": n, "reliability": rel,
        "reliability_fa": _RELIABILITY_FA[rel],
    }


# ---------------------------------------------------------------------------
# کش دیسک
# ---------------------------------------------------------------------------
def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.upper()}_{timeframe}"


def _load_disk_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": 1, "entries": {}}


def _save_disk_cache(data: dict):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, CACHE_FILE)


def _store_entry(symbol: str, timeframe: str, r_values: list,
                 source: str, split_at=None):
    entry = {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "threshold": config.EV_BACKTEST_THRESHOLD,
        "source": source,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "split_at": split_at,
        "r_values_oos": [round(float(r), 4) for r in r_values],
        "n_trades": len(r_values),
    }
    _MEM_CACHE[(symbol.upper(), timeframe)] = entry
    data = _load_disk_cache()
    data["entries"][_cache_key(symbol, timeframe)] = entry
    _save_disk_cache(data)
    return entry


def _entry_from_memory_or_disk(symbol: str, timeframe: str):
    key = (symbol.upper(), timeframe)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    entry = _load_disk_cache()["entries"].get(_cache_key(symbol, timeframe))
    if entry is not None:
        _MEM_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# فایل‌های نتیجه‌ی ذخیره‌شده (ماتریس / نتیجه‌ی تکی)
# ---------------------------------------------------------------------------
def _stored_result_paths(symbol: str, timeframe: str) -> list:
    sym = symbol.upper()
    parent = os.path.dirname(_PACKAGE_DIR)   # پوشه‌ی crypto_analyzer_v2/
    paths = [
        # ماتریس فارکس (افزودنی): برای نمادهای فارکس تنها منبع ذخیره‌شده
        # همین است؛ برای کریپتو این فایل‌ها وجود ندارند و اثری ندارند.
        os.path.join(parent, "matrix_forex", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix_forex", f"{sym}_{timeframe}.json"),
        # ماتریس نهاییِ برنامه‌ی بهبود (پشته‌ی نهایی) اول است: بعد از
        # برنامه، رکورد معتبر هر ترکیب همین است؛ ماتریس اصلی فقط fallback
        # تاریخی می‌ماند (اگر فایل نهایی نباشد).
        os.path.join(parent, "matrix_final", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix_final", f"{sym}_{timeframe}.json"),
        os.path.join(parent, "matrix", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix", f"{sym}_{timeframe}.json"),
    ]
    if sym == "BTC" and timeframe == "4h":
        paths.append(os.path.join(parent, "btc4h_backtest_result.json"))
        paths.append(os.path.join(_PACKAGE_DIR, "btc4h_backtest_result.json"))
    return paths


def _oos_r_values_from_result(result: dict):
    """فقط معاملات segment="out" آستانه‌ی EV — هرگز in-sample."""
    th = str(config.EV_BACKTEST_THRESHOLD)
    data = (result.get("thresholds") or {}).get(th)
    if data is None:
        # آستانه‌ی مبنا در فایل نیست؛ به آستانه‌ی دیگری دست نمی‌زنیم.
        return None
    trades = data.get("trades") or []
    return [float(t["r"]) for t in trades
            if t.get("segment") == "out" and t.get("r") is not None]


def _entry_from_stored_files(symbol: str, timeframe: str):
    for path in _stored_result_paths(symbol, timeframe):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(result.get("symbol", "")).upper() != symbol.upper() \
                or result.get("timeframe") != timeframe:
            continue
        r_values = _oos_r_values_from_result(result)
        if r_values is None:
            continue
        source = "stored_matrix" if "matrix" in path else "stored_result"
        return _store_entry(symbol, timeframe, r_values, source,
                            split_at=result.get("split_at"))
    return None


def _entry_from_fresh_backtest(symbol: str, timeframe: str):
    """اجرای تازه‌ی بک‌تست سیستمی (گران است؛ فقط با درخواست صریح)."""
    from backtest.system_backtest import run_system_backtest
    result = run_system_backtest(
        symbol, timeframe,
        thresholds=(config.EV_BACKTEST_THRESHOLD,))
    r_values = _oos_r_values_from_result(result)
    if r_values is None:
        r_values = []
    return _store_entry(symbol, timeframe, r_values, "system_backtest",
                        split_at=result.get("split_at"))


# ---------------------------------------------------------------------------
# API اصلی EV
# ---------------------------------------------------------------------------
def ev_for_candidate(symbol: str, timeframe: str, direction: str = None,
                     refresh: bool = False, allow_compute: bool = True):
    """
    EV خالص خارج‌ازنمونه‌ی یک ترکیب ارز/تایم‌فریم را برمی‌گرداند
    (خروجی compute_ev + فراداده). اگر هیچ منبعی نباشد و محاسبه هم مجاز
    نباشد، None برمی‌گردد.

    direction پذیرفته می‌شود تا صدا‌زننده راحت باشد، ولی طبق محدودیت
    مستندشده در docstring ماژول، آمار به‌تفکیک جهت نداریم: EV برای هر دو
    جهت یکی است و کلید direction_specific=False این را صریح می‌گوید.
    refresh=True کش و فایل‌های ذخیره‌شده را رد می‌کند و بک‌تست تازه
    می‌گیرد (بازنشانی صریح کش).
    """
    entry = None
    if refresh:
        entry = _entry_from_fresh_backtest(symbol, timeframe)
    else:
        entry = _entry_from_memory_or_disk(symbol, timeframe)
        if entry is None:
            entry = _entry_from_stored_files(symbol, timeframe)
        if entry is None and allow_compute:
            entry = _entry_from_fresh_backtest(symbol, timeframe)
    if entry is None:
        return None
    ev = compute_ev(entry.get("r_values_oos") or [])
    ev.update({
        "symbol": symbol.upper(), "timeframe": timeframe,
        "threshold": entry.get("threshold", config.EV_BACKTEST_THRESHOLD),
        "source": entry.get("source"),
        "computed_at": entry.get("computed_at"),
        "split_at": entry.get("split_at"),
        "direction_specific": False,
    })
    return ev


def refresh_ev_cache(symbols: list, timeframes: list, progress: bool = True) -> dict:
    """
    بازنشانی صریح کش EV: برای هر ترکیب بک‌تست تازه اجرا و در کش ذخیره
    می‌شود (کند است — هر ترکیب حدود یک دقیقه). خروجی: خلاصه‌ی موفقیت/خطا.
    """
    done, failed = [], []
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال بازنشانی EV برای {symbol} {timeframe} ...", flush=True)
            try:
                ev = ev_for_candidate(symbol, timeframe, refresh=True)
                done.append({"symbol": symbol.upper(), "timeframe": timeframe,
                             "n_trades": ev["n_trades"] if ev else 0,
                             "expectancy_r": ev["expectancy_r"] if ev else None})
            except Exception as exc:
                failed.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
    return {"refreshed": done, "errors": failed}


# ---------------------------------------------------------------------------
# قواعد تصمیم
# ---------------------------------------------------------------------------
def decide_candidate(candidate: dict, ev: dict = None) -> dict:
    """
    وضعیت تصمیم یک نامزد را از روی (امتیاز، جهت، حد ضرر/هدف، EV) تعیین
    می‌کند. قواعد دقیق و قطعی (همه‌ی آستانه‌ها در config):

    ۱) حد ضرر یا هدف نباشد → NO_TRADE (نبود نقطه‌ی ابطال/هدف = رد اجباری)
    ۲) EV موجود نباشد → NO_TRADE (بدون آمار خارج‌ازنمونه تصمیم نمی‌گیریم)
    ۳) نمونه ناکافی (n < EV_SAMPLE_INSUFFICIENT_MAX) → NO_TRADE
    ۴) EV خالص ≤ ۰ → NO_TRADE (رد اجباری؛ هر امتیازی را خنثی می‌کند)
    ۵) امتیاز ≥ EV_STRONG_SCORE و EV > ۰ و نمونه «قابل اتکا» →
       STRONG_BUY / STRONG_SELL (بر اساس جهت نامزد)
    ۶) امتیاز ≥ EV_BUY_SCORE و EV > ۰ و نمونه «قابل اتکا» → BUY / SELL
    ۷) بقیه‌ی حالت‌های EV > ۰ (امتیاز زیر آستانه‌ی خرید، یا نمونه «کم»
       با امتیاز ≥ آستانه‌ی خرید) → WAIT (جهت محتمل است ولی شرایط کامل
       نیست)

    خروجی: {"decision", "decision_fa", "decision_reason"} — دلیل همیشه
    فارسی و صریح است، مخصوصاً برای NO_TRADE.
    """
    if ev is None:
        ev = candidate.get("ev")
    direction = candidate.get("direction") or "long"
    score = candidate.get("score") or 0.0

    def out(decision, reason):
        return {"decision": decision,
                "decision_fa": DECISION_FA[decision],
                "decision_reason": reason}

    if candidate.get("stop") is None or candidate.get("target") is None:
        return out("NO_TRADE",
                   "رد: حد ضرر یا هدف مشخص نیست — بدون نقطه‌ی ابطال و هدف، معامله تعریف نمی‌شود.")
    if ev is None:
        return out("NO_TRADE",
                   "رد: آمار EV خارج‌ازنمونه برای این ترکیب موجود نیست (کش EV خالی است) — "
                   "بدون آمار تصمیم نمی‌گیریم؛ کش را بازنشانی کنید.")
    n = ev.get("n_trades") or 0
    if ev.get("reliability") == "insufficient" or n < config.EV_SAMPLE_INSUFFICIENT_MAX:
        return out("NO_TRADE",
                   f"رد: نمونه‌ی خارج‌ازنمونه ناکافی است ({n} معامله؛ حداقل "
                   f"{config.EV_SAMPLE_INSUFFICIENT_MAX}) — آمار قابل اتکا نیست.")
    expectancy = ev.get("expectancy_r")
    if expectancy is None or expectancy <= 0:
        return out("NO_TRADE",
                   f"رد اجباری: امید ریاضی خالص خارج‌ازنمونه نامثبت است ({expectancy}R ≤ 0) — "
                   "امتیاز بالا این رد را خنثی نمی‌کند.")
    # از اینجا EV > 0 با نمونه‌ی حداقل «کم» داریم.
    if ev.get("reliability") == "ok" and score >= config.EV_STRONG_SCORE:
        dec = "STRONG_BUY" if direction == "long" else "STRONG_SELL"
        return out(dec,
                   f"EV خالص مثبت ({expectancy:+}R، {n} معامله‌ی خارج‌ازنمونه) و امتیاز "
                   f"{score}/100 در بازه‌ی قوی است.")
    if ev.get("reliability") == "ok" and score >= config.EV_BUY_SCORE:
        dec = "BUY" if direction == "long" else "SELL"
        return out(dec,
                   f"EV خالص مثبت ({expectancy:+}R، {n} معامله‌ی خارج‌ازنمونه) و امتیاز "
                   f"{score}/100 از آستانه‌ی خرید گذشته است.")
    if ev.get("reliability") != "ok":
        return out("WAIT",
                   f"EV مثبت است ({expectancy:+}R) ولی نمونه کم است ({n} معامله) — "
                   "برای تصمیم خرید/فروش کافی نیست؛ منتظر داده‌ی بیشتر بمانید.")
    return out("WAIT",
               f"EV مثبت است ({expectancy:+}R) ولی امتیاز فعلی ({score}/100) به آستانه‌ی "
               f"خرید ({config.EV_BUY_SCORE:.0f}) نرسیده است.")


def format_ev_fa(ev: dict) -> str:
    """یک خط فارسی آماده‌ی نمایش برای آمار EV (یا پیام نبود آن)."""
    if ev is None:
        return "EV: موجود نیست (کش EV خالی)"
    if ev.get("expectancy_r") is None:
        # ورودی کش با صفر معامله (مثلاً ترکیب تازه‌ی فارکس قبل از این‌که
        # بک‌تستش معامله‌ای بسازد): آماری برای نمایش نیست؛ قبل از این گارد،
        # قالب‌بندی :+ روی None می‌ترکید (باگ نمایش، نه محاسبه).
        return (f"EV: هنوز معامله‌ی خارج‌ازنمونه‌ای برای این ترکیب ثبت "
                f"نشده است (نمونه: {ev.get('n_trades', 0)} معامله — "
                f"{ev.get('reliability_fa', 'نمونه ناکافی')})")
    pf = ev.get("profit_factor")
    pf_txt = "∞" if pf == float("inf") else ("—" if pf is None else f"{pf}")
    return (f"EV خالص (خارج‌ازنمونه): {ev['expectancy_r']:+}R | "
            f"نرخ برد: {ev['win_rate_pct']}٪ | نرخ برد سربه‌سر: "
            f"{ev['break_even_win_rate_pct']}٪ | پروفیت فاکتور: {pf_txt} | "
            f"نمونه: {ev['n_trades']} معامله ({ev['reliability_fa']})")
