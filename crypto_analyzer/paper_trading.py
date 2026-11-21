"""
paper_trading.py — صرافی کاغذی (Paper Trading) برای تست زنده‌ی ربات اصلی.

کاملاً افزودنی است: هیچ محاسبه‌ی موجودی (اندیکاتور، الگو، اسکنر، بک‌تست)
تغییر نمی‌کند. این ماژول فقط «مصرف‌کننده»‌ی خروجی‌های موجود است:
- قیمت لحظه‌ای از اندپوینت عمومی بایننس (/api/v3/ticker/price) با همان
  ترتیب دامنه‌های data/binance_client.py و کش ۳ ثانیه‌ای.
- حالت خودکار فقط خروجی scanner.score_candidate (رفتار پیش‌فرض) را می‌خواند.

مدل حسابداری (مارجینی و متقارن — مستند):
- مبلغ پوزیشن (amount_usdt) همان مارجین/ارزش مبنای معامله است؛
  qty = amount / entry_price.
- سود/زیان ناخالص: لانگ (exit-entry)/entry × amount و شورت دقیقاً قرینه:
  (entry-exit)/entry × amount. برای شورت وثیقه‌ی جداگانه‌ای مدل نمی‌شود؛
  همان مبلغ مبناست و PnL متقارن است.
- کارمزد: PAPER_FEE_PCT درصد از مبلغ پوزیشن، در هر طرف (باز و بسته‌شدن)،
  جدا کسر می‌شود — همان مدل ساده‌ی بک‌تست سیستمی.
- موجودی (balance) = وجه آزاد؛ هنگام بازشدن، مبلغ + کارمزد ورود از آن کم
  می‌شود و هنگام بسته‌شدن مبلغ + PnL ناخالص − کارمزد خروج برمی‌گردد.
- R هر معامله‌ی بسته: net_pnl / (|entry-stop| × qty) — فقط وقتی حد ضرر
  دارد؛ بدون حد ضرر R برابر null است.

وضعیت هر حساب در paper_state.json کنار همین فایل (نوشتن اتمیک) نگه داشته
می‌شود؛ حساب‌ها با chat_id از هم جدا هستند تا ربات به هر کاربر حساب خودش
را بدهد. موتور پس‌زمینه ندارد: evaluate() باید از بیرون (تعامل کاربر یا
جاب دوره‌ای ربات) صدا زده شود.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

import config
from data.binance_client import BinanceClient, BinanceError

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "paper_state.json")
_LOCK = threading.Lock()

_TF_SECONDS = {"1h": 3600, "4h": 4 * 3600, "1d": 86400, "1w": 7 * 86400}


class PaperError(Exception):
    """خطای معاملات کاغذی (ورودی نامعتبر، موجودی ناکافی و …)."""
    pass


# ---------------------------------------------------------------------------
# فید قیمت (با امکان تزریق تابع قیمت برای تست آفلاین)
# ---------------------------------------------------------------------------
_price_fn = None          # اگر ست شود، به‌جای شبکه استفاده می‌شود (تست)
_price_cache = {}         # symbol -> (epoch, price)


def set_price_function(fn):
    """تزریق تابع قیمت برای تست آفلاین؛ با None به حالت شبکه برمی‌گردد."""
    global _price_fn
    _price_fn = fn
    _price_cache.clear()


def get_price(symbol: str) -> float:
    """
    قیمت لحظه‌ای اسپات از /api/v3/ticker/price بایننس (بدون API key) با
    همان ترتیب fallback دامنه‌های BINANCE_BASE_URLS و کش چندثانیه‌ای
    (PAPER_PRICE_CACHE_SECONDS). نماد مثل «BTC» یا «BTCUSDT» پذیرفته می‌شود.

    افزودنی فارکس: نمادهای مجموعه‌ی فارکس (config.FOREX_SYMBOLS) به
    بایننس نمی‌روند؛ قیمتشان بسته‌شدن آخرین کندل کامل فید Dukascopy
    است — با تأخیر (FOREX_FA.md) و فقط برای معاملات کاغذی دستی.
    """
    from data import forex_client
    if forex_client.is_forex_symbol(symbol):
        price, _ts = forex_client.latest_price(symbol)
        return float(price)
    sym = BinanceClient.normalize_symbol(symbol)
    if _price_fn is not None:
        return float(_price_fn(sym))
    now = time.time()
    cached = _price_cache.get(sym)
    ttl = getattr(config, "PAPER_PRICE_CACHE_SECONDS", 3)
    if cached and now - cached[0] < ttl:
        return cached[1]
    timeout = getattr(config, "BINANCE_REQUEST_TIMEOUT_SECONDS", 15)
    last_error = None
    for base in getattr(config, "BINANCE_BASE_URLS",
                        ["https://data-api.binance.vision"]):
        try:
            resp = requests.get(f"{base}/api/v3/ticker/price",
                                params={"symbol": sym}, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            continue
        if resp.status_code == 200:
            price = float(resp.json()["price"])
            _price_cache[sym] = (now, price)
            return price
        if resp.status_code == 400:
            raise BinanceError(f"نماد '{sym}' در بایننس یافت نشد.")
        last_error = BinanceError(f"HTTP {resp.status_code} از {base}")
    raise BinanceError(f"گرفتن قیمت {sym} ناموفق بود: {last_error}")


# ---------------------------------------------------------------------------
# ذخیره‌سازی وضعیت (paper_state.json — نوشتن اتمیک)
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        if isinstance(state, dict) and isinstance(state.get("accounts"), dict):
            return state
    except (OSError, ValueError):
        pass
    return {"accounts": {}}


def _save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_FILE)


def _new_account() -> dict:
    return {
        "balance": float(config.PAPER_START_BALANCE_USDT),
        "realized_pnl": 0.0,
        "positions": [],
        "history": [],
        "next_id": 1,
        "auto": {
            "enabled": False,
            "timeframe": config.PAPER_AUTO_TIMEFRAME,
            "threshold": float(config.PAPER_AUTO_THRESHOLD),
            "risk_pct": float(config.PAPER_AUTO_RISK_PCT),
            "max_open": int(config.PAPER_AUTO_MAX_OPEN),
        },
        "cooldowns": {},   # symbol -> epoch بسته‌شدن آخر (برای حالت خودکار)
    }


def _account_in(state: dict, chat_id) -> dict:
    key = str(chat_id)
    acct = state["accounts"].get(key)
    if acct is None:
        acct = _new_account()
        state["accounts"][key] = acct
    return acct


def all_chat_ids() -> list:
    """همه‌ی chat_id هایی که حساب کاغذی دارند (برای جاب دوره‌ای ربات)."""
    with _LOCK:
        return list(_load_state()["accounts"].keys())


# ---------------------------------------------------------------------------
# محاسبات داخلی پوزیشن
# ---------------------------------------------------------------------------
def _fee(amount: float) -> float:
    return amount * float(config.PAPER_FEE_PCT) / 100.0


def _gross_pnl(pos: dict, price: float) -> float:
    """سود/زیان ناخالص در قیمت داده‌شده — متقارن برای لانگ/شورت."""
    entry, amount = pos["entry_price"], pos["amount_usdt"]
    if pos["direction"] == "long":
        return (price - entry) / entry * amount
    return (entry - price) / entry * amount


def _unrealized_net(pos: dict, price: float) -> float:
    """PnL تحقق‌نیافته‌ی خالص (بعد از کارمزد ورود و کارمزد خروج برآوردی)."""
    return _gross_pnl(pos, price) - pos["entry_fee"] - _fee(pos["amount_usdt"])


def _close_record(acct: dict, pos: dict, exit_price: float, reason: str) -> dict:
    """پوزیشن را با قیمت/دلیل داده‌شده می‌بندد، حساب را به‌روز و رکورد بسته برمی‌گرداند."""
    gross = _gross_pnl(pos, exit_price)
    exit_fee = _fee(pos["amount_usdt"])
    net = gross - pos["entry_fee"] - exit_fee
    acct["balance"] += pos["amount_usdt"] + gross - exit_fee
    acct["realized_pnl"] += net
    r_mult = None
    if pos.get("stop") is not None:
        risk_usdt = abs(pos["entry_price"] - pos["stop"]) * pos["qty"]
        if risk_usdt > 0:
            r_mult = round(net / risk_usdt, 3)
    record = dict(pos)
    record.update({
        "status": "closed",
        "exit_price": exit_price,
        "exit_reason": reason,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "gross_pnl": round(gross, 4),
        "exit_fee": round(exit_fee, 4),
        "net_pnl": round(net, 4),
        "r_multiple": r_mult,
    })
    acct["positions"] = [p for p in acct["positions"] if p["id"] != pos["id"]]
    acct["history"].append(record)
    acct["cooldowns"][pos["symbol"]] = time.time()
    return record


def _effective_stop(pos: dict):
    """
    (حد ضرر مؤثر, دلیل) را با ترکیب حد ضرر ثابت و تریلینگ برمی‌گرداند.
    best_price بهترین قیمت به نفع پوزیشن از زمان بازشدن است؛ تریلینگ‌استاپ
    روی آن سوار است: لانگ best×(1-trail٪)، شورت best×(1+trail٪).

    ⚠️ هم‌ترازی با بک‌تست (افزودنی، تست A/B در TRAILING_AB_FA.md):
    پارامترهای پذیرفته‌شده‌ی A/B در config.SYSBT_TRAILING_ADOPTED بر
    حسب R هستند (فعال‌سازی 1.0R، فاصله 0.5R پشت بهترین قیمت) و مبنای
    پیش‌فرض حالت «خودکار»‌اند — تابع open_position همان‌جا که
    trailing_pct داده نشده باشد برای پوزیشن خودکارِ دارای حد ضرر،
    فاصله‌ی R-محور را به درصدِ معادلِ همان لحظه (0.5×|ورود−استاپ|/ورود
    ×۱۰۰) تبدیل و ذخیره می‌کند تا موتور درصدی این ماژول همان فاصله را
    اجرا کند. دو انحراف باقی می‌ماند و مستند است: (۱) این ماژول درصدی
    است و فاصله با دورشدن قیمت از ورود، بر حسب R منقبض می‌شود؛ (۲)
    آستانه‌ی فعال‌سازی ندارد و از ورود فعال است. پوزیشن‌های دستی
    دقیقاً مثل قبل‌اند: بدون trailing_pct یعنی بدون تریلینگ.
    """
    stop, reason = pos.get("stop"), "stop"
    trail = pos.get("trailing_pct")
    if trail:
        best = pos.get("best_price", pos["entry_price"])
        if pos["direction"] == "long":
            tstop = best * (1 - trail / 100.0)
            if stop is None or tstop > stop:
                stop, reason = tstop, "trailing"
        else:
            tstop = best * (1 + trail / 100.0)
            if stop is None or tstop < stop:
                stop, reason = tstop, "trailing"
    return stop, reason


def _fmt_price(value) -> str:
    if value is None:
        return "—"
    v = float(value)
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.4f}"
    if v >= 0.01:
        return f"{v:.6f}"
    return f"{v:.10f}".rstrip("0")


def _fmt_usdt(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}"


_DIR_FA = {"long": "لانگ (خرید)", "short": "شورت (فروش)"}
_SOURCE_FA = {"manual": "✋ دستی", "auto": "🤖 خودکار"}
_REASON_FA = {"manual": "بستن دستی", "stop": "حد ضرر", "target": "هدف",
              "trailing": "تریلینگ‌استاپ"}


# ---------------------------------------------------------------------------
# API حساب
# ---------------------------------------------------------------------------
def get_account(chat_id) -> dict:
    """رکورد حساب (موجودی، پوزیشن‌های باز، تاریخچه، تنظیمات خودکار)."""
    with _LOCK:
        state = _load_state()
        acct = _account_in(state, chat_id)
        _save_state(state)
        return acct


def reset_account(chat_id) -> dict:
    """حساب را به موجودی شروع برمی‌گرداند و همه‌چیز را پاک می‌کند."""
    with _LOCK:
        state = _load_state()
        state["accounts"][str(chat_id)] = _new_account()
        _save_state(state)
        return state["accounts"][str(chat_id)]


def open_position(chat_id, symbol: str, direction: str, amount,
                  stop=None, target=None, trailing_pct=None,
                  source: str = "manual") -> dict:
    """
    پوزیشن مارکت باز می‌کند. amount مبلغ مبنا (USDT) است؛ کارمزد ورود از
    موجودی آزاد کسر می‌شود. stop/target/trailing_pct اختیاری‌اند و جهتشان
    نسبت به قیمت ورود اعتبارسنجی می‌شود.
    source منشأ پوزیشن است: "manual" (بازشده با دست کاربر) یا "auto"
    (حالت خودکار) — در لیست پوزیشن‌ها و تاریخچه با برچسب نشان داده می‌شود.
    """
    direction = str(direction).lower()
    if direction not in ("long", "short"):
        raise PaperError("جهت باید long یا short باشد.")
    amount = float(amount)
    if amount <= 0:
        raise PaperError("مبلغ باید مثبت باشد.")
    from data import forex_client
    if forex_client.is_forex_symbol(symbol):
        # افزودنی فارکس: پوزیشن فارکس فقط دستی است (خودکار کریپتویی
        # می‌ماند) و قیمتش از فید با تأخیر می‌آید — برچسب market روی
        # پوزیشن ذخیره می‌شود تا در نمایش‌ها معلوم باشد.
        if source != "manual":
            raise PaperError("پوزیشن فارکس فقط به‌صورت دستی باز می‌شود.")
        sym = forex_client.normalize_forex_symbol(symbol)
    else:
        sym = BinanceClient.normalize_symbol(symbol).replace("USDT", "")
    entry = get_price(sym)
    if stop is not None:
        stop = float(stop)
        if direction == "long" and not stop < entry:
            raise PaperError("برای لانگ، حد ضرر باید پایین‌تر از قیمت ورود باشد.")
        if direction == "short" and not stop > entry:
            raise PaperError("برای شورت، حد ضرر باید بالاتر از قیمت ورود باشد.")
    if target is not None:
        target = float(target)
        if direction == "long" and not target > entry:
            raise PaperError("برای لانگ، هدف باید بالاتر از قیمت ورود باشد.")
        if direction == "short" and not target < entry:
            raise PaperError("برای شورت، هدف باید پایین‌تر از قیمت ورود باشد.")
    if trailing_pct is not None:
        trailing_pct = float(trailing_pct)
        if not 0 < trailing_pct < 100:
            raise PaperError("درصد تریلینگ باید بین ۰ و ۱۰۰ باشد.")
    elif source == "auto" and stop is not None:
        # هم‌ترازی با A/B (افزودنی): پوزیشن خودکار بدون تریلینگ صریح،
        # فاصله‌ی پذیرفته‌شده‌ی R-محور بک‌تست را می‌گیرد — تبدیل به درصد
        # معادل در لحظه‌ی ورود: trail_r × |ورود − استاپ| / ورود × ۱۰۰.
        # فقط حالت خودکار و فقط وقتی حد ضرر هست؛ دستی دست نخورده.
        adopted = getattr(config, "SYSBT_TRAILING_ADOPTED", None)
        if adopted:
            trailing_pct = round(
                float(adopted["trail_r"]) * abs(entry - stop) / entry * 100.0, 4)
            if not 0 < trailing_pct < 100:
                trailing_pct = None
    entry_fee = _fee(amount)
    with _LOCK:
        state = _load_state()
        acct = _account_in(state, chat_id)
        if acct["balance"] < amount + entry_fee:
            raise PaperError(
                f"موجودی آزاد کافی نیست: {_fmt_usdt(acct['balance'])} USDT "
                f"(لازم: {_fmt_usdt(amount + entry_fee)})")
        acct["balance"] -= amount + entry_fee
        pos = {
            "id": acct["next_id"],
            "symbol": sym,
            "direction": direction,
            "amount_usdt": amount,
            "qty": amount / entry,
            "entry_price": entry,
            "entry_fee": round(entry_fee, 4),
            "stop": stop,
            "target": target,
            "trailing_pct": trailing_pct,
            "best_price": entry,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "source": source if source in ("manual", "auto") else "manual",
            "market": "forex" if forex_client.is_forex_symbol(sym) else "crypto",
        }
        acct["next_id"] += 1
        acct["positions"].append(pos)
        _save_state(state)
        return pos


def close_position(chat_id, pos_id) -> dict:
    """پوزیشن باز را با قیمت لحظه‌ای و دلیل «دستی» می‌بندد."""
    with _LOCK:
        state = _load_state()
        acct = _account_in(state, chat_id)
        pos = next((p for p in acct["positions"] if p["id"] == int(pos_id)), None)
        if pos is None:
            raise PaperError(f"پوزیشن باز با شناسه‌ی {pos_id} یافت نشد.")
        record = _close_record(acct, pos, get_price(pos["symbol"]), "manual")
        _save_state(state)
        return record


def evaluate(chat_id, price_fn=None) -> list:
    """
    موتور ارزیابی: برای هر پوزیشن باز، best_price را با قیمت فعلی به‌روز
    می‌کند و اگر حد ضرر (ثابت یا تریلینگ) یا هدف خورده باشد می‌بندد.
    خروجی: لیست رویدادهای بسته‌شدن (دیکشنری با کلید message فارسی).
    price_fn فقط برای تست است؛ در حالت عادی get_price شبکه/تزریقی صدا می‌شود.
    """
    events = []
    with _LOCK:
        state = _load_state()
        acct = _account_in(state, chat_id)
        for pos in list(acct["positions"]):
            price = float(price_fn(pos["symbol"])) if price_fn else get_price(pos["symbol"])
            if pos["direction"] == "long":
                pos["best_price"] = max(pos.get("best_price", pos["entry_price"]), price)
            else:
                pos["best_price"] = min(pos.get("best_price", pos["entry_price"]), price)
            eff_stop, stop_reason = _effective_stop(pos)
            reason = None
            if pos["direction"] == "long":
                if eff_stop is not None and price <= eff_stop:
                    reason = stop_reason
                elif pos.get("target") is not None and price >= pos["target"]:
                    reason = "target"
            else:
                if eff_stop is not None and price >= eff_stop:
                    reason = stop_reason
                elif pos.get("target") is not None and price <= pos["target"]:
                    reason = "target"
            if reason:
                record = _close_record(acct, pos, price, reason)
                events.append({
                    "type": "closed",
                    "position": record,
                    "message": (f"🔔 پوزیشن #{record['id']} {record['symbol']}/USDT "
                                f"({_DIR_FA[record['direction']]}, "
                                f"{_SOURCE_FA.get(record.get('source', 'manual'), '✋ دستی')}) بسته شد — دلیل: "
                                f"{_REASON_FA[reason]} | قیمت خروج: {_fmt_price(price)} | "
                                f"سود/زیان خالص: {_fmt_usdt(record['net_pnl'])} USDT"),
                })
        _save_state(state)
    return events


def list_positions(chat_id) -> list:
    """پوزیشن‌های باز همراه با قیمت فعلی و PnL زنده‌ی هرکدام."""
    with _LOCK:
        acct = _account_in(_load_state(), chat_id)
        out = []
        for pos in acct["positions"]:
            price = get_price(pos["symbol"])
            item = dict(pos)
            item["current_price"] = price
            item["gross_pnl"] = round(_gross_pnl(pos, price), 4)
            item["net_pnl"] = round(_unrealized_net(pos, price), 4)
            eff_stop, _ = _effective_stop(pos)
            item["effective_stop"] = eff_stop
            out.append(item)
        return out


def portfolio_summary(chat_id) -> dict:
    """
    خلاصه‌ی سبد: موجودی آزاد، equity (آزاد + مارجین پوزیشن‌ها + PnL
    تحقق‌نیافته‌ی ناخالص)، PnL باز خالص، PnL تحقق‌یافته، نرخ برد و تعداد
    معاملات بسته.
    """
    with _LOCK:
        acct = _account_in(_load_state(), chat_id)
    open_pnl_net = 0.0
    margin_locked = 0.0
    for pos in acct["positions"]:
        price = get_price(pos["symbol"])
        margin_locked += pos["amount_usdt"] + _gross_pnl(pos, price)
        open_pnl_net += _unrealized_net(pos, price)
    hist = acct["history"]
    wins = sum(1 for h in hist if h["net_pnl"] > 0)
    return {
        "balance": round(acct["balance"], 4),
        "equity": round(acct["balance"] + margin_locked, 4),
        "open_pnl": round(open_pnl_net, 4),
        "realized_pnl": round(acct["realized_pnl"], 4),
        "open_positions": len(acct["positions"]),
        "trades": len(hist),
        "wins": wins,
        "win_rate_pct": round(wins / len(hist) * 100.0, 1) if hist else None,
        "auto_enabled": bool(acct["auto"]["enabled"]),
    }


def history(chat_id, limit: int = 20) -> list:
    """آخرین معاملات بسته‌شده (جدیدترین اول)."""
    with _LOCK:
        acct = _account_in(_load_state(), chat_id)
        return list(reversed(acct["history"]))[:limit]


def set_auto(chat_id, enabled=None, **settings) -> dict:
    """روشن/خاموش‌کردن و تنظیم حالت خودکار حساب؛ تنظیمات برگشتی را برمی‌گرداند."""
    with _LOCK:
        state = _load_state()
        acct = _account_in(state, chat_id)
        auto = acct["auto"]
        if enabled is not None:
            auto["enabled"] = bool(enabled)
        for key in ("timeframe", "threshold", "risk_pct", "max_open"):
            if key in settings and settings[key] is not None:
                auto[key] = settings[key]
        _save_state(state)
        return dict(auto)


# ---------------------------------------------------------------------------
# حالت خودکار — تست زنده‌ی ربات اصلی روی حساب کاغذی
# ---------------------------------------------------------------------------
def auto_tick(chat_id) -> list:
    """
    یک قدم حالت خودکار:
    ۱) اول evaluate — پوزیشن‌های رسیده به حد/هدف بسته می‌شوند.
    ۲) اگر خودکار روشن باشد: برای هر ارز SCAN_SYMBOLS روی تایم‌فریم تنظیم‌شده
       همان امتیازدهی پیش‌فرض اسکنر (score_candidate بدون هیچ پارامتر افزودنی)
       اجرا می‌شود. نامزدی که امتیازش ≥ آستانه باشد، در آن ارز پوزیشن باز
       نداشته باشد، کول‌داون ۱۲ کندلی‌اش نگذشته باشد و سقف پوزیشن هم‌زمان
       پر نشده باشد → با حد ضرر و هدف خود نامزد باز می‌شود؛ مبلغ طوری است
       که ریسک تا حد ضرر = risk_pct درصد از equity باشد.
       دروازه‌ی EV (افزودنی): علاوه بر آستانه‌ی امتیاز، تصمیم لایه‌ی EV
       نامزد (scanner.attach_ev_decision — فقط از کش خوانده می‌شود) باید
       BUY/SELL/STRONG_* باشد؛ یعنی EV خالص خارج‌ازنمونه‌ی آن ترکیب
       مثبت و نمونه کافی باشد. NO_TRADE و WAIT خودکار باز نمی‌شوند.
       این دروازه فقط حالت خودکار را محدود می‌کند؛ بازکردن دستی کاربر
       عمداً بدون هیچ دروازه‌ای می‌ماند.
    خروجی: لیست رشته‌های رویداد (باز/بسته‌شدن) برای اطلاع‌رسانی ربات.
    """
    actions = [ev["message"] for ev in evaluate(chat_id)]
    with _LOCK:
        acct = _account_in(_load_state(), chat_id)
        auto = dict(acct["auto"])
    if not auto.get("enabled"):
        return actions

    import scanner
    from analysis import analyzer

    tf = auto["timeframe"]
    threshold = float(auto["threshold"])
    risk_pct = float(auto["risk_pct"])
    max_open = int(auto["max_open"])
    tf_seconds = _TF_SECONDS.get(tf, 4 * 3600)
    cooldown_secs = int(config.PAPER_AUTO_COOLDOWN_CANDLES) * tf_seconds

    for symbol in config.SCAN_SYMBOLS:
        with _LOCK:
            acct = _account_in(_load_state(), chat_id)
            open_syms = {p["symbol"] for p in acct["positions"]}
            open_count = len(acct["positions"])
            last_close = acct["cooldowns"].get(symbol.upper())
        if open_count >= max_open:
            break
        if symbol.upper() in open_syms:
            continue
        if last_close and time.time() - float(last_close) < cooldown_secs:
            continue
        try:
            df = analyzer.fetch_candles_binance(symbol, tf)
            report = analyzer.build_report(df, symbol, tf)
            cand = scanner.score_candidate(df, report, symbol, tf)
        except Exception:
            continue  # خطای یک ارز، تیک خودکار را متوقف نمی‌کند
        # دروازه‌ی EV: تصمیم باید قابل اقدام باشد (EV مثبت + نمونه کافی)؛
        # هر دو شرط «آستانه‌ی امتیاز حساب» و «دروازه‌ی EV» با هم لازم‌اند.
        try:
            scanner.attach_ev_decision(cand)
        except Exception:
            continue
        if cand.get("decision") not in ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL"):
            continue
        if cand["score"] < threshold or cand.get("stop") is None:
            continue
        entry, stop = cand["entry"], cand["stop"]
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            continue
        try:
            equity = portfolio_summary(chat_id)["equity"]
        except Exception:
            continue
        amount = (equity * risk_pct / 100.0) / risk_per_unit * entry
        if amount < float(config.PAPER_AUTO_MIN_AMOUNT_USDT):
            continue
        try:
            pos = open_position(chat_id, symbol, cand["direction"], round(amount, 2),
                                stop=stop, target=cand.get("target"), source="auto")
        except Exception:
            continue  # موجودی/قیمت/اعتبارسنجی یک ارز، تیک خودکار را متوقف نمی‌کند
        actions.append(
            f"🤖 حالت خودکار: پوزیشن #{pos['id']} {pos['symbol']}/USDT "
            f"({_DIR_FA[pos['direction']]}) باز شد — امتیاز {cand['score']}/100 ({tf}) | "
            f"مبلغ: {_fmt_usdt(pos['amount_usdt'])} USDT | ورود: {_fmt_price(pos['entry_price'])} | "
            f"حد ضرر: {_fmt_price(pos['stop'])} | هدف: {_fmt_price(pos['target'])}")
    return actions


# ---------------------------------------------------------------------------
# متن‌های فارسی آماده برای نمایش (ربات و CLI)
# ---------------------------------------------------------------------------
def summary_text(chat_id) -> str:
    s = portfolio_summary(chat_id)
    lines = [
        "═══ 🧪 حساب کاغذی ═══",
        f"موجودی آزاد: {_fmt_usdt(s['balance'])} USDT",
        f"ارزش کل (Equity): {_fmt_usdt(s['equity'])} USDT",
        f"سود/زیان باز (خالص): {_fmt_usdt(s['open_pnl'])} USDT",
        f"سود/زیان تحقق‌یافته: {_fmt_usdt(s['realized_pnl'])} USDT",
        f"پوزیشن باز: {s['open_positions']} | معاملات بسته: {s['trades']}",
    ]
    if s["win_rate_pct"] is not None:
        lines.append(f"نرخ برد: {s['win_rate_pct']}٪ ({s['wins']} از {s['trades']})")
    lines.append(f"حالت خودکار: {'روشن 🤖' if s['auto_enabled'] else 'خاموش'}")
    lines.append("⚠️ معاملات کاغذی است؛ پول واقعی درگیر نیست.")
    return "\n".join(lines)


def positions_text(chat_id) -> str:
    positions = list_positions(chat_id)
    if not positions:
        return "📂 پوزیشن بازی نداری."
    lines = ["═══ 📂 پوزیشن‌های باز ═══"]
    for p in positions:
        lines.append(
            f"#{p['id']} {p['symbol']}{'' if p.get('market') == 'forex' else '/USDT'}"
            f"{' (فارکس — قیمت با تأخیر)' if p.get('market') == 'forex' else ''}"
            f" — {_DIR_FA[p['direction']]} | "
            f"{_SOURCE_FA.get(p.get('source', 'manual'), '✋ دستی')} | "
            f"مبلغ: {_fmt_usdt(p['amount_usdt'])} USDT\n"
            f"    ورود: {_fmt_price(p['entry_price'])} | فعلی: {_fmt_price(p['current_price'])} | "
            f"PnL زنده (خالص): {_fmt_usdt(p['net_pnl'])} USDT\n"
            f"    حد ضرر: {_fmt_price(p['effective_stop'])} | هدف: {_fmt_price(p['target'])}"
            + (f" | تریلینگ: {p['trailing_pct']}٪" if p.get("trailing_pct") else ""))
    return "\n".join(lines)


def history_text(chat_id, limit: int = 10) -> str:
    hist = history(chat_id, limit)
    if not hist:
        return "📜 هنوز معامله‌ی بسته‌شده‌ای نداری."
    lines = ["═══ 📜 تاریخچه‌ی معاملات ═══"]
    for h in hist:
        r_part = f" | {h['r_multiple']}R" if h.get("r_multiple") is not None else ""
        lines.append(
            f"#{h['id']} {h['symbol']}/USDT — {_DIR_FA[h['direction']]} | "
            f"{_SOURCE_FA.get(h.get('source', 'manual'), '✋ دستی')} | "
            f"{_REASON_FA.get(h['exit_reason'], h['exit_reason'])} | "
            f"خالص: {_fmt_usdt(h['net_pnl'])} USDT{r_part}")
    return "\n".join(lines)
