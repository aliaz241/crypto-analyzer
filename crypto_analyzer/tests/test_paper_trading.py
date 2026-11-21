#!/usr/bin/env python3
"""
tests/test_paper_trading.py
تست آفلاین موتور معاملات کاغذی با قیمت‌های جعلی تزریقی — بدون اینترنت:

    python tests/test_paper_trading.py

پوشش: بازکردن لانگ و رسیدن به هدف (PnL خالص از کارمزد)، حد ضرر،
تریلینگ‌استاپ، تقارن شورت، و مکانیک حالت خودکار (با اسکنر stub).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paper_trading as paper

# فایل وضعیت موقت + تابع قیمت قابل‌کنترل
paper.STATE_FILE = os.path.join(tempfile.mkdtemp(), "paper_state_test.json")
PRICES = {"BTCUSDT": 100.0}
paper.set_price_function(lambda sym: PRICES[sym])

CHAT = "test-chat"
fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- ۱) لانگ تا هدف -------------------------------------------------------
paper.reset_account(CHAT)
pos = paper.open_position(CHAT, "BTC", "long", 1000, stop=90, target=120)
check("باز شدن لانگ", pos["qty"] == 10.0 and pos["entry_price"] == 100.0)
check("منشأ پیش‌فرض: دستی", pos.get("source") == "manual")
acct = paper.get_account(CHAT)
check("کسر مبلغ + کارمزد ورود از موجودی", approx(acct["balance"], 10000 - 1000 - 1.0),
      f"balance={acct['balance']}")
PRICES["BTCUSDT"] = 120.0
events = paper.evaluate(CHAT)
check("بسته‌شدن با هدف", len(events) == 1 and events[0]["position"]["exit_reason"] == "target")
rec = events[0]["position"]
# ناخالص ۲۰۰، کارمزد ورود ۱، کارمزد خروج ۱ → خالص ۱۹۸ ؛ R = 198 / (10×10) = 1.98
check("PnL خالص از کارمزد", approx(rec["net_pnl"], 198.0), f"net={rec['net_pnl']}")
check("R معامله", approx(rec["r_multiple"], 1.98), f"R={rec['r_multiple']}")
acct = paper.get_account(CHAT)
check("برگشت موجودی", approx(acct["balance"], 10198.0), f"balance={acct['balance']}")
s = paper.portfolio_summary(CHAT)
check("خلاصه سبد: برد و نرخ برد", s["trades"] == 1 and s["win_rate_pct"] == 100.0)

# --- ۲) حد ضرر -------------------------------------------------------------
PRICES["BTCUSDT"] = 100.0
paper.open_position(CHAT, "BTC", "long", 1000, stop=90)
PRICES["BTCUSDT"] = 89.0
events = paper.evaluate(CHAT)
rec = events[0]["position"]
# ناخالص ‎−۱۱۰، کارمزدها ۲ → خالص ‎−۱۱۲
check("بسته‌شدن با حد ضرر", rec["exit_reason"] == "stop" and approx(rec["net_pnl"], -112.0),
      f"reason={rec['exit_reason']} net={rec['net_pnl']}")

# --- ۳) تریلینگ‌استاپ -------------------------------------------------------
PRICES["BTCUSDT"] = 100.0
paper.open_position(CHAT, "BTC", "long", 1000, trailing_pct=5)
PRICES["BTCUSDT"] = 110.0
check("تریلینگ: هنوز باز در اوج", paper.evaluate(CHAT) == [])
PRICES["BTCUSDT"] = 105.0   # استاپ تریلینگ = 104.5 → هنوز باز
check("تریلینگ: بالای استاپ شناور", paper.evaluate(CHAT) == [])
PRICES["BTCUSDT"] = 104.0   # پایین‌تر از 104.5 → بسته
events = paper.evaluate(CHAT)
rec = events[0]["position"]
check("تریلینگ: بسته‌شدن", rec["exit_reason"] == "trailing" and approx(rec["net_pnl"], 38.0),
      f"reason={rec['exit_reason']} net={rec['net_pnl']}")

# --- ۴) تقارن شورت ----------------------------------------------------------
PRICES["BTCUSDT"] = 100.0
pos = paper.open_position(CHAT, "BTC", "short", 1000, stop=110, target=80)
PRICES["BTCUSDT"] = 80.0
events = paper.evaluate(CHAT)
rec = events[0]["position"]
check("شورت: هدف و تقارن PnL", rec["exit_reason"] == "target" and approx(rec["net_pnl"], 198.0),
      f"net={rec['net_pnl']}")
# بستن دستی شورت در سود کمتر
PRICES["BTCUSDT"] = 100.0
pos = paper.open_position(CHAT, "BTC", "short", 1000)
PRICES["BTCUSDT"] = 90.0
rec = paper.close_position(CHAT, pos["id"])
check("شورت: بستن دستی", rec["exit_reason"] == "manual" and approx(rec["net_pnl"], 98.0),
      f"net={rec['net_pnl']}")
# بدون حد ضرر → R باید null باشد
check("بدون حد ضرر: R خالی", rec["r_multiple"] is None)

# --- ۵) اعتبارسنجی ------------------------------------------------------------
try:
    paper.open_position(CHAT, "BTC", "long", 500, stop=150)  # استاپ بالای ورود برای لانگ
    check("اعتبارسنجی جهت استاپ", False)
except paper.PaperError:
    check("اعتبارسنجی جهت استاپ", True)
try:
    paper.open_position(CHAT, "BTC", "long", 10 ** 9)
    check("اعتبارسنجی موجودی", False)
except paper.PaperError:
    check("اعتبارسنجی موجودی", True)

# --- ۶) مکانیک حالت خودکار (اسکنر stub) --------------------------------------
import scanner
from analysis import analyzer

paper.reset_account(CHAT)
paper.set_auto(CHAT, enabled=True, timeframe="4h", threshold=60, risk_pct=1.0, max_open=3)
PRICES["BTCUSDT"] = 100.0

_orig_fetch, _orig_build, _orig_score = (analyzer.fetch_candles_binance,
                                         analyzer.build_report,
                                         scanner.score_candidate)
analyzer.fetch_candles_binance = lambda sym, tf: None
analyzer.build_report = lambda df, sym, tf: {}
scanner.score_candidate = lambda df, rep, sym, tf: {
    "symbol": sym.upper(), "timeframe": tf, "direction": "long",
    "direction_fa": "لانگ (خرید)", "score": 75.0, "entry": 100.0,
    "stop": 90.0, "target": 120.0, "price": 100.0,
    # قرارداد جدید نامزد بعد از لایه‌ی EV: تصمیم قابل اقدام + آمار EV
    # (attach_ev_decision برای نامزدی که هر دو کلید را دارد دست نمی‌زند)
    "decision": "BUY",
    "ev": {"expectancy_r": 0.3, "n_trades": 40, "reliability": "ok"},
}
try:
    actions = paper.auto_tick(CHAT)
finally:
    analyzer.fetch_candles_binance, analyzer.build_report, scanner.score_candidate = (
        _orig_fetch, _orig_build, _orig_score)
opened = [a for a in actions if "باز شد" in a]
# اسکن ۲۰ ارز است ولی سقف هم‌زمان ۳ است؛ همه روی BTCUSDT قیمت دارند ولی
# ارزهای دیگر هم با همان stub امتیاز می‌گیرند — قیمتشان از _price_fn می‌آید
# و برای نمادهای دیگر KeyError می‌دهد و رد می‌شوند؛ فقط BTC باز می‌شود.
check("خودکار: بازشدن پوزیشن از روی امتیاز", len(opened) >= 1, f"actions={actions}")
positions = paper.list_positions(CHAT)
if positions:
    p0 = positions[0]
    # ریسک ۱٪ از equity ‏(۱۰۰۰۰) = ۱۰۰ → مبلغ = 100/(10) × 100 = 1000
    check("خودکار: سایز بر اساس ریسک", approx(p0["amount_usdt"], 1000.0, tol=1.0),
          f"amount={p0['amount_usdt']}")
    check("خودکار: استاپ/هدف از نامزد", p0["stop"] == 90.0 and p0["target"] == 120.0)
    check("خودکار: برچسب منشأ auto", p0.get("source") == "auto")
    check("برچسب منشأ در متن پوزیشن‌ها", "خودکار" in paper.positions_text(CHAT))
# هم‌زیستی دستی و خودکار در یک حساب + برچسب در تاریخچه
paper.open_position(CHAT, "BTC", "long", 100)   # دستی، کنار پوزیشن خودکار
check("هم‌زیستی دستی و خودکار در یک حساب",
      {p.get("source") for p in paper.list_positions(CHAT)} == {"manual", "auto"})
PRICES["BTCUSDT"] = 130.0
paper.evaluate(CHAT)   # هر دو با هدف بسته می‌شوند (دستی بدون هدف می‌ماند → دستی بسته نمی‌شود)
hist = paper.history(CHAT)
check("برچسب منشأ در تاریخچه", all("source" in h for h in hist) and
      any(h["source"] == "auto" for h in hist))
check("برچسب منشأ در متن تاریخچه", "خودکار" in paper.history_text(CHAT))
check("خودکار: بدون پوزیشن تکراری در همان ارز",
      len([p for p in paper.list_positions(CHAT) if p["symbol"] == "BTC"]) <= 1)

print()
if fails:
    print(f"❌ {len(fails)} تست ناموفق: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های معاملات کاغذی موفق بود.\n")
