#!/usr/bin/env python3
"""
tests/test_paper_trading.py
Offline test of the paper trading engine with injected fake prices — no internet:

    python tests/test_paper_trading.py

Coverage: opening a long and reaching the target (net PnL after fees), stop
loss, trailing stop, short symmetry, and the auto-mode mechanics (with a stub scanner).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paper_trading as paper

# Temporary state file + a controllable price function
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


# --- 1) Long to target ---
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
# Gross 200, entry fee 1, exit fee 1 → net 198 ; R = 198 / (10×10) = 1.98
check("PnL خالص از کارمزد", approx(rec["net_pnl"], 198.0), f"net={rec['net_pnl']}")
check("R معامله", approx(rec["r_multiple"], 1.98), f"R={rec['r_multiple']}")
acct = paper.get_account(CHAT)
check("برگشت موجودی", approx(acct["balance"], 10198.0), f"balance={acct['balance']}")
s = paper.portfolio_summary(CHAT)
check("خلاصه سبد: برد و نرخ برد", s["trades"] == 1 and s["win_rate_pct"] == 100.0)

# --- 2) Stop loss ---
PRICES["BTCUSDT"] = 100.0
paper.open_position(CHAT, "BTC", "long", 1000, stop=90)
PRICES["BTCUSDT"] = 89.0
events = paper.evaluate(CHAT)
rec = events[0]["position"]
# Gross −110, fees 2 → net −112
check("بسته‌شدن با حد ضرر", rec["exit_reason"] == "stop" and approx(rec["net_pnl"], -112.0),
      f"reason={rec['exit_reason']} net={rec['net_pnl']}")

# --- 3) Trailing stop ---
PRICES["BTCUSDT"] = 100.0
paper.open_position(CHAT, "BTC", "long", 1000, trailing_pct=5)
PRICES["BTCUSDT"] = 110.0
check("تریلینگ: هنوز باز در اوج", paper.evaluate(CHAT) == [])
PRICES["BTCUSDT"] = 105.0   # Trailing stop = 104.5 → still open
check("تریلینگ: بالای استاپ شناور", paper.evaluate(CHAT) == [])
PRICES["BTCUSDT"] = 104.0   # Below 104.5 → closed
events = paper.evaluate(CHAT)
rec = events[0]["position"]
check("تریلینگ: بسته‌شدن", rec["exit_reason"] == "trailing" and approx(rec["net_pnl"], 38.0),
      f"reason={rec['exit_reason']} net={rec['net_pnl']}")

# --- 4) Short symmetry ---
PRICES["BTCUSDT"] = 100.0
pos = paper.open_position(CHAT, "BTC", "short", 1000, stop=110, target=80)
PRICES["BTCUSDT"] = 80.0
events = paper.evaluate(CHAT)
rec = events[0]["position"]
check("شورت: هدف و تقارن PnL", rec["exit_reason"] == "target" and approx(rec["net_pnl"], 198.0),
      f"net={rec['net_pnl']}")
# Manually closing the short at a smaller profit
PRICES["BTCUSDT"] = 100.0
pos = paper.open_position(CHAT, "BTC", "short", 1000)
PRICES["BTCUSDT"] = 90.0
rec = paper.close_position(CHAT, pos["id"])
check("شورت: بستن دستی", rec["exit_reason"] == "manual" and approx(rec["net_pnl"], 98.0),
      f"net={rec['net_pnl']}")
# Without a stop loss → R must be null
check("بدون حد ضرر: R خالی", rec["r_multiple"] is None)

# --- 5) Validation ---
try:
    paper.open_position(CHAT, "BTC", "long", 500, stop=150)  # Stop above entry for a long
    check("اعتبارسنجی جهت استاپ", False)
except paper.PaperError:
    check("اعتبارسنجی جهت استاپ", True)
try:
    paper.open_position(CHAT, "BTC", "long", 10 ** 9)
    check("اعتبارسنجی موجودی", False)
except paper.PaperError:
    check("اعتبارسنجی موجودی", True)

# --- 6) Auto-mode mechanics (stub scanner) ---
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
    # The candidate's new contract after the EV layer: an actionable decision + EV statistics
    # (attach_ev_decision does not touch a candidate that has both keys)
    "decision": "BUY",
    "ev": {"expectancy_r": 0.3, "n_trades": 40, "reliability": "ok"},
}
try:
    actions = paper.auto_tick(CHAT)
finally:
    analyzer.fetch_candles_binance, analyzer.build_report, scanner.score_candidate = (
        _orig_fetch, _orig_build, _orig_score)
opened = [a for a in actions if "باز شد" in a]
# The scan is 20 symbols but the simultaneous cap is 3; all have a price on BTCUSDT but
# other symbols also get a score with the same stub — their price comes from _price_fn
# and for other symbols it raises KeyError and they are skipped; only BTC opens.
check("خودکار: بازشدن پوزیشن از روی امتیاز", len(opened) >= 1, f"actions={actions}")
positions = paper.list_positions(CHAT)
if positions:
    p0 = positions[0]
    # Risk 1% of equity (10000) = 100 → amount = 100/(10) × 100 = 1000
    check("خودکار: سایز بر اساس ریسک", approx(p0["amount_usdt"], 1000.0, tol=1.0),
          f"amount={p0['amount_usdt']}")
    check("خودکار: استاپ/هدف از نامزد", p0["stop"] == 90.0 and p0["target"] == 120.0)
    check("خودکار: برچسب منشأ auto", p0.get("source") == "auto")
    check("برچسب منشأ در متن پوزیشن‌ها", "خودکار" in paper.positions_text(CHAT))
# Manual and auto coexisting in one account + the label in the history
paper.open_position(CHAT, "BTC", "long", 100)   # Manual, next to the auto position
check("هم‌زیستی دستی و خودکار در یک حساب",
      {p.get("source") for p in paper.list_positions(CHAT)} == {"manual", "auto"})
PRICES["BTCUSDT"] = 130.0
paper.evaluate(CHAT)   # Both are closed by target (the manual one has no target → the manual one is not closed)
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
