#!/usr/bin/env python3
"""
tests/test_ev_layer.py
Offline test of the EV decision layer — no internet:

    python tests/test_ev_layer.py

Coverage:
1) EV math against hand-computed values (win rate, average win/loss, expected
   value, profit factor, break-even win rate) + the no-loss and all-loss edges,
   aggregate input, and sample confidence boundaries.
2) All branches of the decision rules (decide_candidate): strong, buy/sell,
   mandatory rejection of non-positive EV even with a score of 95, insufficient
   sample, absent EV, absent stop loss/target, and the two WAIT cases.
3) The cache-reading path from a temporary file (no computation, no network).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ev as ev_mod

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def approx(a, b, tol=1e-3):
    return a is not None and b is not None and abs(a - b) <= tol


# --- 1) EV math with a hand-computed answer ---
# R values: wins 2, 1, 0.5 (sum 3.5, mean 1.1667) — losses -1, -1, -0.5
# (sum -2.5, absolute mean 0.8333) — win rate 0.5
# Expected value = (3.5-2.5)/6 = 0.1667 | profit factor = 3.5/2.5 = 1.4
# Break-even = 0.8333 / (1.1667+0.8333) = 0.4167
e = ev_mod.compute_ev([2.0, 1.0, -1.0, -1.0, 0.5, -0.5])
check("نرخ برد", approx(e["win_rate"], 0.5), str(e["win_rate"]))
check("میانگین برد", approx(e["avg_win_r"], 1.1667), str(e["avg_win_r"]))
check("میانگین باخت (قدرمطلق)", approx(e["avg_loss_r"], 0.8333), str(e["avg_loss_r"]))
check("امید ریاضی", approx(e["expectancy_r"], 0.1667), str(e["expectancy_r"]))
check("پروفیت فاکتور", approx(e["profit_factor"], 1.4), str(e["profit_factor"]))
check("نرخ برد سربه‌سر", approx(e["break_even_win_rate"], 0.4167),
      str(e["break_even_win_rate"]))
check("تعداد معاملات", e["n_trades"] == 6)
check("نمونه‌ی ۶تایی = ناکافی", e["reliability"] == "insufficient")

# Trade-dict input (the same form as the backtest log)
e2 = ev_mod.compute_ev({"trades": [{"r": 1.0}, {"r": -1.0}]})
check("ورودی trades: امید ریاضی صفر", approx(e2["expectancy_r"], 0.0))
check("ورودی trades: سربه‌سر ۰٫۵", approx(e2["break_even_win_rate"], 0.5))

# Edge: no losses → infinite profit factor, break-even zero
e3 = ev_mod.compute_ev([1.0, 2.0, 3.0])
check("بدون باخت: PF بی‌نهایت", e3["profit_factor"] == float("inf"))
check("بدون باخت: میانگین باخت صفر", e3["avg_loss_r"] == 0.0)
check("بدون باخت: سربه‌سر صفر", e3["break_even_win_rate"] == 0.0)
check("بدون باخت: امید ریاضی ۲", approx(e3["expectancy_r"], 2.0))

# Edge: all losses → PF zero, break-even one
e4 = ev_mod.compute_ev([-1.0, -2.0])
check("همه باخت: PF صفر", e4["profit_factor"] == 0.0)
check("همه باخت: سربه‌سر یک", e4["break_even_win_rate"] == 1.0)
check("همه باخت: نرخ برد صفر", e4["win_rate"] == 0.0)

# Empty
e5 = ev_mod.compute_ev([])
check("خالی: n صفر و EV تهی", e5["n_trades"] == 0 and e5["expectancy_r"] is None)

# Ready aggregate input: 60% win, average win 2, loss 1 → EV = 1.2-0.4 = 0.8
e6 = ev_mod.compute_ev({"win_rate": 0.6, "avg_win_r": 2.0,
                        "avg_loss_r": 1.0, "n_trades": 50})
check("تجمیعی: امید ریاضی", approx(e6["expectancy_r"], 0.8), str(e6["expectancy_r"]))
check("تجمیعی: سربه‌سر ۱/۳", approx(e6["break_even_win_rate"], 1 / 3, 1e-3))
check("تجمیعی: PF", approx(e6["profit_factor"], 3.0), str(e6["profit_factor"]))
check("تجمیعی: نمونه ۵۰ = قابل اتکا", e6["reliability"] == "ok")

# Sample confidence boundaries: <10 insufficient, 10..29 low, >=30 reliable
check("مرز اطمینان: ۹ ناکافی",
      ev_mod.compute_ev([0.1] * 9)["reliability"] == "insufficient")
check("مرز اطمینان: ۱۰ کم",
      ev_mod.compute_ev([0.1] * 10)["reliability"] == "low")
check("مرز اطمینان: ۲۹ کم",
      ev_mod.compute_ev([0.1] * 29)["reliability"] == "low")
check("مرز اطمینان: ۳۰ قابل اتکا",
      ev_mod.compute_ev([0.1] * 30)["reliability"] == "ok")

# --- 2) Decision rules ---
def cand(score=70.0, direction="long", stop=90.0, target=120.0):
    return {"symbol": "TEST", "timeframe": "4h", "direction": direction,
            "score": score, "stop": stop, "target": target}


def ev(exp=0.3, n=40, rel="ok"):
    return {"expectancy_r": exp, "n_trades": n, "reliability": rel}


d = ev_mod.decide_candidate(cand(85, "long"), ev())
check("قوی لانگ", d["decision"] == "STRONG_BUY")
d = ev_mod.decide_candidate(cand(85, "short"), ev())
check("قوی شورت", d["decision"] == "STRONG_SELL")
d = ev_mod.decide_candidate(cand(65, "long"), ev())
check("خرید", d["decision"] == "BUY")
d = ev_mod.decide_candidate(cand(65, "short"), ev())
check("فروش", d["decision"] == "SELL")
d = ev_mod.decide_candidate(cand(95, "long"), ev(exp=-0.5))
check("رد اجباری: EV منفی با امتیاز ۹۵", d["decision"] == "NO_TRADE"
      and "رد اجباری" in d["decision_reason"])
d = ev_mod.decide_candidate(cand(95, "long"), ev(exp=0.0))
check("رد اجباری: EV صفر", d["decision"] == "NO_TRADE")
d = ev_mod.decide_candidate(cand(90, "long"), ev(n=5, rel="insufficient"))
check("نمونه ناکافی → NO_TRADE", d["decision"] == "NO_TRADE")
d = ev_mod.decide_candidate(cand(90, "long"), None)
check("EV غایب → NO_TRADE", d["decision"] == "NO_TRADE")
d = ev_mod.decide_candidate(cand(90, "long", stop=None), ev())
check("حد ضرر غایب → NO_TRADE", d["decision"] == "NO_TRADE")
d = ev_mod.decide_candidate(cand(90, "long", target=None), ev())
check("هدف غایب → NO_TRADE", d["decision"] == "NO_TRADE")
d = ev_mod.decide_candidate(cand(70, "long"), ev(n=15, rel="low"))
check("نمونه کم + امتیاز کافی → WAIT", d["decision"] == "WAIT")
d = ev_mod.decide_candidate(cand(50, "long"), ev())
check("امتیاز زیر آستانه + EV مثبت → WAIT", d["decision"] == "WAIT")
d = ev_mod.decide_candidate(cand(85, "long"), ev(n=15, rel="low"))
check("امتیاز قوی ولی نمونه کم → WAIT (نه STRONG)", d["decision"] == "WAIT")
check("برچسب فارسی تصمیم", ev_mod.decide_candidate(cand(), ev())["decision_fa"] == "خرید")

# --- 3) Cache path from a temporary file ---
tmp = tempfile.mkdtemp()
cache_path = os.path.join(tmp, "ev_cache.json")
entry = {"symbol": "TESTX", "timeframe": "4h", "threshold": 60,
         "source": "test", "computed_at": "2026-01-01T00:00:00+00:00",
         "split_at": None,
         "r_values_oos": [1.0, 2.0, -1.0] * 10, "n_trades": 30}
with open(cache_path, "w", encoding="utf-8") as fh:
    json.dump({"version": 1, "entries": {"TESTX_4h": entry}}, fh)
_orig_cache_file = ev_mod.CACHE_FILE
ev_mod.CACHE_FILE = cache_path
ev_mod._MEM_CACHE.clear()
try:
    got = ev_mod.ev_for_candidate("TESTX", "4h", allow_compute=False)
finally:
    ev_mod.CACHE_FILE = _orig_cache_file
    ev_mod._MEM_CACHE.clear()
# 30 trades: 20 wins (mean 1.5) and 10 losses (-1) → EV = (30-10)/30 = 0.6667
check("کش: خوانده شد", got is not None)
check("کش: امید ریاضی از R های کش", got is not None and approx(got["expectancy_r"], 0.6667),
      str(got and got["expectancy_r"]))
check("کش: n و اطمینان", got is not None and got["n_trades"] == 30
      and got["reliability"] == "ok")
check("کش: جهت‌خاص نیست", got is not None and got["direction_specific"] is False)
missing = ev_mod.ev_for_candidate("NOPE", "4h", allow_compute=False)
check("کش: ترکیب غایب + بدون محاسبه = None", missing is None)

print()
if fails:
    print(f"❌ {len(fails)} تست ناموفق: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های لایه‌ی EV موفق بود.\n")
