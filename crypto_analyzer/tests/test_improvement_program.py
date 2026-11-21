#!/usr/bin/env python3
"""
tests/test_improvement_program.py
Offline test of the improvement program wiring (steps 1 to 3) — no internet:

    python tests/test_improvement_program.py

Coverage:
1) Config constants of the accepted state: gate on with n_min=8 and a smooth
   edge, regime and MTF defaults on, and max_hold unchanged (step 3 was
   rejected → IMPROVE_SYSBT_MAX_HOLD=None and the engine constant stays 40).
2) The EV gate margin on the live path (scanner.attach_ev_decision): the
   additive ev_gate key with the exact semantics of step 1 — n≥8 and realized
   expectancy <0 → suppressed=True; n<8 → allowed (insufficient sample);
   expectancy ≥0 → allowed. And consistency with the mandatory decision:
   wherever the gate suppresses, decide_candidate also gives NO_TRADE.
3) Resolving scan_market defaults from config: with an offline monkeypatch of
   fetch and score_candidate — a call with no arguments must activate
   apply_regime=True and the MTF path, and the daily direction is computed
   without the regime; and an explicit off (False) must override the config.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
import scanner
from analysis import ev as ev_mod

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# --- 1) Accepted constants ---
check("دروازه: حالت on", config.IMPROVE_GATE_MODE == "on",
      str(config.IMPROVE_GATE_MODE))
check("دروازه: n_min = ۸", config.IMPROVE_GATE_N_MIN == 8)
check("دروازه: لبه‌ی smooth = −۰٫۲R", config.IMPROVE_GATE_SMOOTH_EDGE_R == -0.2)
check("رژیم: پیش‌فرض روشن", config.IMPROVE_REGIME_DEFAULT is True)
check("MTF: پیش‌فرض روشن", config.IMPROVE_MTF_DEFAULT is True)
check("مرحله ۳ رد شد: IMPROVE_SYSBT_MAX_HOLD خالی", config.IMPROVE_SYSBT_MAX_HOLD is None)
check("ثابت موتور max_hold همان ۴۰", config.SYSBT_MAX_HOLD_CANDLES == 40)

# --- 2) Gate margin in attach_ev_decision ---
def cand_with_ev(n, exp):
    return {"symbol": "XXX", "timeframe": "4h", "direction": "long",
            "score": 70.0, "entry": 100.0, "stop": 90.0, "target": 120.0,
            "ev": {"n_trades": n, "expectancy_r": exp,
                   "reliability": "low" if n < 30 else "ok"},
            "decision": "WAIT", "decision_fa": "—", "decision_reason": "—"}

c = scanner.attach_ev_decision(cand_with_ev(12, -0.30))
check("دروازه: n=۱۲ و امید −۰٫۳۰ → suppress", c["ev_gate"]["suppressed"] is True,
      str(c.get("ev_gate")))
check("دروازه: حالت و n_min در حاشیه ثبت است",
      c["ev_gate"]["mode"] == "on" and c["ev_gate"]["n_min"] == 8)
c = scanner.attach_ev_decision(cand_with_ev(12, 0.20))
check("دروازه: امید مثبت → اجازه", c["ev_gate"]["suppressed"] is False)
c = scanner.attach_ev_decision(cand_with_ev(5, -0.50))
check("دروازه: n<۸ → اجازه (سابقه ناکافی)", c["ev_gate"]["suppressed"] is False)
c = scanner.attach_ev_decision(cand_with_ev(8, -0.01))
check("دروازه: مرز n=۸ و امید منفی → suppress", c["ev_gate"]["suppressed"] is True)

# Consistency: every gate suppression must come with the EV layer's mandatory NO_TRADE
for n, exp in ((12, -0.30), (25, -0.05), (40, -1.20)):
    cc = cand_with_ev(n, exp)
    dec = ev_mod.decide_candidate(cc, cc["ev"])
    check(f"سازگاری تصمیم: n={n} exp={exp} → NO_TRADE",
          dec["decision"] == "NO_TRADE", dec["decision"])

# --- 3) Resolving scan_market defaults from config ---
calls = []

def fake_fetch(symbol, timeframe, limit=None):
    idx = pd.date_range("2026-01-01", periods=5, freq="4h", tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": 1.0}, index=idx)

def fake_score(df, report, symbol, timeframe, **kwargs):
    calls.append({"symbol": symbol, "timeframe": timeframe, "kwargs": kwargs})
    return {"symbol": symbol, "timeframe": timeframe, "direction": "long",
            "score": 50.0, "entry": 100.0, "stop": 90.0, "target": 120.0}

orig_fetch = scanner.analyzer.fetch_candles_binance
orig_build = scanner.analyzer.build_report
orig_score = scanner.score_candidate
scanner.analyzer.fetch_candles_binance = fake_fetch
scanner.analyzer.build_report = lambda df, s, t: {}
scanner.score_candidate = fake_score
try:
    calls.clear()
    out = scanner.scan_market(symbols=["BTC"], timeframes=["4h"],
                              progress=False, limit=5)
    main_calls = [c for c in calls if c["timeframe"] == "4h"]
    daily_calls = [c for c in calls if c["timeframe"] == "1d"]
    check("اسکن پیش‌فرض: apply_regime=True روی نامزد 4h",
          any(c["kwargs"].get("apply_regime") is True for c in main_calls),
          str(calls))
    check("اسکن پیش‌فرض: مسیر MTF فعال (فراخوانی دوم با جهت بالاتر)",
          any("higher_tf_direction" in c["kwargs"] for c in main_calls),
          str(calls))
    check("اسکن پیش‌فرض: جهت روزانه بدون رژیم (سمانتیک مرحله ۲)",
          daily_calls and all("apply_regime" not in c["kwargs"] for c in daily_calls),
          str(calls))
    check("اسکن پیش‌فرض: حاشیه‌ی ev_gate روی نامزد هست",
          bool(out["candidates"]) and "ev_gate" in out["candidates"][0])

    calls.clear()
    scanner.scan_market(symbols=["BTC"], timeframes=["4h"], progress=False,
                        limit=5, use_mtf=False, apply_regime=False)
    main_calls = [c for c in calls if c["timeframe"] == "4h"]
    check("خاموش صریح: apply_regime=False بر کانفیگ غلبه می‌کند",
          main_calls and all(c["kwargs"].get("apply_regime") is False
                             for c in main_calls), str(calls))
    check("خاموش صریح: هیچ فراخوانی MTF/روزانه نیست",
          not any("higher_tf_direction" in c["kwargs"] for c in calls)
          and not any(c["timeframe"] == "1d" for c in calls), str(calls))
finally:
    scanner.analyzer.fetch_candles_binance = orig_fetch
    scanner.analyzer.build_report = orig_build
    scanner.score_candidate = orig_score

print()
if fails:
    print(f"❌ {len(fails)} تست ناموفق: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های برنامه‌ی بهبود موفق بود.\n")
