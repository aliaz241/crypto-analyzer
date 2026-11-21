#!/usr/bin/env python3
"""
tests/test_ported_features.py
تست آفلاین چهار قابلیت پورت‌شده از پک خارجی — بدون اینترنت:

    python tests/test_ported_features.py

پوشش:
۱) تفسیر فاندینگ‌ریت و ربع‌های قیمت×بهره‌ی باز (همه‌ی شاخه‌ها) —
   data/futures_client.py
۲) همبستگی بازده آلت با BTC در برابر پیرسونِ دستی —
   scanner._btc_corr_from_frames
۳) تحلیل حذف تک‌تک مؤلفه‌ها (LOO) روی لاگ مصنوعی با جواب دستی —
   backtest/calibration.loo_analysis
۴) امتیاز واریانت وزن‌ها — backtest/system_backtest._variant_score
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import scanner
from backtest import calibration
from backtest.system_backtest import _variant_score
from data.futures_client import (interpret_funding_rate,
                                 interpret_oi_price_relationship)

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# --- ۱) تفسیر فاندینگ‌ریت (همه‌ی شاخه‌ها) -----------------------------------
check("فاندینگ بالا = ازدحام لانگ", interpret_funding_rate(0.07)["label"] == "crowded_long")
check("فاندینگ روی آستانه = ازدحام لانگ", interpret_funding_rate(0.05)["label"] == "crowded_long")
check("فاندینگ منفی بالا = ازدحام شورت", interpret_funding_rate(-0.07)["label"] == "crowded_short")
check("فاندینگ منفی روی آستانه = ازدحام شورت", interpret_funding_rate(-0.05)["label"] == "crowded_short")
check("فاندینگ عادی = خنثی", interpret_funding_rate(0.01)["label"] == "neutral")
check("فاندینگ صفر = خنثی", interpret_funding_rate(0.0)["label"] == "neutral")
check("آستانه‌ی سفارشی", interpret_funding_rate(0.03, threshold_pct=0.02)["label"] == "crowded_long")
check("برچسب فارسی فاندینگ", interpret_funding_rate(0.07)["fa"] == "ازدحام پوزیشن لانگ")

# --- ۱ب) ربع‌های قیمت × بهره‌ی باز ------------------------------------------
check("قیمت↑ OI↑ = روند صعودی سالم",
      interpret_oi_price_relationship(2.0, 3.0)["label"] == "healthy_uptrend")
check("قیمت↑ OI↓ = short covering",
      interpret_oi_price_relationship(2.0, -3.0)["label"] == "short_covering")
check("قیمت↓ OI↑ = روند نزولی سالم",
      interpret_oi_price_relationship(-2.0, 3.0)["label"] == "healthy_downtrend")
check("قیمت↓ OI↓ = خروج لانگ",
      interpret_oi_price_relationship(-2.0, -3.0)["label"] == "long_liquidation_or_exit")

# --- ۲) همبستگی در برابر پیرسون دستی -----------------------------------------
def closes_from_returns(rets, start=100.0):
    closes = [start]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    return pd.DataFrame({"close": closes})


def manual_pearson(xs, ys):
    """پیرسون با فرمول مجموع‌ها — مستقل از pandas/numpy."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den


btc_rets = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.03, 0.015, 0.0, -0.005] * 4
alt_same = closes_from_returns(btc_rets)          # بازده یکسان → همبستگی دقیقاً ۱
alt_opp = closes_from_returns([-r for r in btc_rets])  # بازده قرینه → دقیقاً ‎-۱
btc_df = closes_from_returns(btc_rets)

corr_same = scanner._btc_corr_from_frames(alt_same, btc_df, window=30)
check("همبستگی بازده یکسان = 1.0", approx(corr_same, 1.0, 1e-6), str(corr_same))
corr_opp = scanner._btc_corr_from_frames(alt_opp, btc_df, window=30)
check("همبستگی بازده قرینه = -1.0", approx(corr_opp, -1.0, 1e-6), str(corr_opp))

# حالت عمومی: مقایسه با پیرسون دستی روی ۳۰ بازده‌ی آخر
alt_mix_rets = [r * 0.5 + (0.004 if i % 3 == 0 else -0.002)
                for i, r in enumerate(btc_rets)]
alt_mix = closes_from_returns(alt_mix_rets)
corr_mix = scanner._btc_corr_from_frames(alt_mix, btc_df, window=30)
expected = manual_pearson(alt_mix_rets[-30:], btc_rets[-30:])
check("همبستگی عمومی = پیرسون دستی", approx(corr_mix, round(expected, 3), 1e-6),
      f"{corr_mix} vs {expected:.4f}")
check("داده‌ی ناکافی = None",
      scanner._btc_corr_from_frames(closes_from_returns([0.01] * 5), btc_df, window=30) is None)

# --- ۳) LOO روی لاگ مصنوعی با جواب دستی ---------------------------------------
# سه معامله‌ی out با آستانه‌ی ۵۵. صفرکردن «روند» (۲۵) هر سه را زیر آستانه
# می‌اندازد جز سومی؛ صفرکردن «حجم» (۰) هیچ‌کدام را حذف نمی‌کند.
def _trade(score, comps, r, segment="out"):
    return {"score": score, "components": comps, "r": r, "segment": segment,
            "outcome": "target" if r > 0 else "stop", "pattern": None}

_trades = [
    _trade(60.0, {"trend": 25.0, "momentum": 10.0, "pattern": 15.0,
                  "risk_reward": 10.0, "volume": 0.0}, 1.0),
    _trade(58.0, {"trend": 25.0, "momentum": 8.0, "pattern": 15.0,
                  "risk_reward": 10.0, "volume": 0.0}, -1.0),
    _trade(90.0, {"trend": 25.0, "momentum": 20.0, "pattern": 25.0,
                  "risk_reward": 10.0, "volume": 10.0}, 2.0),
]
_fake_result = {"symbol": "TEST", "timeframe": "4h",
                "thresholds": {"55": {"trades": _trades}}}
loo = calibration.loo_analysis(_fake_result, "55", "out")
# پایه: میانگین (1 - 1 + 2)/3 = 0.667
check("LOO پایه", approx(loo["base"]["expectancy_r"], 0.667, 1e-9),
      str(loo["base"]["expectancy_r"]))
# بدون روند: فقط معامله‌ی سوم می‌ماند (90-25=65) → امید ریاضی 2.0، دلتا +1.333
ct = loo["components"]["trend"]
check("LOO بدون روند: مانده‌ها", ct["kept_trades"] == 1)
check("LOO بدون روند: دلتا", approx(ct["delta_r"], 1.333, 1e-9), str(ct["delta_r"]))
# بدون حجم: هر سه می‌مانند (حجم دو معامله‌ی اول صفر است) → دلتا 0
cv = loo["components"]["volume"]
check("LOO بدون حجم: مانده‌ها", cv["kept_trades"] == 3)
check("LOO بدون حجم: دلتا صفر", approx(cv["delta_r"], 0.0, 1e-9), str(cv["delta_r"]))
check("متن فارسی LOO ساخته می‌شود", "LOO" in calibration.format_loo_fa(_fake_result, "55", "out"))

# --- ۴) امتیاز واریانت وزن‌ها --------------------------------------------------
_full = {"trend": 25.0, "momentum": 20.0, "pattern": 25.0,
         "risk_reward": 20.0, "volume": 10.0}
check("واریانت خالی = امتیاز پایه", approx(_variant_score(_full, {}), 100.0, 1e-9))
check("دوبرابرکردن سقف روند", approx(_variant_score(_full, {"trend": 50.0}), 125.0, 1e-9))
check("صفرکردن الگو", approx(_variant_score(_full, {"pattern": 0.0}), 75.0, 1e-9))
check("مقیاس نسبی نصف", approx(_variant_score({"trend": 12.5}, {"trend": 50.0}), 25.0, 1e-9))

print()
if fails:
    print(f"❌ {len(fails)} تست رد نشد: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های قابلیت‌های پورت‌شده رد شدند.")
