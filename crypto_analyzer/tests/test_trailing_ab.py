#!/usr/bin/env python3
"""
tests/test_trailing_ab.py
Offline test of the system backtest trailing stop — no internet:

    python tests/test_trailing_ab.py

Coverage (all with synthetic candles and a hand-computed answer):
1) Activation at A×R and profit locking with a distance of D×R (long).
2) Anti-lookahead: a stop moved by candle t is not applied to candle t itself.
3) Ratchet: the floating stop never loosens.
4) One-candle conflict (floating stop and target together) → stop first.
5) Short symmetry.
6) No activation → behavior exactly like the fixed stop; and with no trailing
   parameter → the engine's previous output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from backtest.system_backtest import _simulate_trade
import config

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def df_of(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


TR = {"activation_r": 0.5, "trail_r": 1.0}
# Common long basis: entry 100, stop 90, target 120 → R=10
# Activation 0.5R → price 105 ; distance 1R → 10 units behind the best price

# --- 1) Activation and profit lock ---
# Candle 1: high 105 (activation) → floating stop 95 from candle 2
# Candle 2: high 112 → floating stop 102 from candle 3
# Candle 3: low 101 ≤ 102 → trailing exit at 102
df = df_of([(100, 100, 100, 100),
            (100, 105, 102, 104),
            (104, 112, 103, 110),
            (110, 111, 101, 105)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 10, trailing=TR)
check("لانگ: خروج تریلینگ روی استاپ جابه‌جاشده", outcome == "trailing" and exit_price == 102.0,
      f"{outcome} @ {exit_price}")

# --- 2) Anti-lookahead ---
# Candle 1 high 106 (active) and low 94: the hypothetical floating stop 96 must not be applied
# to candle 1 itself (low 94 is below 96 but above the initial stop 90) →
# the trade stays alive and times out at the end of the window.
df = df_of([(100, 100, 100, 100),
            (100, 106, 94, 100),
            (100, 101, 99, 100)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 2, trailing=TR)
check("ضد نشت: استاپ کندل t روی خودش اعمال نمی‌شود", outcome == "timeout",
      f"{outcome} @ {exit_price}")

# --- 3) Ratchet only forward ---
# Candle 1: high 106 → stop 96. Candle 2: a lower high (103) must not loosen
# the stop (93); low 95 ≤ 96 → exit at 96.
df = df_of([(100, 100, 100, 100),
            (100, 106, 100, 105),
            (105, 103, 95, 100)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("رچت: استاپ شناور شل نمی‌شود", outcome == "trailing" and exit_price == 96.0,
      f"{outcome} @ {exit_price}")

# --- 4) Conflict in one candle: stop first ---
# Candle 1: high 112 (below target 120) → stop 102. Candle 2: both the target (121) and
# the floating stop (low 100) are touched → conservatively the stop wins.
df = df_of([(100, 100, 100, 100),
            (100, 112, 104, 110),
            (110, 121, 100, 115)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("تعارض هم‌زمان: استاپ شناور اول", outcome == "trailing" and exit_price == 102.0,
      f"{outcome} @ {exit_price}")

# --- 5) Short symmetry ---
# Entry 100, stop 110, target 80 → R=10; activation at low 95, distance 10
# Candle 1: low 95 → floating stop 105 from candle 2
# Candle 2: low 88 → floating stop 98 from candle 3
# Candle 3: high 99 ≥ 98 → trailing exit at 98 (locked profit of +0.2R before costs)
df = df_of([(100, 100, 100, 100),
            (100, 98, 95, 96),
            (96, 97, 88, 90),
            (90, 99, 89, 95)])
exit_price, outcome = _simulate_trade(df, 0, "short", 100, 110, 80, 10, trailing=TR)
check("شورت: تقارن تریلینگ", outcome == "trailing" and exit_price == 98.0,
      f"{outcome} @ {exit_price}")

# --- 6) No activation / no trailing ---
# The price never reaches 105 → the fixed stop 90 works as before.
df = df_of([(100, 100, 100, 100),
            (100, 103, 95, 100),
            (100, 102, 89, 95)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("بدون فعال‌سازی: استاپ ثابت", outcome == "stop" and exit_price == 90,
      f"{outcome} @ {exit_price}")
# trailing=None on the same data as test 1 → no trailing, timeout with the last candle's close
df = df_of([(100, 100, 100, 100),
            (100, 105, 102, 104),
            (104, 112, 103, 110),
            (110, 111, 101, 105)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 10)
check("trailing=None: رفتار قبلی موتور", outcome == "timeout" and exit_price == 105.0,
      f"{outcome} @ {exit_price}")

# --- 7) Tighter distance (0.5R) ---
# Activation 1R → 110; candle 1 high 112 → stop 107; candle 2 low 106 → exit 107
df = df_of([(100, 100, 100, 100),
            (100, 112, 104, 110),
            (110, 111, 106, 108)])
exit_price, outcome = _simulate_trade(
    df, 0, "long", 100, 90, 120, 5,
    trailing={"activation_r": 1.0, "trail_r": 0.5})
check("واریانت تنگ: فعال‌سازی ۱R و فاصله ۰٫۵R", outcome == "trailing" and exit_price == 107.0,
      f"{outcome} @ {exit_price}")

# --- 8) Paper alignment with the accepted A/B parameter ---
# An auto position with a stop loss and no explicit trailing_pct must take the
# accepted R-based distance as an equivalent percent:
# 0.5 × |100 − 90| / 100 × 100 = 5% — and a manual position must get nothing.
check("ثابت پذیرفته‌شده‌ی A/B تنظیم است",
      config.SYSBT_TRAILING_ADOPTED == {"activation_r": 1.0, "trail_r": 0.5},
      str(config.SYSBT_TRAILING_ADOPTED))
import tempfile
import paper_trading as paper

paper.STATE_FILE = os.path.join(tempfile.mkdtemp(), "paper_state_trail.json")
paper.set_price_function(lambda sym: 100.0)
CHAT_T = "trail-align-chat"
paper.reset_account(CHAT_T)
pos_auto = paper.open_position(CHAT_T, "BTC", "long", 1000, stop=90, target=120,
                               source="auto")
check("خودکار: تریلینگ هم‌تراز A/B (۵٪)", pos_auto.get("trailing_pct") == 5.0,
      f"trailing_pct={pos_auto.get('trailing_pct')}")
pos_man = paper.open_position(CHAT_T, "BTC", "long", 100, stop=90, target=120)
check("دستی: بدون تریلینگ خودکار", pos_man.get("trailing_pct") is None,
      f"trailing_pct={pos_man.get('trailing_pct')}")
pos_auto2 = paper.open_position(CHAT_T, "ETH", "short", 1000, stop=110,
                                trailing_pct=3, source="auto")
check("خودکار با تریلینگ صریح: همان می‌ماند", pos_auto2.get("trailing_pct") == 3.0,
      f"trailing_pct={pos_auto2.get('trailing_pct')}")

print()
if fails:
    print(f"❌ {len(fails)} تست ناموفق: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های تریلینگ‌استاپ (A/B) موفق بود.\n")
