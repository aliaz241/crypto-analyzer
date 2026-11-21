#!/usr/bin/env python3
"""
tests/test_trailing_ab.py
تست آفلاین تریلینگ‌استاپ بک‌تست سیستمی — بدون اینترنت:

    python tests/test_trailing_ab.py

پوشش (همه با کندل مصنوعی و جواب دستی):
۱) فعال‌سازی در A×R و قفل سود با فاصله‌ی D×R (لانگ).
۲) ضد نشت آینده: استاپ جابه‌جاشده با کندل t روی خود کندل t اعمال نمی‌شود.
۳) رچت: استاپ شناور هرگز شل نمی‌شود.
۴) تعارض یک کندل (استاپ شناور و هدف با هم) → استاپ اول.
۵) تقارن شورت.
۶) بدون فعال‌سازی → رفتار دقیقاً مثل استاپ ثابت؛ و بدون پارامتر
   trailing → همان خروجی قبلی موتور.
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
# مبنای مشترک لانگ: ورود ۱۰۰، استاپ ۹۰، هدف ۱۲۰ → R=۱۰
# فعال‌سازی ۰٫۵R → قیمت ۱۰۵ ؛ فاصله ۱R → ۱۰ واحد پشت بهترین قیمت

# --- ۱) فعال‌سازی و قفل سود --------------------------------------------------
# کندل۱: سقف ۱۰۵ (فعال‌سازی) → استاپ شناور ۹۵ از کندل۲
# کندل۲: سقف ۱۱۲ → استاپ شناور ۱۰۲ از کندل۳
# کندل۳: کف ۱۰۱ ≤ ۱۰۲ → خروج trailing روی ۱۰۲
df = df_of([(100, 100, 100, 100),
            (100, 105, 102, 104),
            (104, 112, 103, 110),
            (110, 111, 101, 105)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 10, trailing=TR)
check("لانگ: خروج تریلینگ روی استاپ جابه‌جاشده", outcome == "trailing" and exit_price == 102.0,
      f"{outcome} @ {exit_price}")

# --- ۲) ضد نشت آینده ----------------------------------------------------------
# کندل۱ سقف ۱۰۶ (فعال) و کف ۹۴: استاپ شناورِ فرضی ۹۶ نباید روی خود کندل۱
# اعمال شود (کف ۹۴ از ۹۶ پایین‌تر است ولی از استاپ اولیه ۹۰ بالاتر) →
# معامله زنده می‌ماند و در پایان پنجره timeout می‌شود.
df = df_of([(100, 100, 100, 100),
            (100, 106, 94, 100),
            (100, 101, 99, 100)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 2, trailing=TR)
check("ضد نشت: استاپ کندل t روی خودش اعمال نمی‌شود", outcome == "timeout",
      f"{outcome} @ {exit_price}")

# --- ۳) رچت فقط به جلو ---------------------------------------------------------
# کندل۱: سقف ۱۰۶ → استاپ ۹۶. کندل۲: سقف پایین‌تر (۱۰۳) نباید استاپ را
# شل کند (۹۳)؛ کف ۹۵ ≤ ۹۶ → خروج روی ۹۶.
df = df_of([(100, 100, 100, 100),
            (100, 106, 100, 105),
            (105, 103, 95, 100)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("رچت: استاپ شناور شل نمی‌شود", outcome == "trailing" and exit_price == 96.0,
      f"{outcome} @ {exit_price}")

# --- ۴) تعارض در یک کندل: استاپ اول ---------------------------------------------
# کندل۱: سقف ۱۱۲ (زیر هدف ۱۲۰) → استاپ ۱۰۲. کندل۲: هم هدف (۱۲۱) و هم
# استاپ شناور (کف ۱۰۰) لمس می‌شوند → محافظه‌کارانه استاپ برنده است.
df = df_of([(100, 100, 100, 100),
            (100, 112, 104, 110),
            (110, 121, 100, 115)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("تعارض هم‌زمان: استاپ شناور اول", outcome == "trailing" and exit_price == 102.0,
      f"{outcome} @ {exit_price}")

# --- ۵) تقارن شورت ---------------------------------------------------------------
# ورود ۱۰۰، استاپ ۱۱۰، هدف ۸۰ → R=۱۰؛ فعال‌سازی روی کف ۹۵، فاصله ۱۰
# کندل۱: کف ۹۵ → استاپ شناور ۱۰۵ از کندل۲
# کندل۲: کف ۸۸ → استاپ شناور ۹۸ از کندل۳
# کندل۳: سقف ۹۹ ≥ ۹۸ → خروج trailing روی ۹۸ (سود قفل‌شده‌ی ‎+۰٫۲R پیش از هزینه)
df = df_of([(100, 100, 100, 100),
            (100, 98, 95, 96),
            (96, 97, 88, 90),
            (90, 99, 89, 95)])
exit_price, outcome = _simulate_trade(df, 0, "short", 100, 110, 80, 10, trailing=TR)
check("شورت: تقارن تریلینگ", outcome == "trailing" and exit_price == 98.0,
      f"{outcome} @ {exit_price}")

# --- ۶) بدون فعال‌سازی / بدون تریلینگ ----------------------------------------------
# قیمت هرگز به ۱۰۵ نمی‌رسد → استاپ ثابت ۹۰ مثل قبل عمل می‌کند.
df = df_of([(100, 100, 100, 100),
            (100, 103, 95, 100),
            (100, 102, 89, 95)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 5, trailing=TR)
check("بدون فعال‌سازی: استاپ ثابت", outcome == "stop" and exit_price == 90,
      f"{outcome} @ {exit_price}")
# trailing=None روی همان داده‌ی تست ۱ → بدون تریلینگ، timeout با بسته‌شدن کندل آخر
df = df_of([(100, 100, 100, 100),
            (100, 105, 102, 104),
            (104, 112, 103, 110),
            (110, 111, 101, 105)])
exit_price, outcome = _simulate_trade(df, 0, "long", 100, 90, 120, 10)
check("trailing=None: رفتار قبلی موتور", outcome == "timeout" and exit_price == 105.0,
      f"{outcome} @ {exit_price}")

# --- ۷) فاصله‌ی تنگ‌تر (۰٫۵R) -------------------------------------------------------
# فعال‌سازی ۱R → ۱۱۰؛ کندل۱ سقف ۱۱۲ → استاپ ۱۰۷؛ کندل۲ کف ۱۰۶ → خروج ۱۰۷
df = df_of([(100, 100, 100, 100),
            (100, 112, 104, 110),
            (110, 111, 106, 108)])
exit_price, outcome = _simulate_trade(
    df, 0, "long", 100, 90, 120, 5,
    trailing={"activation_r": 1.0, "trail_r": 0.5})
check("واریانت تنگ: فعال‌سازی ۱R و فاصله ۰٫۵R", outcome == "trailing" and exit_price == 107.0,
      f"{outcome} @ {exit_price}")

# --- ۸) هم‌ترازی پیپر با پارامتر پذیرفته‌شده‌ی A/B -----------------------------------
# پوزیشن خودکارِ دارای حد ضرر و بدون trailing_pct صریح باید فاصله‌ی
# R-محور پذیرفته‌شده را به‌صورت درصد معادل بگیرد:
# 0.5 × |۱۰۰ − ۹۰| / ۱۰۰ × ۱۰۰ = ۵٪ — و پوزیشن دستی نباید چیزی بگیرد.
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
