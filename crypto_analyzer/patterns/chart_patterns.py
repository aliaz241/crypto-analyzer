"""
patterns/chart_patterns.py
تشخیص الگوهای نموداری (هندسی) بر پایه‌ی swing high/low.

برخلاف الگوهای کندلی که رویدادهای تک‌کندلی هستند، این الگوها چند کندل را
در بر می‌گیرند. هر الگوی شناسایی‌شده به صورت دیکشنری با کلیدهای زیر برگردانده
می‌شود تا هم برای گزارش فعلی و هم برای بک‌تست قابل استفاده باشد:
    name, fa, bias, start_index, completion_index, neckline, target, pattern_price

⚠️ نسخه‌ی ۲ — تأیید خط گردن (رفع مشکل ممیزی شماره‌ی ۲): یک شکل‌گیری (سقف/کف
دوقلو یا سر و شانه) فقط وقتی «کامل‌شده» محسوب می‌شود که قیمت بعد از نقطه‌ی
سوئینگ دوم با بسته‌شدن کندل، خط گردن را در جهت bias الگو بشکند.
completion_index کندل تأیید (شکست) است، نه کندل سوئینگ. شکل‌گیریِ تأییدنشده
نه در گزارش الگوهای اخیر می‌آید و نه در نمونه‌های بک‌تست شمرده می‌شود.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.levels import find_swing_points


# ---------------------------------------------------------------------------
# Swing sequence helper
# ---------------------------------------------------------------------------
def get_alternating_swings(df: pd.DataFrame, lookback: int = None) -> list:
    """
    لیستی زیگزاگی از swing point ها برمی‌گرداند که بین high و low تناوب دارند.
    اگر دو swing پشت‌سرهم از یک نوع باشند، فقط شدیدترین (بالاترین/پایین‌ترین) حفظ می‌شود.
    خروجی: لیستی از دیکشنری {"pos": موقعیت عددی در df, "index": ایندکس df, "price": قیمت, "type": "high"/"low"}
    """
    swing_df = find_swing_points(df, lookback)
    points = []
    for pos in range(len(swing_df)):
        if swing_df["is_swing_high"].iloc[pos]:
            points.append({"pos": pos, "index": swing_df.index[pos],
                            "price": swing_df["high"].iloc[pos], "type": "high"})
        elif swing_df["is_swing_low"].iloc[pos]:
            points.append({"pos": pos, "index": swing_df.index[pos],
                            "price": swing_df["low"].iloc[pos], "type": "low"})

    if not points:
        return []

    alternating = [points[0]]
    for p in points[1:]:
        last = alternating[-1]
        if p["type"] == last["type"]:
            # دو swing هم‌نوع پشت سر هم — فقط شدیدترین را نگه دار
            if p["type"] == "high" and p["price"] > last["price"]:
                alternating[-1] = p
            elif p["type"] == "low" and p["price"] < last["price"]:
                alternating[-1] = p
        else:
            alternating.append(p)
    return alternating


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / b * 100


def _find_neckline_confirmation(df: pd.DataFrame, from_pos: int, neckline: float,
                                invalidation: float, direction: str):
    """
    تأیید شکست خط گردن (رفع مشکل ممیزی شماره‌ی ۲).
    از کندل بعد از نقطه‌ی سوئینگ دوم (from_pos) به جلو می‌رود:
    - جهت "above" (الگوهای صعودی): اولین کندلی که close آن بالای خط گردن بسته شود.
    - جهت "below" (الگوهای نزولی): اولین کندلی که close آن پایین خط گردن بسته شود.
    اگر قبل از شکست خط گردن، قیمت از سطح بطلان عبور کند (close آن‌طرف سطح
    بطلان)، شکل‌گیری مرده است و None برمی‌گردد؛ اگر تا انتهای داده هم شکست رخ
    ندهد باز None برمی‌گردد — یعنی الگو هنوز «کامل‌شده» نیست.
    خروجی: موقعیت عددی کندل تأیید (pos) یا None.
    """
    closes = df["close"].values
    for pos in range(from_pos + 1, len(df)):
        c = closes[pos]
        if direction == "above":
            if c < invalidation:
                return None  # قبل از تأیید، الگو باطل شد
            if c > neckline:
                return pos
        else:  # below
            if c > invalidation:
                return None  # قبل از تأیید، الگو باطل شد
            if c < neckline:
                return pos
    return None


# ---------------------------------------------------------------------------
# Double Top / Double Bottom
# ---------------------------------------------------------------------------
def detect_double_top(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 2):
        a, neck, b = swings[i], swings[i + 1], swings[i + 2]
        if a["type"] == "high" and neck["type"] == "low" and b["type"] == "high":
            if _pct_diff(a["price"], b["price"]) <= tol:
                height = a["price"] - neck["price"]
                if height <= 0:
                    continue
                invalidation = max(a["price"], b["price"])
                # الگو فقط با شکست رو به پایین خط گردن بعد از قله‌ی دوم کامل می‌شود
                conf_pos = _find_neckline_confirmation(
                    df, b["pos"], neck["price"], invalidation, "below")
                if conf_pos is None:
                    continue
                found.append({
                    "name": "double_top", "fa": "سقف دوقلو", "bias": "bearish",
                    "start_index": a["index"], "completion_index": df.index[conf_pos],
                    "formation_index": b["index"],
                    "neckline": neck["price"], "pattern_price": a["price"],
                    "invalidation": invalidation,
                    "target": neck["price"] - height,
                })
    return found


def detect_double_bottom(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 2):
        a, neck, b = swings[i], swings[i + 1], swings[i + 2]
        if a["type"] == "low" and neck["type"] == "high" and b["type"] == "low":
            if _pct_diff(a["price"], b["price"]) <= tol:
                height = neck["price"] - a["price"]
                if height <= 0:
                    continue
                invalidation = min(a["price"], b["price"])
                # الگو فقط با شکست رو به بالای خط گردن بعد از کف دوم کامل می‌شود
                conf_pos = _find_neckline_confirmation(
                    df, b["pos"], neck["price"], invalidation, "above")
                if conf_pos is None:
                    continue
                found.append({
                    "name": "double_bottom", "fa": "کف دوقلو", "bias": "bullish",
                    "start_index": a["index"], "completion_index": df.index[conf_pos],
                    "formation_index": b["index"],
                    "neckline": neck["price"], "pattern_price": a["price"],
                    "invalidation": invalidation,
                    "target": neck["price"] + height,
                })
    return found


# ---------------------------------------------------------------------------
# Head & Shoulders / Inverse Head & Shoulders
# ---------------------------------------------------------------------------
def detect_head_and_shoulders(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 4):
        ls, n1, head, n2, rs = swings[i:i + 5]
        types = [ls["type"], n1["type"], head["type"], n2["type"], rs["type"]]
        if types != ["high", "low", "high", "low", "high"]:
            continue
        if head["price"] <= ls["price"] or head["price"] <= rs["price"]:
            continue
        if _pct_diff(ls["price"], rs["price"]) > tol * 2:
            continue
        neckline = (n1["price"] + n2["price"]) / 2
        height = head["price"] - neckline
        if height <= 0:
            continue
        # الگو فقط با شکست رو به پایین خط گردن بعد از شانه‌ی راست کامل می‌شود
        conf_pos = _find_neckline_confirmation(
            df, rs["pos"], neckline, head["price"], "below")
        if conf_pos is None:
            continue
        found.append({
            "name": "head_and_shoulders", "fa": "سر و شانه", "bias": "bearish",
            "start_index": ls["index"], "completion_index": df.index[conf_pos],
            "formation_index": rs["index"],
            "neckline": neckline, "pattern_price": head["price"],
            "invalidation": head["price"],
            "target": neckline - height,
        })
    return found


def detect_inverse_head_and_shoulders(df: pd.DataFrame, swings: list) -> list:
    found = []
    tol = config.LEVEL_EQUALITY_TOLERANCE_PCT
    for i in range(len(swings) - 4):
        ls, n1, head, n2, rs = swings[i:i + 5]
        types = [ls["type"], n1["type"], head["type"], n2["type"], rs["type"]]
        if types != ["low", "high", "low", "high", "low"]:
            continue
        if head["price"] >= ls["price"] or head["price"] >= rs["price"]:
            continue
        if _pct_diff(ls["price"], rs["price"]) > tol * 2:
            continue
        neckline = (n1["price"] + n2["price"]) / 2
        height = neckline - head["price"]
        if height <= 0:
            continue
        # الگو فقط با شکست رو به بالای خط گردن بعد از شانه‌ی راست کامل می‌شود
        conf_pos = _find_neckline_confirmation(
            df, rs["pos"], neckline, head["price"], "above")
        if conf_pos is None:
            continue
        found.append({
            "name": "inverse_head_and_shoulders", "fa": "سر و شانه‌ی معکوس", "bias": "bullish",
            "start_index": ls["index"], "completion_index": df.index[conf_pos],
            "formation_index": rs["index"],
            "neckline": neckline, "pattern_price": head["price"],
            "invalidation": head["price"],
            "target": neckline + height,
        })
    return found


# ---------------------------------------------------------------------------
# Triangles & Wedges (بر پایه‌ی رگرسیون خطی روی swing ها در یک پنجره‌ی لغزان)
# ---------------------------------------------------------------------------
FLAT_SLOPE_THRESHOLD_PCT = 0.05   # شیب کمتر از این (درصد قیمت در هر کندل) = "تقریباً تخت"


def _fit_line(points: list):
    """points: لیستی از (pos, price). خروجی: (slope, intercept) یا None اگر کافی نبود."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


def _scan_triangle_wedge_windows(df: pd.DataFrame, window: int = 40, step: int = 10) -> list:
    found = []
    swing_df = find_swing_points(df)
    n = len(df)

    starts = list(range(0, n - window, step))
    # مطمئن شو آخرین پنجره دقیقاً به انتهای داده می‌رسد تا کندل‌های اخیر
    # به خاطر گام ثابت از قلم نیفتند
    if n - window >= 0 and (not starts or starts[-1] != n - window):
        starts.append(n - window)

    for start in starts:
        end = start + window
        segment = swing_df.iloc[start:end]
        avg_price = segment["close"].mean()
        if avg_price == 0 or np.isnan(avg_price):
            continue

        highs = [(i, segment["high"].iloc[i - start]) for i in range(start, end)
                  if segment["is_swing_high"].iloc[i - start]]
        lows = [(i, segment["low"].iloc[i - start]) for i in range(start, end)
                 if segment["is_swing_low"].iloc[i - start]]

        if len(highs) < 2 or len(lows) < 2:
            continue

        upper_fit = _fit_line(highs)
        lower_fit = _fit_line(lows)
        if upper_fit is None or lower_fit is None:
            continue

        upper_slope_pct = upper_fit[0] / avg_price * 100
        lower_slope_pct = lower_fit[0] / avg_price * 100

        upper_flat = abs(upper_slope_pct) <= FLAT_SLOPE_THRESHOLD_PCT
        lower_flat = abs(lower_slope_pct) <= FLAT_SLOPE_THRESHOLD_PCT

        completion_index = df.index[end - 1]
        pattern_high = max(p[1] for p in highs)
        pattern_low = min(p[1] for p in lows)
        height = pattern_high - pattern_low
        if height <= 0:
            continue

        last_close = df["close"].iloc[end - 1]

        # invalidation = سطحی که اگر قیمت به آن برگردد، فرضیه‌ی الگو باطل می‌شود
        pattern = None
        if upper_flat and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT:
            pattern = {
                "name": "ascending_triangle", "fa": "مثلث صعودی", "bias": "bullish",
                "target": pattern_high + height, "invalidation": pattern_low,
            }
        elif lower_flat and upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT:
            pattern = {
                "name": "descending_triangle", "fa": "مثلث نزولی", "bias": "bearish",
                "target": pattern_low - height, "invalidation": pattern_high,
            }
        elif upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT:
            # مثلث متقارن ذاتاً بدون جهت است؛ موقعیت قیمت فعلی نسبت به وسط مثلث
            # به عنوان یک «تمایل» عملیاتی در نظر گرفته می‌شود، نه قطعیت کلاسیک.
            leaning_bullish = last_close >= (pattern_high + pattern_low) / 2
            pattern = {
                "name": "symmetrical_triangle",
                "fa": "مثلث متقارن (تمایل صعودی)" if leaning_bullish else "مثلث متقارن (تمایل نزولی)",
                "bias": "bullish" if leaning_bullish else "bearish",
                "target": last_close + height if leaning_bullish else last_close - height,
                "invalidation": pattern_low if leaning_bullish else pattern_high,
            }
        elif upper_slope_pct > FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct > FLAT_SLOPE_THRESHOLD_PCT \
                and lower_slope_pct > upper_slope_pct:
            pattern = {
                "name": "rising_wedge", "fa": "گوه‌ی صعودی", "bias": "bearish",
                "target": pattern_low - height, "invalidation": pattern_high,
            }
        elif upper_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT and lower_slope_pct < -FLAT_SLOPE_THRESHOLD_PCT \
                and upper_slope_pct > lower_slope_pct:
            pattern = {
                "name": "falling_wedge", "fa": "گوه‌ی نزولی", "bias": "bullish",
                "target": pattern_high + height, "invalidation": pattern_low,
            }

        if pattern:
            # تأیید شکست مرز (رفع مشکل ممیزی شماره‌ی ۲ برای مثلث‌ها و گوه‌ها):
            # اسکنر پنجره‌ای خطوط مرز را با رگرسیون روی سوئینگ‌ها می‌سازد و نقطه‌ی
            # «تکمیل» ذاتاً انتهای همان پنجره است؛ پس شکستِ چندکندلیِ جداگانه مثل
            # الگوهای سوئینگی قابل ردیابی تمیز نیست. به‌عنوان حداقل شرط، close
            # کندل آخر پنجره باید از خط مرزیِ نزدیک‌تر در جهت bias الگو عبور
            # کرده باشد (صعودی: بالای خط بالایی / نزولی: پایین خط پایینی).
            # بدون این عبور، شکل‌گیری فقط یک تراکم قیمتی است و «الگوی کامل‌شده»
            # محسوب نمی‌شود — نه در گزارش می‌آید و نه در بک‌تست شمرده می‌شود.
            end_pos = end - 1
            upper_at_end = upper_fit[0] * end_pos + upper_fit[1]
            lower_at_end = lower_fit[0] * end_pos + lower_fit[1]
            if pattern["bias"] == "bullish" and not (last_close > upper_at_end):
                pattern = None
            elif pattern["bias"] == "bearish" and not (last_close < lower_at_end):
                pattern = None

        if pattern:
            pattern.update({
                "start_index": df.index[start],
                "completion_index": completion_index,
                "neckline": pattern_high if pattern["bias"] == "bearish" else pattern_low,
                "pattern_price": pattern["invalidation"],
            })
            found.append(pattern)

    # حذف تشخیص‌های همپوشان: وقتی پنجره‌های پیاپی همان نوع الگو را نشان می‌دهند
    # (چون پنجره‌ها با گام کوچک‌تر از عرضشان جلو می‌روند)، فقط جدیدترین را نگه می‌داریم.
    deduped = []
    for p in found:
        if deduped and deduped[-1]["name"] == p["name"] and \
                (df.index.get_loc(p["completion_index"]) - df.index.get_loc(deduped[-1]["completion_index"])) < window:
            deduped[-1] = p
        else:
            deduped.append(p)

    return deduped


# ---------------------------------------------------------------------------
# Master scan
# ---------------------------------------------------------------------------
def scan_all(df: pd.DataFrame) -> list:
    """
    همه‌ی الگوهای نموداری را روی کل تاریخچه اسکن می‌کند.
    خروجی: لیستی از دیکشنری الگو (برای گزارش فعلی و برای بک‌تست قابل استفاده است).
    """
    swings = get_alternating_swings(df)
    results = []
    results += detect_double_top(df, swings)
    results += detect_double_bottom(df, swings)
    results += detect_head_and_shoulders(df, swings)
    results += detect_inverse_head_and_shoulders(df, swings)
    results += _scan_triangle_wedge_windows(df)

    results.sort(key=lambda p: df.index.get_loc(p["completion_index"]))
    return results


def latest_patterns(df: pd.DataFrame, recent_candles: int = 15) -> list:
    """فقط الگوهایی که completion_index آن‌ها در 'recent_candles' کندل آخر است."""
    all_patterns = scan_all(df)
    cutoff_pos = len(df) - recent_candles
    return [p for p in all_patterns if df.index.get_loc(p["completion_index"]) >= cutoff_pos]
