"""
backtest/calibration.py
گزارش کالیبراسیون — ابزار تحلیلی افزودنی (گام ۷). فقط روی خروجی
backtest/system_backtest کار می‌کند و هیچ محاسبه‌ی موجودی را تغییر نمی‌دهد.

هدف: نشان دهد روی داده‌ی تست‌شده، کدام بازه‌های امتیاز و کدام مؤلفه‌ها
واقعاً امید ریاضی مثبت داشته‌اند و کدام‌ها فقط «ظاهر» داشته‌اند:

- بر اساس باکت امتیاز: 55-60 | 60-70 | 70-80 | 80+
  (لاگ آستانه‌ی پایین‌تر، ابرمجموعه‌ی بقیه است؛ باکت هر معامله از
  score خودش خوانده می‌شود.)
- بر اساس حضور/قدرت هر مؤلفه: معامله‌ها به «مؤلفه قوی» (حداقل نصف سقفش)
  و «بقیه» تقسیم و امید ریاضی هر گروه جدا حساب می‌شود. برای مؤلفه‌ی الگو،
  «حضور الگو» (pattern نام‌دار) هم جدا گزارش می‌شود.

⚠️ این‌ها آمار همان دوره‌ی تاریخی تست‌شده‌اند؛ کالیبراسیون روی گذشته
تضمین نمی‌کند همان مؤلفه در آینده هم بهترین باشد (خطر اورفیت هست —
برای همین بخش out_sample را جدا هم می‌توان گرفت).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backtest.system_backtest import _metrics, run_system_backtest

SCORE_BUCKETS = [(55.0, 60.0, "55-60"), (60.0, 70.0, "60-70"),
                 (70.0, 80.0, "70-80"), (80.0, float("inf"), "80+")]

_COMPONENT_MAX = {
    "trend": config.SCORE_TREND_MAX,
    "momentum": config.SCORE_MOMENTUM_MAX,
    "pattern": config.SCORE_PATTERN_MAX,
    "risk_reward": config.SCORE_RR_MAX,
    "volume": config.SCORE_VOLUME_MAX,
}
_COMPONENT_FA = {
    "trend": "روند", "momentum": "مومنتوم", "pattern": "الگو",
    "risk_reward": "سودبه‌ریسک", "volume": "حجم",
}


def _pick_trades(result: dict, base_threshold: str = "55", segment: str = "all") -> list:
    """لاگ معامله‌های یک آستانه (پیش‌فرض پایین‌ترین = ابرمجموعه) + فیلتر بخش."""
    data = result["thresholds"][base_threshold]
    trades = data["trades"]
    if segment == "in":
        trades = [t for t in trades if t["segment"] == "in"]
    elif segment == "out":
        trades = [t for t in trades if t["segment"] == "out"]
    return trades


def calibrate(result: dict, base_threshold: str = "55", segment: str = "all") -> dict:
    """
    خروجی: {"buckets": {label: metrics}, "components": {name: {"strong": m, "rest": m}},
            "pattern_presence": {"with": m, "without": m}, "trades_used": n}
    """
    trades = _pick_trades(result, base_threshold, segment)

    buckets = {}
    for lo, hi, label in SCORE_BUCKETS:
        buckets[label] = _metrics([t for t in trades if lo <= t["score"] < hi])

    components = {}
    for name, mx in _COMPONENT_MAX.items():
        strong = [t for t in trades if (t["components"].get(name) or 0) >= 0.5 * mx]
        rest = [t for t in trades if (t["components"].get(name) or 0) < 0.5 * mx]
        components[name] = {"strong": _metrics(strong), "rest": _metrics(rest)}

    pattern_presence = {
        "with": _metrics([t for t in trades if t.get("pattern")]),
        "without": _metrics([t for t in trades if not t.get("pattern")]),
    }
    return {"buckets": buckets, "components": components,
            "pattern_presence": pattern_presence, "trades_used": len(trades)}


# ---------------------------------------------------------------------------
# تحلیل حذف تک‌تک مؤلفه‌ها (Leave-One-Out) — افزودنی، پورت از پک خارجی
# ---------------------------------------------------------------------------
def loo_analysis(result: dict, base_threshold: str = "55",
                 segment: str = "out") -> dict:
    """
    روش (سازگار با داده‌ی ما): لاگ معامله‌ی آستانه‌ی پایه برای هر معامله
    «امتیاز کل» و «امتیاز هر مؤلفه» را نگه داشته است. صفرکردن یک مؤلفه فقط
    امتیاز کل را کم می‌کند (جهت، ورود، حد ضرر و هدف به امتیاز کل وابسته
    نیستند)، پس دقیقاً می‌شود از همان لاگ، زیرمجموعه‌ای را نگه داشت که
    بدون آن مؤلفه هم از آستانه می‌گذرد و امید ریاضی‌اش را دوباره حساب کرد.
    - delta مثبت = حذف مؤلفه امید ریاضی را بهتر کرد → مؤلفه در این دوره
      داشته ضرر می‌زده (نویز/وزن نادرست)، نه سیگنال.
    - delta منفی = حذفش اوضاع را بدتر کرد → مؤلفه داشته کار مفید می‌کرده.
    ⚠️ اثر انتخاب (selection) هم داخل delta هست: با حذف مؤلفه بعضی معامله‌ها
    کلاً از مجموعه می‌افتند. و این آمارِ همان یک دوره/نماد است — حکم کلی نیست.
    خروجی: {"base": metrics, "threshold": float, "segment": str,
            "trades_used": n,
            "components": {name: {"kept_trades", "expectancy_r", "delta_r"}}}
    """
    threshold = float(base_threshold)
    trades = _pick_trades(result, base_threshold, segment)
    base = _metrics(trades)
    components = {}
    for name in _COMPONENT_MAX:
        kept = [t for t in trades
                if (t["score"] - (t["components"].get(name) or 0.0)) >= threshold]
        m = _metrics(kept)
        delta = None
        if m["expectancy_r"] is not None and base["expectancy_r"] is not None:
            delta = round(m["expectancy_r"] - base["expectancy_r"], 3)
        components[name] = {"kept_trades": len(kept),
                            "expectancy_r": m["expectancy_r"],
                            "delta_r": delta}
    return {"base": base, "threshold": threshold, "segment": segment,
            "trades_used": len(trades), "components": components}


def format_loo_fa(result: dict, base_threshold: str = "55",
                  segment: str = "out") -> str:
    """جدول فارسی LOO: برای هر مؤلفه، تغییر امید ریاضی هنگام حذف آن."""
    loo = loo_analysis(result, base_threshold, segment)
    seg_fa = {"all": "کل دوره", "in": "داخل نمونه", "out": "خارج نمونه"}[segment]
    b = loo["base"]
    lines = [
        f"— تحلیل حذف تک‌تک مؤلفه‌ها (LOO) — {seg_fa}، آستانه‌ی {base_threshold}",
        (f"سیستم کامل: {b['trades']} معامله | امید ریاضی {b['expectancy_r']}R"
         if b["trades"] else "سیستم کامل: معامله‌ای نیست"),
    ]
    for name in _COMPONENT_MAX:
        c = loo["components"][name]
        if c["kept_trades"] == 0 or c["delta_r"] is None:
            lines.append(f"بدون {_COMPONENT_FA[name]}: معامله‌ای نمی‌ماند")
            continue
        verdict = ("حذفش بهتر کرد → در این دوره ضرر می‌زده"
                   if c["delta_r"] > 0.02 else
                   "حذفش بدتر کرد → در این دوره مفید بوده"
                   if c["delta_r"] < -0.02 else "تأثیر محسوس نداشت")
        lines.append(
            f"بدون {_COMPONENT_FA[name]}: {c['kept_trades']} معامله می‌ماند | "
            f"امید ریاضی {c['expectancy_r']}R | تغییر {c['delta_r']:+}R — {verdict}")
    lines.append("⚠️ LOO روی زیرمجموعه‌ی همان لاگ است (اثر انتخاب دارد) و فقط "
                 "همان دوره/نماد را توصیف می‌کند؛ مبنای تغییر وزن به‌تنهایی نیست.")
    return "\n".join(lines)


def _m_line(label: str, m: dict) -> str:
    if m["trades"] == 0:
        return f"{label}: معامله‌ای نیست"
    return (f"{label}: {m['trades']} معامله | برد {m['win_rate_pct']}٪ | "
            f"امید ریاضی {m['expectancy_r']}R | مجموع {m['total_r']}R")


def format_calibration_fa(result: dict, base_threshold: str = "55",
                          segment: str = "all") -> str:
    """گزارش متنی فارسی کالیبراسیون."""
    cal = calibrate(result, base_threshold, segment)
    seg_fa = {"all": "کل دوره", "in": "داخل نمونه", "out": "خارج نمونه"}[segment]
    lines = [
        f"گزارش کالیبراسیون {result['symbol']}/USDT — {result['timeframe']} "
        f"({seg_fa}، مبنا: لاگ آستانه‌ی {base_threshold}، {cal['trades_used']} معامله)",
        "— باکت‌های امتیاز —",
    ]
    for _, _, label in SCORE_BUCKETS:
        lines.append(_m_line(f"امتیاز {label}", cal["buckets"][label]))
    lines.append("— مؤلفه‌ها (قوی = حداقل نصف سقف مؤلفه) —")
    for name in _COMPONENT_MAX:
        c = cal["components"][name]
        lines.append(_m_line(f"{_COMPONENT_FA[name]} قوی", c["strong"]))
        lines.append(_m_line(f"{_COMPONENT_FA[name]} ضعیف/غایب", c["rest"]))
    lines.append("— حضور الگو —")
    lines.append(_m_line("با الگو", cal["pattern_presence"]["with"]))
    lines.append(_m_line("بدون الگو", cal["pattern_presence"]["without"]))
    lines.append(format_loo_fa(result, base_threshold, segment))
    lines.append("⚠️ آمار تاریخی همان دوره‌ی تست است؛ مبنای تصمیم آینده به‌تنهایی نیست.")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_system_backtest("BTC", "4h")
    print(format_calibration_fa(res, segment="all"))
    print()
    print(format_calibration_fa(res, segment="out"))
