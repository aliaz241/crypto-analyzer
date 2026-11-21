"""
backtest/calibration.py
Calibration report — additive analysis tool (step 7). It only works on the
output of backtest/system_backtest and changes no existing computation.

Goal: show, on the tested data, which score ranges and which components
actually had positive expected value and which only had the "appearance":

- By score bucket: 55-60 | 60-70 | 70-80 | 80+
  (the lower-threshold log is a superset of the rest; each trade's bucket is
  read from its own score.)
- By presence/strength of each component: trades are split into "strong
  component" (at least half its cap) and "the rest", and each group's
  expected value is computed separately. For the pattern component,
  "pattern presence" (a named pattern) is also reported separately.

⚠️ These are statistics of the same tested historical period; calibration on
the past does not guarantee the same component will be best in the future
(there is overfitting risk — that is why the out_sample part can also be
taken separately).
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
    """Trade log of one threshold (default: the lowest = superset) + segment filter."""
    data = result["thresholds"][base_threshold]
    trades = data["trades"]
    if segment == "in":
        trades = [t for t in trades if t["segment"] == "in"]
    elif segment == "out":
        trades = [t for t in trades if t["segment"] == "out"]
    return trades


def calibrate(result: dict, base_threshold: str = "55", segment: str = "all") -> dict:
    """
    Output: {"buckets": {label: metrics}, "components": {name: {"strong": m, "rest": m}},
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
# Leave-One-Out component ablation analysis — additive, ported from an external pack
# ---------------------------------------------------------------------------
def loo_analysis(result: dict, base_threshold: str = "55",
                 segment: str = "out") -> dict:
    """
    Method (compatible with our data): the base-threshold trade log has stored,
    for each trade, the "total score" and "each component's score". Zeroing one
    component only reduces the total score (direction, entry, stop loss and
    target do not depend on the total score), so from the same log we can keep
    exactly the subset that would still pass the threshold without that
    component and recompute its expected value.
    - positive delta = removing the component improved the expected value → in
      this period the component was losing money (noise/wrong weight), not
      signaling.
    - negative delta = removing it made things worse → the component was doing
      useful work.
    ⚠️ The selection effect is also inside delta: with the component removed,
    some trades drop out of the set entirely. And these are statistics of one
    period/symbol — not a general verdict.
    Output: {"base": metrics, "threshold": float, "segment": str,
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
    """Persian LOO table: for each component, the change in expected value when it is removed."""
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
    """Persian text report of the calibration."""
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
