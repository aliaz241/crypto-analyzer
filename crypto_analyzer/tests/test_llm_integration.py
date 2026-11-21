#!/usr/bin/env python3
"""
tests/test_llm_integration.py
Offline test of the LLM layer's integration with the scanner — no internet, with a fake client:

    python tests/test_llm_integration.py

Coverage:
1) Layer off (default): scan_market output exactly equals a baseline dict
   taken without the layer — a full deep-compare (no llm / llm_summary keys,
   no field changed). Data comes from sample_data synthetic candles and a
   cache-replaced fetch, and EV is read from the package's real cache
   (BTC/ETH take NO_TRADE on this data).
2) Layer on with a fake client on direct attach: a WAIT candidate (passed
   the EV gate) gets the "llm" key and NO_TRADE never reaches the client
   (call count = 0). The scan_market path with llm=True on this data's
   candidates are all NO_TRADE, so 0 calls and llm_summary has zero analysis.
3) The LLM_SCAN_TOP_K cap on the internal _attach_llm_to_scan path: of 5
   non-NO_TRADE candidates, exactly 3 are reviewed.
4) An erroring fake client: the scan output stays completely unchanged (the
   candidate gets a fail-safe WAIT or stays untouched, but its previous
   fields do not change).
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import scanner
from analysis import analyzer as analyzer_mod
from analysis import llm_client, llm_engine
from tests.sample_data import generate_synthetic_candles

fails = []


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


class FakeClient:
    """Queue-based fake client — the same FakeClient contract as in test_llm_layer."""

    def __init__(self, queues):
        self.queues = {m: list(v) for m, v in queues.items()}
        self.calls = []

    def chat(self, model, messages, temperature=0.2, max_tokens=None,
             timeout=None):
        self.calls.append({"model": model, "temperature": temperature})
        queue = self.queues.get(model) or []
        if not queue:
            raise AssertionError(f"FakeClient: پاسخ تعریف‌نشده برای مدل {model}")
        content = queue.pop(0)
        if isinstance(content, Exception):
            raise content
        return {"content": content, "finish_reason": "stop", "usage": {}}

    def call_count(self, model=None):
        if model is None:
            return len(self.calls)
        return sum(1 for c in self.calls if c["model"] == model)


P = llm_engine.LLM_PRIMARY_MODEL
C = llm_engine.LLM_CROSSCHECK_MODEL


def make_content(cand, decision="BUY", confidence=70):
    """A valid model output for this same candidate (its numbers are copied from the
    candidate itself so the engine's numeric validation passes)."""
    ev = cand.get("ev") or {}
    return json.dumps({
        "decision": decision,
        "confidence": confidence,
        "entry": cand.get("entry"),
        "stop": cand.get("stop"),
        "target": cand.get("target"),
        "scenarios": [{"name": "ادامه", "probability": 0.6,
                       "description": "رسیدن به هدف"},
                      {"name": "برگشت", "probability": 0.4,
                       "description": "خوردن حد ضرر"}],
        "key_reasons": ["تست یکپارچه‌سازی"],
        "risks": ["تست"],
        "numbers_used": {"price": cand.get("price"),
                         "ev_r": ev.get("expectancy_r"),
                         "win_rate": ev.get("win_rate_pct"),
                         "score": cand.get("score")},
    })


def make_wait_candidate(symbol="XRP", timeframe="4h"):
    """A WAIT candidate in the same shape as the scanner output + attach_ev_decision (this
    combination's real EV data from the final matrix: +0.3666R, n=12 — the same
    shape is used in test_llm_layer too)."""
    return {
        "symbol": symbol, "timeframe": timeframe, "direction": "long",
        "direction_fa": "لانگ (خرید)", "score": 72.0,
        "components": {"trend": 20.0, "momentum": 15.0, "pattern": 18.0,
                       "risk_reward": 12.0, "volume": 7.0},
        "price": 2.5, "entry": 2.5, "stop": 2.4, "target": 2.75,
        "risk_reward": 2.5, "pattern": None, "reason": "دلیل: تست",
        "sizing": None,
        "ev": {"expectancy_r": 0.3666, "win_rate": 0.8333,
               "win_rate_pct": 83.3, "avg_win_r": 0.5577,
               "avg_loss_r": 0.589, "profit_factor": 4.7343,
               "break_even_win_rate": 0.5136, "break_even_win_rate_pct": 51.4,
               "n_trades": 12, "reliability": "low"},
        "decision": "WAIT",
        "decision_fa": "انتظار — فعلاً ورود نکن",
        "decision_reason": "تست",
    }


def make_no_trade_candidate():
    cand = make_wait_candidate(symbol="BTC")
    cand["ev"] = None
    cand["decision"] = "NO_TRADE"
    cand["decision_fa"] = "بدون معامله (NO TRADE)"
    return cand


# ---------------------------------------------------------------------------
# Offline fetch: a fixed synthetic candle for all four regression combinations (separate seed but
# deterministic) so scan_market runs on the same real path without a network.
# ---------------------------------------------------------------------------
_DFS = {}
for _sym, _seed in (("BTC", 42), ("ETH", 43)):
    for _tf in ("4h", "1d"):
        _DFS[(_sym, _tf)] = generate_synthetic_candles(
            n=400, start_price=50000.0 if _sym == "BTC" else 3000.0,
            seed=_seed + (0 if _tf == "4h" else 100))


def _fake_fetch(symbol, timeframe, *args, **kwargs):
    return _DFS[(symbol.upper(), timeframe)]


_orig_fetch = analyzer_mod.fetch_candles_binance
scanner.analyzer.fetch_candles_binance = _fake_fetch
try:
    # --- 1) Off = deep-compare with the baseline ---
    base = scanner.scan_market(["BTC", "ETH"], ["4h", "1d"], 5,
                               progress=False)
    baseline = copy.deepcopy(base)
    check("اسکن مبنا نامزد دارد", len(baseline["candidates"]) > 0,
          f"candidates={len(baseline['candidates'])}")
    again = scanner.scan_market(["BTC", "ETH"], ["4h", "1d"], 5,
                                progress=False)
    check("خاموش: دو اجرای پشت‌سرهم دقیقاً برابرند (deterministic)",
          again == baseline)

    # --- 2) On but no key and no client: disabled summary ---
    no_key = scanner.scan_market(["BTC", "ETH"], ["4h", "1d"], 5,
                                 progress=False, llm=True)
    check("روشنِ بدون کلید: llm_summary غیرفعال با دلیل no_api_key",
          no_key.get("llm_summary", {}).get("active") is False
          and no_key.get("llm_summary", {}).get("reason") == "no_api_key")
    stripped = copy.deepcopy(no_key)
    stripped.pop("llm_summary", None)
    check("روشنِ بدون کلید: بقیه‌ی خروجی دقیقاً برابر مبناست",
          stripped == baseline)
    check("روشنِ بدون کلید: هیچ نامزدی کلید llm نگرفته",
          all("llm" not in c for c in no_key["candidates"]))

    # --- 3) On with a fake client on the same data: all NO_TRADE ---
    # (synthetic BTC/ETH take a NO_TRADE decision on the real EV cache, so the engine
    #  must not be called even once and nobody gets an llm key.)
    fake_idle = FakeClient({P: [], C: []})
    with_fake = scanner.scan_market(["BTC", "ETH"], ["4h", "1d"], 5,
                                    progress=False, llm=True,
                                    llm_client=fake_idle)
    check("داده‌ی NO_TRADE: صفر فراخوانی به کلاینت جعلی",
          fake_idle.call_count() == 0)
    check("داده‌ی NO_TRADE: llm_summary فعال ولی analyzed=0",
          with_fake.get("llm_summary", {}).get("active") is True
          and with_fake.get("llm_summary", {}).get("analyzed") == 0)
    stripped2 = copy.deepcopy(with_fake)
    stripped2.pop("llm_summary", None)
    check("داده‌ی NO_TRADE: بقیه‌ی خروجی دقیقاً برابر مبناست",
          stripped2 == baseline)

    # --- 4) Direct attach: gets WAIT, NO_TRADE never arrives ---
    wait_cand = make_wait_candidate()
    content = make_content(wait_cand, "BUY")
    fake1 = FakeClient({P: [content] * 3, C: [content]})
    out_cand = scanner.attach_llm_analysis(copy.deepcopy(wait_cand),
                                           client=fake1)
    check("attach روی WAIT: کلید llm نشسته و تصمیم BUY است",
          (out_cand.get("llm") or {}).get("llm_decision") == "BUY")
    check("attach روی WAIT: ۳ رأی اصلی + ۱ کراس‌چک صدا زده شد",
          fake1.call_count(P) == 3 and fake1.call_count(C) == 1)

    nt_cand = make_no_trade_candidate()
    fake2 = FakeClient({P: [content] * 9, C: [content] * 9})
    out_nt = scanner.attach_llm_analysis(nt_cand, client=fake2)
    check("attach روی NO_TRADE: کلید llm ندارد", "llm" not in out_nt)
    check("attach روی NO_TRADE: صفر فراخوانی (کف سخت)",
          fake2.call_count() == 0)

    # --- 5) Cap of 3 on _attach_llm_to_scan ---
    five = [make_wait_candidate(symbol=f"S{i}") for i in range(5)]
    queues = {P: [], C: []}
    for c in five[:3]:
        queues[P].extend([make_content(c, "BUY")] * 3)
        queues[C].append(make_content(c, "BUY"))
    fake3 = FakeClient(queues)
    fake_out = {"candidates": copy.deepcopy(five)}
    scanner._attach_llm_to_scan(fake_out, True, client=fake3)
    n_llm = sum(1 for c in fake_out["candidates"] if "llm" in c)
    check("سقف LLM_SCAN_TOP_K: دقیقاً ۳ از ۵ نامزد بازبینی شدند",
          n_llm == config.LLM_SCAN_TOP_K == 3, f"n_llm={n_llm}")
    check("سقف: llm_summary analyzed=3 و decisions BUY=3",
          fake_out["llm_summary"]["analyzed"] == 3
          and fake_out["llm_summary"]["decisions"] == {"BUY": 3})

    # --- 6) Erroring client: the candidate's previous fields are untouched ---
    boom = llm_client.LLMError("شبکه قطع است")
    fake4 = FakeClient({P: [boom] * 3, C: [make_content(wait_cand)]})
    orig = make_wait_candidate()
    snapshot = copy.deepcopy(orig)
    out_err = scanner.attach_llm_analysis(orig, client=fake4)
    changed_keys = [k for k in snapshot
                    if out_err.get(k) != snapshot[k]]
    check("کلاینت خطادار: هیچ فیلد قبلی‌ای تغییر نکرده",
          changed_keys == [], f"changed={changed_keys}")
    check("کلاینت خطادار: fail-safe موتور WAIT محافظه‌کارانه است",
          (out_err.get("llm") or {}).get("llm_decision") == "WAIT"
          and (out_err.get("llm") or {}).get("llm_reason")
          == "llm_validation_failed")

    raiser = FakeClient({P: [RuntimeError("boom")] * 3, C: []})
    out_err2 = scanner.attach_llm_analysis(make_wait_candidate(),
                                           client=raiser)
    check("استثنای غیرمنتظره‌ی کلاینت: attach برمی‌گردد و نمی‌ترکد",
          isinstance(out_err2, dict))

    # --- 7) llm_layer_active off by default ---
    check("llm_layer_active() پیش‌فرض False است",
          scanner.llm_layer_active() is False)
finally:
    scanner.analyzer.fetch_candles_binance = _orig_fetch

print()
if fails:
    print(f"❌ {len(fails)} تست یکپارچه‌سازی LLM شکست خورد: {fails}")
    sys.exit(1)
print("✅ همه‌ی تست‌های یکپارچه‌سازی LLM موفق بود.\n")
