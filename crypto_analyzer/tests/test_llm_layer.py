#!/usr/bin/env python3
"""
tests/test_llm_layer.py
Offline test of the LLM decision layer — no internet, with a fake client (FakeClient):

    python tests/test_llm_layer.py

Coverage:
1) Majority vote of the primary model (3 runs, different temperatures).
2) Vote tie → calling the tiebreak model and its decision.
3) Cross-check direction disagreement → one-step downgrade toward WAIT + the disagreement flag.
4) Numeric validation: an invented EV and an invented price level are rejected.
5) Hard floor: a NO_TRADE candidate means zero LLM calls.
6) No API key → analyze_candidate returns None (layer disabled).
7) Broken JSON: a clean parse error + the fail-safe path to WAIT with reason
   llm_validation_failed when there is not enough valid output.
8) Constants: the tiebreak model is the verified model (not kimi-k2.6, whose
   chat returns 404) and retired V4 Pro is not referenced anywhere.
"""

import http.client
import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import llm_engine
from analysis import llm_client


# ---------------------------------------------------------------------------
# Sample data — with exactly the same fields that the scanner's score_candidate and
# attach_ev_decision produce (see scanner.py and analysis/ev.py)
# ---------------------------------------------------------------------------
def make_candidate(**overrides):
    cand = {
        "symbol": "XRP",
        "timeframe": "4h",
        "direction": "long",
        "direction_fa": "لانگ (خرید)",
        "score": 72.0,
        "components": {"trend": 20.0, "momentum": 15.0, "pattern": 18.0,
                       "risk_reward": 12.0, "volume": 7.0},
        "price": 2.5,
        "entry": 2.5,
        "stop": 2.4,
        "target": 2.75,
        "risk_reward": 2.5,
        "pattern": None,
        "reason": "دلیل: تست",
        "ev": {"expectancy_r": 0.472, "win_rate": 0.857, "win_rate_pct": 85.7,
               "avg_win_r": 1.1, "avg_loss_r": 0.9, "profit_factor": 7.13,
               "break_even_win_rate": 0.457, "break_even_win_rate_pct": 45.7,
               "n_trades": 14, "reliability": "low"},
        "decision": "WAIT",
        "decision_fa": "انتظار — فعلاً ورود نکن",
        "decision_reason": "تست",
    }
    cand.update(overrides)
    return cand


def make_content(decision="BUY", confidence=70, ev_r=0.472, price=2.5,
                 entry=2.5, stop=2.4, target=2.75, score=72.0,
                 win_rate=85.7):
    """One valid model output, per the llm_engine output contract."""
    return json.dumps({
        "decision": decision,
        "confidence": confidence,
        "entry": entry,
        "stop": stop,
        "target": target,
        "scenarios": [
            {"name": "ادامه روند", "probability": 0.6,
             "description": "رسیدن به هدف"},
            {"name": "برگشت", "probability": 0.4,
             "description": "خوردن حد ضرر"},
        ],
        "key_reasons": ["EV مثبت خارج‌ازنمونه"],
        "risks": ["نمونه کم است"],
        "numbers_used": {"price": price, "ev_r": ev_r,
                         "win_rate": win_rate, "score": score},
    })


class FakeClient:
    """Fake client: has one response queue per model and records all calls."""

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
T = llm_engine.LLM_TIEBREAK_MODEL


class TestMajorityVote(unittest.TestCase):
    def test_majority_buy(self):
        client = FakeClient({
            P: [make_content("BUY", 70), make_content("BUY", 80),
                make_content("WAIT", 60)],
            C: [make_content("BUY", 65)],
        })
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "BUY")
        self.assertEqual(result["llm_votes"], {"BUY": 2, "WAIT": 1})
        self.assertTrue(result["llm_validated"])
        self.assertFalse(result["llm_crosscheck_disagreement"])
        self.assertEqual(result["llm_crosscheck"], "BUY")
        # Confidence = the mean of all three valid runs
        self.assertEqual(result["llm_confidence"], 70.0)
        self.assertEqual(client.call_count(P), 3)
        self.assertEqual(client.call_count(C), 1)
        self.assertEqual(client.call_count(T), 0)

    def test_tie_triggers_tiebreak(self):
        client = FakeClient({
            P: [make_content("BUY", 70), make_content("SELL", 70),
                make_content("WAIT", 70)],
            T: [make_content("SELL", 75)],
            C: [make_content("SELL", 65)],
        })
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(client.call_count(T), 1)
        self.assertEqual(result["llm_decision"], "SELL")
        self.assertTrue(result["llm_validated"])


class TestCrosscheckDowngrade(unittest.TestCase):
    def test_buy_downgraded_to_wait_on_sell_crosscheck(self):
        client = FakeClient({
            P: [make_content("BUY", 70)] * 3,
            C: [make_content("SELL", 65)],
        })
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "WAIT")
        self.assertTrue(result["llm_crosscheck_disagreement"])
        self.assertEqual(result["llm_crosscheck"], "SELL")

    def test_strong_buy_downgraded_one_step_on_neutral_crosscheck(self):
        client = FakeClient({
            P: [make_content("STRONG_BUY", 90)] * 3,
            C: [make_content("WAIT", 50)],
        })
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "BUY")
        self.assertTrue(result["llm_crosscheck_disagreement"])


class TestNumericValidation(unittest.TestCase):
    def setUp(self):
        self.source = llm_engine.build_source_data(make_candidate())

    def test_valid_output_passes(self):
        parsed, errors = llm_engine.parse_and_validate(
            make_content("BUY"), self.source)
        self.assertIsNotNone(parsed)
        self.assertEqual(errors, [])

    def test_invented_ev_rejected(self):
        parsed, errors = llm_engine.parse_and_validate(
            make_content("BUY", ev_r=9.99), self.source)
        self.assertIsNone(parsed)
        self.assertTrue(any("ev_r" in e for e in errors), errors)

    def test_invented_level_rejected(self):
        parsed, errors = llm_engine.parse_and_validate(
            make_content("BUY", entry=9.99), self.source)
        self.assertIsNone(parsed)
        self.assertTrue(any("entry" in e for e in errors), errors)

    def test_invented_score_rejected(self):
        parsed, errors = llm_engine.parse_and_validate(
            make_content("BUY", score=12.0), self.source)
        self.assertIsNone(parsed)
        self.assertTrue(any("score" in e for e in errors), errors)

    def test_fenced_json_accepted(self):
        raw = "```json\n" + make_content("BUY") + "\n```"
        parsed, errors = llm_engine.parse_and_validate(raw, self.source)
        self.assertIsNotNone(parsed)
        self.assertEqual(errors, [])

    def test_malformed_json_handled(self):
        parsed, errors = llm_engine.parse_and_validate(
            "این اصلاً JSON نیست", self.source)
        self.assertIsNone(parsed)
        self.assertTrue(errors)


class TestHardFloorAndDisabled(unittest.TestCase):
    def test_no_trade_floor_means_zero_llm_calls(self):
        client = FakeClient({P: [make_content("STRONG_BUY", 95)] * 3,
                             C: [make_content("STRONG_BUY", 95)]})
        cand = make_candidate(decision="NO_TRADE")
        result = llm_engine.analyze_candidate(cand, None, client)
        self.assertEqual(result["llm_decision"], "NO_TRADE")
        self.assertEqual(result["llm_reason"], "ev_hard_floor")
        self.assertEqual(client.call_count(), 0)

    def test_missing_key_returns_none(self):
        old = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            result = llm_engine.analyze_candidate(make_candidate(), None, None)
        finally:
            if old is not None:
                os.environ["NVIDIA_API_KEY"] = old
        self.assertIsNone(result)


class TestFailSafePaths(unittest.TestCase):
    def test_all_invalid_outputs_return_wait(self):
        bad = make_content("BUY", ev_r=9.99)  # All three runs have an invented number
        client = FakeClient({P: [bad] * 3, C: [make_content("BUY")]})
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "WAIT")
        self.assertEqual(result["llm_reason"], "llm_validation_failed")
        self.assertFalse(result["llm_validated"])
        self.assertEqual(client.call_count(C), 0)  # The cross-check is not called at all

    def test_garbage_outputs_return_wait(self):
        client = FakeClient({P: ["خراب", "{ناقص", ""],
                             C: [make_content("BUY")]})
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "WAIT")
        self.assertEqual(result["llm_reason"], "llm_validation_failed")

    def test_client_exception_never_raises(self):
        client = FakeClient({
            P: [llm_client.LLMError("شبکه قطع است")] * 3,
            C: [make_content("BUY")],
        })
        result = llm_engine.analyze_candidate(make_candidate(), None, client)
        self.assertEqual(result["llm_decision"], "WAIT")
        self.assertFalse(result["llm_validated"])


class TestModelConstants(unittest.TestCase):
    def test_verified_models_only(self):
        self.assertEqual(llm_engine.LLM_PRIMARY_MODEL,
                         "deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(llm_engine.LLM_CROSSCHECK_MODEL,
                         "nvidia/nemotron-3-ultra-550b-a55b")
        # kimi-k2.6 is listed in the catalog but its chat returns 404 — forbidden
        self.assertEqual(llm_engine.LLM_TIEBREAK_MODEL,
                         "mistralai/mistral-large-4-1t-a52b")
        self.assertEqual(llm_engine.LLM_SPARE_MODEL, "z-ai/glm-5.3-flash")
        for const in (llm_engine.LLM_PRIMARY_MODEL,
                      llm_engine.LLM_CROSSCHECK_MODEL,
                      llm_engine.LLM_TIEBREAK_MODEL,
                      llm_engine.LLM_SPARE_MODEL):
            self.assertNotIn("kimi", const)
            self.assertNotIn("v4-pro", const)  # V4 Pro is retired

    def test_prompt_uses_only_candidate_numbers(self):
        messages = llm_engine.build_analysis_prompt(make_candidate(), None)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("NEVER INVENT NUMBERS", messages[0]["content"])
        self.assertIn('"expectancy_r": 0.472', messages[1]["content"])


class TestClientKeyResolution(unittest.TestCase):
    def test_env_key_resolution_and_no_key_error(self):
        old = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "YOUR_NVIDIA_API_KEY"
        try:
            self.assertEqual(llm_client.resolve_api_key(),
                             "YOUR_NVIDIA_API_KEY")
            self.assertTrue(llm_client.LLMClient().available)
        finally:
            if old is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old
        empty = llm_client.LLMClient(api_key="")
        self.assertFalse(empty.available)
        with self.assertRaises(llm_client.LLMError):
            empty.chat(P, [{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# Transport-level retry in llm_client.chat — fake transport (patched urlopen)
# Proofs: (1) two disconnects + success = content returns with exactly 3 attempts,
# (2) 410 raises LLMRetiredError on the very first attempt (exactly 1 attempt),
# (3) a persistent disconnect raises LLMError after exactly 3 attempts,
# (4) 5xx is also retried, (5) a non-410 4xx is raised without retry.
# ---------------------------------------------------------------------------
def _ok_payload(content="پاسخ سالم"):
    return {"choices": [{"message": {"content": content},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 10}, "model": "fake"}


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code):
    return urllib.error.HTTPError(
        "https://integrate.api.nvidia.com/v1/chat/completions", code,
        f"fake {code}", hdrs=None, fp=io.BytesIO(b'{"detail":"fake"}'))


class _ScriptedTransport:
    """Fake urlopen: each call returns/raises the next result in the queue."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.attempts = 0

    def __call__(self, req, timeout=None):
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestClientTransportRetry(unittest.TestCase):
    def setUp(self):
        self.client = llm_client.LLMClient(api_key="YOUR_NVIDIA_API_KEY")
        # The real 10/30-second pauses must not run in the test
        sleep_patcher = mock.patch.object(llm_client.time, "sleep",
                                          lambda s: None)
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _chat_with(self, outcomes):
        transport = _ScriptedTransport(outcomes)
        with mock.patch.object(urllib.request, "urlopen", transport):
            result = self.client.chat(
                P, [{"role": "user", "content": "x"}], max_tokens=64)
        return result, transport.attempts

    def test_two_disconnects_then_success_three_attempts(self):
        result, attempts = self._chat_with([
            http.client.RemoteDisconnected("قطع شد"),
            ConnectionResetError("ریست شد"),
            _FakeHTTPResponse(_ok_payload("جواب نهایی")),
        ])
        self.assertEqual(result["content"], "جواب نهایی")
        self.assertEqual(attempts, 3)

    def test_410_raises_immediately_single_attempt(self):
        transport = _ScriptedTransport([_http_error(410)])
        with mock.patch.object(urllib.request, "urlopen", transport):
            with self.assertRaises(llm_client.LLMRetiredError):
                self.client.chat(P, [{"role": "user", "content": "x"}],
                                 max_tokens=64)
        self.assertEqual(transport.attempts, 1)

    def test_persistent_disconnect_raises_after_three_attempts(self):
        transport = _ScriptedTransport(
            [TimeoutError("timeout")] * 5)  # More than the attempt cap
        with mock.patch.object(urllib.request, "urlopen", transport):
            with self.assertRaises(llm_client.LLMError) as ctx:
                self.client.chat(P, [{"role": "user", "content": "x"}],
                                 max_tokens=64)
        self.assertNotIsInstance(ctx.exception, llm_client.LLMRetiredError)
        self.assertIsNone(ctx.exception.status_code)
        self.assertEqual(transport.attempts, 3)

    def test_http_500_retried_then_success(self):
        result, attempts = self._chat_with([
            _http_error(500), _FakeHTTPResponse(_ok_payload("بعد از 500"))])
        self.assertEqual(result["content"], "بعد از 500")
        self.assertEqual(attempts, 2)

    def test_http_401_not_retried(self):
        transport = _ScriptedTransport([_http_error(401)])
        with mock.patch.object(urllib.request, "urlopen", transport):
            with self.assertRaises(llm_client.LLMError) as ctx:
                self.client.chat(P, [{"role": "user", "content": "x"}],
                                 max_tokens=64)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(transport.attempts, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
