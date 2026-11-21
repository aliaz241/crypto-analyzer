"""
analysis/llm_engine.py — LLM decision layer (additive, on top of the EV layer)

This module is a "reasoning review" layer on top of the EV layer's output, not
a replacement for it. Three non-negotiable principles:

1) Hard Floor: if the EV layer's decision for a candidate is NO_TRADE,
   that decision is returned without even one LLM call. The LLM can never
   bypass the EV gate or override the mandatory rejection of EV≤0 /
   insufficient sample — exactly the same philosophy as decide_candidate in
   analysis/ev.py.

2) Invented number = invalid output: every model output must pass programmatic
   numeric validation (parse_and_validate). Every number the model quotes in
   numbers_used or in the entry/stop/target levels must match the candidate's
   real numbers within a relative tolerance of 0.5%; otherwise that run is
   discarded. The model is not allowed to create a new price level — levels
   come only from the candidate itself.

3) fail-safe: any error (network, broken JSON, failed validation, exception)
   ends in a conservative result (WAIT or llm_decision=None) and an exception
   is never raised to the scanner/caller.

Voting mechanics (analyze_candidate):
- The primary model is called 3 times with temperatures 0.2/0.5/0.8 and
  max_tokens=4096; majority vote on decision. Tie → the tiebreak model
  decides once between the tied options.
- The cross-check model is called once. Direction mapping (fixed and documented):
  buy = {STRONG_BUY, BUY}, sell = {STRONG_SELL, SELL},
  neutral = {WAIT, NO_TRADE}. If the cross-check direction differs from the
  majority direction, the result is downgraded exactly one step toward WAIT
  (STRONG_BUY→BUY, BUY→WAIT, STRONG_SELL→SELL, SELL→WAIT)
  and llm_crosscheck_disagreement=True is recorded.
- Fewer than 2 valid outputs from the primary model → WAIT with reason
  "llm_validation_failed" (we do not guess).
- If no API key is set → None is returned; i.e. the layer is
  completely disabled and the system's behavior stays exactly as before.

⚠️ Methodological honesty: the LLM adds no edge to the signal; EV still
comes from the out-of-sample backtest. This layer is reasoning/report quality
and a second independent review — its ceiling is set by EV, not the reverse.
Full details: LLM_LAYER_FA.md at the package root.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis import ev as ev_mod
    from analysis import llm_client as llm_client_mod
except ImportError:  # when the analysis folder itself is on sys.path
    import ev as ev_mod
    import llm_client as llm_client_mod

# ---------------------------------------------------------------------------
# Models — verified against the live NVIDIA Build catalog on 2026-11-07
# (models are added/removed from the catalog quickly; check /v1/models before relying on them.
#  The moonshotai/kimi-k2.6 model is listed in /v1/models but its chat returns 404
#  — deliberately not used. DeepSeek V4 Pro is also retired.)
# ---------------------------------------------------------------------------
LLM_PRIMARY_MODEL = "deepseek-ai/deepseek-v4.1-flash"
LLM_CROSSCHECK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
LLM_TIEBREAK_MODEL = "mistralai/mistral-large-4-1t-a52b"
LLM_SPARE_MODEL = "z-ai/glm-5.3-flash"   # Verified spare; currently not in the path

LLM_PRIMARY_TEMPERATURES = (0.2, 0.5, 0.8)
# Generation budget: 4096 — a live test on 2026-11-08 showed both healthy models
# (cross-check and tiebreak) comfortably fit the full analysis contract within this budget;
# a shorter generation means a faster response and a lower chance of being cut off
# midway on the primary model (which on the 16384 budget repeatedly got cut off on this network).
# Transport-level retry in llm_client complements this here.
LLM_MAX_TOKENS = 4096
LLM_REL_TOLERANCE = 0.005       # Relative tolerance for numeric validation: 0.5%
LLM_MIN_VALID_RUNS = 2          # Minimum valid output of the primary model for voting

DECISIONS = tuple(ev_mod.DECISION_FA.keys())

# Direction mapping for cross-check comparison (documented in the module docstring)
_DECISION_SIDE = {
    "STRONG_BUY": "buy", "BUY": "buy",
    "STRONG_SELL": "sell", "SELL": "sell",
    "WAIT": "neutral", "NO_TRADE": "neutral",
}
# One-step downgrade toward WAIT on cross-check disagreement
_DOWNGRADE_ONE_STEP = {
    "STRONG_BUY": "BUY", "BUY": "WAIT",
    "STRONG_SELL": "SELL", "SELL": "WAIT",
    "WAIT": "WAIT", "NO_TRADE": "NO_TRADE",
}


def decision_side(decision: str) -> str:
    """Direction of a decision: "buy" | "sell" | "neutral" (unknown → "neutral")."""
    return _DECISION_SIDE.get(decision, "neutral")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are the final decision reviewer in an institutional, statistics-first trading analysis pipeline. A deterministic quantitative layer has already computed every number you will see (scanner score, backtest-derived expected value from OUT-OF-SAMPLE trades only, entry/stop/target levels). Your job is disciplined judgement over those numbers — never new numbers.

Decision process:
1. EV FIRST. The candidate's out-of-sample net expectancy (ev.expectancy_r, in R) is the primary evidence. If ev <= 0 the trade is already rejected by a hard rule upstream and will not reach you; if EV is positive but the sample is small (reliability "low" or "insufficient"), confidence MUST be low and WAIT is usually correct.
2. STATISTICS OVER NARRATIVE. Patterns, indicator alignment and story are secondary. A compelling story never outweighs weak statistics.
3. MISSING DATA LOWERS CONFIDENCE. Any null/missing field (no stop, no target, no EV, no pattern stats) reduces confidence and pushes toward WAIT. Never fill gaps with assumptions.
4. LEVELS ARE GIVEN, NOT INVENTED. entry/stop/target in your answer must be exactly the candidate's levels. Do not propose your own prices.
5. NEVER INVENT NUMBERS. Use ONLY the numbers present in the provided JSON. Every figure you rely on must be echoed verbatim in numbers_used.
6. Think in scenarios: give 2-4 mutually exclusive scenarios with probabilities summing to about 1.0.

Output contract: reply with ONLY a single JSON object, no markdown fences, no commentary, with exactly these fields:
{
  "decision": one of "STRONG_BUY" | "BUY" | "WAIT" | "NO_TRADE" | "SELL" | "STRONG_SELL",
  "confidence": integer 0-100,
  "entry": the candidate's entry level (number, copied),
  "stop": the candidate's stop level (number, copied),
  "target": the candidate's target level (number, copied),
  "scenarios": [{"name": str, "probability": number 0..1, "description": str}],
  "key_reasons": [str, ...],
  "risks": [str, ...],
  "numbers_used": {"price": <candidate price>, "ev_r": <ev.expectancy_r>, "win_rate": <ev.win_rate_pct>, "score": <candidate score>}
}"""


def _json_safe(obj):
    """Recursive conversion to a JSON-able structure (NaN/inf → None, the rest → str)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    try:
        return float(obj)   # numpy and the like
    except (TypeError, ValueError):
        return str(obj)


def build_analysis_prompt(candidate: dict, report_data: dict = None) -> list:
    """
    Builds ready chat messages: system = a compact institutional decision process
    (EV first, statistics above narrative, missing data = less confidence, mandatory
    rejection of EV≤0, prohibition on inventing numbers) and user = the candidate +
    the computational report as JSON.
    """
    payload = {
        "candidate": _json_safe(candidate),
        "report": _json_safe(report_data) if report_data is not None else None,
    }
    user = (
        "Candidate and computed report (JSON). Decide strictly from these "
        "numbers and return ONLY the JSON object defined in the contract.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Programmatic validation of the model output
# ---------------------------------------------------------------------------
def build_source_data(candidate: dict) -> dict:
    """
    Builds the reference numbers for validation from the candidate itself — the same
    fields that the scanner's score_candidate and attach_ev_decision produce:
    candidate["price"], ["entry"], ["stop"], ["target"], ["score"] and
    candidate["ev"]["expectancy_r"], ["win_rate"], ["win_rate_pct"].
    """
    ev = candidate.get("ev") or {}
    return {
        "price": candidate.get("price"),
        "entry": candidate.get("entry"),
        "stop": candidate.get("stop"),
        "target": candidate.get("target"),
        "score": candidate.get("score"),
        "ev_r": ev.get("expectancy_r"),
        # The model may quote win_rate in either percent or fraction form;
        # both reference values are kept so validation accepts either
        "win_rate_pct": ev.get("win_rate_pct"),
        "win_rate_frac": ev.get("win_rate"),
    }


def _close_enough(value, source) -> bool:
    """Relative tolerance of 0.5% (with a small absolute floor for near-zero numbers like EV)."""
    if value is None or source is None:
        return False
    try:
        value = float(value)
        source = float(source)
    except (TypeError, ValueError):
        return False
    tol = LLM_REL_TOLERANCE * max(abs(source), 1.0)
    return abs(value - source) <= tol


def _extract_json(raw: str):
    """Extracts JSON from raw text (markdown fences and extra text around it are removed)."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def parse_and_validate(raw_content, source_data: dict):
    """
    Parses the model's raw output and validates it programmatically.

    Output: (parsed, errors) — on any error parsed=None and errors is the list
    of reasons. Rules:
    - Valid JSON and a decision in the EV layer's vocabulary.
    - confidence numeric in [0, 100].
    - numbers_used must be an object and every numeric value in it (price, ev_r,
      win_rate, score) must match the reference value within 0.5% tolerance; a number
      whose reference is not in the source data = invented = error.
    - entry/stop/target: if the candidate has that level, the model's value must be
      within 0.5% of it; if the candidate does not have it and the model gives a
      number = invented level.
    - scenarios: a list of {name, probability∈[0,1], description}.
    """
    errors = []
    parsed = _extract_json(raw_content)
    if parsed is None:
        return None, ["invalid_json: خروجی مدل JSON معتبر نیست"]
    if not isinstance(parsed, dict):
        return None, ["invalid_shape: خروجی باید یک آبجکت JSON باشد"]

    decision = parsed.get("decision")
    if decision not in DECISIONS:
        errors.append(f"invalid_decision: {decision!r}")

    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) \
            or not (0 <= float(conf) <= 100):
        errors.append(f"invalid_confidence: {conf!r}")

    # Price levels: only copying the candidate's levels is allowed
    for field in ("entry", "stop", "target"):
        value = parsed.get(field)
        source = source_data.get(field)
        if value is None:
            continue
        if source is None:
            errors.append(f"invented_level: مدل برای {field} عدد داده ولی "
                          "نامزد این سطح را ندارد")
        elif not _close_enough(value, source):
            errors.append(f"level_mismatch: {field} مدل={value} ولی مقدار "
                          f"نامزد={source} (خارج از تلورانس ۰٫۵٪)")

    # Numbers used: every number must have come verbatim from the source data
    numbers_used = parsed.get("numbers_used")
    if not isinstance(numbers_used, dict):
        errors.append("numbers_used_missing: باید آبجکت باشد")
    else:
        refs = {
            "price": (source_data.get("price"),),
            "ev_r": (source_data.get("ev_r"),),
            "score": (source_data.get("score"),),
            # Percent or fraction — either is accepted if it matches its own reference
            "win_rate": (source_data.get("win_rate_pct"),
                         source_data.get("win_rate_frac")),
        }
        for key, sources in refs.items():
            if key not in numbers_used or numbers_used.get(key) is None:
                continue
            value = numbers_used.get(key)
            known = [s for s in sources if s is not None]
            if not known:
                errors.append(f"invented_number: numbers_used.{key}={value} "
                              "ولی این عدد در داده‌ی منبع نیست")
            elif not any(_close_enough(value, s) for s in known):
                errors.append(f"number_mismatch: numbers_used.{key}={value} "
                              f"با مقدار منبع {known} نمی‌خواند")

    scenarios = parsed.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("scenarios_missing: لیست سناریوها لازم است")
    else:
        for i, sc in enumerate(scenarios):
            if not isinstance(sc, dict) or "name" not in sc \
                    or "probability" not in sc or "description" not in sc:
                errors.append(f"scenario_shape: سناریوی {i} ناقص است")
                continue
            p = sc.get("probability")
            if not isinstance(p, (int, float)) or isinstance(p, bool) \
                    or not (0.0 <= float(p) <= 1.0):
                errors.append(f"scenario_probability: سناریوی {i} احتمال "
                              f"نامعتبر دارد ({p!r})")

    for field in ("key_reasons", "risks"):
        if not isinstance(parsed.get(field), list):
            errors.append(f"{field}_missing: باید لیست باشد")

    if errors:
        return None, errors
    return parsed, []


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _base_result():
    return {
        "llm_decision": None,
        "llm_confidence": None,
        "llm_votes": {},
        "llm_crosscheck": None,
        "llm_crosscheck_disagreement": False,
        "llm_validated": False,
        "llm_reason": None,
        "llm_reason_fa": None,
        "llm_invalid_outputs": 0,
        "llm_error": None,
        "llm_models": {
            "primary": LLM_PRIMARY_MODEL,
            "crosscheck": LLM_CROSSCHECK_MODEL,
            "tiebreak": LLM_TIEBREAK_MODEL,
        },
        "scenarios": [],
        "risks": [],
    }


def _client_has_key(client) -> bool:
    """Detecting whether the client is active: a real client has available/api_key;
    a test (fake) client without them is assumed active."""
    if hasattr(client, "available"):
        return bool(client.available)
    if hasattr(client, "api_key"):
        return bool(client.api_key)
    return True


def _call_validated(client, model, messages, temperature, source_data):
    """One call + validation; output: (parsed|None, error|None)."""
    try:
        resp = client.chat(model=model, messages=messages,
                           temperature=temperature, max_tokens=LLM_MAX_TOKENS)
    except Exception as exc:  # LLMError and any other error — fail-safe
        return None, f"call_error: {exc}"
    parsed, errors = parse_and_validate((resp or {}).get("content"), source_data)
    if parsed is None:
        return None, "; ".join(errors)
    return parsed, None


def analyze_candidate(candidate: dict, report_data: dict = None, client=None):
    """
    LLM review on one scanner candidate (on which attach_ev_decision has
    already been run and which has the ev/decision keys).

    Output: an additive dict with llm_* keys and scenarios/risks — or None
    if no API key is set (layer disabled, the system's previous behavior).
    This function never raises an exception.
    """
    try:
        return _analyze_inner(candidate, report_data, client)
    except Exception as exc:  # Last line of defense: absolute fail-safe
        result = _base_result()
        result["llm_error"] = f"unexpected: {exc}"
        result["llm_reason_fa"] = "خطای غیرمنتظره در لایه‌ی LLM؛ نتیجه‌ی لایه‌ی EV معتبر است."
        return result


def _analyze_inner(candidate, report_data, client):
    result = _base_result()

    # (a) Hard floor: the EV layer's NO_TRADE is final — zero LLM calls
    if candidate.get("decision") == "NO_TRADE":
        result["llm_decision"] = "NO_TRADE"
        result["llm_reason"] = "ev_hard_floor"
        result["llm_reason_fa"] = (
            "کف سخت EV: تصمیم لایه‌ی EV ‏NO_TRADE است و LLM اجازه‌ی تغییر "
            "آن را ندارد (بدون فراخوانی مدل).")
        return result

    # (e) No key → layer disabled, None (with no behavioral change)
    if client is None:
        client = llm_client_mod.LLMClient()
    if not _client_has_key(client):
        return None

    source_data = build_source_data(candidate)
    messages = build_analysis_prompt(candidate, report_data)

    # (b) Primary model x3 and majority vote
    valid_runs = []
    invalid = 0
    for temp in LLM_PRIMARY_TEMPERATURES:
        parsed, _err = _call_validated(client, LLM_PRIMARY_MODEL, messages,
                                       temp, source_data)
        if parsed is None:
            invalid += 1
        else:
            valid_runs.append(parsed)
    result["llm_invalid_outputs"] = invalid

    # (d) Fewer than 2 valid outputs → conservative WAIT, no guessing
    if len(valid_runs) < LLM_MIN_VALID_RUNS:
        result["llm_decision"] = "WAIT"
        result["llm_reason"] = "llm_validation_failed"
        result["llm_reason_fa"] = (
            f"خروجی معتبر کافی از مدل اصلی نیامد ({len(valid_runs)} از "
            f"{len(LLM_PRIMARY_TEMPERATURES)})؛ تصمیم محافظه‌کارانه: انتظار.")
        return result

    votes = Counter(p["decision"] for p in valid_runs)
    result["llm_votes"] = dict(votes)
    top_count = max(votes.values())
    leaders = [d for d, c in votes.items() if c == top_count]

    if len(leaders) == 1:
        majority = leaders[0]
    else:
        # Tie → the tiebreak model decides between the tied options
        tie_messages = messages + [{
            "role": "user",
            "content": ("The primary panel is tied between these decisions: "
                        + ", ".join(sorted(leaders))
                        + ". You are the tiebreaker: choose exactly one of "
                          "them and return the same JSON contract."),
        }]
        parsed_tb, _err = _call_validated(client, LLM_TIEBREAK_MODEL,
                                          tie_messages, 0.2, source_data)
        if parsed_tb is None or parsed_tb["decision"] not in leaders:
            result["llm_decision"] = "WAIT"
            result["llm_reason"] = "llm_tiebreak_failed"
            result["llm_reason_fa"] = (
                "رأی مدل اصلی مساوی شد و مدل tiebreak تصمیم معتبر نداد؛ "
                "تصمیم محافظه‌کارانه: انتظار.")
            return result
        majority = parsed_tb["decision"]
        valid_runs.append(parsed_tb)

    # Majority representative: the most confident run of the same decision (for scenario/risk)
    peers = [p for p in valid_runs if p["decision"] == majority]
    representative = max(peers, key=lambda p: float(p["confidence"]))
    result["llm_confidence"] = round(
        sum(float(p["confidence"]) for p in valid_runs) / len(valid_runs), 1)
    result["scenarios"] = representative.get("scenarios") or []
    result["risks"] = representative.get("risks") or []

    # (c) Independent cross-check: direction disagreement → one-step downgrade toward WAIT
    final = majority
    cross_parsed, _err = _call_validated(client, LLM_CROSSCHECK_MODEL,
                                         messages, 0.2, source_data)
    if cross_parsed is not None:
        cross_decision = cross_parsed["decision"]
        result["llm_crosscheck"] = cross_decision
        if decision_side(cross_decision) != decision_side(majority):
            final = _DOWNGRADE_ONE_STEP[majority]
            result["llm_crosscheck_disagreement"] = True

    result["llm_decision"] = final
    result["llm_validated"] = True
    fa = ev_mod.DECISION_FA.get(final, final)
    note = "؛ کراس‌چک مخالف بود و یک پله تنزل اعمال شد" \
        if result["llm_crosscheck_disagreement"] else ""
    result["llm_reason_fa"] = (
        f"رأی اکثریت مدل اصلی ({dict(votes)}) → {fa}، اطمینان میانگین "
        f"{result['llm_confidence']}٪{note}.")
    return result
