"""
analysis/llm_engine.py — لایه‌ی تصمیم LLM (افزودنی، روی لایه‌ی EV)

این ماژول یک لایه‌ی «بازبینی استدلالی» روی خروجی لایه‌ی EV است، نه جایگزین
آن. سه اصل غیرقابل مذاکره:

۱) کف سخت (Hard Floor): اگر تصمیم لایه‌ی EV برای نامزد NO_TRADE باشد،
   همان تصمیم بدون حتی یک فراخوانی LLM برگردانده می‌شود. LLM هرگز
   نمی‌تواند دروازه‌ی EV را دور بزند یا رد اجباریِ EV≤0 / نمونه‌ی ناکافی
   را خنثی کند — دقیقاً همان فلسفه‌ی decide_candidate در analysis/ev.py.

۲) عدد اختراع‌شده = خروجی باطل: هر خروجی مدل باید از اعتبارسنجی عددیِ
   برنامه‌ای رد شود (parse_and_validate). هر عددی که مدل در numbers_used
   یا در سطوح entry/stop/target نقل می‌کند باید با اعداد واقعی نامزد در
   تلورانس نسبی ۰٫۵٪ یکی باشد؛ وگرنه آن اجرا دور ریخته می‌شود. مدل اجازه
   ندارد سطح قیمتی تازه بسازد — سطوح فقط از خود نامزد می‌آیند.

۳) fail-safe: هر خطا (شبکه، JSON خراب، اعتبارسنجی ناموفق، استثنا) به یک
   نتیجه‌ی محافظه‌کارانه ختم می‌شود (WAIT یا llm_decision=None) و هرگز
   استثنا به اسکنر/صدا‌زننده پرتاب نمی‌شود.

مکانیک رأی‌گیری (analyze_candidate):
- مدل اصلی ۳ بار با دماهای ۰٫۲/۰٫۵/۰٫۸ و max_tokens=4096 صدا زده
  می‌شود؛ رأی اکثریت روی decision. تساوی → مدل tiebreak یک بار بین
  گزینه‌های مساوی تصمیم می‌گیرد.
- مدل کراس‌چک یک بار صدا زده می‌شود. نگاشت جهت (ثابت و مستند):
  buy = {STRONG_BUY, BUY}، sell = {STRONG_SELL, SELL}،
  neutral = {WAIT, NO_TRADE}. اگر جهت کراس‌چک با جهت اکثریت فرق کند،
  نتیجه دقیقاً یک پله به سمت WAIT تنزل می‌کند
  (STRONG_BUY→BUY ،BUY→WAIT ،STRONG_SELL→SELL ،SELL→WAIT)
  و llm_crosscheck_disagreement=True ثبت می‌شود.
- کمتر از ۲ خروجی معتبر از مدل اصلی → WAIT با دلیل
  "llm_validation_failed" (حدس نمی‌زنیم).
- اگر هیچ کلید API تنظیم نشده باشد → None برگردانده می‌شود؛ یعنی لایه
  کاملاً غیرفعال است و رفتار سیستم دقیقاً مثل قبل می‌ماند.

⚠️ صداقت روش‌شناختی: LLM به سیگنال، لبه (edge) اضافه نمی‌کند؛ EV همچنان
از بک‌تست خارج‌ازنمونه می‌آید. این لایه کیفیت استدلال/گزارش و یک بازبینی
مستقل دوم است — سقف آن را EV تعیین می‌کند، نه برعکس.
جزئیات کامل: LLM_LAYER_FA.md در ریشه‌ی پکیج.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis import ev as ev_mod
    from analysis import llm_client as llm_client_mod
except ImportError:  # وقتی پوشه‌ی analysis خودش روی sys.path است
    import ev as ev_mod
    import llm_client as llm_client_mod

# ---------------------------------------------------------------------------
# مدل‌ها — در کاتالوگ زنده‌ی NVIDIA Build در 2026-11-07 راستی‌آزمایی شده‌اند
# (ساختن/حذف مدل‌ها در کاتالوگ سریع است؛ قبل از اتکا، /v1/models را چک کنید.
#  مدل moonshotai/kimi-k2.6 در /v1/models لیست می‌شود ولی chat آن 404
#  می‌دهد — عمداً استفاده نشده است. deepseek V4 Pro هم بازنشسته است.)
# ---------------------------------------------------------------------------
LLM_PRIMARY_MODEL = "deepseek-ai/deepseek-v4.1-flash"
LLM_CROSSCHECK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
LLM_TIEBREAK_MODEL = "mistralai/mistral-large-4-1t-a52b"
LLM_SPARE_MODEL = "z-ai/glm-5.3-flash"   # یدکیِ راستی‌آزمایی‌شده؛ فعلاً در مسیر نیست

LLM_PRIMARY_TEMPERATURES = (0.2, 0.5, 0.8)
# بودجه‌ی تولید: 4096 — تست زنده‌ی 2026-11-08 نشان داد هر دو مدل سالم
# (کراس‌چک و tiebreak) قرارداد کامل تحلیل را راحت داخل همین بودجه جا
# می‌دهند؛ تولید کوتاه‌تر یعنی پاسخ سریع‌تر و احتمال کمترِ قطع‌شدن
# وسط راه روی مدل اصلی (که در بودجه‌ی 16384 مکرراً از این شبکه قطع
# می‌شد). تلاش مجددِ سطح انتقال در llm_client هم همین‌جا مکمل است.
LLM_MAX_TOKENS = 4096
LLM_REL_TOLERANCE = 0.005       # تلورانس نسبی اعتبارسنجی عددی: ۰٫۵٪
LLM_MIN_VALID_RUNS = 2          # حداقل خروجی معتبر مدل اصلی برای رأی‌گیری

DECISIONS = tuple(ev_mod.DECISION_FA.keys())

# نگاشت جهت برای مقایسه‌ی کراس‌چک (مستند در docstring ماژول)
_DECISION_SIDE = {
    "STRONG_BUY": "buy", "BUY": "buy",
    "STRONG_SELL": "sell", "SELL": "sell",
    "WAIT": "neutral", "NO_TRADE": "neutral",
}
# تنزل یک پله به سمت WAIT هنگام اختلاف کراس‌چک
_DOWNGRADE_ONE_STEP = {
    "STRONG_BUY": "BUY", "BUY": "WAIT",
    "STRONG_SELL": "SELL", "SELL": "WAIT",
    "WAIT": "WAIT", "NO_TRADE": "NO_TRADE",
}


def decision_side(decision: str) -> str:
    """جهت یک تصمیم: "buy" | "sell" | "neutral" (نامشخص → "neutral")."""
    return _DECISION_SIDE.get(decision, "neutral")


# ---------------------------------------------------------------------------
# ساخت پرامپت
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
    """تبدیل بازگشتی به ساختار JSON‌پذیر (NaN/inf → None، بقیه → str)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    try:
        return float(obj)   # numpy و امثال آن
    except (TypeError, ValueError):
        return str(obj)


def build_analysis_prompt(candidate: dict, report_data: dict = None) -> list:
    """
    messages آماده‌ی chat را می‌سازد: system = فرایند تصمیم نهادیِ فشرده
    (EV اول، آمار بالای روایت، داده‌ی گمشده = اطمینان کمتر، رد اجباری
    EV≤0، ممنوعیت اختراع عدد) و user = نامزد + گزارش محاسباتی به‌صورت JSON.
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
# اعتبارسنجی برنامه‌ای خروجی مدل
# ---------------------------------------------------------------------------
def build_source_data(candidate: dict) -> dict:
    """
    اعداد مرجعِ اعتبارسنجی را از خود نامزد می‌سازد — همان فیلدهایی که
    score_candidate اسکنر و attach_ev_decision تولید می‌کنند:
    candidate["price"], ["entry"], ["stop"], ["target"], ["score"] و
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
        # win_rate را مدل ممکن است به هر دو شکل درصد یا کسر نقل کند؛
        # هر دو مقدار مرجع نگه داشته می‌شود تا اعتبارسنجی هر کدام را بپذیرد
        "win_rate_pct": ev.get("win_rate_pct"),
        "win_rate_frac": ev.get("win_rate"),
    }


def _close_enough(value, source) -> bool:
    """تلورانس نسبی ۰٫۵٪ (با کف مطلق کوچک برای اعداد نزدیک صفر مثل EV)."""
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
    """JSON را از متن خام بیرون می‌کشد (فنس markdown و متن اضافی دورش حذف می‌شود)."""
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
    خروجی خام مدل را پارس و به‌صورت برنامه‌ای اعتبارسنجی می‌کند.

    خروجی: (parsed, errors) — در صورت هر خطا parsed=None و errors فهرست
    دلایل است. قواعد:
    - JSON معتبر و تصمیم در واژگان لایه‌ی EV.
    - confidence عددی در [0, 100].
    - numbers_used باید آبجکت باشد و هر مقدار عددیِ آن (price, ev_r,
      win_rate, score) با مقدار مرجع در تلورانس ۰٫۵٪ یکی باشد؛ عددی که
      مرجعش در داده‌ی منبع نیست = اختراع‌شده = خطا.
    - entry/stop/target: اگر نامزد آن سطح را دارد، مقدار مدل باید در
      ۰٫۵٪ همان باشد؛ اگر نامزد ندارد و مدل عدد داده = سطح اختراع‌شده.
    - scenarios: لیستِ {name, probability∈[0,1], description}.
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

    # سطوح قیمتی: فقط کپیِ سطوح نامزد مجاز است
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

    # اعداد استفاده‌شده: هر عدد باید عیناً از داده‌ی منبع آمده باشد
    numbers_used = parsed.get("numbers_used")
    if not isinstance(numbers_used, dict):
        errors.append("numbers_used_missing: باید آبجکت باشد")
    else:
        refs = {
            "price": (source_data.get("price"),),
            "ev_r": (source_data.get("ev_r"),),
            "score": (source_data.get("score"),),
            # درصد یا کسر — هر کدام با مرجع خودش یکی باشد قبول است
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
# ارکستراسیون
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
    """تشخیص فعال‌بودن کلاینت: کلاینت واقعی available/api_key دارد؛
    کلاینت آزمایشی (fake) بدون این‌ها، فعال فرض می‌شود."""
    if hasattr(client, "available"):
        return bool(client.available)
    if hasattr(client, "api_key"):
        return bool(client.api_key)
    return True


def _call_validated(client, model, messages, temperature, source_data):
    """یک فراخوانی + اعتبارسنجی؛ خروجی: (parsed|None, error|None)."""
    try:
        resp = client.chat(model=model, messages=messages,
                           temperature=temperature, max_tokens=LLM_MAX_TOKENS)
    except Exception as exc:  # LLMError و هر خطای دیگر — fail-safe
        return None, f"call_error: {exc}"
    parsed, errors = parse_and_validate((resp or {}).get("content"), source_data)
    if parsed is None:
        return None, "; ".join(errors)
    return parsed, None


def analyze_candidate(candidate: dict, report_data: dict = None, client=None):
    """
    بازبینی LLM روی یک نامزد اسکنر (که قبلاً attach_ev_decision رویش
    اجرا شده و کلیدهای ev/decision را دارد).

    خروجی: دیکشنری افزودنی با کلیدهای llm_* و scenarios/risks — یا None
    اگر هیچ کلید API تنظیم نشده باشد (لایه غیرفعال، رفتار قبلی سیستم).
    این تابع هرگز استثنا پرتاب نمی‌کند.
    """
    try:
        return _analyze_inner(candidate, report_data, client)
    except Exception as exc:  # آخرین خط دفاع: fail-safe مطلق
        result = _base_result()
        result["llm_error"] = f"unexpected: {exc}"
        result["llm_reason_fa"] = "خطای غیرمنتظره در لایه‌ی LLM؛ نتیجه‌ی لایه‌ی EV معتبر است."
        return result


def _analyze_inner(candidate, report_data, client):
    result = _base_result()

    # (الف) کف سخت: NO_TRADE لایه‌ی EV نهایی است — صفر فراخوانی LLM
    if candidate.get("decision") == "NO_TRADE":
        result["llm_decision"] = "NO_TRADE"
        result["llm_reason"] = "ev_hard_floor"
        result["llm_reason_fa"] = (
            "کف سخت EV: تصمیم لایه‌ی EV ‏NO_TRADE است و LLM اجازه‌ی تغییر "
            "آن را ندارد (بدون فراخوانی مدل).")
        return result

    # (هـ) بدون کلید → لایه غیرفعال، None (بدون هیچ تغییر رفتاری)
    if client is None:
        client = llm_client_mod.LLMClient()
    if not _client_has_key(client):
        return None

    source_data = build_source_data(candidate)
    messages = build_analysis_prompt(candidate, report_data)

    # (ب) مدل اصلی ×۳ و رأی اکثریت
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

    # (د) کمتر از ۲ خروجی معتبر → WAIT محافظه‌کارانه، بدون حدس
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
        # تساوی → مدل tiebreak بین گزینه‌های مساوی تصمیم می‌گیرد
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

    # نماینده‌ی اکثریت: مطمئن‌ترین اجرای هم‌تصمیم (برای سناریو/ریسک)
    peers = [p for p in valid_runs if p["decision"] == majority]
    representative = max(peers, key=lambda p: float(p["confidence"]))
    result["llm_confidence"] = round(
        sum(float(p["confidence"]) for p in valid_runs) / len(valid_runs), 1)
    result["scenarios"] = representative.get("scenarios") or []
    result["risks"] = representative.get("risks") or []

    # (ج) کراس‌چک مستقل: اختلاف جهت → تنزل یک پله به سمت WAIT
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
