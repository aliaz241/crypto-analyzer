"""
analysis/llm_client.py — کلاینت حداقلی OpenAI-compatible برای NVIDIA Build

فقط با کتابخانه‌ی استاندارد (urllib) نوشته شده تا روی ترموکس و بدون هیچ
وابستگی تازه‌ای کار کند؛ این پکیج عمداً هیچ SDK ای نصب نمی‌کند.

- Base URL: ‏https://integrate.api.nvidia.com/v1‏ (اندپوینت‌های
  chat/completions سازگار با OpenAI).
- کلید API (به ترتیب اولویت):
  ۱) ثابت config.NVIDIA_API_KEY اگر در config تعریف شده و غیرخالی باشد
     (از گام یکپارچه‌سازی در config.py هست و پیش‌فرضش "" است؛ این ماژول
     با getattr می‌خواندش)،
  ۲) متغیر محیطی NVIDIA_API_KEY،
  ۳) None — یعنی لایه‌ی LLM غیرفعال است و صدا‌زننده باید بی‌صدا رد شود.
- کلید هرگز در پیام خطا، لاگ یا خروجی چاپ نمی‌شود.
- timeout پیش‌فرض بلند است (۶۰۰ ثانیه): مدل‌های استدلالی روی این سرویس
  کندند و کندی برای این لایه قابل قبول است، قطع‌شدن پاسخ نه.
- سیاست تلاش مجدد (داخل خود chat): حداکثر ۳ تلاش مجموع. روی خطاهای
  سطح اتصال (قطع اتصال حین خواندن/RemoteDisconnected، خطای اتصال و
  timeout سوکت) و روی HTTP 5xx و 429، با backoff پلکانی ۱۰ و ۳۰ ثانیه
  دوباره تلاش می‌شود؛ چون تست زنده‌ی 2026-11-08 نشان داد مدل اصلی از
  این شبکه روی تولیدهای بلند مکرراً وسط راه قطع می‌شود و یک اجرای
  تحلیل نباید برای موفقیت به سالم‌ماندن تک‌تک فراخوانی‌هایش وابسته
  باشد. روی 410 (مدل بازنشسته/حذف‌شده از کاتالوگ) هرگز تلاش مجدد
  نمی‌شود و همان تلاش اول LLMRetiredError پرتاب می‌شود تا صدا‌زننده
  فوراً بفهمد ID مدل مرده است، نه این‌که سرویس شلوغ است؛ بقیه‌ی 4xx ها
  هم بدون تلاش مجدد همان بار اول پرتاب می‌شوند.
"""

import http.client
import json
import os
import time
import urllib.error
import urllib.request

try:
    import config as _config
except Exception:  # pragma: no cover - وقتی analysis/ خودش روی sys.path است
    _config = None

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TIMEOUT_SECONDS = 600
_MAX_ATTEMPTS = 3                          # مجموع تلاش‌ها داخل یک chat()
_RETRY_BACKOFFS_SECONDS = (10.0, 30.0)     # مکث قبل از تلاش ۲ و ۳
_ERROR_BODY_MAX_CHARS = 300


class LLMError(Exception):
    """خطای لایه‌ی LLM؛ status_code کد HTTP است (None برای خطای شبکه)."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class LLMRetiredError(LLMError):
    """مدل از کاتالوگ NVIDIA حذف/بازنشسته شده است (HTTP 410) — تلاش مجدد ممنوع."""


def resolve_api_key():
    """
    کلید NVIDIA را طبق ترتیب مستندشده در docstring ماژول برمی‌گرداند
    یا None اگر هیچ‌جا تنظیم نشده باشد.
    """
    if _config is not None:
        key = getattr(_config, "NVIDIA_API_KEY", None)
        if isinstance(key, str) and key.strip():
            return key.strip()
    env_key = os.environ.get("NVIDIA_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


class LLMClient:
    """کلاینت ساده‌ی chat/completions برای NVIDIA Build (فقط stdlib)."""

    def __init__(self, api_key=None, base_url=BASE_URL):
        # api_key=None یعنی «خودت resolve کن»؛ رشته‌ی خالی یعنی عمداً بدون کلید.
        self.api_key = resolve_api_key() if api_key is None else (api_key or None)
        self.base_url = (base_url or BASE_URL).rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    def chat(self, model, messages, temperature=0.2, max_tokens=16384,
             timeout=None):
        """
        یک فراخوانی chat/completions.

        خروجی: {"content": str, "finish_reason": str|None,
                "usage": dict|None, "model": str}
        روی هر خطای HTTP/شبکه LLMError (یا LLMRetiredError برای 410)
        پرتاب می‌شود؛ صدا‌زننده (llm_engine) همه را fail-safe می‌گیرد.
        """
        if not self.api_key:
            raise LLMError("NVIDIA_API_KEY تنظیم نشده است — لایه‌ی LLM غیرفعال است.")
        if timeout is None:
            timeout = DEFAULT_TIMEOUT_SECONDS

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self.base_url + "/chat/completions"

        last_error = None
        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                # backoff پلکانی بین تلاش‌ها: ۱۰ ثانیه، بعد ۳۰ ثانیه
                time.sleep(_RETRY_BACKOFFS_SECONDS[attempt - 1])
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return self._parse_response(raw, model)
            except urllib.error.HTTPError as exc:
                detail = self._safe_body(exc)
                if exc.code == 410:
                    # مدل مرده است؛ تلاش مجدد فقط وقت تلف می‌کند
                    raise LLMRetiredError(
                        f"مدل '{model}' روی NVIDIA Build در دسترس نیست "
                        f"(HTTP 410 — بازنشسته/حذف از کاتالوگ). "
                        f"جزئیات: {detail}",
                        status_code=410) from None
                last_error = LLMError(
                    f"خطای HTTP {exc.code} از NVIDIA Build برای مدل "
                    f"'{model}': {detail}",
                    status_code=exc.code)
                if not (exc.code == 429 or 500 <= exc.code < 600):
                    # بقیه‌ی 4xx ها قطعی‌اند؛ بدون تلاش مجدد
                    raise last_error from None
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as exc:
                # خطای سطح اتصال — شامل RemoteDisconnected (زیرکلاس
                # ConnectionResetError) و قطع‌شدن خواندن پاسخ وسط راه
                last_error = LLMError(
                    f"خطای شبکه در تماس با NVIDIA Build (مدل '{model}'): {exc}",
                    status_code=None)
        raise last_error

    # ------------------------------------------------------------------
    def _safe_body(self, exc) -> str:
        """بدنه‌ی خطای سرور را کوتاه و بدون کلید برمی‌گرداند."""
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if self.api_key:
            text = text.replace(self.api_key, "***")
        return text.strip()[:_ERROR_BODY_MAX_CHARS] or "(بدون جزئیات)"

    @staticmethod
    def _parse_response(raw: str, model: str) -> dict:
        try:
            data = json.loads(raw)
        except ValueError:
            raise LLMError("پاسخ NVIDIA Build JSON معتبر نبود.") from None
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("پاسخ NVIDIA Build بدون choices برگشت.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # فرم چندبخشی بعضی مدل‌ها: فقط بخش‌های متنی را به هم می‌چسبانیم
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict))
        if not content:
            # بعضی مدل‌های استدلالی متن را در reasoning_content می‌گذارند
            content = message.get("reasoning_content") or ""
        return {
            "content": content,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model") or model,
        }


def get_default_client() -> LLMClient:
    """کلاینت پیش‌فرض با کلید resolveشده از config/محیط."""
    return LLMClient()
