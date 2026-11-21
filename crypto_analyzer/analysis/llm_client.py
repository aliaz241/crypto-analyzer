"""
analysis/llm_client.py — Minimal OpenAI-compatible client for NVIDIA Build

Written with the standard library only (urllib) so it works on Termux with no
new dependencies; this package deliberately installs no SDK.

- Base URL: https://integrate.api.nvidia.com/v1 (OpenAI-compatible
  chat/completions endpoints).
- API key (in priority order):
  1) The config.NVIDIA_API_KEY constant, if defined in config and non-empty
     (it has been in config.py since the integration step and defaults to "";
     this module reads it with getattr),
  2) The NVIDIA_API_KEY environment variable,
  3) None — meaning the LLM layer is disabled and the caller must skip silently.
- The key is never printed in error messages, logs or output.
- The default timeout is long (600 seconds): reasoning models on this service
  are slow, and slowness is acceptable for this layer; a cut-off response is not.
- Retry policy (inside chat itself): at most 3 attempts in total. On
  connection-level errors (connection cut while reading/RemoteDisconnected,
  connection errors and socket timeouts) and on HTTP 5xx and 429, it retries
  with a stepped backoff of 10 and 30 seconds; because a live test on
  2026-11-08 showed the primary model on this network repeatedly gets cut off
  midway on long generations, and one analysis run should not depend on every
  single one of its calls surviving for success. On 410 (model retired/removed
  from the catalog) it never retries and LLMRetiredError is raised on the first
  attempt, so the caller immediately understands the model ID is dead, not that
  the service is busy; other 4xx errors are also raised on the first attempt
  without retry.
"""

import http.client
import json
import os
import time
import urllib.error
import urllib.request

try:
    import config as _config
except Exception:  # pragma: no cover - when analysis/ itself is on sys.path
    _config = None

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TIMEOUT_SECONDS = 600
_MAX_ATTEMPTS = 3                          # Total attempts inside one chat()
_RETRY_BACKOFFS_SECONDS = (10.0, 30.0)     # Pause before attempts 2 and 3
_ERROR_BODY_MAX_CHARS = 300


class LLMError(Exception):
    """LLM-layer error; status_code is the HTTP code (None for a network error)."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class LLMRetiredError(LLMError):
    """The model has been removed/retired from the NVIDIA catalog (HTTP 410) — retry forbidden."""


def resolve_api_key():
    """
    Returns the NVIDIA key according to the order documented in the module
    docstring, or None if it is not set anywhere.
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
    """Simple chat/completions client for NVIDIA Build (stdlib only)."""

    def __init__(self, api_key=None, base_url=BASE_URL):
        # api_key=None means 'resolve it yourself'; an empty string means deliberately no key.
        self.api_key = resolve_api_key() if api_key is None else (api_key or None)
        self.base_url = (base_url or BASE_URL).rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    def chat(self, model, messages, temperature=0.2, max_tokens=16384,
             timeout=None):
        """
        One chat/completions call.

        Output: {"content": str, "finish_reason": str|None,
                "usage": dict|None, "model": str}
        On any HTTP/network error, LLMError (or LLMRetiredError for 410)
        is raised; the caller (llm_engine) treats all of them fail-safe.
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
                # Stepped backoff between attempts: 10 seconds, then 30 seconds
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
                    # The model is dead; retrying only wastes time
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
                    # Other 4xx errors are definitive; no retry
                    raise last_error from None
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as exc:
                # Connection-level error — including RemoteDisconnected (a subclass of
                # ConnectionResetError) and the response read being cut off midway
                last_error = LLMError(
                    f"خطای شبکه در تماس با NVIDIA Build (مدل '{model}'): {exc}",
                    status_code=None)
        raise last_error

    # ------------------------------------------------------------------
    def _safe_body(self, exc) -> str:
        """Returns the server error body, shortened and without the key."""
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
            # Multipart form of some models: we only concatenate the text parts
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict))
        if not content:
            # Some reasoning models put the text in reasoning_content
            content = message.get("reasoning_content") or ""
        return {
            "content": content,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model") or model,
        }


def get_default_client() -> LLMClient:
    """Default client with the key resolved from config/environment."""
    return LLMClient()
