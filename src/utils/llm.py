import random
import re
import time

import litellm
from dotenv import load_dotenv

from src.utils.logger import get_logger, log_llm_call

load_dotenv()

# Strip <think>...</think> blocks that reasoning models (qwen3, deepseek-r1, etc.)
# embed in their content. Without this, ICL stores the full chain-of-thought in the
# example buffer — 5 examples × ~3k thinking tokens = 15k tokens of useless prompt noise.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

litellm.drop_params = True  # silently ignore params unsupported by a provider

_log = get_logger(__name__)

_DEFAULT_TIMEOUT = 60     # per-call timeout (s); overridden via model params in models.yaml
_MAX_RETRIES     = 3      # attempts after the first failure
_RETRY_BASE_S    = 2.0    # initial backoff delay; doubles each retry + jitter

# Transient errors worth retrying — anything else is a hard failure (auth, bad request, etc.)
_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.APIError,           # catches 5xx upstream errors
    litellm.exceptions.InternalServerError,
)

_PROVIDER_MAP = {
    "openrouter/": "openrouter",
    "claude":      "anthropic",
    "anthropic":   "anthropic",
}
_ENDPOINT_MAP = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "anthropic":  "https://api.anthropic.com/v1/messages",
    "openai":     "https://api.openai.com/v1/chat/completions",
}


def _provider(model: str) -> str:
    for prefix, name in _PROVIDER_MAP.items():
        if model.startswith(prefix) or prefix in model:
            return name
    return "openai"


def complete(model: str, messages: list[dict], **kwargs) -> str:
    """Call any model via LiteLLM with automatic retry on transient failures.

    Timeout defaults to _DEFAULT_TIMEOUT but can be overridden per-model via
    params.timeout in models.yaml.
    """
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _call_once(model, messages, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            delay = _RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 1)
            _log.warning(
                "Transient error on attempt %d/%d for %s (%s: %s) — retrying in %.1fs",
                attempt + 1, _MAX_RETRIES + 1, model,
                type(exc).__name__, str(exc)[:120], delay,
            )
            time.sleep(delay)

    _log.error("All %d attempts failed for %s: %s", _MAX_RETRIES + 1, model, last_exc)
    raise last_exc


def _call_once(model: str, messages: list[dict], **kwargs) -> str:
    """Single (non-retried) LiteLLM call with logging."""
    _log.debug("→ %s  messages=%d", model, len(messages))
    t0 = time.monotonic()

    response = litellm.completion(model=model, messages=messages, **kwargs)

    latency_ms = (time.monotonic() - t0) * 1000
    msg = response.choices[0].message
    # Reasoning models (R1, o1) may return content=None; fall back to reasoning_content.
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = _THINK_RE.sub("", text).strip()
    usage = dict(response.usage) if response.usage else {}

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    provider = _provider(model)
    log_llm_call(
        model=model,
        provider=provider,
        endpoint=_ENDPOINT_MAP.get(provider, "unknown"),
        messages=messages,
        response_text=text,
        usage=usage,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
    _log.debug(
        "← %s  tokens=%s  %.0fms  $%.5f",
        model, usage.get("total_tokens", "?"), latency_ms, cost or 0,
    )
    return text
