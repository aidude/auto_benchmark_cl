import concurrent.futures as _cf
import random
import re
import time

import litellm
from dotenv import load_dotenv

from src.utils.logger import get_logger, log_llm_call

load_dotenv()

# Reasoning models (qwen3, deepseek-r1, sarvam, etc.) wrap chain-of-thought in
# <think>...</think>. Without stripping, ICL stores the full trace in the example
# buffer — 5 examples x ~3k thinking tokens = 15k tokens of noise per call.
#
# Safe strip strategy:
#   1. Remove <think>...</think>; keep what remains.
#   2. If nothing remains (model buried the answer inside the block), grab
#      everything after </think> — some models emit the final answer there.
#   3. Last resort: return original so callers never receive an empty string.
_THINK_RE    = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_AFTER_THINK = re.compile(r"</think>(.*)",       re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    if stripped:
        return stripped
    m = _AFTER_THINK.search(text)
    if m:
        after = m.group(1).strip()
        if after:
            return after
    _log.warning("think-strip produced empty response; returning raw text (len=%d)", len(text))
    return text.strip()

litellm.drop_params = True   # silently ignore params unsupported by a provider
litellm.modify_params = True  # allow litellm to adapt params (e.g. max_tokens → max_completion_tokens)

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
    "sarvam/":     "sarvam",
}
_ENDPOINT_MAP = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "anthropic":  "https://api.anthropic.com/v1/messages",
    "openai":     "https://api.openai.com/v1/chat/completions",
    "sarvam":     "https://api.sarvam.ai/v1/chat/completions",
}


def _provider(model: str) -> str:
    for prefix, name in _PROVIDER_MAP.items():
        if model.startswith(prefix) or prefix in model:
            return name
    return "openai"


def extract_answer(response) -> str:
    """Extract final answer from a LiteLLM response object.

    Closed reasoning (gpt-5.4, o3, o4): chain is server-side and hidden.
    message.content is always the clean final answer. No special handling needed.

    Open reasoning (kimi-k2.6, minimax-m2.7, qwen3, deepseek-r1): chain is in
    the response stream. LiteLLM separates it into message.reasoning_content.
    message.content = the final answer after the chain. _strip_thinking() removes
    any residual <think> tags. If content is empty (chain truncated by max_tokens
    or model failed to conclude): fall back to the last paragraph of
    reasoning_content and log a warning — this signals a configuration problem.
    """
    message = response.choices[0].message
    answer = _strip_thinking((message.content or "").strip())

    if not answer:
        reasoning = getattr(message, "reasoning_content", None) or ""
        if reasoning.strip():
            paragraphs = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
            answer = paragraphs[-1] if paragraphs else reasoning.strip()
            _log.warning(
                "Empty content — used reasoning_content fallback for %s. "
                "This usually means max_tokens truncated the chain. "
                "Ensure max_tokens is NOT capped for open reasoning models.",
                getattr(response, "model", "?"),
            )
    return answer


def _call_with_hard_timeout(call_kwargs: dict, timeout_s: int):
    """Run litellm.completion() in a thread with a hard OS-level timeout.

    litellm's own timeout param is not always respected by all providers
    (notably sarvam). This wrapper guarantees the call dies at timeout_s
    seconds regardless, using concurrent.futures thread cancellation.
    """
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(litellm.completion, **call_kwargs)
        try:
            return future.result(timeout=timeout_s)
        except _cf.TimeoutError:
            raise TimeoutError(
                f"Hard timeout: {call_kwargs.get('model', '?')} exceeded {timeout_s}s. "
                f"Provider ignored litellm timeout param."
            )


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
    # timeout is popped and passed as an explicit named arg so it is never
    # silently swallowed when buried inside **kwargs on older litellm builds.
    # reasoning_effort stays in kwargs; supporting models (kimi, minimax, o3)
    # use it, and drop_params removes it for all others.
    timeout = kwargs.pop("timeout", _DEFAULT_TIMEOUT)
    _log.debug("→ %s  messages=%d", model, len(messages))
    t0 = time.monotonic()

    response = _call_with_hard_timeout(
        {"model": model, "messages": messages, **kwargs}, timeout
    )

    latency_ms = (time.monotonic() - t0) * 1000
    text = extract_answer(response)
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
