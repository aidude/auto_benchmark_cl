import time
import litellm
from dotenv import load_dotenv
from src.utils.logger import get_logger, log_llm_call

load_dotenv()

litellm.drop_params = True  # silently ignore params unsupported by a provider

_log = get_logger(__name__)

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
    """Call any model via LiteLLM. Returns assistant text."""
    _log.debug("→ %s  messages=%d", model, len(messages))
    t0 = time.monotonic()

    response = litellm.completion(model=model, messages=messages, **kwargs)

    latency_ms = (time.monotonic() - t0) * 1000
    msg = response.choices[0].message
    # Reasoning models (R1, o1) may return content=None; fall back to reasoning_content.
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
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
