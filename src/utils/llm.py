import time
import litellm
from dotenv import load_dotenv
from src.utils.logger import get_logger, log_llm_call

load_dotenv()

litellm.drop_params = True  # ignore params unsupported by a provider

_log = get_logger(__name__)


def _call(model: str, messages: list[dict], **kwargs) -> tuple[str, dict]:
    """Core call — logs every request to the LLM audit file."""
    _log.debug("→ %s  messages=%d", model, len(messages))
    t0 = time.monotonic()

    response = litellm.completion(model=model, messages=messages, **kwargs)

    latency_ms = (time.monotonic() - t0) * 1000
    text = response.choices[0].message.content
    usage = dict(response.usage) if response.usage else {}

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    log_llm_call(
        model=model,
        messages=messages,
        response_text=text,
        usage=usage,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
    _log.debug(
        "← %s  tokens=%s  latency=%.0fms  cost=$%.5f",
        model, usage.get("total_tokens", "?"), latency_ms, cost or 0,
    )
    return text, usage


def complete(model: str, messages: list[dict], **kwargs) -> str:
    """Call any model via LiteLLM. Returns assistant text."""
    text, _ = _call(model, messages, **kwargs)
    return text


def complete_with_usage(model: str, messages: list[dict], **kwargs) -> tuple[str, dict]:
    """Returns (text, usage_dict) for token tracking."""
    return _call(model, messages, **kwargs)
