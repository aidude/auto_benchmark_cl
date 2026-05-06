import os
import litellm
from dotenv import load_dotenv

load_dotenv()

litellm.drop_params = True  # silently ignore params unsupported by a provider


def complete(model: str, messages: list[dict], **kwargs) -> str:
    """Call any model via LiteLLM. Returns assistant text."""
    response = litellm.completion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def complete_with_usage(
    model: str, messages: list[dict], **kwargs
) -> tuple[str, dict]:
    """Returns (text, usage_dict) for token tracking."""
    response = litellm.completion(model=model, messages=messages, **kwargs)
    text = response.choices[0].message.content
    usage = dict(response.usage) if response.usage else {}
    return text, usage
