from typing import Any
from src.systems.base import BaseSystem
from src.utils.llm import complete


class ICLSystem(BaseSystem):
    """In-context learning — carries a rolling example buffer in the prompt."""

    def __init__(self, model: str, max_examples: int = 5, system_prompt: str = "",
                 llm_kwargs: dict | None = None):
        # max_examples=5: sales_prediction observations are ~1200-1800 tokens each;
        # 10 examples pushed input tokens to 14k-18k per call and hit context limits
        # on smaller models (e.g. qwen3.6-27b, LFM). 5 keeps the buffer under ~9k tokens.
        self.model = model
        self.max_examples = max_examples
        self.system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs or {}
        self._examples: list[dict] = []

    def act(self, observation: dict) -> Any:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for ex in self._examples:
            messages.append({"role": "user",      "content": ex["observation"]})
            messages.append({"role": "assistant",  "content": ex["action"]})
        messages.append({"role": "user", "content": observation["text"]})
        return complete(self.model, messages, **self.llm_kwargs)

    # Store only the TAIL of the response, not the head. Step-by-step reasoning
    # models (kimi, minimax, qwen3, etc.) emit their final answer at the end of
    # a long chain-of-thought. Storing the first N chars captures "Let me think
    # step by step..." instead of the answer; storing the last N chars captures
    # the actual prediction that future episodes should learn from.
    _MAX_ACTION_CHARS = 200

    def update(self, observation: dict, action: Any, reward: float) -> None:
        stored_action = str(action)[-self._MAX_ACTION_CHARS:]
        self._examples.append({"observation": observation["text"], "action": stored_action})
        if len(self._examples) > self.max_examples:
            self._examples.pop(0)

    def reset(self) -> None:
        self._examples.clear()
