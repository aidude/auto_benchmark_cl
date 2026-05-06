from typing import Any
from src.systems.base import BaseSystem
from src.utils.llm import complete


class ICLSystem(BaseSystem):
    """In-context learning — carries a rolling example buffer in the prompt."""

    def __init__(self, model: str, max_examples: int = 10, system_prompt: str = "",
                 llm_kwargs: dict | None = None):
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

    def update(self, observation: dict, action: Any, reward: float) -> None:
        self._examples.append({"observation": observation["text"], "action": str(action)})
        if len(self._examples) > self.max_examples:
            self._examples.pop(0)

    def reset(self) -> None:
        self._examples.clear()
