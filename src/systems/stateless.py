from typing import Any
from src.systems.base import BaseSystem
from src.utils.llm import complete


class StatelessSystem(BaseSystem):
    """No memory — pure prompt → response each turn. Episodes are order-independent."""

    parallel_safe = True

    def __init__(self, model: str, system_prompt: str = "", llm_kwargs: dict | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs or {}

    def act(self, observation: dict) -> Any:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": observation["text"]})
        return complete(self.model, messages, **self.llm_kwargs)

    def update(self, observation: dict, action: Any, reward: float) -> None:
        pass

    def reset(self) -> None:
        pass
