from typing import Any
from src.systems.base import BaseSystem
from src.utils.llm import complete


class StatelessSystem(BaseSystem):
    """No memory — pure prompt → response each turn."""

    def __init__(self, model: str, system_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt

    def act(self, observation: dict) -> Any:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": observation["text"]})
        return complete(self.model, messages)

    def update(self, observation: dict, action: Any, reward: float) -> None:
        pass

    def reset(self) -> None:
        pass
