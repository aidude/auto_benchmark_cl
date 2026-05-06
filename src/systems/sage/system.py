# SAGESystem — neuromodulated continual learning agent.
# Implementation TBD; scaffold wires together the sub-modules.
from typing import Any
from src.systems.base import BaseSystem


class SAGESystem(BaseSystem):
    """Stub — to be implemented with neuromod + memory + lyapunov."""

    def act(self, observation: dict) -> Any:
        raise NotImplementedError

    def update(self, observation: dict, action: Any, reward: float) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
