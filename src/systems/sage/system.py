# SAGESystem — neuromodulated continual learning agent.
# Implementation TBD; scaffold wires together the sub-modules.
from src.systems.base import BaseSystem


class SAGESystem(BaseSystem):
    """Stub — to be implemented with neuromod + memory + lyapunov."""

    def act(self, observation: dict):
        raise NotImplementedError

    def update(self, observation: dict, action, reward: float) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
