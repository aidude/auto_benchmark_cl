from abc import ABC, abstractmethod
from typing import Any


class BaseSystem(ABC):
    """Abstract base for all CL agent systems."""

    @abstractmethod
    def act(self, observation: dict) -> Any:
        """Produce an action / response given the current observation."""

    @abstractmethod
    def update(self, observation: dict, action: Any, reward: float) -> None:
        """Update internal state after receiving feedback."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state; called between tasks."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
