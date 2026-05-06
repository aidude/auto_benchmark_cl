# Episodic buffer + consolidation logic.
# Implementation TBD.
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Episode:
    observation: dict
    action: Any
    reward: float
    salience: float = 0.0


class EpisodicBuffer:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._buffer: list[Episode] = []

    def add(self, episode: Episode) -> None:
        self._buffer.append(episode)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
