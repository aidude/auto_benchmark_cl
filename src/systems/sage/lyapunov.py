# Lyapunov-based forgetting early-warning signal.
# Tracks performance variance to detect catastrophic forgetting before it peaks.
# Implementation TBD.


class ForgettingMonitor:
    def __init__(self, window: int = 20, threshold: float = 0.2):
        self.window = window
        self.threshold = threshold
        self._scores: list[float] = []

    def record(self, score: float) -> None:
        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores.pop(0)

    @property
    def is_forgetting(self) -> bool:
        if len(self._scores) < 2:
            return False
        variance = sum((s - sum(self._scores) / len(self._scores)) ** 2 for s in self._scores)
        return variance > self.threshold
