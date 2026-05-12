"""
Mini Sales Prediction task — 5 episodes per regime instead of 30.
Used for quick smoke tests and token-budget validation before full runs.
"""
from src.tasks.sales_prediction import SalesPredictionTask as _Full, Episode
from typing import Iterator

_MINI_EPISODES = 5


class SalesPredictionMiniTask(_Full):
    """5-episode-per-regime variant of SalesPredictionTask for fast validation."""

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from list(self._regimes[task_id])[:_MINI_EPISODES]
