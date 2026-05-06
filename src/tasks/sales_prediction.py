"""
Sales Prediction CL Task.

Four sequential regimes simulating distribution shift:
  0 — Stable growth   (base ~1000, small +trend, low noise)
  1 — Holiday spike   (base ~1800, strong +trend, high noise)
  2 — Market dip      (base ~700,  negative trend)
  3 — Recovery        (base ~1200, moderate growth)

Each regime = 30 episodes.
Each episode: LLM sees 14-day sales history → predicts day 15.
"""
import random
import re
from dataclasses import dataclass
from typing import Iterator


CONTEXT_LEN = 14
EPISODES_PER_REGIME = 30

_REGIMES = [
    # (base,   trend,  noise,  label)
    (1000.0,   5.0,   50.0,  "stable_growth"),
    (1800.0,  10.0,  180.0,  "holiday_spike"),
    ( 700.0,  -8.0,   70.0,  "market_dip"),
    (1200.0,   6.0,   90.0,  "recovery"),
]


@dataclass
class Episode:
    task_id: int
    regime: str
    history: list[tuple[int, float]]
    target: float
    text: str


def _make_series(n: int, base: float, trend: float, noise: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    val, series = base, []
    for _ in range(n):
        val = max(0.0, val + trend + rng.gauss(0, noise))
        series.append(round(val, 1))
    return series


def _format_prompt(history: list[tuple[int, float]]) -> str:
    rows = "\n".join(f"  Day {d}: {int(s)}" for d, s in history)
    next_day = history[-1][0] + 1
    return (
        "You are analyzing retail sales data.\n"
        f"Recent daily sales (units sold):\n{rows}\n\n"
        f"Predict the sales for Day {next_day}. "
        "Respond with a single integer and nothing else."
    )


def _parse(response: str) -> float | None:
    # Take the last 3-6 digit number — handles chain-of-thought answers where
    # the final prediction appears after reasoning text that contains history values.
    matches = re.findall(r"\b(\d{3,6})\b", response.replace(",", ""))
    return float(matches[-1]) if matches else None


class SalesPredictionTask:
    """Four-regime continual learning task with distribution shift."""

    def __init__(self, seed: int = 42):
        self._regimes = self._build(seed)

    def _build(self, seed: int) -> list[list[Episode]]:
        regimes = []
        for task_id, (base, trend, noise, label) in enumerate(_REGIMES):
            series = _make_series(
                EPISODES_PER_REGIME + CONTEXT_LEN,
                base, trend, noise, seed=seed + task_id,
            )
            episodes = []
            for i in range(EPISODES_PER_REGIME):
                window = series[i : i + CONTEXT_LEN]
                target = series[i + CONTEXT_LEN]
                history = [(i + j + 1, v) for j, v in enumerate(window)]
                episodes.append(Episode(
                    task_id=task_id,
                    regime=label,
                    history=history,
                    target=target,
                    text=_format_prompt(history),
                ))
            regimes.append(episodes)
        return regimes

    @property
    def num_tasks(self) -> int:
        return len(self._regimes)

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from self._regimes[task_id]

    @staticmethod
    def score(response: str, target: float) -> float:
        """Reward in [0, 1]. 1 = perfect, 0 = ≥100% relative error or unparseable."""
        pred = _parse(response)
        if pred is None:
            return 0.0
        return max(0.0, 1.0 - abs(pred - target) / max(target, 1.0))
