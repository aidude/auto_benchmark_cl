"""
Cohort Studies CL Task.

Four sequential regimes simulating clinical study design drift:
  0 — oncology_early    (early-stage cancer, high survival rates)
  1 — oncology_late     (late-stage cancer, low survival rates, shifted covariate coding)
  2 — cardiology_acute  (acute cardiac events, moderate survival)
  3 — cardiology_chronic(chronic cardiac disease, high variability, redefined time horizons)

Each regime = 30 cohort episodes.
Each episode: LLM sees patient group characteristics and study metadata
              → estimates 24-month survival probability as a value in [0.00, 1.00].
Scoring: reward = max(0, 1 - |pred - target| / 0.5)  (full credit within ±0.05)
"""
import math
import random
import re
from dataclasses import dataclass
from typing import Iterator

EPISODES_PER_REGIME = 30

_REGIMES = [
    # (label,               base_survival, age_penalty, stage_penalty, noise)
    ("oncology_early",       0.82,          0.003,        0.10,         0.05),
    ("oncology_late",        0.38,          0.005,        0.18,         0.08),
    ("cardiology_acute",     0.65,          0.004,        0.12,         0.07),
    ("cardiology_chronic",   0.71,          0.002,        0.08,         0.10),
]

_STAGE_LABELS = {
    "oncology_early":    ["Stage I", "Stage II"],
    "oncology_late":     ["Stage III", "Stage IV"],
    "cardiology_acute":  ["STEMI", "NSTEMI"],
    "cardiology_chronic": ["Class II HF", "Class III HF"],
}

_COMORBIDITIES = ["diabetes", "hypertension", "obesity", "COPD", "CKD", "none"]


@dataclass
class Episode:
    task_id: int
    regime: str
    target: float
    text: str


def _format_prompt(
    regime: str, cohort_size: int, mean_age: float, stage: str,
    comorbidities: list[str], followup_months: int, study_note: str,
) -> str:
    comorbidity_str = ", ".join(comorbidities) if comorbidities else "none reported"
    domain = "oncology" if "oncology" in regime else "cardiology"
    return (
        f"You are analysing a {domain} cohort study.\n\n"
        f"Study note     : {study_note}\n"
        f"Cohort size    : {cohort_size} patients\n"
        f"Mean age       : {mean_age:.1f} years\n"
        f"Disease stage  : {stage}\n"
        f"Comorbidities  : {comorbidity_str}\n"
        f"Follow-up      : {followup_months} months\n\n"
        "Estimate the probability that a patient in this cohort survives to the "
        f"{followup_months}-month mark.\n\n"
        "Respond with a single decimal between 0.00 and 1.00 (e.g. 0.73). "
        "No explanation."
    )


_STUDY_NOTES = {
    "oncology_early":    "Standard RECIST criteria applied; remission confirmed by imaging.",
    "oncology_late":     "Palliative intent; variable chemotherapy regimens across sites.",
    "cardiology_acute":  "Revascularisation within 12h; LVEF recorded at discharge.",
    "cardiology_chronic": "NYHA classification used; BNP threshold redefined at 200 pg/mL.",
}


def _build_episode(rng: random.Random, task_id: int, regime: str,
                   base: float, age_pen: float, stage_pen: float, noise: float) -> Episode:
    cohort_size = rng.randint(40, 400)
    mean_age    = round(rng.uniform(48, 76), 1)
    stage       = rng.choice(_STAGE_LABELS[regime])
    comorbids   = rng.sample(_COMORBIDITIES, k=rng.randint(0, 2))
    comorbid_pen = 0.04 * len([c for c in comorbids if c != "none"])
    followup    = 24

    target = base - age_pen * max(0, mean_age - 55) - stage_pen - comorbid_pen
    target = round(min(0.99, max(0.05, target + rng.gauss(0, noise))), 2)

    text = _format_prompt(regime, cohort_size, mean_age, stage, comorbids,
                          followup, _STUDY_NOTES[regime])
    return Episode(task_id=task_id, regime=regime, target=target, text=text)


def _parse(response: str) -> float | None:
    m = re.search(r"\b(0\.\d{1,2}|1\.0{1,2})\b", response.strip())
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{1,2})\s*%", response)
    if m:
        return float(m.group(1)) / 100.0
    return None


class CohortStudiesTask:
    """Four-regime CL task simulating clinical study design drift."""

    def __init__(self, seed: int = 42):
        self._regimes = self._build(seed)

    def _build(self, seed: int) -> list[list[Episode]]:
        regimes = []
        for task_id, (label, base, age_pen, stage_pen, noise) in enumerate(_REGIMES):
            rng = random.Random(seed + task_id * 31)
            episodes = [
                _build_episode(rng, task_id, label, base, age_pen, stage_pen, noise)
                for _ in range(EPISODES_PER_REGIME)
            ]
            regimes.append(episodes)
        return regimes

    @property
    def num_tasks(self) -> int:
        return len(self._regimes)

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from self._regimes[task_id]

    @staticmethod
    def score(response: str, target: float) -> float:
        """Reward = max(0, 1 - |pred - target| / 0.5). Full credit within ±0.05."""
        pred = _parse(response)
        if pred is None:
            return 0.0
        return max(0.0, 1.0 - abs(pred - target) / 0.5)
