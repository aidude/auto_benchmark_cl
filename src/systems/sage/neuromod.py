# Neuromodulatory signals: DA (surprise/reward), NE (novelty/arousal),
# ACh (uncertainty/attention), 5-HT (stability/patience).
# Implementation TBD.
from dataclasses import dataclass, field


@dataclass
class NeuromodState:
    da: float = 0.0   # dopamine
    ne: float = 0.0   # norepinephrine
    ach: float = 0.0  # acetylcholine
    sht: float = 0.0  # serotonin
