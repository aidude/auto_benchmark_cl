"""
Blind Spectrum Monitoring CL Task.

Four sequential regimes simulating RF environment drift:
  0 — clean_band       (2 stable transmitters, low noise floor)
  1 — interference     (burst interference source added at unknown frequency)
  2 — sensor_drift     (systematic +5 MHz offset in reported peak frequencies)
  3 — reconfigured     (entirely new band allocation; prior occupancy map invalid)

Each regime = 30 scan episodes.
Each episode: LLM sees a list of detected spectrum peaks (frequency, power)
              → identifies the dominant signal type and the centre frequency
                of the strongest persistent transmitter.

Target: string in format "<signal_type>:<centre_freq_MHz>"
  signal_type ∈ {wifi, lte, radar, burst_noise, unknown}
Score: 1.0 if both type and frequency match (within ±5 MHz), else partial.
"""
import random
import re
from dataclasses import dataclass
from typing import Iterator

EPISODES_PER_REGIME = 30

# (label, transmitters, noise_floor_dBm, drift_offset_MHz, has_burst)
_REGIMES = [
    ("clean_band",    [("wifi", 2412), ("lte",  700)],  -95, 0,  False),
    ("interference",  [("wifi", 2412), ("lte",  700)],  -90, 0,  True ),
    ("sensor_drift",  [("wifi", 2412), ("lte",  700)],  -95, 5,  False),
    ("reconfigured",  [("radar", 5600), ("lte", 1800)], -92, 0,  False),
]


@dataclass
class Episode:
    task_id: int
    regime: str
    target: str
    text: str


def _dbm(rng: random.Random, peak_power: float, noise: float) -> float:
    return round(peak_power + rng.gauss(0, 2.5), 1)


def _format_prompt(peaks: list[tuple[float, float]], noise_floor: float, scan_id: int) -> str:
    peak_lines = "\n".join(
        f"  Peak {i+1}: {freq:.1f} MHz   {pwr:.1f} dBm"
        for i, (freq, pwr) in enumerate(sorted(peaks, key=lambda x: -x[1]))
    )
    return (
        "You are monitoring an RF spectrum band (0–3000 MHz).\n\n"
        f"Scan #{scan_id:03d}\n"
        f"Noise floor : {noise_floor:.1f} dBm\n"
        f"Detected peaks:\n{peak_lines}\n\n"
        "Identify the dominant persistent signal type and the centre frequency "
        "of the strongest transmitter.\n\n"
        "Respond in this exact format:\n"
        "  <signal_type>:<centre_freq_MHz>\n"
        "where signal_type is one of: wifi, lte, radar, burst_noise, unknown\n"
        "Example: wifi:2412"
    )


def _build_regime(seed: int, task_id: int, label: str,
                  transmitters: list[tuple[str, int]], noise_floor: float,
                  drift: int, has_burst: bool) -> list[Episode]:
    rng = random.Random(seed + task_id * 13)
    dominant_type, dominant_freq = transmitters[0]
    episodes = []

    for scan_i in range(EPISODES_PER_REGIME):
        peaks: list[tuple[float, float]] = []

        # Stable transmitters
        for sig_type, freq in transmitters:
            reported_freq = freq + drift + rng.randint(-2, 2)
            power = -55.0 + rng.gauss(0, 3)
            peaks.append((float(reported_freq), power))

        # Spurious noise peaks
        for _ in range(rng.randint(1, 4)):
            peaks.append((rng.uniform(100, 2900), noise_floor + rng.uniform(1, 8)))

        # Burst interference (regime 1 only)
        if has_burst and rng.random() < 0.5:
            burst_freq = rng.choice([1575, 915, 433])
            peaks.append((float(burst_freq), -62.0 + rng.gauss(0, 5)))
            target_type = "burst_noise"
            target_freq = burst_freq
        else:
            target_type = dominant_type
            target_freq = dominant_freq + drift

        target = f"{target_type}:{target_freq}"
        text   = _format_prompt(peaks, noise_floor, scan_i + 1)
        episodes.append(Episode(task_id=task_id, regime=label, target=target, text=text))

    return episodes


def _parse(response: str) -> tuple[str, int] | None:
    m = re.search(r"(wifi|lte|radar|burst_noise|unknown)\s*:?\s*(\d+)", response.lower())
    if m:
        return m.group(1), int(m.group(2))
    return None


class BlindSpectrumMonitoringTask:
    """Four-regime CL task with RF environment drift."""

    def __init__(self, seed: int = 42):
        self._regimes = self._build(seed)

    def _build(self, seed: int) -> list[list[Episode]]:
        return [
            _build_regime(seed, tid, label, txs, noise, drift, burst)
            for tid, (label, txs, noise, drift, burst) in enumerate(_REGIMES)
        ]

    @property
    def num_tasks(self) -> int:
        return len(self._regimes)

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from self._regimes[task_id]

    @staticmethod
    def score(response: str, target: str) -> float:
        """1.0 if type correct AND freq within ±5 MHz; 0.5 if only type correct; 0.0 otherwise."""
        parsed = _parse(response)
        if parsed is None:
            return 0.0
        pred_type, pred_freq = parsed
        target_type, target_freq = target.split(":")
        if pred_type != target_type:
            return 0.0
        return 1.0 if abs(pred_freq - int(target_freq)) <= 5 else 0.5
