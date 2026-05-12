"""Continual learning metrics: Gain, BWT, FWT."""
from src.utils.logger import get_logger

_log = get_logger(__name__)


def gain(scores: list[float], baseline: list[float]) -> float:
    """Average performance improvement over a stateless baseline."""
    assert len(scores) == len(baseline)
    return sum(s - b for s, b in zip(scores, baseline)) / len(scores)


def bwt(perf_matrix: list[list[float]]) -> float:
    """Backward Transfer: how much learning task j affects earlier task i (i < j).

    perf_matrix[i][j] = performance on task i after training on task j.
    """
    n = len(perf_matrix)
    total = sum(
        perf_matrix[i][n - 1] - perf_matrix[i][i]
        for i in range(n - 1)
    )
    return total / (n - 1) if n > 1 else 0.0


def fwt(perf_matrix: list[list[float]], random_baseline: list[float]) -> float:
    """Forward Transfer: influence of past learning on future task performance."""
    n = len(perf_matrix)
    total = sum(
        perf_matrix[i - 1][i] - random_baseline[i]
        for i in range(1, n)
    )
    return total / (n - 1) if n > 1 else 0.0


def compute_cl_metrics(
    perf_matrix: list[list[float | None]],
    stateless_scores: list[float] | None = None,
    stateless_bwt:    float | None = None,
) -> dict:
    """Compute BWT, FWT, forgetting profile, and floor-relative metrics.

    perf_matrix[i][j] = score on regime i after training through regime j.
    Upper triangular + diagonal only. Lower triangle = None.

    stateless_bwt: BWT from the stateless run on the same model + task.
      This is the TASK DIFFICULTY FLOOR — minimum BWT achievable by any system
      because regime distribution shifts cause apparent forgetting even with zero
      memory. A stateless model cannot genuinely forget (no state), so its BWT
      measures only task-inherent regime discontinuity.
      bwt_gain_over_floor = system_bwt - stateless_bwt.
        > 0  →  system forgets LESS than chance  →  real positive CL signal
        < 0  →  system forgets MORE than chance  →  memory is hurting

    stateless_scores: diagonal from stateless run. Used as FWT baseline.
      FWT answers: does training on past regimes help future performance
      BEYOND what a memoryless model achieves?
    """
    T = len(perf_matrix)

    try:
        bwt_val = round(bwt(perf_matrix), 4)
    except Exception as exc:
        _log.warning("BWT computation failed: %s", exc)
        bwt_val = None

    fwt_val = None
    if stateless_scores and len(stateless_scores) == T:
        try:
            fwt_val = round(fwt(perf_matrix, stateless_scores), 4)
        except Exception as exc:
            _log.warning("FWT computation failed: %s", exc)

    # forgetting[i] = R[i][i] - R[i][T-1]
    # Positive = forgot by end of training. Negative = improved (backward transfer).
    forgetting: dict[str, float] = {}
    for i in range(T):
        r_ii = perf_matrix[i][i]
        r_iT = perf_matrix[i][T - 1]
        if r_ii is not None and r_iT is not None:
            forgetting[f"regime_{i}"] = round(r_ii - r_iT, 4)

    max_f = max(forgetting.values()) if forgetting else None

    bwt_gain = None
    if bwt_val is not None and stateless_bwt is not None:
        bwt_gain = round(bwt_val - stateless_bwt, 4)

    def _bwt_label(v: float | None) -> str:
        if v is None:  return "insufficient_data"
        if v < -0.02:  return "forgetting"
        if v >  0.02:  return "positive_transfer"
        return "stable"

    def _floor_label(v: float | None) -> str:
        if v is None:  return "no_baseline"
        if v >  0.01:  return "better_than_floor"
        if v < -0.01:  return "worse_than_floor"
        return "same_as_floor"

    return {
        "bwt":                   bwt_val,
        "bwt_interpretation":    _bwt_label(bwt_val),
        "fwt":                   fwt_val,
        "bwt_gain_over_floor":   bwt_gain,
        "bwt_vs_floor":          _floor_label(bwt_gain),
        "forgetting_profile":    forgetting,
        "max_forgetting":        max_f,
        "any_forgetting":        any(v > 0.05 for v in forgetting.values()),
    }
