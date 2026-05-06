"""Continual learning metrics: Gain, BWT, FWT."""


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
