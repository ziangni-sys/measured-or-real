from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaxTCalibration:
    critical_value: float
    effective_rank: int
    min_eigenvalue: float
    active_statistics: int


def pair_indices(n_models: int) -> list[tuple[int, int]]:
    if n_models < 2:
        raise ValueError("at least two models are required")
    return list(itertools.combinations(range(n_models), 2))


def gaussian_max_t(
    centered: np.ndarray,
    alpha: float,
    repetitions: int,
    seed: int,
    batch_size: int = 2048,
) -> MaxTCalibration:
    """Calibrate a Gaussian max-t statistic from item-level contrasts.

    Each row of ``centered`` is one centered pair-by-protocol item contrast.
    Simulating from its estimated correlation matrix is the parametric
    Gaussian bootstrap for the joint limiting distribution of the t statistics.
    """
    centered = np.asarray(centered, dtype=np.float64)
    if centered.ndim != 2 or centered.shape[1] < 2:
        raise ValueError("centered must have shape (statistics, items) with at least two items")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if repetitions <= 0 or batch_size <= 0:
        raise ValueError("bootstrap repetitions and batch size must be positive")

    sums_sq = np.einsum("ij,ij->i", centered, centered)
    active = sums_sq > np.finfo(np.float64).eps
    if not np.any(active):
        raise ValueError("all pair-by-protocol item contrasts have zero variance")

    x = centered[active]
    scale = np.sqrt(np.einsum("ij,ij->i", x, x))
    corr = (x @ x.T) / np.outer(scale, scale)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    min_eigenvalue = float(eigenvalues.min())
    tolerance = max(corr.shape) * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0)
    keep = eigenvalues > tolerance
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])

    rng = np.random.default_rng(seed)
    maxima = np.empty(repetitions, dtype=np.float64)
    offset = 0
    while offset < repetitions:
        count = min(batch_size, repetitions - offset)
        draws = rng.standard_normal((count, factor.shape[1])) @ factor.T
        maxima[offset : offset + count] = np.max(np.abs(draws), axis=1)
        offset += count

    critical = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
    return MaxTCalibration(
        critical_value=critical,
        effective_rank=int(keep.sum()),
        min_eigenvalue=min_eigenvalue,
        active_statistics=int(active.sum()),
    )


def rank_sets_from_pair_intervals(
    n_models: int,
    pairs: list[tuple[int, int]],
    lower: np.ndarray,
    upper: np.ndarray,
    point_scores: np.ndarray,
) -> list[dict[str, int | float]]:
    """Convert simultaneous intervals for A-B into simultaneous rank sets."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    point_scores = np.asarray(point_scores, dtype=np.float64)
    if lower.shape != (len(pairs),) or upper.shape != (len(pairs),):
        raise ValueError("one lower and upper bound is required per pair")
    if point_scores.shape != (n_models,):
        raise ValueError("point_scores must contain one score per model")
    if np.any(lower > upper):
        raise ValueError("lower bounds cannot exceed upper bounds")

    definitely_above = np.zeros(n_models, dtype=np.int64)
    definitely_below = np.zeros(n_models, dtype=np.int64)
    for k, (a, b) in enumerate(pairs):
        if lower[k] > 0.0:
            definitely_below[a] += 1
            definitely_above[b] += 1
        elif upper[k] < 0.0:
            definitely_above[a] += 1
            definitely_below[b] += 1

    rows: list[dict[str, int | float]] = []
    for model in range(n_models):
        point_rank = 1 + int(np.sum(point_scores > point_scores[model]))
        rows.append(
            {
                "model_index": model,
                "point_score": float(point_scores[model]),
                "point_rank": point_rank,
                "rank_lower": 1 + int(definitely_above[model]),
                "rank_upper": n_models - int(definitely_below[model]),
                "definitely_above": int(definitely_above[model]),
                "definitely_below": int(definitely_below[model]),
            }
        )
    return rows


def _resolved_winner(lower: float, upper: float) -> int:
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def _pair_verdict(lower: np.ndarray, upper: np.ndarray) -> str:
    winners = np.array([_resolved_winner(lo, hi) for lo, hi in zip(lower, upper)], dtype=np.int8)
    if np.all(winners == 1):
        return "stable-a"
    if np.all(winners == -1):
        return "stable-b"
    if np.all(winners != 0) and len(set(winners.tolist())) > 1:
        return "protocol-dependent"
    if np.any(winners != 0):
        return "unresolved-mixed"
    return "unresolved"


def _point_flip_weight(gap_protocol_0: float, gap_protocol_1: float) -> float | None:
    slope = gap_protocol_0 - gap_protocol_1
    if abs(slope) <= np.finfo(np.float64).eps:
        return None
    root = -gap_protocol_1 / slope
    return float(root) if 0.0 <= root <= 1.0 else None


def _symmetric_positive_radius(margin_protocol_0: float, margin_protocol_1: float, reference: float) -> float:
    """Supremum symmetric radius around reference on which a linear margin is positive."""
    max_radius = min(reference, 1.0 - reference)
    margin_reference = reference * margin_protocol_0 + (1.0 - reference) * margin_protocol_1
    if margin_reference <= 0.0:
        return 0.0
    slope = margin_protocol_0 - margin_protocol_1
    if abs(slope) <= np.finfo(np.float64).eps:
        return float(max_radius)
    root = -margin_protocol_1 / slope
    if 0.0 <= root <= 1.0 and abs(root - reference) <= max_radius:
        return float(max(0.0, abs(root - reference)))
    return float(max_radius)


def analyze_scores(
    scores: np.ndarray,
    model_names: list[str],
    *,
    alpha: float = 0.05,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20270722,
    reference_weight: float = 0.5,
    weight_range: tuple[float, float] = (0.0, 1.0),
    protocol_names: tuple[str, str] = ("zero-shot", "few-shot"),
    batch_size: int = 2048,
) -> dict:
    """Run simultaneous paired ranking on model x protocol x item scores.

    The two protocol means are combined as ``w * protocol_0 + (1-w) *
    protocol_1``. Configuration axes must be averaged before this call, making
    the target explicit: the uniform fixed configuration grid.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError("scores must have shape (models, protocols, items)")
    n_models, n_protocols, n_items = scores.shape
    if len(model_names) != n_models or len(set(model_names)) != n_models:
        raise ValueError("model_names must be unique and match the model dimension")
    if n_protocols != 2 or len(protocol_names) != 2:
        raise ValueError("the protocol-weight analysis currently requires exactly two protocols")
    if n_items < 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and contain at least two items")
    if not 0.0 <= reference_weight <= 1.0:
        raise ValueError("reference_weight must lie in [0, 1]")
    weight_min, weight_max = weight_range
    if not 0.0 <= weight_min <= weight_max <= 1.0:
        raise ValueError("weight_range must satisfy 0 <= min <= max <= 1")

    pairs = pair_indices(n_models)
    pair_items = np.stack([scores[a] - scores[b] for a, b in pairs], axis=0)
    gaps = pair_items.mean(axis=2)
    centered = pair_items - gaps[:, :, None]
    standard_errors = pair_items.std(axis=2, ddof=1) / np.sqrt(n_items)

    calibration = gaussian_max_t(
        centered.reshape(len(pairs) * n_protocols, n_items),
        alpha=alpha,
        repetitions=bootstrap_repetitions,
        seed=seed,
        batch_size=batch_size,
    )
    lower = gaps - calibration.critical_value * standard_errors
    upper = gaps + calibration.critical_value * standard_errors

    w0 = reference_weight
    reference_gaps = w0 * gaps[:, 0] + (1.0 - w0) * gaps[:, 1]
    reference_lower = w0 * lower[:, 0] + (1.0 - w0) * lower[:, 1]
    reference_upper = w0 * upper[:, 0] + (1.0 - w0) * upper[:, 1]

    lower_at_min = weight_min * lower[:, 0] + (1.0 - weight_min) * lower[:, 1]
    lower_at_max = weight_max * lower[:, 0] + (1.0 - weight_max) * lower[:, 1]
    upper_at_min = weight_min * upper[:, 0] + (1.0 - weight_min) * upper[:, 1]
    upper_at_max = weight_max * upper[:, 0] + (1.0 - weight_max) * upper[:, 1]
    range_lower = np.minimum(lower_at_min, lower_at_max)
    range_upper = np.maximum(upper_at_min, upper_at_max)

    pair_rows = []
    for k, (a, b) in enumerate(pairs):
        reference_winner = _resolved_winner(reference_lower[k], reference_upper[k])
        if reference_winner == 1:
            radius = _symmetric_positive_radius(lower[k, 0], lower[k, 1], w0)
        elif reference_winner == -1:
            radius = _symmetric_positive_radius(-upper[k, 0], -upper[k, 1], w0)
        else:
            radius = 0.0
        pair_rows.append(
            {
                "model_a_index": a,
                "model_b_index": b,
                "model_a": model_names[a],
                "model_b": model_names[b],
                "gap_protocol_0": float(gaps[k, 0]),
                "lower_protocol_0": float(lower[k, 0]),
                "upper_protocol_0": float(upper[k, 0]),
                "gap_protocol_1": float(gaps[k, 1]),
                "lower_protocol_1": float(lower[k, 1]),
                "upper_protocol_1": float(upper[k, 1]),
                "verdict": _pair_verdict(lower[k], upper[k]),
                "point_flip_weight": _point_flip_weight(gaps[k, 0], gaps[k, 1]),
                "reference_gap": float(reference_gaps[k]),
                "reference_lower": float(reference_lower[k]),
                "reference_upper": float(reference_upper[k]),
                "reference_winner": int(reference_winner),
                "confidence_robustness_radius": radius,
                "range_lower": float(range_lower[k]),
                "range_upper": float(range_upper[k]),
                "range_winner": int(_resolved_winner(range_lower[k], range_upper[k])),
            }
        )

    model_means = scores.mean(axis=2)
    rank_rows = []
    for protocol in range(2):
        rows = rank_sets_from_pair_intervals(
            n_models, pairs, lower[:, protocol], upper[:, protocol], model_means[:, protocol]
        )
        for row in rows:
            row.update(target=protocol_names[protocol], model=model_names[int(row["model_index"])])
            rank_rows.append(row)

    reference_scores = w0 * model_means[:, 0] + (1.0 - w0) * model_means[:, 1]
    for row in rank_sets_from_pair_intervals(
        n_models, pairs, reference_lower, reference_upper, reference_scores
    ):
        row.update(target=f"pooled-w={w0:g}", model=model_names[int(row["model_index"])])
        rank_rows.append(row)

    range_scores = ((weight_min + weight_max) / 2.0) * model_means[:, 0] + (
        1.0 - (weight_min + weight_max) / 2.0
    ) * model_means[:, 1]
    for row in rank_sets_from_pair_intervals(n_models, pairs, range_lower, range_upper, range_scores):
        row.update(
            target=f"weight-robust-[{weight_min:g},{weight_max:g}]",
            model=model_names[int(row["model_index"])],
        )
        rank_rows.append(row)

    verdict_counts: dict[str, int] = {}
    for row in pair_rows:
        verdict = str(row["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    return {
        "n_models": n_models,
        "n_protocols": n_protocols,
        "n_items": n_items,
        "n_pairs": len(pairs),
        "alpha": alpha,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "reference_weight": reference_weight,
        "weight_range": [weight_min, weight_max],
        "protocol_names": list(protocol_names),
        "critical_value": calibration.critical_value,
        "effective_correlation_rank": calibration.effective_rank,
        "min_correlation_eigenvalue": calibration.min_eigenvalue,
        "active_statistics": calibration.active_statistics,
        "verdict_counts": verdict_counts,
        "pairs": pair_rows,
        "ranks": rank_rows,
    }


def analyze_pooled_scores(
    scores: np.ndarray,
    model_names: list[str],
    *,
    alpha: float = 0.05,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20270722,
    protocol_weight: float = 0.5,
    batch_size: int = 2048,
) -> dict:
    """Run a simultaneous paired comparison after pooling two protocols."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1] != 2:
        raise ValueError("scores must have shape (models, 2 protocols, items)")
    n_models, _, n_items = scores.shape
    if len(model_names) != n_models or len(set(model_names)) != n_models:
        raise ValueError("model_names must be unique and match the model dimension")
    if n_items < 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and contain at least two items")
    if not 0.0 <= protocol_weight <= 1.0:
        raise ValueError("protocol_weight must lie in [0, 1]")

    pooled_scores = protocol_weight * scores[:, 0] + (1.0 - protocol_weight) * scores[:, 1]
    pairs = pair_indices(n_models)
    pair_items = np.stack([pooled_scores[a] - pooled_scores[b] for a, b in pairs])
    gaps = pair_items.mean(axis=1)
    centered = pair_items - gaps[:, None]
    standard_errors = pair_items.std(axis=1, ddof=1) / np.sqrt(n_items)
    calibration = gaussian_max_t(
        centered,
        alpha=alpha,
        repetitions=bootstrap_repetitions,
        seed=seed,
        batch_size=batch_size,
    )
    lower = gaps - calibration.critical_value * standard_errors
    upper = gaps + calibration.critical_value * standard_errors

    pair_rows = []
    for k, (a, b) in enumerate(pairs):
        winner = _resolved_winner(float(lower[k]), float(upper[k]))
        pair_rows.append(
            {
                "model_a_index": a,
                "model_b_index": b,
                "model_a": model_names[a],
                "model_b": model_names[b],
                "gap": float(gaps[k]),
                "lower": float(lower[k]),
                "upper": float(upper[k]),
                "winner": int(winner),
                "verdict": "stable-a" if winner == 1 else "stable-b" if winner == -1 else "unresolved",
            }
        )

    rank_rows = rank_sets_from_pair_intervals(n_models, pairs, lower, upper, pooled_scores.mean(axis=1))
    for row in rank_rows:
        row.update(target=f"pooled-w={protocol_weight:g}", model=model_names[int(row["model_index"])])

    return {
        "n_models": n_models,
        "n_items": n_items,
        "n_pairs": len(pairs),
        "alpha": alpha,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "protocol_weight": protocol_weight,
        "critical_value": calibration.critical_value,
        "effective_correlation_rank": calibration.effective_rank,
        "min_correlation_eigenvalue": calibration.min_eigenvalue,
        "active_statistics": calibration.active_statistics,
        "resolved_pairs": int(sum(row["winner"] != 0 for row in pair_rows)),
        "pairs": pair_rows,
        "ranks": rank_rows,
    }
