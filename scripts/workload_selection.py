"""Dataset-independent confidence selection.

The rank, interval search, capacity projection, and uniformity routines are
extracted from the paper's existing sampler. Mixed ordering uses a seed and
method tag, without dataset/model namespaces. No labels or text fields are read.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any
import numpy as np
from dataclasses import dataclass


SEED_SCHEME = "sha256-json-v1"


def canonical_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(
        {"type": type(value).__name__, "value": value},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, default=str, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class SamplerError(RuntimeError):
    """Base class for package errors."""


class InputValidationError(SamplerError):
    """Raised when an input table violates the input contract."""


class InfeasibleError(SamplerError):
    """Raised when the requested sample exceeds total source capacity."""


@dataclass(frozen=True)
class ProjectionResult:
    target_probability: np.ndarray
    desired_continuous_quotas: np.ndarray
    projected_continuous_quotas: np.ndarray
    integer_quotas: np.ndarray
    capacities: np.ndarray
    saturated_bins: np.ndarray
    initial_overflow: np.ndarray
    achieved_probability: np.ndarray
    distortion_metrics: dict[str, float]


def assign_bins(probabilities: Any, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    bins = np.searchsorted(edges, values, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2).astype(np.int64)


def _distortion(target: np.ndarray, achieved: np.ndarray) -> dict[str, float]:
    midpoint = 0.5 * (target + achieved)
    target_mask = target > 0
    achieved_mask = achieved > 0
    kl_target = np.sum(target[target_mask] * np.log2(target[target_mask] / midpoint[target_mask]))
    kl_achieved = np.sum(
        achieved[achieved_mask] * np.log2(achieved[achieved_mask] / midpoint[achieved_mask])
    )
    return {
        "jensen_shannon_divergence": float(0.5 * (kl_target + kl_achieved)),
        "total_variation_distance": float(0.5 * np.abs(target - achieved).sum()),
    }


def largest_remainder_rounding(
    continuous: np.ndarray, capacities: np.ndarray, sample_size: int
) -> np.ndarray:
    continuous = np.asarray(continuous, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    integer = np.floor(continuous).astype(np.int64)
    integer = np.minimum(integer, capacities)
    remaining = int(sample_size - integer.sum())
    fractional = continuous - np.floor(continuous)
    order = np.lexsort((np.arange(len(continuous)), -fractional))
    for index in order:
        if remaining == 0:
            break
        if integer[index] < capacities[index]:
            integer[index] += 1
            remaining -= 1
    if remaining:
        for index in range(len(integer)):
            if remaining == 0:
                break
            spare = int(capacities[index] - integer[index])
            take = min(spare, remaining)
            integer[index] += take
            remaining -= take
    if remaining or integer.sum() != sample_size or np.any(integer > capacities):
        raise InfeasibleError("could not construct exact capacity-safe integer quotas")
    return integer


def project_bin_quotas(
    target_probability: np.ndarray,
    bin_capacities: np.ndarray,
    sample_size: int,
) -> ProjectionResult:
    target = np.asarray(target_probability, dtype=np.float64)
    capacities = np.asarray(bin_capacities, dtype=np.int64)
    if target.ndim != 1 or target.shape != capacities.shape:
        raise ValueError("target_probability and bin_capacities must be same-length vectors")
    if sample_size <= 0 or capacities.sum() < sample_size:
        raise InfeasibleError("total source-bin capacity is below the requested sample size")
    if np.any(target < 0) or not np.isclose(target.sum(), 1.0, atol=1e-10):
        raise ValueError("target_probability must be nonnegative and sum to one")
    desired = sample_size * target
    initial_overflow = np.maximum(0.0, desired - capacities)
    projected = np.zeros_like(target)
    active = (target > 0) & (capacities > 0)
    remaining = float(sample_size)
    while np.any(active):
        weights = target[active]
        proposed = remaining * weights / weights.sum()
        indexes = np.flatnonzero(active)
        overflow = proposed > capacities[indexes] + 1e-12
        if not np.any(overflow):
            projected[indexes] = proposed
            remaining = 0.0
            break
        saturated = indexes[overflow]
        projected[saturated] = capacities[saturated]
        remaining -= float(capacities[saturated].sum())
        active[saturated] = False
    if remaining > 1e-7:
        raise InfeasibleError("positive target support lacks sufficient capacity")
    integer = largest_remainder_rounding(projected, capacities, sample_size)
    achieved = integer.astype(float) / sample_size
    saturated_bins = np.flatnonzero(
        (projected >= capacities - 1e-9) & (capacities > 0)
    ).astype(np.int64)
    return ProjectionResult(
        target_probability=target.copy(),
        desired_continuous_quotas=desired,
        projected_continuous_quotas=projected,
        integer_quotas=integer,
        capacities=capacities.copy(),
        saturated_bins=saturated_bins,
        initial_overflow=initial_overflow,
        achieved_probability=achieved,
        distortion_metrics=_distortion(target, achieved),
    )


def exact_rank_indices(
    probabilities: Any,
    row_ids: Any,
    sample_size: int,
    *,
    descending: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(probabilities, dtype=np.float64)
    ids = np.asarray(row_ids, dtype=object)
    if sample_size <= 0 or sample_size > len(values):
        raise ValueError("sample_size must be between one and the row count")
    order = sorted(
        range(len(values)),
        key=lambda index: ((-values[index] if descending else values[index]), canonical_value(ids[index])),
    )
    selected = np.asarray(order[:sample_size], dtype=np.int64)
    unselected = np.asarray(order[sample_size:], dtype=np.int64)
    threshold = float(values[selected[-1]])
    tied = int(np.sum(values == threshold))
    selected_tied = int(np.sum(values[selected] == threshold))
    metadata = {
        "direction": "descending" if descending else "ascending",
        "tie_breaking_method": "probability_then_canonical_row_id_ascending",
        "threshold_probability": threshold,
        "rows_tied_at_threshold": tied,
        "selected_from_threshold_tie": selected_tied,
        "minimum_selected_probability": float(values[selected].min()),
        "maximum_selected_probability": float(values[selected].max()),
        "minimum_unselected_probability": float(values[unselected].min()) if len(unselected) else None,
        "maximum_unselected_probability": float(values[unselected].max()) if len(unselected) else None,
    }
    return selected, metadata


@dataclass(frozen=True)
class UniformIntervalCandidate:
    """Capacity-projected uniform target on one candidate score interval."""

    lower_quantile_index: int
    upper_quantile_index: int
    score_minimum: float
    score_maximum: float
    edges: np.ndarray
    centers: np.ndarray
    projection: ProjectionResult
    support_mode: str = "continuous_interval"

    @property
    def span(self) -> float:
        return self.score_maximum - self.score_minimum

    @property
    def bin_count(self) -> int:
        return int(self.projection.capacities.size)


@dataclass(frozen=True)
class UniformityMetrics:
    """Fine-histogram and continuous-CDF distance from a uniform target."""

    fine_bin_total_variation: float
    kolmogorov_smirnov_distance: float
    normalized_wasserstein_distance: float
    maximum_gap_bin_widths: float


@dataclass(frozen=True)
class FineUniformSearchResult:
    """Strict and best-effort realizations from one interval search."""

    strict_candidate: UniformIntervalCandidate | None
    strict_indices: np.ndarray | None
    strict_metrics: UniformityMetrics | None
    best_effort_candidate: UniformIntervalCandidate
    best_effort_indices: np.ndarray
    best_effort_metrics: UniformityMetrics
    candidate_count: int
    evaluated_candidate_count: int
    quantile_endpoint_count: int


def select_mixed_indices(
    scores: Any,
    row_ids: Any,
    sample_size: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select equal exact low/high tails and deterministically shuffle them."""
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(row_ids, dtype=object)
    if sample_size <= 0 or sample_size % 2:
        raise InputValidationError("top-low sample_size must be a positive even integer")
    if sample_size > values.size:
        raise InputValidationError("top-low sample_size exceeds the row count")
    tail_size = sample_size // 2
    low, low_metadata = exact_rank_indices(values, ids, tail_size, descending=False)
    high, high_metadata = exact_rank_indices(values, ids, tail_size, descending=True)
    overlap = np.intersect1d(low, high)
    if overlap.size:
        # Overlap is possible only through a broad threshold tie when K <= N.
        # Keep the exact low tie choices and use other rows from the same high
        # rank order. This preserves score-optimal high membership and uniqueness.
        low_set = set(low.tolist())
        high_order = sorted(
            range(values.size),
            key=lambda index: (-values[index], canonical_value(ids[index])),
        )
        high = np.asarray(
            [index for index in high_order if index not in low_set][:tail_size],
            dtype=np.int64,
        )
        if high.size != tail_size:
            raise AssertionError("could not construct disjoint score-optimal tails")
    selected = np.concatenate((low, high))
    shuffle_seed = stable_seed(seed, "Mixed", "shuffle")
    selected = selected[np.random.default_rng(shuffle_seed).permutation(sample_size)]
    metadata = {
        "selection_operator": "exact_equal_low_high_tails_then_deterministic_shuffle",
        "low_count": tail_size,
        "high_count": tail_size,
        "low_rank": low_metadata,
        "high_rank": high_metadata,
        "initial_cross_tail_overlap": int(overlap.size),
        "cross_tail_overlap_resolution": (
            "keep_low_canonical_ties_then_choose_next_disjoint_high_rank_rows"
            if overlap.size
            else "none"
        ),
        "shuffle_seed": shuffle_seed,
        "seed_scheme": SEED_SCHEME,
        "rank_membership_depends_on_seed": False,
        "output_order_depends_on_seed": True,
    }
    return selected.astype(np.int64), metadata


def search_uniform_interval_candidates(
    scores: Any,
    sample_size: int,
    *,
    target_bin_count: int = 60,
    interval_grid_size: int = 201,
    strict_projection_tv: float = 0.10,
    candidate_limit: int = 100,
    support_mode: str = "continuous_interval",
) -> tuple[list[UniformIntervalCandidate], dict[str, int]]:
    """Return a bounded union of widest-strict and lowest-distortion intervals."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < sample_size or not np.all(np.isfinite(values)):
        raise InputValidationError("scores must be finite and contain at least sample_size rows")
    if sample_size <= 0 or target_bin_count < 2:
        raise InputValidationError("sample_size must be positive and target_bin_count at least two")
    if interval_grid_size < 3 or candidate_limit < 1:
        raise InputValidationError("interval_grid_size must be >=3 and candidate_limit positive")
    if float(np.ptp(values)) <= 0:
        raise InputValidationError("positive score span is required for uniform sampling")
    if support_mode not in {"continuous_interval", "occupied_bins"}:
        raise InputValidationError(f"unknown uniform support mode: {support_mode!r}")

    sorted_values = np.sort(values)
    endpoint_values = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, interval_grid_size), method="linear")
    )
    all_candidates: list[UniformIntervalCandidate] = []
    for lower_index in range(endpoint_values.size - 1):
        lower = float(endpoint_values[lower_index])
        for upper_index in range(lower_index + 1, endpoint_values.size):
            upper = float(endpoint_values[upper_index])
            if not upper > lower:
                continue
            interval_rows = int(
                np.searchsorted(sorted_values, upper, side="right")
                - np.searchsorted(sorted_values, lower, side="left")
            )
            if interval_rows < sample_size:
                continue
            edges = np.linspace(lower, upper, target_bin_count + 1, dtype=np.float64)
            positions = np.searchsorted(sorted_values, edges, side="left")
            capacities = np.diff(positions).astype(np.int64)
            capacities[-1] = int(
                np.searchsorted(sorted_values, upper, side="right") - positions[-2]
            )
            if support_mode == "occupied_bins":
                occupied = capacities > 0
                if int(occupied.sum()) < 2:
                    continue
                target = np.zeros(target_bin_count, dtype=np.float64)
                target[occupied] = 1.0 / int(occupied.sum())
            else:
                target = np.full(
                    target_bin_count, 1.0 / target_bin_count, dtype=np.float64
                )
            try:
                projection = project_bin_quotas(target, capacities, sample_size)
            except InfeasibleError:
                continue
            all_candidates.append(
                UniformIntervalCandidate(
                    lower_quantile_index=lower_index,
                    upper_quantile_index=upper_index,
                    score_minimum=lower,
                    score_maximum=upper,
                    edges=edges,
                    centers=(edges[:-1] + edges[1:]) / 2,
                    projection=projection,
                    support_mode=support_mode,
                )
            )
    if not all_candidates:
        raise InputValidationError("no positive-span interval contains the requested sample size")

    strict_pool = [
        candidate
        for candidate in all_candidates
        if candidate.projection.distortion_metrics["total_variation_distance"]
        <= strict_projection_tv + 1e-12
    ]
    strict_pool.sort(
        key=lambda candidate: (
            -candidate.span,
            candidate.projection.distortion_metrics["total_variation_distance"],
            candidate.score_minimum,
            candidate.lower_quantile_index,
        )
    )
    best_pool = sorted(
        all_candidates,
        key=lambda candidate: (
            candidate.projection.distortion_metrics["total_variation_distance"],
            -candidate.span,
            candidate.score_minimum,
            candidate.lower_quantile_index,
        ),
    )
    retained: dict[tuple[int, int], UniformIntervalCandidate] = {}
    for candidate in strict_pool[:candidate_limit] + best_pool[:candidate_limit]:
        retained[(candidate.lower_quantile_index, candidate.upper_quantile_index)] = candidate
    candidates = list(retained.values())
    return candidates, {
        "candidate_count": len(all_candidates),
        "retained_candidate_count": len(candidates),
        "quantile_endpoint_count": int(endpoint_values.size),
    }


def match_uniform_target_indices(
    scores: Any,
    row_ids: Any,
    candidate: UniformIntervalCandidate,
) -> np.ndarray:
    """Choose unique rows nearest to evenly spaced targets within each fine bin."""
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(row_ids, dtype=object)
    assignments = assign_bins(values, candidate.edges)
    inside = (values >= candidate.score_minimum) & (values <= candidate.score_maximum)
    pieces: list[np.ndarray] = []
    for bin_id, quota_value in enumerate(candidate.projection.integer_quotas):
        quota = int(quota_value)
        if quota == 0:
            continue
        eligible = np.flatnonzero(inside & (assignments == bin_id))
        eligible = np.asarray(
            sorted(
                eligible.tolist(),
                key=lambda index: (values[index], canonical_value(ids[index])),
            ),
            dtype=np.int64,
        )
        if eligible.size < quota:
            raise AssertionError("candidate quota exceeds realized bin capacity")
        eligible_values = values[eligible]
        lower, upper = candidate.edges[bin_id : bin_id + 2]
        targets = lower + (np.arange(quota, dtype=float) + 0.5) * (upper - lower) / quota
        selected_local: list[int] = []
        previous = -1
        for target_index, target in enumerate(targets):
            minimum = previous + 1
            maximum = eligible.size - (quota - target_index)
            insertion = int(np.searchsorted(eligible_values, target, side="left"))
            possible = {
                min(max(insertion, minimum), maximum),
                min(max(insertion - 1, minimum), maximum),
            }
            chosen = min(
                possible,
                key=lambda local: (
                    abs(float(eligible_values[local]) - float(target)),
                    canonical_value(ids[eligible[local]]),
                ),
            )
            selected_local.append(chosen)
            previous = chosen
        pieces.append(eligible[np.asarray(selected_local, dtype=np.int64)])
    selected = np.concatenate(pieces)
    if selected.size != int(candidate.projection.integer_quotas.sum()):
        raise AssertionError("target matching did not realize the exact quota total")
    if np.unique(selected).size != selected.size:
        raise AssertionError("target matching produced duplicate source rows")
    return selected.astype(np.int64)


def evaluate_uniformity(
    selected_scores: Any,
    candidate: UniformIntervalCandidate,
) -> UniformityMetrics:
    """Measure against the candidate's continuous- or occupied-support target."""
    values = np.sort(np.asarray(selected_scores, dtype=np.float64))
    sample_size = values.size
    if sample_size == 0 or candidate.span <= 0:
        raise InputValidationError("uniformity evaluation requires values and positive span")
    counts, _ = np.histogram(values, bins=candidate.edges)
    achieved = counts.astype(float) / sample_size
    target = candidate.projection.target_probability
    fine_tv = float(0.5 * np.abs(achieved - target).sum())
    if candidate.support_mode == "occupied_bins":
        active = target > 0
        achieved_active = achieved[active]
        target_active = target[active]
        cumulative_difference = np.abs(
            np.cumsum(achieved_active) - np.cumsum(target_active)
        )
        ks = float(cumulative_difference.max(initial=0.0))
        wasserstein = float(
            np.mean(cumulative_difference[:-1])
            if cumulative_difference.size > 1
            else 0.0
        )
        selected_active = counts[active] > 0
        maximum_gap_bins = 0.0 if np.all(selected_active) else float("inf")
        return UniformityMetrics(
            fine_bin_total_variation=fine_tv,
            kolmogorov_smirnov_distance=ks,
            normalized_wasserstein_distance=wasserstein,
            maximum_gap_bin_widths=maximum_gap_bins,
        )
    scaled = np.clip((values - candidate.score_minimum) / candidate.span, 0.0, 1.0)
    empirical_high = np.arange(1, sample_size + 1, dtype=float) / sample_size
    empirical_low = np.arange(sample_size, dtype=float) / sample_size
    ks = float(max(np.max(np.abs(empirical_high - scaled)), np.max(np.abs(empirical_low - scaled))))
    ideal = (np.arange(sample_size, dtype=float) + 0.5) / sample_size
    wasserstein = float(np.mean(np.abs(scaled - ideal)))
    gaps = np.diff(np.concatenate(([candidate.score_minimum], values, [candidate.score_maximum])))
    maximum_gap_bins = float(np.max(gaps) / (candidate.span / candidate.bin_count))
    return UniformityMetrics(
        fine_bin_total_variation=fine_tv,
        kolmogorov_smirnov_distance=ks,
        normalized_wasserstein_distance=wasserstein,
        maximum_gap_bin_widths=maximum_gap_bins,
    )


def uniformity_passes(
    metrics: UniformityMetrics,
    *,
    maximum_tv: float = 0.10,
    maximum_ks: float = 0.02,
    maximum_wasserstein: float = 0.01,
    maximum_gap_bin_widths: float = 2.0,
    check_maximum_gap: bool = True,
) -> bool:
    return bool(
        metrics.fine_bin_total_variation <= maximum_tv + 1e-12
        and metrics.kolmogorov_smirnov_distance <= maximum_ks + 1e-12
        and metrics.normalized_wasserstein_distance <= maximum_wasserstein + 1e-12
        and (
            not check_maximum_gap
            or metrics.maximum_gap_bin_widths <= maximum_gap_bin_widths + 1e-12
        )
    )


def find_fine_uniform_selections(
    scores: Any,
    row_ids: Any,
    sample_size: int,
    *,
    target_bin_count: int = 60,
    interval_grid_size: int = 201,
    candidate_limit: int = 100,
    maximum_tv: float = 0.10,
    maximum_ks: float = 0.02,
    maximum_wasserstein: float = 0.01,
    maximum_gap_bin_widths: float = 2.0,
    support_mode: str = "continuous_interval",
) -> FineUniformSearchResult:
    """Find the widest strict selection plus the closest best-effort selection."""
    values = np.asarray(scores, dtype=np.float64)
    candidates, search_metadata = search_uniform_interval_candidates(
        values,
        sample_size,
        target_bin_count=target_bin_count,
        interval_grid_size=interval_grid_size,
        strict_projection_tv=maximum_tv,
        candidate_limit=candidate_limit,
        support_mode=support_mode,
    )
    realized: list[tuple[UniformIntervalCandidate, np.ndarray, UniformityMetrics]] = []
    for candidate in candidates:
        indices = match_uniform_target_indices(values, row_ids, candidate)
        metrics = evaluate_uniformity(values[indices], candidate)
        realized.append((candidate, indices, metrics))

    passing = [
        item
        for item in realized
        if uniformity_passes(
            item[2],
            maximum_tv=maximum_tv,
            maximum_ks=maximum_ks,
            maximum_wasserstein=maximum_wasserstein,
            maximum_gap_bin_widths=maximum_gap_bin_widths,
            check_maximum_gap=support_mode == "continuous_interval",
        )
    ]
    passing.sort(
        key=lambda item: (
            -item[0].span,
            item[2].fine_bin_total_variation,
            item[2].kolmogorov_smirnov_distance,
            item[2].normalized_wasserstein_distance,
            item[0].score_minimum,
        )
    )
    best_ranked = sorted(
        realized,
        key=lambda item: (
            item[2].fine_bin_total_variation,
            item[2].kolmogorov_smirnov_distance,
            item[2].normalized_wasserstein_distance,
            item[2].maximum_gap_bin_widths,
            -item[0].span,
            item[0].score_minimum,
        ),
    )
    strict = passing[0] if passing else None
    best = best_ranked[0]
    return FineUniformSearchResult(
        strict_candidate=strict[0] if strict else None,
        strict_indices=strict[1] if strict else None,
        strict_metrics=strict[2] if strict else None,
        best_effort_candidate=best[0],
        best_effort_indices=best[1],
        best_effort_metrics=best[2],
        candidate_count=search_metadata["candidate_count"],
        evaluated_candidate_count=len(realized),
        quantile_endpoint_count=search_metadata["quantile_endpoint_count"],
    )


def normalize_ids(values, description="IDs"):
    """Preserve string identities, including leading zeros and literal NA."""
    if values.empty or values.isna().any():
        raise ValueError(f"{description} must be nonempty and non-null")
    normalized = values.astype(str)
    if normalized.str.strip().eq("").any() or normalized.duplicated().any():
        raise ValueError(f"{description} must be nonempty and unique")
    return normalized


def shuffle_indices(indices, *, seed=42, method="Uniform"):
    """Shuffle output order while leaving deterministic membership unchanged."""
    selected = np.asarray(indices, dtype=np.int64)
    shuffle_seed = stable_seed(seed, method, "shuffle")
    return selected[np.random.default_rng(shuffle_seed).permutation(selected.size)], shuffle_seed
