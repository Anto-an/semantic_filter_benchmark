"""Exact kappa optimizers from the paper; plotting and experiment orchestration removed."""

from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import resource
import sqlite3
import time
from typing import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


HIGHS_THREADS = min(8, os.cpu_count() or 1)


HIGHS_SOLVER_LOG = False


HIGHS_TIME_LIMIT: float | None = None


MIP_BACKEND = "auto"


DECOMPOSITION_WORKERS = 1


DECOMPOSITION_MEMORY_FRACTION = 0.70


DECOMPOSITION_PILOT_BRANCHES: int | None = None


INITIAL_LEAF_TIME_LIMIT = 5.0


RECURSIVE_LEAF_TIME_LIMIT = 10.0


MAX_RECURSION_DEPTH = 8


RECURSIVE_PRIORITY_POLICY = "gap"


ADAPTIVE_VERBOSE = True


ADAPTIVE_PROGRESS_INTERVAL = 100


PREPRUNE_RECURSIVE_CHILDREN = True


RECURSIVE_PILOT_PARENTS = 0


RECURSIVE_PILOT_BRANCHES: int | None = None


UNION_RELAXATION_TIME_LIMIT = 0.0


DECOMPOSITION_INFLIGHT_MULTIPLIER = 3


DECOMPOSITION_CHECKPOINT_INTERVAL = 1


DECOMPOSITION_CHECKPOINT_PATH: Path | None = None


DECOMPOSITION_WORKER_AFFINITY = "none"


@dataclass(frozen=True)
class ScoreGroups:
    scores: np.ndarray
    counts: np.ndarray
    positives: np.ndarray

    @property
    def size(self) -> int:
        return int(self.counts.sum())

    @property
    def distinct(self) -> int:
        return len(self.scores)

    def prefixes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        documents = np.concatenate(([0], np.cumsum(self.counts, dtype=np.int64)))
        positives = np.concatenate(([0], np.cumsum(self.positives, dtype=np.int64)))
        negatives = np.concatenate(
            ([0], np.cumsum(self.counts - self.positives, dtype=np.int64))
        )
        return documents, positives, negatives


@dataclass(frozen=True)
class CompressedThresholdDomain:
    """Exact legal prefix domain for one stage/predicted-class pair."""

    stage: int
    label: int
    scores_descending: np.ndarray
    group_counts: np.ndarray
    group_errors: np.ndarray
    candidate_prefixes: np.ndarray
    thresholds_before_budget_pruning: int
    thresholds_removed_by_budget_pruning: int
    # Explicit representative thresholds for candidate_prefixes.  Exact
    # domains use the observed boundary value.  Restricted descendant domains
    # retain the selected root-grid value even when rows removed by an earlier
    # stage make several thresholds behaviorally identical.
    candidate_thresholds: np.ndarray | None = None

    @property
    def original_unique_confidence_count(self) -> int:
        return len(self.scores_descending)

    @property
    def retained_threshold_count(self) -> int:
        return len(self.candidate_prefixes)

    @property
    def n_acceptance_buckets(self) -> int:
        return max(0, len(self.candidate_prefixes) - 1)


@dataclass(frozen=True)
class RestrictedGridSpec:
    """Configuration for an exact solve over a legal threshold subset."""

    size: int
    strategy: str = "quantile"


def score_groups(frame: pd.DataFrame) -> ScoreGroups:
    grouped = (
        frame.groupby("proxy_score", sort=True, observed=True)["oracle_label"]
        .agg([("count", "size"), ("positive", "sum")])
        .reset_index()
    )
    return ScoreGroups(
        scores=grouped["proxy_score"].to_numpy(dtype=float),
        counts=grouped["count"].to_numpy(dtype=np.int64),
        positives=grouped["positive"].to_numpy(dtype=np.int64),
    )


def required_correct_count(alpha: Decimal, n_documents: int) -> int:
    return int((alpha * Decimal(n_documents)).to_integral_value(rounding=ROUND_CEILING))


def boundary_thresholds(groups: ScoreGroups, i: int, j: int) -> tuple[float, float]:
    lower = -math.inf if i == 0 else float(groups.scores[i - 1])
    upper = math.inf if j == groups.distinct else float(groups.scores[j])
    if not lower < upper:
        raise AssertionError(f"invalid representative thresholds for boundaries {(i, j)}")
    return lower, upper


def candidate_counts(
    prefixes: tuple[np.ndarray, np.ndarray, np.ndarray], i: int, j: int
) -> dict[str, int]:
    documents, positives, negatives = prefixes
    n_documents = int(documents[-1])
    total_negatives = int(negatives[-1])
    proxy_negative_count = int(documents[i])
    oracle_count = int(documents[j] - documents[i])
    proxy_positive_count = int(n_documents - documents[j])
    false_negative_count = int(positives[i])
    false_positive_count = int(total_negatives - negatives[j])
    error_count = false_negative_count + false_positive_count
    correct_count = n_documents - error_count
    assert proxy_negative_count + oracle_count + proxy_positive_count == n_documents
    return {
        "proxy_negative_count": proxy_negative_count,
        "oracle_count": oracle_count,
        "proxy_positive_count": proxy_positive_count,
        "false_negative_count": false_negative_count,
        "false_positive_count": false_positive_count,
        "error_count": error_count,
        "correct_count": correct_count,
    }


def optimal_policy_for_required_correct(
    groups: ScoreGroups,
    required: int,
    prefixes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, int]:
    """Return the minimum-oracle policy meeting an integer accuracy requirement.

    Among policies with the minimum oracle count, the representative minimizes
    errors and then uses the existing deterministic boundary-order tie break.
    Supplying precomputed prefixes avoids rebuilding them during Pareto search.
    """

    if not 0 <= required <= groups.size:
        raise ValueError(
            f"required correct count must lie in [0, {groups.size}], found {required}"
        )
    documents, positives, negatives = prefixes or groups.prefixes()
    n_documents = groups.size
    total_negatives = int(negatives[-1])
    error_budget = n_documents - required
    valid_i = np.flatnonzero(positives <= error_budget)
    targets = positives[valid_i] + total_negatives - error_budget
    first_feasible_j = np.searchsorted(negatives, targets, side="left")
    js = np.maximum(valid_i, first_feasible_j)
    valid = js <= groups.distinct
    is_ = valid_i[valid]
    js = js[valid]
    if not len(is_):
        raise AssertionError("oracle-only policy must always be feasible")
    oracle_counts = documents[js] - documents[is_]
    errors = positives[is_] + total_negatives - negatives[js]
    minimum_oracle = int(oracle_counts.min())
    optimal_avoidance = oracle_counts == minimum_oracle
    number_of_optima = int(optimal_avoidance.sum())
    best_error = int(errors[optimal_avoidance].min())
    representatives = np.flatnonzero(optimal_avoidance & (errors == best_error))
    position = int(representatives[0])
    i, j = int(is_[position]), int(js[position])
    return {
        "theta_low_boundary_index": i,
        "theta_high_boundary_index": j,
        "minimum_oracle_count": minimum_oracle,
        "best_error_count": best_error,
        "number_of_optimal_threshold_pairs": number_of_optima,
    }


def optimize_groups(
    groups: ScoreGroups,
    alphas: Sequence[Decimal],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    documents, positives, negatives = groups.prefixes()
    n_documents = groups.size
    total_negatives = int(negatives[-1])
    positive_ratio = float(positives[-1] / n_documents)
    majority_accuracy = max(positive_ratio, 1.0 - positive_ratio)
    records = []
    previous_kappa = math.inf
    for alpha in alphas:
        required = required_correct_count(alpha, n_documents)
        optimum = optimal_policy_for_required_correct(groups, required, (documents, positives, negatives))
        i = optimum["theta_low_boundary_index"]
        j = optimum["theta_high_boundary_index"]
        number_of_optima = optimum["number_of_optimal_threshold_pairs"]
        counts = candidate_counts((documents, positives, negatives), i, j)
        lower, upper = boundary_thresholds(groups, i, j)
        achieved_accuracy = counts["correct_count"] / n_documents
        proxy_negative_fraction = counts["proxy_negative_count"] / n_documents
        oracle_routing_fraction = counts["oracle_count"] / n_documents
        proxy_positive_fraction = counts["proxy_positive_count"] / n_documents
        oracle_avoidance = proxy_negative_fraction + proxy_positive_fraction
        assert counts["correct_count"] >= required
        assert math.isclose(achieved_accuracy, 1 - counts["error_count"] / n_documents, abs_tol=1e-15)
        assert math.isclose(
            proxy_negative_fraction + oracle_routing_fraction + proxy_positive_fraction,
            1.0,
            abs_tol=1e-15,
        )
        assert math.isclose(oracle_avoidance, 1 - oracle_routing_fraction, abs_tol=1e-15)
        assert oracle_avoidance <= previous_kappa + 1e-15
        previous_kappa = oracle_avoidance
        records.append(
            {
                **metadata,
                "n_documents": n_documents,
                "n_distinct_scores": groups.distinct,
                "positive_ratio": positive_ratio,
                "majority_accuracy": majority_accuracy,
                "prior_trivial": bool(max(int(positives[-1]), total_negatives) >= required),
                "alpha": format(alpha, "f"),
                "required_correct_count": required,
                "theta_low_raw": lower,
                "theta_high_raw": upper,
                "theta_low_boundary_index": i,
                "theta_high_boundary_index": j,
                **counts,
                "proxy_negative_fraction": proxy_negative_fraction,
                "oracle_routing_fraction": oracle_routing_fraction,
                "proxy_positive_fraction": proxy_positive_fraction,
                "achieved_accuracy": achieved_accuracy,
                "oracle_avoidance": oracle_avoidance,
                "kappa": oracle_avoidance,
                "number_of_optimal_threshold_pairs": number_of_optima,
                "n_proxy_stages": 1,
                "cascade_length": 2,
                "policy_type": "positive_score_two_threshold",
                "stage_thresholds_json": None,
                "optimization_method": "exact_prefix_scan",
                "optimality_proven": True,
                "solver_status": "exact_scan",
                "solver_mip_gap": 0.0,
                "solver_mip_nodes": 0,
            }
        )
    return records


def _json_number(value: float) -> float | str:
    return "Infinity" if math.isinf(value) and value > 0 else (
        "-Infinity" if math.isinf(value) else float(value)
    )


def _threshold_values_for_prefixes(
    scores_descending: np.ndarray, prefixes: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            math.inf
            if int(prefix) == 0
            else float(scores_descending[int(prefix) - 1])
            for prefix in prefixes
        ],
        dtype=float,
    )


def _restricted_grid_indices(
    domain: CompressedThresholdDomain,
    size: int,
    strategy: str,
    *,
    native_threshold: float | None = None,
) -> np.ndarray:
    """Choose a deterministic nested subset of an exact legal domain.

    The maximin ordering is a hierarchical quantile grid: every additional
    point fills the largest uncovered interval in accepted document mass.
    Taking prefixes of this ordering makes grids nested for arbitrary K.
    """

    count = domain.retained_threshold_count
    if count <= size:
        return np.arange(count, dtype=np.int64)
    if strategy not in {"quantile", "candidate_rank"}:
        raise ValueError(f"unsupported approximate grid strategy {strategy!r}")
    prefixes = domain.candidate_prefixes
    mandatory = {0, count - 1}
    finite = np.flatnonzero(prefixes > 0)
    if len(finite):
        mandatory.add(int(finite[0]))
    if int(prefixes[-1]) == len(domain.scores_descending):
        mandatory.add(count - 1)
    if native_threshold is not None:
        raw_prefix = int(
            np.count_nonzero(domain.scores_descending >= float(native_threshold))
        )
        native_index = int(np.searchsorted(prefixes, raw_prefix, side="left"))
        mandatory.add(min(native_index, count - 1))
    if len(mandatory) > size:
        raise ValueError(
            f"approx-grid-size={size} is smaller than {len(mandatory)} "
            f"mandatory states for stage={domain.stage} class={domain.label}"
        )
    if strategy == "quantile":
        cumulative = np.concatenate(
            ([0], np.cumsum(domain.group_counts, dtype=np.int64))
        )
        coordinates = cumulative[prefixes].astype(float)
        scale = max(1.0, float(coordinates[-1]))
        coordinates /= scale
    else:
        coordinates = np.linspace(0.0, 1.0, count)

    selected = set(mandatory)
    while len(selected) < size:
        chosen = min(
            (index for index in range(count) if index not in selected),
            key=lambda index: (
                -min(
                    abs(float(coordinates[index]) - float(coordinates[other]))
                    for other in selected
                ),
                index,
            ),
        )
        selected.add(chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def _stage_thresholds_json(
    stage_metadata: Sequence[dict[str, object]], thresholds: Sequence[tuple[float, float]]
) -> str:
    payload = []
    for metadata, (threshold_0, threshold_1) in zip(stage_metadata, thresholds):
        payload.append(
            {
                "stage": int(metadata["stage"]),
                "surrogate": str(metadata["surrogate"]),
                "model": str(metadata["model"]),
                "requested_fraction": float(metadata["requested_fraction"]),
                "threshold_0": _json_number(threshold_0),
                "threshold_1": _json_number(threshold_1),
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _task_policy_counts(
    frame: pd.DataFrame, thresholds: Sequence[tuple[float, float]]
) -> dict[str, int]:
    n_documents = len(frame)
    oracle = frame["oracle_label"].to_numpy(np.int8)
    active = np.ones(n_documents, dtype=bool)
    final = oracle.copy()
    proxy_negative = np.zeros(n_documents, dtype=bool)
    proxy_positive = np.zeros(n_documents, dtype=bool)
    for stage, (threshold_0, threshold_1) in enumerate(thresholds, start=1):
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        confidence = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
        cutoff = np.where(prediction == 0, threshold_0, threshold_1)
        accepted = active & (confidence >= cutoff)
        final[accepted] = prediction[accepted]
        proxy_negative[accepted & (prediction == 0)] = True
        proxy_positive[accepted & (prediction == 1)] = True
        active[accepted] = False
    false_negative = int(np.count_nonzero(proxy_negative & (oracle == 1)))
    false_positive = int(np.count_nonzero(proxy_positive & (oracle == 0)))
    error_count = int(np.count_nonzero(final != oracle))
    if error_count != false_negative + false_positive:
        raise AssertionError("TaskCascades error decomposition disagrees with replay")
    return {
        "proxy_negative_count": int(proxy_negative.sum()),
        "oracle_count": int(active.sum()),
        "proxy_positive_count": int(proxy_positive.sum()),
        "false_negative_count": false_negative,
        "false_positive_count": false_positive,
        "error_count": error_count,
        "correct_count": n_documents - error_count,
    }


def _subproblem_results_json(
    results: Sequence[dict[str, object]] | None,
) -> str | None:
    if results is None:
        return None
    fields = (
        "branch_id",
        "depth",
        "split_label",
        "candidate_index",
        "threshold",
        "second_label",
        "second_candidate_index",
        "second_threshold",
        "forced_routed",
        "forced_errors",
        "remaining_N",
        "remaining_budget",
        "G",
        "H",
        "upper_bound_routed",
        "old_upper_bound_routed",
        "stronger_upper_bound_routed",
        "union_relaxation_upper_bound_routed",
        "P_after_compression",
        "binary_variables",
        "total_variables",
        "constraints",
        "sparse_nonzeros",
        "status",
        "incumbent_routed",
        "incumbent_errors",
        "best_bound_routed",
        "absolute_gap",
        "mip_gap",
        "mip_nodes",
        "solver_status",
        "elapsed_seconds",
        "cpu_utilization_percent",
        "peak_rss_mb",
        "root_LP_finished",
        "propagation_time",
        "threshold_pruning_time",
        "bucket_build_time",
        "pattern_aggregation_time",
        "model_build_time",
        "incumbent_time",
        "presolve_time",
        "solve_time",
        "checkpoint_time",
        "total_time",
        "remaining_rows",
        "remaining_patterns",
        "remaining_threshold_candidates",
        "threshold_candidates_before",
        "threshold_candidates_after",
        "candidates_removed_by_poison_pruning",
        "additional_candidates_removed_by_future_poison_pruning",
        "union_relaxation_exact",
        "union_relaxation_nodes",
        "union_relaxation_time",
    )
    payload = []
    for result in results:
        row = {}
        for field in fields:
            if field not in result:
                continue
            value = result[field]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = _json_number(value)
            row[field] = value
        payload.append(row)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _task_record(
    metadata: dict[str, object],
    frame: pd.DataFrame,
    alpha: Decimal,
    thresholds: Sequence[tuple[float, float]],
    stage_metadata: Sequence[dict[str, object]],
    n_distinct_scores: int,
    optimization_method: str,
    solver_status: str,
    solver_mip_gap: float,
    solver_mip_nodes: int,
    number_of_optima: int | None = None,
    *,
    optimality_proven: bool = True,
    upper_bound_kappa: float | None = None,
    exact_optimality_certificate: str | None = None,
    decomposition_certificate: dict[str, object] | None = None,
    subproblem_results: Sequence[dict[str, object]] | None = None,
    kappa_method: str = "exact",
    kappa_mode: str = "exact",
    restricted_domain_optimality_certificate: bool = False,
    approx_grid_size_requested: int | None = None,
    approx_grid_strategy: str | None = None,
    model_statistics: dict[str, object] | None = None,
    solve_time_seconds: float | None = None,
    preprocessing_time_seconds: float | None = None,
    total_time_seconds: float | None = None,
) -> dict[str, object]:
    counts = _task_policy_counts(frame, thresholds)
    n_documents = len(frame)
    required = required_correct_count(alpha, n_documents)
    if counts["correct_count"] < required:
        raise AssertionError(
            f"TaskCascades replay misses alpha={alpha}: {counts['correct_count']} < {required}"
        )
    labels = frame["oracle_label"].to_numpy(np.int8)
    positive_ratio = float(labels.mean())
    proxy_negative_fraction = counts["proxy_negative_count"] / n_documents
    oracle_routing_fraction = counts["oracle_count"] / n_documents
    proxy_positive_fraction = counts["proxy_positive_count"] / n_documents
    oracle_avoidance = 1.0 - oracle_routing_fraction
    statistics = model_statistics or {}
    exact_total = int(statistics.get("exact_candidate_count_total", 0))
    restricted_total = int(
        statistics.get("restricted_candidate_count_total", exact_total)
    )
    return {
        **metadata,
        "n_documents": n_documents,
        "n_distinct_scores": n_distinct_scores,
        "positive_ratio": positive_ratio,
        "majority_accuracy": max(positive_ratio, 1.0 - positive_ratio),
        "prior_trivial": bool(max(int(labels.sum()), n_documents - int(labels.sum())) >= required),
        "alpha": format(alpha, "f"),
        "required_correct_count": required,
        "theta_low_raw": np.nan,
        "theta_high_raw": np.nan,
        "theta_low_boundary_index": np.nan,
        "theta_high_boundary_index": np.nan,
        **counts,
        "proxy_negative_fraction": proxy_negative_fraction,
        "oracle_routing_fraction": oracle_routing_fraction,
        "proxy_positive_fraction": proxy_positive_fraction,
        "achieved_accuracy": counts["correct_count"] / n_documents,
        "oracle_avoidance": oracle_avoidance,
        "kappa": oracle_avoidance,
        "routed_count": counts["proxy_negative_count"] + counts["proxy_positive_count"],
        "kappa_method": kappa_method,
        "kappa_mode": kappa_mode,
        "is_global_exact": bool(optimality_proven),
        "is_restricted_domain_exact": bool(
            restricted_domain_optimality_certificate
        ),
        "is_lower_bound_on_unrestricted_kappa": kappa_method == "restricted_grid",
        "restricted_domain_optimality_certificate": bool(
            restricted_domain_optimality_certificate
        ),
        "approx_grid_size_requested": approx_grid_size_requested,
        "approx_grid_strategy": approx_grid_strategy,
        "exact_candidate_count_total": exact_total,
        "restricted_candidate_count_total": restricted_total,
        "exact_candidate_counts_json": json.dumps(
            statistics.get("exact_candidate_counts", []), sort_keys=True
        ),
        "restricted_candidate_counts_json": json.dumps(
            statistics.get("restricted_candidate_counts", []), sort_keys=True
        ),
        "threshold_reduction_ratio": (
            1.0 - restricted_total / exact_total if exact_total else 0.0
        ),
        "patterns_after_compression": statistics.get("P_after_compression"),
        "total_variables": statistics.get("total_variables"),
        "sparse_nonzeros": statistics.get("sparse_nonzeros"),
        "solve_time_seconds": solve_time_seconds,
        "preprocessing_time_seconds": preprocessing_time_seconds,
        "total_time_seconds": total_time_seconds,
        "kappa_approx": np.nan,
        "kappa_exact": np.nan,
        "kappa_gap_exact_minus_approx": np.nan,
        "kappa_gap_pp": np.nan,
        "number_of_optimal_threshold_pairs": number_of_optima,
        "n_proxy_stages": len(stage_metadata),
        "cascade_length": len(stage_metadata) + 1,
        "policy_type": "class_confidence_multistage",
        "stage_thresholds_json": _stage_thresholds_json(stage_metadata, thresholds),
        "optimization_method": optimization_method,
        "optimality_proven": optimality_proven,
        "solver_status": solver_status,
        "solver_mip_gap": solver_mip_gap,
        "solver_mip_nodes": solver_mip_nodes,
        "incumbent_kappa": oracle_avoidance,
        "upper_bound_kappa": (
            oracle_avoidance if upper_bound_kappa is None else upper_bound_kappa
        ),
        "absolute_kappa_gap": max(
            0.0,
            (oracle_avoidance if upper_bound_kappa is None else upper_bound_kappa)
            - oracle_avoidance,
        ),
        "relative_gap": solver_mip_gap,
        "exact_optimality_certificate": optimality_proven,
        "optimality_certificate_detail": exact_optimality_certificate,
        "decomposition_certificate_json": (
            json.dumps(decomposition_certificate, sort_keys=True)
            if decomposition_certificate is not None
            else None
        ),
        "subproblem_results_json": _subproblem_results_json(subproblem_results),
    }


def optimize_single_task_stage(
    frame: pd.DataFrame,
    alphas: Sequence[Decimal],
    metadata: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Optimize one class-confidence stage by joining its two exact prefixes."""

    oracle = frame["oracle_label"].to_numpy(np.int8)
    prediction = frame["stage_01_prediction"].to_numpy(np.int8)
    confidence = frame["stage_01_confidence"].to_numpy(float)
    prefixes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    n_distinct = 0
    for label in (0, 1):
        subset = pd.DataFrame(
            {
                "confidence": confidence[prediction == label],
                "error": (oracle[prediction == label] != label).astype(np.int64),
            }
        )
        grouped = (
            subset.groupby("confidence", sort=False)["error"]
            .agg([("count", "size"), ("errors", "sum")])
            .sort_index(ascending=False)
        )
        scores = grouped.index.to_numpy(float)
        documents = np.concatenate(([0], np.cumsum(grouped["count"].to_numpy(np.int64))))
        errors = np.concatenate(([0], np.cumsum(grouped["errors"].to_numpy(np.int64))))
        prefixes[label] = scores, documents, errors
        n_distinct += len(scores)

    records = []
    for alpha in alphas:
        error_budget = len(frame) - required_correct_count(alpha, len(frame))
        scores0, documents0, errors0 = prefixes[0]
        scores1, documents1, errors1 = prefixes[1]
        candidates = []
        for k0 in range(len(documents0)):
            remaining = error_budget - int(errors0[k0])
            if remaining < 0:
                continue
            k1 = int(np.searchsorted(errors1, remaining, side="right") - 1)
            accepted = int(documents0[k0] + documents1[k1])
            errors = int(errors0[k0] + errors1[k1])
            candidates.append((accepted, errors, k0, k1))
        if not candidates:
            raise AssertionError("oracle-only TaskCascades policy must be feasible")
        maximum_accepted = max(item[0] for item in candidates)
        number_of_optima = sum(item[0] == maximum_accepted for item in candidates)
        accepted, _, k0, k1 = min(
            (item for item in candidates if item[0] == maximum_accepted),
            key=lambda item: (item[1], item[2], item[3]),
        )
        threshold_0 = math.inf if k0 == 0 else float(scores0[k0 - 1])
        threshold_1 = math.inf if k1 == 0 else float(scores1[k1 - 1])
        record = _task_record(
            metadata,
            frame,
            alpha,
            [(threshold_0, threshold_1)],
            stage_metadata,
            n_distinct,
            "exact_class_prefix_scan",
            "exact_scan",
            0.0,
            0,
            number_of_optima,
        )
        if record["proxy_negative_count"] + record["proxy_positive_count"] != accepted:
            raise AssertionError("one-stage TaskCascades optimum disagrees with replay")
        records.append(record)
    return records


def _compress_multistage_thresholds(
    frame: pd.DataFrame,
    n_stages: int,
    error_budget: int,
    *,
    report: bool = True,
    grid_spec: RestrictedGridSpec | None = None,
    native_thresholds: dict[tuple[int, int], float] | None = None,
    allowed_thresholds: dict[tuple[int, int], tuple[float, ...]] | None = None,
    stage_offset: int = 0,
) -> tuple[
    list[dict[int, CompressedThresholdDomain]],
    list[np.ndarray],
    dict[str, int | float],
]:
    """Compress and budget-prune exact threshold-prefix domains.

    A retained prefix ends immediately before a confidence tie group containing
    an error.  Prefixes ending before correct-only groups are dominated by the
    next retained prefix.  Budget pruning uses the all-wrong-prefix certificate
    described in Section 3.1's exact solver notes.
    """

    threshold_pruning_seconds = 0.0
    bucket_build_seconds = 0.0
    oracle = frame["oracle_label"].to_numpy(np.int8)
    all_wrong = np.ones(len(frame), dtype=bool)
    all_wrong_prefix: list[np.ndarray] = []
    for stage in range(1, n_stages + 1):
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        eligible_column = f"stage_{stage:02d}_eligible"
        eligible = (
            frame[eligible_column].to_numpy(bool)
            if eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        all_wrong &= (~eligible) | (prediction != oracle)
        all_wrong_prefix.append(all_wrong.copy())

    domains: list[dict[int, CompressedThresholdDomain]] = []
    document_buckets: list[np.ndarray] = []
    total_original = 0
    total_before_pruning = 0
    total_after_pruning = 0
    total_solver_candidates = 0
    total_bucket_variables = 0
    exact_counts: list[dict[str, int]] = []
    solver_counts: list[dict[str, int]] = []
    for stage in range(1, n_stages + 1):
        predictions = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        confidences = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
        eligible_column = f"stage_{stage:02d}_eligible"
        eligible = (
            frame[eligible_column].to_numpy(bool)
            if eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        stage_domains: dict[int, CompressedThresholdDomain] = {}
        stage_exact_counts: dict[str, int] = {"stage": stage_offset + stage}
        stage_solver_counts: dict[str, int] = {"stage": stage_offset + stage}
        stage_buckets = np.full(len(frame), -1, dtype=np.int32)
        for label in (0, 1):
            pruning_started = time.monotonic()
            mask = (predictions == label) & eligible
            label_scores = confidences[mask]
            scores_descending = np.unique(label_scores)[::-1]
            n_groups = len(scores_descending)
            if n_groups:
                ranks = np.searchsorted(-scores_descending, -label_scores)
                counts = np.bincount(ranks, minlength=n_groups).astype(np.int64)
                errors = np.bincount(
                    ranks,
                    weights=(oracle[mask] != label).astype(np.int64),
                    minlength=n_groups,
                ).astype(np.int64)
                unavoidable = np.bincount(
                    ranks,
                    weights=all_wrong_prefix[stage - 1][mask].astype(np.int64),
                    minlength=n_groups,
                ).astype(np.int64)
                retained = np.unique(
                    np.concatenate(
                        (
                            np.array([0, n_groups], dtype=np.int64),
                            np.flatnonzero(errors > 0).astype(np.int64),
                        )
                    )
                )
                cumulative_unavoidable = np.concatenate(
                    ([0], np.cumsum(unavoidable, dtype=np.int64))
                )
                before_pruning = len(retained)
                retained = retained[cumulative_unavoidable[retained] <= error_budget]
                removed = before_pruning - len(retained)
                if len(retained) == 0 or retained[0] != 0:
                    raise AssertionError("reject-all threshold must remain feasible")
                exact_retained = retained.copy()
                exact_thresholds = _threshold_values_for_prefixes(
                    scores_descending, exact_retained
                )
                original_stage = stage_offset + stage
                if allowed_thresholds is not None:
                    allowed = allowed_thresholds[(original_stage, label)]
                    by_prefix: dict[int, float] = {}
                    for threshold in allowed:
                        raw_prefix = (
                            0
                            if math.isinf(float(threshold))
                            else int(
                                np.count_nonzero(
                                    scores_descending >= float(threshold)
                                )
                            )
                        )
                        if cumulative_unavoidable[raw_prefix] > error_budget:
                            continue
                        previous = by_prefix.get(raw_prefix)
                        if previous is None or float(threshold) > previous:
                            by_prefix[raw_prefix] = float(threshold)
                    if 0 not in by_prefix:
                        by_prefix[0] = math.inf
                    ordered_prefixes = sorted(by_prefix)
                    cumulative_errors = np.concatenate(
                        ([0], np.cumsum(errors, dtype=np.int64))
                    )
                    kept_prefixes: list[int] = []
                    for position, raw_prefix in enumerate(ordered_prefixes):
                        if position == 0 or position == len(ordered_prefixes) - 1:
                            kept_prefixes.append(raw_prefix)
                            continue
                        next_prefix = ordered_prefixes[position + 1]
                        if (
                            cumulative_errors[next_prefix]
                            > cumulative_errors[raw_prefix]
                        ):
                            kept_prefixes.append(raw_prefix)
                    retained = np.asarray(kept_prefixes, dtype=np.int64)
                    candidate_thresholds = np.asarray(
                        [by_prefix[int(prefix)] for prefix in retained], dtype=float
                    )
                elif grid_spec is not None:
                    temporary_domain = CompressedThresholdDomain(
                        stage=original_stage,
                        label=label,
                        scores_descending=scores_descending,
                        group_counts=counts,
                        group_errors=errors,
                        candidate_prefixes=exact_retained,
                        thresholds_before_budget_pruning=before_pruning,
                        thresholds_removed_by_budget_pruning=removed,
                        candidate_thresholds=exact_thresholds,
                    )
                    indices = _restricted_grid_indices(
                        temporary_domain,
                        grid_spec.size,
                        grid_spec.strategy,
                        native_threshold=(native_thresholds or {}).get(
                            (original_stage, label)
                        ),
                    )
                    retained = exact_retained[indices]
                    candidate_thresholds = exact_thresholds[indices]
                else:
                    candidate_thresholds = exact_thresholds
                threshold_pruning_seconds += time.monotonic() - pruning_started
                bucket_started = time.monotonic()
                interval_ends = retained[1:]
                local_buckets = np.searchsorted(interval_ends, ranks, side="right")
                local_buckets[local_buckets == len(interval_ends)] = -1
                stage_buckets[mask] = local_buckets.astype(np.int32)
                bucket_build_seconds += time.monotonic() - bucket_started
            else:
                counts = np.empty(0, dtype=np.int64)
                errors = np.empty(0, dtype=np.int64)
                retained = np.array([0], dtype=np.int64)
                exact_retained = retained.copy()
                candidate_thresholds = np.array([math.inf], dtype=float)
                before_pruning = 1
                removed = 0
                threshold_pruning_seconds += time.monotonic() - pruning_started

            domain = CompressedThresholdDomain(
                stage=stage,
                label=label,
                scores_descending=scores_descending,
                group_counts=counts,
                group_errors=errors,
                candidate_prefixes=retained,
                thresholds_before_budget_pruning=before_pruning,
                thresholds_removed_by_budget_pruning=removed,
                candidate_thresholds=candidate_thresholds,
            )
            stage_domains[label] = domain
            total_original += n_groups
            total_before_pruning += before_pruning
            total_after_pruning += len(exact_retained)
            total_solver_candidates += len(retained)
            total_bucket_variables += domain.n_acceptance_buckets
            stage_exact_counts[f"class_{label}"] = len(exact_retained)
            stage_solver_counts[f"class_{label}"] = len(retained)
            if report:
                ratio = len(retained) / max(1, n_groups + 1)
                print(
                    f"[threshold-compression] stage={stage} class={label} "
                    f"original_unique_confidence_count={n_groups} "
                    f"retained_threshold_count={len(retained)} "
                    f"compression_ratio={ratio:.6f} "
                    f"before_budget_pruning={before_pruning} "
                    f"after_budget_pruning={len(exact_retained)} removed={removed} "
                    f"solver_candidate_count={len(retained)}",
                    flush=True,
                )
        domains.append(stage_domains)
        document_buckets.append(stage_buckets)
        exact_counts.append(stage_exact_counts)
        solver_counts.append(stage_solver_counts)

    diagnostics = {
        "D_original": total_original,
        "D_compressed": total_bucket_variables,
        "thresholds_before_budget_pruning": total_before_pruning,
        "thresholds_after_budget_pruning": total_after_pruning,
        "exact_candidate_count_total": total_after_pruning,
        "restricted_candidate_count_total": total_solver_candidates,
        "exact_candidate_counts": exact_counts,
        "restricted_candidate_counts": solver_counts,
        "number_removed_by_budget_pruning": total_before_pruning
        - total_after_pruning,
        "threshold_pruning_time": threshold_pruning_seconds,
        "bucket_build_time": bucket_build_seconds,
    }
    if report:
        print(
            "[threshold-compression-total] "
            + " ".join(f"{key}={value}" for key, value in diagnostics.items()),
            flush=True,
        )
    return domains, document_buckets, diagnostics


def _aggregate_compressed_patterns(
    frame: pd.DataFrame,
    n_stages: int,
    document_buckets: Sequence[np.ndarray],
    *,
    report: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    started = time.monotonic()
    raw_keys = ["oracle_label"]
    compressed = pd.DataFrame({"oracle_label": frame["oracle_label"].to_numpy(np.int8)})
    compressed_keys = ["oracle_label"]
    for stage in range(1, n_stages + 1):
        prediction_key = f"stage_{stage:02d}_prediction"
        confidence_key = f"stage_{stage:02d}_confidence"
        bucket_key = f"stage_{stage:02d}_bucket"
        raw_keys.extend((prediction_key, confidence_key))
        compressed[prediction_key] = frame[prediction_key].to_numpy(np.int8)
        compressed[bucket_key] = np.asarray(document_buckets[stage - 1], dtype=np.int32)
        compressed_keys.extend((prediction_key, bucket_key))
    p_original = int(frame.groupby(raw_keys, sort=False, dropna=False).ngroups)
    patterns = (
        compressed.groupby(compressed_keys, sort=False, dropna=False)
        .size()
        .reset_index(name="weight")
    )
    diagnostics: dict[str, int | float] = {
        "N": len(frame),
        "P_original": p_original,
        "P_after_compression": len(patterns),
        "aggregation_ratio": len(patterns) / max(1, p_original),
        "pattern_aggregation_time": time.monotonic() - started,
    }
    if report:
        print(
            "[pattern-aggregation] "
            + " ".join(f"{key}={value}" for key, value in diagnostics.items()),
            flush=True,
        )
    return patterns, diagnostics


def _build_multistage_milp(
    frame: pd.DataFrame,
    n_stages: int,
    error_budget: int,
    *,
    report: bool = True,
    grid_spec: RestrictedGridSpec | None = None,
    native_thresholds: dict[tuple[int, int], float] | None = None,
    allowed_thresholds: dict[tuple[int, int], tuple[float, ...]] | None = None,
    stage_offset: int = 0,
) -> dict[str, object]:
    """Build the compressed O(P L) exact survival-state MILP."""

    started = time.monotonic()
    domains, document_buckets, compression = _compress_multistage_thresholds(
        frame,
        n_stages,
        error_budget,
        report=report,
        grid_spec=grid_spec,
        native_thresholds=native_thresholds,
        allowed_thresholds=allowed_thresholds,
        stage_offset=stage_offset,
    )
    patterns, aggregation = _aggregate_compressed_patterns(
        frame, n_stages, document_buckets, report=report
    )
    n_patterns = len(patterns)

    acceptance_layout: list[dict[int, np.ndarray]] = []
    n_acceptance = 0
    for stage_domains in domains:
        layout: dict[int, np.ndarray] = {}
        for label in (0, 1):
            count = stage_domains[label].n_acceptance_buckets
            layout[label] = np.arange(
                n_acceptance, n_acceptance + count, dtype=np.int64
            )
            n_acceptance += count
        acceptance_layout.append(layout)

    pattern_acceptance: list[np.ndarray] = []
    for stage_index, stage in enumerate(range(1, n_stages + 1)):
        predictions = patterns[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        buckets = patterns[f"stage_{stage:02d}_bucket"].to_numpy(np.int32)
        indices = np.full(n_patterns, -1, dtype=np.int64)
        for label in (0, 1):
            mask = (predictions == label) & (buckets >= 0)
            indices[mask] = acceptance_layout[stage_index][label][buckets[mask]]
        pattern_acceptance.append(indices)

    n_survival = n_patterns * n_stages
    n_variables = n_acceptance + n_survival
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    next_row = 0

    def add_rows(
        columns: Sequence[np.ndarray],
        values: Sequence[float | np.ndarray],
        lower: float | np.ndarray,
        upper: float | np.ndarray,
    ) -> None:
        nonlocal next_row
        size = len(columns[0])
        rows = np.arange(next_row, next_row + size, dtype=np.int64)
        for column, value in zip(columns, values):
            row_parts.append(rows)
            column_parts.append(np.asarray(column, dtype=np.int64))
            if np.isscalar(value):
                value_parts.append(np.full(size, float(value), dtype=float))
            else:
                value_parts.append(np.asarray(value, dtype=float))
        lower_parts.append(
            np.full(size, float(lower), dtype=float)
            if np.isscalar(lower)
            else np.asarray(lower, dtype=float)
        )
        upper_parts.append(
            np.full(size, float(upper), dtype=float)
            if np.isscalar(upper)
            else np.asarray(upper, dtype=float)
        )
        next_row += size

    # Prefix monotonicity: accepting a lower-confidence bucket requires all
    # higher-confidence buckets to be accepted.
    for stage_layout in acceptance_layout:
        for indices in stage_layout.values():
            if len(indices) > 1:
                add_rows([indices[:-1], indices[1:]], [-1.0, 1.0], -np.inf, 0.0)

    pattern_index = np.arange(n_patterns, dtype=np.int64)
    for stage_index in range(n_stages):
        survival = n_acceptance + pattern_index * n_stages + stage_index
        acceptance = pattern_acceptance[stage_index]
        has_acceptance = acceptance >= 0
        no_acceptance = ~has_acceptance
        if stage_index == 0:
            # s_1 = 1-a_1.  Constant a=0 rows simply fix s_1=1.
            if has_acceptance.any():
                add_rows(
                    [survival[has_acceptance], acceptance[has_acceptance]],
                    [1.0, 1.0],
                    1.0,
                    1.0,
                )
            if no_acceptance.any():
                add_rows([survival[no_acceptance]], [1.0], 1.0, 1.0)
        else:
            previous = n_acceptance + pattern_index * n_stages + stage_index - 1
            # s_l <= s_{l-1}
            add_rows([survival, previous], [1.0, -1.0], -np.inf, 0.0)
            if has_acceptance.any():
                # s_l <= 1-a_l and s_l >= s_{l-1}-a_l.
                add_rows(
                    [survival[has_acceptance], acceptance[has_acceptance]],
                    [1.0, 1.0],
                    -np.inf,
                    1.0,
                )
                add_rows(
                    [
                        survival[has_acceptance],
                        previous[has_acceptance],
                        acceptance[has_acceptance],
                    ],
                    [1.0, -1.0, 1.0],
                    0.0,
                    np.inf,
                )
            if no_acceptance.any():
                # Together with s_l <= s_{l-1}, this fixes equality.
                add_rows(
                    [survival[no_acceptance], previous[no_acceptance]],
                    [1.0, -1.0],
                    0.0,
                    np.inf,
                )

    weights = patterns["weight"].to_numpy(np.int64)
    oracle = patterns["oracle_label"].to_numpy(np.int8)
    objective = np.zeros(n_variables, dtype=float)
    final_survival = n_acceptance + pattern_index * n_stages + n_stages - 1
    objective[final_survival] = weights

    error_coefficients: dict[int, float] = {}
    error_constant = 0.0
    for stage_index, stage in enumerate(range(1, n_stages + 1)):
        prediction = patterns[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        wrong_weights = weights * (prediction != oracle)
        current = n_acceptance + pattern_index * n_stages + stage_index
        if stage_index == 0:
            error_constant += float(wrong_weights.sum())
        else:
            previous = current - 1
            for column, value in zip(previous, wrong_weights):
                if value:
                    error_coefficients[int(column)] = (
                        error_coefficients.get(int(column), 0.0) + float(value)
                    )
        for column, value in zip(current, wrong_weights):
            if value:
                error_coefficients[int(column)] = (
                    error_coefficients.get(int(column), 0.0) - float(value)
                )
    error_columns = np.fromiter(error_coefficients.keys(), dtype=np.int64)
    error_values = np.fromiter(error_coefficients.values(), dtype=float)
    if len(error_columns):
        row_parts.append(np.full(len(error_columns), next_row, dtype=np.int64))
        column_parts.append(error_columns)
        value_parts.append(error_values)
        lower_parts.append(np.array([-np.inf]))
        upper_parts.append(np.array([float(error_budget) - error_constant]))
        next_row += 1
    elif error_constant > error_budget:
        raise AssertionError("oracle-only policy unexpectedly violates the error budget")

    matrix = coo_matrix(
        (
            np.concatenate(value_parts) if value_parts else np.empty(0),
            (
                np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64),
                np.concatenate(column_parts)
                if column_parts
                else np.empty(0, dtype=np.int64),
            ),
        ),
        shape=(next_row, n_variables),
    ).tocsr()
    matrix.eliminate_zeros()
    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:n_acceptance] = 1
    elapsed = time.monotonic() - started
    preprocessing_seconds = (
        float(compression["threshold_pruning_time"])
        + float(compression["bucket_build_time"])
        + float(aggregation["pattern_aggregation_time"])
    )
    model_memory_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
        + objective.nbytes
        + integrality.nbytes
        + sum(part.nbytes for part in lower_parts)
        + sum(part.nbytes for part in upper_parts)
        + int(patterns.memory_usage(index=True, deep=True).sum())
    )
    statistics = {
        **compression,
        **aggregation,
        "L": n_stages,
        "error_budget_B": error_budget,
        "total_variables": n_variables,
        "binary_variables": n_acceptance,
        "continuous_variables": n_survival,
        "constraints": next_row,
        "sparse_nonzeros": matrix.nnz,
        "model_construction_seconds": elapsed,
        "model_build_time": max(0.0, elapsed - preprocessing_seconds),
        "model_memory_estimate_mb": model_memory_bytes / (1024**2),
        "solver_threads": HIGHS_THREADS,
    }
    if report:
        print(
            "[survival-milp] "
            + " ".join(f"{key}={value}" for key, value in statistics.items()),
            flush=True,
        )
    variable_lower = np.zeros(n_variables, dtype=float)
    variable_upper = np.ones(n_variables, dtype=float)
    return {
        "patterns": patterns,
        "domains": domains,
        "acceptance_layout": acceptance_layout,
        "pattern_acceptance": pattern_acceptance,
        "n_acceptance": n_acceptance,
        "matrix": matrix,
        "lower": np.concatenate(lower_parts) if lower_parts else np.empty(0),
        "upper": np.concatenate(upper_parts) if upper_parts else np.empty(0),
        "objective": objective,
        "integrality": integrality,
        "variable_lower": variable_lower,
        "variable_upper": variable_upper,
        "bounds": Bounds(variable_lower, variable_upper),
        "statistics": statistics,
        "n_documents": len(frame),
        "n_stages": n_stages,
        "error_budget": error_budget,
        "restricted_grid_spec": grid_spec,
    }


def _prefixes_to_thresholds(
    domains: Sequence[dict[int, CompressedThresholdDomain]],
    prefix_lengths: Sequence[Sequence[int]],
) -> list[tuple[float, float]]:
    thresholds: list[tuple[float, float]] = []
    for stage_index, stage_domains in enumerate(domains):
        stage_thresholds = []
        for label in (0, 1):
            domain = stage_domains[label]
            q = int(prefix_lengths[stage_index][label])
            if q < 0 or q >= len(domain.candidate_prefixes):
                raise ValueError("compressed prefix index is outside its legal domain")
            stage_thresholds.append(_threshold_for_prefix_candidate(domain, q))
        thresholds.append((stage_thresholds[0], stage_thresholds[1]))
    return thresholds


def _solution_to_thresholds(
    model: dict[str, object], solution: np.ndarray
) -> list[tuple[float, float]]:
    prefixes: list[list[int]] = []
    for stage_index, stage_layout in enumerate(model["acceptance_layout"]):
        stage_prefixes = []
        for label in (0, 1):
            indices = stage_layout[label]
            selected = solution[indices] >= 0.5
            if selected.any():
                first_rejected = np.flatnonzero(~selected)
                if len(first_rejected) and selected[first_rejected[0] :].any():
                    raise AssertionError("MILP returned a non-monotone threshold policy")
            stage_prefixes.append(int(selected.sum()))
        prefixes.append(stage_prefixes)
    return _prefixes_to_thresholds(model["domains"], prefixes)


def _evaluate_compressed_prefixes(
    model: dict[str, object], prefix_lengths: Sequence[Sequence[int]]
) -> dict[str, int]:
    patterns = model["patterns"]
    n_patterns = len(patterns)
    n_stages = int(model["n_stages"])
    oracle = patterns["oracle_label"].to_numpy(np.int8)
    weights = patterns["weight"].to_numpy(np.int64)
    route = np.full(n_patterns, -1, dtype=np.int16)
    routed_prediction = oracle.copy()
    for stage_index, stage in enumerate(range(1, n_stages + 1)):
        prediction = patterns[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        bucket = patterns[f"stage_{stage:02d}_bucket"].to_numpy(np.int32)
        limits = np.where(
            prediction == 0,
            int(prefix_lengths[stage_index][0]),
            int(prefix_lengths[stage_index][1]),
        )
        accepted = (route < 0) & (bucket >= 0) & (bucket < limits)
        route[accepted] = stage_index
        routed_prediction[accepted] = prediction[accepted]
    proxy = route >= 0
    return {
        "coverage": int(weights[proxy].sum()),
        "errors": int(weights[proxy & (routed_prediction != oracle)].sum()),
    }


def _normalized_native_prefixes(
    model: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
) -> list[list[int]] | None:
    if any(
        "native_threshold_0" not in metadata
        or "native_threshold_1" not in metadata
        for metadata in stage_metadata
    ):
        return None
    prefixes: list[list[int]] = []
    for stage_index, stage_domains in enumerate(model["domains"]):
        stage_prefixes = []
        for label in (0, 1):
            domain = stage_domains[label]
            threshold = float(stage_metadata[stage_index][f"native_threshold_{label}"])
            raw_prefix = int(np.count_nonzero(domain.scores_descending >= threshold))
            q = int(np.searchsorted(domain.candidate_prefixes, raw_prefix, side="left"))
            if q >= len(domain.candidate_prefixes):
                return None
            stage_prefixes.append(q)
        prefixes.append(stage_prefixes)
    return prefixes


def _greedy_multistage_incumbent(model: dict[str, object]) -> tuple[list[list[int]], dict[str, int]]:
    """Build a feasible incumbent using true incremental cascade routing."""

    prefixes = [[0, 0] for _ in range(int(model["n_stages"]))]
    current = _evaluate_compressed_prefixes(model, prefixes)
    budget = int(model["error_budget"])
    while True:
        moves = []
        for stage_index, stage_domains in enumerate(model["domains"]):
            for label in (0, 1):
                if prefixes[stage_index][label] >= stage_domains[label].n_acceptance_buckets:
                    continue
                trial = [values.copy() for values in prefixes]
                trial[stage_index][label] += 1
                result = _evaluate_compressed_prefixes(model, trial)
                if result["errors"] > budget:
                    continue
                delta_coverage = result["coverage"] - current["coverage"]
                delta_errors = result["errors"] - current["errors"]
                if delta_coverage <= 0 and delta_errors >= 0:
                    continue
                if delta_errors < 0:
                    priority = (3, -delta_errors, delta_coverage)
                elif delta_errors == 0:
                    priority = (2, delta_coverage, 0)
                else:
                    priority = (1, delta_coverage / delta_errors, delta_coverage)
                moves.append((priority, -stage_index, -label, trial, result))
        if not moves:
            break
        _, _, _, prefixes, current = max(moves, key=lambda item: item[:3])
    return prefixes, current


def _initial_multistage_incumbent(
    frame: pd.DataFrame,
    model: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
    *,
    report: bool = True,
    use_greedy: bool = True,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    native_prefixes = _normalized_native_prefixes(model, stage_metadata)
    if native_prefixes is not None:
        native_result = _evaluate_compressed_prefixes(model, native_prefixes)
        if native_result["errors"] <= int(model["error_budget"]):
            candidates.append(
                {
                    "source": "native_taskcascades",
                    "prefixes": native_prefixes,
                    **native_result,
                }
            )
    if use_greedy:
        greedy_prefixes, greedy_result = _greedy_multistage_incumbent(model)
        candidates.append(
            {"source": "greedy", "prefixes": greedy_prefixes, **greedy_result}
        )
    else:
        prefixes = [[0, 0] for _ in range(int(model["n_stages"]))]
        final_scan = _exact_final_stage_prefix_scan(
            frame,
            int(model["error_budget"]),
            stage=int(model["n_stages"]),
        )
        final_stage_index = int(model["n_stages"]) - 1
        for label, threshold in enumerate(final_scan["thresholds"]):
            domain = model["domains"][final_stage_index][label]
            raw_prefix = (
                0
                if math.isinf(float(threshold))
                else int(np.count_nonzero(domain.scores_descending >= threshold))
            )
            q = int(
                np.searchsorted(
                    domain.candidate_prefixes, raw_prefix, side="left"
                )
            )
            prefixes[final_stage_index][label] = min(
                q, len(domain.candidate_prefixes) - 1
            )
        final_result = _evaluate_compressed_prefixes(model, prefixes)
        if final_result["errors"] <= int(model["error_budget"]):
            candidates.append(
                {
                    "source": "analytical_final_stage",
                    "prefixes": prefixes,
                    **final_result,
                }
            )
        reject_prefixes = [[0, 0] for _ in range(int(model["n_stages"]))]
        candidates.append(
            {
                "source": "reject_all",
                "prefixes": reject_prefixes,
                **_evaluate_compressed_prefixes(model, reject_prefixes),
            }
        )
    incumbent = max(
        candidates,
        key=lambda item: (int(item["coverage"]), -int(item["errors"])),
    )
    incumbent["thresholds"] = _prefixes_to_thresholds(
        model["domains"], incumbent["prefixes"]
    )
    if report:
        print(
            f"[initial-incumbent] initial_incumbent_source={incumbent['source']} "
            f"initial_proxy_coverage={incumbent['coverage']} "
            f"initial_errors={incumbent['errors']} "
            f"initial_kappa={int(incumbent['coverage']) / len(frame):.12f}",
            flush=True,
        )
    return incumbent


def _all_wrong_coverage_upper_bound(model: dict[str, object]) -> dict[str, int | float]:
    patterns = model["patterns"]
    oracle = patterns["oracle_label"].to_numpy(np.int8)
    all_wrong = np.ones(len(patterns), dtype=bool)
    for stage in range(1, int(model["n_stages"]) + 1):
        bucket = patterns[f"stage_{stage:02d}_bucket"].to_numpy(np.int32)
        prediction = patterns[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        all_wrong &= (bucket < 0) | (prediction != oracle)
    h = int(patterns.loc[all_wrong, "weight"].sum())
    n_documents = int(model["n_documents"])
    budget = int(model["error_budget"])
    coverage = n_documents - h + min(h, budget)
    return {
        "all_wrong_H": h,
        "coverage_upper_bound": coverage,
        "kappa_upper_bound": coverage / n_documents,
    }


def _domain_threshold_allowlist(
    domains: Sequence[dict[int, CompressedThresholdDomain]],
) -> dict[tuple[int, int], tuple[float, ...]]:
    return {
        (int(domain.stage), label): tuple(
            _threshold_for_prefix_candidate(domain, index)
            for index in range(domain.retained_threshold_count)
        )
        for stage_domains in domains
        for label, domain in stage_domains.items()
    }


def _threshold_coupled_coverage_upper_bound(
    frame: pd.DataFrame,
    domains: Sequence[dict[int, CompressedThresholdDomain]],
    error_budget: int,
    *,
    coverage_offset: int = 0,
) -> dict[str, int]:
    """Return a safe bound that couples correct coverage to poison prefixes.

    A record can be counted as error-free proxy coverage only if at least one
    remaining stage predicts it correctly and a legal threshold accepting it
    does not, by itself, admit more globally-all-wrong records than the budget.
    Every other routed record is optimistically charged one error.  This
    ignores harmful first-accept interactions and threshold sharing, so it is
    a relaxation and therefore an upper bound on the real cascade optimum.
    """

    n_stages = len(domains)
    oracle = frame["oracle_label"].to_numpy(np.int8)
    all_wrong = np.ones(len(frame), dtype=bool)
    eligible_by_stage: list[np.ndarray] = []
    for stage in range(1, n_stages + 1):
        eligible_column = f"stage_{stage:02d}_eligible"
        eligible = (
            frame[eligible_column].to_numpy(bool)
            if eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        eligible_by_stage.append(eligible)
        all_wrong &= (~eligible) | (prediction != oracle)

    correctly_routable = np.zeros(len(frame), dtype=bool)
    for stage_index, stage in enumerate(range(1, n_stages + 1)):
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        confidence = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
        eligible = eligible_by_stage[stage_index]
        for label in (0, 1):
            domain = domains[stage_index][label]
            if not len(domain.scores_descending):
                continue
            mask = eligible & (prediction == label)
            ranks = np.searchsorted(
                -domain.scores_descending, -confidence[mask]
            )
            poison_by_group = np.bincount(
                ranks,
                weights=all_wrong[mask].astype(np.int64),
                minlength=len(domain.scores_descending),
            ).astype(np.int64)
            cumulative_poison = np.concatenate(
                ([0], np.cumsum(poison_by_group, dtype=np.int64))
            )
            feasible_prefixes = domain.candidate_prefixes[
                cumulative_poison[domain.candidate_prefixes] <= error_budget
            ]
            maximum_feasible_prefix = (
                int(feasible_prefixes[-1]) if len(feasible_prefixes) else 0
            )
            correct_mask = mask & (oracle == label)
            if maximum_feasible_prefix:
                correct_ranks = np.searchsorted(
                    -domain.scores_descending, -confidence[correct_mask]
                )
                correct_indices = np.flatnonzero(correct_mask)
                correctly_routable[
                    correct_indices[correct_ranks < maximum_feasible_prefix]
                ] = True

    free_correct = int(np.count_nonzero(correctly_routable))
    must_spend_error = len(frame) - free_correct
    local_bound = free_correct + min(must_spend_error, max(0, error_budget))
    return {
        "coverage_upper_bound": coverage_offset + local_bound,
        "correctly_routable": free_correct,
        "must_spend_error_if_routed": must_spend_error,
        "all_wrong": int(np.count_nonzero(all_wrong)),
    }


def _boolean_mask_bits(mask: np.ndarray) -> int:
    return int.from_bytes(
        np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes(),
        "little",
    )


def _exact_union_relaxation_upper_bound(
    frame: pd.DataFrame,
    domains: Sequence[dict[int, CompressedThresholdDomain]],
    error_budget: int,
    *,
    coverage_offset: int = 0,
    time_limit: float | None = None,
) -> dict[str, int | float | bool]:
    """Solve the optimistic threshold-union relaxation with exact bitsets.

    The relaxation ignores harmful first-accept ordering and charges errors
    only for records that are wrong at every remaining eligible stage.  Any
    real policy maps to a feasible relaxation policy with at least as much
    relaxed coverage, so a completed search is a valid upper bound.
    """

    started = time.monotonic()
    oracle = frame["oracle_label"].to_numpy(np.int8)
    all_wrong = np.ones(len(frame), dtype=bool)
    dimensions: list[list[int]] = []
    for stage_index, stage in enumerate(range(1, len(domains) + 1)):
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        confidence = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
        eligible_column = f"stage_{stage:02d}_eligible"
        eligible = (
            frame[eligible_column].to_numpy(bool)
            if eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        all_wrong &= (~eligible) | (prediction != oracle)
        for label in (0, 1):
            domain = domains[stage_index][label]
            accepted_values = []
            for candidate_index in range(domain.retained_threshold_count):
                threshold = _threshold_for_prefix_candidate(
                    domain, candidate_index
                )
                accepted_values.append(
                    _boolean_mask_bits(
                        eligible
                        & (prediction == label)
                        & (confidence >= threshold)
                    )
                )
            dimensions.append(accepted_values)

    poison_bits = _boolean_mask_bits(all_wrong)
    dimensions.sort(key=len)
    suffix_union = [0] * (len(dimensions) + 1)
    for index in range(len(dimensions) - 1, -1, -1):
        suffix_union[index] = suffix_union[index + 1] | dimensions[index][-1]

    best = 0
    nodes = 0
    timed_out = False
    seen: set[tuple[int, int]] = set()

    def search(index: int, accepted_union: int) -> None:
        nonlocal best, nodes, timed_out
        nodes += 1
        if (
            time_limit is not None
            and nodes % 1024 == 0
            and time.monotonic() - started > time_limit
        ):
            timed_out = True
            return
        if (accepted_union | suffix_union[index]).bit_count() <= best:
            return
        state = (index, accepted_union)
        if state in seen:
            return
        seen.add(state)
        if index == len(dimensions):
            best = max(best, accepted_union.bit_count())
            return

        choices: list[int] = []
        last_poison: int | None = None
        for accepted in dimensions[index]:
            combined = accepted_union | accepted
            combined_poison = combined & poison_bits
            if combined_poison.bit_count() > error_budget:
                continue
            # Within one nested threshold chain, the most aggressive choice
            # with an identical poison union dominates all earlier choices.
            if last_poison == combined_poison:
                choices[-1] = combined
            else:
                choices.append(combined)
                last_poison = combined_poison
        choices.sort(key=int.bit_count, reverse=True)
        for combined in choices:
            search(index + 1, combined)
            if timed_out:
                return

    search(0, 0)
    return {
        "coverage_upper_bound": coverage_offset + best,
        "local_relaxation_optimum": best,
        "exact": not timed_out,
        "nodes": nodes,
        "elapsed_seconds": time.monotonic() - started,
    }


def _exact_final_stage_prefix_scan(
    frame: pd.DataFrame,
    error_budget: int,
    *,
    stage: int = 1,
    domains: dict[int, CompressedThresholdDomain] | None = None,
) -> dict[str, object]:
    """Optimize the two class-specific prefixes of one final stage exactly."""

    if error_budget < 0:
        return {
            "feasible": False,
            "coverage": -1,
            "errors": -1,
            "thresholds": (math.inf, math.inf),
            "number_of_optima": 0,
        }
    oracle = frame["oracle_label"].to_numpy(np.int8)
    prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
    confidence = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
    eligible_column = f"stage_{stage:02d}_eligible"
    eligible = (
        frame[eligible_column].to_numpy(bool)
        if eligible_column in frame
        else np.ones(len(frame), dtype=bool)
    )
    prefix_data = {}
    for label in (0, 1):
        mask = eligible & (prediction == label)
        label_scores = confidence[mask]
        scores = np.unique(label_scores)[::-1]
        if len(scores):
            ranks = np.searchsorted(-scores, -label_scores)
            counts = np.bincount(ranks, minlength=len(scores)).astype(np.int64)
            errors = np.bincount(
                ranks,
                weights=(oracle[mask] != label).astype(np.int64),
                minlength=len(scores),
            ).astype(np.int64)
        else:
            counts = np.empty(0, dtype=np.int64)
            errors = np.empty(0, dtype=np.int64)
        coverage_prefix = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        error_prefix = np.concatenate(
            ([0], np.cumsum(errors, dtype=np.int64))
        )
        legal_prefixes = (
            domains[label].candidate_prefixes
            if domains is not None
            else np.arange(len(coverage_prefix), dtype=np.int64)
        )
        prefix_data[label] = (
            scores,
            coverage_prefix[legal_prefixes],
            error_prefix[legal_prefixes],
            legal_prefixes,
        )

    scores0, coverage0, errors0, legal0 = prefix_data[0]
    scores1, coverage1, errors1, legal1 = prefix_data[1]
    candidates = []
    for k0 in range(len(coverage0)):
        remaining = error_budget - int(errors0[k0])
        if remaining < 0:
            break
        k1 = int(np.searchsorted(errors1, remaining, side="right") - 1)
        candidates.append(
            (
                int(coverage0[k0] + coverage1[k1]),
                int(errors0[k0] + errors1[k1]),
                k0,
                k1,
            )
        )
    if not candidates:
        raise AssertionError("reject-all final-stage policy must be feasible")
    best_coverage = max(value[0] for value in candidates)
    optimal = [value for value in candidates if value[0] == best_coverage]
    coverage, errors, k0, k1 = min(
        optimal, key=lambda value: (value[1], value[2], value[3])
    )
    return {
        "feasible": True,
        "coverage": coverage,
        "errors": errors,
        "thresholds": (
            _threshold_for_prefix_candidate(domains[0], k0)
            if domains is not None
            else (math.inf if int(legal0[k0]) == 0 else float(scores0[int(legal0[k0]) - 1])),
            _threshold_for_prefix_candidate(domains[1], k1)
            if domains is not None
            else (math.inf if int(legal1[k1]) == 0 else float(scores1[int(legal1[k1]) - 1])),
        ),
        "prefixes": (k0, k1),
        "number_of_optima": len(optimal),
    }


def _solve_compressed_scipy_milp(model: dict[str, object]) -> dict[str, object]:
    options: dict[str, object] = {
        "presolve": True,
        "mip_abs_gap": 0.999,
        "threads": HIGHS_THREADS,
        "disp": HIGHS_SOLVER_LOG,
    }
    if HIGHS_TIME_LIMIT is not None:
        options["time_limit"] = HIGHS_TIME_LIMIT
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*", category=RuntimeWarning
        )
        result = milp(
            np.asarray(model["objective"], dtype=float),
            integrality=np.asarray(model["integrality"], dtype=np.int8),
            bounds=model["bounds"],
            constraints=LinearConstraint(
                model["matrix"], model["lower"], model["upper"]
            ),
            options=options,
        )
    elapsed = time.monotonic() - started
    primal = float(getattr(result, "fun", math.inf))
    dual = float(getattr(result, "mip_dual_bound", -math.inf))
    absolute_gap = primal - dual
    exact = bool(
        result.x is not None
        and math.isfinite(absolute_gap)
        and absolute_gap < 1.0 - 1e-7
    )
    return {
        "x": result.x,
        "exact": exact,
        "infeasible": result.status == 2,
        "status": str(result.message),
        "mip_gap": (
            math.inf
            if getattr(result, "mip_gap", None) is None
            else float(result.mip_gap)
        ),
        "mip_nodes": (
            0
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        "primal_oracle_count": primal,
        "dual_oracle_count": dual,
        "absolute_objective_gap": absolute_gap,
        "elapsed_seconds": elapsed,
        "backend": "scipy_highs",
    }


def _prefixes_to_solution_vector(
    model: dict[str, object], prefix_lengths: Sequence[Sequence[int]]
) -> np.ndarray:
    n_variables = len(model["objective"])
    n_acceptance = int(model["n_acceptance"])
    n_stages = int(model["n_stages"])
    solution = np.zeros(n_variables, dtype=float)
    for stage_index, layout in enumerate(model["acceptance_layout"]):
        for label in (0, 1):
            q = int(prefix_lengths[stage_index][label])
            solution[layout[label][:q]] = 1.0
    active = np.ones(len(model["patterns"]), dtype=bool)
    for stage_index, acceptance in enumerate(model["pattern_acceptance"]):
        accepted = active & (acceptance >= 0)
        accepted[accepted] &= solution[acceptance[accepted]] >= 0.5
        active[accepted] = False
        survival = (
            n_acceptance
            + np.arange(len(active), dtype=np.int64) * n_stages
            + stage_index
        )
        solution[survival] = active.astype(float)
    return solution


def _solve_compressed_highspy_milp(
    model: dict[str, object], incumbent: dict[str, object] | None
) -> dict[str, object]:
    """Solve through the installed native highspy 1.x interface."""

    import highspy

    # HiGHS keeps a process-global scheduler.  Reset it before applying this
    # worker's thread count; this matters in tests and sequential pilots that
    # deliberately change threads between solves.
    highspy.Highs.resetGlobalScheduler(True)
    matrix = model["matrix"].tocsr()
    lp = highspy.HighsLp()
    lp.num_col_ = matrix.shape[1]
    lp.num_row_ = matrix.shape[0]
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = np.asarray(model["objective"], dtype=np.float64)
    lp.col_lower_ = np.asarray(model["variable_lower"], dtype=np.float64)
    lp.col_upper_ = np.asarray(model["variable_upper"], dtype=np.float64)
    lp.row_lower_ = np.asarray(model["lower"], dtype=np.float64)
    lp.row_upper_ = np.asarray(model["upper"], dtype=np.float64)
    lp.integrality_ = [
        highspy.HighsVarType.kInteger
        if value
        else highspy.HighsVarType.kContinuous
        for value in np.asarray(model["integrality"], dtype=np.int8)
    ]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.num_col_ = matrix.shape[1]
    lp.a_matrix_.num_row_ = matrix.shape[0]
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32, copy=False)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32, copy=False)
    lp.a_matrix_.value_ = matrix.data.astype(np.float64, copy=False)

    solver = highspy.Highs()
    solver.setOptionValue("threads", HIGHS_THREADS)
    solver.setOptionValue("presolve", "on")
    solver.setOptionValue("mip_abs_gap", 0.999)
    solver.setOptionValue("output_flag", HIGHS_SOLVER_LOG)
    if HIGHS_TIME_LIMIT is not None:
        solver.setOptionValue("time_limit", HIGHS_TIME_LIMIT)
    pass_status = solver.passModel(lp)
    if pass_status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"highspy rejected the compressed model: {pass_status}")

    if incumbent is not None:
        incumbent_vector = _prefixes_to_solution_vector(model, incumbent["prefixes"])
        indices = np.arange(len(incumbent_vector), dtype=np.int32)
        start_status = solver.setSolution(len(indices), indices, incumbent_vector)
        if start_status != highspy.HighsStatus.kOk:
            warnings.warn(f"highspy rejected the feasible MIP start: {start_status}")

    last_report = [-math.inf]

    def progress_callback(callback_type, message, output, callback_input, user_data):
        del callback_type, message, callback_input, user_data
        elapsed = float(output.running_time)
        if elapsed - last_report[0] < 10.0:
            return
        last_report[0] = elapsed
        primal = float(output.mip_primal_bound)
        dual = float(output.mip_dual_bound)
        remaining_documents = int(model["n_documents"])
        coverage_offset = int(model.get("coverage_offset", 0))
        total_documents = int(model.get("total_documents", remaining_documents))
        incumbent_coverage = (
            coverage_offset + remaining_documents - primal
            if math.isfinite(primal)
            else math.nan
        )
        best_coverage_bound = (
            coverage_offset + remaining_documents - dual
            if math.isfinite(dual)
            else math.nan
        )
        absolute_coverage_gap = (
            best_coverage_bound - incumbent_coverage
            if math.isfinite(incumbent_coverage) and math.isfinite(best_coverage_bound)
            else math.nan
        )
        print(
            f"[mip-progress] partition={model.get('partition_label', 'full')} "
            f"elapsed_seconds={elapsed:.1f} "
            f"incumbent_coverage={incumbent_coverage:.6g} "
            f"incumbent_kappa={incumbent_coverage / total_documents:.12g} "
            f"best_bound={best_coverage_bound:.6g} "
            f"absolute_coverage_gap={absolute_coverage_gap:.6g} "
            f"relative_mip_gap={float(output.mip_gap):.6g} "
            f"node_count={int(output.mip_node_count)}",
            flush=True,
        )

    solver.setCallback(progress_callback, None)
    solver.startCallback(highspy.cb.HighsCallbackType.kCallbackLogging)
    solver.startCallback(highspy.cb.HighsCallbackType.kCallbackMipLogging)
    started = time.monotonic()
    solver.run()
    elapsed = time.monotonic() - started
    status = solver.getModelStatus()
    info = solver.getInfo()
    solution = solver.getSolution()
    primal = float(info.objective_function_value)
    dual = float(info.mip_dual_bound)
    absolute_gap = primal - dual
    exact = bool(
        solution.value_valid
        and math.isfinite(absolute_gap)
        and absolute_gap < 1.0 - 1e-7
    )
    infeasible = status == highspy.HighsModelStatus.kInfeasible
    return {
        "x": np.asarray(solution.col_value, dtype=float)
        if solution.value_valid
        else None,
        "exact": exact,
        "infeasible": infeasible,
        "status": solver.modelStatusToString(status),
        "mip_gap": float(info.mip_gap),
        "mip_nodes": int(info.mip_node_count),
        "primal_oracle_count": primal,
        "dual_oracle_count": dual,
        "absolute_objective_gap": absolute_gap,
        "elapsed_seconds": elapsed,
        "backend": f"highspy_{solver.version()}",
    }


def _solve_compressed_milp(
    model: dict[str, object], incumbent: dict[str, object] | None
) -> dict[str, object]:
    if MIP_BACKEND not in {"auto", "highspy", "scipy"}:
        raise ValueError(f"unsupported MILP backend {MIP_BACKEND!r}")
    highspy_available = False
    if MIP_BACKEND != "scipy":
        try:
            import highspy  # noqa: F401

            highspy_available = True
        except ImportError:
            if MIP_BACKEND == "highspy":
                raise RuntimeError(
                    "--solver-backend=highspy requested, but highspy is not installed"
                )
    if highspy_available:
        return _solve_compressed_highspy_milp(model, incumbent)
    return _solve_compressed_scipy_milp(model)


_FORKED_DECOMPOSITION_FRAME: pd.DataFrame | None = None


_FORKED_DECOMPOSITION_STAGE_METADATA: Sequence[dict[str, object]] | None = None


_FORKED_GLOBAL_INCUMBENT = None


_FORKED_THRESHOLD_ALLOWLIST: dict[tuple[int, int], tuple[float, ...]] | None = None


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _linux_physical_cpu_ids() -> list[int]:
    topology_root = Path("/sys/devices/system/cpu")
    representatives: dict[tuple[int, int], int] = {}
    try:
        for cpu_path in topology_root.glob("cpu[0-9]*"):
            cpu = int(cpu_path.name[3:])
            topology = cpu_path / "topology"
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
            key = (package, core)
            representatives[key] = min(cpu, representatives.get(key, cpu))
    except (OSError, ValueError):
        return []
    return [representatives[key] for key in sorted(representatives)]


def _linux_numa_topology() -> dict[str, str]:
    topology = {}
    try:
        for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
            topology[node.name] = (node / "cpulist").read_text().strip()
    except OSError:
        return {}
    return topology


def _initialize_solver_worker(counter, cpu_ids: Sequence[int]) -> None:
    """Keep each solver process single-threaded and optionally pin a core."""

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    with counter.get_lock():
        index = int(counter.value)
        counter.value += 1
    os.sched_setaffinity(0, {int(cpu_ids[index % len(cpu_ids)])})


def _initialize_worker_affinity(counter, cpu_ids: Sequence[int]) -> None:
    """Compatibility alias retained for older profiling entry points."""

    _initialize_solver_worker(counter, cpu_ids)


def _threshold_for_prefix_candidate(
    domain: CompressedThresholdDomain, candidate_index: int
) -> float:
    if domain.candidate_thresholds is not None:
        return float(domain.candidate_thresholds[candidate_index])
    raw_prefix = int(domain.candidate_prefixes[candidate_index])
    return (
        math.inf
        if raw_prefix == 0
        else float(domain.scores_descending[raw_prefix - 1])
    )


def _stage1_split_candidates(
    frame: pd.DataFrame,
    domains: Sequence[dict[int, CompressedThresholdDomain]],
    error_budget: int,
    *,
    report: bool = True,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Score both stage-1 class dimensions and return the exact chosen domain."""

    oracle = frame["oracle_label"].to_numpy(np.int8)
    prediction = frame["stage_01_prediction"].to_numpy(np.int8)
    summaries = []
    per_class_tables = []
    for label in (0, 1):
        domain = domains[0][label]
        cumulative_count = np.concatenate(
            ([0], np.cumsum(domain.group_counts, dtype=np.int64))
        )
        cumulative_errors = np.concatenate(
            ([0], np.cumsum(domain.group_errors, dtype=np.int64))
        )
        records = []
        for candidate_index, raw_prefix in enumerate(domain.candidate_prefixes):
            accepted = int(cumulative_count[int(raw_prefix)])
            errors = int(cumulative_errors[int(raw_prefix)])
            records.append(
                {
                    "class": label,
                    "candidate_index": candidate_index,
                    "raw_prefix": int(raw_prefix),
                    "threshold": _threshold_for_prefix_candidate(
                        domain, candidate_index
                    ),
                    "accepted_count": accepted,
                    "rejected_count": int(np.count_nonzero(prediction == label))
                    - accepted,
                    "stage1_error_count": errors,
                    "immediately_infeasible": errors > error_budget,
                }
            )
        table = pd.DataFrame(records)
        per_class_tables.append(table)
        affected = int(np.count_nonzero(prediction == label))
        routing_span = int(table["accepted_count"].max() - table["accepted_count"].min())
        mean_accepted = float(table["accepted_count"].mean())
        # This deliberately rewards routing impact and expected document
        # removal, with only a square-root penalty for domain size.  It does
        # not mechanically choose the smaller candidate set.
        impact_score = routing_span * max(1.0, mean_accepted) / math.sqrt(max(1, len(table)))
        summaries.append(
            {
                "class": label,
                "K": len(table),
                "affected_records": affected,
                "routing_span": routing_span,
                "mean_accepted": mean_accepted,
                "min_stage1_errors": int(table["stage1_error_count"].min()),
                "max_stage1_errors": int(table["stage1_error_count"].max()),
                "impact_score": impact_score,
                "correct_stage1_records": int(
                    np.count_nonzero((prediction == label) & (oracle == label))
                ),
                "wrong_stage1_records": int(
                    np.count_nonzero((prediction == label) & (oracle != label))
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    chosen_label = int(
        summary.sort_values(
            ["impact_score", "affected_records", "K"],
            ascending=[False, False, True],
        ).iloc[0]["class"]
    )
    candidates = per_class_tables[chosen_label].copy()
    if report:
        for row in summary.itertuples(index=False):
            print(
                f"[split-dimension-candidate] stage=1 class={row.class_ if hasattr(row, 'class_') else row[0]} "
                f"K={row.K} affected_records={row.affected_records} "
                f"routing_span={row.routing_span} mean_accepted={row.mean_accepted:.1f} "
                f"stage1_error_range={row.min_stage1_errors}-{row.max_stage1_errors} "
                f"impact_score={row.impact_score:.3f}",
                flush=True,
            )
        print(
            f"[split-dimension-selected] stage=1 class={chosen_label} "
            f"K={len(candidates)} reason=maximum_routing_impact_score",
            flush=True,
        )
    return chosen_label, candidates, summary


def _first_level_branch_descriptors(
    frame: pd.DataFrame,
    split_label: int,
    candidates: pd.DataFrame,
    error_budget: int,
) -> list[dict[str, object]]:
    oracle = frame["oracle_label"].to_numpy(np.int8)
    stage1_prediction = frame["stage_01_prediction"].to_numpy(np.int8)
    stage1_confidence = frame["stage_01_confidence"].to_numpy(float)
    later_has_correct = np.zeros(len(frame), dtype=bool)
    n_stages = sum(
        column.startswith("stage_") and column.endswith("_prediction")
        for column in frame.columns
    )
    for stage in range(2, n_stages + 1):
        later_has_correct |= (
            frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8) == oracle
        )
    descriptors = []
    for row in candidates.to_dict(orient="records"):
        threshold = float(row["threshold"])
        accepted = (stage1_prediction == split_label) & (
            stage1_confidence >= threshold
        )
        remaining = ~accepted
        forced_routed = int(accepted.sum())
        forced_errors = int(np.count_nonzero(accepted & (stage1_prediction != oracle)))
        remaining_budget = error_budget - forced_errors
        # The fixed class cannot accept the rejected records at stage 1.  The
        # other class remains a live threshold dimension.
        stage1_other_class_correct = (
            (stage1_prediction != split_label) & (stage1_prediction == oracle)
        )
        has_free_correct_route = later_has_correct | stage1_other_class_correct
        good = int(np.count_nonzero(remaining & has_free_correct_route))
        all_wrong = int(np.count_nonzero(remaining & ~has_free_correct_route))
        upper_bound = (
            forced_routed + good + min(all_wrong, max(0, remaining_budget))
            if remaining_budget >= 0
            else -1
        )
        descriptors.append(
            {
                **row,
                "branch_id": f"stage1_class{split_label}_q{int(row['candidate_index'])}",
                "parent_id": None,
                "depth": 1,
                "recursion_depth": 0,
                "split_label": split_label,
                "fixed_thresholds": [
                    {
                        "stage": 1,
                        "label": split_label,
                        "candidate_index": int(row["candidate_index"]),
                        "threshold": threshold,
                    }
                ],
                "forced_routed": forced_routed,
                "forced_errors": forced_errors,
                "remaining_N": int(remaining.sum()),
                "remaining_budget": remaining_budget,
                "G": good,
                "H": all_wrong,
                "upper_bound_routed": upper_bound,
                "status": "pending" if remaining_budget >= 0 else "infeasible",
            }
        )
    return descriptors


def _propagate_first_level_branch(
    frame: pd.DataFrame, descriptor: dict[str, object]
) -> pd.DataFrame:
    split_label = int(descriptor["split_label"])
    threshold = float(descriptor["threshold"])
    prediction = frame["stage_01_prediction"].to_numpy(np.int8)
    confidence = frame["stage_01_confidence"].to_numpy(float)
    accepted = (prediction == split_label) & (confidence >= threshold)
    remaining = frame.loc[~accepted].copy().reset_index(drop=True)
    eligible = np.ones(len(remaining), dtype=bool)
    eligible[
        remaining["stage_01_prediction"].to_numpy(np.int8) == split_label
    ] = False
    remaining["stage_01_eligible"] = eligible
    return remaining


def _drop_resolved_first_stage(frame: pd.DataFrame, n_stages: int) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "document_id": frame["document_id"].to_numpy(),
            "oracle_label": frame["oracle_label"].to_numpy(np.int8),
        }
    )
    for old_stage in range(2, n_stages + 1):
        new_stage = old_stage - 1
        for suffix in ("prediction", "confidence"):
            output[f"stage_{new_stage:02d}_{suffix}"] = frame[
                f"stage_{old_stage:02d}_{suffix}"
            ].to_numpy()
        eligible_column = f"stage_{old_stage:02d}_eligible"
        if eligible_column in frame:
            output[f"stage_{new_stage:02d}_eligible"] = frame[
                eligible_column
            ].to_numpy(bool)
    return output


def _second_level_branch_descriptors(
    frame: pd.DataFrame,
    parent: dict[str, object],
    n_stages: int,
) -> list[dict[str, object]]:
    first_remaining = _propagate_first_level_branch(frame, parent)
    second_label = 1 - int(parent["split_label"])
    domains, _, _ = _compress_multistage_thresholds(
        first_remaining,
        n_stages,
        int(parent["remaining_budget"]),
        report=False,
    )
    domain = domains[0][second_label]
    cumulative_count = np.concatenate(
        ([0], np.cumsum(domain.group_counts, dtype=np.int64))
    )
    cumulative_errors = np.concatenate(
        ([0], np.cumsum(domain.group_errors, dtype=np.int64))
    )
    oracle = first_remaining["oracle_label"].to_numpy(np.int8)
    prediction = first_remaining["stage_01_prediction"].to_numpy(np.int8)
    confidence = first_remaining["stage_01_confidence"].to_numpy(float)
    later_has_correct = np.zeros(len(first_remaining), dtype=bool)
    for stage in range(2, n_stages + 1):
        later_has_correct |= (
            first_remaining[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
            == oracle
        )
    second_mask = prediction == second_label
    if len(domain.scores_descending):
        second_ranks = np.searchsorted(
            -domain.scores_descending, -confidence[second_mask]
        )
        correct_future_by_group = np.bincount(
            second_ranks,
            weights=later_has_correct[second_mask].astype(np.int64),
            minlength=len(domain.scores_descending),
        ).astype(np.int64)
    else:
        correct_future_by_group = np.empty(0, dtype=np.int64)
    cumulative_correct_future = np.concatenate(
        ([0], np.cumsum(correct_future_by_group, dtype=np.int64))
    )
    total_future_correct = int(np.count_nonzero(later_has_correct))
    children = []
    for candidate_index, raw_prefix in enumerate(domain.candidate_prefixes):
        threshold = _threshold_for_prefix_candidate(domain, candidate_index)
        additional_routed = int(cumulative_count[int(raw_prefix)])
        additional_errors = int(cumulative_errors[int(raw_prefix)])
        forced_routed = int(parent["forced_routed"]) + additional_routed
        forced_errors = int(parent["forced_errors"]) + additional_errors
        remaining_budget = int(parent["remaining_budget"]) - additional_errors
        remaining_n = len(first_remaining) - additional_routed
        good = total_future_correct - int(
            cumulative_correct_future[int(raw_prefix)]
        )
        all_wrong = remaining_n - good
        upper_bound = (
            forced_routed + good + min(all_wrong, max(0, remaining_budget))
            if remaining_budget >= 0
            else -1
        )
        children.append(
            {
                **parent,
                "branch_id": f"{parent['branch_id']}_class{second_label}_q{candidate_index}",
                "parent_id": str(parent["branch_id"]),
                "depth": 2,
                "recursion_depth": int(parent.get("recursion_depth", 0)) + 1,
                "root_error_budget": int(
                    parent.get(
                        "root_error_budget",
                        int(parent["remaining_budget"])
                        + int(parent["forced_errors"]),
                    )
                ),
                "root_n_stages": int(parent.get("root_n_stages", n_stages)),
                "fixed_thresholds": [
                    *list(parent.get("fixed_thresholds", [])),
                    {
                        "stage": 1,
                        "label": second_label,
                        "candidate_index": candidate_index,
                        "threshold": threshold,
                    },
                ],
                "second_label": second_label,
                "second_candidate_index": candidate_index,
                "second_raw_prefix": int(raw_prefix),
                "second_threshold": threshold,
                "additional_forced_routed": additional_routed,
                "additional_forced_errors": additional_errors,
                "forced_routed": forced_routed,
                "forced_errors": forced_errors,
                "remaining_N": remaining_n,
                "remaining_budget": remaining_budget,
                "G": good,
                "H": all_wrong,
                "upper_bound_routed": upper_bound,
                "status": "pending" if remaining_budget >= 0 else "infeasible",
            }
        )
    return children


def _materialize_recursive_branch(
    frame: pd.DataFrame,
    descriptor: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
) -> tuple[pd.DataFrame, Sequence[dict[str, object]], dict[str, int]]:
    """Replay fixed decisions and return the smallest causally valid suffix.

    Decisions are allowed only at the earliest unresolved stage.  Once both
    class thresholds at a stage are fixed, accepted rows are removed and the
    stage is dropped.  With one class fixed, rows rejected by that class stay
    in the suffix but are marked ineligible for that already-fixed decision.
    """

    n_stages = len(stage_metadata)
    decisions = {
        (int(item["stage"]), int(item["label"])): float(item["threshold"])
        for item in descriptor.get("fixed_thresholds", [])
    }
    root_error_budget = int(
        descriptor.get(
            "root_error_budget",
            int(descriptor.get("remaining_budget", 0))
            + int(descriptor.get("forced_errors", 0)),
        )
    )
    alive = np.ones(len(frame), dtype=bool)
    oracle = frame["oracle_label"].to_numpy(np.int8)
    forced_routed = 0
    forced_errors = 0
    first_remaining_stage = n_stages + 1
    partially_fixed: set[int] = set()
    for stage in range(1, n_stages + 1):
        fixed_labels = {
            label for label in (0, 1) if (stage, label) in decisions
        }
        if not fixed_labels:
            first_remaining_stage = stage
            break
        prediction = frame[f"stage_{stage:02d}_prediction"].to_numpy(np.int8)
        confidence = frame[f"stage_{stage:02d}_confidence"].to_numpy(float)
        accepted = np.zeros(len(frame), dtype=bool)
        for label in fixed_labels:
            accepted |= (
                alive
                & (prediction == label)
                & (confidence >= decisions[(stage, label)])
            )
        forced_routed += int(np.count_nonzero(accepted))
        forced_errors += int(np.count_nonzero(accepted & (prediction != oracle)))
        alive &= ~accepted
        if len(fixed_labels) < 2:
            first_remaining_stage = stage
            partially_fixed = fixed_labels
            break
    remaining_indices = np.flatnonzero(alive)
    output = pd.DataFrame(
        {
            "document_id": frame["document_id"].to_numpy()[remaining_indices],
            "oracle_label": oracle[remaining_indices],
        }
    )
    if first_remaining_stage <= n_stages:
        for old_stage in range(first_remaining_stage, n_stages + 1):
            new_stage = old_stage - first_remaining_stage + 1
            for suffix in ("prediction", "confidence"):
                output[f"stage_{new_stage:02d}_{suffix}"] = frame[
                    f"stage_{old_stage:02d}_{suffix}"
                ].to_numpy()[remaining_indices]
            original_eligible = f"stage_{old_stage:02d}_eligible"
            eligible = (
                frame[original_eligible].to_numpy(bool)[remaining_indices]
                if original_eligible in frame
                else np.ones(len(remaining_indices), dtype=bool)
            )
            if old_stage == first_remaining_stage and partially_fixed:
                prediction = output[f"stage_{new_stage:02d}_prediction"].to_numpy(
                    np.int8
                )
                eligible &= ~np.isin(prediction, list(partially_fixed))
            if not eligible.all():
                output[f"stage_{new_stage:02d}_eligible"] = eligible
    return (
        output.reset_index(drop=True),
        stage_metadata[first_remaining_stage - 1 :],
        {
            "forced_routed": forced_routed,
            "forced_errors": forced_errors,
            "remaining_budget": root_error_budget - forced_errors,
            "stage_offset": first_remaining_stage - 1,
        },
    )


def _materialize_fixed_branch(
    frame: pd.DataFrame,
    descriptor: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
) -> tuple[pd.DataFrame, Sequence[dict[str, object]]]:
    """Backward-compatible wrapper around generic recursive materialization."""

    if "fixed_thresholds" not in descriptor:
        remaining = _propagate_first_level_branch(frame, descriptor)
        if int(descriptor.get("depth", 1)) == 1:
            return remaining, stage_metadata
        second_label = int(descriptor["second_label"])
        prediction = remaining["stage_01_prediction"].to_numpy(np.int8)
        confidence = remaining["stage_01_confidence"].to_numpy(float)
        accepted = (prediction == second_label) & (
            confidence >= float(descriptor["second_threshold"])
        )
        remaining = remaining.loc[~accepted].copy().reset_index(drop=True)
        return (
            _drop_resolved_first_stage(remaining, len(stage_metadata)),
            stage_metadata[1:],
        )
    remaining, local_metadata, _ = _materialize_recursive_branch(
        frame, descriptor, stage_metadata
    )
    return remaining, local_metadata


def _assemble_branch_thresholds(
    local_thresholds: Sequence[tuple[float, float]],
    descriptor: dict[str, object],
) -> list[tuple[float, float]]:
    if "fixed_thresholds" in descriptor:
        total_stages = int(
            descriptor.get(
                "root_n_stages",
                max(
                    [int(item["stage"]) for item in descriptor["fixed_thresholds"]]
                    + [len(local_thresholds)]
                ),
            )
        )
        thresholds = [[math.inf, math.inf] for _ in range(total_stages)]
        fixed = set()
        for item in descriptor["fixed_thresholds"]:
            stage = int(item["stage"])
            label = int(item["label"])
            thresholds[stage - 1][label] = float(item["threshold"])
            fixed.add((stage, label))
        stage_offset = int(descriptor.get("stage_offset", total_stages - len(local_thresholds)))
        for local_stage, values in enumerate(local_thresholds, start=1):
            original_stage = stage_offset + local_stage
            for label in (0, 1):
                if (original_stage, label) not in fixed:
                    thresholds[original_stage - 1][label] = float(values[label])
        return [tuple(values) for values in thresholds]
    if int(descriptor.get("depth", 1)) == 1:
        thresholds = [list(values) for values in local_thresholds]
    else:
        thresholds = [[math.inf, math.inf]] + [
            list(values) for values in local_thresholds
        ]
        thresholds[0][int(descriptor["second_label"])] = float(
            descriptor["second_threshold"]
        )
    thresholds[0][int(descriptor["split_label"])] = float(descriptor["threshold"])
    return [(float(values[0]), float(values[1])) for values in thresholds]


def _recursive_split_statistics(
    frame: pd.DataFrame,
    domains: Sequence[dict[int, CompressedThresholdDomain]],
    stage_offset: int,
    fixed_thresholds: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Score every remaining dimension; only causally earliest ones may split."""

    fixed = {
        (int(item["stage"]), int(item["label"])) for item in fixed_thresholds
    }
    rows: list[dict[str, object]] = []
    for stage_index, stage_domains in enumerate(domains):
        local_stage = stage_index + 1
        original_stage = stage_offset + local_stage
        predictions = frame[f"stage_{local_stage:02d}_prediction"].to_numpy(
            np.int8
        )
        eligible_column = f"stage_{local_stage:02d}_eligible"
        eligible = (
            frame[eligible_column].to_numpy(bool)
            if eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        for label in (0, 1):
            domain = stage_domains[label]
            mask = eligible & (predictions == label)
            cumulative = np.concatenate(
                ([0], np.cumsum(domain.group_counts, dtype=np.int64))
            )
            states = cumulative[domain.candidate_prefixes]
            routing_span = int(states.max() - states.min()) if len(states) else 0
            distinct_states = int(len(np.unique(states))) if len(states) else 1
            candidate_count = int(domain.retained_threshold_count)
            affected = int(np.count_nonzero(mask))
            early_weight = 1.0 / local_stage
            impact_score = (
                early_weight
                * max(1, affected)
                * max(1, routing_span)
                * max(1, distinct_states)
                / math.sqrt(max(1, candidate_count))
            )
            rows.append(
                {
                    "local_stage": local_stage,
                    "original_stage": original_stage,
                    "label": label,
                    "candidate_count": candidate_count,
                    "affected_records": affected,
                    "routing_span": routing_span,
                    "number_of_distinct_acceptance_states": distinct_states,
                    "impact_score": impact_score,
                    # Later-stage routing cannot be propagated until the
                    # earliest stage is completely fixed.
                    "selectable": local_stage == 1
                    and (original_stage, label) not in fixed,
                }
            )
    return rows


def _recursive_split_children(
    frame: pd.DataFrame,
    model: dict[str, object],
    descriptor: dict[str, object],
    propagated: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Partition one branch over a complete compressed threshold domain."""

    split_rows = _recursive_split_statistics(
        frame,
        model["domains"],
        int(propagated["stage_offset"]),
        descriptor.get("fixed_thresholds", []),
    )
    selectable = [row for row in split_rows if bool(row["selectable"])]
    if not selectable:
        raise AssertionError("unresolved branch has no causally valid split dimension")
    chosen = max(
        selectable,
        key=lambda row: (
            float(row["impact_score"]),
            int(row["affected_records"]),
            int(row["routing_span"]),
            -int(row["candidate_count"]),
            -int(row["label"]),
        ),
    )
    local_stage = int(chosen["local_stage"])
    original_stage = int(chosen["original_stage"])
    label = int(chosen["label"])
    domain = model["domains"][local_stage - 1][label]
    prediction = frame[f"stage_{local_stage:02d}_prediction"].to_numpy(np.int8)
    confidence = frame[f"stage_{local_stage:02d}_confidence"].to_numpy(float)
    oracle = frame["oracle_label"].to_numpy(np.int8)
    eligible_column = f"stage_{local_stage:02d}_eligible"
    eligible = (
        frame[eligible_column].to_numpy(bool)
        if eligible_column in frame
        else np.ones(len(frame), dtype=bool)
    )
    later_has_correct = np.zeros(len(frame), dtype=bool)
    for later_stage in range(local_stage + 1, len(model["domains"]) + 1):
        later_eligible_column = f"stage_{later_stage:02d}_eligible"
        later_eligible = (
            frame[later_eligible_column].to_numpy(bool)
            if later_eligible_column in frame
            else np.ones(len(frame), dtype=bool)
        )
        later_has_correct |= later_eligible & (
            frame[f"stage_{later_stage:02d}_prediction"].to_numpy(np.int8)
            == oracle
        )
    other_label = 1 - label
    current_other_correct = (
        eligible & (prediction == other_label) & (prediction == oracle)
    )
    free_correct_route = later_has_correct | current_other_correct
    children: list[dict[str, object]] = []
    for candidate_index in range(domain.retained_threshold_count):
        threshold = _threshold_for_prefix_candidate(domain, candidate_index)
        accepted = (
            eligible & (prediction == label) & (confidence >= threshold)
        )
        additional_routed = int(np.count_nonzero(accepted))
        additional_errors = int(
            np.count_nonzero(accepted & (prediction != oracle))
        )
        forced_routed = int(propagated["forced_routed"]) + additional_routed
        forced_errors = int(propagated["forced_errors"]) + additional_errors
        remaining_budget = int(descriptor["root_error_budget"]) - forced_errors
        remaining = ~accepted
        remaining_n = int(np.count_nonzero(remaining))
        good = int(np.count_nonzero(remaining & free_correct_route))
        all_wrong = remaining_n - good
        upper_bound = (
            forced_routed + good + min(all_wrong, max(0, remaining_budget))
            if remaining_budget >= 0
            else -1
        )
        # Before materializing hundreds of child models, reuse the parent's
        # (superset) legal domains to obtain the same safe threshold-coupled
        # relaxation on this child's exact surviving rows.  Parent domains
        # can only be looser than domains rebuilt after fixing this threshold,
        # so this remains an upper bound.  Mark rejected records of the fixed
        # class ineligible at the current stage so the relaxation respects the
        # fixed decision exactly.
        if PREPRUNE_RECURSIVE_CHILDREN and remaining_budget >= 0:
            child_frame = frame.loc[remaining].copy().reset_index(drop=True)
            child_prediction = child_frame[
                f"stage_{local_stage:02d}_prediction"
            ].to_numpy(np.int8)
            child_eligible_column = f"stage_{local_stage:02d}_eligible"
            if child_eligible_column in child_frame:
                child_eligible = child_frame[
                    child_eligible_column
                ].to_numpy(bool, copy=True)
            else:
                child_eligible = np.ones(len(child_frame), dtype=bool)
            child_eligible &= child_prediction != label
            child_frame[child_eligible_column] = child_eligible
            coupled_bound = _threshold_coupled_coverage_upper_bound(
                child_frame,
                model["domains"],
                remaining_budget,
                coverage_offset=forced_routed,
            )
            upper_bound = min(
                upper_bound, int(coupled_bound["coverage_upper_bound"])
            )
        child_id = (
            f"{descriptor['branch_id']}_s{original_stage}c{label}q{candidate_index}"
        )
        children.append(
            {
                "branch_id": child_id,
                "parent_id": str(descriptor["branch_id"]),
                "root_id": str(descriptor.get("root_id", descriptor["branch_id"])),
                "recursion_depth": int(descriptor.get("recursion_depth", 0)) + 1,
                "root_error_budget": int(descriptor["root_error_budget"]),
                "root_n_stages": int(descriptor["root_n_stages"]),
                "fixed_thresholds": [
                    *list(descriptor.get("fixed_thresholds", [])),
                    {
                        "stage": original_stage,
                        "label": label,
                        "candidate_index": candidate_index,
                        "threshold": threshold,
                    },
                ],
                "split_candidate_index": candidate_index,
                "split_dimension": f"stage_{original_stage}_class_{label}",
                "forced_routed": forced_routed,
                "forced_errors": forced_errors,
                "remaining_budget": remaining_budget,
                "remaining_N": remaining_n,
                "G": good,
                "H": all_wrong,
                "upper_bound_routed": upper_bound,
                "split_preprune_bound": upper_bound,
                "status": "pending" if remaining_budget >= 0 else "infeasible",
                **(
                    {
                        "force_split_until_depth": int(
                            descriptor["force_split_until_depth"]
                        )
                    }
                    if "force_split_until_depth" in descriptor
                    else {}
                ),
            }
        )
    split = {
        "chosen_dimension": f"stage_{original_stage}_class_{label}",
        "candidate_count": int(chosen["candidate_count"]),
        "affected_records": int(chosen["affected_records"]),
        "routing_span": int(chosen["routing_span"]),
        "number_of_distinct_acceptance_states": int(
            chosen["number_of_distinct_acceptance_states"]
        ),
        "impact_score": float(chosen["impact_score"]),
        "dimension_statistics": split_rows,
        "child_ids": [str(child["branch_id"]) for child in children],
        "candidate_indices": list(range(domain.retained_threshold_count)),
        "candidate_thresholds": [
            float(_threshold_for_prefix_candidate(domain, candidate_index))
            for candidate_index in range(domain.retained_threshold_count)
        ],
    }
    return split, children


def _update_shared_incumbent(coverage: int) -> None:
    if _FORKED_GLOBAL_INCUMBENT is None:
        return
    with _FORKED_GLOBAL_INCUMBENT.get_lock():
        if coverage > _FORKED_GLOBAL_INCUMBENT.value:
            _FORKED_GLOBAL_INCUMBENT.value = coverage


def _fixed_threshold_worker(
    descriptor: dict[str, object], solver_threads: int
) -> dict[str, object]:
    global HIGHS_THREADS, HIGHS_TIME_LIMIT
    if _FORKED_DECOMPOSITION_FRAME is None or _FORKED_DECOMPOSITION_STAGE_METADATA is None:
        raise RuntimeError("fixed-threshold worker state was not inherited")
    HIGHS_THREADS = solver_threads
    recursion_depth = int(descriptor.get("recursion_depth", 0))
    HIGHS_TIME_LIMIT = (
        INITIAL_LEAF_TIME_LIMIT
        if recursion_depth == 0
        else RECURSIVE_LEAF_TIME_LIMIT
    )
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    global_lower_bound = (
        int(_FORKED_GLOBAL_INCUMBENT.value)
        if _FORKED_GLOBAL_INCUMBENT is not None
        else -1
    )
    if int(descriptor["upper_bound_routed"]) <= global_lower_bound:
        return {
            **descriptor,
            "status": "pruned_by_upper_bound",
            "incumbent_routed": -1,
            "best_bound_routed": int(descriptor["upper_bound_routed"]),
            "exact": True,
            "mip_nodes": 0,
            "mip_gap": 0.0,
            "absolute_gap": 0,
            "solver_status": "not_started",
            "thresholds": None,
        }

    propagation_started = time.monotonic()
    if "fixed_thresholds" in descriptor:
        remaining, local_stage_metadata, propagated = _materialize_recursive_branch(
            _FORKED_DECOMPOSITION_FRAME,
            descriptor,
            _FORKED_DECOMPOSITION_STAGE_METADATA,
        )
        descriptor = {
            **descriptor,
            **propagated,
            "remaining_N": len(remaining),
            "stage_offset": int(propagated["stage_offset"]),
        }
    else:
        remaining, local_stage_metadata = _materialize_fixed_branch(
            _FORKED_DECOMPOSITION_FRAME,
            descriptor,
            _FORKED_DECOMPOSITION_STAGE_METADATA,
        )
        propagated = {
            "forced_routed": int(descriptor["forced_routed"]),
            "forced_errors": int(descriptor["forced_errors"]),
            "remaining_budget": int(descriptor["remaining_budget"]),
            "stage_offset": len(_FORKED_DECOMPOSITION_STAGE_METADATA)
            - len(local_stage_metadata),
        }
    propagation_time = time.monotonic() - propagation_started
    if int(propagated["remaining_budget"]) < 0:
        return {
            **descriptor,
            "status": "infeasible",
            "exact": True,
            "incumbent_routed": -1,
            "best_bound_routed": -1,
            "absolute_gap": 0,
            "mip_gap": 0.0,
            "mip_nodes": 0,
            "solver_status": "not_started",
            "thresholds": None,
            "propagation_time": propagation_time,
            "total_time": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        }

    # Once only the terminal stage remains, its two tied prefix choices are
    # solved analytically and HiGHS is bypassed completely.
    if len(local_stage_metadata) <= 1:
        if local_stage_metadata:
            final_domains = None
            if _FORKED_THRESHOLD_ALLOWLIST is not None:
                final_domains, _, _ = _compress_multistage_thresholds(
                    remaining,
                    1,
                    int(propagated["remaining_budget"]),
                    report=False,
                    allowed_thresholds=_FORKED_THRESHOLD_ALLOWLIST,
                    stage_offset=int(propagated["stage_offset"]),
                )
            final_scan = _exact_final_stage_prefix_scan(
                remaining,
                int(propagated["remaining_budget"]),
                stage=1,
                domains=final_domains[0] if final_domains is not None else None,
            )
            local_thresholds = [final_scan["thresholds"]]
        else:
            final_scan = {"coverage": 0, "errors": 0, "number_of_optima": 1}
            local_thresholds = []
        local_descriptor = {
            **descriptor,
            "stage_offset": int(propagated["stage_offset"]),
        }
        thresholds = _assemble_branch_thresholds(local_thresholds, local_descriptor)
        replay = _task_policy_counts(_FORKED_DECOMPOSITION_FRAME, thresholds)
        incumbent_routed = len(_FORKED_DECOMPOSITION_FRAME) - replay["oracle_count"]
        if replay["error_count"] > int(descriptor["root_error_budget"]):
            raise AssertionError("analytical final-stage solution violates budget")
        _update_shared_incumbent(incumbent_routed)
        elapsed = time.monotonic() - started_wall
        cpu_seconds = time.process_time() - started_cpu
        return {
            **descriptor,
            "status": "proven_optimal_final_stage",
            "exact": True,
            "incumbent_routed": incumbent_routed,
            "incumbent_errors": replay["error_count"],
            "best_bound_routed": incumbent_routed,
            "absolute_gap": 0,
            "mip_gap": 0.0,
            "mip_nodes": 0,
            "solver_status": "analytical_final_stage",
            "thresholds": thresholds,
            "final_stage_number_of_optima": int(final_scan["number_of_optima"]),
            "propagation_time": propagation_time,
            "threshold_pruning_time": 0.0,
            "bucket_build_time": 0.0,
            "pattern_aggregation_time": 0.0,
            "model_build_time": 0.0,
            "incumbent_time": 0.0,
            "presolve_time": 0.0,
            "solve_time": 0.0,
            "checkpoint_time": 0.0,
            "total_time": elapsed,
            "cpu_seconds": cpu_seconds,
            "remaining_rows": len(remaining),
            "remaining_patterns": len(remaining),
            "remaining_threshold_candidates": 0,
            "total_variables": 0,
            "constraints": 0,
            "sparse_nonzeros": 0,
            "cpu_utilization_percent": 100.0
            * cpu_seconds
            / max(elapsed, 1e-9),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "root_LP_finished": True,
        }
    model = _build_multistage_milp(
        remaining,
        len(local_stage_metadata),
        int(propagated["remaining_budget"]),
        report=False,
        allowed_thresholds=_FORKED_THRESHOLD_ALLOWLIST,
        stage_offset=int(propagated["stage_offset"]),
    )
    model["partition_label"] = str(descriptor["branch_id"])
    model["coverage_offset"] = int(propagated["forced_routed"])
    model["total_documents"] = len(_FORKED_DECOMPOSITION_FRAME)
    statistics = model["statistics"]
    stronger_bound_started = time.monotonic()
    stronger_bound = _threshold_coupled_coverage_upper_bound(
        remaining,
        model["domains"],
        int(propagated["remaining_budget"]),
        coverage_offset=int(propagated["forced_routed"]),
    )
    stronger_bound_time = time.monotonic() - stronger_bound_started
    old_upper_bound = int(descriptor["upper_bound_routed"])
    new_upper_bound = min(
        old_upper_bound, int(stronger_bound["coverage_upper_bound"])
    )
    relaxation = {
        "exact": False,
        "nodes": 0,
        "elapsed_seconds": 0.0,
        "coverage_upper_bound": new_upper_bound,
    }
    if new_upper_bound > global_lower_bound and UNION_RELAXATION_TIME_LIMIT > 0:
        relaxation = _exact_union_relaxation_upper_bound(
            remaining,
            model["domains"],
            int(propagated["remaining_budget"]),
            coverage_offset=int(propagated["forced_routed"]),
            time_limit=UNION_RELAXATION_TIME_LIMIT,
        )
        if bool(relaxation["exact"]):
            new_upper_bound = min(
                new_upper_bound, int(relaxation["coverage_upper_bound"])
            )
    descriptor = {
        **descriptor,
        "old_upper_bound_routed": old_upper_bound,
        "upper_bound_routed": new_upper_bound,
        "stronger_upper_bound_time": stronger_bound_time,
        "stronger_bound_correctly_routable": int(
            stronger_bound["correctly_routable"]
        ),
        "stronger_bound_must_spend_error": int(
            stronger_bound["must_spend_error_if_routed"]
        ),
        "union_relaxation_exact": bool(relaxation["exact"]),
        "union_relaxation_nodes": int(relaxation["nodes"]),
        "union_relaxation_time": float(relaxation["elapsed_seconds"]),
    }
    global_lower_bound = (
        int(_FORKED_GLOBAL_INCUMBENT.value)
        if _FORKED_GLOBAL_INCUMBENT is not None
        else global_lower_bound
    )
    if new_upper_bound <= global_lower_bound:
        elapsed = time.monotonic() - started_wall
        cpu_seconds = time.process_time() - started_cpu
        return {
            **descriptor,
            **statistics,
            "status": "pruned_by_stronger_upper_bound",
            "incumbent_routed": -1,
            "best_bound_routed": new_upper_bound,
            "absolute_gap": 0,
            "mip_gap": 0.0,
            "mip_nodes": 0,
            "solver_status": "not_started",
            "exact": True,
            "thresholds": None,
            "propagation_time": propagation_time,
            "threshold_pruning_time": float(statistics["threshold_pruning_time"]),
            "bucket_build_time": float(statistics["bucket_build_time"]),
            "pattern_aggregation_time": float(
                statistics["pattern_aggregation_time"]
            ),
            "model_build_time": float(statistics["model_build_time"]),
            "incumbent_time": 0.0,
            "presolve_time": None,
            "solve_time": 0.0,
            "checkpoint_time": 0.0,
            "total_time": elapsed,
            "cpu_seconds": cpu_seconds,
            "remaining_rows": len(remaining),
            "remaining_patterns": int(statistics["P_after_compression"]),
            "remaining_threshold_candidates": int(
                statistics["thresholds_after_budget_pruning"]
            ),
            "threshold_candidates_before": int(
                statistics["thresholds_before_budget_pruning"]
            ),
            "threshold_candidates_after": int(
                statistics["thresholds_after_budget_pruning"]
            ),
            "candidates_removed_by_poison_pruning": int(
                statistics["number_removed_by_budget_pruning"]
            ),
            "additional_candidates_removed_by_future_poison_pruning": 0,
            "cpu_utilization_percent": 100.0
            * cpu_seconds
            / max(elapsed, 1e-9),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "root_LP_finished": False,
        }
    incumbent_started = time.monotonic()
    local_incumbent = _initial_multistage_incumbent(
        remaining,
        model,
        local_stage_metadata,
        report=False,
        use_greedy=False,
    )
    local_thresholds = _assemble_branch_thresholds(
        local_incumbent["thresholds"], descriptor
    )
    replay = _task_policy_counts(_FORKED_DECOMPOSITION_FRAME, local_thresholds)
    incumbent_routed = len(_FORKED_DECOMPOSITION_FRAME) - replay["oracle_count"]
    incumbent_time = time.monotonic() - incumbent_started
    _update_shared_incumbent(incumbent_routed)
    if ADAPTIVE_VERBOSE:
        print(
            f"[subproblem-built] branch={descriptor['branch_id']} "
            f"fixed_threshold_count={len(descriptor.get('fixed_thresholds', []))} "
            f"forced_routed={descriptor['forced_routed']} "
            f"forced_errors={descriptor['forced_errors']} remaining_N={len(remaining)} "
            f"remaining_budget={descriptor['remaining_budget']} "
            f"remaining_patterns={statistics['P_after_compression']} "
            f"remaining_threshold_variables={statistics['binary_variables']} "
            f"variables={statistics['total_variables']} constraints={statistics['constraints']} "
            f"nonzeros={statistics['sparse_nonzeros']}",
            flush=True,
        )

    force_split = recursion_depth < int(
        descriptor.get("force_split_until_depth", -1)
    )
    if not force_split and incumbent_routed == int(descriptor["upper_bound_routed"]):
        solver_result = None
        status = "proven_optimal_by_upper_bound"
        best_bound = incumbent_routed
        exact = True
        nodes = 0
        mip_gap = 0.0
        solver_status = "not_started"
    else:
        solver_result = (
            {
                "x": None,
                "dual_oracle_count": -math.inf,
                "exact": False,
                "infeasible": False,
                "mip_nodes": 0,
                "mip_gap": math.inf,
                "status": "forced_timeout_for_test",
                "elapsed_seconds": 0.0,
            }
            if force_split
            else _solve_compressed_milp(model, local_incumbent)
        )
        if solver_result["x"] is not None:
            solved_local = _solution_to_thresholds(
                model, np.asarray(solver_result["x"], dtype=float)
            )
            solved_thresholds = _assemble_branch_thresholds(
                solved_local, descriptor
            )
            solved_replay = _task_policy_counts(
                _FORKED_DECOMPOSITION_FRAME, solved_thresholds
            )
            solved_routed = (
                len(_FORKED_DECOMPOSITION_FRAME) - solved_replay["oracle_count"]
            )
            if solved_routed >= incumbent_routed:
                incumbent_routed = solved_routed
                local_thresholds = solved_thresholds
                replay = solved_replay
                _update_shared_incumbent(incumbent_routed)
        dual = float(solver_result["dual_oracle_count"])
        solver_bound = (
            int(descriptor["forced_routed"])
            + len(remaining)
            - math.ceil(dual - 1e-7)
            if math.isfinite(dual)
            else int(descriptor["upper_bound_routed"])
        )
        best_bound = min(int(descriptor["upper_bound_routed"]), solver_bound)
        exact = bool(solver_result["exact"] or solver_result["infeasible"])
        status = "proven_optimal" if exact else "solver_time_limit"
        nodes = int(solver_result["mip_nodes"])
        mip_gap = float(solver_result["mip_gap"])
        solver_status = str(solver_result["status"])

    current_global_lower_bound = (
        int(_FORKED_GLOBAL_INCUMBENT.value)
        if _FORKED_GLOBAL_INCUMBENT is not None
        else incumbent_routed
    )
    split: dict[str, object] | None = None
    split_children: list[dict[str, object]] = []
    if not exact and best_bound <= current_global_lower_bound:
        status = "closed_by_solver_bound"
        exact = True
    elif (
        not exact
        and recursion_depth < MAX_RECURSION_DEPTH
        and "fixed_thresholds" in descriptor
    ):
        split, split_children = _recursive_split_children(
            remaining, model, descriptor, propagated
        )
        status = "split"
        if ADAPTIVE_VERBOSE:
            print(
                f"[recursive-split] branch={descriptor['branch_id']} "
                f"chosen_dimension={split['chosen_dimension']} "
                f"candidate_count={split['candidate_count']} "
                f"affected_records={split['affected_records']} "
                f"routing_span={split['routing_span']} "
                f"impact_score={split['impact_score']:.6g}",
                flush=True,
            )

    elapsed = time.monotonic() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    return {
        **descriptor,
        **statistics,
        "propagation_time": propagation_time,
        "threshold_pruning_time": float(statistics["threshold_pruning_time"]),
        "bucket_build_time": float(statistics["bucket_build_time"]),
        "pattern_aggregation_time": float(statistics["pattern_aggregation_time"]),
        "model_build_time": float(statistics["model_build_time"]),
        "incumbent_time": incumbent_time,
        "presolve_time": None,
        "solve_time": (
            0.0
            if solver_result is None
            else float(solver_result["elapsed_seconds"])
        ),
        "checkpoint_time": 0.0,
        "total_time": elapsed,
        "cpu_seconds": cpu_seconds,
        "remaining_rows": len(remaining),
        "remaining_patterns": int(statistics["P_after_compression"]),
        "remaining_threshold_candidates": int(
            statistics["thresholds_after_budget_pruning"]
        ),
        "threshold_candidates_before": int(
            statistics["thresholds_before_budget_pruning"]
        ),
        "threshold_candidates_after": int(
            statistics["thresholds_after_budget_pruning"]
        ),
        "candidates_removed_by_poison_pruning": int(
            statistics["number_removed_by_budget_pruning"]
        ),
        "additional_candidates_removed_by_future_poison_pruning": 0,
        "status": status,
        "incumbent_routed": incumbent_routed,
        "incumbent_errors": replay["error_count"],
        "best_bound_routed": best_bound,
        "absolute_gap": max(0, best_bound - incumbent_routed),
        "mip_gap": mip_gap,
        "mip_nodes": nodes,
        "solver_status": solver_status,
        "exact": exact,
        "thresholds": local_thresholds,
        "elapsed_seconds": elapsed,
        "cpu_utilization_percent": 100.0 * cpu_seconds / max(elapsed, 1e-9),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "root_LP_finished": bool(exact or nodes > 0),
        "split": split,
        "split_children": split_children,
    }


class _SubproblemCheckpoint:
    """Transactional per-branch checkpoint store for exact decomposition."""

    def __init__(self, path: Path | None, identity: str):
        self.connection: sqlite3.Connection | None = None
        self.pending_commits = 0
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS results "
            "(branch_id TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if existing is not None and existing[0] != identity:
            connection.close()
            raise ValueError(
                f"decomposition checkpoint identity mismatch for {path}"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('identity', ?)",
            (identity,),
        )
        connection.commit()
        self.connection = connection

    def load(self) -> dict[str, dict[str, object]]:
        if self.connection is None:
            return {}
        return {
            str(branch_id): pickle.loads(payload)
            for branch_id, payload in self.connection.execute(
                "SELECT branch_id, payload FROM results"
            )
        }

    def put(self, result: dict[str, object]) -> float:
        if self.connection is None:
            return 0.0
        started = time.monotonic()
        self.connection.execute(
            "INSERT OR REPLACE INTO results(branch_id, payload) VALUES(?, ?)",
            (
                str(result["branch_id"]),
                sqlite3.Binary(pickle.dumps(result, protocol=5)),
            ),
        )
        self.pending_commits += 1
        if self.pending_commits >= DECOMPOSITION_CHECKPOINT_INTERVAL:
            self.connection.commit()
            self.pending_commits = 0
        return time.monotonic() - started

    def put_many(self, records: Sequence[dict[str, object]]) -> float:
        """Persist a split parent and all queued children in one transaction."""

        if self.connection is None or not records:
            return 0.0
        started = time.monotonic()
        self.connection.executemany(
            "INSERT OR REPLACE INTO results(branch_id, payload) VALUES(?, ?)",
            [
                (
                    str(record["branch_id"]),
                    sqlite3.Binary(pickle.dumps(record, protocol=5)),
                )
                for record in records
            ],
        )
        self.connection.commit()
        self.pending_commits = 0
        return time.monotonic() - started

    def put_metadata(self, key: str, value: object) -> None:
        if self.connection is None:
            return
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> object | None:
        if self.connection is None:
            return None
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None


def _bounded_fixed_threshold_map(
    descriptors: Sequence[dict[str, object]],
    workers: int,
    solver_threads: int,
    shared_incumbent,
    checkpoint: _SubproblemCheckpoint,
    *,
    result_prefix: str,
) -> list[dict[str, object]]:
    """Run fixed-threshold jobs with bounded in-flight futures and resume."""

    saved = checkpoint.load()
    results: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    for descriptor in descriptors:
        branch_id = str(descriptor["branch_id"])
        if branch_id in saved:
            result = saved[branch_id]
            results.append(result)
            if int(result.get("incumbent_routed", -1)) >= 0:
                _update_shared_incumbent(int(result["incumbent_routed"]))
        elif int(descriptor["upper_bound_routed"]) <= int(shared_incumbent.value):
            result = {
                **descriptor,
                "status": "pruned_by_upper_bound",
                "incumbent_routed": -1,
                "best_bound_routed": int(descriptor["upper_bound_routed"]),
                "exact": True,
                "thresholds": None,
                "mip_nodes": 0,
                "mip_gap": 0.0,
                "absolute_gap": 0,
                "solver_status": "not_started",
            }
            result["checkpoint_time"] = checkpoint.put(result)
            results.append(result)
        else:
            pending.append(descriptor)
    if not pending:
        return results

    maximum_inflight = max(
        workers,
        workers * max(1, DECOMPOSITION_INFLIGHT_MULTIPLIER),
    )
    iterator = iter(pending)
    context = mp.get_context("fork")
    physical_cpus = (
        _linux_physical_cpu_ids()
        if DECOMPOSITION_WORKER_AFFINITY == "physical"
        else []
    )
    affinity_counter = context.Value("i", 0)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker_affinity if physical_cpus else None,
        initargs=(affinity_counter, physical_cpus) if physical_cpus else (),
    ) as executor:
        futures = {}

        def submit_one() -> bool:
            descriptor = next(iterator, None)
            if descriptor is None:
                return False
            if int(descriptor["upper_bound_routed"]) <= int(
                shared_incumbent.value
            ):
                result = {
                    **descriptor,
                    "status": "pruned_by_upper_bound",
                    "incumbent_routed": -1,
                    "best_bound_routed": int(descriptor["upper_bound_routed"]),
                    "exact": True,
                    "thresholds": None,
                    "mip_nodes": 0,
                    "mip_gap": 0.0,
                    "absolute_gap": 0,
                    "solver_status": "not_started",
                }
                result["checkpoint_time"] = checkpoint.put(result)
                results.append(result)
                return True
            future = executor.submit(
                _fixed_threshold_worker, descriptor, solver_threads
            )
            futures[future] = descriptor
            return True

        while len(futures) < maximum_inflight and submit_one():
            pass
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                result["checkpoint_time"] = checkpoint.put(result)
                results.append(result)
                print(
                    f"[{result_prefix}] branch={result['branch_id']} "
                    f"status={result['status']} incumbent={result['incumbent_routed']} "
                    f"bound={result['best_bound_routed']} "
                    f"nodes={result.get('mip_nodes', 0)}",
                    flush=True,
                )
            while len(futures) < maximum_inflight and submit_one():
                pass
    return results


def _branch_priority(
    descriptor: dict[str, object], incumbent: int, sequence: int
) -> tuple[float, float, int]:
    gap = max(0, int(descriptor.get("upper_bound_routed", incumbent)) - incumbent)
    size = float(
        descriptor.get(
            "remaining_patterns", descriptor.get("remaining_N", math.inf)
        )
    )
    if RECURSIVE_PRIORITY_POLICY == "cheapest":
        return (size, gap, sequence)
    if RECURSIVE_PRIORITY_POLICY == "potential":
        return (-float(descriptor.get("upper_bound_routed", incumbent)), size, sequence)
    return (gap, size, sequence)


def _adaptive_recursive_map(
    descriptors: Sequence[dict[str, object]],
    workers: int,
    solver_threads: int,
    shared_incumbent,
    checkpoint: _SubproblemCheckpoint,
) -> list[dict[str, object]]:
    """Close or exhaustively split branches using one bounded worker pool."""

    workers = max(1, min(128, int(workers)))
    if solver_threads != 1:
        raise ValueError("adaptive recursive solving requires one HiGHS thread per worker")
    saved = checkpoint.load()
    for record in saved.values():
        if int(record.get("incumbent_routed", -1)) >= 0:
            _update_shared_incumbent(int(record["incumbent_routed"]))

    root_ids = [str(descriptor["branch_id"]) for descriptor in descriptors]
    checkpoint.put_metadata("root_ids", root_ids)
    results_by_id: dict[str, dict[str, object]] = dict(saved)
    pending_heap: list[tuple[float, float, int, dict[str, object]]] = []
    sequence = 0

    def enqueue(descriptor: dict[str, object]) -> None:
        nonlocal sequence
        branch_id = str(descriptor["branch_id"])
        existing = results_by_id.get(branch_id)
        if existing is not None and existing.get("status") not in {
            "queued",
            "solver_time_limit",
        }:
            return
        sequence += 1
        priority = _branch_priority(
            descriptor, int(shared_incumbent.value), sequence
        )
        heapq.heappush(pending_heap, (*priority, descriptor))

    # Resume queued descendants before regenerating unsaved roots.
    for record in saved.values():
        if record.get("status") == "queued":
            enqueue(record)
        elif (
            record.get("status") == "solver_time_limit"
            and int(record.get("recursion_depth", 0)) < MAX_RECURSION_DEPTH
        ):
            enqueue(record)
    root_iterator = iter(descriptors)
    exhausted_roots = False
    maximum_inflight = max(
        workers, workers * max(1, DECOMPOSITION_INFLIGHT_MULTIPLIER)
    )
    context = mp.get_context("fork")
    physical_cpus = (
        _linux_physical_cpu_ids()
        if DECOMPOSITION_WORKER_AFFINITY == "physical"
        else []
    )
    affinity_counter = context.Value("i", 0)

    def next_descriptor() -> dict[str, object] | None:
        nonlocal exhausted_roots
        if pending_heap:
            return heapq.heappop(pending_heap)[-1]
        while not exhausted_roots:
            descriptor = next(root_iterator, None)
            if descriptor is None:
                exhausted_roots = True
                break
            saved_record = results_by_id.get(str(descriptor["branch_id"]))
            if saved_record is None:
                return descriptor
            if saved_record.get("status") in {"queued", "solver_time_limit"}:
                return saved_record
        return None

    completed_count = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_solver_worker,
        initargs=(affinity_counter, physical_cpus),
    ) as executor:
        futures: dict[object, dict[str, object]] = {}

        def submit_one() -> bool:
            descriptor = next_descriptor()
            if descriptor is None:
                return False
            branch_id = str(descriptor["branch_id"])
            incumbent = int(shared_incumbent.value)
            if int(descriptor["upper_bound_routed"]) <= incumbent:
                result = {
                    **descriptor,
                    "status": "pruned_by_upper_bound",
                    "incumbent_routed": -1,
                    "best_bound_routed": int(descriptor["upper_bound_routed"]),
                    "exact": True,
                    "thresholds": None,
                    "mip_nodes": 0,
                    "mip_gap": 0.0,
                    "absolute_gap": 0,
                    "solver_status": "not_started",
                }
                result["checkpoint_time"] = checkpoint.put(result)
                results_by_id[branch_id] = result
                return True
            future = executor.submit(_fixed_threshold_worker, descriptor, 1)
            futures[future] = descriptor
            return True

        while len(futures) < maximum_inflight and submit_one():
            pass
        while futures or pending_heap or not exhausted_roots:
            if not futures:
                while len(futures) < maximum_inflight and submit_one():
                    pass
                if not futures:
                    break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                descriptor = futures.pop(future)
                result = future.result()
                children = list(result.pop("split_children", []))
                branch_id = str(result["branch_id"])
                if int(result.get("incumbent_routed", -1)) >= 0:
                    _update_shared_incumbent(int(result["incumbent_routed"]))
                if result["status"] == "split":
                    split = dict(result["split"])
                    incumbent = int(shared_incumbent.value)
                    queued_children = []
                    child_records = []
                    closed_candidates: list[dict[str, object]] = []
                    for child in children:
                        if int(child.get("remaining_budget", 0)) < 0:
                            closed_candidates.append(
                                {
                                    "candidate_index": int(
                                        child["split_candidate_index"]
                                    ),
                                    "status": "infeasible",
                                    "maximum_bound_routed": -1,
                                }
                            )
                            continue
                        if int(child["upper_bound_routed"]) <= incumbent:
                            closed_candidates.append(
                                {
                                    "candidate_index": int(
                                        child["split_candidate_index"]
                                    ),
                                    "status": "pruned_by_upper_bound",
                                    "maximum_bound_routed": int(
                                        child["upper_bound_routed"]
                                    ),
                                }
                            )
                            continue
                        child_record = {
                            **child,
                            "status": "queued",
                            "exact": False,
                        }
                        queued_children.append(child_record)
                        child_records.append(child_record)

                    # Consecutive candidates closed by the same proof kind are
                    # one disjoint certified range.  This keeps exact domain
                    # coverage without writing millions of redundant rows.
                    closed_ranges: list[dict[str, object]] = []
                    for item in closed_candidates:
                        if (
                            closed_ranges
                            and item["status"] == closed_ranges[-1]["status"]
                            and int(item["candidate_index"])
                            == int(closed_ranges[-1]["end_index"]) + 1
                        ):
                            closed_ranges[-1]["end_index"] = int(
                                item["candidate_index"]
                            )
                            closed_ranges[-1]["maximum_bound_routed"] = max(
                                int(closed_ranges[-1]["maximum_bound_routed"]),
                                int(item["maximum_bound_routed"]),
                            )
                        else:
                            closed_ranges.append(
                                {
                                    "start_index": int(item["candidate_index"]),
                                    "end_index": int(item["candidate_index"]),
                                    "status": item["status"],
                                    "maximum_bound_routed": int(
                                        item["maximum_bound_routed"]
                                    ),
                                }
                            )
                    result["child_ids"] = [
                        str(child["branch_id"]) for child in child_records
                    ]
                    split["child_ids"] = list(result["child_ids"])
                    split["closed_candidate_ranges"] = closed_ranges
                    result["split"] = split
                    result["checkpoint_time"] = checkpoint.put_many(
                        [result, *child_records]
                    )
                    results_by_id[branch_id] = result
                    for child in child_records:
                        results_by_id[str(child["branch_id"])] = child
                    for child in queued_children:
                        enqueue(child)
                else:
                    result["checkpoint_time"] = checkpoint.put(result)
                    results_by_id[branch_id] = result
                completed_count += 1
                if ADAPTIVE_VERBOSE or (
                    ADAPTIVE_PROGRESS_INTERVAL > 0
                    and completed_count % ADAPTIVE_PROGRESS_INTERVAL == 0
                ):
                    print(
                        f"[adaptive-result] completed={completed_count} "
                        f"branch={branch_id} "
                        f"depth={result.get('recursion_depth', 0)} "
                        f"status={result['status']} "
                        f"incumbent={result.get('incumbent_routed', -1)} "
                        f"bound={result.get('best_bound_routed', result.get('upper_bound_routed'))} "
                        f"children={len(children)} queued={len(pending_heap)} "
                        f"running={len(futures)}",
                        flush=True,
                    )
            while len(futures) < maximum_inflight and submit_one():
                pass
    return list(results_by_id.values())


def verify_recursive_certificate(
    records: Sequence[dict[str, object]],
    root_ids: Sequence[str],
    final_incumbent_routed: int,
) -> dict[str, object]:
    """Verify recursive domain closure from persisted records without HiGHS."""

    by_id = {str(record["branch_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate branch IDs in recursive certificate")
    closed_statuses = {
        "proven_optimal",
        "proven_optimal_by_upper_bound",
        "proven_optimal_final_stage",
        "infeasible",
        "pruned_by_upper_bound",
        "pruned_by_stronger_upper_bound",
        "closed_by_solver_bound",
    }
    unresolved: set[str] = set()
    visiting: set[str] = set()
    verified: dict[str, bool] = {}

    def visit(branch_id: str) -> bool:
        if branch_id in verified:
            return verified[branch_id]
        if branch_id in visiting:
            raise ValueError(f"cycle in recursive certificate at {branch_id}")
        visiting.add(branch_id)
        record = by_id.get(branch_id)
        if record is None:
            unresolved.add(branch_id)
            result = False
        elif record.get("status") == "split":
            split = record.get("split") or {}
            child_ids = [str(value) for value in record.get("child_ids", [])]
            candidate_count = int(split.get("candidate_count", -1))
            if len(set(child_ids)) != len(child_ids):
                raise ValueError(f"overlapping duplicate children at {branch_id}")
            indices = []
            for child_id in child_ids:
                child = by_id.get(child_id)
                if child is None or str(child.get("parent_id")) != branch_id:
                    raise ValueError(f"invalid parent/child link {branch_id}->{child_id}")
                indices.append(int(child.get("split_candidate_index", -1)))
            for closed_range in split.get("closed_candidate_ranges", []):
                start = int(closed_range.get("start_index", -1))
                end = int(closed_range.get("end_index", -1))
                if start < 0 or end < start:
                    raise ValueError(f"invalid closed range at {branch_id}")
                status = str(closed_range.get("status"))
                if status not in {"infeasible", "pruned_by_upper_bound"}:
                    raise ValueError(f"invalid range proof at {branch_id}")
                if status == "pruned_by_upper_bound" and int(
                    closed_range.get("maximum_bound_routed", math.inf)
                ) > int(final_incumbent_routed):
                    raise ValueError(f"invalid range bound closure at {branch_id}")
                indices.extend(range(start, end + 1))
            if candidate_count < 1 or sorted(indices) != list(range(candidate_count)):
                raise ValueError(f"non-disjoint threshold partition at {branch_id}")
            child_results = [visit(child_id) for child_id in child_ids]
            result = all(child_results)
        elif record.get("status") in closed_statuses:
            if not bool(record.get("exact", False)):
                raise ValueError(f"closed branch lacks exact flag: {branch_id}")
            if record.get("status") in {
                "pruned_by_upper_bound",
                "pruned_by_stronger_upper_bound",
                "closed_by_solver_bound",
            } and int(record.get("best_bound_routed", math.inf)) > int(
                final_incumbent_routed
            ):
                raise ValueError(f"invalid bound closure at {branch_id}")
            result = True
        else:
            unresolved.add(branch_id)
            result = False
        visiting.remove(branch_id)
        verified[branch_id] = result
        return result

    roots_closed = [visit(str(root_id)) for root_id in root_ids]
    maximum_remaining = max(
        [int(final_incumbent_routed)]
        + [
            int(by_id[branch_id].get("best_bound_routed", by_id[branch_id].get("upper_bound_routed", len(by_id))))
            for branch_id in unresolved
            if branch_id in by_id
        ]
    )
    return {
        "root_count": len(root_ids),
        "closed_root_count": int(sum(roots_closed)),
        "unresolved_count": len(unresolved),
        "unresolved_branch_ids": sorted(unresolved),
        "maximum_remaining_upper_bound": maximum_remaining,
        "exact_optimality_certificate": all(roots_closed)
        and not unresolved
        and maximum_remaining <= int(final_incumbent_routed),
    }


def _solve_adaptive_fixed_threshold_decomposition(
    frame: pd.DataFrame,
    model: dict[str, object],
    incumbent: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
    requested_workers: int,
) -> dict[str, object]:
    """Run the arbitrary-depth exact partition solver."""

    initial_coverage = int(incumbent["coverage"])
    split_label, candidates, split_summary = _stage1_split_candidates(
        frame, model["domains"], int(model["error_budget"]), report=True
    )
    descriptors = _first_level_branch_descriptors(
        frame, split_label, candidates, int(model["error_budget"])
    )
    for descriptor in descriptors:
        descriptor.update(
            {
                "root_id": str(descriptor["branch_id"]),
                "root_error_budget": int(model["error_budget"]),
                "root_n_stages": len(stage_metadata),
            }
        )
    descriptors.sort(
        key=lambda branch: (
            -int(branch["upper_bound_routed"]),
            int(branch["remaining_N"]),
            int(branch["forced_errors"]),
        )
    )
    selected_limit = (
        len(descriptors)
        if DECOMPOSITION_PILOT_BRANCHES is None
        else min(len(descriptors), DECOMPOSITION_PILOT_BRANCHES)
    )
    selected = descriptors[:selected_limit]
    omitted = descriptors[selected_limit:]
    static_records: list[dict[str, object]] = []
    runnable: list[dict[str, object]] = []
    for descriptor in selected:
        if descriptor["status"] == "infeasible":
            static_records.append(
                {
                    **descriptor,
                    "status": "infeasible",
                    "exact": True,
                    "best_bound_routed": -1,
                    "incumbent_routed": -1,
                    "thresholds": None,
                }
            )
        elif int(descriptor["upper_bound_routed"]) <= initial_coverage:
            static_records.append(
                {
                    **descriptor,
                    "status": "pruned_by_upper_bound",
                    "exact": True,
                    "best_bound_routed": int(descriptor["upper_bound_routed"]),
                    "incumbent_routed": -1,
                    "thresholds": None,
                }
            )
        else:
            runnable.append(descriptor)
    static_records.extend(
        {
            **descriptor,
            "status": "not_run_pilot",
            "exact": False,
            "best_bound_routed": int(descriptor["upper_bound_routed"]),
            "incumbent_routed": -1,
            "thresholds": None,
        }
        for descriptor in omitted
    )

    available_memory = _available_memory_bytes()
    memory_workers = max(
        1,
        int(available_memory * DECOMPOSITION_MEMORY_FRACTION)
        // int(0.75 * 1024**3),
    )
    workers = max(
        1,
        min(
            128,
            requested_workers,
            len(runnable) or 1,
            len(_linux_physical_cpu_ids()) or (os.cpu_count() or 1),
            memory_workers,
        ),
    )
    print(
        f"[adaptive-decomposition] roots={len(descriptors)} "
        f"scheduled={len(runnable)} pilot={DECOMPOSITION_PILOT_BRANCHES is not None} "
        f"workers={workers} solver_threads_per_worker=1 "
        f"initial_limit={INITIAL_LEAF_TIME_LIMIT} "
        f"recursive_limit={RECURSIVE_LEAF_TIME_LIMIT} "
        f"max_depth={MAX_RECURSION_DEPTH} priority={RECURSIVE_PRIORITY_POLICY}",
        flush=True,
    )

    identity_columns = ["oracle_label"]
    for stage in range(1, len(stage_metadata) + 1):
        identity_columns.extend(
            [f"stage_{stage:02d}_prediction", f"stage_{stage:02d}_confidence"]
        )
    frame_digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame[identity_columns], index=False
        ).to_numpy(np.uint64).tobytes()
    ).hexdigest()
    checkpoint_identity = json.dumps(
        {
            "version": 3,
            "N": len(frame),
            "L": len(stage_metadata),
            "error_budget": int(model["error_budget"]),
            "frame_digest": frame_digest,
            "initial_leaf_time_limit": INITIAL_LEAF_TIME_LIMIT,
            "recursive_leaf_time_limit": RECURSIVE_LEAF_TIME_LIMIT,
            "max_recursion_depth": MAX_RECURSION_DEPTH,
            "restricted_grid": (
                hashlib.sha256(
                    repr(_domain_threshold_allowlist(model["domains"])).encode()
                ).hexdigest()
                if model.get("restricted_grid_spec") is not None
                else None
            ),
        },
        sort_keys=True,
    )
    checkpoint = _SubproblemCheckpoint(
        DECOMPOSITION_CHECKPOINT_PATH, checkpoint_identity
    )
    if static_records:
        checkpoint.put_many(static_records)
    context = mp.get_context("fork")
    shared_incumbent = context.Value("q", initial_coverage)
    global _FORKED_DECOMPOSITION_FRAME, _FORKED_DECOMPOSITION_STAGE_METADATA
    global _FORKED_GLOBAL_INCUMBENT, _FORKED_THRESHOLD_ALLOWLIST
    _FORKED_DECOMPOSITION_FRAME = frame
    _FORKED_DECOMPOSITION_STAGE_METADATA = stage_metadata
    _FORKED_GLOBAL_INCUMBENT = shared_incumbent
    _FORKED_THRESHOLD_ALLOWLIST = (
        _domain_threshold_allowlist(model["domains"])
        if model.get("restricted_grid_spec") is not None
        else None
    )
    try:
        adaptive_records = _adaptive_recursive_map(
            runnable, workers, 1, shared_incumbent, checkpoint
        )
    finally:
        _FORKED_DECOMPOSITION_FRAME = None
        _FORKED_DECOMPOSITION_STAGE_METADATA = None
        _FORKED_GLOBAL_INCUMBENT = None
        _FORKED_THRESHOLD_ALLOWLIST = None
        checkpoint.close()
    by_id = {
        str(record["branch_id"]): record
        for record in [*static_records, *adaptive_records]
    }
    records = list(by_id.values())
    feasible = [
        record
        for record in records
        if record.get("thresholds") is not None
        and int(record.get("incumbent_routed", -1)) >= initial_coverage
    ]
    final_coverage = max(
        [initial_coverage]
        + [int(record["incumbent_routed"]) for record in feasible]
    )
    best_thresholds = incumbent["thresholds"]
    if feasible:
        best = max(feasible, key=lambda record: int(record["incumbent_routed"]))
        if int(best["incumbent_routed"]) >= final_coverage:
            best_thresholds = best["thresholds"]
    # A final incumbent improvement can close previously stored bounds.
    for record in records:
        if record.get("status") == "solver_time_limit" and int(
            record.get("best_bound_routed", math.inf)
        ) <= final_coverage:
            record["status"] = "closed_by_solver_bound"
            record["exact"] = True

    root_ids = [str(descriptor["branch_id"]) for descriptor in descriptors]
    verified = verify_recursive_certificate(records, root_ids, final_coverage)
    counts = {
        status: sum(record.get("status") == status for record in records)
        for status in {
            "proven_optimal",
            "proven_optimal_by_upper_bound",
            "proven_optimal_final_stage",
            "infeasible",
            "pruned_by_upper_bound",
            "pruned_by_stronger_upper_bound",
            "closed_by_solver_bound",
            "solver_time_limit",
            "not_run_pilot",
            "split",
        }
    }
    certificate = {
        **verified,
        "alpha_error_budget": int(model["error_budget"]),
        "N": len(frame),
        "initial_incumbent": initial_coverage,
        "final_incumbent": final_coverage,
        "decomposition_dimension_1": f"stage_1_class_{split_label}",
        "number_first_level_subproblems": len(descriptors),
        "number_solved_optimal": counts["proven_optimal"]
        + counts["proven_optimal_by_upper_bound"]
        + counts["proven_optimal_final_stage"],
        "number_infeasible": counts["infeasible"],
        "number_pruned_analytically": counts["pruned_by_upper_bound"]
        + counts["pruned_by_stronger_upper_bound"],
        "number_closed_by_solver_bound": counts["closed_by_solver_bound"],
        "number_recursively_split": counts["split"],
        "number_unresolved": int(verified["unresolved_count"]),
    }
    certificate["final_optimal_routed"] = (
        final_coverage if certificate["exact_optimality_certificate"] else None
    )
    certificate["final_kappa"] = (
        final_coverage / len(frame)
        if certificate["exact_optimality_certificate"]
        else None
    )
    certificate["kappa_lower_bound"] = final_coverage / len(frame)
    certificate["kappa_upper_bound"] = (
        int(certificate["maximum_remaining_upper_bound"]) / len(frame)
    )
    return {
        "thresholds": best_thresholds,
        "exact": bool(certificate["exact_optimality_certificate"]),
        "status": "adaptive recursive fixed-threshold decomposition",
        "mip_gap": (
            0.0
            if certificate["exact_optimality_certificate"]
            else (
                int(certificate["maximum_remaining_upper_bound"])
                - final_coverage
            )
            / max(1, final_coverage)
        ),
        "mip_nodes": sum(int(record.get("mip_nodes", 0)) for record in records),
        "coverage_upper_bound": int(certificate["maximum_remaining_upper_bound"]),
        "certificate": json.dumps(certificate, sort_keys=True),
        "decomposition_certificate": certificate,
        "subproblem_results": records,
        "split_candidate_statistics": candidates,
        "split_dimension_statistics": split_summary,
        "backend": "adaptive_recursive_decomposition",
    }


def _solve_fixed_threshold_decomposition(
    frame: pd.DataFrame,
    model: dict[str, object],
    incumbent: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
    requested_workers: int,
) -> dict[str, object]:
    """Exact disjoint decomposition with every branch rebuilt from raw rows."""

    initial_replay = _task_policy_counts(frame, incumbent["thresholds"])
    initial_coverage = len(frame) - initial_replay["oracle_count"]
    if (
        initial_coverage != int(incumbent["coverage"])
        or initial_replay["error_count"] > int(model["error_budget"])
    ):
        raise AssertionError("global incumbent failed full-workload replay")

    return _solve_adaptive_fixed_threshold_decomposition(
        frame, model, incumbent, stage_metadata, requested_workers
    )

    split_label, candidates, split_summary = _stage1_split_candidates(
        frame, model["domains"], int(model["error_budget"]), report=True
    )
    descriptors = _first_level_branch_descriptors(
        frame, split_label, candidates, int(model["error_budget"])
    )
    descriptors.sort(
        key=lambda branch: (
            -int(branch["upper_bound_routed"]),
            int(branch["remaining_N"]),
            int(branch["forced_errors"]),
        )
    )

    results: list[dict[str, object]] = []
    pending = []
    for descriptor in descriptors:
        if descriptor["status"] == "infeasible":
            results.append(
                {
                    **descriptor,
                    "best_bound_routed": -1,
                    "incumbent_routed": -1,
                    "status": "infeasible",
                    "exact": True,
                    "thresholds": None,
                }
            )
        elif int(descriptor["upper_bound_routed"]) <= initial_coverage:
            results.append(
                {
                    **descriptor,
                    "best_bound_routed": int(descriptor["upper_bound_routed"]),
                    "incumbent_routed": -1,
                    "status": "pruned_by_upper_bound",
                    "exact": True,
                    "thresholds": None,
                }
            )
        else:
            pending.append(descriptor)

    selected = (
        pending[:DECOMPOSITION_PILOT_BRANCHES]
        if DECOMPOSITION_PILOT_BRANCHES is not None
        else pending
    )
    unselected = pending[len(selected) :]
    results.extend(
        {
            **descriptor,
            "best_bound_routed": int(descriptor["upper_bound_routed"]),
            "incumbent_routed": -1,
            "status": "not_run_pilot",
            "exact": False,
            "thresholds": None,
        }
        for descriptor in unselected
    )

    available_cores = os.cpu_count() or 1
    available_memory = _available_memory_bytes()
    memory_budget = int(available_memory * DECOMPOSITION_MEMORY_FRACTION)
    estimated_worker_memory = 3 * 1024**3
    memory_workers = max(1, memory_budget // estimated_worker_memory)
    workers = max(
        1,
        min(
            requested_workers,
            max(1, available_cores // max(1, HIGHS_THREADS)),
            memory_workers,
            max(1, len(selected)),
        ),
    )
    print(
        f"[fixed-threshold-decomposition] stage=1 class={split_label} "
        f"subproblems={len(descriptors)} analytically_pruned={len(results) - len(unselected)} "
        f"scheduled={len(selected)} pilot={DECOMPOSITION_PILOT_BRANCHES is not None} "
        f"workers={workers} solver_threads_per_worker={HIGHS_THREADS} "
        f"memory_fraction={DECOMPOSITION_MEMORY_FRACTION:.2f}",
        flush=True,
    )
    print(
        f"[worker-topology] logical_cpus={os.cpu_count() or 1} "
        f"physical_cores_detected={len(_linux_physical_cpu_ids())} "
        f"affinity={DECOMPOSITION_WORKER_AFFINITY} "
        f"numa={json.dumps(_linux_numa_topology(), sort_keys=True)}",
        flush=True,
    )

    context = mp.get_context("fork")
    shared_incumbent = context.Value("q", initial_coverage)
    identity_columns = ["oracle_label"]
    for stage in range(1, len(stage_metadata) + 1):
        identity_columns.extend(
            [f"stage_{stage:02d}_prediction", f"stage_{stage:02d}_confidence"]
        )
    frame_digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame[identity_columns], index=False
        ).to_numpy(np.uint64).tobytes()
    ).hexdigest()
    checkpoint_identity = json.dumps(
        {
            "version": 1,
            "N": len(frame),
            "L": len(stage_metadata),
            "error_budget": int(model["error_budget"]),
            "frame_digest": frame_digest,
        },
        sort_keys=True,
    )
    checkpoint = _SubproblemCheckpoint(
        DECOMPOSITION_CHECKPOINT_PATH, checkpoint_identity
    )
    global _FORKED_DECOMPOSITION_FRAME, _FORKED_DECOMPOSITION_STAGE_METADATA
    global _FORKED_GLOBAL_INCUMBENT
    _FORKED_DECOMPOSITION_FRAME = frame
    _FORKED_DECOMPOSITION_STAGE_METADATA = stage_metadata
    _FORKED_GLOBAL_INCUMBENT = shared_incumbent
    try:
        results.extend(
            _bounded_fixed_threshold_map(
                selected,
                workers,
                HIGHS_THREADS,
                shared_incumbent,
                checkpoint,
                result_prefix="subproblem-result",
            )
        )
    finally:
        _FORKED_DECOMPOSITION_FRAME = None
        _FORKED_DECOMPOSITION_STAGE_METADATA = None
        _FORKED_GLOBAL_INCUMBENT = None

    hard_parents = sorted(
        [result for result in results if result.get("status") == "solver_time_limit"],
        key=lambda result: (
            -int(result["best_bound_routed"]),
            int(result["remaining_N"]),
        ),
    )[:RECURSIVE_PILOT_PARENTS]
    recursive_selected: list[dict[str, object]] = []
    for parent in hard_parents:
        children = _second_level_branch_descriptors(
            frame, parent, len(stage_metadata)
        )
        children.sort(
            key=lambda branch: (
                -int(branch["upper_bound_routed"]),
                int(branch["remaining_N"]),
                int(branch["forced_errors"]),
            )
        )
        parent["status"] = "recursively_split"
        parent["exact"] = True
        current_lower_bound = int(shared_incumbent.value)
        viable_children = []
        for child in children:
            if child["status"] == "infeasible":
                results.append(
                    {
                        **child,
                        "best_bound_routed": -1,
                        "incumbent_routed": -1,
                        "status": "infeasible",
                        "exact": True,
                        "thresholds": None,
                    }
                )
            elif int(child["upper_bound_routed"]) <= current_lower_bound:
                results.append(
                    {
                        **child,
                        "best_bound_routed": int(child["upper_bound_routed"]),
                        "incumbent_routed": -1,
                        "status": "pruned_by_upper_bound",
                        "exact": True,
                        "thresholds": None,
                    }
                )
            else:
                viable_children.append(child)
        selected_children = (
            viable_children[:RECURSIVE_PILOT_BRANCHES]
            if RECURSIVE_PILOT_BRANCHES is not None
            else viable_children
        )
        recursive_selected.extend(selected_children)
        results.extend(
            {
                **child,
                "best_bound_routed": int(child["upper_bound_routed"]),
                "incumbent_routed": -1,
                "status": "not_run_recursive_pilot",
                "exact": False,
                "thresholds": None,
            }
            for child in viable_children[len(selected_children) :]
        )
        print(
            f"[recursive-split] parent={parent['branch_id']} "
            f"dimension=stage1/class{1 - split_label} children={len(children)} "
            f"scheduled={len(selected_children)}",
            flush=True,
        )

    if recursive_selected:
        recursive_workers = min(workers, len(recursive_selected))
        _FORKED_DECOMPOSITION_FRAME = frame
        _FORKED_DECOMPOSITION_STAGE_METADATA = stage_metadata
        _FORKED_GLOBAL_INCUMBENT = shared_incumbent
        try:
            results.extend(
                _bounded_fixed_threshold_map(
                    recursive_selected,
                    recursive_workers,
                    HIGHS_THREADS,
                    shared_incumbent,
                    checkpoint,
                    result_prefix="recursive-result",
                )
            )
        finally:
            _FORKED_DECOMPOSITION_FRAME = None
            _FORKED_DECOMPOSITION_STAGE_METADATA = None
            _FORKED_GLOBAL_INCUMBENT = None

    checkpoint.close()

    feasible = [
        result
        for result in results
        if result.get("thresholds") is not None
        and int(result.get("incumbent_routed", -1)) >= initial_coverage
    ]
    best_thresholds = incumbent["thresholds"]
    final_coverage = initial_coverage
    if feasible:
        best = max(feasible, key=lambda result: int(result["incumbent_routed"]))
        if int(best["incumbent_routed"]) > final_coverage:
            final_coverage = int(best["incumbent_routed"])
            best_thresholds = best["thresholds"]

    for result in results:
        if (
            result["status"]
            in {"solver_time_limit", "not_run_pilot", "not_run_recursive_pilot"}
            and int(result["best_bound_routed"]) <= final_coverage
        ):
            result["status"] = "closed_by_solver_bound"
            result["exact"] = True
    unresolved = [result for result in results if not bool(result.get("exact", False))]
    closed = [result for result in results if bool(result.get("exact", False))]
    maximum_remaining_bound = max(
        [final_coverage]
        + [int(result["best_bound_routed"]) for result in unresolved]
    )
    maximum_closed_bound = max(
        [final_coverage]
        + [
            int(result["best_bound_routed"])
            for result in closed
            if "best_bound_routed" in result
            and result.get("status") != "recursively_split"
        ]
    )
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in {
            "proven_optimal",
            "proven_optimal_by_upper_bound",
            "infeasible",
            "pruned_by_upper_bound",
            "pruned_by_stronger_upper_bound",
            "closed_by_solver_bound",
            "solver_time_limit",
            "not_run_pilot",
            "not_run_recursive_pilot",
            "recursively_split",
        }
    }
    certificate = {
        "alpha_error_budget": int(model["error_budget"]),
        "N": len(frame),
        "initial_incumbent": initial_coverage,
        "final_incumbent": final_coverage,
        "decomposition_dimension_1": f"stage_1_class_{split_label}",
        "number_first_level_subproblems": len(descriptors),
        "number_solved_optimal": counts.get("proven_optimal", 0)
        + counts.get("proven_optimal_by_upper_bound", 0),
        "number_infeasible": counts.get("infeasible", 0),
        "number_pruned_analytically": counts.get("pruned_by_upper_bound", 0)
        + counts.get("pruned_by_stronger_upper_bound", 0),
        "number_pruned_by_stronger_upper_bound": counts.get(
            "pruned_by_stronger_upper_bound", 0
        ),
        "number_closed_by_solver_bound": counts.get("closed_by_solver_bound", 0),
        "number_recursively_split": counts.get("recursively_split", 0),
        "number_unresolved": len(unresolved),
        "maximum_remaining_upper_bound": maximum_remaining_bound,
        "maximum_closed_upper_bound": maximum_closed_bound,
        "exact_optimality_certificate": len(unresolved) == 0
        and maximum_remaining_bound <= final_coverage
        and maximum_closed_bound <= final_coverage,
    }
    certificate["final_optimal_routed"] = (
        final_coverage if certificate["exact_optimality_certificate"] else None
    )
    certificate["final_kappa"] = (
        final_coverage / len(frame)
        if certificate["exact_optimality_certificate"]
        else None
    )
    certificate["kappa_lower_bound"] = final_coverage / len(frame)
    certificate["kappa_upper_bound"] = maximum_remaining_bound / len(frame)
    return {
        "thresholds": best_thresholds,
        "exact": bool(certificate["exact_optimality_certificate"]),
        "status": "fixed-threshold decomposition",
        "mip_gap": (
            0.0
            if certificate["exact_optimality_certificate"]
            else (maximum_remaining_bound - final_coverage) / max(1, final_coverage)
        ),
        "mip_nodes": sum(int(result.get("mip_nodes", 0)) for result in results),
        "coverage_upper_bound": maximum_remaining_bound,
        "certificate": json.dumps(certificate, sort_keys=True),
        "decomposition_certificate": certificate,
        "subproblem_results": results,
        "split_candidate_statistics": candidates,
        "split_dimension_statistics": split_summary,
        "backend": "fixed_threshold_decomposition",
    }


def _optimize_exact_task_cascade(
    frame: pd.DataFrame,
    alphas: Sequence[Decimal],
    metadata: dict[str, object],
    stage_metadata: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    n_stages = len(stage_metadata)
    if n_stages == 0:
        records = []
        for alpha in alphas:
            records.append(
                _task_record(
                    metadata,
                    frame,
                    alpha,
                    [],
                    stage_metadata,
                    0,
                    "trivial_oracle_only",
                    "exact_trivial",
                    0.0,
                    0,
                    1,
                )
            )
        return records
    if n_stages == 1:
        return optimize_single_task_stage(frame, alphas, metadata, stage_metadata)

    records = []
    for alpha in alphas:
        total_started = time.monotonic()
        error_budget = len(frame) - required_correct_count(alpha, len(frame))
        model = _build_multistage_milp(frame, n_stages, error_budget)
        preprocessing_time = time.monotonic() - total_started
        solve_started = time.monotonic()
        n_distinct = int(model["statistics"]["D_original"])
        incumbent = _initial_multistage_incumbent(
            frame, model, stage_metadata
        )
        upper_bound = _all_wrong_coverage_upper_bound(model)
        print(
            f"[global-upper-bound] all_wrong_H={upper_bound['all_wrong_H']} "
            f"coverage_upper_bound={upper_bound['coverage_upper_bound']} "
            f"kappa_upper_bound={upper_bound['kappa_upper_bound']:.12f}",
            flush=True,
        )
        if int(incumbent["coverage"]) == int(upper_bound["coverage_upper_bound"]):
            thresholds = incumbent["thresholds"]
            diagnostics = {
                "exact": True,
                "status": "analytical all-wrong upper bound equals feasible incumbent",
                "mip_gap": 0.0,
                "mip_nodes": 0,
                "backend": "analytical_upper_bound",
                "coverage_upper_bound": int(upper_bound["coverage_upper_bound"]),
                "certificate": "feasible incumbent equals global all-wrong coverage upper bound",
            }
            print(
                "[exact-certificate] incumbent equals analytical upper bound; MILP skipped",
                flush=True,
            )
        else:
            diagnostics = _solve_fixed_threshold_decomposition(
                frame,
                model,
                incumbent,
                stage_metadata,
                DECOMPOSITION_WORKERS,
            )
            diagnostics["decomposition_certificate"]["alpha"] = format(alpha, "f")
            diagnostics["certificate"] = json.dumps(
                diagnostics["decomposition_certificate"], sort_keys=True
            )
            thresholds = diagnostics["thresholds"]
            print(
                f"[solver-result] backend={diagnostics['backend']} "
                f"exact={diagnostics['exact']} nodes={diagnostics['mip_nodes']} "
                f"relative_mip_gap={diagnostics['mip_gap']} "
                f"coverage_upper_bound={diagnostics['coverage_upper_bound']}",
                flush=True,
            )
        replay = _task_policy_counts(frame, thresholds)
        coverage = len(frame) - replay["oracle_count"]
        if bool(diagnostics["exact"]):
            coverage_bound = coverage
        else:
            coverage_bound = max(
                coverage, int(diagnostics["coverage_upper_bound"])
            )
        record = _task_record(
            metadata,
            frame,
            alpha,
            thresholds,
            stage_metadata,
            n_distinct,
            f"exact_compressed_survival_{diagnostics['backend']}",
            str(diagnostics["status"]),
            float(diagnostics["mip_gap"]),
            int(diagnostics["mip_nodes"]),
            optimality_proven=bool(diagnostics["exact"]),
            upper_bound_kappa=coverage_bound / len(frame),
            exact_optimality_certificate=diagnostics["certificate"],
            decomposition_certificate=diagnostics.get("decomposition_certificate"),
            subproblem_results=diagnostics.get("subproblem_results"),
            model_statistics=model["statistics"],
            solve_time_seconds=time.monotonic() - solve_started,
            preprocessing_time_seconds=preprocessing_time,
            total_time_seconds=time.monotonic() - total_started,
        )
        records.append(record)
    return records
