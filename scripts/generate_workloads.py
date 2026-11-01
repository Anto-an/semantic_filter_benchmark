"""Generate Low, High, Mixed, and strict Uniform ID selections from any confidence CSV."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from workload_selection import (
    InputValidationError,
    InfeasibleError,
    exact_rank_indices,
    find_fine_uniform_selections,
    normalize_ids,
    select_mixed_indices,
    shuffle_indices,
)

METHODS = ("Low", "High", "Mixed", "Uniform")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict): return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(item) for item in value]
    if isinstance(value, np.ndarray): return json_value(value.tolist())
    if isinstance(value, np.generic): return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def read_confidence(path, id_column, confidence_column, logprob_column=None):
    columns = [id_column, confidence_column]
    if logprob_column is not None: columns.append(logprob_column)
    if len(set(columns)) != len(columns):
        raise ValueError("ID, confidence, and optional log-probability columns must be distinct")
    frame = pd.read_csv(path, usecols=columns, dtype={id_column: str}, keep_default_na=False)
    ids = normalize_ids(frame[id_column], "Source IDs")
    scores = pd.to_numeric(frame[confidence_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Confidence must contain only finite predicted-label probabilities in [0, 1]")
    crosscheck = None
    if logprob_column is not None:
        logprobs = pd.to_numeric(frame[logprob_column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(logprobs).all() or np.any(logprobs > 1e-12):
            raise ValueError("Chosen-label log-probabilities must be finite and nonpositive")
        error = float(np.max(np.abs(scores - np.exp(logprobs))))
        if error > 1e-9:
            raise ValueError(f"Confidence differs from exp(log-probability); maximum absolute error is {error}")
        crosscheck = {"maximum_absolute_error": error, "tolerance": 1e-9, "passed": True}
    return ids.to_numpy(dtype=object), scores, crosscheck


def validate_configuration(args):
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Methods must be distinct")
    if args.sample_size < 1 or args.uniform_sample_size < 1:
        raise ValueError("Sample sizes must be positive")
    if "Mixed" in args.methods and args.sample_size % 2:
        raise ValueError("Mixed requires an even --sample-size for equal tails")
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if args.uniform_target_bins < 2 or args.uniform_interval_grid_size < 3 or args.uniform_candidate_limit < 1:
        raise ValueError("Uniform requires at least 2 bins, a grid size of 3, and a positive candidate limit")
    for value in (args.uniform_maximum_tv, args.uniform_maximum_ks,
                  args.uniform_maximum_wasserstein, args.uniform_maximum_gap_bin_widths):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Uniform acceptance limits must be finite and nonnegative")
    destinations = [args.output_dir / method / "ids.csv" for method in METHODS]
    destinations.append(args.output_dir / "manifest.json")
    if any(path.resolve() == args.input.resolve() for path in destinations):
        raise ValueError("Output files must differ from the input CSV")
    if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in destinations):
        raise ValueError("Managed output paths must be regular files")
    if any(path.exists() for path in destinations) and not args.overwrite:
        raise FileExistsError("Output files already exist; choose a new --output-dir or pass --overwrite")
    manifest = args.output_dir / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if not isinstance(previous, dict) or previous.get("tool") != "generate_workloads":
            raise ValueError("Output directory has an unrelated manifest; choose a dedicated workload output directory")


def uniform_parameters(args):
    return {
        "target_bin_count": args.uniform_target_bins,
        "interval_grid_size": args.uniform_interval_grid_size,
        "candidate_limit": args.uniform_candidate_limit,
        "maximum_tv": args.uniform_maximum_tv,
        "maximum_ks": args.uniform_maximum_ks,
        "maximum_wasserstein": args.uniform_maximum_wasserstein,
        "maximum_gap_bin_widths": args.uniform_maximum_gap_bin_widths,
    }


def select_uniform(scores, ids, size, parameters, seed):
    record = {"requested_rows": size, "status": "infeasible", "rows": 0}
    try:
        result = find_fine_uniform_selections(scores, ids, size, **parameters)
    except (InputValidationError, InfeasibleError) as error:
        record["reason"] = str(error)
        return None, record
    record.update(candidate_count=result.candidate_count,
                  evaluated_candidate_count=result.evaluated_candidate_count,
                  quantile_endpoint_count=result.quantile_endpoint_count)
    if result.strict_indices is None:
        record.update(reason="No searched confidence interval passed every strict uniformity limit",
                      closest_candidate_metrics=asdict(result.best_effort_metrics))
        return None, record
    candidate = result.strict_candidate
    selected, shuffle_seed = shuffle_indices(result.strict_indices, seed=seed, method="Uniform")
    record.update(status="generated", rows=len(selected),
                  confidence_interval=[candidate.score_minimum, candidate.score_maximum],
                  metrics=asdict(result.strict_metrics),
                  bin_edges=candidate.edges, bin_capacities=candidate.projection.capacities,
                  bin_quotas=candidate.projection.integer_quotas, shuffle_seed=shuffle_seed,
                  membership_depends_on_seed=False, output_order_depends_on_seed=True,
                  selection_operator="capacity_projected_fine_uniform_nearest_target_matching")
    return selected, record


def generate(args):
    validate_configuration(args)
    # Hash around reading so an input edited concurrently cannot receive false provenance.
    source_hash = sha256(args.input)
    ids, scores, crosscheck = read_confidence(args.input, args.id_column, args.confidence_column, args.logprob_column)
    if sha256(args.input) != source_hash:
        raise ValueError("Input CSV changed while it was being read")
    sizes = {method: args.uniform_sample_size if method == "Uniform" else args.sample_size for method in args.methods}
    if any(size > len(ids) for size in sizes.values()):
        raise ValueError("Input contains fewer unique rows than a requested sample size")
    parameters = uniform_parameters(args)
    records, selections = {}, {}
    for method in args.methods:
        if method == "Uniform":
            selected, record = select_uniform(scores, ids, sizes[method], parameters, args.seed)
        else:
            if method == "Mixed":
                selected, detail = select_mixed_indices(scores, ids, sizes[method], seed=args.seed)
            else:
                selected, detail = exact_rank_indices(scores, ids, sizes[method], descending=method == "High")
            record = {"status": "generated", "requested_rows": sizes[method], "rows": len(selected),
                      "selection_details": detail}
        if selected is not None:
            if len(selected) != sizes[method] or len(np.unique(ids[selected])) != sizes[method]:
                raise AssertionError("Selection did not produce the requested number of unique IDs")
            selections[method] = ids[selected]
            record["path"] = f"{method}/ids.csv"
        records[method] = record
        print(f"{method}: {record['status']} ({record['rows']}/{sizes[method]} rows)", flush=True)
    manifest = {
        "tool": "generate_workloads", "format_version": 1,
        "complete": len(selections) == len(args.methods),
        "source": {"file": args.input.name, "sha256": source_hash, "rows": len(ids),
                   "id_column": args.id_column, "confidence_column": args.confidence_column,
                   "confidence_semantics": "provided_predicted_label_probability",
                   "logprob_column": args.logprob_column, "logprob_crosscheck": crosscheck},
        "seed": args.seed, "requested_methods": list(args.methods),
        "uniform_parameters": parameters if "Uniform" in args.methods else None,
        "methods": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".workloads-", dir=args.output_dir) as temporary:
        staging = Path(temporary)
        for method, selected_ids in selections.items():
            path = staging / method / "ids.csv"
            path.parent.mkdir()
            pd.DataFrame({"id": selected_ids}).to_csv(path, index=False, lineterminator="\n")
            records[method]["sha256"] = sha256(path)
        (staging / "manifest.json").write_text(json.dumps(json_value(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n")
        for method in METHODS:
            target = args.output_dir / method / "ids.csv"
            if method in selections:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / method / "ids.csv", target)
            elif args.overwrite:
                target.unlink(missing_ok=True)
        os.replace(staging / "manifest.json", args.output_dir / "manifest.json")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV or CSV.gz with IDs and predicted-label confidence")
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--confidence-column", required=True)
    parser.add_argument("--logprob-column", help="Optional chosen-label log-probability cross-check")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", type=str.capitalize, choices=METHODS, default=list(METHODS))
    parser.add_argument("--sample-size", type=int, default=10_000, help="Size for Low, High, and Mixed")
    parser.add_argument("--uniform-sample-size", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uniform-target-bins", type=int, default=60)
    parser.add_argument("--uniform-interval-grid-size", type=int, default=201)
    parser.add_argument("--uniform-candidate-limit", type=int, default=100, help="Retained candidates per search pool")
    parser.add_argument("--uniform-maximum-tv", type=float, default=0.10)
    parser.add_argument("--uniform-maximum-ks", type=float, default=0.02)
    parser.add_argument("--uniform-maximum-wasserstein", type=float, default=0.01)
    parser.add_argument("--uniform-maximum-gap-bin-widths", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace this run's four managed ID files and manifest")
    return parser, parser.parse_args(argv)


def main():
    parser, args = parse_args()
    try:
        manifest = generate(args)
    except (ValueError, OSError, InputValidationError, InfeasibleError) as error:
        parser.error(str(error))
    if not manifest["complete"]:
        print("Some requested workloads were infeasible; see manifest.json. No best-effort IDs were emitted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
