"""Calculate full-information kappa from fixed proxy scores and oracle labels."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

import kappa_core as core
from workload_selection import normalize_ids


def read_inputs(path, id_column, oracle_column, ids=None, selection_id_column="id"):
    if id_column == oracle_column:
        raise ValueError("ID and oracle-label columns must be distinct")
    frame = pd.read_csv(path, dtype={id_column: str}, keep_default_na=False)
    if id_column not in frame or oracle_column not in frame:
        raise ValueError("The score CSV must contain the requested ID and oracle-label columns")
    frame[id_column] = normalize_ids(frame[id_column], "Score IDs")
    collisions = {"document_id", "oracle_label"}.intersection(frame.columns) - {id_column, oracle_column}
    if collisions:
        raise ValueError(f"Rename columns conflicting with normalized ID/label fields: {sorted(collisions)}")
    if ids is not None:
        selected = pd.read_csv(ids, dtype={selection_id_column: str}, keep_default_na=False)
        if list(selected.columns) != [selection_id_column]:
            raise ValueError(f"Selection CSV must contain only the {selection_id_column!r} column")
        selected_ids = normalize_ids(selected[selection_id_column], "Selection IDs")
        if not set(selected_ids) <= set(frame[id_column]):
            raise ValueError("Selection IDs are absent from the score CSV; both inputs must use the same record identities")
        frame = frame.set_index(id_column).loc[selected_ids].reset_index()
    labels = pd.to_numeric(frame[oracle_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Oracle labels must be binary 0/1")
    frame[oracle_column] = labels.astype(np.int8)
    return frame.rename(columns={id_column: "document_id", oracle_column: "oracle_label"})


def calculate(frame, alphas, mode="single", score_column="proxy_score", stage_metadata=None):
    if mode not in {"single", "cascade"}:
        raise ValueError("Mode must be single or cascade")
    if frame.empty or "oracle_label" not in frame:
        raise ValueError("A nonempty frame with oracle_label is required")
    labels = pd.to_numeric(frame["oracle_label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Oracle labels must be binary 0/1")
    frame = frame.copy()
    frame["oracle_label"] = labels.astype(np.int8)
    if mode == "single":
        if score_column not in frame:
            raise ValueError(f"Missing score column: {score_column}")
        scores = pd.to_numeric(frame[score_column], errors="raise")
        if not np.isfinite(scores).all():
            raise ValueError("Proxy scores must be finite; missing scores must be resolved before kappa calculation")
        frame = frame.copy()
        frame["proxy_score"] = scores
        return core.optimize_groups(core.score_groups(frame), alphas, {})
    if not isinstance(stage_metadata, (list, tuple)):
        raise ValueError("Stage metadata must be a list describing the fixed cascade")
    stage_fields = {"prediction": set(), "confidence": set()}
    for column in frame:
        if not column.startswith("stage_") or not column.endswith(("_prediction", "_confidence")):
            continue
        match = re.fullmatch(r"stage_(\d+)_(prediction|confidence)", column)
        if match is None or column != f"stage_{int(match[1]):02d}_{match[2]}":
            raise ValueError("Stage fields must use names such as stage_01_prediction and stage_01_confidence")
        stage_fields[match[2]].add(int(match[1]))
    if stage_fields["prediction"] != stage_fields["confidence"]:
        raise ValueError("Each stage must have both a prediction and confidence column")
    stages = sorted(stage_fields["prediction"])
    if stages != list(range(1, len(stages) + 1)) or len(stage_metadata) != len(stages):
        raise ValueError("Stages must be contiguous from 01 and match the stage metadata")
    frame = frame.copy()
    for stage, metadata in zip(stages, stage_metadata):
        required = {"stage", "surrogate", "model", "requested_fraction"}
        if not isinstance(metadata, dict) or not required <= set(metadata) or metadata["stage"] != stage:
            raise ValueError("Stage metadata needs stage, surrogate, model, and requested_fraction in stage order")
        try:
            fraction = float(metadata["requested_fraction"])
        except (TypeError, ValueError) as error:
            raise ValueError("Stage document fractions must be numeric") from error
        if not 0 < fraction <= 1:
            raise ValueError("Stage document fractions must lie in (0, 1]")
        p, c = f"stage_{stage:02d}_prediction", f"stage_{stage:02d}_confidence"
        if p not in frame or c not in frame:
            raise ValueError(f"Stage {stage} requires {p} and {c}")
        prediction = pd.to_numeric(frame[p], errors="raise")
        confidence = pd.to_numeric(frame[c], errors="raise")
        if not prediction.isin([0, 1]).all() or not np.isfinite(confidence).all() or not confidence.between(0, 1).all():
            raise ValueError("Each stage needs binary predictions and finite confidence in [0, 1]")
        frame[p], frame[c] = prediction.astype(np.int8), confidence.astype(float)
    results = core._optimize_exact_task_cascade(frame, alphas, {}, stage_metadata)
    for row in results:
        if not row["optimality_proven"]:
            # Keep the feasible policy and bounds, without calling it exact kappa.
            row["kappa"] = None
    return results


def json_value(value):
    if isinstance(value, dict): return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, np.generic): return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="CSV containing fixed scores and oracle labels")
    parser.add_argument("--alpha", nargs="+", default=["0.95"])
    parser.add_argument("--mode", choices=["single", "cascade"], default="single")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--oracle-column", default="oracle_label")
    parser.add_argument("--score-column", default="proxy_score")
    parser.add_argument("--ids", type=Path, help="Optional ID-only confidence selection")
    parser.add_argument("--selection-id-column", default="id", help="ID column in the one-column --ids CSV")
    parser.add_argument("--stage-metadata", type=Path, help="JSON list describing the fixed cascade stages")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-backend", choices=["auto", "highspy", "scipy"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        alphas = sorted(set(Decimal(value) for value in args.alpha))
        if not alphas or any(not a.is_finite() or not 0 < a <= 1 for a in alphas):
            raise ValueError("Accuracy targets must lie in (0, 1]")
        if args.workers < 1: raise ValueError("Worker count must be positive")
        if args.mode == "cascade" and args.stage_metadata is None:
            raise ValueError("--stage-metadata is required in cascade mode")
        if args.output.resolve() in {p.resolve() for p in (args.scores, args.ids, args.stage_metadata) if p is not None}:
            raise ValueError("Output must differ from input files")
        if args.mode == "single" and args.score_column in {args.id_column, args.oracle_column}:
            raise ValueError("Proxy score, ID, and oracle-label columns must be distinct")
        frame = read_inputs(args.scores, args.id_column, args.oracle_column, args.ids, args.selection_id_column)
        metadata = json.loads(args.stage_metadata.read_text()) if args.stage_metadata else None
        core.DECOMPOSITION_WORKERS = args.workers
        core.MIP_BACKEND = args.solver_backend
        results = calculate(frame, alphas, args.mode, args.score_column, metadata)
    except (ValueError, KeyError, ArithmeticError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value(results), indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} kappa result(s) for {len(frame)} records to {args.output}")


if __name__ == "__main__":
    main()
