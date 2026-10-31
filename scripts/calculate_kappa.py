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


def read_inputs(path, id_column, oracle_column, ids=None):
    frame = pd.read_csv(path, dtype={id_column: str})
    if id_column not in frame or oracle_column not in frame:
        raise ValueError("The score CSV must contain the requested ID and oracle-label columns")
    if frame.empty or frame[id_column].isna().any() or frame[id_column].eq("").any() or frame[id_column].duplicated().any():
        raise ValueError("Score rows must have nonempty, unique IDs")
    if ids is not None:
        selected = pd.read_csv(ids, dtype={"id": str})
        if list(selected.columns) != ["id"]:
            raise ValueError("Selection CSV must contain only the id column")
        if selected.empty or selected.id.isna().any() or selected.id.eq("").any() or selected.id.duplicated().any():
            raise ValueError("Selection IDs must be nonempty and unique")
        if not set(selected.id) <= set(frame[id_column]):
            raise ValueError("Selection IDs are absent from the score CSV; use the public baseline id identity")
        frame = frame.set_index(id_column).loc[selected.id].reset_index()
    labels = pd.to_numeric(frame[oracle_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Oracle labels must be binary 0/1")
    return frame.rename(columns={id_column: "document_id", oracle_column: "oracle_label"})


def calculate(frame, alphas, mode="single", score_column="proxy_score", stage_metadata=None):
    if mode == "single":
        if score_column not in frame:
            raise ValueError(f"Missing score column: {score_column}")
        scores = pd.to_numeric(frame[score_column], errors="raise")
        if not np.isfinite(scores).all():
            raise ValueError("Proxy scores must be finite; missing scores must be resolved before kappa calculation")
        frame = frame.copy()
        frame["proxy_score"] = scores
        return core.optimize_groups(core.score_groups(frame), alphas, {})
    if stage_metadata is None:
        raise ValueError("Provide the fixed cascade's stage metadata")
    stages = sorted(int(match[1]) for col in frame for match in [re.fullmatch(r"stage_(\d+)_prediction", col)] if match)
    if stages != list(range(1, len(stages) + 1)) or len(stage_metadata) != len(stages):
        raise ValueError("Stages must be contiguous from 01 and match the stage metadata")
    frame = frame.copy()
    for stage, metadata in zip(stages, stage_metadata):
        required = {"stage", "surrogate", "model", "requested_fraction"}
        if not required <= set(metadata) or metadata["stage"] != stage:
            raise ValueError("Stage metadata needs stage, surrogate, model, and requested_fraction in stage order")
        if not 0 < float(metadata["requested_fraction"]) <= 1:
            raise ValueError("Stage document fractions must lie in (0, 1]")
        p, c = f"stage_{stage:02d}_prediction", f"stage_{stage:02d}_confidence"
        if c not in frame:
            raise ValueError(f"Missing confidence column: {c}")
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
        frame = read_inputs(args.scores, args.id_column, args.oracle_column, args.ids)
        metadata = json.loads(args.stage_metadata.read_text()) if args.stage_metadata else None
        core.DECOMPOSITION_WORKERS = args.workers
        core.MIP_BACKEND = args.solver_backend
        results = calculate(frame, alphas, args.mode, args.score_column, metadata)
    except (ValueError, KeyError, ArithmeticError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value(results), indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} kappa result(s) for {len(frame)} records to {args.output}")


if __name__ == "__main__":
    main()
