# Semantic Filter Benchmark

This artifact accompanies the paper's study of LOTUS, BARGAIN, ScaleDoc, and TaskCascade. It provides the ten baseline datasets without annotations, the oracle-confidence workload IDs from Section 6.2, dataset-independent tools for generating confidence workloads and calculating cascadeability **κ**, and a patch that adds an **accuracy target to the LOTUS interface**. The technical report is available as [TR.pdf](TR.pdf). Experimental settings are documented below; experiment orchestration and figure-generation pipelines are omitted.

## Contents

```text
README.md
TR.pdf
LICENSE
requirements.txt
lotus-accuracy.patch
scripts/
  generate_workloads.py
  workload_selection.py
  calculate_kappa.py
  kappa_core.py
data/
  manifest.json
  predicates.csv
  baseline/{PubMed,BigPatent,FEVER,GovReport,Court,Legal,Enron,Movie,Wiki,Steam}.csv.gz
  selected_10k/
    gemma4_12b/{Wiki,Steam}/{Low,High,Mixed,Uniform}/ids.csv
    qwen36_27b/{Wiki,Steam}/{Low,High,Mixed,Uniform}/ids.csv
```

`kappa_core.py` retains the paper's exact optimization routines and their dependencies. Plotting, report generation, historical-result checks, and experiment runners have been removed. The artifact contains no labels, saved model responses, model weights, or implementations of the other three systems.

## Data and predicates

| Dataset | Records | Predicate input |
|---|---:|---|
| PubMed | 10,000 | `sentence` |
| BigPatent | 10,000 | `abstract` |
| FEVER | 15,981 | `text` |
| GovReport | 10,000 | `summary` |
| Court | 16,902 | `text` |
| Legal | 489 | `text` |
| Enron | 9,997 | `text` |
| Movie | 1,000 | `text` |
| Wiki | 90,000 | `text` |
| Steam | 90,000 | `review` |
| **Total** | **254,369** | |

Each dataset is paired with one complete natural-language predicate in `data/predicates.csv`, with columns `dataset,predicate_id,predicate_column,predicate`. Dataset names and confidence-variant directories follow the paper. Each baseline is stored in one gzip-compressed CSV named after the dataset, such as `Enron.csv.gz`. Document order is preserved; Enron contains 9,997 canonical records.

The baselines retain document content, source identifiers, and source metadata. Model labels, probabilities, log-probabilities, token usage, parsing diagnostics, GPT annotations, and original task annotations are removed. In particular, the release excludes FEVER's `supported`, PubMed's source `label`, CUAD's annotated contract fields, Court's decision-direction/vote annotations, and Steam's `recommended` field. The manifest lists the exact retained columns.

Eight workloads use `source_row_number` as their benchmark identity. Wiki and Steam use string `id`. Steam additionally retains `review_id` as source metadata for joining older score exports.

### Confidence workloads: Section 6.2

Only the two oracle selection models are included: **`gemma4_12b`** and **`qwen36_27b`**. Each has Wiki and Steam selections under the paper's variant names:

| Directory | Records | Membership rule |
|---|---:|---|
| `Low` | 10,000 | Lowest predicted-label confidence |
| `High` | 10,000 | Highest predicted-label confidence |
| `Mixed` | 10,000 | 5,000 records from each confidence tail |
| `Uniform` | 3,000 | Records distributed across the confidence range |

There are **16 ID-only files**, containing **132,000 selection entries** in total. Every file contains exactly one column named `id`, with unique IDs in the original selection order. These selections reuse the released Wiki/Steam documents; overlapping selections do not introduce additional source records. Class-ratio variants and selections made by the smaller proxy models are omitted.

## Generate workloads from any dataset

`scripts/generate_workloads.py` accepts any CSV or CSV.gz containing unique record IDs and **the probability assigned to the model's predicted label**. Choose the column names explicitly; there are no built-in dataset names, model names, predicate fields, or configuration directories. Text and label columns are not required or copied to the outputs.

For an input table such as `records.csv.gz`, with columns `record_key` and `answer_confidence`:

```bash
python -m pip install -r requirements.txt
python scripts/generate_workloads.py records.csv.gz \
  --id-column record_key --confidence-column answer_confidence \
  --output-dir outputs/workloads
```

This requests all four methods. `--methods Low High Mixed Uniform` selects an explicit subset or ordering. Names are case-insensitive at the command line and use the paper's capitalization in output directories. `--sample-size` controls Low/High/Mixed (default 10,000); `--uniform-sample-size` controls Uniform (default 3,000). For example:

```bash
python scripts/generate_workloads.py records.csv.gz \
  --id-column record_key --confidence-column answer_confidence \
  --methods Low Mixed --sample-size 2000 --seed 17 \
  --output-dir outputs/custom_workloads
```

The input must contain enough unique records for each requested method. IDs are read as strings, preserving leading zeros and literal IDs such as `NA`. Missing, blank, or duplicate IDs and nonfinite/out-of-range confidence values are rejected; rows are never silently dropped. Confidence must be in `[0,1]`. The tool uses the supplied confidence directly, without clipping, quantile calibration, class-probability conversion, or implicit model-specific normalization. `P(True)` and quantile-calibrated proxy scores are not substitutes for predicted-label confidence. If necessary, compute the intended confidence with your model interface first.

An optional `--logprob-column answer_logprob` checks that confidence equals the exponential of the provided chosen-label log-probability, within an absolute tolerance of `1e-9`. This does not change the selection scores.

### Selection rules

| Method | Rule |
|---|---|
| Low | Select the exact lowest-confidence records. |
| High | Select the exact highest-confidence records. |
| Mixed | Select equal, disjoint low/high tails; sample size must be even. |
| Uniform | Find a capacity-feasible confidence interval and select records approximating a continuous uniform distribution within it, subject to all strict checks below. |

Ranking breaks confidence ties by canonical string ID. When tied tails overlap, Mixed retains the low-tail choices and takes the next disjoint records in high-rank order. Uniform uses capacity-projected quotas and nearest-target matching within bins. The default seed is 42; it controls the final Mixed/Uniform output order, not their deterministic membership. Given the same ID/confidence pairs, changing source-row order does not change the selections.

Strict Uniform searches quantile-defined interval candidates and chooses the widest passing interval among the searched candidates. It does **not** promise uniformity over the whole `[0,1]` range, prove that no possible interval exists when its search fails, or silently substitute a disconnected-support or best-effort distribution.

| Uniform option | Default |
|---|---:|
| `--uniform-target-bins` | 60 |
| `--uniform-interval-grid-size` | 201 quantile endpoints |
| `--uniform-candidate-limit` | 100 candidates per pool; up to 200 retained |
| `--uniform-maximum-tv` | 0.10 fine-bin total variation |
| `--uniform-maximum-ks` | 0.02 Kolmogorov–Smirnov distance |
| `--uniform-maximum-wasserstein` | 0.01 interval-normalized Wasserstein distance |
| `--uniform-maximum-gap-bin-widths` | 2.0 |

### Generated files and infeasibility

```text
outputs/workloads/
  Low/ids.csv
  High/ids.csv
  Mixed/ids.csv
  Uniform/ids.csv
  manifest.json
```

Each successful method emits exactly one column named `id`, regardless of the source ID-column name. The manifest records the source checksum, input columns, seed, requested/achieved counts, method status, output checksums, tie-breaking details, and Uniform settings, selected interval, quotas, and metrics. The published data manifest is separate from these generated-run manifests.

If no searched Uniform interval passes all limits, the manifest records `Uniform.status = "infeasible"`, its reason, and available diagnostics. No Uniform ID file or best-effort ID file is emitted. Other successful requested methods are retained. A constant-confidence input is also Uniform-infeasible, while the rank methods remain available.

Exit code `0` means all requested methods succeeded; `1` means a requested workload was infeasible; `2` indicates invalid inputs, configuration, or output errors. Existing managed outputs are refused unless `--overwrite` is specified. On overwrite, the run replaces its four tool-owned ID files and manifest, removing stale files for infeasible or unrequested methods while preserving unrelated files. An unrelated existing manifest is never overwritten. Use a dedicated output directory, such as `outputs/workloads`, to keep historical paper artifacts intact.

The generator does not require or modify the bundled datasets. The existing released selection IDs remain historical paper artifacts; new inputs and settings produce new workloads.

### Reading data and materializing selections

Each released baseline is a single complete gzip-compressed CSV. `data/manifest.json` records its path, row count, columns, compressed size, and SHA-256 checksums for both the compressed file and its CSV content. Pandas reads CSV.gz directly. For your own records, join the normalized output IDs back to the source using the explicitly chosen source ID column:

```python
import pandas as pd

records = pd.read_csv("records.csv.gz", dtype={"record_key": str}, keep_default_na=False)
ids = pd.read_csv("outputs/workloads/Low/ids.csv", dtype={"id": str}, keep_default_na=False)
assert records["record_key"].is_unique and ids["id"].isin(records["record_key"]).all()
selected = records.set_index("record_key").loc[ids["id"]].reset_index()
```

This preserves selection order. The same join applies to the bundled ID files using their documented baseline identities. Source documents and annotations are not embedded in the generated `ids.csv` files.

## LOTUS accuracy-target interface

Use Python 3.11 on Linux. The patch is based on LOTUS revision `136ae4f4a344a2f75d89f811e516dfcb0de30e46` (package version 1.2.4). Apply it to that revision and install the modified package:

```bash
git clone https://github.com/lotus-data/lotus.git lotus-source
git -C lotus-source checkout 136ae4f4a344a2f75d89f811e516dfcb0de30e46
git -C lotus-source apply ../lotus-accuracy.patch
python -m pip install -e ./lotus-source
```

The patch modifies `lotus/types.py`, `lotus/sem_ops/cascade_utils.py`, `lotus/sem_ops/sem_filter.py`, and `lotus/models/lm.py`. It adds `CascadeArgs(accuracy_target=...)` while retaining the original precision/recall behavior when the accuracy target is omitted. It also exposes probability-resolution and cascade diagnostics used by this benchmark.

Accuracy is agreement with the chosen oracle, rather than agreement with independent human annotations. LOTUS normalizes the proxy's True/False token probabilities when both are available; otherwise it uses the emitted binary label as a deterministic score. It applies quantile calibration, importance sampling with replacement, and joint threshold selection using the implementation's normal-approximation lower bound. This estimated bound is not an unconditional finite-sample guarantee.

Use LOTUS directly with your own records and predicate. Set `PROXY_MODEL`, `ORACLE_MODEL`, `PROXY_API_BASE`, `ORACLE_API_BASE`, `MODEL_API_KEY`, and `MAX_CTX_LEN` for your serving setup. Set `FILTER_PREDICATE` to the natural-language predicate with placeholders matching your input columns, such as `{content}`. The same code works independently of the dataset and model family:

```python
import os
import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs

ID_COLUMN = "record_key"  # Change to the identity column in your records.
df = pd.read_csv("records.csv.gz", dtype={ID_COLUMN: str}, keep_default_na=False)
predicate = os.environ["FILTER_PREDICATE"]

def served_model(role):
    return LM(
        model=os.environ[f"{role}_MODEL"],
        api_base=os.environ[f"{role}_API_BASE"],
        api_key=os.environ["MODEL_API_KEY"],
        temperature=0, max_tokens=4,
        max_ctx_len=int(os.environ["MAX_CTX_LEN"]),
    )

oracle, proxy = served_model("ORACLE"), served_model("PROXY")
lotus.settings.configure(lm=oracle, helper_lm=proxy, enable_cache=False)
result, stats = df.sem_filter(
    predicate,
    cascade_args=CascadeArgs(
        accuracy_target=0.95,
        sampling_percentage=0.10,
        failure_probability=0.05,
        cascade_IS_weight=0.9,
        cascade_IS_random_seed=42,
        cascade_num_calibration_quantiles=50,
        cascade_IS_max_sample_range=len(df),
    ),
    return_all=True,
    return_stats=True,
)
```

Endpoints must return log-probabilities and have chain-of-thought reasoning disabled. For local servers that do not require authentication, set `MODEL_API_KEY` to the value expected by the server. Use model-specific serving options for strict True/False output and sufficient context; do not truncate or silently discard overlength records. The explicit sample-range setting above samples from the entire workload, rather than LOTUS's smaller upstream default range.

`result["_filter"]` contains cascade decisions. `stats["proxy_scores"]` contains fixed quantile-calibrated scores for κ calculation, rather than the oracle's predicted-label confidence required for workload generation. Calibration sampling has duplicate draws, so physical oracle requests and unique labeled documents differ. Keep calibration costs separate from inference routing reduction.

## Calculating κ

κ is a retrospective diagnostic: the largest empirical fraction of documents that can bypass the oracle under the fixed cascade's threshold rule while achieving accuracy at least α. It requires **complete oracle labels** for the evaluated population. Those labels and model scores are intentionally absent from the public datasets and must be generated separately. They must not be used to select the practical calibration sample or tune its thresholds.

For a conventional two-threshold proxy, the script optimizes negative acceptance below the lower threshold, positive acceptance above the upper threshold, and oracle fallback between them. It searches all distinct score boundaries without splitting tied scores. The allowed error count is `N - ceil(αN)`.

Install the calculation dependencies:

```bash
python -m pip install -r requirements.txt
```

`scripts/calculate_kappa.py` is a calculation utility for an individual supplied score table; it does not run the benchmark experiments. A conventional score CSV contains:

```text
id,oracle_label,proxy_score
```

IDs must be unique, labels must be binary 0/1, and scores must be finite. Use the actual fixed scores of the system being studied: quantile-calibrated LOTUS scores, normalized BARGAIN scores, or ScaleDoc's learned proxy scores. Do not recalibrate scores separately for a selected subset if the diagnostic is meant to examine a proxy fixed on the baseline population.

For LOTUS, obtain complete reference labels separately through the oracle interface and combine them with the previously saved scores:

```python
reference = df.sem_filter(predicate, return_all=True)  # Oracle only; no cascade_args.
score_table = pd.DataFrame({
    "id": df[ID_COLUMN].astype(str),
    "oracle_label": reference["_filter"].astype(int),
    "proxy_score": stats["proxy_scores"],
})
score_table.to_csv("scores.csv", index=False)
```

The full-oracle pass is extra work for retrospective diagnosis, not part of the practical cascade's calibration cost. Keep the same documents, order, predicate, oracle model, and model configuration across the two passes.

Examples of individual κ calculations, using the normalized score table above and any generated ID selection:

```bash
python scripts/calculate_kappa.py scores.csv \
  --alpha 0.95 --output outputs/kappa.json
python scripts/calculate_kappa.py scores.csv \
  --ids outputs/workloads/Low/ids.csv \
  --alpha 0.95 --output outputs/low_kappa.json
```

Existing score tables do not need to rename their columns. For example, a table using `record_key`, `reference_answer`, and `surrogate_score` can be evaluated directly:

```bash
python scripts/calculate_kappa.py custom_scores.csv.gz \
  --id-column record_key --oracle-column reference_answer \
  --score-column surrogate_score --ids outputs/workloads/High/ids.csv \
  --output outputs/high_kappa.json
```

Generated selections always use `id`. For an external one-column selection CSV with a different header, pass `--selection-id-column chosen_key`. Both inputs must identify the same records; column-name mapping does not translate between different identity systems. IDs are preserved as strings, and selection-file order is retained. Invalid or duplicate IDs, missing selected records, nonbinary oracle labels, and nonfinite proxy scores are rejected. Oracle labels are normalized to numeric 0/1 before optimization. Internal normalized names `document_id` and `oracle_label` must not conflict with other input columns.

### Fixed multi-stage cascades

For a fixed multi-stage cascade, including TaskCascade, supply all fixed proxy stages in their execution order, with columns:

```text
id,oracle_label,stage_01_prediction,stage_01_confidence,stage_02_prediction,stage_02_confidence,...
```

Every stage must have predictions and confidences for **every record**, including records that the original policy accepted earlier. The stage predictions are binary; confidence is the predicted-label probability in `[0,1]`. A native routed-only log is insufficient. The oracle is the terminal fallback and is not included as a proxy stage.

Provide a JSON list describing the fixed stages, for example:

```json
[
  {"stage": 1, "surrogate": "task-1", "model": "your-proxy-model", "requested_fraction": 0.25},
  {"stage": 2, "surrogate": "task-2", "model": "your-proxy-model", "requested_fraction": 0.5}
]
```

Replace these illustrative values with the actual fixed cascade's task/model/fraction metadata. Optional `native_threshold_0` and `native_threshold_1` fields supply an initial policy; they do not restrict the optimization.

```bash
python scripts/calculate_kappa.py task_scores.csv --mode cascade \
  --stage-metadata stages.json --alpha 0.95 --workers 1 \
  --output outputs/task_kappa.json
```

The retained optimizer uses exact threshold-prefix domains, compression, and certified subproblem decomposition. It accepts a record at the first stage whose class-specific threshold is met. Its default backend is HiGHS when available, with SciPy as a fallback (`--solver-backend scipy`). Exact multi-stage optimization can be expensive. Results retain solver bounds and proof status; if optimality is unresolved, the public `kappa` field is `null` and `incumbent_kappa`/`upper_bound_kappa` report the available bounds. An unresolved incumbent is not presented as exact κ.

## Paper configurations

| Setting | Configuration |
|---|---|
| Gemma family | Proxy: Gemma 4 E4B; oracle: Gemma 4 12B |
| Qwen family | Proxy: Qwen 3.5 9B; oracle: Qwen 3.6 27B |
| Serving | SGLang, temperature 0, chain-of-thought disabled |
| Accuracy target | α = 0.95 by default; measured against the family-specific oracle |
| Failure probability | δ = 0.05 where supported |
| Calibration set | 10% of the evaluated document population |
| LOTUS | 50 quantile buckets; importance mixture 0.9; sampling with replacement over the entire workload |
| Canonical seed | 42 |
| Target satisfaction rate | 100 independent runs/query for LOTUS, BARGAIN, and ScaleDoc; one run/query for TaskCascade |
| BARGAIN | Accuracy-target variant; raw True/False probabilities normalized; symmetric two-threshold interpretation |
| ScaleDoc embeddings | NV-Embed-v2 |
| ScaleDoc training | A disjoint 10% training set; 70/30 training/validation split; 50 + 30 epochs; learning rate 0.001; batch size 128 |
| ScaleDoc calibration | 64 histogram bins; smoothing window 5; 33 candidate thresholds |
| TaskCascade generation | o4-mini; three agent iterations; five surrogate requests/iteration |
| TaskCascade document fractions | 0.1, 0.25, 0.5, 1.0 |

The two large-model directories under `selected_10k` identify which **oracle** selected the confidence workload. The smaller models above are runtime proxies, not additional released selection directories.

The LOTUS labeling/live runs used a four-token output cap. Recorded contexts were 4,096 tokens for PubMed, BigPatent, and GovReport; 8,192 for FEVER; and 16,384 for Wiki and Steam. Court and Legal used 131,072. Enron and Movie used 131,072 for Gemma E4B and 262,144 for the other models. Preserve complete predicate inputs and confirm that the server/tokenizer supports the required context.

For monetary-cost calculations, the paper uses the following USD prices per million tokens:

| Model | Input | Output |
|---|---:|---:|
| Qwen 3.6 27B | 0.29 | 2.50 |
| Qwen 3.5 9B | 0.08 | 0.13 |
| Gemma 4 12B | 0.10 | 0.30 |
| Gemma 4 E4B | 0.02 | 0.10 |

These are the paper's experimental cost constants, not a statement of current service pricing. Report achieved accuracy together with oracle reduction; a target violation is not a feasible realization gap. κ omits proxy construction/calibration costs and must not be interpreted as an end-to-end cost reduction.

## Attribution and validation

The LOTUS patch derives from [lotus-data/lotus](https://github.com/lotus-data/lotus) at the revision given above. Its original Apache-2.0 license text is retained in `LICENSE`. The κ routines are extracted from the paper's existing implementation without changing their optimization logic. The generic workload selection core is derived from the original sampler's exact rank and strict Uniform routines; dataset/model seed namespaces and configuration-specific entry points are removed. The other systems' implementations are not bundled; the technical report is provided as [TR.pdf](TR.pdf).

Dataset families follow the paper's bibliography: [PubMed 200k RCT](https://aclanthology.org/I17-2052/), [BIGPATENT](https://aclanthology.org/P19-1212/), [Enron](https://doi.org/10.1007/978-3-540-30115-8_22), [GovReport](https://aclanthology.org/2021.naacl-main.112/), [SCOTUS Opinions](https://www.kaggle.com/datasets/gqfiddler/scotus-opinions), [CUAD](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/6ea9ab1baa0efb9e19094440c317e21b-Paper-round1.pdf), [FEVER](https://aclanthology.org/N18-1074/), [Movie Scripts Corpus](https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus), [ConvoKit/Wikipedia Talk](https://aclanthology.org/2020.sigdial-1.8/), and [Steam Reviews 2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021). The underlying documents retain their source terms; the code license does not relicense them. Separate original data license/notice files were not present in the supplied canonical directory.

Data checks compare checksums, row counts, retained source values, selection identities, and selection order. Selection checks compare the generic routines against the original sampler and cover exact ranks, tied/disjoint tails, strict Uniform success/failure, string IDs, reproducibility, reordered inputs, overwrite behavior, and generator-to-κ interoperability. κ checks compare the retained routines with the original implementation and brute-force solutions on small tied-score and multi-stage inputs. Validation fixtures are kept outside this minimal release. Model inference and full paper experiments are not rerun as part of packaging.
