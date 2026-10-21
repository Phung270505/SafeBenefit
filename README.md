# SafeBenefit

## 1. Research overview

SafeBenefit selects a transfer plan at one safe point in a dynamic distributed graph query. The objective is to minimize **total endpoint MAC wire bytes**, including payload, control, retries and switching, while preserving the canonical result and graph version.

| Method | Decision rule |
|---|---|
| `STATIC` | Keep the initial plan selected on calibration-train. |
| `SWITCH_COST_THRESHOLD` | Require savings to cover switching cost and a calibrated margin. |
| `LINUCB` | Frozen contextual bandit with SafeBenefit fallback for missing/OOD features. |
| `SAFE_BENEFIT` | Require savings to cover switching cost, uncertainty and a safety margin. |

Three plans are available: `RAW_ROWS`, `LOCAL_REDUCE`, and `COMPRESSED_REDUCE`. Each scale has separate calibration, training, policies and evaluation: **96 calibration queries, 192 evaluation queries, four query classes, three graph checkpoints, seed 42**. There are 768 main executions per scale; results are not pooled across scales. Ties and plan dominance are valid outcomes.

Byte costs, logical operations, modeled CPU/I/O service and simulated completion time are reported separately. Reference capacities are **10^7 work units/s** and **10^9 I/O bytes/s**. Host runtime is diagnostic only. This is an **OMNeT++/INET simulation of materialized query-result exchange**, not a physical cloud or complete graph-engine benchmark.

Known limitation: the retained LinUCB implementation has differences between Python training and C++ deployment in the compression-ratio feature and fallback uncertainty calculation. Its results do not establish a general ranking of learned versus heuristic optimizers.

## 2. Data

Only **LDBC SNB Interactive v1, Datagen 1.0.0: SF0.1, SF1 and SF10** are active. Obtain CsvBasic–LongDateFormatter snapshots and official update streams from the [LDBC dataset catalogue](https://ldbcouncil.org/benchmarks/snb/datasets/). These are three scales of one synthetic social-graph domain. The queries are custom microbenchmarks, not the full official LDBC query suite.

Datasets and results are excluded from this repository. Keep original inputs under `data/raw/`. Each directory below must contain `snapshot/vertices.csv`, `snapshot/edges.csv`, `updates/calibration/events.jsonl`, `updates/evaluation/events.jsonl`, and `metadata/manifest.json` with matching checksums:

```text
data/processed/ldbc_snb_i1_v100_sf01/
data/processed/ldbc_snb_i1_v100_sf1/
data/processed/ldbc_snb_i1_v100_sf10/
```

Evaluation continues from the graph state after calibration, using its own update stream. **The runner does not download or preprocess data.** Adapter code is in `src/datasets/ldbc.py`, but exact reproduction of the retained experiment requires matching prepared inputs/manifests from the authors; a verified one-command raw-to-prepared reconstruction for every retained scale is not provided.

## 3. Setup and run

Requirements: **Python 3.12+**, Windows/WSL2 distribution `OMNeT-opp_env`, **OMNeT++ 6.4.0**, and **INET 4.7.0**. Pinned Python dependencies and simulator metadata are in `environment/`.

The Windows preflight expects `/home/opp_env/.venv/bin/opp_env` and simulator installations under `/home/opp_env/default_workspace/`. Provision that environment and the prepared data before running.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r environment/requirements.txt
# Git does not retain empty data/runtime directories.
New-Item -ItemType Directory -Force -Path data/raw,data/processed,data/fixtures,output/current | Out-Null
.venv\Scripts\python.exe run.py --preflight
.venv\Scripts\python.exe run.py
```

The default runs validation, materialization, forced calibration plans, training/freeze, evaluation, scalability, stress and reporting sequentially for each scale. Compatible runs resume; incompatible outputs are rejected. Do not edit code/config or start another run during execution.

Results are written separately under `output/current/ldbc_sf01/`, `ldbc_sf1/` and `ldbc_sf10/`, with `metrics/per_query.jsonl`, `metrics/per_query.csv`, `metrics/method_summary.csv`, `metrics/statistics.json` and `report/final_results.md`.

**Restart:** `python run.py --fresh` deletes only `output/current`. Preserve any needed results first.

Local checks (not a full INET experiment):

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python.exe -m ruff check --no-cache src tests run.py
```

Source layout: `src/` (processing, methods, workflow and analysis), `simulator/omnetpp/` (C++/NED and shell helpers), `config/study.json` (experiment design), `tests/` (regression checks). `data/manifests/datasets.json` is the small dataset registry, not dataset content. Papers, figures, outputs, environments and caches are excluded from Git.
