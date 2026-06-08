# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`gmstrat` is a research pipeline for **stratified sampling of redistricting plans** on grid graphs. The core idea: Markov chain samplers (Cycle Walk, ReCom) generate ensembles of valid districting plans; this pipeline then clusters districts into representative "letters," combines them into "words" (representative plans), assigns each plan to its nearest words via a partition of unity, and computes stratum weights and a flux matrix for EMUS-style umbrella sampling.

The current research focus is applying the pipeline to **m × n grid graphs** (as opposed to real states like Connecticut or North Carolina), where small cases admit enumeration of ground truth and systematic variation of parameters.

## Environment setup

**Preferred (venv):**
```bash
brew install gdal geos proj libspatialindex   # macOS native deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name gmstrat-venv --display-name "Python (gmstrat-venv)"
```

**Conda alternative:**
```bash
conda env create -f environment.yml && conda activate gmstrat
```

**Julia (for sampling):** requires Julia 1.11+ via `juliaup`. The sampling script runs under `sampling/runCycleWalkEnv/`.

## Running sampling (CycleWalk)

```bash
./sampling/run.sh \
  --map-file data/graph/grid_graph_5_by_5.json \
  --output-file local/output/grid5x5/atlas.jsonl.gz \
  --cycle-walk-steps 1e5 \
  --pop-dev 0.1
```

Output is a gzipped JSONL file. `local/` is gitignored — put all sampling output and scratch work there. See `sampling/run_cyclewalk.jl` for additional flags (score selections, output frequency via `cycle_walk_out_freq`).

## Key Python modules

| File | Role |
|---|---|
| `data.py` | Grid graph generation (`generate_grid_graph`) and GeoPandas shape generation (`generate_grid_shape`) |
| `sample.py` | `SampleProcessor` — reads `.jsonl.gz` sample files, deduplicates districts, computes pairwise distances, and builds/saves intermediate `.feather` files and distance/linkage `.npy` files |
| `hccfit.py` | `HccLinkage` — HccUltraFit hierarchical clustering algorithm that fits an ultrametric to pairwise district distances |
| `hierachical.py` | `HClusters` — wraps a saved linkage matrix; cuts the dendrogram at K clusters, computes centroids (letters), and optionally refines with k-means |
| `word.py` | Beam search (`nearest_words`) for finding the L nearest words (unordered tuples of letter indices) to each observed plan |
| `utils.py` | Shared utilities: `weighted_l1` distance (capped, sparse or dense), `vec_to_str`/`str_to_vec` for district encoding, and all `plot_*` visualization functions |

## Data flow

```
data/graph/<grid>.json          # graph spec with node/adjacency/population
    ↓  (sampling/run.sh)
local/output/<run>/atlas.jsonl.gz
    ↓  (SampleProcessor.process_samples)
local/<run>/samples.feather
local/<run>/plans.feather
local/<run>/districts.feather
local/<run>/distributions.feather
local/<run>/distance_matrix.npy
    ↓  (SampleProcessor.ensure_linkage → HccLinkage)
local/<run>/linkage.npy
    ↓  (HClusters + word.py)
letters (cluster centroids), words (nearest_words beam search), POU, flux matrix
```

Intermediate `.feather` and `.npy` files are cached on disk; re-running a stage skips recomputation if the file already exists.

## Notebooks

- `demo.ipynb` — canonical end-to-end walkthrough of the full pipeline
- `grid_experiments.ipynb` — current active experiments on grid graphs (branch `grid-experiments`)
- `exhaustive_3x3.ipynb` — exhaustive enumeration experiments on 3×3 grids

## Key distances and parameters

- **District distance:** population-weighted capped ℓ¹, `d_max = 2 * total_pop / num_districts`. Stored as `int32`.
- **Plan distance:** minimum-cost matching between districts under district distance.
- **Letters (K):** chosen by the elbow method on within-cluster dispersion; `HClusters.update_clusters(K)` cuts at K.
- **Word degree (L):** number of nearest words per plan in the beam search.
- **POU temperature T and kernel κ:** control overlap between strata.

## Grid graph files

Pre-generated grids live in `data/graph/`. Naming convention: `grid_NxN_k<districts>.json`. The `generate_grid_graph(N, filename, num_districts=k)` function in `data.py` produces new ones. All nodes have population 1 by default (uniform); a custom `population_matrix` can be passed.
