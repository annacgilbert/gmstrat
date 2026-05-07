# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python (required):**
```bash
conda env create -f environment.yml && conda activate gmstrat
```

**Julia (required for sampling only):**
```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.11 && juliaup default 1.11
```

## Running the Pipeline

**1. Generate samples (Julia):**
```bash
./sampling/run.sh --map-file data/graph/grid_graph_5_by_5.json \
  --output-file local/output/grid5x5/atlas.jsonl.gz \
  --cycle-walk-steps 1e5 --pop-dev 0.1
```
Dumps a sample every 100 steps; override with `cycle_walk_out_freq`. See `sampling/run_cyclewalk.jl` for all options (score selection, etc.).

**2. Analysis (Python):** Run `demo.ipynb` in Jupyter — it contains the full pipeline.

There is no automated test suite. The project is research code run interactively.

## Architecture

This project explores the state space of redistricting plans on small grid graphs using MCMC sampling followed by hierarchical clustering analysis.

### Pipeline Flow

```
Grid JSON → Julia CycleWalk MCMC → JSONL.gz samples
         → SampleProcessor (Python)
         → Feather files (districts catalog, plans)
         → Distance matrix (.npy)
         → HCC linkage matrix (.npy)
         → Word/sequence analysis & visualization (notebook)
```

### Key Data Representations

- **District vector:** numpy array of precinct indices belonging to one district
- **Plan vector:** numpy array assigning each precinct to a district_id
- **District UIDs:** integer IDs assigned after deduplication across all samples
- **Plans:** stored as sequences of district UIDs (e.g., `[5, 12, 8, 22]`)

### Key Modules

- `sample.py` — `SampleProcessor` orchestrates all processing: reads JSONL.gz, deduplicates districts, builds Feather catalogs, computes distance matrices. `SampleStoragePaths` manages output paths.
- `data.py` — generates grid graphs and GeoDataFrames from graph JSON files
- `utils.py` — vector/string conversions, visualization helpers, population-weighted L1 distance
- `word.py` — "word" abstraction: sequences of HCC cluster IDs used for stratification; candidate merging and nearest-word algorithms
- `hccfit.py` — fits hierarchical cluster-center (HCC) linkage (adapted from arXiv:2409.01010)
- `hierachical.py` — `HccLinkage` and `HClusters` wrappers for centroid-criterion clustering

### File Formats

| Format | Contents |
|--------|----------|
| `data/graph/*.json` | Graph input: nodes with `precinct_id_str`, `population`; adjacency lists |
| `local/output/**/*.jsonl.gz` | MCMC samples: JSON lines, gzipped; header line contains `districts` count |
| `local/output/**/*.feather` | Districts catalog and plan summaries (Apache Arrow columnar) |
| `local/output/**/*.npy` | Distance matrices, linkage matrices, pairwise distance edges |

Output is written to `local/` (git-ignored). Expensive computations (distance matrix, linkage) are computed once and cached as `.npy` files.

### Sampling Parameters

- `--pop-dev`: population deviation tolerance (e.g., `0.1` = ±10%)
- `--cycle-walk-steps`: total MCMC steps
- Samples output every 100 steps by default (`cycle_walk_out_freq`)
- Scores default to all-zeros; configurable in `run_cyclewalk.jl`
