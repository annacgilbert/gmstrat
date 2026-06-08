# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python — preferred: venv**
```bash
brew install gdal geos proj libspatialindex   # macOS, one-time
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name gmstrat-venv --display-name "Python (gmstrat-venv)"
```

**Python — alternative: conda/mamba**
```bash
conda env create -f environment.yml && conda activate gmstrat
```

**Julia (required for sampling only):**
```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.11 && juliaup default 1.11
```

The project currently runs under **pyenv Python 3.13.0** when no venv is active. `scikit-learn` is in `requirements.txt` and is used for MDS embeddings.

## Running the Pipeline

**0. Generate a grid graph JSON (if needed):**
```python
from data import generate_grid_graph
generate_grid_graph(N=6, filename="data/graph/grid_6x6_k2.json", num_districts=2)
```
Existing JSONs are in `data/graph/`. The file name convention is `grid_NxN_kI.json` (N=grid side, I=districts).

**1. Generate samples (Julia):**
```bash
./sampling/run.sh --map-file data/graph/grid_graph_5_by_5.json \
  --output-file local/output/grid5x5/atlas.jsonl.gz \
  --cycle-walk-steps 1e5 --pop-dev 0.1
```
Dumps a sample every 100 steps; override with `--cycle-walk-out-freq N`. See `sampling/run_cyclewalk.jl` for all options (score selection, RNG seed, etc.).

**2. Analysis (Python):** Three notebooks cover different stages:
- `demo.ipynb` — full pipeline: `SampleProcessor` → clustering → words → flux
- `exhaustive_3x3.ipynb` — complete enumeration of the 3×3 grid for ground-truth comparison
- `grid_experiments.ipynb` — systematic experiments varying grid size, with extensive visualizations (district grids, distance matrix, dendrograms, MDS embeddings, Poincaré disk)

There is no automated test suite. The project is research code run interactively.

## Architecture

This project explores the state space of redistricting plans on small grid graphs via CycleWalk MCMC sampling, then stratifies the resulting ensemble using hierarchical clustering.

### Pipeline Flow

```
data.py: generate_grid_graph()
  → data/graph/*.json

sampling/run.sh (Julia CycleWalk)
  → local/output/**/*.jsonl.gz

SampleProcessor.process_samples()       # slow: reads JSONL, deduplicates districts
  → *.feather (districts catalog, plans, distributions)
  → pdist_edges.npy (distance quantile edges)

SampleProcessor.load_distance_matrix()  # slow: O(N²) pairwise distances, cached
  → distance_matrix.npy

SampleProcessor.ensure_linkage()        # slow: HccLinkage fit, cached
  → linkage.npy

HClusters.update_clusters(K)            # fast: cuts dendrogram at K clusters
  → centroids and cluster_densities in memory

PlanWordBuilder.build()                 # assigns plans to nearest K-letter words
  → PlanWordResult (df_plans, df_words, district_cluster_distances)

WordStat(plan_word, temp)
  → stationary_distribution(): stratum weights z_s
  → flux_matrix(): row-stochastic F_{s,s'}
```

After `process_samples()` runs once, use `load_processed()` to reload from cached Feather files without re-parsing the JSONL.

### Key Data Representations

- **District (sparse):** numpy array of sorted precinct indices in the district (e.g. `[0, 1, 5]`)
- **District (dense/vector):** binary numpy array over all precincts
- **District string:** `"."` -joined sorted precinct indices (used as dedup key and feather storage)
- **Plan vector:** numpy int array mapping each precinct index → district_id (0-indexed)
- **Plan (as UIDs):** sorted list of district UIDs, stored as `"."` -joined string in feather
- **Maximum distance:** `2 * total_pop // num_districts` — distances are capped at this value to prevent unmatched districts from inflating the metric
- **Letters:** cluster centroids — precincts where ≥50% of cluster members include that precinct (population-weighted majority vote under L¹)
- **Words:** sorted I-tuples of letter indices (unordered, like a plan), found via beam search in `nearest_words()`

### Key Modules

- `sample.py` — `SampleProcessor` is the central object; `SampleStoragePaths` manages all output paths. Two main entry points: `process_samples()` (full ingestion) and `load_processed()` (reload from cache).
- `data.py` — `generate_grid_graph(N, filename, num_districts=K)` writes a graph JSON; `generate_grid_shape(N)` returns a GeoDataFrame for visualization.
- `hccfit.py` — `HccLinkage` implements the HccUltraFit algorithm (arXiv:2409.01010); call `hcc.learn_UM()` then save `hcc.Z` as the linkage matrix.
- `hierachical.py` — `HClusters` wraps a saved linkage: `update_clusters(K)` cuts the dendrogram, `update_centroids()` computes letter representatives, `kcentroids()` refines with k-medoids iterations. Note: filename is intentionally `hierachical.py` (one 'r'). `get_cluster_from_linkage(K, Z)` is a standalone helper that returns `(c2i, i2c)` for the tree-cut without a full `HClusters` object.
- `word.py` — `PlanWordBuilder.build()` runs beam search to assign plans to nearest words; `WordStat` computes stratum weights and the flux matrix F. `WordStat` also exposes `adjacency_matrix()`, `laplacian_matrix()`, `spectral_stats()`, `stationary_table()`, and `flux_dataframe()`.
- `utils.py` — `weighted_l1()` (sparse and dense modes), `vec_to_str()`/`str_to_vec()`, and all plotting helpers (`plot_plan`, `plot_district`, `plot_distribution`, `plot_words_list`, `plot_words_combined`, `plot_words_centroids`).

### HClusters Key Attributes

After `update_clusters(K)` and `kcentroids()`:
- `hc.i2c` — `np.ndarray` shape `(n_districts,)`: district_uid → cluster_id
- `hc.c2i` — `dict[int, list[int]]`: cluster_id → list of member district_uids
- `hc.cluster_densities` — `np.ndarray` shape `(K, n_precincts)`: continuous centroid density (fraction of cluster members containing each precinct)
- `hc.centroids` — `list[np.ndarray]`: sparse majority-vote centroid for each cluster (precincts with density ≥ 0.5)

`kcentroids()` reassigns districts to their nearest centroid and updates all four. To get the pure tree-cut assignments (before k-medoids refinement), use `scipy.cluster.hierarchy.fcluster(Z, t=K, criterion='maxclust') - 1`.

### Distance Formula

`SampleProcessor.compute_distance(x, y)` calls `weighted_l1(x, y, population, maximum_distance, sparse=True)`:

```
d(A, B) = sum of population[i] for i in symmetric_difference(A, B)
         capped at maximum_distance = 2 * total_pop // num_districts
```

For uniform-population grids this equals the number of precincts that differ between the two districts. The cap prevents unmatched/incomparable districts from inflating the metric.

### File Formats

| Format | Contents |
|--------|----------|
| `data/graph/*.json` | Graph: nodes with `precinct_id_str`, `population`; `adjacency` lists; `num_districts` |
| `local/output/**/*.jsonl.gz` | CycleWalk samples: gzipped JSON lines; line 2 is a header with `districts` count; each data line has `name` (step number) and `districting` |
| `local/output/**/*.feather` | `districts.feather` (uid + district_str), `plans.feather`, `samples.feather`, `distributions.feather` |
| `local/output/**/*.npy` | `distance_matrix.npy`, `linkage.npy`, `pdist_edges.npy` |

`distributions.feather` schema: `sample_tag`, `plan_str` (dot-joined district UIDs), `count`, `freq`, `plan_vector`. To compute per-district marginal frequency, iterate over rows and accumulate each plan's count into both of its district UIDs.

Output is written to `local/` (git-ignored). Distance matrix and linkage are computed once and cached; delete the `.npy` files to recompute.

### Sampling Parameters

- `--pop-dev`: population deviation tolerance (e.g., `0.1` = ±10%)
- `--cycle-walk-steps`: total MCMC steps (e.g., `1e5`, `1e6`)
- `--cycle-walk-out-freq`: dump sample every N steps (default 100)
- `--rng-seed`: integer seed for reproducibility
- `--gamma`, `--iso-weight`: score function weights (default all-zeros = uniform target)

## Plotting Gotchas

All plot helpers (`plot_district`, `plot_distribution`, etc.) call `ax.axis("off")`, which suppresses axis labels including `set_xlabel()` and `set_ylabel()`. To place text below a subplot:

```python
# Wrong — hidden by axis("off")
ax.set_xlabel("label")

# Correct — use fig.text() in figure coordinates
fig.canvas.draw()   # must flush first so ax.get_position() is accurate
pos = ax.get_position()
fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.03, "label", ha="center", va="top")
```

For inset axes aligned to dendrogram leaves, convert data coordinates to figure coordinates after `fig.canvas.draw()`:
```python
fig.canvas.draw()
x_px, y_px = ax.transData.transform((x_data, y_data))
xf, yf = fig.transFigure.inverted().transform((x_px, y_px))
ax_in = fig.add_axes([xf - w/2, yf - h, w, h])
```
Call `ax.set_ylim(bottom=0)` before the flush to eliminate matplotlib's default y-axis padding so leaf positions map exactly to the axes bottom edge.
