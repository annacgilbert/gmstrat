# Environment choices — `venv` vs `conda`

This project supports both lightweight Python `venv` workflows and full conda/mamba environment reproduction. I use `venv` locally; below are concise instructions for both approaches so you (or contributors) can reproduce the environment.

## I prefer `venv` (recommended for local/dev)

- Ensure system libs are installed (macOS Homebrew):

```bash
brew install gdal geos proj libspatialindex
```

- Create and activate a virtual environment in the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

- Install Python packages from `requirements.txt` (created from the active venv):

```bash
pip install -r requirements.txt
```

- Register the kernel for notebooks (optional):

```bash
python -m ipykernel install --user --name gmstrat-venv --display-name "Python (gmstrat-venv)"
```

Notes:
- Use the `source .venv/bin/activate` step before running tests or notebooks.
- The file `requirements.txt` was generated from the working `.venv` and pins versions.

## Conda / Mamba (reproducible environments)

If you prefer conda/mamba, `environment.yml` is included. It pins Python 3.13 and lists pip packages under `pip:` for exact reproduction.

Create with:

```bash
# using mamba (recommended):
mamba env create -f environment.yml

# or with conda:
conda env create -f environment.yml

# activate:
conda activate gmstrat
```

## Which to use?
- Use `venv` for a small, fast developer workflow on your machine (you already use this).
- Use `conda`/`mamba` when you need fully reproducible builds across machines or CI, or to avoid compiling GIS native deps locally.

If you'd like, I can add a short script `setup_venv.sh` to automate the venv+install steps — tell me if you want that.
# Requirements

## Python
`conda env create -f environment.yml && conda activate gmstrat`

## Julia

Requires Julia 1.11+ (recommended: `juliaup`):

- macOS/Linux: `curl -fsSL https://install.julialang.org | sh`
- Windows (PowerShell): `winget install --id Julialang.Juliaup -e`
- Then: `juliaup add 1.11 && juliaup default 1.11`

## Sampling
See the `Prepare` section in the demo notebook for how to generate grid json (containing graph information) needed for sampling.

`./sampling/run.sh --map-file data/graph/grid_graph_5_by_5.json --output-file local/output/grid5x5/atlas.jsonl.gz --cycle-walk-steps 1e5 --pop-dev 0.1`

The above samples 1e5 step, where a sample is dumped every 100 steps with a population deviation of 0.1 (i.e. ten percent). The frequency can be overriden with `cycle_walk_out_freq`. Check `run_cyclewalk.kl` for more options, including score selections (default to all-zeros).

## Analysis
Rest of the pipeline is contained in `demo.ipynb`.

The focused fixed-clump experiment for the tree-count conjectures is a
reproducible command-line workflow rather than a notebook.  See
[`experiments/README.md`](experiments/README.md).

## Acknowledgements
The sampling code is adapted from
`https://github.com/jonmjonm/CycleWalk.jl`

The hccfit implementation is adapted from
`https://arxiv.org/pdf/2409.01010` (I can't find the github repo.)
