# Fixed-clump L1-neighborhood experiment

This directory implements experiment 1 in Section 6, “A focused
experimental program,” of the conjectures paper.  It is deliberately separate
from the letters/words notebook: the experiment needs the paper's **uncapped,
normalized, unlabeled plan metric**, not the capped district metric used by the
clustering pipeline.

## Experimental contract

The driver enforces the design choices that matter for interpreting the
result.

- Pilot and evaluation atlases must have different paths.  New atlases record
  their RNG seeds, and reused pilot/evaluation seeds are rejected.
- A component of the pilot `d1 < eta` neighbor graph is extended to the full
  continuum plan space as a finite union of **open** `d1` balls.  Each ball has
  radius at most `eta/2`, so extensions of distinct components do not overlap.
  The centers and component membership are serialized before evaluation.
- One pilot sample at `pilot_reference_n` defines the same continuum clumps
  for every lattice size.  Cross-resolution distances are computed exactly
  from overlaps of the two cell grids.
- All reported masses use only held-out chains.  Errors use a Bartlett-HAC
  variance and are no smaller than the empirical between-chain variance.
- The loader requires CycleWalk's tree-count plan marginal: `gamma=0` and
  `iso_weight=0`.  CycleWalk is a forest chain, so `gamma=0` is the setting
  whose projection to partitions has weight equal to the product of district
  spanning-tree counts.
- For configured small grids, exhaustive enumeration computes integer
  restricted partition functions with an exact matrix-tree determinant.
- For `k=2`, vertical and horizontal distances are distances to the **whole
  straight-cut modes**, including the offset interval when the fixed balance
  tolerance is positive.  The pilot data at each `n` estimate `r_jn`; held-out
  data alone estimate the `H_jn` curves.

The clumps are data-dependent but, conditional on the pilot, fixed finite
unions of continuum balls.  This is the “one fixed continuum clump” option in
Section 6; no learned-set sandwich assumption is needed.

## Run

From the repository root, inspect the independent sampling jobs:

```bash
python -m experiments.sample_fixed_clump \
  --config experiments/configs/fixed_clump_k2.example.json
```

Launch only the missing jobs (this is intentionally expensive):

```bash
python -m experiments.sample_fixed_clump \
  --config experiments/configs/fixed_clump_k2.example.json \
  --execute
```

Then run the analysis:

```bash
MPLCONFIGDIR=local/matplotlib python -m experiments.run_fixed_clump \
  --config experiments/configs/fixed_clump_k2.example.json \
  --output local/experiments/fixed_clump_k2/results
```

`sample_fixed_clump` skips existing atlases.  It overwrites them only when
both `--execute` and `--force` are explicit.

## Outputs

The results directory contains:

- `frozen_clumps.json` and `frozen_clump_centers.npz`: the Borel clump
  definitions learned from the pilot;
- `fixed_clump_estimates.csv`: held-out probabilities, HAC errors, effective
  sample sizes, `log(P)/n`, and exact-comparison columns;
- `surface_rate_fits.csv`: weighted fits of `log P(A) = b - n I_hat(A)`;
- `exact_restricted_sums.csv`: exact `Z_n`, `Z_n(A)`, and their ratio;
- `mode_tube_curves.csv`: fixed-radius `H_jn(eta)` estimates;
- `mode_tube_collapse.csv`: held-out curves at `eta = s r_jn`, with `r_jn`
  learned on the pilot;
- two diagnostic PNGs and `manifest.json` recording all input files and chain
  counts.

Tune radii and component filters using pilot output only.  If a held-out clump
has zero hits, the table reports a one-sided 95% upper bound rather than
pretending its error is zero.  A zero-hit clump cannot be used in the
log-linear surface-rate fit; it needs more held-out sampling or a targeted
rare-event method.
