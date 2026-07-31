"""Run experiment 1 from Section 6: fixed-clump L1 neighborhoods.

Usage (from the repository root)::

    python -m experiments.run_fixed_clump \
        --config experiments/configs/fixed_clump_k2.example.json \
        --output local/experiments/fixed_clump_k2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from experiments.fixed_clump import (
    FrozenClump,
    FrozenClumpModel,
    GridSpec,
    PlanSamples,
    axis_mode_separation,
    exact_restricted_sums,
    fit_surface_rate,
    learn_frozen_clumps,
    mode_distances,
    probability_estimate,
    ratio_estimate,
    read_cyclewalk_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _file_paths(entries: Sequence[str | dict[str, Any]]) -> list[Path]:
    return [_resolve(entry["path"] if isinstance(entry, dict) else entry) for entry in entries]


def _load_run_samples(
    run: dict[str, Any], kind: str, grid: GridSpec, strict_target: bool
) -> PlanSamples:
    return read_cyclewalk_samples(
        _file_paths(run[f"{kind}_files"]),
        grid,
        min_step=int(run.get(f"{kind}_min_step", run.get("min_step", 0))),
        thin=int(run.get(f"{kind}_thin", run.get("thin", 1))),
        max_samples_per_file=run.get(
            f"max_{kind}_samples_per_file", run.get("max_samples_per_file")
        ),
        require_tree_count_target=strict_target,
    )


def _validate_config(config: dict[str, Any]) -> None:
    if not config.get("runs"):
        raise ValueError("config must contain at least one run")
    ns = [int(run["n"]) for run in config["runs"]]
    if len(ns) != len(set(ns)):
        raise ValueError("each lattice scale n must occur exactly once")
    reference_n = int(config["pilot_reference_n"])
    if reference_n not in ns:
        raise ValueError("pilot_reference_n must name one of the configured runs")
    for run in config["runs"]:
        pilot = set(_file_paths(run["pilot_files"]))
        heldout = set(_file_paths(run["heldout_files"]))
        overlap = pilot & heldout
        if overlap:
            raise ValueError(
                "pilot and held-out atlases must be distinct; overlap: "
                + ", ".join(str(path) for path in sorted(overlap))
            )
        pilot_seeds = {
            int(entry["seed"])
            for entry in run["pilot_files"]
            if isinstance(entry, dict) and "seed" in entry
        }
        heldout_seeds = {
            int(entry["seed"])
            for entry in run["heldout_files"]
            if isinstance(entry, dict) and "seed" in entry
        }
        if pilot_seeds & heldout_seeds:
            raise ValueError("pilot and held-out chains must use disjoint RNG seeds")

    mode = config.get("mode_tubes")
    if mode and mode.get("enabled", True):
        delta = float(config.get("balance_tolerance", 0.0))
        separation = axis_mode_separation(delta)
        eta0 = float(mode["eta0"])
        if not 0.0 < eta0 < separation / 3.0:
            raise ValueError(
                f"mode eta0 must lie in (0, Delta/3)=({0}, {separation / 3})"
            )
        if any(not 0.0 < float(eta) < eta0 for eta in mode["eta_values"]):
            raise ValueError("each fixed mode eta must lie in (0, eta0)")


def _serialize_model(
    model: FrozenClumpModel, output: Path, reference_n: int
) -> None:
    np.savez_compressed(output / "frozen_clump_centers.npz", centers=model.centers)
    metadata = {
        "definition": "finite unions of open d1 balls around frozen pilot plans",
        "reference_n": reference_n,
        "reference_graph": str(model.center_grid.graph_path),
        "centers_file": "frozen_clump_centers.npz",
        "distance_normalization": "d1 = 1 - maximum matched overlap fraction",
        "clumps": [
            {
                "name": clump_name(clump),
                "eta": clump.eta,
                "open_ball_radius": clump.radius,
                "component_id": clump.component_id,
                "pilot_observations": clump.pilot_observations,
                "center_indices": clump.center_indices.tolist(),
            }
            for clump in model.clumps
        ],
    }
    (output / "frozen_clumps.json").write_text(json.dumps(metadata, indent=2) + "\n")


def clump_name(clump: FrozenClump) -> str:
    eta = np.format_float_positional(clump.eta, trim="-")
    return f"eta_{eta}_component_{clump.component_id}"


def _subsample(plans: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or len(plans) <= int(maximum):
        return plans
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(plans), size=int(maximum), replace=False))
    return plans[selected]


def _learn_model(
    config: dict[str, Any], loaded: dict[int, dict[str, Any]]
) -> FrozenClumpModel:
    reference_n = int(config["pilot_reference_n"])
    reference = loaded[reference_n]
    pilot = _subsample(
        reference["pilot"].plans,
        config.get("max_reference_pilot_plans"),
        int(config.get("random_seed", 20260731)),
    )
    return learn_frozen_clumps(
        pilot,
        reference["grid"],
        config["etas"],
        extension_radius_factor=float(config.get("extension_radius_factor", 0.5)),
        min_component_observations=int(
            config.get("min_component_observations", 2)
        ),
        max_components_per_eta=config.get("max_components_per_eta"),
    )


def _fixed_clump_rows(
    model: FrozenClumpModel, loaded: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in sorted(loaded):
        heldout = loaded[n]["heldout"]
        distances = model.distance_matrix(heldout.plans, loaded[n]["grid"])
        for clump in model.clumps:
            indicator = model.membership(clump, distances)
            row = {
                "clump": clump_name(clump),
                "eta": clump.eta,
                "open_ball_radius": clump.radius,
                "component_id": clump.component_id,
                "n": n,
                **probability_estimate(indicator, heldout.chain_ids),
            }
            probability = float(row["probability"])
            standard_error = float(row["standard_error"])
            row["log_probability_over_n"] = (
                math.log(probability) / n if probability > 0 else float("-inf")
            )
            row["log_rate_standard_error"] = (
                standard_error / (n * probability)
                if probability > 0
                else float("nan")
            )
            rows.append(row)
    return rows


def _surface_rate_rows(fixed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names = sorted({str(row["clump"]) for row in fixed_rows})
    for name in names:
        selected = [row for row in fixed_rows if row["clump"] == name]
        result.append(
            {
                "clump": name,
                "eta": selected[0]["eta"],
                "component_id": selected[0]["component_id"],
                **fit_surface_rate(selected),
            }
        )
    return result


def _positive_pilot_scale(
    distances: np.ndarray, eta0: float, quantile: float
) -> tuple[float, bool, int]:
    inside = distances[distances < eta0]
    if not len(inside):
        return float("nan"), False, 0
    scale = float(np.quantile(inside, quantile))
    floored = False
    if scale <= 0.0:
        positive = inside[inside > 0.0]
        if not len(positive):
            return float("nan"), False, len(inside)
        scale = float(positive.min())
        floored = True
    return scale, floored, len(inside)


def _mode_rows(
    config: dict[str, Any], loaded: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = config.get("mode_tubes")
    if not mode or not mode.get("enabled", True):
        return [], []
    delta = float(config.get("balance_tolerance", 0.0))
    eta0 = float(mode["eta0"])
    quantile = float(mode.get("scale_quantile", 0.5))
    fixed_rows: list[dict[str, Any]] = []
    collapsed_rows: list[dict[str, Any]] = []

    for n in sorted(loaded):
        grid = loaded[n]["grid"]
        pilot = loaded[n]["pilot"]
        heldout = loaded[n]["heldout"]
        for orientation in ("vertical", "horizontal"):
            pilot_distances = mode_distances(
                pilot.plans, grid, orientation, delta
            )
            heldout_distances = mode_distances(
                heldout.plans, grid, orientation, delta
            )
            scale, floored, scale_count = _positive_pilot_scale(
                pilot_distances, eta0, quantile
            )
            denominator = heldout_distances < eta0

            for eta in mode["eta_values"]:
                eta = float(eta)
                estimate = ratio_estimate(
                    heldout_distances < eta, denominator, heldout.chain_ids
                )
                fixed_rows.append(
                    {
                        "orientation": orientation,
                        "n": n,
                        "eta": eta,
                        "eta0": eta0,
                        "pilot_scale": scale,
                        "pilot_scale_quantile": quantile,
                        "pilot_scale_floored": floored,
                        "pilot_tube_count": scale_count,
                        **estimate,
                    }
                )

            if not math.isfinite(scale):
                continue
            for s in mode["rescaled_s"]:
                s = float(s)
                eta = s * scale
                if eta >= eta0:
                    continue
                estimate = ratio_estimate(
                    heldout_distances < eta, denominator, heldout.chain_ids
                )
                collapsed_rows.append(
                    {
                        "orientation": orientation,
                        "n": n,
                        "s": s,
                        "eta": eta,
                        "eta0": eta0,
                        "pilot_scale": scale,
                        "pilot_scale_quantile": quantile,
                        "pilot_scale_floored": floored,
                        "pilot_tube_count": scale_count,
                        **estimate,
                    }
                )
    return fixed_rows, collapsed_rows


def _exact_rows(
    config: dict[str, Any],
    loaded: dict[int, dict[str, Any]],
    model: FrozenClumpModel,
) -> list[dict[str, Any]]:
    requested = {int(n) for n in config.get("exact_n", [])}
    if not requested:
        return []
    unknown = requested - set(loaded)
    if unknown:
        raise ValueError(f"exact_n includes unconfigured scales: {sorted(unknown)}")
    delta = float(config.get("balance_tolerance", 0.0))
    rows: list[dict[str, Any]] = []
    for n in sorted(requested):
        grid = loaded[n]["grid"]
        events = {}
        for clump in model.clumps:
            def event(plan: np.ndarray, selected: FrozenClump = clump) -> bool:
                distances = model.distance_matrix(plan[None, :], grid)
                return bool(model.membership(selected, distances)[0])

            events[clump_name(clump)] = event
        exact = exact_restricted_sums(grid, delta, events)
        for name in events:
            rows.append(
                {
                    "clump": name,
                    "n": n,
                    "valid_plan_count": exact["plan_count"],
                    "partition_function": str(exact["partition_function"]),
                    "restricted_partition_function": str(
                        exact["restricted_partition_functions"][name]
                    ),
                    "exact_mass": exact["masses"][name],
                }
            )
    return rows


def _add_exact_comparison(
    fixed_rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]]
) -> None:
    exact = {(row["clump"], row["n"]): row for row in exact_rows}
    for row in fixed_rows:
        match = exact.get((row["clump"], row["n"]))
        if match is None:
            row["exact_mass"] = float("nan")
            row["heldout_minus_exact"] = float("nan")
            row["heldout_exact_z"] = float("nan")
            continue
        difference = float(row["probability"]) - float(match["exact_mass"])
        se = float(row["standard_error"])
        row["exact_mass"] = match["exact_mass"]
        row["heldout_minus_exact"] = difference
        row["heldout_exact_z"] = difference / se if se > 0 else float("nan")


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(output / name, index=False)


def _plot_fixed_rates(output: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for name in sorted({str(row["clump"]) for row in rows}):
        selected = sorted(
            (row for row in rows if row["clump"] == name), key=lambda row: row["n"]
        )
        positive = [row for row in selected if float(row["probability"]) > 0]
        if not positive:
            continue
        x = np.asarray([row["n"] for row in positive])
        y = np.log(np.asarray([row["probability"] for row in positive]))
        error = np.asarray(
            [row["standard_error"] / row["probability"] for row in positive]
        )
        ax.errorbar(x, y, yerr=1.96 * error, marker="o", capsize=2, label=name)
    ax.set_xlabel("surface scale n")
    ax.set_ylabel("log held-out clump mass")
    ax.set_title("Fixed-clump surface-rate diagnostic")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output / "fixed_clump_rates.png", dpi=180)
    plt.close(fig)


def _plot_mode_collapse(output: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, orientation in zip(axes, ("vertical", "horizontal")):
        for n in sorted({int(row["n"]) for row in rows}):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["orientation"] == orientation and int(row["n"]) == n
                ),
                key=lambda row: row["s"],
            )
            if not selected:
                continue
            ax.errorbar(
                [row["s"] for row in selected],
                [row["ratio"] for row in selected],
                yerr=[1.96 * row["standard_error"] for row in selected],
                marker="o",
                capsize=2,
                label=f"n={n}",
            )
        ax.set_title(f"{orientation} mode")
        ax.set_xlabel(r"rescaled radius $s=\eta/r_{j,n}$")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(r"$H_{j,n}(s r_{j,n})$")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "mode_tube_collapse.png", dpi=180)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def run(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    strict_target = bool(config.get("require_tree_count_target", True))

    loaded: dict[int, dict[str, Any]] = {}
    manifest_runs: list[dict[str, Any]] = []
    for run_config in config["runs"]:
        n = int(run_config["n"])
        grid = GridSpec.from_json(_resolve(run_config["graph"]))
        if grid.surface_scale != n:
            raise ValueError(
                f"run n={n}, but {grid.graph_path} has surface scale {grid.surface_scale}"
            )
        pilot = _load_run_samples(run_config, "pilot", grid, strict_target)
        heldout = _load_run_samples(run_config, "heldout", grid, strict_target)
        pilot_header_seeds = {
            int(metadata["rng_seed"])
            for metadata in pilot.chain_metadata
            if "rng_seed" in metadata
        }
        heldout_header_seeds = {
            int(metadata["rng_seed"])
            for metadata in heldout.chain_metadata
            if "rng_seed" in metadata
        }
        if pilot_header_seeds & heldout_header_seeds:
            raise ValueError(
                f"n={n} pilot and held-out atlas headers reuse RNG seeds: "
                f"{sorted(pilot_header_seeds & heldout_header_seeds)}"
            )
        loaded[n] = {"grid": grid, "pilot": pilot, "heldout": heldout}
        manifest_runs.append(
            {
                "n": n,
                "graph": str(grid.graph_path),
                "pilot_files": [str(path) for path in pilot.source_files],
                "pilot_samples": len(pilot.plans),
                "pilot_rng_seeds": sorted(pilot_header_seeds),
                "heldout_files": [str(path) for path in heldout.source_files],
                "heldout_samples": len(heldout.plans),
                "heldout_chains": int(len(np.unique(heldout.chain_ids))),
                "heldout_rng_seeds": sorted(heldout_header_seeds),
            }
        )

    model = _learn_model(config, loaded)
    _serialize_model(model, output, int(config["pilot_reference_n"]))
    fixed_rows = _fixed_clump_rows(model, loaded)
    exact_rows = _exact_rows(config, loaded, model)
    _add_exact_comparison(fixed_rows, exact_rows)
    rate_rows = _surface_rate_rows(fixed_rows)
    mode_rows, collapsed_rows = _mode_rows(config, loaded)

    _write_csv(output, "fixed_clump_estimates.csv", fixed_rows)
    _write_csv(output, "surface_rate_fits.csv", rate_rows)
    _write_csv(output, "exact_restricted_sums.csv", exact_rows)
    _write_csv(output, "mode_tube_curves.csv", mode_rows)
    _write_csv(output, "mode_tube_collapse.csv", collapsed_rows)
    _plot_fixed_rates(output, fixed_rows)
    _plot_mode_collapse(output, collapsed_rows)

    manifest = {
        "experiment": config.get("experiment_name", "fixed_clump_l1"),
        "config": str(config_path.resolve()),
        "pilot_evaluation_policy": (
            "clumps and r_jn learned from pilot chains; all reported masses and "
            "H-curves evaluated on disjoint held-out chains"
        ),
        "balance_tolerance": float(config.get("balance_tolerance", 0.0)),
        "tree_count_target_required": strict_target,
        "reference_n": int(config["pilot_reference_n"]),
        "frozen_clumps": len(model.clumps),
        "runs": manifest_runs,
        "outputs": [
            "frozen_clumps.json",
            "frozen_clump_centers.npz",
            "fixed_clump_estimates.csv",
            "surface_rate_fits.csv",
            "exact_restricted_sums.csv",
            "mode_tube_curves.csv",
            "mode_tube_collapse.csv",
            "fixed_clump_rates.png",
            "mode_tube_collapse.png",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), _resolve(args.output))


if __name__ == "__main__":
    main()
