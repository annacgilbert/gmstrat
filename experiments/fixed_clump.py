"""Core machinery for the fixed-clump L1-neighborhood experiment.

The objects in this module deliberately use the continuum normalization from
Section 2 of the conjectures paper.  In particular, distances are *uncapped*
and plans are unlabeled.  The existing letter pipeline's capped district
metric is therefore not used here.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GridSpec:
    """A regular rectangular grid embedded cellwise in the unit square."""

    graph_path: Path
    num_districts: int
    population: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]
    precinct_keys: tuple[str, ...]
    node_grid: np.ndarray

    @classmethod
    def from_json(cls, filename: str | Path) -> "GridSpec":
        path = Path(filename).resolve()
        with path.open() as handle:
            raw = json.load(handle)

        nodes = raw["nodes"]
        row_values = sorted({node["x_location"] for node in nodes})
        col_values = sorted({node["y_location"] for node in nodes})
        row_index = {value: idx for idx, value in enumerate(row_values)}
        col_index = {value: idx for idx, value in enumerate(col_values)}
        node_grid = np.full((len(row_values), len(col_values)), -1, dtype=np.int32)
        for node_idx, node in enumerate(nodes):
            row = row_index[node["x_location"]]
            col = col_index[node["y_location"]]
            if node_grid[row, col] >= 0:
                raise ValueError(f"duplicate grid location in {path}: {(row, col)}")
            node_grid[row, col] = node_idx
        if np.any(node_grid < 0):
            raise ValueError(f"{path} is not a complete regular rectangular grid")

        areas = np.asarray([float(node.get("area", 1.0)) for node in nodes])
        if not np.allclose(areas, areas[0]):
            raise ValueError(
                "fixed_clump currently requires equal-area grid cells so that "
                "the cellwise embedding has the paper's continuum normalization"
            )

        adjacency = tuple(
            tuple(int(neighbor["id"]) for neighbor in neighbors)
            for neighbors in raw["adjacency"]
        )
        return cls(
            graph_path=path,
            num_districts=int(raw["num_districts"]),
            population=np.asarray(
                [int(node["population"]) for node in nodes], dtype=np.int64
            ),
            adjacency=adjacency,
            precinct_keys=tuple(str(node["precinct_id_str"]) for node in nodes),
            node_grid=node_grid,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.node_grid.shape)

    @property
    def num_cells(self) -> int:
        return int(self.node_grid.size)

    @property
    def surface_scale(self) -> int:
        rows, cols = self.shape
        if rows != cols:
            raise ValueError(
                "the current experiment uses square refinements, so the surface "
                f"scale is ambiguous for shape {self.shape}"
            )
        return rows

    def as_matrix(self, plan: np.ndarray) -> np.ndarray:
        plan_arr = np.asarray(plan, dtype=np.int16)
        if plan_arr.shape != (self.num_cells,):
            raise ValueError(
                f"plan has shape {plan_arr.shape}; expected {(self.num_cells,)}"
            )
        return plan_arr[self.node_grid]


@dataclass(frozen=True)
class PlanSamples:
    plans: np.ndarray
    steps: np.ndarray
    chain_ids: np.ndarray
    source_files: tuple[Path, ...]
    chain_metadata: tuple[dict, ...]


def canonicalize_plan(plan: np.ndarray) -> np.ndarray:
    """Relabel districts by first occurrence, preserving the unlabeled plan."""

    plan = np.asarray(plan)
    relabel: dict[int, int] = {}
    canonical = np.empty(plan.shape, dtype=np.int16)
    for idx, raw_label in enumerate(plan.tolist()):
        label = int(raw_label)
        if label not in relabel:
            relabel[label] = len(relabel)
        canonical[idx] = relabel[label]
    return canonical


def _open_atlas(path: Path):
    if ".gz" in path.suffixes:
        return gzip.open(path, "rt")
    return path.open()


def _atlas_is_tree_count_target(header: dict) -> bool:
    """Return whether a CycleWalk atlas has the tree-count plan marginal.

    CycleWalk samples forests.  With no additional energy (gamma=0 and
    iso_weight=0), forgetting the trees gives plan mass proportional to the
    product of the district spanning-tree counts.  Any nonzero energy changes
    that target.
    """

    weights = header.get("energy weights", [])
    if any(not math.isclose(float(weight), 0.0) for weight in weights):
        return False
    for key in ("gamma", "iso_weight"):
        if key in header and not math.isclose(float(header[key]), 0.0):
            return False
    return True


def read_cyclewalk_samples(
    files: Sequence[str | Path],
    grid: GridSpec,
    *,
    min_step: int = 0,
    thin: int = 1,
    max_samples_per_file: int | None = None,
    require_tree_count_target: bool = True,
) -> PlanSamples:
    """Read CycleWalk atlases while retaining chain identities for errors."""

    if thin < 1:
        raise ValueError("thin must be at least 1")
    key_to_idx = {key: idx for idx, key in enumerate(grid.precinct_keys)}
    plans: list[np.ndarray] = []
    steps: list[int] = []
    chain_ids: list[int] = []
    chain_metadata: list[dict] = []
    resolved = tuple(Path(file).resolve() for file in files)

    for chain_id, path in enumerate(resolved):
        kept_from_file = 0
        eligible = 0
        with _open_atlas(path) as handle:
            for line_idx, line in enumerate(handle):
                record = json.loads(line)
                if line_idx == 2:
                    if int(record.get("districts", -1)) != grid.num_districts:
                        raise ValueError(
                            f"{path} has {record.get('districts')} districts, "
                            f"but {grid.graph_path} has {grid.num_districts}"
                        )
                    if require_tree_count_target and not _atlas_is_tree_count_target(
                        record
                    ):
                        raise ValueError(
                            f"{path} has nonzero CycleWalk energy weights and is not "
                            "a tree-count-plan sample"
                        )
                    chain_metadata.append(record)
                    continue
                if line_idx < 3:
                    continue

                step = int(str(record["name"]).removeprefix("step"))
                if step <= min_step:
                    continue
                if eligible % thin:
                    eligible += 1
                    continue
                eligible += 1

                assignment = np.full(grid.num_cells, -1, dtype=np.int16)
                for item in record["districting"]:
                    raw_key = next(iter(item))
                    key = raw_key
                    if raw_key.startswith("[") and raw_key.endswith("]"):
                        try:
                            key = str(json.loads(raw_key)[0])
                        except (json.JSONDecodeError, IndexError, TypeError):
                            pass
                    assignment[key_to_idx[str(key)]] = int(item[raw_key])
                if np.any(assignment < 0):
                    raise ValueError(f"incomplete districting at step {step} in {path}")

                labels = np.unique(assignment)
                if len(labels) != grid.num_districts:
                    raise ValueError(
                        f"step {step} in {path} uses {len(labels)} district labels"
                    )
                assignment = canonicalize_plan(assignment)
                plans.append(assignment)
                steps.append(step)
                chain_ids.append(chain_id)
                kept_from_file += 1
                if (
                    max_samples_per_file is not None
                    and kept_from_file >= max_samples_per_file
                ):
                    break

    if not plans:
        raise ValueError("no samples remain after burn-in/thinning")
    return PlanSamples(
        plans=np.stack(plans),
        steps=np.asarray(steps, dtype=np.int64),
        chain_ids=np.asarray(chain_ids, dtype=np.int32),
        source_files=resolved,
        chain_metadata=tuple(chain_metadata),
    )


@lru_cache(maxsize=None)
def _uniform_overlap(size_a: int, size_b: int) -> np.ndarray:
    """Exact 1D overlap lengths between two uniform partitions of [0,1]."""

    left_a = np.arange(size_a, dtype=float) / size_a
    right_a = np.arange(1, size_a + 1, dtype=float) / size_a
    left_b = np.arange(size_b, dtype=float) / size_b
    right_b = np.arange(1, size_b + 1, dtype=float) / size_b
    return np.maximum(
        0.0,
        np.minimum(right_a[:, None], right_b[None, :])
        - np.maximum(left_a[:, None], left_b[None, :]),
    )


def plan_d1(
    plan_a: np.ndarray,
    grid_a: GridSpec,
    plan_b: np.ndarray,
    grid_b: GridSpec,
) -> float:
    """Normalized unlabeled continuum L1 distance between cellwise plans."""

    if grid_a.num_districts != grid_b.num_districts:
        raise ValueError("plans with different district counts cannot be compared")
    k = grid_a.num_districts
    matrix_a = grid_a.as_matrix(plan_a)
    matrix_b = grid_b.as_matrix(plan_b)
    row_overlap = _uniform_overlap(matrix_a.shape[0], matrix_b.shape[0])
    col_overlap = _uniform_overlap(matrix_a.shape[1], matrix_b.shape[1])
    contingency = np.zeros((k, k), dtype=float)

    row_pairs = np.argwhere(row_overlap > 0)
    col_pairs = np.argwhere(col_overlap > 0)
    for row_a, row_b in row_pairs:
        row_weight = row_overlap[row_a, row_b]
        for col_a, col_b in col_pairs:
            contingency[
                matrix_a[row_a, col_a], matrix_b[row_b, col_b]
            ] += row_weight * col_overlap[col_a, col_b]

    rows, cols = linear_sum_assignment(-contingency)
    matched_area = float(contingency[rows, cols].sum())
    distance = 1.0 - matched_area
    return float(np.clip(distance, 0.0, 1.0))


def cross_d1_matrix(
    plans_a: np.ndarray,
    grid_a: GridSpec,
    plans_b: np.ndarray,
    grid_b: GridSpec,
) -> np.ndarray:
    """All cross distances, with an exact vectorized k=2 implementation."""

    plans_a = np.asarray(plans_a)
    plans_b = np.asarray(plans_b)
    if grid_a.num_districts != grid_b.num_districts:
        raise ValueError("plans with different district counts cannot be compared")
    if grid_a.num_districts != 2:
        return np.asarray(
            [
                [plan_d1(left, grid_a, right, grid_b) for right in plans_b]
                for left in plans_a
            ],
            dtype=float,
        )

    common_rows = math.lcm(grid_a.shape[0], grid_b.shape[0])
    common_cols = math.lcm(grid_a.shape[1], grid_b.shape[1])
    common_cells = common_rows * common_cols

    def packed_refinement(plans: np.ndarray, grid: GridSpec) -> np.ndarray:
        matrices = plans[:, grid.node_grid.ravel()].reshape(
            len(plans), grid.shape[0], grid.shape[1]
        )
        refined = np.repeat(matrices, common_rows // grid.shape[0], axis=1)
        refined = np.repeat(refined, common_cols // grid.shape[1], axis=2)
        indicators = (refined.reshape(len(plans), common_cells) == 0)
        padding = (-common_cells) % 64
        if padding:
            indicators = np.pad(indicators, ((0, 0), (0, padding)))
        packed_bytes = np.packbits(indicators, axis=1, bitorder="little")
        return np.ascontiguousarray(packed_bytes).view(np.uint64)

    packed_a = packed_refinement(plans_a, grid_a)
    packed_b = packed_refinement(plans_b, grid_b)
    distances = np.empty((len(plans_a), len(plans_b)), dtype=float)
    words = packed_a.shape[1]
    # Limit the temporary broadcasted XOR array to about 16 MB.
    block_size = max(1, 2_000_000 // max(1, len(plans_b) * words))
    bitwise_count = getattr(np, "bitwise_count", None)
    if bitwise_count is None:
        raise RuntimeError("the vectorized k=2 metric requires NumPy >= 2.0")
    for start in range(0, len(plans_a), block_size):
        stop = min(len(plans_a), start + block_size)
        xor = np.bitwise_xor(
            packed_a[start:stop, None, :], packed_b[None, :, :]
        )
        mismatch = bitwise_count(xor).sum(axis=2, dtype=np.int32)
        unlabeled_mismatch = np.minimum(mismatch, common_cells - mismatch)
        distances[start:stop] = unlabeled_mismatch / common_cells
    return distances


def pairwise_d1(plans: np.ndarray, grid: GridSpec) -> np.ndarray:
    return cross_d1_matrix(plans, grid, plans, grid)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


@dataclass(frozen=True)
class FrozenClump:
    eta: float
    radius: float
    component_id: int
    center_indices: np.ndarray
    pilot_observations: int


@dataclass(frozen=True)
class FrozenClumpModel:
    """Finite unions of open balls centered at frozen pilot plans."""

    centers: np.ndarray
    center_grid: GridSpec
    clumps: tuple[FrozenClump, ...]

    def distance_matrix(self, plans: np.ndarray, grid: GridSpec) -> np.ndarray:
        return cross_d1_matrix(plans, grid, self.centers, self.center_grid)

    @staticmethod
    def membership(clump: FrozenClump, distances: np.ndarray) -> np.ndarray:
        # Strict inequality realizes the prescribed union of open balls.
        return np.min(distances[:, clump.center_indices], axis=1) < clump.radius


def learn_frozen_clumps(
    pilot_plans: np.ndarray,
    grid: GridSpec,
    etas: Sequence[float],
    *,
    extension_radius_factor: float = 0.5,
    min_component_observations: int = 2,
    max_components_per_eta: int | None = None,
) -> FrozenClumpModel:
    """Learn eta-neighbor components and freeze disjoint ball-union clumps."""

    if not 0.0 < extension_radius_factor <= 0.5:
        raise ValueError("extension_radius_factor must lie in (0, 0.5]")
    if not etas or any(float(eta) <= 0 for eta in etas):
        raise ValueError("etas must be a nonempty sequence of positive radii")

    canonical_plans = np.stack(
        [canonicalize_plan(plan) for plan in np.asarray(pilot_plans)]
    )
    centers, multiplicities = np.unique(
        canonical_plans,
        axis=0,
        return_counts=True,
    )
    distances = pairwise_d1(centers, grid)
    learned: list[FrozenClump] = []

    for eta in sorted({float(value) for value in etas}):
        union_find = _UnionFind(len(centers))
        left, right = np.where(np.triu(distances < eta, k=1))
        for i, j in zip(left.tolist(), right.tolist()):
            union_find.union(i, j)

        components: dict[int, list[int]] = {}
        for idx in range(len(centers)):
            components.setdefault(union_find.find(idx), []).append(idx)
        ranked = sorted(
            components.values(),
            key=lambda component: (
                -int(multiplicities[component].sum()),
                min(component),
            ),
        )
        eligible = [
            component
            for component in ranked
            if int(multiplicities[component].sum()) >= min_component_observations
        ]
        if max_components_per_eta is not None:
            eligible = eligible[:max_components_per_eta]
        for component_id, component in enumerate(eligible):
            learned.append(
                FrozenClump(
                    eta=eta,
                    radius=eta * extension_radius_factor,
                    component_id=component_id,
                    center_indices=np.asarray(component, dtype=np.int32),
                    pilot_observations=int(multiplicities[component].sum()),
                )
            )

    if not learned:
        raise ValueError("no pilot component met min_component_observations")
    return FrozenClumpModel(
        centers=centers,
        center_grid=grid,
        clumps=tuple(learned),
    )


def _newey_west_variance_of_mean(values: np.ndarray) -> float:
    """Bartlett-HAC variance for a chain mean with an automatic lag window."""

    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float("nan")
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    if gamma0 == 0.0:
        return 0.0
    max_lag = min(n - 1, max(1, int(math.floor(4 * (n / 100) ** (2 / 9)))))
    long_run = gamma0
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1)
        long_run += 2.0 * weight * covariance
    return max(long_run, 0.0) / n


def stratified_mean_and_variance(
    values: np.ndarray, chain_ids: np.ndarray
) -> tuple[float, float]:
    """Mean and HAC variance, combining independent chains by sample count."""

    values = np.asarray(values, dtype=float)
    chain_ids = np.asarray(chain_ids)
    if values.shape != chain_ids.shape:
        raise ValueError("values and chain_ids must have the same shape")
    total = len(values)
    mean = float(values.mean())
    within_variance = 0.0
    chain_means: list[float] = []
    chain_weights: list[float] = []
    for chain_id in np.unique(chain_ids):
        chain = values[chain_ids == chain_id]
        chain_means.append(float(chain.mean()))
        chain_weights.append(len(chain) / total)
        chain_variance = _newey_west_variance_of_mean(chain)
        if math.isnan(chain_variance):
            continue
        within_variance += (len(chain) / total) ** 2 * chain_variance

    # A stuck chain can make the within-chain HAC error spuriously zero.  With
    # multiple independent chains, retain the larger of the within-chain
    # estimate and the empirical between-chain variance of their weighted mean.
    between_variance = 0.0
    if len(chain_means) > 1:
        weights = np.asarray(chain_weights, dtype=float)
        means = np.asarray(chain_means, dtype=float)
        sum_squared_weights = float(np.dot(weights, weights))
        weighted_sample_variance = float(
            np.dot(weights, (means - mean) ** 2) / (1.0 - sum_squared_weights)
        )
        between_variance = weighted_sample_variance * sum_squared_weights
    return mean, max(within_variance, between_variance)


def probability_estimate(
    indicator: np.ndarray, chain_ids: np.ndarray
) -> dict[str, float | int]:
    indicator = np.asarray(indicator, dtype=float)
    probability, variance = stratified_mean_and_variance(indicator, chain_ids)
    standard_error = math.sqrt(max(variance, 0.0))
    n = len(indicator)
    iid_variance = probability * (1.0 - probability)
    if variance > 0 and iid_variance > 0:
        effective_n = min(float(n), iid_variance / variance)
    else:
        effective_n = float(n)

    lower = max(0.0, probability - 1.96 * standard_error)
    upper = min(1.0, probability + 1.96 * standard_error)
    if probability == 0.0:
        upper = 1.0 - 0.05 ** (1.0 / effective_n)
    elif probability == 1.0:
        lower = 0.05 ** (1.0 / effective_n)
    chain_probabilities = np.asarray(
        [indicator[chain_ids == chain_id].mean() for chain_id in np.unique(chain_ids)]
    )
    return {
        "count": int(indicator.sum()),
        "sample_size": n,
        "probability": probability,
        "standard_error": standard_error,
        "effective_sample_size": effective_n,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "chains": len(chain_probabilities),
        "chain_probability_min": float(chain_probabilities.min()),
        "chain_probability_max": float(chain_probabilities.max()),
        "chain_probability_sd": (
            float(chain_probabilities.std(ddof=1))
            if len(chain_probabilities) > 1
            else 0.0
        ),
    }


def ratio_estimate(
    numerator: np.ndarray, denominator: np.ndarray, chain_ids: np.ndarray
) -> dict[str, float | int]:
    """HAC delta-method estimate of E[numerator] / E[denominator]."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    p_num = float(numerator.mean())
    p_den = float(denominator.mean())
    if p_den == 0.0:
        return {
            "numerator_count": int(numerator.sum()),
            "denominator_count": 0,
            "ratio": float("nan"),
            "standard_error": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
        }
    ratio = p_num / p_den
    influence = (numerator - ratio * denominator) / p_den
    _, variance = stratified_mean_and_variance(influence, chain_ids)
    standard_error = math.sqrt(max(variance, 0.0))
    chain_ratios = []
    for chain_id in np.unique(chain_ids):
        selected = chain_ids == chain_id
        chain_denominator = float(denominator[selected].mean())
        if chain_denominator > 0:
            chain_ratios.append(
                float(numerator[selected].mean()) / chain_denominator
            )
    return {
        "numerator_count": int(numerator.sum()),
        "denominator_count": int(denominator.sum()),
        "ratio": ratio,
        "standard_error": standard_error,
        "ci95_lower": max(0.0, ratio - 1.96 * standard_error),
        "ci95_upper": min(1.0, ratio + 1.96 * standard_error),
        "chains_with_denominator_hits": len(chain_ratios),
        "chain_ratio_min": min(chain_ratios) if chain_ratios else float("nan"),
        "chain_ratio_max": max(chain_ratios) if chain_ratios else float("nan"),
    }


def axis_mode_separation(balance_tolerance: float) -> float:
    """Distance between vertical and horizontal k=2 straight-cut modes."""

    delta = float(balance_tolerance)
    if not 0.0 <= delta < 0.5:
        raise ValueError("balance_tolerance must lie in [0, 0.5)")
    return (1.0 - delta**2) / 2.0


def axis_mode_distance(
    plan: np.ndarray,
    grid: GridSpec,
    orientation: str,
    balance_tolerance: float,
) -> float:
    """Distance to the whole vertical or horizontal straight-cut mode."""

    if grid.num_districts != 2:
        raise ValueError("axis mode tubes are defined here only for k=2")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    delta = float(balance_tolerance)
    low = (1.0 - delta) / 2.0
    high = (1.0 + delta) / 2.0
    matrix = grid.as_matrix(plan)
    axis_size = matrix.shape[1] if orientation == "vertical" else matrix.shape[0]
    candidates = [low, high]
    candidates.extend(
        breakpoint
        for breakpoint in np.arange(axis_size + 1, dtype=float) / axis_size
        if low <= breakpoint <= high
    )

    district = matrix == 0
    district_area = float(district.mean())
    best = 1.0
    for cut in candidates:
        if orientation == "vertical":
            full = int(math.floor(cut * matrix.shape[1] + 1e-12))
            fractional = cut * matrix.shape[1] - full
            intersection = float(district[:, :full].sum()) / matrix.size
            if full < matrix.shape[1] and fractional > 1e-12:
                intersection += (
                    fractional * float(district[:, full].sum()) / matrix.size
                )
        else:
            full = int(math.floor(cut * matrix.shape[0] + 1e-12))
            fractional = cut * matrix.shape[0] - full
            intersection = float(district[:full, :].sum()) / matrix.size
            if full < matrix.shape[0] and fractional > 1e-12:
                intersection += (
                    fractional * float(district[full, :].sum()) / matrix.size
                )
        mismatch = district_area + cut - 2.0 * intersection
        best = min(best, mismatch, 1.0 - mismatch)
    return float(np.clip(best, 0.0, 1.0))


def mode_distances(
    plans: np.ndarray,
    grid: GridSpec,
    orientation: str,
    balance_tolerance: float,
) -> np.ndarray:
    return np.asarray(
        [
            axis_mode_distance(plan, grid, orientation, balance_tolerance)
            for plan in plans
        ],
        dtype=float,
    )


def _connected(vertices: frozenset[int], adjacency: Sequence[Sequence[int]]) -> bool:
    if not vertices:
        return False
    start = next(iter(vertices))
    visited = {start}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if neighbor in vertices and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(vertices)


def enumerate_balanced_connected_plans(
    grid: GridSpec, balance_tolerance: float
) -> Iterator[np.ndarray]:
    """Enumerate each unlabeled balanced connected plan exactly once.

    This is intentionally a small-n validator.  The canonical rule that each
    next district contains the smallest remaining vertex removes label
    permutations.
    """

    delta = float(balance_tolerance)
    total_population = int(grid.population.sum())
    ideal = total_population / grid.num_districts
    min_population = (1.0 - delta) * ideal
    max_population = (1.0 + delta) * ideal
    vertices = frozenset(range(grid.num_cells))

    def population(subset: Iterable[int]) -> int:
        indices = np.fromiter(subset, dtype=np.int32)
        return int(grid.population[indices].sum())

    def recurse(
        remaining: frozenset[int], districts: tuple[frozenset[int], ...]
    ) -> Iterator[tuple[frozenset[int], ...]]:
        districts_left = grid.num_districts - len(districts)
        remaining_population = population(remaining)
        if not (
            districts_left * min_population - 1e-9
            <= remaining_population
            <= districts_left * max_population + 1e-9
        ):
            return
        if districts_left == 1:
            if (
                min_population - 1e-9
                <= remaining_population
                <= max_population + 1e-9
                and _connected(remaining, grid.adjacency)
            ):
                yield districts + (remaining,)
            return

        ordered = sorted(remaining)
        first = ordered[0]
        max_size = len(ordered) - (districts_left - 1)
        for size in range(1, max_size + 1):
            for others in combinations(ordered[1:], size - 1):
                district = frozenset((first, *others))
                district_population = population(district)
                if not (
                    min_population - 1e-9
                    <= district_population
                    <= max_population + 1e-9
                ):
                    continue
                if not _connected(district, grid.adjacency):
                    continue
                yield from recurse(remaining - district, districts + (district,))

    for districts in recurse(vertices, ()):
        plan = np.empty(grid.num_cells, dtype=np.int16)
        for label, district in enumerate(districts):
            plan[list(district)] = label
        yield plan


def _bareiss_determinant(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if size == 0:
        return 1
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_idx in range(size - 1):
        if work[pivot_idx][pivot_idx] == 0:
            swap = next(
                (row for row in range(pivot_idx + 1, size) if work[row][pivot_idx]),
                None,
            )
            if swap is None:
                return 0
            work[pivot_idx], work[swap] = work[swap], work[pivot_idx]
            sign *= -1
        pivot = work[pivot_idx][pivot_idx]
        for row in range(pivot_idx + 1, size):
            for col in range(pivot_idx + 1, size):
                numerator = (
                    work[row][col] * pivot
                    - work[row][pivot_idx] * work[pivot_idx][col]
                )
                work[row][col] = numerator // previous
        previous = pivot
        for row in range(pivot_idx + 1, size):
            work[row][pivot_idx] = 0
    return sign * work[-1][-1]


def spanning_tree_count(
    vertices: frozenset[int], adjacency: Sequence[Sequence[int]]
) -> int:
    """Exact matrix-tree count for an induced subgraph."""

    ordered = sorted(vertices)
    if len(ordered) <= 1:
        return 1
    local = {vertex: idx for idx, vertex in enumerate(ordered)}
    laplacian = [[0 for _ in ordered] for _ in ordered]
    for vertex in ordered:
        row = local[vertex]
        for neighbor in adjacency[vertex]:
            if neighbor not in vertices:
                continue
            col = local[neighbor]
            laplacian[row][row] += 1
            laplacian[row][col] -= 1
    cofactor = [row[:-1] for row in laplacian[:-1]]
    return _bareiss_determinant(cofactor)


def exact_restricted_sums(
    grid: GridSpec,
    balance_tolerance: float,
    events: dict[str, Callable[[np.ndarray], bool]],
) -> dict[str, object]:
    """Enumerate Z and restricted tree-count sums for named hard clumps."""

    cache: dict[frozenset[int], int] = {}
    restricted = {name: 0 for name in events}
    total = 0
    plan_count = 0
    for plan in enumerate_balanced_connected_plans(grid, balance_tolerance):
        weight = 1
        for label in range(grid.num_districts):
            district = frozenset(np.flatnonzero(plan == label).tolist())
            if district not in cache:
                cache[district] = spanning_tree_count(district, grid.adjacency)
            weight *= cache[district]
        total += weight
        plan_count += 1
        for name, event in events.items():
            if event(plan):
                restricted[name] += weight
    if total == 0:
        raise ValueError("the exact state space is empty")
    return {
        "plan_count": plan_count,
        "partition_function": total,
        "restricted_partition_functions": restricted,
        "masses": {name: value / total for name, value in restricted.items()},
    }


def fit_surface_rate(rows: Sequence[dict]) -> dict[str, float | int]:
    """Fit log P(A) = intercept - rate*n using held-out estimates."""

    usable = [
        row
        for row in rows
        if float(row["probability"]) > 0.0
        and math.isfinite(float(row["probability"]))
    ]
    if len(usable) < 2:
        return {"points": len(usable), "rate": float("nan")}
    x = np.asarray([float(row["n"]) for row in usable])
    y = np.log(np.asarray([float(row["probability"]) for row in usable]))
    log_errors = np.asarray(
        [
            max(
                float(row["standard_error"]) / float(row["probability"]),
                1e-6,
            )
            for row in usable
        ]
    )
    weights = 1.0 / log_errors**2
    design = np.column_stack([np.ones_like(x), x])
    xtwx = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(xtwx)
    coefficients = covariance @ design.T @ (weights * y)
    fitted = design @ coefficients
    residual = y - fitted
    if len(x) > 2:
        scale = float(np.sum(weights * residual**2) / (len(x) - 2))
        covariance *= scale
    slope_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    total_variation = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / total_variation
        if total_variation > 0
        else 1.0
    )
    return {
        "points": len(usable),
        "rate": -float(coefficients[1]),
        "rate_standard_error": slope_error,
        "intercept": float(coefficients[0]),
        "r_squared": r_squared,
    }
