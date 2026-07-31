from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fixed_clump import (
    GridSpec,
    axis_mode_distance,
    axis_mode_separation,
    exact_restricted_sums,
    learn_frozen_clumps,
    plan_d1,
    probability_estimate,
    spanning_tree_count,
)
from experiments.run_fixed_clump import run


def write_grid(path: Path, size: int = 2, districts: int = 2) -> None:
    nodes = []
    adjacency = []
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            nodes.append(
                {
                    "id": idx,
                    "precinct_id": idx,
                    "precinct_id_str": f"({row},{col})",
                    "x_location": row,
                    "y_location": col,
                    "area": 1,
                    "population": 1,
                }
            )
            neighbors = []
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                new_row, new_col = row + drow, col + dcol
                if 0 <= new_row < size and 0 <= new_col < size:
                    neighbors.append({"id": new_row * size + new_col, "length": 1})
            adjacency.append(neighbors)
    path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "num_districts": districts,
                "nodes": nodes,
                "adjacency": adjacency,
            }
        )
    )


def write_atlas(path: Path, plans: list[np.ndarray], seed: int) -> None:
    size = int(round(len(plans[0]) ** 0.5))
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps("atlas") + "\n")
        handle.write(json.dumps({}) + "\n")
        handle.write(
            json.dumps(
                {
                    "districts": 2,
                    "energy weights": [],
                    "gamma": 0.0,
                    "iso_weight": 0.0,
                    "rng_seed": seed,
                }
            )
            + "\n"
        )
        for step, plan in enumerate(plans, start=1):
            districting = [
                {f"({idx // size},{idx % size})": int(label) + 1}
                for idx, label in enumerate(plan)
            ]
            handle.write(
                json.dumps(
                    {
                        "name": f"step{step * 100}",
                        "districting": districting,
                    }
                )
                + "\n"
            )


class FixedClumpCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph2 = self.root / "grid2.json"
        self.graph4 = self.root / "grid4.json"
        write_grid(self.graph2, 2)
        write_grid(self.graph4, 4)
        self.grid2 = GridSpec.from_json(self.graph2)
        self.grid4 = GridSpec.from_json(self.graph4)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def vertical(size: int) -> np.ndarray:
        return np.asarray(
            [int(col >= size // 2) for row in range(size) for col in range(size)],
            dtype=np.int16,
        )

    @staticmethod
    def horizontal(size: int) -> np.ndarray:
        return np.asarray(
            [int(row >= size // 2) for row in range(size) for col in range(size)],
            dtype=np.int16,
        )

    def test_d1_is_unlabeled_normalized_and_cross_resolution(self) -> None:
        vertical2 = self.vertical(2)
        horizontal2 = self.horizontal(2)
        self.assertAlmostEqual(
            plan_d1(vertical2, self.grid2, 1 - vertical2, self.grid2), 0.0
        )
        self.assertAlmostEqual(
            plan_d1(vertical2, self.grid2, horizontal2, self.grid2), 0.5
        )
        self.assertAlmostEqual(
            plan_d1(vertical2, self.grid2, self.vertical(4), self.grid4), 0.0
        )

    def test_pilot_components_freeze_open_ball_unions(self) -> None:
        vertical = self.vertical(2)
        horizontal = self.horizontal(2)
        model = learn_frozen_clumps(
            np.stack([vertical, 1 - vertical, horizontal]),
            self.grid2,
            [0.4],
            min_component_observations=1,
        )
        self.assertEqual(len(model.centers), 2)
        self.assertEqual(len(model.clumps), 2)
        distances = model.distance_matrix(np.stack([vertical, horizontal]), self.grid2)
        memberships = [model.membership(clump, distances) for clump in model.clumps]
        self.assertTrue(any(np.array_equal(value, [True, False]) for value in memberships))
        self.assertTrue(any(np.array_equal(value, [False, True]) for value in memberships))

    def test_whole_axis_mode_distance(self) -> None:
        vertical = self.vertical(4)
        horizontal = self.horizontal(4)
        self.assertAlmostEqual(axis_mode_distance(vertical, self.grid4, "vertical", 0), 0)
        self.assertAlmostEqual(
            axis_mode_distance(horizontal, self.grid4, "vertical", 0), 0.5
        )
        quarter_cut = np.asarray(
            [int(col >= 1) for row in range(4) for col in range(4)], dtype=np.int16
        )
        self.assertAlmostEqual(
            axis_mode_distance(quarter_cut, self.grid4, "vertical", 0.4), 0.05
        )
        self.assertAlmostEqual(axis_mode_separation(0.0), 0.5)
        self.assertAlmostEqual(axis_mode_separation(0.4), 0.42)

    def test_exact_tree_count_restricted_sum(self) -> None:
        vertical = self.vertical(2)
        exact = exact_restricted_sums(
            self.grid2,
            0.0,
            {
                "vertical": lambda plan: plan_d1(
                    plan, self.grid2, vertical, self.grid2
                )
                < 0.1
            },
        )
        self.assertEqual(exact["plan_count"], 2)
        self.assertEqual(exact["partition_function"], 2)
        self.assertEqual(exact["restricted_partition_functions"]["vertical"], 1)
        self.assertAlmostEqual(exact["masses"]["vertical"], 0.5)
        self.assertEqual(
            spanning_tree_count(frozenset(range(4)), self.grid2.adjacency), 4
        )

    def test_error_uses_between_chain_dispersion(self) -> None:
        estimate = probability_estimate(
            np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 1])
        )
        self.assertAlmostEqual(estimate["probability"], 0.5)
        self.assertAlmostEqual(estimate["standard_error"], 0.5)


class FixedClumpEndToEndTest(unittest.TestCase):
    def test_driver_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
            graph = root / "grid2.json"
            write_grid(graph, 2)
            vertical = np.asarray([0, 1, 0, 1], dtype=np.int16)
            horizontal = np.asarray([0, 0, 1, 1], dtype=np.int16)
            pilot = root / "pilot.jsonl.gz"
            heldout1 = root / "heldout1.jsonl.gz"
            heldout2 = root / "heldout2.jsonl.gz"
            write_atlas(pilot, [vertical, horizontal] * 2, 1)
            write_atlas(heldout1, [vertical, horizontal] * 2, 2)
            write_atlas(heldout2, [horizontal, vertical] * 2, 3)

            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "pilot_reference_n": 2,
                        "balance_tolerance": 0.0,
                        "etas": [0.4],
                        "min_component_observations": 1,
                        "exact_n": [2],
                        "mode_tubes": {"enabled": False},
                        "runs": [
                            {
                                "n": 2,
                                "graph": str(graph),
                                "pilot_files": [{"path": str(pilot), "seed": 1}],
                                "heldout_files": [
                                    {"path": str(heldout1), "seed": 2},
                                    {"path": str(heldout2), "seed": 3},
                                ],
                            }
                        ],
                    }
                )
            )
            output = root / "results"
            run(config, output)

            manifest = json.loads((output / "manifest.json").read_text())
            estimates = pd.read_csv(output / "fixed_clump_estimates.csv")
            exact = pd.read_csv(output / "exact_restricted_sums.csv")
            self.assertEqual(manifest["frozen_clumps"], 2)
            self.assertEqual(len(estimates), 2)
            self.assertTrue(np.allclose(estimates["probability"], 0.5))
            self.assertTrue(np.allclose(exact["exact_mass"], 0.5))
            self.assertTrue((output / "fixed_clump_rates.png").is_file())


if __name__ == "__main__":
    unittest.main()
