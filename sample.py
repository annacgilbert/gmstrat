from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hccfit import HccLinkage
from utils import read_json, str_to_vec, vec_to_str, weighted_l1


@dataclass(frozen=True)
class SampleStoragePaths:
    root: Path

    @property
    def samples(self) -> Path:
        return self.root / "samples.feather"

    @property
    def plans(self) -> Path:
        return self.root / "plans.feather"

    @property
    def districts(self) -> Path:
        return self.root / "districts.feather"

    @property
    def distributions(self) -> Path:
        return self.root / "distributions.feather"

    @property
    def distance(self) -> Path:
        return self.root / "distance_matrix.npy"

    @property
    def linkage(self) -> Path:
        return self.root / "linkage.npy"

    @property
    def pdist_edges(self) -> Path:
        return self.root / "pdist_edges.npy"


def _open_sample_file(filename, mode="rt"):
    fn = Path(filename)
    if ".gz" in fn.suffixes:
        return gzip.open(fn, mode)
    return open(fn, mode)


class SampleProcessor:
    def __init__(self, precinct_fn, save_dir):
        precinct_data = read_json(precinct_fn)
        self.num_districts = int(precinct_data["num_districts"])

        df_precincts = pd.DataFrame(precinct_data["nodes"]).reset_index(drop=True)
        self.df_precincts = df_precincts
        self.num_precincts = len(df_precincts)

        self.population = np.asarray(
            df_precincts["population"].to_numpy(), dtype=np.int32
        )
        self.total_pop = int(self.population.sum(dtype=np.int32))
        precinct_keys = df_precincts["precinct_id_str"].astype(str).to_list()
        self.prec_to_idx = {key: idx for idx, key in enumerate(precinct_keys)}

        self.maximum_distance = int((self.total_pop * 2) // self.num_districts)

        self.paths = SampleStoragePaths(root=Path(save_dir))
        self.paths.root.mkdir(parents=True, exist_ok=True)

        self.all_districts = None

    def clean(self):
        shutil.rmtree(self.paths.root, ignore_errors=True)
        self.paths.root.mkdir(parents=True, exist_ok=True)

    def load_processed(self):
        self.df_plans = pd.read_feather(self.paths.plans)
        self.df_districts = pd.read_feather(self.paths.districts)
        self.df_samples = pd.read_feather(self.paths.samples)
        self.df_distributions = pd.read_feather(self.paths.distributions)
        self.num_districts = len(self.df_plans.plan.iloc[0])
        self.maximum_distance = int((self.total_pop * 2) // self.num_districts)
        try:
            self.pdist_edges = np.load(self.paths.pdist_edges, allow_pickle=False)
        except FileNotFoundError:
            self.pdist_edges = self._compute_pdist_edges()
            np.save(self.paths.pdist_edges, self.pdist_edges, allow_pickle=False)
        self.all_districts = None

    def process_samples(
        self,
        samples_fns,
        max_length,
        min_step: int = 0,
        *,
        pdist_edge_samples: int = 100,
        pdist_edge_bins: int = 50,
    ):
        df_samples = self._read_samples(samples_fns, max_length, min_step)
        df_samples_expanded = self._expand_samples(df_samples)
        df_districts = self._build_district_catalog(df_samples_expanded)
        df_samples_expanded = self._attach_district_ids(
            df_samples_expanded, df_districts
        )
        df_plans = self._summarize_plans(df_samples_expanded)
        df_distributions = self._calculate_distributions(df_plans)

        self.df_samples = df_samples
        self.df_districts = df_districts
        self.df_plans = df_plans
        self.df_distributions = df_distributions

        self._write_feather(df_samples, self.paths.samples)
        self._write_feather(df_districts, self.paths.districts)
        self._write_feather(df_plans, self.paths.plans)
        self._write_feather(df_distributions, self.paths.distributions)
        self.pdist_edges = self._compute_pdist_edges(
            n_samples=pdist_edge_samples,
            n_bins=pdist_edge_bins,
        )
        np.save(self.paths.pdist_edges, self.pdist_edges, allow_pickle=False)
        self.all_districts = None

    def _read_samples(
        self, samples_fns: Iterable[Path | str], max_length: int, min_step: int
    ):
        records = []
        for fn in samples_fns:
            path = Path(fn)
            print(f"Reading {path}")
            sample_tag = path.stem.split(".")[0]
            file_count = 0
            with _open_sample_file(path) as handle:
                with tqdm(
                    total=max_length, desc=f"{sample_tag}", unit="samples"
                ) as pbar:
                    for line_idx, line in enumerate(handle):
                        if file_count >= max_length:
                            break
                        if line_idx == 2:
                            header = json.loads(line)
                            if isinstance(header, dict) and "districts" in header:
                                self.num_districts = int(header["districts"])
                                self.maximum_distance = int(
                                    (self.total_pop * 2) // self.num_districts
                                )
                        if line_idx < 3:
                            continue
                        data = json.loads(line)
                        step = int(str(data["name"]).removeprefix("step"))
                        if step <= min_step:
                            continue
                        plan_vector = self._parse_plan_vector(data["districting"]) - 1
                        records.append(
                            {
                                "step": step,
                                "plan_vector": plan_vector,
                                "sample_tag": sample_tag,
                            }
                        )
                        file_count += 1
                        pbar.update(1)
        return pd.DataFrame.from_records(records)

    def _parse_plan_vector(self, districting: Sequence[dict]) -> np.ndarray:
        plan_vector = np.zeros(self.num_precincts, dtype=np.int32)
        for assignment in districting:
            key = next(iter(assignment.keys()))
            precinct = key
            if isinstance(key, str) and key.startswith("[") and key.endswith("]"):
                try:
                    precinct = json.loads(key)[0]
                except Exception:
                    precinct = key
            plan_vector[self.prec_to_idx[str(precinct)]] = int(assignment[key])
        return plan_vector

    def _expand_samples(self, df_samples: pd.DataFrame) -> pd.DataFrame:
        expanded_rows = []
        for plan_id, row in tqdm(
            df_samples.iterrows(), total=len(df_samples), desc="plans"
        ):
            pvec = row.plan_vector
            for district_id in range(self.num_districts):
                dvec = np.where(pvec == district_id)[0]
                expanded_rows.append(
                    {
                        "step": row.step,
                        "sample_tag": row.sample_tag,
                        "plan_id": plan_id,
                        "district_id": district_id,
                        "district_vector": dvec,
                    }
                )
        return pd.DataFrame.from_records(expanded_rows)

    def _build_district_catalog(
        self, df_samples_expanded: pd.DataFrame
    ) -> pd.DataFrame:
        df_samples_expanded["district_str"] = df_samples_expanded.district_vector.apply(
            vec_to_str
        )
        df_districts = df_samples_expanded.drop_duplicates(subset=["district_str"])[
            ["district_str"]
        ].reset_index(drop=True)
        df_districts["district_uid"] = df_districts.index
        return df_districts

    def _attach_district_ids(
        self, df_samples_expanded: pd.DataFrame, df_districts: pd.DataFrame
    ):
        district_map = dict(zip(df_districts.district_str, df_districts.district_uid))
        df_samples_expanded = df_samples_expanded.copy()
        df_samples_expanded["district_uid"] = df_samples_expanded.district_str.map(
            district_map
        )
        return df_samples_expanded

    def _summarize_plans(self, df_samples_expanded: pd.DataFrame) -> pd.DataFrame:
        df_plans = (
            df_samples_expanded.groupby(["step", "plan_id", "sample_tag"])[
                "district_uid"
            ]
            .apply(list)
            .reset_index(name="plan")
        )
        df_plans["plan_str"] = df_plans.plan.apply(
            lambda x: ".".join([str(i) for i in sorted(x)])
        )
        return df_plans

    def _calculate_distributions(self, df_plans: pd.DataFrame) -> pd.DataFrame:
        df_distributions = (
            df_plans.groupby(["sample_tag", "plan_str"])
            .size()
            .reset_index(name="count")
        )
        df_distributions["freq"] = df_distributions.groupby("sample_tag")[
            "count"
        ].transform(lambda x: x / x.sum())
        df_distributions["plan_vector"] = df_distributions.plan_str.apply(str_to_vec)
        return df_distributions

    def _write_feather(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(path)

    def _compute_pdist_edges(
        self, n_samples: int = 500, n_bins: int = 400
    ) -> np.ndarray:
        df = self.df_districts
        n = min(int(n_samples), int(len(df)))
        sampled = df.sample(n=n, random_state=0)
        districts = sampled["district_str"].apply(str_to_vec).to_list()

        dist_list = []
        for i in range(n):
            for j in range(i):
                d = int(self.compute_distance(districts[i], districts[j]))
                if d < self.maximum_distance:
                    dist_list.append(d)

        right_edge = self.maximum_distance + 1
        _, edges = pd.qcut(
            dist_list + [-1, right_edge],
            q=int(n_bins),
            retbins=True,
            duplicates="drop",
        )
        return np.unique(np.asarray(edges, dtype=np.int32))

    def compute_distance(self, x, y):
        return weighted_l1(x, y, self.population, self.maximum_distance, sparse=True)

    def compute_distance_matrix(self, districts):
        num_districts = len(districts)
        distance_matrix = np.zeros((num_districts, num_districts), dtype=np.int32)
        total = num_districts * (num_districts - 1) // 2
        pbar = tqdm(total=total)
        for i in range(num_districts):
            for j in range(i):
                d = int(self.compute_distance(districts[i], districts[j]))
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
                pbar.update(1)
        return distance_matrix

    def load_distance_matrix(self):
        if not self.paths.distance.is_file():
            distance_matrix = self.compute_distance_matrix(self.get_all_districts())
            np.save(self.paths.distance, distance_matrix)
        return np.load(self.paths.distance, allow_pickle=False)

    def ensure_linkage(self) -> None:
        if self.paths.linkage.is_file():
            return
        distance_matrix = self.load_distance_matrix()
        hcc = HccLinkage(distance_matrix)
        hcc.learn_UM()
        np.save(self.paths.linkage, hcc.Z.astype(np.int32))

    def get_all_districts(self):
        if self.all_districts is None:
            self.all_districts = self.df_districts.district_str.apply(
                str_to_vec
            ).to_list()
        return self.all_districts

    def encode_district(self, district):
        return (
            self.df_districts[self.df_districts.district_str == vec_to_str(district)]
            .iloc[0]
            .district_uid
        )

    def decode_district(self, district_uid):
        return self.get_all_districts()[district_uid]

    def decode_district_vector(self, district):
        district_vector = np.zeros(self.num_precincts, dtype=np.int32)
        district_vector[district] += 1
        return district_vector

    def decode_plan_vector(self, plan):
        all_districts = self.get_all_districts()
        plan_vector = np.zeros(self.num_precincts, dtype=np.int32)
        for district_id, district_uid in enumerate(plan):
            for precinct in all_districts[district_uid]:
                plan_vector[precinct] = district_id
        return plan_vector
