from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import vec_to_str

WordCandidate = Tuple[float, List[int]]
LetterCandidate = Tuple[float, int]
CentroidNorm = Literal["l1", "l2"]


def merge_words(
    word_arr: Sequence[WordCandidate],
    letter_arr: Sequence[LetterCandidate],
    word_degree: int,
    max_distance: float | None = None,
) -> List[WordCandidate]:
    merged: List[Tuple[float, List[int]]] = []
    for word_idx in range(min(word_degree, len(word_arr))):
        word_dist, word = word_arr[word_idx]
        if max_distance is not None and word_dist > max_distance:
            break
        for letter_idx in range(min(word_degree - word_idx, len(letter_arr))):
            letter_dist, letter = letter_arr[letter_idx]
            new_dist = word_dist + letter_dist
            if max_distance is not None and new_dist > max_distance:
                break
            new_word = word + [letter]
            candidate = (-new_dist, new_word)
            if len(merged) < word_degree:
                heapq.heappush(merged, candidate)
            else:
                heapq.heappushpop(merged, candidate)
    return sorted([(-dist, word) for dist, word in merged], key=lambda x: x[0])


def nearest_words(
    letter_distances: Sequence[Sequence[LetterCandidate]],
    word_degree: int,
    *,
    sort_word: bool = True,
    max_distance: float | None = None,
) -> List[WordCandidate]:
    word_arr: List[WordCandidate] = [(0, [])]
    for letter_arr in letter_distances:
        word_arr = merge_words(
            word_arr,
            letter_arr,
            word_degree,
            max_distance=max_distance,
        )
        if not word_arr:
            break
    if sort_word:
        return [(dist, sorted(word)) for dist, word in word_arr]
    return [(dist, list(word)) for dist, word in word_arr]


@dataclass
class PlanWordResult:
    df_plans: pd.DataFrame
    df_words: pd.DataFrame
    district_cluster_distances: List[List[LetterCandidate]]


class PlanWordBuilder:
    def __init__(
        self,
        sp,
        clusters,
        word_degree: int,
        verbose: bool = True,
        *,
        centroid_norm: CentroidNorm = "l1",
        letter_subset: Sequence[int] | None = None,
        dedupe: bool = True,
        max_distance: float | None = None,
    ):
        self.sp = sp
        self.clusters = clusters
        self.word_degree = word_degree
        self.verbose = verbose
        self.centroid_norm = centroid_norm
        self.letter_subset = (
            None if letter_subset is None else [int(x) for x in letter_subset]
        )
        self.dedupe = bool(dedupe)
        self.max_distance = None if max_distance is None else float(max_distance)

    def _resolve_letter_ids(self, num_letters: int) -> np.ndarray:
        if self.letter_subset is None:
            return np.arange(int(num_letters), dtype=np.int32)
        return np.asarray(list(dict.fromkeys(self.letter_subset)), dtype=np.int32)

    def build(self) -> PlanWordResult:
        df_plans = self._prepare_plans()
        if self.centroid_norm == "l2":
            district_cluster_distances = self._district_to_density_distances()
        else:
            district_cluster_distances = self._district_to_centroid_distances()
        df_words = self._enumerate_words(df_plans, district_cluster_distances)
        return PlanWordResult(
            df_plans=df_plans,
            df_words=df_words,
            district_cluster_distances=district_cluster_distances,
        )

    def _prepare_plans(self) -> pd.DataFrame:
        df_plans = self.sp.df_distributions.copy().reset_index(drop=True)
        df_plans["plan_uid"] = df_plans.index
        df_plans["plan_vector"] = df_plans.plan_vector.apply(
            lambda vec: np.array(vec, dtype=np.int32)
            if not isinstance(vec, np.ndarray)
            else vec.astype(np.int32)
        )
        return df_plans

    def _district_to_density_distances(self) -> List[List[LetterCandidate]]:
        densities = np.asarray(self.clusters.cluster_densities, dtype=float)
        letter_ids = self._resolve_letter_ids(densities.shape[0])
        densities = densities[letter_ids]

        weights = np.asarray(self.sp.population, dtype=float)
        max_dist = (
            None
            if self.sp.maximum_distance is None
            else float(self.sp.maximum_distance)
        )

        total_dw = densities @ weights

        all_districts = self.sp.get_all_districts()
        distances: List[List[LetterCandidate]] = []

        district_iter = all_districts
        if self.verbose:
            district_iter = tqdm(
                all_districts,
                total=len(all_districts),
                desc="Computing letter distances",
            )

        for district_vec in district_iter:
            idx = np.asarray(district_vec, dtype=np.intp)
            if idx.size == 0:
                sum_w = 0.0
                sum_dw = np.zeros(densities.shape[0], dtype=float)
            else:
                w_idx = weights[idx]
                sum_w = float(w_idx.sum())
                sum_dw = densities[:, idx] @ w_idx

            dist_vec = total_dw + sum_w - 2.0 * sum_dw
            dist_vec = np.maximum(dist_vec, 0.0)
            if max_dist is not None:
                dist_vec = np.minimum(dist_vec, max_dist)

            entries: List[LetterCandidate] = [
                (float(dist_vec[j]), int(letter_ids[j]))
                for j in range(dist_vec.shape[0])
            ]
            entries.sort(key=lambda x: x[0])
            distances.append(entries)

        return distances

    def _district_to_centroid_distances(self) -> List[List[LetterCandidate]]:
        centroids = self.clusters.centroids
        letter_ids = self._resolve_letter_ids(len(centroids))
        all_districts = self.sp.get_all_districts()
        distances: List[List[LetterCandidate]] = []

        weights = np.asarray(self.sp.population, dtype=np.int64)
        max_dist = (
            None if self.sp.maximum_distance is None else int(self.sp.maximum_distance)
        )

        num_centroids = int(letter_ids.size)
        centroid_members = np.zeros(
            (num_centroids, self.sp.num_precincts), dtype=np.bool_
        )
        centroid_weight_sum = np.zeros(num_centroids, dtype=np.int64)
        for row_idx, centroid_id in enumerate(letter_ids.tolist()):
            idx = np.asarray(centroids[int(centroid_id)], dtype=np.intp)
            if idx.size:
                centroid_members[row_idx, idx] = True
                centroid_weight_sum[row_idx] = weights[idx].sum(dtype=np.int64)

        district_iter = all_districts
        if self.verbose:
            district_iter = tqdm(
                all_districts,
                total=len(all_districts),
                desc="Computing letter distances",
            )

        for district_vec in district_iter:
            idx = np.asarray(district_vec, dtype=np.intp)
            if idx.size:
                w_idx = weights[idx]
                sum_w = int(w_idx.sum(dtype=np.int64))
                inter = centroid_members[:, idx] @ w_idx
            else:
                sum_w = 0
                inter = np.zeros(num_centroids, dtype=np.int64)

            dist_vec = sum_w + centroid_weight_sum - 2 * inter
            dist_vec = np.maximum(dist_vec, 0)
            if max_dist is not None:
                dist_vec = np.minimum(dist_vec, max_dist)

            entries: List[LetterCandidate] = [
                (float(dist_vec[j]), int(letter_ids[j])) for j in range(num_centroids)
            ]
            entries.sort(key=lambda x: x[0])
            distances.append(entries)

        return distances

    def _enumerate_words(
        self,
        df_plans: pd.DataFrame,
        district_cluster_distances: Sequence[Sequence[LetterCandidate]],
    ) -> pd.DataFrame:
        records: List[dict] = []
        plan_iter = df_plans.itertuples()
        if self.verbose:
            plan_iter = tqdm(
                plan_iter,
                total=len(df_plans),
                desc="Enumerating words",
            )

        for row in plan_iter:
            letters = [
                district_cluster_distances[district_uid]
                for district_uid in row.plan_vector
            ]
            word_candidates = nearest_words(
                letters,
                self.word_degree,
                sort_word=self.dedupe,
                max_distance=self.max_distance,
            )
            if not word_candidates:
                continue
            min_distance = word_candidates[0][0]
            for dist, word in word_candidates:
                records.append(
                    {
                        "plan_uid": row.plan_uid,
                        "plan": row.plan_vector,
                        "word": np.array(word, dtype=np.int32),
                        "distance": dist,
                        "min_distance": min_distance,
                    }
                )

        df_words = pd.DataFrame.from_records(records)
        if df_words.empty:
            return df_words

        df_words["plan_str"] = df_words.plan.apply(vec_to_str)
        df_words["word_str"] = df_words.word.apply(vec_to_str)
        if self.dedupe:
            df_words = df_words.drop_duplicates(
                subset=["plan_str", "word_str"]
            ).reset_index(drop=True)
        else:
            df_words = df_words.reset_index(drop=True)
        df_words["word_uid"] = df_words.word_str.astype("category").cat.codes
        return df_words


class WordStat:
    def __init__(self, plan_word: PlanWordResult, temp: float, verbose: bool = True):
        self.plan_word = plan_word
        self.df_words = plan_word.df_words.copy()
        self.df_plans = plan_word.df_plans.copy()
        self.temp = temp
        self.verbose = verbose

        self._prepare_plan_metadata()
        self._word_index: pd.DataFrame | None = None
        self._weights_computed = False
        self._flux_matrix: np.ndarray | None = None
        self._stationary: np.ndarray | None = None

    def _prepare_plan_metadata(self) -> None:
        if "plan_uid" not in self.df_plans.columns:
            self.df_plans = self.df_plans.reset_index().rename(
                columns={"index": "plan_uid"}
            )
        self.plan_freq = dict(zip(self.df_plans.plan_uid, self.df_plans.freq))

    def compute_weights(self) -> pd.DataFrame:
        if self._weights_computed:
            return self.df_words

        df_words = self.df_words
        df_words["plan_uid"] = df_words.plan_uid.astype(int)
        df_words["plan_freq"] = df_words.plan_uid.map(self.plan_freq)
        distance = df_words.distance.to_numpy(dtype=float)
        min_distance = df_words.min_distance.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = distance / min_distance
        ratio = np.where((min_distance == 0) & (distance == 0), 1.0, ratio)
        ratio = np.where(np.isnan(ratio), 1.0, ratio)
        df_words["phi"] = np.exp(-self.temp * (ratio - 1.0))
        df_words["phi_normalized"] = df_words.groupby("plan_uid")["phi"].transform(
            lambda x: x / x.sum()
        )
        df_words["phi_freq"] = df_words.phi_normalized * df_words.plan_freq
        if self._word_index is None:
            self._word_index = (
                df_words[["word_uid", "word_str", "word"]]
                .drop_duplicates(subset=["word_uid"])
                .sort_values("word_uid")
                .reset_index(drop=True)
            )
        self._weights_computed = True
        return df_words

    def stationary_distribution(self) -> np.ndarray:
        if self._stationary is not None:
            return self._stationary
        df_words = self.compute_weights()
        stationary = (
            df_words.groupby("word_uid")["phi_freq"].sum().sort_index().to_numpy()
        )
        self._stationary = stationary
        return stationary

    def flux_matrix(self) -> np.ndarray:
        if self._flux_matrix is not None:
            return self._flux_matrix

        df_words = self.compute_weights()
        plan_freq = self.plan_freq

        num_words = df_words.word_uid.max() + 1
        flux = np.zeros((num_words, num_words), dtype=float)

        phi_support = df_words.groupby("word_uid")["plan_uid"].apply(set)
        phi_lookup = dict(
            zip(
                zip(df_words.word_uid, df_words.plan_uid),
                df_words.phi_normalized,
            )
        )

        word_iter = range(num_words)
        if self.verbose:
            word_iter = tqdm(word_iter, total=num_words, desc="Computing flux")

        for word1 in word_iter:
            plans1 = phi_support[word1]
            for word2 in range(word1 + 1):
                total = 0.0
                shared = plans1.intersection(phi_support[word2])
                for plan_uid in shared:
                    total += (
                        phi_lookup[(word1, plan_uid)]
                        * phi_lookup[(word2, plan_uid)]
                        * plan_freq[plan_uid]
                    )
                flux[word1, word2] = total
                flux[word2, word1] = total

        stationary = self.stationary_distribution()
        stationary_matrix = stationary.reshape(-1, 1)
        flux = flux / stationary_matrix
        self._flux_matrix = flux
        return flux

    def word_metadata(self) -> pd.DataFrame:
        if self._word_index is None:
            self.compute_weights()
        return self._word_index.copy()

    def stationary_table(self, descending: bool = True) -> pd.DataFrame:
        metadata = self.word_metadata()
        stationary = self.stationary_distribution()
        table = metadata.copy()
        table["stationary"] = stationary
        table = table.sort_values(
            "stationary", ascending=not descending, ignore_index=True
        )
        return table

    def flux_dataframe(self) -> pd.DataFrame:
        metadata = self.word_metadata()
        labels = metadata.sort_values("word_uid")["word_str"].tolist()
        flux = self.flux_matrix()
        return pd.DataFrame(flux, index=labels, columns=labels)

    def adjacency_matrix(self) -> np.ndarray:
        return (self.flux_matrix() > 0).astype(int)

    def laplacian_matrix(self) -> np.ndarray:
        adj = self.adjacency_matrix()
        deg = np.diag(adj.sum(axis=1))
        return deg - adj

    def spectral_stats(self) -> dict:
        flux = self.flux_matrix()
        lap = self.laplacian_matrix()
        eigen_flux = np.linalg.eigvals(flux)
        eigen_lap = np.linalg.eigvalsh(lap)
        return {
            "flux_eigenvalues": eigen_flux,
            "laplacian_eigenvalues": eigen_lap,
        }
