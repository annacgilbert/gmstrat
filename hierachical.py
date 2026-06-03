from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from utils import str_to_vec


def get_cluster_from_linkage(
    k: int, linkage: np.ndarray
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    num_vertices = len(linkage) + 1
    membership = {i: [i] for i in range(num_vertices)}

    for merge_idx, merge in enumerate(linkage[:, :2], start=0):
        x, y = int(merge[0]), int(merge[1])
        membership[merge_idx + num_vertices] = membership.pop(x) + membership.pop(y)
        if len(membership) == k:
            break

    c2i: Dict[int, List[int]] = defaultdict(list)
    i2c = np.zeros(num_vertices, dtype=np.int32)
    for cluster_id, cluster_name in enumerate(list(membership.keys())):
        for member in membership[cluster_name]:
            c2i[cluster_id].append(member)
            i2c[member] = cluster_id
    return c2i, i2c


class HClusters:
    def __init__(self, sp):
        self.sp = sp
        self.linkage = np.load(sp.paths.linkage)
        self.num_vertices = len(self.linkage) + 1
        self.centroids: List[np.ndarray] = []
        self.cluster_densities: np.ndarray | None = None

    def update_clusters(self, num_clusters: int, update_centroids: bool = True) -> None:
        self.num_clusters = num_clusters
        self.c2i, self.i2c = get_cluster_from_linkage(self.num_clusters, self.linkage)
        if update_centroids:
            self.update_centroids()

    def kcentroids(
        self, convergence_thres: float = 0.01, max_iter: int = 50, verbose=True
    ) -> None:
        num_clusters = int(getattr(self, "num_clusters", 0))
        all_districts = self.sp.get_all_districts()
        prev_assignments = np.asarray(self.i2c, dtype=np.int32).copy()
        prev_wcss = self._compute_wcss_centroid(all_districts, prev_assignments)

        for epoch in tqdm(range(max_iter)):
            new_assignments = self._assign_to_nearest_centroids(
                all_districts,
                centroids=self.centroids,
                previous_assignments=prev_assignments,
                num_clusters=num_clusters,
            )
            assignments_unchanged = np.array_equal(new_assignments, prev_assignments)
            self.i2c = new_assignments
            self.c2i = self._build_c2i_from_assignments(new_assignments, num_clusters)
            self.update_centroids()

            new_wcss = self._compute_wcss_centroid(all_districts, new_assignments)
            improvement = prev_wcss - new_wcss
            if improvement <= 0.0:
                break

            rel_improvement = improvement / prev_wcss if prev_wcss > 0.0 else 0.0
            prev_assignments = new_assignments
            prev_wcss = new_wcss
            if assignments_unchanged or (rel_improvement < convergence_thres):
                break
            if verbose:
                print(f"epoch {epoch}: rel_improvement = {rel_improvement}")

    def _build_c2i_from_assignments(
        self, assignments: np.ndarray, num_clusters: int
    ) -> Dict[int, List[int]]:
        c2i: Dict[int, List[int]] = defaultdict(list)
        for district_uid, cluster_id in enumerate(assignments):
            cid = int(cluster_id)
            c2i[cid].append(int(district_uid))
        return c2i

    def _assign_to_nearest_centroids(
        self,
        districts: Sequence[np.ndarray],
        *,
        centroids: Sequence[np.ndarray],
        previous_assignments: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        weights = np.asarray(self.sp.population, dtype=np.int64)
        max_dist = (
            None
            if self.sp.maximum_distance is None
            else int(np.int64(self.sp.maximum_distance))
        )

        centroid_members = np.zeros(
            (num_clusters, self.sp.num_precincts), dtype=np.bool_
        )
        centroid_weight_sum = np.zeros(num_clusters, dtype=np.int64)
        for cluster_id, centroid in enumerate(centroids):
            idx = np.asarray(centroid, dtype=np.intp)
            if idx.size:
                centroid_members[cluster_id, idx] = True
                centroid_weight_sum[cluster_id] = weights[idx].sum(dtype=np.int64)

        num_points = len(districts)
        assignments = np.empty(num_points, dtype=np.int32)

        for district_uid, district in enumerate(districts):
            idx = np.asarray(district, dtype=np.intp)
            if idx.size:
                w_idx = weights[idx]
                sum_w = int(w_idx.sum(dtype=np.int64))
                inter = centroid_members[:, idx] @ w_idx
            else:
                sum_w = 0
                inter = np.zeros(num_clusters, dtype=np.int64)

            dist_vec = sum_w + centroid_weight_sum - 2 * inter
            if max_dist is not None:
                dist_vec = np.minimum(dist_vec, max_dist)

            min_dist = int(dist_vec.min())
            candidates = np.flatnonzero(dist_vec == min_dist)

            current = int(previous_assignments[district_uid])
            if np.any(candidates == current):
                chosen = current
            else:
                chosen = int(candidates[0])

            assignments[district_uid] = chosen

        return assignments

    def _compute_wcss_centroid(
        self, districts: Sequence[np.ndarray], assignments: np.ndarray
    ) -> float:
        num_clusters = int(getattr(self, "num_clusters", 0))
        weights = np.asarray(self.sp.population, dtype=np.int64)
        max_dist = (
            None
            if self.sp.maximum_distance is None
            else int(np.int64(self.sp.maximum_distance))
        )

        centroid_members = np.zeros(
            (num_clusters, self.sp.num_precincts), dtype=np.bool_
        )
        centroid_weight_sum = np.zeros(num_clusters, dtype=np.int64)
        for cluster_id, centroid in enumerate(self.centroids):
            idx = np.asarray(centroid, dtype=np.intp)
            if idx.size:
                centroid_members[cluster_id, idx] = True
                centroid_weight_sum[cluster_id] = weights[idx].sum(dtype=np.int64)

        total = np.int64(0)
        for district_uid, district in enumerate(districts):
            cluster_id = int(assignments[district_uid])
            idx = np.asarray(district, dtype=np.intp)
            if idx.size:
                w_idx = weights[idx]
                sum_w = int(w_idx.sum(dtype=np.int64))
                inter = int(centroid_members[cluster_id, idx] @ w_idx)
            else:
                sum_w = 0
                inter = 0
            dist = sum_w + int(centroid_weight_sum[cluster_id]) - 2 * inter
            if max_dist is not None and dist > max_dist:
                dist = max_dist
            total += dist
        return float(total)

    def update_centroids(self) -> None:
        all_districts = self.sp.get_all_districts()
        centroids: List[np.ndarray] = []
        densities: List[np.ndarray] = []

        num_clusters = int(getattr(self, "num_clusters", max(self.i2c) + 1))
        for cluster_id in range(num_clusters):
            members = self.c2i[cluster_id]
            cls_size = len(members)
            if cls_size == 0:
                densities.append(np.zeros(self.sp.num_precincts, dtype=float))
                centroids.append(np.array([], dtype=int))
                continue

            counts = np.zeros(self.sp.num_precincts, dtype=np.int32)
            for district_uid in members:
                counts[all_districts[district_uid]] += 1

            dens = counts / cls_size
            centroid = np.flatnonzero(dens >= 0.5)
            densities.append(dens)
            centroids.append(centroid)

        self.cluster_densities = np.vstack(densities)
        self.centroids = centroids

    def compute_wcss(self):
        df = self.sp.df_districts.copy()
        df["dvec"] = df.district_str.apply(str_to_vec)
        df["centroid"] = df.district_uid.apply(lambda x: self.centroids[self.i2c[x]])
        df["letter"] = df.district_uid.apply(lambda x: self.i2c[x])
        df["dcentroid"] = df.apply(
            lambda row: self.sp.compute_distance(row.dvec, row.centroid), axis=1
        )
        return df
