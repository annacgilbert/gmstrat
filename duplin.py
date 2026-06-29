"""Duplin-Onslow exhaustive enumeration: adapters and fast HCC.

The Duplin-Onslow system (from the Duke Quantifying Gerrymandering
project) is a 25-unit dual graph (24 Onslow precincts + Duplin County
as a single unit) together with an exhaustive enumeration of all
17,653 population-balanced contiguous 3-district plans. Because the
plan space is fully enumerated, every quantity in the pipeline
(letters, words, POU, stratum weights, flux) is exact: there is no
sampling error anywhere.

This module provides
  1. converters that put the raw data into the formats SampleProcessor
     expects (gmstrat graph JSON + CycleWalk-style JSONL),
  2. a memory-lean reimplementation of the HccUltraFit merge loop
     (`hcc_linkage_fast`) that produces the same Z matrix as
     hccfit.HccLinkage but scales to the 14,345 unique districts here,
  3. election utilities: per-district vote totals and per-plan seat
     counts for the 22 races attached to the graph.

Raw inputs live in data/duplin_onslow/; derived artifacts are cached
under local/duplin/ like any other SampleProcessor run.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "duplin_onslow"
RAW_GRAPH = DATA_DIR / "Duplin_Onslow.json"
RAW_ENUM = DATA_DIR / "EnumeratedDuplin_C_Onslow_P_PopOnly.csv"
GRAPH_FN = DATA_DIR / "duplin_onslow_gmstrat.json"
SAMPLE_FN = DATA_DIR / "enumerated_plans.jsonl.gz"
NUM_DISTRICTS = 3


def convert_graph(force: bool = False) -> Path:
    """Write a gmstrat-format graph JSON from the Duke node-link file."""
    if GRAPH_FN.is_file() and not force:
        return GRAPH_FN
    with open(RAW_GRAPH) as f:
        raw = json.load(f)
    nodes = []
    for nd in raw["nodes"]:
        out = dict(nd)
        out["precinct_id_str"] = str(nd["GEOID"])
        out["population"] = int(nd["POPULATION"])
        nodes.append(out)
    with open(GRAPH_FN, "w") as f:
        json.dump({"num_districts": NUM_DISTRICTS, "nodes": nodes}, f)
    return GRAPH_FN


def convert_enumeration(force: bool = False) -> Path:
    """Write the enumerated plans as CycleWalk-style JSONL.gz.

    Mirrors the format SampleProcessor._read_samples expects: two
    ignored lines, a header line with the district count, then one
    record per plan with 1-indexed district labels.
    """
    if SAMPLE_FN.is_file() and not force:
        return SAMPLE_FN
    df = pd.read_csv(RAW_ENUM)
    cols = list(df.columns)
    with gzip.open(SAMPLE_FN, "wt") as f:
        f.write(json.dumps({"comment": "exhaustive Duplin-Onslow enumeration"}) + "\n")
        f.write(json.dumps({"comment": "all population-balanced contiguous plans"}) + "\n")
        f.write(json.dumps({"districts": NUM_DISTRICTS}) + "\n")
        for step, row in enumerate(df.itertuples(index=False), start=1):
            districting = [{col: int(val)} for col, val in zip(cols, row)]
            f.write(json.dumps({"name": f"step{step}", "districting": districting}) + "\n")
    return SAMPLE_FN


def compute_distance_matrix_fast(districts, population, maximum_distance,
                                 block: int = 2048) -> np.ndarray:
    """Vectorized pairwise capped population-weighted l1 distance.

    Districts are index vectors (as from SampleProcessor.get_all_districts).
    For binary membership vectors the weighted l1 distance is
        d(u, v) = pop(u) + pop(v) - 2 * pop(u & v),
    computed blockwise as an inner product of weighted membership rows.
    Matches SampleProcessor.compute_distance exactly (int32, capped).
    """
    population = np.asarray(population, dtype=np.int64)
    n = len(districts)
    p = population.shape[0]
    B = np.zeros((n, p), dtype=np.float64)
    for i, idx in enumerate(districts):
        B[i, np.asarray(idx, dtype=np.intp)] = 1.0
    Bw = B * population[None, :]
    pops = Bw.sum(axis=1)

    out = np.empty((n, n), dtype=np.int32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        inter = Bw[start:stop] @ B.T
        d = pops[start:stop, None] + pops[None, :] - 2.0 * inter
        np.minimum(d, float(maximum_distance), out=d)
        out[start:stop] = np.rint(d).astype(np.int32)
    np.fill_diagonal(out, 0)
    return out


def hcc_linkage_fast(d: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Memory-lean HccUltraFit linkage, equivalent to hccfit.HccLinkage.

    Reproduces learn_UM exactly (same edge order, same merge condition,
    same Z output) but avoids the dense (n x n) Python edge list and the
    dense (2n x 2n) M matrix:
      - edges sorted with numpy argsort over the condensed upper triangle
        (ties resolved in the same (i, j) i>j row-major order as
        get_edge_seq's stable sort),
      - N kept as (n, 2n) int32, H as (n, 2n) bool,
      - M kept as a dict-of-dicts over active roots only,
      - the per-merge recomputation of H[:, r] vectorized over vertices.

    Returns Z with rows [k, l, distance, new_size] indexed by r - n,
    identical to HccLinkage.Z.
    """
    n = d.shape[0]
    # get_edge_seq emits (i, j) with i > j in row-major order and then
    # stable-sorts by distance; tril_indices yields the same (i, j)
    # sequence, so a stable argsort reproduces the tie order exactly.
    il, jl = np.tril_indices(n, k=-1)
    vals = d[il, jl]
    order = np.argsort(vals, kind="stable")
    ei = il[order].astype(np.int32)
    ej = jl[order].astype(np.int32)
    del il, jl, vals, order

    # counts fit int16: N[v, c] <= |c| <= n = 14345 < 32767
    N = np.zeros((n, 2 * n), dtype=np.int16)
    H = np.zeros((n, 2 * n), dtype=bool)
    S = np.ones(2 * n, dtype=np.int64)
    membership = np.arange(n, dtype=np.int64)
    M: dict[int, dict[int, int]] = {i: {} for i in range(n)}
    Z = np.zeros((n - 1, 4))
    heights = np.zeros(2 * n)
    next_root = n
    num_clusters = n

    def m_get(k: int, l: int) -> int:
        return M[k].get(l, 0)

    def m_inc(k: int, l: int) -> None:
        M[k][l] = M[k].get(l, 0) + 1

    t = 0
    total_edges = ei.shape[0]
    report = max(total_edges // 20, 1)
    while num_clusters > 1 and t < total_edges:
        i = int(ei[t])
        j = int(ej[t])
        k = int(membership[i])
        l = int(membership[j])
        N[i, l] += 1
        N[j, k] += 1
        if k != l:
            if not H[i, l] and 2 * N[i, l] >= S[l]:
                H[i, l] = True
                m_inc(k, l)
            if not H[j, k] and 2 * N[j, k] >= S[k]:
                H[j, k] = True
                m_inc(l, k)
            if m_get(k, l) + m_get(l, k) == S[k] + S[l]:
                # ---- merge clusters k and l into new root r ----
                r = next_root
                next_root += 1
                distance = float(d[i, j])
                new_size = int(S[k] + S[l])
                S[r] = new_size
                N[:, r] = N[:, k] + N[:, l]
                in_r = (membership == k) | (membership == l)
                hv = 2 * N[:, r] >= new_size
                H[:, r] = hv
                # members of each active cluster that hear r
                m_r_rows = np.bincount(
                    membership[hv & ~in_r].astype(np.intp), minlength=0
                )
                for c in np.flatnonzero(m_r_rows):
                    cc = int(c)
                    if S[cc] > 0 and cc != k and cc != l:
                        M[cc][r] = int(m_r_rows[cc])
                # r's own row: union of k's and l's rows
                row_k = M.pop(k, {})
                row_l = M.pop(l, {})
                merged = dict(row_k)
                for key, v in row_l.items():
                    merged[key] = merged.get(key, 0) + v
                merged.pop(k, None)
                merged.pop(l, None)
                M[r] = merged
                # vertices of r that hear r itself count into M[r][r]?
                # hccfit increments M[membership[v], r] for ALL v with
                # 2N >= size, including members of k and l (then sums
                # M[r,:] = M[k,:] + M[l,:] BEFORE membership update, so
                # those go to M[k, r] / M[l, r] and land in M[r][r]).
                self_hear = int(np.count_nonzero(hv & in_r))
                if self_hear:
                    merged[r] = merged.get(r, 0) + self_hear
                membership[in_r] = r
                S[k] = 0
                S[l] = 0
                heights[r] = distance
                Z[r - n] = [k, l, distance, new_size]
                num_clusters -= 1
                if verbose and num_clusters % 1000 == 0:
                    print(f"  {num_clusters} clusters left, edge {t}/{total_edges}")
        t += 1
    if num_clusters != 1:
        raise RuntimeError(f"merge loop exhausted edges with {num_clusters} clusters")
    return Z


# ---------------------------------------------------------------------------
# Elections


def race_columns(df_precincts: pd.DataFrame) -> list[str]:
    """Race prefixes (e.g. EL16G_USS) with both _D and _R columns."""
    cols = set(df_precincts.columns)
    races = sorted(
        {c[:-2] for c in cols if c.startswith("EL") and c.endswith("_D")}
    )
    return [r for r in races if f"{r}_R" in cols]


def district_seat_table(sp, races: list[str] | None = None) -> pd.DataFrame:
    """Democratic win indicator and vote shares per unique district."""
    df_p = sp.df_precincts
    if races is None:
        races = race_columns(df_p)
    districts = sp.get_all_districts()
    rows = {}
    for race in races:
        dvotes = df_p[f"{race}_D"].to_numpy(dtype=np.int64)
        rvotes = df_p[f"{race}_R"].to_numpy(dtype=np.int64)
        dsum = np.array([dvotes[idx].sum() for idx in districts])
        rsum = np.array([rvotes[idx].sum() for idx in districts])
        rows[f"{race}_dem_win"] = (dsum > rsum).astype(np.int32)
        rows[f"{race}_dem_share"] = dsum / np.maximum(dsum + rsum, 1)
    out = pd.DataFrame(rows)
    out.index.name = "district_uid"
    return out


def plan_seat_counts(sp, df_seats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Democratic seat count (0-3) per plan per race, plus BVAP stats."""
    if df_seats is None:
        df_seats = district_seat_table(sp)
    races = sorted({c[:-8] for c in df_seats.columns if c.endswith("_dem_win")})
    win_cols = {r: df_seats[f"{r}_dem_win"].to_numpy() for r in races}

    plans = sp.df_distributions.plan_vector.to_list()
    plan_arr = np.array([np.asarray(p, dtype=np.intp) for p in plans])
    out = pd.DataFrame(index=np.arange(len(plans)))
    out.index.name = "plan_uid"
    for r in races:
        out[f"{r}_seats"] = win_cols[r][plan_arr].sum(axis=1)

    # Black voting-age population share of the most-Black district
    df_pcts = sp.df_precincts
    bvap = df_pcts["PL10VA_AP_B"].to_numpy(dtype=np.int64)
    vap = df_pcts["PL10VA_TOT"].to_numpy(dtype=np.int64)
    districts = sp.get_all_districts()
    bshare = np.array(
        [bvap[idx].sum() / max(vap[idx].sum(), 1) for idx in districts]
    )
    out["max_bvap_share"] = bshare[plan_arr].max(axis=1)
    return out
