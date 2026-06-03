from __future__ import annotations

import json
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors


def read_json(fn):
    with open(fn, "r") as f:
        return json.load(f)


def vec_to_str(vec):
    return ".".join([str(int(i)) for i in sorted(vec)])


def str_to_vec(s):
    if s == "":
        return np.array([], dtype=np.int32)
    return np.array(s.split("."), dtype=np.int32)


def weighted_l1(
    v1, v2, weight, maximum_distance=np.iinfo(np.int32).max, *, sparse: bool
):
    if sparse:
        w = np.asarray(weight)
        diff = np.setxor1d(np.asarray(v1, dtype=np.intp), np.asarray(v2, dtype=np.intp))
        dist = w[diff].sum(dtype=np.int64)
        if maximum_distance is not None:
            dist = min(dist, np.int64(maximum_distance))
        return np.int32(dist)

    v1_f = np.asarray(v1, dtype=float)
    v2_f = np.asarray(v2, dtype=float)
    w_f = np.asarray(weight, dtype=float)
    dist = float(np.dot(np.abs(v1_f - v2_f), w_f))
    if maximum_distance is not None:
        dist = min(dist, float(maximum_distance))
    return dist


def plot_plan(gdf, labels, ax=None, cmap="tab20"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    gdf.assign(cluster=labels).plot(
        column="cluster", cmap=cmap, ax=ax, edgecolor="black", linewidth=0.2
    )
    ax.axis("off")
    return ax


def plot_district(gdf, mask, ax=None, color="red"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    mask_arr = np.asarray(mask)
    if mask_arr.dtype.kind in "iu" and mask_arr.ndim == 1 and mask_arr.size != len(gdf):
        sel = np.zeros(len(gdf), dtype=bool)
        sel[mask_arr] = True
    else:
        sel = mask_arr.astype(bool)

    gdf.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.2)
    gdf[sel].plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)
    ax.axis("off")
    return ax


def plot_distribution(
    gdf,
    values,
    *,
    ax=None,
    cmap="Blues",
    axis=False,
    colorbar=False,
    vmin=None,
    vmax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    gdf.assign(_value=np.asarray(values, dtype=float)).plot(
        column="_value",
        cmap=cmap,
        ax=ax,
        edgecolor="black",
        linewidth=0.2,
        legend=bool(colorbar),
        vmin=vmin,
        vmax=vmax,
    )
    if not axis:
        ax.axis("off")
    return ax


def plot_words_list(
    gdf,
    cluster_densities,
    word,
    *,
    cmap="Blues",
    cols=3,
    figsize=(10, 4),
    show_colorbar=True,
):
    word = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)
    n_letters = int(len(word))

    cols = int(min(int(cols), max(n_letters, 1)))
    rows = int(math.ceil(n_letters / cols)) if n_letters else 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    selected = cluster_densities[word] if n_letters else np.asarray([0.0])
    vmin = float(np.min(selected))
    vmax = float(np.max(selected))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    for idx, centroid_id in enumerate(word.tolist()):
        values = cluster_densities[int(centroid_id)]
        plot_distribution(
            gdf,
            values,
            ax=axes_flat[idx],
            cmap=cmap,
            axis=True,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        axes_flat[idx].set_title(f"letter {idx}: {int(centroid_id)}")

    for ax in axes_flat[n_letters:]:
        ax.axis("off")

    if show_colorbar and n_letters:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes_flat[:n_letters].tolist(),
            fraction=0.035,
            pad=0.04,
            location="right",
        )

    return fig, axes


def plot_words_combined(
    gdf,
    cluster_densities,
    word,
    *,
    cmap="tab10",
    base_color="lightgrey",
    edgecolor="black",
    linewidth=0.2,
    ltr=False,
    ax=None,
    figsize=(8, 8),
):
    word = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    gdf.plot(
        ax=ax, color=base_color, edgecolor=edgecolor, linewidth=linewidth, alpha=0.3
    )

    if ltr and len(word) > 1:
        x_centers = gdf.geometry.centroid.x.to_numpy()
        xs = []
        for centroid_id in word.tolist():
            dens = cluster_densities[int(centroid_id)]
            total = float(dens.sum())
            xs.append(
                float("inf") if total <= 0 else float(np.sum(x_centers * dens) / total)
            )
        order = np.argsort(np.asarray(xs, dtype=float), kind="mergesort")
        word = word[order]

    cmap_obj = plt.get_cmap(cmap)
    n_letters = max(len(word), 1)
    for idx, centroid_id in enumerate(word.tolist()):
        values = cluster_densities[int(centroid_id)]
        if np.all(values == 0):
            continue
        color = cmap_obj(idx / max(1, n_letters - 1))
        rgba = np.tile(color, (len(values), 1))
        if rgba.shape[1] == 3:
            rgba = np.column_stack([rgba, np.ones(len(values))])
        alpha = values / (values.max() if values.max() > 0 else 1.0)
        rgba[:, -1] = np.clip(alpha, 0.0, 1.0)
        gdf.plot(ax=ax, color=rgba, edgecolor=edgecolor, linewidth=linewidth)

    ax.axis("off")
    return fig, ax


def plot_words_centroids(
    gdf,
    cluster_densities,
    word,
    *,
    threshold=0.5,
    cmap="tab10",
    base_color="lightgrey",
    base_alpha=0.3,
    centroid_alpha=0.9,
    edgecolor="black",
    linewidth=0.2,
    overlap_edgecolor="black",
    overlap_linewidth=1.2,
    ltr=False,
    ax=None,
    figsize=(8, 8),
):
    word_arr = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if ltr and len(word_arr) > 1:
        x_centers = gdf.geometry.centroid.x.to_numpy()
        xs = []
        for centroid_id in word_arr.tolist():
            dens = cluster_densities[int(centroid_id)]
            mask = dens >= float(threshold)
            xs.append(float(np.mean(x_centers[mask])) if np.any(mask) else float("inf"))
        order = np.argsort(np.asarray(xs, dtype=float), kind="mergesort")
        word_arr = word_arr[order]

    if len(word_arr) == 0:
        gdf.plot(
            ax=ax,
            color=base_color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=base_alpha,
        )
        ax.axis("off")
        return fig, ax

    densities = cluster_densities[word_arr]
    mask = densities >= float(threshold)
    weights = densities * mask
    totals = weights.sum(axis=0)
    has_any = totals > 0
    overlap = mask.sum(axis=0) > 1

    cmap_obj = plt.get_cmap(cmap)
    n_letters = max(len(word_arr), 1)
    letter_rgba = np.array(
        [cmap_obj(i / max(1, n_letters - 1)) for i in range(len(word_arr))],
        dtype=float,
    )
    letter_rgb = letter_rgba[:, :3]

    rgb_num = weights.T @ letter_rgb
    rgb = np.zeros_like(rgb_num)
    np.divide(rgb_num, totals[:, None], out=rgb, where=has_any[:, None])

    base_rgba = np.array(mcolors.to_rgba(base_color, alpha=base_alpha), dtype=float)
    rgba = np.tile(base_rgba, (len(gdf), 1))
    rgba[has_any, :3] = rgb[has_any]
    rgba[has_any, 3] = float(centroid_alpha)

    gdf.plot(ax=ax, color=rgba, edgecolor=edgecolor, linewidth=linewidth)
    if np.any(overlap):
        gdf[overlap].boundary.plot(
            ax=ax, color=overlap_edgecolor, linewidth=overlap_linewidth
        )

    ax.axis("off")
    return fig, ax
