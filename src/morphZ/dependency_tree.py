#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde

try:
    import pandas as pd  # optional, only for CSV labels
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


###############################################################################
#                                  UTILITIES                                  #
###############################################################################

def _safe_gaussian_kde(
    data: np.ndarray,
    bw_method: str | float | None,
    rng: np.random.Generator,
    max_retries: int = 2,
) -> gaussian_kde:
    """
    Fit ``gaussian_kde`` robustly, adding tiny Gaussian jitter if a
    singular covariance is encountered.

    Args:
        data: Array of shape (d, n) as expected by ``gaussian_kde``.
        bw_method: Bandwidth method or scalar factor; passed through to SciPy.
        rng: Numpy Generator used to draw jitter when needed.
        max_retries: Maximum number of jitter retries before raising.

    Returns:
        A fitted ``gaussian_kde`` instance.

    Notes:
        - If your data contains duplicated columns/samples or has near-zero
          variance along a dimension, consider starting with a numeric
          ``bw_method`` slightly larger than the default (e.g., 1.2) to reduce
          the chance of singular matrices.
    """
    attempt = 0
    jitter_scale = 1e-9
    while True:
        try:
            return gaussian_kde(data, bw_method=bw_method)
        except np.linalg.LinAlgError:
            attempt += 1
            if attempt > max_retries:
                raise
            warnings.warn(
                f"Singular covariance detected (attempt {attempt}). Adding jitter and retrying.",
                RuntimeWarning,
            )
            data = data + rng.normal(scale=jitter_scale, size=data.shape)
            jitter_scale *= 10  # back‑off


def _eval_log_pdf_at_samples(
    kde: gaussian_kde,
    samples_T: np.ndarray,
    loo: bool,
    eps: float = 1e-300,
) -> np.ndarray:
    """
    Evaluate log‑pdf at the training samples with optional leave‑one‑out
    correction to reduce self‑contribution bias.

    Args:
        kde: A fitted ``gaussian_kde``.
        samples_T: Sample matrix with shape (d, n) matching the kde, i.e.
            transposed with samples in columns.
        loo: If True, apply a simple leave‑one‑out adjustment.
        eps: Lower clip bound to avoid ``log(0)``.

    Returns:
        Array of shape (n,) containing log‑densities for each sample.
    """
    pdf_vals = kde.evaluate(samples_T)
    if loo:
        n = samples_T.shape[1]
        try:
            norm_factor = kde._norm_factor  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover – older SciPy fallback
            d = kde.d
            cov_det = np.linalg.det(kde.covariance)
            norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        self_term = norm_factor * (1.0 / n)
        pdf_vals = (n * pdf_vals - self_term) / (n - 1)
    return np.log(np.clip(pdf_vals, eps, None))


def compute_pairwise_mi(
    samples: np.ndarray,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    eps: float = 1e-300,
) -> np.ndarray:
    """
    Compute a symmetric matrix of pairwise mutual informations using KDE.

    Args:
        samples (np.ndarray): 2D array of shape (n_samples, n_dims).
        bw_method (str or float, optional): Bandwidth for KDE. Use a SciPy
            method name like "silverman" or "scott", or a positive scalar
            factor. Defaults to "silverman".
        loo (bool, optional): If True, use leave-one-out correction. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        eps (float, optional): Small value to clip PDF values to avoid log(0). Defaults to 1e-300.

    Returns:
        np.ndarray: A symmetric matrix of pairwise mutual informations.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2‑D array (n_samples, n_dims).")
    n_samples, n_dims = samples.shape

    # 1‑D densities
    log_p = np.empty((n_dims, n_samples))
    for i in range(n_dims):
        kde_i = _safe_gaussian_kde(samples[:, i][None, :], bw_method, rng)
        log_p[i] = _eval_log_pdf_at_samples(kde_i, samples[:, i][None, :], loo, eps)

    # 2‑D densities & MI
    mi = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            kde_ij = _safe_gaussian_kde(samples[:, [i, j]].T, bw_method, rng)
            log_pxy = _eval_log_pdf_at_samples(kde_ij, samples[:, [i, j]].T, loo, eps)
            mi_ij = float(np.mean(log_pxy - log_p[i] - log_p[j]))
            mi[i, j] = mi[j, i] = max(0.0, mi_ij)  # ensure non‑negativity
    return mi

###############################################################################
#                                VISUALISATION                                #
###############################################################################

def plot_mi_heatmap(
    mi: np.ndarray,
    labels: Sequence[str],
    outfile: str = "mi_heatmap.png",
    cmap: str = "inferno",
) -> None:
    """
    Save a heat-map of the MI matrix.

    Args:
        mi (np.ndarray): The mutual information matrix.
        labels (Sequence[str]): The labels for the axes.
        outfile (str, optional): Path where the image is saved. Defaults to "mi_heatmap.png".
        cmap (str, optional): Matplotlib colormap name. Nice options: "magma",
            "inferno", "viridis". Defaults to "inferno".
    """
    n_labels = len(labels)
    # Dynamically adjust figure size and font size for readability
    # These factors can be tuned for better aesthetics
    fig_width = max(8, n_labels * 0.15)
    fig_height = max(6, n_labels * 0.15)
    font_size = max(4, 12 - n_labels * 0.08)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(mi, cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mutual Information", rotation=-90, va="bottom")

    ax.set_title("Pairwise Mutual Information")
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_chow_liu_tree(
    tree: nx.Graph,
    labels: Sequence[str],
    outfile: str = "chow_liu_tree.png",
    root: int = 0,
) -> None:
    """
    Draw the Chow–Liu tree using a spring layout for better aesthetics.

    Args:
        tree (nx.Graph): The Chow–Liu maximum spanning tree over variables.
        labels (Sequence[str]): Labels per node index.
        outfile (str, optional): Path where the image is saved. Defaults to "chow_liu_tree.png".
        root (int, optional): Root index used by the BFS orientation and for
            hierarchical layout when Graphviz is available. Defaults to 0.

    Tips:
        - For clean hierarchical layouts, install ``pygraphviz``.
        - Edge labels show MI weights; consider post‑processing or thresholding
          very small weights if the figure is cluttered.
    """
    try:
        # Use graphviz for a hierarchical layout that avoids overlaps
        pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
    except ImportError:
        logger.warning(
            "PyGraphviz not found, falling back to spring_layout. For a better layout, please install pygraphviz."
        )
        pos = nx.spring_layout(tree, seed=42) # spring layout for equal branches


    n_dims = len(labels)
    # Scale font size to be smaller with more dimensions
    node_font_size = max(4, 10 - 0.1 * n_dims)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(tree, pos, node_size=800, node_color="#89CFF0")
    nx.draw_networkx_edges(tree, pos, width=2.5, alpha=0.7)
    nx.draw_networkx_labels(tree, pos, labels=dict(enumerate(labels)), font_size=node_font_size)
    
    edge_lbls = {(u, v): f"{tree[u][v]['weight']:.3f}" for u, v in tree.edges()}
    edge_labels_drawn = nx.draw_networkx_edge_labels(
        tree, pos, edge_labels=edge_lbls, font_color="darkred", font_size=11, font_weight='bold'
    )
    # Add a white bounding box to edge labels for readability
    for label in edge_labels_drawn.values():
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.3))

    plt.title("Chow–Liu Dependency Tree", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

###############################################################################
#                             DEPENDENCY EXTRACTION                            #
###############################################################################

def extract_dependency_edges(
    tree: nx.Graph,
    labels: Sequence[str],
    root: int = 0,
) -> List[Tuple[str, str]]:
    """
    Return a (parent, child) edge list in BFS order with string labels.

    Args:
        tree: Undirected Chow–Liu tree over indices.
        labels: Human‑readable names for nodes (``labels[i]`` for node i).
        root: Index to use as BFS root when orienting edges.

    Returns:
        List of pairs of strings representing parent -> child relationships.
    """
    directed = nx.bfs_tree(tree, root)  # oriented from root outward
    return [(str(labels[parent]), str(labels[child])) for parent, child in directed.edges()]

###############################################################################
#                            MAIN ORCHESTRATION                                #
###############################################################################

def compute_and_plot_mi_tree(
    samples: np.ndarray,
    names: Optional[Iterable[str]] = None,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    root: int = 0,
    out_path: Optional[str] = None,
    morph_type: Optional[str] = None,
) -> Tuple[np.ndarray, nx.Graph, List[Tuple[str, str]]]:
    """
    Compute MI, visualise, and return tree plus *(parent, child)* edge list.

    Args:
        samples (np.ndarray): Array of shape (n_samples, n_dims).
        names (Iterable[str] | None): Optional names for each dimension. If None,
            names default to ``["x0", "x1", ...]``.
        bw_method (str | float, optional): KDE bandwidth. Use a SciPy method name
            ("silverman", "scott") or a numeric factor. Defaults to "silverman".
        loo (bool, optional): Apply leave‑one‑out correction in KDE evaluations. Defaults to True.
        seed (int, optional): RNG seed used for robust KDE fitting. Defaults to 0.
        root (int, optional): Root index when orienting the tree. Defaults to 0.
        out_path (str | None, optional): Directory to save visualisations and
            JSON outputs. If None, images are saved to CWD and JSON is skipped.
        morph_type (str | None, optional): If provided, controls which artifacts
            are saved. ``chow_liu_tree.png`` and ``tree.json`` are saved only
            when ``morph_type == 'tree'``. When ``None`` (backward compatible),
            tree artifacts are saved.

    Returns:
        Tuple[np.ndarray, nx.Graph, List[Tuple[str, str]]]: The MI matrix, the
        maximum spanning tree, and the list of (parent, child) edges.

    Suggestions:
        - If you plan to use ``TreeKDE`` later, pass ``out_path`` so that the
          generated ``tree.json`` is available without recomputation.
        - For very high dimensions, reduce the number of samples used to speed
          up the MI estimates and plotting.
    """
    n_dims = samples.shape[1]
    if names is not None:
        labels = [str(n) for n in names]  # ensure JSON‑serialisable strings
    else:
        labels = [f"x{i}" for i in range(n_dims)]

    if len(labels) != n_dims:
        raise ValueError("Length of `names` must match number of dimensions.")

    mi = compute_pairwise_mi(samples, bw_method=bw_method, loo=loo, seed=seed)

    # Decide whether to save tree artifacts based on morph_type
    save_tree_artifacts = (morph_type == "tree") if morph_type is not None else True

    heatmap_path = "mi_heatmap.png"
    tree_path = "chow_liu_tree.png"
    json_path = None

    if out_path:
        p = Path(out_path)
        p.mkdir(parents=True, exist_ok=True)
        heatmap_path = p / "mi_heatmap.png"
        tree_path = p / "chow_liu_tree.png"
        json_path = p / "tree.json"

    plot_mi_heatmap(mi, labels, outfile=heatmap_path)

    # Build fully‑connected weighted graph
    complete = nx.Graph()
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            complete.add_edge(i, j, weight=float(mi[i, j]))

    mst = nx.maximum_spanning_tree(complete, weight="weight")
    # Save the tree image only when requested (morph_type == 'tree')
    if save_tree_artifacts and out_path:
        plot_chow_liu_tree(mst, labels, outfile=tree_path, root=root)

    dep_edges = extract_dependency_edges(mst, labels, root=root)

    # print("\nDependency list (parent -> child):")
    # for p, c in dep_edges:
    #     print(f"  {p} -> {c}")

    # Save dependency list JSON only when requested (morph_type == 'tree')
    if save_tree_artifacts and json_path:
        with open(json_path, "w", encoding="utf8") as fp:
            json.dump(dep_edges, fp, indent=2, default=str)
        logger.info("Dependency list saved to %s", json_path)

    # Top‑10 MI pairs for reference
    tri = np.triu_indices_from(mi, k=1)
    all_pairs = sorted(zip(tri[0], tri[1], mi[tri]), key=lambda x: x[2], reverse=True)
    top10 = all_pairs[:10]
    logger.info("Top 10 MI pairs:")
    for i, j, val in top10:
        logger.info("  %s — %s : %.6f", labels[i], labels[j], val)

    if out_path:
        mi_path = Path(out_path) / "params_MI.json"
        mi_data = [[labels[i], labels[j], val] for i, j, val in all_pairs]
        with open(mi_path, "w", encoding="utf8") as f:
            json.dump(mi_data, f, indent=2)
        logger.info("All MI pairs saved to %s", mi_path)


    return mi, mst, dep_edges
