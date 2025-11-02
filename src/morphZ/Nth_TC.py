#!/usr/bin/env python3
"""
Compute n-order Total Correlation (TC) from continuous posterior samples via Gaussian KDE.
For n=2, this is equivalent to pairwise Mutual Information (MI).
Plots an MI heatmap for n=2, and saves the TC values for n>=2.

Dependencies: numpy, scipy, matplotlib, pandas (optional).
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
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
    """Fit ``gaussian_kde`` robustly, adding jitter if covariance is singular."""
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
    """Evaluate log‑pdf at training samples with optional leave‑one‑out correction."""
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


def compute_total_correlation(
    samples: np.ndarray,
    n_order: int = 2,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    eps: float = 1e-300,
) -> np.ndarray | list[tuple[tuple[int, ...], float]]:
    """
    Compute a symmetric matrix of pairwise mutual informations (n_order=2)
    or a list of n-order total correlations using KDE.

    Args:
        samples (np.ndarray): 2D array of shape (n_samples, n_dims).
        n_order (int, optional): Order of total correlation. Defaults to 2 (pairwise MI).
        bw_method (str or float, optional): Bandwidth method for KDE. Defaults to "silverman".
        loo (bool, optional): If True, use leave-one-out correction. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        eps (float, optional): Small value to clip PDF values to avoid log(0). Defaults to 1e-300.

    Returns:
        np.ndarray | list: For n_order=2, a symmetric matrix of pairwise MIs.
                           For n_order>2, a list of (indices, tc_value) tuples.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2‑D array (n_samples, n_dims).")
    n_samples, n_dims = samples.shape

    if not 2 <= n_order <= n_dims:
        raise ValueError(f"`n_order` must be between 2 and {n_dims}.")

    # 1‑D marginal densities
    log_p = np.empty((n_dims, n_samples))
    for i in range(n_dims):
        kde_i = _safe_gaussian_kde(samples[:, i][None, :], bw_method, rng)
        log_p[i] = _eval_log_pdf_at_samples(kde_i, samples[:, i][None, :], loo, eps)

    # N‑D densities & TC
    if n_order == 2:
        mi = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                kde_ij = _safe_gaussian_kde(samples[:, [i, j]].T, bw_method, rng)
                log_pxy = _eval_log_pdf_at_samples(kde_ij, samples[:, [i, j]].T, loo, eps)
                mi_ij = float(np.mean(log_pxy - log_p[i] - log_p[j]))
                mi[i, j] = mi[j, i] = max(0.0, mi_ij)  # ensure non‑negativity
        return mi
    else:
        results = []
        n_combin = 0
        for indices in combinations(range(n_dims), n_order):
            subset_samples = samples[:, list(indices)].T
            kde_n = _safe_gaussian_kde(subset_samples, bw_method, rng)
            log_p_joint = _eval_log_pdf_at_samples(kde_n, subset_samples, loo, eps)
            
            log_p_marginals_sum = np.sum(log_p[list(indices), :], axis=0)
            
            tc_value = float(np.mean(log_p_joint - log_p_marginals_sum))
            results.append((indices, max(0.0, tc_value)))
            n_combin += 1
        logger.info("The number of estimated Total correlations: %s", n_combin)
        return results


###############################################################################
#                                VISUALISATION                                #
###############################################################################

def plot_tc_heatmap(
    mi: np.ndarray,
    labels: Sequence[str],
    outfile: str = "mi_heatmap.png",
    cmap: str = "magma",
) -> None:
    """
    Save a heat-map of the MI matrix (only for n_order=2).

    Args:
        mi (np.ndarray): The mutual information matrix.
        labels (Sequence[str]): The labels for the axes.
        outfile (str, optional): The output file path. Defaults to "mi_heatmap.png".
        cmap (str, optional): The colormap to use. Defaults to "magma".
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



###############################################################################
#                            MAIN ORCHESTRATION                                #
###############################################################################

def compute_and_save_tc(
    samples: np.ndarray,
    names: Optional[Iterable[str]] = None,
    n_order: int = 2,
    bw_method: str | float = "silverman",
    loo: bool = True,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> np.ndarray | list:
    """
    Compute TC, visualise (for n=2), and save all TC values.

    Args:
        samples (np.ndarray): 2D array of shape (n_samples, n_dims).
        names (Optional[Iterable[str]], optional): Names of the parameters. Defaults to None.
        n_order (int, optional): Order of total correlation. Defaults to 2.
        bw_method (str | float, optional): Bandwidth method for KDE. Defaults to "silverman".
        loo (bool, optional): If True, use leave-one-out correction. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.
        out_path (Optional[str], optional): The output directory path. Defaults to None.

    Returns:
        np.ndarray | list: The total correlation results.
    """
    n_dims = samples.shape[1]
    if names is not None:
        labels = [str(n) for n in names]  # ensure JSON‑serialisable strings
    else:
        labels = [f"x{i}" for i in range(n_dims)]

    if len(labels) != n_dims:
        raise ValueError("Length of `names` must match number of dimensions.")

    results = compute_total_correlation(
        samples, n_order=n_order, bw_method=bw_method, loo=loo, seed=seed
    )
    
    if out_path:
        p = Path(out_path)
        p.mkdir(parents=True, exist_ok=True)
    else:
        p = Path(".")

    if n_order == 2:
        mi_matrix = results
        heatmap_path = p / "mi_heatmap.png"
        plot_tc_heatmap(mi_matrix, labels, outfile=heatmap_path)

        tri = np.triu_indices_from(mi_matrix, k=1)
        all_pairs = sorted(zip(tri[0], tri[1], mi_matrix[tri]), key=lambda x: x[2], reverse=True)
        
        logger.info("Top 10 MI pairs:")
        for i, j, val in all_pairs[:10]:
            logger.info("  %s — %s : %.6f", labels[i], labels[j], val)

        mi_path = p / f"params_{n_order}-order_TC.json"
        mi_data = [[labels[i], labels[j], val] for i, j, val in all_pairs]
        with open(mi_path, "w", encoding="utf8") as f:
            json.dump(mi_data, f, indent=2)
        logger.info("All MI pairs saved to %s", mi_path)
    else:
        # Sort results for n > 2
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 %s-order Total Correlation groups:", n_order)
        for indices, val in sorted_results[:10]:
            group_labels = [labels[i] for i in indices]
            logger.info("  %s : %.6f", ", ".join(group_labels), val)

        tc_path = p / f"params_{n_order}-order_TC.json"
        tc_data = [
            [[labels[i] for i in indices], val] for indices, val in sorted_results
        ]
        with open(tc_path, "w", encoding="utf8") as f:
            json.dump(tc_data, f, indent=2)
        logger.info("All %s-order TC values saved to %s", n_order, tc_path)

    return results
