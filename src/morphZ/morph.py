import json
import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
try:
    # Literal is available in Python 3.8+
    from typing import Literal
except Exception:  # pragma: no cover
    Literal = lambda *args, **kwargs: str  # type: ignore

try:
    from sksparse.cholmod import CholmodNotPositiveDefiniteError
except ImportError:
    # Define a dummy exception if sksparse is not available,
    # so the code doesn't crash if the dependency is missing.
    class CholmodNotPositiveDefiniteError(RuntimeError):
        pass

from scipy.stats import norm
from scipy.special import logsumexp
from . import utils
from .morph_indep import Morph_Indep
from .morph_tree import Morph_Tree
from .morph_pairwise import Morph_Pairwise
from . import dependency_tree
from .morph_group import Morph_Group
from . import Nth_TC
from .bw_method import compute_and_save_bandwidths
from .bridge import bridge_sampling_ln, compute_bridge_rmse

logger = logging.getLogger(__name__)


def _save_corner_plot(
    posterior_samples: np.ndarray,
    proposal_samples: np.ndarray,
    param_names: Optional[List[str]],
    output_path: str,
    morph_type: str,
    verbose: bool,
    prefer_corner: bool,
) -> None:
    """Render and save a corner plot comparing posterior and proposal samples."""
    if not prefer_corner:
        if verbose:
            logger.info("Skipping corner plot because prefer_corner=False")
        return

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        msg = f"Matplotlib is required to save corner plots: {exc}"
        warnings.warn(msg, RuntimeWarning)
        if verbose:
            logger.warning(msg)
        return

    try:
        import corner as corner_lib  # type: ignore
    except ImportError:
        msg = "corner package is not installed; skipping corner plot."
        warnings.warn(msg, RuntimeWarning)
        if verbose:
            logger.warning("%s (morph type: %s)", msg, morph_type)
        return

    def _downsample(arr: np.ndarray, max_points: int = 50000) -> np.ndarray:
        n = arr.shape[0]
        if n <= max_points:
            return arr
        idx = np.random.choice(n, size=max_points, replace=False)
        return arr[idx]

    post_plot = _downsample(posterior_samples)
    prop_plot = _downsample(proposal_samples)
    names = param_names if param_names is not None else [f"param_{j}" for j in range(post_plot.shape[1])]

    fig = corner_lib.corner(
        post_plot,
        bins=50,
        color="black",
        labels=names,
        label_kwargs={"fontsize": 7},
        hist_kwargs={"density": True},
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_fmt=".2f",
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.5, 0.8, 0.95),
        smooth=0.9,
        pcolor_kwargs=dict(alpha=0.06),
        contour_kwargs=dict(alpha=0.4, colors=["black"]),
    )
    corner_lib.corner(
        prop_plot,
        bins=50,
        color="red",
        labels=names,
        label_kwargs={"fontsize": 7},
        hist_kwargs={"density": True},
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_fmt=".2f",
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.5, 0.8, 0.95),
        smooth=0.9,
        pcolor_kwargs=dict(alpha=0.04),
        contour_kwargs=dict(alpha=0.4, colors=["red"]),
        fig=fig,
    )

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=6, label="Posterior samples"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, label="Morph samples"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", frameon=False)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    corner_fname = os.path.join(output_path, f"corner_{morph_type}.png")
    fig.savefig(corner_fname, dpi=150)
    plt.close(fig)
    if verbose:
        logger.info("Saved corner plot via 'corner' to %s", corner_fname)

# ----- Typing helpers for better IDE hovers -----
# Bandwidth methods supported by bw_method.py
BandwidthMethod = Literal["scott", "silverman", "isj", "cv_iso", "cv_diag"]

# Common proposal types
MorphTypeBase = Literal["indep", "pair", "tree"]
# Frequently used grouped variants (extend if you use others regularly)
MorphTypeGroup = Literal["2_group", "3_group", "4_group", "5_group"]
# Final type shown to users in hovers; still accept arbitrary strings at runtime
MorphType = Union[MorphTypeBase, MorphTypeGroup, str]


def evidence(
    post_samples: np.ndarray,
    log_posterior_values: np.ndarray,
    log_posterior_function: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    thin: int = 1,
    kde_fraction: float = 0.5,
    bridge_start_fraction: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-2,
    morph_type: MorphType = "indep",
    param_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    n_estimations: int = 1,
    kde_bw: Optional[Union[BandwidthMethod, float, Dict[str, float]]] = None,
    verbose: bool = False,
    top_k_greedy: int = None,
    plot: bool = True,
    prefer_corner: bool = True,
) -> List[List[float]]:
    """
    Compute log evidence using morphological bridge sampling with KDE proposals.

    This orchestrates proposal construction (independent, pairwise, grouped, or
    tree‑structured KDE), draws proposal samples, and performs bridge sampling.

    Args:
        post_samples (ndarray): Posterior samples of shape ``(N, d)``.
        log_posterior_values (ndarray): Log posterior values for ``post_samples``;
            shape ``(N,)``. Used to avoid re‑evaluating expensive log posteriors.
        log_posterior_function (callable): Callable taking a single vector
            ``theta`` with shape ``(d,)`` and returning the log target density.
        n_resamples (int): Number of proposal samples per estimation. Typical
            range 500–5000 depending on dimensionality.
        thin (int): Thinning stride applied to both ``post_samples`` and
            ``log_posterior_values`` to reduce autocorrelation. Use ``thin>1`` if
            your MCMC is highly autocorrelated.
        kde_fraction (float): Fraction of the (thinned) chain used to fit the
            proposal KDE(s). Remaining samples are reserved for the bridge.
            Values between 0.3 and 0.7 are common.
        bridge_start_fraction (float): Fraction of (thinned) chain index from
            which the bridge uses the posterior samples. E.g., ``0.5`` starts at
            the latter half to reduce dependence with KDE fitting.
        max_iter (int): Max bridge iterations.
        tol (float): Convergence tolerance for the bridge fixed‑point update.
        morph_type (MorphType): Proposal family. Options shown on hover:
            - ``"indep"``: Product of 1D KDEs (fast, robust; assumes weak deps).
            - ``"pair"``: Greedy pairwise KDEs using MI ranking (good for moderate deps).
            - ``"tree"``: Chow–Liu tree KDE (captures a single dependency tree).
            - ``"{k}_group"``: Group KDE using k‑order total correlation groups
              from ``Nth_TC`` (e.g., ``"3_group"``). Common values: ``"2_group"``,
              ``"3_group"``, ``"4_group"``, ``"5_group"``.
        param_names (list[str] | None): Optional names for parameters; used for
            bandwidth JSONs and reporting. Defaults to ``["param_i"]``.
        output_path (str | None): Directory for artifacts (bandwidth JSONs,
            dependency files, and results). Defaults to ``"log_MorphZ"``.
        n_estimations (int): Number of independent bridge estimates to run. Use
            >1 to gauge variability; results are saved as a 2‑column text file
            with ``logz`` and ``err`` per row.
        kde_bw (BandwidthMethod | float | dict | None): Bandwidth selector/factor.
            Supported selectors from ``bw_method.py``: ``'scott'``, ``'silverman'``,
            ``'isj'`` (Botev’s ISJ), ``'cv_iso'`` (isotropic CV), ``'cv_diag'``
            (diagonal CV → scalar factor). You can also pass a number (e.g., 0.9)
            or a ``{name: value}`` dict to override specific parameters when
            using ``bw_json_path``.
        verbose (bool): Print fitting details for KDE components.
        top_k_greedy (int): For ``pair`` and ``*_group`` morph types, run K
            seeded greedy selections starting from each of the top‑K candidates
            (by MI or TC), and keep the selection with the highest total score.
            Default is 1 (single greedy pass as before).
        plot (bool): If True, saves a corner plot comparing posterior samples
            (black) and proposal samples (red) as a PNG in ``output_path``.
            The plot is not displayed.
        prefer_corner (bool): If True, attempts to render the corner plot using
            the ``corner`` package with filled contours and the following style:
            ``bins=50``, ``quantiles=(0.05,0.5,0.95)``, ``show_titles=True``,
            ``title_fmt='.2f'``, ``plot_datapoints=True``, ``fill_contours=True``,
            ``levels=(0.5,0.8,0.95)``, ``smooth=0.9``. If ``corner`` is not
            installed, a warning is emitted and the plot is skipped.

    Returns:
        list[[float, float]]: A list of ``[logz, err]`` for each estimation.

    Suggestions:
        - Start with ``morph_type='indep'`` for speed. If diagnostics look poor,
          try ``'pair'`` or ``'tree'`` to capture dependencies.
        - For bandwidths, try ``'silverman'`` for speed, ``'cv_iso'`` for tighter
          fits, or ``'isj'`` as a robust nonparametric choice.
        - Use ``n_estimations>=3`` to assess stability and report mean/SE.
    """

    
    kde_bw_name = kde_bw
    samples = post_samples[::thin, :]
    log_prob = log_posterior_values[::thin]

    tot_len, ndim = samples.shape

    if output_path is None:
        output_path = "log_MorphZ"
    
    os.makedirs(output_path, exist_ok=True)


    kde_samples = samples[:int(tot_len * kde_fraction), :]

    # Use user-provided kde_bw or default to "silverman"
    if kde_bw is None:
        kde_bw = "silverman"
    
    # Detect numeric bandwidths; if numeric, skip bw_method/JSON logic and pass directly
    bw_is_numeric = isinstance(kde_bw, (float, int, np.floating))
    
    if top_k_greedy is None:
        from math import comb
        if morph_type == "pair":
            top_k_greedy = comb(ndim, 2)
            if verbose:
                logger.info("Setting top_k_greedy to %s for pairs selection.", top_k_greedy)
        elif "group" in morph_type:
            n_order = int(morph_type.split("_")[0])
            top_k_greedy = int(np.sqrt(comb(ndim, n_order))) # sqrt of number of possible groups
            if verbose:
                logger.info("Setting top_k_greedy to %s for %s-groups selection.", top_k_greedy, n_order)

    if param_names is None:
        param_names = [f"param_{i}" for i in range(ndim)]

    if morph_type == "indep":
        logger.info("Using independent KDE for proposal distribution.")
        if bw_is_numeric:
            if verbose:
                logger.info("KDE bandwidth method: %s (numeric: %s)", kde_bw, bw_is_numeric)
            # Pass numeric bandwidth directly; do not compute or load JSON
            target_kde = Morph_Indep(kde_samples, kde_bw=kde_bw, param_names=param_names, verbose=verbose, bw_json_path=None)
        else:
            method_name = kde_bw  # Store original method name
            bw_json_path = f"{output_path}/bw_{method_name}_1D.json"

            if not os.path.exists(bw_json_path):
                logger.info("BW file not found at %s. Running Bw with %s...", bw_json_path, method_name)

                kde_bw = compute_and_save_bandwidths(
                    kde_samples,
                    method=method_name,
                    param_names=param_names,
                    n_order=1,
                    output_path=output_path,
                )
            target_kde = Morph_Indep(
                kde_samples,
                kde_bw=kde_bw,
                param_names=param_names,
                verbose=verbose,
                bw_json_path=bw_json_path,
            )
        log_proposal_pdf = target_kde.logpdf_kde

    elif morph_type == "pair":
        logger.info("Using Morph_Pairwise for proposal distribution.")
        mi_file = f"{output_path}/params_MI.json"
        if param_names is None:
            param_names = [f"param_{i}" for i in range(ndim)]

        if not os.path.exists(mi_file):
            logger.info("MI file not found at %s. Running dependency tree computation...", mi_file)
            dependency_tree.compute_and_plot_mi_tree(samples, names=param_names, out_path=output_path, morph_type="pair")

        if bw_is_numeric:
            # Direct numeric bandwidth; skip JSON computation
            target_kde = Morph_Pairwise(
                kde_samples,
                param_mi=mi_file,
                param_names=param_names,
                kde_bw=kde_bw,
                verbose=verbose,
                bw_json_path=None,
                top_k_greedy=top_k_greedy,
            )
            # Save selected pairs/singles
            try:
                selected_pairs_path = os.path.join(output_path, "selected_pairs.json")
                sel = {
                    "pairs": [{"names": [a, b], "mi": float(mi)} for (a, b, mi) in getattr(target_kde, "pairs", [])],
                    "singles": list(getattr(target_kde, "singles", [])),
                }
                # Build JSON string first to avoid truncating file on failure
                content = json.dumps(sel, indent=2)
                with open(selected_pairs_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:  # pragma: no cover
                if verbose:
                    logger.warning("Failed to write selected_pairs.json: %s", e)
        else:
            method_name = kde_bw  # Store original method name

            bw_json_path = f"{output_path}/bw_{method_name}_2D.json"
            
            if not os.path.exists(bw_json_path):
                logger.info("BW file not found at %s. Running Bw with %s...", bw_json_path, method_name)
                kde_bw = compute_and_save_bandwidths(
                    kde_samples,
                    method=method_name,
                    param_names=param_names,
                    output_path=output_path,
                    n_order=2,
                    in_path=mi_file,
                    group_format="pairs",
                    top_k_greedy=top_k_greedy,
                )
            # Pass the JSON path to KDE class for automatic bandwidth loading
            target_kde = Morph_Pairwise(
                kde_samples,
                param_mi=mi_file,
                param_names=param_names,
                kde_bw=kde_bw,
                verbose=verbose,
                bw_json_path=bw_json_path,
                top_k_greedy=top_k_greedy,
            )
            # Save selected pairs/singles
            try:
                selected_pairs_path = os.path.join(output_path, "selected_pairs.json")
                sel = {
                    "pairs": [{"names": [a, b], "mi": float(mi)} for (a, b, mi) in getattr(target_kde, "pairs", [])],
                    "singles": list(getattr(target_kde, "singles", [])),
                }
                content = json.dumps(sel, indent=2)
                with open(selected_pairs_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:  # pragma: no cover
                if verbose:
                    logger.warning("Failed to write selected_pairs.json: %s", e)
        log_proposal_pdf = target_kde.logpdf

    elif "group" in morph_type:
        logger.info("Using Morph_Group for proposal distribution.")
        n_order = int(morph_type.split("_")[0])
        group_file = f"{output_path}/params_{n_order}-order_TC.json"
        if param_names is None:
            param_names = [f"param_{i}" for i in range(ndim)]
        if not os.path.exists(group_file):
            logger.info("Group file not found at %s. Running total correlation computation...", group_file)
            Nth_TC.compute_and_save_tc(samples, names=param_names, n_order=n_order, out_path=output_path)

        if bw_is_numeric:
            if verbose:
                logger.info("KDE bandwidth method: %s (numeric: %s)", kde_bw, bw_is_numeric)
            target_kde = Morph_Group(
                kde_samples,
                group_file,
                param_names=param_names,
                kde_bw=kde_bw,
                verbose=verbose,
                bw_json_path=None,
                top_k_greedy=top_k_greedy,
            )
            # Save selected groups/singles
            try:
                selected_group_path = os.path.join(output_path, f"selected_{n_order}-order_group.json")
                sel = {
                    "groups": [{"names": list(g.get("names", ())), "tc": float(g.get("tc", 0.0))} for g in getattr(target_kde, "groups", [])],
                    "singles": list(getattr(target_kde, "singles", [])),
                    "n_order": int(n_order),
                }
                content = json.dumps(sel, indent=2)
                with open(selected_group_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:  # pragma: no cover
                if verbose:
                    logger.warning("Failed to write selected_%s-order_group.json: %s", n_order, e)
        else:
            method_name = kde_bw  # Store original method name
            bw_json_path = f"{output_path}/bw_{method_name}_{n_order}D.json"

            if not os.path.exists(bw_json_path):
                logger.info("BW file not found at %s. Running Bw with %s...", bw_json_path, method_name)
                kde_bw = compute_and_save_bandwidths(
                    kde_samples,
                    method=method_name,
                    param_names=param_names,
                    n_order=n_order,
                    output_path=output_path,
                    in_path=group_file,
                    group_format="groups",
                    top_k_greedy=top_k_greedy,
                )

            target_kde = Morph_Group(
                kde_samples,
                group_file,
                param_names=param_names,
                kde_bw=kde_bw,
                verbose=verbose,
                bw_json_path=bw_json_path,
                top_k_greedy=top_k_greedy,
            )
            # Save selected groups/singles
            try:
                selected_group_path = os.path.join(output_path, f"selected_{n_order}-order_group.json")
                sel = {
                    "groups": [{"names": list(g.get("names", ())), "tc": float(g.get("tc", 0.0))} for g in getattr(target_kde, "groups", [])],
                    "singles": list(getattr(target_kde, "singles", [])),
                    "n_order": int(n_order),
                }
                content = json.dumps(sel, indent=2)
                with open(selected_group_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:  # pragma: no cover
                if verbose:
                    logger.warning("Failed to write selected_%s-order_group.json: %s", n_order, e)
        log_proposal_pdf = target_kde.logpdf

    elif morph_type == "tree":
        logger.info("Using Morph_Tree for proposal distribution.")
        tree_file = f"{output_path}/tree.json"
        if param_names is None:
            param_names = [f"param_{i}" for i in range(ndim)]
        if not os.path.exists(tree_file):
            logger.info(
                "Tree file not found at %s. Running dependency tree computation... might take a while for higher dimensions. for faster results, use fewer samples per param.",
                tree_file,
            )
            dependency_tree.compute_and_plot_mi_tree(samples, names=param_names, out_path=output_path, morph_type="tree")

        if bw_is_numeric:
            # Direct numeric bandwidth; do not compute JSON bandwidths
            target_kde = Morph_Tree(kde_samples, tree_file=tree_file, param_names=param_names, kde_bw=kde_bw, bw_json_path=None)
        else:
            method_name = kde_bw  # Store original method name
            kde_bw = compute_and_save_bandwidths(
                kde_samples,
                method=method_name,
                param_names=param_names,
                n_order=2,
                output_path=output_path,
            )
            # Pass the JSON path to KDE class for automatic bandwidth loading
            bw_json_path = f"{output_path}/bw_{method_name}_2D.json"
            target_kde = Morph_Tree(kde_samples, tree_file=tree_file, param_names=param_names, kde_bw=kde_bw, bw_json_path=bw_json_path)
        log_proposal_pdf = target_kde.logpdf
    else:
        raise ValueError(f"Unknown morph_type: {morph_type}. Supported types are 'indep', 'pair', and 'tree'.")

    bridge_start_index = int(tot_len * bridge_start_fraction)
    samples_mor = samples[bridge_start_index:, :]
    log_post = log_prob[bridge_start_index:]
    # Optional: corner-style comparison plot of posterior vs proposal
        
    if plot :
        samples_prop = target_kde.resample(n_resamples)

        _save_corner_plot(
            samples,
            samples_prop,
            param_names,
            output_path,
            morph_type,
            verbose,
            prefer_corner,
        )
    logz_path = f"{output_path}/logz_morph_z_{morph_type}_{kde_bw_name}.txt"
    header = "logz err"
    all_log_z_results = []
    for i in range(n_estimations):

        samples_prop = target_kde.resample(n_resamples)
        estimation_label = f"Estimation {i+1}/{n_estimations}"
        if i > 0:
            print()
        logger.debug(estimation_label)
        log_z_results = bridge_sampling_ln(
            log_posterior_function,
            log_proposal_pdf,
            samples_mor,
            log_post,
            samples_prop,
            tol=tol,
            max_iter=max_iter,
            estimation_label=estimation_label,
            verbose=verbose,
        )
        all_log_z_results.append(log_z_results)

        # Persist partial results every iteration so progress survives interruptions.
        row = np.array([log_z_results], dtype=float)
        mode = "w" if i == 0 else "a"
        with open(logz_path, mode, encoding="utf-8") as f:
            if i == 0:
                f.write(f"{header}\n")
            np.savetxt(f, row, fmt="%f", comments="", delimiter=" ")

    final_msg = f"\nSaved log(z) to {logz_path}"
    print(final_msg)
    logger.debug(final_msg.strip())

    return all_log_z_results
