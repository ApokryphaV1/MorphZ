import numpy as np
import json
import os
from typing import Callable, Dict, List, Optional, Union
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
    max_iter: int = 50000,
    tol: float = 1e-4,
    morph_type: MorphType = "indep",
    param_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    n_estimations: int = 1,
    kde_bw: Optional[Union[BandwidthMethod, float, Dict[str, float]]] = None,
    verbose: bool = False,
    top_k_greedy: int = None,
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
             print(f"Setting top_k_greedy to {top_k_greedy} for pairs selection.")
        elif "group" in morph_type:
            n_order = int(morph_type.split("_")[0])
            top_k_greedy = int(np.sqrt(comb(ndim, n_order))) # sqrt of number of possible groups
            if verbose:
                print(f"Setting top_k_greedy to {top_k_greedy} for {n_order}-groups selection.")

    if param_names is None:
                param_names = [f'param_{i}' for i in range(ndim)]

    if morph_type == "indep":
        print("\nUsing independent KDE for proposal distribution.")
        if bw_is_numeric:
            if verbose:
                print(f"\nKDE bandwidth method: {kde_bw} (numeric: {bw_is_numeric})")
            # Pass numeric bandwidth directly; do not compute or load JSON
            target_kde = Morph_Indep(kde_samples, kde_bw=kde_bw, param_names=param_names, verbose=verbose, bw_json_path=None)
        else:
            method_name = kde_bw  # Store original method name
            bw_json_path= f"{output_path}/bw_{method_name}_1D.json"

            if not os.path.exists(bw_json_path):
                print(f"BW file not found at {bw_json_path}. Running Bw with {method_name}...")

                kde_bw = compute_and_save_bandwidths(kde_samples, method=method_name, param_names=param_names,n_order=1, output_path=output_path)
            target_kde = Morph_Indep(kde_samples, kde_bw=kde_bw, param_names=param_names,verbose=verbose, bw_json_path=bw_json_path)
        log_proposal_pdf = target_kde.logpdf_kde

    elif morph_type == "pair":
        print("\nUsing Morph_Pairwise for proposal distribution.")
        mi_file = f"{output_path}/params_MI.json"
        if param_names is None:
                param_names = [f'param_{i}' for i in range(ndim)]

        if not os.path.exists(mi_file):
            print(f"MI file not found at {mi_file}. Running dependency tree computation...")
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
                    print(f"Warning: failed to write selected_pairs.json: {e}")
        else:
            method_name = kde_bw  # Store original method name

            bw_json_path= f"{output_path}/bw_{method_name}_2D.json"
            
            if not os.path.exists(bw_json_path):
                print(f"BW file not found at {bw_json_path}. Running Bw with {method_name}...")
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
                    print(f"Warning: failed to write selected_pairs.json: {e}")
        log_proposal_pdf = target_kde.logpdf

    elif "group" in morph_type:
        print("\nUsing Morph_Group for proposal distribution.")
        n_order = int(morph_type.split("_")[0])
        group_file = f"{output_path}/params_{n_order}-order_TC.json"
        if param_names is None:
                param_names = [f'param_{i}' for i in range(ndim)]
        if not os.path.exists(group_file):
            print(f"Group file not found at {group_file}. Running total correlation computation...")

            Nth_TC.compute_and_save_tc(samples,names=param_names,n_order=n_order,out_path=output_path)

        # Convert group file format if needed for bandwidth computation
        import json
        with open(group_file, 'r') as f:
            group_data = json.load(f)
        if bw_is_numeric:
            if verbose:
                print(f"\nKDE bandwidth method: {kde_bw} (numeric: {bw_is_numeric})")
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
                    print(f"Warning: failed to write selected_{n_order}-order_group.json: {e}")
        else:
            method_name = kde_bw  # Store original method name
            bw_json_path= f"{output_path}/bw_{method_name}_{n_order}D.json"

            if not os.path.exists(bw_json_path):
                print(f"BW file not found at {bw_json_path}. Running Bw with {method_name}...")
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
                    print(f"Warning: failed to write selected_{n_order}-order_group.json: {e}")
        log_proposal_pdf = target_kde.logpdf

    elif morph_type == "tree":
        print("\nUsing Morph_Tree for proposal distribution.")
        tree_file = f"{output_path}/tree.json"
        if param_names is None:
            param_names = [f'param_{i}' for i in range(ndim)]
        if not os.path.exists(tree_file):
            print(f"Tree file not found at {tree_file}. "
                  "Running dependency tree computation... might take a while for higher dimensions. for faster results, use fewer samples per param.")
            dependency_tree.compute_and_plot_mi_tree(samples, names=param_names, out_path=output_path, morph_type="tree")

        if bw_is_numeric:
            # Direct numeric bandwidth; do not compute JSON bandwidths
            target_kde = Morph_Tree(kde_samples, tree_file=tree_file, param_names=param_names, kde_bw=kde_bw, bw_json_path=None)
        else:
            method_name = kde_bw  # Store original method name
            kde_bw = compute_and_save_bandwidths(kde_samples, method=method_name, param_names=param_names,n_order= 2, output_path=output_path)
            # Pass the JSON path to KDE class for automatic bandwidth loading
            bw_json_path = f"{output_path}/bw_{method_name}_2D.json"
            target_kde = Morph_Tree(kde_samples, tree_file=tree_file, param_names=param_names, kde_bw=kde_bw, bw_json_path=bw_json_path)
        log_proposal_pdf = target_kde.logpdf
    else:
        raise ValueError(f"Unknown morph_type: {morph_type}. Supported types are 'indep', 'pair', and 'tree'.")

    bridge_start_index = int(tot_len * bridge_start_fraction)
    samples_mor = samples[bridge_start_index:, :]
    log_post = log_prob[bridge_start_index:]

    all_log_z_results = []
    for i in range(n_estimations):
        
        samples_prop = target_kde.resample(n_resamples)
        print(f"\nEstimation {i+1}/{n_estimations}")
        log_z_results = bridge_sampling_ln(
            log_posterior_function,
            log_proposal_pdf,
            samples_mor,
            log_post,
            samples_prop,
            tol=tol,
            max_iter=max_iter
        )
        all_log_z_results.append(log_z_results)
        
    # Save log(z) and error
    logz_path = f"{output_path}/logz_morph_z_{morph_type}_{kde_bw_name}.txt"
    header = "logz err"
    np.savetxt(logz_path, np.array(all_log_z_results), header=header, fmt='%f', comments='')
    print(f"\nSaved log(z) to {logz_path}")

    return all_log_z_results
