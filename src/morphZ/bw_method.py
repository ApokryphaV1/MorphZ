"""
Bandwidth selection utilities compatible with SciPy's `gaussian_kde`.

This module produces bandwidth *factors* (the scalar `kde.factor`) that are
compatible with `scipy.stats.gaussian_kde`'s convention where

    kernel_covariance = data_covariance * factor**2

Functions provided aim to:
- Return SciPy-compatible **scalar factors** for Scott and Silverman rules.
- Convert per-dimension bandwidths (e.g. from ISJ or diagonal CV) to a single
  SciPy factor by using the geometric-mean ratio `factor = exp(mean(log(h_j/s_j)))`,
  where `h_j` is the per-dimension kernel standard deviation and `s_j` is the
  sample standard deviation (computed with `ddof=1` to match `np.cov`).
- Provide cross-validation routines that directly optimize the log-likelihood
  objective *using* SciPy's `gaussian_kde` for isotropic search, and a
  coordinate-descent diagonal search that scores with `gaussian_kde` on scaled
  data. The diagonal search returns both the per-dimension bandwidths and the
  equivalent SciPy scalar factor.

Notes
-----
- SciPy's `gaussian_kde` expects dataset arrays of shape `(d, n)` where `d`
  is dimensionality and `n` is number of samples. The utilities here accept
  the more common `(n, d)` shape and transpose when calling SciPy.
- For diagonal (per-dimension) bandwidths there's no exact single scalar that
  reproduces arbitrary `h_j`, so we use the geometric-mean conversion which
  yields the factor `f` minimizing multiplicative error in a log-sense.

"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from typing import Literal, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
from scipy.fft import dct
from scipy.optimize import brentq
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold


ArrayLike = Union[np.ndarray, list, tuple]


# -------------------------
# Helpers
# -------------------------


def _to_2d(x: ArrayLike) -> np.ndarray:
    """Return data as shape (n_samples, n_features)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("data must be 1D or 2D array-like")
    return x


def _sample_std_ddof1(x: np.ndarray) -> np.ndarray:
    """Sample standard deviations per-column with ddof=1 (matches np.cov).

    Returns an array of shape (n_features,).
    """
    x = np.asarray(x)
    return np.sqrt(np.var(x, axis=0, ddof=1))


def _robust_scale_1d(x: np.ndarray) -> float:
    """
    Robust per-dimension scale used by Silverman's rule.
    Returns min(std, IQR/1.34) with a tiny epsilon to avoid zeros.
    Note: uses population std (ddof=0) consistent with many references for
    the rule-of-thumb constant; use _sample_std_ddof1 when converting to
    SciPy factors (which use sample covariance internally).
    """
    x = np.asarray(x)
    std = float(np.std(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(std, iqr / 1.34 if iqr > 0 else np.inf)
    return float(max(scale, 1e-12))


# -------------------------
# SciPy-style factors
# -------------------------


def scott_factor(data: ArrayLike) -> float:
    """
    Return the exact factor used by SciPy's gaussian_kde with bw_method='scott'.

    This delegates to gaussian_kde(...).factor to ensure identical behavior
    to the installed SciPy version. No fallback is provided; gaussian_kde is
    required for this project.
    """
    X = _to_2d(data)
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")
    kde = gaussian_kde(X.T, bw_method="scott")
    return float(kde.factor)


def silverman_factor(data: ArrayLike) -> float:
    """
    Return the exact factor used by SciPy's gaussian_kde with bw_method='silverman'.

    This delegates to gaussian_kde(...).factor to ensure identical behavior
    to the installed SciPy version. No fallback is provided; gaussian_kde is
    required for this project.
    """
    X = _to_2d(data)
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")
    kde = gaussian_kde(X.T, bw_method="silverman")
    return float(kde.factor)


# -------------------------
# Scott/Silverman legacy (per-dimension) helpers
# -------------------------


def scott_rule(data: ArrayLike) -> Union[float, np.ndarray]:
    """
    Legacy Scott rule: returns per-dimension numeric bandwidth(s) h_j in data
    units such that h_j = factor * sigma_j where factor = scott_factor and
    sigma_j = std(X[:, j]) (population std).
    """
    X = _to_2d(data)
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")
    factor = scott_factor(X)
    sigmas = np.std(X, axis=0)
    h = factor * sigmas
    if d == 1:
        return float(h[0])
    return np.maximum(h, 1e-12)


def silverman_rule(data: ArrayLike) -> Union[float, np.ndarray]:
    """
    Legacy Silverman rule (robust): per-dimension bandwidths in data units.

    For 1D uses: h = 0.9 * min(std, IQR/1.34) * n**(-1/5)
    For ND uses robust per-dim scales with the multivariate constant.
    """
    X = _to_2d(data)
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")
    if d == 1:
        scale = _robust_scale_1d(X[:, 0])
        return float(0.9 * scale * n ** (-1.0 / 5.0))
    c_d = (4.0 / (d + 2)) ** (1.0 / (d + 4))
    scales = np.array([_robust_scale_1d(X[:, j]) for j in range(d)])
    h = c_d * (n ** (-1.0 / (d + 4))) * scales
    return np.maximum(h, 1e-12)


# ---------------------------------------------
# Botev ISJ (1D) + conversion to SciPy factor
# ---------------------------------------------


def _botev_isj_bandwidth_1d(x: np.ndarray, n_bins: int = 2 ** 12, limits: Tuple[float, float] | None = None) -> float:
    """
    1D ISJ bandwidth (standard deviation of kernel) in data units.
    Implementation follows Botev et al. (2010) with DCT-based fixed-point.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")

    m = int(2 ** np.ceil(np.log2(n_bins)))

    if limits is None:
        xmin, xmax = float(x.min()), float(x.max())
        Dx = xmax - xmin
        if Dx <= 0:
            return 1e-12
        xmin -= Dx / 10.0
        xmax += Dx / 10.0
    else:
        xmin, xmax = float(limits[0]), float(limits[1])
        Dx = xmax - xmin
        if Dx <= 0:
            raise ValueError("limits must define a positive range")

    counts, _ = np.histogram(x, bins=m, range=(xmin, xmax))
    p = counts.astype(float) / N

    a = dct(p, type=2, norm="ortho")
    a2 = (a / 2.0) ** 2

    k = np.arange(1, m, dtype=float)
    k2 = k * k

    def xi_gamma(t: float, L: int = 7) -> float:
        f = 2.0 * (np.pi ** (2 * L)) * np.sum((k2 ** L) * a2[1:] * np.exp(- (np.pi ** 2) * k2 * t))
        for s in range(L - 1, 1, -1):
            K = np.prod(np.arange(1, 2 * s, 2)) / math.sqrt(2.0 * math.pi)
            C = (1.0 + 2.0 ** (-(s + 0.5))) / 3.0
            t = (2.0 * C * K / (N * f)) ** (2.0 / (3.0 + 2.0 * s))
            f = 2.0 * (np.pi ** (2 * s)) * np.sum((k2 ** s) * a2[1:] * np.exp(- (np.pi ** 2) * k2 * t))
        return (2.0 * N * math.sqrt(math.pi) * f) ** (-2.0 / 5.0)

    try:
        t_star = brentq(lambda t: t - xi_gamma(t), 0.0, 0.1)
    except Exception:
        sigma = float(np.std(x))
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        scale = min(sigma, iqr / 1.34 if iqr > 0 else sigma)
        return float(0.9 * max(scale, 1e-12) * (N ** (-1.0 / 5.0)))

    return float(np.sqrt(t_star) * Dx)


def botev_isj_bandwidth(data: ArrayLike, n_bins: int = 2 ** 12, limits: Tuple[float, float] | None = None) -> Union[float, np.ndarray]:
    """
    ISJ bandwidth(s) in data units. For ND returns per-dimension h_j.
    Use `botev_isj_factor` to convert to a SciPy-compatible scalar factor.
    """
    X = _to_2d(data)
    n, d = X.shape
    if d == 1:
        return _botev_isj_bandwidth_1d(X[:, 0], n_bins=n_bins, limits=limits)
    hs = np.array([_botev_isj_bandwidth_1d(X[:, j], n_bins=n_bins, limits=None) for j in range(d)])
    return np.maximum(hs, 1e-12)


def botev_isj_factor(data: ArrayLike, n_bins: int = 2 ** 12) -> float:
    """
    Convert ISJ per-dimension bandwidths to a SciPy `factor`.

    factor = exp(mean(log(h_j / s_j))) where s_j is sample std (ddof=1).
    For 1D this is simply h / s.
    """
    X = _to_2d(data)
    n, d = X.shape
    hs = botev_isj_bandwidth(X, n_bins=n_bins, limits=None)
    if np.isscalar(hs):
        hs = np.array([hs], dtype=float)
    s = _sample_std_ddof1(X)
    s = np.maximum(s, 1e-12)
    ratios = np.maximum(hs, 1e-12) / s
    factor = float(np.exp(np.mean(np.log(ratios))))
    return float(max(factor, 1e-12))


# ---------------------------------------------------
# Cross-validated bandwidth (SciPy-compatible factors)
# ---------------------------------------------------


def cross_validation_bandwidth(
    data: ArrayLike,
    kind: Literal["isotropic", "diagonal"] = "isotropic",
    factor_grid: np.ndarray | None = None,
    cv: int = 10,
    random_state: int | None = None,
    return_diag_h: bool = False,
) -> Union[float, np.ndarray]:
    """
    Cross-validation bandwidth selection that returns a SciPy-compatible
    scalar `factor` by optimizing the summed log-likelihood using
    `scipy.stats.gaussian_kde`.

    Parameters
    ----------
    data : (n_samples, n_features)
    kind : 'isotropic' or 'diagonal'
      - 'isotropic': searches scalar factors directly (best match for SciPy).
      - 'diagonal': performs coordinate-descent search for per-dimension
        bandwidths `h_j` (in data units) by scoring with `gaussian_kde` on
        scaled data; returns the equivalent SciPy scalar factor via geometric mean.
    factor_grid : optional grid of candidate *factors* for isotropic search.
    cv : number of folds
    random_state : RNG seed
    return_diag_h : if True and kind=='diagonal', also return the per-dim h vector

    Returns
    -------
    float (best SciPy factor) or (factor, h_vector) when return_diag_h=True.
    """
    X = _to_2d(data)
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least two samples to compute bandwidth.")

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if kind == "isotropic":
        base = scott_factor(X)
        if factor_grid is None:
            factor_grid = np.logspace(np.log10(base * 0.2), np.log10(base * 5.0), 40)
        else:
            factor_grid = np.asarray(factor_grid, dtype=float)

        best_factor = None
        best_score = -np.inf

        for f in factor_grid:
            total = 0.0
            for train_idx, test_idx in kf.split(X):
                train = X[train_idx].T  # shape (d, n_train)
                test = X[test_idx].T
                kde = gaussian_kde(train)
                # set scalar factor directly
                kde.set_bandwidth(bw_method=float(f))
                if hasattr(kde, "logpdf"):
                    scores = kde.logpdf(test)
                else:
                    # fallback: avoid log(0)
                    vals = kde.evaluate(test)
                    scores = np.log(np.maximum(vals, 1e-300))
                total += float(np.sum(scores))
            if total > best_score:
                best_score = total
                best_factor = float(f)
        return float(best_factor)

    if kind != "diagonal":
        raise ValueError("kind must be 'isotropic' or 'diagonal'")

    # Diagonal search: coordinate descent over multiplicative factors applied to a base
    rng = np.random.RandomState(random_state)
    base_scales = np.array([_robust_scale_1d(X[:, j]) for j in range(d)])
    base_scales[base_scales <= 0] = 1e-3

    if factor_grid is None:
        # multiplicative grid around 1.0
        factor_grid = np.logspace(-0.6, 0.6, 15)
    else:
        factor_grid = np.asarray(factor_grid, dtype=float)

    # initialize h with Silverman per-dim (data units)
    h = silverman_rule(X)
    if np.isscalar(h):
        h = np.repeat(float(h), d)
    else:
        h = np.array(h, dtype=float)

    def cv_score_for_h(h_vec: np.ndarray) -> float:
        h_vec = np.maximum(h_vec, 1e-12)
        X_scaled = X / h_vec.reshape(1, -1)
        total = 0.0
        for train_idx, test_idx in kf.split(X_scaled):
            train = X_scaled[train_idx].T
            test = X_scaled[test_idx].T
            kde = gaussian_kde(train)
            # force factor=1.0 since bandwidth encoded by scaling
            kde.set_bandwidth(bw_method=1.0)
            if hasattr(kde, "logpdf"):
                scores = kde.logpdf(test)
            else:
                vals = kde.evaluate(test)
                scores = np.log(np.maximum(vals, 1e-300))
            total += float(np.sum(scores))
        return float(total)

    best_score = cv_score_for_h(h)
    improved = True
    max_outer_iters = 5

    for _ in range(max_outer_iters):
        if not improved:
            break
        improved = False
        for j in range(d):
            current = float(h[j])
            candidates = base_scales[j] * factor_grid
            if not np.any(np.isclose(candidates, current)):
                candidates = np.unique(np.sort(np.append(candidates, current)))
            best_j = current
            best_j_score = best_score
            for cand in candidates:
                h_try = h.copy()
                h_try[j] = float(max(cand, 1e-12))
                s = cv_score_for_h(h_try)
                if s > best_j_score + 1e-12:
                    best_j_score = s
                    best_j = float(cand)
            if not np.isclose(best_j, current):
                h[j] = best_j
                best_score = best_j_score
                improved = True

    # convert per-dimension h to SciPy factor f via geometric mean of ratios h_j / s_j
    s = _sample_std_ddof1(X)
    s = np.maximum(s, 1e-12)
    ratios = np.maximum(h, 1e-12) / s
    factor = float(np.exp(np.mean(np.log(ratios))))

    if return_diag_h:
        return float(factor), np.maximum(h, 1e-12)
    return float(factor)


# -------------------------
# Unified selector + saver (SciPy-compatible factors)
# -------------------------


def select_bandwidth(
    data: ArrayLike,
    method: Literal["scott", "silverman", "isj", "cv_iso", "cv_diag"] = "silverman",
) -> Union[float, np.ndarray]:
    """
    Unified bandwidth selector returning SciPy-compatible *scalar factors*.

    - 'scott': returns scott_factor
    - 'silverman': returns silverman_factor
    - 'isj': returns botev_isj_factor
    - 'cv_iso': cross-validated isotropic factor (optimizes SciPy log-lik)
    - 'cv_diag': diagonal CV -> returns SciPy factor (geometric mean); use
      `select_bandwidth(...);` with `method='cv_diag'` to get scalar factor, or
      call `cross_validation_bandwidth(..., kind='diagonal', return_diag_h=True)`
      to also get per-dimension h_j.
    """
    if method == "scott":
        return scott_factor(data)
    if method == "silverman":
        return silverman_factor(data)
    if method == "isj":
        return botev_isj_factor(data)
    if method == "cv_iso":
        return cross_validation_bandwidth(data, kind="isotropic")
    if method == "cv_diag":
        return cross_validation_bandwidth(data, kind="diagonal")
    raise ValueError("Unknown method: %r" % (method,))


def compute_and_save_bandwidths(
    data: ArrayLike,
    method: Literal["scott", "silverman", "isj", "cv_iso", "cv_diag"],
    param_names: list[str],
    output_path: str,
    n_order: int,
    in_path: str | None = None,
    group_format: Literal["pairs", "groups"] = "pairs",
    top_k_greedy: int = 1,
    
) -> list[list[Union[str, float]]]:
    """
    Compute SciPy-compatible scalar factors for each parameter or group and
    save a simplified JSON file. Each entry is [param1, ..., factor].

    Note: for group-level joint estimation we produce a single scalar `factor`
    that can be passed to `gaussian_kde.set_bandwidth(factor)` for subsequent
    evaluations. For diagonal/group-per-dimension exact control consider
    returning per-dim h_j and applying them with a custom KDE wrapper.
    
    Parameters
    ----------
    data : (n_samples, n_features)
    method : {'scott','silverman','isj','cv_iso','cv_diag'}
    param_names : list[str]
    output_path : str
    n_order : int
        Group order (2 for pairs, k for k-groups) used only for output filename.
    in_path : str | None
        Optional path to MI/TC JSON (pairs or groups). When provided, groups
        are selected greedily from this candidate list.
    group_format : {'pairs','groups'}
        Format of `in_path` entries.
    top_k_greedy : int
        Run K seeded greedy selections starting from each of the top-K
        candidates (by MI or TC), then keep the selection with the largest
        total score. Default 1 (single greedy pass).
    """
    X = _to_2d(data)
    n_samples, n_features = X.shape

    if len(param_names) != n_features:
        raise ValueError("Number of parameter names must match number of features")

    if method not in ["scott", "silverman", "isj", "cv_iso", "cv_diag"]:
        raise ValueError("Unsupported method")

    # parse groups if provided
    selected_groups = []  # list of dicts {"names": tuple(...), "tc": float}
    singles = list(param_names)

    if in_path is not None:
        with open(in_path, "r", encoding="utf-8") as f:
            raw_groups = json.load(f)

        parsed = []
        if group_format == "pairs":
            for entry in raw_groups:
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    raise ValueError("Each entry in pairs format must be [p1, p2, value]")
                p1, p2, val = entry
                parsed.append(([p1, p2], float(val)))
        else:
            for entry in raw_groups:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError("Each entry in groups format must be [group, tc]")
                group, tc = entry
                if not isinstance(group, (list, tuple)):
                    raise ValueError("Group must be a list/tuple of parameter identifiers")
                parsed.append((list(group), float(tc)))

        param_map = {name: i for i, name in enumerate(param_names)}

        canonical = []
        for group, tc in parsed:
            def _to_name(x):
                if isinstance(x, (int, np.integer)):
                    idx = int(x)
                    if idx < 0 or idx >= n_features:
                        raise IndexError("Index out of bounds")
                    return param_names[idx]
                if isinstance(x, str):
                    if x not in param_map:
                        raise KeyError(f"Parameter name {x!r} not found")
                    return x
                raise TypeError("Unsupported parameter identifier type")

            named_group = tuple(sorted([_to_name(p) for p in group]))
            canonical.append((named_group, float(tc)))

        canonical.sort(key=lambda t: -t[1])

        # Seeded greedy selection aligned with Morph_Pairwise/Morph_Group
        def _run_seed(seed_idx: int):
            pool = set(param_names)
            selection = []
            total = 0.0
            if 0 <= seed_idx < len(canonical):
                names, val = canonical[seed_idx]
                if all(name in pool for name in names):
                    selection.append({"names": names, "tc": float(val)})
                    for n in names:
                        pool.remove(n)
                    total += float(val)
            for names, val in canonical:
                if all(name in pool for name in names):
                    selection.append({"names": names, "tc": float(val)})
                    for n in names:
                        pool.remove(n)
                    total += float(val)
            return selection, sorted(pool), total

        if top_k_greedy is None or top_k_greedy <= 1 or len(canonical) == 0:
            pool = set(param_names)
            for names, val in canonical:
                if all(name in pool for name in names):
                    selected_groups.append({"names": names, "tc": float(val)})
                    for n in names:
                        pool.remove(n)
            singles = sorted(pool)
        else:
            K = min(int(top_k_greedy), len(canonical))
            best_sel, best_singles = [], param_names[:]  # placeholders
            best_key = None  # (total, n_groups, -seed_idx)
            for seed_idx in range(K):
                sel_i, singles_i, total_i = _run_seed(seed_idx)
                key = (total_i, len(sel_i), -seed_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_sel, best_singles = sel_i, singles_i
            selected_groups, singles = best_sel, best_singles
    

    # Compute factors and build structured payload mirroring selected_* files
    bandwidths: dict[str, float] = {}
    groups_out: list[dict] = []

    # groups/pairs factors
    for group_info in selected_groups:
        group_names = list(group_info["names"])
        indices = [param_names.index(n) for n in group_names]
        group_data = X[:, indices]

        if method == "scott":
            joint_factor = scott_factor(group_data)
        elif method == "silverman":
            joint_factor = silverman_factor(group_data)
        elif method == "isj":
            joint_factor = botev_isj_factor(group_data)
        elif method == "cv_iso":
            joint_factor = float(cross_validation_bandwidth(group_data, kind="isotropic"))
        elif method == "cv_diag":
            fac = cross_validation_bandwidth(group_data, kind="diagonal")
            joint_factor = float(fac)
        else:
            raise ValueError("Unsupported method for groups")

        groups_out.append({"names": group_names, "bw": float(joint_factor)})
        for name in group_names:
            bandwidths[name] = float(joint_factor)

    # singles factors
    singles_out: list[dict] = []
    for param_name in singles:
        param_idx = param_names.index(param_name)
        param_data = X[:, param_idx]

        if method == "scott":
            bw = scott_factor(param_data)
        elif method == "silverman":
            bw = silverman_factor(param_data)
        elif method == "isj":
            bw = botev_isj_factor(param_data)
        elif method == "cv_iso":
            bw = float(cross_validation_bandwidth(param_data, kind="isotropic"))
        elif method == "cv_diag":
            bw = float(cross_validation_bandwidth(X, kind="diagonal"))
        else:
            raise ValueError("Unsupported method")

        singles_out.append({"name": param_name, "bw": float(bw)})
        bandwidths[param_name] = float(bw)

    # Assemble payload: use 'pairs' when n_order==2 and group_format=='pairs'; else 'groups'
    top_key = "pairs" if (n_order == 2 and group_format == "pairs") else "groups"
    payload = {
        top_key: groups_out,
        "singles": singles_out,
    }
    if top_key == "groups":
        payload["n_order"] = int(n_order)

    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f"bw_{method}_{n_order}D.json")
    # Serialize first to avoid partial file on error
    content = json.dumps(payload, indent=2)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Bandwidths saved to %s", output_filename)
    if in_path is not None:
        logger.info(
            "Processed %s %s and %s single parameters",
            len(selected_groups),
            top_key,
            len(singles),
        )

    # Return bandwidth mapping for optional immediate use (used as overrides)
    return bandwidths


# quick smoke test when run as script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState(0)
    x = rng.normal(size=200)
    logger.info("Scott factor (1D): %s", scott_factor(x))
    logger.info("Silverman factor (1D): %s", silverman_factor(x))
    logger.info("Legacy Scott h (1D): %s", scott_rule(x))
    logger.info("Legacy Silverman h (1D): %s", silverman_rule(x))
    logger.info("ISJ h (1D): %s", botev_isj_bandwidth(x, n_bins=2 ** 10))
    logger.info("ISJ factor (1D): %s", botev_isj_factor(x, n_bins=2 ** 10))
    X = rng.normal(size=(200, 3))
    logger.info("CV isotropic factor (3D): %s", cross_validation_bandwidth(X, kind="isotropic", cv=4))
    logger.info("CV diagonal factor (3D): %s", cross_validation_bandwidth(X, kind="diagonal", cv=4))
