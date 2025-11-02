"""
pairwise_kde.py -- greedy pairwise KDE approximation (see module docstring in code).
"""
from __future__ import annotations
import json
import logging
import os
from typing import List, Tuple, Union, Dict, Any
import numpy as np
from scipy.stats import gaussian_kde
from .kde_base import KDEBase

logger = logging.getLogger(__name__)


class Morph_Pairwise(KDEBase):
    """
    Greedy pairwise KDE approximation guided by mutual information (MI).

    Pairs with highest MI are modeled jointly with 2D KDEs; remaining variables
    are modeled as independent 1D KDEs. This balances flexibility and speed.

    Use ``dependency_tree.compute_and_plot_mi_tree`` or ``Nth_TC.compute_and_save_tc``
    to generate MI pairs if you need persistent files for reuse.
    """

    def __init__(
        self,
        data: np.ndarray,
        param_mi: Union[str, List],
        param_names: List[str] = None,
        kde_bw: Union[str, float, Dict[str, float]] = "silverman",
        min_mi: float = None,
        verbose: bool = False,
        bw_json_path: str = None,
        bw_method: Union[str, float, Dict[str, float], None] = None,
        top_k_greedy: int = 1,
    ):
        """
        Initialize and fit pairwise/independent KDE components.

        Args:
            data (ndarray): Samples with shape ``(n_samples, n_params)``.
            param_mi (str | list): Either a path to ``params_MI.json`` (as
                saved by dependency utilities) or a list of ``[a, b, mi]``
                entries where ``a`` and ``b`` are names or indices.
            param_names (list[str] | None): Optional parameter names; if None,
                defaults to ``["param_i"]``.
            kde_bw (str | float | dict): Bandwidth method or numeric factor. If a
                dict, values override JSON bandwidths per parameter name.
            bw_method (str | float | dict | None): Backward‑compat alias for ``kde_bw``.
            min_mi (float | None): Filter out pairs with MI below this value
                before selection.
            verbose (bool): Print helpful fitting logs.
            bw_json_path (str | None): Path to bandwidth JSON produced by
                ``compute_and_save_bandwidths(..., n_order=2)``. When provided,
                speeds up and standardizes bandwidths. You can still override
                individual parameters via the ``kde_bw`` dict.
            top_k_greedy (int): Try K seeded greedy runs. Run i in [0..K-1]
                seeds the selection with the i-th highest MI pair from the
                sorted candidate list, then completes selection greedily. The
                run with the largest total MI is kept. Defaults to 1 (current
                behavior).

        Notes:
            - When ``kde_bw`` is a list per pair, the geometric mean is used to
              obtain a scalar suitable for 2D ``gaussian_kde``.
            - Supported selectors from ``bw_method.py`` for SciPy‑compatible
              factors: ``'scott'``, ``'silverman'``, ``'isj'`` (Botev’s ISJ),
              ``'cv_iso'`` (isotropic CV), ``'cv_diag'`` (diagonal CV → scalar
              factor via geometric mean). Try ``'cv_iso'`` when pairwise
              components need tighter fits; prefer ``'silverman'`` for speed.
        """
        self.verbose = verbose
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("`data` must be 2D (n_samples, n_params).")
        self.n_samples, self.n_params = data.shape
        self.data = data
        # Backward compatibility: allow bw_method alias
        if bw_method is not None and kde_bw == 'silverman':
            kde_bw = bw_method
        elif bw_method is not None and bw_method != kde_bw:
            raise ValueError("Specify only one of 'kde_bw' or 'bw_method', not both.")
        self.kde_bw = kde_bw
        self.min_mi = min_mi
        self.bw_json_path = bw_json_path
        self.top_k_greedy = int(top_k_greedy) if top_k_greedy is not None else 1
        if param_names is None:
            param_names = [f"param_{i}" for i in range(self.n_params)]
        if len(param_names) != self.n_params:
            raise ValueError("Length of param_names must match number of columns in data.")
        self.param_names = [str(p) for p in param_names]
        self.param_map = {name: i for i, name in enumerate(self.param_names)}
        if isinstance(param_mi, str):
            with open(param_mi, "r", encoding="utf-8") as f:
                raw_pairs = json.load(f)
        else:
            raw_pairs = param_mi
        
        parsed = []
        for entry in raw_pairs:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                raise ValueError("Each entry in param_mi must be a list/tuple of at least 3 elements (a, b, mi).")
            a, b, mi = entry[0], entry[1], float(entry[2])
            parsed.append((a, b, mi))
        canonical = []
        for a, b, mi in parsed:
            def _to_name_or_index(x):
                if isinstance(x, (int, np.integer)):
                    idx = int(x)
                    if idx < 0 or idx >= self.n_params:
                        raise IndexError(f"Index {idx} out of bounds for param count {self.n_params}.")
                    return self.param_names[idx]
                if isinstance(x, str):
                    return x
                try:
                    idx = int(x)
                    if 0 <= idx < self.n_params:
                        return self.param_names[idx]
                except Exception:
                    pass
                return str(x)
            na = _to_name_or_index(a)
            nb = _to_name_or_index(b)
            if na not in self.param_map or nb not in self.param_map:
                raise KeyError(f"Parameter name {na!r} or {nb!r} not found in param_names.")
            canonical.append((na, nb, float(mi)))
        if self.min_mi is not None:
            canonical = [t for t in canonical if t[2] >= self.min_mi]
        canonical.sort(key=lambda t: -t[2])

        # Helper to run one seeded greedy pass starting from seed_idx
        def _run_seed(seed_idx: int):
            pool = set(self.param_names)
            selection = []
            total = 0.0
            if 0 <= seed_idx < len(canonical):
                na, nb, mi = canonical[seed_idx]
                if na in pool and nb in pool:
                    selection.append((na, nb, mi))
                    pool.remove(na); pool.remove(nb)
                    total += float(mi)
            for na, nb, mi in canonical:
                if na in pool and nb in pool:
                    selection.append((na, nb, mi))
                    pool.remove(na); pool.remove(nb)
                    total += float(mi)
            singles = sorted(pool)
            return selection, singles, total

        # Best-of-K seeded greedy
        if self.top_k_greedy is None or self.top_k_greedy <= 1 or len(canonical) == 0:
            pool = set(self.param_names)
            self.pairs = []
            for na, nb, mi in canonical:
                if na in pool and nb in pool:
                    self.pairs.append((na, nb, mi))
                    pool.remove(na); pool.remove(nb)
                    if self.verbose:
                        logger.info("Selected pair: %s, %s (MI=%.4g)", na, nb, mi)
            self.singles = sorted(pool)
            if self.verbose:
                logger.info("Pairs selected (%s): %s", len(self.pairs), self.pairs)
                logger.info("Singles (%s): %s", len(self.singles), self.singles)
        else:
            K = min(int(self.top_k_greedy), len(canonical))
            best_pairs, best_singles = [], self.param_names[:]  # placeholders
            best_key = None  # (total_mi, n_pairs, -seed_idx) for max-compare
            for seed_idx in range(K):
                pairs_i, singles_i, total_i = _run_seed(seed_idx)
                key = (total_i, len(pairs_i), -seed_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pairs, best_singles = pairs_i, singles_i
            self.pairs, self.singles = best_pairs, best_singles
            if self.verbose:
                total_best = best_key[0] if best_key is not None else 0.0
                logger.info("Best-of-%s seeded greedy total MI: %.4g; pairs=%s", K, total_best, len(self.pairs))
                for na, nb, mi in self.pairs:
                    logger.info("Selected pair: %s, %s (MI=%.4g)", na, nb, mi)
                logger.info("Singles (%s): %s", len(self.singles), self.singles)
        # Save selection next to source MI JSON if available
        try:
            if isinstance(param_mi, str):
                out_dir = os.path.dirname(os.path.abspath(param_mi))
                out_path = os.path.join(out_dir, "selected_pairs.json")
                payload = {
                    "pairs": [{"names": [a, b], "mi": float(mi)} for (a, b, mi) in getattr(self, "pairs", [])],
                    "singles": list(getattr(self, "singles", [])),
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                if self.verbose:
                    logger.info("Saved selection to %s", out_path)
        except Exception as e:  # pragma: no cover
            if self.verbose:
                logger.warning("Failed to write selected_pairs.json: %s", e)

        self._fit_kdes()

    def _fit_kdes(self):
        """Fit 2D KDEs for selected pairs and 1D KDEs for remaining singles."""
        # Prepare bandwidth dictionary from JSON and user overrides
        bandwidth_dict = self._prepare_bandwidth_dict(self.kde_bw, self.bw_json_path, self.param_names)

        self.pair_kdes = []
        self.single_kdes = {}
        for na, nb, mi in self.pairs:
            i = self.param_map[na]; j = self.param_map[nb]
            arr = self.data[:, [i, j]].T.copy()

            # Determine bandwidth for the pair using improved logic
            bw = self._get_bandwidth_for_params([na, nb], bandwidth_dict, self.kde_bw)

            # For 2D KDE, gaussian_kde expects a single scalar bandwidth or specific methods
            if isinstance(bw, list) and len(bw) == 2:
                # Use geometric mean of the two bandwidths for the pair
                bw_scalar = np.exp(np.mean(np.log(bw)))
            elif isinstance(bw, (int, float)):
                # Single scalar value
                bw_scalar = float(bw)
            else:
                # String method like 'silverman' or 'scott'
                bw_scalar = bw
            if self.verbose:
                logger.info("approx kde for pair%s with bw: %s", [na, nb], bw_scalar)
            kde2 = gaussian_kde(arr, bw_method=bw_scalar)
            self.pair_kdes.append({"names": (na, nb), "indices": (i, j), "mi": mi, "kde": kde2})

        for name in self.singles:
            i = self.param_map[name]
            arr = self.data[:, i][None, :].copy()

            # Determine bandwidth for the single parameter using improved logic
            bw = self._get_bandwidth_for_params([name], bandwidth_dict, self.kde_bw)

            # For 1D KDE, extract scalar value if it's a list
            if isinstance(bw, list) and len(bw) == 1:
                bw_scalar = bw[0]
            elif isinstance(bw, (int, float)):
                bw_scalar = float(bw)
            else:
                bw_scalar = bw

            kde1 = gaussian_kde(arr, bw_method=bw_scalar)
            self.single_kdes[name] = {"index": i, "kde": kde1}

    def logpdf(self, x: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluate the joint log density under the pairwise KDE approximation.

        Args:
            x: Either a single point with shape ``(d,)`` or a batch with
                shape ``(N, d)`` (both accepted; columns are dimensions).

        Returns:
            float or ndarray: Log density for each point; returns a scalar for
            a single input point.
        """
        x_arr = np.asarray(x.T, dtype=float)
        was_1d = x_arr.ndim == 1

        if was_1d:
            if x_arr.shape[0] != self.n_params:
                raise ValueError(f"Point has incorrect dimensionality. Expected {self.n_params}, got {x_arr.shape[0]}")
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim != 2 or x_arr.shape[1] != self.n_params:
            raise ValueError(f"Points have incorrect dimensionality. Expected (N, {self.n_params}), got {x_arr.shape}")

        n_points = x_arr.shape[0]
        logp = np.zeros(n_points, dtype=float)

        for entry in self.pair_kdes:
            i, j = entry["indices"]
            kde2 = entry["kde"]
            xy = x_arr[:, [i, j]].T
            logp += kde2.logpdf(xy)

        for name, info in self.single_kdes.items():
            idx = info["index"]
            kde1 = info["kde"]
            xx = x_arr[:, [idx]].T
            logp += kde1.logpdf(xx)
        
        if was_1d:
            return logp[0]
        else:
            return logp

    def pdf(self, x: Union[np.ndarray, List[float]]):
        """Return ``exp(logpdf(x))`` as a convenience wrapper."""
        return np.exp(self.logpdf(x))

    def resample(self, n_resamples: int, random_state: Union[int, None] = None) -> np.ndarray:
        """
        Draw i.i.d. samples from the fitted approximation.

        Args:
            n_resamples: Number of samples to generate.
            random_state: Optional integer seed for reproducibility.

        Returns:
            np.ndarray: Samples with shape ``(n_resamples, d)``.
        """
        if random_state is not None:
            old_state = np.random.get_state()
            np.random.seed(int(random_state))
        try:
            out = np.zeros((n_resamples, self.n_params), dtype=float)
            for entry in self.pair_kdes:
                na, nb = entry["names"]
                i, j = entry["indices"]
                kde2 = entry["kde"]
                samp = kde2.resample(n_resamples)
                out[:, i] = samp[0, :]
                out[:, j] = samp[1, :]
            for name, info in self.single_kdes.items():
                idx = info["index"]
                kde1 = info["kde"]
                samp = kde1.resample(n_resamples)
                out[:, idx] = samp.reshape(-1)
        finally:
            if random_state is not None:
                np.random.set_state(old_state)
        return out
