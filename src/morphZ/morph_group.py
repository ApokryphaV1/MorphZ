"""
morph_group.py -- greedy group-based KDE approximation.
"""
from __future__ import annotations
import json
import logging
import os
import re
from typing import List, Tuple, Union, Dict, Any
import numpy as np
from scipy.stats import gaussian_kde
from .kde_base import KDEBase

logger = logging.getLogger(__name__)


class Morph_Group(KDEBase):
    """
    Group‑based KDE approximation guided by N‑th order Total Correlation (TC).

    Disjoint groups with largest TC are modeled by multi‑dimensional KDEs; any
    leftovers are modeled as independent 1D KDEs. Use
    ``Nth_TC.compute_and_save_tc`` to generate and persist groups.
    """

    def __init__(
        self,
        data: np.ndarray,
        param_tc: Union[str, List],
        param_names: List[str] = None,
        kde_bw: Union[str, float, Dict[str, float]] = "silverman",
        min_tc: float = None,
        verbose: bool = False,
        bw_json_path: str = None,
        bw_method: Union[str, float, Dict[str, float], None] = None,
        top_k_greedy: int = 1,
    ):
        """
        Initialize and fit group/independent KDE components.

        Args:
            data (ndarray): Samples with shape ``(n_samples, n_params)``.
            param_tc (str | list): Either a path to a TC JSON file with entries
                like ``[[group_names], tc]`` or an in‑memory list of such pairs.
            param_names (list[str] | None): Optional names for parameters.
            kde_bw (str | float | dict): Bandwidth method/factor or per‑name overrides.
            bw_method (str | float | dict | None): Backward‑compat alias for ``kde_bw``.
            min_tc (float | None): Discard groups with TC below this threshold.
            verbose (bool): Print helpful fitting logs.
            bw_json_path (str | None): Path to bandwidth JSON (e.g., produced by
                ``compute_and_save_bandwidths(..., n_order=k, group_format="groups")``).
            top_k_greedy (int): Try K seeded greedy runs. Run i in [0..K-1]
                seeds the selection with the i-th highest‑TC group from the
                sorted candidate list, then completes selection greedily. The
                run with the largest total TC is kept. Defaults to 1 (current
                behavior).

        Notes:
            - For group KDEs (dim>1), a single scalar bandwidth is required by
              ``gaussian_kde``; we use the geometric mean when a list is given.
            - Supported selectors from ``bw_method.py`` for SciPy‑compatible
              factors: ``'scott'``, ``'silverman'``, ``'isj'`` (Botev’s ISJ),
              ``'cv_iso'`` (isotropic CV), ``'cv_diag'`` (diagonal CV → scalar
              factor via geometric mean). ``'cv_iso'`` often improves multi‑D
              group fits but is slower.
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
        self.min_tc = min_tc
        self.bw_json_path = bw_json_path
        self.top_k_greedy = int(top_k_greedy) if top_k_greedy is not None else 1
        if param_names is None:
            param_names = [f"param_{i}" for i in range(self.n_params)]
        if len(param_names) != self.n_params:
            raise ValueError("Length of param_names must match number of columns in data.")
        self.param_names = [str(p) for p in param_names]
        self.param_map = {name: i for i, name in enumerate(self.param_names)}
        if isinstance(param_tc, str):
            with open(param_tc, "r", encoding="utf-8") as f:
                raw_groups = json.load(f)
        else:
            raw_groups = param_tc
        
        parsed = []
        for entry in raw_groups:
            # Accept both formats:
            # 1) [["p1","p2"], tc]
            # 2) ["p1","p2", tc]
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError("Each entry in param_tc must be a list/tuple with a group and a TC value.")

            if len(entry) == 2 and isinstance(entry[0], (list, tuple)):
                group, tc = entry[0], float(entry[1])
            else:
                # Assume last element is tc and preceding elements are names
                group, tc = list(entry[:-1]), float(entry[-1])

            if not isinstance(group, (list, tuple)):
                raise ValueError("The group specification must be a list/tuple of parameter names/indices.")
            parsed.append((group, tc))

        canonical = []
        for group, tc in parsed:
            def _to_name(x):
                if isinstance(x, (int, np.integer)):
                    idx = int(x)
                    if idx < 0 or idx >= self.n_params:
                        raise IndexError(f"Index {idx} out of bounds for param count {self.n_params}.")
                    return self.param_names[idx]
                if isinstance(x, str):
                    if x not in self.param_map:
                        raise KeyError(f"Parameter name {x!r} not found in param_names.")
                    return x
                raise TypeError(f"Unsupported parameter identifier type: {type(x)}")
            
            named_group = tuple(sorted([_to_name(p) for p in group]))
            canonical.append((named_group, float(tc)))

        if self.min_tc is not None:
            canonical = [g for g in canonical if g[1] >= self.min_tc]
        
        canonical.sort(key=lambda t: -t[1])

        # Helper to run one seeded greedy pass starting from seed_idx
        def _run_seed(seed_idx: int):
            pool = set(self.param_names)
            selection = []  # list of dicts with {names, tc}
            total = 0.0
            if 0 <= seed_idx < len(canonical):
                names, tc = canonical[seed_idx]
                if all(name in pool for name in names):
                    selection.append({"names": names, "tc": float(tc)})
                    for n in names:
                        pool.remove(n)
                    total += float(tc)
            for names, tc in canonical:
                if all(name in pool for name in names):
                    selection.append({"names": names, "tc": float(tc)})
                    for n in names:
                        pool.remove(n)
                    total += float(tc)
            singles = sorted(pool)
            return selection, singles, total

        # Best-of-K seeded greedy
        if self.top_k_greedy is None or self.top_k_greedy <= 1 or len(canonical) == 0:
            pool = set(self.param_names)
            self.groups = []
            for names, tc in canonical:
                if all(name in pool for name in names):
                    self.groups.append({"names": names, "tc": tc})
                    for name in names:
                        pool.remove(name)
                    if self.verbose:
                        logger.info("Selected group: %s (TC=%.4g)", ", ".join(names), tc)

            self.singles = sorted(pool)
            if self.verbose:
                logger.info(
                    "Groups selected (%s): %s",
                    len(self.groups),
                    [g["names"] for g in self.groups],
                )
                logger.info("Singles (%s): %s", len(self.singles), self.singles)
        else:
            K = min(int(self.top_k_greedy), len(canonical))
            best_groups, best_singles = [], self.param_names[:]  # placeholders
            best_key = None  # (total_tc, n_groups, -seed_idx) for max-compare
            for seed_idx in range(K):
                groups_i, singles_i, total_i = _run_seed(seed_idx)
                key = (total_i, len(groups_i), -seed_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_groups, best_singles = groups_i, singles_i
            self.groups, self.singles = best_groups, best_singles
            if self.verbose:
                total_best = best_key[0] if best_key is not None else 0.0
                logger.info(
                    "Best-of-%s seeded greedy total TC: %.4g; groups=%s",
                    K,
                    total_best,
                    len(self.groups),
                )
                for g in self.groups:
                    logger.info(
                        "Selected group: %s (TC=%.4g)",
                        ", ".join(g["names"]),
                        g["tc"],
                    )
                logger.info("Singles (%s): %s", len(self.singles), self.singles)
        # Save selection next to source TC JSON if available
        try:
            if isinstance(param_tc, str):
                out_dir = os.path.dirname(os.path.abspath(param_tc))
                # Try to extract n_order from filename like params_3-order_TC.json
                base = os.path.basename(param_tc)
                m = re.search(r"params_(\d+)-order_TC\.json$", base)
                if m:
                    n_order_str = m.group(1)
                    out_name = f"selected_{n_order_str}-order_group.json"
                else:
                    out_name = "selected_group.json"
                out_path = os.path.join(out_dir, out_name)
                payload = {
                    "groups": [
                        {"names": list(g.get("names", ())), "tc": float(g.get("tc", 0.0))}
                        for g in getattr(self, "groups", [])
                    ],
                    "singles": list(getattr(self, "singles", [])),
                }
                # Include n_order when discoverable
                if m:
                    payload["n_order"] = int(m.group(1))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                if self.verbose:
                    logger.info("Saved selection to %s", out_path)
        except Exception as e:  # pragma: no cover
            if self.verbose:
                logger.warning("Failed to write selected group file: %s", e)

        self._fit_kdes()

    def _fit_kdes(self):
        """Fit multi‑dimensional KDEs for selected groups and 1D KDEs for singles."""
        # Prepare bandwidth dictionary from JSON and user overrides
        bandwidth_dict = self._prepare_bandwidth_dict(self.kde_bw, self.bw_json_path, self.param_names)

        self.group_kdes = []
        self.single_kdes = {}
        for group_info in self.groups:
            names = group_info["names"]
            indices = tuple(self.param_map[name] for name in names)
            arr = self.data[:, list(indices)].T.copy()

            # Determine bandwidth for the group using improved logic
            bw = self._get_bandwidth_for_params(names, bandwidth_dict, self.kde_bw)

            # For multi-dimensional KDE (groups with >1 parameter), gaussian_kde expects a single scalar bandwidth
            if isinstance(bw, list) and len(bw) > 1:
                # Use geometric mean of the individual bandwidths for the group
                bw_scalar = np.exp(np.mean(np.log(bw)))
            elif isinstance(bw, (int, float)):
                # Single scalar value
                bw_scalar = float(bw)
            else:
                # String method like 'silverman' or 'scott'
                bw_scalar = bw
            if self.verbose:
                logger.info("approx kde for group%s with bw: %s", names, bw_scalar)
            kde = gaussian_kde(arr, bw_method=bw_scalar)
            self.group_kdes.append({"names": names, "indices": indices, "tc": group_info["tc"], "kde": kde})

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
            if self.verbose:
                logger.info("approx kde for singles%s with bw: %s", name, bw_scalar)
            kde1 = gaussian_kde(arr, bw_method=bw_scalar)
            self.single_kdes[name] = {"index": i, "kde": kde1}

    def logpdf(self, x: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluate the joint log density under the group KDE approximation.

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

        for entry in self.group_kdes:
            indices = entry["indices"]
            kde = entry["kde"]
            points_subset = x_arr[:, list(indices)].T
            logp += kde.logpdf(points_subset)

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
            for entry in self.group_kdes:
                indices = entry["indices"]
                kde = entry["kde"]
                samp = kde.resample(n_resamples)
                for i, idx in enumerate(indices):
                    out[:, idx] = samp[i, :]
            
            for name, info in self.single_kdes.items():
                idx = info["index"]
                kde1 = info["kde"]
                samp = kde1.resample(n_resamples)
                out[:, idx] = samp.reshape(-1)
        finally:
            if random_state is not None:
                np.random.set_state(old_state)
        return out
