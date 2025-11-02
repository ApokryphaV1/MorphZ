import json
import logging
from typing import Union, Dict, Any

import emcee
import numpy as np
from scipy.stats import gaussian_kde
from .kde_base import KDEBase

logger = logging.getLogger(__name__)


class Morph_Tree(KDEBase):
    """
    A class to approximate a multivariate distribution using a Chow-Liu tree-based decomposition.
    The joint PDF is approximated as a product of conditional probabilities based on the tree structure.
    p(x_1, ..., x_n) ≈ p(x_root) * product_{i in children} p(x_i | x_parent(i))
    """
    def __init__(self, data, tree_file, param_names=None, kde_bw='silverman', bw_json_path=None, bw_method=None):
        """
        Initializes the Morph_Tree object by building the KDEs based on the provided tree structure.

        Args:
            data (array-like): 2D array of shape (n_samples, n_params) used to fit the KDEs.
            tree_file (str): Path to the JSON file containing the tree structure.
            param_names (list of str, optional): Names of the parameters. If None, parameters are named by their index.
            kde_bw (str, scalar, or dict, optional): The method used to estimate the bandwidth of the KDE.
                                                     Can also be used to override specific parameter bandwidths
                                                     when using JSON files. Defaults to 'silverman'.
            bw_json_path (str, optional): Path to JSON file containing bandwidth values. If provided,
                                         bandwidths will be loaded from this file and kde_bw can be used
                                         to override specific parameters.
            bw_method (str, scalar, or dict, optional): Backward‑compat alias for ``kde_bw``.
        
        Suggestions:
            - Generate ``tree_file`` via ``dependency_tree.compute_and_plot_mi_tree``
              to ensure consistency with MI weights and labels used elsewhere.
            - When passing a dict for ``kde_bw``, keys must match ``param_names``.
            - Supported selectors from ``bw_method.py`` for SciPy‑compatible
              factors: ``'scott'``, ``'silverman'``, ``'isj'`` (Botev’s ISJ),
              ``'cv_iso'`` (isotropic CV), ``'cv_diag'`` (diagonal CV → scalar
              factor via geometric mean). For trees, ``'cv_iso'`` can offer a
              good accuracy/runtime trade‑off.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array-like object with shape (n_samples, n_params)")

        n_samples, n_params = data.shape
        logger.info("Morph_Tree initialized with %s dimensional data shape.", n_params)

        if param_names is None:
            param_names = [f"param_{i}" for i in range(n_params)]
        elif len(param_names) != n_params:
            raise ValueError("Length of param_names must equal the number of parameters (columns) in data")

        self.param_names = param_names
        self.param_map = {name: i for i, name in enumerate(param_names)}
        self.bw_json_path = bw_json_path

        # Backward compatibility: allow bw_method alias
        if bw_method is not None and kde_bw == 'silverman':
            kde_bw = bw_method
        elif bw_method is not None and bw_method != kde_bw:
            raise ValueError("Specify only one of 'kde_bw' or 'bw_method', not both.")

        with open(tree_file, 'r') as f:
            self.tree = json.load(f)

        # Prepare bandwidth dictionary from JSON and user overrides
        bandwidth_dict = self._prepare_bandwidth_dict(kde_bw, bw_json_path, param_names)

        # after loading self.tree = json.load(f)
        all_parents = {p for p, c in self.tree}
        all_children = {c for p, c in self.tree}
        roots = list(all_parents - all_children)
        if len(roots) != 1:
            raise ValueError("Tree JSON must have exactly one root")
        self.root = roots[0]

        # Create 1D KDE for the root
        root_idx = self.param_map[self.root]
        root_bw = self._get_bandwidth_for_params([self.root], bandwidth_dict, kde_bw)
        # Ensure bw is a scalar or valid method string for 1D KDE
        if isinstance(root_bw, list) and len(root_bw) == 1:
            root_bw_scalar = root_bw[0]
        elif isinstance(root_bw, (int, float)):
            root_bw_scalar = float(root_bw)
        else:
            root_bw_scalar = root_bw
        self.root_kde = gaussian_kde(data[:, root_idx], bw_method=root_bw_scalar)

        # Create 2D KDEs for the branches
        self.branch_kdes = {}
        for parent, child in self.tree:
            parent_idx = self.param_map[parent]
            child_idx = self.param_map[child]

            # Determine bandwidth for the joint distribution using improved logic
            joint_bw = self._get_bandwidth_for_params([child, parent], bandwidth_dict, kde_bw)
            parent_bw = self._get_bandwidth_for_params([parent], bandwidth_dict, kde_bw)

            # Convert joint bandwidth to a scalar/method acceptable by gaussian_kde
            if isinstance(joint_bw, list) and len(joint_bw) == 2:
                # Geometric mean of per-dimension factors (mirrors PairwiseKDE)
                joint_bw_scalar = np.exp(np.mean(np.log(joint_bw)))
            elif isinstance(joint_bw, (int, float)):
                joint_bw_scalar = float(joint_bw)
            else:
                joint_bw_scalar = joint_bw  # e.g., 'silverman' or 'scott'

            # 2D KDE for the joint distribution p(child, parent)
            joint_kde = gaussian_kde(data[:, [child_idx, parent_idx]].T, bw_method=joint_bw_scalar)

            # 1D KDE for the parent p(parent) — ensure scalar/method
            if isinstance(parent_bw, list) and len(parent_bw) == 1:
                parent_bw_scalar = parent_bw[0]
            elif isinstance(parent_bw, (int, float)):
                parent_bw_scalar = float(parent_bw)
            else:
                parent_bw_scalar = parent_bw
            parent_kde = gaussian_kde(data[:, parent_idx], bw_method=parent_bw_scalar)

            self.branch_kdes[(parent, child)] = (joint_kde, parent_kde)

    def logpdf(self, point):
        """
        Computes the log probability density of a given point using the tree-based approximation.

        Args:
            point (array-like): The point at which to evaluate the log probability density.

        Returns:
            float: The log probability density at the given point.
        """
        logpdf = 0
        
        # Add the logpdf of the root
        root_idx = self.param_map[self.root]
        logpdf += self.root_kde.logpdf(point[root_idx])
         

        # Add the logpdf of the conditional distributions
        for parent, child in self.tree:
            parent_idx = self.param_map[parent]
            child_idx = self.param_map[child]
            
            joint_kde, parent_kde = self.branch_kdes[(parent, child)]
            
            # log p(child|parent) = log p(child, parent) - log p(parent)
            log_p_conditional = joint_kde.logpdf(point[[child_idx, parent_idx]]) - parent_kde.logpdf(point[parent_idx])
            logpdf += log_p_conditional

        return logpdf

    def resample(self, n_resamples, nwalkers=None, progress=True):
        """
        Resamples from the approximated multivariate distribution using emcee.
        The number of walkers and thinning are adapted to generate n_resamples.

        Args:
            n_resamples (int): The desired number of resamples.
            nwalkers (int, optional): The number of walkers. Defaults to 2 * ndim.
            progress (bool, optional): Whether to display a progress bar. Defaults to False.

        Returns:
            np.ndarray: An array of resampled points with shape (n_resamples, n_dims).
        
        Tips:
            - If too few independent samples are produced, increase ``n_steps`` by
              raising ``n_resamples`` or override the initial autocorrelation guess.
            - For speed, consider using ``PairwiseKDE``/``GroupKDE`` resampling
              when the tree structure is not essential.
        """
        ndim = len(self.param_names)
        if nwalkers is None:
            nwalkers = 2 * ndim

        # initial guess for autocorrelation time
        initial_autocorr_time = 5
        
        # Calculate the number of steps needed
        #n_steps = (n_resamples * initial_autocorr_time) // nwalkers
        n_steps = max(10, (n_resamples * initial_autocorr_time + nwalkers - 1) // nwalkers)
        burn_in = int(0.3 * n_steps)
        
        initial_state = self._initial_resample(nwalkers).T

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logpdf)
        sampler.run_mcmc(initial_state, n_steps, progress=progress)

        # Discard burn-in
        samples = sampler.get_chain(discard=burn_in, flat=False)

        # Estimate autocorrelation time
        try:
            autocorr_time = np.mean(sampler.get_autocorr_time(tol=0))
            if np.isnan(autocorr_time):
                autocorr_time = initial_autocorr_time
                #print(f"Warning: Autocorrelation time is NaN. Using initial guess: {autocorr_time}")
        except emcee.autocorr.AutocorrError:
            autocorr_time = initial_autocorr_time
            #print(f"Warning: Autocorrelation time could not be estimated. Using initial guess: {autocorr_time}")

        # Thin the chain
        thin_factor = max(1, int(np.ceil(autocorr_time)))
        thinned_samples = samples[::thin_factor, :, :].reshape(-1, ndim)

        # If not enough samples, warn the user and sample with replacement
        if thinned_samples.shape[0] == 0:
            raise ValueError("No independent samples were generated. The MCMC chain may be too short or the autocorrelation time is severely underestimated.")
        
        if thinned_samples.shape[0] < n_resamples:
            logger.warning(
                "Not enough independent samples generated (%s/%s). The autocorrelation time may be underestimated. Sampling with replacement to get %s samples.",
                thinned_samples.shape[0],
                n_resamples,
                n_resamples,
            )
            return thinned_samples[np.random.choice(thinned_samples.shape[0], n_resamples, replace=True)]

        return thinned_samples[np.random.choice(thinned_samples.shape[0], n_resamples, replace=False)]

    def _initial_resample(self, size=1):
        """
        Helper function to generate initial samples for the MCMC walkers.
        This is the old resample method.
        """
        if size == 1:
            resampled = np.zeros(len(self.param_names))
        else:
            resampled = np.zeros((len(self.param_names), size))

        # Resample the root
        root_idx = self.param_map[self.root]
        resampled[root_idx] = self.root_kde.resample(size)

        # Resample the children conditioned on the parent
        for parent, child in self.tree:
            parent_idx = self.param_map[parent]
            child_idx = self.param_map[child]
            
            child_kde = gaussian_kde(self.branch_kdes[(parent, child)][0].dataset[0])
            resampled[child_idx] = child_kde.resample(size)
            
        return resampled
