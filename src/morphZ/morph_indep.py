import logging
from typing import Union, Dict, Any

import numpy as np
from scipy.stats import gaussian_kde
from .kde_base import KDEBase

logger = logging.getLogger(__name__)


class Morph_Indep(KDEBase):
    """
    A class to approximate a multivariate distribution using a product of 1D Gaussian Kernel Density Estimators (KDEs).
    This assumes independence between the parameters.
    """
    def __init__(self, data, kde_bw='silverman', param_names=None, bw_json_path=None, verbose=False, bw_method=None):
        """
        Initializes the Morph_Indep object by fitting a 1D KDE to each parameter (column) in the data.

        Args:
            data (array-like): 2D array of shape (n_samples, n_params) used to fit the KDEs.
            kde_bw (str, scalar, or dict, optional): The method used to estimate the bandwidth of the KDE.
                                                     Can also be used to override specific parameter bandwidths
                                                     when using JSON files. Defaults to 'silverman'.
            bw_method (str, scalar, or dict, optional): Backward‑compat alias for ``kde_bw``.
            param_names (list of str, optional): Names of the parameters. If None, parameters are named by their index.
            bw_json_path (str, optional): Path to JSON file containing bandwidth values. If provided,
                                         bandwidths will be loaded from this file and kde_bw can be used
                                         to override specific parameters.
            verbose (bool, optional): Print fitting details per parameter.

        Suggestions:
            - ``kde_bw`` can be a number (e.g., 0.8) for slightly tighter KDEs
              or a dict mapping parameter names to custom values.
            - Supported selectors from ``bw_method.py`` for SciPy‑compatible
              factors: ``'scott'``, ``'silverman'``, ``'isj'`` (Botev’s ISJ),
              ``'cv_iso'`` (isotropic CV), ``'cv_diag'`` (diagonal CV → scalar
              factor via geometric mean). Choose ``'cv_iso'``/``'cv_diag'`` for
              higher fidelity at extra cost.
            - Use ``bw_json_path`` with values from
              ``compute_and_save_bandwidths`` to ensure consistent settings
              between different KDE approximations.
        """
        self.verbose = verbose
        # Backward compatibility: allow bw_method alias
        if bw_method is not None and kde_bw == 'silverman':
            kde_bw = bw_method
        elif bw_method is not None and bw_method != kde_bw:
            raise ValueError("Specify only one of 'kde_bw' or 'bw_method', not both.")
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array-like object with shape (n_samples, n_params)")

        n_samples, n_params = data.shape

        if param_names is None:
            param_names = [f"{i}" for i in range(n_params)]
        elif len(param_names) != n_params:
            raise ValueError("Length of param_names must equal the number of parameters (columns) in data")

        self.param_names = param_names
        self.param_map = {name: i for i, name in enumerate(param_names)}

        # Prepare bandwidth dictionary from JSON and user overrides
        bandwidth_dict = self._prepare_bandwidth_dict(kde_bw, bw_json_path, param_names)

        self.kde_dict = {}
        for i, name in enumerate(param_names):
            # Extract the i-th parameter (all samples for that parameter)
            param_data = data[:, i]

            # Determine the bandwidth for the current parameter using improved logic
            bw = self._get_bandwidth_for_params([name], bandwidth_dict, kde_bw)
            if self.verbose:
                logger.info("morph_indep for %s with bw: %s", name, bw)
            # Fit a Gaussian KDE
            kde_obj = gaussian_kde(param_data, bw_method=bw)
            # Save the callable KDE object in the dictionary
            self.kde_dict[name] = kde_obj

    def logpdf_kde(self,point):
        """
        Computes the log probability density of a given point.

        The total log probability is the sum of the log probabilities from each 1D KDE,
        assuming independence.

        Args:
            point (array-like): The point at which to evaluate the log probability density.

        Returns:
            float: The log probability density at the given point.
        """
        logpdf = 0
        for name, kde_obj in self.kde_dict.items():
            # Use the parameter index from param_map to access the correct coordinate
            param_idx = self.param_map[name]
            logpdf += kde_obj.logpdf(point[param_idx])
        return logpdf

    def resample(self, size=1):
        """
        Resamples from the approximated multivariate distribution.

        Args:
            size (int, optional): The number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: An array of resampled points with shape (size, n_params).
        """
        n_params = len(self.param_names)
        resampled = np.zeros((size, n_params))

        for i, name in enumerate(self.param_names):
            kde_obj = self.kde_dict[name]
            resampled[:, i] = kde_obj.resample(size)

        return resampled
