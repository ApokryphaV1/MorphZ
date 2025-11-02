"""
morphZ - KDE-based density estimation and approximation package.
"""

import logging
from typing import Optional

from .kde_base import KDEBase
from .morph_indep import Morph_Indep
KDE_approx = Morph_Indep  # backward-compat alias
from .morph_pairwise import Morph_Pairwise
PairwiseKDE = Morph_Pairwise  # backward-compat alias
from .morph_group import Morph_Group
GroupKDE = Morph_Group  # backward-compat alias
from .morph_tree import Morph_Tree
TreeKDE = Morph_Tree  # backward-compat alias
from .bw_method import (
    select_bandwidth,
    compute_and_save_bandwidths,
    scott_rule,
    silverman_rule,
    botev_isj_bandwidth,
    cross_validation_bandwidth
)
from .bridge import (
    bridge_sampling_ln,
    compute_bridge_rmse,
)
from .morph import (
    evidence
)
from . import utils
from . import dependency_tree
from . import Nth_TC


def setup_logging(level: int = logging.INFO, stream=None, fmt: Optional[str] = None) -> None:
    """
    Configure root logging for morphZ consumers.

    Args:
        level: Logging level passed to :func:`logging.basicConfig`.
        stream: Optional stream override (``sys.stdout`` by default).
        fmt: Optional logging format string. Defaults to a concise pattern.
    """
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        stream=stream,
        format=fmt or "%(levelname)s:%(name)s:%(message)s",
    )

# Version is provided by setuptools_scm via git tags. When installed, importlib
# metadata has the version; during builds sdist/wheels include _version.py.
try:  # prefer file written by setuptools_scm at build time
    from ._version import version as __version__  # type: ignore
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # Python 3.8+
        __version__ = _pkg_version("morphZ")
    except Exception:  # fallback for source tree without tags/metadata
        __version__ = "0.0.0"
__all__ = [
    "KDEBase",
    "Morph_Indep",
    "KDE_approx",
    "Morph_Pairwise",
    "PairwiseKDE",
    "Morph_Group",
    "GroupKDE",
    "Morph_Tree",
    "TreeKDE",
    "select_bandwidth",
    "compute_and_save_bandwidths",
    "scott_rule",
    "silverman_rule",
    "botev_isj_bandwidth",
    "cross_validation_bandwidth",
    "bridge_sampling_ln",
    "compute_bridge_rmse",
    "evidence",
    "utils",
    "dependency_tree",
    "Nth_TC",
    "setup_logging",
]
