## MorphZ

Morph-Z for high accuracy marginal likelihood estimation and morphological density approximation toolkit for scientific workflows, with utilities for dependency analysis.

- Flexible Morph backends: independent, pairwise, grouped, and tree-structured.
- Bandwidth selection: Scott, Silverman, Botev ISJ, and cross-validation variants.
- Evidence estimation via bridge sampling with robust diagnostics.
- Mutual information and Total correlation estimation.
- Mutual information and Chow–Liu dependency tree visualisation.

## Installation

Python 3.8+ is recommended.

```bash
pip install morphz
```

From source (editable):

```bash
pip install -e .
```

## Run The Examples

Interactive notebooks live in `examples/`:

- `examples/eggbox.ipynb`
- `examples/gaussian shell.ipynb`
- `examples/peak_sampling_new.ipynb`


## Quick Starts

Minimal Morph fit and evaluate:

```python
import numpy as np
from morphZ import Morph_Indep, select_bandwidth

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 2))

bw = select_bandwidth(X, method="silverman")
morph_indep = Morph_Indep(X, kde_bw=bw)

pts = rng.normal(size=(5, 2))
print(morph_indep.logpdf(pts))
```

Compute MI heatmap and a Chow–Liu tree (artifacts saved to `out_dir`):

```python
import numpy as np
from morphZ import dependency_tree

X = np.random.default_rng(0).normal(size=(1000, 4))
mi, tree, edges = dependency_tree.compute_and_plot_mi_tree(
    X, names=["x0", "x1", "x2", "x3"], out_path="out_dir", morph_type="tree"
)
print("Edges (parent -> child):", edges)
```

Compute n‑order Total Correlation (TC) and save results:

```python
import numpy as np
from morphZ import Nth_TC

X = np.random.default_rng(0).normal(size=(1000, 5))
Nth_TC.compute_and_save_tc(
    X, names=[f"x{i}" for i in range(X.shape[1])], n_order=3, out_path="out_dir"
)
```

End‑to‑end morphological evidence with bridge sampling:

```python
import numpy as np
from morphZ import evidence

rng = np.random.default_rng(0)
dim = 2

# Toy "posterior": standard Normal, known up to a constant
def log_target(theta: np.ndarray) -> float:
    return -0.5 * np.dot(theta, theta)

# Pretend these came from an MCMC chain
post_samples = rng.normal(size=(5000, dim))
log_post_vals = -0.5 * np.sum(post_samples**2, axis=1)

results = evidence(
    post_samples=post_samples,
    log_posterior_values=log_post_vals,
    log_posterior_function=log_target,
    n_resamples=2000,
    morph_type="tree",          # "indep" | "pair" | "tree" | "3_group" | ...
    kde_bw="isj",             # "scott" | "silverman" | "isj" | "cv_iso" | "cv_diag" | numeric
    param_names=[f"x{i}" for i in range(dim)],
    output_path="examples/morphZ_gaussian_demo",
    n_estimations=2,
    verbose=True,
)

print("log(z), err per run:\n", np.array(results))
```

Artifacts will be saved under `examples/morphZ_gaussian_demo/` (bandwidths, MI/Tree files as needed, and `logz_morph_z_<morph_type>_<bw_method>.txt`).

## API Highlights

- Morphs: `Morph_Indep`, `Morph_Pairwise`, `Morph_Tree`, `Morph_Group`.
- Bandwidths: `select_bandwidth`, `compute_and_save_bandwidths`.
- Evidence: `evidence`, `bridge_sampling_ln` (lower‑level), `compute_bridge_rmse`.
- Dependency analysis: `dependency_tree.compute_and_plot_mi_tree`.
- Total correlation: `Nth_TC.compute_and_save_tc`.

Notes:

- If you pass a numeric `kde_bw` (e.g., `0.9`) the library skips bandwidth JSONs.
- Tree/group proposals will compute and cache `tree.json`/`params_*_TC.json` on first use.

## Dependencies

- Core: `numpy`, `scipy`, `matplotlib`, `corner`, `networkx`, `emcee`, `statsmodels`, `scikit-learn`
- Optional: `pandas` (CSV labels), `pygraphviz` (nicer tree layout), `scikit-sparse` (optional exception type)

## Development

- Build wheels/sdist: `python -m build`
- Check metadata: `twine check dist/*`
- Tests live in `tests/`

## Versioning & Release

Versioning is derived from git tags via `setuptools_scm`.

- Tag a release: `git tag vX.Y.Z && git push --tags`
- CI: publishes to TestPyPI on pushes to `main`/`master`; to PyPI on `v*` tags.
- Uses PyPI/TestPyPI Trusted Publishing (OIDC). You can also use API tokens if preferred.

## License

BSD-3-Clause. See `LICENSE` for details.
