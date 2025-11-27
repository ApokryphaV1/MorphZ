import numpy as np
from tempfile import TemporaryDirectory
import pytest
from morphZ import evidence

@pytest.mark.parametrize("verbose, expect_output", [
    (False, ""),   # no prints
    (True,  "something"),  # prints expected
])
def test_verbosity(capsys, verbose, expect_output):
    np.random.seed(42)
    n_samples = 20

    post_samples = np.random.randn(n_samples, 1)
    log_posterior_values = np.zeros(n_samples)

    with TemporaryDirectory() as temp_dir:
        evidence(
            post_samples=post_samples,
            log_posterior_values=log_posterior_values,
            log_posterior_function=lambda x: 0.0,
            n_resamples=5,
            morph_type="indep",
            param_names=["p"],
            output_path=temp_dir,
            verbose=verbose,
        )

    out, err = capsys.readouterr()
    if verbose:
        assert out != "", "Expected printed output when verbose=True."
    else:
        assert out == "", "Expected no printed output when verbose=False."
        assert err == ""