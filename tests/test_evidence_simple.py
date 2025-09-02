#!/usr/bin/env python3
"""
Simple test script to verify that the evidence function works with the updated bandwidth integration.
"""
import numpy as np
import sys
import os
import tempfile

# Add the src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import evidence

def mock_log_posterior(theta):
    """Mock log posterior function for testing."""
    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    # Multivariate Gaussian log probability
    n_dims = theta.shape[0]
    cov = np.eye(n_dims) * 0.1
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))

    log_prob = -0.5 * (n_dims * np.log(2 * np.pi) + log_det_cov)

    if theta.ndim == 1:
        diff = theta.flatten()
        log_prob -= 0.5 * diff.T @ inv_cov @ diff

    return float(log_prob)

def test_evidence_simple():
    """Simple test of evidence function with independent KDE."""
    print("Testing evidence function with independent KDE...")

    # Create test posterior samples
    np.random.seed(42)
    n_samples = 500
    n_params = 2
    post_samples = np.random.randn(n_samples, n_params)
    log_posterior_values = np.array([mock_log_posterior(post_samples[i]) for i in range(n_samples)])
    param_names = ['param1', 'param2']

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        try:
            # Test evidence with independent KDE
            results = evidence(
                post_samples=post_samples,
                log_posterior_values=log_posterior_values,
                log_posterior_function=mock_log_posterior,
                n_resamples=50,
                thin=1,
                kde_fraction=0.8,
                bridge_start_fraction=0.5,
                max_iter=500,
                tol=1e-3,
                morph_type="indep",
                param_names=param_names,
                output_path=temp_dir,
                n_estimations=1,
                kde_bw="silverman",
                verbose=False
            )

            print(f"Evidence results: {results}")

            # Check that files were created
            bw_file = os.path.join(temp_dir, "bw_silverman_1D.json")
            logz_file = os.path.join(temp_dir, "logz_morph_z_indep_silverman.txt")

            assert os.path.exists(bw_file), "Bandwidth file not created"
            assert os.path.exists(logz_file), "LogZ file not created"

            # Verify results structure
            assert isinstance(results, list), "Results should be a list"
            assert len(results) == 1, "Should have one estimation"
            assert len(results[0]) == 2, "Each result should be [logZ, error]"

            logz, error = results[0]
            assert isinstance(logz, (int, float)), "logZ should be numeric"
            assert isinstance(error, (int, float)), "Error should be numeric"
            assert not np.isnan(logz), "logZ should not be NaN"
            assert not np.isnan(error), "Error should not be NaN"

            print(".4f")
            return True

        except Exception as e:
            print(f"❌ Evidence test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Simple evidence function test...")
    print("="*50)

    success = test_evidence_simple()

    if success:
        print("\n🎉 Evidence function test passed!")
        print("The evidence function works correctly with the updated bandwidth integration.")
    else:
        print("\n❌ Evidence function test failed!")
        sys.exit(1)
