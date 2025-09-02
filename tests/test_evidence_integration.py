#!/usr/bin/env python3
"""
Test script to verify that the evidence function works correctly with the updated bandwidth integration.
"""
import numpy as np
import sys
import os
import tempfile
import json

# Add the src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import evidence

def mock_log_posterior(theta):
    """
    Mock log posterior function for testing.
    Returns log probability for a multivariate Gaussian centered at origin.
    """
    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    # Multivariate Gaussian log probability
    n_dims = theta.shape[0]
    cov = np.eye(n_dims) * 0.1  # Small covariance
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))

    log_prob = -0.5 * (n_dims * np.log(2 * np.pi) + log_det_cov)

    if theta.ndim == 1:
        diff = theta.flatten()
        log_prob -= 0.5 * diff.T @ inv_cov @ diff
    else:
        # Handle multiple points
        log_probs = np.zeros(theta.shape[1])
        for i in range(theta.shape[1]):
            diff = theta[:, i]
            log_probs[i] = -0.5 * diff.T @ inv_cov @ diff
        log_prob = log_prob + log_probs

    return float(log_prob) if theta.ndim == 1 else log_probs

def test_evidence_indep_kde():
    """Test evidence function with independent KDE."""
    print("Testing evidence function with independent KDE...")

    # Create test posterior samples
    np.random.seed(42)
    n_samples = 1000
    n_params = 3
    post_samples = np.random.randn(n_samples, n_params)
    log_posterior_values = np.array([mock_log_posterior(post_samples[i]) for i in range(n_samples)])
    param_names = ['param1', 'param2', 'param3']

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        try:
            # Test evidence with independent KDE
            results = evidence(
                post_samples=post_samples,
                log_posterior_values=log_posterior_values,
                log_posterior_function=mock_log_posterior,
                n_resamples=100,
                thin=1,
                kde_fraction=0.8,
                bridge_start_fraction=0.5,
                max_iter=1000,
                tol=1e-3,
                morph_type="indep",
                param_names=param_names,
                output_path=temp_dir,
                n_estimations=1,
                kde_bw="silverman",
                verbose=False
            )

            print(f"   Evidence results: {results}")

            # Check that bandwidth file was created
            bw_file = os.path.join(temp_dir, "bw_silverman_1D.json")
            assert os.path.exists(bw_file), "Bandwidth file not created"

            # Check that logZ file was created
            logz_file = os.path.join(temp_dir, "logz_morph_z_indep_silverman.txt")
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
            print(f"❌ Independent KDE test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_evidence_pair_kde():
    """Test evidence function with pairwise KDE."""
    print("\n" + "="*50)
    print("Testing evidence function with pairwise KDE...")

    # Create test posterior samples
    np.random.seed(42)
    n_samples = 800
    n_params = 4
    post_samples = np.random.randn(n_samples, n_params)
    log_posterior_values = np.array([mock_log_posterior(post_samples[i]) for i in range(n_samples)])
    param_names = ['A', 'B', 'C', 'D']

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Create MI file for pairwise KDE
        mi_data = [
            ['A', 'B', 0.5],
            ['A', 'C', 0.3],
            ['B', 'D', 0.2],
        ]
        mi_file = os.path.join(temp_dir, "params_MI.json")
        with open(mi_file, 'w') as f:
            json.dump(mi_data, f, indent=2)

        try:
            # Test evidence with pairwise KDE
            results = evidence(
                post_samples=post_samples,
                log_posterior_values=log_posterior_values,
                log_posterior_function=mock_log_posterior,
                n_resamples=100,
                thin=1,
                kde_fraction=0.8,
                bridge_start_fraction=0.5,
                max_iter=1000,
                tol=1e-3,
                morph_type="pair",
                param_names=param_names,
                output_path=temp_dir,
                n_estimations=1,
                kde_bw="silverman",
                verbose=False
            )

            print(f"   Evidence results: {results}")

            # Check that bandwidth file was created
            bw_file = os.path.join(temp_dir, "bw_silverman_2D.json")
            assert os.path.exists(bw_file), "Bandwidth file not created"

            # Check that logZ file was created
            logz_file = os.path.join(temp_dir, "logz_morph_z_pair_silverman.txt")
            assert os.path.exists(logz_file), "LogZ file not created"

            logz, error = results[0]
            assert not np.isnan(logz), "logZ should not be NaN"
            assert not np.isnan(error), "Error should not be NaN"

            print(".4f")
            return True

        except Exception as e:
            print(f"❌ Pairwise KDE test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_evidence_tree_kde():
    """Test evidence function with tree KDE."""
    print("\n" + "="*50)
    print("Testing evidence function with tree KDE...")

    # Create test posterior samples
    np.random.seed(42)
    n_samples = 600
    n_params = 3
    post_samples = np.random.randn(n_samples, n_params)
    log_posterior_values = np.array([mock_log_posterior(post_samples[i]) for i in range(n_samples)])
    param_names = ['X', 'Y', 'Z']

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Create a simple tree structure (adjacency list format)
        tree_data = {
            "nodes": ["X", "Y", "Z"],
            "edges": [["X", "Y"], ["Y", "Z"]]
        }
        tree_file = os.path.join(temp_dir, "tree.json")
        with open(tree_file, 'w') as f:
            json.dump(tree_data, f, indent=2)

        try:
            # Test evidence with tree KDE
            results = evidence(
                post_samples=post_samples,
                log_posterior_values=log_posterior_values,
                log_posterior_function=mock_log_posterior,
                n_resamples=100,
                thin=1,
                kde_fraction=0.8,
                bridge_start_fraction=0.5,
                max_iter=1000,
                tol=1e-3,
                morph_type="tree",
                param_names=param_names,
                output_path=temp_dir,
                n_estimations=1,
                kde_bw="silverman",
                verbose=False
            )

            print(f"   Evidence results: {results}")

            # Check that bandwidth file was created
            bw_file = os.path.join(temp_dir, "bw_silverman_2D.json")
            assert os.path.exists(bw_file), "Bandwidth file not created"

            logz, error = results[0]
            assert not np.isnan(logz), "logZ should not be NaN"
            assert not np.isnan(error), "Error should not be NaN"

            print(".4f")
            return True

        except Exception as e:
            print(f"❌ Tree KDE test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Testing evidence function integration with updated bandwidth system...")
    print("="*70)

    success1 = test_evidence_indep_kde()
    success2 = test_evidence_pair_kde()
    success3 = test_evidence_tree_kde()

    if success1 and success2 and success3:
        print("\n🎉 All evidence function tests passed!")
        print("The evidence function works correctly with the updated bandwidth integration.")
    else:
        print("\n❌ Some evidence function tests failed!")
        sys.exit(1)
