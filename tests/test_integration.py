#!/usr/bin/env python3
"""
Test script to demonstrate the fixed KDE bandwidth integration.
"""
import numpy as np
import sys
import os
import tempfile
import shutil

# Add the project src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import KDE_approx, compute_and_save_bandwidths

def test_bandwidth_integration():
    """Test that bandwidth computation and KDE usage work together seamlessly."""
    print("Testing bandwidth computation and KDE integration...")

    # Create test data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    param_names = ['param1', 'param2', 'param3']

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Compute and save bandwidths (simulating what evidence() function does)
        print("\n1. Computing bandwidths...")
        bw_dict = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            n_order=1
        )
        print(f"   Computed bandwidths: {bw_dict}")

        # Check that the JSON file was created
        json_file = os.path.join(temp_dir, "bw_silverman_1D.json")
        assert os.path.exists(json_file), f"JSON file not created: {json_file}"

        # Step 2: Load and verify JSON content
        print("\n2. Loading JSON file...")
        import json
        with open(json_file, 'r') as f:
            json_content = json.load(f)
        print(f"   JSON content: {json_content}")

        # Step 3: Create KDE using the JSON path (simulating what KDE classes do)
        print("\n3. Creating KDE with JSON path...")
        kde = KDE_approx(
            data,
            kde_bw="silverman",  # fallback method
            param_names=param_names,
            bw_json_path=json_file  # this should now load from JSON
        )

        # Step 4: Test KDE functionality
        print("\n4. Testing KDE functionality...")
        test_point = np.array([0.0, 0.0, 0.0])
        logpdf_val = kde.logpdf_kde(test_point)
        print(f"   Log PDF at [0,0,0]: {logpdf_val}")

        samples = kde.resample(10)
        print(f"   Resampled shape: {samples.shape}")

        print("\n✅ Integration test passed! Bandwidth computation and KDE usage work seamlessly.")

if __name__ == "__main__":
    test_bandwidth_integration()
    print("\n🎉 Bandwidth integration test completed successfully!")
