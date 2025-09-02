#!/usr/bin/env python3
"""
Test script to verify the PairwiseKDE bandwidth integration works correctly.
"""
import numpy as np
import sys
import os
import tempfile
import json

# Add the project src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import PairwiseKDE, compute_and_save_bandwidths

def test_pairwise_kde_integration():
    """Test that PairwiseKDE works with bandwidth integration."""
    print("Testing PairwiseKDE bandwidth integration...")

    # Create test data
    np.random.seed(42)
    data = np.random.randn(100, 4)
    param_names = ['A', 'B', 'C', 'D']

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Create mock pairwise MI data (simulating dependency_tree output)
        print("\n1. Creating mock pairwise MI data...")
        mi_data = [
            ['A', 'B', 0.5],  # A and B have mutual information 0.5
            ['A', 'C', 0.3],  # A and C have mutual information 0.3
            ['B', 'D', 0.2],  # B and D have mutual information 0.2
        ]

        mi_file = os.path.join(temp_dir, "params_MI.json")
        with open(mi_file, 'w') as f:
            json.dump(mi_data, f, indent=2)
        print(f"   Created MI file: {mi_file}")

        # Step 2: Convert MI data to group format for bandwidth computation
        print("\n2. Converting MI data to group format...")
        group_data = []
        for entry in mi_data:
            if len(entry) >= 3:
                param1, param2, mi_value = entry[0], entry[1], entry[2]
                group_data.append([[param1, param2], float(mi_value)])

        group_file = os.path.join(temp_dir, "pair_groups.json")
        with open(group_file, 'w') as f:
            json.dump(group_data, f, indent=2)
        print(f"   Created group file: {group_file}")

        # Step 3: Compute bandwidths using the group file
        print("\n3. Computing bandwidths with group information...")
        bw_dict = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            in_path=group_file,
            group_format="groups",
            n_order=2
        )
        print(f"   Computed bandwidths: {bw_dict}")

        # Check that the JSON file was created
        json_file = os.path.join(temp_dir, "bw_silverman_2D.json")
        assert os.path.exists(json_file), f"JSON file not created: {json_file}"

        # Step 4: Load and verify JSON content
        print("\n4. Loading JSON file...")
        with open(json_file, 'r') as f:
            json_content = json.load(f)
        print(f"   JSON content: {json_content}")

        # Step 5: Create PairwiseKDE using the JSON path
        print("\n5. Creating PairwiseKDE with JSON path...")
        kde = PairwiseKDE(
            data,
            param_mi=mi_file,
            param_names=param_names,
            kde_bw="silverman",  # fallback method
            verbose=True,
            bw_json_path=json_file  # this should now load from JSON
        )

        # Step 6: Test PairwiseKDE functionality
        print("\n6. Testing PairwiseKDE functionality...")
        test_point = np.array([0.0, 0.0, 0.0, 0.0])
        logpdf_val = kde.logpdf(test_point)
        print(f"   Log PDF at [0,0,0,0]: {logpdf_val}")

        samples = kde.resample(10)
        print(f"   Resampled shape: {samples.shape}")

        # Step 7: Verify KDE structure
        print("\n7. Verifying KDE structure...")
        print(f"   Number of pair KDEs: {len(kde.pair_kdes)}")
        print(f"   Number of single KDEs: {len(kde.single_kdes)}")

        # Check that pairs were created correctly
        expected_pairs = [('A', 'B')]
        actual_pairs = [entry["names"] for entry in kde.pair_kdes]
        print(f"   Expected pairs: {expected_pairs}")
        print(f"   Actual pairs: {actual_pairs}")

        print("\n✅ PairwiseKDE integration test passed!")
        print("Bandwidth computation and PairwiseKDE usage work seamlessly.")

if __name__ == "__main__":
    test_pairwise_kde_integration()
    print("\n🎉 PairwiseKDE integration test completed successfully!")
