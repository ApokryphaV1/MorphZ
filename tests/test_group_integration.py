#!/usr/bin/env python3
"""
Test script to verify the GroupKDE bandwidth integration works correctly with groups of 3 elements.
"""
import numpy as np
import sys
import os
import tempfile
import json

# Add the src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import GroupKDE, compute_and_save_bandwidths

def test_group_kde_3_elements():
    """Test that GroupKDE works with groups of 3 elements (3D KDEs)."""
    print("Testing GroupKDE bandwidth integration with groups of 3 elements...")

    # Create test data with more samples for better KDE estimation
    np.random.seed(42)
    data = np.random.randn(200, 6)  # 6 parameters, more samples
    param_names = ['A', 'B', 'C', 'D', 'E', 'F']

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Create mock total correlation (TC) data (simulating Nth_TC output)
        print("\n1. Creating mock TC data with groups of 3 elements...")
        tc_data = [
            [['A', 'B', 'C'], 0.8],  # Group of 3 parameters with high correlation
            [['D', 'E', 'F'], 0.7],  # Another group of 3 parameters
        ]

        tc_file = os.path.join(temp_dir, "params_3-order_TC.json")
        with open(tc_file, 'w') as f:
            json.dump(tc_data, f, indent=2)
        print(f"   Created TC file: {tc_file}")

        # Step 2: Compute bandwidths using the group file
        print("\n2. Computing bandwidths with group information...")
        bw_dict = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            in_path=tc_file,
            group_format="groups",
            n_order=3
        )
        print(f"   Computed bandwidths: {bw_dict}")

        # Check that the JSON file was created
        json_file = os.path.join(temp_dir, "bw_silverman_3D.json")
        assert os.path.exists(json_file), f"JSON file not created: {json_file}"

        # Step 3: Load and verify JSON content
        print("\n3. Loading JSON file...")
        with open(json_file, 'r') as f:
            json_content = json.load(f)
        print(f"   JSON content: {json_content}")

        # Step 4: Create GroupKDE using the JSON path
        print("\n4. Creating GroupKDE with JSON path...")
        kde = GroupKDE(
            data,
            param_tc=tc_file,
            param_names=param_names,
            kde_bw="silverman",  # fallback method
            verbose=True,
            bw_json_path=json_file  # this should now load from JSON
        )

        # Step 5: Test GroupKDE functionality
        print("\n5. Testing GroupKDE functionality...")
        test_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        logpdf_val = kde.logpdf(test_point)
        print(f"   Log PDF at [0,0,0,0,0,0]: {logpdf_val}")

        samples = kde.resample(10)
        print(f"   Resampled shape: {samples.shape}")

        # Step 6: Verify KDE structure
        print("\n6. Verifying KDE structure...")
        print(f"   Number of group KDEs: {len(kde.group_kdes)}")
        print(f"   Number of single KDEs: {len(kde.single_kdes)}")

        # Check that groups were created correctly
        expected_groups = [('A', 'B', 'C'), ('D', 'E', 'F')]
        actual_groups = [tuple(entry["names"]) for entry in kde.group_kdes]
        print(f"   Expected groups: {expected_groups}")
        print(f"   Actual groups: {actual_groups}")

        # Verify group dimensions
        for i, entry in enumerate(kde.group_kdes):
            group_size = len(entry["names"])
            print(f"   Group {i+1} size: {group_size} parameters")
            assert group_size == 3, f"Expected group size 3, got {group_size}"

        print("\n✅ GroupKDE 3-element integration test passed!")
        print("Bandwidth computation and GroupKDE usage work seamlessly with 3D KDEs.")

def test_group_kde_mixed_sizes():
    """Test GroupKDE with mixed group sizes (2 and 3 elements)."""
    print("\n" + "="*60)
    print("Testing GroupKDE with mixed group sizes...")

    # Create test data
    np.random.seed(42)
    data = np.random.randn(150, 5)  # 5 parameters
    param_names = ['P1', 'P2', 'P3', 'P4', 'P5']

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Create mixed group data (2 and 3 elements)
        print("\n1. Creating mixed group data...")
        tc_data = [
            [['P1', 'P2', 'P3'], 0.8],  # Group of 3
            [['P4', 'P5'], 0.6],        # Group of 2
        ]

        tc_file = os.path.join(temp_dir, "params_mixed_TC.json")
        with open(tc_file, 'w') as f:
            json.dump(tc_data, f, indent=2)

        # Compute bandwidths
        print("\n2. Computing bandwidths...")
        bw_dict = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            in_path=tc_file,
            group_format="groups",
            n_order=3
        )
        print(f"   Computed bandwidths: {bw_dict}")

        # Create GroupKDE
        print("\n3. Creating GroupKDE...")
        json_file = os.path.join(temp_dir, "bw_silverman_3D.json")
        kde = GroupKDE(
            data,
            param_tc=tc_file,
            param_names=param_names,
            kde_bw="silverman",
            verbose=True,
            bw_json_path=json_file
        )

        # Test functionality
        print("\n4. Testing functionality...")
        test_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        logpdf_val = kde.logpdf(test_point)
        print(f"   Log PDF: {logpdf_val}")

        samples = kde.resample(5)
        print(f"   Resampled shape: {samples.shape}")

        # Verify structure
        print("\n5. Verifying structure...")
        print(f"   Groups: {len(kde.group_kdes)}")
        print(f"   Singles: {len(kde.single_kdes)}")

        group_sizes = [len(entry["names"]) for entry in kde.group_kdes]
        print(f"   Group sizes: {group_sizes}")

        assert 3 in group_sizes, "Expected a group of size 3"
        assert 2 in group_sizes, "Expected a group of size 2"

        print("\n✅ Mixed group sizes test passed!")

if __name__ == "__main__":
    test_group_kde_3_elements()
    test_group_kde_mixed_sizes()
    print("\n🎉 All GroupKDE tests completed successfully!")
