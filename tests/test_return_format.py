#!/usr/bin/env python3
"""
Test script to verify that compute_and_save_bandwidths returns the same format as saved to JSON.
"""
import numpy as np
import sys
import os
import tempfile
import json

# Add the project src directory to the path so we can import morphZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morphZ import compute_and_save_bandwidths

def test_return_format_matches_json():
    """Test that the return value matches what's saved to JSON."""
    print("Testing that compute_and_save_bandwidths return format matches JSON...")

    # Create test data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    param_names = ['param1', 'param2', 'param3']

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Compute bandwidths
        print("\n1. Computing bandwidths...")
        returned_value = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            n_order=1
        )
        print(f"   Returned value: {returned_value}")

        # Step 2: Load and verify JSON content
        print("\n2. Loading JSON file...")
        json_file = os.path.join(temp_dir, "bw_silverman_1D.json")
        with open(json_file, 'r') as f:
            json_content = json.load(f)
        print(f"   JSON content: {json_content}")

        # Step 3: Verify they match
        print("\n3. Verifying formats match...")
        if returned_value == json_content:
            print("✅ Return value matches JSON content exactly!")
            return True
        else:
            print("❌ Return value does not match JSON content!")
            print(f"   Returned: {type(returned_value)} - {returned_value}")
            print(f"   JSON:     {type(json_content)} - {json_content}")
            return False

def test_with_groups():
    """Test the return format with grouped parameters."""
    print("\n" + "="*60)
    print("Testing with grouped parameters...")

    # Create test data
    np.random.seed(42)
    data = np.random.randn(100, 4)
    param_names = ['A', 'B', 'C', 'D']

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock group data
        group_data = [
            [['A', 'B'], 0.8],  # Group of 2 parameters
            [['C', 'D'], 0.6],  # Another group of 2 parameters
        ]

        group_file = os.path.join(temp_dir, "test_groups.json")
        with open(group_file, 'w') as f:
            json.dump(group_data, f, indent=2)

        # Compute bandwidths with groups
        returned_value = compute_and_save_bandwidths(
            data,
            method="silverman",
            param_names=param_names,
            output_path=temp_dir,
            in_path=group_file,
            group_format="groups",
            n_order=2
        )
        print(f"   Returned value: {returned_value}")

        # Load JSON
        json_file = os.path.join(temp_dir, "bw_silverman_2D.json")
        with open(json_file, 'r') as f:
            json_content = json.load(f)
        print(f"   JSON content: {json_content}")

        # Verify match
        if returned_value == json_content:
            print("✅ Grouped return value matches JSON content!")
            return True
        else:
            print("❌ Grouped return value does not match JSON content!")
            return False

if __name__ == "__main__":
    success1 = test_return_format_matches_json()
    success2 = test_with_groups()

    if success1 and success2:
        print("\n🎉 All tests passed! compute_and_save_bandwidths now returns the same format as saved to JSON.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
