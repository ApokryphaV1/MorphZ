#!/usr/bin/env python3
"""
Regression tests for Nth_TC output formats.
Ensures the 2-order export matches the structure used by higher orders.
"""
import json
import os
import sys
import tempfile

import numpy as np

# Allow importing morphZ from the repo without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from morphZ import Nth_TC  # noqa: E402


def _load_json(path: str):
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)


def test_two_order_tc_json_matches_higher_order_structure():
    """2-order TC JSON should use the same [[names...], value] structure as n>2."""
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(150, 4))
    names = [f"param_{i}" for i in range(samples.shape[1])]

    with tempfile.TemporaryDirectory() as tmpdir:
        Nth_TC.compute_and_save_tc(
            samples, names=names, n_order=2, out_path=tmpdir, seed=0
        )
        path_2d = os.path.join(tmpdir, "params_2-order_TC.json")
        assert os.path.exists(path_2d), "2-order TC JSON not created"
        tc_2d = _load_json(path_2d)
        assert tc_2d, "Expected non-empty 2-order TC data"

        for entry in tc_2d[:5]:  # spot-check a few entries
            assert isinstance(entry, list) and len(entry) == 2
            labels, value = entry
            assert isinstance(labels, list) and len(labels) == 2
            assert all(isinstance(label, str) for label in labels)
            assert isinstance(value, (int, float))

        # Sanity-check that higher-order output remains unchanged
        Nth_TC.compute_and_save_tc(
            samples, names=names, n_order=3, out_path=tmpdir, seed=0
        )
        path_3d = os.path.join(tmpdir, "params_3-order_TC.json")
        assert os.path.exists(path_3d), "3-order TC JSON not created"
        tc_3d = _load_json(path_3d)
        assert tc_3d, "Expected non-empty 3-order TC data"

        for entry in tc_3d[:5]:
            assert isinstance(entry, list) and len(entry) == 2
            labels, value = entry
            assert isinstance(labels, list) and len(labels) == 3
            assert all(isinstance(label, str) for label in labels)
            assert isinstance(value, (int, float))


if __name__ == "__main__":
    test_two_order_tc_json_matches_higher_order_structure()
    print("✅ Nth_TC JSON format tests passed!")
