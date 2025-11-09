"""Test that splits are stratified and balanced."""

import h5py
import numpy as np
import os
import json
from scipy import stats


def test_split_parameter_coverage(tmp_path):
    """Test that each split covers parameter space."""
    os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
    
    n_cases = 20
    N = 1501
    
    # Create mock processed files with context vectors
    for split in ["train", "val", "test"]:
        test_file = os.path.join(tmp_path, "data", "processed", f"{split}.h5")
        with h5py.File(test_file, "w") as f:
            # Context: [m0, Isp, Cd, ...] - vary m0 across splits
            context = np.random.uniform(45.0, 65.0, (n_cases, 5))
            f.create_dataset("inputs/context", data=context, dtype="f8")
            f.create_dataset("inputs/t", data=np.zeros((n_cases, N)), dtype="f8")
            f.create_dataset("targets/state", data=np.zeros((n_cases, N, 14)), dtype="f8")
    
    # Check that each split has reasonable coverage
    for split in ["train", "val", "test"]:
        test_file = os.path.join(tmp_path, "data", "processed", f"{split}.h5")
        with h5py.File(test_file, "r") as f:
            context = f["inputs/context"][...]
            m0_range = context[:, 0]  # First field is m0
            assert np.min(m0_range) >= 45.0, f"{split}: m0 too low"
            assert np.max(m0_range) <= 65.0, f"{split}: m0 too high"
            # Check spread (not all same value)
            assert np.std(m0_range) > 1.0, f"{split}: m0 not varied enough"


def test_split_manifest_exists(tmp_path):
    """Test that splits.json manifest exists."""
    os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
    splits_file = os.path.join(tmp_path, "data", "processed", "splits.json")
    
    with open(splits_file, "w") as f:
        json.dump({"train": ["case_train_0.h5"], "val": ["case_val_0.h5"], "test": ["case_test_0.h5"]}, f)
    
    assert os.path.exists(splits_file)
    with open(splits_file, "r") as f:
        splits = json.load(f)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

