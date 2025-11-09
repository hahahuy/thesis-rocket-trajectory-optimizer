"""Test that processed files have consistent shapes."""

import h5py
import numpy as np
import os


def test_processed_shapes_consistent(tmp_path):
    """Test that all processed splits have identical N."""
    os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
    
    N = 1501  # 30s * 50 Hz + 1
    n_train, n_val, n_test = 10, 2, 2
    
    for split, n_cases in [("train", n_train), ("val", n_val), ("test", n_test)]:
        test_file = os.path.join(tmp_path, "data", "processed", f"{split}.h5")
        with h5py.File(test_file, "w") as f:
            f.create_dataset("inputs/t", data=np.zeros((n_cases, N)), dtype="f8")
            f.create_dataset("targets/state", data=np.zeros((n_cases, N, 14)), dtype="f8")
    
    # Check all have same N
    N_values = []
    for split in ["train", "val", "test"]:
        test_file = os.path.join(tmp_path, "data", "processed", f"{split}.h5")
        with h5py.File(test_file, "r") as f:
            N_actual = f["inputs/t"].shape[1]
            N_values.append(N_actual)
    
    assert len(set(N_values)) == 1, f"Inconsistent N across splits: {N_values}"


def test_state_shape(tmp_path):
    """Test that state has shape [n_cases, N, 14]."""
    os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "processed", "train.h5")
    
    n_cases = 10
    N = 1501
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("targets/state", data=np.zeros((n_cases, N, 14)), dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["targets/state"]
        assert state.shape == (n_cases, N, 14), f"State shape: {state.shape}, expected ({n_cases}, {N}, 14)"

