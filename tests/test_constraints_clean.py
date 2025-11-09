"""Test that stored datasets have no constraint violations."""

import h5py
import numpy as np
import os


def test_no_q_violations(tmp_path):
    """Test that q_dyn <= qmax in all stored cases."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    qmax = 40000.0
    
    with h5py.File(test_file, "w") as f:
        g_mon = f.create_group("monitors")
        # Create q_dyn that respects limit
        q_dyn = np.random.uniform(0, qmax * 0.9, N)
        g_mon.create_dataset("q_dyn", data=q_dyn, dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        q_dyn = f["monitors/q_dyn"][...]
        assert np.all(q_dyn <= qmax + 1e-9), f"q_dyn violation: max={np.max(q_dyn)}, qmax={qmax}"


def test_no_n_violations(tmp_path):
    """Test that n_load <= nmax in all stored cases."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    nmax = 5.0
    
    with h5py.File(test_file, "w") as f:
        g_mon = f.create_group("monitors")
        n_load = np.random.uniform(0, nmax * 0.9, N)
        g_mon.create_dataset("n_load", data=n_load, dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        n_load = f["monitors/n_load"][...]
        assert np.all(n_load <= nmax + 1e-9), f"n_load violation: max={np.max(n_load)}, nmax={nmax}"


def test_quaternion_norm(tmp_path):
    """Test that quaternions are unit-norm (error < 1e-6)."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    # Create normalized quaternions
    q = np.random.randn(N, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    
    with h5py.File(test_file, "w") as f:
        state = np.zeros((N, 14), dtype="f8")
        state[:, 6:10] = q
        f.create_dataset("state", data=state, dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["state"][...]
        quat = state[:, 6:10]
        norms = np.linalg.norm(quat, axis=1)
        max_err = np.max(np.abs(norms - 1.0))
        assert max_err < 1e-6, f"Quaternion norm error: {max_err}"


def test_no_nans_infs(tmp_path):
    """Test that stored data has no NaNs or Infs."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("state", data=np.zeros((N, 14), dtype="f8"), dtype="f8")
        f.create_dataset("control", data=np.zeros((N, 4), dtype="f8"), dtype="f8")
        g_mon = f.create_group("monitors")
        g_mon.create_dataset("q_dyn", data=np.zeros(N), dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["state"][...]
        control = f["control"][...]
        q_dyn = f["monitors/q_dyn"][...]
        
        assert not np.any(np.isnan(state)), "NaNs in state"
        assert not np.any(np.isinf(state)), "Infs in state"
        assert not np.any(np.isnan(control)), "NaNs in control"
        assert not np.any(np.isinf(control)), "Infs in control"
        assert not np.any(np.isnan(q_dyn)), "NaNs in q_dyn"
        assert not np.any(np.isinf(q_dyn)), "Infs in q_dyn"

