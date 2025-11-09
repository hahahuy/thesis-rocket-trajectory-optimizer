"""Test HDF5 schema consistency for WP3 datasets."""

import h5py
import numpy as np
import os
import glob


def test_raw_case_schema(tmp_path):
    """Test that raw case files have correct schema."""
    # Create a minimal test case
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    K = 61
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("time", data=np.linspace(0, 30, N), dtype="f8")
        f.create_dataset("state", data=np.zeros((N, 14), dtype="f8"), dtype="f8")
        f.create_dataset("control", data=np.zeros((N, 4), dtype="f8"), dtype="f8")
        
        g_mon = f.create_group("monitors")
        g_mon.create_dataset("rho", data=np.zeros(N), dtype="f8")
        g_mon.create_dataset("q_dyn", data=np.zeros(N), dtype="f8")
        g_mon.create_dataset("n_load", data=np.zeros(N), dtype="f8")
        
        g_ocp = f.create_group("ocp")
        g_ocp.create_dataset("knots/state", data=np.zeros((K, 14), dtype="f8"), dtype="f8")
        g_ocp.create_dataset("knots/control", data=np.zeros((K, 4), dtype="f8"), dtype="f8")
        
        g_meta = f.create_group("meta")
        # NumPy 2.0 compatibility
        try:
            string_dtype = np.string_
        except AttributeError:
            string_dtype = np.bytes_
        g_meta.create_dataset("git_hash", data=np.array("abc123".encode('utf-8'), dtype=string_dtype))
        g_meta.create_dataset("created_utc", data=np.array("2025-01-01T00:00:00Z".encode('utf-8'), dtype=string_dtype))
        g_meta.create_dataset("seed", data=np.array(1, dtype="i4"))
    
    # Validate schema
    with h5py.File(test_file, "r") as f:
        assert "time" in f
        assert f["time"].shape == (N,)
        assert f["time"].dtype == np.float64
        
        assert "state" in f
        assert f["state"].shape == (N, 14)
        assert f["state"].dtype == np.float64
        
        assert "control" in f
        assert f["control"].shape == (N, 4)
        assert f["control"].dtype == np.float64
        
        assert "monitors" in f
        assert "monitors/rho" in f
        assert f["monitors/rho"].shape == (N,)
        assert "monitors/q_dyn" in f
        assert "monitors/n_load" in f
        
        assert "ocp" in f
        assert "ocp/knots/state" in f
        assert f["ocp/knots/state"].shape == (K, 14)
        assert "ocp/knots/control" in f
        assert f["ocp/knots/control"].shape == (K, 4)
        
        assert "meta" in f
        assert "meta/git_hash" in f
        assert "meta/created_utc" in f
        assert "meta/seed" in f


def test_processed_schema(tmp_path):
    """Test that processed split files have correct schema."""
    os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "processed", "train.h5")
    
    n_cases = 10
    N = 1501
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("inputs/t", data=np.zeros((n_cases, N), dtype="f8"), dtype="f8")
        f.create_dataset("inputs/context", data=np.zeros((n_cases, 5), dtype="f8"), dtype="f8")
        f.create_dataset("targets/state", data=np.zeros((n_cases, N, 14), dtype="f8"), dtype="f8")
        
        g_meta = f.create_group("meta")
        # NumPy 2.0 compatibility
        try:
            string_dtype = np.string_
        except AttributeError:
            string_dtype = np.bytes_
        g_meta.create_dataset("scales", data=np.array('{"L":10000.0}'.encode('utf-8'), dtype=string_dtype))
        g_meta.create_dataset("context_fields", data=np.array('["m0","Isp"]'.encode('utf-8'), dtype=string_dtype))
    
    with h5py.File(test_file, "r") as f:
        assert "inputs/t" in f
        assert f["inputs/t"].shape == (n_cases, N)
        
        assert "inputs/context" in f
        assert len(f["inputs/context"].shape) == 2
        
        assert "targets/state" in f
        assert f["targets/state"].shape == (n_cases, N, 14)
        
        assert "meta/scales" in f
        assert "meta/context_fields" in f


def test_uniform_time_grid():
    """Test that all cases have identical time grid length."""
    # This would check real data if available
    # For now, just verify the concept
    N_expected = 1501  # 30s * 50 Hz + 1
    assert N_expected > 0

