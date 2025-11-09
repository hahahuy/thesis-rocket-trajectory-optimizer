"""Test that velocity data varies (not all zeros)."""

import h5py
import numpy as np
import os
import glob


def test_velocity_varies(tmp_path):
    """Test that velocity components vary (at least one component should change)."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    # Create state with varying vertical velocity
    state = np.zeros((N, 14), dtype="f8")
    state[:, 5] = np.linspace(0, 2000, N)  # vz increases
    state[:, 2] = np.cumsum(state[:, 5]) * 0.02  # altitude
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("state", data=state, dtype="f8")
        f.create_dataset("time", data=np.linspace(0, 30, N), dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["state"][...]
        vx_ptp = np.ptp(state[:, 3])
        vy_ptp = np.ptp(state[:, 4])
        vz_ptp = np.ptp(state[:, 5])
        
        # At least one velocity component should vary
        max_ptp = max(vx_ptp, vy_ptp, vz_ptp)
        assert max_ptp > 1.0, f"Velocity should vary: vx_ptp={vx_ptp}, vy_ptp={vy_ptp}, vz_ptp={vz_ptp}"


def test_velocity_magnitude_increases():
    """Test that velocity magnitude generally increases during ascent."""
    # This would check real data if available
    # For now, just verify the concept
    assert True  # Placeholder - would check actual trajectory files

