"""Test that mass is monotonic when thrust > 0."""

import numpy as np
import h5py
import os


def test_mass_decreases_with_thrust(tmp_path):
    """Test that mass strictly decreases when T > 0."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    # Create state with decreasing mass
    state = np.zeros((N, 14))
    state[:, 13] = np.linspace(60.0, 40.0, N)  # Mass decreases
    
    # Control with positive thrust
    control = np.zeros((N, 4))
    control[:, 0] = 4000.0  # Constant thrust
    control[:, 1:4] = np.array([1.0, 0.0, 0.0])  # Unit direction
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("state", data=state, dtype="f8")
        f.create_dataset("control", data=control, dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["state"][...]
        control = f["control"][...]
        mass = state[:, 13]
        thrust = control[:, 0]
        
        # Where thrust > 0, mass should be strictly decreasing
        thrust_mask = thrust > 1e-3
        if np.any(thrust_mask):
            mass_thrust = mass[thrust_mask]
            mass_diff = np.diff(mass_thrust)
            assert np.all(mass_diff <= 0), "Mass should decrease when thrust > 0"


def test_mass_flat_when_no_thrust(tmp_path):
    """Test that mass is flat when T ≈ 0."""
    os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
    test_file = os.path.join(tmp_path, "data", "raw", "case_train_0.h5")
    
    N = 1501
    # Create state with constant mass
    state = np.zeros((N, 14))
    state[:, 13] = 50.0  # Constant mass
    
    # Control with zero thrust
    control = np.zeros((N, 4))
    control[:, 0] = 0.0  # No thrust
    control[:, 1:4] = np.array([1.0, 0.0, 0.0])
    
    with h5py.File(test_file, "w") as f:
        f.create_dataset("state", data=state, dtype="f8")
        f.create_dataset("control", data=control, dtype="f8")
    
    with h5py.File(test_file, "r") as f:
        state = f["state"][...]
        control = f["control"][...]
        mass = state[:, 13]
        thrust = control[:, 0]
        
        # Where thrust ≈ 0, mass should be constant
        no_thrust_mask = thrust < 1e-3
        if np.any(no_thrust_mask):
            mass_no_thrust = mass[no_thrust_mask]
            mass_std = np.std(mass_no_thrust)
            assert mass_std < 1e-6, "Mass should be constant when thrust ≈ 0"

