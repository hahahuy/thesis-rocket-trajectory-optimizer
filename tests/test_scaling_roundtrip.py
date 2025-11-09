"""Test scaling round-trip accuracy."""

import numpy as np
from src.data.preprocess import to_nd, from_nd, Scales


def test_scaling_roundtrip():
    """Test that to_nd -> from_nd round-trip error < 1e-9."""
    scales = Scales(L=10000.0, V=313.0, T=31.62, M=50.0, F=490.0, W=0.0316)
    
    N = 100
    # Random state [x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, wx,wy,wz, m]
    state = np.random.randn(N, 14).astype(np.float64)
    state[:, 0:3] *= 5000.0  # positions in m
    state[:, 3:6] *= 200.0  # velocities in m/s
    state[:, 6:10] = state[:, 6:10] / np.linalg.norm(state[:, 6:10], axis=1, keepdims=True)  # normalized quat
    state[:, 10:13] *= 0.1  # angular rates in rad/s
    state[:, 13:14] *= 50.0  # mass in kg
    
    # Random control [T, uTx, uTy, uTz]
    control = np.random.randn(N, 4).astype(np.float64)
    control[:, 0:1] *= 2000.0  # thrust in N
    control[:, 1:4] = control[:, 1:4] / np.linalg.norm(control[:, 1:4], axis=1, keepdims=True)  # unit direction
    
    # Random time
    t = np.linspace(0, 30, N).astype(np.float64)
    
    # Round-trip
    state_nd, control_nd, t_nd = to_nd(state, control, t, scales)
    state_rt, control_rt, t_rt = from_nd(state_nd, control_nd, t_nd, scales)
    
    # Check errors
    state_err = np.abs(state - state_rt)
    control_err = np.abs(control - control_rt)
    t_err = np.abs(t - t_rt)
    
    max_state_err = np.max(state_err)
    max_control_err = np.max(control_err)
    max_t_err = np.max(t_err)
    
    assert max_state_err < 1e-9, f"State round-trip error: {max_state_err}"
    assert max_control_err < 1e-9, f"Control round-trip error: {max_control_err}"
    assert max_t_err < 1e-9, f"Time round-trip error: {max_t_err}"


def test_angular_rate_scaling():
    """Test that angular rates are scaled by W."""
    scales = Scales(L=10000.0, V=313.0, T=31.62, M=50.0, F=490.0, W=0.0316)
    
    N = 10
    state = np.zeros((N, 14))
    state[:, 10:13] = 1.0  # 1 rad/s
    
    control = np.zeros((N, 4))
    t = np.linspace(0, 1, N)
    
    state_nd, _, _ = to_nd(state, control, t, scales)
    
    # Angular rates should be scaled by W
    expected = 1.0 / scales.W
    actual = state_nd[0, 10]  # First angular rate
    
    assert np.isclose(actual, expected, rtol=1e-6), f"Angular rate scaling: expected {expected}, got {actual}"


def test_quaternion_unscaled():
    """Test that quaternions are NOT scaled."""
    scales = Scales(L=10000.0, V=313.0, T=31.62, M=50.0, F=490.0, W=0.0316)
    
    N = 10
    state = np.zeros((N, 14))
    state[:, 6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # Unit quaternion
    
    control = np.zeros((N, 4))
    t = np.linspace(0, 1, N)
    
    state_nd, _, _ = to_nd(state, control, t, scales)
    
    # Quaternion should be unchanged
    quat_nd = state_nd[:, 6:10]
    assert np.allclose(quat_nd, state[:, 6:10], atol=1e-9), "Quaternions should not be scaled"

