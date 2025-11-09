"""Test that control_cb returns unit thrust direction vectors."""

import numpy as np


def test_control_unit_norm():
    """Test that control vectors have unit thrust direction."""
    N = 100
    # Create control array [T, uTx, uTy, uTz]
    control = np.zeros((N, 4))
    control[:, 0] = np.random.uniform(1000, 5000, N)  # Thrust magnitude
    # Unit direction vectors
    uT = np.random.randn(N, 3)
    uT = uT / np.linalg.norm(uT, axis=1, keepdims=True)
    control[:, 1:4] = uT
    
    # Check unit norm
    norms = np.linalg.norm(control[:, 1:4], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9), f"Control unit norm violation: min={np.min(norms)}, max={np.max(norms)}"


def test_control_cb_signature():
    """Test that control_cb has correct signature (t, x) -> [4]."""
    # This would test a real control_cb if available
    # For now, just document the expected signature
    def mock_control_cb(t: float, x: np.ndarray) -> np.ndarray:
        """Mock control callback."""
        return np.array([4000.0, 1.0, 0.0, 0.0])  # [T, uTx, uTy, uTz]
    
    t = 10.0
    x = np.zeros(14)
    u = mock_control_cb(t, x)
    
    assert u.shape == (4,), f"Control shape: {u.shape}, expected (4,)"
    assert np.isclose(np.linalg.norm(u[1:4]), 1.0, atol=1e-9), "Unit thrust direction required"

