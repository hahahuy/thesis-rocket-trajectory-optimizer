"""
Direct collocation transcription for optimal control problem.

Converts continuous OCP to discrete NLP using Hermite-Simpson method.
"""

import casadi as ca
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from .collocation import compute_hermite_simpson_step
from .dynamics_casadi import compute_dynamics, compute_dynamic_pressure, compute_load_factor


class DirectCollocation:
    """
    Direct collocation transcription using Hermite-Simpson method.
    """
    
    def __init__(
        self,
        nx: int,
        nu: int,
        N: int,
        params: Dict,
        f: Optional[Callable] = None,
        scaling: Optional[Dict] = None
    ):
        """
        Initialize direct collocation transcription.
        
        Args:
            nx: State dimension
            nu: Control dimension
            N: Number of collocation intervals
            params: Physical parameters
            f: Dynamics function (default: compute_dynamics)
            scaling: Scaling factors for states/controls
        """
        self.nx = nx
        self.nu = nu
        self.N = N
        self.params = params
        self.f = f if f is not None else compute_dynamics
        self.scaling = scaling or {}
        
        # Scaling factors (default to 1.0, ensure float/numeric types)
        x_scale_raw = self.scaling.get('x_scale', np.ones(nx))
        u_scale_raw = self.scaling.get('u_scale', np.ones(nu))
        t_scale_raw = self.scaling.get('t_scale', 1.0)
        
        # Convert to numpy arrays with float dtype
        if isinstance(x_scale_raw, np.ndarray):
            self.x_scale = x_scale_raw.astype(float)
        else:
            self.x_scale = np.array([float(x) for x in x_scale_raw], dtype=float)
        
        if isinstance(u_scale_raw, np.ndarray):
            self.u_scale = u_scale_raw.astype(float)
        else:
            self.u_scale = np.array([float(u) for u in u_scale_raw], dtype=float)
        
        self.t_scale = float(t_scale_raw) if isinstance(t_scale_raw, (int, float, np.number)) else float(t_scale_raw)
        
        # NLP variables
        self.X = None  # States (nx, N+1)
        self.U = None  # Controls (nu, N)
        self.tf = None  # Final time (scalar)
        
        # NLP structure
        self.nlp_vars = None
        self.nlp_obj = None
        self.nlp_g = None
        
    def create_nlp_variables(self, tf_free: bool = False) -> Tuple[ca.MX, ca.MX, ca.MX]:
        """
        Create NLP decision variables.
        
        Args:
            tf_free: Whether final time is free (optimization variable)
            
        Returns:
            X: State variables (nx, N+1)
            U: Control variables (nu, N)
            tf: Final time variable (scalar or fixed)
        """
        # State variables at all nodes
        self.X = ca.MX.sym('X', self.nx, self.N + 1)
        
        # Control variables at all intervals
        self.U = ca.MX.sym('U', self.nu, self.N)
        
        # Final time (free or fixed)
        if tf_free:
            self.tf = ca.MX.sym('tf', 1, 1)
        else:
            # Get from params or ocp_config
            tf_val = self.params.get('tf_fixed', None)
            if tf_val is None:
                # Try to get from config (will be set later)
                tf_val = 100.0
            self.tf = float(tf_val)
        
        return self.X, self.U, self.tf
    
    def compute_defect_constraints(self) -> ca.MX:
        """
        Compute all defect constraints.
        
        Returns:
            g_defect: Defect constraints (nx * N,)
        """
        if self.X is None or self.U is None:
            raise ValueError("NLP variables not created. Call create_nlp_variables() first.")
        
        dt = self.tf / self.N
        
        # Apply scaling to time (ensure numeric type)
        t_scale_val = float(self.t_scale) if isinstance(self.t_scale, (int, float, np.number)) else float(self.t_scale)
        dt_scaled = dt / t_scale_val
        
        # Apply scaling to states and controls
        # Convert scales to float array if numpy, ensure proper type
        if isinstance(self.x_scale, np.ndarray):
            x_scale_vals = self.x_scale.astype(float)
        else:
            x_scale_vals = np.array([float(x) for x in self.x_scale], dtype=float)
        
        if isinstance(self.u_scale, np.ndarray):
            u_scale_vals = self.u_scale.astype(float)
        else:
            u_scale_vals = np.array([float(u) for u in self.u_scale], dtype=float)
        
        # Scale states (element-wise division with CasADi)
        X_scaled = ca.MX.zeros(self.nx, self.N + 1)
        for i in range(self.nx):
            scale_val = float(x_scale_vals[i])
            X_scaled[i, :] = self.X[i, :] / scale_val
        
        # Scale controls
        U_scaled = ca.MX.zeros(self.nu, self.N)
        for i in range(self.nu):
            scale_val = float(u_scale_vals[i])
            U_scaled[i, :] = self.U[i, :] / scale_val
        
        defects = []
        
        for k in range(self.N):
            # Scaled states and controls
            x_k_scaled = X_scaled[:, k]
            x_kp1_scaled = X_scaled[:, k + 1]
            u_k_scaled = U_scaled[:, k]
            u_kp1_scaled = U_scaled[:, k] if k == self.N - 1 else U_scaled[:, k + 1]
            
            # Compute defect using scaled variables
            defect_scaled = compute_hermite_simpson_step(
                self.f, x_k_scaled, u_k_scaled, x_kp1_scaled, u_kp1_scaled,
                dt_scaled, self.params
            )
            
            # Unscale defect (vectorized multiplication for better AD)
            # Convert scale to CasADi DM for element-wise multiplication
            scale_vec = ca.DM(x_scale_vals)
            # Ensure no zero scales (replace with 1.0)
            scale_vec_safe = ca.fmax(ca.fabs(scale_vec), 1e-10)
            # Vectorized unscaling
            defect_unscaled = defect_scaled * scale_vec_safe
            defects.append(defect_unscaled)
        
        # Stack all defects (vertcat creates column vector automatically)
        if len(defects) == 0:
            g_defect = ca.MX.zeros(0, 1)
        else:
            # defects is list of (nx,) vectors, vertcat stacks them vertically
            g_defect = ca.vertcat(*defects)  # Results in (nx*N, 1) column vector
        
        return g_defect
    
    def compute_objective(self, objective_type: str = "fuel_minimization") -> ca.MX:
        """
        Compute objective function.
        
        Args:
            objective_type: Type of objective ('fuel_minimization', 'time_minimization', 'weighted')
            
        Returns:
            J: Objective value (scalar)
        """
        if self.X is None:
            raise ValueError("NLP variables not created. Call create_nlp_variables() first.")
        
        m0 = self.X[13, 0]  # Initial mass
        mf = self.X[13, -1]  # Final mass
        
        if objective_type == "fuel_minimization":
            # J = m(0) - m(tf)
            J = m0 - mf
            
        elif objective_type == "time_minimization":
            # J = tf
            J = self.tf
            
        elif objective_type == "weighted":
            # J = lambda1 * tf + lambda2 * (m0 - mf)
            lambda1 = self.params.get('lambda1', 1.0)
            lambda2 = self.params.get('lambda2', 0.0)
            J = lambda1 * self.tf + lambda2 * (m0 - mf)
            
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        return J
    
    def compute_path_constraints(self, constraint_types: Dict[str, bool]) -> Tuple[ca.MX, ca.MX, ca.MX]:
        """
        Compute path constraints.
        
        Args:
            constraint_types: Dictionary of constraint flags
                {'dynamic_pressure': True, 'load_factor': True, 'mass': True}
        
        Returns:
            g_q: Dynamic pressure constraints (N+1,) or None
            g_n: Load factor constraints (N+1,) or None
            g_m: Mass constraints (N+1,) or None
        """
        if self.X is None or self.U is None:
            raise ValueError("NLP variables not created. Call create_nlp_variables() first.")
        
        g_q_list = []
        g_n_list = []
        g_m_list = []
        
        for k in range(self.N + 1):
            x_k = self.X[:, k]
            u_k = self.U[:, k] if k < self.N else self.U[:, -1]
            
            # Dynamic pressure constraint
            if constraint_types.get('dynamic_pressure', False):
                q = compute_dynamic_pressure(x_k, self.params)
                q_max = self.params.get('q_max', 50000.0)
                g_q_list.append(q - q_max)
            
            # Load factor constraint
            if constraint_types.get('load_factor', False):
                n = compute_load_factor(x_k, u_k, self.params)
                n_max = self.params.get('n_max', 10.0)
                g_n_list.append(n - n_max)
            
            # Mass constraint
            if constraint_types.get('mass', False):
                m = x_k[13]
                m_dry = self.params.get('m_dry', 1000.0)
                g_m_list.append(m_dry - m)
        
        # Ensure column vectors (CasADi vertcat already creates column vectors)
        g_q = ca.vertcat(*g_q_list) if g_q_list else None
        g_n = ca.vertcat(*g_n_list) if g_n_list else None
        g_m = ca.vertcat(*g_m_list) if g_m_list else None
        
        return g_q, g_n, g_m
    
    def create_nlp(
        self,
        objective_type: str = "fuel_minimization",
        constraint_types: Optional[Dict[str, bool]] = None,
        tf_free: bool = False
    ) -> Dict:
        """
        Create complete NLP problem.
        
        Args:
            objective_type: Type of objective
            constraint_types: Dictionary of constraint flags
            tf_free: Whether final time is free
        
        Returns:
            nlp: Dictionary with 'x', 'f', 'g', 'p' keys
        """
        # Create variables
        X, U, tf = self.create_nlp_variables(tf_free=tf_free)
        
        # Compute objective
        J = self.compute_objective(objective_type)
        
        # Compute defect constraints
        g_defect = self.compute_defect_constraints()
        # g_defect should already be a column vector from vertcat
        
        # Compute path constraints
        constraint_types = constraint_types or {
            'dynamic_pressure': True,
            'load_factor': True,
            'mass': True
        }
        g_q, g_n, g_m = self.compute_path_constraints(constraint_types)
        
        # Collect all constraints (ensure all are column vectors)
        g_list = []
        
        # Defect constraints
        if g_defect is not None:
            g_list.append(g_defect)
        
        # Path constraints (already column vectors from compute_path_constraints)
        if g_q is not None:
            g_list.append(g_q)
        if g_n is not None:
            g_list.append(g_n)
        if g_m is not None:
            g_list.append(g_m)
        
        if len(g_list) == 0:
            g = ca.MX.zeros(0, 1)
        elif len(g_list) == 1:
            g = g_list[0]
        else:
            g = ca.vertcat(*g_list)
        
        # Collect all decision variables
        if tf_free:
            x_vars = [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), tf]
        else:
            x_vars = [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)]
        
        x = ca.vertcat(*x_vars)
        
        # Create NLP dictionary
        nlp = {
            'x': x,
            'f': J,
            'g': g
        }
        
        return nlp

