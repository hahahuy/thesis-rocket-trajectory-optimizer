"""
Neural network models for PINN training.
"""

from .pinn import PINN
from .residual_net import ResidualNet
from .direction_d_pinn import DirectionDPINN, DirectionDPINN_D1
from .latent_ode import (
    RocketLatentODEPINN,
    LatentDynamicsNet,
    LatentODEBlock,
    LatentODEBlockRK4,
)
from .sequence_pinn import RocketSequencePINN
from .hybrid_pinn import (
    RocketHybridPINN,
    RocketHybridPINNC1,
    RocketHybridPINNC2,
    RocketHybridPINNC3,
)
from .branches import (
    MassBranch,
    RotationBranch,
    TranslationBranch,
    MonotonicMassBranch,
    RotationBranchMinimal,
    PhysicsAwareTranslationBranch,
    rotation_vector_to_quaternion,
)
from .coordination import CoordinatedBranches, AerodynamicCouplingModule
from .physics_layers import PhysicsComputationLayer
from .shared_stem import SharedStem
from .architectures import (
    ContextEncoder,
    DeepContextEncoder,
    FourierFeatures,
    MLP,
    MLPBlock,
    OutputHeads,
    TimeEmbedding,
    normalize_quaternion,
)

__all__ = [
    "PINN",
    "ResidualNet",
    "DirectionDPINN",
    "DirectionDPINN_D1",
    "RocketLatentODEPINN",
    "RocketSequencePINN",
    "RocketHybridPINN",
    "RocketHybridPINNC1",
    "RocketHybridPINNC2",
    "RocketHybridPINNC3",
    "TranslationBranch",
    "RotationBranch",
    "MassBranch",
    "MonotonicMassBranch",
    "RotationBranchMinimal",
    "PhysicsAwareTranslationBranch",
    "SharedStem",
    "LatentDynamicsNet",
    "LatentODEBlock",
    "LatentODEBlockRK4",
    "CoordinatedBranches",
    "AerodynamicCouplingModule",
    "PhysicsComputationLayer",
    "ContextEncoder",
    "DeepContextEncoder",
    "FourierFeatures",
    "MLP",
    "MLPBlock",
    "OutputHeads",
    "TimeEmbedding",
    "normalize_quaternion",
    "rotation_vector_to_quaternion",
]

