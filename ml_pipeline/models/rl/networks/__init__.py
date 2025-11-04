"""
Neural network architectures for RL agents.
"""

from .modular_ppo import (
    ConfigEncoder,
    PortfolioEncoder,
    ExecutionEncoder,
    OpportunityEncoder,
    FusionLayer,
    ActorHead,
    CriticHead,
    ModularPPONetwork,
)

__all__ = [
    'ConfigEncoder',
    'PortfolioEncoder',
    'ExecutionEncoder',
    'OpportunityEncoder',
    'FusionLayer',
    'ActorHead',
    'CriticHead',
    'ModularPPONetwork',
]
