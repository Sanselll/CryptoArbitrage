"""
Reward Function Configuration (Simplified RL-v2 Approach)

Simplified reward function inspired by RL-v2 philosophy:
"Reward what you want (profit), not how to get it"

The reward function consists of only 3 components:
1. Hourly P&L reward (main signal) - Agent learns from actual outcomes
2. Entry cost penalty - Discourages overtrading
3. Stop-loss penalty - Basic risk management

This module provides a dataclass for configuring reward scales, enabling:
- Grid search over reward parameters
- PBT tuning of reward function
- Reproducible reward configurations
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RewardConfig:
    """
    Configuration for reward function components (Simplified RL-v2 approach).

    Philosophy: Let agent learn optimal behavior from P&L outcomes, not from
    artificial incentives. Agent discovers when to enter/exit through experience.

    All parameters are multiplicative scales applied to normalized reward components.
    """

    # Main reward signal: Hourly P&L percentage change
    # Agent learns from actual outcomes - holding winners accumulates P&L,
    # exiting early or holding losers reduces P&L
    pnl_reward_scale: float = 3.0

    # Entry cost penalty: Applied to entry fee percentage
    # CRITICAL: High penalty to prevent overtrading
    # With 0.08% entry fee, penalty = -0.16 per trade (makes agent selective)
    entry_penalty_scale: float = 2.0

    # Stop-loss trigger penalty: Fixed penalty when position hits stop-loss
    # Simple risk management to avoid catastrophic losses
    stop_loss_penalty: float = -1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'pnl_reward_scale': self.pnl_reward_scale,
            'entry_penalty_scale': self.entry_penalty_scale,
            'stop_loss_penalty': self.stop_loss_penalty,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'RewardConfig':
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def sample_random(cls,
                      pnl_range=(1.0, 5.0),
                      entry_range=(0.5, 2.0),
                      stop_range=(-3.0, -0.5)) -> 'RewardConfig':
        """
        Sample random reward configuration for PBT exploration.

        Args:
            pnl_range: Range for pnl_reward_scale
            entry_range: Range for entry_penalty_scale
            stop_range: Range for stop_loss_penalty

        Returns:
            RewardConfig with randomly sampled parameters
        """
        return cls(
            pnl_reward_scale=np.random.uniform(*pnl_range),
            entry_penalty_scale=np.random.uniform(*entry_range),
            stop_loss_penalty=np.random.uniform(*stop_range),
        )

    def perturb(self, factor: float = 0.2) -> 'RewardConfig':
        """
        Perturb parameters for PBT exploration.

        Args:
            factor: Perturbation factor (default: 0.2 = ±20%)

        Returns:
            New RewardConfig with perturbed parameters
        """
        multiplier = np.random.choice([1.0 - factor, 1.0 + factor])

        return RewardConfig(
            pnl_reward_scale=self.pnl_reward_scale * multiplier,
            entry_penalty_scale=self.entry_penalty_scale * multiplier,
            stop_loss_penalty=self.stop_loss_penalty * multiplier,
        )

    def __repr__(self) -> str:
        return (
            f"RewardConfig(pnl={self.pnl_reward_scale:.2f}, "
            f"entry={self.entry_penalty_scale:.2f}, "
            f"stop={self.stop_loss_penalty:.2f})"
        )


# Preset configurations for common scenarios

DEFAULT_CONFIG = RewardConfig()

CONSERVATIVE_CONFIG = RewardConfig(
    pnl_reward_scale=2.0,      # Lower reward → less aggressive learning
    entry_penalty_scale=2.0,   # Higher penalty → fewer trades
    stop_loss_penalty=-2.0,    # Stronger stop-loss avoidance
)

AGGRESSIVE_CONFIG = RewardConfig(
    pnl_reward_scale=5.0,      # Higher reward → more aggressive learning
    entry_penalty_scale=0.5,   # Lower penalty → more trades
    stop_loss_penalty=-0.5,    # Weaker stop-loss avoidance (higher risk tolerance)
)
