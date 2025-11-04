"""
Reward Function Configuration

As per IMPLEMENTATION_PLAN.md (lines 259-304), the reward function consists of:
1. Hourly P&L reward (main signal)
2. Entry cost penalty
3. Liquidation risk penalty
4. Stop-loss penalty

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
    Configuration for reward function components.

    All parameters are multiplicative scales applied to normalized reward components.
    """

    # Main reward signal: Hourly P&L percentage change
    pnl_reward_scale: float = 3.0

    # Entry cost penalty: Applied to entry fee percentage
    entry_penalty_scale: float = 3.0

    # Liquidation risk penalty: Multiplier for distance below buffer
    liquidation_penalty_scale: float = 20.0

    # Stop-loss trigger penalty: Fixed penalty when position hits stop-loss
    stop_loss_penalty: float = -2.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'pnl_reward_scale': self.pnl_reward_scale,
            'entry_penalty_scale': self.entry_penalty_scale,
            'liquidation_penalty_scale': self.liquidation_penalty_scale,
            'stop_loss_penalty': self.stop_loss_penalty,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'RewardConfig':
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def sample_random(cls,
                      pnl_range=(1.0, 5.0),
                      entry_range=(1.0, 5.0),
                      liq_range=(10.0, 30.0),
                      stop_range=(-5.0, -0.5)) -> 'RewardConfig':
        """
        Sample random reward configuration for PBT exploration.

        Args:
            pnl_range: Range for pnl_reward_scale
            entry_range: Range for entry_penalty_scale
            liq_range: Range for liquidation_penalty_scale
            stop_range: Range for stop_loss_penalty

        Returns:
            RewardConfig with randomly sampled parameters
        """
        return cls(
            pnl_reward_scale=np.random.uniform(*pnl_range),
            entry_penalty_scale=np.random.uniform(*entry_range),
            liquidation_penalty_scale=np.random.uniform(*liq_range),
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
            liquidation_penalty_scale=self.liquidation_penalty_scale * multiplier,
            stop_loss_penalty=self.stop_loss_penalty * multiplier,
        )

    def __repr__(self) -> str:
        return (
            f"RewardConfig(pnl={self.pnl_reward_scale:.2f}, "
            f"entry={self.entry_penalty_scale:.2f}, "
            f"liq={self.liquidation_penalty_scale:.2f}, "
            f"stop={self.stop_loss_penalty:.2f})"
        )


# Preset configurations for common scenarios

DEFAULT_CONFIG = RewardConfig()

CONSERVATIVE_CONFIG = RewardConfig(
    pnl_reward_scale=2.0,      # Lower reward → less aggressive
    entry_penalty_scale=4.0,   # Higher penalty → fewer trades
    liquidation_penalty_scale=30.0,  # Higher penalty → more cautious
    stop_loss_penalty=-3.0,    # Stronger stop-loss avoidance
)

AGGRESSIVE_CONFIG = RewardConfig(
    pnl_reward_scale=5.0,      # Higher reward → more aggressive
    entry_penalty_scale=2.0,   # Lower penalty → more trades
    liquidation_penalty_scale=15.0,  # Lower penalty → higher risk tolerance
    stop_loss_penalty=-1.0,    # Weaker stop-loss avoidance
)
