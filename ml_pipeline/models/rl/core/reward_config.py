"""
Reward Function Configuration (Pure RL-v2 Approach)

Pure outcome-based reward function following true RL-v2 philosophy:
"Reward what you want (profit), not how to get it"

The reward function consists of 2 components:
1. Funding P&L reward (main signal, 5x weight) - Focus on funding fees
2. Price P&L reward (secondary signal, 1x weight) - Total outcome matters
3. Liquidation risk penalty (safety critical) - Prevent capital loss

Removed behavioral shaping:
- No entry penalty (learn selectivity through outcomes)
- No stop-loss penalty (natural negative P&L is sufficient)
- No opportunity cost (learn through missed profits)

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
    Configuration for reward function components (Pure RL-v2 approach).

    Philosophy: Agent learns optimal behavior purely from P&L outcomes.
    Emphasis on funding P&L (real profit source) over price movements (noise).
    Only exception: liquidation penalty for safety during exploration.

    All parameters are multiplicative scales applied to normalized reward components.
    """

    # Funding P&L reward: 10x scale to overcome entropy noise
    # V3.1: Increased from 1.0 to 10.0 to provide stronger learning signal
    # Funding fees are paid every 1-8 hours and are the actual arbitrage profit
    # Equal weights let agent learn natural balance between funding and price
    funding_reward_scale: float = 10.0

    # Price P&L reward: 10x scale to overcome entropy noise
    # V3.1: Increased from 1.0 to 10.0 to provide stronger learning signal
    # Both components matter for total P&L outcome
    # Equal weights prevent over-optimization on either component
    price_reward_scale: float = 10.0

    # Liquidation risk penalty: Critical safety mechanism
    # Applied when position approaches liquidation threshold
    # Prevents capital loss during exploration phase
    # Scale: penalty = (buffer - distance) * scale
    # Set to 10 for trained agents (less conservative), 0 to disable
    liquidation_penalty_scale: float = 10.0

    # Opportunity cost penalty: DISABLED BY DEFAULT
    # ISSUE: Cumulative penalties over long episodes (3088 steps) overwhelm P&L signal
    # Example: -0.0125 per step × 3088 steps = -38.6 total (vs ~7% P&L)
    # Result: Agent overtrades desperately to avoid accumulation (1545 trades, 12-min holds)
    #
    # RECOMMENDED APPROACH: Rely on APR comparison features (Phase 2) for rotation learning
    # - Features already in observation: current_position_apr, best_available_apr, apr_advantage
    # - Agent learns rotation from P&L outcomes (RL-v2: "Reward what you want, not how to get it")
    #
    # If rotation learning is insufficient, consider one-time entry penalty instead
    # Set to 0.0 to disable (recommended), 0.03-0.10 for experimentation only
    opportunity_cost_scale: float = 0.0

    # Negative funding exit reward: Bonus for exiting positions with negative estimated funding
    # When estimated_funding_8h_pct < 0, the position is paying funding (losing money)
    # Reward agent for recognizing and exiting these losing positions quickly
    # Scale: reward = abs(estimated_funding_8h_pct) * scale (applied on EXIT action)
    # Recommended: 1.0-5.0 (1.0 = neutral, 5.0 = strong incentive)
    # Set to 0.0 to disable
    negative_funding_exit_reward_scale: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'funding_reward_scale': self.funding_reward_scale,
            'price_reward_scale': self.price_reward_scale,
            'liquidation_penalty_scale': self.liquidation_penalty_scale,
            'opportunity_cost_scale': self.opportunity_cost_scale,
            'negative_funding_exit_reward_scale': self.negative_funding_exit_reward_scale,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'RewardConfig':
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def sample_random(cls,
                      funding_range=(0.5, 2.0),
                      price_range=(0.5, 2.0),
                      liquidation_range=(5.0, 20.0),
                      opportunity_cost_range=(0.0, 0.15),
                      negative_funding_exit_range=(0.0, 5.0)) -> 'RewardConfig':
        """
        Sample random reward configuration for PBT exploration.

        Args:
            funding_range: Range for funding_reward_scale
            price_range: Range for price_reward_scale
            liquidation_range: Range for liquidation_penalty_scale
            opportunity_cost_range: Range for opportunity_cost_scale
            negative_funding_exit_range: Range for negative_funding_exit_reward_scale

        Returns:
            RewardConfig with randomly sampled parameters
        """
        return cls(
            funding_reward_scale=np.random.uniform(*funding_range),
            price_reward_scale=np.random.uniform(*price_range),
            liquidation_penalty_scale=np.random.uniform(*liquidation_range),
            opportunity_cost_scale=np.random.uniform(*opportunity_cost_range),
            negative_funding_exit_reward_scale=np.random.uniform(*negative_funding_exit_range),
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
            funding_reward_scale=self.funding_reward_scale * multiplier,
            price_reward_scale=self.price_reward_scale * multiplier,
            liquidation_penalty_scale=self.liquidation_penalty_scale * multiplier,
            opportunity_cost_scale=max(0.0, self.opportunity_cost_scale * multiplier),  # Keep >= 0
            negative_funding_exit_reward_scale=max(0.0, self.negative_funding_exit_reward_scale * multiplier),  # Keep >= 0
        )

    def __repr__(self) -> str:
        return (
            f"RewardConfig(funding={self.funding_reward_scale:.2f}, "
            f"price={self.price_reward_scale:.2f}, "
            f"liq={self.liquidation_penalty_scale:.0f}, "
            f"opp_cost={self.opportunity_cost_scale:.2f}, "
            f"neg_fund_exit={self.negative_funding_exit_reward_scale:.2f})"
        )


# Preset configurations for common scenarios

DEFAULT_CONFIG = RewardConfig()

CONSERVATIVE_CONFIG = RewardConfig(
    funding_reward_scale=0.5,      # Lower reward → slower learning
    price_reward_scale=0.5,        # Balanced with funding
    liquidation_penalty_scale=20.0,  # Stronger safety
    opportunity_cost_scale=0.0,     # Disabled (see docstring above)
    negative_funding_exit_reward_scale=0.0,  # Disabled by default
)

AGGRESSIVE_CONFIG = RewardConfig(
    funding_reward_scale=2.0,      # Higher reward → faster learning
    price_reward_scale=2.0,        # Balanced with funding
    liquidation_penalty_scale=5.0,   # Weaker safety (explore more)
    opportunity_cost_scale=0.0,     # Disabled (see docstring above)
    negative_funding_exit_reward_scale=0.0,  # Disabled by default
)
