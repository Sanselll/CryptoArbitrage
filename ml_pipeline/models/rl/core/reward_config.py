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

    # Funding P&L reward: 15x scale for stronger learning signal
    # V6: Increased from 10.0 to 15.0 with higher final entropy (0.05) for balance
    # Funding fees are paid every 1-8 hours and are the actual arbitrage profit
    # Equal weights let agent learn natural balance between funding and price
    funding_reward_scale: float = 15.0

    # Price P&L reward: 15x scale for stronger learning signal
    # V6: Increased from 10.0 to 15.0 with higher final entropy (0.05) for balance
    # Both components matter for total P&L outcome
    # Equal weights prevent over-optimization on either component
    price_reward_scale: float = 15.0

    # Liquidation risk penalty: Critical safety mechanism
    # Applied when position approaches liquidation threshold
    # Prevents capital loss during exploration phase
    # Scale: penalty = (buffer - distance) * scale
    # Set to 10 for trained agents (less conservative), 0 to disable
    liquidation_penalty_scale: float = 10.0

    # Opportunity cost penalty: V6 ENABLED (was disabled)
    # V6 CHANGES: Now uses 200% APR threshold (was 10%) to avoid overtrading
    # Only penalizes when holding SIGNIFICANTLY inferior positions (e.g., 0% vs 1239%)
    #
    # Previous issue: -0.0125 per step × 3088 steps = -38.6 total (vs ~7% P&L)
    # V6 fix: Only penalize for gaps > 200%, with /1000 scaling
    # Example: 1239% gap → -0.03 * 1.039 * 0.3 ≈ -0.01 per step (much smaller)
    #
    # Features already in observation: current_position_apr, best_available_apr, apr_advantage
    # V6 also improved apr_advantage with log scale for large gaps
    #
    # Set to 0.03 for V6 (balanced), 0.0 to disable
    opportunity_cost_scale: float = 0.03

    # Negative funding exit reward: Bonus for exiting positions to rotate to better opportunities
    # Triggers when: fund_apr < 0 OR fund_apr << best_available_apr
    # V6.1: Increased to 2.0 (was 0.5) to encourage rotation behavior
    # Combined with /500 divisor in environment.py for stronger signal
    # Set to 0.0 to disable
    negative_funding_exit_reward_scale: float = 2.0

    # Trade diversity bonus: V6.1 - Encourage exploration through trading
    # Small bonus for each completed trade to prevent "hold forever" strategy
    # Also includes inactivity penalty for holding same positions too long
    # WARNING: Too high causes overtrading (400 trades/ep at 0.5)
    # Set to 0.0 to disable
    trade_diversity_bonus: float = 0.1  # V6.2: Reduced from 0.5 to 0.1 to prevent overtrading
    inactivity_penalty_hours: float = 48.0  # V6.2: Increased from 24 to 48 hours
    inactivity_penalty_scale: float = 0.005  # V6.2: Reduced from 0.01 to 0.005

    # V7: Negative APR penalty - Hourly penalty for holding positions with negative APR
    # Applied each step when position has current_position_apr < 0
    # Encourages faster exit when funding rate flips against position
    # Example: -100% APR → penalty = 100/1000 * 0.02 = 0.002 per step
    # Set to 0.0 to disable
    negative_apr_penalty_scale: float = 0.02

    # V7: APR flip exit bonus - Extra reward for exiting positions when APR flipped
    # Applied when EXIT action is taken and entry_apr > 0 but current_apr < 0
    # Reinforces learning to exit when funding direction changes
    # Example: +687% entry → -105% current = flip_bonus up to 2.0 * scale
    # Set to 0.0 to disable
    apr_flip_exit_bonus_scale: float = 1.5

    # V7: Opportunity cost threshold - Lower threshold for opportunity cost penalty
    # Original was 100% APR gap, reduced to 50% to be more sensitive
    # Penalizes holding inferior positions more aggressively
    opportunity_cost_threshold: float = 50.0  # V7: Lowered from 100 (was 200 in V6)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'funding_reward_scale': self.funding_reward_scale,
            'price_reward_scale': self.price_reward_scale,
            'liquidation_penalty_scale': self.liquidation_penalty_scale,
            'opportunity_cost_scale': self.opportunity_cost_scale,
            'negative_funding_exit_reward_scale': self.negative_funding_exit_reward_scale,
            'trade_diversity_bonus': self.trade_diversity_bonus,
            'inactivity_penalty_hours': self.inactivity_penalty_hours,
            'inactivity_penalty_scale': self.inactivity_penalty_scale,
            # V7 parameters
            'negative_apr_penalty_scale': self.negative_apr_penalty_scale,
            'apr_flip_exit_bonus_scale': self.apr_flip_exit_bonus_scale,
            'opportunity_cost_threshold': self.opportunity_cost_threshold,
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
            f"neg_fund_exit={self.negative_funding_exit_reward_scale:.2f}, "
            f"neg_apr_pen={self.negative_apr_penalty_scale:.2f}, "
            f"flip_exit={self.apr_flip_exit_bonus_scale:.2f})"
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
