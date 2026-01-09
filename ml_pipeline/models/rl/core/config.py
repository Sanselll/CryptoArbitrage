"""
Trading Configuration for RL Agent

User-configurable parameters that define trading constraints and risk management.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingConfig:
    """
    Configuration parameters for the trading agent.

    These parameters can be set by users or sampled during training
    to teach the agent to adapt to different risk profiles.
    """

    # Position sizing and leverage (V9: single position only)
    max_leverage: float = 3.0                    # Maximum leverage allowed (1-10x)
    target_utilization: float = 0.6              # Target capital utilization (0-1, e.g., 0.6 = 60%)
    max_positions: int = 1                        # Maximum concurrent positions (V9: single position only)

    # Risk management
    stop_loss_threshold: float = -0.10           # P&L% threshold for automatic exit (e.g., -0.10 = -10%)
    liquidation_buffer: float = 0.15             # Minimum safe distance to liquidation (e.g., 0.15 = 15%)

    def to_array(self) -> np.ndarray:
        """
        Convert config to numpy array for neural network input.

        Returns:
            5-dimensional array of configuration parameters
        """
        return np.array([
            self.max_leverage,
            self.target_utilization,
            float(self.max_positions),
            self.stop_loss_threshold,
            self.liquidation_buffer
        ], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> 'TradingConfig':
        """
        Create TradingConfig from numpy array.

        Args:
            arr: 5-dimensional array

        Returns:
            TradingConfig instance
        """
        return TradingConfig(
            max_leverage=float(arr[0]),
            target_utilization=float(arr[1]),
            max_positions=int(arr[2]),
            stop_loss_threshold=float(arr[3]),
            liquidation_buffer=float(arr[4])
        )

    @staticmethod
    def sample_random(seed: Optional[int] = None) -> 'TradingConfig':
        """
        Sample random configuration for training diversity.

        This allows the agent to learn to trade under various constraints
        and generalize to any user-specified configuration.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Randomly sampled TradingConfig
        """
        if seed is not None:
            np.random.seed(seed)

        return TradingConfig(
            max_leverage=np.random.uniform(1.0, 10.0),
            target_utilization=np.random.uniform(0.3, 0.8),
            max_positions=1,  # V9: single position only
            stop_loss_threshold=np.random.uniform(-0.05, -0.01),
            liquidation_buffer=np.random.uniform(0.1, 0.3)
        )

    @staticmethod
    def sample_moderate(seed: Optional[int] = None) -> 'TradingConfig':
        """
        Sample moderate configuration for Phase 2 curriculum learning.

        Samples from a narrower range than sample_random() to focus
        on moderate risk profiles during generalization phase.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Moderately sampled TradingConfig
        """
        if seed is not None:
            np.random.seed(seed)

        return TradingConfig(
            max_leverage=np.random.uniform(1.0, 5.0),      # 1-5x (vs 1-10x in random)
            target_utilization=np.random.uniform(0.4, 0.7), # 40-70% (vs 30-80%)
            max_positions=1,                                # V9: single position only
            stop_loss_threshold=np.random.uniform(-0.03, -0.015),  # -3% to -1.5%
            liquidation_buffer=np.random.uniform(0.12, 0.20)       # 12-20%
        )

    @staticmethod
    def get_conservative() -> 'TradingConfig':
        """Get conservative trading configuration (low risk)."""
        return TradingConfig(
            max_leverage=1.0,
            target_utilization=0.4,
            max_positions=1,              # V9: single position only
            stop_loss_threshold=-0.01,  # -1% stop-loss
            liquidation_buffer=0.25      # 25% buffer
        )

    @staticmethod
    def get_moderate() -> 'TradingConfig':
        """Get moderate trading configuration (balanced)."""
        return TradingConfig(
            max_leverage=3.0,
            target_utilization=0.6,
            max_positions=1,              # V9: single position only
            stop_loss_threshold=-0.02,  # -2% stop-loss
            liquidation_buffer=0.15      # 15% buffer
        )

    @staticmethod
    def get_aggressive() -> 'TradingConfig':
        """Get aggressive trading configuration (high risk, V9)."""
        return TradingConfig(
            max_leverage=5.0,
            target_utilization=0.75,
            max_positions=1,              # V9: single position only
            stop_loss_threshold=-0.03,  # -3% stop-loss
            liquidation_buffer=0.10      # 10% buffer
        )

    def validate(self) -> bool:
        """
        Validate configuration parameters are within acceptable ranges (V9).

        Returns:
            True if valid, False otherwise
        """
        if not (1.0 <= self.max_leverage <= 10.0):
            return False
        if not (0.0 <= self.target_utilization <= 1.0):
            return False
        if self.max_positions != 1:  # V9: single position only
            return False
        if not (-0.10 <= self.stop_loss_threshold <= 0.0):
            return False
        if not (0.05 <= self.liquidation_buffer <= 0.50):
            return False
        return True

    def __repr__(self) -> str:
        """String representation for logging."""
        return (f"TradingConfig(leverage={self.max_leverage:.1f}x, "
                f"util={self.target_utilization:.1%}, "
                f"max_pos={self.max_positions}, "
                f"stop_loss={self.stop_loss_threshold:.2%}, "
                f"liq_buffer={self.liquidation_buffer:.1%})")
