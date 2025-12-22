"""
Unit tests for TradingConfig (V9: single position mode)

Tests configuration validation, serialization, and random sampling.
"""

import pytest
import numpy as np
from models.rl.core.config import TradingConfig


class TestTradingConfig:
    """Test suite for TradingConfig class."""

    def test_default_config(self):
        """Test default configuration values (V9)."""
        config = TradingConfig()

        assert config.max_leverage == 3.0
        assert config.target_utilization == 0.6
        assert config.max_positions == 1  # V9: single position only
        assert config.stop_loss_threshold == -0.02
        assert config.liquidation_buffer == 0.15

    def test_custom_config(self):
        """Test custom configuration (V9: max_positions must be 1)."""
        config = TradingConfig(
            max_leverage=5.0,
            target_utilization=0.7,
            max_positions=1,  # V9: single position only
            stop_loss_threshold=-0.03,
            liquidation_buffer=0.2
        )

        assert config.max_leverage == 5.0
        assert config.target_utilization == 0.7
        assert config.max_positions == 1
        assert config.stop_loss_threshold == -0.03
        assert config.liquidation_buffer == 0.2

    def test_to_array(self):
        """Test conversion to numpy array."""
        config = TradingConfig(
            max_leverage=3.0,
            target_utilization=0.6,
            max_positions=1,  # V9: single position
            stop_loss_threshold=-0.025,
            liquidation_buffer=0.18
        )

        arr = config.to_array()

        assert arr.shape == (5,)
        assert arr.dtype == np.float32
        np.testing.assert_almost_equal(arr[0], 3.0, decimal=5)  # max_leverage
        np.testing.assert_almost_equal(arr[1], 0.6, decimal=5)  # target_utilization
        np.testing.assert_almost_equal(arr[2], 1.0, decimal=5)  # max_positions (V9)
        np.testing.assert_almost_equal(arr[3], -0.025, decimal=5)  # stop_loss_threshold
        np.testing.assert_almost_equal(arr[4], 0.18, decimal=5)  # liquidation_buffer

    def test_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([2.0, 0.55, 1.0, -0.015, 0.12], dtype=np.float32)
        config = TradingConfig.from_array(arr)

        np.testing.assert_almost_equal(config.max_leverage, 2.0, decimal=5)
        np.testing.assert_almost_equal(config.target_utilization, 0.55, decimal=5)
        assert config.max_positions == 1
        np.testing.assert_almost_equal(config.stop_loss_threshold, -0.015, decimal=5)
        np.testing.assert_almost_equal(config.liquidation_buffer, 0.12, decimal=5)

    def test_round_trip_conversion(self):
        """Test array conversion round trip."""
        original = TradingConfig(
            max_leverage=4.5,
            target_utilization=0.65,
            max_positions=1,  # V9: single position
            stop_loss_threshold=-0.04,
            liquidation_buffer=0.25
        )

        arr = original.to_array()
        restored = TradingConfig.from_array(arr)

        np.testing.assert_almost_equal(restored.max_leverage, original.max_leverage, decimal=5)
        np.testing.assert_almost_equal(restored.target_utilization, original.target_utilization, decimal=5)
        assert restored.max_positions == original.max_positions
        np.testing.assert_almost_equal(restored.stop_loss_threshold, original.stop_loss_threshold, decimal=5)
        np.testing.assert_almost_equal(restored.liquidation_buffer, original.liquidation_buffer, decimal=5)

    def test_sample_random(self):
        """Test random config sampling (V9: max_positions always 1)."""
        config = TradingConfig.sample_random(seed=42)

        # Check ranges
        assert 1.0 <= config.max_leverage <= 10.0
        assert 0.3 <= config.target_utilization <= 0.8
        assert config.max_positions == 1  # V9: always 1
        assert -0.05 <= config.stop_loss_threshold <= -0.01
        assert 0.1 <= config.liquidation_buffer <= 0.3

    def test_sample_random_reproducible(self):
        """Test random sampling is reproducible with seed."""
        config1 = TradingConfig.sample_random(seed=123)
        config2 = TradingConfig.sample_random(seed=123)

        assert config1.max_leverage == config2.max_leverage
        assert config1.target_utilization == config2.target_utilization
        assert config1.max_positions == config2.max_positions
        assert config1.stop_loss_threshold == config2.stop_loss_threshold
        assert config1.liquidation_buffer == config2.liquidation_buffer

    def test_sample_random_different(self):
        """Test random sampling produces different configs."""
        config1 = TradingConfig.sample_random(seed=1)
        config2 = TradingConfig.sample_random(seed=2)

        # At least one value should be different (excluding max_positions which is always 1)
        different = (
            config1.max_leverage != config2.max_leverage or
            config1.target_utilization != config2.target_utilization or
            config1.stop_loss_threshold != config2.stop_loss_threshold or
            config1.liquidation_buffer != config2.liquidation_buffer
        )
        assert different

    def test_conservative_preset(self):
        """Test conservative configuration preset (V9)."""
        config = TradingConfig.get_conservative()

        assert config.max_leverage == 1.0
        assert config.target_utilization == 0.4
        assert config.max_positions == 1  # V9: single position
        assert config.stop_loss_threshold == -0.01
        assert config.liquidation_buffer == 0.25

    def test_moderate_preset(self):
        """Test moderate configuration preset (V9)."""
        config = TradingConfig.get_moderate()

        assert config.max_leverage == 3.0
        assert config.target_utilization == 0.6
        assert config.max_positions == 1  # V9: single position
        assert config.stop_loss_threshold == -0.02
        assert config.liquidation_buffer == 0.15

    def test_aggressive_preset(self):
        """Test aggressive configuration preset (V9)."""
        config = TradingConfig.get_aggressive()

        assert config.max_leverage == 5.0
        assert config.target_utilization == 0.75
        assert config.max_positions == 1  # V9: single position
        assert config.stop_loss_threshold == -0.03
        assert config.liquidation_buffer == 0.10

    def test_validate_valid_config(self):
        """Test validation accepts valid config (V9)."""
        config = TradingConfig(
            max_leverage=5.0,
            target_utilization=0.6,
            max_positions=1,  # V9: must be 1
            stop_loss_threshold=-0.02,
            liquidation_buffer=0.15
        )

        assert config.validate() is True

    def test_validate_invalid_leverage_low(self):
        """Test validation rejects leverage < 1."""
        config = TradingConfig(max_leverage=0.5)
        assert config.validate() is False

    def test_validate_invalid_leverage_high(self):
        """Test validation rejects leverage > 10."""
        config = TradingConfig(max_leverage=15.0)
        assert config.validate() is False

    def test_validate_invalid_utilization_low(self):
        """Test validation rejects utilization < 0."""
        config = TradingConfig(target_utilization=-0.1)
        assert config.validate() is False

    def test_validate_invalid_utilization_high(self):
        """Test validation rejects utilization > 1."""
        config = TradingConfig(target_utilization=1.5)
        assert config.validate() is False

    def test_validate_invalid_max_positions(self):
        """Test validation rejects max_positions != 1 (V9)."""
        # max_positions > 1 is invalid in V9
        config = TradingConfig(max_positions=2)
        assert config.validate() is False

        # max_positions < 1 is also invalid
        config2 = TradingConfig(max_positions=0)
        assert config2.validate() is False

    def test_validate_invalid_stop_loss(self):
        """Test validation rejects invalid stop-loss."""
        # Too low
        config1 = TradingConfig(stop_loss_threshold=-0.15)
        assert config1.validate() is False

        # Positive (invalid)
        config2 = TradingConfig(stop_loss_threshold=0.01)
        assert config2.validate() is False

    def test_validate_invalid_liquidation_buffer(self):
        """Test validation rejects invalid liquidation buffer."""
        # Too low
        config1 = TradingConfig(liquidation_buffer=0.01)
        assert config1.validate() is False

        # Too high
        config2 = TradingConfig(liquidation_buffer=0.8)
        assert config2.validate() is False

    def test_repr(self):
        """Test string representation."""
        config = TradingConfig(
            max_leverage=2.5,
            target_utilization=0.55,
            max_positions=1,  # V9
            stop_loss_threshold=-0.025,
            liquidation_buffer=0.18
        )

        repr_str = repr(config)

        assert "2.5x" in repr_str
        assert "55.0%" in repr_str or "55%" in repr_str
        assert "max_pos=1" in repr_str
        assert "-2.50%" in repr_str or "-2.5%" in repr_str
        assert "18.0%" in repr_str or "18%" in repr_str
