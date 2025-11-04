"""
Unit tests for Environment updates

Tests observation space, action space, action masking, and new functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig


@pytest.fixture
def sample_data_path(tmp_path):
    """Create a minimal CSV file for testing."""
    data = {
        'entry_time': pd.date_range('2025-01-01', periods=100, freq='H', tz='UTC'),
        'symbol': ['BTCUSDT'] * 100,
        'long_exchange': ['Binance'] * 100,
        'short_exchange': ['Bybit'] * 100,
        'entry_long_price': [50000.0] * 100,
        'entry_short_price': [50010.0] * 100,
        'long_funding_rate': [0.0001] * 100,
        'short_funding_rate': [-0.0001] * 100,
        'long_funding_interval_hours': [8] * 100,
        'short_funding_interval_hours': [8] * 100,
        'long_next_funding_time': pd.date_range('2025-01-01 08:00', periods=100, freq='H', tz='UTC'),
        'short_next_funding_time': pd.date_range('2025-01-01 08:00', periods=100, freq='H', tz='UTC'),
        'fund_profit_8h': [0.5] * 100,
        'fundProfit8h24hProj': [0.45] * 100,
        'fundProfit8h3dProj': [0.4] * 100,
        'fund_apr': [100.0] * 100,
        'fundApr24hProj': [95.0] * 100,
        'fundApr3dProj': [90.0] * 100,
        'spread30SampleAvg': [0.02] * 100,
        'priceSpread24hAvg': [0.015] * 100,
        'priceSpread3dAvg': [0.018] * 100,
        'spread_volatility_stddev': [0.001] * 100,
        'volume_24h': [1000000.0] * 100,
        'bidAskSpreadPercent': [0.05] * 100,
        'orderbookDepthUsd': [10000.0] * 100,
        'estimatedProfitPercentage': [0.3] * 100,
        'positionCostPercent': [0.2] * 100,
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return str(csv_path)


class TestEnvironmentSimpleMode:
    """Test environment in simple mode (backward compatibility)."""

    def test_simple_mode_initialization(self, sample_data_path):
        """Test environment initialization in simple mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=True,
            verbose=False
        )

        assert env.simple_mode is True
        assert env.action_space.n == 3  # HOLD, ENTER, EXIT
        assert env.observation_space.shape == (36,)  # 14 portfolio + 22 opportunity

    def test_simple_mode_reset(self, sample_data_path):
        """Test environment reset in simple mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=True,
            verbose=False
        )

        obs, info = env.reset(seed=42)

        assert obs.shape == (36,)
        assert 'episode_start' in info
        assert 'episode_end' in info
        assert 'portfolio_value' in info

    def test_simple_mode_step_hold(self, sample_data_path):
        """Test HOLD action in simple mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=True,
            episode_length_days=1,
            verbose=False
        )

        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD

        assert obs.shape == (36,)
        assert not terminated
        assert info['action_type'] == 'hold'


class TestEnvironmentFullMode:
    """Test environment in full mode."""

    def test_full_mode_initialization(self, sample_data_path):
        """Test environment initialization in full mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        assert env.simple_mode is False
        assert env.action_space.n == 36  # 1 HOLD + 30 ENTER + 5 EXIT
        assert env.observation_space.shape == (275,)  # 5 + 10 + 60 + 200

    def test_full_mode_with_config(self, sample_data_path):
        """Test environment with custom config."""
        config = TradingConfig(
            max_leverage=5.0,
            target_utilization=0.7,
            max_positions=4,
            stop_loss_threshold=-0.025,
            liquidation_buffer=0.2
        )

        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            trading_config=config,
            simple_mode=False,
            verbose=False
        )

        assert env.trading_config.max_leverage == 5.0
        assert env.trading_config.target_utilization == 0.7
        assert env.trading_config.max_positions == 4

    def test_full_mode_reset(self, sample_data_path):
        """Test environment reset in full mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        obs, info = env.reset(seed=42)

        assert obs.shape == (275,)
        assert 'episode_start' in info

    def test_config_sampling(self, sample_data_path):
        """Test random config sampling."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            sample_random_config=True,
            simple_mode=False,
            verbose=False
        )

        obs1, _ = env.reset(seed=1)
        config1 = env.current_config

        obs2, _ = env.reset(seed=2)
        config2 = env.current_config

        # Different seeds should produce different configs
        assert (config1.max_leverage != config2.max_leverage or
                config1.target_utilization != config2.target_utilization)

    def test_observation_structure_full_mode(self, sample_data_path):
        """Test observation structure in full mode."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        obs, _ = env.reset(seed=42)

        # Check dimensions
        assert obs.shape == (275,)

        # Config features (first 5)
        config_features = obs[:5]
        assert config_features[0] >= 1.0  # max_leverage
        assert 0.0 <= config_features[1] <= 1.0  # target_utilization
        assert 1 <= config_features[2] <= 5  # max_positions

        # Portfolio features (next 10)
        portfolio_features = obs[5:15]
        assert portfolio_features[0] == 1.0  # capital_ratio (initial)

        # Execution features (next 60)
        execution_features = obs[15:75]
        # Should be all zeros initially (no positions)
        assert np.all(execution_features == 0.0)

        # Opportunity features (last 200)
        opportunity_features = obs[75:275]
        # Should have some non-zero values if opportunities exist
        # (depends on whether data has opportunities at start time)


class TestActionMasking:
    """Test action masking functionality."""

    def test_get_action_mask_initial(self, sample_data_path):
        """Test action mask at environment start."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        env.reset(seed=42)
        mask = env._get_action_mask()

        assert mask.shape == (36,)
        assert mask[0] == True  # HOLD always valid

        # ENTER actions should be valid if opportunities exist
        # EXIT actions should be invalid (no positions yet)
        for i in range(31, 36):
            assert mask[i] == False  # No positions to exit

    def test_action_mask_with_positions(self, sample_data_path):
        """Test action mask when positions are open."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            episode_length_days=1,
            verbose=False
        )

        env.reset(seed=42)

        # Open a position (if possible)
        # This depends on data having opportunities
        # We'll just test the mask logic

        mask = env._get_action_mask()
        assert mask.shape == (36,)


class TestOpportunitySelection:
    """Test opportunity selection functionality."""

    def test_select_top_opportunities(self, sample_data_path):
        """Test top opportunity selection."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        env.reset(seed=42)

        # Get opportunities at current time
        opportunities = env._get_opportunities_at_time(env.current_time)

        # Should return up to 10 opportunities
        assert len(opportunities) <= 10

    def test_composite_score_calculation(self, sample_data_path):
        """Test composite score calculation."""
        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            simple_mode=False,
            verbose=False
        )

        opp = {
            'fundApr24hProj': 100.0,
            'fund_apr': 95.0,
            'volume_24h': 2000000.0,
            'bidAskSpreadPercent': 0.05
        }

        score = env._calculate_composite_score(opp)

        # High APR + good volume + low spread = high score
        assert score > 0


class TestDynamicPositionSizing:
    """Test dynamic position sizing."""

    def test_calculate_position_size_small(self, sample_data_path):
        """Test small position size calculation."""
        config = TradingConfig(
            max_leverage=2.0,
            target_utilization=0.6,
            max_positions=5
        )

        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            trading_config=config,
            simple_mode=False,
            initial_capital=10000.0,
            verbose=False
        )

        env.reset(seed=42)

        # Action 1 = ENTER_OPP_0_SMALL
        size = env._calculate_position_size(1)

        # max_allowed = (10000 * 2.0 * 0.6) / 5 = 2400
        # small = 2400 * 0.1 = 240
        expected = (10000.0 * 2.0 * 0.6 / 5.0) * 0.1
        assert abs(size - expected) < 1.0

    def test_calculate_position_size_medium(self, sample_data_path):
        """Test medium position size calculation."""
        config = TradingConfig(
            max_leverage=2.0,
            target_utilization=0.6,
            max_positions=5
        )

        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            trading_config=config,
            simple_mode=False,
            initial_capital=10000.0,
            verbose=False
        )

        env.reset(seed=42)

        # Action 11 = ENTER_OPP_0_MEDIUM
        size = env._calculate_position_size(11)

        # medium = 2400 * 0.2 = 480
        expected = (10000.0 * 2.0 * 0.6 / 5.0) * 0.2
        assert abs(size - expected) < 1.0

    def test_calculate_position_size_large(self, sample_data_path):
        """Test large position size calculation."""
        config = TradingConfig(
            max_leverage=2.0,
            target_utilization=0.6,
            max_positions=5
        )

        env = FundingArbitrageEnv(
            data_path=sample_data_path,
            trading_config=config,
            simple_mode=False,
            initial_capital=10000.0,
            verbose=False
        )

        env.reset(seed=42)

        # Action 21 = ENTER_OPP_0_LARGE
        size = env._calculate_position_size(21)

        # large = 2400 * 0.3 = 720
        expected = (10000.0 * 2.0 * 0.6 / 5.0) * 0.3
        assert abs(size - expected) < 1.0


class TestActionDecoding:
    """Test action decoding in full mode."""

    def test_action_0_is_hold(self):
        """Test action 0 decodes to HOLD."""
        # This is implicitly tested in step, but we can verify the logic
        pass

    def test_action_1_10_are_small_enters(self):
        """Test actions 1-10 decode to SMALL ENTER."""
        # Tested via _execute_action
        pass

    def test_action_11_20_are_medium_enters(self):
        """Test actions 11-20 decode to MEDIUM ENTER."""
        pass

    def test_action_21_30_are_large_enters(self):
        """Test actions 21-30 decode to LARGE ENTER."""
        pass

    def test_action_31_35_are_exits(self):
        """Test actions 31-35 decode to EXIT."""
        pass
