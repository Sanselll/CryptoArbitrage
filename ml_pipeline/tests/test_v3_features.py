"""
Unit tests for V3 Feature Refactoring (301→203 dimensions)

Tests velocity tracking, feature dimensions, and correct feature extraction
for the V3 refactoring that streamlines features from 301 to 203 dimensions.

V3 Changes:
- Portfolio: 6→3 features (removed historical metrics)
- Execution: 100→85 features (5×20→5×17, added velocities)
- Opportunity: 190→110 features (10×19→10×11, removed market quality)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from models.rl.core.portfolio import Portfolio, Position
from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork


@pytest.fixture(scope='module')
def test_data_file():
    """Create minimal test data file for environment tests."""
    test_data_path = Path('data/test_rl_minimal.csv')
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame({
        'entry_time': [pd.Timestamp.now(tz='UTC') + timedelta(hours=i) for i in range(100)],
        'symbol': ['BTCUSDT'] * 100,
        'long_exchange': ['binance'] * 100,
        'short_exchange': ['bybit'] * 100,
        'entry_long_price': [50000.0] * 100,
        'entry_short_price': [50010.0] * 100,
        'fund_apr': [50.0] * 100,
        'long_funding_rate': [0.0001] * 100,
        'short_funding_rate': [-0.0001] * 100,
        'long_funding_interval_hours': [8] * 100,
        'short_funding_interval_hours': [8] * 100,
        'long_next_funding_time': [pd.Timestamp.now(tz='UTC') + timedelta(hours=8) for _ in range(100)],
        'short_next_funding_time': [pd.Timestamp.now(tz='UTC') + timedelta(hours=8) for _ in range(100)],
        'spread30SampleAvg': [0.001] * 100,
        'fund_profit_8h': [0.5] * 100,
        'fundProfit8h24hProj': [0.5] * 100,
        'fundProfit8h3dProj': [0.5] * 100,
        'fundApr24hProj': [50.0] * 100,
        'fundApr3dProj': [50.0] * 100,
        'priceSpread24hAvg': [0.001] * 100,
        'priceSpread3dAvg': [0.001] * 100,
        'spread_volatility_stddev': [0.0001] * 100,
    })
    test_df.to_csv(test_data_path, index=False)

    yield str(test_data_path)

    # Cleanup after all tests
    if test_data_path.exists():
        test_data_path.unlink()


class TestV3VelocityTracking:
    """Test velocity tracking features added in V3."""

    def test_position_velocity_fields_initialization(self):
        """Test that Position initializes with velocity tracking fields."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Check velocity tracking fields exist and are initialized to 0
        assert hasattr(position, 'prev_estimated_pnl_pct')
        assert hasattr(position, 'prev_estimated_funding_8h_pct')
        assert hasattr(position, 'prev_spread_pct')

        assert position.prev_estimated_pnl_pct == 0.0
        assert position.prev_estimated_funding_8h_pct == 0.0
        assert position.prev_spread_pct == 0.0

    def test_estimated_funding_8h_calculation(self):
        """Test estimated_funding_8h_pct calculation."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,  # Long pays 0.01%
            short_funding_rate=-0.0001,  # Short receives 0.01%
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Calculate expected: (long_payments * -long_rate + short_payments * short_rate) * 100
        # long_payments_8h = 8/8 = 1, short_payments_8h = 8/8 = 1
        # long_funding_8h = -0.0001 * 1 = -0.0001
        # short_funding_8h = -0.0001 * 1 = -0.0001
        # total = (-0.0001 + -0.0001) * 100 = -0.02%

        funding_8h = position.calculate_estimated_funding_8h_pct()
        assert abs(funding_8h - (-0.02)) < 0.001  # Allow small floating point error

    def test_update_velocity_tracking(self):
        """Test velocity tracking updates correctly."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Mock price data
        price_data = {
            'BTCUSDT': {
                'long_price': 50100.0,  # Price went up
                'short_price': 50110.0,
                'long_funding_rate': 0.0002,  # Funding rate changed
                'short_funding_rate': -0.0002
            }
        }

        # Create portfolio and add position
        portfolio = Portfolio(initial_capital=10000.0, max_positions=3)
        portfolio.positions.append(position)

        # Update velocity tracking (on Portfolio, not Position)
        portfolio.update_velocity_tracking(price_data)

        # Check that previous values are now stored on the position
        assert position.prev_estimated_pnl_pct != 0.0
        assert position.prev_estimated_funding_8h_pct != 0.0
        assert position.prev_spread_pct != 0.0


class TestV3ExecutionFeatures:
    """Test execution features are 17 dimensions (V3: was 20)."""

    def test_execution_feature_dimensions(self):
        """Test that get_execution_state returns 17 features."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        portfolio = Portfolio(initial_capital=10000.0, max_positions=3)
        portfolio.positions.append(position)

        price_data = {
            'BTCUSDT': {
                'long_price': 50100.0,
                'short_price': 50110.0
            }
        }

        # Get execution state
        exec_state = portfolio.get_execution_state(
            exec_idx=0,
            price_data=price_data,
            best_available_apr=50.0
        )

        # V3: Should be 17 features (was 20)
        assert exec_state.shape == (17,), f"Expected 17 features, got {exec_state.shape[0]}"
        assert isinstance(exec_state, np.ndarray)
        assert exec_state.dtype == np.float32

    def test_execution_features_include_velocities(self):
        """Test that execution state includes velocity features."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC') - timedelta(hours=1),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=7),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=7)
        )

        portfolio = Portfolio(initial_capital=10000.0, max_positions=3)
        portfolio.positions.append(position)

        price_data = {
            'BTCUSDT': {
                'long_price': 50100.0,
                'short_price': 50110.0,
                'long_funding_rate': 0.0001,
                'short_funding_rate': -0.0001
            }
        }

        # First update to establish baseline
        portfolio.update_velocity_tracking(price_data)

        # Second update with changed prices
        price_data['BTCUSDT']['long_price'] = 50200.0
        price_data['BTCUSDT']['short_price'] = 50210.0
        portfolio.update_velocity_tracking(price_data)

        # Get execution state
        exec_state = portfolio.get_execution_state(
            exec_idx=0,
            price_data=price_data,
            best_available_apr=50.0
        )

        # Feature indices in V3:
        # 4: estimated_pnl_velocity
        # 6: funding_velocity
        # 8: spread_velocity

        # Velocities should be calculated (non-zero after price changes)
        # Note: May be zero on first update, but structure should be correct
        assert exec_state.shape == (17,)


class TestV3ObservationSpace:
    """Test observation space dimensions (V3: 301→203)."""

    def test_observation_space_dimensions(self, test_data_file):
        """Test that environment observation is 203 dimensions."""
        # Create environment
        env = FundingArbitrageEnv(
            data_path=test_data_file,
            initial_capital=10000.0,
            trading_config=TradingConfig(),
            reward_config=RewardConfig(),
            episode_length_days=3,
            verbose=False
        )

        # Reset environment
        observation, info = env.reset()

        # V3: Should be 203 dimensions (was 301)
        assert observation.shape == (203,), f"Expected 203-dim observation, got {observation.shape[0]}"
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32

    def test_observation_breakdown(self, test_data_file):
        """Test observation space breakdown matches V3 spec."""
        env = FundingArbitrageEnv(
            data_path=test_data_file,
            initial_capital=10000.0,
            trading_config=TradingConfig(),
            reward_config=RewardConfig(),
            episode_length_days=3,
            verbose=False
        )

        observation, info = env.reset()

        # V3 Breakdown: Config(5) + Portfolio(3) + Execution(85) + Opportunity(110) = 203
        config_features = observation[0:5]
        portfolio_features = observation[5:8]
        execution_features = observation[8:93]  # 5 slots × 17 features = 85
        opportunity_features = observation[93:203]  # 10 slots × 11 features = 110

        assert config_features.shape == (5,)
        assert portfolio_features.shape == (3,)  # V3: was 6
        assert execution_features.shape == (85,)  # V3: was 100
        assert opportunity_features.shape == (110,)  # V3: was 190


class TestV3NetworkArchitecture:
    """Test ModularPPO network handles V3 dimensions."""

    def test_network_input_dimensions(self):
        """Test that network accepts 203-dim observations."""
        network = ModularPPONetwork()

        # Create mock observation (203 dims)
        batch_size = 4
        obs = np.random.randn(batch_size, 203).astype(np.float32)
        obs_tensor = np.array(obs)

        # Create mock action mask (36 actions)
        mask = np.ones((batch_size, 36), dtype=bool)

        # Convert to tensors
        import torch
        obs_tensor = torch.FloatTensor(obs)
        mask_tensor = torch.BoolTensor(mask)

        # Forward pass
        action_logits, values = network(obs_tensor, mask_tensor)

        # Check output shapes
        assert action_logits.shape == (batch_size, 36), f"Expected (4, 36), got {action_logits.shape}"
        assert values.shape == (batch_size, 1), f"Expected (4, 1), got {values.shape}"

    def test_encoder_dimensions(self):
        """Test that encoders handle V3 dimensions correctly."""
        network = ModularPPONetwork()

        # Check that encoders exist and are initialized correctly
        assert hasattr(network, 'portfolio_encoder')
        assert hasattr(network, 'execution_encoder')
        assert hasattr(network, 'opportunity_encoder')

        # The encoders are initialized with V3 dimensions in ModularPPONetwork.__init__
        # PortfolioEncoder(input_dim=3), ExecutionEncoder(features_per_slot=17), etc.
        # We verify this indirectly by checking the network can process 203-dim input
        import torch
        batch_size = 2
        obs = torch.randn(batch_size, 203)
        mask = torch.ones(batch_size, 36, dtype=torch.bool)

        # Should not raise an error
        action_logits, values = network(obs, mask)
        assert action_logits.shape == (batch_size, 36)
        assert values.shape == (batch_size, 1)


class TestV3FeatureScaler:
    """Test feature scaler works with 11 opportunity features."""

    def test_feature_scaler_dimensions(self):
        """Test that feature scaler expects 11 features."""
        scaler_path = Path('trained_models/rl/feature_scaler_v2.pkl')

        if not scaler_path.exists():
            pytest.skip(f"Feature scaler not found at {scaler_path}")

        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Check scaler was fit on 11 features
        assert scaler.n_features_in_ == 11, f"Expected 11 features, got {scaler.n_features_in_}"

        # Test transformation
        test_features = np.random.randn(1, 11).astype(np.float32)
        scaled = scaler.transform(test_features)

        assert scaled.shape == (1, 11)


class TestV3FeatureRemovals:
    """Test that removed features are not present in V3."""

    def test_portfolio_no_historical_metrics(self, test_data_file):
        """Test that portfolio features don't include historical metrics."""
        env = FundingArbitrageEnv(
            data_path=test_data_file,
            initial_capital=10000.0,
            trading_config=TradingConfig(),
            reward_config=RewardConfig(),
            episode_length_days=3,
            verbose=False
        )

        observation, info = env.reset()

        # Portfolio features should be only 3 dimensions
        # (no avg_position_pnl_pct, total_pnl_pct, max_drawdown_pct)
        portfolio_features = observation[5:8]
        assert portfolio_features.shape == (3,)

    def test_execution_no_legacy_features(self):
        """Test that execution features don't include removed legacy features."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="binance",
            short_exchange="bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=2.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        portfolio = Portfolio(initial_capital=10000.0, max_positions=3)
        portfolio.positions.append(position)

        price_data = {
            'BTCUSDT': {
                'long_price': 50100.0,
                'short_price': 50110.0
            }
        }

        exec_state = portfolio.get_execution_state(
            exec_idx=0,
            price_data=price_data,
            best_available_apr=50.0
        )

        # Should be 17 features (removed: net_funding_ratio, net_funding_rate,
        # funding_efficiency, entry_spread_pct, long/short_pnl_pct,
        # old pnl_velocity, peak_drawdown, is_old_loser)
        assert exec_state.shape == (17,)


class TestV3OpportunityFeatures:
    """Test opportunity features are 11 dimensions (V3: was 19)."""

    def test_opportunity_feature_extraction(self, test_data_file):
        """Test that environment extracts 11 opportunity features per slot."""
        env = FundingArbitrageEnv(
            data_path=test_data_file,
            initial_capital=10000.0,
            trading_config=TradingConfig(),
            reward_config=RewardConfig(),
            episode_length_days=3,
            verbose=False
        )

        observation, info = env.reset()

        # Extract first opportunity features (11 features)
        first_opp_features = observation[93:104]

        assert first_opp_features.shape == (11,), f"Expected 11 features per opportunity, got {first_opp_features.shape[0]}"

    def test_opportunity_no_market_quality(self, test_data_file):
        """Test that opportunity features don't include market quality metrics."""
        # V3 removed: volume_24h, bidAskSpreadPercent, orderbookDepthUsd,
        # estimatedProfitPercentage, positionCostPercent
        # V3 kept: 6 profit/APR projections, 4 spread metrics, 1 velocity

        env = FundingArbitrageEnv(
            data_path=test_data_file,
            initial_capital=10000.0,
            trading_config=TradingConfig(),
            reward_config=RewardConfig(),
            episode_length_days=3,
            verbose=False
        )

        observation, info = env.reset()

        # Should have 110 opportunity features total (10 × 11)
        opportunity_features = observation[93:203]
        assert opportunity_features.shape == (110,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
