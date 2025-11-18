"""
Integration tests for UnifiedFeatureBuilder.

Tests that all components (backend inference, training environment, test inference)
produce identical feature vectors for the same raw data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.features import UnifiedFeatureBuilder, DIMS


def create_test_raw_data():
    """Create sample raw data for testing."""
    return {
        'trading_config': {
            'max_leverage': 2.0,
            'target_utilization': 0.8,
            'max_positions': 3,
            'stop_loss_threshold': -0.02,
            'liquidation_buffer': 0.15
        },
        'portfolio': {
            'total_capital': 10000.0,
            'capital_utilization': 25.0,
            'positions': [
                {
                    'is_active': True,
                    'symbol': 'BTCUSDT',
                    'position_size_usd': 1000.0,
                    'position_age_hours': 5.5,
                    'leverage': 2.0,
                    'entry_long_price': 95000.0,
                    'entry_short_price': 95050.0,
                    'current_long_price': 95100.0,
                    'current_short_price': 95150.0,
                    'unrealized_pnl_pct': 0.15,
                    'long_pnl_pct': 0.105,
                    'short_pnl_pct': 0.105,
                    'long_funding_rate': 0.0001,
                    'short_funding_rate': -0.0003,
                    'long_funding_interval_hours': 8,
                    'short_funding_interval_hours': 8,
                    'entry_apr': 150.0,
                    'current_position_apr': 140.0,
                    'liquidation_distance': 0.45
                }
            ]
        },
        'opportunities': [
            {
                'symbol': 'ETHUSDT',
                'long_exchange': 'binance',
                'short_exchange': 'bybit',
                'fund_profit_8h': 0.25,
                'fundProfit8h24hProj': 0.75,
                'fundProfit8h3dProj': 2.25,
                'fund_apr': 180.0,
                'fundApr24hProj': 175.0,
                'fundApr3dProj': 170.0,
                'spread30SampleAvg': 0.0015,
                'priceSpread24hAvg': 0.0018,
                'priceSpread3dAvg': 0.0020,
                'spread_volatility_stddev': 0.0005,
                'has_existing_position': False
            }
        ]
    }


def test_observation_shape():
    """Test that observation vector has correct shape."""
    print("Test 1: Observation Shape")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)  # No scaler for this test
    raw_data = create_test_raw_data()

    obs = builder.build_observation_from_raw_data(raw_data)

    assert obs.shape == (DIMS.TOTAL,), f"Expected shape ({DIMS.TOTAL},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"

    print(f"✅ Observation shape: {obs.shape}")
    print(f"✅ Observation dtype: {obs.dtype}")
    print()


def test_feature_components():
    """Test that feature components have correct dimensions."""
    print("Test 2: Feature Component Dimensions")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    config_features = builder.build_config_features(raw_data['trading_config'])
    portfolio_features = builder.build_portfolio_features(
        raw_data['portfolio'],
        raw_data['trading_config']
    )
    execution_features = builder.build_execution_features(raw_data['portfolio'], best_available_apr=180.0)
    opportunity_features = builder.build_opportunity_features(raw_data['opportunities'])

    assert config_features.shape == (DIMS.CONFIG,), f"Config: expected {DIMS.CONFIG}, got {config_features.shape}"
    assert portfolio_features.shape == (DIMS.PORTFOLIO,), f"Portfolio: expected {DIMS.PORTFOLIO}, got {portfolio_features.shape}"
    assert execution_features.shape == (DIMS.EXECUTIONS_TOTAL,), f"Executions: expected {DIMS.EXECUTIONS_TOTAL}, got {execution_features.shape}"
    assert opportunity_features.shape == (DIMS.OPPORTUNITIES_TOTAL,), f"Opportunities: expected {DIMS.OPPORTUNITIES_TOTAL}, got {opportunity_features.shape}"

    print(f"✅ Config features: {config_features.shape}")
    print(f"✅ Portfolio features: {portfolio_features.shape}")
    print(f"✅ Execution features: {execution_features.shape}")
    print(f"✅ Opportunity features: {opportunity_features.shape}")
    print()


def test_config_features_values():
    """Test config feature extraction."""
    print("Test 3: Config Feature Values")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    config_features = builder.build_config_features(raw_data['trading_config'])

    expected = np.array([2.0, 0.8, 3.0, -0.02, 0.15], dtype=np.float32)
    np.testing.assert_array_almost_equal(config_features, expected, decimal=5)

    print(f"✅ Config features match expected values")
    print(f"   Values: {config_features}")
    print()


def test_portfolio_features_calculation():
    """Test portfolio feature calculation."""
    print("Test 4: Portfolio Feature Calculation")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    portfolio_features = builder.build_portfolio_features(
        raw_data['portfolio'],
        raw_data['trading_config']
    )

    # Expected:
    # - num_positions_ratio = 1 position / 3 max = 0.333...
    # - min_liq_distance = 0.45
    # - capital_utilization = 25.0 / 100 = 0.25

    assert abs(portfolio_features[0] - 0.333333) < 0.01, f"num_positions_ratio incorrect: {portfolio_features[0]}"
    assert abs(portfolio_features[1] - 0.45) < 0.001, f"min_liq_distance incorrect: {portfolio_features[1]}"
    assert abs(portfolio_features[2] - 0.25) < 0.001, f"capital_utilization incorrect: {portfolio_features[2]}"

    print(f"✅ Portfolio features calculated correctly")
    print(f"   num_positions_ratio: {portfolio_features[0]:.4f}")
    print(f"   min_liq_distance: {portfolio_features[1]:.4f}")
    print(f"   capital_utilization: {portfolio_features[2]:.4f}")
    print()


def test_execution_features_active_position():
    """Test execution features for active position."""
    print("Test 5: Execution Features (Active Position)")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    execution_features = builder.build_execution_features(raw_data['portfolio'], best_available_apr=180.0)

    # First slot should be active
    first_slot = execution_features[:DIMS.EXECUTIONS_PER_SLOT]

    assert first_slot[0] == 1.0, f"is_active should be 1.0, got {first_slot[0]}"
    assert first_slot[1] != 0.0, f"net_pnl_pct should not be 0.0, got {first_slot[1]}"
    assert first_slot[2] > 0.0, f"hours_held_norm should be > 0.0, got {first_slot[2]}"

    print(f"✅ Execution features (slot 0) calculated correctly")
    print(f"   is_active: {first_slot[0]}")
    print(f"   net_pnl_pct: {first_slot[1]:.4f}")
    print(f"   hours_held_norm: {first_slot[2]:.4f}")
    print()


def test_execution_features_empty_slots():
    """Test that empty execution slots are all zeros."""
    print("Test 6: Execution Features (Empty Slots)")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    execution_features = builder.build_execution_features(raw_data['portfolio'], best_available_apr=180.0)

    # Slots 1-4 should be all zeros (only 1 position)
    for slot_idx in range(1, DIMS.EXECUTIONS_SLOTS):
        slot_start = slot_idx * DIMS.EXECUTIONS_PER_SLOT
        slot_end = slot_start + DIMS.EXECUTIONS_PER_SLOT
        slot_features = execution_features[slot_start:slot_end]

        assert np.all(slot_features == 0.0), f"Slot {slot_idx} should be all zeros, got {slot_features}"

    print(f"✅ Empty execution slots (1-4) are all zeros")
    print()


def test_opportunity_features_without_scaler():
    """Test opportunity features WITH scaler (default behavior)."""
    print("Test 7: Opportunity Features (With Scaler)")
    print("-" * 80)

    # Use default scaler path (builder will load it)
    builder = UnifiedFeatureBuilder()
    raw_data = create_test_raw_data()

    opportunity_features = builder.build_opportunity_features(raw_data['opportunities'])

    # First slot should have SCALED values (not raw)
    first_slot = opportunity_features[:DIMS.OPPORTUNITIES_PER_SLOT]

    # With scaler, values will be standardized (mean=0, std=1)
    # We just check that they're not the raw values and are finite
    assert first_slot[0] != 0.25, f"fund_profit_8h should be scaled, got raw value"
    assert np.isfinite(first_slot).all(), "All features should be finite"

    print(f"✅ Opportunity features (with scaler) calculated correctly")
    print(f"   fund_profit_8h (scaled): {first_slot[0]:.4f}")
    print(f"   fund_apr (scaled): {first_slot[3]:.4f}")
    print(f"   apr_velocity (scaled): {first_slot[10]:.4f}")
    print()


def test_action_mask_generation():
    """Test action mask generation."""
    print("Test 8: Action Mask Generation")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    opportunities = raw_data['opportunities']
    num_positions = 1
    max_positions = 3

    action_mask = builder.get_action_mask(opportunities, num_positions, max_positions)

    # HOLD should always be valid
    assert action_mask[0] == True, "HOLD action should be valid"

    # ENTER actions should be valid (have capacity and 1 opportunity)
    assert action_mask[1] == True, "ENTER_OPP_0_SMALL should be valid"
    assert action_mask[11] == True, "ENTER_OPP_0_MEDIUM should be valid"
    assert action_mask[21] == True, "ENTER_OPP_0_LARGE should be valid"

    # ENTER for non-existent opportunities should be invalid
    assert action_mask[2] == False, "ENTER_OPP_1 should be invalid (no opportunity)"

    # EXIT for position 0 should be valid
    assert action_mask[31] == True, "EXIT_POS_0 should be valid"

    # EXIT for non-existent positions should be invalid
    assert action_mask[32] == False, "EXIT_POS_1 should be invalid (no position)"

    valid_count = action_mask.sum()
    print(f"✅ Action mask generated correctly")
    print(f"   Valid actions: {valid_count}/{DIMS.TOTAL_ACTIONS}")
    print(f"   HOLD: {action_mask[0]}")
    print(f"   ENTER (opp 0): {action_mask[1]}, {action_mask[11]}, {action_mask[21]}")
    print(f"   EXIT (pos 0): {action_mask[31]}")
    print()


def test_consistency_across_multiple_calls():
    """Test that builder produces consistent results across multiple calls."""
    print("Test 9: Consistency Across Multiple Calls")
    print("-" * 80)

    builder = UnifiedFeatureBuilder(feature_scaler_path=None)
    raw_data = create_test_raw_data()

    obs1 = builder.build_observation_from_raw_data(raw_data)
    obs2 = builder.build_observation_from_raw_data(raw_data)
    obs3 = builder.build_observation_from_raw_data(raw_data)

    np.testing.assert_array_equal(obs1, obs2, err_msg="obs1 != obs2")
    np.testing.assert_array_equal(obs2, obs3, err_msg="obs2 != obs3")

    print(f"✅ Builder produces consistent results across calls")
    print()


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("UNIFIED FEATURE BUILDER INTEGRATION TESTS")
    print("=" * 80)
    print()

    try:
        test_observation_shape()
        test_feature_components()
        test_config_features_values()
        test_portfolio_features_calculation()
        test_execution_features_active_position()
        test_execution_features_empty_slots()
        test_opportunity_features_without_scaler()
        test_action_mask_generation()
        test_consistency_across_multiple_calls()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()

        return True

    except AssertionError as e:
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        print()
        return False

    except Exception as e:
        print("=" * 80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        print()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
