"""
Test that the training environment works with UnifiedFeatureBuilder.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from common.features import DIMS


def test_environment_initialization():
    """Test that environment initializes with UnifiedFeatureBuilder."""
    print("Test 1: Environment Initialization")
    print("-" * 80)

    # Use a small test dataset
    env = FundingArbitrageEnv(
        data_path='data/rl_test.csv',
        feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl',
        initial_capital=10000.0,
        trading_config=TradingConfig.get_moderate(),
        episode_length_days=1,
        verbose=True
    )

    print(f"✅ Environment initialized successfully")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n}")
    print(f"   Feature builder: {type(env.feature_builder).__name__}")
    print()

    return env


def test_reset_and_observation():
    """Test that reset produces correct observation shape."""
    print("Test 2: Reset and Observation")
    print("-" * 80)

    env = FundingArbitrageEnv(
        data_path='data/rl_test.csv',
        feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl',
        initial_capital=10000.0,
        episode_length_days=1,
        verbose=False
    )

    obs, info = env.reset()

    assert obs.shape == (DIMS.TOTAL,), f"Expected shape ({DIMS.TOTAL},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    assert np.isfinite(obs).all(), "Observation contains non-finite values"

    print(f"✅ Reset produced valid observation")
    print(f"   Shape: {obs.shape}")
    print(f"   Dtype: {obs.dtype}")
    print(f"   All finite: {np.isfinite(obs).all()}")
    print()

    return env, obs


def test_step():
    """Test that step produces correct observation."""
    print("Test 3: Step Function")
    print("-" * 80)

    env = FundingArbitrageEnv(
        data_path='data/rl_test.csv',
        feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl',
        initial_capital=10000.0,
        episode_length_days=1,
        verbose=False
    )

    obs, info = env.reset()

    # Take HOLD action
    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (DIMS.TOTAL,), f"Expected shape ({DIMS.TOTAL},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    assert np.isfinite(obs).all(), "Observation contains non-finite values"
    assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"

    print(f"✅ Step produced valid observation")
    print(f"   Shape: {obs.shape}")
    print(f"   Reward: {reward:.4f}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}")
    print()


def test_multiple_steps():
    """Test multiple steps."""
    print("Test 4: Multiple Steps")
    print("-" * 80)

    env = FundingArbitrageEnv(
        data_path='data/rl_test.csv',
        feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl',
        initial_capital=10000.0,
        episode_length_days=1,
        verbose=False
    )

    obs, info = env.reset()

    # Take 10 HOLD actions
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(0)

        if terminated or truncated:
            break

    print(f"✅ Completed {i+1} steps successfully")
    print(f"   Final observation shape: {obs.shape}")
    print(f"   Episode terminated: {terminated}")
    print(f"   Episode truncated: {truncated}")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRAINING ENVIRONMENT + UNIFIED FEATURE BUILDER TESTS")
    print("=" * 80)
    print()

    try:
        # Check if data file exists
        data_file = Path('data/rl_test.csv')
        if not data_file.exists():
            print(f"⚠️  Test data file not found: {data_file}")
            print(f"   Skipping environment tests (data dependency)")
            print()
            return True

        test_environment_initialization()
        test_reset_and_observation()
        test_step()
        test_multiple_steps()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()

        return True

    except FileNotFoundError as e:
        print("=" * 80)
        print(f"⚠️  TEST SKIPPED: Data file not found")
        print(f"   {e}")
        print("=" * 80)
        print()
        return True  # Not a failure, just missing test data

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
