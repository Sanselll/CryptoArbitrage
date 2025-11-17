"""
Quick test to verify parallel environment implementation works correctly.

Usage:
    python test_parallel_env.py
"""

import sys
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.core.vec_env import ParallelEnv


def create_test_env():
    """Create a simple test environment."""
    trading_config = TradingConfig(
        max_leverage=2.0,
        target_utilization=0.8,
        max_positions=2,
    )

    reward_config = RewardConfig(
        funding_reward_scale=1.0,
        price_reward_scale=1.0,
        liquidation_penalty_scale=10.0,
        opportunity_cost_scale=0.0,
    )

    env = FundingArbitrageEnv(
        data_path='data/rl_train.csv',
        price_history_path='data/symbol_data',
        feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl',
        initial_capital=10000.0,
        trading_config=trading_config,
        reward_config=reward_config,
        sample_random_config=False,
        episode_length_days=5,
        step_hours=5/60.0,  # 5 minutes
        use_full_range_episodes=False,
        verbose=False,
    )

    return env


def test_parallel_env(n_envs=4):
    """Test parallel environment with multiple workers."""
    print(f"\n{'='*80}")
    print(f"Testing Parallel Environment with {n_envs} workers")
    print(f"{'='*80}\n")

    # Create environment factories
    env_fns = [lambda: create_test_env() for _ in range(n_envs)]

    print(f"Creating {n_envs} parallel environments...")
    vec_env = ParallelEnv(env_fns)
    print(f"✅ Parallel environments created successfully\n")

    # Test reset
    print("Testing reset()...")
    obs, infos = vec_env.reset()
    print(f"✅ Reset successful")
    print(f"   Observations shape: {obs.shape}")
    print(f"   Number of info dicts: {len(infos)}\n")

    # Test step
    print("Testing step() with random actions...")
    actions = np.random.randint(0, 36, size=n_envs)
    print(f"   Actions: {actions}")

    next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    print(f"✅ Step successful")
    print(f"   Next obs shape: {next_obs.shape}")
    print(f"   Rewards: {rewards}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}\n")

    # Test action masks
    print("Testing get_action_masks()...")
    masks = vec_env.get_action_masks()
    if masks is not None:
        print(f"✅ Action masks retrieved")
        print(f"   Masks shape: {masks.shape}")
    else:
        print(f"⚠️  Action masks not available (this is OK if env doesn't support them)\n")

    # Test multiple steps
    print("Testing 10 random steps...")
    for i in range(10):
        actions = np.random.randint(0, 36, size=n_envs)
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        # Check for episode completions
        for j, info in enumerate(infos):
            if info.get('episode_ended', False):
                print(f"   Environment {j} completed an episode at step {i+1}")

    print(f"✅ 10 steps completed successfully\n")

    # Test cleanup
    print("Testing cleanup...")
    vec_env.close()
    print(f"✅ Environments closed successfully\n")

    print(f"{'='*80}")
    print("✅ ALL TESTS PASSED!")
    print(f"{'='*80}\n")


def test_single_vs_parallel():
    """Compare single and parallel environment behavior."""
    print(f"\n{'='*80}")
    print("Comparing Single vs Parallel Environment")
    print(f"{'='*80}\n")

    # Single environment
    print("Testing single environment...")
    single_env = create_test_env()
    obs_single, _ = single_env.reset()
    print(f"✅ Single env obs shape: {obs_single.shape}")

    # Parallel environment
    print("\nTesting parallel environment (4 workers)...")
    env_fns = [lambda: create_test_env() for _ in range(4)]
    vec_env = ParallelEnv(env_fns)
    obs_parallel, _ = vec_env.reset()
    print(f"✅ Parallel env obs shape: {obs_parallel.shape}")

    # Compare shapes
    print(f"\nShape comparison:")
    print(f"  Single:   {obs_single.shape}")
    print(f"  Parallel: {obs_parallel.shape}")
    print(f"  Expected: (4, {obs_single.shape[0]})")

    assert obs_parallel.shape == (4, obs_single.shape[0]), "Shape mismatch!"
    print(f"\n✅ Shapes match!\n")

    # Cleanup
    vec_env.close()
    single_env.close()


if __name__ == '__main__':
    try:
        # Run basic parallel env test
        test_parallel_env(n_envs=4)

        # Run comparison test
        test_single_vs_parallel()

        print("🎉 All tests completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
