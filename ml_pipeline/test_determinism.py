"""
Test that evaluation episodes are deterministic with fixed seeds.
"""
import sys
sys.path.insert(0, 'src')

from rl.environment import FundingArbitrageEnv

# Create environment
env = FundingArbitrageEnv(
    data_path='data/rl_test.csv',
    price_history_path='data/price_history',
    initial_capital=10000,
    episode_length_days=3
)

print("Testing deterministic episode generation with fixed seeds...")
print("="*80)

# Test seed 10000 three times
seed = 10000
episode_starts = []

for i in range(3):
    obs, info = env.reset(seed=seed)
    episode_starts.append(env.episode_start)
    print(f"Run {i+1} with seed {seed}: Episode start = {env.episode_start}")

# Check if all are the same
if episode_starts[0] == episode_starts[1] == episode_starts[2]:
    print(f"\n✅ SUCCESS: All episodes with seed {seed} start at the same time!")
else:
    print(f"\n❌ FAILURE: Episodes with seed {seed} have different start times!")
    print(f"   Run 1: {episode_starts[0]}")
    print(f"   Run 2: {episode_starts[1]}")
    print(f"   Run 3: {episode_starts[2]}")

print("\n" + "="*80)
print("Testing that different seeds produce different episodes...")

# Test different seeds
seeds_to_test = [10000, 10001, 10002]
starts = {}

for seed in seeds_to_test:
    obs, info = env.reset(seed=seed)
    starts[seed] = env.episode_start
    print(f"Seed {seed}: Episode start = {env.episode_start}")

# Check that different seeds give different episodes
if len(set(starts.values())) == len(seeds_to_test):
    print(f"\n✅ SUCCESS: Different seeds produce different episodes!")
else:
    print(f"\n❌ FAILURE: Different seeds produced same episodes!")

print("\n" + "="*80)
print("Determinism test complete!")
