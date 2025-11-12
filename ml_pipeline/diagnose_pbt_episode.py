"""
Diagnostic to mimic exactly what PBT parallel training does
"""
import sys
import torch
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.algorithms.ppo_trainer import PPOTrainer
from models.rl.networks.modular_ppo import ModularPPONetwork

# Exactly match PBT settings
trading_config = TradingConfig.sample_random()
print("Trading Config:")
print(f"  Max leverage: {trading_config.max_leverage:.2f}")
print(f"  Target utilization: {trading_config.target_utilization:.2f}")
print(f"  Max positions: {trading_config.max_positions}")

reward_config = RewardConfig(
    funding_reward_scale=5.0,
    price_reward_scale=1.0,
    liquidation_penalty_scale=1000.0,
)
print("\nReward Config:")
print(f"  Funding scale: {reward_config.funding_reward_scale}")
print(f"  Price scale: {reward_config.price_reward_scale}")
print(f"  Liquidation penalty: {reward_config.liquidation_penalty_scale}")

env = FundingArbitrageEnv(
    data_path="data/rl_train.csv",
    initial_capital=10000,
    trading_config=trading_config,
    reward_config=reward_config,
    episode_length_days=7,
    step_hours=5.0 / 60.0,  # 5 minutes
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    verbose=False,
)

print(f"\nEnvironment:")
print(f"  Episode length: {env.episode_length_hours} hours")
print(f"  Step hours: {env.step_hours}")
print(f"  Max steps: ~{int(env.episode_length_hours / env.step_hours)}")

# Create trainer
network = ModularPPONetwork()
trainer = PPOTrainer(
    network=network,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    n_epochs=4,
    batch_size=64,
    device='cpu',
)

print("\nRunning 5 episodes with max_steps=1000...")
print("="*80)

episode_rewards = []
episode_lengths = []

for ep in range(5):
    stats = trainer.train_episode(env, max_steps=1000)
    episode_rewards.append(stats['episode_reward'])
    episode_lengths.append(stats['episode_length'])

    print(f"Episode {ep+1}:")
    print(f"  Reward: {stats['episode_reward']:.2f}")
    print(f"  Length: {stats['episode_length']} steps")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Mean reward: {np.mean(episode_rewards):.2f}")
print(f"Mean length: {np.mean(episode_lengths):.1f} steps")
print(f"Reward range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
