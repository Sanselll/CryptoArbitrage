"""Diagnose where test_inference and server predictor diverge during episode."""

import numpy as np
import torch
from pathlib import Path

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.algorithms.ppo_trainer import PPOTrainer
from models.rl.networks.modular_ppo import ModularPPONetwork
from server.inference.rl_predictor import ModularRLPredictor
from test_server_vs_inference_comparison import convert_portfolio_to_server_format

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create environment
trading_config = TradingConfig(
    max_leverage=2.0,
    target_utilization=0.8,
    max_positions=2,
    stop_loss_threshold=-0.02,
    liquidation_buffer=0.15,
)
reward_config = RewardConfig(
    funding_reward_scale=1.0,
    price_reward_scale=1.0,
    liquidation_penalty_scale=10.0,
    opportunity_cost_scale=0.0,
)

env = FundingArbitrageEnv(
    data_path="data/rl_test.csv",
    initial_capital=10000.0,
    trading_config=trading_config,
    reward_config=reward_config,
    episode_length_days=7,
    step_hours=5/60,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    use_full_range_episodes=True,  # Match test_inference.py default
    verbose=False,
)

# Load models
print("Loading models...")
import sys
import os
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

network = ModularPPONetwork()
trainer = PPOTrainer(network=network, learning_rate=3e-4, device='cpu')
trainer.load("checkpoints/best_model.pt")
trainer.network.eval()  # Set to eval mode for inference

predictor = ModularRLPredictor("checkpoints/best_model.pt", device='cpu')

sys.stdout = old_stdout
print("✅ Models loaded\n")

# Reset environment
np.random.seed(seed)
obs_env, _ = env.reset(seed=seed)

# Run first 20 steps and compare
print(f"{'='*100}")
print(f"STEP-BY-STEP COMPARISON: Environment vs Server Predictor")
print(f"{'='*100}\n")

divergence_step = None
for step in range(20):
    # === ENVIRONMENT PATH ===
    obs_env_step = env._get_observation()
    action_mask_env = env._get_action_mask()
    action_env, _, _ = trainer.select_action(obs_env_step, action_mask_env, deterministic=True)

    # === SERVER PATH ===
    opportunities = [dict(opp) for opp in env.current_opportunities[:10]]
    price_data = env._get_current_prices()
    portfolio_dict = convert_portfolio_to_server_format(env.portfolio, price_data=price_data)
    config_dict = {
        'max_leverage': env.current_config.max_leverage,
        'target_utilization': env.current_config.target_utilization,
        'max_positions': env.current_config.max_positions,
        'stop_loss_threshold': env.current_config.stop_loss_threshold,
        'liquidation_buffer': env.current_config.liquidation_buffer,
    }

    obs_server = predictor._build_observation(config_dict, portfolio_dict, opportunities)
    positions = portfolio_dict.get('positions', [])
    num_positions = sum(1 for p in positions if p.get('symbol', '') != '')
    max_positions = config_dict.get('max_positions', 3)
    action_mask_server = predictor._get_action_mask(opportunities, num_positions, max_positions)
    action_server, _, _ = trainer.select_action(obs_server, action_mask_server, deterministic=True)

    # === COMPARE ===
    obs_diff = np.abs(obs_env_step - obs_server).max()
    mask_diff = np.sum(action_mask_env != action_mask_server)
    action_match = (action_env == action_server)

    status = "✅" if action_match else "❌"
    print(f"Step {step:3d}: {status}  Action: ENV={action_env:2d}  SERVER={action_server:2d}  "
          f"ObsDiff={obs_diff:.6f}  MaskDiff={mask_diff}  Positions={len(env.portfolio.positions)}")

    if not action_match and divergence_step is None:
        divergence_step = step
        print(f"\n{'='*100}")
        print(f"🚨 FIRST DIVERGENCE AT STEP {step}")
        print(f"{'='*100}")
        print(f"\nEnvironment Action: {action_env}")
        print(f"Server Action: {action_server}")

        # Save observations to file for detailed analysis
        np.save('/tmp/obs_env_divergence.npy', obs_env_step)
        np.save('/tmp/obs_server_divergence.npy', obs_server)
        np.save('/tmp/mask_env_divergence.npy', action_mask_env)
        np.save('/tmp/mask_server_divergence.npy', action_mask_server)
        print(f"\n📁 Saved observations to /tmp/obs_*_divergence.npy")

        # Test with exact same observation using both trainer instances
        action_test1, _, _ = trainer.select_action(obs_env_step, action_mask_env, deterministic=True)
        action_test2, _, _ = predictor.trainer.select_action(obs_env_step, action_mask_env, deterministic=True)
        print(f"\n🧪 Testing with ENV observation:")
        print(f"   Trainer (from test_inference): action={action_test1}")
        print(f"   Predictor.trainer (from server): action={action_test2}")
        print(f"   Match: {action_test1 == action_test2}")
        print(f"\nMax Observation Difference: {obs_diff:.6f}")
        print(f"Action Mask Differences: {mask_diff}")

        # Show mask differences
        if mask_diff > 0:
            print(f"\nAction Mask Comparison:")
            for i in range(36):
                if action_mask_env[i] != action_mask_server[i]:
                    print(f"  Action {i:2d}: ENV={action_mask_env[i]}  SERVER={action_mask_server[i]}")

        # Show observation differences > 0.001
        obs_diff_arr = np.abs(obs_env_step - obs_server)
        large_diffs = np.where(obs_diff_arr > 0.001)[0]
        if len(large_diffs) > 0:
            print(f"\nObservation Differences > 0.001:")
            for idx in large_diffs[:20]:
                print(f"  [{idx:3d}]: ENV={obs_env_step[idx]:10.6f}  SERVER={obs_server[idx]:10.6f}  diff={obs_diff_arr[idx]:10.6f}")

        print(f"\n{'='*100}\n")

    # Step environment (use environment action to keep environments in sync)
    obs_env, reward, terminated, truncated, info = env.step(action_env)

    if terminated or truncated:
        print(f"\n✅ Episode ended at step {step}")
        break

if divergence_step is None:
    print(f"\n{'='*100}")
    print(f"✅ NO DIVERGENCE FOUND in first 20 steps!")
    print(f"{'='*100}")
else:
    print(f"\nFirst divergence occurred at step {divergence_step}")
