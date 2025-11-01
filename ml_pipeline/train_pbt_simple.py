#!/usr/bin/env python3
"""
Simplified PBT implementation using multiprocessing instead of Ray Tune.

This avoids Ray's complex runtime environment setup and works directly
with the local filesystem.
"""

import argparse
import os
import sys
import multiprocessing as mp
from datetime import datetime
import json
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from rl.environment import FundingArbitrageEnv


class PopulationAgent:
    """Single agent in the PBT population."""

    def __init__(self, agent_id, hyperparams, train_data, eval_data,
                 price_history, feature_scaler, save_dir):
        self.agent_id = agent_id
        self.hyperparams = hyperparams
        self.train_data = train_data
        self.eval_data = eval_data
        self.price_history = price_history
        self.feature_scaler = feature_scaler
        self.save_dir = save_dir
        self.best_reward = -np.inf

    def train(self, timesteps_per_iteration):
        """Train for one PBT iteration."""
        # Create environments
        train_env = DummyVecEnv([lambda: Monitor(FundingArbitrageEnv(
            data_path=self.train_data,
            price_history_path=self.price_history,
            feature_scaler_path=self.feature_scaler,
            initial_capital=10000.0,
            episode_length_days=3,
            max_positions=3,
            max_opportunities_per_hour=5,
            use_full_range_episodes=False,
            quality_entry_bonus=0.0,
            quality_entry_penalty=0.0,
        ))])

        eval_env = Monitor(FundingArbitrageEnv(
            data_path=self.eval_data,
            price_history_path=self.price_history,
            feature_scaler_path=self.feature_scaler,
            initial_capital=10000.0,
            max_positions=3,
            max_opportunities_per_hour=5,
            use_full_range_episodes=True,  # Full range for evaluation (episode_length_days ignored)
            quality_entry_bonus=0.0,
            quality_entry_penalty=0.0,
        ))

        # Always create a new model with current hyperparameters
        # This ensures hyperparameter mutations from PBT take effect
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self.hyperparams['learning_rate'],
            gamma=self.hyperparams['gamma'],
            gae_lambda=self.hyperparams['gae_lambda'],
            ent_coef=self.hyperparams['ent_coef'],
            clip_range=self.hyperparams['clip_range'],
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0,
        )

        # Load weights from checkpoint if it exists (for continuing training)
        model_path = os.path.join(self.save_dir, f'agent_{self.agent_id}_model.zip')
        if os.path.exists(model_path):
            # Load parameters (weights) from saved model
            saved_model = PPO.load(model_path)
            model.policy.load_state_dict(saved_model.policy.state_dict())
            del saved_model  # Free memory

        # Train
        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)

        # Evaluate
        obs = eval_env.reset()[0]
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # Extract metrics (use episode-level metrics from final info)
        pnl_pct = info.get('episode_pnl_pct', 0.0)
        win_rate = info.get('episode_win_rate', 0.0)  # Fixed: was 'win_rate'
        trades = info.get('episode_trades_count', 0)   # Fixed: was 'trades_count'

        # Log evaluation details for verification
        if self.agent_id == 0:  # Only log for first agent to reduce spam
            print(f"  Agent {self.agent_id} evaluation: {steps} steps, {trades} trades")

        # Save model
        model.save(model_path)

        # Save hyperparameters
        hyperparam_path = os.path.join(self.save_dir, f'agent_{self.agent_id}_hyperparams.json')
        with open(hyperparam_path, 'w') as f:
            json.dump(self.hyperparams, f, indent=2)

        train_env.close()
        eval_env.close()

        return {
            'agent_id': self.agent_id,
            'episode_reward': episode_reward,
            'pnl_pct': pnl_pct,
            'win_rate': win_rate,
            'trades': trades,
            'hyperparams': self.hyperparams.copy()
        }


def train_agent_worker(args):
    """Worker function for multiprocessing."""
    agent, timesteps = args
    try:
        return agent.train(timesteps)
    except Exception as e:
        import traceback
        print(f"Agent {agent.agent_id} failed: {e}")
        print(traceback.format_exc())
        return None


def mutate_hyperparams(hyperparams, mutation_rate=0.2):
    """Mutate hyperparameters for exploration."""
    new_params = hyperparams.copy()

    # Perturb each parameter with some probability
    if np.random.rand() < 0.5:
        # Resample learning rate
        new_params['learning_rate'] = 10 ** np.random.uniform(-5, -3)
    else:
        # Perturb learning rate by ±20%
        factor = np.random.uniform(1 - mutation_rate, 1 + mutation_rate)
        new_params['learning_rate'] = np.clip(hyperparams['learning_rate'] * factor, 1e-5, 1e-3)

    if np.random.rand() < 0.5:
        new_params['gamma'] = np.random.choice([0.95, 0.97, 0.99, 0.995])

    if np.random.rand() < 0.5:
        new_params['gae_lambda'] = np.random.uniform(0.9, 0.99)
    else:
        factor = np.random.uniform(1 - mutation_rate, 1 + mutation_rate)
        new_params['gae_lambda'] = np.clip(hyperparams['gae_lambda'] * factor, 0.9, 0.99)

    if np.random.rand() < 0.5:
        new_params['ent_coef'] = np.random.uniform(0.01, 0.1)
    else:
        factor = np.random.uniform(1 - mutation_rate, 1 + mutation_rate)
        new_params['ent_coef'] = np.clip(hyperparams['ent_coef'] * factor, 0.01, 0.1)

    if np.random.rand() < 0.5:
        new_params['clip_range'] = np.random.uniform(0.1, 0.4)
    else:
        factor = np.random.uniform(1 - mutation_rate, 1 + mutation_rate)
        new_params['clip_range'] = np.clip(hyperparams['clip_range'] * factor, 0.1, 0.4)

    return new_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--perturbation-interval', type=int, default=20000)
    parser.add_argument('--train-data', type=str, default='data/rl_train.csv')
    parser.add_argument('--eval-data', type=str, default='data/rl_test.csv')  # Use test set
    parser.add_argument('--price-history', type=str, default='data/price_history')
    parser.add_argument('--feature-scaler', type=str, default='models/rl/feature_scaler.pkl')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'models/pbt_{timestamp}'

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SIMPLIFIED PBT FOR CRYPTO ARBITRAGE")
    print("="*70)
    print(f"Population: {args.population}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Perturbation interval: {args.perturbation_interval:,}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")

    # Initialize population with random hyperparameters
    agents = []
    for i in range(args.population):
        hyperparams = {
            'learning_rate': 10 ** np.random.uniform(-5, -3),
            'gamma': np.random.choice([0.95, 0.97, 0.99, 0.995]),
            'gae_lambda': np.random.uniform(0.9, 0.99),
            'ent_coef': np.random.uniform(0.01, 0.1),
            'clip_range': np.random.uniform(0.1, 0.4),
        }
        agent = PopulationAgent(
            agent_id=i,
            hyperparams=hyperparams,
            train_data=args.train_data,
            eval_data=args.eval_data,
            price_history=args.price_history,
            feature_scaler=args.feature_scaler,
            save_dir=args.output_dir
        )
        agents.append(agent)

    # PBT loop
    num_iterations = args.timesteps // args.perturbation_interval

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*70}\n")

        # Train all agents in parallel
        with mp.Pool(processes=min(args.population, mp.cpu_count())) as pool:
            results = pool.map(train_agent_worker, [(agent, args.perturbation_interval) for agent in agents])

        # Filter out failed agents
        results = [r for r in results if r is not None]

        if not results:
            print("❌ All agents failed!")
            break

        # Sort by performance
        results.sort(key=lambda x: x['episode_reward'], reverse=True)

        # Print results
        print("\nAgent Performance:")
        print(f"{'ID':<5} {'Reward':<12} {'P&L%':<10} {'WinRate':<10} {'Trades':<8}")
        print("-" * 60)
        for r in results:
            print(f"{r['agent_id']:<5} {r['episode_reward']:+11.2f} {r['pnl_pct']:+9.2f}% {r['win_rate']:9.1f}% {r['trades']:<8}")

        # PBT: Bottom 25% copy from top 25%
        n_bottom = max(1, len(results) // 4)
        n_top = max(1, len(results) // 4)

        for i in range(n_bottom):
            # Bottom agent copies from random top agent
            bottom_id = results[-(i+1)]['agent_id']
            top_id = results[np.random.randint(n_top)]['agent_id']

            if bottom_id != top_id:
                print(f"\n🔄 Agent {bottom_id} copying from Agent {top_id}")

                # Copy model weights
                src_model = os.path.join(args.output_dir, f'agent_{top_id}_model.zip')
                dst_model = os.path.join(args.output_dir, f'agent_{bottom_id}_model.zip')
                import shutil
                shutil.copy(src_model, dst_model)

                # Mutate hyperparameters
                top_params = results[np.random.randint(n_top)]['hyperparams']
                new_params = mutate_hyperparams(top_params)
                agents[bottom_id].hyperparams = new_params

                print(f"  New hyperparams: lr={new_params['learning_rate']:.6f}, "
                      f"gamma={new_params['gamma']}, ent_coef={new_params['ent_coef']:.4f}")

    # Final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest agent: {results[0]['agent_id']}")
    print(f"Episode Reward: {results[0]['episode_reward']:+.2f}")
    print(f"Episode P&L: {results[0]['pnl_pct']:+.2f}%")
    print(f"Win Rate: {results[0]['win_rate']:.1f}%")
    print(f"Trades: {results[0]['trades']}")
    print(f"\nBest model: {args.output_dir}/agent_{results[0]['agent_id']}_model.zip")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
