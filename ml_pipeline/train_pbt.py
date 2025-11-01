#!/usr/bin/env python3
"""
Population Based Training (PBT) for Crypto Arbitrage RL Agent

This script uses Ray Tune's PopulationBasedTraining scheduler to automatically
discover optimal hyperparameters while training multiple PPO agents in parallel.

PBT works by:
1. Training a population of agents with different hyperparameters
2. Periodically evaluating all agents
3. Weak agents copy weights from strong agents (exploitation)
4. Perturbing hyperparameters for exploration (mutation)

This addresses the seed variance problem and automatically tunes hyperparameters.

Usage:
    python train_pbt.py --population 8 --timesteps 500000
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, 'src')

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl.environment import FundingArbitrageEnv


class RayTuneCallback(BaseCallback):
    """
    Callback that reports metrics to Ray Tune after each evaluation.
    """
    def __init__(self, eval_env, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run full-range evaluation episode
            obs = self.eval_env.reset()[0]
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            # Get trading metrics from final info
            final_pnl_pct = info.get('episode_pnl_pct', 0.0)
            final_portfolio = info.get('portfolio_value', 10000.0)
            trades_count = info.get('trades_count', 0)
            win_rate = info.get('win_rate', 0.0)

            # Report to Ray Tune
            # Note: Ray Tune will use 'episode_reward' for PBT fitness by default
            tune.report(
                timesteps=self.num_timesteps,
                episode_reward=episode_reward,
                episode_pnl_pct=final_pnl_pct,
                portfolio_value=final_portfolio,
                trades_count=trades_count,
                win_rate=win_rate,
                episode_length=episode_length
            )

            # Track best model
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward

        return True


def create_env(data_path: str, price_history_path: str = None,
               feature_scaler_path: str = None, use_full_range_episodes: bool = False,
               **reward_kwargs):
    """Create and wrap a single environment."""
    env = FundingArbitrageEnv(
        data_path=data_path,
        price_history_path=price_history_path,
        feature_scaler_path=feature_scaler_path,
        initial_capital=10000.0,
        episode_length_days=3,  # Ignored if use_full_range_episodes=True
        max_positions=3,
        max_opportunities_per_hour=5,
        use_full_range_episodes=use_full_range_episodes,
        **reward_kwargs
    )
    return Monitor(env)


def make_env(data_path, price_history_path, feature_scaler_path, **reward_kwargs):
    """Factory function for creating environments (needed for SubprocVecEnv)."""
    def _init():
        return create_env(data_path, price_history_path, feature_scaler_path,
                         use_full_range_episodes=False, **reward_kwargs)
    return _init


def train_ppo_pbt(config, checkpoint_dir=None):
    """
    Training function for a single agent in the PBT population.

    This function is called by Ray Tune for each agent in the population.

    Args:
        config: Hyperparameter configuration from Ray Tune
        checkpoint_dir: Directory to restore from (for exploitation)
    """
    # Extract paths and settings from config
    train_data_path = config['train_data_path']
    eval_data_path = config['eval_data_path']
    price_history_path = config['price_history_path']
    feature_scaler_path = config['feature_scaler_path']
    n_envs = config['n_envs']
    eval_freq = config['eval_freq']
    total_timesteps = config['total_timesteps']

    # Reward parameters (pure P&L learning)
    reward_kwargs = {
        'quality_entry_bonus': 0.0,
        'quality_entry_penalty': 0.0,
    }

    # Create training environments (random episodes)
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(train_data_path, price_history_path, feature_scaler_path, **reward_kwargs)
            for _ in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(train_data_path, price_history_path, feature_scaler_path, **reward_kwargs)
        ])

    # Create evaluation environment (full-range episodes)
    eval_env = create_env(eval_data_path, price_history_path, feature_scaler_path,
                         use_full_range_episodes=True, **reward_kwargs)

    # PPO hyperparameters from config
    learning_rate = config['learning_rate']
    gamma = config['gamma']
    gae_lambda = config['gae_lambda']
    ent_coef = config['ent_coef']
    clip_range = config['clip_range']

    # Create or restore model
    if checkpoint_dir:
        # Restore from checkpoint (exploitation in PBT)
        model_path = os.path.join(checkpoint_dir, "model.zip")
        model = PPO.load(model_path, env=env)

        # Update hyperparameters (PBT mutation)
        model.learning_rate = learning_rate
        model.gamma = gamma
        model.gae_lambda = gae_lambda
        model.ent_coef = ent_coef
        model.clip_range = clip_range

        print(f"Restored model from {checkpoint_dir}")
        print(f"Updated hyperparameters: lr={learning_rate:.6f}, gamma={gamma}, "
              f"ent_coef={ent_coef:.4f}, clip_range={clip_range}")
    else:
        # Create new model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0,
            tensorboard_log=None,  # Ray Tune handles logging
        )

    # Create Ray Tune callback
    ray_callback = RayTuneCallback(eval_env, eval_freq=eval_freq)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=ray_callback,
        reset_num_timesteps=False,  # Continue from checkpoint if restoring
    )

    # Final checkpoint - save model
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        model_path = os.path.join(temp_checkpoint_dir, "model.zip")
        model.save(model_path)

        # Create checkpoint for Ray Tune
        checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report(metrics={"done": True}, checkpoint=checkpoint)

    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Population Based Training for Crypto Arbitrage')
    parser.add_argument('--population', type=int, default=8,
                      help='Population size (number of parallel agents)')
    parser.add_argument('--timesteps', type=int, default=500000,
                      help='Total timesteps per agent')
    parser.add_argument('--n-envs', type=int, default=1,
                      help='Number of parallel environments per agent (reduce if population is large)')
    parser.add_argument('--perturbation-interval', type=int, default=20000,
                      help='Timesteps between PBT perturbations')
    parser.add_argument('--eval-freq', type=int, default=5000,
                      help='Timesteps between evaluations')
    parser.add_argument('--train-data', type=str, default='data/rl_train.csv',
                      help='Path to training data')
    parser.add_argument('--eval-data', type=str, default='data/rl_eval.csv',
                      help='Path to evaluation data')
    parser.add_argument('--price-history', type=str, default='data/price_history',
                      help='Path to price history directory')
    parser.add_argument('--feature-scaler', type=str, default='models/rl/feature_scaler.pkl',
                      help='Path to feature scaler')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for Ray Tune results')

    args = parser.parse_args()

    # Create output directory (must be absolute path for Ray Tune)
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.abspath(f'ray_results/pbt_{timestamp}')
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("POPULATION BASED TRAINING (PBT) FOR CRYPTO ARBITRAGE")
    print("="*70)
    print(f"Population size: {args.population}")
    print(f"Timesteps per agent: {args.timesteps:,}")
    print(f"Parallel envs per agent: {args.n_envs}")
    print(f"Perturbation interval: {args.perturbation_interval:,} steps")
    print(f"Evaluation frequency: {args.eval_freq:,} steps")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")

    # Initialize Ray with runtime environment
    # This ensures workers can import from 'src' directory
    # Exclude large model files but keep source code and data
    runtime_env = {
        "env_vars": {"PYTHONPATH": os.path.join(os.getcwd(), "src")},
        "working_dir": os.getcwd(),
        "excludes": [
            "models/rl*/",          # Exclude all RL model directories
            "models/xgboost/",       # Exclude XGBoost models
            "ray_results/",          # Exclude previous Ray results
            "data/opportunities/",   # Exclude raw opportunities (large)
            "*.log",                 # Exclude log files
            "training_log*.txt",     # Exclude training logs
            "evaluation_*.csv",      # Exclude evaluation outputs
            "__pycache__/",
            ".git/",
        ]
    }
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    # Define hyperparameter search space
    # Convert all paths to absolute paths for Ray workers
    config = {
        # Data paths (must be absolute for Ray workers)
        'train_data_path': os.path.abspath(args.train_data),
        'eval_data_path': os.path.abspath(args.eval_data),
        'price_history_path': os.path.abspath(args.price_history),
        'feature_scaler_path': os.path.abspath(args.feature_scaler),

        # Training settings
        'n_envs': args.n_envs,
        'eval_freq': args.eval_freq,
        'total_timesteps': args.timesteps,

        # Hyperparameters to tune (initial values and ranges)
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'gamma': tune.choice([0.95, 0.97, 0.99, 0.995]),
        'gae_lambda': tune.uniform(0.90, 0.99),
        'ent_coef': tune.uniform(0.01, 0.1),
        'clip_range': tune.uniform(0.1, 0.4),
    }

    # Configure PBT scheduler
    pbt_scheduler = PopulationBasedTraining(
        time_attr='timesteps',
        perturbation_interval=args.perturbation_interval,
        hyperparam_mutations={
            # Resample from original distribution
            'learning_rate': tune.loguniform(1e-5, 1e-3),
            'gamma': tune.choice([0.95, 0.97, 0.99, 0.995]),
            'gae_lambda': tune.uniform(0.90, 0.99),
            'ent_coef': tune.uniform(0.01, 0.1),
            'clip_range': tune.uniform(0.1, 0.4),
        },
        quantile_fraction=0.25,  # Bottom 25% are replaced
        resample_probability=0.25,  # 25% chance to resample instead of perturb
        log_config=True,  # Log PBT decisions
    )

    # Configure progress reporter
    reporter = CLIReporter(
        metric_columns=['timesteps', 'episode_reward', 'episode_pnl_pct', 'win_rate',
                       'trades_count', 'learning_rate', 'gamma', 'ent_coef'],
        max_progress_rows=args.population,
        max_report_frequency=30,  # Update every 30 seconds
    )

    # Run PBT
    print("Starting PBT training...")
    print("Ray Dashboard: http://127.0.0.1:8265")
    print("\nNote: Initial startup may take a few minutes as population initializes.\n")

    analysis = tune.run(
        train_ppo_pbt,
        name='pbt_crypto_arbitrage',
        scheduler=pbt_scheduler,
        metric='episode_reward',  # Fitness metric for PBT
        mode='max',  # Maximize episode reward
        num_samples=args.population,  # Population size
        config=config,
        storage_path=args.output_dir,  # Updated from local_dir
        progress_reporter=reporter,
        verbose=1,
        # Resource allocation
        resources_per_trial={'cpu': args.n_envs},  # Each agent gets n_envs CPUs
    )

    # Print results
    print("\n" + "="*70)
    print("PBT TRAINING COMPLETE")
    print("="*70)

    # Get best trial
    best_trial = analysis.get_best_trial('episode_reward', 'max', 'last')

    print(f"\nBest agent:")
    print(f"  Episode Reward: {best_trial.last_result['episode_reward']:+.2f}")
    print(f"  Episode P&L: {best_trial.last_result['episode_pnl_pct']:+.2f}%")
    print(f"  Win Rate: {best_trial.last_result['win_rate']:.1f}%")
    print(f"  Trades: {best_trial.last_result['trades_count']}")
    print(f"\n  Best Hyperparameters:")
    print(f"    learning_rate: {best_trial.config['learning_rate']:.6f}")
    print(f"    gamma: {best_trial.config['gamma']}")
    print(f"    gae_lambda: {best_trial.config['gae_lambda']:.4f}")
    print(f"    ent_coef: {best_trial.config['ent_coef']:.4f}")
    print(f"    clip_range: {best_trial.config['clip_range']:.4f}")

    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Best trial directory: {best_trial.logdir}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Evaluate best model on test set (Oct 22-28):")
    print(f"   python train_rl_agent.py --evaluate --model-path {best_trial.logdir}/model.zip")
    print("\n2. View detailed results and hyperparameter evolution:")
    print(f"   tensorboard --logdir {args.output_dir}")
    print("\n3. Compare all agents from final population:")
    print("   python evaluate_pbt_population.py --results-dir", args.output_dir)
    print("="*70 + "\n")

    # Cleanup
    ray.shutdown()


if __name__ == '__main__':
    main()
