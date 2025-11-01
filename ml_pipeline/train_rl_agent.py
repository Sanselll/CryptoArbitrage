"""
Train PPO Agent for Funding Rate Arbitrage

This script trains a Proximal Policy Optimization (PPO) agent to learn
optimal trading strategies for funding rate arbitrage.
"""

import sys
sys.path.insert(0, 'src')

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

from rl.environment import FundingArbitrageEnv


class EntropyAnnealingCallback(BaseCallback):
    """
    Custom callback to anneal entropy coefficient during training.

    Linearly decreases ent_coef from initial_ent to final_ent over total_timesteps.
    """
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps

    def _on_rollout_end(self) -> None:
        """Update entropy coefficient at the end of each rollout."""
        # Calculate progress (0.0 to 1.0)
        progress = self.num_timesteps / self.total_timesteps

        # Linear annealing: initial_ent → final_ent
        new_ent_coef = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        new_ent_coef = max(new_ent_coef, self.final_ent)  # Don't go below final value

        # Update model's entropy coefficient
        self.model.ent_coef = new_ent_coef

        # Log to tensorboard
        self.logger.record("train/ent_coef", new_ent_coef)

    def _on_step(self) -> bool:
        return True


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics to TensorBoard.

    Logs P&L, win rate, trade count, and reward-P&L alignment metrics
    for real-time monitoring of agent performance.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called after each environment step to log episode-level metrics."""
        # Check if any environments finished an episode
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                # Episode-level metrics are only available when episode ends
                if 'episode_pnl_pct' in info:
                    episode_reward = info.get('episode', {}).get('r', 0.0)
                    pnl_pct = info['episode_pnl_pct']
                    trades_count = info['episode_trades_count']
                    win_rate = info['episode_win_rate']
                    final_value = info['episode_final_value']

                    # Log to TensorBoard
                    self.logger.record('trading/episode_pnl_pct', pnl_pct)
                    self.logger.record('trading/episode_reward', episode_reward)
                    self.logger.record('trading/trades_count', trades_count)
                    self.logger.record('trading/win_rate', win_rate)
                    self.logger.record('trading/final_portfolio_value', final_value)

                    # Alignment metric: reward per 1% P&L (should be consistent)
                    if pnl_pct != 0:
                        reward_per_pnl = episode_reward / pnl_pct
                        self.logger.record('trading/reward_per_pnl', reward_per_pnl)

        return True


class DeterministicEvalCallback(BaseCallback):
    """
    Custom evaluation callback that uses FULL-RANGE episodes for consistent evaluation.

    In full-range mode, each evaluation runs a SINGLE episode covering the ENTIRE eval dataset.
    This ensures completely deterministic and comprehensive evaluation at every checkpoint.
    """
    def __init__(
        self,
        eval_env,
        eval_freq: int = 5000,
        best_model_save_path: str = None,
        log_path: str = None,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic

        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

        # Create save directory
        if best_model_save_path is not None:
            os.makedirs(best_model_save_path, exist_ok=True)
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each training step."""
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation on FULL-RANGE episode (no seed needed, env uses full data range)
            obs = self.eval_env.reset()[0]
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            mean_reward = episode_reward
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f}, episode_length={episode_length}")

            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", episode_length)

            # Save best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.best_mean_reward = mean_reward

                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        return True


def create_env(data_path: str, price_history_path: str = None, feature_scaler_path: str = None, seed: int = None, use_full_range_episodes: bool = False, **reward_kwargs):
    """Create and wrap the environment."""
    env = FundingArbitrageEnv(
        data_path=data_path,
        price_history_path=price_history_path,
        feature_scaler_path=feature_scaler_path,
        initial_capital=10000.0,
        episode_length_days=3,  # 72 hours per episode (ignored if use_full_range_episodes=True)
        max_positions=3,  # FIXED: Consistent with environment default
        max_opportunities_per_hour=5,  # Reduced for clearer signal
        use_full_range_episodes=use_full_range_episodes,  # Full-range episode mode
        **reward_kwargs  # Pass through reward shaping parameters
    )

    # Wrap in Monitor for tracking
    env = Monitor(env)

    return env


def make_env(data_path: str, price_history_path: str = None, feature_scaler_path: str = None, seed: int = None, rank: int = 0, **reward_kwargs):
    """Factory function for parallel environment creation (needed for SubprocVecEnv)."""
    def _init():
        env = create_env(data_path, price_history_path, feature_scaler_path, seed=seed + rank if seed else None, **reward_kwargs)
        return env
    return _init


def train_ppo_agent(
    data_path: str = 'data/rl_train.csv',
    eval_data_path: str = 'data/rl_test.csv',
    price_history_path: str = 'data/price_history',
    feature_scaler_path: str = 'models/rl/feature_scaler.pkl',
    total_timesteps: int = 1000000,  # INCREASED: 500k → 1M for better learning with noisy data
    save_dir: str = 'models/rl',
    # UPDATED PPO hyperparameters (based on analysis)
    learning_rate: float = 5.989e-05,
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 17,
    gamma: float = 0.99,  # V3: INCREASED 0.96 → 0.99 to value future rewards more (hold longer)
    gae_lambda: float = 0.98,  # V3: ADJUSTED 0.9888 → 0.98 for faster credit assignment
    clip_range: float = 0.243,
    ent_coef_initial: float = 0.08,  # INCREASED: 0.0139 → 0.08 for more exploration
    ent_coef_final: float = 0.02,  # INCREASED: 0.001 → 0.02 to maintain exploration
    # UPDATED reward shaping parameters (v2: reduced variance, proportional quality signals)
    pnl_reward_scale: float = 1.0,  # REDUCED: 3.0 → 1.0 to reduce variance, let quality signals dominate
    hold_bonus: float = 0.0,  # REMOVED: 0.212 → 0.0 to avoid inaction bias
    quality_entry_bonus: float = 0.5,  # Multiplier for expected profit (50% bonus)
    quality_entry_penalty: float = -0.5,  # Multiplier for expected profit (-50% penalty)
    seed: int = 42
):
    """
    Train PPO agent for funding arbitrage.

    Args:
        data_path: Path to RL opportunities CSV
        eval_data_path: Path to evaluation opportunities CSV
        price_history_path: Path to price history directory
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Optimization epochs per rollout
        gamma: Discount factor
        seed: Random seed
    """
    print("="*80)
    print("TRAINING PPO AGENT FOR FUNDING ARBITRAGE")
    print("="*80)
    print(f"Data: {data_path}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {seed}")
    print()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"ppo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare reward shaping parameters
    reward_kwargs = {
        'pnl_reward_scale': pnl_reward_scale,
        'hold_bonus': hold_bonus,
        'quality_entry_bonus': quality_entry_bonus,
        'quality_entry_penalty': quality_entry_penalty
    }

    # Create training environment with parallel workers (8 cores for 10-core CPU)
    n_envs = 8  # Leave 2 cores for system
    print(f"Creating training environment with {n_envs} parallel workers...")
    train_env = SubprocVecEnv([make_env(data_path, price_history_path, feature_scaler_path, seed, i, **reward_kwargs) for i in range(n_envs)])

    # Create evaluation environment (separate for unbiased eval)
    # Use full-range episodes for comprehensive, deterministic evaluation
    print("Creating evaluation environment (full-range mode)...")
    eval_env = create_env(eval_data_path, price_history_path, feature_scaler_path, seed=None, use_full_range_episodes=True, **reward_kwargs)

    # Create PPO agent with improved architecture
    print("\nInitializing PPO agent...")

    # CPU is faster than MPS for MLP policies with small batches
    # MPS overhead (memory transfers) outweighs benefits for non-CNN models
    device = "cpu"
    print("💻 Using CPU (faster than MPS for MLP + small batches)")

    # Policy kwargs: Medium network (256x256)
    # Note: Feature normalization is handled by StandardScaler in environment
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=False,
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        ent_coef=ent_coef_initial,  # Will be annealed by callback
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        seed=seed
    )

    print(f"\nModel architecture:")
    print(f"  Policy network: MLP (256 → 256 hidden layers)")
    print(f"  Value network: MLP (256 → 256 hidden layers)")
    print(f"  Total parameters: ~200K per network")
    print(f"  Observation space: {train_env.observation_space.shape[0]} dimensions")
    print(f"  Action space: {train_env.action_space.n} discrete actions")
    print(f"  Entropy: {ent_coef_initial:.4f} → {ent_coef_final:.4f} (annealed via callback)")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  GAE Lambda: {gae_lambda:.4f}")
    print(f"  Clip range: {clip_range:.3f}")
    print(f"  Reward scale: {pnl_reward_scale:.3f}")
    print(f"  Hold bonus: {hold_bonus:.3f}")
    print(f"  Quality entry bonus: {quality_entry_bonus:.3f}")
    print(f"  Quality entry penalty: {quality_entry_penalty:.3f}")
    print()

    # Create callbacks
    # Deterministic evaluation callback - evaluate every 20000 steps on FULL-RANGE episode
    # Note: Increased freq from 5000 to 20000 since full-range eval takes ~17× longer
    eval_callback = DeterministicEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=20000,  # Reduced frequency due to longer eval time
        deterministic=True,
        verbose=1
    )

    # Checkpoint callback - save every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_checkpoint"
    )

    # Entropy annealing callback
    entropy_callback = EntropyAnnealingCallback(
        initial_ent=ent_coef_initial,
        final_ent=ent_coef_final,
        total_timesteps=total_timesteps
    )

    # Trading metrics callback for TensorBoard logging
    trading_metrics_callback = TradingMetricsCallback()

    callbacks = [eval_callback, checkpoint_callback, entropy_callback, trading_metrics_callback]

    # Train the agent
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Progress will be logged to: {run_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(run_dir, "final_model")
    model.save(final_model_path)
    print(f"\n✅ Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(run_dir, 'best_model')}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model, run_dir


def evaluate_agent(model_path: str, data_path: str, price_history_path: str = None, feature_scaler_path: str = None, use_full_range: bool = True, n_episodes: int = 1, seed: int = 999):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to saved model
        data_path: Path to evaluation data
        price_history_path: Path to price history directory
        feature_scaler_path: Path to feature scaler pickle file
        use_full_range: If True, runs single episode covering entire data range (default)
        n_episodes: Number of episodes (ignored if use_full_range=True)
        seed: Random seed (ignored if use_full_range=True)
    """
    print("\n" + "="*80)
    print("EVALUATING TRAINED AGENT")
    print("="*80)

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create evaluation environment with full-range episodes for comprehensive testing
    env = create_env(data_path, price_history_path, feature_scaler_path, seed=None, use_full_range_episodes=use_full_range)

    # Run evaluation (single full-range episode or multiple random episodes)
    num_runs = 1 if use_full_range else n_episodes
    episode_rewards = []
    episode_pnls = []
    episode_lengths = []
    all_trades = []  # Track all trades across episodes

    # Import for action probability extraction
    from stable_baselines3.common.utils import obs_as_tensor

    for episode in range(num_runs):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        # Track action data for ENTER and EXIT actions only
        enter_actions = {}  # key: step, value: (action, prob, hold_prob)
        exit_actions = {}   # key: step, value: (action, prob, hold_prob)

        while not done:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)

            # Extract action probabilities for this step
            # Add batch dimension if needed (PPO policy expects batched input)
            obs_for_policy = obs.reshape(1, -1) if obs.ndim == 1 else obs
            obs_tensor = obs_as_tensor(obs_for_policy, model.policy.device)
            distribution = model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]

            # Track ENTER or EXIT actions with their probabilities
            if 1 <= action <= 5:  # ENTER action
                enter_actions[steps] = (action, float(action_probs[action]), float(action_probs[0]))
            elif 6 <= action <= 8:  # EXIT action
                exit_actions[steps] = (action, float(action_probs[action]), float(action_probs[0]))

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_pnls.append(info['total_pnl_pct'])
        episode_lengths.append(steps)

        # Collect closed positions from this episode
        if hasattr(env.unwrapped, 'portfolio') and hasattr(env.unwrapped.portfolio, 'closed_positions'):
            for i, pos in enumerate(env.unwrapped.portfolio.closed_positions):
                # Try to find entry/exit actions - use position order as approximation
                # Since positions close in order, we can match them sequentially
                entry_data = None
                exit_data = None

                # Get the i-th ENTER and EXIT actions
                if i < len(enter_actions):
                    enter_step = sorted(enter_actions.keys())[i] if enter_actions else None
                    if enter_step is not None:
                        action, prob, hold_prob = enter_actions[enter_step]
                        entry_data = {'action': action, 'prob': prob, 'hold_prob': hold_prob}

                if i < len(exit_actions):
                    exit_step = sorted(exit_actions.keys())[i] if exit_actions else None
                    if exit_step is not None:
                        action, prob, hold_prob = exit_actions[exit_step]
                        exit_data = {'action': action, 'prob': prob, 'hold_prob': hold_prob}

                all_trades.append({
                    'episode': episode + 1,
                    'symbol': pos.symbol,
                    'entry_time': pos.entry_time,
                    'exit_time': pos.exit_time,
                    'pnl_pct': pos.realized_pnl_pct,
                    'pnl_usd': pos.realized_pnl_usd,
                    'duration_hours': (pos.exit_time - pos.entry_time).total_seconds() / 3600,
                    'entry_fees_usd': pos.entry_fees_paid_usd,
                    'exit_fees_usd': pos.exit_fees_paid_usd,
                    'position_size_usd': pos.position_size_usd,
                    'entry_action': entry_data['action'] if entry_data else None,
                    'entry_probability': entry_data['prob'] if entry_data else None,
                    'entry_hold_prob': entry_data['hold_prob'] if entry_data else None,
                    'exit_action': exit_data['action'] if exit_data else None,
                    'exit_probability': exit_data['prob'] if exit_data else None,
                    'exit_hold_prob': exit_data['hold_prob'] if exit_data else None
                })

        mode_str = "Full-range" if use_full_range else f"{episode+1}/{num_runs}"
        print(f"Episode ({mode_str}): "
              f"Reward={episode_reward:+.2f}, "
              f"P&L={info['total_pnl_pct']:+.2f}%, "
              f"Steps={steps}")

    # Summary statistics
    print("\n" + "─"*60)
    print("EVALUATION SUMMARY")
    print("─"*60)
    if use_full_range:
        print(f"Mode: Full-range episode (entire dataset)")
        print(f"Total Reward: {episode_rewards[0]:+.2f}")
        print(f"Total P&L: {episode_pnls[0]:+.2f}%")
        print(f"Episode Length: {episode_lengths[0]} steps")
    else:
        print(f"Mode: {num_runs} random episodes")
        print(f"Average Reward: {np.mean(episode_rewards):+.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average P&L: {np.mean(episode_pnls):+.2f}% ± {np.std(episode_pnls):.2f}%")
        print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
        print(f"Win Rate: {sum(1 for pnl in episode_pnls if pnl > 0) / num_runs * 100:.1f}%")
    print("─"*60)

    # Show top profitable trades
    if all_trades:
        sorted_trades = sorted(all_trades, key=lambda x: x['pnl_usd'], reverse=True)
        top_n = min(5, len(sorted_trades))

        print(f"\nTOP {top_n} MOST PROFITABLE TRADES:")
        print("─"*80)
        for i, trade in enumerate(sorted_trades[:top_n], 1):
            print(f"{i}. {trade['symbol']} | "
                  f"Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
                  f"P&L: ${trade['pnl_usd']:+.2f} ({trade['pnl_pct']:+.2f}%) | "
                  f"Duration: {trade['duration_hours']:.1f}h")
        print("─"*80)

    # Save all trades to CSV for detailed analysis
    if all_trades:
        import pandas as pd
        trades_df = pd.DataFrame(all_trades)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'evaluation_trades_{timestamp}.csv'
        trades_df.to_csv(csv_filename, index=False)
        print(f"\n✅ All trades saved to: {csv_filename}")
        print(f"   Total trades: {len(all_trades)}")
        print(f"   Winning trades: {len([t for t in all_trades if t['pnl_usd'] > 0])}")
        print(f"   Losing trades: {len([t for t in all_trades if t['pnl_usd'] <= 0])}")

    env.close()

    return {
        'episode_rewards': episode_rewards,
        'episode_pnls': episode_pnls,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'mean_pnl': np.mean(episode_pnls),
        'win_rate': sum(1 for pnl in episode_pnls if pnl > 0) / n_episodes,
        'all_trades': all_trades
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO agent for funding arbitrage')
    parser.add_argument('--data-path', type=str, default='data/rl_train.csv',
                        help='Path to training RL opportunities CSV')
    parser.add_argument('--eval-data-path', type=str, default='data/rl_test.csv',
                        help='Path to evaluation RL opportunities CSV')
    parser.add_argument('--price-history-path', type=str, default='data/price_history',
                        help='Path to price history directory')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models/rl',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path to model for evaluation only (skip training)')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')

    args = parser.parse_args()

    # Default feature scaler path
    default_scaler_path = 'models/rl/feature_scaler.pkl'

    if args.eval_only:
        # Evaluation only
        evaluate_agent(
            model_path=args.eval_only,
            data_path=args.eval_data_path,  # Use eval_data_path, not data_path
            price_history_path=args.price_history_path,
            feature_scaler_path=default_scaler_path,
            n_episodes=args.n_eval_episodes
        )
    else:
        # Train agent
        model, run_dir = train_ppo_agent(
            data_path=args.data_path,
            eval_data_path=args.eval_data_path,
            price_history_path=args.price_history_path,
            feature_scaler_path=default_scaler_path,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            learning_rate=args.lr,
            seed=args.seed
        )

        # Evaluate trained agent on test set
        best_model_path = os.path.join(run_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            print("\nEvaluating best model on test set...")
            evaluate_agent(
                model_path=best_model_path,
                data_path=args.eval_data_path,  # Use specified eval data
                price_history_path=args.price_history_path,
                feature_scaler_path=default_scaler_path,
                n_episodes=args.n_eval_episodes
            )
