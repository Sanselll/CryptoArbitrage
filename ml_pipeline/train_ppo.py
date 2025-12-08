"""
Train PPO agent on Funding Arbitrage Environment

IMPORTANT: Uses separate train and test data to prevent data leakage.
- Trains on train_data_path
- Evaluates on test_data_path (held-out test set)

Example usage:
    python train_ppo.py --num-episodes 1000
    python train_ppo.py --train-data-path data/rl_train.csv --test-data-path data/rl_test.csv --num-episodes 2000
"""

import argparse
import os
from pathlib import Path
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO agent on Funding Arbitrage')

    # Data
    parser.add_argument('--train-data-path', type=str, default='data/rl_train.csv',
                        help='Path to training opportunities CSV file')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test opportunities CSV file (for evaluation)')
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')
    parser.add_argument('--feature-scaler-path', type=str, default='trained_models/rl/feature_scaler_v3.pkl',
                        help='Path to fitted feature scaler pickle (V5.4: StandardScaler with 12 features)')

    # Environment (V3: 301→203 dimensions)
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital in USD')
    parser.add_argument('--episode-length-days', type=int, default=3,
                        help='Episode length in days (for training episodes)')
    parser.add_argument('--step-minutes', type=int, default=5,
                        help='Minutes per prediction step (default: 5 = 5-minute intervals)')
    parser.add_argument('--sample-random-config', action='store_true',
                        help='Sample random TradingConfig each episode for diversity')
    parser.add_argument('--eval-full-test', action='store_true', default=False,
                        help='Use entire test dataset for evaluation (V6 default: False = random 3-day windows)')
    parser.add_argument('--no-eval-full-test', action='store_false', dest='eval_full_test',
                        help='Use episode-length-days for evaluation instead of full test dataset')

    # Trading config (if not sampling random)
    parser.add_argument('--max-leverage', type=float, default=2.0,
                        help='Maximum leverage (1-10x)')
    parser.add_argument('--target-utilization', type=float, default=0.8,
                        help='Target capital utilization (0-1)')
    parser.add_argument('--max-positions', type=int, default=2,
                        help='Maximum concurrent positions (1-5)')

    # Reward config (V7: Balanced RL approach with APR flip detection)
    parser.add_argument('--funding-reward-scale', type=float, default=15.0,
                        help='Scale for funding P&L reward (V6: 15.0 for stronger learning signal)')
    parser.add_argument('--price-reward-scale', type=float, default=15.0,
                        help='Scale for price P&L reward (V6: 15.0 for stronger learning signal)')
    parser.add_argument('--liquidation-penalty-scale', type=float, default=10.0,
                        help='Penalty scale for approaching liquidation (10 for less conservative, 0 to disable)')
    parser.add_argument('--opportunity-cost-scale', type=float, default=0.0,
                        help='Opportunity cost penalty (DISABLED by default - can cause overtrading, use 0.0)')
    parser.add_argument('--negative-funding-exit-reward-scale', type=float, default=2.0,
                        help='Exit reward scale for positions with negative funding (V6: 2.0 for rotation)')
    # V7: New reward parameters for APR direction flip detection
    parser.add_argument('--negative-apr-penalty-scale', type=float, default=0.02,
                        help='Hourly penalty for holding positions with negative APR (V7: 0.02)')
    parser.add_argument('--apr-flip-exit-bonus-scale', type=float, default=1.5,
                        help='Bonus for exiting positions when APR direction flipped (V7: 1.5)')
    parser.add_argument('--opportunity-cost-threshold', type=float, default=50.0,
                        help='APR gap threshold for opportunity cost penalty (V7: 50%%)')
    # V6.1: Trade diversity (DISABLED by default - causes overtrading)
    parser.add_argument('--trade-diversity-bonus', type=float, default=0.0,
                        help='Bonus per completed trade (WARNING: >0.1 causes overtrading, use 0.0)')
    parser.add_argument('--inactivity-penalty-hours', type=float, default=48.0,
                        help='Hours after which inactivity penalty starts (V6.2: 48h)')
    parser.add_argument('--inactivity-penalty-scale', type=float, default=0.005,
                        help='Scale for inactivity penalty (V6.2: 0.005)')

    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['constant', 'linear', 'cosine'],
                        help='Learning rate schedule (default: constant)')
    parser.add_argument('--final-learning-rate', type=float, default=None,
                        help='Final learning rate for linear/cosine decay (default: learning_rate / 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.20,
                        help='Entropy coefficient (used if no decay schedule)')
    parser.add_argument('--initial-entropy-coef', type=float, default=0.20,
                        help='Initial entropy coefficient for decay schedule (V7: 0.20 for better exploration)')
    parser.add_argument('--final-entropy-coef', type=float, default=0.10,
                        help='Final entropy coefficient for decay schedule (V7: 0.10 to maintain exploration)')
    parser.add_argument('--entropy-decay-episodes', type=int, default=2000,
                        help='Number of episodes over which to decay entropy (V7: 2000 slower decay)')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Mini-batch size (default: 256, was 64)')

    # Training
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes to average for evaluation (V6: 10 random 3-day windows)')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (cpu, cuda, or mps)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N episodes')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Enable TensorBoard logging (default: True)')
    parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard',
                        help='Disable TensorBoard logging')
    parser.add_argument('--tensorboard-dir', type=str, default='runs',
                        help='Directory for TensorBoard logs (default: runs)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results (default: 42)')

    # Parallel environments
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of parallel environments (1=sequential, 4-8=parallel for speedup)')
    parser.add_argument('--parallel-start-method', type=str, default=None,
                        choices=[None, 'spawn', 'fork'],
                        help='Multiprocessing start method (None=default, spawn=safer, fork=faster on Unix)')

    return parser.parse_args()


def create_environment(args, data_path: str, is_test_env: bool = False, verbose: bool = True):
    """Create environment (train or test).

    Args:
        args: Command-line arguments
        data_path: Path to data CSV file
        is_test_env: If True, this is a test environment (may use full range)
        verbose: Print environment info
    """
    # Create TradingConfig
    if args.sample_random_config:
        trading_config = None  # Will sample random each episode
    else:
        trading_config = TradingConfig(
            max_leverage=args.max_leverage,
            target_utilization=args.target_utilization,
            max_positions=args.max_positions,
        )

    # Create RewardConfig (V7: APR flip detection + balanced rewards)
    reward_config = RewardConfig(
        funding_reward_scale=args.funding_reward_scale,
        price_reward_scale=args.price_reward_scale,
        liquidation_penalty_scale=args.liquidation_penalty_scale,
        opportunity_cost_scale=args.opportunity_cost_scale,
        negative_funding_exit_reward_scale=args.negative_funding_exit_reward_scale,
        # V7: New parameters for APR direction flip detection
        negative_apr_penalty_scale=args.negative_apr_penalty_scale,
        apr_flip_exit_bonus_scale=args.apr_flip_exit_bonus_scale,
        opportunity_cost_threshold=args.opportunity_cost_threshold,
        # V6.1: Trade diversity (DISABLED by default to prevent overtrading)
        trade_diversity_bonus=args.trade_diversity_bonus,
        inactivity_penalty_hours=args.inactivity_penalty_hours,
        inactivity_penalty_scale=args.inactivity_penalty_scale,
    )

    # Convert step minutes to hours for the environment
    step_hours = args.step_minutes / 60.0

    # For test environments, optionally use full range
    use_full_range = is_test_env and args.eval_full_test

    # Create environment
    env = FundingArbitrageEnv(
        data_path=data_path,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        initial_capital=args.initial_capital,
        trading_config=trading_config,
        reward_config=reward_config,
        sample_random_config=args.sample_random_config,
        episode_length_days=args.episode_length_days,
        step_hours=step_hours,
        use_full_range_episodes=use_full_range,
        verbose=verbose,
    )

    if verbose:
        if use_full_range:
            # Calculate actual data range for display
            import pandas as pd
            df = pd.read_csv(data_path)
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            data_days = (df['entry_time'].max() - df['entry_time'].min()).total_seconds() / 86400
            total_steps = int((data_days * 24 * 60) / args.step_minutes)
            print(f"Step interval: {args.step_minutes} minute(s) ({step_hours:.4f} hours)")
            print(f"🌐 FULL TEST RANGE: {data_days:.1f} days = ~{total_steps:,} steps")
        else:
            total_steps = int((args.episode_length_days * 24 * 60) / args.step_minutes)
            print(f"Step interval: {args.step_minutes} minute(s) ({step_hours:.4f} hours)")
            print(f"Episode: {args.episode_length_days} days = {total_steps} steps")

    return env


def evaluate(trainer, env, num_episodes: int = 5):
    """Evaluate the agent with detailed trading metrics."""
    # CRITICAL: Set network to eval mode for evaluation (disables dropout)
    # This ensures evaluation metrics match production inference
    was_training = trainer.network.training
    trainer.network.eval()

    # Episode-level metrics
    eval_rewards = []
    eval_lengths = []

    # Trading metrics
    total_pnls = []
    total_pnl_pcts = []
    num_trades = []
    num_winning_trades = []
    num_losing_trades = []
    avg_trade_durations = []
    max_drawdowns = []
    opportunities_seen = []
    trades_executed = []

    # Profit factor metrics
    all_winning_pnl = []
    all_losing_pnl = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        opportunities_count = 0

        while not done:
            # Count opportunities available at this step
            if hasattr(env, 'current_opportunities'):
                opportunities_count += len(env.current_opportunities)

            # Get action mask
            if hasattr(env, '_get_action_mask'):
                action_mask = env._get_action_mask()
            else:
                action_mask = None

            # Select action (deterministic)
            action, _, _ = trainer.select_action(obs, action_mask, deterministic=True)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # Collect portfolio metrics after episode
        portfolio = env.portfolio

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        total_pnls.append(portfolio.total_pnl_usd)
        total_pnl_pcts.append(portfolio.total_pnl_pct)
        max_drawdowns.append(portfolio.max_drawdown_pct)
        opportunities_seen.append(opportunities_count)

        # Trade statistics
        total_closed = len(portfolio.closed_positions)
        num_trades.append(total_closed)

        if total_closed > 0:
            winning = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd > 0)
            losing = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd <= 0)
            num_winning_trades.append(winning)
            num_losing_trades.append(losing)

            # Average trade duration in hours
            durations = [p.hours_held for p in portfolio.closed_positions]
            avg_trade_durations.append(np.mean(durations))
        else:
            num_winning_trades.append(0)
            num_losing_trades.append(0)
            avg_trade_durations.append(0.0)

        # Trades executed (total positions opened = currently open + closed)
        trades_executed.append(len(portfolio.positions) + len(portfolio.closed_positions))

        # Collect P&L for profit factor calculation
        for position in portfolio.closed_positions:
            if position.realized_pnl_usd > 0:
                all_winning_pnl.append(position.realized_pnl_usd)
            else:
                all_losing_pnl.append(abs(position.realized_pnl_usd))

    # Calculate aggregate statistics
    total_trades_sum = sum(num_trades)
    total_winning = sum(num_winning_trades)
    total_losing = sum(num_losing_trades)

    win_rate = (total_winning / total_trades_sum * 100) if total_trades_sum > 0 else 0.0

    # Calculate profit factor (total wins / total losses)
    total_wins = sum(all_winning_pnl) if all_winning_pnl else 0.0
    total_losses = sum(all_losing_pnl) if all_losing_pnl else 0.001  # Avoid division by zero
    profit_factor = total_wins / total_losses

    # Restore original training mode
    if was_training:
        trainer.network.train()

    return {
        # Episode metrics
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'mean_length': np.mean(eval_lengths),

        # P&L metrics
        'mean_pnl_usd': np.mean(total_pnls),
        'mean_pnl_pct': np.mean(total_pnl_pcts),
        'total_pnl_usd': np.sum(total_pnls),

        # Trade metrics
        'total_trades': total_trades_sum,
        'mean_trades_per_episode': np.mean(num_trades),
        'total_winning_trades': total_winning,
        'total_losing_trades': total_losing,
        'win_rate': win_rate,
        'profit_factor': profit_factor,

        # Duration metrics
        'mean_trade_duration_hours': np.mean([d for d in avg_trade_durations if d > 0]) if any(avg_trade_durations) else 0.0,

        # Opportunity metrics
        'mean_opportunities_per_episode': np.mean(opportunities_seen),
        'mean_trades_executed_per_episode': np.mean(trades_executed),

        # Risk metrics
        'mean_max_drawdown_pct': np.mean(max_drawdowns),
    }


def train(args):
    """Main training loop."""
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"🎲 Random seed: {args.seed}\n")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    writer = None
    if args.tensorboard:
        # Create unique run name with timestamp
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_path = Path(args.tensorboard_dir) / run_name
        writer = SummaryWriter(str(tensorboard_path))
        print(f"📊 TensorBoard logging enabled: {tensorboard_path}")
        print(f"   Run: tensorboard --logdir={args.tensorboard_dir}")

        # Log hyperparameters
        hparams = {
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'entropy_coef': args.entropy_coef,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'episode_length_days': args.episode_length_days,
            'step_minutes': args.step_minutes,
            'funding_reward_scale': args.funding_reward_scale,
            'price_reward_scale': args.price_reward_scale,
        }
        for key, value in hparams.items():
            writer.add_text('hyperparameters', f'{key}: {value}', 0)

    # Create train and test environments (separate data to prevent leakage)
    print("\n" + "=" * 80)
    print("CREATING TRAIN ENVIRONMENT")
    print("=" * 80)

    # Check if we should use parallel environments
    use_parallel = args.n_envs > 1

    if use_parallel:
        from models.rl.core.vec_env import ParallelEnv

        print(f"🚀 Using {args.n_envs} parallel environments for training")
        print(f"   Start method: {args.parallel_start_method or 'default'}")

        # Create environment factory functions
        def make_train_env(seed_offset):
            def _init():
                env = create_environment(args, data_path=args.train_data_path, is_test_env=False, verbose=False)
                # Set different seed for each environment
                env.seed = args.seed + seed_offset
                return env
            return _init

        env_fns = [make_train_env(i) for i in range(args.n_envs)]
        train_env = ParallelEnv(env_fns, start_method=args.parallel_start_method)

        print(f"✅ {args.n_envs} parallel environments created")
    else:
        print("Using single environment (sequential training)")
        train_env = create_environment(args, data_path=args.train_data_path, is_test_env=False, verbose=True)

    print("\n" + "=" * 80)
    print("CREATING TEST ENVIRONMENT (for evaluation)")
    print("=" * 80)
    test_env = create_environment(args, data_path=args.test_data_path, is_test_env=True, verbose=True)

    # Create network
    print("\n" + "=" * 80)
    print("CREATING NETWORK")
    print("=" * 80)
    network = ModularPPONetwork()
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {total_params:,}")

    # Create trainer
    print("\n" + "=" * 80)
    print("CREATING TRAINER")
    print("=" * 80)
    trainer = PPOTrainer(
        network=network,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        initial_entropy_coef=args.initial_entropy_coef,  # V3.1
        final_entropy_coef=args.final_entropy_coef,      # V3.1
        entropy_decay_episodes=args.entropy_decay_episodes,  # V3.1
        lr_schedule=args.lr_schedule,
        lr_schedule_total_episodes=args.num_episodes,
        final_learning_rate=args.final_learning_rate,
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer.load(args.resume_from)
        start_episode = trainer.num_updates  # Approximate

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Training for {args.num_episodes} episodes")
    print(f"Device: {args.device}")
    if use_parallel:
        print(f"Parallel mode: {args.n_envs} environments")
    print()

    best_composite_score = -float('inf')
    start_time = time.time()

    for episode in range(start_episode, args.num_episodes):
        episode_start_time = time.time()

        # Train one episode (on TRAIN data)
        # max_steps should exceed episode length to allow natural episode termination
        # 5-day episode at 5-min steps = 1440 steps, so use 2000 for safety
        max_episode_steps = int((args.episode_length_days * 24 * 60) / args.step_minutes) + 500
        if use_parallel:
            stats = trainer.train_episode_vectorized(train_env, max_steps=max_episode_steps)
        else:
            stats = trainer.train_episode(train_env, max_steps=max_episode_steps)

        episode_time = time.time() - episode_start_time

        # Calculate FPS (needed for both console and TensorBoard logging)
        elapsed_time = time.time() - start_time
        fps = trainer.total_timesteps / elapsed_time if elapsed_time > 0 else 0

        # Logging
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"  Reward: {stats['episode_reward']:8.2f}  |  Length: {stats['episode_length']:4d}  |  Mean(100): {stats['mean_reward_100']:8.2f}")
            print(f"  Loss: {stats['loss']:6.4f}  |  Policy: {stats['policy_loss']:6.4f}  |  Value: {stats['value_loss']:6.4f}  |  Entropy: {stats['entropy']:6.4f}")
            print(f"  KL: {stats['approx_kl']:6.4f}  |  Clipfrac: {stats['clipfrac']:6.4f}")
            # V3.1: Show entropy coefficient if using decay
            if 'entropy_coef' in stats:
                print(f"  Entropy Coef: {stats['entropy_coef']:6.4f}")
            # Show learning rate if using scheduling
            if 'learning_rate' in stats and args.lr_schedule != 'constant':
                print(f"  Learning Rate: {stats['learning_rate']:.2e}")
            print(f"  Time: {episode_time:.2f}s  |  FPS: {fps:.0f}  |  Total steps: {trainer.total_timesteps}")
            print()

        # TensorBoard logging (every episode for smooth curves)
        if writer is not None:
            # Episode metrics
            writer.add_scalar('train/episode_reward', stats['episode_reward'], episode)
            writer.add_scalar('train/episode_length', stats['episode_length'], episode)
            writer.add_scalar('train/mean_reward_100', stats['mean_reward_100'], episode)

            # Loss metrics
            writer.add_scalar('train/loss', stats['loss'], episode)
            writer.add_scalar('train/policy_loss', stats['policy_loss'], episode)
            writer.add_scalar('train/value_loss', stats['value_loss'], episode)
            writer.add_scalar('train/entropy', stats['entropy'], episode)
            writer.add_scalar('train/approx_kl', stats['approx_kl'], episode)
            writer.add_scalar('train/clipfrac', stats['clipfrac'], episode)

            # Entropy coefficient (if using decay)
            if 'entropy_coef' in stats:
                writer.add_scalar('train/entropy_coef', stats['entropy_coef'], episode)

            # Learning rate (if using schedule)
            if 'learning_rate' in stats:
                writer.add_scalar('train/learning_rate', stats['learning_rate'], episode)

            # Exit statistics (from new reward system)
            if 'agent_closes' in stats:
                writer.add_scalar('exits/agent_closes', stats['agent_closes'], episode)
                writer.add_scalar('exits/forced_closes', stats['forced_closes'], episode)
                writer.add_scalar('exits/stop_loss_closes', stats['stop_loss_closes'], episode)
                total_closes = stats['agent_closes'] + stats['forced_closes'] + stats['stop_loss_closes']
                if total_closes > 0:
                    agent_close_ratio = stats['agent_closes'] / total_closes
                    writer.add_scalar('exits/agent_close_ratio', agent_close_ratio, episode)

            # Performance metrics
            writer.add_scalar('perf/fps', fps, episode)
            writer.add_scalar('perf/total_timesteps', trainer.total_timesteps, episode)

        # Evaluation (on TEST data to prevent overfitting)
        if (episode + 1) % args.eval_every == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION ON TEST SET (Episode {episode + 1})")
            print(f"{'='*80}")
            eval_stats = evaluate(trainer, test_env, num_episodes=args.eval_episodes)

            # Display comprehensive metrics
            print(f"\n📊 Episode Metrics:")
            print(f"  Mean Reward:     {eval_stats['mean_reward']:8.2f} ± {eval_stats['std_reward']:.2f}")

            # Show length in appropriate unit
            if args.step_minutes == 1:
                print(f"  Mean Length:     {eval_stats['mean_length']:8.1f} minutes")
            elif args.step_minutes == 60:
                print(f"  Mean Length:     {eval_stats['mean_length']:8.1f} hours")
            else:
                print(f"  Mean Length:     {eval_stats['mean_length']:8.1f} steps ({args.step_minutes} min/step)")

            print(f"\n💰 P&L Metrics:")
            print(f"  Mean P&L (USD):  ${eval_stats['mean_pnl_usd']:8.2f}")
            print(f"  Mean P&L (%):    {eval_stats['mean_pnl_pct']:8.2f}%")
            print(f"  Total P&L:       ${eval_stats['total_pnl_usd']:8.2f}")

            print(f"\n📈 Trading Metrics:")
            print(f"  Total Trades:    {eval_stats['total_trades']:8.0f}")
            print(f"  Winning Trades:  {eval_stats['total_winning_trades']:8.0f}")
            print(f"  Losing Trades:   {eval_stats['total_losing_trades']:8.0f}")
            print(f"  Win Rate:        {eval_stats['win_rate']:8.1f}%")
            print(f"  Avg Duration:    {eval_stats['mean_trade_duration_hours']:8.1f} hours")

            print(f"\n🎯 Opportunity Metrics:")
            print(f"  Opportunities/Ep: {eval_stats['mean_opportunities_per_episode']:7.0f}")
            print(f"  Trades/Episode:   {eval_stats['mean_trades_per_episode']:7.1f}")
            print(f"  Execution Rate:   {(eval_stats['mean_trades_per_episode'] / eval_stats['mean_opportunities_per_episode'] * 100) if eval_stats['mean_opportunities_per_episode'] > 0 else 0:.1f}%")

            print(f"\n⚠️  Risk Metrics:")
            print(f"  Max Drawdown:    {eval_stats['mean_max_drawdown_pct']:8.2f}%")

            print(f"\n📊 Profitability Metrics:")
            print(f"  Profit Factor:   {eval_stats['profit_factor']:8.2f}")

            # Calculate composite score for model selection (IMPROVED VERSION)
            # Weights: 50% P&L, 30% Profit Factor, 20% Low Drawdown

            # 1. P&L Score: Bounded using tanh to prevent domination
            #    tanh(x/5) maps: 5% → 0.76, 10% → 0.96, 20% → 0.999
            pnl_score = np.tanh(eval_stats['mean_pnl_pct'] / 5.0)

            # 2. Profit Factor Score: Normalize (2.0 = 1.0, capped at 1.0)
            #    Profit Factor > 2.0 is excellent, > 1.5 is good
            profit_factor_score = min(eval_stats['profit_factor'] / 2.0, 1.0)

            # 3. Drawdown Score: Lower is better, floored at 0.0
            drawdown_score = max(0.0, 1.0 - (eval_stats['mean_max_drawdown_pct'] / 100.0))

            composite_score = (
                0.50 * pnl_score +
                0.30 * profit_factor_score +
                0.20 * drawdown_score
            )

            print(f"\n🎯 Composite Score: {composite_score:.4f}")
            print(f"   (P&L: {pnl_score:.3f} | ProfitFactor: {profit_factor_score:.3f} | Drawdown: {drawdown_score:.3f})")
            print()

            # TensorBoard evaluation logging
            if writer is not None:
                writer.add_scalar('eval/mean_reward', eval_stats['mean_reward'], episode)
                writer.add_scalar('eval/mean_pnl_pct', eval_stats['mean_pnl_pct'], episode)
                writer.add_scalar('eval/mean_pnl_usd', eval_stats['mean_pnl_usd'], episode)
                writer.add_scalar('eval/total_trades', eval_stats['total_trades'], episode)
                writer.add_scalar('eval/win_rate', eval_stats['win_rate'], episode)
                writer.add_scalar('eval/profit_factor', eval_stats['profit_factor'], episode)
                writer.add_scalar('eval/mean_trade_duration_hours', eval_stats['mean_trade_duration_hours'], episode)
                writer.add_scalar('eval/max_drawdown_pct', eval_stats['mean_max_drawdown_pct'], episode)
                writer.add_scalar('eval/composite_score', composite_score, episode)

                # Score components
                writer.add_scalar('eval/score_pnl', pnl_score, episode)
                writer.add_scalar('eval/score_profit_factor', profit_factor_score, episode)
                writer.add_scalar('eval/score_drawdown', drawdown_score, episode)

            # Save best model based on composite score
            if composite_score > best_composite_score:
                prev_best = best_composite_score
                best_composite_score = composite_score
                best_path = checkpoint_dir / 'best_model.pt'
                trainer.save(str(best_path))
                print(f"✅ New best model saved: {best_path}")
                if prev_best > -float('inf'):
                    print(f"   Score improved: {prev_best:.4f} → {composite_score:.4f} (+{composite_score - prev_best:.4f})")
                print()

        # Save checkpoint
        if (episode + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode + 1}.pt'
            trainer.save(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}\n")

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    trainer.save(str(final_path))

    # Cleanup parallel environments
    if use_parallel:
        print("\n🧹 Closing parallel environments...")
        train_env.close()

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("📊 TensorBoard logs saved")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total timesteps: {trainer.total_timesteps:,}")
    print(f"Best composite score: {best_composite_score:.4f}")
    print(f"Final model saved: {final_path}")
    if use_parallel:
        speedup = args.n_envs * 0.7  # Approximate speedup (70% efficiency)
        print(f"Parallel speedup: ~{speedup:.1f}x with {args.n_envs} environments")


if __name__ == '__main__':
    args = parse_args()
    train(args)
