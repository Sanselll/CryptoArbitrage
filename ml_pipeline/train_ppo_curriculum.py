"""
Train PPO with Curriculum Learning

Implements 3-phase curriculum learning as per IMPLEMENTATION_PLAN.md:
- Phase 1 (Episodes 0-500): Simple fixed config, 72h episodes
- Phase 2 (Episodes 500-1500): Moderate sampled configs, 120h episodes
- Phase 3 (Episodes 1500-3000): Full config range, 168h episodes

Usage:
    python train_ppo_curriculum.py --num-episodes 3000
"""

import argparse
from pathlib import Path
import time
from datetime import datetime
import numpy as np

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.curriculum import CurriculumScheduler
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO with Curriculum Learning')

    # Data
    parser.add_argument('--train-data-path', type=str, default='data/rl_train.csv')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')
    parser.add_argument('--feature-scaler-path', type=str, default='trained_models/rl/feature_scaler_v2.pkl',
                        help='Path to fitted StandardScaler pickle (default: trained_models/rl/feature_scaler_v2.pkl)')

    # Checkpoint loading
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/v3/best_model.pt)')
    parser.add_argument('--start-episode', type=int, default=0,
                        help='Starting episode number (used when resuming from checkpoint)')

    # Curriculum
    parser.add_argument('--phase1-end', type=int, default=500,
                        help='Episode when Phase 1 ends')
    parser.add_argument('--phase2-end', type=int, default=1500,
                        help='Episode when Phase 2 ends')
    parser.add_argument('--num-episodes', type=int, default=3000,
                        help='Total training episodes')

    # Reward config (V3 Pure RL-v2 - matching train_ppo.py)
    parser.add_argument('--funding-reward-scale', type=float, default=1.0)
    parser.add_argument('--price-reward-scale', type=float, default=1.0)
    parser.add_argument('--liquidation-penalty-scale', type=float, default=10.0)
    parser.add_argument('--opportunity-cost-scale', type=float, default=0.0)  # Disabled in V3
    parser.add_argument('--negative-funding-exit-reward-scale', type=float, default=0.0)  # Disabled in V3

    # PPO hyperparameters (V3 - matching train_ppo.py)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.05)
    parser.add_argument('--initial-entropy-coef', type=float, default=None)
    parser.add_argument('--final-entropy-coef', type=float, default=None)
    parser.add_argument('--entropy-decay-episodes', type=int, default=2000)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)

    # Training
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/curriculum')
    parser.add_argument('--log-interval', type=int, default=10)

    return parser.parse_args()


def evaluate(trainer, env, num_episodes: int = 1):
    """Evaluate the agent with detailed trading metrics."""
    # Set network to eval mode
    was_training = trainer.network.training
    trainer.network.eval()

    # Episode-level metrics
    eval_rewards = []
    eval_lengths = []
    total_pnls = []
    total_pnl_pcts = []
    num_trades = []
    num_winning_trades = []
    num_losing_trades = []
    avg_trade_durations = []
    max_drawdowns = []
    opportunities_seen = []
    all_winning_pnl = []
    all_losing_pnl = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        opportunities_count = 0

        while not done:
            if hasattr(env, 'current_opportunities'):
                opportunities_count += len(env.current_opportunities)

            action_mask = env._get_action_mask() if hasattr(env, '_get_action_mask') else None
            action, _, _ = trainer.select_action(obs, action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # Collect portfolio metrics
        portfolio = env.portfolio
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        total_pnls.append(portfolio.total_pnl_usd)
        total_pnl_pcts.append(portfolio.total_pnl_pct)
        max_drawdowns.append(portfolio.max_drawdown_pct)
        opportunities_seen.append(opportunities_count)

        total_closed = len(portfolio.closed_positions)
        num_trades.append(total_closed)

        if total_closed > 0:
            winning = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd > 0)
            losing = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd <= 0)
            num_winning_trades.append(winning)
            num_losing_trades.append(losing)

            durations = [p.hours_held for p in portfolio.closed_positions]
            avg_trade_durations.append(np.mean(durations))

            for position in portfolio.closed_positions:
                if position.realized_pnl_usd > 0:
                    all_winning_pnl.append(position.realized_pnl_usd)
                else:
                    all_losing_pnl.append(abs(position.realized_pnl_usd))
        else:
            num_winning_trades.append(0)
            num_losing_trades.append(0)
            avg_trade_durations.append(0.0)

    # Calculate stats
    total_trades_sum = sum(num_trades)
    total_winning = sum(num_winning_trades)
    total_losing = sum(num_losing_trades)
    win_rate = (total_winning / total_trades_sum * 100) if total_trades_sum > 0 else 0.0

    total_wins = sum(all_winning_pnl) if all_winning_pnl else 0.0
    total_losses = sum(all_losing_pnl) if all_losing_pnl else 0.001
    profit_factor = total_wins / total_losses

    # Restore training mode
    if was_training:
        trainer.network.train()

    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'mean_length': np.mean(eval_lengths),
        'mean_pnl_usd': np.mean(total_pnls),
        'mean_pnl_pct': np.mean(total_pnl_pcts),
        'total_pnl_usd': np.sum(total_pnls),
        'total_trades': total_trades_sum,
        'mean_trades_per_episode': np.mean(num_trades),
        'total_winning_trades': total_winning,
        'total_losing_trades': total_losing,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'mean_trade_duration_hours': np.mean([d for d in avg_trade_durations if d > 0]) if any(avg_trade_durations) else 0.0,
        'mean_opportunities_per_episode': np.mean(opportunities_seen),
        'mean_max_drawdown_pct': np.mean(max_drawdowns),
    }


def main():
    args = parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create curriculum scheduler
    curriculum = CurriculumScheduler(
        phase1_end=args.phase1_end,
        phase2_end=args.phase2_end,
        phase3_end=args.num_episodes
    )

    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING - PPO TRAINING")
    print("=" * 80)
    print(curriculum)
    print()

    # Create reward config (V3 Pure RL-v2)
    reward_config = RewardConfig(
        funding_reward_scale=args.funding_reward_scale,
        price_reward_scale=args.price_reward_scale,
        liquidation_penalty_scale=args.liquidation_penalty_scale,
        opportunity_cost_scale=args.opportunity_cost_scale,
        negative_funding_exit_reward_scale=args.negative_funding_exit_reward_scale,
    )

    print("Reward Configuration (V3):")
    print(f"  {reward_config}")
    print()

    # Create network and trainer (these persist across all phases)
    network = ModularPPONetwork()
    trainer = PPOTrainer(
        network=network,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        initial_entropy_coef=args.initial_entropy_coef,
        final_entropy_coef=args.final_entropy_coef,
        entropy_decay_episodes=args.entropy_decay_episodes,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        trainer.load(args.checkpoint)
        print(f"✅ Checkpoint loaded! Resuming from episode {args.start_episode}")
        print()

    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()

    # Create TRAINING environment ONCE (loads data only once)
    print("=" * 80)
    print("INITIALIZING TRAINING ENVIRONMENT")
    print("=" * 80)

    # Start with Phase 1 config
    initial_config = curriculum.get_config(0)
    initial_episode_length = curriculum.get_episode_length_days(0)

    env = FundingArbitrageEnv(
        data_path=args.train_data_path,
        initial_capital=args.initial_capital,
        trading_config=initial_config,
        reward_config=reward_config,
        episode_length_days=initial_episode_length,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        verbose=False,
    )
    print("✅ Training environment initialized (data cached in memory)")
    print()

    # Create TEST environment for evaluation
    print("=" * 80)
    print("INITIALIZING TEST ENVIRONMENT")
    print("=" * 80)

    test_env = FundingArbitrageEnv(
        data_path=args.test_data_path,
        initial_capital=args.initial_capital,
        trading_config=initial_config,
        reward_config=reward_config,
        episode_length_days=initial_episode_length,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        verbose=False,
    )
    print("✅ Test environment initialized (data cached in memory)")
    print()

    # Training loop with curriculum
    best_composite_score = -float('inf')
    start_time = time.time()

    print("=" * 80)
    print("STARTING CURRICULUM TRAINING")
    print("=" * 80)
    if args.start_episode > 0:
        print(f"Resuming from episode {args.start_episode}")
    print()

    for episode in range(args.start_episode, args.num_episodes):
        episode_start = time.time()

        # Get curriculum phase config
        phase = curriculum.get_current_phase(episode)
        trading_config = curriculum.get_config(episode)
        episode_length_days = curriculum.get_episode_length_days(episode)

        # Update environment config for this episode (no reload, just config update)
        env.current_config = trading_config
        env.episode_length_hours = episode_length_days * 24

        # Train one episode (resets internally)
        stats = trainer.train_episode(env, max_steps=1000)
        episode_time = time.time() - episode_start

        # Log progress
        if (episode + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = trainer.total_timesteps / elapsed

            phase_name, progress = curriculum.get_phase_progress(episode)

            print(f"Episode {episode + 1}/{args.num_episodes} | Phase: {phase_name} ({progress:.0f}%)")
            print(f"  Reward: {stats['episode_reward']:8.2f}  |  Mean(100): {stats['mean_reward_100']:8.2f}")
            print(f"  Config: {trading_config.max_leverage:.1f}x, {trading_config.target_utilization:.0%} util, {trading_config.max_positions} pos")
            print(f"  Time: {episode_time:.1f}s  |  FPS: {fps:.0f}")
            print()

        # Evaluation on TEST data
        if (episode + 1) % args.eval_every == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION ON TEST SET (Episode {episode + 1})")
            print(f"{'='*80}")

            # Update test environment config to match current curriculum stage
            test_env.current_config = trading_config
            test_env.episode_length_hours = episode_length_days * 24

            eval_stats = evaluate(trainer, test_env, num_episodes=1)

            # Display comprehensive metrics
            print(f"\n📊 Episode Metrics:")
            print(f"  Mean Reward:     {eval_stats['mean_reward']:8.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Length:     {eval_stats['mean_length']:8.1f} steps")

            print(f"\n💰 P&L Metrics:")
            print(f"  Mean P&L (USD):  ${eval_stats['mean_pnl_usd']:8.2f}")
            print(f"  Mean P&L (%):    {eval_stats['mean_pnl_pct']:8.2f}%")

            print(f"\n📈 Trading Metrics:")
            print(f"  Total Trades:    {eval_stats['total_trades']:8.0f}")
            print(f"  Winning Trades:  {eval_stats['total_winning_trades']:8.0f}")
            print(f"  Losing Trades:   {eval_stats['total_losing_trades']:8.0f}")
            print(f"  Win Rate:        {eval_stats['win_rate']:8.1f}%")
            print(f"  Avg Duration:    {eval_stats['mean_trade_duration_hours']:8.1f} hours")

            print(f"\n🎯 Opportunity Metrics:")
            print(f"  Opportunities/Ep: {eval_stats['mean_opportunities_per_episode']:7.0f}")
            print(f"  Trades/Episode:   {eval_stats['mean_trades_per_episode']:7.1f}")

            print(f"\n⚠️  Risk Metrics:")
            print(f"  Max Drawdown:    {eval_stats['mean_max_drawdown_pct']:8.2f}%")
            print(f"  Profit Factor:   {eval_stats['profit_factor']:8.2f}")

            # Calculate composite score
            pnl_score = np.tanh(eval_stats['mean_pnl_pct'] / 5.0)
            profit_factor_score = np.clip(eval_stats['profit_factor'] / 3.0, 0.0, 1.0)
            drawdown_score = np.clip(1.0 + (eval_stats['mean_max_drawdown_pct'] / 10.0), 0.0, 1.0)
            composite_score = 0.5 * pnl_score + 0.3 * profit_factor_score + 0.2 * drawdown_score

            print(f"\n🏆 Composite Score: {composite_score:.4f}")

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_model_path = checkpoint_dir / 'best_model.pt'
                trainer.save(str(best_model_path))
                print(f"   ⭐ NEW BEST MODEL! Saved to {best_model_path}")

            print(f"{'='*80}\n")

        # Save checkpoint every save_every episodes
        if (episode + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode + 1}.pt'
            trainer.save(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}\n")

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    trainer.save(str(final_path))

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total timesteps: {trainer.total_timesteps:,}")
    print(f"Final model: {final_path}")


if __name__ == '__main__':
    main()
