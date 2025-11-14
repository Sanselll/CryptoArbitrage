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

    # Curriculum
    parser.add_argument('--phase1-end', type=int, default=500,
                        help='Episode when Phase 1 ends')
    parser.add_argument('--phase2-end', type=int, default=1500,
                        help='Episode when Phase 2 ends')
    parser.add_argument('--num-episodes', type=int, default=3000,
                        help='Total training episodes')

    # Reward config (Simplified RL-v2 approach)
    parser.add_argument('--pnl-reward-scale', type=float, default=3.0)
    parser.add_argument('--entry-penalty-scale', type=float, default=1.0)
    parser.add_argument('--stop-loss-penalty', type=float, default=-1.0)

    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)

    # Training
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/curriculum')
    parser.add_argument('--log-interval', type=int, default=10)

    return parser.parse_args()


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

    # Create reward config (Simplified RL-v2 approach)
    reward_config = RewardConfig(
        pnl_reward_scale=args.pnl_reward_scale,
        entry_penalty_scale=args.entry_penalty_scale,
        stop_loss_penalty=args.stop_loss_penalty,
    )

    print("Reward Configuration:")
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
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()

    # Training loop with curriculum
    best_composite_score = -float('inf')
    start_time = time.time()

    print("=" * 80)
    print("STARTING CURRICULUM TRAINING")
    print("=" * 80)
    print()

    for episode in range(args.num_episodes):
        episode_start = time.time()

        # Get curriculum phase config
        phase = curriculum.get_current_phase(episode)
        trading_config = curriculum.get_config(episode)
        episode_length_days = curriculum.get_episode_length_days(episode)

        # Create environment for this episode (config changes each episode in Phase 2 & 3)
        env = FundingArbitrageEnv(
            data_path=args.train_data_path,
            initial_capital=args.initial_capital,
            trading_config=trading_config,
            reward_config=reward_config,
            episode_length_days=episode_length_days,
            price_history_path=args.price_history_path,
            feature_scaler_path=args.feature_scaler_path,
            verbose=False,
        )

        # Train one episode
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

        # Periodic evaluation and checkpointing done in parallel script
        # (Simplified for readability)

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
