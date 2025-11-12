"""
Population-Based Training (PBT) for PPO

Implements PBT as per IMPLEMENTATION_PLAN.md (lines 771-790):
- Population: 8 agents
- Perturbation interval: Every 100 episodes
- Exploit: Bottom 25% copies from top 25%
- Explore: Perturb hyperparameters ±20%

Tunes both PPO hyperparameters AND reward parameters simultaneously.

Usage:
    python train_ppo_pbt.py --population 8 --episodes-per-agent 1000
"""

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import copy
import multiprocessing as mp
from functools import partial
import torch

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.core.curriculum import CurriculumScheduler
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


def evaluate(trainer, env, num_episodes: int = 1):
    """Evaluate the agent with detailed trading metrics (same as train_ppo.py)."""
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


@dataclass
class PBTHyperparameters:
    """Hyperparameters that PBT will tune."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5

    # Reward hyperparameters (Balanced RL-v2 approach)
    funding_reward_scale: float = 1.0  # Funding P&L weight
    price_reward_scale: float = 1.0    # Price P&L weight (equal balance)
    liquidation_penalty_scale: float = 10.0  # Safety penalty (10 for less conservative, 0 to disable)

    def perturb(self, factor: float = 0.2) -> 'PBTHyperparameters':
        """
        Perturb hyperparameters for exploration.

        Args:
            factor: Perturbation factor (default: 0.2 = ±20%)

        Returns:
            New PBTHyperparameters with perturbed values
        """
        multiplier = np.random.choice([1.0 - factor, 1.0 + factor])

        return PBTHyperparameters(
            learning_rate=np.clip(self.learning_rate * multiplier, 1e-5, 3e-3),
            gamma=np.clip(self.gamma * multiplier, 0.95, 0.995),
            gae_lambda=np.clip(self.gae_lambda * multiplier, 0.90, 0.98),
            clip_range=np.clip(self.clip_range * multiplier, 0.15, 0.30),
            entropy_coef=np.clip(self.entropy_coef * multiplier, 0.001, 0.02),
            vf_coef=np.clip(self.vf_coef * multiplier, 0.5, 1.0),
            # Reward hyperparameters (Balanced RL-v2)
            funding_reward_scale=np.clip(self.funding_reward_scale * multiplier, 0.5, 2.0),
            price_reward_scale=np.clip(self.price_reward_scale * multiplier, 0.5, 2.0),
            liquidation_penalty_scale=np.clip(self.liquidation_penalty_scale * multiplier, 0.0, 20.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'PBTHyperparameters':
        return cls(**d)


@dataclass
class AgentState:
    """State of a single PBT agent."""
    agent_id: int
    hyperparams: PBTHyperparameters
    trainer: PPOTrainer
    episodes_completed: int = 0
    total_timesteps: int = 0
    mean_reward_100: float = 0.0
    composite_score: float = -float('inf')  # Composite score from test evaluation
    recent_rewards: List[float] = None

    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = []


def train_agent_parallel(agent_id: int,
                         hyperparams_dict: dict,
                         network_state_dict: dict,
                         recent_rewards: List[float],
                         episodes_completed: int,
                         num_episodes: int,
                         data_path: str,
                         initial_capital: float,
                         use_curriculum: bool,
                         curriculum_state: dict,
                         price_history_path: str,
                         feature_scaler_path: str,
                         device: str) -> Tuple[int, List[float], int, int, dict]:
    """
    Train a single agent for N episodes (parallel-safe function).

    This function is called in parallel by multiprocessing.Pool.

    Returns:
        Tuple of (agent_id, recent_rewards, episodes_completed, total_timesteps, network_state_dict)
    """
    # Recreate hyperparameters
    hyperparams = PBTHyperparameters.from_dict(hyperparams_dict)

    # Create network and load state
    network = ModularPPONetwork()
    network.load_state_dict(network_state_dict)

    # Create trainer
    trainer = PPOTrainer(
        network=network,
        learning_rate=hyperparams.learning_rate,
        gamma=hyperparams.gamma,
        gae_lambda=hyperparams.gae_lambda,
        clip_range=hyperparams.clip_range,
        value_coef=hyperparams.vf_coef,
        entropy_coef=hyperparams.entropy_coef,
        n_epochs=4,
        batch_size=64,
        device=device,
    )

    # Recreate curriculum if needed
    curriculum = None
    if use_curriculum:
        curriculum = CurriculumScheduler()
        # Restore curriculum state if provided
        if curriculum_state:
            curriculum.__dict__.update(curriculum_state)

    # Train for num_episodes
    for ep in range(num_episodes):
        episode = episodes_completed

        # Progress logging every 10 episodes
        if ep % 10 == 0 and ep > 0:
            import sys
            print(f"    Agent {agent_id}: {ep}/{num_episodes} episodes, Mean(10)={np.mean(recent_rewards[-10:]):.2f}")
            sys.stdout.flush()

        # Get config from curriculum
        if use_curriculum and curriculum:
            trading_config = curriculum.get_config(episode)
            episode_length_days = curriculum.get_episode_length_days(episode)
            step_hours = 5.0 / 60.0  # 5 minutes (consistent with non-curriculum)
        else:
            trading_config = TradingConfig.sample_random()
            episode_length_days = 7
            step_hours = 5.0 / 60.0  # 5 minutes

        # Create reward config from agent's hyperparameters (Pure RL-v2)
        reward_config = RewardConfig(
            funding_reward_scale=hyperparams.funding_reward_scale,
            price_reward_scale=hyperparams.price_reward_scale,
            liquidation_penalty_scale=hyperparams.liquidation_penalty_scale,
        )

        # Create environment
        env = FundingArbitrageEnv(
            data_path=data_path,
            initial_capital=initial_capital,
            trading_config=trading_config,
            reward_config=reward_config,
            episode_length_days=episode_length_days,
            step_hours=step_hours,
            price_history_path=price_history_path,
            feature_scaler_path=feature_scaler_path,
            verbose=False,
        )

        # Train episode (match regular PPO training max_steps)
        stats = trainer.train_episode(env, max_steps=1000)

        # Update state
        episodes_completed += 1
        recent_rewards.append(stats['episode_reward'])

        # Keep only last 100 rewards
        if len(recent_rewards) > 100:
            recent_rewards = recent_rewards[-100:]

    # Return updated state
    return (
        agent_id,
        recent_rewards,
        episodes_completed,
        trainer.total_timesteps,
        network.state_dict()
    )


class PBTManager:
    """Manages population-based training."""

    def __init__(self,
                 population_size: int,
                 data_path: str,
                 test_data_path: str,
                 initial_capital: float,
                 perturbation_interval: int,
                 device: str = 'cpu',
                 checkpoint_dir: str = 'checkpoints/pbt',
                 use_curriculum: bool = True,
                 price_history_path: str = None,
                 feature_scaler_path: str = 'trained_models/rl/feature_scaler.pkl',
                 num_processes: int = None):
        """
        Initialize PBT manager.

        Args:
            population_size: Number of agents (default: 8)
            data_path: Path to training data
            test_data_path: Path to test data
            initial_capital: Initial capital
            perturbation_interval: Episodes between perturbations (default: 100)
            device: Training device
            checkpoint_dir: Directory for checkpoints
            use_curriculum: Whether to use curriculum learning
            price_history_path: Path to price history directory
            feature_scaler_path: Path to fitted StandardScaler pickle
            num_processes: Number of parallel processes (default: min(population_size, cpu_count))
        """
        self.population_size = population_size
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.initial_capital = initial_capital
        self.perturbation_interval = perturbation_interval
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.price_history_path = price_history_path
        self.feature_scaler_path = feature_scaler_path

        # Parallel training
        if num_processes is None:
            num_processes = min(population_size, mp.cpu_count())
        self.num_processes = num_processes

        # Curriculum learning
        self.use_curriculum = use_curriculum
        if use_curriculum:
            self.curriculum = CurriculumScheduler()

        # Create test environment for evaluation
        self.test_env = self._create_environment(test_data_path, verbose=False)

        # Initialize population
        self.agents: List[AgentState] = []
        self._initialize_population()

        # Tracking
        self.generation = 0
        self.best_agent_id = 0
        self.best_score = -float('inf')

    def _create_environment(self, data_path: str, verbose: bool = False):
        """Create an environment (train or test)."""
        # Use default reward config (Pure RL-v2 approach)
        reward_config = RewardConfig(
            funding_reward_scale=5.0,
            price_reward_scale=1.0,
            liquidation_penalty_scale=1000.0,
        )

        # Use default trading config
        trading_config = TradingConfig(
            max_leverage=1.0,
            target_utilization=0.9,
            max_positions=5,
        )

        env = FundingArbitrageEnv(
            data_path=data_path,
            price_history_path=self.price_history_path,
            feature_scaler_path=self.feature_scaler_path,
            initial_capital=self.initial_capital,
            trading_config=trading_config,
            reward_config=reward_config,
            episode_length_days=7,
            step_hours=5.0 / 60.0,  # 5 minutes
            verbose=verbose,
        )

        return env

    def _initialize_population(self):
        """Initialize population with diverse hyperparameters."""
        print("=" * 80)
        print("INITIALIZING PBT POPULATION")
        print("=" * 80)
        print(f"Population size: {self.population_size}")
        print()

        for i in range(self.population_size):
            # Sample random hyperparameters for diversity
            if i == 0:
                # First agent uses default hyperparameters
                hyperparams = PBTHyperparameters()
            else:
                # Others use perturbed defaults
                base = PBTHyperparameters()
                hyperparams = base.perturb(factor=0.3)  # Larger initial diversity

            # Create network
            network = ModularPPONetwork()

            # Create trainer with agent's hyperparameters
            trainer = PPOTrainer(
                network=network,
                learning_rate=hyperparams.learning_rate,
                gamma=hyperparams.gamma,
                gae_lambda=hyperparams.gae_lambda,
                clip_range=hyperparams.clip_range,
                value_coef=hyperparams.vf_coef,
                entropy_coef=hyperparams.entropy_coef,
                n_epochs=4,
                batch_size=64,
                device=self.device,
            )

            agent = AgentState(
                agent_id=i,
                hyperparams=hyperparams,
                trainer=trainer,
            )

            self.agents.append(agent)

            print(f"Agent {i}: LR={hyperparams.learning_rate:.2e}, "
                  f"Funding={hyperparams.funding_reward_scale:.1f}, "
                  f"Price={hyperparams.price_reward_scale:.1f}, "
                  f"Liq={hyperparams.liquidation_penalty_scale:.0f}")

        print()

    def train_agent(self, agent: AgentState, num_episodes: int):
        """Train a single agent for N episodes."""
        for _ in range(num_episodes):
            episode = agent.episodes_completed

            # Get config from curriculum
            if self.use_curriculum:
                trading_config = self.curriculum.get_config(episode)
                episode_length_days = self.curriculum.get_episode_length_days(episode)
            else:
                trading_config = TradingConfig.sample_random()
                episode_length_days = 7

            # Create reward config from agent's hyperparameters (Pure RL-v2)
            reward_config = RewardConfig(
                funding_reward_scale=agent.hyperparams.funding_reward_scale,
                price_reward_scale=agent.hyperparams.price_reward_scale,
                liquidation_penalty_scale=agent.hyperparams.liquidation_penalty_scale,
            )

            # Create environment
            env = FundingArbitrageEnv(
                data_path=self.data_path,
                initial_capital=self.initial_capital,
                trading_config=trading_config,
                reward_config=reward_config,
                episode_length_days=episode_length_days,
                step_hours=5.0 / 60.0,  # 5 minutes (consistent with parallel training)
                price_history_path=self.price_history_path,
                feature_scaler_path=self.feature_scaler_path,  # For 19 features per opportunity
                verbose=False,
            )

            # Train episode
            stats = agent.trainer.train_episode(env, max_steps=1000)

            # Update agent state
            agent.episodes_completed += 1
            agent.total_timesteps = agent.trainer.total_timesteps
            agent.recent_rewards.append(stats['episode_reward'])

            # Keep only last 100 rewards
            if len(agent.recent_rewards) > 100:
                agent.recent_rewards = agent.recent_rewards[-100:]

            agent.mean_reward_100 = np.mean(agent.recent_rewards)

    def train_agents_parallel(self, num_episodes: int, num_processes: int = None):
        """
        Train all agents in parallel using multiprocessing.

        Args:
            num_episodes: Number of episodes to train each agent
            num_processes: Number of processes (default: population_size, max: CPU count)
        """
        if num_processes is None:
            num_processes = min(self.population_size, mp.cpu_count())

        # Get curriculum state if using curriculum
        curriculum_state = None
        if self.use_curriculum:
            curriculum_state = self.curriculum.__dict__.copy()

        # Prepare arguments for each agent
        training_args = []
        for agent in self.agents:
            args = (
                agent.agent_id,
                agent.hyperparams.to_dict(),
                agent.trainer.network.state_dict(),
                agent.recent_rewards.copy(),
                agent.episodes_completed,
                num_episodes,
                self.data_path,
                self.initial_capital,
                self.use_curriculum,
                curriculum_state,
                self.price_history_path,
                self.feature_scaler_path,
                self.device,
            )
            training_args.append(args)

        # Train agents in parallel
        print(f"Training {self.population_size} agents in parallel using {num_processes} processes...")
        print(f"Each agent will train for {num_episodes} episodes (updates every 10 episodes)")
        import sys
        sys.stdout.flush()

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(train_agent_parallel, training_args)

        print(f"✅ All agents completed {num_episodes} episodes")

        # Update agent states from results
        for (agent_id, recent_rewards, episodes_completed, total_timesteps, network_state_dict) in results:
            agent = self.agents[agent_id]
            agent.recent_rewards = recent_rewards
            agent.episodes_completed = episodes_completed
            agent.total_timesteps = total_timesteps
            agent.mean_reward_100 = np.mean(recent_rewards) if recent_rewards else 0.0

            # Update network weights
            agent.trainer.network.load_state_dict(network_state_dict)

    def exploit_and_explore(self):
        """PBT exploit and explore step."""
        self.generation += 1

        print("\n" + "=" * 80)
        print(f"PBT GENERATION {self.generation}: EXPLOIT & EXPLORE")
        print("=" * 80)

        # Evaluate all agents on test set to get composite scores
        print("\nEvaluating agents on test set...")
        for agent in self.agents:
            eval_stats = evaluate(agent.trainer, self.test_env, num_episodes=1)

            # Calculate composite score (same as train_ppo.py)
            pnl_score = np.tanh(eval_stats['mean_pnl_pct'] / 5.0)
            profit_factor_score = min(eval_stats['profit_factor'] / 2.0, 1.0)
            drawdown_score = max(0.0, 1.0 - (eval_stats['mean_max_drawdown_pct'] / 100.0))

            agent.composite_score = (
                0.50 * pnl_score +
                0.30 * profit_factor_score +
                0.20 * drawdown_score
            )

        # Rank agents by composite score (test performance)
        agents_sorted = sorted(self.agents, key=lambda a: a.composite_score, reverse=True)

        # Display rankings
        print("\nAgent Rankings (by Composite Score on Test Set):")
        for rank, agent in enumerate(agents_sorted):
            print(f"  {rank+1}. Agent {agent.agent_id}: "
                  f"Score={agent.composite_score:.4f}, "
                  f"TrainReward={agent.mean_reward_100:7.2f}, "
                  f"Episodes={agent.episodes_completed}")
        print()

        # Identify top and bottom quartiles (at least 1 agent each for small populations)
        quartile_size = max(1, self.population_size // 4)
        top_agents = agents_sorted[:quartile_size]  # Top 25%
        bottom_agents = agents_sorted[-quartile_size:]  # Bottom 25%

        # Exploit: Bottom agents copy from top agents
        print("Exploit phase:")
        for bottom_agent in bottom_agents:
            # Select random top agent to copy from
            top_agent = np.random.choice(top_agents)

            print(f"  Agent {bottom_agent.agent_id} copies from Agent {top_agent.agent_id}")

            # Copy network weights
            bottom_agent.trainer.network.load_state_dict(
                top_agent.trainer.network.state_dict()
            )
            bottom_agent.trainer.optimizer.load_state_dict(
                top_agent.trainer.optimizer.state_dict()
            )

            # Copy hyperparameters
            bottom_agent.hyperparams = copy.deepcopy(top_agent.hyperparams)

        print()

        # Explore: Perturb hyperparameters of copied agents
        print("Explore phase:")
        for bottom_agent in bottom_agents:
            old_hyperparams = bottom_agent.hyperparams
            new_hyperparams = old_hyperparams.perturb(factor=0.2)
            bottom_agent.hyperparams = new_hyperparams

            # Update trainer with new hyperparameters
            for param_group in bottom_agent.trainer.optimizer.param_groups:
                param_group['lr'] = new_hyperparams.learning_rate

            print(f"  Agent {bottom_agent.agent_id}: "
                  f"LR {old_hyperparams.learning_rate:.2e}→{new_hyperparams.learning_rate:.2e}, "
                  f"Funding {old_hyperparams.funding_reward_scale:.1f}→{new_hyperparams.funding_reward_scale:.1f}")

        print()

        # Update best agent (based on composite score)
        best_agent = agents_sorted[0]
        if best_agent.composite_score > self.best_score:
            self.best_score = best_agent.composite_score
            self.best_agent_id = best_agent.agent_id

            # Save best agent
            best_path = self.checkpoint_dir / 'best_agent.pt'
            best_agent.trainer.save(str(best_path))

            print(f"✅ New best agent: Agent {best_agent.agent_id}")
            print(f"   Composite Score: {best_agent.composite_score:.4f}")
            print(f"   Train Reward: {best_agent.mean_reward_100:.2f}")
            print(f"   Saved to: {best_path}\n")

    def train(self, episodes_per_agent: int):
        """Run PBT training."""
        print("\n" + "=" * 80)
        print("STARTING PBT TRAINING")
        print("=" * 80)
        print(f"Episodes per agent: {episodes_per_agent}")
        print(f"Perturbation interval: {self.perturbation_interval}")
        print(f"Total generations: {episodes_per_agent // self.perturbation_interval}")
        print()

        start_time = time.time()

        total_rounds = episodes_per_agent // self.perturbation_interval

        for round_num in range(total_rounds):
            round_start = time.time()

            print(f"\n{'='*80}")
            print(f"ROUND {round_num + 1}/{total_rounds}")
            print(f"{'='*80}\n")

            # Train all agents in parallel for perturbation_interval episodes
            self.train_agents_parallel(self.perturbation_interval, num_processes=self.num_processes)

            # Print results
            for agent in self.agents:
                print(f"  Agent {agent.agent_id}: Episodes={agent.episodes_completed}, "
                      f"Mean(100)={agent.mean_reward_100:.2f}")

            round_time = time.time() - round_start
            print(f"Round {round_num + 1} completed in {round_time/60:.1f} minutes")

            # Exploit and explore
            if round_num < total_rounds - 1:  # Don't perturb on last round
                self.exploit_and_explore()

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("PBT TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best agent: Agent {self.best_agent_id}")
        print(f"Best score: {self.best_score:.2f}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description='Population-Based Training for PPO')

    # PBT parameters
    parser.add_argument('--population', type=int, default=8,
                        help='Population size (default: 8)')
    parser.add_argument('--episodes-per-agent', type=int, default=1000,
                        help='Episodes each agent trains for')
    parser.add_argument('--perturbation-interval', type=int, default=100,
                        help='Episodes between exploit/explore (default: 100)')
    parser.add_argument('--num-processes', type=int, default=None,
                        help='Number of parallel processes (default: min(population, cpu_count))')

    # Data
    parser.add_argument('--train-data-path', type=str, default='data/rl_train.csv')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')
    parser.add_argument('--feature-scaler-path', type=str, default='trained_models/rl/feature_scaler.pkl',
                        help='Path to fitted StandardScaler pickle (default: trained_models/rl/feature_scaler.pkl)')

    # Training
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/pbt')
    parser.add_argument('--use-curriculum', action='store_true',
                        help='Use curriculum learning')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results (default: 42)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"🎲 Random seed: {args.seed}\n")

    # Create PBT manager
    pbt = PBTManager(
        population_size=args.population,
        data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        initial_capital=args.initial_capital,
        perturbation_interval=args.perturbation_interval,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_curriculum=args.use_curriculum,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        num_processes=args.num_processes,
    )

    # Run PBT training
    pbt.train(episodes_per_agent=args.episodes_per_agent)

    print(f"Best agent saved to: {pbt.checkpoint_dir}/best_agent.pt")


if __name__ == '__main__':
    main()
