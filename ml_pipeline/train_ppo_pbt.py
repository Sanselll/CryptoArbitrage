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

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.core.curriculum import CurriculumScheduler
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


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

    # Reward hyperparameters (NEW!)
    pnl_reward_scale: float = 3.0
    entry_penalty_scale: float = 3.0
    liquidation_penalty_scale: float = 20.0
    stop_loss_penalty: float = -2.0

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
            # Reward hyperparameters
            pnl_reward_scale=np.clip(self.pnl_reward_scale * multiplier, 1.0, 5.0),
            entry_penalty_scale=np.clip(self.entry_penalty_scale * multiplier, 1.0, 5.0),
            liquidation_penalty_scale=np.clip(self.liquidation_penalty_scale * multiplier, 10.0, 30.0),
            stop_loss_penalty=np.clip(self.stop_loss_penalty * multiplier, -5.0, -0.5),
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
    recent_rewards: List[float] = None

    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = []


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
                 price_history_path: str = None):
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

        # Curriculum learning
        self.use_curriculum = use_curriculum
        if use_curriculum:
            self.curriculum = CurriculumScheduler()

        # Initialize population
        self.agents: List[AgentState] = []
        self._initialize_population()

        # Tracking
        self.generation = 0
        self.best_agent_id = 0
        self.best_score = -float('inf')

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
                  f"PNL_scale={hyperparams.pnl_reward_scale:.1f}, "
                  f"Entry_scale={hyperparams.entry_penalty_scale:.1f}")

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
                episode_length_days = 3

            # Create reward config from agent's hyperparameters
            reward_config = RewardConfig(
                pnl_reward_scale=agent.hyperparams.pnl_reward_scale,
                entry_penalty_scale=agent.hyperparams.entry_penalty_scale,
                liquidation_penalty_scale=agent.hyperparams.liquidation_penalty_scale,
                stop_loss_penalty=agent.hyperparams.stop_loss_penalty,
            )

            # Create environment
            env = FundingArbitrageEnv(
                data_path=self.data_path,
                initial_capital=self.initial_capital,
                trading_config=trading_config,
                reward_config=reward_config,
                episode_length_days=episode_length_days,
                price_history_path=self.price_history_path,
                simple_mode=False,
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

    def exploit_and_explore(self):
        """PBT exploit and explore step."""
        self.generation += 1

        print("\n" + "=" * 80)
        print(f"PBT GENERATION {self.generation}: EXPLOIT & EXPLORE")
        print("=" * 80)

        # Rank agents by performance (mean_reward_100)
        agents_sorted = sorted(self.agents, key=lambda a: a.mean_reward_100, reverse=True)

        # Display rankings
        print("\nAgent Rankings:")
        for rank, agent in enumerate(agents_sorted):
            print(f"  {rank+1}. Agent {agent.agent_id}: "
                  f"Mean(100)={agent.mean_reward_100:7.2f}, "
                  f"Episodes={agent.episodes_completed}")
        print()

        # Identify top and bottom quartiles
        quartile_size = self.population_size // 4
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
                  f"PNL {old_hyperparams.pnl_reward_scale:.1f}→{new_hyperparams.pnl_reward_scale:.1f}")

        print()

        # Update best agent
        best_agent = agents_sorted[0]
        if best_agent.mean_reward_100 > self.best_score:
            self.best_score = best_agent.mean_reward_100
            self.best_agent_id = best_agent.agent_id

            # Save best agent
            best_path = self.checkpoint_dir / 'best_agent.pt'
            best_agent.trainer.save(str(best_path))

            print(f"✅ New best agent: Agent {best_agent.agent_id}")
            print(f"   Score: {best_agent.mean_reward_100:.2f}")
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

            # Train all agents for perturbation_interval episodes
            for agent in self.agents:
                print(f"Training Agent {agent.agent_id}...")
                self.train_agent(agent, self.perturbation_interval)
                print(f"  Episodes: {agent.episodes_completed}, "
                      f"Mean(100): {agent.mean_reward_100:.2f}\n")

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

    # Data
    parser.add_argument('--train-data-path', type=str, default='data/rl_train.csv')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')

    # Training
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/pbt')
    parser.add_argument('--use-curriculum', action='store_true',
                        help='Use curriculum learning')

    return parser.parse_args()


def main():
    args = parse_args()

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
    )

    # Run PBT training
    pbt.train(episodes_per_agent=args.episodes_per_agent)

    print(f"Best agent saved to: {pbt.checkpoint_dir}/best_agent.pt")


if __name__ == '__main__':
    main()
