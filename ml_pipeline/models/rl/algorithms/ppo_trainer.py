"""
PPO (Proximal Policy Optimization) Trainer with Action Masking

Implements PPO algorithm for training the ModularPPONetwork on the
FundingArbitrageEnv environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

from models.rl.networks.modular_ppo import ModularPPONetwork


class RolloutBuffer:
    """
    Buffer for storing rollout data during episode collection.
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

    def add(self,
            obs: np.ndarray,
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            done: bool,
            action_mask: np.ndarray):
        """Add a timestep to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        return {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'dones': torch.FloatTensor(self.dones),
            'action_masks': torch.BoolTensor(np.array(self.action_masks)),
        }

    def clear(self):
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO Trainer with action masking support.
    """

    def __init__(self,
                 network: ModularPPONetwork,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        """
        Initialize PPO trainer.

        Args:
            network: ModularPPONetwork instance
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Mini-batch size for updates
            device: Device to use ('cpu' or 'cuda')
        """
        self.network = network.to(device)
        self.device = device

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Training statistics
        self.total_timesteps = 0
        self.num_updates = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def select_action(self,
                      obs: np.ndarray,
                      action_mask: Optional[np.ndarray] = None,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select an action using the policy.

        Args:
            obs: Observation (275,)
            action_mask: Action mask (36,) boolean array
            deterministic: If True, select argmax action instead of sampling

        Returns:
            action: Selected action
            value: State value estimate
            log_prob: Log probability of action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, 275)

            if action_mask is not None:
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)  # (1, 36)
            else:
                mask_tensor = None

            # Forward pass
            action_logits, value = self.network(obs_tensor, mask_tensor)

            # Get action distribution
            dist = torch.distributions.Categorical(logits=action_logits)

            # Sample or select max
            if deterministic:
                action = torch.argmax(action_logits, dim=1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            return action.item(), value.item(), log_prob.item()

    def compute_returns_and_advantages(self,
                                        rewards: torch.Tensor,
                                        values: torch.Tensor,
                                        dones: torch.Tensor,
                                        last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE.

        Args:
            rewards: (T,) rewards
            values: (T,) value estimates
            dones: (T,) done flags
            last_value: Value estimate for last state

        Returns:
            returns: (T,) discounted returns
            advantages: (T,) GAE advantages
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)

        # GAE computation
        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
            else:
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return returns, advantages

    def update(self, rollout_buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            rollout_buffer: Buffer containing rollout data
            last_value: Value estimate for final state

        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        data = rollout_buffer.get()
        obs = data['observations'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        old_values = data['values'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)
        action_masks = data['action_masks'].to(self.device)

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, old_values, dones, last_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0
        n_batches = 0

        # Multiple epochs
        for epoch in range(self.n_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(obs), device=self.device)

            # Mini-batch updates
            for start_idx in range(0, len(obs), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                # Get mini-batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # Forward pass
                values, log_probs, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions, batch_action_masks
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                values_clipped = old_values[batch_indices] + torch.clamp(
                    values.squeeze() - old_values[batch_indices],
                    -self.clip_range,
                    self.clip_range
                )
                value_loss1 = (values.squeeze() - batch_returns) ** 2
                value_loss2 = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clipfrac += clipfrac.item()
                n_batches += 1

        self.num_updates += 1

        return {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
            'approx_kl': total_approx_kl / n_batches,
            'clipfrac': total_clipfrac / n_batches,
        }

    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """
        Collect a full episode and update the policy.

        Args:
            env: Gymnasium environment (FundingArbitrageEnv)
            max_steps: Maximum steps per episode

        Returns:
            Dictionary of episode statistics
        """
        buffer = RolloutBuffer()
        obs, info = env.reset()

        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # Get action mask from environment
            if hasattr(env, '_get_action_mask'):
                action_mask = env._get_action_mask()
            else:
                action_mask = None

            # Select action
            action, value, log_prob = self.select_action(obs, action_mask)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.add(obs, action, reward, value, log_prob, done, action_mask)

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1

            if done:
                break

        # Get last value for bootstrapping
        if done:
            last_value = 0.0
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, last_value_tensor = self.network(obs_tensor, None)
                last_value = last_value_tensor.item()

        # Update policy
        update_stats = self.update(buffer, last_value)

        # Track episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_reward_100': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_length_100': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            **update_stats
        }

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.num_updates = checkpoint.get('num_updates', 0)


if __name__ == "__main__":
    # Test PPO trainer
    print("Testing PPO Trainer...")

    # Create network
    from models.rl.networks.modular_ppo import ModularPPONetwork
    network = ModularPPONetwork()

    # Create trainer
    trainer = PPOTrainer(
        network=network,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        n_epochs=4,
        batch_size=32,
    )

    print(f"✅ Trainer created with {sum(p.numel() for p in network.parameters()):,} parameters")

    # Test action selection
    obs = np.random.randn(275)
    action_mask = np.ones(36, dtype=bool)
    action_mask[31:36] = False  # Mask exit actions

    action, value, log_prob = trainer.select_action(obs, action_mask)
    print(f"✅ Action selection works: action={action}, value={value:.3f}, log_prob={log_prob:.3f}")

    # Test rollout buffer
    buffer = RolloutBuffer()
    for i in range(10):
        buffer.add(
            obs=np.random.randn(275),
            action=i % 31,
            reward=np.random.randn(),
            value=np.random.randn(),
            log_prob=np.random.randn(),
            done=False,
            action_mask=action_mask
        )

    print(f"✅ Buffer works: {len(buffer)} transitions stored")

    # Test update
    update_stats = trainer.update(buffer, last_value=0.0)
    print(f"✅ Update works: loss={update_stats['loss']:.4f}")

    print("\n✅ PPO Trainer test passed!")
