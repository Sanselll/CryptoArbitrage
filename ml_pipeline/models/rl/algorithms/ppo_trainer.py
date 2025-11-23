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
                 device: str = 'cpu',
                 initial_entropy_coef: Optional[float] = None,
                 final_entropy_coef: Optional[float] = None,
                 entropy_decay_episodes: int = 2000,
                 compile_model: bool = True):
        """
        Initialize PPO trainer.

        Args:
            network: ModularPPONetwork instance
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient (used if no decay)
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Mini-batch size for updates
            device: Device to use ('cpu' or 'cuda')
            initial_entropy_coef: Starting entropy coefficient (for decay schedule)
            final_entropy_coef: Final entropy coefficient (for decay schedule)
            entropy_decay_episodes: Number of episodes over which to decay entropy
            compile_model: Whether to use torch.compile (disable for production inference)
        """
        self.network = network.to(device)

        # Compile network for faster training (PyTorch 2.0+)
        # Disabled by default for production inference (requires C++ compiler)
        if compile_model:
            try:
                self.network = torch.compile(self.network, mode='reduce-overhead')
                print("✅ Model compiled with torch.compile for faster training")
            except Exception as e:
                print(f"⚠️  torch.compile not available (requires PyTorch 2.0+): {e}")
        else:
            print("ℹ️  torch.compile disabled (compile_model=False)")

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

        # Entropy decay schedule (V3.1)
        self.use_entropy_decay = initial_entropy_coef is not None and final_entropy_coef is not None
        if self.use_entropy_decay:
            self.initial_entropy_coef = initial_entropy_coef
            self.final_entropy_coef = final_entropy_coef
            self.entropy_decay_episodes = entropy_decay_episodes
            print(f"✅ Entropy decay enabled: {initial_entropy_coef:.3f} → {final_entropy_coef:.3f} over {entropy_decay_episodes} episodes")
        else:
            self.initial_entropy_coef = entropy_coef
            self.final_entropy_coef = entropy_coef
            self.entropy_decay_episodes = 0

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Training statistics
        self.total_timesteps = 0
        self.num_updates = 0
        self.num_episodes = 0  # V3.1: Track episode count for entropy decay
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def get_entropy_coef(self, episode: int) -> float:
        """
        Get the current entropy coefficient based on episode number.

        V3.1: Linear decay from initial_entropy_coef to final_entropy_coef
        over entropy_decay_episodes.

        Args:
            episode: Current episode number

        Returns:
            Current entropy coefficient
        """
        if not self.use_entropy_decay:
            return self.entropy_coef

        # Linear decay: initial - (initial - final) * progress
        progress = min(episode / self.entropy_decay_episodes, 1.0)
        current_coef = self.initial_entropy_coef - (self.initial_entropy_coef - self.final_entropy_coef) * progress
        return current_coef

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

    def update(self, rollout_buffer: RolloutBuffer, last_value: float = 0.0, episode: int = 0) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            rollout_buffer: Buffer containing rollout data
            last_value: Value estimate for final state
            episode: Current episode number (for entropy decay)

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

        # Get current entropy coefficient (with decay)
        current_entropy_coef = self.get_entropy_coef(episode)

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

                # Total loss (V3.1: use current_entropy_coef instead of self.entropy_coef)
                loss = policy_loss + self.value_coef * value_loss + current_entropy_coef * entropy_loss

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
            'entropy_coef': current_entropy_coef,  # V3.1: Include current entropy coefficient
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
        # Explicitly set network to training mode
        self.network.train()

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

        # Update policy (V3.1: pass episode number for entropy decay)
        update_stats = self.update(buffer, last_value, episode=self.num_episodes)
        self.num_episodes += 1  # Increment episode counter

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

    def train_episode_vectorized(self, vec_env, max_steps: int = 1000) -> Dict[str, float]:
        """
        Collect rollouts from multiple parallel environments and update the policy.

        This is significantly faster than train_episode() as it collects data from
        multiple environments simultaneously.

        Args:
            vec_env: ParallelEnv instance (multiple environments in parallel)
            max_steps: Maximum steps per episode (applied to each environment)

        Returns:
            Dictionary of episode statistics (averaged across all environments)
        """
        from models.rl.core.vec_env import ParallelEnv

        n_envs = len(vec_env)
        buffer = RolloutBuffer()

        # Reset all environments
        obs, infos = vec_env.reset()  # (n_envs, obs_dim)

        # Track per-environment episode stats
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs)
        completed_episodes = []

        for step in range(max_steps):
            # Get action masks from all environments
            action_masks = vec_env.get_action_masks()  # (n_envs, n_actions) or None

            # Select actions for all environments (batch inference)
            actions = []
            values = []
            log_probs = []

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)  # (n_envs, obs_dim)

                if action_masks is not None:
                    mask_tensor = torch.BoolTensor(action_masks).to(self.device)
                else:
                    mask_tensor = None

                # Batch forward pass
                for i in range(n_envs):
                    obs_i = obs_tensor[i:i+1]  # (1, obs_dim)
                    mask_i = mask_tensor[i:i+1] if mask_tensor is not None else None

                    action, value, log_prob = self.select_action(obs_i[0].cpu().numpy(),
                                                                 mask_i[0].cpu().numpy() if mask_i is not None else None)
                    actions.append(action)
                    values.append(value)
                    log_probs.append(log_prob)

            actions = np.array(actions)

            # Step all environments
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = terminated | truncated

            # Store transitions for all environments
            for i in range(n_envs):
                mask_i = action_masks[i] if action_masks is not None else None
                buffer.add(obs[i], actions[i], rewards[i], values[i], log_probs[i], dones[i], mask_i)

                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                self.total_timesteps += 1

                # Track completed episodes
                if infos[i].get('episode_ended', False):
                    completed_episodes.append({
                        'reward': episode_rewards[i],
                        'length': episode_lengths[i]
                    })
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0.0

            # Update observations
            obs = next_obs

            # Early stopping if all environments done (unlikely in practice)
            if np.all(dones):
                break

        # Get last values for bootstrapping (for all environments)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            # For simplicity, compute average last value across environments
            # In practice, each env may need its own bootstrap value
            last_values = []
            for i in range(n_envs):
                _, last_value_tensor = self.network(obs_tensor[i:i+1], None)
                last_values.append(last_value_tensor.item())
            last_value = np.mean(last_values)

        # Update policy with collected rollouts
        update_stats = self.update(buffer, last_value)

        # Calculate episode statistics (from completed episodes)
        if completed_episodes:
            mean_reward = np.mean([ep['reward'] for ep in completed_episodes])
            mean_length = np.mean([ep['length'] for ep in completed_episodes])

            # Track for rolling average
            for ep in completed_episodes:
                self.episode_rewards.append(ep['reward'])
                self.episode_lengths.append(ep['length'])
        else:
            # Use current (incomplete) episode stats
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)

        return {
            'episode_reward': mean_reward,
            'episode_length': mean_length,
            'mean_reward_100': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_length_100': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'n_completed_episodes': len(completed_episodes),
            'n_envs': n_envs,
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

        # Handle checkpoints saved with torch.compile (have _orig_mod. prefix)
        state_dict = checkpoint['network_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Strip _orig_mod. prefix for compatibility with non-compiled models
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            print("ℹ️  Stripped '_orig_mod.' prefix from checkpoint (was saved with torch.compile)")

        self.network.load_state_dict(state_dict)
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
