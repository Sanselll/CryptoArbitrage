"""
Modular PPO Network Architecture for Multi-Opportunity, Multi-Position Trading

Implements Option B: Modular Encoders with Cross-Attention Fusion
- ConfigEncoder: 5 → 16
- PortfolioEncoder: 10 → 32
- ExecutionEncoder: 5×12 → 64 (with self-attention)
- OpportunityEncoder: 10×20 → 128 (with self-attention)
- FusionLayer: Cross-attention → 256
- Actor/Critic heads with action masking support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ConfigEncoder(nn.Module):
    """
    Encodes trading configuration into a dense representation.

    Input: 5 features (max_leverage, target_utilization, max_positions, stop_loss, liq_buffer)
    Output: 16-dimensional embedding
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 5) config features
        Returns:
            (batch_size, 16) config embedding
        """
        return self.net(x)


class PortfolioEncoder(nn.Module):
    """
    Encodes portfolio state into a dense representation.

    Input: 10 features (capital_ratio, available_ratio, margin_util, etc.)
    Output: 32-dimensional embedding
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 10) portfolio features
        Returns:
            (batch_size, 32) portfolio embedding
        """
        return self.net(x)


class ExecutionEncoder(nn.Module):
    """
    Encodes execution states with self-attention over position slots.

    Input: 60 features (5 positions × 12 features each)
    Output: 64-dimensional embedding
    """

    def __init__(self,
                 num_slots: int = 5,
                 features_per_slot: int = 12,
                 embedding_dim: int = 32,
                 num_heads: int = 2,
                 output_dim: int = 64):
        super().__init__()

        self.num_slots = num_slots
        self.features_per_slot = features_per_slot
        self.embedding_dim = embedding_dim

        # Per-slot embedding
        self.slot_encoder = nn.Sequential(
            nn.Linear(features_per_slot, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Self-attention over slots
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embedding_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim * num_slots, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 60) execution features (5 slots × 12 features)
        Returns:
            (batch_size, 64) execution embedding
        """
        batch_size = x.shape[0]

        # Reshape to (batch_size, num_slots, features_per_slot)
        x = x.view(batch_size, self.num_slots, self.features_per_slot)

        # Encode each slot
        slot_embeddings = self.slot_encoder(x)  # (batch, 5, 32)

        # Self-attention over slots
        attn_out, _ = self.attention(slot_embeddings, slot_embeddings, slot_embeddings)

        # Residual connection + normalization
        slot_embeddings = self.norm(slot_embeddings + attn_out)

        # Flatten and project
        flattened = slot_embeddings.view(batch_size, -1)  # (batch, 160)
        output = self.output_proj(flattened)  # (batch, 64)

        return output


class OpportunityEncoder(nn.Module):
    """
    Encodes opportunity states with self-attention over opportunity slots.

    Input: 200 features (10 opportunities × 20 features each)
    Output: 128-dimensional embedding
    """

    def __init__(self,
                 num_slots: int = 10,
                 features_per_slot: int = 20,
                 embedding_dim: int = 64,
                 num_heads: int = 4,
                 output_dim: int = 128):
        super().__init__()

        self.num_slots = num_slots
        self.features_per_slot = features_per_slot
        self.embedding_dim = embedding_dim

        # Per-opportunity embedding
        self.opp_encoder = nn.Sequential(
            nn.Linear(features_per_slot, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Self-attention over opportunities
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embedding_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim * num_slots, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 200) opportunity features (10 slots × 20 features)
        Returns:
            (batch_size, 128) opportunity embedding
        """
        batch_size = x.shape[0]

        # Reshape to (batch_size, num_slots, features_per_slot)
        x = x.view(batch_size, self.num_slots, self.features_per_slot)

        # Encode each opportunity
        opp_embeddings = self.opp_encoder(x)  # (batch, 10, 64)

        # Self-attention over opportunities
        attn_out, _ = self.attention(opp_embeddings, opp_embeddings, opp_embeddings)

        # Residual connection + normalization
        opp_embeddings = self.norm(opp_embeddings + attn_out)

        # Flatten and project
        flattened = opp_embeddings.view(batch_size, -1)  # (batch, 640)
        output = self.output_proj(flattened)  # (batch, 128)

        return output


class FusionLayer(nn.Module):
    """
    Fuses config, portfolio, execution, and opportunity embeddings using cross-attention.

    Input: config (16), portfolio (32), executions (64), opportunities (128)
    Output: 256-dimensional fused representation
    """

    def __init__(self,
                 config_dim: int = 16,
                 portfolio_dim: int = 32,
                 exec_dim: int = 64,
                 opp_dim: int = 128,
                 fusion_dim: int = 256):
        super().__init__()

        # Project all embeddings to same dimension for attention
        self.config_proj = nn.Linear(config_dim, fusion_dim)
        self.portfolio_proj = nn.Linear(portfolio_dim, fusion_dim)
        self.exec_proj = nn.Linear(exec_dim, fusion_dim)
        self.opp_proj = nn.Linear(opp_dim, fusion_dim)

        # Cross-attention: opportunities attend to portfolio+executions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

    def forward(self,
                config: torch.Tensor,
                portfolio: torch.Tensor,
                executions: torch.Tensor,
                opportunities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            config: (batch_size, 16)
            portfolio: (batch_size, 32)
            executions: (batch_size, 64)
            opportunities: (batch_size, 128)
        Returns:
            (batch_size, 256) fused embedding
        """
        batch_size = config.shape[0]

        # Project all to fusion dimension
        config_emb = self.config_proj(config)  # (batch, 256)
        portfolio_emb = self.portfolio_proj(portfolio)  # (batch, 256)
        exec_emb = self.exec_proj(executions)  # (batch, 256)
        opp_emb = self.opp_proj(opportunities)  # (batch, 256)

        # Stack into sequence for attention
        # Sequence: [config, portfolio, executions, opportunities]
        sequence = torch.stack([config_emb, portfolio_emb, exec_emb, opp_emb], dim=1)  # (batch, 4, 256)

        # Cross-attention (each element attends to all others)
        attn_out, _ = self.cross_attention(sequence, sequence, sequence)

        # Residual + norm
        sequence = self.norm1(sequence + attn_out)

        # Feed-forward
        ffn_out = self.ffn(sequence)

        # Residual + norm
        sequence = self.norm2(sequence + ffn_out)

        # Mean pooling over sequence
        fused = sequence.mean(dim=1)  # (batch, 256)

        return fused


class ActorHead(nn.Module):
    """
    Actor network: outputs action logits with masking support.

    Input: 256-dimensional fused embedding
    Output: 36 action logits (before masking)
    """

    def __init__(self, input_dim: int = 256, num_actions: int = 36):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 256) fused embedding
            action_mask: (batch_size, 36) boolean mask (True = valid, False = invalid)
        Returns:
            (batch_size, 36) action logits (masked invalid actions set to -inf)
        """
        logits = self.net(x)  # (batch, 36)

        if action_mask is not None:
            # Mask invalid actions by setting logits to -inf
            logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))

        return logits


class CriticHead(nn.Module):
    """
    Critic network: outputs state value estimate.

    Input: 256-dimensional fused embedding
    Output: Scalar value estimate
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 256) fused embedding
        Returns:
            (batch_size, 1) value estimate
        """
        return self.net(x)


class ModularPPONetwork(nn.Module):
    """
    Complete modular PPO network combining all components.

    Processes 301-dim observation (Phase 2: Added APR comparison features):
    - Config (5) → ConfigEncoder → 16
    - Portfolio (6) → PortfolioEncoder → 32
    - Executions (100) → ExecutionEncoder → 64  # Phase 2: +3 APR comparison features per position
    - Opportunities (190) → OpportunityEncoder → 128
    - Fusion → 256
    - Actor → 36 action logits
    - Critic → 1 value
    """

    def __init__(self):
        super().__init__()

        # Encoders
        self.config_encoder = ConfigEncoder(input_dim=5, output_dim=16)
        self.portfolio_encoder = PortfolioEncoder(input_dim=6, output_dim=32)
        self.execution_encoder = ExecutionEncoder(
            num_slots=5,
            features_per_slot=20,  # Phase 2: +3 APR comparison features (was 17)
            embedding_dim=32,
            num_heads=2,
            output_dim=64
        )
        self.opportunity_encoder = OpportunityEncoder(
            num_slots=10,
            features_per_slot=19,
            embedding_dim=64,
            num_heads=4,
            output_dim=128
        )

        # Fusion
        self.fusion = FusionLayer(
            config_dim=16,
            portfolio_dim=32,
            exec_dim=64,
            opp_dim=128,
            fusion_dim=256
        )

        # Heads
        self.actor = ActorHead(input_dim=256, num_actions=36)
        self.critic = CriticHead(input_dim=256)

    def forward(self,
                obs: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through entire network.

        Args:
            obs: (batch_size, 301) observation (Phase 2: APR comparison features)
            action_mask: (batch_size, 36) boolean mask (optional)

        Returns:
            action_logits: (batch_size, 36) masked action logits
            value: (batch_size, 1) state value estimate
        """
        # Split observation into components
        config = obs[:, 0:5]            # (batch, 5)
        portfolio = obs[:, 5:11]        # (batch, 6)
        executions = obs[:, 11:111]     # (batch, 100) - Phase 2: +15 dims
        opportunities = obs[:, 111:301] # (batch, 190)

        # Encode each component
        config_emb = self.config_encoder(config)
        portfolio_emb = self.portfolio_encoder(portfolio)
        exec_emb = self.execution_encoder(executions)
        opp_emb = self.opportunity_encoder(opportunities)

        # Fuse embeddings
        fused = self.fusion(config_emb, portfolio_emb, exec_emb, opp_emb)

        # Generate outputs
        action_logits = self.actor(fused, action_mask)
        value = self.critic(fused)

        return action_logits, value

    def get_action_distribution(self,
                                 obs: torch.Tensor,
                                 action_mask: Optional[torch.Tensor] = None) -> torch.distributions.Categorical:
        """
        Get action distribution for sampling.

        Args:
            obs: (batch_size, 301) observation (Phase 2: APR comparison)
            action_mask: (batch_size, 36) boolean mask (optional)

        Returns:
            Categorical distribution over actions
        """
        action_logits, _ = self.forward(obs, action_mask)
        return torch.distributions.Categorical(logits=action_logits)

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.

        Args:
            obs: (batch_size, 301) observations (Phase 2: APR comparison)
            actions: (batch_size,) actions taken
            action_mask: (batch_size, 36) boolean mask (optional)

        Returns:
            values: (batch_size, 1) state values
            log_probs: (batch_size,) log probabilities of actions
            entropy: (batch_size,) entropy of action distributions
        """
        action_logits, values = self.forward(obs, action_mask)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_probs, entropy


if __name__ == "__main__":
    # Test the network
    print("Testing ModularPPONetwork...")

    # Create network
    net = ModularPPONetwork()

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 301)  # Phase 2: APR comparison features (was 286)

    # Create action mask (all valid)
    action_mask = torch.ones(batch_size, 36, dtype=torch.bool)
    # Mask out some actions for testing
    action_mask[:, 31:36] = False  # No positions to exit

    # Forward pass
    action_logits, values = net(obs, action_mask)

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Values shape: {values.shape}")

    # Test action distribution
    dist = net.get_action_distribution(obs, action_mask)
    actions = dist.sample()
    print(f"\nSampled actions: {actions}")

    # Test action evaluation
    values, log_probs, entropy = net.evaluate_actions(obs, actions, action_mask)
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")

    print("\n✅ Network test passed!")
