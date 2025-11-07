"""Test initial network behavior to diagnose premature convergence."""
import torch
import numpy as np
from models.rl.networks.modular_ppo import ModularPPONetwork

# Create network
net = ModularPPONetwork()

# Create zero observation (simulating empty state)
obs = np.zeros(275)
mask = np.ones(36, dtype=bool)
mask[31:36] = False  # No positions to exit

obs_t = torch.FloatTensor(obs).unsqueeze(0)
mask_t = torch.BoolTensor(mask).unsqueeze(0)

with torch.no_grad():
    logits, value = net(obs_t, mask_t)
    dist = torch.distributions.Categorical(logits=logits)
    probs = dist.probs

    print("="*80)
    print("INITIAL NETWORK BIAS TEST")
    print("="*80)
    print(f"\nAction probabilities (first 10 ENTER actions):")
    for i in range(10):
        print(f"  Action {i:2d}: {probs[0, i].item():.6f}")

    print(f"\nKey statistics:")
    print(f"  HOLD probability (action 0):        {probs[0, 0].item():.6f}")
    print(f"  Total ENTER probability (1-30):     {probs[0, 1:31].sum().item():.6f}")
    print(f"  Entropy:                            {dist.entropy().item():.6f}")
    print(f"  Max action probability:             {probs[0].max().item():.6f}")
    print(f"  Min action probability (non-exit):  {probs[0, :31].min().item():.6f}")

    # Sample 100 actions to see distribution
    samples = []
    for _ in range(100):
        action = dist.sample()
        samples.append(action.item())

    print(f"\nSampling 100 actions:")
    print(f"  HOLD (0):         {samples.count(0)}/100")
    print(f"  ENTER SMALL (1-10):   {sum(1 for s in samples if 1 <= s <= 10)}/100")
    print(f"  ENTER MEDIUM (11-20): {sum(1 for s in samples if 11 <= s <= 20)}/100")
    print(f"  ENTER LARGE (21-30):  {sum(1 for s in samples if 21 <= s <= 30)}/100")
    print()
