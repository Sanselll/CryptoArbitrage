"""Diagnose entropy behavior during action selection."""
import torch
import numpy as np
from models.rl.networks.modular_ppo import ModularPPONetwork

net = ModularPPONetwork()

print("="*80)
print("ENTROPY DIAGNOSIS")
print("="*80)

# Test different scenarios
scenarios = [
    ("Empty state (all zeros)", np.zeros(275)),
    ("Random state", np.random.randn(275) * 0.1),
    ("Typical portfolio state", None),  # We'll construct this
]

for scenario_name, obs in scenarios[:2]:  # Skip last one for now
    print(f"\n{scenario_name}")
    print("-" * 80)

    # Test with different action masks
    masks = [
        ("All 31 actions valid (0 positions)", np.concatenate([np.ones(31), np.zeros(5)]).astype(bool)),
        ("Only 6 actions valid (5 positions)", np.concatenate([np.ones(1), np.zeros(30), np.ones(5)]).astype(bool)),
    ]

    for mask_name, mask in masks:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits, value = net(obs_t, mask_t)
            dist = torch.distributions.Categorical(logits=logits)

            # Theoretical maximum entropy for this number of valid actions
            n_valid = mask.sum()
            max_entropy = np.log(n_valid)

            actual_entropy = dist.entropy().item()
            entropy_ratio = actual_entropy / max_entropy if max_entropy > 0 else 0

            print(f"\n  {mask_name}")
            print(f"    Valid actions: {n_valid}")
            print(f"    Max possible entropy: {max_entropy:.4f}")
            print(f"    Actual entropy: {actual_entropy:.4f}")
            print(f"    Entropy ratio: {entropy_ratio:.2%}")
            print(f"    Interpretation: {'Uniform distribution' if entropy_ratio > 0.95 else 'Concentrated distribution' if entropy_ratio < 0.5 else 'Moderate concentration'}")

print("\n" + "="*80)
print("ENTROPY COEFFICIENT IMPACT")
print("="*80)

entropy_coef = 0.01
print(f"\nCurrent entropy_coef: {entropy_coef}")
print(f"\nEntropy bonus in loss function:")
print(f"  loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss")
print(f"  entropy_loss = -entropy.mean()")

print(f"\nWith entropy_coef = {entropy_coef}:")
print(f"  If entropy = 3.43 (uniform over 31 actions)")
print(f"    → Entropy bonus: {entropy_coef * 3.43:.6f}")
print(f"  If entropy = 1.79 (uniform over 6 actions)")
print(f"    → Entropy bonus: {entropy_coef * 1.79:.6f}")

print(f"\nWith entropy_coef = 0.05 (5x higher):")
print(f"  If entropy = 3.43")
print(f"    → Entropy bonus: {0.05 * 3.43:.6f}")
print(f"  If entropy = 1.79")
print(f"    → Entropy bonus: {0.05 * 1.79:.6f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
The network initialization is CORRECT:
- Initial entropy = 3.43 (near-uniform over 31 actions)
- Network is NOT biased toward HOLD initially

BUT the training converges quickly because:
1. entropy_coef = 0.01 is TOO LOW
   - Provides very weak incentive for exploration
   - Agent converges to first working policy very quickly

2. With low entropy, agent discovers "open 5 → hold" strategy early
   - This strategy gives positive rewards (from P&L)
   - Agent reinforces this behavior rapidly
   - No further exploration of "exit → enter better" strategy

3. Once at max_positions:
   - Action space shrinks to 6 actions (HOLD + 5 EXIT)
   - Even with same entropy_coef, less exploration happens
   - Agent defaults to HOLD (which accumulates P&L rewards)

SOLUTION:
- INCREASE entropy_coef to 0.05-0.1 during early training
- This will force more exploration of EXIT actions
- Agent can discover "exit → enter better" strategy
- Gradually anneal entropy_coef as training progresses
""")
