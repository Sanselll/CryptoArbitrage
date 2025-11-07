"""Test action masking logic to understand why agent makes so few trades."""
import numpy as np
from models.rl.core.config import TradingConfig
from models.rl.core.portfolio import Portfolio

# Simulate different scenarios
print("="*80)
print("ACTION MASKING ANALYSIS")
print("="*80)

config = TradingConfig(max_positions=5)
portfolio = Portfolio(initial_capital=10000, max_positions=5, max_position_size_pct=33.3)

print("\nScenario 1: Fresh portfolio, 10 opportunities available")
print(f"  Current positions: {len(portfolio.positions)}/{config.max_positions}")
print(f"  Available margin: ${portfolio.available_margin:,.2f}")
print(f"  Has capacity: {len(portfolio.positions) < config.max_positions}")

# Simulate action mask calculation
mask = np.zeros(36, dtype=bool)
mask[0] = True  # HOLD always valid

num_positions = len(portfolio.positions)
max_positions = config.max_positions
has_capacity = num_positions < max_positions

if has_capacity:
    print(f"\n  ✓ Has capacity for new positions")
    print(f"  ENTER actions enabled: indices 1-30 (for 10 opportunities)")
    for i in range(10):  # 10 opportunities
        mask[1 + i] = True      # SMALL
        mask[11 + i] = True     # MEDIUM
        mask[21 + i] = True     # LARGE
else:
    print(f"\n  ✗ NO capacity for new positions")
    print(f"  All ENTER actions disabled")

print(f"\n  Total valid actions: {mask.sum()}")
print(f"    - HOLD: 1")
print(f"    - ENTER: {mask[1:31].sum()}")
print(f"    - EXIT: {mask[31:36].sum()}")

print("\n" + "="*80)
print("Scenario 2: 5 positions open (at max_positions limit)")
print("="*80)

# Simulate 5 open positions
portfolio.positions = [None] * 5  # Mock 5 positions

print(f"  Current positions: {len(portfolio.positions)}/{config.max_positions}")
print(f"  Has capacity: {len(portfolio.positions) < config.max_positions}")

mask2 = np.zeros(36, dtype=bool)
mask2[0] = True

has_capacity2 = len(portfolio.positions) < max_positions
if has_capacity2:
    print(f"\n  ✓ Has capacity")
    for i in range(10):
        mask2[1 + i] = True
        mask2[11 + i] = True
        mask2[21 + i] = True
else:
    print(f"\n  ✗ NO capacity - all ENTER actions masked!")

# EXIT actions
for i in range(5):
    if i < len(portfolio.positions):
        mask2[31 + i] = True

print(f"\n  Total valid actions: {mask2.sum()}")
print(f"    - HOLD: 1")
print(f"    - ENTER: {mask2[1:31].sum()}")
print(f"    - EXIT: {mask2[31:36].sum()}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
If the agent opens 5 positions early in the episode:
1. All 30 ENTER actions get MASKED (invalid)
2. Only 6 actions remain valid: HOLD (1) + EXIT (5)
3. Agent is FORCED to either HOLD or EXIT existing positions
4. This creates a "position limit trap" - agent can't explore new opportunities!

This explains the ~5 trades per episode behavior:
- Agent opens 5 positions early (while exploring)
- Gets trapped at max_positions limit
- Holds positions for ~6 days
- Episode ends with only those 5 trades executed

SOLUTION: The agent needs to learn to:
1. Exit positions strategically to free up capacity for better opportunities
2. Or use lower max_positions during training to avoid this trap
3. Or increase entropy to encourage more EXIT actions when at capacity
""")
