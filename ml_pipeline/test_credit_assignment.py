"""Test credit assignment problem - why agent doesn't learn to exit & re-enter."""
import numpy as np

print("="*80)
print("CREDIT ASSIGNMENT PROBLEM ANALYSIS")
print("="*80)

# Simulation parameters
pnl_reward_scale = 5.0
entry_penalty_scale = 0.3
gamma = 0.99  # Discount factor

print("\nScenario: Agent has 5 open positions, all mediocre (0.05%/hour)")
print("Two strategies available:")
print("  A) HOLD all 5 positions for remaining 100 hours")
print("  B) EXIT 2 mediocre, ENTER 2 better (0.15%/hour) after 10 hours")

print("\n" + "-"*80)
print("STRATEGY A: Hold mediocre positions")
print("-"*80)

# 5 positions × 0.05%/hour × 100 hours
pnl_per_hour_A = 5 * 0.05
total_hours_A = 100
cumulative_pnl_A = pnl_per_hour_A * total_hours_A
reward_A = cumulative_pnl_A * pnl_reward_scale

print(f"  P&L per hour: {pnl_per_hour_A:.2f}%")
print(f"  Total hours: {total_hours_A}")
print(f"  Cumulative P&L: {cumulative_pnl_A:.2f}%")
print(f"  Total reward: {reward_A:.2f}")

print("\n" + "-"*80)
print("STRATEGY B: Exit & re-enter with better opportunities")
print("-"*80)

# Phase 1: 5 mediocre for 10 hours
phase1_pnl = 5 * 0.05 * 10
phase1_reward = phase1_pnl * pnl_reward_scale

# Phase 2: EXIT 2 (no immediate reward)
# Phase 3: ENTER 2 better (entry penalty)
entry_penalty = -0.04 * entry_penalty_scale * 2  # 2 new positions

# Phase 4: 3 mediocre + 2 better for 90 hours
phase2_pnl = (3 * 0.05 + 2 * 0.15) * 90
phase2_reward = phase2_pnl * pnl_reward_scale

total_reward_B = phase1_reward + entry_penalty + phase2_reward

print(f"  Phase 1 (10h, 5 mediocre): {phase1_reward:.2f}")
print(f"  Phase 2 (EXIT 2): 0.00 (no immediate reward)")
print(f"  Phase 3 (ENTER 2 better): {entry_penalty:.4f} (entry penalty)")
print(f"  Phase 4 (90h, 3 mediocre + 2 better): {phase2_reward:.2f}")
print(f"  Total reward: {total_reward_B:.2f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"  Strategy A (hold): {reward_A:.2f}")
print(f"  Strategy B (exit & re-enter): {total_reward_B:.2f}")
print(f"  Difference: {total_reward_B - reward_A:.2f} ({(total_reward_B - reward_A) / reward_A * 100:+.1f}%)")

if total_reward_B > reward_A:
    print("\n  ✓ Strategy B is BETTER - agent SHOULD learn to exit & re-enter")
else:
    print("\n  ✗ Strategy A is BETTER - agent correctly learns to hold")

print("\n" + "="*80)
print("CREDIT ASSIGNMENT CHALLENGE")
print("="*80)

# Calculate discounted returns for EXIT action at t=10
discount_factor = gamma ** 10  # Reward 10 steps in future
immediate_return = 0.0  # EXIT has no immediate reward
future_benefit = (phase2_reward - phase1_reward / 10 * 9) / 90 * 10  # Average benefit over next 10 hours
discounted_benefit = future_benefit * discount_factor

print(f"\nAt timestep 10, agent considers EXIT action:")
print(f"  Immediate return: {immediate_return:.4f}")
print(f"  Future benefit (next 10h): ~{future_benefit:.4f}")
print(f"  Discount factor (γ^10): {discount_factor:.4f}")
print(f"  Discounted benefit: {discounted_benefit:.4f}")

print(f"\nVs. HOLD action at timestep 10:")
hold_immediate = (5 * 0.05) * pnl_reward_scale
print(f"  Immediate return: {hold_immediate:.4f}")

print("\n" + "="*80)
print("THE PROBLEM")
print("="*80)
print(f"""
Why agent struggles to learn EXIT → ENTER strategy:

1. IMMEDIATE REWARD BIAS
   - HOLD: immediate reward = {hold_immediate:.4f}
   - EXIT: immediate reward = 0.00
   - Agent prefers actions with immediate positive rewards

2. TEMPORAL CREDIT ASSIGNMENT
   - Benefit from EXIT appears {10}+ steps in the future
   - Discounted by γ^10 = {discount_factor:.4f}
   - Makes future benefit look {1/discount_factor:.2f}x smaller

3. EXPLORATION REQUIRED
   - Agent must EXIT (no reward)
   - Then ENTER (penalty: {entry_penalty:.4f})
   - Then wait 10+ hours to see benefit
   - With low entropy ({0.01}), this sequence rarely happens randomly

4. VALUE FUNCTION MUST LEARN
   - V(state with mediocre positions) < V(state with better positions)
   - But agent rarely visits "state after EXIT" due to low exploration
   - Value function can't learn the difference

SOLUTION:
- HIGHER ENTROPY (0.05-0.1) forces exploration of EXIT actions
- More episodes in "exited → looking for better" state
- Value function learns: "having capacity is valuable"
- Policy learns: "exit mediocre → enter better → higher cumulative reward"
""")
