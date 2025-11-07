"""Analyze reward structure to understand if it encourages the 5-trade behavior."""
import numpy as np

print("="*80)
print("REWARD STRUCTURE ANALYSIS")
print("="*80)

# From reward_config.py
pnl_reward_scale = 5.0
entry_penalty_scale = 0.3
stop_loss_penalty = -1.0

print("\nReward Component 1: Hourly P&L Reward")
print("  Formula: (pnl_change / total_execution_size) * 100 * pnl_reward_scale")
print(f"  Scale: {pnl_reward_scale}")

# Simulate a profitable position earning 0.1% per hour (realistic funding rate)
pnl_per_hour_pct = 0.1  # 0.1% per hour
hourly_reward = pnl_per_hour_pct * pnl_reward_scale
print(f"\n  Example: Position earning 0.1%/hour")
print(f"    → Hourly reward: {hourly_reward:.4f}")
print(f"    → Over 144 hours (6 days): {hourly_reward * 144:.2f} cumulative reward")

print("\nReward Component 2: Entry Penalty")
print("  Formula: -(entry_fee_pct * 100) * entry_penalty_scale")
print(f"  Scale: {entry_penalty_scale}")

# Typical entry fee: 0.02% on both sides = 0.04%
entry_fee_pct = 0.04
entry_penalty = -entry_fee_pct * entry_penalty_scale
print(f"\n  Example: Entry fee 0.04% (0.02% per side)")
print(f"    → Entry penalty: {entry_penalty:.4f}")

print("\nReward Component 3: Stop-Loss Penalty")
print(f"  Fixed penalty: {stop_loss_penalty}")

print("\n" + "="*80)
print("BEHAVIORAL INCENTIVES")
print("="*80)

print("\nScenario A: Open 5 positions, hold for 6 days")
total_entry_penalty = entry_penalty * 5
total_hourly_rewards = hourly_reward * 144 * 5  # 5 positions × 144 hours
total_reward_A = total_entry_penalty + total_hourly_rewards

print(f"  Entry penalties: {total_entry_penalty:.4f} (5 positions)")
print(f"  Hourly rewards: {total_hourly_rewards:.2f} (5 positions × 144 hours)")
print(f"  TOTAL REWARD: {total_reward_A:.2f}")

print("\nScenario B: Open 20 positions (4 per day), hold each for 1.5 days")
total_entry_penalty_B = entry_penalty * 20
total_hourly_rewards_B = hourly_reward * 36 * 20  # 20 positions × 36 hours
total_reward_B = total_entry_penalty_B + total_hourly_rewards_B

print(f"  Entry penalties: {total_entry_penalty_B:.4f} (20 positions)")
print(f"  Hourly rewards: {total_hourly_rewards_B:.2f} (20 positions × 36 hours)")
print(f"  TOTAL REWARD: {total_reward_B:.2f}")

print("\n" + "="*80)
print("KEY INSIGHT: Reward Structure Comparison")
print("="*80)
print(f"""
Scenario A (5 trades, long hold): {total_reward_A:.2f}
Scenario B (20 trades, short hold): {total_reward_B:.2f}

The reward structure is CORRECT and should favor more trades IF:
1. The agent can identify better opportunities to enter
2. The agent exits unprofitable positions to free capacity
3. Entry penalties are low enough (0.3x scale is reasonable)

BUT if the agent:
- Opens 5 positions immediately (while exploring)
- All 30 ENTER actions get MASKED (max_positions reached)
- Only 6 actions valid: HOLD + 5 EXIT
- Agent has NO incentive to EXIT (no immediate reward for exiting)
- Agent learns to HOLD because:
  a) HOLD accumulates hourly P&L rewards
  b) EXIT has no immediate reward (reward comes from cumulative P&L)
  c) Without seeing new opportunities after EXIT, agent can't learn "exit → enter better"

This creates a CREDIT ASSIGNMENT problem:
- Agent needs to learn: "Exit mediocre position → Free capacity → Enter better position"
- But reward only comes from P&L, not from strategic position management
- Agent gets stuck in "open 5 → hold forever" local optimum
""")

print("\n" + "="*80)
print("ROOT CAUSE DIAGNOSIS")
print("="*80)
print("""
The agent exhibits this behavior because:

1. ENTROPY IS TOO LOW (0.01)
   - Not enough exploration at episode 50
   - Agent converges too quickly to suboptimal policy
   - Should be 0.05-0.1 for adequate exploration

2. ACTION MASKING TRAP
   - Opening max_positions (5) masks all ENTER actions
   - Agent can't explore "exit → enter better" strategy
   - Gets stuck in "hold forever" local optimum

3. NO EXIT INCENTIVE
   - RL-v2 philosophy: "No immediate exit reward"
   - Relies on agent discovering exit → better entry through exploration
   - But with low entropy + action masking, exploration never happens!

4. CREDIT ASSIGNMENT FAILURE
   - Agent needs multi-step reasoning: "Exit now → Enter better later → Higher P&L"
   - But with low exploration, agent never discovers this pattern
   - PPO with low entropy converges to first working strategy: "hold everything"
""")
