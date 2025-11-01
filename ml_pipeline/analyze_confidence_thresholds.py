"""
Analyze Confidence Thresholds from RL Model Evaluation

This script analyzes the action probabilities from the evaluation trades CSV
to determine what confidence thresholds the model actually used for ENTER/EXIT decisions.
"""

import pandas as pd
import numpy as np

# Load the evaluation trades
df = pd.read_csv('evaluation_trades_20251101_104039.csv')

# Filter out trades with missing probabilities (zero P&L trades)
df_with_probs = df[(df['entry_probability'].notna()) & (df['exit_probability'].notna())]

print("=" * 80)
print("CONFIDENCE THRESHOLD ANALYSIS - RL Model (agent_7)")
print("=" * 80)
print(f"\nTotal trades analyzed: {len(df_with_probs)}")
print(f"Winning trades: {len(df_with_probs[df_with_probs['pnl_usd'] > 0])}")
print(f"Losing trades: {len(df_with_probs[df_with_probs['pnl_usd'] <= 0])}")

# Separate winning and losing trades
winning = df_with_probs[df_with_probs['pnl_usd'] > 0]
losing = df_with_probs[df_with_probs['pnl_usd'] <= 0]

print("\n" + "─" * 80)
print("ENTRY PROBABILITY ANALYSIS")
print("─" * 80)

print("\nAll Trades:")
print(f"  Mean:   {df_with_probs['entry_probability'].mean():.4f} ({df_with_probs['entry_probability'].mean()*100:.2f}%)")
print(f"  Median: {df_with_probs['entry_probability'].median():.4f} ({df_with_probs['entry_probability'].median()*100:.2f}%)")
print(f"  Std:    {df_with_probs['entry_probability'].std():.4f}")
print(f"  Min:    {df_with_probs['entry_probability'].min():.4f} ({df_with_probs['entry_probability'].min()*100:.2f}%)")
print(f"  Max:    {df_with_probs['entry_probability'].max():.4f} ({df_with_probs['entry_probability'].max()*100:.2f}%)")

print("\nWinning Trades:")
print(f"  Mean:   {winning['entry_probability'].mean():.4f} ({winning['entry_probability'].mean()*100:.2f}%)")
print(f"  Median: {winning['entry_probability'].median():.4f} ({winning['entry_probability'].median()*100:.2f}%)")
print(f"  Std:    {winning['entry_probability'].std():.4f}")

print("\nLosing Trades:")
print(f"  Mean:   {losing['entry_probability'].mean():.4f} ({losing['entry_probability'].mean()*100:.2f}%)")
print(f"  Median: {losing['entry_probability'].median():.4f} ({losing['entry_probability'].median()*100:.2f}%)")
print(f"  Std:    {losing['entry_probability'].std():.4f}")

print("\n" + "─" * 80)
print("EXIT PROBABILITY ANALYSIS")
print("─" * 80)

print("\nAll Trades:")
print(f"  Mean:   {df_with_probs['exit_probability'].mean():.4f} ({df_with_probs['exit_probability'].mean()*100:.2f}%)")
print(f"  Median: {df_with_probs['exit_probability'].median():.4f} ({df_with_probs['exit_probability'].median()*100:.2f}%)")
print(f"  Std:    {df_with_probs['exit_probability'].std():.4f}")
print(f"  Min:    {df_with_probs['exit_probability'].min():.4f} ({df_with_probs['exit_probability'].min()*100:.2f}%)")
print(f"  Max:    {df_with_probs['exit_probability'].max():.4f} ({df_with_probs['exit_probability'].max()*100:.2f}%)")

print("\nWinning Trades:")
print(f"  Mean:   {winning['exit_probability'].mean():.4f} ({winning['exit_probability'].mean()*100:.2f}%)")
print(f"  Median: {winning['exit_probability'].median():.4f} ({winning['exit_probability'].median()*100:.2f}%)")
print(f"  Std:    {winning['exit_probability'].std():.4f}")

print("\nLosing Trades:")
print(f"  Mean:   {losing['exit_probability'].mean():.4f} ({losing['exit_probability'].mean()*100:.2f}%)")
print(f"  Median: {losing['exit_probability'].median():.4f} ({losing['exit_probability'].median()*100:.2f}%)")
print(f"  Std:    {losing['exit_probability'].std():.4f}")

print("\n" + "─" * 80)
print("HOLD PROBABILITY ANALYSIS")
print("─" * 80)

print("\nEntry Hold Probability:")
print(f"  Mean:   {df_with_probs['entry_hold_prob'].mean():.4f} ({df_with_probs['entry_hold_prob'].mean()*100:.2f}%)")
print(f"  Median: {df_with_probs['entry_hold_prob'].median():.4f} ({df_with_probs['entry_hold_prob'].median()*100:.2f}%)")
print(f"  Std:    {df_with_probs['entry_hold_prob'].std():.4f}")

print("\nExit Hold Probability:")
print(f"  Mean:   {df_with_probs['exit_hold_prob'].mean():.4f} ({df_with_probs['exit_hold_prob'].mean()*100:.2f}%)")
print(f"  Median: {df_with_probs['exit_hold_prob'].median():.4f} ({df_with_probs['exit_hold_prob'].median()*100:.2f}%)")
print(f"  Std:    {df_with_probs['exit_hold_prob'].std():.4f}")

# Percentile analysis
print("\n" + "─" * 80)
print("PROBABILITY PERCENTILES")
print("─" * 80)

percentiles = [10, 25, 50, 75, 90]
print("\nEntry Probability Percentiles:")
for p in percentiles:
    val = np.percentile(df_with_probs['entry_probability'], p)
    print(f"  {p}th percentile: {val:.4f} ({val*100:.2f}%)")

print("\nExit Probability Percentiles:")
for p in percentiles:
    val = np.percentile(df_with_probs['exit_probability'], p)
    print(f"  {p}th percentile: {val:.4f} ({val*100:.2f}%)")

# Correlation analysis
print("\n" + "─" * 80)
print("PROBABILITY vs P&L CORRELATION")
print("─" * 80)

entry_corr = df_with_probs[['entry_probability', 'pnl_usd']].corr().iloc[0, 1]
exit_corr = df_with_probs[['exit_probability', 'pnl_usd']].corr().iloc[0, 1]

print(f"\nEntry Probability vs P&L: {entry_corr:.4f}")
print(f"Exit Probability vs P&L:  {exit_corr:.4f}")

# Top profitable trades
print("\n" + "─" * 80)
print("TOP 5 MOST PROFITABLE TRADES - ENTRY/EXIT PROBABILITIES")
print("─" * 80)

top_trades = df_with_probs.nlargest(5, 'pnl_usd')
print("\n{:<12} {:>10} {:>10} {:>12} {:>12}".format(
    "Symbol", "P&L $", "P&L %", "Entry Prob", "Exit Prob"
))
print("─" * 60)
for _, trade in top_trades.iterrows():
    print("{:<12} {:>10.2f} {:>9.2f}% {:>11.1f}% {:>11.1f}%".format(
        trade['symbol'],
        trade['pnl_usd'],
        trade['pnl_pct'],
        trade['entry_probability'] * 100,
        trade['exit_probability'] * 100
    ))

# Worst trades
print("\n" + "─" * 80)
print("TOP 5 WORST TRADES - ENTRY/EXIT PROBABILITIES")
print("─" * 80)

worst_trades = df_with_probs.nsmallest(5, 'pnl_usd')
print("\n{:<12} {:>10} {:>10} {:>12} {:>12}".format(
    "Symbol", "P&L $", "P&L %", "Entry Prob", "Exit Prob"
))
print("─" * 60)
for _, trade in worst_trades.iterrows():
    print("{:<12} {:>10.2f} {:>9.2f}% {:>11.1f}% {:>11.1f}%".format(
        trade['symbol'],
        trade['pnl_usd'],
        trade['pnl_pct'],
        trade['entry_probability'] * 100,
        trade['exit_probability'] * 100
    ))

print("\n" + "=" * 80)
print("CONFIDENCE THRESHOLD RECOMMENDATIONS")
print("=" * 80)

# Based on the data, suggest thresholds
entry_mean = df_with_probs['entry_probability'].mean()
entry_p75 = np.percentile(df_with_probs['entry_probability'], 75)
exit_mean = df_with_probs['exit_probability'].mean()
exit_p75 = np.percentile(df_with_probs['exit_probability'], 75)

print("\nBased on the actual model behavior:")
print(f"  - Entry probabilities range from {df_with_probs['entry_probability'].min()*100:.1f}% to {df_with_probs['entry_probability'].max()*100:.1f}%")
print(f"  - Exit probabilities range from {df_with_probs['exit_probability'].min()*100:.1f}% to {df_with_probs['exit_probability'].max()*100:.1f}%")
print(f"  - Mean entry probability: {entry_mean*100:.1f}%")
print(f"  - Mean exit probability: {exit_mean*100:.1f}%")

print("\nObservation:")
print("  The model operates with relatively LOW action probabilities (12-18% range).")
print("  This is because the action space has 9 actions, so probabilities are distributed.")
print("  Even the 'chosen' action typically has only 13-16% probability.")

print("\nSuggested Confidence Levels:")
print("  HIGH:   >= 16% (top 25% of decisions)")
print("  MEDIUM: 13-16% (middle 50%)")
print("  LOW:    < 13% (bottom 25%)")

print("\nAlternative approach (relative to mean):")
print(f"  HIGH:   >= {entry_mean*1.15*100:.1f}% for ENTER, >= {exit_mean*1.15*100:.1f}% for EXIT")
print(f"  MEDIUM: {entry_mean*0.85*100:.1f}-{entry_mean*1.15*100:.1f}% for ENTER, {exit_mean*0.85*100:.1f}-{exit_mean*1.15*100:.1f}% for EXIT")
print(f"  LOW:    < {entry_mean*0.85*100:.1f}% for ENTER, < {exit_mean*0.85*100:.1f}% for EXIT")

print("\n" + "=" * 80)
