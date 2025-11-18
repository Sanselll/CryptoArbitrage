#!/usr/bin/env python3
"""Analyze checkpoint test results and find top performers."""

import os
import re
from pathlib import Path

def extract_metrics(filepath):
    """Extract key metrics from a result file."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    metrics = {}

    # Checkpoint name
    filename = os.path.basename(filepath)
    checkpoint_match = re.search(r'checkpoint_(ep\d+)', filename)
    if checkpoint_match:
        metrics['checkpoint'] = checkpoint_match.group(1)

    # Mean P&L (%)
    pnl_match = re.search(r'Mean P\&L \(%\):\s+([-\d.]+)%', content)
    if pnl_match:
        metrics['pnl_pct'] = float(pnl_match.group(1))

    # Calculate APR from P&L % (2.9 days test period)
    if 'pnl_pct' in metrics:
        metrics['apr'] = (metrics['pnl_pct'] / 2.9) * 365

    # Total Trades
    trades_match = re.search(r'Total Trades:\s+(\d+)', content)
    if trades_match:
        metrics['trades'] = int(trades_match.group(1))

    # Win Rate
    win_rate_match = re.search(r'Win Rate:\s+([-\d.]+)%', content)
    if win_rate_match:
        metrics['win_rate'] = float(win_rate_match.group(1))

    # Average Duration
    avg_hold_match = re.search(r'Avg Duration:\s+([-\d.]+)\s+hours', content)
    if avg_hold_match:
        metrics['avg_hold_hours'] = float(avg_hold_match.group(1))

    # Composite Score
    composite_match = re.search(r'Composite Score:\s+([-\d.]+)', content)
    if composite_match:
        metrics['composite_score'] = float(composite_match.group(1))

    # Return metrics if we have the essentials
    if 'pnl_pct' in metrics and 'trades' in metrics:
        return metrics

    return None

def main():
    results_dir = Path('test_results')

    # Collect all metrics
    all_metrics = []
    for filepath in results_dir.glob('*_checkpoint_ep*_results.txt'):
        metrics = extract_metrics(filepath)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        print("No valid results found!")
        return

    # Sort by P&L % (descending)
    all_metrics.sort(key=lambda x: x.get('pnl_pct', -9999), reverse=True)

    # Display top 5
    print("\n" + "="*100)
    print("TOP 5 CHECKPOINTS BY P&L")
    print("="*100)
    print(f"{'Checkpoint':<15} {'P&L %':<12} {'Trades':<10} {'Win Rate':<12} {'Avg Hold Hours':<15}")
    print("-"*100)

    for i, metrics in enumerate(all_metrics[:5], 1):
        checkpoint = metrics.get('checkpoint', 'N/A')
        pnl_pct = metrics.get('pnl_pct', 0)
        trades = metrics.get('trades', 0)
        win_rate = metrics.get('win_rate', 0)
        avg_hold = metrics.get('avg_hold_hours', 0)

        print(f"{checkpoint:<15} {pnl_pct:>10.2f}%  {trades:<10} {win_rate:>9.2f}%   {avg_hold:>13.2f}")

    print("="*100)
    print(f"\nTotal checkpoints analyzed: {len(all_metrics)}")

    # Show all results sorted
    print("\n" + "="*100)
    print("ALL CHECKPOINTS (sorted by P&L %)")
    print("="*100)
    print(f"{'Checkpoint':<15} {'P&L %':<12} {'Trades':<10} {'Win Rate':<12} {'Avg Hold Hours':<15}")
    print("-"*100)

    for metrics in all_metrics:
        checkpoint = metrics.get('checkpoint', 'N/A')
        pnl_pct = metrics.get('pnl_pct', 0)
        trades = metrics.get('trades', 0)
        win_rate = metrics.get('win_rate', 0)
        avg_hold = metrics.get('avg_hold_hours', 0)

        print(f"{checkpoint:<15} {pnl_pct:>10.2f}%  {trades:<10} {win_rate:>9.2f}%   {avg_hold:>13.2f}")

    print("="*100)

if __name__ == '__main__':
    main()
