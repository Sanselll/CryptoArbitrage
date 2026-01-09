#!/usr/bin/env python3
"""
Test all checkpoints in a directory on full data range.
Outputs summary CSV with P&L, trades, and metrics for each checkpoint.
Uses parallel processing for speed.
"""

import subprocess
import sys
import os
import re
import csv
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_metrics_from_output(output: str) -> dict:
    """Extract key metrics from test_inference.py output."""
    metrics = {
        'pnl_usd': 0.0,
        'pnl_pct': 0.0,
        'trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'max_drawdown': 0.0,
        'composite_score': 0.0,
        'avg_trade_pnl': 0.0,
    }

    # Extract P&L (format: "P&L=$ 190.97 (19.10%)")
    pnl_match = re.search(r'P&L=\$\s*([-\d.,]+)\s*\(([-\d.]+)%\)', output)
    if pnl_match:
        metrics['pnl_usd'] = float(pnl_match.group(1).replace(',', ''))
        metrics['pnl_pct'] = float(pnl_match.group(2))

    # Extract trades
    trades_match = re.search(r'Total trades:\s*(\d+)', output)
    if trades_match:
        metrics['trades'] = int(trades_match.group(1))

    # Extract win rate
    win_match = re.search(r'Win Rate:\s*([\d.]+)%', output)
    if win_match:
        metrics['win_rate'] = float(win_match.group(1))

    # Extract profit factor
    pf_match = re.search(r'Profit Factor:\s*([\d.]+|inf)', output)
    if pf_match:
        val = pf_match.group(1)
        metrics['profit_factor'] = 999.0 if val == 'inf' else float(val)

    # Extract max drawdown
    dd_match = re.search(r'Max Drawdown:\s*([\d.]+)%', output)
    if dd_match:
        metrics['max_drawdown'] = float(dd_match.group(1))

    # Extract composite score
    score_match = re.search(r'Composite Score:\s*([\d.]+)', output)
    if score_match:
        metrics['composite_score'] = float(score_match.group(1))

    # Calculate average trade P&L
    if metrics['trades'] > 0:
        metrics['avg_trade_pnl'] = metrics['pnl_usd'] / metrics['trades']

    return metrics


def test_checkpoint_worker(args_tuple):
    """Worker function for parallel checkpoint testing."""
    checkpoint_path, data_path, price_history_path, start_time, end_time, cwd = args_tuple

    cmd = [
        'python', 'test_inference.py',
        '--test-data-path', data_path,
        '--price-history-path', price_history_path,
        '--leverage', '2',
        '--initial-capital', '1000',
        '--checkpoint', checkpoint_path,
        '--no-early-termination',
    ]
    if start_time:
        cmd.extend(['--start-time', start_time])
    if end_time:
        cmd.extend(['--end-time', end_time])

    checkpoint_name = Path(checkpoint_path).name

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per checkpoint
            cwd=cwd
        )
        output = result.stdout + result.stderr
        metrics = extract_metrics_from_output(output)
        metrics['status'] = 'success' if result.returncode == 0 else 'error'
        metrics['error'] = '' if result.returncode == 0 else result.stderr[:200]
    except subprocess.TimeoutExpired:
        metrics = {'status': 'timeout', 'error': 'Timeout after 600s'}
    except Exception as e:
        metrics = {'status': 'error', 'error': str(e)[:200]}

    metrics['checkpoint'] = checkpoint_name

    # Extract episode number
    ep_match = re.search(r'ep(\d+)', checkpoint_name)
    metrics['episode'] = int(ep_match.group(1)) if ep_match else 0

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test all checkpoints in a directory')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--test-data-path', type=str, default='data/production/rl_opportunities.csv',
                        help='Path to test data CSV')
    parser.add_argument('--price-history-path', type=str, default='data/production/price_history',
                        help='Path to price history directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: checkpoint_results_<timestamp>.csv)')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Start time for filtering test data (e.g., "2026-01-01 00:00:00")')
    parser.add_argument('--end-time', type=str, default=None,
                        help='End time for filtering test data (e.g., "2026-01-05 23:59:59")')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    # Find all checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_ep*.pt'),
                        key=lambda x: int(re.search(r'ep(\d+)', x.name).group(1)))

    # Add best_model and final_model if they exist
    for special in ['best_model.pt', 'final_model.pt']:
        special_path = checkpoint_dir / special
        if special_path.exists():
            checkpoints.append(special_path)

    num_workers = args.workers or min(cpu_count(), 8)  # Cap at 8 workers

    print(f"Found {len(checkpoints)} checkpoints to test")
    print(f"Data: {args.test_data_path}")
    print(f"Price history: {args.price_history_path}")
    if args.start_time or args.end_time:
        print(f"Date range: {args.start_time or 'start'} to {args.end_time or 'end'}")
    print(f"Workers: {num_workers} (parallel)")
    print("=" * 80)

    # Output file
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'checkpoint_results_{timestamp}.csv'

    # CSV fieldnames
    fieldnames = ['checkpoint', 'episode', 'status', 'pnl_usd', 'pnl_pct', 'trades',
                  'win_rate', 'profit_factor', 'max_drawdown', 'composite_score',
                  'avg_trade_pnl', 'error']

    # Prepare worker arguments
    cwd = os.path.dirname(os.path.abspath(__file__))
    work_items = [
        (str(cp), args.test_data_path, args.price_history_path,
         args.start_time, args.end_time, cwd)
        for cp in checkpoints
    ]

    # Run in parallel
    print(f"\nTesting {len(checkpoints)} checkpoints with {num_workers} workers...\n")

    results = []
    completed = 0

    with Pool(num_workers) as pool:
        for metrics in pool.imap_unordered(test_checkpoint_worker, work_items):
            completed += 1
            results.append(metrics)

            # Print progress
            if metrics.get('status') == 'success':
                print(f"[{completed}/{len(checkpoints)}] {metrics['checkpoint']}: "
                      f"P&L=${metrics['pnl_usd']:.2f} ({metrics['pnl_pct']:.1f}%) | "
                      f"Trades: {metrics['trades']} | Score: {metrics['composite_score']:.3f}", flush=True)
            else:
                print(f"[{completed}/{len(checkpoints)}] {metrics['checkpoint']}: "
                      f"{metrics.get('status')} - {metrics.get('error', '')[:50]}", flush=True)

    # Sort results by episode number for CSV
    results.sort(key=lambda x: x.get('episode', 0))

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")

    # Print top 5 by composite score
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        top5 = sorted(successful, key=lambda x: x.get('composite_score', 0), reverse=True)[:5]
        print("\nTop 5 by Composite Score:")
        for r in top5:
            print(f"  {r['checkpoint']}: Score={r['composite_score']:.3f}, "
                  f"P&L=${r['pnl_usd']:.2f} ({r['pnl_pct']:.1f}%), Trades={r['trades']}")

        # Also show top 5 by P&L
        top5_pnl = sorted(successful, key=lambda x: x.get('pnl_pct', 0), reverse=True)[:5]
        print("\nTop 5 by P&L %:")
        for r in top5_pnl:
            print(f"  {r['checkpoint']}: P&L={r['pnl_pct']:.1f}%, "
                  f"Score={r['composite_score']:.3f}, Trades={r['trades']}")


if __name__ == '__main__':
    main()
