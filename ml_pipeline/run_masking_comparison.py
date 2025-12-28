#!/usr/bin/env python3
"""
Run checkpoint tests with different masking configurations.
Compares: No masking vs APR only vs APR + 30min funding
"""

import subprocess
import os
import sys
from pathlib import Path

CHECKPOINT_DIR = "checkpoints/v10_5000"
TEST_DATA = "data/production/rl_opportunities.csv"
START_TIME = "2025-12-24 09:40:00"
END_TIME = "2025-12-27 17:20:00"
INITIAL_CAPITAL = 100

def run_single_test(checkpoint_path: str) -> dict:
    """Run a single checkpoint test and return results."""
    cmd = [
        "python", "test_inference.py",
        "--checkpoint", checkpoint_path,
        "--test-data-path", TEST_DATA,
        "--start-time", START_TIME,
        "--end-time", END_TIME,
        "--initial-capital", str(INITIAL_CAPITAL)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Parse results
        pnl = None
        trades = None
        winrate = None

        for line in output.split('\n'):
            if 'Total P&L:' in line and '$' in line:
                try:
                    pnl = float(line.split('$')[1].split()[0])
                except:
                    pass
            elif 'Total Trades:' in line:
                try:
                    trades = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Win Rate:' in line and '%' in line:
                try:
                    winrate = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass

        return {'pnl': pnl, 'trades': trades, 'winrate': winrate}
    except Exception as e:
        return {'pnl': None, 'trades': None, 'winrate': None, 'error': str(e)}

def update_masking(enable_apr: bool, enable_time: bool):
    """Update masking configuration in unified_feature_builder.py"""

    filepath = Path("common/features/unified_feature_builder.py")
    content = filepath.read_text()

    # Find the get_action_mask method and update it
    if enable_apr and enable_time:
        # Full masking: APR + time to funding
        new_mask_code = '''        mask = np.zeros(DIMS.TOTAL_ACTIONS, dtype=bool)

        # HOLD is always valid
        mask[DIMS.ACTION_HOLD] = True

        # ENTER actions with APR and time masking
        effective_max_positions = min(max_positions, DIMS.EXECUTIONS_SLOTS)
        has_capacity = num_positions < effective_max_positions

        if has_capacity:
            for i in range(DIMS.OPPORTUNITIES_SLOTS):
                if i < len(opportunities):
                    opp = opportunities[i]

                    if opp.get('has_existing_position', False):
                        continue

                    # APR masking
                    fund_apr = opp.get('fund_apr', 0.0)
                    if fund_apr < min_apr:
                        continue

                    # Time to funding masking
                    time_to_funding_norm = self._calc_time_to_profitable_funding(opp, current_time)
                    minutes_to_funding = time_to_funding_norm * 480.0
                    if minutes_to_funding > max_minutes_to_funding:
                        continue

                    mask[1 + i] = True
                    mask[6 + i] = True
                    mask[11 + i] = True

        # EXIT actions
        for i in range(DIMS.EXECUTIONS_SLOTS):
            if i < num_positions:
                mask[DIMS.ACTION_EXIT_START + i] = True

        return mask'''
    elif enable_apr:
        # APR masking only
        new_mask_code = '''        mask = np.zeros(DIMS.TOTAL_ACTIONS, dtype=bool)

        # HOLD is always valid
        mask[DIMS.ACTION_HOLD] = True

        # ENTER actions with APR masking only
        effective_max_positions = min(max_positions, DIMS.EXECUTIONS_SLOTS)
        has_capacity = num_positions < effective_max_positions

        if has_capacity:
            for i in range(DIMS.OPPORTUNITIES_SLOTS):
                if i < len(opportunities):
                    opp = opportunities[i]

                    if opp.get('has_existing_position', False):
                        continue

                    # APR masking only
                    fund_apr = opp.get('fund_apr', 0.0)
                    if fund_apr < min_apr:
                        continue

                    mask[1 + i] = True
                    mask[6 + i] = True
                    mask[11 + i] = True

        # EXIT actions
        for i in range(DIMS.EXECUTIONS_SLOTS):
            if i < num_positions:
                mask[DIMS.ACTION_EXIT_START + i] = True

        return mask'''
    else:
        # No masking
        new_mask_code = '''        mask = np.zeros(DIMS.TOTAL_ACTIONS, dtype=bool)

        # HOLD is always valid
        mask[DIMS.ACTION_HOLD] = True

        # MASKING DISABLED - Allow all actions based only on basic constraints
        effective_max_positions = min(max_positions, DIMS.EXECUTIONS_SLOTS)
        has_capacity = num_positions < effective_max_positions

        if has_capacity:
            for i in range(DIMS.OPPORTUNITIES_SLOTS):
                if i < len(opportunities):
                    opp = opportunities[i]

                    if opp.get('has_existing_position', False):
                        continue

                    mask[1 + i] = True
                    mask[6 + i] = True
                    mask[11 + i] = True

        # EXIT actions
        for i in range(DIMS.EXECUTIONS_SLOTS):
            if i < num_positions:
                mask[DIMS.ACTION_EXIT_START + i] = True

        return mask'''

    # Find and replace the mask generation code
    import re
    pattern = r'(        mask = np\.zeros\(DIMS\.TOTAL_ACTIONS.*?return mask)'
    content = re.sub(pattern, new_mask_code, content, flags=re.DOTALL)

    filepath.write_text(content)
    print(f"Updated masking: APR={enable_apr}, Time={enable_time}")

def run_all_checkpoints(config_name: str) -> dict:
    """Run all checkpoints and return results."""
    results = {}
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("*.pt"))
    total = len(checkpoints)

    print(f"\n{'='*60}")
    print(f"Testing {total} checkpoints with config: {config_name}")
    print(f"{'='*60}")

    for i, ckpt in enumerate(checkpoints):
        result = run_single_test(str(ckpt))
        results[ckpt.name] = result

        if result['pnl'] is not None:
            print(f"[{i+1}/{total}] {ckpt.name}: PNL={result['pnl']:.2f}%, WinRate={result['winrate']:.1f}%, Trades={result['trades']}")
        else:
            print(f"[{i+1}/{total}] {ckpt.name}: FAILED")

    return results

def main():
    os.chdir(Path(__file__).parent)

    all_results = {}

    # Test 1: No masking
    print("\n" + "="*80)
    print("TEST 1: NO MASKING")
    print("="*80)
    update_masking(enable_apr=False, enable_time=False)
    all_results['no_masking'] = run_all_checkpoints("No Masking")

    # Test 2: APR masking only
    print("\n" + "="*80)
    print("TEST 2: APR MASKING ONLY (min 2500%)")
    print("="*80)
    update_masking(enable_apr=True, enable_time=False)
    all_results['apr_only'] = run_all_checkpoints("APR Only")

    # Test 3: APR + Time masking
    print("\n" + "="*80)
    print("TEST 3: APR + TIME MASKING (min 2500%, max 30min)")
    print("="*80)
    update_masking(enable_apr=True, enable_time=True)
    all_results['apr_and_time'] = run_all_checkpoints("APR + Time")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON: TOP 10 BY P&L FOR EACH CONFIG")
    print("="*80)

    for config_name, results in all_results.items():
        valid = [(k, v) for k, v in results.items() if v['pnl'] is not None]
        sorted_results = sorted(valid, key=lambda x: x[1]['pnl'], reverse=True)[:10]

        print(f"\n{config_name.upper()}:")
        print(f"{'Checkpoint':<30} {'P&L%':>10} {'WinRate%':>10} {'Trades':>8}")
        print("-" * 60)
        for name, r in sorted_results:
            print(f"{name:<30} {r['pnl']:>10.2f} {r['winrate']:>10.1f} {r['trades']:>8}")

    # Save results to file
    with open("masking_comparison_results.txt", "w") as f:
        for config_name, results in all_results.items():
            valid = [(k, v) for k, v in results.items() if v['pnl'] is not None]
            sorted_results = sorted(valid, key=lambda x: x[1]['pnl'], reverse=True)

            f.write(f"\n{'='*60}\n{config_name.upper()}\n{'='*60}\n")
            for name, r in sorted_results:
                f.write(f"{name}|{r['pnl']:.2f}|{r['winrate']:.1f}|{r['trades']}\n")

    print("\nResults saved to masking_comparison_results.txt")

if __name__ == "__main__":
    main()
