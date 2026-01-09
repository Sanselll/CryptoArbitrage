#!/bin/bash

# Test all checkpoints in parallel and find the best models
# Usage: ./test_all_checkpoints.sh

set -e

CHECKPOINT_DIR="${1:-checkpoints_v5}"
RESULTS_FILE="checkpoint_results_$(basename $CHECKPOINT_DIR)_full.txt"
PARALLEL_JOBS=4

# Common test parameters
TEST_DATA="/Users/sansel/Projects/CryptoArbitrage/ml_pipeline/data/production/rl_opportunities.csv"
PRICE_HISTORY="data/production/price_history"
LEVERAGE=2
START_TIME="2025-12-03 01:20:00"
END_TIME="2026-01-08 09:50:00"
INITIAL_CAPITAL=1000

echo "========================================================================"
echo "TESTING ALL CHECKPOINTS IN PARALLEL"
echo "========================================================================"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Time range: $START_TIME to $END_TIME"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Clear previous results
> "$RESULTS_FILE"

# Find all checkpoint files
CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f | sort)
TOTAL=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')

echo "Found $TOTAL checkpoints to test"
echo ""

# Function to test a single checkpoint
test_checkpoint() {
    local checkpoint="$1"
    local basename=$(basename "$checkpoint")

    echo "[$(date '+%H:%M:%S')] Testing: $basename"

    # Run test and capture output
    local output=$(python test_inference.py \
        --test-data-path "$TEST_DATA" \
        --price-history-path "$PRICE_HISTORY" \
        --leverage "$LEVERAGE" \
        --start-time "$START_TIME" \
        --end-time "$END_TIME" \
        --initial-capital "$INITIAL_CAPITAL" \
        --stop-loss -0.10 \
        --checkpoint "$checkpoint" \
        2>&1)

    # Extract metrics from output
    local pnl=$(echo "$output" | grep "Mean P&L (%):" | grep -oE '[-+]?[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
    local winrate=$(echo "$output" | grep "Win Rate:" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
    local trades=$(echo "$output" | grep "Total Trades:" | grep -oE '[0-9]+' | head -1)
    local avg_hours=$(echo "$output" | grep "Avg Duration:" | grep -oE '[0-9]+\.[0-9]+' | head -1)

    # Handle empty values
    pnl=${pnl:-0.0}
    winrate=${winrate:-0.0}
    trades=${trades:-0}
    avg_hours=${avg_hours:-0.0}

    # Save results
    echo "$basename|$pnl|$winrate|$trades|$avg_hours" >> "$RESULTS_FILE"

    echo "[$(date '+%H:%M:%S')] ✓ $basename: PNL=$pnl%, WinRate=$winrate%, Trades=$trades"
}

# Export function and variables for parallel execution
export -f test_checkpoint
export TEST_DATA PRICE_HISTORY LEVERAGE START_TIME END_TIME INITIAL_CAPITAL RESULTS_FILE

# Run tests in parallel
echo "$CHECKPOINTS" | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'test_checkpoint "$@"' _ {}

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE - ANALYZING RESULTS"
echo "========================================================================"
echo ""

# Sort by PNL (descending) and display top 5
echo "TOP 5 MODELS BY P&L:"
echo "------------------------------------------------------------------------"
printf "%-30s %10s %10s %10s %12s\n" "CHECKPOINT" "P&L %" "WIN RATE %" "TRADES" "AVG HOURS"
echo "------------------------------------------------------------------------"

sort -t'|' -k2 -n -r "$RESULTS_FILE" | head -5 | while IFS='|' read -r checkpoint pnl winrate trades avg_hours; do
    printf "%-30s %9s%% %9s%% %10s %11sh\n" "$checkpoint" "$pnl" "$winrate" "$trades" "$avg_hours"
done

echo "------------------------------------------------------------------------"
echo ""
echo "Full results saved to: $RESULTS_FILE"
echo ""
