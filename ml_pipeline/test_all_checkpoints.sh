#!/bin/bash

# Script to test all checkpoints in v3-3 and v3-4 folders in parallel
# Usage: ./test_all_checkpoints.sh

set -e

# Configuration
TEST_DATA_PATH="data/production/rl_opportunities.csv"
PRICE_HISTORY_PATH="data/production/symbol_data"
LEVERAGE=2
OUTPUT_DIR="test_results"
MAX_PARALLEL=4  # Number of tests to run in parallel

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Testing All Checkpoints in v3-3 and v3-4${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to test a single checkpoint
test_checkpoint() {
    local checkpoint_path=$1
    local checkpoint_name=$(basename "$checkpoint_path" .pt)
    local folder_name=$(basename $(dirname "$checkpoint_path"))
    local output_file="${OUTPUT_DIR}/${folder_name}_${checkpoint_name}_results.txt"

    echo -e "${YELLOW}Testing: ${folder_name}/${checkpoint_name}${NC}"

    # Run test and capture output
    python test_inference.py \
        --checkpoint "$checkpoint_path" \
        --test-data-path "$TEST_DATA_PATH" \
        --price-history-path "$PRICE_HISTORY_PATH" \
        --leverage "$LEVERAGE" \
        > "$output_file" 2>&1

    echo -e "${GREEN}✓ Completed: ${folder_name}/${checkpoint_name}${NC}"
}

export -f test_checkpoint
export TEST_DATA_PATH PRICE_HISTORY_PATH LEVERAGE OUTPUT_DIR GREEN YELLOW NC

# Collect all checkpoints
checkpoints=()

if [ -d "checkpoints/v3-3" ]; then
    for checkpoint in checkpoints/v3-3/checkpoint_ep*.pt; do
        if [ -f "$checkpoint" ]; then
            checkpoints+=("$checkpoint")
        fi
    done
fi

if [ -d "checkpoints/v3-4" ]; then
    for checkpoint in checkpoints/v3-4/checkpoint_ep*.pt; do
        if [ -f "$checkpoint" ]; then
            checkpoints+=("$checkpoint")
        fi
    done
fi

total_checkpoints=${#checkpoints[@]}
echo -e "${BLUE}Found $total_checkpoints checkpoints to test${NC}"
echo -e "${BLUE}Running $MAX_PARALLEL tests in parallel${NC}"
echo ""

# Run tests in parallel using xargs
printf '%s\n' "${checkpoints[@]}" | xargs -n 1 -P $MAX_PARALLEL -I {} bash -c 'test_checkpoint "$@"' _ {}

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}All checkpoints tested!${NC}"
echo -e "${BLUE}Results saved in: ${OUTPUT_DIR}/${NC}"
echo -e "${BLUE}================================================${NC}"

# Create summary
echo ""
echo -e "${BLUE}Generating summary...${NC}"
echo ""

summary_file="${OUTPUT_DIR}/summary.txt"
echo "Checkpoint Test Summary" > "$summary_file"
echo "======================" >> "$summary_file"
echo "Generated: $(date)" >> "$summary_file"
echo "" >> "$summary_file"

# Extract key metrics from each result file (sorted by checkpoint name)
for result_file in $(ls ${OUTPUT_DIR}/*_results.txt | sort -V); do
    if [ -f "$result_file" ]; then
        checkpoint_name=$(basename "$result_file" _results.txt)
        echo "--- $checkpoint_name ---" >> "$summary_file"

        # Extract final stats section
        sed -n '/Final Portfolio Stats/,/Test Results:/p' "$result_file" >> "$summary_file" 2>/dev/null || echo "No results found" >> "$summary_file"
        echo "" >> "$summary_file"
    fi
done

echo -e "${GREEN}Summary saved to: $summary_file${NC}"
echo ""
echo -e "${BLUE}Top 5 checkpoints by P&L:${NC}"
grep "Total P&L:" "$summary_file" | sort -t: -k2 -rn | head -5
