#!/bin/bash

LOG_FILE="training_v3.log"
LAST_SIZE=0

echo "🔍 Monitoring training progress..."
echo "Waiting for Episode 50 evaluation..."
echo ""

while true; do
    # Check if process is still running
    if ! ps aux | grep "train_ppo.py" | grep -q "10230"; then
        echo "⚠️  Training process stopped!"
        exit 1
    fi
    
    # Check if log has new content
    CURRENT_SIZE=$(wc -c < "$LOG_FILE" 2>/dev/null || echo "0")
    
    if [ "$CURRENT_SIZE" -gt "$LAST_SIZE" ]; then
        # New content available
        tail -50 "$LOG_FILE" | tail -20
        echo "---"
        
        # Check if Episode 50 evaluation appeared
        if grep -q "EVALUATION ON TEST SET (Episode 50)" "$LOG_FILE"; then
            echo ""
            echo "✅ Episode 50 evaluation found!"
            grep -A 40 "EVALUATION ON TEST SET (Episode 50)" "$LOG_FILE"
            exit 0
        fi
    fi
    
    LAST_SIZE=$CURRENT_SIZE
    sleep 60  # Check every minute
done
