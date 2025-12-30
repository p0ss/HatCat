#!/usr/bin/env bash
# Helper script to run training with timestamped logs in logs/ directory

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/training_${TIMESTAMP}.log"

echo "Starting training... Log file: $LOGFILE"

# Activate venv and run training with all passed arguments
. .venv/bin/activate
nohup python scripts/train_sumo_classifiers.py "$@" > "$LOGFILE" 2>&1 &

# Get the PID
PID=$!

echo "Training started with PID: $PID"
echo "Monitor progress with: tail -f $LOGFILE"
echo "Stop training with: kill $PID"
