#!/bin/bash
# Quick progress checker

echo "=== Stage 1.5 Self-Diversity ==="
tail -n 3 stage1_5_1k.log 2>/dev/null || echo "Not started yet"

echo ""
echo "=== File Sizes ==="
ls -lh data/processed/encyclopedia_stage*.h5 2>/dev/null | tail -n 5

echo ""
echo "=== Running Processes ==="
ps aux | grep -E "(stage_1_5|train_interpreter)" | grep python | head -n 3
