#!/bin/bash
# Test inter-run variability of tripole lenses

for i in {1..5}; do
  echo "=========================================="
  echo "Run $i/5"
  echo "=========================================="
  poetry run python scripts/test_tripole_single_simplex.py 2>&1 | grep -A 20 "^RESULTS"
  echo ""
done
