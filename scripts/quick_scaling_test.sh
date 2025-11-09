#!/bin/bash
#
# Quick scaling test: Compare definitional vs relational approaches
#
# Key comparison:
#   10 concepts × (1 def + 9 rels)  vs  10 concepts × 10 defs
#

set -e

CONCEPT_GRAPH="data/concept_graph/wordnet_v2_top10.json"
MODEL="google/gemma-3-4b-pt"
OUTPUT_DIR="results/scaling_quick"

mkdir -p "$OUTPUT_DIR"

echo "======================================================================="
echo "QUICK SCALING TEST"
echo "======================================================================="
echo ""
echo "Testing:"
echo "  1. 10 concepts × (1 def + 9 rels)"
echo "  2. 10 concepts × 10 defs"
echo ""

# Test 1: 10 concepts × (1 def + 9 rels)
echo "Running Test 1: 10 concepts × (1 def + 9 rels)..."
poetry run python scripts/scaling_study.py \
  --concept-graph "$CONCEPT_GRAPH" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --single \
  --n-concepts 10 \
  --n-definitions 1 \
  --n-relationships 9

# Test 2: 10 concepts × 10 defs
echo ""
echo "Running Test 2: 10 concepts × 10 defs (no relationships)..."
poetry run python scripts/scaling_study.py \
  --concept-graph "$CONCEPT_GRAPH" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --single \
  --n-concepts 10 \
  --n-definitions 10 \
  --n-relationships 0

echo ""
echo "======================================================================="
echo "COMPARISON"
echo "======================================================================="

# Extract and compare results
TEST1=$(cat "$OUTPUT_DIR/scaling_c10_d1_r9.json" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['results']['mean_val_acc']:.1%}\")")
TEST2=$(cat "$OUTPUT_DIR/scaling_c10_d10_r0.json" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['results']['mean_val_acc']:.1%}\")")

echo ""
echo "  10 concepts × (1 def + 9 rels): $TEST1"
echo "  10 concepts × 10 defs:           $TEST2"
echo ""

if [ $(echo "$TEST1 > $TEST2" | bc -l) -eq 1 ]; then
    echo "✓ Relational approach wins!"
else
    echo "✓ Definitional approach wins!"
fi
