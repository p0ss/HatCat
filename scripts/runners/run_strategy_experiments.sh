#!/bin/bash
# Run all three adaptive scaling strategies on 100-concept set

MODEL="google/gemma-3-4b-pt"
GRAPH="data/concept_graph/wordnet_v2_top100.json"
N_CONCEPTS=100

echo "========================================================================"
echo "ADAPTIVE SCALING STRATEGY EXPERIMENTS"
echo "========================================================================"
echo "Testing 3 strategies on 100 concepts:"
echo "  1. SYMMETRIC: X(C+R) - 1 def + 1 rel per iteration"
echo "  2. HALF-SCALED: X(C(N/2)) - max(1, N/2) defs per iteration"
echo "  3. RELFIRST-PURE: X(C*N) - N defs per iteration"
echo ""

# Strategy 1: Symmetric
echo "Starting SYMMETRIC strategy..."
poetry run python scripts/adaptive_scaling_strategies.py \
  --concept-graph "$GRAPH" \
  --model "$MODEL" \
  --strategy symmetric \
  --output-dir results/strategy_symmetric_100 \
  --n-concepts "$N_CONCEPTS" \
  --target-accuracy 0.95 \
  --max-data-size 200 \
  --memory-threshold 2.0 \
  --device cuda \
  > results/strategy_symmetric_100.log 2>&1

echo "✓ SYMMETRIC complete"
echo ""

# Strategy 2: Half-scaled
echo "Starting HALF-SCALED strategy..."
poetry run python scripts/adaptive_scaling_strategies.py \
  --concept-graph "$GRAPH" \
  --model "$MODEL" \
  --strategy half-scaled \
  --output-dir results/strategy_halfscaled_100 \
  --n-concepts "$N_CONCEPTS" \
  --target-accuracy 0.95 \
  --max-data-size 200 \
  --memory-threshold 2.0 \
  --device cuda \
  > results/strategy_halfscaled_100.log 2>&1

echo "✓ HALF-SCALED complete"
echo ""

# Strategy 3: RelFirst-Pure
echo "Starting RELFIRST-PURE strategy..."
poetry run python scripts/adaptive_scaling_strategies.py \
  --concept-graph "$GRAPH" \
  --model "$MODEL" \
  --strategy relfirst-pure \
  --output-dir results/strategy_relfirstpure_100 \
  --n-concepts "$N_CONCEPTS" \
  --target-accuracy 0.95 \
  --max-data-size 200 \
  --memory-threshold 2.0 \
  --device cuda \
  > results/strategy_relfirstpure_100.log 2>&1

echo "✓ RELFIRST-PURE complete"
echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Results:"
echo "  - results/strategy_symmetric_100/"
echo "  - results/strategy_halfscaled_100/"
echo "  - results/strategy_relfirstpure_100/"
echo ""
echo "Logs:"
echo "  - results/strategy_symmetric_100.log"
echo "  - results/strategy_halfscaled_100.log"
echo "  - results/strategy_relfirstpure_100.log"
