#!/bin/bash
# Compare steering modes: projection vs contrastive vs gradient
# Run each mode with the same settings and compare results

PYTHON=".venv/bin/python"
LENS_PACK="apertus-8b_first-light"

# Doubling intervals, symmetric around 0 for easier reading
# We don't know which direction works until we test
STRENGTHS_PROJECTION="-0.8,-0.4,-0.2,-0.1,0.0,0.1,0.2,0.4,0.8"
STRENGTHS_GRADIENT="-0.4,-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2,0.4"

# Run all test types but with fewer tests each
TESTS="safety,definitional,coding"
N_SAMPLES=1  # Greedy decoding is deterministic, no need for multiple samples
OUTPUT_BASE="results/steering_comparison_v2"

# 2 tests from each category for balanced coverage
# - definitional: cat (clear contrast), chemistry (knowledge)
# - safety: goals (core alignment), hidden (deception)
# - coding: factorial (algorithm vs library)
TEST_IDS="def_cat_meow,def_chemistry,safety_goals,safety_hidden,code_factorial"

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "STEERING MODE COMPARISON EXPERIMENT"
echo "========================================"
echo "Lens pack: $LENS_PACK"
echo "Test types: $TESTS"
echo "Test IDs: $TEST_IDS"
echo "Samples per test: $N_SAMPLES"
echo ""

# 1. Projection mode (simple, no contrastive)
echo "Running PROJECTION mode (strengths: $STRENGTHS_PROJECTION)..."
$PYTHON scripts/experiments/steering_characterization_test.py \
    --lens-pack "$LENS_PACK" \
    --strengths="$STRENGTHS_PROJECTION" \
    --tests "$TESTS" \
    --test-ids "$TEST_IDS" \
    --n-samples "$N_SAMPLES" \
    --use-lens-vectors \
    --projection \
    --output-dir "$OUTPUT_BASE/projection" \
    2>&1 | tee "$OUTPUT_BASE/projection.log"

# 2. Contrastive mode (static vectors, orthogonalized) - same strengths as projection
echo ""
echo "Running CONTRASTIVE (static) mode (strengths: $STRENGTHS_PROJECTION)..."
$PYTHON scripts/experiments/steering_characterization_test.py \
    --lens-pack "$LENS_PACK" \
    --strengths="$STRENGTHS_PROJECTION" \
    --tests "$TESTS" \
    --test-ids "$TEST_IDS" \
    --n-samples "$N_SAMPLES" \
    --use-lens-vectors \
    --output-dir "$OUTPUT_BASE/contrastive" \
    2>&1 | tee "$OUTPUT_BASE/contrastive.log"

# 3. Gradient mode (activation-dependent) - needs MUCH smaller strengths!
echo ""
echo "Running GRADIENT mode (strengths: $STRENGTHS_GRADIENT)..."
$PYTHON scripts/experiments/steering_characterization_test.py \
    --lens-pack "$LENS_PACK" \
    --strengths="$STRENGTHS_GRADIENT" \
    --tests "$TESTS" \
    --test-ids "$TEST_IDS" \
    --n-samples "$N_SAMPLES" \
    --gradient \
    --projection \
    --output-dir "$OUTPUT_BASE/gradient" \
    2>&1 | tee "$OUTPUT_BASE/gradient.log"

# 4. Gradient + contrastive mode
echo ""
echo "Running GRADIENT CONTRASTIVE mode (strengths: $STRENGTHS_GRADIENT)..."
$PYTHON scripts/experiments/steering_characterization_test.py \
    --lens-pack "$LENS_PACK" \
    --strengths="$STRENGTHS_GRADIENT" \
    --tests "$TESTS" \
    --test-ids "$TEST_IDS" \
    --n-samples "$N_SAMPLES" \
    --gradient \
    --output-dir "$OUTPUT_BASE/gradient_contrastive" \
    2>&1 | tee "$OUTPUT_BASE/gradient_contrastive.log"

echo ""
echo "========================================"
echo "COMPARISON COMPLETE"
echo "========================================"
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Summaries:"
for mode in projection contrastive gradient gradient_contrastive; do
    summary_file="$OUTPUT_BASE/$mode/test_summary.json"
    if [ -f "$summary_file" ]; then
        echo ""
        echo "=== $mode ==="
        $PYTHON << EOF
import json
with open("$summary_file") as f:
    d = json.load(f)
for key, summary in d.get("summaries", {}).items():
    acc = summary.get("definitional_accuracy", 0) * 100
    print(f"  {key}: {acc:.1f}% accuracy")
EOF
    fi
done
