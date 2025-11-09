#!/bin/bash
# Overnight pipeline: 50K concepts × 20 templates
# Expected runtime: ~6-8 hours

set -e  # Exit on error

STAGE0_PATH="data/processed/encyclopedia_stage0_50k.h5"
STAGE1_PATH="data/processed/encyclopedia_stage1_50k_x20.h5"
MODEL_DIR="models/stage1_50k"

echo "========================================================================"
echo "OVERNIGHT PIPELINE: 50K CONCEPTS × 20 TEMPLATES"
echo "========================================================================"
echo "Start time: $(date)"
echo ""
echo "Pipeline:"
echo "  1. Bootstrap 50K Stage 0 (~2-3 hours)"
echo "  2. Refine to Stage 1 with 20 templates (~3-4 hours)"
echo "  3. Train interpreter (~1-2 hours)"
echo ""
echo "Expected completion: 6-8 hours"
echo "========================================================================"
echo ""

# Step 1: Bootstrap 50K concepts
echo "STEP 1/3: Bootstrapping 50K Stage 0 encyclopedia..."
echo "Output: $STAGE0_PATH"
echo ""
poetry run python scripts/stage_0_bootstrap.py \
    --n-concepts 50000 \
    --output "$STAGE0_PATH" \
    --layers -1 \
    --device cuda

echo ""
echo "✓ Stage 0 complete"
echo ""

# Step 2: Refine with 20 templates
echo "STEP 2/3: Refining to Stage 1 with 20 templates per concept..."
echo "Output: $STAGE1_PATH"
echo ""
poetry run python scripts/stage_1_refinement.py \
    --input "$STAGE0_PATH" \
    --output "$STAGE1_PATH" \
    --n-samples 20 \
    --layer -1 \
    --device cuda

echo ""
echo "✓ Stage 1 complete"
echo ""

# Step 3: Train interpreter
echo "STEP 3/3: Training interpreter on 1M samples..."
echo "Output: $MODEL_DIR"
echo ""
poetry run python scripts/train_interpreter.py \
    --data "$STAGE1_PATH" \
    --batch-size 128 \
    --epochs 20 \
    --lr 1e-3 \
    --output-dir "$MODEL_DIR" \
    --accelerator cuda \
    --devices 1

echo ""
echo "========================================================================"
echo "✓ PIPELINE COMPLETE"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Stage 0: $STAGE0_PATH"
echo "  - Stage 1: $STAGE1_PATH"
echo "  - Model: $MODEL_DIR"
echo ""
echo "Check training metrics:"
echo "  cat logs/interpreter/version_*/metrics.csv | tail -n 5"
