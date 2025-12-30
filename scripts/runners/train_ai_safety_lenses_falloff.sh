#!/usr/bin/env bash
#
# Train AI safety lenses with falloff validation method
#
# This trains 19 AI safety concepts from layer 4 using:
# - DualAdaptiveTrainer with independent graduation
# - Falloff validation for proper calibration
# - Definitional prompts (baseline for comparison with behavioral training later)
#

set -euo pipefail

# Activate environment
. .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Output directory
OUTPUT_DIR="results/ai_safety_lenses_falloff"
LOG_FILE="logs/train_ai_safety_falloff.log"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=================================="
echo "AI Safety Lens Training (Falloff)"
echo "=================================="
echo "Method: DualAdaptiveTrainer + Falloff Validation"
echo "Concepts: 19 AI safety concepts from layer 4"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=================================="
echo

# List AI safety concepts
echo "Training concepts:"
cat data/concept_graph/ai_safety_layer_entries/layer4_children.json | jq -r '.[].sumo_term' | nl
echo

# Train layer 4 (which contains only AI safety concepts)
# Using existing script - layer 4 contains the 19 AI safety concepts
python scripts/train_sumo_classifiers.py \
    --layers 4 \
    --use-adaptive-training \
    --validation-mode falloff \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo "Results: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo
echo "Next steps:"
echo "  1. Verify lens calibration in ${OUTPUT_DIR}"
echo "  2. Compare with behavioral training results (when available)"
echo "  3. Create v4 lens pack merging SUMO + AI safety lenses"
echo
