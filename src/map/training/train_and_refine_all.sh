#!/bin/bash
# Train remaining layer 6, then refine ALL layers 0-6
# This is a one-shot script for unattended execution

set -e  # Exit on error

# Go to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT/src"

PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "========================================"
echo "PHASE 1: Finish training layer 6"
echo "========================================"
echo "Working dir: $(pwd)"
echo "Python: $PYTHON"

"$PYTHON" -m training.train_concept_pack_lenses \
    --concept-pack first-light \
    --model swiss-ai/Apertus-8B-2509 \
    --output-dir ../lens_packs/apertus-8b_first-light \
    --training-manifest ../lens_packs/apertus-8b_first-light/training_manifest.json \
    --layers 6 \
    --no-sibling-refinement

echo ""
echo "========================================"
echo "PHASE 2: Sibling refinement on ALL layers (0-6)"
echo "========================================"

"$PYTHON" -m training.train_concept_pack_lenses \
    --concept-pack first-light \
    --model swiss-ai/Apertus-8B-2509 \
    --output-dir ../lens_packs/apertus-8b_first-light \
    --refine-only \
    --layers 0 1 2 3 4 5 6

echo ""
echo "========================================"
echo "COMPLETE"
echo "========================================"
echo "Training and refinement finished."
echo "Ready for calibration analysis."
