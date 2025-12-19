#!/bin/bash
# Production calibration script with proper logging

LOG_FILE="/tmp/calibration_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

cd /home/poss/Documents/Code/HatCat

.venv/bin/python -m src.training.calibration.cycle \
    --lens-pack lens_packs/apertus-8b_first-light_calibration-test \
    --concept-pack concept_packs/first-light \
    --model swiss-ai/Apertus-8B-2509 \
    --production \
    --max-cycles 3 \
    --max-finetune-epochs 30 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Calibration complete. Log saved to: $LOG_FILE"
