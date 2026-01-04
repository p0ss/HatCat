#!/bin/bash
# FineLlama-3.1-8B Lens Pack Training
# Designed for multi-day unattended run

set -e

# Configuration
MODEL="mlabonne/FineLlama-3.1-8B"
CONCEPT_PACK="concept_packs/first-light"
RUN_NAME="finellama-3.1-8b_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="lens_packs/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training parameters - balanced for quality vs time
N_TRAIN_POS=50
N_TRAIN_NEG=50
N_TEST_POS=20
N_TEST_NEG=20
VALIDATION_MODE="falloff"

# Change to project directory
cd /home/poss/Documents/Code/HatCatDev

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "  FineLlama-3.1-8B Lens Pack Training" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "Concept Pack: ${CONCEPT_PACK}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Samples per concept: ${N_TRAIN_POS}+ / ${N_TRAIN_NEG}- train, ${N_TEST_POS}+ / ${N_TEST_NEG}- test" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Activate virtual environment
source .venv/bin/activate

# Set CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
echo "Starting training at $(date)..." | tee -a "${LOG_FILE}"
python scripts/training/train_full_lens_pack.py \
    --model "${MODEL}" \
    --concept-pack "${CONCEPT_PACK}" \
    --device cuda \
    --layers 0 1 2 3 4 5 \
    --n-train-pos ${N_TRAIN_POS} \
    --n-train-neg ${N_TRAIN_NEG} \
    --n-test-pos ${N_TEST_POS} \
    --n-test-neg ${N_TEST_NEG} \
    --validation-mode ${VALIDATION_MODE} \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "  Training Complete" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Finished: $(date)" | tee -a "${LOG_FILE}"
echo "Results saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
