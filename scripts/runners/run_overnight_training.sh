#!/bin/bash
# Overnight training run for all SUMO abstraction layers
# With improved synset coverage (100% mapped) and adaptive training

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/overnight_training_${TIMESTAMP}"
LOG_FILE="overnight_training_${TIMESTAMP}.log"

echo "=================================="
echo "OVERNIGHT TRAINING RUN"
echo "=================================="
echo ""
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Layers: 0-5 (5,582 total concepts)"
echo "  Layer 0: 14 concepts"
echo "  Layer 1: 276 concepts"
echo "  Layer 2: 1,059 concepts"
echo "  Layer 3: 991 concepts"
echo "  Layer 4: 3,221 concepts"
echo "  Layer 5: 21 concepts"
echo ""
echo "Features:"
echo "  ✓ 100% synset coverage (527 new mappings)"
echo "  ✓ 96.5% WordNet relationships"
echo "  ✓ Adaptive training with independent graduation"
echo "  ✓ CamelCase splitting with quotations"
echo "  ✓ AI-symmetry hard negatives"
echo "  ✓ Varied temperature (0.3-0.9, empirically validated)"
echo ""
echo "Starting at: $(date)"
echo ""

# Activate virtual environment
. .venv/bin/activate

# Run training for all layers
python scripts/train_sumo_classifiers.py \
    --layers 0 1 2 3 4 5 \
    --model google/gemma-3-4b-pt \
    --device cuda \
    --n-train-pos 10 \
    --n-train-neg 10 \
    --n-test-pos 20 \
    --n-test-neg 20 \
    --output-dir "${OUTPUT_DIR}" \
    --use-adaptive-training \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

echo ""
echo "=================================="
echo "TRAINING COMPLETE"
echo "=================================="
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully"

    # Generate summary report
    echo ""
    echo "Generating summary report..."
    python -c "
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
total_trained = 0
total_failed = 0

for layer in range(6):
    layer_dir = output_dir / f'layer{layer}'
    if layer_dir.exists():
        classifiers = list(layer_dir.glob('*_classifier.pt'))
        total_trained += len(classifiers)
        print(f'Layer {layer}: {len(classifiers)} classifiers trained')

print(f'')
print(f'Total: {total_trained} classifiers trained')
"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
    echo "Check log for details: ${LOG_FILE}"
fi

exit ${EXIT_CODE}
