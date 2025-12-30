#!/bin/bash
# Run token length experiment by temporarily patching source code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_FILE="$PROJECT_ROOT/src/map/training/sumo_classifiers.py"
BACKUP_FILE="$SRC_FILE.tokenexp_backup"

# Backup original
cp "$SRC_FILE" "$BACKUP_FILE"

cleanup() {
    echo "Restoring original file..."
    mv "$BACKUP_FILE" "$SRC_FILE"
}

trap cleanup EXIT

# Test each token length
for TOKENS in 10 20 40; do
    echo ""
    echo "========================================================================"
    echo "TESTING max_new_tokens=$TOKENS"
    echo "========================================================================"

    # Restore original and patch with new value
    cp "$BACKUP_FILE" "$SRC_FILE"
    sed -i "s/max_new_tokens: int = 20/max_new_tokens: int = $TOKENS/" "$SRC_FILE"

    # Verify patch
    echo "Patched extract_activations to use max_new_tokens=$TOKENS"
    grep "max_new_tokens: int = $TOKENS" "$SRC_FILE" || echo "WARNING: Patch may have failed"

    # Train single concept
    cd "$PROJECT_ROOT"
    poetry run python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from src.map.training.sumo_classifiers import train_layer, load_layer_concepts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-2b-it',
    torch_dtype=torch.bfloat16,
    device_map='cuda',
)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

# Load layer 2 to filter to just Carnivore
layer_concepts, concept_map = load_layer_concepts(2)

# Filter to just Carnivore
import json
filtered_path = Path('data/concept_graph/abstraction_layers/layer2_carnivore_only.json')
filtered_path.parent.mkdir(parents=True, exist_ok=True)

with open(filtered_path, 'w') as f:
    json.dump({
        'layer': 2,
        'concepts': [concept_map['Carnivore']]
    }, f, indent=2)

print(f'Created filtered layer with only Carnivore')

# Train just this concept
result = train_layer(
    layer=2,
    model=model,
    tokenizer=tokenizer,
    n_train_pos=10,
    n_train_neg=10,
    n_test_pos=5,
    n_test_neg=5,
    device='cuda',
    output_dir=Path('results/token_length_experiment/tokens_${TOKENS}'),
    use_adaptive_training=True,
    validation_mode='falloff',
)

print(f'Training complete for tokens={$TOKENS}')
" 2>&1 | tee "results/token_length_experiment/run_${TOKENS}_tokens.log"

done

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
echo "Results saved to results/token_length_experiment/"
