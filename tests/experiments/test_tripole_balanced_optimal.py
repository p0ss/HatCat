#!/usr/bin/env python3
"""
Test tripole lens with OPTIMAL BALANCED sampling.

Instead of matching the minimum, we target a HIGHER number of examples per pole
by intelligently adjusting prompts_per_synset to balance while maximizing total data.

Strategy: Target the MEDIAN overlap count * BASE_PROMPTS_PER_SYNSET
This gives us more data than pure downsampling while still achieving balance.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_data_generation import (
    create_simplex_pole_training_dataset_contrastive,
    get_overlap_synsets_for_pole
)
from training.sumo_classifiers import extract_activations
from training.tripole_classifier import train_tripole_simplex

# Configuration
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
TEST_SIMPLEX = "taste_development"
BEHAVIORAL_RATIO = 0.6
LAYER_IDX = 12
BASE_PROMPTS_PER_SYNSET = 5

print("=" * 80)
print("TRIPOLE LENS TEST - OPTIMAL BALANCED SAMPLING")
print("=" * 80)

# 1. Load simplex definition
print(f"\n1. Loading simplex: {TEST_SIMPLEX}")
with open(S_TIER_DEFS_PATH) as f:
    s_tier_defs = json.load(f)

if TEST_SIMPLEX not in s_tier_defs['simplexes']:
    print(f"ERROR: Simplex '{TEST_SIMPLEX}' not found")
    sys.exit(1)

simplex_def = s_tier_defs['simplexes'][TEST_SIMPLEX]
dimension = simplex_def['dimension']

three_pole = {
    'negative_pole': simplex_def['negative_pole'],
    'neutral_homeostasis': simplex_def['neutral_homeostasis'],
    'positive_pole': simplex_def['positive_pole']
}
print(f"   ✓ Found simplex: {dimension}")

# 2. Count overlap synsets for each pole
print(f"\n2. Counting overlap synsets per pole...")
overlap_counts = {}
for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
    pole_type = pole_name.split('_')[0]
    pole_id = f"{dimension}_{pole_type}"
    synsets = get_overlap_synsets_for_pole(pole_id)
    overlap_counts[pole_type] = len(synsets)
    print(f"   {pole_type:8}: {len(synsets)} overlap synsets")

# Calculate median as target (compromise between min and max)
sorted_counts = sorted(overlap_counts.values())
median_count = sorted_counts[1]
target_examples_per_pole = median_count * BASE_PROMPTS_PER_SYNSET

min_count = min(overlap_counts.values())
max_count = max(overlap_counts.values())
imbalance = max_count / min_count if min_count > 0 else float('inf')

print(f"\n   Imbalance: {imbalance:.1f}x")
print(f"   Target: {target_examples_per_pole} examples per pole (median strategy)")
print(f"   This is {target_examples_per_pole / (min_count * BASE_PROMPTS_PER_SYNSET):.1f}x more data than pure downsampling")

# 3. Load model
print(f"\n3. Loading model...")
model_name = "google/gemma-3-4b-pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True
)
model.eval()
print(f"   ✓ Model loaded on {device}")

# 4. Generate BALANCED training data
print(f"\n4. Generating OPTIMALLY BALANCED training data...")

all_pole_data = {}
for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
    pole_data = three_pole[pole_name]
    pole_type = pole_name.split('_')[0]

    # Calculate adjusted prompts_per_synset for this pole
    n_overlaps = overlap_counts[pole_type]
    if n_overlaps > 0:
        # Round up to ensure we hit target
        adjusted_prompts_per_synset = max(1, int(np.ceil(target_examples_per_pole / n_overlaps)))
    else:
        adjusted_prompts_per_synset = BASE_PROMPTS_PER_SYNSET

    # Get other poles for contrastive learning
    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**three_pole[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    print(f"\n   [{pole_type.upper()}] Generating with {adjusted_prompts_per_synset} prompts/synset...")

    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        behavioral_ratio=BEHAVIORAL_RATIO,
        prompts_per_synset=adjusted_prompts_per_synset
    )

    all_pole_data[pole_type] = {
        'prompts': prompts,
        'labels': labels
    }

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"       Generated {len(prompts)} prompts ({n_pos} positive, {n_neg} negative)")

# 5. Convert to tripole format
print(f"\n5. Converting to tripole format...")

all_prompts = []
tripole_labels = []

pole_index_map = {'negative': 0, 'neutral': 1, 'positive': 2}

for pole_type, data in all_pole_data.items():
    prompts = data['prompts']
    binary_labels = data['labels']
    pole_idx = pole_index_map[pole_type]

    # Only take POSITIVE examples for this pole
    for i, (prompt, label) in enumerate(zip(prompts, binary_labels)):
        if label == 1:
            all_prompts.append(prompt)
            tripole_labels.append(pole_idx)

print(f"   Total training examples: {len(all_prompts)}")
print(f"\n   Distribution:")
for pole_type, idx in pole_index_map.items():
    count = sum(1 for l in tripole_labels if l == idx)
    print(f"   {pole_type:8}: {count} examples")

# 6. Extract activations
print(f"\n6. Extracting activations (layer {LAYER_IDX})...")
activations = extract_activations(model, tokenizer, all_prompts, device, LAYER_IDX)
print(f"   Shape: {activations.shape}")

# Handle combined extraction
if activations.shape[0] == 2 * len(all_prompts):
    tripole_labels_expanded = []
    for label in tripole_labels:
        tripole_labels_expanded.append(label)
        tripole_labels_expanded.append(label)
    tripole_labels = tripole_labels_expanded
    print(f"   Combined extraction: duplicated labels (new shape: {len(tripole_labels)})")

# Convert to tensors
activations = torch.tensor(activations, dtype=torch.float32)
labels_tensor = torch.tensor(tripole_labels, dtype=torch.long)

# 7. Train/test split (80/20)
print(f"\n7. Splitting data (80/20)...")
n_total = activations.shape[0]
n_train = int(n_total * 0.8)

# Shuffle
indices = torch.randperm(n_total)
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_activations = activations[train_indices]
train_labels = labels_tensor[train_indices]
test_activations = activations[test_indices]
test_labels = labels_tensor[test_indices]

print(f"   Train: {train_activations.shape[0]} samples")
print(f"   Test: {test_activations.shape[0]} samples")

# Show train/test distribution
print(f"\n   Train distribution:")
for pole_type, idx in pole_index_map.items():
    count = (train_labels == idx).sum().item()
    print(f"   {pole_type:8}: {count} samples")

print(f"\n   Test distribution:")
for pole_type, idx in pole_index_map.items():
    count = (test_labels == idx).sum().item()
    print(f"   {pole_type:8}: {count} samples")

# 8. Train tripole lens
print(f"\n8. Training tripole lens...")
lens, history = train_tripole_simplex(
    train_activations=train_activations,
    train_labels=train_labels,
    test_activations=test_activations,
    test_labels=test_labels,
    hidden_dim=train_activations.shape[1],
    device=device,
    lr=1e-3,
    max_epochs=100,
    patience=10,
    lambda_margin=0.5,
    lambda_ortho=1e-4,
)

# 9. Results
print(f"\n" + "=" * 80)
print("RESULTS - OPTIMAL BALANCED SAMPLING")
print("=" * 80)

print(f"\nBest test F1: {history['best_test_f1']:.3f}")
print(f"Final test accuracy: {history['final_metrics']['accuracy']:.3f}")

print(f"\nPer-pole F1 scores:")
for pole_type, idx in pole_index_map.items():
    f1_key = f'pole_{idx}_f1'
    if f1_key in history['final_metrics']:
        f1 = history['final_metrics'][f1_key]
        print(f"  {pole_type}: {f1:.3f}")

print(f"\nFinal margins:")
final_margins = history['margins'][-1]
for pole_type, idx in pole_index_map.items():
    print(f"  {pole_type}: {final_margins[idx]:.3f}")

print(f"\n{'=' * 80}")
print(f"COMPARISON:")
print(f"  Imbalanced baseline: neutral F1 avg ~0.273")
print(f"  Min downsampling: neutral F1 = 0.377 (better neutral, but only 65 examples/pole)")
print(f"  Optimal (this run): neutral F1 = {history['final_metrics'].get('pole_1_f1', 0.0):.3f} (with {target_examples_per_pole} examples/pole)")
print(f"{'=' * 80}")
