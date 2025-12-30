#!/usr/bin/env python3
"""
Test hierarchical training hypothesis: Should parents include children's synsets?

This script tests three conditions:
1. Baseline: Only concept's own synsets (canonical_synset only)
2. Direct Children: Include direct children's synsets
3. Recursive Descendants: Include all descendant synsets

For each condition, we train Layer 0 lenses and measure:
- Training success rate
- Calibration accuracy
- Child detection ability
- Sibling discrimination
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_classifiers import train_layer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def collect_descendant_synsets(
    concept_name: str,
    all_concepts: Dict[str, Dict],
    visited: Set[str] = None
) -> List[str]:
    """
    Recursively collect all descendant synsets.

    Args:
        concept_name: SUMO concept name
        all_concepts: Map of concept name -> concept dict
        visited: Set of already visited concepts (prevents cycles)

    Returns:
        List of synset IDs from this concept and all descendants
    """
    if visited is None:
        visited = set()

    if concept_name in visited:
        return []

    visited.add(concept_name)

    concept = all_concepts.get(concept_name)
    if not concept:
        return []

    # Start with this concept's own synsets
    synsets = list(concept.get('synsets', []))

    # Recursively add children's synsets
    for child_name in concept.get('category_children', []):
        child_synsets = collect_descendant_synsets(child_name, all_concepts, visited)
        synsets.extend(child_synsets)

    return synsets


def prepare_experimental_conditions(layer: int = 0) -> Dict[str, Dict]:
    """
    Prepare three experimental conditions for Layer 0.

    Returns:
        Dict mapping condition name -> modified layer data
    """
    # Load original layer data
    layer_file = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / f"layer{layer}.json"
    with open(layer_file) as f:
        original_data = json.load(f)

    # Build concept map for recursive lookup
    all_layers_data = {}
    for l in range(7):  # Layers 0-6
        try:
            with open(PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / f"layer{l}.json") as f:
                all_layers_data[l] = json.load(f)
        except FileNotFoundError:
            continue

    # Flatten to concept map
    all_concepts = {}
    for layer_data in all_layers_data.values():
        for concept in layer_data['concepts']:
            all_concepts[concept['sumo_term']] = concept

    conditions = {}

    # Condition 1: Baseline (canonical_synset only)
    baseline_data = json.loads(json.dumps(original_data))  # Deep copy
    for concept in baseline_data['concepts']:
        # Keep only canonical_synset
        canonical = concept.get('canonical_synset')
        if canonical:
            concept['synsets'] = [canonical]
        else:
            concept['synsets'] = []
    conditions['baseline'] = baseline_data

    # Condition 2: Direct Children (current abstraction layer state)
    direct_children_data = json.loads(json.dumps(original_data))  # Deep copy
    # Already has direct children's synsets from our updates
    conditions['direct_children'] = direct_children_data

    # Condition 3: Recursive Descendants
    recursive_data = json.loads(json.dumps(original_data))  # Deep copy
    for concept in recursive_data['concepts']:
        concept_name = concept['sumo_term']
        # Collect all descendant synsets recursively
        descendant_synsets = collect_descendant_synsets(concept_name, all_concepts)
        # Deduplicate
        concept['synsets'] = list(set(descendant_synsets))
    conditions['recursive_descendants'] = recursive_data

    return conditions


def run_experimental_condition(
    condition_name: str,
    layer_data: Dict,
    model,
    tokenizer,
    device: str,
    output_dir: Path
) -> Dict:
    """
    Train Layer 0 with a specific experimental condition.

    Returns:
        Results dict with training and calibration metrics
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENTAL CONDITION: {condition_name.upper()}")
    print(f"{'='*80}")

    # Save modified layer data temporarily
    temp_layer_file = output_dir / "layer0_modified.json"
    with open(temp_layer_file, 'w') as f:
        json.dump(layer_data, f, indent=2)

    # Report synset counts
    print(f"\nSynset counts per concept:")
    for concept in layer_data['concepts']:
        synset_count = len(concept.get('synsets', []))
        print(f"  {concept['sumo_term']}: {synset_count} synsets")

    # Train using modified data
    # We need to temporarily replace the layer file
    original_layer_file = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer0.json"
    backup_file = original_layer_file.with_suffix('.json.backup')

    try:
        # Backup original
        import shutil
        shutil.copy2(original_layer_file, backup_file)

        # Replace with experimental condition
        shutil.copy2(temp_layer_file, original_layer_file)

        # Train
        results = train_layer(
            layer=0,
            model=model,
            tokenizer=tokenizer,
            n_train_pos=10,  # Small for quick testing
            n_train_neg=10,
            n_test_pos=5,
            n_test_neg=5,
            device=device,
            output_dir=output_dir / "lenses",
            use_adaptive_training=True,
            validation_mode='falloff',
        )

    finally:
        # Restore original
        if backup_file.exists():
            shutil.copy2(backup_file, original_layer_file)
            backup_file.unlink()

    return results


def main():
    print("="*80)
    print("HIERARCHICAL TRAINING HYPOTHESIS TEST")
    print("="*80)
    print("\nTesting whether parents should include children's synsets in training data")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "hierarchical_training_experiment" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Load model
    print("\nLoading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    print("✓ Model loaded")

    # Prepare experimental conditions
    print("\nPreparing experimental conditions...")
    conditions = prepare_experimental_conditions(layer=0)

    # Run each condition
    all_results = {}
    for condition_name, layer_data in conditions.items():
        condition_output = output_dir / condition_name
        condition_output.mkdir(exist_ok=True)

        try:
            results = run_experimental_condition(
                condition_name=condition_name,
                layer_data=layer_data,
                model=model,
                tokenizer=tokenizer,
                device=device,
                output_dir=condition_output
            )
            all_results[condition_name] = results

        except Exception as e:
            print(f"\n✗ ERROR in {condition_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[condition_name] = {'error': str(e)}

    # Save combined results
    results_file = output_dir / "experimental_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'conditions': all_results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Run calibration on each lens set")
    print(f"2. Compare child detection rates")
    print(f"3. Analyze statistical significance")
    print(f"4. Document findings in HIERARCHICAL_TRAINING_DECISION.md")


if __name__ == '__main__':
    main()
