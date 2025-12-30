#!/usr/bin/env python3
"""
Test Jacobian-based concept extraction vs trained classifiers.

This script:
1. Loads trained classifiers for Layer 0 concepts
2. Computes Jacobian concept vectors for same concepts
3. Measures cosine similarity (alignment) between the two directions
4. Generates detailed output for analysis

The goal is to understand the relationship between:
- Jacobian: Local sensitivity direction (context-dependent geometric prior)
- Classifier: Learned separating hyperplane (trained on synthetic data)
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hat.steering.detached_jacobian import extract_concept_vector_jacobian


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(np.dot(vec1_norm, vec2_norm))


def load_trained_classifier(concept_name: str, layer: int, lens_dir: Path) -> dict:
    """
    Load trained classifier weights (lens direction).

    Returns dict with 'weights', 'metadata', 'path' or None if not found.
    """
    layer_dir = lens_dir / f"layer{layer}"

    if not layer_dir.exists():
        return None

    # Try multiple possible filenames
    possible_paths = [
        layer_dir / f"{concept_name}_classifier.pt",
        layer_dir / f"{concept_name}_classifier.pth",
        layer_dir / f"{concept_name}_lens.pt",
        layer_dir / f"{concept_name}_lens.pth",
        layer_dir / f"{concept_name}.pt",
        layer_dir / f"{concept_name}.pth",
    ]

    classifier_path = None
    for path in possible_paths:
        if path.exists():
            classifier_path = path
            break

    if classifier_path is None:
        return None

    # Load classifier
    try:
        state_dict = torch.load(classifier_path, map_location='cpu')
    except Exception as e:
        print(f"  Warning: Failed to load {classifier_path}: {e}")
        return None

    # Get weight vector (first layer weights)
    weights = None

    # Try different keys
    if 'weight' in state_dict:
        weights = state_dict['weight'].cpu().numpy()
    elif '0.weight' in state_dict:
        weights = state_dict['0.weight'].cpu().numpy()
    elif 'model_state_dict' in state_dict:
        # Nested state dict
        nested = state_dict['model_state_dict']
        if 'weight' in nested:
            weights = nested['weight'].cpu().numpy()
        elif '0.weight' in nested:
            weights = nested['0.weight'].cpu().numpy()
    else:
        # Try to find first linear layer
        for key in state_dict.keys():
            if 'weight' in key and isinstance(state_dict[key], torch.Tensor):
                if state_dict[key].dim() >= 1:
                    weights = state_dict[key].cpu().numpy()
                    break

    if weights is None:
        return None

    # Handle MLP: for multi-layer classifiers, compute effective input direction
    # by taking the gradient of output w.r.t. input at zero (linear approximation)
    if weights.ndim == 2 and weights.shape[0] < weights.shape[1]:
        # Shape: (hidden_dim, input_dim) - this is an MLP first layer
        # For Jacobian comparison, we want the dominant input direction
        # Use SVD on first layer weights to get principal direction
        U, S, Vh = np.linalg.svd(weights, full_matrices=False)
        # Take top right singular vector (input space direction)
        weights = Vh[0]  # Shape: (input_dim,)
    else:
        # Single layer or already correct shape - flatten
        weights = weights.flatten()

    # Normalize
    weights = weights / (np.linalg.norm(weights) + 1e-8)

    # Extract metadata if available
    metadata = {}
    if 'metadata' in state_dict:
        metadata = state_dict['metadata']

    return {
        'weights': weights,
        'metadata': metadata,
        'path': str(classifier_path)
    }


def load_layer_concepts(layer: int, layers_dir: Path) -> list:
    """Load concept names from layer JSON."""
    layer_file = layers_dir / f"layer{layer}.json"

    if not layer_file.exists():
        return []

    with open(layer_file) as f:
        data = json.load(f)

    # Extract concept names
    concepts = []
    for item in data:
        if 'sumo_term' in item:
            concepts.append(item['sumo_term'])

    return concepts


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test Jacobian vs Classifier alignment"
    )
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                        help='Model name')
    parser.add_argument('--layer', type=int, default=0,
                        help='Abstraction layer to test (0-5)')
    parser.add_argument('--model-layer', type=int, default=6,
                        help='Model layer index for Jacobian computation')
    parser.add_argument('--lens-dir', type=str,
                        default='lens_packs/gemma-3-4b-pt_sumo-wordnet-v1',
                        help='Lens pack directory')
    parser.add_argument('--layers-dir', type=str,
                        default='data/concept_graph/abstraction_layers',
                        help='Concept definitions directory')
    parser.add_argument('--concepts', type=str, nargs='*',
                        help='Specific concepts to test (default: all in layer)')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Maximum number of concepts to test')
    parser.add_argument('--output', type=str,
                        default='results/jacobian_classifier_alignment.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print("=" * 80)
    print("JACOBIAN VS CLASSIFIER ALIGNMENT TEST")
    print("=" * 80)
    print()
    print("Testing hypothesis:")
    print("  - Jacobian: Local sensitivity direction (context-dependent)")
    print("  - Classifier: Learned separating hyperplane (data-driven)")
    print()
    print("Measuring cosine similarity to understand their relationship.")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # Use BF16 for memory efficiency
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device} (dtype: bfloat16)")

    lens_dir = Path(args.lens_dir)
    layers_dir = Path(args.layers_dir)

    if not lens_dir.exists():
        print(f"\n✗ Lens directory not found: {lens_dir}")
        print("  Train classifiers first using training script")
        return 1

    if not layers_dir.exists():
        print(f"\n✗ Layers directory not found: {layers_dir}")
        return 1

    print(f"\nLens directory: {lens_dir}")
    print(f"Abstraction layer: {args.layer}")
    print(f"Model layer: {args.model_layer}")

    # Get concepts to test
    if args.concepts:
        concepts = args.concepts
        print(f"Testing specific concepts: {concepts}")
    else:
        concepts = load_layer_concepts(args.layer, layers_dir)
        print(f"Loaded {len(concepts)} concepts from layer {args.layer}")

        if args.max_concepts and len(concepts) > args.max_concepts:
            concepts = concepts[:args.max_concepts]
            print(f"Limited to first {args.max_concepts} concepts")

    if not concepts:
        print("\n✗ No concepts found to test")
        return 1

    # Test each concept
    results = []
    success_count = 0
    no_classifier_count = 0
    jacobian_fail_count = 0

    print(f"\n{'=' * 80}")
    print("TESTING CONCEPTS")
    print(f"{'=' * 80}\n")

    for i, concept in enumerate(concepts, 1):
        print(f"[{i}/{len(concepts)}] {concept}")

        result = {
            'concept': concept,
            'layer': args.layer,
            'model_layer': args.model_layer,
            'status': 'unknown',
            'jacobian_time': None,
            'alignment': None,
            'jacobian_norm': None,
            'classifier_norm': None,
        }

        # Load trained classifier
        print("  Loading classifier...", end=" ")
        classifier_data = load_trained_classifier(concept, args.layer, lens_dir)

        if classifier_data is None:
            print("✗ Not found")
            result['status'] = 'no_classifier'
            no_classifier_count += 1
            results.append(result)
            print()
            continue

        print(f"✓ ({classifier_data['path']})")
        classifier_vector = classifier_data['weights']
        result['classifier_norm'] = float(np.linalg.norm(classifier_vector))

        # Compute Jacobian
        print("  Computing Jacobian...", end=" ")
        start_time = time.time()

        try:
            jacobian_vector = extract_concept_vector_jacobian(
                model=model,
                tokenizer=tokenizer,
                concept=concept,
                device=device,
                layer_idx=args.model_layer,
                prompt_template="The concept of {concept} means"
            )

            jacobian_time = time.time() - start_time
            result['jacobian_time'] = jacobian_time
            result['jacobian_norm'] = float(np.linalg.norm(jacobian_vector))

            print(f"✓ ({jacobian_time:.2f}s)")

        except Exception as e:
            jacobian_time = time.time() - start_time
            print(f"✗ Failed after {jacobian_time:.2f}s")
            print(f"     Error: {e}")
            result['status'] = 'jacobian_failed'
            result['error'] = str(e)
            jacobian_fail_count += 1
            results.append(result)
            print()
            continue

        # Compute alignment
        alignment = cosine_similarity(jacobian_vector, classifier_vector)
        result['alignment'] = float(alignment)
        result['status'] = 'success'

        print(f"  Alignment: {alignment:.4f}")

        success_count += 1
        results.append(result)
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total concepts tested: {len(concepts)}")
    print(f"  Successful: {success_count}")
    print(f"  No classifier: {no_classifier_count}")
    print(f"  Jacobian failed: {jacobian_fail_count}")

    if success_count > 0:
        successful = [r for r in results if r['status'] == 'success']
        alignments = [r['alignment'] for r in successful]
        times = [r['jacobian_time'] for r in successful]

        print()
        print("Alignment Statistics:")
        print(f"  Mean:   {np.mean(alignments):.4f}")
        print(f"  Median: {np.median(alignments):.4f}")
        print(f"  Std:    {np.std(alignments):.4f}")
        print(f"  Min:    {np.min(alignments):.4f}")
        print(f"  Max:    {np.max(alignments):.4f}")

        print()
        print("Alignment Distribution:")
        bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in bins:
            count = sum(1 for a in alignments if low <= a < high)
            pct = 100 * count / len(alignments)
            bar = "█" * int(pct / 2)
            print(f"  [{low:.1f}-{high:.1f}): {count:3d} ({pct:5.1f}%) {bar}")

        print()
        print("Timing Statistics:")
        print(f"  Mean:   {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")
        print(f"  Total:  {np.sum(times):.1f}s")

        # Top/bottom alignments
        successful_sorted = sorted(successful, key=lambda x: x['alignment'], reverse=True)

        print()
        print("Top 5 Highest Alignment:")
        for r in successful_sorted[:5]:
            print(f"  {r['concept']:30s}: {r['alignment']:.4f}")

        print()
        print("Top 5 Lowest Alignment:")
        for r in successful_sorted[-5:]:
            print(f"  {r['concept']:30s}: {r['alignment']:.4f}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'model': args.model,
            'abstraction_layer': args.layer,
            'model_layer': args.model_layer,
            'lens_dir': str(lens_dir),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_concepts': len(concepts),
            'successful': success_count,
            'no_classifier': no_classifier_count,
            'jacobian_failed': jacobian_fail_count,
        },
        'results': results
    }

    if success_count > 0:
        output_data['statistics'] = {
            'alignment': {
                'mean': float(np.mean(alignments)),
                'median': float(np.median(alignments)),
                'std': float(np.std(alignments)),
                'min': float(np.min(alignments)),
                'max': float(np.max(alignments)),
            },
            'timing': {
                'mean': float(np.mean(times)),
                'median': float(np.median(times)),
                'total': float(np.sum(times)),
            }
        }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
