#!/usr/bin/env python3
"""
Hierarchical Lens Pack Calibration Test

Tests lenses against their actual hierarchical relationships:
1. **Positive samples**: All synsets for the target concept
2. **Sibling samples**: Synsets from sibling concepts (should NOT fire)
3. **Parent samples**: Synsets from parent concepts (may partially fire - general concepts)
4. **Child samples**: Synsets from child concepts (may partially fire - specific instances)
5. **Distant samples**: Synsets from unrelated branches (should NOT fire)

This tests whether lenses correctly distinguish:
- Concept from siblings (specificity)
- Concept from parents (abstraction level)
- Concept from children (generalization)
- Concept from unrelated concepts (semantic distance)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


def get_wordnet_definitions(synsets: List[str], max_samples: int = 10) -> List[str]:
    """Get WordNet definitions for synset IDs."""
    definitions = []
    for synset_id in synsets[:max_samples * 2]:  # Try more to get enough valid ones
        if len(definitions) >= max_samples:
            break
        try:
            synset = wn.synset(synset_id)
            definition = synset.definition()
            if definition and len(definition) > 10:  # Skip very short definitions
                definitions.append(definition)
        except:
            continue
    return definitions


def get_definitions_from_stored(definitions: List[str], max_samples: int = 10) -> List[str]:
    """Get definitions from pre-stored definition strings (for v4 concepts)."""
    result = []
    for definition in definitions[:max_samples]:
        if definition and len(definition) > 10:  # Skip very short definitions
            result.append(definition)
    return result


def capture_activation(
    text: str,
    model,
    tokenizer,
    target_layer: int = -1
) -> torch.Tensor:
    """
    Capture activation for a text sample at the target layer.
    Returns: [1, hidden_dim] tensor
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last token's hidden state from target layer
        hidden_states = outputs.hidden_states[target_layer]  # [1, seq_len, hidden_dim]
        activation = hidden_states[:, -1, :]  # [1, hidden_dim]

    return activation.float()  # Convert to float32 for lenses


def evaluate_lens_hierarchical(
    lens_key: Tuple[str, int],
    lens,
    concept_name: str,
    concept_definitions: List[str],  # Definitions for target concept
    siblings_definitions: List[List[str]],  # List of sibling concept definition lists
    parent_definitions: List[str],
    children_definitions: List[List[str]],  # List of child concept definition lists
    distant_definitions: List[List[str]],  # List of distant concept definition lists
    model,
    tokenizer,
    n_samples_per_category: int = 10,
    use_wordnet: bool = False  # Whether to try WordNet lookup (for legacy synsets)
) -> Dict:
    """
    Evaluate lens against hierarchical relationships.

    Returns metrics for each category:
    - positive (target concept): should fire (high scores)
    - siblings: should NOT fire (low scores)
    - parent: may partially fire (medium scores acceptable)
    - children: may partially fire (medium scores acceptable)
    - distant: should NOT fire (low scores)
    """
    # Get definitions for each category
    # For v4 concepts, definitions are passed directly; for legacy, we'd use WordNet
    if use_wordnet:
        positive_defs = get_wordnet_definitions(concept_definitions, n_samples_per_category)
    else:
        positive_defs = get_definitions_from_stored(concept_definitions, n_samples_per_category)

    sibling_defs = []
    for sibling_def_list in siblings_definitions[:5]:  # Sample up to 5 siblings
        if use_wordnet:
            sibling_defs.extend(get_wordnet_definitions(sibling_def_list, max_samples=2))
        else:
            sibling_defs.extend(get_definitions_from_stored(sibling_def_list, max_samples=2))
    sibling_defs = sibling_defs[:n_samples_per_category]

    if use_wordnet:
        parent_defs = get_wordnet_definitions(parent_definitions, n_samples_per_category)
    else:
        parent_defs = get_definitions_from_stored(parent_definitions, n_samples_per_category)

    child_defs = []
    for child_def_list in children_definitions[:5]:  # Sample up to 5 children
        if use_wordnet:
            child_defs.extend(get_wordnet_definitions(child_def_list, max_samples=2))
        else:
            child_defs.extend(get_definitions_from_stored(child_def_list, max_samples=2))
    child_defs = child_defs[:n_samples_per_category]

    distant_defs = []
    for distant_def_list in distant_definitions[:5]:  # Sample up to 5 distant concepts
        if use_wordnet:
            distant_defs.extend(get_wordnet_definitions(distant_def_list, max_samples=2))
        else:
            distant_defs.extend(get_definitions_from_stored(distant_def_list, max_samples=2))
    distant_defs = distant_defs[:n_samples_per_category]

    # Capture activations
    def get_scores(definitions: List[str]) -> List[float]:
        if not definitions:
            return []
        scores = []
        for definition in definitions:
            activation = capture_activation(definition, model, tokenizer)
            logit = lens(activation)
            score = torch.sigmoid(logit).item()  # Convert logit to probability
            scores.append(score)
        return scores

    with torch.inference_mode():
        positive_scores = get_scores(positive_defs)
        sibling_scores = get_scores(sibling_defs)
        parent_scores = get_scores(parent_defs)
        child_scores = get_scores(child_defs)
        distant_scores = get_scores(distant_defs)

        # Single term test
        single_term_activation = capture_activation(concept_name, model, tokenizer)
        single_term_logit = lens(single_term_activation)
        single_term_score = torch.sigmoid(single_term_logit).item()

    # Calculate metrics
    threshold = 0.5

    # Positive metrics (should fire)
    tp = sum(1 for s in positive_scores if s >= threshold)
    fn = sum(1 for s in positive_scores if s < threshold)
    tp_rate = tp / len(positive_scores) if positive_scores else 0

    # Negative metrics (siblings + distant should NOT fire)
    all_negative_scores = sibling_scores + distant_scores
    fp = sum(1 for s in all_negative_scores if s >= threshold)
    tn = sum(1 for s in all_negative_scores if s < threshold)
    fp_rate = fp / len(all_negative_scores) if all_negative_scores else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp_rate': float(tp_rate),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn / len(positive_scores) if positive_scores else 0),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'single_term_score': float(single_term_score),
        'positive_scores': [float(s) for s in positive_scores],
        'sibling_scores': [float(s) for s in sibling_scores],
        'parent_scores': [float(s) for s in parent_scores],
        'child_scores': [float(s) for s in child_scores],
        'distant_scores': [float(s) for s in distant_scores],
        'avg_positive': float(np.mean(positive_scores)) if positive_scores else None,
        'avg_sibling': float(np.mean(sibling_scores)) if sibling_scores else None,
        'avg_parent': float(np.mean(parent_scores)) if parent_scores else None,
        'avg_child': float(np.mean(child_scores)) if child_scores else None,
        'avg_distant': float(np.mean(distant_scores)) if distant_scores else None,
        'sample_counts': {
            'positive': len(positive_scores),
            'sibling': len(sibling_scores),
            'parent': len(parent_scores),
            'child': len(child_scores),
            'distant': len(distant_scores)
        }
    }


def categorize_lens(metrics: Dict) -> str:
    """
    Categorize lens performance based on hierarchical metrics.
    """
    if metrics['f1'] < 0.3:
        return 'broken'
    elif metrics['fp_rate'] > 0.5:
        return 'over_firing'
    elif metrics['tp_rate'] < 0.5:
        return 'under_firing'
    elif metrics['f1'] > 0.7:
        return 'well_calibrated'
    else:
        return 'marginal'


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical calibration test for lens pack"
    )
    parser.add_argument('--lens-pack', default='gemma-3-4b-pt_sumo-wordnet-v2',
                       help='Lens pack ID to calibrate')
    parser.add_argument('--model', default='google/gemma-3-4b-pt',
                       help='Model to use for activation capture')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of samples per category')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output-dir', default='results/lens_calibration',
                       help='Output directory')
    parser.add_argument('--max-concepts', type=int, default=None,
                       help='Limit number of concepts to test (for quick testing)')
    parser.add_argument('--layers-dir', type=Path, default=None,
                       help='Directory with layer JSON files (for v4 concept packs)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HIERARCHICAL LENS PACK CALIBRATION TEST")
    print("="*80)
    print(f"Lens pack: {args.lens_pack}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("✓ Model loaded")

    # Load lens manager (don't load base layers yet - load on demand)
    print("\nLoading lens manager...")
    lenses_dir = Path(f"lens_packs/{args.lens_pack}")
    layers_dir = args.layers_dir if args.layers_dir else Path("data/concept_graph/abstraction_layers")

    manager = DynamicLensManager(
        lenses_dir=lenses_dir,
        layers_data_dir=layers_dir,
        base_layers=[],  # Don't load any lenses upfront - load on demand
        device=args.device
    )
    print(f"✓ Total concepts in metadata: {len(manager.concept_metadata)}")
    print(f"✓ Parent-child relationships: {len(manager.parent_to_children)}")

    # Load definition and synset data from layer files
    print(f"\nLoading concept data from {manager.layers_data_dir}...")
    concept_definitions = {}  # (sumo_term, layer) -> [definitions]
    concept_synsets = {}  # (sumo_term, layer) -> [synsets] - for legacy WordNet lookup
    concept_parents = {}  # (concept, layer) -> list of parent names
    layer_files = sorted(manager.layers_data_dir.glob("layer*.json"))

    for layer_file in layer_files:
        with open(layer_file) as f:
            layer_data = json.load(f)
            # Support both old format (metadata.layer) and v4 format (layer at top level)
            if 'metadata' in layer_data:
                layer = layer_data['metadata']['layer']
            else:
                layer = layer_data.get('layer', int(layer_file.stem.replace('layer', '')))

            for concept in layer_data.get('concepts', []):
                sumo_term = concept['sumo_term']
                key = (sumo_term, layer)

                # Collect definitions (v4 format has 'definition' field)
                definition = concept.get('definition', '')
                sumo_definition = concept.get('sumo_definition', '')

                if key not in concept_definitions:
                    concept_definitions[key] = []

                # Add the main definition if present
                if definition and len(definition) > 10:
                    concept_definitions[key].append(definition)
                # Add sumo_definition as secondary if different
                if sumo_definition and len(sumo_definition) > 10 and sumo_definition != definition:
                    concept_definitions[key].append(sumo_definition)

                # Also add lemmas as short definition-like samples
                lemmas = concept.get('lemmas', [])
                for lemma in lemmas[:3]:  # Take up to 3 lemmas
                    if lemma and len(lemma) > 3:
                        concept_definitions[key].append(f"A type of {lemma}")

                # For layer 6: collect individual synsets from each entry (legacy)
                if layer == 6:
                    if key not in concept_synsets:
                        concept_synsets[key] = []
                    synset = concept.get('synset')  # Note: singular 'synset' field in layer 6
                    if synset:
                        concept_synsets[key].append(synset)
                else:
                    # For layers 0-5: use synsets list directly (legacy)
                    synsets = concept.get('synsets', [])
                    if key not in concept_synsets:
                        concept_synsets[key] = synsets

                # Get parent concepts (category_children in reverse)
                category_children = concept.get('category_children', [])
                # Note: We'd need to build reverse mapping for parents
                # For now, we'll use the parent_to_children mapping from manager

    print(f"✓ Loaded definition data for {len(concept_definitions)} (concept, layer) pairs")
    print(f"✓ Loaded synset data for {len(concept_synsets)} (concept, layer) pairs")

    # Build concept lookup by layer - SCAN lens files directly for all available lenses
    concepts_by_layer = defaultdict(list)
    total_lenses_found = 0

    for layer_dir in sorted(lenses_dir.glob("layer*")):
        if not layer_dir.is_dir():
            continue
        layer = int(layer_dir.name.replace("layer", ""))

        for lens_file in layer_dir.glob("*_classifier.pt"):
            concept_name = lens_file.stem.replace("_classifier", "")
            definitions = concept_definitions.get((concept_name, layer), [])
            metadata = manager.concept_metadata.get((concept_name, layer), {})
            concepts_by_layer[layer].append((concept_name, layer, definitions, metadata))
            total_lenses_found += 1

    # Test concepts
    all_concepts = []
    for layer in sorted(concepts_by_layer.keys()):
        all_concepts.extend(concepts_by_layer[layer])

    print(f"Found {total_lenses_found} lenses across {len(concepts_by_layer)} layers:")
    for layer in sorted(concepts_by_layer.keys()):
        print(f"  Layer {layer}: {len(concepts_by_layer[layer])} lenses")

    if args.max_concepts:
        all_concepts = all_concepts[:args.max_concepts]

    print(f"\nTesting {len(all_concepts)} concepts with hierarchical relationships...")
    print(f"Generating {args.n_samples} samples per category (positive, siblings, parent, children, distant)")

    results = {}
    categories = defaultdict(int)

    # Cache for loaded lenses (load directly, not through manager)
    loaded_lenses = {}

    for concept_name, layer, definitions, metadata in tqdm(all_concepts, desc="Testing lenses"):
        lens_key = (concept_name, layer)

        # Load lens directly from file
        lens_file = lenses_dir / f"layer{layer}" / f"{concept_name}_classifier.pt"
        if not lens_file.exists():
            continue

        if lens_key not in loaded_lenses:
            try:
                lens_data = torch.load(lens_file, map_location=args.device)
                # Lens is a 3-layer MLP from BinaryClassifier:
                # 0: Linear(input_dim, 128)
                # 1: ReLU
                # 2: Dropout
                # 3: Linear(128, 64)
                # 4: ReLU
                # 5: Dropout
                # 6: Linear(64, 1)
                # Keys are: 0.weight, 0.bias, 3.weight, 3.bias, 6.weight, 6.bias
                input_dim = lens_data['0.weight'].shape[1]
                hidden1 = lens_data['0.weight'].shape[0]
                hidden2 = lens_data['3.weight'].shape[0]
                output_dim = lens_data['6.weight'].shape[0]

                lens = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden1),   # 0
                    torch.nn.ReLU(),                        # 1
                    torch.nn.Dropout(0.1),                  # 2
                    torch.nn.Linear(hidden1, hidden2),      # 3
                    torch.nn.ReLU(),                        # 4
                    torch.nn.Dropout(0.1),                  # 5
                    torch.nn.Linear(hidden2, output_dim)    # 6
                ).to(args.device)
                lens.load_state_dict(lens_data)
                lens.eval()
                loaded_lenses[lens_key] = lens
            except Exception as e:
                print(f"Failed to load lens {concept_name} L{layer}: {e}")
                continue

        lens = loaded_lenses[lens_key]

        # Get hierarchical relationships using definitions
        # Siblings: concepts with same parent (or same layer if no parent info)
        siblings_definitions = []
        for sib_name, sib_layer, sib_defs, _ in concepts_by_layer[layer]:
            if sib_name != concept_name:
                siblings_definitions.append(sib_defs)

        # Parent: concepts from layer-1
        parent_definitions = []
        if layer > 0:
            parent_concepts = concepts_by_layer[layer - 1]
            for parent_name, _, parent_defs, _ in parent_concepts[:3]:  # Sample a few parents
                parent_definitions.extend(parent_defs)

        # Children: concepts from layer+1 that are children of this concept
        children_definitions = []
        children_keys = manager.parent_to_children.get(lens_key, [])  # Returns list of (name, layer) tuples
        for child_key in children_keys[:5]:  # Sample up to 5 children
            child_defs = concept_definitions.get(child_key, [])
            if child_defs:
                children_definitions.append(child_defs)

        # Distant: concepts from very different layers
        distant_definitions = []
        distant_layer = (layer + 3) % 5  # Pick a distant layer (0-4)
        if distant_layer in concepts_by_layer:
            for dist_name, _, dist_defs, _ in concepts_by_layer[distant_layer][:5]:
                distant_definitions.append(dist_defs)

        # Evaluate lens
        metrics = evaluate_lens_hierarchical(
            lens_key,
            lens,
            concept_name,
            definitions,
            siblings_definitions,
            parent_definitions,
            children_definitions,
            distant_definitions,
            model,
            tokenizer,
            args.n_samples
        )

        # Categorize
        category = categorize_lens(metrics)
        categories[category] += 1

        results[f"{concept_name}_L{layer}"] = {
            'concept': concept_name,
            'layer': layer,
            'category': category,
            'metrics': metrics
        }

        # Unload lens to free memory (keep cache small)
        if len(loaded_lenses) > 100:
            # Clear oldest lenses
            keys_to_remove = list(loaded_lenses.keys())[:50]
            for k in keys_to_remove:
                del loaded_lenses[k]
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'hierarchical_calibration_{timestamp}.json'

    summary = {
        'lens_pack': args.lens_pack,
        'model': args.model,
        'timestamp': timestamp,
        'total_lenses': len(results),
        'categories': dict(categories),
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total lenses tested: {len(results)}")
    print(f"\nCategories:")
    for category, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        print(f"  {category:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
