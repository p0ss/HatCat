#!/usr/bin/env python3
"""
Fast Calibration Analysis

Speed optimizations over original analysis.py:
1. No generation - just forward pass on concept prompt (10x faster)
2. Uses DynamicLensManager for realistic hierarchical competition

For each concept:
1. Run forward pass on concept prompt (no generation)
2. Extract activation at last token
3. Run through DynamicLensManager with hierarchical loading
4. Check if target concept appears in top-k

This tests the full production pipeline:
- Ancestors must fire to load children
- Target competes against all loaded lenses (50-500)
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class BatchedAnalysisResult:
    """Result of batched calibration analysis."""
    lens_pack_id: str
    concept_pack_id: str
    model_id: str
    timestamp: str
    mode: str
    batch_size: int
    top_k: int
    total_concepts: int
    total_batches: int
    avg_in_top_k_rate: float
    concepts_needing_boost: int
    concepts_over_firing: int
    lens_reports: Dict
    under_firing: List[str]
    over_firing: List[str]
    well_calibrated: List[str]


def create_batches(
    concepts: List[Tuple[str, int]],  # (concept_name, layer)
    batch_size: int = 5,
    shuffle: bool = True,
    repeats: int = 1,
) -> List[List[Tuple[str, int]]]:
    """
    Create batches of concepts for testing.

    Args:
        concepts: List of (concept_name, layer) tuples
        batch_size: Number of concepts per batch
        shuffle: Randomize order
        repeats: How many times each concept appears (in different batches)

    Returns:
        List of batches, each batch is a list of (concept_name, layer) tuples
    """
    all_concepts = concepts * repeats
    if shuffle:
        random.shuffle(all_concepts)

    batches = []
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i + batch_size]
        if len(batch) == batch_size:  # Only full batches
            batches.append(batch)

    # Handle remainder
    remainder = all_concepts[len(batches) * batch_size:]
    if remainder:
        # Pad with random concepts to make a full batch
        while len(remainder) < batch_size:
            remainder.append(random.choice(concepts))
        batches.append(remainder)

    return batches


def create_hierarchy_batches(
    concept_pack_dir: Path,
    layers: List[int],
    max_concepts: Optional[int] = None,
) -> List[List[Tuple[str, int, List[str]]]]:
    """
    Create batches where each batch is a hierarchy path (root → leaf).

    Each batch contains concepts from the same lineage:
    [Layer0 root, Layer1 child, Layer2 grandchild, ...]

    At each position, the concept at that position should be rank 0,
    and its ancestors should fire but rank below it.

    Returns:
        List of batches, each batch is [(concept_name, layer, ancestors), ...]
    """
    # Load hierarchy to find parent-child relationships
    hierarchy = {}  # concept -> {layer, parents, children}

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', [])
        for c in concept_list:
            term = c.get('sumo_term') or c.get('term')
            if not term:
                continue

            parents = c.get('parent_concepts', [])
            children = c.get('child_concepts', []) or c.get('category_children', [])

            hierarchy[term] = {
                'layer': layer,
                'parents': parents,
                'children': children,
            }

    # Build paths from leaves to roots
    def get_ancestor_path(concept: str) -> List[Tuple[str, int]]:
        """Get path from concept up to root."""
        path = []
        current = concept
        visited = set()

        while current and current in hierarchy and current not in visited:
            visited.add(current)
            info = hierarchy[current]
            path.append((current, info['layer']))

            # Move to first parent
            parents = info['parents']
            current = parents[0] if parents else None

        return list(reversed(path))  # Root first

    # Find leaf concepts (no children or at max layer)
    max_layer = max(layers)
    leaves = []
    for term, info in hierarchy.items():
        if info['layer'] == max_layer or not info['children']:
            leaves.append(term)

    if max_concepts:
        random.shuffle(leaves)
        leaves = leaves[:max_concepts]

    # Create batches - each batch is an ancestor path
    batches = []
    for leaf in leaves:
        path = get_ancestor_path(leaf)
        if len(path) >= 2:  # Need at least 2 for meaningful comparison
            # Add ancestor list to each entry
            batch_with_ancestors = []
            for i, (term, layer) in enumerate(path):
                ancestors = [p[0] for p in path[:i]]  # All concepts before this one
                batch_with_ancestors.append((term, layer, ancestors))
            batches.append(batch_with_ancestors)

    return batches


def create_triple_criteria_batches(
    concept_pack_dir: Path,
    layers: List[int],
    lens_paths: Dict[str, Tuple[Path, int]],
    num_random_batches: int = 3,
    max_concepts: Optional[int] = None,
) -> Tuple[List[dict], Dict]:
    """
    Create batches for triple-criteria testing:
    1. Ancestor batches: target + nearest ancestors (must be rank 0)
    2. Random batches: target + 4 random concepts (must be in top 5)
    3. Sibling batches: target + siblings (same parent) (must be rank 0)

    This replaces the separate sibling refinement process by integrating
    sibling competition into the main calibration loop.

    Returns:
        batches: List of batch dicts with 'type', 'target', 'concepts'
        hierarchy: Dict of concept -> {layer, parents, children}
    """
    # Load hierarchy
    hierarchy = {}

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', [])
        for c in concept_list:
            term = c.get('sumo_term') or c.get('term')
            if not term:
                continue

            parents = c.get('parent_concepts', [])
            children = c.get('child_concepts', []) or c.get('category_children', [])

            hierarchy[term] = {
                'layer': layer,
                'parents': parents,
                'children': children,
            }

    def get_nearest_ancestors(concept: str, n: int = 4) -> List[str]:
        """Get n nearest ancestors (closest first)."""
        ancestors = []
        current = concept
        visited = set([concept])

        while len(ancestors) < n:
            if current not in hierarchy:
                break
            parents = hierarchy[current]['parents']
            if not parents:
                break
            parent = parents[0]
            if parent in visited:
                break
            visited.add(parent)
            ancestors.append(parent)
            current = parent

        return ancestors

    # Get all concepts that have lenses
    all_concepts = [c for c in hierarchy.keys() if c in lens_paths]

    if max_concepts:
        random.shuffle(all_concepts)
        all_concepts = all_concepts[:max_concepts]

    print(f"  Building batches for {len(all_concepts)} concepts...")

    batches = []

    for concept in all_concepts:
        layer = hierarchy[concept]['layer']

        # 1. Ancestor batch: target + 4 nearest ancestors
        ancestors = get_nearest_ancestors(concept, 4)
        # Filter to those with lenses
        ancestors = [a for a in ancestors if a in lens_paths]

        if ancestors:  # Need at least 1 ancestor for meaningful test
            ancestor_batch = {
                'type': 'ancestor',
                'target': concept,
                'target_layer': layer,
                'concepts': ancestors + [concept],  # Ancestors first, target last
                'ancestors': ancestors,
            }
            batches.append(ancestor_batch)

        # 2. Random batches: target + 4 random (non-ancestor) concepts
        non_ancestors = [c for c in all_concepts
                        if c != concept and c not in ancestors
                        and c in lens_paths]

        for i in range(num_random_batches):
            if len(non_ancestors) >= 4:
                randoms = random.sample(non_ancestors, 4)
                random_batch = {
                    'type': 'random',
                    'target': concept,
                    'target_layer': layer,
                    'concepts': randoms + [concept],
                    'ancestors': [],
                }
                batches.append(random_batch)

        # 3. Sibling batch: target + siblings (same parent, same layer)
        concept_parents = hierarchy[concept]['parents']
        if concept_parents:
            # Find siblings: same parent(s), different concept, has lens
            siblings = []
            for other_concept, other_info in hierarchy.items():
                if other_concept == concept:
                    continue
                if other_concept not in lens_paths:
                    continue
                # Check if they share at least one parent
                other_parents = other_info.get('parents', [])
                if any(p in concept_parents for p in other_parents):
                    siblings.append(other_concept)

            if siblings:
                # Limit to ~10 siblings to keep batches manageable
                if len(siblings) > 10:
                    siblings = random.sample(siblings, 10)
                sibling_batch = {
                    'type': 'sibling',
                    'target': concept,
                    'target_layer': layer,
                    'concepts': siblings + [concept],
                    'siblings': siblings,
                }
                batches.append(sibling_batch)

    print(f"  Created {len(batches)} total batches")
    ancestor_count = sum(1 for b in batches if b['type'] == 'ancestor')
    random_count = sum(1 for b in batches if b['type'] == 'random')
    sibling_count = sum(1 for b in batches if b['type'] == 'sibling')
    print(f"    Ancestor batches: {ancestor_count}")
    print(f"    Random batches: {random_count}")
    print(f"    Sibling batches: {sibling_count}")

    return batches, hierarchy


def run_triple_criteria_analysis(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    num_random_batches: int = 3,
    max_concepts: Optional[int] = None,
    layer_idx: int = 15,
) -> BatchedAnalysisResult:
    """
    Run triple-criteria calibration analysis.

    For each concept:
    1. Ancestor batch: must be RANK 0 (beat all ancestors)
    2. Random batches: must be in TOP 5 (fire consistently)
    3. Sibling batch: must be RANK 0 among siblings (beat siblings)

    A concept is well-calibrated if it passes ALL THREE criteria.
    This replaces the separate sibling refinement process.
    """
    from src.hat.monitoring.lens_manager import SimpleMLP

    print(f"\n{'='*80}")
    print("TRIPLE-CRITERIA CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Random batches per concept: {num_random_batches}")
    print(f"  Criteria:")
    print(f"    - Ancestor batches: must be RANK 0 (beat ancestors)")
    print(f"    - Random batches: must be in TOP 5 (fire consistently)")
    print(f"    - Sibling batches: must be RANK 0 (beat siblings)")

    # Detect hidden dim and build lens paths
    print(f"\nDetecting hidden dim...")
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            lens_files = list(layer_dir.glob("*_classifier.pt"))
            if lens_files:
                state_dict = torch.load(lens_files[0], map_location='cpu')
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension")

    print(f"  Hidden dim: {hidden_dim}")

    # Build lens path lookup
    lens_paths = {}
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            for lens_file in layer_dir.glob("*_classifier.pt"):
                concept_name = lens_file.stem.replace("_classifier", "")
                lens_paths[concept_name] = (lens_file, layer)

    print(f"  Found {len(lens_paths)} lens files")

    # Create triple-criteria batches
    print(f"\nCreating batches...")
    batches, hierarchy = create_triple_criteria_batches(
        concept_pack_dir, layers, lens_paths,
        num_random_batches=num_random_batches,
        max_concepts=max_concepts,
    )

    # Create layer norm
    layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    # Track results per concept
    concept_results = defaultdict(lambda: {
        'ancestor_rank_0': 0,
        'ancestor_total': 0,
        'random_in_top5': 0,
        'random_total': 0,
        'sibling_rank_0': 0,
        'sibling_total': 0,
        'layer': 0,
        'activations': [],
        'failed_ancestors': [],  # Ancestors that outranked this concept
        'failed_siblings': [],   # Siblings that outranked this concept
    })

    # Track ancestor over-firing
    ancestor_over_fire_counts = defaultdict(int)
    # Track sibling over-firing
    sibling_over_fire_counts = defaultdict(int)

    print(f"\nRunning triple-criteria probes...")
    model.eval()

    for batch in tqdm(batches, desc="Batches"):
        batch_type = batch['type']
        target = batch['target']
        concepts = batch['concepts']
        target_layer = batch['target_layer']

        # Load lenses for this batch
        batch_lenses = {}
        for concept_name in concepts:
            if concept_name not in lens_paths:
                continue
            lens_path, _ = lens_paths[concept_name]
            try:
                state_dict = torch.load(lens_path, map_location='cpu')
                lens = SimpleMLP(hidden_dim).to(device)
                if not list(state_dict.keys())[0].startswith('net.'):
                    state_dict = {f'net.{k}': v for k, v in state_dict.items()}
                lens.load_state_dict(state_dict)
                lens.eval()
                batch_lenses[concept_name] = lens
            except:
                pass

        if target not in batch_lenses or len(batch_lenses) < 2:
            continue

        # Forward pass on target concept name
        inputs = tokenizer(target, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            activation = hidden_states[0, -1, :].float()
            activation = layer_norm(activation.unsqueeze(0)).squeeze(0)

        # Run all batch lenses
        scores = {}
        for lens_name, lens in batch_lenses.items():
            with torch.no_grad():
                score = lens(activation.unsqueeze(0)).item()
                scores[lens_name] = score

        # Rank
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        target_rank = next((i for i, (name, _) in enumerate(ranked) if name == target), len(ranked))
        target_score = scores.get(target, 0)

        # Record results based on batch type
        concept_results[target]['layer'] = target_layer
        concept_results[target]['activations'].append(target_score)

        if batch_type == 'ancestor':
            concept_results[target]['ancestor_total'] += 1
            if target_rank == 0:
                concept_results[target]['ancestor_rank_0'] += 1
            else:
                # Record which ancestors beat us
                for ancestor in batch['ancestors']:
                    if ancestor in scores and scores[ancestor] > target_score:
                        concept_results[target]['failed_ancestors'].append(ancestor)
                        ancestor_over_fire_counts[ancestor] += 1

        elif batch_type == 'random':
            concept_results[target]['random_total'] += 1
            if target_rank < 5:  # In top 5
                concept_results[target]['random_in_top5'] += 1

        elif batch_type == 'sibling':
            concept_results[target]['sibling_total'] += 1
            if target_rank == 0:
                concept_results[target]['sibling_rank_0'] += 1
            else:
                # Record which siblings beat us
                for sibling in batch.get('siblings', []):
                    if sibling in scores and scores[sibling] > target_score:
                        concept_results[target]['failed_siblings'].append(sibling)
                        sibling_over_fire_counts[sibling] += 1

        # Clear batch lenses
        del batch_lenses
        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\nAggregating results...")

    lens_reports = {}
    under_firing = []
    over_firing = []
    well_calibrated = []

    for concept_name, results in concept_results.items():
        ancestor_rate = results['ancestor_rank_0'] / results['ancestor_total'] if results['ancestor_total'] > 0 else 1.0
        random_rate = results['random_in_top5'] / results['random_total'] if results['random_total'] > 0 else 1.0
        sibling_rate = results['sibling_rank_0'] / results['sibling_total'] if results['sibling_total'] > 0 else 1.0
        avg_activation = np.mean(results['activations']) if results['activations'] else 0

        # Triple criteria: must pass ALL THREE
        passes_ancestor = ancestor_rate >= 0.8  # 80% rank-0 vs ancestors
        passes_random = random_rate >= 0.8      # 80% in top-5 vs randoms
        passes_sibling = sibling_rate >= 0.8    # 80% rank-0 vs siblings

        # Check if this concept over-fires as an ancestor or sibling
        times_over_fired_ancestor = ancestor_over_fire_counts.get(concept_name, 0)
        times_over_fired_sibling = sibling_over_fire_counts.get(concept_name, 0)
        times_over_fired = times_over_fired_ancestor + times_over_fired_sibling

        lens_reports[concept_name] = {
            'concept': concept_name,
            'layer': results['layer'],
            'ancestor_rank_0_rate': ancestor_rate,
            'ancestor_tests': results['ancestor_total'],
            'random_top5_rate': random_rate,
            'random_tests': results['random_total'],
            'sibling_rank_0_rate': sibling_rate,
            'sibling_tests': results['sibling_total'],
            'avg_activation': avg_activation,
            'passes_ancestor_criterion': passes_ancestor,
            'passes_random_criterion': passes_random,
            'passes_sibling_criterion': passes_sibling,
            'failed_ancestors': list(set(results['failed_ancestors']))[:5],
            'failed_siblings': list(set(results['failed_siblings']))[:5],
            'over_fire_count': times_over_fired,
        }

        if not passes_ancestor or not passes_random or not passes_sibling:
            under_firing.append(concept_name)
        elif times_over_fired > 3:
            over_firing.append(concept_name)
        else:
            well_calibrated.append(concept_name)

    # Calculate overall pass rates
    total_concepts = len(concept_results)
    ancestor_passers = sum(1 for c, r in lens_reports.items() if r['passes_ancestor_criterion'])
    random_passers = sum(1 for c, r in lens_reports.items() if r['passes_random_criterion'])
    sibling_passers = sum(1 for c, r in lens_reports.items() if r['passes_sibling_criterion'])
    all_passers = sum(1 for c, r in lens_reports.items()
                      if r['passes_ancestor_criterion'] and r['passes_random_criterion'] and r['passes_sibling_criterion'])

    result = BatchedAnalysisResult(
        lens_pack_id=lens_pack_dir.name,
        concept_pack_id=concept_pack_dir.name,
        model_id=str(model.config._name_or_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode='triple_criteria',
        batch_size=5,
        top_k=5,
        total_concepts=total_concepts,
        total_batches=len(batches),
        avg_in_top_k_rate=all_passers / total_concepts if total_concepts > 0 else 0,
        concepts_needing_boost=len(under_firing),
        concepts_over_firing=len(over_firing),
        lens_reports=lens_reports,
        under_firing=under_firing,
        over_firing=over_firing,
        well_calibrated=well_calibrated,
    )

    # Print summary
    print(f"\n  Pass rates:")
    print(f"    Ancestor criterion (rank 0): {ancestor_passers}/{total_concepts} ({ancestor_passers/total_concepts*100:.1f}%)")
    print(f"    Random criterion (top 5): {random_passers}/{total_concepts} ({random_passers/total_concepts*100:.1f}%)")
    print(f"    Sibling criterion (rank 0): {sibling_passers}/{total_concepts} ({sibling_passers/total_concepts*100:.1f}%)")
    print(f"    All three criteria: {all_passers}/{total_concepts} ({all_passers/total_concepts*100:.1f}%)")

    return result


# Backward-compatible aliases
create_dual_criteria_batches = create_triple_criteria_batches
run_dual_criteria_analysis = run_triple_criteria_analysis


def run_production_analysis(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    top_k: int = 10,
    max_concepts: Optional[int] = None,
    layer_idx: int = 15,
) -> BatchedAnalysisResult:
    """
    Run calibration analysis using DynamicLensManager's normal loading behavior.

    For each concept:
    1. Get activation for concept's name
    2. Run detect_and_expand() against loaded lenses (base layers + dynamic expansion)
    3. Check if target is in top-k (absolute criterion)
    4. Check if target beats all its ancestors (ancestor criterion)

    This tests realistic production conditions with normal lens loading.
    """
    from src.hat.monitoring.lens_manager import DynamicLensManager

    print(f"\n{'='*80}")
    print("PRODUCTION CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Top-k threshold: {top_k}")
    print(f"  Testing against DynamicLensManager loaded lenses")

    # Initialize DynamicLensManager with lenses_dir (legacy structure)
    print(f"\nInitializing DynamicLensManager...")

    manager = DynamicLensManager(
        lenses_dir=lens_pack_dir,  # Points to layer0/, layer1/, etc.
        layers_data_dir=concept_pack_dir / "hierarchy",
        base_layers=layers[:3] if len(layers) >= 3 else layers,  # Base layers for initial load
        device=device,
        max_loaded_lenses=500,  # Reasonable limit
        normalize_hidden_states=True,
    )

    print(f"  Base lenses loaded: {len(manager.loaded_lenses)}")

    # Build hierarchy lookup for ancestor checking
    print(f"\nBuilding hierarchy...")
    hierarchy = {}  # concept -> list of ancestors
    all_concepts = {}  # concept -> layer

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue
        with open(layer_file) as f:
            layer_data = json.load(f)
        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])
        for c in concept_list:
            term = c.get('sumo_term') or c.get('term')
            if term:
                all_concepts[term] = layer
                parent = c.get('parent')
                ancestors = []
                if parent:
                    ancestors.append(parent)
                    current = parent
                    while current in hierarchy:
                        parent_ancestors = hierarchy[current]
                        if parent_ancestors:
                            ancestors.extend(parent_ancestors)
                        break
                hierarchy[term] = ancestors

    # Get concepts to test - all concepts that have lenses
    concepts_to_test = []
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue
        for lens_file in layer_dir.glob("*_classifier.pt"):
            concept_name = lens_file.stem.replace("_classifier", "")
            if concept_name in all_concepts:
                concepts_to_test.append((concept_name, all_concepts[concept_name]))

    print(f"  Concepts to test: {len(concepts_to_test)}")

    if max_concepts and len(concepts_to_test) > max_concepts:
        random.shuffle(concepts_to_test)
        concepts_to_test = concepts_to_test[:max_concepts]
        print(f"  Limited to {max_concepts} concepts for testing")

    # Run analysis
    print(f"\nRunning production probes...")
    model.eval()

    concept_results = {}
    over_fire_counts = defaultdict(int)  # Track how often each probe over-fires

    for concept_name, concept_layer in tqdm(concepts_to_test, desc="Testing"):
        # Get activation for concept name
        inputs = tokenizer(concept_name, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            activation = hidden_states[0, -1, :].float()

            # Use DynamicLensManager's detect_and_expand
            detected, _ = manager.detect_and_expand(activation, top_k=top_k * 2)

        # Get scores for all loaded lenses
        scores = dict(manager.lens_scores)
        num_loaded = len(manager.loaded_lenses)

        # Find target score and rank among loaded lenses
        target_key = (concept_name, concept_layer)
        target_score = scores.get(target_key, 0)

        # Sort loaded lens scores to find rank
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_rank = next((i for i, (k, v) in enumerate(sorted_scores) if k == target_key), 999)

        # Check ancestor criterion
        ancestors = hierarchy.get(concept_name, [])
        ancestor_set = set(ancestors)
        beats_all_ancestors = True
        failed_ancestors = []
        for ancestor in ancestors:
            # Find ancestor in any layer
            for l in layers:
                ancestor_key = (ancestor, l)
                if ancestor_key in scores:
                    if scores[ancestor_key] >= target_score:
                        beats_all_ancestors = False
                        failed_ancestors.append(ancestor)
                    break

        # Track over-firing: probes in top-k that aren't target or ancestors
        top_k_probes = sorted_scores[:top_k]
        intruders = []
        for (probe_name, probe_layer), score in top_k_probes:
            if probe_name != concept_name and probe_name not in ancestor_set:
                over_fire_counts[probe_name] += 1
                intruders.append(probe_name)

        concept_results[concept_name] = {
            'layer': concept_layer,
            'rank': target_rank,
            'score': target_score,
            'in_top_k': target_rank < top_k,
            'beats_ancestors': beats_all_ancestors,
            'failed_ancestors': failed_ancestors[:5],
            'intruders': intruders,  # All probes that shouldn't be in top-k
            'num_competing_lenses': num_loaded,
        }

    # Aggregate results
    print(f"\nAggregating results...")

    lens_reports = {}
    under_firing = []
    over_firing = []
    well_calibrated = []

    # Identify chronic over-firers (appear in top-k for many unrelated concepts)
    # Threshold: over-fires on more than 1% of tested concepts
    over_fire_threshold = max(10, len(concept_results) // 100)
    chronic_over_firers = {
        name: count for name, count in over_fire_counts.items()
        if count >= over_fire_threshold
    }

    # Build reverse map: for each over-firer, which concepts did it incorrectly fire on?
    over_fire_on_map = defaultdict(list)
    for concept_name, results in concept_results.items():
        for intruder in results.get('intruders', []):
            over_fire_on_map[intruder].append(concept_name)

    for concept_name, results in concept_results.items():
        passes_top_k = results['in_top_k']
        passes_ancestor = results['beats_ancestors']

        lens_reports[concept_name] = {
            'concept': concept_name,
            'layer': results['layer'],
            'rank': results['rank'],
            'score': results['score'],
            'in_top_k': passes_top_k,
            'passes_ancestor_criterion': passes_ancestor,
            'failed_ancestors': results['failed_ancestors'],
            'intruders': results.get('intruders', []),
            'num_competing_lenses': results['num_competing_lenses'],
            'over_fire_count': over_fire_counts.get(concept_name, 0),
            'over_fire_on': over_fire_on_map.get(concept_name, []),  # For finetune suppression
        }

        if not passes_top_k or not passes_ancestor:
            under_firing.append(concept_name)
        else:
            well_calibrated.append(concept_name)

    # Over-firing list is now chronic over-firers (not random criterion failures)
    over_firing = sorted(chronic_over_firers.keys(), key=lambda x: -chronic_over_firers[x])

    # Calculate pass rates
    total_concepts = len(concept_results)
    top_k_passers = sum(1 for r in lens_reports.values() if r['in_top_k'])
    ancestor_passers = sum(1 for r in lens_reports.values() if r['passes_ancestor_criterion'])
    both_passers = sum(1 for r in lens_reports.values()
                       if r['in_top_k'] and r['passes_ancestor_criterion'])

    avg_rank = np.mean([r['rank'] for r in lens_reports.values()])
    avg_competing = np.mean([r['num_competing_lenses'] for r in lens_reports.values()])

    result = BatchedAnalysisResult(
        lens_pack_id=lens_pack_dir.name,
        concept_pack_id=concept_pack_dir.name,
        model_id=str(model.config._name_or_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode='production',
        batch_size=0,  # N/A for production mode
        top_k=top_k,
        total_concepts=total_concepts,
        total_batches=0,  # N/A
        avg_in_top_k_rate=top_k_passers / total_concepts if total_concepts > 0 else 0,
        concepts_needing_boost=len(under_firing),
        concepts_over_firing=len(over_firing),
        lens_reports=lens_reports,
        under_firing=under_firing,
        over_firing=over_firing,
        well_calibrated=well_calibrated,
    )

    # Print summary
    print(f"\n  Results (avg {avg_competing:.0f} competing lenses):")
    print(f"    Top-{top_k} criterion: {top_k_passers}/{total_concepts} ({top_k_passers/total_concepts*100:.1f}%)")
    print(f"    Ancestor criterion: {ancestor_passers}/{total_concepts} ({ancestor_passers/total_concepts*100:.1f}%)")
    print(f"    Both criteria: {both_passers}/{total_concepts} ({both_passers/total_concepts*100:.1f}%)")
    print(f"    Average rank: {avg_rank:.1f}")
    print(f"    Chronic over-firers (>{over_fire_threshold} intrusions): {len(chronic_over_firers)}")

    if chronic_over_firers:
        print(f"\n  Worst over-firers:")
        for name in over_firing[:10]:
            print(f"    {name}: {chronic_over_firers[name]} intrusions")

    return result


def run_hierarchy_analysis(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    max_concepts: Optional[int] = None,
    layer_idx: int = 15,
    load_threshold: float = 0.3,
) -> BatchedAnalysisResult:
    """
    Run hierarchical calibration analysis.

    Tests that for each leaf concept:
    1. The leaf is rank 0 at its position
    2. Ancestors fire above threshold but rank below leaf
    3. Detects ancestor over-firing (when ancestor outranks descendant)
    """
    from src.hat.monitoring.lens_manager import SimpleMLP

    print(f"\n{'='*80}")
    print("HIERARCHICAL CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Load threshold: {load_threshold}")

    # Create hierarchy batches
    print(f"\nBuilding hierarchy paths...")
    batches = create_hierarchy_batches(concept_pack_dir, layers, max_concepts)
    print(f"  Created {len(batches)} hierarchy paths")

    if not batches:
        raise ValueError("No hierarchy paths found")

    # Show sample path
    sample = batches[0]
    print(f"  Sample path: {' → '.join([c[0] for c in sample])}")

    # Detect hidden dim and load lens paths
    print(f"\nDetecting hidden dim...")
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            lens_files = list(layer_dir.glob("*_classifier.pt"))
            if lens_files:
                state_dict = torch.load(lens_files[0], map_location='cpu')
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension")

    print(f"  Hidden dim: {hidden_dim}")

    # Build lens path lookup
    lens_paths = {}
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            for lens_file in layer_dir.glob("*_classifier.pt"):
                concept_name = lens_file.stem.replace("_classifier", "")
                lens_paths[concept_name] = (lens_file, layer)

    print(f"  Found {len(lens_paths)} lens files")

    # Create layer norm
    layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    # Track results
    concept_results = defaultdict(lambda: {
        'in_top_k': 0,
        'is_rank_0': 0,
        'total': 0,
        'ranks': [],
        'activations': [],
        'ancestor_over_fires': [],  # Cases where ancestor ranked higher
    })

    over_fire_counts = defaultdict(lambda: defaultdict(int))

    print(f"\nRunning hierarchical probes...")
    model.eval()

    for batch in tqdm(batches, desc="Paths"):
        # Load lenses for this path
        path_lenses = {}
        for concept_name, layer, ancestors in batch:
            if concept_name not in lens_paths:
                continue
            lens_path, _ = lens_paths[concept_name]
            try:
                state_dict = torch.load(lens_path, map_location='cpu')
                lens = SimpleMLP(hidden_dim).to(device)
                if not list(state_dict.keys())[0].startswith('net.'):
                    state_dict = {f'net.{k}': v for k, v in state_dict.items()}
                lens.load_state_dict(state_dict)
                lens.eval()
                path_lenses[concept_name] = lens
            except:
                pass

        if len(path_lenses) < 2:
            continue

        # For each concept in the path, probe with just that concept name
        for target_concept, target_layer, ancestors in batch:
            if target_concept not in path_lenses:
                continue

            # Create prompt for this concept
            prompt = target_concept

            # Tokenize and run forward pass
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                activation = hidden_states[0, -1, :].float()  # Last token
                activation = layer_norm(activation.unsqueeze(0)).squeeze(0)

            # Run all path lenses and rank
            scores = {}
            for lens_name, lens in path_lenses.items():
                with torch.no_grad():
                    score = lens(activation.unsqueeze(0)).item()
                    scores[lens_name] = score

            # Rank by score
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            rank_0_concept = ranked[0][0] if ranked else None

            # Find target's rank
            target_rank = next((i for i, (name, _) in enumerate(ranked) if name == target_concept), len(ranked))
            target_score = scores.get(target_concept, 0)

            # Check ancestor over-firing
            ancestor_over_fires = []
            for ancestor in ancestors:
                if ancestor in scores and scores[ancestor] > target_score:
                    ancestor_over_fires.append(ancestor)
                    over_fire_counts[ancestor][target_concept] += 1

            # Record results
            concept_results[target_concept]['in_top_k'] += 1 if target_rank < 5 else 0
            concept_results[target_concept]['is_rank_0'] += 1 if target_rank == 0 else 0
            concept_results[target_concept]['total'] += 1
            concept_results[target_concept]['ranks'].append(target_rank)
            concept_results[target_concept]['activations'].append(target_score)
            concept_results[target_concept]['layer'] = target_layer
            if ancestor_over_fires:
                concept_results[target_concept]['ancestor_over_fires'].extend(ancestor_over_fires)

        # Clear lenses
        del path_lenses
        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\nAggregating results...")

    lens_reports = {}
    under_firing = []
    over_firing = []
    well_calibrated = []

    total_rank_0 = 0
    total_probes = 0

    for concept_name, results in concept_results.items():
        rank_0_rate = results['is_rank_0'] / results['total'] if results['total'] > 0 else 0
        in_top_k_rate = results['in_top_k'] / results['total'] if results['total'] > 0 else 0
        avg_rank = np.mean(results['ranks']) if results['ranks'] else 999
        avg_activation = np.mean(results['activations']) if results['activations'] else 0

        total_rank_0 += results['is_rank_0']
        total_probes += results['total']

        # Under-firing: not rank 0 often enough
        needs_boost = rank_0_rate < 0.5

        # Over-firing: this concept appears as over-firing ancestor for others
        times_over_fired = sum(over_fire_counts[concept_name].values())
        is_over_firing = times_over_fired > 3

        lens_reports[concept_name] = {
            'concept': concept_name,
            'layer': results.get('layer', 0),
            'probe_count': results['total'],
            'rank_0_count': results['is_rank_0'],
            'rank_0_rate': rank_0_rate,
            'in_top_k_rate': in_top_k_rate,
            'avg_rank': avg_rank,
            'avg_activation': avg_activation,
            'needs_boost': needs_boost,
            'over_fire_count': times_over_fired,
            'ancestor_over_fires': list(set(results['ancestor_over_fires']))[:5],
        }

        if needs_boost:
            under_firing.append(concept_name)
        elif is_over_firing:
            over_firing.append(concept_name)
        else:
            well_calibrated.append(concept_name)

    avg_rank_0_rate = total_rank_0 / total_probes if total_probes > 0 else 0

    result = BatchedAnalysisResult(
        lens_pack_id=lens_pack_dir.name,
        concept_pack_id=concept_pack_dir.name,
        model_id=str(model.config._name_or_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode='hierarchy',
        batch_size=0,  # Variable path lengths
        top_k=5,
        total_concepts=len(concept_results),
        total_batches=len(batches),
        avg_in_top_k_rate=avg_rank_0_rate,  # Using rank_0 as primary metric
        concepts_needing_boost=len(under_firing),
        concepts_over_firing=len(over_firing),
        lens_reports=lens_reports,
        under_firing=under_firing,
        over_firing=over_firing,
        well_calibrated=well_calibrated,
    )

    return result


def find_concept_positions(
    tokenizer,
    batch_prompt: str,
    concept_names: List[str],
) -> List[int]:
    """
    Find the token position for each concept in the batch prompt.

    Returns the position of the last token of each concept name.
    """
    tokens = tokenizer.encode(batch_prompt, add_special_tokens=True)

    positions = []
    search_start = 0

    for concept in concept_names:
        # Tokenize just the concept
        concept_tokens = tokenizer.encode(concept, add_special_tokens=False)

        # Find where this concept appears in the full token sequence
        # Search from search_start to handle repeated concepts
        found = False
        for i in range(search_start, len(tokens) - len(concept_tokens) + 1):
            if tokens[i:i + len(concept_tokens)] == concept_tokens:
                # Position is the last token of this concept
                positions.append(i + len(concept_tokens) - 1)
                search_start = i + len(concept_tokens)
                found = True
                break

        if not found:
            # Fallback: try to find partial match or use -1
            positions.append(-1)

    return positions


def run_batched_analysis(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    batch_size: int = 5,
    top_k: int = 5,
    repeats: int = 1,
    max_concepts: Optional[int] = None,
    layer_idx: int = 15,
) -> BatchedAnalysisResult:
    """
    Run batched calibration analysis.
    """
    from src.hat.monitoring.lens_manager import DynamicLensManager

    print(f"\n{'='*80}")
    print("BATCHED CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Top-k threshold: {top_k}")
    print(f"  Repeats per concept: {repeats}")

    # Load concepts from hierarchy
    print(f"\nLoading concepts...")
    concepts = []
    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])
        for c in concept_list:
            term = c.get('sumo_term') or c.get('term')
            if term:
                concepts.append((term, layer))

    print(f"  Loaded {len(concepts)} concepts")

    if max_concepts and len(concepts) > max_concepts:
        random.shuffle(concepts)
        concepts = concepts[:max_concepts]
        print(f"  Limited to {max_concepts} concepts for testing")

    # Create batches
    batches = create_batches(concepts, batch_size=batch_size, repeats=repeats)
    print(f"  Created {len(batches)} batches")

    # We'll load lenses per-batch to avoid OOM
    # First, detect hidden dim from a sample lens
    print(f"\nDetecting hidden dim...")
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            lens_files = list(layer_dir.glob("*_classifier.pt"))
            if lens_files:
                state_dict = torch.load(lens_files[0], map_location='cpu')
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension")

    print(f"  Hidden dim: {hidden_dim}")

    # Create layer norm for normalization
    layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    # Build lens path lookup
    lens_paths = {}
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            for lens_file in layer_dir.glob("*_classifier.pt"):
                concept_name = lens_file.stem.replace("_classifier", "")
                lens_paths[concept_name] = (lens_file, layer)

    print(f"  Found {len(lens_paths)} lens files")

    # Track results per concept
    concept_results = defaultdict(lambda: {
        'in_top_k': 0,
        'total': 0,
        'ranks': [],
        'activations': [],
    })

    # Track over-firing: which lenses fire when they shouldn't
    over_fire_counts = defaultdict(lambda: defaultdict(int))

    print(f"\nRunning batched probes...")
    model.eval()

    # Import SimpleMLP for loading lenses
    from src.hat.monitoring.lens_manager import SimpleMLP

    for batch in tqdm(batches, desc="Batches"):
        concept_names = [c[0] for c in batch]
        concept_layers = {c[0]: c[1] for c in batch}

        # Load lenses for this batch only
        batch_lenses = {}
        for concept_name in concept_names:
            if concept_name not in lens_paths:
                continue
            lens_path, layer = lens_paths[concept_name]
            try:
                state_dict = torch.load(lens_path, map_location='cpu')
                lens = SimpleMLP(hidden_dim).to(device)
                # Handle key format
                if not list(state_dict.keys())[0].startswith('net.'):
                    state_dict = {f'net.{k}': v for k, v in state_dict.items()}
                lens.load_state_dict(state_dict)
                lens.eval()
                batch_lenses[concept_name] = lens
            except Exception as e:
                pass

        if len(batch_lenses) < 2:
            continue

        # Create batch prompt
        batch_prompt = ", ".join(concept_names)

        # Tokenize
        inputs = tokenizer(batch_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Find positions for each concept
        positions = find_concept_positions(tokenizer, batch_prompt, concept_names)

        # Forward pass (no generation!)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]

        # For each concept position, run ALL batch lenses and rank
        for target_concept, pos in zip(concept_names, positions):
            if pos < 0 or pos >= hidden_states.shape[1]:
                continue
            if target_concept not in batch_lenses:
                continue

            # Extract activation at this position
            activation = hidden_states[0, pos, :].float()

            # Normalize
            activation = layer_norm(activation.unsqueeze(0)).squeeze(0)

            # Run all batch lenses and rank
            scores = {}
            for lens_name, lens in batch_lenses.items():
                with torch.no_grad():
                    score = lens(activation.unsqueeze(0)).item()
                    scores[lens_name] = score

            # Rank by score (within batch)
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            top_k_names = [name for name, _ in ranked[:top_k]]

            # Check if target concept is in top-k (of the batch)
            in_top_k = target_concept in top_k_names

            # Find rank within batch
            rank = next((i for i, (name, _) in enumerate(ranked) if name == target_concept), len(ranked))

            # Record result
            concept_results[target_concept]['in_top_k'] += 1 if in_top_k else 0
            concept_results[target_concept]['total'] += 1
            concept_results[target_concept]['ranks'].append(rank)
            concept_results[target_concept]['activations'].append(scores.get(target_concept, 0))
            concept_results[target_concept]['layer'] = concept_layers[target_concept]

            # Track over-firing: which OTHER lenses scored higher than target?
            for other_name in top_k_names:
                if other_name != target_concept and scores[other_name] > scores[target_concept]:
                    over_fire_counts[other_name][target_concept] += 1

        # Clear batch lenses to free memory
        del batch_lenses
        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\nAggregating results...")

    lens_reports = {}
    under_firing = []
    over_firing = []
    well_calibrated = []

    total_in_top_k = 0
    total_probes = 0

    for concept_name, results in concept_results.items():
        in_top_k_rate = results['in_top_k'] / results['total'] if results['total'] > 0 else 0
        avg_rank = np.mean(results['ranks']) if results['ranks'] else 999
        avg_activation = np.mean(results['activations']) if results['activations'] else 0

        total_in_top_k += results['in_top_k']
        total_probes += results['total']

        # Determine status
        needs_boost = in_top_k_rate < 0.5  # Less than 50% in top-k = needs boost

        # Check over-firing
        over_fire_on = []
        for target_concept, count in over_fire_counts[concept_name].items():
            if count >= 2:  # Appeared in top-k for another concept multiple times
                over_fire_on.append(target_concept)

        is_over_firing = len(over_fire_on) > 5  # Over-fires on more than 5 other concepts

        lens_reports[concept_name] = {
            'concept': concept_name,
            'layer': results.get('layer', 0),
            'probe_count': results['total'],
            'in_top_k_count': results['in_top_k'],
            'in_top_k_rate': in_top_k_rate,
            'avg_rank': avg_rank,
            'avg_activation': avg_activation,
            'needs_boost': needs_boost,
            'over_fire_count': len(over_fire_on),
            'over_fire_on': over_fire_on[:10],  # Limit to first 10
        }

        if needs_boost:
            under_firing.append(concept_name)
        elif is_over_firing:
            over_firing.append(concept_name)
        else:
            well_calibrated.append(concept_name)

    avg_in_top_k_rate = total_in_top_k / total_probes if total_probes > 0 else 0

    result = BatchedAnalysisResult(
        lens_pack_id=lens_pack_dir.name,
        concept_pack_id=concept_pack_dir.name,
        model_id=str(model.config._name_or_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode='batched',
        batch_size=batch_size,
        top_k=top_k,
        total_concepts=len(concept_results),
        total_batches=len(batches),
        avg_in_top_k_rate=avg_in_top_k_rate,
        concepts_needing_boost=len(under_firing),
        concepts_over_firing=len(over_firing),
        lens_reports=lens_reports,
        under_firing=under_firing,
        over_firing=over_firing,
        well_calibrated=well_calibrated,
    )

    return result


def print_batched_analysis_summary(result: BatchedAnalysisResult):
    """Print human-readable summary."""
    print(f"\n{'='*80}")
    print(f"CALIBRATION ANALYSIS SUMMARY ({result.mode.upper()} MODE)")
    print(f"{'='*80}")
    print(f"  Lens pack: {result.lens_pack_id}")
    print(f"  Concepts probed: {result.total_concepts}")
    print(f"  Total batches: {result.total_batches}")
    print(f"  Batch size: {result.batch_size}")
    print(f"  Top-k: {result.top_k}")
    print()

    # Mode-specific output
    if result.mode == 'production':
        # Production mode: real DynamicLensManager population
        top_k_passers = sum(1 for r in result.lens_reports.values() if r.get('in_top_k', False))
        ancestor_passers = sum(1 for r in result.lens_reports.values() if r.get('passes_ancestor_criterion', False))
        both_passers = sum(1 for r in result.lens_reports.values()
                          if r.get('in_top_k', False) and r.get('passes_ancestor_criterion', False))
        avg_rank = np.mean([r.get('rank', 999) for r in result.lens_reports.values()])
        avg_competing = np.mean([r.get('num_competing_lenses', 0) for r in result.lens_reports.values()])

        print(f"  Competing against: ~{avg_competing:.0f} lenses (avg)")
        print(f"  Pass rates:")
        print(f"    Top-{result.top_k} criterion: {top_k_passers}/{result.total_concepts} ({top_k_passers/result.total_concepts*100:.1f}%)")
        print(f"    Ancestor criterion: {ancestor_passers}/{result.total_concepts} ({ancestor_passers/result.total_concepts*100:.1f}%)")
        print(f"    Both criteria: {both_passers}/{result.total_concepts} ({both_passers/result.total_concepts*100:.1f}%)")
        print(f"  Average rank: {avg_rank:.1f}")
        print()
        print(f"  Concepts needing boost: {result.concepts_needing_boost}")
        print(f"  Well calibrated: {len(result.well_calibrated)}")

        # Show worst under-firing
        if result.under_firing:
            print(f"\n  Top 20 under-firing concepts:")
            sorted_under = sorted(
                [(c, result.lens_reports[c]) for c in result.under_firing if c in result.lens_reports],
                key=lambda x: (x[1].get('rank', 999), -x[1].get('score', 0))
            )
            for concept, report in sorted_under[:20]:
                rank = report.get('rank', 999)
                score = report.get('score', 0)
                in_top_k = "✓" if report.get('in_top_k', False) else "✗"
                beats_anc = "✓" if report.get('passes_ancestor_criterion', False) else "✗"
                failed_anc = report.get('failed_ancestors', [])[:3]
                failed_str = f" (beaten by: {', '.join(failed_anc)})" if failed_anc else ""
                print(f"    {concept}: rank={rank}, top-k={in_top_k}, ancestors={beats_anc}{failed_str}")

    elif result.mode in ('dual_criteria', 'triple_criteria'):
        # Triple-criteria mode: different metrics
        ancestor_passers = sum(1 for r in result.lens_reports.values() if r.get('passes_ancestor_criterion', False))
        random_passers = sum(1 for r in result.lens_reports.values() if r.get('passes_random_criterion', False))
        sibling_passers = sum(1 for r in result.lens_reports.values() if r.get('passes_sibling_criterion', False))
        all_passers = sum(1 for r in result.lens_reports.values()
                         if r.get('passes_ancestor_criterion', False)
                         and r.get('passes_random_criterion', False)
                         and r.get('passes_sibling_criterion', False))

        print(f"  Pass rates:")
        print(f"    Ancestor criterion (rank 0 vs ancestors): {ancestor_passers}/{result.total_concepts} ({ancestor_passers/result.total_concepts*100:.1f}%)")
        print(f"    Random criterion (top 5 vs randoms): {random_passers}/{result.total_concepts} ({random_passers/result.total_concepts*100:.1f}%)")
        print(f"    Sibling criterion (rank 0 vs siblings): {sibling_passers}/{result.total_concepts} ({sibling_passers/result.total_concepts*100:.1f}%)")
        print(f"    All three criteria: {all_passers}/{result.total_concepts} ({all_passers/result.total_concepts*100:.1f}%)")
        print()
        print(f"  Concepts needing boost: {result.concepts_needing_boost}")
        print(f"  Concepts over-firing: {result.concepts_over_firing}")
        print(f"  Well calibrated: {len(result.well_calibrated)}")

        # Show worst under-firing (failed triple criteria)
        if result.under_firing:
            print(f"\n  Top 20 under-firing concepts:")
            sorted_under = sorted(
                [(c, result.lens_reports[c]) for c in result.under_firing if c in result.lens_reports],
                key=lambda x: (x[1].get('ancestor_rank_0_rate', 0), x[1].get('random_top5_rate', 0), x[1].get('sibling_rank_0_rate', 0))
            )
            for concept, report in sorted_under[:20]:
                anc_rate = report.get('ancestor_rank_0_rate', 0)
                rand_rate = report.get('random_top5_rate', 0)
                sib_rate = report.get('sibling_rank_0_rate', 0)
                failed_anc = report.get('failed_ancestors', [])[:2]
                failed_sib = report.get('failed_siblings', [])[:2]
                details = []
                if failed_anc:
                    details.append(f"anc:{','.join(failed_anc)}")
                if failed_sib:
                    details.append(f"sib:{','.join(failed_sib)}")
                detail_str = f" ({'; '.join(details)})" if details else ""
                print(f"    {concept}: anc={anc_rate:.0%}, rand={rand_rate:.0%}, sib={sib_rate:.0%}{detail_str}")

    else:
        # Standard mode: rank distribution
        all_ranks = []
        for concept, report in result.lens_reports.items():
            all_ranks.extend(report.get('ranks', []))

        if all_ranks:
            rank_counts = {}
            for r in all_ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1
            total = len(all_ranks)

            print(f"  Rank distribution (0=best):")
            for r in sorted(rank_counts.keys()):
                pct = rank_counts[r] / total * 100
                bar = "█" * int(pct / 2)
                print(f"    Rank {r}: {rank_counts[r]:4d} ({pct:5.1f}%) {bar}")

            avg_rank = np.mean(all_ranks)
            print(f"\n  Average rank: {avg_rank:.2f}")
            print(f"  Rank 0 rate: {rank_counts.get(0, 0) / total:.1%}")

        print()
        print(f"  Average in-top-{result.top_k} rate: {result.avg_in_top_k_rate:.1%}")
        print(f"  Concepts needing boost (under-firing): {result.concepts_needing_boost}")
        print(f"  Concepts over-firing: {result.concepts_over_firing}")
        print(f"  Well calibrated: {len(result.well_calibrated)}")

        # Show worst under-firing
        if result.under_firing:
            print(f"\n  Top 20 under-firing concepts (need boost):")
            sorted_under = sorted(
                [(c, result.lens_reports[c]) for c in result.under_firing if c in result.lens_reports],
                key=lambda x: x[1].get('in_top_k_rate', 0)
            )
            for concept, report in sorted_under[:20]:
                in_top_k = report.get('in_top_k_rate', 0)
                avg_rank = report.get('avg_rank', 0)
                print(f"    {concept}: {in_top_k:.0%} in top-{result.top_k}, avg rank {avg_rank:.0f}")

    # Show worst over-firing (same for all modes)
    if result.over_firing:
        print(f"\n  Top 10 over-firing concepts:")
        sorted_over = sorted(
            [(c, result.lens_reports[c]) for c in result.over_firing if c in result.lens_reports],
            key=lambda x: -x[1].get('over_fire_count', 0)
        )
        for concept, report in sorted_over[:10]:
            print(f"    {concept}: over-fires on {report.get('over_fire_count', 0)} concepts")


def main():
    parser = argparse.ArgumentParser(description='Fast calibration analysis')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to analyze (default: all)')
    parser.add_argument('--batch-size', type=int, default=5, help='Concepts per batch (random mode)')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k threshold')
    parser.add_argument('--repeats', type=int, default=1, help='Repeats per concept')
    parser.add_argument('--max-concepts', type=int, default=None, help='Limit concepts for testing')
    parser.add_argument('--layer-idx', type=int, default=15, help='Model layer for activations')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--hierarchy', action='store_true',
                        help='Use hierarchy mode: test ancestor→descendant paths')
    parser.add_argument('--dual-criteria', action='store_true',
                        help='Use triple-criteria mode: ancestor (rank 0) + random (top 5) + sibling (rank 0) batches')
    parser.add_argument('--production', action='store_true',
                        help='Use production mode: test against full DynamicLensManager population')
    parser.add_argument('--num-random-batches', type=int, default=3,
                        help='Number of random batches per concept (dual-criteria mode)')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)

    # Determine layers
    if args.layers is None:
        layers = []
        for layer_dir in lens_pack_dir.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace('layer', ''))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()
    else:
        layers = args.layers

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    if args.production:
        result = run_production_analysis(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            top_k=args.top_k,
            max_concepts=args.max_concepts,
            layer_idx=args.layer_idx,
        )
        suffix = "_production"
    elif args.dual_criteria:
        result = run_triple_criteria_analysis(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            num_random_batches=args.num_random_batches,
            max_concepts=args.max_concepts,
            layer_idx=args.layer_idx,
        )
        suffix = "_triple_criteria"
    elif args.hierarchy:
        result = run_hierarchy_analysis(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            max_concepts=args.max_concepts,
            layer_idx=args.layer_idx,
        )
        suffix = "_hierarchy"
    else:
        result = run_batched_analysis(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            batch_size=args.batch_size,
            top_k=args.top_k,
            repeats=args.repeats,
            max_concepts=args.max_concepts,
            layer_idx=args.layer_idx,
        )
        suffix = "_batched"

    print_batched_analysis_summary(result)
    output_path = Path(args.output) if args.output else lens_pack_dir / f"calibration_analysis{suffix}.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n  Saved analysis to: {output_path}")


if __name__ == '__main__':
    main()
