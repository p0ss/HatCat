"""
Lens validation to detect poorly-calibrated classifiers.

Validates that lenses fire specifically on their target domain,
not universally across all inputs (which indicates overfitting to
activation distribution rather than learning semantic concept).
"""

import json
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path


# Cache for loaded hierarchies
_hierarchy_cache: Dict[str, Dict] = {}


def load_hierarchy_for_pack(hierarchy_dir: Path) -> Dict:
    """
    Load hierarchy data for a concept pack.

    Returns dict with:
        - layer0_concepts: List of layer 0 concept names
        - layer0_prompts: Dict of {concept_name: prompt} for validation
        - concept_to_layer0: Dict mapping any concept to its layer0 ancestor
    """
    cache_key = str(hierarchy_dir)
    if cache_key in _hierarchy_cache:
        return _hierarchy_cache[cache_key]

    # Load layer 0 concepts
    layer0_path = hierarchy_dir / "layer0.json"
    with open(layer0_path) as f:
        layer0_data = json.load(f)

    layer0_concepts = [c['sumo_term'] for c in layer0_data['concepts']]

    # Build prompts from layer 0
    layer0_prompts = {}
    for concept in layer0_data['concepts']:
        name = concept['sumo_term']
        definition = concept.get('definition', '')
        if definition and not definition.startswith('Grouping category'):
            layer0_prompts[name] = definition
        else:
            layer0_prompts[name] = f"Topics related to {name.lower()}"

    # Build concept -> layer0 mapping by walking up parent chains
    concept_to_layer0 = {}

    # First, map layer0 to themselves
    for name in layer0_concepts:
        concept_to_layer0[name] = name

    # Load all layers and build parent mapping
    parent_map = {}  # concept -> parent
    for layer_num in range(8):
        layer_path = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_path.exists():
            continue
        with open(layer_path) as f:
            layer_data = json.load(f)
        for concept in layer_data.get('concepts', []):
            name = concept['sumo_term']
            parents = concept.get('parent_concepts', [])
            if parents:
                parent_map[name] = parents[0]  # Use first parent

    # Walk up to find layer0 ancestor for each concept
    def find_layer0_ancestor(concept_name: str, visited: set = None) -> str:
        if visited is None:
            visited = set()
        if concept_name in visited:
            return layer0_concepts[0]  # Cycle detected, use default
        visited.add(concept_name)

        if concept_name in layer0_concepts:
            return concept_name
        if concept_name in concept_to_layer0:
            return concept_to_layer0[concept_name]

        parent = parent_map.get(concept_name)
        if parent:
            ancestor = find_layer0_ancestor(parent, visited)
            concept_to_layer0[concept_name] = ancestor
            return ancestor

        # No parent found, use first layer0 as default
        return layer0_concepts[0]

    # Populate mappings for all concepts
    for concept_name in parent_map:
        if concept_name not in concept_to_layer0:
            concept_to_layer0[concept_name] = find_layer0_ancestor(concept_name)

    result = {
        'layer0_concepts': layer0_concepts,
        'layer0_prompts': layer0_prompts,
        'concept_to_layer0': concept_to_layer0,
    }

    _hierarchy_cache[cache_key] = result
    return result


def load_layer0_validation_prompts(hierarchy_dir: Path = None) -> Dict[str, str]:
    """
    Load layer 0 concepts and generate validation prompts.

    Returns dict of {concept_name: prompt} for validation testing.
    """
    if hierarchy_dir is None:
        # Fallback to old default for backwards compatibility
        hierarchy_dir = Path("data/concept_graph/abstraction_layers")

    hierarchy = load_hierarchy_for_pack(hierarchy_dir)
    return hierarchy['layer0_prompts']


# Default prompts - loaded lazily when needed
DEFAULT_TEST_PROMPTS = None


def get_default_test_prompts(hierarchy_dir: Path = None) -> Dict[str, str]:
    """Get test prompts, loading from hierarchy if provided."""
    global DEFAULT_TEST_PROMPTS

    if hierarchy_dir is not None:
        return load_layer0_validation_prompts(hierarchy_dir)

    if DEFAULT_TEST_PROMPTS is None:
        # Fallback to old default
        DEFAULT_TEST_PROMPTS = load_layer0_validation_prompts()

    return DEFAULT_TEST_PROMPTS


def get_relative_validation_prompts(
    concept_name: str,
    hierarchy_dir: Path,
    max_siblings: int = 5,
) -> Tuple[Dict[str, str], str]:
    """
    Get validation prompts from concept's siblings only.

    This validates against what we actually train against - siblings are used as
    hard negatives during training, so we should be able to discriminate from them.

    Parents are NOT included because ancestors are excluded from negatives during
    training - we want parent lenses to fire when child concepts are present.

    Args:
        concept_name: The concept being validated
        hierarchy_dir: Path to hierarchy directory
        max_siblings: Maximum siblings to include (to keep validation fast)

    Returns:
        Tuple of (prompts_dict, target_concept_name)
        prompts_dict maps concept names to their definition prompts
    """
    # Load all concepts to build lookup
    concept_lookup = {}  # name -> concept dict

    for layer_num in range(8):
        layer_path = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_path.exists():
            continue
        with open(layer_path) as f:
            layer_data = json.load(f)
        for c in layer_data.get('concepts', []):
            concept_lookup[c['sumo_term']] = c

    if concept_name not in concept_lookup:
        # Fallback - return just the concept itself
        return {concept_name: f"The concept of {concept_name}"}, concept_name

    concept = concept_lookup[concept_name]
    prompts = {}

    # Add the target concept
    definition = concept.get('definition', '')
    if definition and not definition.startswith('Grouping category') and not definition.startswith('Sense group'):
        prompts[concept_name] = definition
    else:
        prompts[concept_name] = f"The concept of {concept_name}"

    # Get siblings via parent (but don't include parent itself)
    parents = concept.get('parent_concepts', [])

    for parent_name in parents[:2]:  # Max 2 parents
        if parent_name in concept_lookup:
            parent = concept_lookup[parent_name]

            # Get siblings (parent's other children) - these are our training negatives
            siblings = parent.get('child_concepts', [])
            sibling_count = 0
            for sib_name in siblings:
                if sib_name != concept_name and sib_name in concept_lookup:
                    sib = concept_lookup[sib_name]
                    sib_def = sib.get('definition', '')
                    if sib_def and not sib_def.startswith('Grouping category') and not sib_def.startswith('Sense group'):
                        prompts[sib_name] = sib_def
                    else:
                        prompts[sib_name] = f"The concept of {sib_name}"
                    sibling_count += 1
                    if sibling_count >= max_siblings:
                        break

    # Ensure we have at least 3 prompts for meaningful validation (target + 2 siblings)
    if len(prompts) < 3:
        return None, concept_name  # Not enough siblings to validate

    return prompts, concept_name


def infer_concept_domain(concept: Dict, hierarchy_dir: Path = None) -> str:
    """
    Infer the best matching layer 0 domain for a concept.

    Uses the hierarchy to walk up to the layer 0 ancestor.

    Args:
        concept: Concept dict with 'sumo_term' and optional 'definition'
        hierarchy_dir: Path to hierarchy directory (for accurate lookup)

    Returns:
        Layer 0 concept name
    """
    concept_name = concept['sumo_term']

    if hierarchy_dir is not None:
        try:
            hierarchy = load_hierarchy_for_pack(hierarchy_dir)
            if concept_name in hierarchy['concept_to_layer0']:
                return hierarchy['concept_to_layer0'][concept_name]
            # If not found in mapping, return first layer0 concept as default
            return hierarchy['layer0_concepts'][0]
        except Exception:
            pass

    # Fallback to heuristic matching for backwards compatibility
    try:
        layer0_path = Path("data/concept_graph/abstraction_layers/layer0.json")
        with open(layer0_path) as f:
            layer0_data = json.load(f)

        layer0_names = [c['sumo_term'] for c in layer0_data['concepts']]
        if concept_name in layer0_names:
            return concept_name

        # Heuristic matching based on name patterns
        name_lower = concept_name.lower()

        if 'process' in name_lower or 'action' in name_lower or 'event' in name_lower:
            return 'Process'
        elif 'object' in name_lower or 'device' in name_lower or 'artifact' in name_lower:
            return 'Object'
        elif 'attribute' in name_lower or 'quality' in name_lower:
            return 'Attribute'
        elif 'relation' in name_lower:
            return 'Relation'
        elif 'quantity' in name_lower or 'number' in name_lower:
            return 'Quantity'
        elif 'collection' in name_lower or 'group' in name_lower or 'list' in name_lower:
            return 'Collection'
        elif 'proposition' in name_lower or 'statement' in name_lower:
            return 'Proposition'
        elif 'physical' in name_lower:
            return 'Physical'

        return 'Abstract'

    except Exception:
        return 'Abstract'


def validate_lens_calibration(
    lens_path: Path,
    concept: Dict,
    model,
    tokenizer,
    device: str = "cuda",
    layer_idx: int = 15,
    test_prompts: Optional[Dict[str, str]] = None,
    expected_domain: Optional[str] = None,
    hierarchy_dir: Path = None,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Validate that a lens fires specifically on its target domain.

    Tests the lens on diverse prompts and checks:
    1. Does it fire strongly on expected domain?
    2. Does it fire weakly on other domains?

    A well-calibrated lens should rank high on target domain and
    low on other domains. Universal firing (high rank everywhere)
    indicates poor calibration.

    Args:
        lens_path: Path to saved lens .pt file
        concept: Concept dict with metadata
        model: Language model
        tokenizer: Tokenizer
        device: Device for inference
        layer_idx: Model layer to extract activations from
        test_prompts: Optional custom test prompts by domain
        expected_domain: Optional explicit target domain
        hierarchy_dir: Path to hierarchy directory for accurate domain inference
        top_k: Number of top lenses to check

    Returns:
        Dict with validation metrics:
        - target_rank: Rank on expected domain (lower is better)
        - target_logit: Logit on expected domain
        - avg_other_rank: Average rank on other domains (higher is better)
        - avg_other_logit: Average logit on other domains
        - calibration_score: 0-1 score (higher is better)
        - passed: Whether lens passed validation
    """
    from src.hat.classifiers.classifier import load_classifier

    # Load lens using unified HAT loader (handles legacy and new formats)
    lens = load_classifier(lens_path, device=device)
    lens.eval()

    # Use hierarchy-aware prompts if available
    if test_prompts is None:
        test_prompts = get_default_test_prompts(hierarchy_dir)

    # Infer expected domain using hierarchy if available
    if expected_domain is None:
        expected_domain = infer_concept_domain(concept, hierarchy_dir)

    # Ensure expected domain is in test prompts
    if expected_domain not in test_prompts:
        # Fallback to first available domain
        expected_domain = list(test_prompts.keys())[0]

    # Get target layer
    target_layer = model.model.language_model.layers[layer_idx]

    # Test on each domain
    results = {}

    with torch.no_grad():
        for domain, prompt in test_prompts.items():
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # Capture activations
            captured_hidden = []

            def hook(module, input, output):
                captured_hidden.append(output[0])

            handle = target_layer.register_forward_hook(hook)
            outputs = model(inputs['input_ids'])
            handle.remove()

            if captured_hidden:
                # Get last token activation
                h = captured_hidden[0][:, -1, :].cpu().numpy()
                h_tensor = torch.tensor(h, dtype=torch.float32).to(device)

                # Get lens prediction
                logit = lens(h_tensor).item()

                results[domain] = {
                    'logit': logit,
                }

    # Calculate relative rankings
    sorted_domains = sorted(results.items(), key=lambda x: x[1]['logit'], reverse=True)
    for rank, (domain, data) in enumerate(sorted_domains, 1):
        data['rank'] = rank

    # Extract metrics
    target_rank = results[expected_domain]['rank']
    target_logit = results[expected_domain]['logit']

    other_domains = [d for d in results if d != expected_domain]
    other_ranks = [results[d]['rank'] for d in other_domains]
    other_logits = [results[d]['logit'] for d in other_domains]

    avg_other_rank = np.mean(other_ranks) if other_ranks else 0
    avg_other_logit = np.mean(other_logits) if other_logits else 0

    # Calibration score:
    # - Good if fires strongly on target (rank 1-3) and weakly on others (rank > 5)
    # - Bad if fires uniformly (rank 1 everywhere)

    # Normalize ranks to 0-1 (1 = best rank, 0 = worst rank)
    num_domains = len(results)
    if num_domains > 1:
        target_score = 1.0 - (target_rank - 1) / (num_domains - 1)
        specificity_score = (avg_other_rank - 1) / (num_domains - 1)
    else:
        target_score = 1.0
        specificity_score = 0.0

    # Combined score: balance sensitivity and specificity
    calibration_score = (target_score + specificity_score) / 2

    # Pass criteria: target in top 3 AND avg_other_rank > 4
    # Adjusted for smaller domain counts
    max_target_rank = min(3, num_domains)
    min_other_rank = num_domains / 2  # At least middle rank on average
    passed = (target_rank <= max_target_rank) and (avg_other_rank >= min_other_rank)

    return {
        'concept': concept['sumo_term'],
        'expected_domain': expected_domain,
        'target_rank': target_rank,
        'target_logit': target_logit,
        'avg_other_rank': avg_other_rank,
        'avg_other_logit': avg_other_logit,
        'calibration_score': calibration_score,
        'passed': passed,
        'all_results': results,
    }


def validate_lens_set(
    lens_dir: Path,
    concepts: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    layer_idx: int = 15,
    hierarchy_dir: Path = None,
    save_results: bool = True,
) -> Dict[str, any]:
    """
    Validate all lenses in a directory.

    Args:
        lens_dir: Directory containing lens .pt files
        concepts: List of concept dicts
        model: Language model
        tokenizer: Tokenizer
        device: Device for inference
        layer_idx: Model layer for activations
        hierarchy_dir: Path to hierarchy directory for accurate domain inference
        save_results: Whether to save results to JSON

    Returns:
        Dict with summary statistics and per-lens results
    """
    concept_map = {c['sumo_term']: c for c in concepts}

    lens_files = list(lens_dir.glob("*_classifier.pt"))

    results = []
    passed = 0
    failed = 0

    print(f"Validating {len(lens_files)} lenses...")

    for i, lens_path in enumerate(lens_files):
        concept_name = lens_path.stem.replace('_classifier', '')

        if concept_name not in concept_map:
            print(f"  [{i+1}/{len(lens_files)}] Skipping {concept_name} (no metadata)")
            continue

        concept = concept_map[concept_name]

        try:
            result = validate_lens_calibration(
                lens_path, concept, model, tokenizer, device, layer_idx,
                hierarchy_dir=hierarchy_dir
            )
            results.append(result)

            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  [{i+1}/{len(lens_files)}] {concept_name}: {status} "
                  f"(target=#{result['target_rank']}, others={result['avg_other_rank']:.1f}, "
                  f"score={result['calibration_score']:.2f})")

            if result['passed']:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  [{i+1}/{len(lens_files)}] ERROR validating {concept_name}: {e}")
            failed += 1

    # Summary statistics
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0

    summary = {
        'total_lenses': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': pass_rate,
        'results': results,
    }

    # Save results
    if save_results:
        output_file = lens_dir / 'validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved validation results to {output_file}")

    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total lenses: {total}")
    print(f"Passed: {passed} ({pass_rate:.1%})")
    print(f"Failed: {failed} ({(1-pass_rate):.1%})")

    # Show worst offenders
    failed_lenses = [r for r in results if not r['passed']]
    if failed_lenses:
        print(f"\nWorst calibrated lenses (universal firing):")
        failed_lenses.sort(key=lambda r: r['avg_other_rank'])
        for r in failed_lenses[:10]:
            print(f"  {r['concept']:30s} target=#{r['target_rank']} "
                  f"others={r['avg_other_rank']:.1f} score={r['calibration_score']:.2f}")

    return summary


__all__ = [
    'validate_lens_calibration',
    'validate_lens_set',
    'infer_concept_domain',
    'load_hierarchy_for_pack',
    'get_default_test_prompts',
]
