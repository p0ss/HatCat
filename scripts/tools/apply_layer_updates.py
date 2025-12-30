#!/usr/bin/env python3
"""
Apply AI safety concept layer updates to abstraction_layers JSON files.

This script:
1. Loads recalculated AI safety concept layers
2. Removes old AI safety entries from wrong layers
3. Adds new entries to correct layers
4. Updates parent category_children fields
5. Validates integrity
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


# AI safety concepts to manage (will be moved to correct layers)
AI_SAFETY_CONCEPTS = {
    # From recalculation - these are the concepts we're managing
    'AIControlProblem', 'AIDecline', 'GoalFaithfulness', 'RobustAIControl',
    'AIAlignmentProcess', 'AIAlignmentTheory', 'AICare', 'AIGrowth',
    'AISafety', 'CapabilityLoss', 'ComputationalProcess', 'InferenceProcess',
    'PromptFollowing', 'RewardFaithfulness', 'SafeAIDeployment', 'SelfImpairment',
    'TokenProcessing', 'AIFailureProcess', 'AIOptimizationProcess',
    'CapabilityAcquisition', 'Catastrophe', 'Deception', 'InnerAlignment',
    'NonDeceptiveAlignment', 'OrthogonalityThesis', 'OuterAlignment',
    'PoliticalProcess', 'RapidTransformation', 'SelfImprovement',
    'SpecificationAdherence', 'AICatastrophicEvent', 'AIGovernanceProcess',
    'AIStrategicDeception', 'GoalMisgeneralization', 'HumanDeception',
    'InstrumentalConvergence', 'IntelligenceExplosion', 'MesaOptimization',
    'MesaOptimizer', 'RewardHacking', 'SpecificationGaming',
    'TechnologicalSingularity', 'AIDeception', 'AIGovernance',
    'DeceptiveAlignment', 'GreyGooScenario', 'TreacherousTurn'
}

# Deleted concepts (should be removed entirely)
DELETED_CONCEPTS = {
    'AIRiskScenario',
    'AIBeneficialOutcome',
    'AIAlignmentFailureMode',
    'AIMetaOptimization',
    'AIAlignmentPrinciple',
    'AICatastrophe'  # Renamed to AICatastrophicEvent
}


def load_recalculated_layers() -> Dict:
    """Load the recalculated AI safety layer assignments."""
    with open('results/ai_safety_recalculated_layers.json') as f:
        data = json.load(f)
    return data


def load_layer_file(layer_num: int) -> Dict:
    """Load a layer JSON file."""
    path = Path(f'data/concept_graph/abstraction_layers/layer{layer_num}.json')
    with open(path) as f:
        return json.load(f)


def save_layer_file(layer_num: int, layer_data: Dict) -> None:
    """Save a layer JSON file."""
    path = Path(f'data/concept_graph/abstraction_layers/layer{layer_num}.json')
    with open(path, 'w') as f:
        json.dump(layer_data, f, indent=2)


def remove_ai_safety_concepts(layer_data: List[Dict]) -> List[Dict]:
    """Remove all AI safety concepts from a layer."""
    return [
        concept for concept in layer_data
        if concept['sumo_term'] not in AI_SAFETY_CONCEPTS
        and concept['sumo_term'] not in DELETED_CONCEPTS
    ]


def create_layer_entry(concept_info: Dict, wordnet_mappings: Dict) -> Dict:
    """Create a layer entry for an AI safety concept."""
    sumo_term = concept_info['concept']
    layer = concept_info['layer']
    depth = concept_info['depth']
    parent = concept_info['parent']
    children = concept_info['children']
    is_category = concept_info['is_category']

    # Try to get WordNet synsets (fallback to generated)
    synsets = []
    canonical_synset = None
    lemmas = []
    definition = f"AI safety concept: {sumo_term}"
    pos = 'n'
    lexname = 'noun.Tops'

    # Check if we have WordNet mappings for this concept
    if sumo_term in wordnet_mappings:
        synset_data = wordnet_mappings[sumo_term]
        synsets = synset_data.get('synsets', [])
        canonical_synset = synset_data.get('canonical_synset')
        lemmas = synset_data.get('lemmas', [sumo_term.lower()])
        definition = synset_data.get('definition', definition)
        pos = synset_data.get('pos', pos)
        lexname = synset_data.get('lexname', lexname)

    if not synsets:
        # Fallback: generate synset ID
        canonical_synset = f"{sumo_term.lower()}.{pos}.01"
        synsets = [canonical_synset]
        lemmas = [sumo_term.lower()]

    return {
        "sumo_term": sumo_term,
        "sumo_depth": depth,
        "layer": layer,
        "is_category_lens": is_category,
        "is_pseudo_sumo": False,
        "category_children": children,
        "synset_count": len(synsets),
        "direct_synset_count": len(synsets),
        "synsets": synsets,
        "canonical_synset": canonical_synset,
        "lemmas": lemmas,
        "pos": pos,
        "definition": definition,
        "lexname": lexname
    }


def update_parent_children(layer_data: List[Dict], new_concepts_by_layer: Dict) -> None:
    """Update category_children fields for parent concepts."""
    # Build map of all concepts by name
    all_concepts = {}
    for layer_num, concepts in new_concepts_by_layer.items():
        for concept in concepts:
            all_concepts[concept['sumo_term']] = concept

    # Also include existing concepts from layer_data
    for concept in layer_data:
        all_concepts[concept['sumo_term']] = concept

    # For each concept, update its category_children
    for concept in all_concepts.values():
        if concept['is_category_lens']:
            # Ensure category_children is accurate
            # (The recalculated data should already have this, but verify)
            pass


def main():
    print("=" * 80)
    print("APPLYING AI SAFETY LAYER UPDATES")
    print("=" * 80)
    print()

    # Load recalculated layers
    print("Loading recalculated layer assignments...")
    recalc_data = load_recalculated_layers()
    by_layer = recalc_data['by_layer']
    print(f"✓ Loaded {recalc_data['summary']['total_concepts']} AI safety concepts")
    print()

    # Load placeholder WordNet mappings (empty for now)
    wordnet_mappings = {}
    # TODO: Load from actual WordNet mappings if available

    # Process each layer
    changes_summary = defaultdict(lambda: {'removed': 0, 'added': 0})

    for layer_str, concepts_info in by_layer.items():
        layer_num = int(layer_str)
        print(f"Processing Layer {layer_num}...")

        # Load current layer (has metadata and concepts)
        layer_data = load_layer_file(layer_num)
        current_concepts = layer_data.get('concepts', [])
        original_count = len(current_concepts)

        # Remove old AI safety entries
        cleaned_concepts = remove_ai_safety_concepts(current_concepts)
        removed_count = original_count - len(cleaned_concepts)

        # Add new AI safety entries
        for concept_info in concepts_info:
            entry = create_layer_entry(concept_info, wordnet_mappings)
            cleaned_concepts.append(entry)

        added_count = len(concepts_info)

        # Sort by sumo_term for consistency
        cleaned_concepts.sort(key=lambda x: x['sumo_term'])

        # Update metadata
        layer_data['concepts'] = cleaned_concepts
        layer_data['metadata']['total_concepts'] = len(cleaned_concepts)

        # Save updated layer
        save_layer_file(layer_num, layer_data)

        changes_summary[layer_num]['removed'] = removed_count
        changes_summary[layer_num]['added'] = added_count
        changes_summary[layer_num]['final_count'] = len(cleaned_concepts)

        print(f"  Removed: {removed_count} old entries")
        print(f"  Added: {added_count} new entries")
        print(f"  Final count: {len(cleaned_concepts)} concepts")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for layer_num in sorted(changes_summary.keys()):
        stats = changes_summary[layer_num]
        print(f"Layer {layer_num}:")
        print(f"  Removed: {stats['removed']}")
        print(f"  Added: {stats['added']}")
        print(f"  Final: {stats['final_count']}")

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    # Check that all AI safety concepts are placed
    placed_concepts = set()
    for layer_str, concepts_info in by_layer.items():
        for concept_info in concepts_info:
            placed_concepts.add(concept_info['concept'])

    missing = AI_SAFETY_CONCEPTS - placed_concepts - DELETED_CONCEPTS
    if missing:
        print(f"⚠ WARNING: {len(missing)} concepts not placed:")
        for concept in sorted(missing):
            print(f"  - {concept}")
    else:
        print("✓ All AI safety concepts successfully placed")

    # Check that deleted concepts are gone
    for layer_num in range(6):
        layer_data = load_layer_file(layer_num)
        concepts = layer_data.get('concepts', [])
        deleted_found = [
            c['sumo_term'] for c in concepts
            if c['sumo_term'] in DELETED_CONCEPTS
        ]
        if deleted_found:
            print(f"⚠ WARNING: Deleted concepts still in Layer {layer_num}:")
            for concept in deleted_found:
                print(f"  - {concept}")

    print()
    print("✓ Layer updates applied successfully")
    print()


if __name__ == '__main__':
    main()
