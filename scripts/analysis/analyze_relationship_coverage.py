#!/usr/bin/env python3
"""
Analyze relationship coverage and identify concepts that need more children loaded.

This script helps prioritize Layer 5 expansion by:
1. Finding parent concepts with many unloaded children
2. Scoring children by AI safety relevance (manual input needed)
3. Recommending which children to add to Layer 5
"""

import json
from pathlib import Path
from collections import defaultdict
from nltk.corpus import wordnet as wn
from typing import Dict, List, Set, Tuple


def load_our_concepts() -> Dict[int, Dict]:
    """Load all concepts from our layers."""
    layer_dir = Path(__file__).parent.parent / "data" / "concept_graph" / "abstraction_layers"
    concepts_by_layer = {}

    for layer_num in range(6):
        layer_file = layer_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concepts_by_layer[layer_num] = {
            'metadata': layer_data.get('metadata', {}),
            'concepts': layer_data['concepts']
        }

    return concepts_by_layer


def get_our_synsets(concepts_by_layer: Dict) -> Set[str]:
    """Get all synsets we currently have loaded."""
    all_synsets = set()

    for layer_data in concepts_by_layer.values():
        for concept in layer_data['concepts']:
            all_synsets.update(concept.get('synsets', []))

    return all_synsets


def analyze_parent_children_coverage(concepts_by_layer: Dict) -> List[Dict]:
    """Find parents with unloaded children."""
    our_synsets = get_our_synsets(concepts_by_layer)

    # Convert to synset objects
    our_synset_objects = set()
    for synset_name in our_synsets:
        if '.n.' in synset_name:  # Focus on nouns
            try:
                our_synset_objects.add(wn.synset(synset_name))
            except:
                pass

    # Analyze each concept
    coverage_analysis = []

    for layer_num, layer_data in concepts_by_layer.items():
        if layer_num >= 5:  # Skip Layer 5+ for parent analysis
            continue

        for concept in layer_data['concepts']:
            sumo_term = concept['sumo_term']
            synsets = concept.get('synsets', [])

            if not synsets:
                continue

            # For each synset in this concept, check its children
            for synset_name in synsets:
                if '.n.' not in synset_name:
                    continue

                try:
                    synset = wn.synset(synset_name)
                except:
                    continue

                # Get hyponyms (children)
                hyponyms = synset.hyponyms()

                if not hyponyms:
                    continue

                # Check how many children are loaded
                loaded_children = [h for h in hyponyms if h in our_synset_objects]
                unloaded_children = [h for h in hyponyms if h not in our_synset_objects]

                if unloaded_children:
                    coverage_analysis.append({
                        'layer': layer_num,
                        'sumo_term': sumo_term,
                        'synset': synset_name,
                        'definition': synset.definition(),
                        'total_children': len(hyponyms),
                        'loaded_children': len(loaded_children),
                        'unloaded_children': len(unloaded_children),
                        'coverage_pct': len(loaded_children) / len(hyponyms) * 100 if hyponyms else 0,
                        'unloaded_list': [(h.name(), h.definition()) for h in unloaded_children[:10]],  # Sample first 10
                        'lexname': synset.lexname(),
                    })

    return coverage_analysis


def prioritize_by_relationship_density(coverage_analysis: List[Dict]) -> List[Dict]:
    """Sort by number of unloaded children (relationship density)."""
    return sorted(coverage_analysis, key=lambda x: x['unloaded_children'], reverse=True)


def prioritize_by_domain(coverage_analysis: List[Dict], priority_domains: List[str]) -> List[Dict]:
    """Filter and sort by semantic domain priority."""
    filtered = [c for c in coverage_analysis if c['lexname'] in priority_domains]
    return sorted(filtered, key=lambda x: (
        priority_domains.index(c['lexname']),
        -x['unloaded_children']
    ))


def print_coverage_report(coverage_analysis: List[Dict], top_n: int = 20):
    """Print human-readable coverage report."""
    print("=" * 100)
    print("RELATIONSHIP COVERAGE ANALYSIS")
    print("=" * 100)
    print()
    print(f"Concepts with unloaded children: {len(coverage_analysis)}")
    print()

    print(f"Top {top_n} parents by unloaded children count:")
    print("-" * 100)
    print(f"{'Layer':<6} {'SUMO Concept':<30} {'Synset':<25} {'Children':<10} {'Loaded':<8} {'Missing':<8} {'%':<6}")
    print("-" * 100)

    for entry in coverage_analysis[:top_n]:
        print(f"{entry['layer']:<6} "
              f"{entry['sumo_term']:<30} "
              f"{entry['synset']:<25} "
              f"{entry['total_children']:<10} "
              f"{entry['loaded_children']:<8} "
              f"{entry['unloaded_children']:<8} "
              f"{entry['coverage_pct']:>5.1f}%")

    print()
    print("=" * 100)


def print_detailed_analysis(coverage_analysis: List[Dict], concept_name: str):
    """Print detailed analysis for a specific concept."""
    matches = [c for c in coverage_analysis if concept_name.lower() in c['sumo_term'].lower()]

    if not matches:
        print(f"No matches found for '{concept_name}'")
        return

    for entry in matches:
        print("=" * 100)
        print(f"DETAILED ANALYSIS: {entry['sumo_term']}")
        print("=" * 100)
        print(f"Layer: {entry['layer']}")
        print(f"Synset: {entry['synset']}")
        print(f"Definition: {entry['definition']}")
        print(f"Lexname: {entry['lexname']}")
        print(f"Total children: {entry['total_children']}")
        print(f"Loaded: {entry['loaded_children']} ({entry['coverage_pct']:.1f}%)")
        print(f"Unloaded: {entry['unloaded_children']}")
        print()
        print(f"Sample unloaded children (first 10):")
        for child_name, child_def in entry['unloaded_list']:
            print(f"  - {child_name}")
            print(f"    {child_def}")
        print()


def identify_tier1_candidates(coverage_analysis: List[Dict]) -> List[Dict]:
    """Identify Tier 1 critical gaps based on heuristics."""

    # Tier 1: noun.motive domain + noun.Tops + critical safety domains
    tier1_domains = ['noun.motive', 'noun.Tops', 'noun.feeling', 'noun.communication']
    tier1_keywords = ['motivation', 'ethical', 'conscience', 'deception', 'lying', 'emotion']

    tier1 = []

    for entry in coverage_analysis:
        # Domain-based inclusion
        if entry['lexname'] in tier1_domains:
            tier1.append(entry)
            continue

        # Keyword-based inclusion
        text = f"{entry['sumo_term']} {entry['definition']}".lower()
        if any(kw in text for kw in tier1_keywords):
            tier1.append(entry)
            continue

    return tier1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze relationship coverage')
    parser.add_argument('--detail', type=str, help='Show detailed analysis for concept name')
    parser.add_argument('--top', type=int, default=20, help='Show top N concepts')
    parser.add_argument('--tier1', action='store_true', help='Show Tier 1 critical gaps only')
    parser.add_argument('--domain', type=str, help='Filter by semantic domain (e.g., noun.motive)')
    args = parser.parse_args()

    print("Loading concepts from layers...")
    concepts_by_layer = load_our_concepts()

    print("Analyzing relationship coverage...")
    coverage_analysis = analyze_parent_children_coverage(concepts_by_layer)

    # Sort by relationship density
    coverage_analysis = prioritize_by_relationship_density(coverage_analysis)

    if args.detail:
        print_detailed_analysis(coverage_analysis, args.detail)
    elif args.tier1:
        tier1 = identify_tier1_candidates(coverage_analysis)
        print(f"\nTier 1 Critical Gaps: {len(tier1)} concepts")
        print_coverage_report(tier1, top_n=len(tier1))
    elif args.domain:
        domain_filtered = [c for c in coverage_analysis if c['lexname'] == args.domain]
        print(f"\nDomain: {args.domain} ({len(domain_filtered)} concepts)")
        print_coverage_report(domain_filtered, top_n=len(domain_filtered))
    else:
        print_coverage_report(coverage_analysis, top_n=args.top)

    # Summary statistics
    print()
    print("SUMMARY STATISTICS")
    print("=" * 100)

    total_parents = len(coverage_analysis)
    total_missing_children = sum(c['unloaded_children'] for c in coverage_analysis)
    avg_missing = total_missing_children / total_parents if total_parents > 0 else 0

    # Categorize by missing children count
    low_missing = len([c for c in coverage_analysis if c['unloaded_children'] <= 5])
    med_missing = len([c for c in coverage_analysis if 5 < c['unloaded_children'] <= 15])
    high_missing = len([c for c in coverage_analysis if c['unloaded_children'] > 15])

    print(f"Total parent concepts with unloaded children: {total_parents}")
    print(f"Total unloaded children across all parents: {total_missing_children}")
    print(f"Average unloaded children per parent: {avg_missing:.1f}")
    print()
    print("Distribution:")
    print(f"  1-5 unloaded children: {low_missing} parents")
    print(f"  6-15 unloaded children: {med_missing} parents")
    print(f"  >15 unloaded children: {high_missing} parents")
    print()

    # Tier 1 summary
    tier1 = identify_tier1_candidates(coverage_analysis)
    print(f"Tier 1 critical gaps identified: {len(tier1)} concepts")
    print(f"  Estimated training time: ~{len(tier1) * 18}s ({len(tier1) * 18 / 60:.1f} minutes)")
    print()
