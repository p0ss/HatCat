#!/usr/bin/env python3
"""
Analyze bad sibling concepts and their meld cascade.

"Bad" concepts are those with <20% overlap between:
- True siblings (children of same parent in hierarchy)
- Training siblings (concepts at same layer, used during training)

This mismatch means they were trained against the wrong cohort.

The meld protocol requires retraining:
- Bad concepts themselves
- Their siblings (will be retrained with correct cohort)
- Their immediate parents (children changed)

Optional cascade (for full correctness):
- All ancestors
- Ancestor siblings
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_bad_sibling_cascade(hierarchy_path: Path, verbose: bool = False) -> dict:
    """Analyze bad concepts and compute cascade scope."""

    with open(hierarchy_path) as f:
        hier = json.load(f)

    # Build layer mapping
    current_layers = {}
    for c in set(hier['child_to_parent'].keys()) | set(hier.get('root_concepts', [])):
        name, layer = c.rsplit(':', 1)
        current_layers[c] = int(layer)

    layer_concepts = defaultdict(list)
    for c, layer in current_layers.items():
        layer_concepts[layer].append(c)

    # Find bad concepts (<20% sibling overlap)
    bad_concepts = set()
    bad_details = []

    for c, parent in hier['child_to_parent'].items():
        c_layer = current_layers[c]
        true_siblings = set(hier['parent_to_children'].get(parent, [])) - {c}
        training_siblings = set(layer_concepts[c_layer]) - {c}

        if len(true_siblings) == 0:
            continue

        overlap = len(true_siblings & training_siblings)
        ratio = overlap / len(true_siblings)

        if ratio < 0.2:
            bad_concepts.add(c)
            if verbose:
                bad_details.append({
                    "concept": c,
                    "parent": parent,
                    "true_siblings": len(true_siblings),
                    "training_siblings": len(training_siblings),
                    "overlap": overlap,
                    "ratio": round(ratio, 3)
                })

    # Trace the cascade step by step
    print('Step-by-step meld cascade:')
    print(f'Step 0 - Seed bad concepts: {len(bad_concepts)}')

    # Step 1: Add old siblings
    step1 = set(bad_concepts)
    for c in bad_concepts:
        if c in hier['child_to_parent']:
            parent = hier['child_to_parent'][c]
            siblings = set(hier['parent_to_children'].get(parent, []))
            step1.update(siblings)
    print(f'Step 1 - + old siblings: {len(step1)} (+{len(step1)-len(bad_concepts)})')

    # Step 2: Add immediate parents
    step2 = set(step1)
    for c in bad_concepts:
        if c in hier['child_to_parent']:
            step2.add(hier['child_to_parent'][c])
    print(f'Step 2 - + immediate parents: {len(step2)} (+{len(step2)-len(step1)})')

    # Step 3: Recursively add all ancestors
    step3 = set(step2)
    to_check = [c for c in step2 if c in hier['child_to_parent']]
    while to_check:
        c = to_check.pop()
        if c in hier['child_to_parent']:
            parent = hier['child_to_parent'][c]
            if parent not in step3:
                step3.add(parent)
                to_check.append(parent)
    print(f'Step 3 - + all ancestors: {len(step3)} (+{len(step3)-len(step2)})')

    # Step 4: For each ancestor, add its siblings too
    step4 = set(step3)
    ancestors_added = step3 - step1
    for a in ancestors_added:
        if a in hier['child_to_parent']:
            parent = hier['child_to_parent'][a]
            siblings = set(hier['parent_to_children'].get(parent, []))
            step4.update(siblings)
    print(f'Step 4 - + ancestor siblings: {len(step4)} (+{len(step4)-len(step3)})')

    total_concepts = len(current_layers)
    print()
    print(f'Total meld blast: {len(step4)} ({100*len(step4)/total_concepts:.1f}%)')

    # Conservative scope (skip ancestor cascade)
    print()
    print('=== CONSERVATIVE: Skip ancestor cascade ===')
    conservative = set(bad_concepts)
    for c in bad_concepts:
        if c in hier['child_to_parent']:
            parent = hier['child_to_parent'][c]
            siblings = set(hier['parent_to_children'].get(parent, []))
            conservative.update(siblings)
            conservative.add(parent)  # Just immediate parent, no cascade

    print(f'Bad + siblings + immediate parent: {len(conservative)} ({100*len(conservative)/total_concepts:.1f}%)')

    # Return structured results
    return {
        "total_concepts": total_concepts,
        "bad_concepts": len(bad_concepts),
        "step1_with_siblings": len(step1),
        "step2_with_parents": len(step2),
        "step3_with_ancestors": len(step3),
        "step4_full_blast": len(step4),
        "conservative_scope": len(conservative),
        "full_blast_pct": round(100 * len(step4) / total_concepts, 1),
        "conservative_pct": round(100 * len(conservative) / total_concepts, 1),
        "bad_concept_list": sorted(bad_concepts),
        "conservative_list": sorted(conservative),
        "full_blast_list": sorted(step4),
        "bad_details": bad_details if verbose else []
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze bad sibling cascade")
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=Path("concept_packs/sumo-wordnet-v4/hierarchy.json"),
        help="Path to hierarchy.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed bad concept info"
    )

    args = parser.parse_args()

    if not args.hierarchy.exists():
        print(f"Error: Hierarchy not found: {args.hierarchy}")
        return 1

    print(f"Hierarchy: {args.hierarchy}")
    print()

    results = analyze_bad_sibling_cascade(args.hierarchy, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
