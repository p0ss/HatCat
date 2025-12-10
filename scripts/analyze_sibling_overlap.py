#!/usr/bin/env python3
"""
Analyze sibling overlap between current hierarchy and trained lenses.

For each trained concept, compares:
1. The siblings it was trained against (from lens pack metadata)
2. The siblings it should have (from current hierarchy)

This identifies concepts that need retraining due to hierarchy changes.

Output categories:
- good (>80% overlap): Training cohort matches current hierarchy
- ok (50-80% overlap): Minor drift, may benefit from retraining
- poor (20-50% overlap): Significant drift, should retrain
- bad (<20% overlap): Trained against wrong cohort, must retrain
- new: No existing lens, needs first training
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_hierarchy(hierarchy_path: Path) -> Dict:
    """Load hierarchy.json."""
    with open(hierarchy_path) as f:
        return json.load(f)


def load_lens_metadata(lens_pack_dir: Path) -> Dict[str, Dict]:
    """Load metadata for all trained lenses."""
    metadata = {}

    for layer_dir in lens_pack_dir.glob("layer*"):
        if not layer_dir.is_dir():
            continue

        layer = int(layer_dir.name.replace("layer", ""))

        # Check for metadata files
        for meta_file in layer_dir.glob("*_metadata.json"):
            with open(meta_file) as f:
                meta = json.load(f)

            concept_name = meta.get("concept_name", meta_file.stem.replace("_metadata", ""))
            concept_key = f"{concept_name}:{layer}"
            metadata[concept_key] = meta

        # Also check for classifier files without metadata
        for clf_file in layer_dir.glob("*_classifier.pt"):
            concept_name = clf_file.stem.replace("_classifier", "")
            concept_key = f"{concept_name}:{layer}"
            if concept_key not in metadata:
                metadata[concept_key] = {"concept_name": concept_name, "layer": layer}

    return metadata


def get_hierarchy_siblings(concept_key: str, hierarchy: Dict) -> Set[str]:
    """Get current siblings for a concept from hierarchy."""
    # Find parent
    parent = hierarchy.get("child_to_parent", {}).get(concept_key)
    if not parent:
        # Root concept or orphan - no siblings
        return set()

    # Get all children of parent (siblings)
    siblings = set(hierarchy.get("parent_to_children", {}).get(parent, []))
    siblings.discard(concept_key)  # Remove self

    return siblings


def get_trained_siblings(concept_key: str, lens_metadata: Dict[str, Dict]) -> Set[str]:
    """Get siblings that were used during training from lens metadata."""
    meta = lens_metadata.get(concept_key, {})

    # Try different metadata fields that might contain sibling info
    trained_siblings = set()

    # Check for explicit sibling list
    if "training_siblings" in meta:
        trained_siblings = set(meta["training_siblings"])
    elif "negative_concepts" in meta:
        trained_siblings = set(meta["negative_concepts"])
    elif "sibling_concepts" in meta:
        trained_siblings = set(meta["sibling_concepts"])

    return trained_siblings


def compute_overlap(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard-like overlap between two sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = perfect match
    if not set_a or not set_b:
        return 0.0  # One empty = no overlap

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def analyze_sibling_overlap(
    hierarchy_path: Path,
    lens_pack_path: Path,
    concept_pack_path: Path
) -> Dict:
    """Analyze sibling overlap for all concepts."""

    # Load data
    hierarchy = load_hierarchy(hierarchy_path)
    lens_metadata = load_lens_metadata(lens_pack_path) if lens_pack_path.exists() else {}

    # Get all concepts from hierarchy
    all_concepts = set(hierarchy.get("child_to_parent", {}).keys())
    all_concepts |= set(hierarchy.get("root_concepts", []))

    results = {
        "good": [],      # >80% overlap
        "ok": [],        # 50-80%
        "poor": [],      # 20-50%
        "bad": [],       # <20%
        "new": [],       # No trained lens
        "stats": {},
        "by_layer": defaultdict(lambda: {"good": 0, "ok": 0, "poor": 0, "bad": 0, "new": 0})
    }

    for concept_key in sorted(all_concepts):
        name, layer = concept_key.rsplit(":", 1)
        layer = int(layer)

        # Get current hierarchy siblings
        hier_siblings = get_hierarchy_siblings(concept_key, hierarchy)

        # Check if we have a trained lens
        if concept_key not in lens_metadata:
            results["new"].append({
                "concept": concept_key,
                "layer": layer,
                "current_siblings": len(hier_siblings)
            })
            results["by_layer"][layer]["new"] += 1
            continue

        # Get trained siblings
        trained_siblings = get_trained_siblings(concept_key, lens_metadata)

        # If no sibling info in metadata, we can't compare - assume needs recheck
        if not trained_siblings and hier_siblings:
            # No metadata about training siblings - conservative: mark as poor
            results["poor"].append({
                "concept": concept_key,
                "layer": layer,
                "overlap": 0.0,
                "current_siblings": len(hier_siblings),
                "trained_siblings": 0,
                "note": "No training sibling metadata"
            })
            results["by_layer"][layer]["poor"] += 1
            continue

        # Compute overlap
        overlap = compute_overlap(hier_siblings, trained_siblings)

        entry = {
            "concept": concept_key,
            "layer": layer,
            "overlap": round(overlap, 3),
            "current_siblings": len(hier_siblings),
            "trained_siblings": len(trained_siblings),
            "new_siblings": len(hier_siblings - trained_siblings),
            "removed_siblings": len(trained_siblings - hier_siblings)
        }

        if overlap >= 0.8:
            results["good"].append(entry)
            results["by_layer"][layer]["good"] += 1
        elif overlap >= 0.5:
            results["ok"].append(entry)
            results["by_layer"][layer]["ok"] += 1
        elif overlap >= 0.2:
            results["poor"].append(entry)
            results["by_layer"][layer]["poor"] += 1
        else:
            results["bad"].append(entry)
            results["by_layer"][layer]["bad"] += 1

    # Compute stats
    total = len(all_concepts)
    results["stats"] = {
        "total_concepts": total,
        "good_count": len(results["good"]),
        "ok_count": len(results["ok"]),
        "poor_count": len(results["poor"]),
        "bad_count": len(results["bad"]),
        "new_count": len(results["new"]),
        "good_pct": round(100 * len(results["good"]) / total, 1) if total else 0,
        "ok_pct": round(100 * len(results["ok"]) / total, 1) if total else 0,
        "poor_pct": round(100 * len(results["poor"]) / total, 1) if total else 0,
        "bad_pct": round(100 * len(results["bad"]) / total, 1) if total else 0,
        "new_pct": round(100 * len(results["new"]) / total, 1) if total else 0,
        "needs_training": len(results["bad"]) + len(results["new"]),
        "should_retrain": len(results["poor"]),
    }

    return results


def compute_meld_blast_radius(results: Dict, hierarchy: Dict) -> Dict:
    """Compute full retraining scope using meld protocol.

    For concepts needing retraining:
    - Add their current siblings (new cohort)
    - Add their immediate parents
    """

    bad_concepts = set(r["concept"] for r in results["bad"])
    new_concepts = set(r["concept"] for r in results["new"])

    # Start with direct targets
    must_retrain = bad_concepts | new_concepts

    # Add siblings of bad concepts
    sibling_additions = set()
    for concept_key in bad_concepts:
        parent = hierarchy.get("child_to_parent", {}).get(concept_key)
        if parent:
            siblings = set(hierarchy.get("parent_to_children", {}).get(parent, []))
            sibling_additions |= siblings

    # Add immediate parents of bad concepts
    parent_additions = set()
    for concept_key in bad_concepts | new_concepts:
        parent = hierarchy.get("child_to_parent", {}).get(concept_key)
        if parent:
            parent_additions.add(parent)

    total_scope = must_retrain | sibling_additions | parent_additions

    return {
        "direct_targets": len(must_retrain),
        "sibling_additions": len(sibling_additions - must_retrain),
        "parent_additions": len(parent_additions - must_retrain - sibling_additions),
        "total_scope": len(total_scope),
        "concepts": sorted(total_scope)
    }


def print_report(results: Dict, verbose: bool = False):
    """Print analysis report."""
    stats = results["stats"]

    print("=" * 60)
    print("SIBLING OVERLAP ANALYSIS")
    print("=" * 60)

    print(f"\nTotal concepts: {stats['total_concepts']}")
    print()

    print("--- OVERLAP CATEGORIES ---")
    print(f"Good (>80%):    {stats['good_count']:4d} ({stats['good_pct']:.1f}%)")
    print(f"OK (50-80%):    {stats['ok_count']:4d} ({stats['ok_pct']:.1f}%)")
    print(f"Poor (20-50%):  {stats['poor_count']:4d} ({stats['poor_pct']:.1f}%)")
    print(f"Bad (<20%):     {stats['bad_count']:4d} ({stats['bad_pct']:.1f}%)")
    print(f"New (no lens):  {stats['new_count']:4d} ({stats['new_pct']:.1f}%)")

    print()
    print("--- TRAINING REQUIRED ---")
    print(f"Must train (bad + new):  {stats['needs_training']}")
    print(f"Should retrain (poor):   {stats['should_retrain']}")

    print()
    print("--- BY LAYER ---")
    for layer in sorted(results["by_layer"].keys()):
        layer_stats = results["by_layer"][layer]
        total = sum(layer_stats.values())
        print(f"  Layer {layer}: {total} concepts "
              f"(good={layer_stats['good']}, ok={layer_stats['ok']}, "
              f"poor={layer_stats['poor']}, bad={layer_stats['bad']}, new={layer_stats['new']})")

    if verbose and results["bad"]:
        print()
        print("--- WORST BAD CONCEPTS (sample) ---")
        for entry in sorted(results["bad"], key=lambda x: x["overlap"])[:20]:
            print(f"  {entry['concept']}: {entry['overlap']*100:.0f}% overlap "
                  f"(+{entry.get('new_siblings', '?')} new, -{entry.get('removed_siblings', '?')} removed)")


def main():
    parser = argparse.ArgumentParser(description="Analyze sibling overlap for trained lenses")
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=Path("concept_packs/sumo-wordnet-v4/hierarchy.json"),
        help="Path to hierarchy.json"
    )
    parser.add_argument(
        "--lens-pack",
        type=Path,
        default=Path("lens_packs/apertus-8b_sumo-wordnet-v4.2"),
        help="Path to lens pack directory"
    )
    parser.add_argument(
        "--concept-pack",
        type=Path,
        default=Path("concept_packs/sumo-wordnet-v4"),
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--blast-radius",
        action="store_true",
        help="Compute full meld blast radius"
    )

    args = parser.parse_args()

    if not args.hierarchy.exists():
        # Try alternate location
        alt_hier = args.concept_pack / "hierarchy.json"
        if alt_hier.exists():
            args.hierarchy = alt_hier
        else:
            print(f"Error: Hierarchy not found: {args.hierarchy}")
            return 1

    print(f"Hierarchy: {args.hierarchy}")
    print(f"Lens pack: {args.lens_pack}")

    results = analyze_sibling_overlap(
        args.hierarchy,
        args.lens_pack,
        args.concept_pack
    )

    print_report(results, verbose=args.verbose)

    if args.blast_radius:
        hierarchy = load_hierarchy(args.hierarchy)
        blast = compute_meld_blast_radius(results, hierarchy)
        print()
        print("--- MELD BLAST RADIUS ---")
        print(f"Direct targets (bad + new): {blast['direct_targets']}")
        print(f"+ Sibling additions:        {blast['sibling_additions']}")
        print(f"+ Parent additions:         {blast['parent_additions']}")
        print(f"= Total scope:              {blast['total_scope']}")
        results["blast_radius"] = blast

    if args.output:
        # Convert defaultdict to regular dict for JSON serialization
        results["by_layer"] = dict(results["by_layer"])
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
