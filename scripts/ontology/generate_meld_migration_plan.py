#!/usr/bin/env python3
"""
Generate migration plan for lens pack after melds have been applied.

Reads the concept pack to identify:
1. New concepts added by melds (need training)
2. Parent concepts that got new children (must retrain - negative samples changed)
3. Sibling concepts (should retrain - contrastive examples changed)
4. Unchanged concepts (can copy lenses directly)

Usage:
    python scripts/generate_meld_migration_plan.py \
        --concept-pack first-light \
        --source-lens-pack lens_packs/apertus-8b_sumo-wordnet-v4.2 \
        --output results/first-light_migration_plan.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_hierarchy(concept_pack_dir: Path) -> dict:
    """Load the concept hierarchy from the pack."""
    hierarchy_dir = concept_pack_dir / "hierarchy"
    hierarchy = {}

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)
            for concept in layer_data.get("concepts", []):
                # Handle both old ("term") and new ("sumo_term") formats
                term = concept.get("sumo_term") or concept.get("term")
                if term:
                    hierarchy[term] = concept

    return hierarchy


def load_pack_info(concept_pack_dir: Path) -> dict:
    """Load pack.json to get meld history."""
    pack_file = concept_pack_dir / "pack.json"
    with open(pack_file) as f:
        return json.load(f)


def get_trained_concepts(lens_pack_dir: Path) -> set:
    """Get set of concepts that have trained lenses."""
    trained = set()
    for layer_dir in lens_pack_dir.glob("layer*"):
        for lens_file in layer_dir.glob("*_classifier.pt"):
            concept = lens_file.stem.replace("_classifier", "")
            trained.add(concept)
    return trained


def get_parent_child_map(hierarchy: dict) -> tuple:
    """Build parent->children and child->parent maps."""
    parent_to_children = defaultdict(set)
    child_to_parent = {}

    for term, concept in hierarchy.items():
        parents = concept.get("parent_concepts", [])
        for parent in parents:
            parent_to_children[parent].add(term)
            child_to_parent[term] = parent

    return parent_to_children, child_to_parent


def get_siblings(concept: str, parent_to_children: dict, child_to_parent: dict) -> set:
    """Get siblings of a concept (children of the same parent)."""
    parent = child_to_parent.get(concept)
    if not parent:
        return set()

    siblings = parent_to_children.get(parent, set()).copy()
    siblings.discard(concept)  # Remove self
    return siblings


def identify_new_concepts(pack_info: dict, melds_since: str = None) -> set:
    """Identify concepts added by recent melds."""
    new_concepts = set()

    melds = pack_info.get("melds_applied", [])

    for meld in melds:
        # If melds_since specified, only consider melds after that date
        if melds_since:
            meld_date = meld.get("melded_at", "")
            if meld_date < melds_since:
                continue

        # For each meld, we need to look at the source meld file to get concepts
        # For now, we'll use the meld request ID to identify recent melds
        # and mark all concepts from recent melds as new
        meld_id = meld.get("meld_request_id", "")

        # Heuristic: melds from the restructuring batch have @0.1.0 version
        # and specific naming patterns
        if "@0.1.0" in meld_id and any(pattern in meld_id for pattern in [
            "-melds", "abstractentity", "appliances", "artifact", "bodypart",
            "building", "cognitive", "container", "device", "electronics",
            "elemental", "field", "financial", "geographic", "internal",
            "kitchen", "land", "machines", "measuring", "misc", "motion",
            "organization", "proposition", "region", "relation", "science",
            "service", "social", "speech", "sport", "stating", "stationary",
            "text", "time", "translocation", "vehicle", "water"
        ]):
            concepts_added = meld.get("concepts_added", 0)
            if concepts_added > 0:
                # Mark this meld for concept extraction
                new_concepts.add(meld_id)

    return new_concepts


def main():
    parser = argparse.ArgumentParser(description="Generate lens migration plan")
    parser.add_argument("--concept-pack", required=True, help="Target concept pack name")
    parser.add_argument("--source-lens-pack", required=True, help="Source lens pack directory")
    parser.add_argument("--output", required=True, help="Output migration plan JSON")
    parser.add_argument("--melds-since", help="Only consider melds after this ISO date")
    args = parser.parse_args()

    concept_pack_dir = Path(f"concept_packs/{args.concept_pack}")
    source_lens_dir = Path(args.source_lens_pack)
    output_path = Path(args.output)

    print("Loading concept hierarchy...")
    hierarchy = load_hierarchy(concept_pack_dir)
    print(f"  Loaded {len(hierarchy)} concepts")

    print("Loading pack info...")
    pack_info = load_pack_info(concept_pack_dir)

    print("Getting trained concepts from source lens pack...")
    trained_concepts = get_trained_concepts(source_lens_dir)
    print(f"  Found {len(trained_concepts)} trained lenses")

    print("Building parent-child maps...")
    parent_to_children, child_to_parent = get_parent_child_map(hierarchy)

    # Identify new concepts (those in hierarchy but not in trained set)
    all_concepts = set(hierarchy.keys())
    new_concepts = all_concepts - trained_concepts
    print(f"  New concepts (not in source): {len(new_concepts)}")

    # Identify parents that need retraining
    # A parent needs retraining if it has new children
    must_retrain = set()
    for new_concept in new_concepts:
        parent = child_to_parent.get(new_concept)
        if parent and parent in trained_concepts:
            must_retrain.add(parent)
    print(f"  Parents to retrain: {len(must_retrain)}")

    # Identify siblings that should be retrained
    should_retrain = set()
    for new_concept in new_concepts:
        siblings = get_siblings(new_concept, parent_to_children, child_to_parent)
        for sibling in siblings:
            if sibling in trained_concepts and sibling not in new_concepts:
                should_retrain.add(sibling)

    # Also add siblings of retrained parents' children
    for parent in must_retrain:
        for child in parent_to_children.get(parent, []):
            if child in trained_concepts and child not in new_concepts:
                should_retrain.add(child)

    # Remove overlap
    should_retrain -= must_retrain
    print(f"  Siblings to retrain: {len(should_retrain)}")

    # Concepts that can be copied as-is
    copy_concepts = trained_concepts - new_concepts - must_retrain - should_retrain
    print(f"  Concepts to copy unchanged: {len(copy_concepts)}")

    # Organize by layer for the migration tool
    lenses_to_copy = defaultdict(list)
    for concept in copy_concepts:
        layer = hierarchy.get(concept, {}).get("layer", 0)
        lenses_to_copy[str(layer)].append(concept)

    # Build migration plan
    migration_plan = {
        "created": datetime.now().isoformat(),
        "source_concept_pack": "sumo-wordnet-v4",
        "target_concept_pack": args.concept_pack,
        "source_lens_pack": str(source_lens_dir),
        "summary": {
            "total_concepts": len(all_concepts),
            "new_count": len(new_concepts),
            "must_retrain_count": len(must_retrain),
            "should_retrain_count": len(should_retrain),
            "copy_count": len(copy_concepts),
            "total_to_train": len(new_concepts) + len(must_retrain) + len(should_retrain),
        },
        "new_concepts": sorted(list(new_concepts)),
        "must_retrain": sorted(list(must_retrain)),
        "should_retrain": sorted(list(should_retrain)),
        "lenses_to_copy": {k: sorted(v) for k, v in lenses_to_copy.items()},
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(migration_plan, f, indent=2)

    print()
    print("=" * 70)
    print("MIGRATION PLAN SUMMARY")
    print("=" * 70)
    print(f"Total concepts in target:     {len(all_concepts)}")
    print(f"New concepts to train:        {len(new_concepts)}")
    print(f"Parents to retrain:           {len(must_retrain)}")
    print(f"Siblings to retrain:          {len(should_retrain)}")
    print(f"Lenses to copy unchanged:     {len(copy_concepts)}")
    print(f"Total needing training:       {len(new_concepts) + len(must_retrain) + len(should_retrain)}")
    print()
    print(f"Migration plan written to: {output_path}")


if __name__ == "__main__":
    main()
