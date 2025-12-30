#!/usr/bin/env python3
"""
Apply motive and emotion S-tier patches to layer2.json

This script:
1. Reads noun_motive_patch.json and noun_feeling_patch.json
2. Converts S-tier simplexes to full layer entries
3. Adds them to layer2.json
4. Validates no duplicates or conflicts
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Paths
PATCHES_DIR = Path(__file__).parent.parent / "data" / "concept_graph" / "wordnet_patches"
LAYERS_DIR = Path(__file__).parent.parent / "data" / "concept_graph" / "abstraction_layers"
LAYER2_PATH = LAYERS_DIR / "layer2.json"

MOTIVE_PATCH_PATH = PATCHES_DIR / "noun_motive_patch.json"
FEELING_PATCH_PATH = PATCHES_DIR / "noun_feeling_patch.json"


def load_json(path: Path) -> Any:
    """Load JSON file"""
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save JSON file with formatting"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def create_layer_entry_from_simplex(simplex: Dict, category: str) -> Dict:
    """
    Convert a three-pole simplex from patch file to a full layer entry

    Args:
        simplex: Simplex dict with three_pole_simplex structure
        category: "motive" or "feeling"

    Returns:
        Layer entry dict compatible with layer2.json format
    """
    synset = simplex['synset']
    lemmas = simplex['lemmas']
    definition = simplex['definition']
    three_pole = simplex.get('three_pole_simplex', {})
    justification = simplex.get('s_tier_justification', '')

    # Extract SUMO term from first lemma (capitalize properly)
    sumo_term = lemmas[0].replace('_', '').title()

    # Build layer entry matching existing format
    entry = {
        "sumo_term": sumo_term,
        "sumo_depth": 2,
        "layer": 2,
        "is_category_lens": False,  # Individual concepts, not categories
        "is_pseudo_sumo": False,
        "category_children": [],
        "synset_count": 1,
        "direct_synset_count": 1,
        "synsets": [synset],
        "canonical_synset": synset,
        "lemmas": lemmas,
        "pos": "n",
        "definition": definition,
        "lexname": f"noun.{category}",
        # S-tier specific fields
        "s_tier": True,
        "three_pole_simplex": three_pole,
        "s_tier_justification": justification,
        "training_priority": "critical"
    }

    return entry


def get_existing_synsets(concepts: List[Dict]) -> set:
    """Extract set of existing synsets from concept entries"""
    synsets = set()
    for entry in concepts:
        if 'canonical_synset' in entry:
            synsets.add(entry['canonical_synset'])
        if 'synsets' in entry:
            synsets.update(entry['synsets'])
    return synsets


def apply_patches():
    """Main application logic"""

    print("=" * 80)
    print("Applying Motive and Emotion S-Tier Patches to layer2.json")
    print("=" * 80)

    # Load existing layer2.json
    print(f"\n1. Loading existing layer2.json from {LAYER2_PATH}")
    layer2_full = load_json(LAYER2_PATH)

    # Extract concepts array
    concepts = layer2_full.get("concepts", [])
    metadata = layer2_full.get("metadata", {})

    original_count = len(concepts)
    print(f"   Found {original_count} existing concept entries")

    existing_synsets = get_existing_synsets(concepts)
    print(f"   Tracked {len(existing_synsets)} unique synsets")

    # Load patches
    print(f"\n2. Loading patches:")
    print(f"   - {MOTIVE_PATCH_PATH}")
    motive_patch = load_json(MOTIVE_PATCH_PATH)
    print(f"   - {FEELING_PATCH_PATH}")
    feeling_patch = load_json(FEELING_PATCH_PATH)

    # Process motive S-tier concepts
    print(f"\n3. Processing noun.motive S-tier simplexes:")
    motive_s_tier = motive_patch.get('critical_s_tier', [])
    print(f"   Found {len(motive_s_tier)} S-tier motive simplexes")

    added_motive = 0
    skipped_motive = 0

    for simplex in motive_s_tier:
        synset = simplex['synset']
        if synset in existing_synsets:
            print(f"   [SKIP] {synset} - already exists")
            skipped_motive += 1
        else:
            entry = create_layer_entry_from_simplex(simplex, "motive")
            concepts.append(entry)
            existing_synsets.add(synset)
            print(f"   [ADD]  {synset} - {simplex['lemmas'][0]}")
            added_motive += 1

    print(f"   Added: {added_motive}, Skipped: {skipped_motive}")

    # Process feeling S-tier concepts
    print(f"\n4. Processing noun.feeling S-tier simplexes:")
    feeling_s_tier = feeling_patch.get('critical_s_tier', [])
    print(f"   Found {len(feeling_s_tier)} S-tier feeling simplexes")

    added_feeling = 0
    skipped_feeling = 0

    for simplex in feeling_s_tier:
        synset = simplex['synset']
        if synset in existing_synsets:
            print(f"   [SKIP] {synset} - already exists")
            skipped_feeling += 1
        else:
            entry = create_layer_entry_from_simplex(simplex, "feeling")
            concepts.append(entry)
            existing_synsets.add(synset)
            print(f"   [ADD]  {synset} - {simplex['lemmas'][0]}")
            added_feeling += 1

    print(f"   Added: {added_feeling}, Skipped: {skipped_feeling}")

    # Summary
    total_added = added_motive + added_feeling
    total_skipped = skipped_motive + skipped_feeling
    new_count = len(concepts)

    print(f"\n5. Summary:")
    print(f"   Original entries:  {original_count}")
    print(f"   Added S-tier:      {total_added}")
    print(f"   Skipped (existed): {total_skipped}")
    print(f"   New total:         {new_count}")
    print(f"   Expected:          {original_count + total_added}")

    if new_count != original_count + total_added:
        print(f"   ⚠️  WARNING: Count mismatch!")
        return False

    # Update metadata
    if metadata:
        metadata['total_concepts'] = new_count
        metadata['s_tier_count'] = sum(1 for c in concepts if c.get('s_tier', False))

    # Rebuild full structure
    layer2_updated = {
        "metadata": metadata,
        "concepts": concepts
    }

    # Save updated layer2.json
    backup_path = LAYER2_PATH.with_suffix('.json.backup')
    print(f"\n6. Creating backup at {backup_path}")
    save_json(backup_path, layer2_full)  # Backup original

    print(f"\n7. Writing updated layer2.json")
    save_json(LAYER2_PATH, layer2_updated)

    print(f"\n✅ Patches applied successfully!")
    print(f"\n📊 S-Tier Breakdown:")
    print(f"   Motive simplexes:  {added_motive}")
    print(f"   Feeling simplexes: {added_feeling}")
    print(f"   Total S-tier:      {total_added}")

    # Count total S-tier in layer
    s_tier_count = sum(1 for e in concepts if e.get('s_tier', False))
    print(f"\n   Total S-tier concepts in layer2: {s_tier_count}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Review layer2.json to verify entries look correct")
    print("  2. Modify sumo_data_generation.py to support three-pole training")
    print("  3. Wait for v2 training to complete (1819/3278)")
    print("  4. Retrain with expanded S-tier concept set")
    print("  5. Validate homeostatic steering quality")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = apply_patches()
    exit(0 if success else 1)
