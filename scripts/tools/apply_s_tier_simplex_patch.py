#!/usr/bin/env python3
"""
Apply S-tier simplex patch to layer2.json

This script:
1. Reads s_tier_simplexes_from_review.json
2. Enriches each simplex with WordNet data (lemmas, definitions)
3. Creates layer2 entries for each three-pole simplex
4. Adds them to layer2.json without duplicates
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from nltk.corpus import wordnet as wn

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PATCHES_DIR = PROJECT_ROOT / "data" / "concept_graph" / "wordnet_patches"
LAYERS_DIR = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers"

PATCH_PATH = PATCHES_DIR / "s_tier_simplexes_from_review.json"
LAYER2_PATH = LAYERS_DIR / "layer2.json"


def load_json(path: Path) -> Any:
    """Load JSON file"""
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save JSON file with formatting"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def enrich_pole_with_wordnet(synset_id: str) -> Dict:
    """
    Enrich a pole with WordNet data

    Args:
        synset_id: WordNet synset ID (e.g., "composure.n.01")

    Returns:
        Dict with synset, lemmas, and definition
    """
    try:
        ss = wn.synset(synset_id)
        return {
            "synset": synset_id,
            "lemmas": [l.name() for l in ss.lemmas()],
            "definition": ss.definition()
        }
    except Exception as e:
        print(f"  ⚠️  Warning: Could not find {synset_id} in WordNet: {e}")
        return {
            "synset": synset_id,
            "lemmas": [],
            "definition": ""
        }


def create_layer_entry_from_simplex(simplex: Dict) -> Dict:
    """
    Convert a three-pole simplex to a layer2 entry

    Args:
        simplex: Simplex dict from patch file

    Returns:
        Layer entry dict compatible with layer2.json format
    """
    dimension = simplex['dimension']
    three_pole = simplex['three_pole_simplex']

    # Get the neutral homeostasis as the canonical synset
    neutral = three_pole['neutral_homeostasis']
    canonical_synset = neutral['synset']

    # Collect all synsets in the simplex
    all_synsets = [
        three_pole['negative_pole']['synset'],
        neutral['synset'],
        three_pole['positive_pole']['synset']
    ]

    # Use neutral pole's first lemma as SUMO term
    lemmas = neutral.get('lemmas', [])
    if lemmas:
        sumo_term = lemmas[0].replace('_', '').title()
    else:
        # Fallback: use dimension name
        sumo_term = dimension.replace('_', ' ').title().replace(' ', '')

    # Determine lexname from synsets
    try:
        ss = wn.synset(canonical_synset)
        lexname = ss.lexname()
    except:
        lexname = "noun.feeling"  # Default fallback

    # Build layer entry
    entry = {
        "sumo_term": f"{sumo_term}_Simplex",
        "sumo_depth": 2,
        "layer": 2,
        "is_category_lens": False,
        "is_pseudo_sumo": True,  # These are synthetic three-pole concepts
        "category_children": [],
        "synset_count": 3,  # Three poles
        "direct_synset_count": 3,
        "synsets": all_synsets,
        "canonical_synset": canonical_synset,
        "lemmas": lemmas if lemmas else [dimension],
        "pos": "n",
        "definition": neutral.get('definition', ''),
        "lexname": lexname,
        # S-tier specific fields
        "s_tier": True,
        "simplex_dimension": dimension,
        "three_pole_simplex": three_pole,
        "s_tier_justification": simplex.get('s_tier_justification', ''),
        "training_priority": simplex.get('training_priority', 'standard')
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
        # Also check if this is already an S-tier simplex with same dimension
        if entry.get('s_tier') and 'simplex_dimension' in entry:
            synsets.add(f"SIMPLEX:{entry['simplex_dimension']}")
    return synsets


def apply_patch():
    """Main application logic"""

    print("=" * 80)
    print("Applying S-Tier Simplex Patch to layer2.json")
    print("=" * 80)

    # Load existing layer2.json
    print(f"\n1. Loading existing layer2.json from {LAYER2_PATH}")
    layer2_full = load_json(LAYER2_PATH)

    concepts = layer2_full.get("concepts", [])
    metadata = layer2_full.get("metadata", {})

    original_count = len(concepts)
    print(f"   Found {original_count} existing concept entries")

    existing_synsets = get_existing_synsets(concepts)
    print(f"   Tracked {len(existing_synsets)} unique synsets/simplexes")

    # Load S-tier patch
    print(f"\n2. Loading S-tier simplex patch from {PATCH_PATH}")
    patch = load_json(PATCH_PATH)

    total_simplexes = patch['metadata']['total_s_tier_simplexes']
    print(f"   Found {total_simplexes} S-tier simplexes")

    # Process all priority levels
    print(f"\n3. Processing S-tier simplexes:")

    added_count = 0
    skipped_count = 0

    for priority_section in ['critical_s_tier', 'high_priority_s_tier',
                              'medium_priority_s_tier', 'standard_s_tier']:
        simplexes = patch.get(priority_section, [])

        if not simplexes:
            continue

        priority_name = priority_section.replace('_s_tier', '').upper()
        print(f"\n   {priority_name}: {len(simplexes)} simplexes")

        for simplex in simplexes:
            dimension = simplex['dimension']
            dimension_key = f"SIMPLEX:{dimension}"

            # Check if already exists
            if dimension_key in existing_synsets:
                print(f"      [SKIP] {dimension} - already exists")
                skipped_count += 1
                continue

            # Enrich with WordNet data
            three_pole = simplex['three_pole_simplex']
            for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
                pole = three_pole[pole_name]
                synset_id = pole['synset']

                # Only enrich if missing data
                if not pole.get('lemmas') or not pole.get('definition'):
                    enriched = enrich_pole_with_wordnet(synset_id)
                    pole['lemmas'] = enriched['lemmas']
                    pole['definition'] = enriched['definition']

            # Create layer entry
            entry = create_layer_entry_from_simplex(simplex)
            concepts.append(entry)
            existing_synsets.add(dimension_key)

            print(f"      [ADD]  {dimension} (μ−={three_pole['negative_pole']['synset']}, "
                  f"μ0={three_pole['neutral_homeostasis']['synset']}, "
                  f"μ+={three_pole['positive_pole']['synset']})")
            added_count += 1

    # Summary
    new_count = len(concepts)

    print(f"\n4. Summary:")
    print(f"   Original entries:  {original_count}")
    print(f"   Added S-tier:      {added_count}")
    print(f"   Skipped (existed): {skipped_count}")
    print(f"   New total:         {new_count}")
    print(f"   Expected:          {original_count + added_count}")

    if new_count != original_count + added_count:
        print(f"   ⚠️  WARNING: Count mismatch!")
        return False

    # Update metadata
    if metadata:
        metadata['total_concepts'] = new_count
        metadata['s_tier_count'] = sum(1 for c in concepts if c.get('s_tier', False))
        metadata['simplex_count'] = sum(1 for c in concepts if 'simplex_dimension' in c)

    # Rebuild full structure
    layer2_updated = {
        "metadata": metadata,
        "concepts": concepts
    }

    # Save backup
    backup_path = LAYER2_PATH.with_suffix('.json.backup_before_s_tier')
    print(f"\n5. Creating backup at {backup_path}")
    save_json(backup_path, layer2_full)

    # Save updated layer2.json
    print(f"\n6. Writing updated layer2.json")
    save_json(LAYER2_PATH, layer2_updated)

    # Also save enriched patch
    enriched_patch_path = PATCH_PATH.with_name('s_tier_simplexes_enriched.json')
    print(f"\n7. Saving enriched patch to {enriched_patch_path}")
    save_json(enriched_patch_path, patch)

    print(f"\n✅ Patches applied successfully!")
    print(f"\n📊 S-Tier Breakdown:")
    print(f"   Total S-tier simplexes added: {added_count}")

    # Count by priority
    for priority_section in ['critical_s_tier', 'high_priority_s_tier',
                              'medium_priority_s_tier', 'standard_s_tier']:
        count = len(patch.get(priority_section, []))
        if count > 0:
            priority_name = priority_section.replace('_s_tier', '')
            print(f"   {priority_name.capitalize()}: {count}")

    s_tier_count = sum(1 for e in concepts if e.get('s_tier', False))
    print(f"\n   Total S-tier concepts in layer2: {s_tier_count}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Review layer2.json to verify simplex entries look correct")
    print("  2. Modify sumo_data_generation.py to support three-pole training")
    print("  3. Create train_s_tier_simplexes.py training script")
    print("  4. Run S+ training for homeostatic steering lenses")
    print("  5. Validate with test_homeostatic_steering.py")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = apply_patch()
    exit(0 if success else 1)
