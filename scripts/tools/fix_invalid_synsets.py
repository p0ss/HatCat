#!/usr/bin/env python3
"""
Find alternative valid synsets for the invalid ones.
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn

# Manual replacements for invalid synsets
REPLACEMENTS = {
    # Layer 2
    'app.n.01': None,  # Remove - too informal/modern
    'meat-eater.n.01': 'carnivore.n.02',  # Any animal that feeds on flesh
    'computer_window.n.01': None,  # Remove - not in WordNet
    'confining.n.01': 'restraint.n.01',  # The act of controlling by restraining
    'directing.n.01': 'direction.n.03',  # The act of managing something
    'weathering.n.01': None,  # Remove - not in WordNet, but erosion.n.01 already present
    'reaping.n.01': None,  # Remove - not in WordNet, but harvesting.n.01 already present

    # Layer 3
    'adrenaline.n.02': None,  # Remove - only 1 sense exists
    'buttermilk.n.02': None,  # Remove - only 1 sense exists
    'goat_milk.n.01': None,  # Remove - not in WordNet, but milk.n.01 already present
    'jacuzzi.n.02': 'jacuzzi.n.01',  # A large whirlpool bathtub
    'travel_document.n.01': None,  # Remove - not in WordNet, but passport.n.01 and identification.n.02 already present
    'road_vehicle.n.01': None,  # Remove - not in WordNet, but car.n.01 and truck.n.01 already present
    'dusk.n.02': None,  # Remove - only 1 sense exists, but sunset.n.01 already present
    'waste_container.n.01': None,  # Remove - not in WordNet, but wastebasket.n.01 and trash_can.n.01 already present
    'whirlpool_bath.n.01': 'hot_tub.n.01',  # Already in list, remove duplicate
    'cultured_milk.n.01': None,  # Remove - not in WordNet, but yogurt.n.01 and dairy_product.n.01 already present
}


def fix_invalid_synsets(suggestions_file: str, output_file: str):
    """Fix invalid synsets by replacing or removing them."""

    # Load suggestions
    with open(suggestions_file, 'r') as f:
        suggestions = json.load(f)

    print(f"Loaded {len(suggestions)} concept suggestions")
    print("=" * 80)

    fixed_count = 0
    removed_count = 0

    # Fix each concept
    for concept in suggestions:
        sumo_term = concept['sumo_term']
        suggested_synsets = concept['suggested_synsets']

        new_synsets = []
        modified = False

        for synset_data in suggested_synsets:
            synset_name = synset_data['synset']

            # Check if this synset needs fixing
            if synset_name in REPLACEMENTS:
                replacement = REPLACEMENTS[synset_name]

                if replacement is None:
                    # Remove this synset
                    print(f"Removing {synset_name} from {sumo_term}")
                    removed_count += 1
                    modified = True
                    continue
                else:
                    # Replace with alternative
                    try:
                        synset = wn.synset(replacement)
                        print(f"Replacing {synset_name} with {replacement} for {sumo_term}")
                        synset_data['synset'] = replacement
                        synset_data['definition'] = synset.definition()
                        fixed_count += 1
                        modified = True
                    except Exception as e:
                        print(f"ERROR: Replacement {replacement} is also invalid: {e}")
                        continue

            new_synsets.append(synset_data)

        if modified:
            concept['suggested_synsets'] = new_synsets

    # Save fixed suggestions
    with open(output_file, 'w') as f:
        json.dump(suggestions, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Synsets replaced: {fixed_count}")
    print(f"Synsets removed: {removed_count}")
    print(f"Total changes: {fixed_count + removed_count}")
    print(f"\nFixed suggestions saved to: {output_file}")


if __name__ == '__main__':
    suggestions_file = Path(__file__).parent.parent / 'results' / 'wordnet_patch_suggestions.json'
    output_file = suggestions_file  # Overwrite original

    print(f"Fixing invalid synsets in: {suggestions_file}")
    print()

    fix_invalid_synsets(suggestions_file, output_file)
