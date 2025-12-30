#!/usr/bin/env python3
"""
Fix duplicate concepts in hierarchy tree.

Three strategies:
1. DISAMBIGUATE: Rename one occurrence to include parent (e.g., Balloon -> BalloonContainer)
2. REMOVE_SHALLOW: Remove the shallower occurrence, keep deeper
3. REMOVE_DEEP: Remove the deeper occurrence, keep shallower

Usage:
    python scripts/ontology/fix_tree_duplicates.py --dry-run
    python scripts/ontology/fix_tree_duplicates.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# === DUPLICATE RESOLUTION PLAN ===
# Format: concept -> (action, details)
# Actions: 'disambiguate', 'remove_shallow', 'remove_deep', 'keep_both'

RESOLUTION_PLAN = {
    # Category 1: Legitimate multi-homing - disambiguate by appending parent
    'Balloon': ('disambiguate', {'rename_path_contains': 'FluidContainer', 'new_name': 'BalloonContainer'}),
    'Cellulose': ('disambiguate', {'rename_path_contains': 'Carbohydrate', 'new_name': 'CelluloseCarbohydrate'}),
    'Diamond': ('disambiguate', {'rename_path_contains': 'Mineral', 'new_name': 'DiamondMineral'}),
    'Dish': ('disambiguate', {'rename_path_contains': 'PreparedFood', 'new_name': 'DishFood'}),
    'Electronics': ('disambiguate', {'rename_path_contains': 'FieldOfStudy', 'new_name': 'ElectronicsEngineering'}),
    'Mathematics': ('disambiguate', {'rename_path_contains': 'FieldOfStudy', 'new_name': 'MathematicsField'}),
    'Physics': ('disambiguate', {'rename_path_contains': 'FieldOfStudy', 'new_name': 'PhysicsField'}),
    'Statistics': ('disambiguate', {'rename_path_contains': 'FieldOfStudy', 'new_name': 'StatisticsField'}),
    'PowerElectronics': ('disambiguate', {'rename_path_contains': 'FieldOfStudy', 'new_name': 'PowerElectronicsField'}),
    'Clapping': ('disambiguate', {'rename_path_contains': 'RadiatingSound', 'new_name': 'ClappingSound'}),
    'ConsumerPriceIndex': ('disambiguate', {'rename_path_contains': 'PerformanceMeasure', 'new_name': 'CPIMeasure'}),
    'InterestRate': ('disambiguate', {'rename_path_contains': 'CurrencyMeasure', 'new_name': 'InterestRateMeasure'}),
    'HydraulicFluid': ('disambiguate', {'rename_path_contains': 'Mixture', 'new_name': 'HydraulicFluidMixture'}),
    'MotorOil': ('disambiguate', {'rename_path_contains': 'PetroleumProduct', 'new_name': 'MotorOilPetroleum'}),
    'PhosphoricAcid': ('disambiguate', {'rename_path_contains': 'CompoundSubstance', 'new_name': 'PhosphoricAcidCompound'}),
    'Seed': ('disambiguate', {'rename_path_contains': 'EdibleSeed', 'new_name': 'EdibleSeed'}),
    'HurricaneSeason': ('disambiguate', {'rename_path_contains': 'SeasonOfYear', 'new_name': 'HurricaneSeasonTime'}),
    'Work': ('disambiguate', {'rename_path_contains': 'FieldMeasure', 'new_name': 'WorkMeasure'}),

    # Category 2: Sibling generation duplicates - remove shallow (wrong), keep deep (correct)
    'MesaOptimization': ('remove_shallow', {}),
    'RewardHacking': ('remove_shallow', {}),
    'Submarine': ('remove_shallow', {}),
    'HumanoidRobot': ('remove_shallow', {}),

    # InnerAlignment/OuterAlignment - keep under AIAlignmentProcess (d3), remove from AIAlignmentTheory (d5)
    'InnerAlignment': ('remove_deep', {}),
    'OuterAlignment': ('remove_deep', {}),

    # Category 3: Structural fixes
    # Perception/Reasoning - keep in MindsAndAgents (cognitive), remove from LivingThings (biological)
    'Perception': ('remove_deep', {}),  # Keep d2 MindsAndAgents, remove d3 LivingThings
    'Reasoning': ('remove_deep', {}),   # Keep d2 MindsAndAgents, remove d4 LivingThings

    # Infrastructure - keep as StationaryArtifact (d3), remove from Region (d2)
    'Infrastructure': ('remove_shallow', {}),

    # PhysicalActions - keep in AgentAction (d2), remove from OrganismProcess (d4)
    'PhysicalActions': ('remove_deep', {}),

    # PhysicalAttrs - appears twice at same depth under different parents - keep both for now
    'PhysicalAttrs': ('keep_both', {}),

    # Military/info operations - complex, need careful review
    'CyberOperation': ('remove_deep', {}),  # Keep d2 Security, remove d4 InformationOperation leaf
    'DeceptionOperation': ('remove_deep', {}),  # Keep d4 InformationOperation, remove d6
    'PsychologicalOperation': ('remove_deep', {}),  # Keep d3 MilitaryProcess (has children)
    'OffensiveInformationOperation': ('remove_shallow', {}),  # Keep d5 with children
    'DisinformationCampaign': ('remove_deep', {}),  # Keep d4 InfluenceProcess

    # DeceptiveSpeech - keep under SpeechAct (has children), remove under TellingALie
    'DeceptiveSpeech': ('remove_deep', {}),

    # WaterSports - obvious duplicate, keep shallower
    'WaterSports': ('remove_deep', {}),

    # _layer - internal marker, remove duplicates
    '_layer': ('remove_deep', {}),
}


def strip_markers(key):
    """Remove all marker prefixes from a key."""
    markers = ['NEW:', 'MOVED:', 'ORPHAN:', 'RENAMED:', 'ELEVATED:', 'ABSORBED:']
    result = key
    for m in markers:
        result = result.replace(m, '')
    return result


def find_all_occurrences(node, path=[], depth=0):
    """Find all concept occurrences with their paths and depths."""
    occurrences = defaultdict(list)

    if not isinstance(node, dict):
        return occurrences

    for k, v in node.items():
        clean_k = strip_markers(k)
        current_path = path + [{'key': k, 'clean': clean_k}]

        occurrences[clean_k].append({
            'original_key': k,
            'path': current_path,
            'path_str': ' -> '.join([p['clean'] for p in current_path]),
            'depth': depth,
            'is_leaf': not isinstance(v, dict),
            'has_children': isinstance(v, dict) and len(v) > 0
        })

        if isinstance(v, dict):
            child_occurrences = find_all_occurrences(v, current_path, depth + 1)
            for concept, occ_list in child_occurrences.items():
                occurrences[concept].extend(occ_list)

    return occurrences


def navigate_to_parent(tree, path):
    """Navigate to the parent node of the given path."""
    if len(path) <= 1:
        return tree, path[0]['key'] if path else None

    current = tree
    for step in path[:-1]:
        if step['key'] not in current:
            return None, None
        current = current[step['key']]

    return current, path[-1]['key']


def remove_node(tree, path):
    """Remove a node from the tree given its path."""
    parent, key = navigate_to_parent(tree, path)
    if parent is None or key is None:
        return False
    if key in parent:
        del parent[key]
        return True
    return False


def rename_node(tree, path, new_name):
    """Rename a node in the tree."""
    parent, old_key = navigate_to_parent(tree, path)
    if parent is None or old_key is None:
        return False
    if old_key in parent:
        value = parent[old_key]
        del parent[old_key]
        parent[new_name] = value
        return True
    return False


def apply_resolution(tree, concept, occurrences, action, details, dry_run=False):
    """Apply a resolution action to a duplicate concept."""
    sorted_occs = sorted(occurrences, key=lambda x: x['depth'])

    if action == 'keep_both':
        return None, "Keeping both occurrences"

    elif action == 'remove_shallow':
        # Remove shallowest, keep deepest
        to_remove = sorted_occs[0]
        to_keep = sorted_occs[-1]
        if not dry_run:
            remove_node(tree, to_remove['path'])
        return to_remove, f"Removed d{to_remove['depth']}, kept d{to_keep['depth']}"

    elif action == 'remove_deep':
        # Remove deepest, keep shallowest
        to_remove = sorted_occs[-1]
        to_keep = sorted_occs[0]
        if not dry_run:
            remove_node(tree, to_remove['path'])
        return to_remove, f"Removed d{to_remove['depth']}, kept d{to_keep['depth']}"

    elif action == 'disambiguate':
        # Find occurrence matching path_contains and rename it
        path_contains = details.get('rename_path_contains', '')
        new_name = details.get('new_name', concept + 'Alt')

        for occ in occurrences:
            if path_contains in occ['path_str']:
                if not dry_run:
                    rename_node(tree, occ['path'], new_name)
                return occ, f"Renamed to {new_name} (was at d{occ['depth']})"

        return None, f"WARNING: Could not find occurrence with '{path_contains}' in path"

    return None, f"Unknown action: {action}"


def main():
    parser = argparse.ArgumentParser(description='Fix duplicate concepts in hierarchy tree')
    parser.add_argument('--input', default='concept_packs/first-light/hierarchy/hierarchy_tree_v12.json',
                        help='Input tree file')
    parser.add_argument('--output', default='concept_packs/first-light/hierarchy/hierarchy_tree_v13.json',
                        help='Output tree file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    args = parser.parse_args()

    # Load tree
    with open(args.input) as f:
        tree = json.load(f)
    print(f"Loaded tree from {args.input}")

    # Find all occurrences
    occurrences = find_all_occurrences(tree)
    duplicates = {k: v for k, v in occurrences.items() if len(v) > 1}

    print(f"Found {len(duplicates)} concepts with duplicates")

    # Check for unplanned duplicates
    unplanned = set(duplicates.keys()) - set(RESOLUTION_PLAN.keys())
    if unplanned:
        print(f"\nWARNING: {len(unplanned)} duplicates not in resolution plan:")
        for concept in sorted(unplanned):
            print(f"  - {concept}")

    # Apply resolutions
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Applying resolutions...")

    results = {
        'disambiguated': [],
        'removed': [],
        'kept_both': [],
        'errors': []
    }

    for concept, occ_list in sorted(duplicates.items()):
        if concept not in RESOLUTION_PLAN:
            results['errors'].append(f"{concept}: No resolution plan")
            continue

        action, details = RESOLUTION_PLAN[concept]
        affected, msg = apply_resolution(tree, concept, occ_list, action, details, dry_run=args.dry_run)

        if action == 'disambiguate' and affected:
            results['disambiguated'].append(f"{concept}: {msg}")
        elif action in ('remove_shallow', 'remove_deep') and affected:
            results['removed'].append(f"{concept}: {msg}")
        elif action == 'keep_both':
            results['kept_both'].append(f"{concept}: {msg}")
        elif 'WARNING' in msg:
            results['errors'].append(f"{concept}: {msg}")

        print(f"  {concept}: {msg}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Disambiguated: {len(results['disambiguated'])}")
    print(f"  Removed: {len(results['removed'])}")
    print(f"  Kept both: {len(results['kept_both'])}")
    print(f"  Errors: {len(results['errors'])}")

    if results['errors']:
        print(f"\nErrors:")
        for e in results['errors']:
            print(f"  - {e}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write to {args.output}")
    else:
        with open(args.output, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"\n✓ Wrote fixed tree to {args.output}")


if __name__ == '__main__':
    main()
