#!/usr/bin/env python3
"""
Test WordNet patch loading and querying.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map.data.wordnet_patch_loader import WordNetPatchLoader


def main():
    print("=" * 80)
    print("TESTING WORDNET PATCH SYSTEM")
    print("=" * 80)
    print()

    # Load patches
    loader = WordNetPatchLoader()
    patch_dir = Path("data/concept_graph/wordnet_patches")

    print(f"Loading patches from {patch_dir}...")
    count = loader.load_all_patches(patch_dir)
    print(f"✓ Loaded {count} patches")
    print()

    # Display summary
    summary = loader.export_summary()
    print("Summary:")
    print("-" * 80)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Test queries
    print("=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)
    print()

    # Test role variants
    test_synset = "satisfaction.n.01"
    print(f"Query: Role variants of '{test_synset}'")
    print("-" * 80)
    role_variants = loader.get_role_variants(test_synset)

    if role_variants:
        print(f"Found {len(role_variants)} role variants:")
        for related, meta in role_variants:
            role1 = meta.get('role1', '?')
            role2 = meta.get('role2', '?')
            sumo1 = meta.get('sumo_term1', '')
            sumo2 = meta.get('sumo_term2', '')
            print(f"  {related}")
            print(f"    Roles: {role1} → {role2}")
            print(f"    SUMO: {sumo1} → {sumo2}")
    else:
        print("  (none found)")
    print()

    # Test antonyms
    print(f"Query: Antonyms of '{test_synset}'")
    print("-" * 80)
    antonyms = loader.get_antonyms(test_synset)

    if antonyms:
        print(f"Found {len(antonyms)} antonyms:")
        for related, meta in antonyms:
            axis = meta.get('axis', '?')
            pole1 = meta.get('pole1', '?')
            pole2 = meta.get('pole2', '?')
            print(f"  {related}")
            print(f"    Axis: {axis} ({pole1} ↔ {pole2})")
    else:
        print("  (none found)")
    print()

    # Test all relationships
    print(f"Query: All relationships for '{test_synset}'")
    print("-" * 80)
    all_rels = loader.get_all_relationships(test_synset)

    for rel_type, rels in all_rels.items():
        print(f"  {rel_type}: {len(rels)} relationships")
    print()

    # Test multiple synsets
    print("=" * 80)
    print("TESTING MULTIPLE SYNSETS")
    print("=" * 80)
    print()

    test_synsets = [
        "satisfaction.n.01",
        "dissatisfaction.n.01",
        "excitation.n.01",
        "calmness.n.01",
        "altruism.n.01",
        "hostility.n.01",
        "dominance.n.03",
        "submissiveness.n.02",
        "receptiveness.n.01",
        "narrowmindedness.n.01"
    ]

    for synset in test_synsets:
        all_rels = loader.get_all_relationships(synset)
        total = sum(len(rels) for rels in all_rels.values())
        print(f"{synset:30s} - {total} relationships")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
