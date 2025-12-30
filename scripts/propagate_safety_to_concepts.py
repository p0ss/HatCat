#!/usr/bin/env python3
"""
Propagate Safety Metadata to Concept Pack

Takes the consolidated safety registry and updates the concept pack's
individual concept JSON files with:
- safety_tags (risk_level, impacts, treaty_relevant, harness_relevant)
- training_hints (if available from melds and missing in concept)
- simplex_mapping (if available)

This makes the concept pack the authoritative source, which then
flows to lens packs and the server.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_registry(registry_path: Path) -> dict:
    """Load the safety registry."""
    with open(registry_path) as f:
        return json.load(f)


def find_concept_file(concepts_dir: Path, term: str) -> Path | None:
    """Find the concept file for a given term."""
    # Normalize term to lowercase filename
    filename = term.lower() + ".json"

    for layer_dir in concepts_dir.iterdir():
        if not layer_dir.is_dir():
            continue
        concept_file = layer_dir / filename
        if concept_file.exists():
            return concept_file

    return None


def load_meld_training_data(melds_dir: Path) -> dict:
    """Load training examples from applied melds."""
    training_data = {}

    for meld_file in melds_dir.glob("*.json"):
        if 'remediation' in meld_file.name:
            continue

        try:
            with open(meld_file) as f:
                meld = json.load(f)

            for candidate in meld.get('candidates', []):
                term = candidate.get('term', '')
                hints = candidate.get('training_hints', {})

                if hints.get('positive_examples') and hints.get('negative_examples'):
                    training_data[term] = {
                        'positive_examples': hints['positive_examples'],
                        'negative_examples': hints['negative_examples'],
                        'disambiguation': hints.get('disambiguation', ''),
                    }

                # Also capture simplex_mapping
                simplex = candidate.get('simplex_mapping', {})
                if simplex.get('status') == 'mapped':
                    if term not in training_data:
                        training_data[term] = {}
                    training_data[term]['simplex_mapping'] = simplex

        except Exception as e:
            print(f"Warning: Failed to parse {meld_file.name}: {e}")

    return training_data


def update_concept_file(
    concept_file: Path,
    registry_entry: dict,
    training_data: dict,
    dry_run: bool = False,
) -> tuple[bool, list[str]]:
    """
    Update a concept file with safety metadata.

    Returns (was_updated, list of changes made)
    """
    with open(concept_file) as f:
        concept = json.load(f)

    term = concept.get('term', '')
    changes = []

    # Update safety_tags
    if 'safety_tags' not in concept:
        concept['safety_tags'] = {}

    safety_tags = concept['safety_tags']

    # Risk level
    new_risk = registry_entry.get('risk_level', 'low')
    old_risk = safety_tags.get('risk_level', 'low')
    if new_risk != old_risk and new_risk in ['high', 'critical', 'medium']:
        safety_tags['risk_level'] = new_risk
        changes.append(f"risk_level: {old_risk} → {new_risk}")

    # Impacts
    new_impacts = registry_entry.get('impacts', [])
    if new_impacts and not safety_tags.get('impacts'):
        safety_tags['impacts'] = new_impacts
        changes.append(f"impacts: {new_impacts}")

    # Treaty/harness relevance
    if registry_entry.get('treaty_relevant') and not safety_tags.get('treaty_relevant'):
        safety_tags['treaty_relevant'] = True
        changes.append("treaty_relevant: True")

    if registry_entry.get('harness_relevant') and not safety_tags.get('harness_relevant'):
        safety_tags['harness_relevant'] = True
        changes.append("harness_relevant: True")

    # Training hints from melds
    meld_data = training_data.get(term, {})

    if 'training_hints' not in concept:
        concept['training_hints'] = {}

    hints = concept['training_hints']

    # Only add training examples if missing
    if meld_data.get('positive_examples') and not hints.get('positive_examples'):
        hints['positive_examples'] = meld_data['positive_examples']
        changes.append(f"positive_examples: {len(meld_data['positive_examples'])} added")

    if meld_data.get('negative_examples') and not hints.get('negative_examples'):
        hints['negative_examples'] = meld_data['negative_examples']
        changes.append(f"negative_examples: {len(meld_data['negative_examples'])} added")

    if meld_data.get('disambiguation') and not hints.get('disambiguation'):
        hints['disambiguation'] = meld_data['disambiguation']
        changes.append("disambiguation added")

    # Simplex mapping
    if meld_data.get('simplex_mapping') and not concept.get('simplex_mapping'):
        concept['simplex_mapping'] = meld_data['simplex_mapping']
        changes.append(f"simplex_mapping: {meld_data['simplex_mapping'].get('monitor')}")

    # Write back if changes were made
    if changes and not dry_run:
        with open(concept_file, 'w') as f:
            json.dump(concept, f, indent=2)

    return bool(changes), changes


def main():
    parser = argparse.ArgumentParser(
        description="Propagate safety metadata from registry to concept pack"
    )
    parser.add_argument(
        "--registry", "-r",
        default="data/safety_registry.json",
        help="Path to safety registry"
    )
    parser.add_argument(
        "--concepts", "-c",
        default="concept_packs/first-light/concepts",
        help="Path to concepts directory"
    )
    parser.add_argument(
        "--melds", "-m",
        default="melds/applied",
        help="Path to applied melds (for training data)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details for each concept"
    )

    args = parser.parse_args()

    registry_path = Path(args.registry)
    concepts_dir = Path(args.concepts)
    melds_dir = Path(args.melds)

    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        print("Run: python scripts/build_safety_registry.py first")
        return

    if not concepts_dir.exists():
        print(f"Concepts directory not found: {concepts_dir}")
        return

    # Load data
    print("Loading registry...")
    registry = load_registry(registry_path)
    print(f"  {len(registry)} concepts in registry")

    print("Loading training data from melds...")
    training_data = load_meld_training_data(melds_dir)
    print(f"  {len(training_data)} concepts with training data")

    # Process concepts
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Updating concept files...")

    stats = defaultdict(int)

    for term, entry in registry.items():
        concept_file = find_concept_file(concepts_dir, term)

        if not concept_file:
            stats['not_found'] += 1
            continue

        updated, changes = update_concept_file(
            concept_file,
            entry,
            training_data,
            dry_run=args.dry_run
        )

        if updated:
            stats['updated'] += 1
            if args.verbose:
                print(f"  {term}: {', '.join(changes)}")
        else:
            stats['unchanged'] += 1

    # Summary
    print(f"\n=== Summary ===")
    print(f"Updated: {stats['updated']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Not found in pack: {stats['not_found']}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Remove --dry-run to apply changes.")


if __name__ == "__main__":
    main()
