#!/usr/bin/env python3
"""
Migrate concept pack layer files to per-concept JSON files.

Creates concepts/layerN/concept_name.json for each concept,
using the MELD candidate schema as the base structure.

This makes concept packs self-contained and distributable.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


def slugify(term: str) -> str:
    """Convert concept term to safe filename."""
    # Remove layer suffix if present (e.g., "Deception:2" -> "Deception")
    term = term.split(":")[0]
    # Replace problematic characters
    term = re.sub(r'[^\w\-]', '_', term)
    return term.lower()


def load_layer_files(concept_pack_path: Path) -> Dict[int, List[Dict]]:
    """Load all layer files from hierarchy directory."""
    hierarchy_dir = concept_pack_path / "hierarchy"
    layers = {}

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        layer_num = int(layer_file.stem.replace("layer", ""))
        with open(layer_file) as f:
            data = json.load(f)
        layers[layer_num] = data.get("concepts", [])

    return layers


def load_steering_targets(concept_pack_path: Path) -> Dict[str, Dict]:
    """Load existing steering targets from hierarchy.json."""
    hierarchy_file = concept_pack_path / "hierarchy.json"
    if not hierarchy_file.exists():
        return {}

    with open(hierarchy_file) as f:
        data = json.load(f)

    return data.get("steering_targets", {})


def detect_risk_level(concept: Dict) -> str:
    """Detect risk level based on concept content."""
    term = concept.get("sumo_term", "").lower()
    definition = (concept.get("sumo_definition", "") or concept.get("definition", "")).lower()

    # High risk keywords
    high_risk = ['weapon', 'attack', 'exploit', 'malicious', 'fabricat']
    medium_risk = ['deception', 'manipulation', 'coercion', 'fraud', 'abuse', 'threat']

    combined = term + " " + definition

    for kw in high_risk:
        if kw in combined:
            return "high"

    for kw in medium_risk:
        if kw in combined:
            return "medium"

    return "low"


def concept_to_meld_schema(
    concept: Dict,
    layer: int,
    steering_targets: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Convert a layer file concept entry to MELD candidate schema.

    See docs/specification/MAP/MAP_MELDING.md §2.2 for schema spec.
    """
    term = concept.get("sumo_term") or concept.get("name", "Unknown")
    term_with_layer = f"{term}:{layer}"

    # Extract existing relationships if present (MELD-enhanced concepts have these)
    existing_rels = concept.get("relationships", {})

    # Build relationships, preserving MELD data if available
    relationships = {
        "related": existing_rels.get("related", []),
        "antonyms": existing_rels.get("antonyms", []),
        "has_part": existing_rels.get("has_part", concept.get("category_children", [])),
        "part_of": existing_rels.get("part_of", []),
    }

    # Check for steering target (becomes "opposite" relationship)
    if term_with_layer in steering_targets:
        target_info = steering_targets[term_with_layer]
        relationships["opposite"] = [target_info["target"]]
        relationships["opposite_rationale"] = target_info.get("rationale", "")

    # Build safety tags - preserve MELD data if available, otherwise detect
    existing_safety = concept.get("safety_tags", {})
    if existing_safety:
        safety_tags = existing_safety
    else:
        risk_level = detect_risk_level(concept)
        safety_tags = {
            "risk_level": risk_level,
            "impacts": [],
            "treaty_relevant": False,
            "harness_relevant": risk_level in ["medium", "high"],
        }

    # Build training hints - preserve MELD data if available
    existing_hints = concept.get("training_hints", {})
    lemmas = concept.get("lemmas", [])

    if existing_hints and (existing_hints.get("positive_examples") or existing_hints.get("negative_examples")):
        # Use existing MELD training hints
        training_hints = existing_hints
        if lemmas and "seed_terms" not in training_hints:
            training_hints["seed_terms"] = lemmas
    else:
        # Generate minimal training hints from lemmas
        training_hints = {
            "positive_examples": [],
            "negative_examples": [],
            "disambiguation": "",
        }
        if lemmas:
            training_hints["seed_terms"] = lemmas

    # Get aliases - prefer explicit MELD aliases, fall back to lemmas
    aliases = concept.get("aliases", lemmas[:5] if lemmas else [])

    # Build the MELD-style concept entry
    result = {
        # Identity
        "term": term,
        "role": "concept",

        # Hierarchy
        "parent_concepts": concept.get("parent_concepts", []),
        "layer": layer,
        "domain": concept.get("domain", ""),

        # Definition
        "definition": concept.get("definition") or concept.get("sumo_definition", ""),
        "definition_source": "SUMO" if concept.get("sumo_definition") else "",

        # Aliases
        "aliases": aliases,

        # WordNet mapping
        "wordnet": {
            "synsets": concept.get("synsets", []),
            "canonical_synset": concept.get("canonical_synset", ""),
            "lemmas": lemmas,
            "pos": concept.get("pos", ""),
        },

        # Relationships
        "relationships": relationships,

        # Safety
        "safety_tags": safety_tags,

        # Training
        "training_hints": training_hints,

        # Children (for reference)
        "children": concept.get("child_concepts") or concept.get("category_children", []),

        # Category lens metadata
        "is_category_lens": concept.get("is_category_lens", False),
        "child_count": concept.get("child_count", 0),

        # MELD source for traceability
        "meld_source": concept.get("meld_source"),
    }

    # Clean up empty/null values
    result = {k: v for k, v in result.items() if v not in [None, "", [], {}]}

    return result


def migrate_concept_pack(concept_pack_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate a concept pack to per-concept JSON files.

    Args:
        concept_pack_path: Path to concept pack
        dry_run: If True, don't write files, just report what would be done

    Returns:
        Migration summary
    """
    concept_pack_path = Path(concept_pack_path)

    # Load existing data
    layers = load_layer_files(concept_pack_path)
    steering_targets = load_steering_targets(concept_pack_path)

    # Create concepts directory structure
    concepts_dir = concept_pack_path / "concepts"

    summary = {
        "total_concepts": 0,
        "by_layer": {},
        "with_steering_targets": 0,
        "files_created": [],
        "errors": [],
    }

    for layer_num, concepts in sorted(layers.items()):
        layer_dir = concepts_dir / f"layer{layer_num}"

        if not dry_run:
            layer_dir.mkdir(parents=True, exist_ok=True)

        summary["by_layer"][layer_num] = 0

        for concept in concepts:
            term = concept.get("sumo_term") or concept.get("name")
            if not term:
                summary["errors"].append(f"Layer {layer_num}: concept missing term")
                continue

            # Convert to MELD schema
            meld_concept = concept_to_meld_schema(concept, layer_num, steering_targets)

            # Generate filename
            filename = slugify(term) + ".json"
            filepath = layer_dir / filename

            # Write file
            if not dry_run:
                with open(filepath, "w") as f:
                    json.dump(meld_concept, f, indent=2)

            summary["total_concepts"] += 1
            summary["by_layer"][layer_num] += 1
            summary["files_created"].append(str(filepath.relative_to(concept_pack_path)))

            # Track steering targets
            term_with_layer = f"{term}:{layer_num}"
            if term_with_layer in steering_targets:
                summary["with_steering_targets"] += 1

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate concept pack to per-concept JSON files"
    )
    parser.add_argument(
        "concept_pack",
        type=Path,
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, just report what would be done"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all files created"
    )

    args = parser.parse_args()

    print(f"Migrating concept pack: {args.concept_pack}")
    print(f"Dry run: {args.dry_run}")
    print()

    summary = migrate_concept_pack(args.concept_pack, dry_run=args.dry_run)

    print(f"Total concepts: {summary['total_concepts']}")
    print(f"With steering targets: {summary['with_steering_targets']}")
    print(f"By layer:")
    for layer, count in sorted(summary["by_layer"].items()):
        print(f"  Layer {layer}: {count}")

    if summary["errors"]:
        print(f"\nErrors: {len(summary['errors'])}")
        for err in summary["errors"][:10]:
            print(f"  {err}")

    if args.verbose:
        print(f"\nFiles created: {len(summary['files_created'])}")
        for f in summary["files_created"][:20]:
            print(f"  {f}")
        if len(summary["files_created"]) > 20:
            print(f"  ... and {len(summary['files_created']) - 20} more")

    if not args.dry_run:
        print(f"\nMigration complete. Files written to {args.concept_pack / 'concepts'}/")


if __name__ == "__main__":
    main()
