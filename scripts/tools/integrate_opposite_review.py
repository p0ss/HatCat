#!/usr/bin/env python3
"""
Integrate opposite review results into hierarchy.json steering_targets.

Takes the output from run_agentic_opposite_review.py and updates the
concept pack's hierarchy.json with steering_targets mappings.

Usage:
    python scripts/tools/integrate_opposite_review.py \
        --review results/opposite_review.json \
        --pack concept_packs/first-light \
        --min-confidence 6
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime


def load_review_results(review_file: Path) -> dict:
    """Load the agentic review results."""
    with open(review_file) as f:
        data = json.load(f)
    return data


def load_hierarchy(pack_dir: Path) -> dict:
    """Load the hierarchy.json from a concept pack."""
    hierarchy_file = pack_dir / "hierarchy.json"
    with open(hierarchy_file) as f:
        return json.load(f)


def save_hierarchy(pack_dir: Path, hierarchy: dict):
    """Save updated hierarchy.json."""
    hierarchy_file = pack_dir / "hierarchy.json"
    with open(hierarchy_file, 'w') as f:
        json.dump(hierarchy, f, indent=2)


def get_existing_concepts(pack_dir: Path) -> set:
    """Get set of all concept names in the pack."""
    hierarchy_dir = pack_dir / "hierarchy"
    concepts = set()

    for layer_file in hierarchy_dir.glob("layer*.json"):
        with open(layer_file) as f:
            data = json.load(f)
        for concept in data.get("concepts", []):
            concepts.add(concept["sumo_term"])

    return concepts


def integrate_results(
    review_data: dict,
    hierarchy: dict,
    existing_concepts: set,
    min_confidence: int = 6,
    require_exists: bool = True
) -> tuple[dict, dict]:
    """
    Integrate review results into hierarchy steering_targets.

    Args:
        review_data: The agentic review output
        hierarchy: Current hierarchy.json data
        existing_concepts: Set of concepts in the pack
        min_confidence: Minimum confidence score to accept
        require_exists: Only add if opposite exists in pack

    Returns:
        (updated_hierarchy, stats)
    """
    results = review_data.get("results", [])

    # Initialize steering_targets if not present
    if "steering_targets" not in hierarchy:
        hierarchy["steering_targets"] = {}

    existing_targets = set(hierarchy["steering_targets"].keys())

    stats = {
        "total_reviewed": len(results),
        "errors": 0,
        "low_confidence": 0,
        "no_opposite": 0,
        "opposite_missing": 0,
        "already_exists": 0,
        "added": 0,
        "added_concepts": [],
        "missing_opposites": [],  # Opposites that should be added to pack
        "flag_distribution": Counter(),
    }

    for result in results:
        sumo_term = result.get("sumo_term", "")
        layer = result.get("layer", 0)
        term_with_layer = f"{sumo_term}:{layer}"

        # Skip errors
        if "error" in result:
            stats["errors"] += 1
            continue

        recommendation = result.get("recommendation", {})
        selected = recommendation.get("selected")
        confidence = recommendation.get("confidence", 0)
        reasoning = recommendation.get("reasoning", "")
        flags = recommendation.get("flags", [])

        # Track flags
        for flag in flags:
            stats["flag_distribution"][flag] += 1

        # Skip if no opposite selected
        if not selected:
            stats["no_opposite"] += 1
            continue

        # Skip low confidence
        if confidence < min_confidence:
            stats["low_confidence"] += 1
            continue

        # Skip if already has a target
        if term_with_layer in existing_targets:
            stats["already_exists"] += 1
            continue

        # Check if opposite exists in pack
        if selected not in existing_concepts:
            stats["opposite_missing"] += 1
            # Track for potential addition
            if "should_add_to_layers" in str(result.get("candidates", [])):
                stats["missing_opposites"].append({
                    "concept": sumo_term,
                    "opposite": selected,
                    "reasoning": reasoning
                })

            if require_exists:
                continue

        # Add steering target
        hierarchy["steering_targets"][term_with_layer] = {
            "target": selected,
            "rationale": reasoning[:200] if reasoning else f"Semantic opposite of {sumo_term}",
            "confidence": confidence,
            "source": "agentic_review"
        }

        stats["added"] += 1
        stats["added_concepts"].append(term_with_layer)

    return hierarchy, stats


def print_stats(stats: dict):
    """Print integration statistics."""
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal reviewed: {stats['total_reviewed']}")
    print(f"Errors: {stats['errors']}")
    print(f"No opposite identified: {stats['no_opposite']}")
    print(f"Low confidence (skipped): {stats['low_confidence']}")
    print(f"Opposite not in pack (skipped): {stats['opposite_missing']}")
    print(f"Already had target: {stats['already_exists']}")
    print(f"\nNewly added: {stats['added']}")

    if stats["flag_distribution"]:
        print("\nFlag distribution:")
        for flag, count in stats["flag_distribution"].most_common():
            print(f"  {flag}: {count}")

    if stats["missing_opposites"]:
        print(f"\nOpposites that should be added to pack ({len(stats['missing_opposites'])}):")
        for item in stats["missing_opposites"][:20]:
            print(f"  {item['concept']} -> {item['opposite']}")
        if len(stats["missing_opposites"]) > 20:
            print(f"  ... and {len(stats['missing_opposites']) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate opposite review results into hierarchy.json"
    )
    parser.add_argument(
        "--review", type=Path, required=True,
        help="Path to opposite_review.json"
    )
    parser.add_argument(
        "--pack", type=Path, required=True,
        help="Path to concept pack directory"
    )
    parser.add_argument(
        "--min-confidence", type=int, default=6,
        help="Minimum confidence score to accept (default: 6)"
    )
    parser.add_argument(
        "--include-missing", action="store_true",
        help="Include opposites even if they don't exist in pack"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--save-missing", type=Path,
        help="Save list of missing opposites to file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("INTEGRATE OPPOSITE REVIEW")
    print("=" * 60)

    # Load data
    print(f"\nLoading review results from: {args.review}")
    review_data = load_review_results(args.review)
    print(f"  Loaded {len(review_data.get('results', []))} results")

    print(f"\nLoading hierarchy from: {args.pack}")
    hierarchy = load_hierarchy(args.pack)
    existing_targets = len(hierarchy.get("steering_targets", {}))
    print(f"  Existing steering targets: {existing_targets}")

    print("\nGetting existing concepts...")
    existing_concepts = get_existing_concepts(args.pack)
    print(f"  Found {len(existing_concepts)} concepts in pack")

    # Integrate
    print(f"\nIntegrating with min_confidence={args.min_confidence}...")
    updated_hierarchy, stats = integrate_results(
        review_data,
        hierarchy,
        existing_concepts,
        min_confidence=args.min_confidence,
        require_exists=not args.include_missing
    )

    # Print stats
    print_stats(stats)

    # Save missing opposites if requested
    if args.save_missing and stats["missing_opposites"]:
        with open(args.save_missing, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "count": len(stats["missing_opposites"]),
                "missing_opposites": stats["missing_opposites"]
            }, f, indent=2)
        print(f"\nSaved missing opposites to: {args.save_missing}")

    # Save or show dry run
    if args.dry_run:
        print("\n[DRY RUN] Would have updated hierarchy.json")
        print(f"  New steering targets: {existing_targets} -> {len(updated_hierarchy['steering_targets'])}")
    else:
        save_hierarchy(args.pack, updated_hierarchy)
        print(f"\n✓ Updated hierarchy.json")
        print(f"  Steering targets: {existing_targets} -> {len(updated_hierarchy['steering_targets'])}")


if __name__ == "__main__":
    main()
