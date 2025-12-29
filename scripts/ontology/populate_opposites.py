#!/usr/bin/env python3
"""
Populate concept pack with opposite/antonym relationships from LLM review.

Reads the opposite_review.json and updates concept files with the selected opposites.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Populate concept pack with opposites")
    parser.add_argument("--review-file", required=True, help="Path to opposite_review.json")
    parser.add_argument("--concept-pack", required=True, help="Path to concept pack")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    review_path = Path(args.review_file)
    concept_pack_path = Path(args.concept_pack)
    concepts_dir = concept_pack_path / "concepts"

    # Load review results
    print(f"Loading review from {review_path}...")
    with open(review_path) as f:
        review_data = json.load(f)

    results = review_data["results"]
    print(f"  Total concepts reviewed: {len(results)}")

    # Process each result
    updated = 0
    skipped_no_opposite = 0
    skipped_not_found = 0
    errors = []

    for r in results:
        sumo_term = r["sumo_term"]
        layer = r["layer"]
        rec = r.get("recommendation", {})
        selected = rec.get("selected", "")

        # Skip if no opposite selected
        if not selected or selected.lower() in ["none", "null", ""]:
            skipped_no_opposite += 1
            continue

        # Find concept file
        concept_file = concepts_dir / f"layer{layer}" / f"{sumo_term.lower()}.json"
        if not concept_file.exists():
            # Try exact case
            concept_file = concepts_dir / f"layer{layer}" / f"{sumo_term}.json"
        if not concept_file.exists():
            skipped_not_found += 1
            errors.append(f"Not found: {sumo_term} L{layer}")
            continue

        # Load concept
        try:
            with open(concept_file) as f:
                concept = json.load(f)
        except Exception as e:
            errors.append(f"Error loading {sumo_term}: {e}")
            continue

        # Update opposite relationship (for Fisher-LDA contrastive steering)
        if "relationships" not in concept:
            concept["relationships"] = {}
        if "opposite" not in concept["relationships"]:
            concept["relationships"]["opposite"] = []

        # Add opposite if not already present
        current_opposites = concept["relationships"]["opposite"]
        if selected not in current_opposites:
            current_opposites.append(selected)

            # Also store the reasoning for reference
            if "opposite_reasoning" not in concept:
                concept["opposite_reasoning"] = rec.get("reasoning", "")

            if args.dry_run:
                print(f"  Would update: {sumo_term} L{layer} → {selected}")
            else:
                with open(concept_file, "w") as f:
                    json.dump(concept, f, indent=2)
            updated += 1
        else:
            # Already has this antonym
            pass

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Concepts updated: {updated}")
    print(f"  Skipped (no opposite): {skipped_no_opposite}")
    print(f"  Skipped (file not found): {skipped_not_found}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    if args.dry_run:
        print("\n  [DRY RUN - no files were modified]")
    else:
        print(f"\n  ✓ Updated {updated} concept files")


if __name__ == "__main__":
    main()
