#!/usr/bin/env python3
"""
Clean up skeleton labels by removing institutional prefixes.

Removes prefixes like "School of", "University of", "Department of", etc.
that would bias the probes toward detecting the word "school" rather than
the actual concept.

Usage:
    python scripts/cleanup_skeleton_labels.py results/ontology_skeleton_v2.json
    python scripts/cleanup_skeleton_labels.py results/ontology_skeleton_v2.json --dry-run
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


# Prefixes to remove (case-insensitive)
PREFIXES_TO_REMOVE = [
    # Institutional
    r"^(?:The\s+)?University\s+of\s+",
    r"^(?:The\s+)?College\s+of\s+",
    r"^(?:The\s+)?School\s+of\s+",
    r"^(?:The\s+)?Department\s+of\s+",
    r"^(?:The\s+)?Institute\s+of\s+",
    r"^(?:The\s+)?Academy\s+of\s+",
    r"^(?:The\s+)?Faculty\s+of\s+",
    r"^(?:The\s+)?Center\s+for\s+",
    r"^(?:The\s+)?Centre\s+for\s+",
    r"^(?:The\s+)?Program\s+in\s+",
    r"^(?:The\s+)?Programme\s+in\s+",
    # Variations without "of"
    r"^(?:The\s+)?University\s+",
    r"^(?:The\s+)?College\s+",
    r"^(?:The\s+)?School\s+",
    r"^(?:The\s+)?Department\s+",
    r"^(?:The\s+)?Institute\s+",
    r"^(?:The\s+)?Academy\s+",
    # Suffixes that might appear
    r"\s+School$",
    r"\s+Institute$",
    r"\s+Academy$",
    r"\s+College$",
    r"\s+University$",
    r"\s+Department$",
    r"\s+Center$",
    r"\s+Centre$",
    r"\s+Program$",
    r"\s+Programme$",
    r"\s+Lab$",
    r"\s+Laboratory$",
]


def clean_label(label: str) -> str:
    """Remove institutional prefixes/suffixes from a label."""
    original = label

    # Apply each pattern
    for pattern in PREFIXES_TO_REMOVE:
        label = re.sub(pattern, "", label, flags=re.IGNORECASE)

    # Clean up whitespace
    label = label.strip()

    # Handle edge cases where the whole thing was a prefix
    if not label:
        return original

    return label


def clean_node(node: dict, stats: Counter) -> dict:
    """Recursively clean a skeleton node and its children."""
    original_label = node.get("label", "")
    cleaned_label = clean_label(original_label)

    if cleaned_label != original_label:
        stats["cleaned"] += 1
        stats[f"L{node.get('level', '?')}_cleaned"] += 1
    else:
        stats["unchanged"] += 1

    node["label"] = cleaned_label

    # Also clean the ID to match (kebab-case)
    if cleaned_label != original_label:
        node["id"] = cleaned_label.lower().replace(" ", "-").replace("&", "and")

    # Recurse into children
    if "children" in node:
        node["children"] = [clean_node(child, stats) for child in node["children"]]

    return node


def main():
    parser = argparse.ArgumentParser(description="Clean institutional prefixes from skeleton labels")
    parser.add_argument("skeleton_file", type=Path, help="Path to skeleton JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying file")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: overwrite input)")

    args = parser.parse_args()

    if not args.skeleton_file.exists():
        print(f"Error: {args.skeleton_file} not found")
        return 1

    # Load skeleton
    with open(args.skeleton_file) as f:
        skeleton = json.load(f)

    stats = Counter()

    # Show some examples before cleaning
    print("Example labels before cleaning:")
    examples = []
    def collect_examples(node, depth=0):
        if len(examples) < 10:
            examples.append((node.get("level", "?"), node.get("label", "")))
        for child in node.get("children", []):
            collect_examples(child, depth + 1)

    for root in skeleton.get("roots", []):
        collect_examples(root)

    for level, label in examples[:10]:
        cleaned = clean_label(label)
        marker = " → " + cleaned if cleaned != label else ""
        print(f"  L{level}: {label}{marker}")

    print()

    # Clean all nodes
    skeleton["roots"] = [clean_node(root, stats) for root in skeleton.get("roots", [])]

    # Report stats
    print(f"Cleaning complete:")
    print(f"  Total cleaned: {stats['cleaned']}")
    print(f"  Unchanged: {stats['unchanged']}")
    for key in sorted(stats.keys()):
        if key.startswith("L") and key.endswith("_cleaned"):
            print(f"    {key}: {stats[key]}")

    if args.dry_run:
        print("\n[DRY RUN - no changes written]")

        # Show some cleaned examples
        print("\nSample cleaned labels:")
        cleaned_examples = []
        def collect_cleaned(node):
            if len(cleaned_examples) < 10:
                cleaned_examples.append((node.get("level", "?"), node.get("label", "")))
            for child in node.get("children", []):
                collect_cleaned(child)

        for root in skeleton.get("roots", []):
            collect_cleaned(root)

        for level, label in cleaned_examples[:10]:
            print(f"  L{level}: {label}")
    else:
        # Write output
        output_path = args.output or args.skeleton_file
        with open(output_path, "w") as f:
            json.dump(skeleton, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
