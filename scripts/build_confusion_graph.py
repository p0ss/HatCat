#!/usr/bin/env python3
"""
Build cross-concept confusion graph from polar MELDs.

This script analyzes the sourced_from_positives_of references across all MELDs
to create a graph showing which concepts are easily confused with which others.
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_all_melds(meld_dir: Path) -> Dict[str, Dict]:
    """Load all MELDs from L1-L3 directories."""
    melds = {}

    for level in ["L1", "L2", "L3"]:
        level_dir = meld_dir / level
        if not level_dir.exists():
            continue

        for meld_file in level_dir.glob("*.json"):
            try:
                data = json.loads(meld_file.read_text())
                term = data["polar_meld"]["term"]
                melds[term] = {
                    "file": str(meld_file),
                    "level": int(level[1]),
                    "node_id": data["node"]["id"],
                    "node_label": data["node"]["label"],
                    "data": data
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {meld_file}: {e}")

    return melds


def extract_confusion_references(melds: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract sourced_from_positives_of references from all MELDs."""
    confusion_refs = {}

    for term, meld_info in melds.items():
        polar_meld = meld_info["data"]["polar_meld"]
        poles = polar_meld.get("poles", {})

        positive_sources = []
        negative_sources = []

        # Extract from positive pole
        if "positive" in poles:
            confusables = poles["positive"].get("confusables", {})
            positive_sources = confusables.get("sourced_from_positives_of", [])

        # Extract from negative pole
        if "negative" in poles:
            confusables = poles["negative"].get("confusables", {})
            negative_sources = confusables.get("sourced_from_positives_of", [])

        confusion_refs[term] = {
            "level": meld_info["level"],
            "node_id": meld_info["node_id"],
            "positive_confusables_from": positive_sources,
            "negative_confusables_from": negative_sources,
            "all_sources": list(set(positive_sources + negative_sources))
        }

    return confusion_refs


def build_confusion_graph(confusion_refs: Dict[str, Dict]) -> Dict:
    """Build bidirectional confusion graph."""

    # Forward: concept -> concepts whose positives are used as confusables
    sources_confusables_from = {}

    # Reverse: concept -> concepts that use this concept's positives as confusables
    is_confusable_for = defaultdict(lambda: {"positive_pole": [], "negative_pole": []})

    # Track unresolved references (concepts mentioned but not in our MELD set)
    unresolved = defaultdict(list)
    all_terms = set(confusion_refs.keys())

    for term, refs in confusion_refs.items():
        sources_confusables_from[term] = {
            "positive_pole": refs["positive_confusables_from"],
            "negative_pole": refs["negative_confusables_from"]
        }

        # Build reverse mapping
        for source in refs["positive_confusables_from"]:
            if source in all_terms:
                is_confusable_for[source]["positive_pole"].append(term)
            else:
                unresolved[source].append((term, "positive"))

        for source in refs["negative_confusables_from"]:
            if source in all_terms:
                is_confusable_for[source]["negative_pole"].append(term)
            else:
                unresolved[source].append((term, "negative"))

    return {
        "sources_confusables_from": sources_confusables_from,
        "is_confusable_for": dict(is_confusable_for),
        "unresolved_references": dict(unresolved)
    }


def compute_statistics(confusion_refs: Dict[str, Dict], graph: Dict) -> Dict:
    """Compute statistics about the confusion graph."""

    # Count concepts by how many they reference
    ref_counts = [len(refs["all_sources"]) for refs in confusion_refs.values()]

    # Count concepts by how often they're referenced
    referenced_counts = defaultdict(int)
    for refs in confusion_refs.values():
        for source in refs["all_sources"]:
            referenced_counts[source] += 1

    # Find most confusable concepts (most frequently referenced)
    most_confusable = sorted(
        referenced_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # Find concepts with most cross-references
    most_referencing = sorted(
        [(term, len(refs["all_sources"])) for term, refs in confusion_refs.items()],
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # Count by level
    level_counts = defaultdict(int)
    for refs in confusion_refs.values():
        level_counts[refs["level"]] += 1

    # Find bidirectional confusions (A confused with B and B confused with A)
    bidirectional = []
    for term, refs in confusion_refs.items():
        for source in refs["all_sources"]:
            if source in confusion_refs:
                if term in confusion_refs[source]["all_sources"]:
                    pair = tuple(sorted([term, source]))
                    if pair not in bidirectional:
                        bidirectional.append(pair)

    return {
        "total_concepts": len(confusion_refs),
        "concepts_by_level": dict(level_counts),
        "total_references": sum(ref_counts),
        "avg_references_per_concept": sum(ref_counts) / len(ref_counts) if ref_counts else 0,
        "max_references": max(ref_counts) if ref_counts else 0,
        "concepts_with_no_references": sum(1 for c in ref_counts if c == 0),
        "most_confusable_concepts": most_confusable,
        "most_referencing_concepts": most_referencing,
        "bidirectional_confusions": len(bidirectional),
        "bidirectional_pairs": bidirectional[:50],  # First 50 pairs
        "unresolved_reference_count": len(graph["unresolved_references"])
    }


def main():
    parser = argparse.ArgumentParser(description="Build confusion graph from polar MELDs")
    parser.add_argument(
        "--meld-dir", "-m",
        type=Path,
        default=Path("results/polar_melds"),
        help="Directory containing polar MELDs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/confusion_graph.json"),
        help="Output file for confusion graph"
    )

    args = parser.parse_args()

    # Load all MELDs
    logger.info(f"Loading MELDs from {args.meld_dir}...")
    melds = load_all_melds(args.meld_dir)
    logger.info(f"Loaded {len(melds)} MELDs")

    # Extract confusion references
    logger.info("Extracting confusion references...")
    confusion_refs = extract_confusion_references(melds)

    # Build graph
    logger.info("Building confusion graph...")
    graph = build_confusion_graph(confusion_refs)

    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(confusion_refs, graph)

    # Combine output
    output = {
        "metadata": {
            "total_melds": len(melds),
            "source_directory": str(args.meld_dir)
        },
        "statistics": stats,
        "graph": graph,
        "concept_references": confusion_refs
    }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print("CONFUSION GRAPH ANALYSIS")
    print(f"{'='*60}")
    print(f"Total concepts: {stats['total_concepts']}")
    print(f"Concepts by level: {stats['concepts_by_level']}")
    print(f"\nCross-references:")
    print(f"  Total references: {stats['total_references']}")
    print(f"  Average per concept: {stats['avg_references_per_concept']:.2f}")
    print(f"  Max references: {stats['max_references']}")
    print(f"  Concepts with no refs: {stats['concepts_with_no_references']}")
    print(f"\nBidirectional confusions: {stats['bidirectional_confusions']}")
    print(f"Unresolved references: {stats['unresolved_reference_count']}")
    print(f"\nMost confusable concepts (most frequently referenced):")
    for concept, count in stats["most_confusable_concepts"][:10]:
        print(f"  {count:3d}x - {concept}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
