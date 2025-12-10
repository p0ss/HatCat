#!/usr/bin/env python3
"""
Analyze hierarchy for Information Architecture violations.

Checks:
1. Branching factor - parents with too many children (>MAX_CHILDREN)
2. Sibling layer span - children at different layers (granularity mismatch)
3. Depth vs layer mismatch - layer != ontological depth from root

Based on HATCAT_ARCHITECTURAL_PRINCIPLES.md:
- Layer should equal ontological depth
- Branching factor should be manageable (cognitive load)
- Siblings should be conceptual peers at same granularity
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


MAX_RECOMMENDED_CHILDREN = 12  # Manageable cognitive choice set


def get_depth(concept_key: str, child_to_parent: Dict[str, str], cache: Dict[str, int] = None) -> int:
    """Calculate depth from root for a concept."""
    if cache is None:
        cache = {}
    if concept_key in cache:
        return cache[concept_key]
    if concept_key not in child_to_parent:
        cache[concept_key] = 0
        return 0
    parent = child_to_parent[concept_key]
    depth = 1 + get_depth(parent, child_to_parent, cache)
    cache[concept_key] = depth
    return depth


def analyze_hierarchy(hierarchy_path: Path) -> Dict:
    """Analyze hierarchy for IA violations."""
    with open(hierarchy_path) as f:
        hier = json.load(f)

    results = {
        "branching_violations": [],
        "layer_span_violations": [],
        "depth_mismatches": [],
        "depth_distribution": {},
        "summary": {}
    }

    # 1. Branching factor analysis
    for parent, children in hier["parent_to_children"].items():
        if len(children) > MAX_RECOMMENDED_CHILDREN:
            results["branching_violations"].append({
                "parent": parent,
                "child_count": len(children),
                "children": children[:10] + ["..."] if len(children) > 10 else children
            })

    results["branching_violations"].sort(key=lambda x: x["child_count"], reverse=True)

    # 2. Sibling layer span analysis
    for parent, children in hier["parent_to_children"].items():
        child_layers = {}
        for c in children:
            name, layer = c.rsplit(":", 1)
            layer = int(layer)
            if layer not in child_layers:
                child_layers[layer] = []
            child_layers[layer].append(c)

        if len(child_layers) > 1:
            layers = sorted(child_layers.keys())
            span = max(layers) - min(layers)
            if span > 0:
                results["layer_span_violations"].append({
                    "parent": parent,
                    "layers": layers,
                    "span": span,
                    "children_by_layer": child_layers
                })

    results["layer_span_violations"].sort(key=lambda x: x["span"], reverse=True)

    # 3. Depth vs layer mismatch analysis
    depth_cache = {}
    all_concepts = set(hier["child_to_parent"].keys()) | set(hier.get("root_concepts", []))

    for concept_key in all_concepts:
        name, layer = concept_key.rsplit(":", 1)
        layer = int(layer)
        depth = get_depth(concept_key, hier["child_to_parent"], depth_cache)

        if layer != depth:
            results["depth_mismatches"].append({
                "concept": concept_key,
                "assigned_layer": layer,
                "actual_depth": depth,
                "difference": depth - layer
            })

    results["depth_mismatches"].sort(key=lambda x: abs(x["difference"]), reverse=True)

    # 4. Depth distribution
    depths = [get_depth(c, hier["child_to_parent"], depth_cache) for c in all_concepts]
    results["depth_distribution"] = dict(Counter(depths))

    # Summary
    results["summary"] = {
        "total_concepts": len(all_concepts),
        "branching_violations": len(results["branching_violations"]),
        "layer_span_violations": len(results["layer_span_violations"]),
        "depth_mismatches": len(results["depth_mismatches"]),
        "max_depth": max(depths) if depths else 0,
        "layers_needed": max(depths) + 1 if depths else 0,
        "worst_branching": results["branching_violations"][0] if results["branching_violations"] else None,
        "worst_layer_span": results["layer_span_violations"][0] if results["layer_span_violations"] else None
    }

    return results


def print_report(results: Dict, verbose: bool = False):
    """Print analysis report."""
    summary = results["summary"]

    print("=" * 60)
    print("HIERARCHY INFORMATION ARCHITECTURE ANALYSIS")
    print("=" * 60)

    print(f"\nTotal concepts: {summary['total_concepts']}")
    print(f"Max depth from root: {summary['max_depth']}")
    print(f"Layers needed for proper depth alignment: {summary['layers_needed']}")

    print(f"\n--- VIOLATIONS SUMMARY ---")
    print(f"Branching factor violations (>{MAX_RECOMMENDED_CHILDREN} children): {summary['branching_violations']}")
    print(f"Layer span violations (siblings at different layers): {summary['layer_span_violations']}")
    print(f"Depth/layer mismatches (layer != depth): {summary['depth_mismatches']}")

    if results["branching_violations"]:
        print(f"\n--- TOP BRANCHING VIOLATIONS ---")
        for v in results["branching_violations"][:10]:
            print(f"  {v['parent']}: {v['child_count']} children")

    if results["layer_span_violations"]:
        print(f"\n--- TOP LAYER SPAN VIOLATIONS ---")
        for v in results["layer_span_violations"][:10]:
            print(f"  {v['parent']}: children at layers {v['layers']} (span={v['span']})")

    if results["depth_mismatches"]:
        print(f"\n--- TOP DEPTH MISMATCHES ---")
        for v in results["depth_mismatches"][:10]:
            print(f"  {v['concept']}: layer={v['assigned_layer']}, depth={v['actual_depth']} (diff={v['difference']:+d})")

    print(f"\n--- DEPTH DISTRIBUTION ---")
    for depth in sorted(results["depth_distribution"].keys()):
        count = results["depth_distribution"][depth]
        bar = "#" * min(50, count // 20)
        print(f"  Depth {depth:2d}: {count:4d} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hierarchy for IA violations")
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="concept_packs/sumo-wordnet-v4/hierarchy/hierarchy.json",
        help="Path to hierarchy.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed violations"
    )

    args = parser.parse_args()

    hierarchy_path = Path(args.hierarchy)
    if not hierarchy_path.exists():
        print(f"Error: Hierarchy not found: {hierarchy_path}")
        return 1

    results = analyze_hierarchy(hierarchy_path)
    print_report(results, args.verbose)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
