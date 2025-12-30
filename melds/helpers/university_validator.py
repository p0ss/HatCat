#!/usr/bin/env python3
"""Validate university-built taxonomies before chunking into melds."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

PLACEHOLDER_TERMS = {"general", "introduction", "overview", "basics", "misc", "other", "foundations", "fundamentals", "miscellaneous"}
CAMEL_CASE_RE = re.compile(r'^[A-Z][A-Za-z0-9]+$')


class NodeInfo:
    def __init__(self, node_id: str, label: str, level: int, path: List[str], child_count: int):
        self.node_id = node_id
        self.label = label
        self.level = level
        self.path = path
        self.child_count = child_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.node_id,
            'label': self.label,
            'level': self.level,
            'path': " > ".join(self.path),
            'child_count': self.child_count
        }


def normalize_label(label: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', label.lower()).strip()


def is_placeholder(label: str) -> bool:
    tokens = normalize_label(label).split()
    return any(t in PLACEHOLDER_TERMS for t in tokens)


def is_camel_case(label: str) -> bool:
    # Allow multi-token CamelCase (no spaces); fallback to simple letters
    return bool(CAMEL_CASE_RE.match(label))


def tokenize(label: str) -> List[str]:
    return [tok for tok in re.split(r'[^a-z0-9]+', label.lower()) if tok]


def jaccard(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def traverse(node: Dict[str, Any], path: List[str], nodes: List[NodeInfo], registry: Dict[str, List[NodeInfo]]):
    children = node.get('children', []) or []
    current_path = path + [node['label']]
    info = NodeInfo(
        node.get('id', ''),
        node.get('label', ''),
        node.get('level', len(current_path)),
        current_path,
        len(children)
    )
    nodes.append(info)
    registry[normalize_label(info.label)].append(info)

    for child in children:
        traverse(child, current_path, nodes, registry)


def detect_naming_issues(nodes: List[NodeInfo]) -> List[Dict[str, Any]]:
    issues = []
    for info in nodes:
        reasons = []
        if is_placeholder(info.label):
            reasons.append('placeholder term detected')
        if ' ' in info.label:
            reasons.append('contains spaces; expected CamelCase/PascalCase')
        if not is_camel_case(info.label):
            reasons.append('not CamelCase/PascalCase')
        if reasons:
            issues.append({
                'node': info.to_dict(),
                'reasons': reasons
            })
    return issues


def detect_duplicates(registry: Dict[str, List[NodeInfo]]) -> List[Dict[str, Any]]:
    duplicates = []
    for normalized, entries in registry.items():
        if len(entries) > 1:
            duplicates.append({
                'label': normalized,
                'occurrences': [e.to_dict() for e in entries]
            })
    return duplicates


def detect_sibling_overlap(nodes_by_parent: Dict[str, List[NodeInfo]]) -> List[Dict[str, Any]]:
    overlap_issues = []
    for parent_path, children in nodes_by_parent.items():
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                a = children[i]
                b = children[j]
                sim = jaccard(tokenize(a.label), tokenize(b.label))
                if sim >= 0.6:
                    overlap_issues.append({
                        'parent_path': parent_path,
                        'child_a': a.to_dict(),
                        'child_b': b.to_dict(),
                        'similarity': round(sim, 2)
                    })
    return overlap_issues


def collect_nodes_by_parent(tree: Dict[str, Any]) -> Dict[str, List[NodeInfo]]:
    nodes_by_parent: Dict[str, List[NodeInfo]] = defaultdict(list)

    def helper(node: Dict[str, Any], path: List[str]):
        children = node.get('children', []) or []
        current_path = path + [node['label']]
        info = NodeInfo(
            node.get('id', ''),
            node.get('label', ''),
            node.get('level', len(current_path)),
            current_path,
            len(children)
        )
        if path:
            nodes_by_parent[" > ".join(path)].append(info)
        for child in children:
            helper(child, current_path)

    helper(tree, [])
    return nodes_by_parent


def leaf_precision_score(nodes: List[NodeInfo], leaf_level_goal: int = 6) -> Dict[str, Any]:
    leaves = [n for n in nodes if n.child_count == 0]
    precise = 0
    short = 0
    level_gap = 0

    for leaf in leaves:
        token_count = len(tokenize(leaf.label))
        if token_count <= 2:
            short += 1
        else:
            precise += 1
        if leaf.level < leaf_level_goal:
            level_gap += 1

    total = len(leaves)
    score = 0.0 if total == 0 else round(precise / total, 3)
    return {
        'leaf_count': total,
        'precise_leaves': precise,
        'short_label_leaves': short,
        'leaves_below_target_level': level_gap,
        'precision_score': score
    }


def identify_regeneration_targets(
    naming_issues: List[Dict[str, Any]],
    duplicates: List[Dict[str, Any]],
    overlap: List[Dict[str, Any]],
    label_counts: Dict[str, int]
) -> List[Dict[str, Any]]:
    targets = {}

    for issue in naming_issues:
        node = issue['node']
        targets[node['id']] = {
            'node': node,
            'reason': 'naming_issue',
            'details': issue['reasons']
        }

    for dup in duplicates:
        for node in dup['occurrences']:
            path_parts = node['path'].split(' > ')
            parent_label = path_parts[-2] if len(path_parts) >= 2 else ''
            suggestion = auto_disambiguate_label(node['label'], parent_label, label_counts)
            targets[node['id']] = {
                'node': node,
                'reason': 'duplicate_label',
                'details': f"Label '{dup['label']}' appears {len(dup['occurrences'])} times",
                'suggested_label': suggestion if suggestion != node['label'] else None
            }

    for ov in overlap:
        for key in ['child_a', 'child_b']:
            node = ov[key]
            targets[node['id']] = {
                'node': node,
                'reason': 'sibling_overlap',
                'details': f"Similarity {ov['similarity']} under {ov['parent_path']}"
            }

    return list(targets.values())


def auto_disambiguate_label(label: str, parent_label: str, existing_normalized: Dict[str, int]) -> str:
    base = label
    normalized = normalize_label(base)
    if existing_normalized.get(normalized, 0) <= 1:
        return label

    candidate = f"{parent_label}{label}"
    norm_candidate = normalize_label(candidate)
    suffix = 1
    while norm_candidate in existing_normalized:
        candidate = f"{parent_label}{label}{suffix}"
        norm_candidate = normalize_label(candidate)
        suffix += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description='Taxonomy lint report generator for university-built trees')
    parser.add_argument('--tree-file', type=Path, required=True, help='Path to the university JSON file')
    parser.add_argument('--report-file', type=Path, help='Optional JSON file to write report')
    parser.add_argument('--regen-file', type=Path, help='Optional JSON file listing nodes to regenerate')

    args = parser.parse_args()

    with args.tree_file.open() as f:
        tree = json.load(f)

    nodes: List[NodeInfo] = []
    registry: Dict[str, List[NodeInfo]] = defaultdict(list)
    traverse(tree, [], nodes, registry)

    naming_issues = detect_naming_issues(nodes)
    duplicates = detect_duplicates(registry)
    nodes_by_parent = collect_nodes_by_parent(tree)
    sibling_overlap = detect_sibling_overlap(nodes_by_parent)
    leaf_stats = leaf_precision_score(nodes)
    label_counts = {norm: len(entries) for norm, entries in registry.items()}
    regen_targets = identify_regeneration_targets(naming_issues, duplicates, sibling_overlap, label_counts)

    report = {
        'summary': {
            'total_nodes': len(nodes),
            'naming_issues': len(naming_issues),
            'duplicate_labels': len(duplicates),
            'sibling_overlaps': len(sibling_overlap),
            'regeneration_targets': len(regen_targets),
            'leaf_precision_score': leaf_stats['precision_score']
        },
        'leaf_stats': leaf_stats,
        'naming_issues': naming_issues,
        'duplicate_labels': duplicates,
        'sibling_overlap': sibling_overlap,
        'regeneration_targets': regen_targets
    }

    print("Taxonomy Lint Report")
    print("---------------------")
    for key, value in report['summary'].items():
        print(f"{key}: {value}")

    if args.report_file:
        with args.report_file.open('w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report written to {args.report_file}")

    if args.regen_file:
        with args.regen_file.open('w') as f:
            json.dump({'targets': regen_targets}, f, indent=2)
        print(f"Regeneration targets written to {args.regen_file}")


if __name__ == '__main__':
    main()
