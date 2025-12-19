#!/usr/bin/env python3
"""Split a concept pack tree into meld-operation chunks per top-level branch."""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def slugify(text: str) -> str:
    """Create a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-') or 'chunk'


def build_operations(
    node: Dict,
    parent: Optional[Dict],
    chunk_key: str,
    operations_by_chunk: Dict[str, List[Dict]]
) -> None:
    """Recursively collect create_meld operations for each non-leaf node."""

    children = node.get('children') or []

    if parent is not None and children:
        operations_by_chunk[chunk_key].append({
            'operation': 'create_meld',
            'target': node.get('label'),
            'target_id': node.get('id'),
            'new_parent': parent.get('label'),
            'parent_id': parent.get('id'),
            'source_concepts': [child.get('label') for child in children],
            'source_concept_ids': [child.get('id') for child in children],
            'target_layer': node.get('level'),
            'child_count': len(children)
        })

    for child in children:
        build_operations(child, node, chunk_key, operations_by_chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description='Chunk concept pack into meld operations')
    parser.add_argument('--concept-pack', type=Path, default=Path('cs_superset_concept_pack.json'),
                        help='Path to the concept pack JSON file')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory to write chunked meld operation files')
    parser.add_argument('--indent', type=int, default=2,
                        help='JSON indentation level for outputs')

    args = parser.parse_args()

    with args.concept_pack.open() as f:
        pack = json.load(f)

    root_label = pack.get('label', 'ROOT')
    top_level_nodes = pack.get('children', []) or []

    operations_by_chunk: Dict[str, List[Dict]] = defaultdict(list)
    chunk_node_info: Dict[str, Dict] = {}

    for node in top_level_nodes:
        label = node.get('label')
        if not label:
            continue
        chunk_node_info[label] = node
        build_operations(node, pack, label, operations_by_chunk)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pack.get('meta', {})
    timestamp = datetime.now(timezone.utc).isoformat()

    for chunk_label, ops in sorted(operations_by_chunk.items()):
        node_info = chunk_node_info.get(chunk_label, {})
        payload = {
            'concept_pack': {
                'id': pack.get('id'),
                'label': root_label,
                'namespace': meta.get('namespace'),
                'source_file': str(args.concept_pack),
                'version': meta.get('version'),
                'generated_at': timestamp
            },
            'chunk': {
                'label': chunk_label,
                'id': node_info.get('id'),
                'level': node_info.get('level'),
                'slug': slugify(chunk_label),
                'operations_count': len(ops),
                'child_count': len(node_info.get('children', []) or [])
            },
            'operations': {
                'create_meld': ops
            }
        }

        output_path = args.output_dir / f"{slugify(chunk_label)}_meld_operations.json"
        with output_path.open('w') as f:
            json.dump(payload, f, indent=args.indent)

        print(f"Wrote {len(ops)} operations to {output_path}")


if __name__ == '__main__':
    main()
