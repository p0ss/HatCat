#!/usr/bin/env python3
"""Generate meld descriptions from a chunk file and emit MAP requests."""

import argparse
import anthropic
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

SYSTEM_PROMPT = """You are an expert ontologist helping to define intermediate concepts for a concept hierarchy.

For each concept you must return valid JSON:
{
  "term": "<concept name>",
  "definition": "<20-200 char definition>",
  "positive_examples": ["sentence", ...],
  "negative_examples": ["near-miss", ...],
  "contrast_concepts": ["sibling", ...]
}

Guidelines:
- Definition must capture what unifies all child concepts without listing them
- Positive examples: 4-6 natural sentences where the concept clearly appears
- Negative examples: 4-6 near misses (siblings, parents, confusing neighbors)
- Contrast concepts: 3-5 related concepts to disambiguate from
- Stay within provided parent scope; do not invent new children
- Use concise everyday language, no markdown
"""

PLACEHOLDER_DEF = "Definition unavailable"


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


def get_domain_for_parent(parent: str) -> str:
    domain_map = {
        "MindsAndAgents": ["Agent", "Cognitive", "Mental", "Psychological", "Social", "Communication", "Intention", "Process", "Action", "Reasoning"],
        "CreatedThings": ["Artifact", "Device", "Tool", "Container", "Machine", "Vehicle", "Furniture", "Weapon", "Instrument", "Building"],
        "PhysicalWorld": ["Physical", "Natural", "Geographic", "Weather", "Material", "Chemical", "Motion", "Force", "Energy"],
        "LivingThings": ["Organism", "Animal", "Plant", "Body", "Biological", "Anatomical"],
        "Information": ["Information", "Content", "Text", "Data", "Document", "Record", "Proposition", "Representation"],
    }

    parent_lower = parent.lower()
    for domain, keywords in domain_map.items():
        for kw in keywords:
            if kw.lower() in parent_lower:
                return domain
    return "CreatedThings"


def load_existing_concepts(concept_pack_dir: Optional[Path]) -> Dict[str, Dict]:
    if not concept_pack_dir:
        return {}

    hierarchy_dir = concept_pack_dir / 'hierarchy'
    if not hierarchy_dir.exists():
        return {}

    concepts = {}
    for layer in range(7):
        layer_file = hierarchy_dir / f'layer{layer}.json'
        if layer_file.exists():
            with open(layer_file) as f:
                data = json.load(f)
            for c in data.get('concepts', []):
                concepts[c['sumo_term']] = c
    return concepts


def build_meld_prompt(op: Dict, existing_concepts: Dict, chunk_label: str) -> str:
    target = op['target']
    parent = op['new_parent']
    sources = op.get('source_concepts', [])
    layer = op.get('target_layer', 3)

    parent_info = existing_concepts.get(parent, {})
    parent_def = parent_info.get('definition', parent_info.get('sumo_definition', PLACEHOLDER_DEF))[:200]

    source_info = []
    for src in sources[:10]:
        info = existing_concepts.get(src, {})
        source_info.append({
            'name': src,
            'definition': info.get('definition', info.get('sumo_definition', PLACEHOLDER_DEF))[:160]
        })

    prompt = f"""Chunk: {chunk_label}
Concept: {target}
Layer: {layer}
Parent: {parent}
Parent definition: {parent_def}

Children to unify ({len(sources)} total):
{json.dumps(source_info, indent=2)}
{f"Note: {len(sources) - 10} additional children not shown" if len(sources) > 10 else ""}

Requirements:
1. Definition must explain what unites all children without naming them
2. Positive examples: 4-6 natural sentences that clearly describe {target}
3. Negative examples: 4-6 near misses (siblings or parent scope items)
4. Contrast concepts: 3-5 related concepts to disambiguate (use siblings)
Return ONLY valid JSON, no markdown fences.
"""
    return prompt


def convert_to_candidate(term: str, meld: Dict) -> Dict:
    parent = meld.get('parent', '')
    layer = meld.get('layer', 3)
    children = meld.get('children', [])

    candidate = {
        'term': term,
        'role': 'concept',
        'parent_concepts': [parent] if parent else [],
        'layer_hint': layer,
        'definition': meld.get('definition', ''),
        'domain': get_domain_for_parent(parent),
        'training_hints': {
            'positive_examples': meld.get('positive_examples', []),
            'negative_examples': meld.get('negative_examples', []),
            'disambiguation': f"Not to be confused with: {', '.join(meld.get('contrast_concepts', []))}"
        },
        'relationships': {
            'related': meld.get('contrast_concepts', []),
            'has_part': children
        },
        'safety_tags': {
            'risk_level': 'low',
            'impacts': [],
            'treaty_relevant': False,
            'harness_relevant': False
        }
    }

    if children:
        candidate['children'] = children

    return candidate


def create_meld_request(parent: str, candidates: List[Dict], target_pack: str, version: str, chunk_label: str) -> Dict:
    slug = f"{slugify(chunk_label)}-{slugify(parent)}-melds"
    meld_id = f"org.hatcat/{slug}@{version}"
    created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    domains = [c.get('domain', 'CreatedThings') for c in candidates]
    primary_domain = max(set(domains), key=domains.count)

    request = {
        'meld_request_id': meld_id,
        'target_pack_spec_id': target_pack,
        'metadata': {
            'name': f"{parent} Melds ({chunk_label})",
            'description': f"Chunk {chunk_label}: new intermediate concepts under {parent}",
            'source': 'university_builder_chunk',
            'author': 'generate_chunk_descriptions',
            'created': created,
            'generator_model': 'claude-sonnet-4-20250514',
            'primary_domain': primary_domain
        },
        'attachment_points': [
            {
                'target_concept_id': f"{target_pack}::concept/{parent}",
                'relationship': 'parent_of',
                'candidate_concept': c['term']
            }
            for c in candidates
        ],
        'candidates': candidates,
        'validation': {
            'status': 'pending',
            'errors': [],
            'warnings': [],
            'validated_at': None
        }
    }
    return request


def generate_description(client: anthropic.Anthropic, op: Dict, existing_concepts: Dict, chunk_label: str, model: str) -> Optional[Dict]:
    prompt = build_meld_prompt(op, existing_concepts, chunk_label)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            parts = content.split("```")
            for part in parts:
                if part.strip().startswith('{'):
                    content = part.strip()
                    break
        result = json.loads(content)
        result['parent'] = op['new_parent']
        result['children'] = op.get('source_concepts', [])
        result['layer'] = op.get('target_layer', 3)
        result['generated'] = True
        return result
    except Exception as e:
        print(f"  !!! Error for {op['target']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate meld descriptions for a concept chunk')
    parser.add_argument('--chunk-file', type=Path, required=True, help='Chunk meld operations JSON')
    parser.add_argument('--concept-pack', type=Path, help='Optional concept pack directory for definitions')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for MAP request outputs')
    parser.add_argument('--checkpoint', type=Path, help='Optional checkpoint JSON path for generated melds')
    parser.add_argument('--target-pack', type=str, default='org.hatcat/sumo-wordnet-v4@5.0.0', help='Target pack spec ID')
    parser.add_argument('--version', type=str, default='0.1.0', help='Meld request version')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', help='Anthropic model')
    parser.add_argument('--batch-size', type=int, default=5, help='Save checkpoint every N melds')
    parser.add_argument('--start-index', type=int, default=0, help='Start index for operations')
    parser.add_argument('--dry-run', action='store_true', help='Preview operations without API calls')

    args = parser.parse_args()

    with args.chunk_file.open() as f:
        chunk = json.load(f)

    operations = chunk.get('operations', {}).get('create_meld', [])
    chunk_label = chunk.get('chunk', {}).get('label', args.chunk_file.stem)
    chunk_slug = chunk.get('chunk', {}).get('slug', slugify(chunk_label))

    print(f"Loaded chunk '{chunk_label}' with {len(operations)} create_meld ops")

    existing_concepts = load_existing_concepts(args.concept_pack)
    if existing_concepts:
        print(f"Loaded {len(existing_concepts)} concept definitions for context")

    output_dir = args.output_dir / chunk_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or (output_dir / f"{chunk_slug}_generated.json")
    results: Dict[str, Dict] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open() as f:
            results = json.load(f).get('melds', {})
        print(f"Resumed {len(results)} existing melds from {checkpoint_path}")

    if args.dry_run:
        for i, op in enumerate(operations[:5]):
            print(f"[{i}] {op['target']} (parent={op['new_parent']}, children={len(op.get('source_concepts', []))})")
        return

    client = anthropic.Anthropic()
    total = len(operations)

    for idx, op in enumerate(operations[args.start_index:], args.start_index):
        target = op['target']
        if target in results:
            print(f"[{idx}/{total}] {target}: cached")
            continue

        print(f"[{idx}/{total}] Generating {target} under {op['new_parent']}")
        result = generate_description(client, op, existing_concepts, chunk_label, args.model)
        if result:
            results[target] = result
            print(f"  ✓ {result['definition'][:70]}...")
        else:
            print("  ✗ generation failed")

        time.sleep(0.5)

        if (idx + 1) % args.batch_size == 0:
            with checkpoint_path.open('w') as f:
                json.dump({'melds': results}, f, indent=2)
            print(f"  Saved checkpoint ({len(results)} melds)")

    with checkpoint_path.open('w') as f:
        json.dump({'melds': results}, f, indent=2)
    print(f"Saved final checkpoint with {len(results)} melds")

    # Convert to MAP requests grouped by parent
    grouped = defaultdict(list)
    for term, meld in results.items():
        candidate = convert_to_candidate(term, meld)
        parent = meld.get('parent', 'Unknown')
        grouped[parent].append(candidate)

    written = 0
    for parent, candidates in grouped.items():
        request = create_meld_request(parent, candidates, args.target_pack, args.version, chunk_label)
        file_path = output_dir / f"{slugify(parent)}_meld_request.json"
        with file_path.open('w') as f:
            json.dump(request, f, indent=2)
        written += 1
        print(f"Wrote MAP request for {parent} -> {file_path}")

    print(f"Done. Generated {len(results)} melds and {written} MAP request files in {output_dir}")


if __name__ == '__main__':
    main()
