#!/usr/bin/env python3
"""
Generate meld descriptions using the Anthropic API.

For each new concept in the meld_operations.json, calls Claude to generate
high-quality definitions and training examples following the HATCAT_MELD_POLICY
and MAP_MELDING specification.

Usage:
    python scripts/generate_meld_descriptions.py \
        --meld-ops lens_packs/apertus-8b_first-light-v1/meld_operations.json \
        --output data/meld_descriptions/generated_melds.json \
        --batch-size 5 \
        --start-index 0
"""

import anthropic
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional


# System prompt with MELD policy context
SYSTEM_PROMPT = """You are an expert ontologist helping to create meld descriptions for a concept hierarchy.

## Context: HatCat Meld System

HatCat is a system for training "lenses" - linear probes that detect when specific concepts are active in a language model's activations. The hierarchy organizes ~7000 concepts across layers (0-4) with parent-child relationships.

A "meld" is a new intermediate concept being created to group existing child concepts. For example, "TransportContainers" melds together [TravelContainer, ShipContainer, ProductPackage, etc.].

## Your Task

For each meld concept, you must provide:

1. **definition** (20-200 characters): A clear, definitional statement ("X is a...").
   - Should NOT be circular (don't just list the children)
   - Should capture the semantic essence that unifies the children
   - First sentence should be definitional

2. **positive_examples** (4-6 examples): Natural language sentences where this concept would activate.
   - Diverse scenarios and phrasings
   - Should be natural, not technical jargon
   - Should genuinely involve the concept

3. **negative_examples** (4-6 examples): Sentences where this concept should NOT activate.
   - Near-misses from the same domain (most important!)
   - Sibling concepts or parent concepts
   - Things that might be confused with this concept

4. **contrast_concepts** (3-5 concepts): Related concepts to discriminate against.
   - Siblings under the same parent
   - Things commonly confused
   - NOT the children (they are positives, not contrasts)

## Style Guidelines

- Prefer shorter terms for readability
- Definitions should be 20-200 characters
- Examples should be complete, natural sentences
- Negative examples should be "near misses" - things close but different
- Think about what would be confusing for a classifier

## Output Format

Return valid JSON with this exact structure:
{
  "term": "<concept name>",
  "definition": "<clear definitional statement>",
  "positive_examples": ["example1", "example2", ...],
  "negative_examples": ["near-miss1", "near-miss2", ...],
  "contrast_concepts": ["sibling1", "sibling2", ...]
}
"""


def build_meld_prompt(meld_op: Dict, existing_concepts: Dict) -> str:
    """Build the prompt for a single meld operation."""

    target = meld_op['target']
    parent = meld_op['new_parent']
    sources = meld_op['source_concepts']
    layer = meld_op['target_layer']

    # Check if this is an over-firer enhancement (has over_fire_on_sample)
    is_overfirer = 'over_fire_on_sample' in meld_op
    over_fire_on = meld_op.get('over_fire_on_sample', [])
    over_fire_count = meld_op.get('over_fire_count', 0)
    existing_def = meld_op.get('existing_definition', '')

    # Get info about source concepts if available
    source_info = []
    for src in sources[:10]:  # Limit to first 10 to avoid prompt bloat
        if src in existing_concepts:
            info = existing_concepts[src]
            source_info.append({
                'name': src,
                'definition': info.get('definition', info.get('sumo_definition', 'N/A'))[:200]
            })
        else:
            source_info.append({'name': src, 'definition': 'N/A'})

    # Get parent info
    parent_info = existing_concepts.get(parent, {})
    parent_def = parent_info.get('definition', parent_info.get('sumo_definition', 'N/A'))[:200]

    # Get info about over-fire concepts for negative examples
    over_fire_info = []
    if is_overfirer:
        for ofc in over_fire_on[:15]:
            if ofc in existing_concepts:
                info = existing_concepts[ofc]
                over_fire_info.append({
                    'name': ofc,
                    'definition': info.get('definition', info.get('sumo_definition', 'N/A'))[:150]
                })
            else:
                over_fire_info.append({'name': ofc, 'definition': 'N/A'})

    if is_overfirer:
        # Build prompt for over-firer enhancement
        prompt = f"""Please generate enhanced training data for this EXISTING concept that is over-firing (incorrectly activating on unrelated concepts):

## Concept: {target}

**Layer**: {layer} (0=most abstract, 6=most specific)
**Parent concept**: {parent}
**Parent definition**: {parent_def}
**Existing definition**: {existing_def if existing_def else "None - needs definition"}

## CRITICAL PROBLEM

This concept's classifier is OVER-FIRING - it incorrectly activates on {over_fire_count} other concepts!

**Concepts it incorrectly fires on** (sample of {len(over_fire_on)}):
{json.dumps(over_fire_info, indent=2)}

## Requirements

1. **definition**: A precise, narrow definition that EXCLUDES the over-fire concepts
2. **positive_examples**: 4-6 sentences that are SPECIFICALLY about {target}, not the over-fire concepts
3. **negative_examples**: 4-6 sentences about the OVER-FIRE CONCEPTS listed above - these are what the classifier needs to learn to REJECT
4. **contrast_concepts**: The over-fire concepts it should NOT activate on

The goal is to make the classifier MORE DISCRIMINATING - it should ONLY fire on genuine {target} content, not on {over_fire_on[0] if over_fire_on else 'other concepts'}, {over_fire_on[1] if len(over_fire_on) > 1 else 'etc'}, etc.

Return ONLY valid JSON, no markdown code blocks."""
    else:
        # Original prompt for new meld concepts
        prompt = f"""Please generate a meld description for this new concept:

## Concept: {target}

**Layer**: {layer} (0=most abstract, 4=most specific)
**Parent concept**: {parent}
**Parent definition**: {parent_def}

**Child concepts being melded** ({len(sources)} total):
{json.dumps(source_info, indent=2)}

{f"Note: {len(sources) - 10} more children not shown" if len(sources) > 10 else ""}

## Requirements

1. The definition should capture what unifies ALL these children
2. Positive examples should be diverse scenarios involving {target}
3. Negative examples should be near-misses - related but NOT {target}
4. Contrast concepts should be siblings of {target} (other children of {parent})

Return ONLY valid JSON, no markdown code blocks."""

    return prompt


def load_existing_concepts(concept_pack_dir: Path) -> Dict:
    """Load existing concept definitions from layer files."""
    concepts = {}

    hierarchy_dir = concept_pack_dir / 'hierarchy'
    if not hierarchy_dir.exists():
        print(f"Warning: Hierarchy dir not found: {hierarchy_dir}")
        return concepts

    for layer in range(7):
        layer_file = hierarchy_dir / f'layer{layer}.json'
        if layer_file.exists():
            with open(layer_file) as f:
                data = json.load(f)
            for c in data.get('concepts', []):
                concepts[c['sumo_term']] = c

    print(f"Loaded {len(concepts)} existing concepts")
    return concepts


def generate_meld_description(
    client: anthropic.Anthropic,
    meld_op: Dict,
    existing_concepts: Dict,
    model: str = "claude-sonnet-4-20250514"
) -> Optional[Dict]:
    """Generate description for a single meld."""

    prompt = build_meld_prompt(meld_op, existing_concepts)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        content = response.content[0].text.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        # Add metadata
        result['parent'] = meld_op['new_parent']
        result['children'] = meld_op['source_concepts']
        result['layer'] = meld_op['target_layer']
        result['generated'] = True

        return result

    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {meld_op['target']}: {e}")
        print(f"  Response was: {content[:500]}...")
        return None
    except Exception as e:
        print(f"  Error for {meld_op['target']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate meld descriptions via API')
    parser.add_argument('--meld-ops', type=str, required=True,
                        help='Path to meld_operations.json')
    parser.add_argument('--concept-pack', type=str, default='concept_packs/sumo-wordnet-v4',
                        help='Path to concept pack directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file for generated descriptions')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Process this many melds before saving checkpoint')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start from this index (for resuming)')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Anthropic model to use')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without calling API')

    args = parser.parse_args()

    # Load meld operations
    print(f"Loading meld operations from: {args.meld_ops}")
    with open(args.meld_ops) as f:
        meld_data = json.load(f)

    create_melds = meld_data['operations']['create_meld']
    print(f"Found {len(create_melds)} melds to generate")

    # Load existing concepts for context
    concept_pack_dir = Path(args.concept_pack)
    existing_concepts = load_existing_concepts(concept_pack_dir)

    # Load any existing output for resuming
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results.get('melds', {}))} existing results")
    else:
        results = {
            'version': 'v5',
            'model': args.model,
            'melds': {}
        }

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for i, meld_op in enumerate(create_melds[args.start_index:args.start_index+5], args.start_index):
            print(f"\n[{i}] {meld_op['target']}")
            print(f"    Parent: {meld_op['new_parent']}")
            print(f"    Layer: {meld_op['target_layer']}")
            print(f"    Children: {meld_op['source_concepts'][:5]}...")
        return

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Process melds
    print(f"\nStarting from index {args.start_index}")

    for i, meld_op in enumerate(create_melds[args.start_index:], args.start_index):
        target = meld_op['target']

        # Skip if already generated
        if target in results['melds']:
            print(f"[{i}/{len(create_melds)}] {target} - already done, skipping")
            continue

        print(f"[{i}/{len(create_melds)}] Generating: {target}")

        result = generate_meld_description(client, meld_op, existing_concepts, args.model)

        if result:
            results['melds'][target] = result
            print(f"  ✓ Generated: {result['definition'][:60]}...")
        else:
            print(f"  ✗ Failed to generate")

        # Rate limiting
        time.sleep(0.5)

        # Checkpoint save
        if (i + 1) % args.batch_size == 0:
            print(f"\nSaving checkpoint ({len(results['melds'])} melds)...")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Final save
    print(f"\nSaving final results ({len(results['melds'])} melds)...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total melds generated: {len(results['melds'])}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
