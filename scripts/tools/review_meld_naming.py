#!/usr/bin/env python3
"""
LLM Review Pass for Meld Term Naming

Reviews auto-generated meld proposals and improves term names to be:
- Semantically meaningful (not just WordNet lemmas)
- Clear when seen in isolation
- Consistent in formatting (PascalCase)
- Not tautological (Parent_Parent)
- Not replicating ancestor concepts

Usage:
    python scripts/review_meld_naming.py \
        --input melds/pending/remediation_meld_20251206_113639.json \
        --output melds/pending/remediation_meld_reviewed.json \
        --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import anthropic

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review and improve meld term naming using LLM"
    )
    parser.add_argument('--input', required=True,
                        help='Input meld JSON file')
    parser.add_argument('--output', required=True,
                        help='Output meld JSON file with improved names')
    parser.add_argument('--model', default='claude-sonnet-4-20250514',
                        help='Anthropic model to use (default: claude-sonnet-4-20250514)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of operations to review per API call')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without making API calls')
    return parser.parse_args()


SYSTEM_PROMPT = """You are reviewing auto-generated concept naming proposals for an ontology.
Your task is to improve the proposed term names to be clearer and more semantically meaningful.

Context: These terms will be shown to humans interpreting AI model activations. A term like
"RadiatingNuclear" might fire when an AI discusses nuclear power OR nuclear weapons - we want
to split it into clearer sub-concepts so operators can distinguish intent.

Guidelines for naming:
1. Use PascalCase consistently (e.g., NuclearPower, NuclearWeapon)
2. Names should be meaningful in isolation - someone seeing just the term should understand it
3. Avoid tautological names (e.g., Brain_Brain should become something like BiologicalBrain vs BrainMetaphor)
4. Avoid POS-based suffixes (_Adv_All, _Noun) - use semantic meaning instead
5. Don't replicate ancestor concepts (Container shouldn't have a child called CreatedThings if that's already a grandparent)
6. Keep names concise but clear (1-3 words ideally)
7. Preserve the semantic distinction the split is trying to capture

For each operation, you'll see:
- The parent concept being split
- The proposed child terms with their synsets and definitions
- Your job is to propose better term names that capture the semantic distinction

Respond with a JSON object mapping old term names to new term names.
Only include terms that need renaming - omit terms that are already good.
If a split doesn't make semantic sense (e.g., the groups aren't meaningfully different),
set the new name to null to flag it for human review."""


def format_operation_for_review(op: Dict) -> str:
    """Format a single operation for LLM review."""
    lines = []
    lines.append(f"## Split: {op['target_concept']}")
    lines.append(f"Rationale: {op['rationale']}")
    lines.append("")

    for split in op.get('split_into', []):
        lines.append(f"### Proposed term: {split['term']}")
        if split.get('representative_label'):
            lines.append(f"Representative: {split['representative_label']}")

        # Show synset details
        details = split.get('synsets_detail', [])
        if details:
            lines.append("Synsets:")
            for d in details[:5]:  # Limit to first 5
                lemma = d.get('lemma', '?')
                defn = d.get('definition', '')[:80]
                lines.append(f"  - {lemma}: {defn}")
            if len(details) > 5:
                lines.append(f"  ... and {len(details) - 5} more")
        lines.append("")

    return "\n".join(lines)


def format_bucket_operation_for_review(op: Dict) -> str:
    """Format a bucket operation for LLM review."""
    lines = []
    lines.append(f"## Bucket: {op['target_concept']}")
    lines.append(f"Rationale: {op['rationale']}")
    lines.append("")

    for bucket in op.get('bucket_suggestions', []):
        lines.append(f"### Proposed bucket: {bucket['bucket_name']}")
        members = bucket.get('members', [])
        lines.append(f"Members ({len(members)}): {', '.join(members[:10])}")
        if len(members) > 10:
            lines.append(f"  ... and {len(members) - 10} more")
        lines.append("")

    return "\n".join(lines)


def review_batch(client: anthropic.Anthropic, operations: List[Dict], model: str) -> Dict[str, Optional[str]]:
    """Review a batch of operations and return name mappings."""

    # Format all operations
    formatted = []
    for op in operations:
        if op.get('split_into'):
            formatted.append(format_operation_for_review(op))
        elif op.get('bucket_suggestions'):
            formatted.append(format_bucket_operation_for_review(op))

    user_message = """Review these concept split/bucket proposals and suggest better term names.

""" + "\n---\n".join(formatted) + """

Respond with a JSON object mapping old term names to improved names.
Example: {"Brain_Brain": "BiologicalBrain", "RadiatingSound_Adv_All": "AcousticEmission", "BadName": null}

Only include terms that need changes. Set value to null if the split doesn't make semantic sense."""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    # Extract JSON from response
    response_text = response.content[0].text

    # Try to find JSON in the response
    try:
        # Look for JSON block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
        print(f"Response was: {response_text[:500]}...")
        return {}


def apply_name_mappings(meld_data: Dict, mappings: Dict[str, Optional[str]]) -> Dict:
    """Apply name mappings to the meld data."""
    flagged_for_review = []

    for op in meld_data.get('structural_operations', []):
        # Handle split operations
        for split in op.get('split_into', []):
            old_name = split.get('term')
            if old_name in mappings:
                new_name = mappings[old_name]
                if new_name is None:
                    # Flag for human review
                    split['needs_review'] = True
                    split['review_reason'] = "LLM flagged as semantically unclear split"
                    flagged_for_review.append(old_name)
                else:
                    split['original_term'] = old_name
                    split['term'] = new_name

        # Handle bucket operations
        for bucket in op.get('bucket_suggestions', []):
            old_name = bucket.get('bucket_name')
            if old_name in mappings:
                new_name = mappings[old_name]
                if new_name is None:
                    bucket['needs_review'] = True
                    bucket['review_reason'] = "LLM flagged as semantically unclear bucket"
                    flagged_for_review.append(old_name)
                else:
                    bucket['original_bucket_name'] = old_name
                    bucket['bucket_name'] = new_name

    if flagged_for_review:
        print(f"  Flagged {len(flagged_for_review)} terms for human review")

    return meld_data


def main():
    args = parse_args()

    # Load input meld
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        meld_data = json.load(f)

    operations = meld_data.get('structural_operations', [])
    print(f"Loaded {len(operations)} operations from {input_path}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for i, op in enumerate(operations[:3]):
            print(f"\nOperation {i+1}:")
            if op.get('split_into'):
                print(format_operation_for_review(op))
            elif op.get('bucket_suggestions'):
                print(format_bucket_operation_for_review(op))
        print(f"\n... and {len(operations) - 3} more operations")
        return

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Process in batches
    all_mappings = {}
    batch_size = args.batch_size

    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(operations) + batch_size - 1) // batch_size

        print(f"Reviewing batch {batch_num}/{total_batches} ({len(batch)} operations)...")

        mappings = review_batch(client, batch, args.model)
        all_mappings.update(mappings)

        renamed = sum(1 for v in mappings.values() if v is not None)
        flagged = sum(1 for v in mappings.values() if v is None)
        print(f"  Renamed: {renamed}, Flagged for review: {flagged}")

    # Apply all mappings
    print(f"\nApplying {len(all_mappings)} name changes...")
    meld_data = apply_name_mappings(meld_data, all_mappings)

    # Add review metadata
    meld_data['metadata']['llm_review'] = {
        'model': args.model,
        'total_renames': sum(1 for v in all_mappings.values() if v is not None),
        'flagged_for_review': sum(1 for v in all_mappings.values() if v is None),
        'mappings': all_mappings
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(meld_data, f, indent=2)

    print(f"\nWrote reviewed meld to: {output_path}")
    print(f"Total renames: {sum(1 for v in all_mappings.values() if v is not None)}")
    print(f"Flagged for human review: {sum(1 for v in all_mappings.values() if v is None)}")


if __name__ == "__main__":
    main()
