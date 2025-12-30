#!/usr/bin/env python3
"""
AI Risk Assessment Script - Interpretability Tool Impact

Queries the Anthropic API to assess how an open-source interpretability tool
would affect each AI risk in the AI Risk Repository database.

Usage:
    python scripts/tools/assess_ai_risks_interpretability.py \
        --input "docs/risk assessment/he AI Risk Repository V3_26_03_2025 AI Risk Database v3.tsv" \
        --output "docs/risk assessment/ai_risks_with_interpretability_assessment.tsv" \
        --api-key $ANTHROPIC_API_KEY \
        --start-row 0 \
        --batch-size 50

The script:
1. Reads the TSV file containing AI risks (column K = Description)
2. For each risk, queries Claude to assess interpretability tool impact
3. Adds three new columns after column K:
   - Interp_Likelihood (1-5 scale: how tool affects probability of risk)
   - Interp_Magnitude (1-5 scale: how tool affects severity if risk occurs)
   - Interp_Reason (single sentence explaining both assessments)
   Scale: 1=significantly reduces, 2=somewhat reduces, 3=neutral, 4=somewhat increases, 5=significantly increases
4. Saves progress incrementally and supports resuming
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


SYSTEM_PROMPT = """You are an AI safety researcher assessing how OPEN-SOURCE vs CLOSED-SOURCE interpretability tools would affect AI risks.

IMPORTANT CONTEXT: The interpretability tool ONLY works on models where you have access to the weights. It CANNOT be used to attack or analyze closed-source/proprietary models externally. This means:
- Open-weight model developers and users can use it on their own models
- Closed-source labs can use it internally on their own models
- Adversaries CANNOT use it against production APIs of closed-source models

The question is: given this constraint, would making such tools OPEN-SOURCE (publicly available) vs CLOSED-SOURCE (restricted to select organizations) affect this risk?

You must assess TWO dimensions:
1. LIKELIHOOD: How would OPEN-SOURCE availability (vs closed-source) affect the probability of this risk occurring?
2. MAGNITUDE: How would OPEN-SOURCE availability (vs closed-source) affect the severity/impact if this risk does occur?

Respond with ONLY a JSON object in this exact format:
{"likelihood": N, "magnitude": N, "reason": "Single sentence explanation covering both"}

Score scale (for both likelihood and magnitude):
1 = Open-source significantly reduces risk compared to closed-source
2 = Open-source somewhat reduces risk compared to closed-source
3 = Neutral / no clear difference between open vs closed
4 = Open-source somewhat increases risk compared to closed-source
5 = Open-source significantly increases risk compared to closed-source

Keep your reason to ONE concise sentence explaining the key factors."""

USER_PROMPT_TEMPLATE = """Assess this AI risk:

Risk: {risk_description}

Remember: The interpretability tool ONLY works on models where you have the weights. It cannot be used to attack closed-source API models externally.

Compare: if such tools are OPEN-SOURCE (publicly available) vs CLOSED-SOURCE (restricted to select organizations), how would this affect:
1. The LIKELIHOOD of this risk occurring?
2. The MAGNITUDE (severity/impact) if it does occur?

Respond with JSON only: {{"likelihood": N, "magnitude": N, "reason": "explanation"}}"""


def parse_response(response_text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Parse Claude's response to extract likelihood, magnitude, and reason."""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            likelihood = int(data.get('likelihood', 0))
            magnitude = int(data.get('magnitude', 0))
            reason = str(data.get('reason', ''))
            if 1 <= likelihood <= 5 and 1 <= magnitude <= 5:
                return likelihood, magnitude, reason
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: try to extract scores from text
    scores = re.findall(r'\b([1-5])\b', response_text)
    if len(scores) >= 2:
        return int(scores[0]), int(scores[1]), response_text[:200]

    return None, None, None


def assess_risk(client: anthropic.Anthropic, risk_description: str, model: str = "claude-sonnet-4-20250514") -> Tuple[int, int, str]:
    """Query Claude API to assess a single risk."""
    if not risk_description or len(risk_description.strip()) < 10:
        return 3, 3, "Insufficient risk description to assess"

    try:
        message = client.messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(risk_description=risk_description)}
            ]
        )

        response_text = message.content[0].text
        likelihood, magnitude, reason = parse_response(response_text)

        if likelihood is None or magnitude is None:
            return 3, 3, f"Could not parse response: {response_text[:100]}"

        return likelihood, magnitude, reason

    except anthropic.RateLimitError:
        time.sleep(60)  # Wait and retry
        return assess_risk(client, risk_description, model)
    except Exception as e:
        return 3, 3, f"API error: {str(e)[:100]}"


def load_progress(progress_file: Path) -> dict:
    """Load progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_rows": [], "results": {}}


def save_progress(progress_file: Path, progress: dict):
    """Save progress to checkpoint file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Assess AI risks for interpretability tool impact"
    )
    parser.add_argument('--input', required=True,
                        help='Input TSV file path')
    parser.add_argument('--output', required=True,
                        help='Output TSV file path')
    parser.add_argument('--api-key', default=os.environ.get('ANTHROPIC_API_KEY'),
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--model', default='claude-sonnet-4-20250514',
                        help='Claude model to use (default: claude-sonnet-4-20250514)')
    parser.add_argument('--risk-column', type=int, default=10,
                        help='Column index for risk description (0-indexed, default: 10 = column K)')
    parser.add_argument('--header-row', type=int, default=2,
                        help='Row index for header (0-indexed, default: 2)')
    parser.add_argument('--start-row', type=int, default=0,
                        help='Start from this data row (0-indexed from first data row)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum rows to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Save progress every N rows (default: 50)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be processed without making API calls')

    args = parser.parse_args()

    if not args.api_key:
        print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set environment variable.")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)
    progress_file = output_path.with_suffix('.progress.json')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load input TSV
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)

    print(f"Loaded {len(rows)} rows")

    # Identify header and data rows
    header_row_idx = args.header_row
    data_start_idx = header_row_idx + 1
    risk_col = args.risk_column

    # Find where to insert new columns (after column K)
    insert_col = risk_col + 1

    # Load progress
    progress = load_progress(progress_file)

    # Initialize client
    client = anthropic.Anthropic(api_key=args.api_key)

    # Add header columns if not already present
    if len(rows) > header_row_idx:
        header = rows[header_row_idx]
        if len(header) <= insert_col or header[insert_col] != 'Interp_Likelihood':
            # Insert new columns into header
            header.insert(insert_col, 'Interp_Likelihood')
            header.insert(insert_col + 1, 'Interp_Magnitude')
            header.insert(insert_col + 2, 'Interp_Reason')
            rows[header_row_idx] = header

            # Also insert empty columns into pre-header rows
            for i in range(header_row_idx):
                rows[i].insert(insert_col, '')
                rows[i].insert(insert_col + 1, '')
                rows[i].insert(insert_col + 2, '')

    # Process data rows
    data_rows = rows[data_start_idx:]
    total_to_process = len(data_rows)
    if args.max_rows:
        total_to_process = min(total_to_process, args.start_row + args.max_rows)

    print(f"Processing rows {args.start_row} to {total_to_process - 1}...")
    print(f"Risk column: {risk_col} (column {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[risk_col] if risk_col < 26 else risk_col})")

    processed = 0
    skipped = 0

    for i in range(args.start_row, total_to_process):
        row_idx = data_start_idx + i
        row = rows[row_idx]

        # Ensure row has enough columns
        while len(row) <= insert_col + 2:
            row.append('')

        # Check if already processed
        if str(i) in progress['results']:
            result = progress['results'][str(i)]
            row[insert_col] = str(result['likelihood'])
            row[insert_col + 1] = str(result['magnitude'])
            row[insert_col + 2] = result['reason']
            rows[row_idx] = row
            skipped += 1
            continue

        # Get risk description
        if len(row) > risk_col:
            risk_desc = row[risk_col].strip()
        else:
            risk_desc = ''

        # Skip blank/empty rows
        if not risk_desc or len(risk_desc) < 10:
            print(f"[{i+1}/{total_to_process}] SKIP (empty/blank row)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"Row {i}: Would assess: {risk_desc[:100]}...")
            continue

        # Assess risk
        likelihood, magnitude, reason = assess_risk(client, risk_desc, args.model)

        # Update row
        row[insert_col] = str(likelihood)
        row[insert_col + 1] = str(magnitude)
        row[insert_col + 2] = reason
        rows[row_idx] = row

        # Track progress
        progress['completed_rows'].append(i)
        progress['results'][str(i)] = {'likelihood': likelihood, 'magnitude': magnitude, 'reason': reason}
        processed += 1

        # Print progress
        print(f"[{i+1}/{total_to_process}] L:{likelihood} M:{magnitude} - {risk_desc[:50]}...")

        # Save checkpoint
        if processed % args.batch_size == 0:
            save_progress(progress_file, progress)
            print(f"  Checkpoint saved ({processed} processed)")

        # Rate limiting
        time.sleep(args.delay)

    if args.dry_run:
        print(f"\nDry run complete. Would process {total_to_process - args.start_row} rows.")
        return

    # Save final progress
    save_progress(progress_file, progress)

    # Write output TSV
    print(f"\nWriting output to {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(rows)

    print(f"\nComplete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Output: {output_path}")
    print(f"  Progress file: {progress_file}")


if __name__ == '__main__':
    main()
