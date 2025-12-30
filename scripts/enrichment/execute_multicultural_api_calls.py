#!/usr/bin/env python3
"""
Execute all multicultural enrichment API calls automatically.

This script loads the enrichment plan and executes all 77 API requests,
saving responses to individual JSON files.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
import os

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
ENRICHMENT_PLAN_PATH = PROJECT_ROOT / "data" / "balance_enrichment_multicultural.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enrichment_responses"

# Check for Anthropic API key
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed!")
    print("Install it with: pip install anthropic")
    sys.exit(1)


def call_claude_api(prompt: str, client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API with the given prompt."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=8000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"   ERROR calling API: {e}")
        return None


def parse_json_response(response_text: str) -> list:
    """Extract JSON array from response text."""
    # Try to find JSON array in response
    start = response_text.find('[')
    end = response_text.rfind(']') + 1

    if start == -1 or end == 0:
        print("   WARNING: No JSON array found in response")
        return None

    json_str = response_text[start:end]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"   WARNING: Failed to parse JSON: {e}")
        return None


def main():
    print("=" * 80)
    print("AUTOMATED MULTICULTURAL ENRICHMENT API EXECUTOR")
    print("=" * 80)

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=API_KEY)
    print("\n✓ Anthropic API client initialized")

    # Load enrichment plan
    print(f"\nLoading enrichment plan from {ENRICHMENT_PLAN_PATH}...")
    with open(ENRICHMENT_PLAN_PATH) as f:
        plan = json.load(f)

    metadata = plan['metadata']
    enrichment_plan = plan['enrichment_plan']

    print(f"✓ Loaded plan: {metadata['total_api_requests']} requests")
    print(f"  - Multicultural: {metadata['multicultural_overlaps']} synsets")
    print(f"  - Balance: {metadata['balance_overlaps']} synsets")
    print(f"  - Total: {metadata['total_new_overlaps_needed']} synsets")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Execute all requests
    print(f"\n{'=' * 80}")
    print("EXECUTING API REQUESTS")
    print("=" * 80)

    request_num = 0
    successful = 0
    failed = 0
    total_synsets = 0

    start_time = time.time()

    for simplex, pole_plans in enrichment_plan.items():
        print(f"\n{simplex}:")

        for pole, pole_plan in pole_plans.items():
            requests = pole_plan['requests']

            for req_idx, request in enumerate(requests):
                request_num += 1

                # Generate output filename (sanitize simplex name for filesystem)
                req_type = request['type']
                batch_num = req_idx + 1
                simplex_safe = simplex.replace('/', '_')  # Replace / with _ for filesystem
                output_file = OUTPUT_DIR / f"{simplex_safe}_{pole}_{req_type}_batch{batch_num}.json"

                # Skip if already exists
                if output_file.exists():
                    print(f"  [{request_num:2d}/{metadata['total_api_requests']}] {pole:8} {req_type:12} batch {batch_num} - SKIPPED (exists)")
                    successful += 1
                    continue

                print(f"  [{request_num:2d}/{metadata['total_api_requests']}] {pole:8} {req_type:12} batch {batch_num} ({request['count']:2d} synsets)...", end=' ', flush=True)

                # Call API
                response_text = call_claude_api(request['prompt'], client)

                if response_text is None:
                    print("FAILED")
                    failed += 1
                    continue

                # Parse response
                synsets = parse_json_response(response_text)

                if synsets is None:
                    print("FAILED (parse error)")
                    failed += 1
                    # Save raw response for debugging
                    with open(output_file.with_suffix('.txt'), 'w') as f:
                        f.write(response_text)
                    continue

                # Save response
                response_data = {
                    'metadata': {
                        'simplex': simplex,
                        'pole': pole,
                        'type': req_type,
                        'batch': request['batch'],
                        'requested_count': request['count'],
                        'actual_count': len(synsets),
                        'timestamp': datetime.now().isoformat()
                    },
                    'synsets': synsets
                }

                with open(output_file, 'w') as f:
                    json.dump(response_data, f, indent=2)

                print(f"OK ({len(synsets)} synsets)")
                successful += 1
                total_synsets += len(synsets)

                # Rate limiting - be nice to the API
                time.sleep(1)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 80}")
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nTotal requests: {metadata['total_api_requests']}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total synsets generated: {total_synsets}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nResponses saved to: {OUTPUT_DIR}/")

    if failed > 0:
        print(f"\nWARNING: {failed} requests failed. Check output directory for details.")

    print("\nNext step:")
    print("  python scripts/merge_enrichment_responses.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
