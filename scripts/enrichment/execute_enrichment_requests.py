#!/usr/bin/env python3
"""
Execute API enrichment requests to generate balanced simplex overlap synsets.

This script reads the enrichment plan from balance_enrichment_requests.json
and executes API calls to generate culturally-diverse synthetic synsets.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
ENRICHMENT_PLAN_PATH = PROJECT_ROOT / "data" / "balance_enrichment_requests.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enrichment_responses"

def main():
    print("=" * 80)
    print("SIMPLEX OVERLAP ENRICHMENT - API REQUEST EXECUTOR")
    print("=" * 80)

    # Load enrichment plan
    print(f"\n1. Loading enrichment plan from {ENRICHMENT_PLAN_PATH}...")
    with open(ENRICHMENT_PLAN_PATH) as f:
        plan = json.load(f)

    metadata = plan['metadata']
    enrichment_plan = plan['enrichment_plan']

    print(f"   Total new overlaps needed: {metadata['total_new_overlaps_needed']}")
    print(f"   Total API requests: {metadata['total_api_requests']}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Display all requests
    print(f"\n2. API Requests to Execute:\n")

    request_num = 0
    for simplex, pole_plans in enrichment_plan.items():
        print(f"\n{simplex}:")
        for pole, pole_plan in pole_plans.items():
            requests = pole_plan['requests']
            for req in requests:
                request_num += 1
                print(f"  [{request_num:2d}] {pole:8} - Batch {req['batch']:3} ({req['count']:2d} synsets)")

    print(f"\n{'=' * 80}")
    print("INSTRUCTIONS FOR MANUAL EXECUTION")
    print("=" * 80)

    print(f"""
This script currently displays the API requests that need to be executed.
You'll need to manually call your LLM API (e.g., Claude, GPT-4) with each prompt.

To execute the requests:

1. The prompts are in: {ENRICHMENT_PLAN_PATH}

2. For each request, find the 'prompt' field in the JSON structure:
   enrichment_plan -> [simplex] -> [pole] -> requests -> [index] -> prompt

3. Send the prompt to your API and save the response as JSON

4. Save responses to: {OUTPUT_DIR}/
   Filename format: [simplex]_[pole]_batch[N].json

5. Example filename: taste_development_neutral_batch1.json

After collecting all responses, run the merge script to integrate them:
   python scripts/merge_enrichment_responses.py

Alternatively, integrate with your preferred API client:
- OpenAI: Use openai.ChatCompletion.create()
- Anthropic: Use anthropic.Anthropic().messages.create()
- Local: Use your inference endpoint

Would you like me to generate a specific API integration script?
""")

    # Generate example code for first request
    print("\n" + "=" * 80)
    print("EXAMPLE: First API Request")
    print("=" * 80)

    # Get first request
    first_simplex = list(enrichment_plan.keys())[0]
    first_pole = list(enrichment_plan[first_simplex].keys())[0]
    first_request = enrichment_plan[first_simplex][first_pole]['requests'][0]

    print(f"\nSimplex: {first_request['simplex']}")
    print(f"Pole: {first_request['pole']}")
    print(f"Batch: {first_request['batch']}")
    print(f"Count: {first_request['count']} synsets")
    print(f"\nPrompt:")
    print("-" * 80)
    print(first_request['prompt'])
    print("-" * 80)

    # Generate request summary JSON for easy reference
    summary_file = OUTPUT_DIR / "request_summary.json"
    summary = []

    request_num = 0
    for simplex, pole_plans in enrichment_plan.items():
        for pole, pole_plan in pole_plans.items():
            for idx, req in enumerate(pole_plan['requests']):
                request_num += 1
                summary.append({
                    'request_id': request_num,
                    'simplex': req['simplex'],
                    'pole': req['pole'],
                    'batch': req['batch'],
                    'count': req['count'],
                    'output_filename': f"{req['simplex']}_{req['pole']}_batch{idx+1}.json",
                    'status': 'pending'
                })

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Request summary saved to: {summary_file}")
    print(f"✓ Use this to track which requests you've completed")

    print(f"\n{'=' * 80}")
    print(f"Next steps:")
    print(f"1. Execute the {metadata['total_api_requests']} API requests")
    print(f"2. Save responses to {OUTPUT_DIR}/")
    print(f"3. Run: python scripts/merge_enrichment_responses.py")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
