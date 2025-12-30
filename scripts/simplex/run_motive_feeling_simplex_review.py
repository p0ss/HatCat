#!/usr/bin/env python3
"""
Run simplex agentic review specifically for all 470 motive/feeling concepts.

This is a high-priority focused review to identify which emotional/motivational
concepts have natural three-pole progressions for S-tier treatment.
"""

import json
import subprocess
from pathlib import Path

# Load the 470 motive/feeling concepts
project_root = Path(__file__).parent.parent
input_file = project_root / "results" / "motive_feeling_concepts_for_simplex_review.json"

with open(input_file) as f:
    concepts = json.load(f)

print(f"Loaded {len(concepts)} motive/feeling concepts")

# Add dummy scores field (required by simplex review script)
# Use high scores to ensure they're all prioritized
for concept in concepts:
    concept['scores'] = {
        'total': 100.0,  # Max priority
        'external_monitoring': 50.0,
        'internal_awareness': 50.0
    }

# Create a tier2-compatible format
tier2_format = {
    'metadata': {
        'source': 'motive_feeling_focused_review',
        'total_concepts': len(concepts),
        'description': 'All 470 noun.motive and noun.feeling concepts from WordNet'
    },
    'all_scored': concepts
}

# Save formatted version
formatted_file = project_root / "results" / "motive_feeling_tier2_format.json"
with open(formatted_file, 'w') as f:
    json.dump(tier2_format, f, indent=2)

print(f"Saved formatted concepts to {formatted_file}")

# Run the simplex agentic review
print("\n" + "="*80)
print("Starting simplex agentic review for 470 motive/feeling concepts")
print("="*80)

# The review script will use the formatted file
# We'll need to pass it via environment or modify the script
print("\nTo run the review, execute:")
print(f"  TIER2_FILE={formatted_file} python scripts/run_simplex_agentic_review.py 470")
print("\nOr we can invoke it directly...")

# Actually, let's just call the async function directly
import sys
import asyncio
sys.path.insert(0, str(project_root / "scripts"))

# Import the main async function
exec(open(project_root / "scripts" / "run_simplex_agentic_review.py").read())
