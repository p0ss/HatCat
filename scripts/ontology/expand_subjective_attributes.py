#!/usr/bin/env python3
"""
Expand Subjective Attribute Concepts Using Claude API

Uses Claude 4.5 to propose finer-grained child concepts for the coarse-grained
SubjectiveWeak/StrongPositive/NegativeAttribute concepts in SUMO.

Similar to the simplex review approach, but focused on creating a richer
taxonomy for subjective evaluative language.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic


# The four main subjective attribute concepts to expand
SUBJECTIVE_ATTRIBUTES = [
    "SubjectiveWeakPositiveAttribute",
    "SubjectiveWeakNegativeAttribute",
    "SubjectiveStrongPositiveAttribute",
    "SubjectiveStrongNegativeAttribute",
]


def load_sumo_documentation(concept: str) -> str:
    """Load SUMO documentation for a concept from the KIF files."""
    kif_files = Path("data/concept_graph/sumo_source").glob("*.kif")

    for kif_file in kif_files:
        with open(kif_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find documentation for this concept
        if f"(documentation {concept}" in content:
            # Extract the documentation string
            start = content.find(f"(documentation {concept}")
            if start != -1:
                end = content.find(")", start)
                # Find the actual doc string (between quotes)
                doc_start = content.find('"', start)
                doc_end = content.find('"', doc_start + 1)
                if doc_start != -1 and doc_end != -1:
                    return content[doc_start+1:doc_end]

    return f"No documentation found for {concept}"


def create_expansion_prompt(concept: str, doc: str) -> str:
    """Create a prompt asking Claude to propose child concepts."""

    return f"""You are helping to expand a coarse-grained ontological concept into more specific child concepts for use in AI safety research.

**Concept to Expand**: {concept}

**SUMO Documentation**:
{doc}

**Context**: This concept is currently a terminal node (no children) in the SUMO ontology. It's used as a catch-all for subjective evaluative language that lacks objective criteria. However, for detecting manipulation and deceptive language patterns in LLMs, we need finer-grained distinctions.

**Your Task**: Propose 8-12 child concepts that would be useful for distinguishing different types of manipulative or evaluative language. Consider:

1. **Manipulation tactics**: flattery, love-bombing, subtle negging, gaslighting language
2. **Intensity levels**: mild praise vs excessive praise, gentle criticism vs harsh criticism
3. **Authenticity markers**: genuine appreciation vs manipulative flattery
4. **Context sensitivity**: appropriate praise vs inappropriate praise (context mismatch)
5. **Semantic clusters**: appearance-based, ability-based, character-based, outcome-based

For each proposed child concept, provide:

1. **Name**: Clear, descriptive name (CamelCase, following SUMO conventions)
2. **Definition**: 1-2 sentence definition explaining what it captures
3. **Example words/phrases**: 5-10 representative terms from WordNet that would map to this concept
4. **Manipulation relevance**: Brief note on why this is useful for detecting manipulation (if applicable)

Return your response as a JSON array of objects with these fields: "name", "definition", "examples", "manipulation_relevance"

Focus on concepts that would help distinguish:
- Genuine positive language from flattery/manipulation
- Constructive feedback from destructive criticism
- Appropriate enthusiasm from excessive/suspicious praise
- Honest negative assessment from bullying/gaslighting language"""


def query_claude(prompt: str, api_key: str, model: str = "claude-opus-4-20250514") -> str:
    """Query Claude API with the expansion prompt."""

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return message.content[0].text


def parse_claude_response(response: str) -> List[Dict]:
    """Parse Claude's JSON response into structured data."""

    # Try to extract JSON from the response
    # Claude might wrap it in markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response was: {response[:500]}...")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Expand subjective attribute concepts using Claude API"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-20250514',
        help='Claude model to use (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--concepts',
        type=str,
        nargs='+',
        default=SUBJECTIVE_ATTRIBUTES,
        help='Specific concepts to expand (default: all subjective attributes)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/subjective_attribute_expansion.json'),
        help='Output JSON file'
    )

    args = parser.parse_args()

    # Get API key
    import os
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: No API key provided. Use --api-key or set ANTHROPIC_API_KEY")
        return 1

    print("="*80)
    print("SUBJECTIVE ATTRIBUTE CONCEPT EXPANSION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Concepts to expand: {', '.join(args.concepts)}")
    print(f"Output: {args.output}")
    print("="*80)

    results = {}

    for concept in args.concepts:
        print(f"\nProcessing: {concept}")
        print("-" * 80)

        # Load SUMO documentation
        doc = load_sumo_documentation(concept)
        print(f"Documentation: {doc[:200]}...")

        # Create prompt
        prompt = create_expansion_prompt(concept, doc)

        # Query Claude
        print(f"Querying {args.model}...")
        response = query_claude(prompt, api_key, args.model)

        # Parse response
        children = parse_claude_response(response)

        if children:
            print(f"✓ Proposed {len(children)} child concepts:")
            for child in children[:3]:  # Show first 3
                print(f"  - {child.get('name', 'Unknown')}: {child.get('definition', 'No definition')[:100]}...")
            if len(children) > 3:
                print(f"  ... and {len(children) - 3} more")
        else:
            print("✗ No valid child concepts parsed")

        results[concept] = {
            'documentation': doc,
            'proposed_children': children,
            'raw_response': response
        }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output}")
    print(f"\nTotal concepts expanded: {len(results)}")
    print(f"Total child concepts proposed: {sum(len(r['proposed_children']) for r in results.values())}")

    return 0


if __name__ == '__main__':
    exit(main())
