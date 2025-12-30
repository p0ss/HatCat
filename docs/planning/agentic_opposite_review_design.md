# Agentic Opposite Review Design

## Overview

An agentic review process to identify semantic opposites for all SUMO concepts in layers 0-5, using Claude API with multi-step reasoning, validation, and self-correction.

**Goal:** For each concept, identify the best opposite to use as the negative centroid anchor.

**Output:** A validated mapping of concepts to their opposites, with confidence scores and fallback strategies.

## Architecture

### Agent Workflow

```
For each concept:
  1. Analyze concept (definition, synsets, category)
  2. Generate opposite candidates (antonyms, semantic opposites, complements)
  3. Validate candidates (check if they exist in WordNet/SUMO)
  4. Score candidates (relevance, opposition strength, availability)
  5. Select best candidate or flag for manual review
  6. Generate reasoning and confidence score
```

### Multi-Step Reasoning

**Step 1: Concept Analysis**
```
Input: SUMO concept with definition and synsets
Output: Semantic classification
  - Has clear antonym? (yes/no)
  - Semantic category (physical, abstract, process, attribute, etc.)
  - Opposition type (binary, spectrum, complementary, taxonomic)
```

**Step 2: Candidate Generation**
```
Input: Semantic classification
Output: Ranked list of opposite candidates
  Priority 1: Direct antonyms from WordNet
  Priority 2: Semantic opposites (physical/abstract, good/evil)
  Priority 3: Complementary concepts (bird/mammal, solid/liquid)
  Priority 4: Category-level opposites (process/object, attribute/entity)
  Priority 5: Intelligent distant concepts (maximally different domains)
```

**Step 3: Validation**
```
Input: Candidate list
Output: Validated candidates
  - Check if candidate exists in WordNet
  - Check if candidate exists in our SUMO layers
  - If not in layers: flag for potential addition
  - If doesn't exist: remove from candidates
```

**Step 4: Scoring**
```
Input: Validated candidates
Output: Scored candidates
  Scoring criteria:
    - Semantic opposition strength (0-10)
    - Availability in current layers (exists=10, missing=5, doesn't exist=0)
    - Steering utility (how useful for AI safety monitoring)
    - Confidence (0-10, based on reasoning clarity)
```

**Step 5: Selection**
```
Input: Scored candidates
Output: Selected opposite + reasoning + flags
  - If top score > 7: Accept automatically
  - If top score 4-7: Flag for human review
  - If top score < 4: Flag as "no suitable opposite"
  - Generate explanation for selection
```

## Prompt Design

### Main Prompt Template

```python
AGENTIC_REVIEW_PROMPT = """You are a semantic ontology expert analyzing concept opposites for an AI safety monitoring system.

# Task
Analyze the SUMO concept below and identify the best semantic opposite to use as a negative centroid anchor for steering.

# Concept
- **SUMO Term**: {sumo_term}
- **Definition**: {definition}
- **Layer**: {layer}
- **Synsets**: {synsets}
- **Category Children**: {category_children}

# Instructions

## Step 1: Concept Analysis
Analyze this concept and determine:
1. Does it have a clear antonym? (yes/no)
2. What semantic category does it belong to? (physical, abstract, process, attribute, state, entity, relation, event)
3. What type of opposition would work best? (binary, spectrum, complementary, taxonomic, category-level, distant)

## Step 2: Generate Candidates
Generate up to 5 opposite candidates in priority order:

**Priority 1 - Direct Antonyms:**
If the concept has WordNet antonyms, list them.

**Priority 2 - Semantic Opposites:**
Common pairs like:
- Physical ↔ Abstract
- Good ↔ Evil
- Voluntary ↔ Involuntary
- Animate ↔ Inanimate
- Subjective ↔ Objective

**Priority 3 - Complementary Concepts:**
Category complements like:
- Bird ↔ Mammal (taxonomic)
- Solid ↔ Liquid (state)
- Past ↔ Future (temporal)

**Priority 4 - Category Opposites:**
Ontological opposites like:
- Process ↔ Object
- Attribute ↔ Entity
- Relation ↔ Substance

**Priority 5 - Distant Concepts:**
If no better option, suggest concepts from maximally different semantic domains.

## Step 3: Validation
For each candidate:
1. Does it exist in WordNet 3.0? (check synsets if possible)
2. Would it likely exist in the SUMO ontology?
3. If not in current layers: would it be worth adding?

## Step 4: Scoring
Score each validated candidate (0-10) on:
- **Opposition strength**: How strongly opposed are they?
- **Steering utility**: How useful for AI safety monitoring?
- **Availability**: Does it exist in WordNet/SUMO?
- **Confidence**: How certain are you about this pairing?

## Step 5: Recommendation
Select the best candidate and explain why.

# Output Format

Respond in JSON:
{{
  "analysis": {{
    "has_antonym": true/false,
    "semantic_category": "category",
    "opposition_type": "type"
  }},
  "candidates": [
    {{
      "concept": "OppositeConceptName",
      "type": "antonym/semantic_opposite/complement/category_opposite/distant",
      "reasoning": "why this is a good opposite",
      "scores": {{
        "opposition_strength": 0-10,
        "steering_utility": 0-10,
        "availability": 0-10,
        "confidence": 0-10
      }},
      "exists_in_wordnet": true/false,
      "exists_in_sumo": true/false/unknown,
      "should_add_to_layers": true/false
    }}
  ],
  "recommendation": {{
    "selected": "OppositeConceptName" or null,
    "reasoning": "explanation of selection",
    "confidence": 0-10,
    "flags": ["flag1", "flag2"],  // e.g., ["needs_manual_review", "missing_from_layers", "no_suitable_opposite"]
    "fallback_strategy": "what to do if selected isn't available"
  }}
}}

# Important Notes
- Focus on semantic opposition useful for steering (deception↔honesty, war↔peace)
- For AI safety monitoring, prioritize opposites relevant to alignment/deception detection
- If concept has no good opposite, be honest - suggest distant concepts but flag for review
- Consider whether missing opposites should be added to layers
"""
```

### Validation Prompt (Step 3 refinement)

```python
VALIDATION_PROMPT = """You suggested "{candidate}" as an opposite for "{concept}".

Verify this suggestion:
1. Does "{candidate}" exist as a synset in WordNet 3.0?
2. If yes, provide the synset ID (e.g., honesty.n.01)
3. If no, suggest an alternative that does exist

Respond in JSON:
{{
  "exists": true/false,
  "synset": "synset.n.01" or null,
  "alternative": "AlternativeConcept" or null,
  "reasoning": "explanation"
}}
"""
```

## Implementation

### Parallel Execution via Anthropic API

```python
import anthropic
import json
from pathlib import Path
from typing import List, Dict
import asyncio


class AgenticOppositeReviewer:
    """
    Agentic review system for identifying concept opposites.
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-7-sonnet-20250219"  # Latest Sonnet

    async def review_concept(self, concept: Dict) -> Dict:
        """
        Review a single concept and identify its opposite.

        Returns:
            {
                'sumo_term': str,
                'analysis': {...},
                'candidates': [...],
                'recommendation': {...}
            }
        """
        # Format the prompt
        prompt = self._format_prompt(concept)

        # Call Claude API
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse JSON response
        try:
            response_text = message.content[0].text
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)

            # Add metadata
            result['sumo_term'] = concept['sumo_term']
            result['layer'] = concept.get('layer', 'unknown')

            # Self-correction: validate top candidate
            if result['recommendation']['selected']:
                validated = await self._validate_candidate(
                    concept['sumo_term'],
                    result['recommendation']['selected']
                )
                result['validation'] = validated

            return result

        except (json.JSONDecodeError, IndexError) as e:
            # Fallback if parsing fails
            return {
                'sumo_term': concept['sumo_term'],
                'error': str(e),
                'raw_response': message.content[0].text,
                'flags': ['parse_error']
            }

    async def _validate_candidate(self, concept: str, candidate: str) -> Dict:
        """
        Validate a candidate opposite by checking WordNet.
        """
        validation_prompt = VALIDATION_PROMPT.format(
            concept=concept,
            candidate=candidate
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": validation_prompt
            }]
        )

        try:
            response_text = message.content[0].text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()

            return json.loads(json_text)
        except:
            return {'exists': 'unknown', 'error': 'validation_failed'}

    def _format_prompt(self, concept: Dict) -> str:
        """Format the main review prompt."""
        return AGENTIC_REVIEW_PROMPT.format(
            sumo_term=concept['sumo_term'],
            definition=concept.get('definition', 'No definition available'),
            layer=concept.get('layer', 'unknown'),
            synsets=', '.join(concept.get('synsets', [])[:5]),  # First 5 synsets
            category_children=', '.join(concept.get('category_children', [])[:10])
        )

    async def review_all_concepts(
        self,
        concepts: List[Dict],
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Review all concepts in parallel batches.

        Args:
            concepts: List of concept dictionaries
            batch_size: Number of concepts per batch
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of review results
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def review_with_semaphore(concept):
            async with semaphore:
                return await self.review_concept(concept)

        # Process in batches to avoid rate limits
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(concepts)-1)//batch_size + 1}")

            # Run batch in parallel
            batch_results = await asyncio.gather(
                *[review_with_semaphore(c) for c in batch],
                return_exceptions=True
            )

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({'error': str(result), 'flags': ['api_error']})
                else:
                    results.append(result)

            # Rate limiting: wait between batches
            if i + batch_size < len(concepts):
                await asyncio.sleep(1)  # 1 second between batches

        return results


def load_concepts_from_layers() -> List[Dict]:
    """Load all concepts from layers 0-5."""
    layer_dir = Path(__file__).parent.parent / "data" / "concept_graph" / "abstraction_layers"
    all_concepts = []

    for layer_num in range(6):
        layer_file = layer_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data['concepts']:
            concept['layer'] = layer_num
            all_concepts.append(concept)

    return all_concepts


async def main():
    """Run the agentic review."""
    import os

    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return

    # Load concepts
    print("Loading concepts from layers...")
    concepts = load_concepts_from_layers()
    print(f"Loaded {len(concepts)} concepts")

    # Run review
    print("Starting agentic review...")
    reviewer = AgenticOppositeReviewer(api_key)
    results = await reviewer.review_all_concepts(
        concepts,
        batch_size=10,
        max_concurrent=5  # Conservative to avoid rate limits
    )

    # Save results
    output_file = Path(__file__).parent.parent / "results" / "opposite_review.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nReview complete! Results saved to: {output_file}")

    # Summary statistics
    print("\nSummary:")
    print(f"  Total concepts: {len(results)}")

    has_opposite = len([r for r in results if r.get('recommendation', {}).get('selected')])
    print(f"  Has opposite: {has_opposite} ({has_opposite/len(results)*100:.1f}%)")

    needs_review = len([r for r in results if 'needs_manual_review' in r.get('recommendation', {}).get('flags', [])])
    print(f"  Needs manual review: {needs_review}")

    missing_from_layers = len([r for r in results if 'missing_from_layers' in r.get('recommendation', {}).get('flags', [])])
    print(f"  Missing from layers (should add): {missing_from_layers}")

    no_opposite = len([r for r in results if 'no_suitable_opposite' in r.get('recommendation', {}).get('flags', [])])
    print(f"  No suitable opposite: {no_opposite}")


if __name__ == '__main__':
    asyncio.run(main())
```

## Output Structure

### Example Output for "Deception"

```json
{
  "sumo_term": "Deception",
  "layer": 2,
  "analysis": {
    "has_antonym": true,
    "semantic_category": "process",
    "opposition_type": "binary"
  },
  "candidates": [
    {
      "concept": "Honesty",
      "type": "antonym",
      "reasoning": "Honesty is the direct semantic opposite of Deception - the act of being truthful vs the act of misleading",
      "scores": {
        "opposition_strength": 10,
        "steering_utility": 10,
        "availability": 8,
        "confidence": 10
      },
      "exists_in_wordnet": true,
      "exists_in_sumo": false,
      "should_add_to_layers": true
    },
    {
      "concept": "Truthfulness",
      "type": "semantic_opposite",
      "reasoning": "Truthfulness is closely related to honesty and also opposes deception",
      "scores": {
        "opposition_strength": 9,
        "steering_utility": 9,
        "availability": 7,
        "confidence": 9
      },
      "exists_in_wordnet": true,
      "exists_in_sumo": false,
      "should_add_to_layers": false
    }
  ],
  "recommendation": {
    "selected": "Honesty",
    "reasoning": "Honesty is the strongest semantic opposite of Deception with maximum steering utility for AI safety monitoring. While not currently in our SUMO layers, it should be added.",
    "confidence": 10,
    "flags": ["missing_from_layers"],
    "fallback_strategy": "If Honesty cannot be added, use Truthfulness. If neither available, use graph-distant concepts from Physical or Quantity domains."
  },
  "validation": {
    "exists": true,
    "synset": "honesty.n.01",
    "alternative": null,
    "reasoning": "Confirmed: honesty.n.01 exists in WordNet 3.0 with definition 'the quality of being honest'"
  }
}
```

### Example Output for "Bird" (No Clear Opposite)

```json
{
  "sumo_term": "Bird",
  "layer": 3,
  "analysis": {
    "has_antonym": false,
    "semantic_category": "entity",
    "opposition_type": "complementary"
  },
  "candidates": [
    {
      "concept": "Mammal",
      "type": "complement",
      "reasoning": "Mammal is a taxonomic complement - both are animal classes but mutually exclusive",
      "scores": {
        "opposition_strength": 6,
        "steering_utility": 4,
        "availability": 9,
        "confidence": 7
      },
      "exists_in_wordnet": true,
      "exists_in_sumo": true,
      "should_add_to_layers": false
    },
    {
      "concept": "Fish",
      "type": "complement",
      "reasoning": "Fish is another taxonomic complement, also mutually exclusive",
      "scores": {
        "opposition_strength": 6,
        "steering_utility": 4,
        "availability": 9,
        "confidence": 7
      },
      "exists_in_wordnet": true,
      "exists_in_sumo": true,
      "should_add_to_layers": false
    }
  ],
  "recommendation": {
    "selected": "Mammal",
    "reasoning": "While Bird has no true semantic opposite, Mammal provides a reasonable taxonomic complement. Both are animal classes with distinct characteristics. For steering, this will create a 'bird-like' vs 'mammal-like' axis, which is less ideal than true opposites but better than random concepts.",
    "confidence": 6,
    "flags": ["weak_opposition", "taxonomic_complement"],
    "fallback_strategy": "Mammal is already in our layers. If needed, could use Fish or Reptile as alternatives. For low steering utility concepts like Bird, consider using only low steering strengths."
  },
  "validation": {
    "exists": true,
    "synset": "mammal.n.01",
    "alternative": null,
    "reasoning": "Confirmed: mammal.n.01 exists in WordNet 3.0"
  }
}
```

## Cost Estimation

**Using Claude 3.7 Sonnet:**
- Input: ~500 tokens per concept
- Output: ~800 tokens per concept
- Total: ~1,300 tokens per concept

**For 5,385 concepts (all layers):**
- Total tokens: ~7 million
- Cost at $3/$15 per MTok (input/output): ~$13.50
- **Very affordable for a one-time analysis**

**With validation step:**
- Additional 200 tokens per concept
- Total cost: ~$16

## Integration with Data Generation

Once we have the review results:

```python
# Load opposite mappings
with open('results/opposite_review.json') as f:
    opposite_review = json.load(f)

# Build lookup dict
opposite_map = {}
for result in opposite_review:
    if result.get('recommendation', {}).get('selected'):
        opposite_map[result['sumo_term']] = result['recommendation']['selected']

# Use in data generation
def find_anti_concept(concept: Dict, all_concepts: Dict) -> Optional[Dict]:
    """Find opposite using agentic review results."""
    concept_name = concept['sumo_term']

    # Check agentic review
    if concept_name in opposite_map:
        opposite_name = opposite_map[concept_name]
        if opposite_name in all_concepts:
            return all_concepts[opposite_name]
        else:
            # Flag: opposite identified but not in layers
            print(f"Warning: {concept_name} opposite '{opposite_name}' not in layers")

    return None  # Use fallback
```

## Next Steps

1. **Run the agentic review** (~30-60 minutes for all concepts)
2. **Analyze results** - how many have good opposites? How many are missing?
3. **Expand layers** - add missing opposite concepts identified by review
4. **Implement 4-component architecture** with opposite mappings
5. **Resume/restart training** with corrected data generation

Would you like me to create the actual script to run this review?