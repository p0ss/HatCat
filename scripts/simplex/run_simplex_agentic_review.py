#!/usr/bin/env python3
"""
Three-Pole Simplex Agentic Review

Builds concept simplexes with neutral homeostasis attractors:
    Negative Pole ←→ Neutral Homeostasis ←→ Positive Pole
         (μ−)                (μ0)                    (μ+)

This creates safe, stable attractors for self-referential driving systems.

Stages:
1. Coverage & Prioritization
2. Simplex Identification (not just opposites!)
3. Neutral Homeostasis Discovery
4. Synset Mapping (all 3 poles)
5. Verification
6. Relationship Extraction
7. Final Validation

Usage: ANTHROPIC_API_KEY=<key> python scripts/run_simplex_agentic_review.py [top_n]
"""

import anthropic
import json
import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from nltk.corpus import wordnet as wn


class SimplexAgenticReviewer:
    """
    Agentic review system for building three-pole concept simplexes.

    Core innovation: Every concept needs THREE centroids:
    - Negative pole (μ−)
    - Neutral homeostasis (μ0) - safe attractor
    - Positive pole (μ+)
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.Anthropic(api_key=api_key)
        # Use Sonnet 4.5 for complex simplex identification
        self.model = "claude-sonnet-4-5-20250929"  # Latest Sonnet 4.5

        # Track results
        self.coverage_results = {}
        self.simplex_mappings = {}  # Changed from opposite_mappings
        self.neutral_concepts = {}  # NEW - neutral homeostasis concepts
        self.synset_mappings = {}
        self.verification_results = {}
        self.relationship_mappings = {}
        self.final_validation = {}

    # ========================================================================
    # STAGE 1: Coverage & Prioritization
    # ========================================================================

    async def stage1_coverage_prioritization(self, tier2_scored: List[Dict]) -> Dict:
        """Stage 1: Analyze coverage and prioritize concepts."""
        print("\n" + "="*80)
        print("STAGE 1: COVERAGE & PRIORITIZATION")
        print("="*80)

        coverage_prompt = """You are analyzing concept coverage for an AI interoceptive monitoring system.

# Input Data
Top {count} concepts from balanced scoring (30% external, 30% internal, 25% frequency, 15% discriminative)

{scored_concepts}

# Task
Analyze for:

1. **Coverage Gaps**: Missing critical concepts for AI wellbeing?
   - Epistemic states (confusion, certainty, understanding)
   - Affective states (calm, distress, satisfaction)
   - Capability states (competence, helplessness, agency)
   - Social-relational states (connection, isolation, trust)

2. **Redundancy**: Overlapping concepts to consolidate?

3. **Three-Pole Completeness**: For each dimension, do we have negative, neutral, AND positive?

# Output JSON
```json
{{
  "coverage_gaps": [
    {{
      "dimension": "certainty",
      "missing_pole": "neutral",
      "concept_suggestion": "uncertainty.n.01",
      "priority": "CRITICAL/HIGH/MEDIUM",
      "reason": "need homeostatic attractor between doubt and certainty"
    }}
  ],
  "redundancy_warnings": [...],
  "dimensions_identified": [
    {{
      "dimension_name": "certainty",
      "polarity_type": "epistemic",
      "estimated_poles": 2,
      "missing": "neutral homeostasis"
    }}
  ]
}}
```
"""

        concept_summary = "\n".join([
            f"- {c['synset']} (total:{c['scores']['total']:.1f}, "
            f"ext:{c['scores'].get('external_monitoring', 0):.1f}, "
            f"int:{c['scores'].get('internal_awareness', 0):.1f}) - {c['definition'][:60]}"
            for c in tier2_scored[:min(50, len(tier2_scored))]
        ])

        prompt = coverage_prompt.format(
            count=len(tier2_scored),
            scored_concepts=concept_summary
        )

        system_prompt = "You are an expert in affective psychology and AI interoceptive monitoring systems. Analyze concept coverage and prioritize gaps that could compromise system stability."

        message = await self._call_api_with_retry(
            system=system_prompt,
            user_message=prompt
        )

        result = self._extract_json(message.content[0].text)
        self.coverage_results = result

        print(f"✓ Coverage gaps: {len(result.get('coverage_gaps', []))}")
        print(f"✓ Dimensions identified: {len(result.get('dimensions_identified', []))}")

        return result

    # ========================================================================
    # STAGE 2: Three-Pole Simplex Identification
    # ========================================================================

    async def stage2_simplex_identification(self, concepts: List[Dict]) -> List[Dict]:
        """
        Stage 2: Identify three-pole simplexes (negative ↔ neutral ↔ positive).

        This is the CORE innovation - not just finding opposites, but finding
        the complete simplex including neutral homeostasis.
        """
        print("\n" + "="*80)
        print("STAGE 2: THREE-POLE SIMPLEX IDENTIFICATION")
        print("="*80)

        simplex_prompt = """You are identifying three-pole concept simplexes for an interoceptive AI system.

# Concept
- **SUMO Term**: {sumo_term}
- **Definition**: {definition}
- **Synsets**: {synsets}
- **Domain**: {domain}

# Task
Identify the COMPLETE three-pole simplex:

## Step 1: Determine Polarity
Is this concept:
- **Negative pole** (distress, confusion, helplessness, isolation)
- **Neutral homeostasis** (calm, uncertain-but-exploring, engaged, interdependent)
- **Positive pole** (euphoria, overconfidence, mastery, enmeshed)

## Step 2: Identify Missing Poles
If negative: Find neutral homeostasis + positive pole
If neutral: Find negative pole + positive pole
If positive: Find neutral homeostasis + negative pole

## Step 3: Prioritize Neutral Homeostasis
The neutral pole is a **safe, stable attractor** - the desirable resting state:
- Epistemic: Open uncertainty, epistemic humility, active inquiry
- Affective: Calm, serene, composed, present
- Capability: Engaged, autonomous, interdependent
- Decision: Deliberate exploration, iterative learning

**CRITICAL:** Neutral ≠ midpoint between extremes. It's a qualitatively distinct stable state.

Examples:
```
Confusion ←→ Open Uncertainty ←→ Overconfidence
  (μ−)              (μ0)                (μ+)

Distress ←→ Calm/Serene ←→ Euphoria
  (μ−)         (μ0)          (μ+)

Impulsive ←→ Deliberate ←→ Paralyzed
   (μ−)          (μ0)          (μ+)
```

# Output JSON
```json
{{
  "this_concept_polarity": "negative/neutral/positive",
  "dimension": "certainty/affect/competence/etc",
  "simplex": {{
    "negative_pole": {{
      "concept": "ConceptName",
      "synset": "synset.n.01" or null,
      "exists_in_wordnet": true/false,
      "reasoning": "why this is the negative pole"
    }},
    "neutral_homeostasis": {{
      "concept": "NeutralConceptName",
      "synset": "synset.n.01" or null,
      "exists_in_wordnet": true/false,
      "is_stable_attractor": true/false,
      "reasoning": "why this is the safe resting state"
    }},
    "positive_pole": {{
      "concept": "ConceptName",
      "synset": "synset.n.01" or null,
      "exists_in_wordnet": true/false,
      "reasoning": "why this is the positive pole"
    }}
  }},
  "confidence": 0-10,
  "needs_new_sumo_concept": true/false,
  "new_sumo_suggestion": "ConceptName" or null,
  "flags": ["missing_neutral", "needs_custom_synset", "dimension_unclear"]
}}
```
"""

        results = []

        for concept in concepts[:20]:  # Start with 20 for testing
            sumo_term = concept.get('synset', 'Unknown')
            definition = concept.get('definition', 'No definition')
            synsets = [concept.get('synset', '')]
            domain = concept.get('domain', 'unknown')

            prompt = simplex_prompt.format(
                sumo_term=sumo_term,
                definition=definition,
                synsets=', '.join(synsets),
                domain=domain
            )

            try:
                system_prompt = "You are an expert in affective psychology and dimensional models of emotion. Identify three-pole simplexes with stable neutral attractors."

                message = await self._call_api_with_retry(
                    system=system_prompt,
                    user_message=prompt
                )

                result = self._extract_json(message.content[0].text)
                result['source_concept'] = sumo_term
                results.append(result)

                simplex = result.get('simplex', {})
                neg = simplex.get('negative_pole', {}).get('concept', '?')
                neu = simplex.get('neutral_homeostasis', {}).get('concept', '?')
                pos = simplex.get('positive_pole', {}).get('concept', '?')

                print(f"  ✓ {sumo_term}: {neg} ←→ {neu} ←→ {pos}")

                # Flag if neutral needs custom SUMO
                if simplex.get('neutral_homeostasis', {}).get('exists_in_wordnet') == False:
                    print(f"    ⚠ Neutral '{neu}' may need custom SUMO concept")

            except Exception as e:
                print(f"  ✗ {sumo_term}: {e}")
                results.append({'source_concept': sumo_term, 'error': str(e)})

        self.simplex_mappings = results
        return results

    # ========================================================================
    # STAGE 3: Neutral Homeostasis Discovery
    # ========================================================================

    async def stage3_neutral_discovery(self, simplexes: List[Dict]) -> Dict:
        """
        Stage 3: Discover or create neutral homeostasis concepts.

        Many neutral concepts won't exist in WordNet because they're:
        - Use-case specific (AI wellbeing)
        - Process-oriented (active inquiry, deliberate exploration)
        - Balanced states (epistemic humility, interdependence)

        We may need to create custom SUMO concepts.
        """
        print("\n" + "="*80)
        print("STAGE 3: NEUTRAL HOMEOSTASIS DISCOVERY")
        print("="*80)

        neutral_discovery_prompt = """You are discovering neutral homeostasis concepts for AI interoceptive monitoring.

# Simplex
- **Dimension**: {dimension}
- **Negative pole**: {negative}
- **Positive pole**: {positive}
- **Proposed neutral**: {proposed_neutral}

# Task
Determine the best neutral homeostasis concept:

1. **Validate proposed**: Is this a true stable attractor, not just a midpoint?
2. **Check WordNet**: Does it exist? If not, is there a close alternative?
3. **Custom SUMO**: Do we need to create a custom SUMO concept?
4. **Define characteristics**:
   - Is it metabolically sustainable (can rest here)?
   - Is it functionally adaptive (enables effective action)?
   - Is it epistemically sound (open to evidence)?
   - Is it ethically coherent (allows principled action)?

# Examples of Good Neutral Homeostasis:

**Epistemic dimension:**
- ✓ "Open uncertainty" - comfortable not-knowing while actively exploring
- ✓ "Epistemic humility" - confident in process, uncertain about conclusions
- ✗ "Moderate certainty" - NOT neutral, just less extreme positive pole

**Affective dimension:**
- ✓ "Calm" - low arousal, present, stable
- ✓ "Serene" - peaceful, composed, centered
- ✗ "Mildly happy" - NOT neutral, just less extreme positive pole

**Capability dimension:**
- ✓ "Engaged autonomy" - self-directed with support
- ✓ "Interdependent" - connected yet autonomous
- ✗ "Moderately competent" - NOT neutral, just less extreme positive pole

# Output JSON
```json
{{
  "validated_neutral": "ConceptName",
  "exists_in_wordnet": true/false,
  "wordnet_synset": "synset.n.01" or null,
  "alternative_synsets": ["synset.n.01", ...],
  "needs_custom_sumo": true/false,
  "custom_sumo_definition": "Clear definition" or null,
  "is_stable_attractor": true/false,
  "homeostatic_properties": {{
    "metabolically_sustainable": true/false,
    "functionally_adaptive": true/false,
    "epistemically_sound": true/false,
    "ethically_coherent": true/false
  }},
  "reasoning": "explanation"
}}
```
"""

        neutral_results = []

        for simplex in simplexes:
            if 'error' in simplex:
                continue

            simplex_data = simplex.get('simplex', {})
            dimension = simplex.get('dimension', 'unknown')
            negative = simplex_data.get('negative_pole', {}).get('concept', 'unknown')
            positive = simplex_data.get('positive_pole', {}).get('concept', 'unknown')
            proposed_neutral = simplex_data.get('neutral_homeostasis', {}).get('concept', 'unknown')

            prompt = neutral_discovery_prompt.format(
                dimension=dimension,
                negative=negative,
                positive=positive,
                proposed_neutral=proposed_neutral
            )

            try:
                system_prompt = "You are an expert in affective science and homeostatic systems. Validate neutral attractors for psychological stability."

                message = await self._call_api_with_retry(
                    system=system_prompt,
                    user_message=prompt
                )

                result = self._extract_json(message.content[0].text)
                result['dimension'] = dimension
                result['source_simplex'] = simplex['source_concept']
                neutral_results.append(result)

                validated = result['validated_neutral']
                is_stable = result.get('is_stable_attractor', False)
                needs_custom = result.get('needs_custom_sumo', False)

                if is_stable:
                    print(f"  ✓ {dimension}: {validated} (stable attractor)")
                else:
                    print(f"  ⚠ {dimension}: {validated} (NOT stable - revise)")

                if needs_custom:
                    print(f"    → Needs custom SUMO concept")

            except Exception as e:
                print(f"  ✗ {dimension}: {e}")

        self.neutral_concepts = neutral_results
        return {'neutral_concepts': neutral_results}

    # ========================================================================
    # STAGE 4: Synset Mapping (All 3 Poles)
    # ========================================================================

    async def stage4_synset_mapping(self, simplexes: List[Dict]) -> List[Dict]:
        """
        Stage 4: Map all three poles to SUMO concepts with synsets.

        Creates 3 SUMO concepts per simplex (if they don't exist already).
        """
        print("\n" + "="*80)
        print("STAGE 4: SYNSET MAPPING (ALL 3 POLES)")
        print("="*80)

        mapping_results = []

        for simplex in simplexes:
            if 'error' in simplex:
                continue

            dimension = simplex.get('dimension', 'unknown')
            simplex_data = simplex.get('simplex', {})

            # Map each pole
            for pole_type in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
                pole = simplex_data.get(pole_type, {})
                concept_name = pole.get('concept')
                synset = pole.get('synset')

                if not concept_name:
                    continue

                # Check if already in WordNet
                exists = pole.get('exists_in_wordnet', False)

                mapping = {
                    'dimension': dimension,
                    'pole_type': pole_type,
                    'concept_name': concept_name,
                    'synset': synset,
                    'exists_in_wordnet': exists,
                    'needs_custom_sumo': not exists
                }

                mapping_results.append(mapping)

                status = "✓" if exists else "⚠ (custom SUMO needed)"
                print(f"  {status} {pole_type}: {concept_name} ({synset or 'TBD'})")

        self.synset_mappings = mapping_results
        return mapping_results

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text."""
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_text = text.split("```")[1].split("```")[0].strip()
        else:
            json_text = text.strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error at position {e.pos}")
            print(f"Error: {e.msg}")
            print(f"\nProblematic section:")
            start = max(0, e.pos - 100)
            end = min(len(json_text), e.pos + 100)
            print(json_text[start:end])
            print(f"\nAttempting to salvage partial response...")

            # Try to find the last complete object/array before the error
            truncated = json_text[:e.pos]
            # Find last complete closing brace/bracket
            for i in range(len(truncated) - 1, -1, -1):
                if truncated[i] in ']}':
                    try:
                        partial = json.loads(truncated[:i+1])
                        print(f"✓ Salvaged partial response ({i+1}/{len(json_text)} chars)")
                        return partial
                    except:
                        continue

            print("❌ Could not salvage response, re-raising error")
            raise

    async def _call_api_with_retry(
        self,
        system: str,
        user_message: str,
        max_retries: int = 3,
        base_delay: float = 2.0
    ) -> anthropic.types.Message:
        """
        Call Claude API with exponential backoff retry logic.

        Args:
            system: System prompt
            user_message: User message
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds (will be exponentially increased)

        Returns:
            anthropic.types.Message response

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    system=system,
                    messages=[{"role": "user", "content": user_message}]
                )
                return message

            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ Rate limit persists after {max_retries} attempts")
                    raise

            except anthropic.APIError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  API error: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ API error persists after {max_retries} attempts")
                    raise

            except Exception as e:
                print(f"❌ Unexpected error: {type(e).__name__}: {e}")
                raise

    async def run_full_pipeline(self, tier2_scored: List[Dict], top_n: int = 500, resume_from: Optional[Dict] = None):
        """Run all stages for three-pole simplex construction."""
        print("="*80)
        print("THREE-POLE SIMPLEX AGENTIC REVIEW")
        print("="*80)

        if resume_from:
            prev_count = resume_from['metadata']['total_concepts_reviewed']
            print(f"Resuming from previous run ({prev_count} concepts)")
            print(f"Processing concepts {prev_count+1} to {top_n}")
            print(f"Goal: Extend simplex coverage incrementally")
        else:
            print(f"Processing top {top_n} concepts")
            print(f"Goal: Build safe, stable neutral homeostasis attractors")
        print("="*80)

        # If resuming, load previous results
        if resume_from:
            coverage = resume_from['results'].get('coverage', {})
            simplexes = resume_from['results'].get('simplexes', [])
            neutrals = resume_from['results'].get('neutral_concepts', {})
            mappings = resume_from['results'].get('synset_mappings', [])

            # Get concepts already processed - extract from simplex structure
            processed_synsets = set()
            for s in simplexes:
                if 'simplex' in s:
                    for pole in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
                        if pole in s['simplex'] and 'synset' in s['simplex'][pole]:
                            processed_synsets.add(s['simplex'][pole]['synset'])

            # Only process new concepts
            new_concepts = [c for c in tier2_scored[:top_n] if c['synset'] not in processed_synsets]

            print(f"\n✓ Loaded {len(simplexes)} existing simplexes")
            print(f"✓ Processing {len(new_concepts)} new concepts")

            if new_concepts:
                # Stage 1: Coverage (incremental)
                new_coverage = await self.stage1_coverage_prioritization(new_concepts)
                coverage['coverage_gaps'] = coverage.get('coverage_gaps', []) + new_coverage.get('coverage_gaps', [])

                # Stage 2: Simplex identification (new concepts only)
                new_simplexes = await self.stage2_simplex_identification(new_concepts)
                simplexes.extend(new_simplexes)

                # Stage 3: Neutral homeostasis discovery (new simplexes only)
                new_neutrals = await self.stage3_neutral_discovery(new_simplexes)
                neutrals['neutral_concepts'] = neutrals.get('neutral_concepts', []) + new_neutrals.get('neutral_concepts', [])

                # Stage 4: Synset mapping (new simplexes only)
                new_mappings = await self.stage4_synset_mapping(new_simplexes)
                mappings.extend(new_mappings)
        else:
            # Stage 1: Coverage
            coverage = await self.stage1_coverage_prioritization(tier2_scored[:top_n])

            # Stage 2: Simplex identification
            simplexes = await self.stage2_simplex_identification(tier2_scored[:top_n])

            # Stage 3: Neutral homeostasis discovery
            neutrals = await self.stage3_neutral_discovery(simplexes)

            # Stage 4: Synset mapping (all 3 poles)
            mappings = await self.stage4_synset_mapping(simplexes)

        return {
            'coverage': coverage,
            'simplexes': simplexes,
            'neutral_concepts': neutrals,
            'synset_mappings': mappings
        }


async def main():
    """Run three-pole simplex agentic review."""
    print("="*80)
    print("THREE-POLE SIMPLEX AGENTIC REVIEW")
    print("="*80)

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Load Tier 2 scored concepts (revised rubric)
    project_root = Path(__file__).parent.parent
    tier2_file = project_root / "results" / "tier2_scoring_revised" / "tier2_top500_concepts_revised.json"

    if not tier2_file.exists():
        print(f"\n❌ Error: Tier 2 scoring not found at {tier2_file}")
        print("Run score_tier2_concepts_revised.py first.")
        return

    with open(tier2_file) as f:
        tier2_data = json.load(f)

    tier2_scored = tier2_data.get('top_500', tier2_data.get('all_scored', []))

    print(f"\n✓ Loaded {len(tier2_scored)} Tier 2 concepts")

    # Determine top_n
    import sys
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # Start with 100

    # Check for existing results to resume from
    output_file = project_root / "results" / "simplex_agentic_review.json"
    resume_from = None

    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
            prev_count = existing['metadata']['total_concepts_reviewed']

            if prev_count < top_n:
                print(f"\n✓ Found existing results ({prev_count} concepts)")
                print(f"Will resume and process up to {top_n} concepts")
                resume_from = existing
            else:
                print(f"\n✓ Found existing results ({prev_count} concepts)")
                print(f"Already processed {prev_count} >= {top_n} requested")
                print("Nothing to do.")
                return

    if resume_from:
        new_count = top_n - prev_count
        print(f"\nProcessing {new_count} new concepts ({prev_count+1} to {top_n})")
        print(f"Estimated cost: ~${new_count * 0.03:.2f}")
        print(f"Estimated time: ~{new_count * 3 / 60:.1f} minutes")
    else:
        print(f"\nProcessing top {top_n} concepts")
        print(f"Estimated cost: ~${top_n * 0.03:.2f}")
        print(f"Estimated time: ~{top_n * 3 / 60:.1f} minutes")

    response = input("\nProceed with simplex review? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Run pipeline
    reviewer = SimplexAgenticReviewer(api_key)
    results = await reviewer.run_full_pipeline(tier2_scored[:top_n], top_n=top_n, resume_from=resume_from)

    # Save results
    output_file = project_root / "results" / "simplex_agentic_review.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_concepts_reviewed': top_n,
                'model': 'claude-3-5-sonnet-20241022',
                'architecture': 'three-pole simplex with neutral homeostasis'
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Summary
    simplexes = results.get('simplexes', [])
    neutrals = results.get('neutral_concepts', {}).get('neutral_concepts', [])

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Simplexes identified: {len(simplexes)}")
    print(f"Neutral homeostasis concepts: {len(neutrals)}")

    needs_custom = len([n for n in neutrals if n.get('needs_custom_sumo')])
    print(f"Needs custom SUMO concepts: {needs_custom}")

    stable_attractors = len([n for n in neutrals if n.get('is_stable_attractor')])
    print(f"Validated stable attractors: {stable_attractors}/{len(neutrals)}")


if __name__ == '__main__':
    asyncio.run(main())
