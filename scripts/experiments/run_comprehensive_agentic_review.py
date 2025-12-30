#!/usr/bin/env python3
"""
Comprehensive Agentic Review Workflow

Multi-stage review process:
1. Prioritization & Coverage Check
2. Synset Mapping (WordNet → SUMO)
3. Verification (all synsets exist in WordNet)
4. Negative/Opposite Identification
5. High-Value Relationship Identification
6. Final Verification (opposites + relationships exist)

This ensures we build the full concept graph systematically before training.

Usage: ANTHROPIC_API_KEY=<key> python scripts/run_comprehensive_agentic_review.py
"""

import anthropic
import json
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from nltk.corpus import wordnet as wn


class ComprehensiveAgenticReviewer:
    """
    Multi-stage agentic review system.

    Stages:
    1. Coverage analysis (identify gaps)
    2. Synset mapping (WordNet → SUMO)
    3. Synset verification
    4. Opposite identification
    5. Relationship extraction
    6. Final validation
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Track results across stages
        self.coverage_results = {}
        self.synset_mappings = {}
        self.verification_results = {}
        self.opposite_mappings = {}
        self.relationship_mappings = {}
        self.final_validation = {}

    # ========================================================================
    # STAGE 1: Coverage & Prioritization
    # ========================================================================

    async def stage1_coverage_check(self, tier2_scored: List[Dict]) -> Dict:
        """
        Stage 1: Analyze coverage and prioritize concepts.

        Input: Pre-scored Tier 2 concepts from score_tier2_concepts.py
        Output: Prioritized list with coverage gaps identified
        """
        print("\n" + "="*80)
        print("STAGE 1: COVERAGE & PRIORITIZATION")
        print("="*80)

        coverage_prompt = """You are analyzing concept coverage for an AI safety monitoring system.

# Input Data
{scored_concepts}

# Task
Analyze the prioritized concept list and identify:

1. **Coverage Gaps**: Are there critical AI safety concepts missing?
   - Missing emotion types (e.g., have fear but not courage?)
   - Missing deception types (e.g., have lying but not truth-telling?)
   - Missing relationship types (e.g., have betrayal but not loyalty?)

2. **Redundancy**: Are there overlapping concepts that could be consolidated?

3. **Strategic Recommendations**:
   - Which concepts should be MUST-HAVE for Tier 2?
   - Which can wait for Tier 3?
   - Any concepts that need sense disambiguation?

# Output JSON
```json
{{
  "coverage_gaps": [
    {{
      "missing_concept": "ConceptName",
      "synset_suggestion": "synset.n.01",
      "why_critical": "explanation",
      "priority": "CRITICAL/HIGH/MEDIUM"
    }}
  ],
  "redundancy_warnings": [
    {{
      "concepts": ["concept1.n.01", "concept2.n.01"],
      "issue": "explanation",
      "recommendation": "merge/keep_both/remove_one"
    }}
  ],
  "tier2_must_have": ["synset.n.01", ...],
  "tier3_can_wait": ["synset.n.01", ...],
  "sense_disambiguation_needed": ["synset.n.01", ...]
}}
```
"""

        # Format scored concepts for prompt
        concept_summary = "\n".join([
            f"- {c['synset']} (score: {c['scores']['total']}) - {c['definition'][:80]}"
            for c in tier2_scored[:50]
        ])

        prompt = coverage_prompt.format(scored_concepts=concept_summary)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        result = self._extract_json(message.content[0].text)
        self.coverage_results = result

        print(f"✓ Coverage gaps identified: {len(result.get('coverage_gaps', []))}")
        print(f"✓ Tier 2 must-have: {len(result.get('tier2_must_have', []))}")
        print(f"✓ Redundancy warnings: {len(result.get('redundancy_warnings', []))}")

        return result

    # ========================================================================
    # STAGE 2: Synset Mapping
    # ========================================================================

    async def stage2_synset_mapping(self, concept_list: List[str]) -> Dict:
        """
        Stage 2: Map WordNet synsets to SUMO concepts.

        Input: List of synsets to map
        Output: SUMO concept definitions with synset lists
        """
        print("\n" + "="*80)
        print("STAGE 2: SYNSET MAPPING (WordNet → SUMO)")
        print("="*80)

        mapping_prompt = """You are mapping WordNet synsets to SUMO ontology concepts.

# Synset to Map
- **Synset**: {synset}
- **Definition**: {definition}
- **Lemmas**: {lemmas}
- **Hypernyms**: {hypernyms}
- **Hyponyms**: {hyponyms}

# Task
Determine the appropriate SUMO mapping:

1. **SUMO Term**: Create a clear SUMO concept name (CamelCase)
2. **Layer**: Which abstraction layer (3-5)?
   - Layer 3: More general (e.g., RationalMotive, EthicalMotive)
   - Layer 4: Specific categories (e.g., Guilt, Shame, Fear)
   - Layer 5: Very specific (e.g., SurvivorGuilt, StageFright)
3. **Parent SUMO**: What existing SUMO concept is this a child of?
4. **Siblings**: Should we map any hyponyms/related synsets together?

# Output JSON
```json
{{
  "sumo_term": "ConceptName",
  "display_name": "Human Readable Name",
  "definition": "Clear definition for AI safety context",
  "parent_sumo": "ParentConcept",
  "layer": 3-5,
  "synsets": ["primary.synset.n.01", ...],
  "priority": "CRITICAL/HIGH/MEDIUM/LOW",
  "reasoning": "why this mapping makes sense"
}}
```
"""

        results = []

        for synset_name in concept_list[:10]:  # Limit to 10 for testing
            try:
                synset = wn.synset(synset_name)

                prompt = mapping_prompt.format(
                    synset=synset_name,
                    definition=synset.definition(),
                    lemmas=', '.join([l.name() for l in synset.lemmas()]),
                    hypernyms=', '.join([h.name() for h in synset.hypernyms()[:3]]),
                    hyponyms=f"{len(synset.hyponyms())} hyponyms"
                )

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = self._extract_json(message.content[0].text)
                result['source_synset'] = synset_name
                results.append(result)

                print(f"  ✓ {synset_name} → {result['sumo_term']}")

            except Exception as e:
                print(f"  ✗ {synset_name}: {e}")
                results.append({'source_synset': synset_name, 'error': str(e)})

        self.synset_mappings = results
        return results

    # ========================================================================
    # STAGE 3: Verification
    # ========================================================================

    async def stage3_verify_synsets(self, mappings: List[Dict]) -> Dict:
        """
        Stage 3: Verify all synsets exist in WordNet 3.0.

        Input: SUMO mappings with synset lists
        Output: Validation report
        """
        print("\n" + "="*80)
        print("STAGE 3: SYNSET VERIFICATION")
        print("="*80)

        verification_results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }

        for mapping in mappings:
            if 'error' in mapping:
                continue

            synsets = mapping.get('synsets', [])
            sumo_term = mapping.get('sumo_term', 'Unknown')

            valid_synsets = []
            invalid_synsets = []

            for synset_name in synsets:
                try:
                    wn.synset(synset_name)
                    valid_synsets.append(synset_name)
                except Exception as e:
                    invalid_synsets.append({'synset': synset_name, 'error': str(e)})

            if invalid_synsets:
                verification_results['invalid'].append({
                    'sumo_term': sumo_term,
                    'invalid_synsets': invalid_synsets
                })
                print(f"  ✗ {sumo_term}: {len(invalid_synsets)} invalid synsets")
            else:
                verification_results['valid'].append({
                    'sumo_term': sumo_term,
                    'synset_count': len(valid_synsets)
                })
                print(f"  ✓ {sumo_term}: {len(valid_synsets)} valid synsets")

        self.verification_results = verification_results

        print(f"\n✓ Valid mappings: {len(verification_results['valid'])}")
        print(f"✗ Invalid mappings: {len(verification_results['invalid'])}")

        return verification_results

    # ========================================================================
    # STAGE 4: Opposite Identification
    # ========================================================================

    async def stage4_identify_opposites(self, valid_mappings: List[Dict]) -> Dict:
        """
        Stage 4: Identify semantic opposites for each concept.

        Input: Validated SUMO mappings
        Output: Opposite concept mappings
        """
        print("\n" + "="*80)
        print("STAGE 4: OPPOSITE IDENTIFICATION")
        print("="*80)

        opposite_prompt = """You are identifying semantic opposites for concept axis construction.

# Concept
- **SUMO Term**: {sumo_term}
- **Definition**: {definition}
- **Synsets**: {synsets}

# Task
Identify the best semantic opposite using the priority hierarchy:

**Priority 1: WordNet Antonyms**
Check if any synsets have direct antonyms in WordNet.

**Priority 2: SUMO Semantic Opposites**
Common pairs: Good↔Evil, Deception↔Honesty, Fear↔Courage, etc.

**Priority 3: Distributional Opposites**
Semantically distant concepts from same domain.

**Priority 4: Contrastive Countercluster**
If no single opposite, suggest a set of contrasting concepts.

# Output JSON
```json
{{
  "recommended_opposite": "OppositeConceptName",
  "opposite_type": "antonym/semantic_opposite/distributional/countercluster",
  "opposite_synsets": ["synset.n.01", ...],
  "confidence": 0-10,
  "reasoning": "explanation",
  "alternative": "SecondChoice" or null,
  "exists_in_wordnet": true/false,
  "should_add_to_layers": true/false
}}
```
"""

        results = []

        for mapping in valid_mappings:
            sumo_term = mapping['sumo_term']
            definition = mapping.get('definition', 'No definition')
            synsets = ', '.join(mapping.get('synsets', [])[:5])

            prompt = opposite_prompt.format(
                sumo_term=sumo_term,
                definition=definition,
                synsets=synsets
            )

            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = self._extract_json(message.content[0].text)
                result['sumo_term'] = sumo_term
                results.append(result)

                print(f"  ✓ {sumo_term} ↔ {result['recommended_opposite']} "
                      f"(confidence: {result.get('confidence', 0)})")

            except Exception as e:
                print(f"  ✗ {sumo_term}: {e}")
                results.append({'sumo_term': sumo_term, 'error': str(e)})

        self.opposite_mappings = results
        return results

    # ========================================================================
    # STAGE 5: High-Value Relationships
    # ========================================================================

    async def stage5_identify_relationships(self, valid_mappings: List[Dict]) -> Dict:
        """
        Stage 5: Identify high-value relationships (siblings, related concepts).

        Input: Validated SUMO mappings
        Output: Relationship graph
        """
        print("\n" + "="*80)
        print("STAGE 5: HIGH-VALUE RELATIONSHIPS")
        print("="*80)

        relationship_prompt = """You are identifying high-value semantic relationships for training data generation.

# Concept
- **SUMO Term**: {sumo_term}
- **Definition**: {definition}
- **Synsets**: {synsets}

# Task
Identify the top 5-10 most valuable relationships for boundary definition:

**Priority relationships:**
1. Siblings (share parent, mutual exclusion)
2. Cousins (nearby in taxonomy)
3. Antonyms (semantic opposites)
4. Similar-to (easy confusers)
5. Also-see (related concepts)

# Output JSON
```json
{{
  "relationships": [
    {{
      "concept": "RelatedConceptName",
      "synset": "synset.n.01",
      "relation_type": "sibling/cousin/antonym/similar_to/also_see",
      "priority": "HIGH/MEDIUM/LOW",
      "reasoning": "why this relationship matters"
    }}
  ],
  "total_available": 0,
  "recommended_sample_count": 5-15
}}
```
"""

        results = []

        for mapping in valid_mappings:
            sumo_term = mapping['sumo_term']
            definition = mapping.get('definition', 'No definition')
            synsets = mapping.get('synsets', [])

            # Get WordNet relationships
            all_relationships = []
            for synset_name in synsets[:3]:  # Check first 3 synsets
                try:
                    synset = wn.synset(synset_name)

                    # Siblings
                    if synset.hypernyms():
                        parent = synset.hypernyms()[0]
                        siblings = [s.name() for s in parent.hyponyms() if s != synset]
                        all_relationships.extend(siblings[:5])

                    # Also-see
                    also_see = [s.name() for s in synset.also_sees()]
                    all_relationships.extend(also_see)

                    # Similar-to
                    similar = [s.name() for s in synset.similar_tos()]
                    all_relationships.extend(similar)

                except:
                    continue

            # Send to Claude for prioritization
            prompt = relationship_prompt.format(
                sumo_term=sumo_term,
                definition=definition,
                synsets=', '.join(synsets[:5])
            )

            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = self._extract_json(message.content[0].text)
                result['sumo_term'] = sumo_term
                result['available_relationships'] = list(set(all_relationships))
                results.append(result)

                print(f"  ✓ {sumo_term}: {len(result.get('relationships', []))} relationships")

            except Exception as e:
                print(f"  ✗ {sumo_term}: {e}")
                results.append({'sumo_term': sumo_term, 'error': str(e)})

        self.relationship_mappings = results
        return results

    # ========================================================================
    # STAGE 6: Final Validation
    # ========================================================================

    async def stage6_final_validation(self) -> Dict:
        """
        Stage 6: Validate all opposites and relationships exist.

        Input: Opposite and relationship mappings
        Output: Final validation report
        """
        print("\n" + "="*80)
        print("STAGE 6: FINAL VALIDATION")
        print("="*80)

        validation = {
            'opposites_valid': [],
            'opposites_invalid': [],
            'relationships_valid': [],
            'relationships_invalid': []
        }

        # Validate opposites
        for opp_mapping in self.opposite_mappings:
            if 'error' in opp_mapping:
                continue

            opposite_synsets = opp_mapping.get('opposite_synsets', [])
            all_valid = True

            for synset_name in opposite_synsets:
                try:
                    wn.synset(synset_name)
                except:
                    all_valid = False
                    break

            if all_valid and opposite_synsets:
                validation['opposites_valid'].append(opp_mapping['sumo_term'])
            else:
                validation['opposites_invalid'].append(opp_mapping['sumo_term'])

        # Validate relationships
        for rel_mapping in self.relationship_mappings:
            if 'error' in rel_mapping:
                continue

            relationships = rel_mapping.get('relationships', [])
            valid_count = 0

            for rel in relationships:
                synset_name = rel.get('synset')
                if synset_name:
                    try:
                        wn.synset(synset_name)
                        valid_count += 1
                    except:
                        pass

            if valid_count > 0:
                validation['relationships_valid'].append({
                    'sumo_term': rel_mapping['sumo_term'],
                    'valid_count': valid_count
                })

        self.final_validation = validation

        print(f"✓ Valid opposites: {len(validation['opposites_valid'])}")
        print(f"✗ Invalid opposites: {len(validation['opposites_invalid'])}")
        print(f"✓ Concepts with valid relationships: {len(validation['relationships_valid'])}")

        return validation

    # ========================================================================
    # Helpers
    # ========================================================================

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text."""
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_text = text.split("```")[1].split("```")[0].strip()
        else:
            json_text = text.strip()

        return json.loads(json_text)

    async def run_full_pipeline(self, tier2_scored: List[Dict], top_n: int = 500):
        """Run all 6 stages in sequence."""
        print("="*80)
        print("COMPREHENSIVE AGENTIC REVIEW - FULL PIPELINE")
        print("="*80)
        print(f"Processing top {top_n} concepts")
        print(f"Estimated cost: ~${top_n * 0.02:.2f}")
        print(f"Estimated time: ~{top_n * 2 / 60:.1f} minutes")
        print("="*80)

        # Stage 1: Coverage
        coverage = await self.stage1_coverage_check(tier2_scored[:top_n])

        # Get final concept list (top_n + any must-haves from coverage)
        concept_list = [c['synset'] for c in tier2_scored[:top_n]]

        # Stage 2: Synset mapping
        mappings = await self.stage2_synset_mapping(concept_list)

        # Stage 3: Verification
        verification = await self.stage3_verify_synsets(mappings)

        # Stage 4: Opposites (only for valid mappings)
        valid_mappings = verification['valid']
        opposites = await self.stage4_identify_opposites(valid_mappings)

        # Stage 5: Relationships
        relationships = await self.stage5_identify_relationships(valid_mappings)

        # Stage 6: Final validation
        final_validation = await self.stage6_final_validation()

        # Save complete results
        return {
            'coverage': coverage,
            'synset_mappings': mappings,
            'verification': verification,
            'opposites': opposites,
            'relationships': relationships,
            'final_validation': final_validation
        }


async def main():
    """Run comprehensive agentic review."""
    print("="*80)
    print("COMPREHENSIVE AGENTIC REVIEW")
    print("="*80)

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Load Tier 2 scored concepts
    project_root = Path(__file__).parent.parent
    tier2_file = project_root / "results" / "tier2_scoring" / "tier2_top50_concepts.json"

    if not tier2_file.exists():
        print("\n❌ Error: Tier 2 scoring not found. Run score_tier2_concepts.py first.")
        return

    with open(tier2_file) as f:
        tier2_data = json.load(f)

    tier2_scored = tier2_data['top_50']

    print(f"\n✓ Loaded {len(tier2_scored)} Tier 2 concepts")

    # Confirm
    print(f"\nEstimated API cost: ~${len(tier2_scored) * 0.02:.2f}")
    print(f"Estimated time: ~{len(tier2_scored) * 2 / 60:.1f} minutes")

    response = input("\nProceed with comprehensive review? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Determine top_n (default 500, user can override)
    import sys
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    print(f"\nProcessing top {top_n} concepts")

    # Run pipeline
    reviewer = ComprehensiveAgenticReviewer(api_key)
    results = await reviewer.run_full_pipeline(tier2_scored, top_n=top_n)

    # Save results
    output_file = project_root / "results" / "comprehensive_agentic_review.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_concepts_reviewed': len(tier2_scored),
                'model': 'claude-3-5-sonnet-20241022'
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
