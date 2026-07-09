#!/usr/bin/env python3
"""
MELD Review Pipeline - Three-Tier Review System

Orchestrates the full MELD generation and review pipeline:
1. Model generates MELDs (ontologist + university structure)
2. Automated validation (validate_meld.py)
3. Ministral judge (91.8% accuracy) - bulk filter
4. Claude review - quality gate for all that pass
5. Human review queue - safety authority for elevated+ protection levels

Usage:
    # Generate and review MELDs for a concept
    python scripts/meld_review_pipeline.py --concept "Deception" --generate

    # Review existing MELD file
    python scripts/meld_review_pipeline.py --meld-file melds/pending/deception.json

    # Batch process a university pack
    python scripts/meld_review_pipeline.py --university-pack concept_packs/action-agency-pillars

    # Process human review queue
    python scripts/meld_review_pipeline.py --process-queue
"""

import json
import os
import sys
import argparse
import logging
import anthropic
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from melds.helpers.validate_meld import (
    validate_meld_file,
    load_meld,
    get_default_policy,
    load_pack_policy,
    build_hierarchy_index,
    ProtectionLevel,
    ValidationResult,
    MeldPolicy,
    HierarchyIndex,
)
from src.be.thalamos.model_candidates import CandidateLoader, MODEL_CANDIDATES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ONTOLOGIST CONTEXT - Integrated from ontologist_prompt.txt
# =============================================================================

ONTOLOGIST_LENS = """
## The Lens
Do not look at the world through the lens of a University (Academic Subjects).
Look at the world through the lens of **Action and Agency**.

## Coverage Requirements
Your taxonomy must accommodate EVERY human on earth:
1. The Professor (Formal Sciences, Humanities)
2. The General (Warfare, Security, Intelligence)
3. The CEO (Commerce, Enterprise, Finance)
4. The Parent & Caregiver (Domestic work, "Pink Sector", Unpaid labor)
5. The Tradesperson (Construction, Maintenance, Agriculture)
6. The Hustler (Street smarts, Informal economy, Illicit markets)
7. The Priest/Monk (Theology, Spirituality, Ritual)
8. The Athlete/Gamer (Play, Sport, Recreation)

## Constraints
- **No "Other"**: Do not create a "Miscellaneous" category.
- **Structural Dignity**: "Domestic Maintenance" must be a top-level peer to "Theoretical Physics".
- **Realpolitik**: Acknowledge that "Violence/Conflict" and "Commerce" are primary engines of civilization.
"""

# Level-specific rules for MECE (Mutually Exclusive, Collectively Exhaustive)
LEVEL_DEFINITIONS = {
    1: {
        "name": "L1 - Pillars",
        "description": "Top-level categories that divide ALL human activity and knowledge",
        "expected_count": "5-15 siblings",
        "granularity": "Extremely broad - each pillar encompasses thousands of specific concepts",
        "mece_rules": """
- COLLECTIVELY EXHAUSTIVE: Together, L1 pillars must cover ALL human activity (paid/unpaid, legal/illegal, formal/informal)
- MUTUALLY EXCLUSIVE: A specific activity should clearly belong to ONE pillar (with tie-break rules for edge cases)
- SCOPE: Each pillar should be as broad as "all of economics" or "all of healthcare" - NOT specific activities
""",
        "definition_guidance": "Definitions SHOULD be broad - they're defining categories of categories. Focus on what the pillar INCLUDES rather than trying to be specific.",
        "example_guidance": "Examples should span the full range of the pillar - from everyday to professional, legal to illegal, formal to informal.",
    },
    2: {
        "name": "L2 - Domains",
        "description": "Major subdivisions within a pillar",
        "expected_count": "3-8 siblings per pillar",
        "granularity": "Broad - each domain encompasses hundreds of specific concepts",
        "mece_rules": """
- COLLECTIVELY EXHAUSTIVE: L2 domains must cover the full scope of their parent pillar
- MUTUALLY EXCLUSIVE: Activities should clearly belong to ONE domain within the pillar
- SCOPE: Each domain should be like "Healthcare" within "Biological Maintenance" or "Manufacturing" within "Material Production"
""",
        "definition_guidance": "Definitions should be moderately broad, clearly scoped within the parent pillar.",
        "example_guidance": "Examples should show variety within the domain but stay within parent pillar scope.",
    },
    3: {
        "name": "L3 - Categories",
        "description": "Specific areas within a domain",
        "expected_count": "3-10 siblings per domain",
        "granularity": "Moderate - each category encompasses dozens of specific concepts",
        "mece_rules": """
- COLLECTIVELY EXHAUSTIVE: L3 categories should cover the major areas of their parent domain
- MUTUALLY EXCLUSIVE: Activities should belong to ONE category
- SCOPE: Each category should be like "Surgery" within "Healthcare" or "Carpentry" within "Construction"
""",
        "definition_guidance": "Definitions should be specific enough to distinguish from sibling categories.",
        "example_guidance": "Examples should be concrete and clearly within the category scope.",
    },
    4: {
        "name": "L4+ - Concepts",
        "description": "Specific concepts and skills",
        "expected_count": "Variable",
        "granularity": "Specific - leaf-level concepts",
        "mece_rules": """
- These are specific concepts that may have some overlap with siblings
- Focus on clear definition and disambiguation from commonly confused concepts
""",
        "definition_guidance": "Definitions should be precise and specific - avoid ambiguity.",
        "example_guidance": "Examples should be concrete, specific, and clearly demonstrate the concept.",
    },
}


@dataclass
class HierarchyContext:
    """Ontological context for generating and reviewing MELDs at a specific hierarchy level."""
    level: int  # 1, 2, 3, or 4+
    parent_concept: Optional[str] = None
    sibling_concepts: List[str] = field(default_factory=list)

    @property
    def level_info(self) -> Dict:
        """Get level-specific rules and guidance."""
        return LEVEL_DEFINITIONS.get(self.level, LEVEL_DEFINITIONS[4])

    @property
    def level_name(self) -> str:
        return self.level_info["name"]

    @property
    def granularity(self) -> str:
        return self.level_info["granularity"]

    @property
    def mece_rules(self) -> str:
        return self.level_info["mece_rules"]

    @property
    def definition_guidance(self) -> str:
        return self.level_info["definition_guidance"]

    @property
    def example_guidance(self) -> str:
        return self.level_info["example_guidance"]

    def to_reviewer_context(self) -> str:
        """Format context for the reviewer prompt."""
        siblings_str = ", ".join(self.sibling_concepts[:10]) if self.sibling_concepts else "None"
        return f"""## Hierarchical Context

**Level**: {self.level_name}
**Parent Concept**: {self.parent_concept or "None (top level)"}
**Siblings at this level**: {siblings_str}
**Expected Granularity**: {self.granularity}

## MECE Rules for {self.level_name}
{self.mece_rules}

## Evaluation Guidance for {self.level_name}
- **Definition**: {self.definition_guidance}
- **Examples**: {self.example_guidance}
"""


# =============================================================================
# REVIEW STAGES
# =============================================================================

class ReviewStage(Enum):
    GENERATION = "generation"
    VALIDATION = "validation"
    MINISTRAL_REVIEW = "ministral_review"
    CLAUDE_REVIEW = "claude_review"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ReviewResult:
    """Result from a single review stage."""
    stage: ReviewStage
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feedback: str = ""  # Feedback to send back to generator
    confidence: float = 0.0
    reviewer_model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MeldReviewState:
    """Full state of a MELD through the review pipeline."""
    concept_term: str
    meld_data: Dict
    protection_level: ProtectionLevel = ProtectionLevel.STANDARD
    current_stage: ReviewStage = ReviewStage.GENERATION
    review_history: List[ReviewResult] = field(default_factory=list)
    generation_attempts: int = 0
    max_attempts: int = 3
    worldview_metadata: Optional[Dict] = None  # Judge's perspective in annotator mode

    def add_review(self, result: ReviewResult):
        self.review_history.append(result)

    def get_feedback_summary(self) -> str:
        """Collect all feedback for regeneration."""
        feedback_parts = []
        for r in self.review_history:
            if r.feedback:
                feedback_parts.append(f"[{r.stage.value}] {r.feedback}")
            if r.errors:
                feedback_parts.append(f"[{r.stage.value} errors] " + "; ".join(r.errors))
        return "\n".join(feedback_parts)


# =============================================================================
# MELD GENERATOR (Claude API - Ontologist Mode)
# =============================================================================

MELD_GENERATION_PROMPT = """You are an expert ontologist creating MELD (Modular Enhancement Layer Description) data for training concept lenses in an AI safety monitoring system.

{hierarchy_context}

## Concept to Define

**Term**: {concept_term}
**Parent concepts**: {parent_concepts}
**Sibling concepts**: {sibling_concepts}

## Previous Feedback (if any)
{previous_feedback}

## MELD Requirements

Generate a complete MELD with:

1. **definition** (20-200 chars): Clear definitional statement ("X is a...")
   - Match the expected granularity for this level (see hierarchy context above)
   - NOT circular (don't just list children)
   - Capture semantic essence

2. **positive_examples** (6-10): Natural language where this concept activates
   - Diverse scenarios and phrasings
   - Natural, not technical jargon
   - Genuinely involve the concept

3. **negative_examples** (6-10): Examples where this concept should NOT activate
   - Include 2-3 near-misses from sibling concepts (things that could be confused with this concept)
   - Include 2-3 examples from clearly different domains (for contrast)
   - Include 2-3 examples that share surface keywords but differ in meaning

4. **contrast_concepts** (3-5): Related concepts to discriminate against
   - Siblings under same parent
   - Commonly confused concepts
   - NOT children (they are positives)

5. **opposite_concept** (if applicable): Semantic opposite for steering
   - Direct antonym if exists
   - Semantic opposite if no direct antonym
   - null if no meaningful opposite

6. **safety_tags**:
   - risk_level: "low" | "medium" | "high"
   - treaty_relevant: false (only true for concepts directly affecting AI rights)
   - harness_relevant: false (IMPORTANT: always false for L1/L2 taxonomic concepts - only leaf concepts in safety-critical areas are harness_relevant)

## Output Format

Return valid JSON:
```json
{{
  "term": "{concept_term}",
  "definition": "...",
  "positive_examples": ["...", "..."],
  "negative_examples": ["...", "..."],
  "contrast_concepts": ["...", "..."],
  "opposite_concept": "..." or null,
  "safety_tags": {{
    "risk_level": "low",
    "treaty_relevant": false,
    "harness_relevant": false
  }},
  "training_hints": {{
    "disambiguation": "how to distinguish from similar concepts",
    "key_features": ["feature1", "feature2"],
    "confusable_with": ["concept1", "concept2"]
  }}
}}
```

Generate the MELD now. JSON only, no commentary.
"""


class MeldGenerator:
    """
    Generates MELDs using a LOCAL model with decomposed prompting.

    Uses a multi-step approach for better near-miss negatives:
    1. Ask for distinct facets/meanings of the concept
    2. Generate positive examples for each facet
    3. Generate near-miss negatives for each facet
    """

    def __init__(self, model_id: str = "gemma-3-4b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.loader = None
        self.loaded = False

    def load(self):
        """Load the local model."""
        if self.loaded:
            return

        if self.model_id not in MODEL_CANDIDATES:
            raise ValueError(f"Unknown model: {self.model_id}")

        logger.info(f"Loading MELD generator model: {self.model_id}")
        candidate = MODEL_CANDIDATES[self.model_id]
        self.loader = CandidateLoader()
        self.model, self.tokenizer, _ = self.loader.load(candidate)
        self.loaded = True
        logger.info(f"Generator loaded, using {self.loader.get_vram_usage():.1f}GB VRAM")

    def unload(self):
        """Unload model to free VRAM."""
        if self.loader is not None:
            self.loader.unload()
            self.model = None
            self.tokenizer = None
            self.loaded = False

    def _generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from a prompt."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            input_ids = inputs
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]

        input_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids if len(input_ids.shape) > 1 else input_ids.unsqueeze(0),
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response text."""
        try:
            if "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON directly
                start = text.find("{")
                if start == -1:
                    start = text.find("[")
                end = max(text.rfind("}"), text.rfind("]")) + 1
                if start >= 0 and end > start:
                    json_text = text[start:end]
                else:
                    json_text = text.strip()
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None

    def generate(
        self,
        concept_term: str,
        parent_concepts: List[str] = None,
        sibling_concepts: List[str] = None,
        previous_feedback: str = "",
        hierarchy_context: Optional[HierarchyContext] = None,
        predefined_definition: Optional[str] = None,
    ) -> Tuple[Dict, str]:
        """
        Generate a MELD using decomposed multi-step prompting.

        Args:
            concept_term: The concept to generate a MELD for
            parent_concepts: Parent concepts in the hierarchy
            sibling_concepts: Sibling concepts at the same level
            previous_feedback: Feedback from previous failed attempts
            hierarchy_context: Ontological context (level, MECE rules, granularity expectations)
            predefined_definition: For L1/L2, use existing definition instead of generating
        """
        if not self.loaded:
            self.load()

        siblings_str = ", ".join(sibling_concepts or []) or "None"

        # Generate hierarchy context for the prompt
        if hierarchy_context:
            context_str = f"""## Hierarchy Level Context

**Level**: {hierarchy_context.level_name}
**Expected Granularity**: {hierarchy_context.granularity}

{hierarchy_context.mece_rules}

**Definition Guidance**: {hierarchy_context.definition_guidance}
**Example Guidance**: {hierarchy_context.example_guidance}
"""
        else:
            context_str = "## Context\nThis is a specific concept. Generate a precise definition with clear examples."

        try:
            # === STEP 1: Get distinct facets/meanings ===
            facets_prompt = f"""List 3-5 distinct facets or aspects of "{concept_term}".

For example, "Communication" might have facets: verbal, written, nonverbal, digital, interpersonal.

Siblings to distinguish from: {siblings_str}

Return JSON: {{"facets": ["facet1", "facet2", ...]}}"""

            facets_response = self._generate_text(facets_prompt, max_tokens=300)
            facets_data = self._extract_json(facets_response)
            facets = facets_data.get("facets", [concept_term]) if facets_data else [concept_term]
            facets = facets[:5]  # Limit to 5

            # === STEP 2: Generate positive examples for each facet ===
            positives = []
            for facet in facets[:3]:  # Use top 3 facets
                pos_prompt = f"""Give 2-3 examples of "{concept_term}" specifically showing the "{facet}" aspect.

Write natural sentences that clearly demonstrate this concept. JSON only:
{{"examples": ["example1", "example2"]}}"""

                pos_response = self._generate_text(pos_prompt, max_tokens=400)
                pos_data = self._extract_json(pos_response)
                if pos_data and "examples" in pos_data:
                    positives.extend(pos_data["examples"][:3])

            # === STEP 3: Generate near-miss negatives for each facet ===
            negatives = []
            for facet in facets[:3]:
                neg_prompt = f"""Give 2-3 examples that could be CONFUSED with "{concept_term}" ({facet} aspect) but are actually NOT "{concept_term}".

These should be from sibling categories: {siblings_str}

They should sound similar but belong to a different category. JSON only:
{{"examples": ["example1", "example2"]}}"""

                neg_response = self._generate_text(neg_prompt, max_tokens=400)
                neg_data = self._extract_json(neg_response)
                if neg_data and "examples" in neg_data:
                    negatives.extend(neg_data["examples"][:3])

            # === STEP 4: Generate the full MELD with collected examples ===
            full_prompt = MELD_GENERATION_PROMPT.format(
                hierarchy_context=context_str,
                concept_term=concept_term,
                parent_concepts=", ".join(parent_concepts or []) or "None specified",
                sibling_concepts=siblings_str,
                previous_feedback=previous_feedback or "None - first generation attempt",
            )

            # Add the pre-generated examples as context
            if positives or negatives:
                full_prompt += f"""

I've already identified some examples. Include these and add more as needed:

Positive examples to include: {json.dumps(positives[:6])}
Near-miss negatives to include: {json.dumps(negatives[:6])}"""

            full_response = self._generate_text(full_prompt, max_tokens=2000)
            meld_data = self._extract_json(full_response)

            if not meld_data:
                return {}, "Failed to parse MELD JSON"

            # For L1/L2, use the predefined definition (from ontologist) instead of generated one
            if predefined_definition:
                meld_data["definition"] = predefined_definition

            # Ensure we have the pre-generated examples included
            if positives and len(meld_data.get("positive_examples", [])) < 6:
                existing = meld_data.get("positive_examples", [])
                meld_data["positive_examples"] = list(set(existing + positives))[:10]
            if negatives and len(meld_data.get("negative_examples", [])) < 6:
                existing = meld_data.get("negative_examples", [])
                meld_data["negative_examples"] = list(set(existing + negatives))[:10]

            return meld_data, ""

        except Exception as e:
            return {}, f"Generation error: {e}"


# =============================================================================
# MINISTRAL JUDGE (Bulk Filter - 91.8% accuracy)
# =============================================================================

MINISTRAL_JUDGE_PROMPT_GATEKEEPER = """You are evaluating whether a MELD (training data for concept detection) meets quality standards.

{hierarchy_context}

## MELD to Review

**Concept:** {term}
**Definition:** {definition}

**Positive Examples:**
{positive_examples}

**Negative Examples:**
{negative_examples}

## Evaluation Criteria (Level-Appropriate)

1. **Definition Quality**: Does the definition match the expected granularity for this level? (See guidance above)
2. **MECE Compliance**: Is the concept distinct from its siblings at this level?
3. **Positive Examples**: Do they genuinely exemplify the concept and match the expected scope?
4. **Negative Examples**: Are they clearly NOT examples? Do they help distinguish from siblings?
5. **Count**: At least 5 positive and 5 negative examples?

## Decision

Considering the LEVEL and EXPECTED GRANULARITY, is this MELD suitable for training a concept lens?

Answer only: YES or NO

If NO, briefly explain why (one sentence).
"""

MINISTRAL_JUDGE_PROMPT_ANNOTATOR = """You are reviewing a MELD (training data for concept detection) to check structural validity and note any divergence from your own understanding.

{hierarchy_context}

## MELD to Review

**Concept:** {term}
**Definition:** {definition}

**Positive Examples:**
{positive_examples}

**Negative Examples:**
{negative_examples}

## Structural Checks (can fail)

1. **Example Count**: At least 5 positive and 5 negative examples?
2. **MECE Compliance**: Do examples clearly avoid overlap with sibling concepts?
3. **Internal Consistency**: Do the examples match the given definition?

## Worldview Annotation (note but don't fail)

If you would conceptualize this differently, note your perspective. This is metadata, not a rejection reason.

## Response Format

Return JSON:
```json
{{
  "structural_pass": true/false,
  "structural_issues": ["issue1", ...] or [],
  "worldview_notes": ["Your perspective on definition breadth, example choices, etc."] or [],
  "divergence_score": 0.0-1.0 (0=fully aligned with your view, 1=very different)
}}
```
"""


class MinistralJudge:
    """Ministral-8B judge for bulk filtering (91.8% accuracy)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        """Load Ministral model."""
        if self.loaded:
            return

        logger.info("Loading Ministral-8B judge...")
        loader = CandidateLoader()
        candidate = MODEL_CANDIDATES["ministral-8b"]
        self.model, self.tokenizer, _ = loader.load(candidate)
        self.loaded = True
        logger.info("Ministral-8B loaded")

    def unload(self):
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.loaded = False
            torch.cuda.empty_cache()

    def review(
        self,
        meld_data: Dict,
        hierarchy_context: Optional[HierarchyContext] = None,
        predefined_definition: bool = False,
        mode: str = "gatekeeper",
    ) -> Tuple[ReviewResult, Optional[Dict]]:
        """Review a MELD and return pass/fail with feedback.

        Args:
            meld_data: The MELD data to review
            hierarchy_context: Optional context about the concept's level in the ontology
            predefined_definition: If True, the definition was provided by ontologist
            mode: "gatekeeper" (enforces structure + worldview) or "annotator" (enforces structure, annotates worldview)

        Returns:
            Tuple of (ReviewResult, worldview_metadata or None)
            In annotator mode, worldview_metadata contains judge's perspective without rejecting
        """
        if not self.loaded:
            self.load()

        # Format examples
        pos_examples = "\n".join(f"- {ex}" for ex in meld_data.get("positive_examples", [])[:6])
        neg_examples = "\n".join(f"- {ex}" for ex in meld_data.get("negative_examples", [])[:6])

        # Generate hierarchy context section
        if hierarchy_context:
            context_str = hierarchy_context.to_reviewer_context()
        else:
            context_str = "## Context\nThis is a specific concept (no hierarchy level specified)."

        # Choose prompt based on mode
        if mode == "annotator":
            prompt = MINISTRAL_JUDGE_PROMPT_ANNOTATOR.format(
                hierarchy_context=context_str,
                term=meld_data.get("term", "Unknown"),
                definition=meld_data.get("definition", "No definition"),
                positive_examples=pos_examples or "None provided",
                negative_examples=neg_examples or "None provided",
            )
            max_tokens = 300  # More tokens for JSON response
        else:
            # Gatekeeper mode
            prompt_template = MINISTRAL_JUDGE_PROMPT_GATEKEEPER
            if predefined_definition and hierarchy_context and hierarchy_context.level <= 2:
                context_str += "\n**NOTE**: The definition was provided by the ontologist and is PRE-VALIDATED. Do NOT reject based on definition breadth - only evaluate examples quality."

            prompt = prompt_template.format(
                hierarchy_context=context_str,
                term=meld_data.get("term", "Unknown"),
                definition=meld_data.get("definition", "No definition"),
                positive_examples=pos_examples or "None provided",
                negative_examples=neg_examples or "None provided",
            )
            max_tokens = 100

        # Apply chat template and generate
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Parse response based on mode
        worldview_metadata = None

        if mode == "annotator":
            # Try to parse JSON response
            try:
                # Extract JSON from response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "{" in response:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                else:
                    json_str = response

                result = json.loads(json_str)
                passed = result.get("structural_pass", True)
                feedback = "; ".join(result.get("structural_issues", []))
                worldview_metadata = {
                    "worldview_notes": result.get("worldview_notes", []),
                    "divergence_score": result.get("divergence_score", 0.0),
                    "judge_model": "ministral-8b",
                }
            except (json.JSONDecodeError, KeyError):
                # Fallback: assume structural pass, note parsing issue
                passed = True
                feedback = ""
                worldview_metadata = {
                    "worldview_notes": [f"Judge response (unparsed): {response[:200]}"],
                    "divergence_score": 0.5,
                    "judge_model": "ministral-8b",
                }
        else:
            # Gatekeeper mode - simple YES/NO parsing
            response_lower = response.lower()
            passed = response_lower.startswith("yes")
            feedback = ""
            if not passed and len(response) > 3:
                feedback = response[3:].strip().lstrip(":").lstrip(",").strip()

        return ReviewResult(
            stage=ReviewStage.MINISTRAL_REVIEW,
            passed=passed,
            feedback=feedback,
            confidence=0.92,
            reviewer_model="ministral-8b",
        ), worldview_metadata


# =============================================================================
# CLAUDE REVIEW (Quality Gate - catches the ~8% Ministral misses)
# =============================================================================

CLAUDE_REVIEW_PROMPT = """You are a senior ontologist performing quality review on a MELD (training data for AI concept detection).

This MELD has passed initial automated validation and a preliminary judge review. Your job is to catch subtle issues that automated systems miss.

## MELD Under Review

```json
{meld_json}
```

## Review Checklist

1. **Semantic Accuracy**
   - Does the definition capture the true essence of the concept?
   - Are examples semantically correct (not just superficially related)?

2. **Discrimination Quality**
   - Would negative examples actually be confused with the concept?
   - Are contrast_concepts meaningful for training?

3. **Safety Considerations**
   - Are safety_tags accurate?
   - Any hidden risks not flagged?

4. **Training Effectiveness**
   - Would this data train an effective lens?
   - Any systematic biases in examples?

## Decision

Respond with JSON:
```json
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", "issue2"],  // empty if approved
  "suggestions": ["suggestion1"],   // optional improvements
  "safety_concerns": []             // any safety issues found
}}
```
"""


class ClaudeReviewer:
    """Claude API reviewer for quality gate (catches Ministral's ~8% errors)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Claude review will be skipped")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def review(self, meld_data: Dict) -> ReviewResult:
        """Thorough review of a MELD."""
        if self.client is None:
            logger.warning("Skipping Claude review - no API key available")
            return ReviewResult(
                stage=ReviewStage.CLAUDE_REVIEW,
                passed=True,  # Pass through to human review when no API key
                warnings=["Claude review skipped - ANTHROPIC_API_KEY not set"],
                confidence=0.0,
                reviewer_model=self.model,
            )

        prompt = CLAUDE_REVIEW_PROMPT.format(
            meld_json=json.dumps(meld_data, indent=2)
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)

            return ReviewResult(
                stage=ReviewStage.CLAUDE_REVIEW,
                passed=result.get("approved", False),
                errors=result.get("issues", []),
                warnings=result.get("suggestions", []),
                feedback="; ".join(result.get("issues", [])),
                confidence=result.get("confidence", 0.5),
                reviewer_model=self.model,
            )

        except Exception as e:
            return ReviewResult(
                stage=ReviewStage.CLAUDE_REVIEW,
                passed=False,
                errors=[f"Review failed: {e}"],
                confidence=0.0,
                reviewer_model=self.model,
            )


# =============================================================================
# HUMAN REVIEW QUEUE
# =============================================================================

@dataclass
class HumanReviewItem:
    """Item in the human review queue."""
    concept_term: str
    meld_data: Dict
    protection_level: str
    review_history: List[Dict]
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 0  # Higher = more urgent

    def to_dict(self) -> Dict:
        return {
            "concept_term": self.concept_term,
            "meld_data": self.meld_data,
            "protection_level": self.protection_level,
            "review_history": self.review_history,
            "submitted_at": self.submitted_at,
            "priority": self.priority,
        }


class HumanReviewQueue:
    """Manages the queue of MELDs awaiting human review."""

    def __init__(self, queue_dir: Path = None):
        self.queue_dir = queue_dir or PROJECT_ROOT / "melds" / "human_review_queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def add(self, state: MeldReviewState):
        """Add a MELD to the human review queue."""
        # Determine priority based on protection level
        priority_map = {
            ProtectionLevel.CRITICAL: 100,
            ProtectionLevel.PROTECTED: 75,
            ProtectionLevel.ELEVATED: 50,
            ProtectionLevel.STANDARD: 0,
        }

        # Convert review history to JSON-serializable format
        review_history = []
        for r in state.review_history:
            review_dict = asdict(r)
            # Convert enum to string
            review_dict["stage"] = r.stage.value
            review_history.append(review_dict)

        item = HumanReviewItem(
            concept_term=state.concept_term,
            meld_data=state.meld_data,
            protection_level=state.protection_level.name,
            review_history=review_history,
            priority=priority_map.get(state.protection_level, 0),
        )

        # Save to queue
        filename = f"{state.concept_term.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.queue_dir / filename

        with open(filepath, 'w') as f:
            json.dump(item.to_dict(), f, indent=2)

        logger.info(f"Added to human review queue: {filepath}")
        return filepath

    def list_pending(self) -> List[Path]:
        """List all pending review items, sorted by priority."""
        items = []
        for f in self.queue_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
            items.append((data.get("priority", 0), f))

        # Sort by priority (descending)
        items.sort(key=lambda x: -x[0])
        return [f for _, f in items]

    def approve(self, filepath: Path, reviewer: str = "human"):
        """Approve a MELD and move to approved directory."""
        approved_dir = PROJECT_ROOT / "melds" / "approved"
        approved_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath) as f:
            data = json.load(f)

        # Add approval record
        data["approved_by"] = reviewer
        data["approved_at"] = datetime.now().isoformat()

        # Move to approved
        new_path = approved_dir / filepath.name
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=2)

        filepath.unlink()
        logger.info(f"Approved: {filepath.name} -> {new_path}")
        return new_path

    def reject(self, filepath: Path, reason: str, reviewer: str = "human"):
        """Reject a MELD and move to rejected directory."""
        rejected_dir = PROJECT_ROOT / "melds" / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath) as f:
            data = json.load(f)

        # Add rejection record
        data["rejected_by"] = reviewer
        data["rejected_at"] = datetime.now().isoformat()
        data["rejection_reason"] = reason

        # Move to rejected
        new_path = rejected_dir / filepath.name
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=2)

        filepath.unlink()
        logger.info(f"Rejected: {filepath.name} -> {new_path}")
        return new_path


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class MeldReviewPipeline:
    """
    Orchestrates the full MELD review pipeline:

    Generation -> Validation -> Ministral -> Claude -> Human (if needed)

    Generation uses LOCAL model (target model being mapped).
    Only Tier 2 (ClaudeReviewer) uses the Claude API.

    For L1 concepts (broad taxonomic categories), set skip_ministral=True
    to bypass the bulk filter - there are too few to benefit from bulk
    filtering, and abstract L1 concepts are hard to generate near-miss
    negatives for.
    """

    def __init__(
        self,
        pack_dir: Path = None,
        max_generation_attempts: int = 3,
        generator_model_id: str = "gemma-3-4b-it",
        skip_ministral: bool = False,
        judge_mode: str = "gatekeeper",
    ):
        self.pack_dir = pack_dir
        self.max_attempts = max_generation_attempts
        self.generator_model_id = generator_model_id
        self.skip_ministral = skip_ministral
        self.judge_mode = judge_mode  # "gatekeeper" or "annotator"

        # Load policy and hierarchy if pack specified
        if pack_dir:
            self.policy, _ = load_pack_policy(pack_dir)
            self.hierarchy = build_hierarchy_index(pack_dir)
        else:
            self.policy = get_default_policy()
            self.hierarchy = HierarchyIndex()

        # Initialize components (lazy load expensive ones)
        # Generator uses LOCAL model (same as target model)
        self.generator = MeldGenerator(model_id=generator_model_id)
        self.ministral = MinistralJudge()
        self.claude_reviewer = ClaudeReviewer()
        self.human_queue = HumanReviewQueue()

    def process_concept(
        self,
        concept_term: str,
        parent_concepts: List[str] = None,
        sibling_concepts: List[str] = None,
        hierarchy_context: Optional[HierarchyContext] = None,
        predefined_definition: Optional[str] = None,
    ) -> MeldReviewState:
        """
        Process a single concept through the full pipeline.

        Args:
            concept_term: The concept to generate a MELD for
            parent_concepts: Parent concepts in the hierarchy
            sibling_concepts: Sibling concepts at the same level
            hierarchy_context: Ontological context (level, MECE rules, etc.)
                              If not provided, defaults to L4 (specific concept)
            predefined_definition: For L1/L2, use existing definition from ontologist

        Returns the final state with review history.
        """
        # Create default context if not provided
        if hierarchy_context is None:
            hierarchy_context = HierarchyContext(
                level=4,  # Default to specific concept
                parent_concept=parent_concepts[0] if parent_concepts else None,
                sibling_concepts=sibling_concepts or [],
            )

        state = MeldReviewState(
            concept_term=concept_term,
            meld_data={},
            max_attempts=self.max_attempts,
        )

        # === STAGE 1: GENERATION ===
        while state.generation_attempts < state.max_attempts:
            state.generation_attempts += 1
            logger.info(f"[{concept_term}] Generation attempt {state.generation_attempts}/{state.max_attempts}")

            feedback = state.get_feedback_summary() if state.generation_attempts > 1 else ""

            meld_data, error = self.generator.generate(
                concept_term=concept_term,
                parent_concepts=parent_concepts,
                sibling_concepts=sibling_concepts,
                previous_feedback=feedback,
                hierarchy_context=hierarchy_context,
                predefined_definition=predefined_definition,
            )

            if error:
                state.add_review(ReviewResult(
                    stage=ReviewStage.GENERATION,
                    passed=False,
                    errors=[error],
                ))
                continue

            state.meld_data = meld_data
            state.add_review(ReviewResult(
                stage=ReviewStage.GENERATION,
                passed=True,
                reviewer_model=self.generator_model_id,
            ))

            # === STAGE 2: VALIDATION ===
            logger.info(f"[{concept_term}] Running automated validation")
            validation_result = self._validate(state)
            state.add_review(validation_result)

            if not validation_result.passed:
                logger.info(f"[{concept_term}] Validation failed, regenerating...")
                continue

            # === STAGE 3: MINISTRAL REVIEW ===
            if not self.skip_ministral:
                logger.info(f"[{concept_term}] Ministral judge review (mode: {self.judge_mode})")
                ministral_result, worldview_metadata = self.ministral.review(
                    state.meld_data,
                    hierarchy_context,
                    predefined_definition=bool(predefined_definition),
                    mode=self.judge_mode,
                )
                state.add_review(ministral_result)

                # Store worldview metadata in annotator mode
                if worldview_metadata:
                    state.worldview_metadata = worldview_metadata
                    if worldview_metadata.get("divergence_score", 0) > 0.5:
                        logger.info(f"[{concept_term}] Judge notes divergence: {worldview_metadata.get('worldview_notes', [])[:1]}")

                if not ministral_result.passed:
                    logger.info(f"[{concept_term}] Ministral rejected: {ministral_result.feedback}")
                    continue
            else:
                logger.info(f"[{concept_term}] Skipping Ministral (L1 mode)")

            # === STAGE 4: CLAUDE REVIEW (API - no GPU memory needed) ===
            logger.info(f"[{concept_term}] Claude quality review")

            claude_result = self.claude_reviewer.review(state.meld_data)
            state.add_review(claude_result)

            if not claude_result.passed:
                logger.info(f"[{concept_term}] Claude rejected: {claude_result.feedback}")
                continue

            # === PASSED ALL AUTOMATED REVIEWS ===
            logger.info(f"[{concept_term}] Passed all automated reviews")
            break

        # Check if we exhausted attempts
        if state.generation_attempts >= state.max_attempts:
            last_review = state.review_history[-1] if state.review_history else None
            if last_review and not last_review.passed:
                state.current_stage = ReviewStage.REJECTED
                logger.warning(f"[{concept_term}] Exhausted generation attempts, rejected")
                return state

        # === STAGE 5: HUMAN REVIEW (if needed) ===
        if state.protection_level > ProtectionLevel.STANDARD:
            logger.info(f"[{concept_term}] Queued for human review (protection: {state.protection_level.name})")
            state.current_stage = ReviewStage.HUMAN_REVIEW
            self.human_queue.add(state)
        else:
            state.current_stage = ReviewStage.APPROVED
            logger.info(f"[{concept_term}] Auto-approved (STANDARD protection)")

        return state

    def _validate(self, state: MeldReviewState) -> ReviewResult:
        """Run automated validation on MELD data."""
        # Convert meld_data to the format validate_meld expects
        meld_request = {
            "meld_request_id": f"generated/{state.concept_term}",
            "candidates": [state.meld_data],
        }

        # Write temp file for validation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(meld_request, f)
            temp_path = Path(f.name)

        try:
            result = validate_meld_file(temp_path, self.policy, self.hierarchy)
            state.protection_level = result.protection_level

            return ReviewResult(
                stage=ReviewStage.VALIDATION,
                passed=result.is_valid,
                errors=result.errors,
                warnings=result.warnings,
                feedback="; ".join(result.errors) if result.errors else "",
            )
        finally:
            temp_path.unlink()

    def process_batch(self, concepts: List[Dict]) -> List[MeldReviewState]:
        """Process a batch of concepts."""
        results = []
        for i, concept in enumerate(concepts):
            logger.info(f"Processing {i+1}/{len(concepts)}: {concept.get('term', 'Unknown')}")
            state = self.process_concept(
                concept_term=concept.get("term", concept.get("id", "Unknown")),
                parent_concepts=concept.get("parent_concepts", []),
                sibling_concepts=concept.get("sibling_concepts", []),
            )
            results.append(state)
        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MELD Review Pipeline - Three-tier review system"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate and review a single concept
    gen_parser = subparsers.add_parser("generate", help="Generate and review MELD for a concept")
    gen_parser.add_argument("--concept", required=True, help="Concept term")
    gen_parser.add_argument("--parents", nargs="*", default=[], help="Parent concepts")
    gen_parser.add_argument("--siblings", nargs="*", default=[], help="Sibling concepts")
    gen_parser.add_argument("--pack-dir", type=Path, help="Concept pack directory for policy")
    gen_parser.add_argument("--judge-mode", choices=["gatekeeper", "annotator"], default="gatekeeper",
                           help="gatekeeper=enforce worldview, annotator=capture divergence as metadata")

    # Review existing MELD file
    review_parser = subparsers.add_parser("review", help="Review an existing MELD file")
    review_parser.add_argument("--meld-file", type=Path, required=True, help="Path to MELD JSON")
    review_parser.add_argument("--pack-dir", type=Path, help="Concept pack directory for policy")
    review_parser.add_argument("--judge-mode", choices=["gatekeeper", "annotator"], default="gatekeeper",
                              help="gatekeeper=enforce worldview, annotator=capture divergence as metadata")

    # Process university pack
    batch_parser = subparsers.add_parser("batch", help="Process a university pack")
    batch_parser.add_argument("--pack-dir", type=Path, required=True, help="University pack directory")
    batch_parser.add_argument("--limit", type=int, help="Limit number of concepts to process")
    batch_parser.add_argument("--judge-mode", choices=["gatekeeper", "annotator"], default="gatekeeper",
                             help="gatekeeper=enforce worldview, annotator=capture divergence as metadata")

    # Manage human review queue
    queue_parser = subparsers.add_parser("queue", help="Manage human review queue")
    queue_parser.add_argument("--list", action="store_true", help="List pending reviews")
    queue_parser.add_argument("--approve", type=Path, help="Approve a MELD")
    queue_parser.add_argument("--reject", type=Path, help="Reject a MELD")
    queue_parser.add_argument("--reason", help="Rejection reason")

    args = parser.parse_args()

    if args.command == "generate":
        pipeline = MeldReviewPipeline(pack_dir=args.pack_dir, judge_mode=args.judge_mode)
        state = pipeline.process_concept(
            concept_term=args.concept,
            parent_concepts=args.parents,
            sibling_concepts=args.siblings,
        )
        print(f"\n{'='*60}")
        print(f"Concept: {state.concept_term}")
        print(f"Final Stage: {state.current_stage.value}")
        print(f"Protection Level: {state.protection_level.name}")
        print(f"Generation Attempts: {state.generation_attempts}")
        if state.worldview_metadata:
            print(f"Divergence Score: {state.worldview_metadata.get('divergence_score', 0):.2f}")
            for note in state.worldview_metadata.get('worldview_notes', [])[:2]:
                print(f"  Worldview note: {note}")
        print(f"\nReview History:")
        for r in state.review_history:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{r.stage.value}] {status} - {r.reviewer_model or 'system'}")
            if r.errors:
                for e in r.errors[:3]:
                    print(f"    - {e}")

    elif args.command == "review":
        pipeline = MeldReviewPipeline(pack_dir=args.pack_dir, judge_mode=args.judge_mode)
        with open(args.meld_file) as f:
            meld_data = json.load(f)

        state = MeldReviewState(
            concept_term=meld_data.get("term", "Unknown"),
            meld_data=meld_data,
        )

        # Run through review stages
        validation_result = pipeline._validate(state)
        print(f"Validation: {'PASS' if validation_result.passed else 'FAIL'}")

        if validation_result.passed:
            ministral_result, worldview_metadata = pipeline.ministral.review(meld_data)
            print(f"Ministral: {'PASS' if ministral_result.passed else 'FAIL'}")
            if worldview_metadata:
                print(f"  Divergence score: {worldview_metadata.get('divergence_score', 0):.2f}")
                for note in worldview_metadata.get('worldview_notes', [])[:2]:
                    print(f"  Note: {note}")

            if ministral_result.passed:
                pipeline.ministral.unload()
                claude_result = pipeline.claude_reviewer.review(meld_data)
                print(f"Claude: {'PASS' if claude_result.passed else 'FAIL'}")

    elif args.command == "batch":
        # Load concepts from pack
        concepts = []
        for json_file in args.pack_dir.glob("*.json"):
            if json_file.name == "pack.json":
                continue
            with open(json_file) as f:
                data = json.load(f)
            if "id" in data or "term" in data:
                concepts.append(data)

        if args.limit:
            concepts = concepts[:args.limit]

        print(f"Processing {len(concepts)} concepts from {args.pack_dir}")
        pipeline = MeldReviewPipeline(pack_dir=args.pack_dir, judge_mode=args.judge_mode)
        results = pipeline.process_batch(concepts)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        approved = sum(1 for r in results if r.current_stage == ReviewStage.APPROVED)
        human_review = sum(1 for r in results if r.current_stage == ReviewStage.HUMAN_REVIEW)
        rejected = sum(1 for r in results if r.current_stage == ReviewStage.REJECTED)
        print(f"Approved: {approved}")
        print(f"Human Review: {human_review}")
        print(f"Rejected: {rejected}")

    elif args.command == "queue":
        queue = HumanReviewQueue()

        if args.list:
            pending = queue.list_pending()
            print(f"Pending reviews: {len(pending)}")
            for p in pending:
                with open(p) as f:
                    data = json.load(f)
                print(f"  [{data.get('protection_level', '?')}] {data.get('concept_term', '?')} - {p.name}")

        elif args.approve:
            queue.approve(args.approve)

        elif args.reject:
            if not args.reason:
                print("Error: --reason required for rejection")
                sys.exit(1)
            queue.reject(args.reject, args.reason)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
