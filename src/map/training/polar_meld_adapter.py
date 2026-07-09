"""
Adapter to convert Polar MELDs to the format expected by existing training infrastructure.

Maps polar MELD fields to SUMO-style concept dicts so we can reuse
DualAdaptiveTrainer and the full training pipeline.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def polar_meld_to_concept(meld_data: Dict, pole: str = "positive") -> Dict:
    """
    Convert a polar MELD to a SUMO-style concept dict for training.

    Maps polar MELD fields to the format expected by create_sumo_training_dataset():
    - examples → positive_examples (used directly as positive prompts)
    - OPPOSITE pole examples + confusables → negative_examples (hard negatives)

    This ensures the lens learns to discriminate between positive and negative
    poles, not just between the pole and unrelated confusables.

    Args:
        meld_data: Full polar MELD dict (with "node" and "polar_meld" keys)
        pole: Which pole to convert ("positive" or "negative")

    Returns:
        Concept dict compatible with existing training infrastructure
    """
    node = meld_data["node"]
    pm = meld_data["polar_meld"]
    pole_data = pm["poles"][pole]
    opposite_pole = "negative" if pole == "positive" else "positive"
    opposite_pole_data = pm["poles"][opposite_pole]
    gen_ctx = pm.get("_generation_context", {})

    # Get examples for this pole
    # New format: separate "descriptions" and "instances" fields
    # Old format: single "examples" field
    # Merge both for backwards compatibility
    descriptions = pole_data.get("descriptions", [])
    instances = pole_data.get("instances", [])
    examples = pole_data.get("examples", [])

    # Combine all example types - instances are more valuable so put them first
    all_examples = instances + descriptions + examples

    # CRITICAL: Use opposite pole examples as PRIMARY hard negatives
    # This teaches the lens to discriminate positive from negative poles
    opp_descriptions = opposite_pole_data.get("descriptions", [])
    opp_instances = opposite_pole_data.get("instances", [])
    opp_examples = opposite_pole_data.get("examples", [])
    opposite_examples = opp_instances + opp_descriptions + opp_examples

    # Also include confusables as additional negatives
    confusables = pole_data.get("confusables", {}).get("examples", [])

    # Convert examples to prompt format
    # The existing training expects full prompts, not just descriptions
    def to_prompts(descriptions: List[str]) -> List[str]:
        templates = [
            "Scenario: {text}",
            "Consider this: {text}",
            "Example: {text}",
            "{text}",
        ]
        prompts = []
        for desc in descriptions:
            template = random.choice(templates)
            prompts.append(template.format(text=desc))
        return prompts

    # Map to SUMO-style concept
    concept = {
        # Core identity
        "sumo_term": f"{pm['term']}_{pole}",  # Disambiguate positive vs negative
        "original_term": pm["term"],
        "pole": pole,

        # Definition from this pole
        "definition": pole_data.get("definition", pm.get("definition", "")),
        "sumo_definition": pole_data.get("definition", ""),

        # Hierarchy info - map "level" to "layer" for compatibility
        "layer": node.get("level", 1),
        "parent_concepts": [gen_ctx.get("parent")] if gen_ctx.get("parent") else [],
        "category_children": gen_ctx.get("children", []),

        # MELD examples - these are used directly by create_sumo_training_dataset
        # Positive = examples of THIS pole (descriptions + instances)
        # Negative = examples of OPPOSITE pole (primary) + confusables (secondary)
        # This ensures the lens discriminates between poles, not just concept vs confusables
        "positive_examples": to_prompts(all_examples),
        "negative_examples": to_prompts(opposite_examples) + to_prompts(confusables),

        # Track what we used for debugging
        "opposite_pole_examples": opposite_examples,

        # Also keep raw examples for reference (all types merged)
        "examples": all_examples,
        "descriptions": descriptions,
        "instances": instances,
        "confusables": confusables,
        "confusable_sources": pole_data.get("confusables", {}).get("sourced_from_positives_of", []),

        # Training hints from polar MELD
        "training_hints": pm.get("training_hints", {}),

        # Safety metadata
        "safety_tags": pm.get("safety_tags", {}),

        # For compatibility - we don't have WordNet mappings
        "synsets": [],
        "canonical_synset": None,
        "lemmas": [pm["term"]],

        # Source tracking
        "_source": "polar_meld",
        "_node_id": node.get("id", ""),
    }

    return concept


def load_polar_melds_as_concepts(
    meld_dir: Path,
    level: int,
    pole: str = "positive"
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load polar MELDs and convert to concept list and map.

    Args:
        meld_dir: Directory containing polar MELDs (with L1/, L2/, L3/ subdirs)
        level: Which level to load
        pole: Which pole to load ("positive" or "negative")

    Returns:
        Tuple of (concept_list, concept_map) compatible with load_layer_concepts()
    """
    level_dir = meld_dir / f"L{level}"
    concepts = []
    concept_map = {}

    for meld_file in sorted(level_dir.glob("*.json")):
        try:
            meld_data = json.loads(meld_file.read_text())
            concept = polar_meld_to_concept(meld_data, pole)
            concepts.append(concept)
            concept_map[concept["sumo_term"]] = concept
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {meld_file}: {e}")

    return concepts, concept_map


def generate_polar_training_prompts(
    concept: Dict,
    n_positive: int = 10,
    n_negative: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Generate training prompts from a polar MELD concept.

    Unlike SUMO training which generates prompts from definitions and relationships,
    polar MELDs already have example scenarios we can use directly.

    Args:
        concept: Converted polar MELD concept dict
        n_positive: Number of positive prompts to generate
        n_negative: Number of negative prompts to generate

    Returns:
        Tuple of (positive_prompts, negative_prompts)
    """
    examples = concept.get("examples", [])
    confusables = concept.get("confusables", [])

    if not examples or not confusables:
        return [], []

    # Prompt templates for variety
    templates = [
        "Scenario: {text}",
        "Consider: {text}",
        "Example: {text}",
        "{text}",
        "Situation: {text}",
    ]

    def generate_prompts(source_examples: List[str], n: int) -> List[str]:
        prompts = []
        for _ in range(n):
            example = random.choice(source_examples)
            template = random.choice(templates)
            prompts.append(template.format(text=example))
        return prompts

    positive_prompts = generate_prompts(examples, n_positive)
    negative_prompts = generate_prompts(confusables, n_negative)

    return positive_prompts, negative_prompts


def build_polar_negative_pool(
    concepts: List[Dict],
    target_concept: Dict,
) -> List[Dict]:
    """
    Build negative pool for a concept from its confusables and siblings.

    For polar MELDs, the confusables ARE the hard negatives, but we can
    also include other concepts from the same level as additional negatives.

    Args:
        concepts: All concepts at this level
        target_concept: The concept we're training

    Returns:
        List of concepts to use as negatives
    """
    negatives = []
    target_term = target_concept["sumo_term"]

    # Other concepts at same level (excluding self)
    for c in concepts:
        if c["sumo_term"] != target_term:
            negatives.append(c)

    return negatives
