"""
Training data generation for concept classification.

Phase 4 methodology: Generate diverse prompts for training binary classifiers.
"""

from typing import List, Dict
import random


def generate_definition_prompt(concept: str) -> str:
    """
    Generate a definitional prompt for a concept.

    Args:
        concept: Concept name

    Returns:
        Definitional prompt string

    Example:
        >>> generate_definition_prompt("person")
        'What is person?'
    """
    return f"What is {concept}?"


def generate_relationship_prompts(
    concept: str,
    relationships: List[str],
    n_samples: int = 5
) -> List[str]:
    """
    Generate relationship-based prompts for a concept.

    Args:
        concept: Concept name
        relationships: List of related concept names
        n_samples: Number of prompts to generate

    Returns:
        List of relationship prompts

    Example:
        >>> generate_relationship_prompts("person", ["human", "individual"], n_samples=2)
        ['person is related to human.', 'person is related to individual.']
    """
    if not relationships:
        return []

    sampled = random.sample(relationships, min(n_samples, len(relationships)))
    return [f"{concept} is related to {rel}." for rel in sampled]


def generate_negative_prompts(
    concept: str,
    negative_concepts: List[str],
    n_samples: int = 10
) -> List[str]:
    """
    Generate negative prompts using semantically distant concepts.

    Phase 4 uses neutral concepts (semantic distance ≥ 5 hops in WordNet)
    to train classifiers that distinguish concept from unrelated content.

    Args:
        concept: Target concept name
        negative_concepts: List of semantically distant concept names
        n_samples: Number of negative prompts to generate

    Returns:
        List of negative definitional prompts

    Example:
        >>> generate_negative_prompts("person", ["object", "substance"], n_samples=2)
        ['What is object?', 'What is substance?']
    """
    if not negative_concepts:
        return []

    sampled = random.sample(negative_concepts, min(n_samples, len(negative_concepts)))
    return [f"What is {neg}?" for neg in sampled]


def create_training_dataset(
    concept: str,
    concept_info: Dict,
    negative_pool: List[str],
    n_positives: int = 10,
    n_negatives: int = 10
) -> tuple[List[str], List[int]]:
    """
    Create a balanced training dataset for binary classification.

    Args:
        concept: Target concept name
        concept_info: Dict with 'related' key containing related concepts
        negative_pool: Pool of negative concept names
        n_positives: Number of positive examples (definitions + relationships)
        n_negatives: Number of negative examples

    Returns:
        (prompts, labels) where labels are 1 for positive, 0 for negative

    Example:
        >>> concept_info = {"related": ["human", "individual"]}
        >>> prompts, labels = create_training_dataset(
        ...     "person", concept_info, ["object", "thing"], n_positives=5, n_negatives=5
        ... )
        >>> len(prompts), sum(labels)
        (10, 5)
    """
    prompts = []
    labels = []

    # Positive examples: definitions + relationships
    prompts.append(generate_definition_prompt(concept))
    labels.append(1)

    related = concept_info.get('related', [])
    if related:
        rel_prompts = generate_relationship_prompts(
            concept, related, n_samples=n_positives - 1
        )
        prompts.extend(rel_prompts)
        labels.extend([1] * len(rel_prompts))

    # Negative examples: distant concepts
    neg_prompts = generate_negative_prompts(concept, negative_pool, n_samples=n_negatives)
    prompts.extend(neg_prompts)
    labels.extend([0] * len(neg_prompts))

    return prompts, labels
