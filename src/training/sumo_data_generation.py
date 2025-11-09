"""
SUMO-aware training data generation for hierarchical concept classification.

Extends the WordNet-based approach to support SUMO category relationships.
"""

from typing import List, Dict, Optional
import random
from nltk.corpus import wordnet as wn


def generate_category_relationship_prompts(
    concept: str,
    category_children: List[str],
    all_concepts_map: Dict[str, Dict],
    n_samples: int = 5
) -> List[str]:
    """
    Generate relationship prompts using SUMO category hierarchy.

    SUMO category_children are analogous to WordNet hyponyms (more specific categories).

    Args:
        concept: SUMO category name
        category_children: List of child SUMO categories
        all_concepts_map: Map of all SUMO categories to their definitions
        n_samples: Number of prompts to generate

    Returns:
        List of relationship prompts

    Example:
        >>> generate_category_relationship_prompts(
        ...     "Physical",
        ...     ["Object", "Process", "Collection"],
        ...     {"Object": {"definition": "a physical thing"}, ...},
        ...     n_samples=2
        ... )
        ['Physical includes the subcategory Object.', 'Physical includes the subcategory Process.']
    """
    if not category_children:
        return []

    prompts = []
    sampled = random.sample(category_children, min(n_samples, len(category_children)))

    for child in sampled:
        # Try multiple prompt formats for robustness
        templates = [
            f"{concept} includes the subcategory {child}.",
            f"{child} is a type of {concept}.",
            f"{concept} has subcategory {child}.",
        ]
        prompts.append(random.choice(templates))

    return prompts


def generate_wordnet_relationship_prompts(
    canonical_synset: str,
    n_samples: int = 5
) -> List[str]:
    """
    Generate relationship prompts using WordNet relationships from canonical synset.

    Args:
        canonical_synset: WordNet synset ID (e.g., "physical_entity.n.01")
        n_samples: Number of prompts to generate

    Returns:
        List of relationship prompts
    """
    if not canonical_synset:
        return []

    try:
        synset = wn.synset(canonical_synset)
    except Exception:
        return []

    # Collect all relationships
    relationships = []

    # Hypernyms (broader categories)
    for hyp in synset.hypernyms():
        relationships.append((hyp.lemma_names()[0], "is a type of"))

    # Hyponyms (more specific types)
    for hypo in synset.hyponyms()[:10]:  # Limit to avoid explosion
        relationships.append((hypo.lemma_names()[0], "has type"))

    # Meronyms (parts)
    for mero in synset.member_meronyms() + synset.part_meronyms():
        relationships.append((mero.lemma_names()[0], "has part"))

    # Holonyms (wholes)
    for holo in synset.member_holonyms() + synset.part_holonyms():
        relationships.append((holo.lemma_names()[0], "is part of"))

    if not relationships:
        return []

    # Sample and format
    sampled = random.sample(relationships, min(n_samples, len(relationships)))
    concept_name = synset.lemma_names()[0].replace('_', ' ')

    prompts = []
    for related_concept, relation_type in sampled:
        related_concept = related_concept.replace('_', ' ')
        prompts.append(f"{concept_name} {relation_type} {related_concept}.")

    return prompts


def create_sumo_training_dataset(
    concept: Dict,
    all_concepts: Dict[str, Dict],
    negative_pool: List[str],
    n_positives: int = 10,
    n_negatives: int = 10,
    use_category_relationships: bool = True,
    use_wordnet_relationships: bool = True
) -> tuple[List[str], List[int]]:
    """
    Create balanced training dataset for SUMO concept binary classification.

    Combines:
    1. SUMO category relationships (category_children)
    2. WordNet relationships (from canonical_synset)
    3. Definitions
    4. Graph-based negatives

    Args:
        concept: SUMO concept dict with keys: 'sumo_term', 'definition',
                 'category_children', 'canonical_synset'
        all_concepts: Map of SUMO term -> concept dict
        negative_pool: Pool of negative concept names
        n_positives: Number of positive examples
        n_negatives: Number of negative examples
        use_category_relationships: Include SUMO category hierarchy
        use_wordnet_relationships: Include WordNet relationships

    Returns:
        (prompts, labels) where labels are 1 for positive, 0 for negative

    Example:
        >>> concept = {
        ...     "sumo_term": "Physical",
        ...     "definition": "an entity that has physical existence",
        ...     "category_children": ["Object", "Process"],
        ...     "canonical_synset": "physical_entity.n.01"
        ... }
        >>> prompts, labels = create_sumo_training_dataset(
        ...     concept, all_concepts, ["Abstract", "Quantity"],
        ...     n_positives=5, n_negatives=5
        ... )
        >>> len(prompts), sum(labels)
        (10, 5)
    """
    prompts = []
    labels = []

    concept_name = concept['sumo_term']
    definition = concept.get('definition', f"SUMO category: {concept_name}")

    # Positive examples: start with definition
    prompts.append(f"What is {concept_name}? {definition}")
    labels.append(1)

    n_remaining = n_positives - 1

    # SUMO category relationships
    category_prompts = []
    if use_category_relationships:
        category_children = concept.get('category_children', [])
        if category_children:
            category_prompts = generate_category_relationship_prompts(
                concept_name,
                category_children,
                all_concepts,
                n_samples=n_remaining // 2 if use_wordnet_relationships else n_remaining
            )

    # WordNet relationships
    wordnet_prompts = []
    if use_wordnet_relationships:
        canonical_synset = concept.get('canonical_synset')
        if canonical_synset:
            n_wordnet = n_remaining - len(category_prompts)
            wordnet_prompts = generate_wordnet_relationship_prompts(
                canonical_synset,
                n_samples=n_wordnet
            )

    # Combine relationship prompts
    rel_prompts = category_prompts + wordnet_prompts

    # Pad if we don't have enough relationships
    while len(rel_prompts) < n_remaining:
        # Fallback: use definition with variations
        variations = [
            f"Describe {concept_name}.",
            f"Explain what {concept_name} means.",
            f"{concept_name} is defined as: {definition}",
        ]
        rel_prompts.append(random.choice(variations))

    # Limit to requested number
    rel_prompts = rel_prompts[:n_remaining]

    prompts.extend(rel_prompts)
    labels.extend([1] * len(rel_prompts))

    # Negative examples: semantically distant concepts
    neg_prompts = []
    sampled_negs = random.sample(negative_pool, min(n_negatives, len(negative_pool)))

    for neg_concept in sampled_negs:
        # Get definition if available
        if neg_concept in all_concepts:
            neg_def = all_concepts[neg_concept].get('definition', f"SUMO category: {neg_concept}")
            neg_prompts.append(f"What is {neg_concept}? {neg_def}")
        else:
            neg_prompts.append(f"What is {neg_concept}?")

    prompts.extend(neg_prompts)
    labels.extend([0] * len(neg_prompts))

    return prompts, labels


def build_sumo_negative_pool(
    all_concepts: List[Dict],
    target_concept: Dict,
    min_layer_distance: int = 0
) -> List[str]:
    """
    Build negative pool using SUMO hierarchy structure.

    Uses a more lenient approach for Layer 0 concepts where all concepts
    are at the same layer. For single-layer scenarios, any concept that's
    not a direct parent/child is considered a valid negative.

    Args:
        all_concepts: All SUMO concepts
        target_concept: Target concept to find negatives for
        min_layer_distance: Minimum layer distance (default 0 = exclude only direct relations)

    Returns:
        List of negative concept names
    """
    target_layer = target_concept['layer']
    target_term = target_concept['sumo_term']
    target_children = set(target_concept.get('category_children', []))

    # Check if target has a parent (appears in someone's category_children)
    target_parents = set()
    for concept in all_concepts:
        if target_term in concept.get('category_children', []):
            target_parents.add(concept['sumo_term'])

    negatives = []

    for concept in all_concepts:
        concept_term = concept['sumo_term']

        # Skip same concept
        if concept_term == target_term:
            continue

        # Skip direct children
        if concept_term in target_children:
            continue

        # Skip direct parents
        if concept_term in target_parents:
            continue

        # Skip if target is in this concept's children (redundant check but safe)
        if target_term in concept.get('category_children', []):
            continue

        # For Layer 0 (all same layer), accept all non-direct-relations
        if target_layer == 0 and concept['layer'] == 0:
            negatives.append(concept_term)
            continue

        # For other layers, use layer distance
        layer_dist = abs(concept['layer'] - target_layer)
        if layer_dist >= min_layer_distance:
            negatives.append(concept_term)

    return negatives


def extract_sumo_relationships(concept: Dict) -> Dict[str, List[str]]:
    """
    Extract all relationships from a SUMO concept for analysis.

    Args:
        concept: SUMO concept dict

    Returns:
        Dict with keys: 'category_children', 'hypernyms', 'hyponyms', etc.
    """
    relationships = {
        'category_children': concept.get('category_children', []),
        'hypernyms': [],
        'hyponyms': [],
        'meronyms': [],
        'holonyms': [],
        'antonyms': []
    }

    canonical_synset = concept.get('canonical_synset')
    if canonical_synset:
        try:
            synset = wn.synset(canonical_synset)

            relationships['hypernyms'] = [h.name() for h in synset.hypernyms()]
            relationships['hyponyms'] = [h.name() for h in synset.hyponyms()[:10]]
            relationships['meronyms'] = [h.name() for h in synset.member_meronyms() + synset.part_meronyms()]
            relationships['holonyms'] = [h.name() for h in synset.member_holonyms() + synset.part_holonyms()]

            # Extract antonyms from lemmas
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    relationships['antonyms'].append(antonym.synset().name())
        except Exception:
            pass

    return relationships
