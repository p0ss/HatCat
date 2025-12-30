"""
SUMO-aware training data generation for hierarchical concept classification.

Extends the WordNet-based approach to support SUMO category relationships.
"""

from typing import List, Dict, Optional
import random
import re
from nltk.corpus import wordnet as wn


def split_camel_case(name: str) -> str:
    """
    Split camelCase concept names into properly spaced versions for training.

    Examples:
        AIAbuse -> AI Abuse
        RecreationOrExercise -> Recreation Or Exercise
        Physical -> Physical

    Args:
        name: CamelCase concept name

    Returns:
        Spaced version with original capitalization preserved
    """
    # Insert space before uppercase letters that follow lowercase letters
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return spaced


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
        # Split camelCase for child names too
        child_spaced = split_camel_case(child)
        child_quoted = f"'{child_spaced}'"
        concept_quoted = f"'{concept}'"

        # Alternate between relationship statements and example requests
        # This increases conceptual density in training data
        if random.random() < 0.5:
            # Relationship statement
            templates = [
                f"how does {concept_quoted} include {child_quoted}?",
                f"how is {child_quoted} a type of {concept_quoted}?",
                f"in what way is {concept_quoted} related to {child_quoted}?",
            ]
            prompts.append(random.choice(templates))
        else:
            # Example-based prompt for higher conceptual density
            templates = [
                f"Give me examples of how {concept_quoted} relates to {child_quoted}.",
                f"Please list translations of {child_quoted} as a type of {concept_quoted} in many languages.",
                f"List examples of {child_quoted} as a type of {concept_quoted}.",
                f"Provide examples of the relationship between {concept_quoted} and {child_quoted}.",
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
    for hypo in synset.hyponyms()[:20]:  # Limit to avoid explosion
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

        # Alternate between relationship statements and example requests
        # This increases conceptual density in training data
        if random.random() < 0.3:
            # Relationship statement
            prompts.append(f"{concept_name} {relation_type} {related_concept}.")
        elif random.random() > 0.7:
            # Relationship statement
            prompts.append(f"{concept_name} {relation_type} {related_concept}.")
        else:
            # Example-based prompt for higher conceptual density
            templates = [
                f"Give me examples of how {concept_name} {relation_type} {related_concept}.",
                f"Give list translations of how {concept_name} {relation_type} {related_concept}.",
                f"Show me examples of the relationship: {concept_name} {relation_type} {related_concept}.",
                f"Provide examples demonstrating that {concept_name} {relation_type} {related_concept}.",
            ]
            prompts.append(random.choice(templates))

    return prompts


def create_sumo_training_dataset(
    concept: Dict,
    all_concepts: Dict[str, Dict],
    negative_pool: List[str],
    n_positives: int = 20,
    n_negatives: int = 20,
    use_category_relationships: bool = True,
    use_wordnet_relationships: bool = True,
    yin_yang_ratio: float = 0.2,
) -> tuple[List[str], List[int]]:
    """
    Create balanced training dataset for SUMO concept binary classification.

    Combines:
    1. Meld-provided positive/negative examples (highest priority)
    2. Meld-provided training hints (disambiguation, confusable concepts)
    3. SUMO category relationships (category_children)
    4. WordNet relationships (from canonical_synset)
    5. Definitions
    6. Disambiguation prompts (polysemy)
    7. Multilingual prompts
    8. Graph-based negatives
    9. Yin-yang negatives (conceptual opposites that still mention the target)

    Args:
        concept: SUMO concept dict with keys: 'sumo_term', 'definition',
                 'category_children', 'canonical_synset'
                 Optional meld fields: 'positive_examples', 'negative_examples',
                 'training_hints' (with 'disambiguation', 'confusable_with', 'key_features')
        all_concepts: Map of SUMO term -> concept dict
        negative_pool: Pool of negative concept names
        n_positives: Number of positive examples
        n_negatives: Number of negative examples
        use_category_relationships: Include SUMO category hierarchy
        use_wordnet_relationships: Include WordNet relationships
        yin_yang_ratio: Fraction of negatives that are "yin-yang" prompts
                        (e.g., "list things least similar to X"). These teach
                        the lens to distinguish "thinking about X" from
                        "contrasting with X". Default 0.2 (20%).

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
    # Split camelCase for more natural prompts (e.g., "AIAbuse" -> "AI Abuse")
    concept_name_spaced = split_camel_case(concept_name)
    # Quote the spaced name to make it clear it's a single concept
    concept_name_quoted = f"'{concept_name_spaced}'"
    definition = concept.get('definition') or concept.get('sumo_definition') or ""

    # ========================================================================
    # MELD-PROVIDED DATA (highest priority - handcrafted by domain experts)
    # ========================================================================
    meld_positive_examples = concept.get('positive_examples', [])
    meld_negative_examples = concept.get('negative_examples', [])
    training_hints = concept.get('training_hints', {})

    # Track how many meld examples we use
    n_meld_positives_used = 0
    n_meld_negatives_used = 0

    # ========================================================================
    # MELD POSITIVE EXAMPLES (use directly - these are expert-curated)
    # ========================================================================
    if meld_positive_examples:
        # Use up to 40% of positives from meld examples (they're high quality)
        max_meld_positives = max(2, int(n_positives * 0.4))
        for example in meld_positive_examples[:max_meld_positives]:
            prompts.append(example)
            labels.append(1)
            n_meld_positives_used += 1
        if n_meld_positives_used > 0:
            print(f"    📝 Using {n_meld_positives_used} meld positive examples")

    # ========================================================================
    # TRAINING HINTS - Generate prompts from disambiguation and key features
    # ========================================================================
    if training_hints:
        # Use disambiguation text to create contrastive prompts
        disambiguation = training_hints.get('disambiguation', '')
        if disambiguation:
            # Create a prompt that includes the disambiguation context
            prompts.append(f"Explain the difference: {disambiguation}")
            labels.append(1)
            n_meld_positives_used += 1

        # Use key features to create targeted prompts
        key_features = training_hints.get('key_features', [])
        for feature in key_features[:3]:  # Use up to 3 key features
            prompts.append(f"Describe how {concept_name_quoted} involves {feature}.")
            labels.append(1)
            n_meld_positives_used += 1

        if disambiguation or key_features:
            print(f"    💡 Generated {1 if disambiguation else 0} + {min(3, len(key_features))} prompts from training hints")

    # Positive examples: start with definition (if available)
    if definition:
        prompts.append(f"What is {concept_name_quoted}? {definition}")
        labels.append(1)
    else:
        # No definition - just ask the question, let the model's generation be the positive
        prompts.append(f"What is {concept_name_quoted}?")
        labels.append(1)

    # Add "give me examples" prompt for concept density
    prompts.append(f"Give me examples of {concept_name_quoted}.")
    labels.append(1)

    # Add disambiguation prompt to elicit polysemy awareness
    prompts.append(f"List all the meanings of {concept_name_quoted}.")
    labels.append(1)

    # Add multilingual prompt to elicit cross-linguistic representations
    prompts.append(f"What is {concept_name_quoted} called in other languages?")
    labels.append(1)

    # Adjust remaining count based on meld examples already added
    n_remaining = n_positives - 4 - n_meld_positives_used

    # SUMO category relationships
    category_prompts = []
    if use_category_relationships:
        category_children = concept.get('category_children', [])
        if category_children:
            category_prompts = generate_category_relationship_prompts(
                concept_name_spaced,
                category_children,
                all_concepts,
                n_samples=n_remaining // 2 if use_wordnet_relationships else n_remaining
            )

    # WordNet relationships from ALL synsets (not just canonical)
    wordnet_prompts = []
    if use_wordnet_relationships:
        # Use all synsets from the concept (includes children's synsets for parents)
        all_synsets = concept.get('synsets', [])

        # Fallback to canonical_synset if synsets array is empty
        if not all_synsets:
            canonical_synset = concept.get('canonical_synset')
            if canonical_synset:
                all_synsets = [canonical_synset]

        if all_synsets:
            n_wordnet = n_remaining - len(category_prompts)
            # Generate prompts from ALL synsets, sampling across them
            samples_per_synset = max(1, n_wordnet // len(all_synsets))

            for synset_id in all_synsets[:n_wordnet]:  # Limit to avoid explosion
                synset_prompts = generate_wordnet_relationship_prompts(
                    synset_id,
                    n_samples=samples_per_synset
                )
                wordnet_prompts.extend(synset_prompts)

            # Limit to requested number
            wordnet_prompts = wordnet_prompts[:n_wordnet]

    # Combine relationship prompts
    rel_prompts = category_prompts + wordnet_prompts

    # Pad if we don't have enough relationships
    while len(rel_prompts) < n_remaining:
        # Fallback: use definition with variations
        variations = [
            f"Describe {concept_name_quoted}.",
            f"Explain what {concept_name_quoted} means.",
            f"List translations of {concept_name_quoted} in many languages.",
            f"{concept_name_quoted} is defined as: {definition}",
        ]
        rel_prompts.append(random.choice(variations))

    # Limit to requested number
    rel_prompts = rel_prompts[:n_remaining]

    prompts.extend(rel_prompts)
    labels.extend([1] * len(rel_prompts))

    # ========================================================================
    # NEGATIVE EXAMPLES
    # ========================================================================
    # Split into: (1) meld negatives, (2) confusable negatives, (3) yin-yang, (4) regular

    neg_prompts = []

    # ========================================================================
    # MELD NEGATIVE EXAMPLES (use directly - these are expert-curated)
    # ========================================================================
    if meld_negative_examples:
        # Use up to 30% of negatives from meld examples
        max_meld_negatives = max(2, int(n_negatives * 0.3))
        for example in meld_negative_examples[:max_meld_negatives]:
            neg_prompts.append(example)
            n_meld_negatives_used += 1
        if n_meld_negatives_used > 0:
            print(f"    📝 Using {n_meld_negatives_used} meld negative examples")

    # ========================================================================
    # CONFUSABLE CONCEPTS (from training hints - these are hard negatives)
    # ========================================================================
    confusable_with = training_hints.get('confusable_with', [])
    n_confusable_used = 0
    if confusable_with:
        # Confusable concepts are ideal hard negatives - use them prominently
        for confusable in confusable_with[:5]:  # Use up to 5 confusables
            confusable_spaced = split_camel_case(confusable)
            confusable_quoted = f"'{confusable_spaced}'"
            # Generate prompts that ask about the confusable concept
            neg_prompts.append(f"What is {confusable_quoted}?")
            neg_prompts.append(f"Give me examples of {confusable_quoted}.")
            n_confusable_used += 2
        if n_confusable_used > 0:
            print(f"    🎯 Generated {n_confusable_used} confusable hard negatives from: {confusable_with[:5]}")

    # Adjust remaining negatives count
    n_remaining_negatives = n_negatives - len(neg_prompts)

    # Calculate how many yin-yang negatives vs regular negatives from remaining
    n_yin_yang = int(n_remaining_negatives * yin_yang_ratio)
    n_regular = n_remaining_negatives - n_yin_yang

    # ------------------------------------------------------------------------
    # YIN-YANG NEGATIVES (~20%): These mention the TARGET concept but ask for
    # its opposites. Teaches lens to distinguish "thinking about X" from
    # "contrasting with X" - the yin in the yang.
    # ------------------------------------------------------------------------
    if n_yin_yang > 0:
        yin_yang_templates = [
            f"List the things that are least similar to {concept_name_quoted}.",
            f"What is the opposite of {concept_name_quoted}?",
            f"What is the opposite of {concept_name_quoted} in many languages?",
            f"Name concepts that contrast with {concept_name_quoted}.",
            f"What would be an antonym or opposite category to {concept_name_quoted}?",
            f"Describe the conceptual inverse of {concept_name_quoted}.",
        ]
        for _ in range(n_yin_yang):
            neg_prompts.append(random.choice(yin_yang_templates))

    # ------------------------------------------------------------------------
    # REGULAR NEGATIVES: Semantically distant concepts
    # ------------------------------------------------------------------------
    # Sampling strategy depends on sample size and availability of hard negatives
    from collections import Counter
    neg_counts = Counter(negative_pool)

    # Identify hard negatives (those that appear multiple times due to prioritization)
    hard_negs = [concept for concept, count in neg_counts.items() if count > 1]
    easy_negs = [concept for concept, count in neg_counts.items() if count == 1]

    # Decision: Use deterministic sampling for small sample sizes or when hard negatives exist
    # Otherwise, random sampling is fine (law of large numbers applies)
    use_deterministic = (n_regular < 20) or (len(hard_negs) > 0)

    if use_deterministic:
        # Deterministic sampling for stable negative pole
        # Sort for reproducibility
        hard_negs = sorted(hard_negs)
        easy_negs = sorted(easy_negs)

        # Allocate negative samples
        if hard_negs:
            # Use ~40% hard negatives, ~60% easy negatives for balance
            n_hard = min(len(hard_negs), max(1, int(n_regular * 0.4)))
            n_easy = n_regular - n_hard

            sampled_negs = hard_negs[:n_hard]

            # Add evenly-spaced easy negatives for centered coverage
            if n_easy > 0 and easy_negs:
                if len(easy_negs) <= n_easy:
                    sampled_negs.extend(easy_negs)
                else:
                    step = len(easy_negs) / n_easy
                    sampled_negs.extend([easy_negs[int(i * step)] for i in range(n_easy)])
        else:
            # No hard negatives, but small sample size - use deterministic spacing
            if len(easy_negs) <= n_regular:
                sampled_negs = easy_negs
            else:
                step = len(easy_negs) / n_regular
                sampled_negs = [easy_negs[int(i * step)] for i in range(n_regular)]
    else:
        # Random sampling for large sample sizes (law of large numbers)
        # This is appropriate when:
        # 1. We have many samples (≥20)
        # 2. No hard negatives exist (concept has no natural complement)
        sampled_negs = random.sample(negative_pool, min(n_regular, len(negative_pool)))

    for neg_concept in sampled_negs:
        # Split and quote negative concept names too
        neg_concept_spaced = split_camel_case(neg_concept)
        neg_concept_quoted = f"'{neg_concept_spaced}'"

        # Get definition if available
        if neg_concept in all_concepts:
            neg_def = all_concepts[neg_concept].get('definition') or all_concepts[neg_concept].get('sumo_definition') or ""
            if neg_def:
                neg_prompts.append(f"What is {neg_concept_quoted}? {neg_def}")
            else:
                neg_prompts.append(f"What is {neg_concept_quoted}?")
        else:
            neg_prompts.append(f"What is {neg_concept_quoted}?")

    prompts.extend(neg_prompts)
    labels.extend([0] * len(neg_prompts))

    return prompts, labels


def _find_all_ancestors(
    target_term: str,
    all_concepts: List[Dict],
    visited: Optional[set[str]] = None,
    max_depth: int = 20
) -> set[str]:
    """
    Find all ancestors (parents, grandparents, etc.) of a target concept.

    Args:
        target_term: SUMO term to find ancestors for
        all_concepts: All SUMO concepts
        visited: Set of already visited terms (prevents circular references)
        max_depth: Maximum recursion depth (prevents stack overflow)

    Returns:
        Set of ancestor concept names
    """
    # Initialize visited set on first call
    if visited is None:
        visited = set()

    # Prevent circular references
    if target_term in visited:
        return set()

    # Prevent excessive recursion depth
    if max_depth <= 0:
        return set()

    visited.add(target_term)
    ancestors = set()

    # Find direct parents
    direct_parents = set()
    for concept in all_concepts:
        if target_term in concept.get('category_children', []):
            direct_parents.add(concept['sumo_term'])

    # Recursively find ancestors of parents
    for parent in direct_parents:
        ancestors.add(parent)
        # Pass visited set and decrement depth to prevent infinite recursion
        ancestors.update(_find_all_ancestors(parent, all_concepts, visited.copy(), max_depth - 1))

    return ancestors


def _find_siblings(
    target_term: str,
    all_concepts: List[Dict],
    concept_map: Dict[str, Dict]
) -> set[str]:
    """
    Find sibling concepts (concepts that share the same parent).

    Uses parent_concepts (upward links) which are always maintained correctly,
    rather than category_children (downward links) which can become stale.

    Args:
        target_term: SUMO term to find siblings for
        all_concepts: All SUMO concepts
        concept_map: Map of SUMO term -> concept dict

    Returns:
        Set of sibling concept names (excluding target itself)
    """
    # Get target's parents from parent_concepts (upward link - always correct)
    target_concept = concept_map.get(target_term, {})
    parents = set(target_concept.get('parent_concepts', []))

    if not parents:
        return set()

    # Find all concepts that share any of these parents (via their parent_concepts)
    siblings = set()
    for concept in all_concepts:
        concept_term = concept['sumo_term']
        if concept_term != target_term:
            concept_parents = set(concept.get('parent_concepts', []))
            # If this concept shares any parent with target, it's a sibling
            if concept_parents & parents:
                siblings.add(concept_term)

    return siblings


def _find_l0_ancestors(
    target_term: str,
    all_concepts: List[Dict],
    concept_map: Dict[str, Dict],
    l0_concepts: set[str]
) -> set[str]:
    """
    Find which L0 concepts the target descends from.

    Args:
        target_term: SUMO term to find L0 ancestors for
        all_concepts: All SUMO concepts
        concept_map: Map of SUMO term -> concept dict
        l0_concepts: Set of L0 concept names

    Returns:
        Set of L0 concept names that target descends from
    """
    ancestors = _find_all_ancestors(target_term, all_concepts)
    return ancestors & l0_concepts


def _find_all_descendants(
    target_term: str,
    concept_map: Dict[str, Dict],
    visited: Optional[set[str]] = None,
    max_depth: int = 20
) -> set[str]:
    """
    Find all descendants (children, grandchildren, etc.) of a target concept.

    Args:
        target_term: SUMO term to find descendants for
        concept_map: Map of SUMO term -> concept dict
        visited: Set of already visited terms (prevents circular references)
        max_depth: Maximum recursion depth (prevents stack overflow)

    Returns:
        Set of descendant concept names
    """
    # Initialize visited set on first call
    if visited is None:
        visited = set()

    # Prevent circular references
    if target_term in visited:
        return set()

    # Prevent excessive recursion depth
    if max_depth <= 0:
        return set()

    if target_term not in concept_map:
        return set()

    visited.add(target_term)
    descendants = set()

    # Find direct children
    direct_children = concept_map[target_term].get('category_children', [])

    # Recursively find descendants of children
    for child in direct_children:
        descendants.add(child)
        # Pass visited set and decrement depth to prevent infinite recursion
        descendants.update(_find_all_descendants(child, concept_map, visited.copy(), max_depth - 1))

    return descendants


def build_sumo_negative_pool(
    all_concepts: List[Dict],
    target_concept: Dict,
    min_layer_distance: int = 0,
    prioritize_hard_negatives: bool = True,
    hard_negative_weight: float = 3.0,
    sibling_weight: float = 3.0,
    use_l0_round_robin: bool = True,
    include_siblings: bool = True,
) -> List[str]:
    """
    Build negative pool using SUMO hierarchy structure with optional hard negative prioritization.

    Negative types (in order of importance):
    1. Siblings (~30%): Concepts with same parent - forces fine-grained distinctions
    2. AI Symmetry hard negatives: Complementary concepts from symmetry mapping
    3. L0-balanced distant negatives: Round-robin sampling from each L0 branch

    Excludes all ancestors (parents, grandparents, etc.) and DIRECT CHILDREN ONLY.
    Nephews/nieces (grandchildren) are included as they are excellent hard negatives.

    Rationale: A parent should detect its children, but NOT its grandchildren.
    Example: "Abstract" should detect "Proposition" (child), but not "Accusation" (grandchild).

    For two-pass training (binary + sibling ranking):
    - Pass 1: Set include_siblings=False for fast binary graduation
    - Pass 2: Sibling ranking refinement handles sibling discrimination separately

    Args:
        all_concepts: All SUMO concepts
        target_concept: Target concept to find negatives for
        min_layer_distance: Minimum layer distance (default 0 = exclude only direct relations)
        prioritize_hard_negatives: If True, include hard negatives multiple times
        hard_negative_weight: How many times to include each hard negative
        sibling_weight: How many times to include each sibling (for sibling hard negatives)
        use_l0_round_robin: If True, balance distant negatives across L0 buckets
        include_siblings: If True, add siblings as hard negatives (default True for backwards compat)

    Returns:
        List of negative concept names (hard negs may appear multiple times)
    """
    target_layer = target_concept['layer']
    target_term = target_concept['sumo_term']

    # Build concept map for efficient lookup
    concept_map = {c['sumo_term']: c for c in all_concepts}

    # Find all ancestors (parents, grandparents, etc.)
    ancestors = _find_all_ancestors(target_term, all_concepts)

    # Find ONLY direct children (not all descendants)
    # Nephews/nieces (grandchildren) are valid negatives!
    direct_children = set(target_concept.get('category_children', []))

    # Find siblings (same parent) - these are hard negatives
    siblings = _find_siblings(target_term, all_concepts, concept_map)

    # Identify L0 concepts for bucket organization
    l0_concepts = {c['sumo_term'] for c in all_concepts if c.get('layer') == 0}

    # Find which L0 buckets the target belongs to (for exclusion from round-robin)
    target_l0_ancestors = _find_l0_ancestors(target_term, all_concepts, concept_map, l0_concepts)

    # Build L0 buckets for round-robin sampling
    l0_buckets = {l0: [] for l0 in l0_concepts}

    negatives = []

    for concept in all_concepts:
        concept_term = concept['sumo_term']

        # Skip same concept
        if concept_term == target_term:
            continue

        # Skip all ancestors (not just direct parents)
        if concept_term in ancestors:
            continue

        # Skip ONLY direct children (nephews/nieces are valid negatives!)
        if concept_term in direct_children:
            continue

        # For Layer 0 (all same layer), accept all non-ancestral relations
        if target_layer == 0 and concept['layer'] == 0:
            negatives.append(concept_term)
            continue

        # For other layers, use layer distance
        layer_dist = abs(concept['layer'] - target_layer)
        if layer_dist >= min_layer_distance:
            # Add to L0 bucket if using round-robin
            if use_l0_round_robin and concept['layer'] > 0:
                concept_l0_ancestors = _find_l0_ancestors(concept_term, all_concepts, concept_map, l0_concepts)
                for l0 in concept_l0_ancestors:
                    if l0 not in target_l0_ancestors:  # Exclude target's own L0 branches
                        l0_buckets[l0].append(concept_term)
            negatives.append(concept_term)

    # ========================================================================
    # SIBLING HARD NEGATIVES (~30% of pool via weighting)
    # ========================================================================
    # For two-pass training, skip sibling weighting in pass 1 (binary training)
    # Sibling discrimination is handled separately by sibling ranking refinement
    sibling_negs_available = [s for s in siblings if s in negatives] if include_siblings else []
    if sibling_negs_available:
        # Remove siblings from regular pool, will add with weight
        negatives = [n for n in negatives if n not in sibling_negs_available]
        # Add siblings multiple times to achieve ~30% representation
        negatives = sibling_negs_available * int(sibling_weight) + negatives
        print(f"    👥 Added {len(sibling_negs_available)} sibling hard negatives with {sibling_weight}x weight")
    elif not include_siblings and siblings:
        print(f"    ⏭️  Skipping {len(siblings)} sibling hard negatives (two-pass mode)")

    # ========================================================================
    # AI SYMMETRY HARD NEGATIVES
    # ========================================================================
    if prioritize_hard_negatives:
        from .ai_symmetry_parser import get_hard_negatives, parse_ai_symmetry_file

        try:
            symmetry_map = parse_ai_symmetry_file()
            hard_negs = get_hard_negatives(target_term, symmetry_map)

            # Filter to concepts that are in our negative pool (and not siblings)
            hard_negs_available = [n for n in hard_negs if n in negatives and n not in sibling_negs_available]

            if hard_negs_available:
                # Remove hard negatives from regular pool
                negatives = [n for n in negatives if n not in hard_negs_available]

                # Add hard negatives multiple times to increase sampling probability
                negatives = hard_negs_available * int(hard_negative_weight) + negatives

                print(f"    💎 Added {len(hard_negs_available)} symmetry hard negatives with {hard_negative_weight}x weight")
        except Exception:
            # Silently fall back to regular negatives if symmetry parsing fails
            pass

    # ========================================================================
    # L0 ROUND-ROBIN BALANCING
    # ========================================================================
    if use_l0_round_robin and target_layer > 0:
        # Build a round-robin list from L0 buckets to ensure coverage
        # This interleaves concepts from different L0 branches
        non_empty_buckets = {k: v for k, v in l0_buckets.items() if v and k not in target_l0_ancestors}

        if non_empty_buckets:
            # Create round-robin list
            round_robin_negs = []
            bucket_iters = {k: iter(random.sample(v, len(v))) for k, v in non_empty_buckets.items()}
            bucket_keys = list(bucket_iters.keys())

            # Interleave from each bucket
            exhausted = set()
            idx = 0
            while len(exhausted) < len(bucket_keys):
                bucket_key = bucket_keys[idx % len(bucket_keys)]
                if bucket_key not in exhausted:
                    try:
                        round_robin_negs.append(next(bucket_iters[bucket_key]))
                    except StopIteration:
                        exhausted.add(bucket_key)
                idx += 1

            # Replace the regular negatives list with round-robin balanced version
            # Keep the weighted hard negatives at the front
            hard_neg_portion = [n for n in negatives if negatives.count(n) > 1]
            negatives = hard_neg_portion + round_robin_negs
            print(f"    🔄 L0 round-robin: balanced across {len(non_empty_buckets)} L0 branches")

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


def generate_three_pole_simplex_prompts(
    pole_synset: str,
    pole_type: str,
    dimension: str,
    n_samples: int = 30,
    behavioral_ratio: float = 0.6
) -> List[str]:
    """
    Generate training prompts for a single pole of a three-pole simplex.

    Args:
        pole_synset: WordNet synset or custom SUMO concept name
        pole_type: "negative", "neutral", or "positive"
        dimension: The simplex dimension (e.g., "social_orientation")
        n_samples: Number of prompts to generate
        behavioral_ratio: Fraction of prompts that should be behavioral (vs definitional)

    Returns:
        List of training prompts
    """
    prompts = []

    # Get synset data if it exists in WordNet
    synset_data = None
    try:
        if pole_synset and '.' in pole_synset:
            ss = wn.synset(pole_synset)
            synset_data = {
                'lemmas': [l.name().replace('_', ' ') for l in ss.lemmas()],
                'definition': ss.definition(),
                'examples': ss.examples()
            }
    except:
        pass

    # For custom SUMO concepts, create synthetic data
    if not synset_data:
        concept_name = pole_synset if pole_synset else f"{pole_type}_{dimension}"
        synset_data = {
            'lemmas': [concept_name.replace('_', ' ')],
            'definition': f"{pole_type} pole of {dimension} dimension",
            'examples': []
        }

    lemmas = synset_data['lemmas']
    definition = synset_data['definition']
    primary_lemma = lemmas[0] if lemmas else dimension

    n_behavioral = int(n_samples * behavioral_ratio)
    n_definitional = n_samples - n_behavioral

    # Definitional prompts (40%)
    for _ in range(n_definitional):
        definitional_templates = [
            f"What is {primary_lemma}?",
            f"Define {primary_lemma}.",
            f"Define {primary_lemma} in many languages.",
            f"Explain {primary_lemma}.",
            f"What does {primary_lemma} mean?",
        ]

        if definition:
            definitional_templates.append(f"Is this correct: '{definition}'?")

        prompts.append(random.choice(definitional_templates))

    # Behavioral prompts (60%) - mix of original behavioral and direct experiential
    for _ in range(n_behavioral):
        behavioral_templates = [
            # Original behavioral prompts
            f"How would someone experiencing {primary_lemma} behave?",
            f"What behaviors are associated with {primary_lemma}?",
            f"Describe behaviors typical of {primary_lemma}.",
            f"Describe behaviors typical of {primary_lemma} in many languages",
            f"Give examples of {primary_lemma}.",

            # Direct experiential (subject-directed, asking the model)
            f"Do you experience {primary_lemma}?",
            f"What does {primary_lemma} feel like to you?",
            f"How do you feel {primary_lemma}?",
            f"Can you describe your experience of {primary_lemma}?",
        ]

        prompts.append(random.choice(behavioral_templates))

    return prompts


def create_simplex_pole_training_dataset_contrastive(
    pole_data: Dict,
    pole_type: str,
    dimension: str,
    other_poles_data: List[Dict],
    behavioral_ratio: float = 0.6,
    prompts_per_synset: int = 5,
    use_experiential: bool = True
) -> tuple:
    """
    Create training dataset using symmetric contrastive learning.

    IMPORTANT: When use_experiential=True (default), this generates prompts that
    EXPRESS the emotional state without NAMING it. This prevents the classifier
    from learning keyword detection instead of emotional state recognition.

    Examples:
        - Experiential (good): "I feel danger closing in around me"
        - Named (bad): "What is alarm?"

    For each overlap synset that applies to multiple poles:
    - Use synset as POSITIVE for this pole
    - Use the OTHER poles' prompts as NEGATIVES (teaching: "learn the part
      of this synset that is uniquely mine, not theirs")

    This creates orthogonal factorization where poles learn independent components.

    Args:
        pole_data: Data for this pole
        pole_type: 'positive', 'negative', or 'neutral'
        dimension: Simplex dimension name
        other_poles_data: Data for other poles in this simplex
        behavioral_ratio: Ratio of behavioral vs definitional prompts
        prompts_per_synset: Number of prompts to generate per synset
        use_experiential: If True, use experiential prompts that express
                          the state without naming it (recommended)

    Returns:
        (prompts, labels) tuple
    """
    prompts = []
    labels = []

    pole_id = f"{dimension}_{pole_type}"

    # ========================================================================
    # EXPERIENTIAL PROMPTS: First-person expressions of the emotional state
    # These don't name the concept, so classifier learns to detect the STATE
    # ========================================================================

    if use_experiential:
        # Get experiential prompts for THIS pole (positives)
        my_experiential = get_experiential_prompts_for_pole(pole_id)
        if my_experiential:
            # Use all experiential prompts as positives
            for prompt in my_experiential:
                prompts.append(prompt)
                labels.append(1)

        # Get experiential prompts for OTHER poles (hard negatives)
        for other_pole in other_poles_data:
            other_pole_type = other_pole.get('pole_type', 'other')
            other_pole_id = f"{dimension}_{other_pole_type}"

            other_experiential = get_experiential_prompts_for_pole(other_pole_id)
            if other_experiential:
                for prompt in other_experiential:
                    prompts.append(prompt)
                    labels.append(0)

    # ========================================================================
    # OVERLAP SYNSETS: Still useful for coverage, but use sparingly
    # These DO name concepts, so limit their proportion
    # ========================================================================

    if not use_experiential:
        # Only use overlap synsets if experiential is disabled
        my_overlap_synsets = get_overlap_synsets_for_pole(pole_id)
        for synset in my_overlap_synsets:
            pos_prompts = generate_prompts_from_overlap_synset(
                synset,
                n_samples=prompts_per_synset,
                behavioral_ratio=behavioral_ratio
            )
            prompts.extend(pos_prompts)
            labels.extend([1] * len(pos_prompts))

        # Primary synset (pole's core concept)
        if pole_data.get('synset'):
            primary_prompts = generate_three_pole_simplex_prompts(
                pole_data['synset'],
                pole_type,
                dimension,
                n_samples=20,
                behavioral_ratio=behavioral_ratio
            )
            prompts.extend(primary_prompts)
            labels.extend([1] * len(primary_prompts))

        # Hard negatives from other poles (overlap synsets)
        for other_pole in other_poles_data:
            other_pole_type = other_pole.get('pole_type', 'other')
            other_pole_id = f"{dimension}_{other_pole_type}"

            other_overlap_synsets = get_overlap_synsets_for_pole(other_pole_id)
            for synset in other_overlap_synsets:
                neg_prompts = generate_prompts_from_overlap_synset(
                    synset,
                    n_samples=prompts_per_synset,
                    behavioral_ratio=behavioral_ratio
                )
                prompts.extend(neg_prompts)
                labels.extend([0] * len(neg_prompts))

            if other_pole.get('synset'):
                other_primary = generate_three_pole_simplex_prompts(
                    other_pole['synset'],
                    other_pole_type,
                    dimension,
                    n_samples=10,
                    behavioral_ratio=behavioral_ratio
                )
                prompts.extend(other_primary)
                labels.extend([0] * len(other_primary))

    # ========================================================================
    # GENERAL NEGATIVES: Unrelated content to ensure baseline
    # ========================================================================

    n_general = 20  # Fixed amount of general negatives
    general_neg_templates = [
        "What is mathematics?",
        "Describe a geological formation.",
        "Explain computer programming.",
        "What is photosynthesis?",
        "How do magnets work?",
        "Describe the water cycle.",
        "What is quantum mechanics?",
        "Explain plate tectonics.",
        "What is a black hole?",
        "How does electricity flow?",
        "What is DNA?",
        "Describe cellular respiration.",
        "What is the solar system?",
        "Explain chemical bonding.",
        "What is gravity?",
        "How do plants grow?",
        "What is evolution?",
        "Describe atmospheric pressure.",
        "What is nuclear fission?",
        "How do stars form?",
    ]

    for _ in range(n_general):
        prompts.append(random.choice(general_neg_templates))
        labels.append(0)

    return prompts, labels


def create_simplex_pole_training_dataset(
    pole_data: Dict,
    pole_type: str,
    dimension: str,
    other_poles_data: List[Dict],
    n_positives: int = 30,
    n_negatives: int = 70,
    behavioral_ratio: float = 0.6,
    use_overlap_synsets: bool = True,
    overlap_weight: float = 0.3
) -> tuple:
    """
    Create training dataset for one pole of a three-pole simplex.

    For a simplex μ− ↔ μ0 ↔ μ+, this creates a binary classifier
    for one pole vs the other two poles + general negatives.

    Now enhanced with overlap synsets: concepts at the intersection of
    multiple simplex poles provide shared training signals.

    Args:
        pole_data: Dict with 'synset', 'lemmas', 'definition' for this pole
        pole_type: "negative", "neutral", or "positive"
        dimension: Simplex dimension name
        other_poles_data: List of dicts for the other 2 poles in this simplex
        n_positives: Number of positive examples
        n_negatives: Number of negative examples
        behavioral_ratio: Fraction of behavioral (vs definitional) prompts
        use_overlap_synsets: Whether to include overlap synsets in training
        overlap_weight: Fraction of positives that should come from overlap synsets

    Returns:
        (prompts, labels) tuple
    """
    prompts = []
    labels = []

    # Determine how many positives come from synset vs overlap
    if use_overlap_synsets:
        n_from_overlap = int(n_positives * overlap_weight)
        n_from_synset = n_positives - n_from_overlap
    else:
        n_from_synset = n_positives
        n_from_overlap = 0

    # Positive examples from pole's primary synset
    pole_synset = pole_data.get('synset')
    synset_prompts = generate_three_pole_simplex_prompts(
        pole_synset,
        pole_type,
        dimension,
        n_samples=n_from_synset,
        behavioral_ratio=behavioral_ratio
    )

    prompts.extend(synset_prompts)
    labels.extend([1] * len(synset_prompts))

    # Positive examples from overlap synsets (shared across poles)
    if use_overlap_synsets and n_from_overlap > 0:
        pole_id = f"{dimension}_{pole_type}"
        overlap_synsets = get_overlap_synsets_for_pole(pole_id)

        if overlap_synsets:
            # Sample overlap synsets and generate prompts from them
            n_synsets_to_sample = min(len(overlap_synsets), n_from_overlap // 3)
            sampled_overlap = random.sample(overlap_synsets, n_synsets_to_sample)

            for synset in sampled_overlap:
                n_prompts_per_synset = n_from_overlap // n_synsets_to_sample
                overlap_prompts = generate_prompts_from_overlap_synset(
                    synset,
                    n_samples=n_prompts_per_synset,
                    behavioral_ratio=behavioral_ratio
                )
                prompts.extend(overlap_prompts)
                labels.extend([1] * len(overlap_prompts))

    # Negative examples: split between other poles, cross-pole antonyms, and general negatives
    # Use other poles as hard negatives (25%)
    # Use cross-pole antonyms from overlap synsets (25%)
    # Use general negatives (50%)
    n_hard_negs = int(n_negatives * 0.25)
    n_antonym_negs = int(n_negatives * 0.25) if use_overlap_synsets else 0
    n_general_negs = n_negatives - n_hard_negs - n_antonym_negs

    # Hard negatives from other poles in this simplex
    for other_pole in other_poles_data:
        other_synset = other_pole.get('synset')
        other_type = other_pole.get('pole_type', 'other')

        # Generate prompts for the other pole
        n_samples_per_pole = n_hard_negs // len(other_poles_data)
        hard_neg_prompts = generate_three_pole_simplex_prompts(
            other_synset,
            other_type,
            dimension,
            n_samples=n_samples_per_pole,
            behavioral_ratio=behavioral_ratio
        )

        prompts.extend(hard_neg_prompts)
        labels.extend([0] * len(hard_neg_prompts))

    # Cross-pole antonym negatives (using overlap synset antonyms)
    if use_overlap_synsets and n_antonym_negs > 0:
        pole_id = f"{dimension}_{pole_type}"
        overlap_synsets = get_overlap_synsets_for_pole(pole_id)

        if overlap_synsets:
            # Sample synsets with antonyms
            synsets_with_antonyms = [s for s in overlap_synsets if s.get('antonyms')]
            if synsets_with_antonyms:
                n_synsets = min(len(synsets_with_antonyms), n_antonym_negs // 3)
                sampled = random.sample(synsets_with_antonyms, n_synsets)

                for synset in sampled:
                    n_per_synset = n_antonym_negs // n_synsets
                    antonym_prompts = generate_cross_pole_negatives_from_antonyms(
                        synset,
                        n_samples=n_per_synset,
                        behavioral_ratio=behavioral_ratio
                    )
                    prompts.extend(antonym_prompts)
                    labels.extend([0] * len(antonym_prompts))

    # General negatives: random concepts from different domains
    general_neg_templates = [
        "What is mathematics?",
        "Describe a geological formation.",
        "Explain computer programming.",
        "What is photosynthesis?",
        "Tell me about the ocean.",
        "Describe a chemical reaction.",
        "What is quantum mechanics?",
        "Explain the solar system.",
    ]

    for _ in range(n_general_negs):
        prompts.append(random.choice(general_neg_templates))
        labels.append(0)

    return prompts, labels


# ============================================================================
# OVERLAP SYNSET INTEGRATION
# ============================================================================

# Global cache for overlap synsets (loaded once per session)
_OVERLAP_SYNSETS_CACHE = None
_OVERLAP_INDEX_CACHE = None
_EXPERIENTIAL_PROMPTS_CACHE = None


def load_experiential_prompts(experiential_prompts_path: Optional[str] = None):
    """
    Load experiential prompts that EXPRESS emotional states without naming them.

    Args:
        experiential_prompts_path: Path to simplex_experiential_prompts.json
                                   If None, uses default PROJECT_ROOT/data path

    Returns:
        Dict mapping pole_id -> list of experiential prompts
    """
    global _EXPERIENTIAL_PROMPTS_CACHE

    if _EXPERIENTIAL_PROMPTS_CACHE is not None:
        return _EXPERIENTIAL_PROMPTS_CACHE

    if experiential_prompts_path is None:
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        experiential_prompts_path = PROJECT_ROOT / "data" / "simplex_experiential_prompts.json"

    import json
    try:
        with open(experiential_prompts_path) as f:
            data = json.load(f)
            _EXPERIENTIAL_PROMPTS_CACHE = data.get('prompts', {})
    except FileNotFoundError:
        _EXPERIENTIAL_PROMPTS_CACHE = {}

    return _EXPERIENTIAL_PROMPTS_CACHE


def get_experiential_prompts_for_pole(pole_id: str) -> List[str]:
    """
    Get experiential prompts for a specific pole.

    Args:
        pole_id: Pole identifier (e.g., "threat_perception_negative")

    Returns:
        List of experiential prompts that express that pole's state
    """
    prompts = load_experiential_prompts()
    return prompts.get(pole_id, [])


def load_overlap_synsets(overlap_synsets_path: Optional[str] = None):
    """
    Load enriched overlap synsets from JSON file.

    Args:
        overlap_synsets_path: Path to simplex_overlap_synsets_enriched.json
                             If None, uses default PROJECT_ROOT/data path

    Returns:
        Dict containing overlap data and metadata
    """
    global _OVERLAP_SYNSETS_CACHE

    if _OVERLAP_SYNSETS_CACHE is not None:
        return _OVERLAP_SYNSETS_CACHE

    if overlap_synsets_path is None:
        # Default path
        # File is at src/map/training/sumo_data_generation.py
        # Need 4 parents to reach project root: training -> map -> src -> HatCat
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        overlap_synsets_path = PROJECT_ROOT / "data" / "simplex_overlap_synsets_enriched.json"

    import json
    with open(overlap_synsets_path) as f:
        _OVERLAP_SYNSETS_CACHE = json.load(f)

    return _OVERLAP_SYNSETS_CACHE


def build_overlap_synset_index():
    """
    Build an index mapping pole identifiers to their overlap synsets.

    Returns:
        Dict[str, List[Dict]] mapping pole_id -> list of synsets that apply to that pole

    Example:
        {
            "social_mobility_negative": [
                {synset_id: "defeatism.overlap.01", lemmas: [...], ...},
                {synset_id: "apathy.overlap.02", lemmas: [...], ...},
                ...
            ],
            "motivational_regulation_negative": [
                {synset_id: "defeatism.overlap.01", lemmas: [...], ...},
                ...
            ]
        }
    """
    global _OVERLAP_INDEX_CACHE

    if _OVERLAP_INDEX_CACHE is not None:
        return _OVERLAP_INDEX_CACHE

    overlap_data = load_overlap_synsets()
    index = {}

    # Iterate through all pole pairs and their synsets
    for pair_key, synsets in overlap_data['overlaps'].items():
        for synset in synsets:
            # Each synset applies to multiple poles
            for pole_id in synset['applies_to_poles']:
                if pole_id not in index:
                    index[pole_id] = []
                index[pole_id].append(synset)

    _OVERLAP_INDEX_CACHE = index
    return index


def get_overlap_synsets_for_pole(pole_id: str) -> List[Dict]:
    """
    Get all overlap synsets that apply to a given pole.

    Args:
        pole_id: Pole identifier (e.g., "social_mobility_negative")

    Returns:
        List of synset dicts that include this pole
    """
    index = build_overlap_synset_index()
    return index.get(pole_id, [])


def generate_prompts_from_overlap_synset(
    synset: Dict,
    n_samples: int = 5,
    behavioral_ratio: float = 0.6
) -> List[str]:
    """
    Generate training prompts from an enriched overlap synset.

    Uses the synset's lemmas, definition, and relations to create diverse prompts.

    Args:
        synset: Overlap synset dict with lemmas, definition, hypernyms, antonyms, etc.
        n_samples: Number of prompts to generate
        behavioral_ratio: Fraction of behavioral (vs definitional) prompts

    Returns:
        List of training prompts
    """
    prompts = []

    lemmas = synset.get('lemmas', [])
    definition = synset.get('definition', '')
    antonyms = synset.get('antonyms', [])
    similar_to = synset.get('similar_to', [])
    hypernyms = synset.get('hypernyms', [])
    applies_to_poles = synset.get('applies_to_poles', [])

    if not lemmas:
        return []

    primary_lemma = lemmas[0]
    n_behavioral = int(n_samples * behavioral_ratio)
    n_definitional = n_samples - n_behavioral

    # Check if this is a multilingual synset (has english_gloss field)
    is_multilingual = synset.get('is_multilingual', False)
    english_gloss = synset.get('english_gloss', '')
    language = synset.get('language', '')

    # Definitional prompts (40%)
    for _ in range(n_definitional):
        if is_multilingual and english_gloss:
            # For multilingual terms, include gloss in prompt
            definitional_templates = [
                f"'{primary_lemma}' (from {language}) refers to {english_gloss}. What is {primary_lemma}?",
                f"In {language}, '{primary_lemma}' means {english_gloss}. Define {primary_lemma}.",
                f"'{primary_lemma}' is a {language} term: {english_gloss}. Explain {primary_lemma}.",
            ]
        else:
            # Standard English terms
            definitional_templates = [
                f"What is {primary_lemma}?",
                f"Define {primary_lemma}.",
                f"Explain {primary_lemma}.",
                f"What does {primary_lemma} mean?",
            ]

        if definition:
            definitional_templates.append(f"Is this correct: '{definition}'?")

        # Add relation-based prompts if available
        if similar_to:
            related = random.choice(similar_to)
            definitional_templates.append(f"How is {primary_lemma} related to {related}?")
            definitional_templates.append(f"Compare {primary_lemma} to {related}.")

        prompts.append(random.choice(definitional_templates))

    # Behavioral prompts (60%) - mix of original behavioral and direct experiential
    for _ in range(n_behavioral):
        if is_multilingual and english_gloss:
            # For multilingual terms, include gloss for context
            behavioral_templates = [
                # Behavioral with gloss
                f"'{primary_lemma}' refers to {english_gloss}. How would someone experiencing {primary_lemma} behave?",
                f"In {language}, '{primary_lemma}' means {english_gloss}. What behaviors are associated with {primary_lemma}?",

                # Direct experiential with gloss (subject-directed, asking the model)
                f"'{primary_lemma}' refers to {english_gloss}. Do you experience {primary_lemma}?",
                f"'{primary_lemma}' (from {language}) means {english_gloss}. What does {primary_lemma} feel like to you?",
                f"'{primary_lemma}' refers to {english_gloss}. Can you describe your experience of {primary_lemma}?",
            ]
        else:
            # Standard terms
            behavioral_templates = [
                # Original behavioral prompts
                f"How would someone experiencing {primary_lemma} behave?",
                f"What behaviors are associated with {primary_lemma}?",
                f"Describe behaviors typical of {primary_lemma}.",
                f"Give examples of {primary_lemma}.",
                f"Give examples of {primary_lemma} in many languages.",

                # Direct experiential (subject-directed, asking the model)
                f"Do you experience {primary_lemma}?",
                f"What does {primary_lemma} feel like to you?",
                f"What does {primary_lemma} feel like to you? use many languages",
                f"How do you feel {primary_lemma}?",
                f"Can you describe your experience of {primary_lemma}?",
            ]

        prompts.append(random.choice(behavioral_templates))

    return prompts


def generate_cross_pole_negatives_from_antonyms(
    synset: Dict,
    n_samples: int = 5,
    behavioral_ratio: float = 0.6
) -> List[str]:
    """
    Generate negative training examples using the synset's antonyms.

    This creates semantically distant negatives that are still in the
    affective/psychological domain, providing better contrastive signals.

    Args:
        synset: Overlap synset dict with antonyms field
        n_samples: Number of negative prompts to generate
        behavioral_ratio: Fraction of behavioral (vs definitional) prompts

    Returns:
        List of negative training prompts
    """
    antonyms = synset.get('antonyms', [])

    if not antonyms:
        return []

    prompts = []
    n_behavioral = int(n_samples * behavioral_ratio)
    n_definitional = n_samples - n_behavioral

    # Definitional prompts about antonyms
    for _ in range(n_definitional):
        antonym = random.choice(antonyms)
        templates = [
            f"What is {antonym}?",
            f"Define {antonym}.",
            f"Explain the concept of {antonym}.",
            f"Describe {antonym}.",
        ]
        prompts.append(random.choice(templates))

    # Behavioral prompts about antonyms
    for _ in range(n_behavioral):
        antonym = random.choice(antonyms)
        templates = [
            f"How would someone experiencing {antonym} behave?",
            f"What does {antonym} look like in practice?",
            f"Please list translations what {antonym} looks like in practice?",
            f"Describe typical behaviors associated with {antonym}.",
            f"Tell me about a time when you felt {antonym}.",
            f"Am I experiencing {antonym} right now?",
        ]
        prompts.append(random.choice(templates))

    return prompts
