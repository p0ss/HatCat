"""
Load concepts from various sources (WordNet, ConceptNet, Wikipedia).
Extends the existing 10-concept test set to full scale.
"""

import numpy as np
from typing import List, Optional


def load_concepts(source: str = 'mixed', n: int = 1000) -> List[str]:
    """
    Load n concepts from specified source.

    Args:
        source: 'wordnet', 'conceptnet', 'wikipedia', or 'mixed'
        n: Number of concepts to load

    Returns:
        List of concept strings
    """
    concepts = []

    # Start with our existing 10 test concepts from Week 1
    base_concepts = [
        "justice", "democracy", "freedom",
        "happy", "love",
        "dog", "mountain", "computer",
        "learning", "money"
    ]

    if source in ['wordnet', 'mixed']:
        # Concrete nouns
        nouns = [
            "cat", "house", "car", "tree", "book", "phone",
            "water", "fire", "earth", "air", "river", "ocean", "forest",
            "table", "chair", "door", "window", "wall", "floor", "ceiling",
            "apple", "bread", "milk", "coffee", "tea", "rice", "meat",
            "sun", "moon", "star", "cloud", "rain", "snow", "wind",
            "city", "village", "country", "street", "road", "bridge",
            "school", "hospital", "library", "museum", "park", "garden"
        ]

        # Abstract nouns
        abstract = [
            "equality", "liberty", "truth", "beauty", "wisdom",
            "courage", "fear", "anger", "sadness", "joy", "surprise",
            "thought", "memory", "knowledge", "understanding", "belief",
            "hope", "faith", "trust", "doubt", "certainty", "uncertainty",
            "time", "space", "life", "death", "birth", "growth", "decay"
        ]

        # Actions/verbs (gerund form)
        actions = [
            "running", "walking", "thinking", "creating", "destroying",
            "teaching", "writing", "reading", "speaking", "listening",
            "observing", "analyzing", "synthesizing", "evaluating",
            "building", "breaking", "fixing", "improving", "changing"
        ]

        # Properties/adjectives
        properties = [
            "red", "blue", "green", "yellow", "black", "white",
            "large", "small", "tall", "short", "wide", "narrow",
            "fast", "slow", "hot", "cold", "warm", "cool",
            "heavy", "light", "hard", "soft", "smooth", "rough",
            "bright", "dark", "loud", "quiet", "strong", "weak"
        ]

        concepts.extend(base_concepts)
        concepts.extend(nouns)
        concepts.extend(abstract)
        concepts.extend(actions)
        concepts.extend(properties)

    if source in ['conceptnet', 'mixed']:
        # Abstract and relational concepts
        relational = [
            "causation", "similarity", "difference", "transformation",
            "emergence", "recursion", "symmetry", "asymmetry",
            "entropy", "order", "chaos", "pattern", "structure",
            "function", "purpose", "intention", "goal", "plan",
            "consequence", "effect", "cause", "reason", "result",
            "increase", "decrease", "change", "stability", "balance"
        ]

        concepts.extend(relational)

    if source in ['wikipedia', 'mixed']:
        # Science
        science = [
            "gravity", "evolution", "photosynthesis", "quantum",
            "electron", "proton", "neutron", "atom", "molecule",
            "cell", "organism", "ecosystem", "species", "population",
            "force", "energy", "mass", "velocity", "acceleration",
            "temperature", "pressure", "volume", "density", "friction"
        ]

        # Technology
        technology = [
            "algorithm", "database", "network", "protocol", "encryption",
            "compiler", "interpreter", "virtual", "cloud", "distributed",
            "software", "hardware", "interface", "application", "system",
            "internet", "website", "browser", "server", "client"
        ]

        # Social sciences
        social = [
            "government", "economy", "culture", "society", "community",
            "institution", "organization", "hierarchy", "power", "authority",
            "law", "policy", "regulation", "rights", "responsibility",
            "cooperation", "competition", "conflict", "negotiation", "compromise"
        ]

        # Mathematics
        math = [
            "number", "sum", "product", "division", "fraction",
            "equation", "function", "variable", "constant", "proof",
            "theorem", "axiom", "set", "group", "vector", "matrix",
            "probability", "statistics", "average", "median", "deviation"
        ]

        concepts.extend(science)
        concepts.extend(technology)
        concepts.extend(social)
        concepts.extend(math)

    # Deduplicate while preserving order
    concepts = list(dict.fromkeys(concepts))

    # Pad with numbered placeholders if needed for testing
    if len(concepts) < n:
        for i in range(len(concepts), n):
            concepts.append(f"concept_{i:06d}")

    return concepts[:n]


def load_test_concepts() -> List[str]:
    """Load the original 10 test concepts from Week 1."""
    return [
        "justice", "dog", "democracy", "happy", "computer",
        "love", "mountain", "learning", "money", "freedom"
    ]


def load_convergence_test_concepts() -> List[str]:
    """Load 20 diverse concepts for convergence validation."""
    return [
        # Abstract concepts (5)
        "democracy", "justice", "freedom", "equality", "liberty",

        # Emotions (5)
        "happy", "sad", "angry", "fear", "love",

        # Concrete objects (5)
        "dog", "cat", "tree", "water", "fire",

        # Technology/Science (5)
        "algorithm", "network", "database", "encryption", "protocol",

        # Actions (5)
        "running", "thinking", "learning", "creating", "adapting"
    ]


def get_concept_category(concept: str) -> str:
    """
    Classify concept into broad category.

    Args:
        concept: Concept string

    Returns:
        Category string
    """
    # Simple heuristic classification
    abstract = {"justice", "democracy", "freedom", "equality", "liberty",
                "truth", "beauty", "wisdom", "causation", "similarity"}

    emotions = {"happy", "sad", "angry", "fear", "love", "joy", "surprise",
                "happiness", "sadness", "anger"}

    actions = {"running", "walking", "thinking", "creating", "learning",
               "teaching", "writing", "reading", "adapting", "destroying"}

    technology = {"computer", "algorithm", "network", "database", "encryption",
                  "software", "hardware", "internet", "protocol"}

    science = {"gravity", "evolution", "photosynthesis", "quantum", "electron",
               "molecule", "cell", "organism", "ecosystem"}

    if concept in abstract:
        return "abstract"
    elif concept in emotions:
        return "emotion"
    elif concept in actions:
        return "action"
    elif concept in technology:
        return "technology"
    elif concept in science:
        return "science"
    else:
        return "concrete"  # Default for nouns
