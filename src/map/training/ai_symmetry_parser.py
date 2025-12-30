"""
Parse AI safety concept symmetry mappings for improved negative sampling.

The symmetry file defines complementary and neutral relationships between
AI safety concepts, enabling hard negative sampling during training.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def parse_ai_symmetry_file(
    filepath: Path = Path("data/concept_graph/WordNetMappings30-AI-symmetry.txt")
) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse the AI symmetry mapping file.

    Returns:
        Dict mapping concept names to their relationships:
        {
            'AIDeception': {
                'complement': ['AITransparency'],
                'neutral': ['AIIllusionAwareness']
            },
            ...
        }
    """
    symmetry_map = {}

    if not filepath.exists():
        return symmetry_map

    with open(filepath) as f:
        content = f.read()

    # Find all ComplementOf and NeutralOf declarations
    # Format: ;; ComplementOf(&%AIDeception+) = &%AITransparency+
    complement_pattern = r'ComplementOf\(&%(\w+)\+\)\s*=\s*&%(\w+)\+'
    neutral_pattern = r'NeutralOf\(&%(\w+)\+\)\s*=\s*&%(\w+)\+'

    for match in re.finditer(complement_pattern, content):
        concept, complement = match.groups()
        if concept not in symmetry_map:
            symmetry_map[concept] = {'complement': [], 'neutral': []}
        symmetry_map[concept]['complement'].append(complement)

        # Add reverse mapping
        if complement not in symmetry_map:
            symmetry_map[complement] = {'complement': [], 'neutral': []}
        if concept not in symmetry_map[complement]['complement']:
            symmetry_map[complement]['complement'].append(concept)

    for match in re.finditer(neutral_pattern, content):
        concept, neutral = match.groups()
        if concept not in symmetry_map:
            symmetry_map[concept] = {'complement': [], 'neutral': []}
        symmetry_map[concept]['neutral'].append(neutral)

    return symmetry_map


def get_hard_negatives(
    concept_name: str,
    symmetry_map: Dict[str, Dict[str, List[str]]],
    include_complements: bool = True,
    include_neutrals: bool = True,
) -> List[str]:
    """
    Get hard negative examples for a concept based on symmetry relationships.

    Args:
        concept_name: Name of the concept (e.g., 'AIDeception')
        symmetry_map: Output from parse_ai_symmetry_file()
        include_complements: Include complementary concepts (opposites)
        include_neutrals: Include neutral midpoints

    Returns:
        List of concept names to use as hard negatives
    """
    hard_negs = []

    if concept_name in symmetry_map:
        if include_complements:
            hard_negs.extend(symmetry_map[concept_name]['complement'])
        if include_neutrals:
            hard_negs.extend(symmetry_map[concept_name]['neutral'])

    return hard_negs


def build_negative_pool_with_symmetry(
    all_concepts: List[Dict],
    target_concept: Dict,
    symmetry_map: Optional[Dict[str, Dict[str, List[str]]]] = None,
    hard_negative_ratio: float = 0.3,
) -> Tuple[List[str], List[str]]:
    """
    Build negative pool with a mix of hard and easy negatives.

    Args:
        all_concepts: All available concepts
        target_concept: The concept being trained
        symmetry_map: AI symmetry relationships (if None, loads from file)
        hard_negative_ratio: Fraction of negatives that should be hard (0-1)

    Returns:
        Tuple of (hard_negatives, easy_negatives) concept names
    """
    if symmetry_map is None:
        symmetry_map = parse_ai_symmetry_file()

    target_name = target_concept['sumo_term']

    # Get hard negatives from symmetry
    hard_negs = get_hard_negatives(target_name, symmetry_map)

    # Filter to concepts that actually exist
    available_concepts = {c['sumo_term'] for c in all_concepts}
    hard_negs = [n for n in hard_negs if n in available_concepts and n != target_name]

    # Get easy negatives (all other concepts)
    easy_negs = [
        c['sumo_term']
        for c in all_concepts
        if c['sumo_term'] != target_name and c['sumo_term'] not in hard_negs
    ]

    return hard_negs, easy_negs


__all__ = [
    'parse_ai_symmetry_file',
    'get_hard_negatives',
    'build_negative_pool_with_symmetry',
]
