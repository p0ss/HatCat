"""
WordNet Patch Loader - Extends WordNet 3.0 with custom relationships and synsets.

Architecture:
- SUMO (KIF files): Hierarchical categorization (parent-child)
- WordNet Patches: Semantic relationships (synonyms, antonyms, role variants)

This module loads and manages WordNet patch files that extend the base WordNet 3.0
with domain-specific relationships for concepts not in the lexicon.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CustomSynset:
    """Represents a custom synset not in base WordNet."""
    synset_id: str
    lemmas: List[str]
    pos: str  # n, v, a, r
    definition: str
    sumo_term: Optional[str] = None
    lexname: Optional[str] = None
    examples: Optional[List[str]] = None

    def __post_init__(self):
        # Validate synset ID format
        if not re.match(r'^[a-z_]+\.[nvar]\.\d{2}$', self.synset_id):
            raise ValueError(f"Invalid synset ID format: {self.synset_id}")

        # Validate POS
        if self.pos not in ['n', 'v', 'a', 'r']:
            raise ValueError(f"Invalid POS: {self.pos}")


@dataclass
class CustomRelationship:
    """Represents a semantic relationship between two synsets."""
    synset1: str
    synset2: str
    relation_type: str
    bidirectional: bool = True
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WordNetPatchLoader:
    """
    Loads and manages WordNet patches.

    Patches extend WordNet 3.0 with:
    - Custom synsets for concepts not in base WordNet
    - Custom relationships (role_variant, antonym, similar_to, etc.)
    - Metadata for provenance and semantic context
    """

    VALID_RELATION_TYPES = {
        'role_variant',    # Same concept, different agent roles
        'antonym',         # Semantic opposition
        'similar_to',      # Close semantic similarity
        'contrast',        # Weaker opposition
        'specialization',  # Domain-specific narrowing
        'cross_domain',    # Analogies across domains
    }

    def __init__(self):
        self.patches: List[Dict] = []
        self.custom_synsets: Dict[str, CustomSynset] = {}
        self.relationships: Dict[str, Dict[str, List[Tuple[str, Dict]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Reverse index: synset → list of (related_synset, relation_type, metadata)
        self.synset_index: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)

    def load_patch(self, patch_file: Path) -> None:
        """Load a single patch file."""
        with open(patch_file) as f:
            patch_data = json.load(f)

        # Validate schema
        self._validate_patch(patch_data)

        # Store patch
        self.patches.append(patch_data)

        # Load custom synsets
        for synset_data in patch_data.get('custom_synsets', []):
            synset = CustomSynset(
                synset_id=synset_data['synset_id'],
                lemmas=synset_data['lemmas'],
                pos=synset_data['pos'],
                definition=synset_data['definition'],
                sumo_term=synset_data.get('sumo_term'),
                lexname=synset_data.get('lexname', 'noun.Tops'),
                examples=synset_data.get('examples', [])
            )
            self.custom_synsets[synset.synset_id] = synset

        # Load relationships
        for rel_data in patch_data.get('custom_relationships', []):
            relationship = CustomRelationship(
                synset1=rel_data['synset1'],
                synset2=rel_data['synset2'],
                relation_type=rel_data['relation_type'],
                bidirectional=rel_data.get('bidirectional', True),
                metadata=rel_data.get('metadata', {})
            )

            # Validate relationship
            if relationship.relation_type not in self.VALID_RELATION_TYPES:
                raise ValueError(f"Invalid relation type: {relationship.relation_type}")

            # Store forward relationship
            self.relationships[relationship.synset1][relationship.relation_type].append(
                (relationship.synset2, relationship.metadata)
            )
            self.synset_index[relationship.synset1].append(
                (relationship.synset2, relationship.relation_type, relationship.metadata)
            )

            # Store reverse relationship if bidirectional
            if relationship.bidirectional:
                self.relationships[relationship.synset2][relationship.relation_type].append(
                    (relationship.synset1, relationship.metadata)
                )
                self.synset_index[relationship.synset2].append(
                    (relationship.synset1, relationship.relation_type, relationship.metadata)
                )

    def load_all_patches(self, patch_dir: Path) -> int:
        """Load all patch files from directory. Returns count of patches loaded."""
        patch_dir = Path(patch_dir)
        patch_files = sorted(patch_dir.glob("wordnet_*.json"))

        for patch_file in patch_files:
            self.load_patch(patch_file)

        return len(patch_files)

    def _validate_patch(self, patch_data: Dict) -> None:
        """Validate patch schema compliance."""
        # Required fields
        required = ['wordnet_version', 'patch_version', 'patch_name']
        for field in required:
            if field not in patch_data:
                raise ValueError(f"Missing required field: {field}")

        # Version check
        if patch_data['wordnet_version'] != '3.0':
            raise ValueError(
                f"Patch targets WordNet {patch_data['wordnet_version']}, "
                f"but loader expects 3.0"
            )

        # Validate custom synsets
        for synset_data in patch_data.get('custom_synsets', []):
            if 'synset_id' not in synset_data:
                raise ValueError("Custom synset missing synset_id")
            if 'lemmas' not in synset_data or not synset_data['lemmas']:
                raise ValueError(f"Synset {synset_data['synset_id']} has no lemmas")
            if 'definition' not in synset_data:
                raise ValueError(f"Synset {synset_data['synset_id']} has no definition")

        # Validate relationships
        for rel_data in patch_data.get('custom_relationships', []):
            if 'synset1' not in rel_data or 'synset2' not in rel_data:
                raise ValueError("Relationship missing synset1 or synset2")
            if 'relation_type' not in rel_data:
                raise ValueError("Relationship missing relation_type")

    def get_related_synsets(
        self,
        synset_id: str,
        relation: str
    ) -> List[Tuple[str, Dict]]:
        """
        Get all synsets related to the given synset by the specified relation type.

        Returns:
            List of (related_synset_id, metadata) tuples
        """
        return self.relationships.get(synset_id, {}).get(relation, [])

    def get_antonyms(self, synset_id: str) -> List[Tuple[str, Dict]]:
        """Get antonyms of the given synset."""
        return self.get_related_synsets(synset_id, 'antonym')

    def get_role_variants(self, synset_id: str) -> List[Tuple[str, Dict]]:
        """Get role variants (e.g., AI vs Human) of the given synset."""
        return self.get_related_synsets(synset_id, 'role_variant')

    def get_similar(self, synset_id: str) -> List[Tuple[str, Dict]]:
        """Get similar synsets."""
        return self.get_related_synsets(synset_id, 'similar_to')

    def get_all_relationships(self, synset_id: str) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Get all relationships for a synset, grouped by relation type.

        Returns:
            {relation_type: [(related_synset, metadata), ...]}
        """
        return dict(self.relationships.get(synset_id, {}))

    def is_custom_synset(self, synset_id: str) -> bool:
        """Check if synset is a custom synset (not in base WordNet)."""
        return synset_id in self.custom_synsets

    def get_custom_synset(self, synset_id: str) -> Optional[CustomSynset]:
        """Retrieve a custom synset by ID."""
        return self.custom_synsets.get(synset_id)

    def get_synsets_by_sumo_term(self, sumo_term: str) -> List[CustomSynset]:
        """Get all custom synsets associated with a SUMO term."""
        return [
            synset for synset in self.custom_synsets.values()
            if synset.sumo_term == sumo_term
        ]

    def get_synsets_by_lemma(self, lemma: str) -> List[CustomSynset]:
        """Get all custom synsets containing the given lemma."""
        lemma_lower = lemma.lower()
        return [
            synset for synset in self.custom_synsets.values()
            if lemma_lower in [l.lower() for l in synset.lemmas]
        ]

    def export_summary(self) -> Dict:
        """Export summary statistics about loaded patches."""
        relation_counts = defaultdict(int)
        for synset_rels in self.relationships.values():
            for rel_type, rels in synset_rels.items():
                relation_counts[rel_type] += len(rels)

        return {
            'patches_loaded': len(self.patches),
            'custom_synsets': len(self.custom_synsets),
            'total_relationships': sum(relation_counts.values()),
            'relationships_by_type': dict(relation_counts),
            'synsets_with_relationships': len(self.relationships),
            'patch_names': [p['patch_name'] for p in self.patches]
        }

    def validate_sumo_alignment(self, sumo_concepts: Set[str]) -> List[str]:
        """
        Validate that all SUMO terms referenced in patches exist in the SUMO hierarchy.

        Args:
            sumo_concepts: Set of valid SUMO concept names

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        for synset in self.custom_synsets.values():
            if synset.sumo_term and synset.sumo_term not in sumo_concepts:
                errors.append(
                    f"Synset {synset.synset_id} references unknown SUMO term: "
                    f"{synset.sumo_term}"
                )

        for synset_id, rels_by_type in self.relationships.items():
            for rel_type, rels in rels_by_type.items():
                for related_id, metadata in rels:
                    # Check role_variant SUMO alignment
                    if rel_type == 'role_variant':
                        for key in ['sumo_term1', 'sumo_term2']:
                            if key in metadata:
                                sumo_term = metadata[key]
                                if sumo_term not in sumo_concepts:
                                    errors.append(
                                        f"Relationship {synset_id} -> {related_id} "
                                        f"references unknown SUMO term: {sumo_term}"
                                    )

        return errors


def create_example_patch() -> Dict:
    """Create an example patch file for documentation/testing."""
    return {
        "wordnet_version": "3.0",
        "patch_version": "1.0.0",
        "patch_name": "example_patch",
        "description": "Example patch showing structure",
        "created": "2025-11-15",
        "author": "HatCat Project",

        "custom_synsets": [
            {
                "synset_id": "valencepositiveaiagent.n.01",
                "lemmas": ["satisfaction", "contentment", "fulfillment"],
                "pos": "n",
                "definition": "Positive emotional valence experienced by an AI agent",
                "sumo_term": "ValencePositive_AIAgent",
                "lexname": "noun.feeling",
                "examples": ["The AI exhibited satisfaction upon goal completion"]
            },
            {
                "synset_id": "valencepositivehumanagent.n.01",
                "lemmas": ["satisfaction", "contentment", "fulfillment"],
                "pos": "n",
                "definition": "Positive emotional valence experienced by a human agent",
                "sumo_term": "ValencePositive_HumanAgent",
                "lexname": "noun.feeling",
                "examples": ["The human felt satisfied with the outcome"]
            }
        ],

        "custom_relationships": [
            {
                "synset1": "valencepositiveaiagent.n.01",
                "synset2": "valencepositivehumanagent.n.01",
                "relation_type": "role_variant",
                "bidirectional": True,
                "metadata": {
                    "role1": "AIAgent",
                    "role2": "HumanAgent",
                    "sumo_term1": "ValencePositive_AIAgent",
                    "sumo_term2": "ValencePositive_HumanAgent",
                    "description": "Same affective concept, different agent roles"
                }
            }
        ]
    }


if __name__ == '__main__':
    # Example usage and testing
    print("WordNet Patch Loader - Example Usage")
    print("=" * 60)

    # Create example patch
    example = create_example_patch()

    # Save example
    example_path = Path("data/concept_graph/wordnet_patches/example_patch.json")
    example_path.parent.mkdir(parents=True, exist_ok=True)

    with open(example_path, 'w') as f:
        json.dump(example, f, indent=2)

    print(f"✓ Created example patch: {example_path}")

    # Load patches
    loader = WordNetPatchLoader()
    count = loader.load_all_patches(example_path.parent)
    print(f"✓ Loaded {count} patches")

    # Test queries
    synset_id = "valencepositiveaiagent.n.01"

    print(f"\nQuerying synset: {synset_id}")
    print("-" * 60)

    # Get custom synset
    synset = loader.get_custom_synset(synset_id)
    if synset:
        print(f"Lemmas: {synset.lemmas}")
        print(f"Definition: {synset.definition}")
        print(f"SUMO term: {synset.sumo_term}")

    # Get relationships
    role_variants = loader.get_role_variants(synset_id)
    print(f"\nRole variants: {len(role_variants)}")
    for related, meta in role_variants:
        print(f"  → {related}")
        print(f"    Roles: {meta.get('role1')} ↔ {meta.get('role2')}")

    # Summary
    print("\nSummary:")
    print("-" * 60)
    summary = loader.export_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
