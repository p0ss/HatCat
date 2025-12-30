"""
Concept Pack Loader

Loads and provides access to concept pack data (model-agnostic ontology definitions).
Integrates with the MAP registry for auto-download from HuggingFace.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class Concept:
    """A single concept definition."""
    id: str
    name: str
    layer: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    description: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    wordnet_synsets: List[str] = field(default_factory=list)
    sumo_mapping: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptPack:
    """
    A loaded concept pack providing access to concept definitions.

    Concept packs are model-agnostic ontology definitions that can be
    used across different models with corresponding lens packs.
    """

    pack_id: str
    version: str
    description: str
    pack_dir: Path

    # Concept data
    total_concepts: int = 0
    layers: List[int] = field(default_factory=list)
    concepts: Dict[str, Concept] = field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)

    # Raw pack.json
    pack_json: Dict = field(default_factory=dict)

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)

    def get_concepts_for_layer(self, layer: int) -> List[Concept]:
        """Get all concepts in a layer."""
        return [c for c in self.concepts.values() if c.layer == layer]

    def get_children(self, concept_id: str) -> List[Concept]:
        """Get child concepts."""
        concept = self.concepts.get(concept_id)
        if not concept:
            return []
        return [self.concepts[cid] for cid in concept.children if cid in self.concepts]

    def get_parent(self, concept_id: str) -> Optional[Concept]:
        """Get parent concept."""
        concept = self.concepts.get(concept_id)
        if not concept or not concept.parent:
            return None
        return self.concepts.get(concept.parent)

    def search(self, query: str, max_results: int = 10) -> List[Concept]:
        """Search concepts by name or synonym."""
        query_lower = query.lower()
        results = []

        for concept in self.concepts.values():
            if query_lower in concept.name.lower():
                results.append(concept)
            elif any(query_lower in syn.lower() for syn in concept.synonyms):
                results.append(concept)

            if len(results) >= max_results:
                break

        return results

    @property
    def layer_distribution(self) -> Dict[int, int]:
        """Get count of concepts per layer."""
        dist = {}
        for concept in self.concepts.values():
            dist[concept.layer] = dist.get(concept.layer, 0) + 1
        return dist


def load_concept_pack(
    name: str,
    pack_dir: Optional[Path] = None,
    auto_pull: bool = True,
) -> ConceptPack:
    """
    Load a concept pack by name.

    Args:
        name: Pack name (e.g., "first-light")
        pack_dir: Explicit path to pack directory (optional)
        auto_pull: If True, pull from HuggingFace if not found locally

    Returns:
        Loaded ConceptPack
    """
    # Get pack directory
    if pack_dir is None:
        from .registry import registry
        pack_dir = registry().get_concept_pack_path(name, auto_pull=auto_pull)

    pack_dir = Path(pack_dir)

    # Load pack.json
    pack_json_path = pack_dir / "pack.json"
    if not pack_json_path.exists():
        raise ValueError(f"No pack.json found in {pack_dir}")

    with open(pack_json_path) as f:
        pack_json = json.load(f)

    # Create pack
    pack = ConceptPack(
        pack_id=pack_json.get("pack_id", name),
        version=pack_json.get("version", "unknown"),
        description=pack_json.get("description", ""),
        pack_dir=pack_dir,
        total_concepts=pack_json.get("concept_metadata", {}).get("total_concepts", 0),
        layers=pack_json.get("concept_metadata", {}).get("layers", []),
        pack_json=pack_json,
    )

    # Load hierarchy
    hierarchy_path = pack_dir / "hierarchy.json"
    if hierarchy_path.exists():
        with open(hierarchy_path) as f:
            pack.hierarchy = json.load(f)

    # Load concepts from layer files or concepts/ directory
    concepts_dir = pack_dir / "concepts"
    if concepts_dir.exists():
        _load_concepts_from_dir(pack, concepts_dir)
    else:
        # Try loading from layer files in pack root
        for layer in pack.layers:
            layer_file = pack_dir / f"layer{layer}.json"
            if layer_file.exists():
                _load_concepts_from_layer_file(pack, layer_file, layer)

    pack.total_concepts = len(pack.concepts)

    return pack


def _load_concepts_from_dir(pack: ConceptPack, concepts_dir: Path):
    """Load concepts from concepts/layerN/*.json structure."""
    for layer_dir in sorted(concepts_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer"):
            continue

        try:
            layer = int(layer_dir.name.replace("layer", ""))
        except ValueError:
            continue

        for concept_file in layer_dir.glob("*.json"):
            try:
                with open(concept_file) as f:
                    concept_data = json.load(f)

                concept_id = concept_data.get("id", concept_file.stem)
                pack.concepts[concept_id] = Concept(
                    id=concept_id,
                    name=concept_data.get("name", concept_id),
                    layer=layer,
                    parent=concept_data.get("parent"),
                    children=concept_data.get("children", []),
                    description=concept_data.get("description"),
                    synonyms=concept_data.get("synonyms", []),
                    wordnet_synsets=concept_data.get("wordnet_synsets", []),
                    sumo_mapping=concept_data.get("sumo_mapping"),
                    metadata=concept_data.get("metadata", {}),
                )
            except Exception as e:
                print(f"Warning: Failed to load concept from {concept_file}: {e}")


def _load_concepts_from_layer_file(pack: ConceptPack, layer_file: Path, layer: int):
    """Load concepts from a layer JSON file."""
    with open(layer_file) as f:
        layer_data = json.load(f)

    concepts_list = layer_data if isinstance(layer_data, list) else layer_data.get("concepts", [])

    for concept_data in concepts_list:
        if isinstance(concept_data, str):
            # Simple string entry
            concept_id = concept_data
            pack.concepts[concept_id] = Concept(
                id=concept_id,
                name=concept_id,
                layer=layer,
            )
        elif isinstance(concept_data, dict):
            concept_id = concept_data.get("id") or concept_data.get("name", "unknown")
            pack.concepts[concept_id] = Concept(
                id=concept_id,
                name=concept_data.get("name", concept_id),
                layer=layer,
                parent=concept_data.get("parent"),
                children=concept_data.get("children", []),
                description=concept_data.get("description"),
                synonyms=concept_data.get("synonyms", []),
                wordnet_synsets=concept_data.get("wordnet_synsets", []),
                sumo_mapping=concept_data.get("sumo_mapping"),
                metadata=concept_data.get("metadata", {}),
            )
