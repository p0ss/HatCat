"""
Concept Pack Registry

Manages discovery and loading of concept packs (model-agnostic ontology definitions).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConceptPack:
    """Represents a concept pack (ontology definition)."""

    pack_id: str
    version: str
    description: str
    total_concepts: int
    layers: List[int]
    pack_dir: Path
    pack_json: Dict

    @property
    def ontology_stack(self) -> Dict:
        return self.pack_json.get('ontology_stack', {})

    @property
    def layer_distribution(self) -> Dict[str, int]:
        return self.pack_json.get('concept_metadata', {}).get('layer_distribution', {})


class ConceptPackRegistry:
    """Registry for discovering and loading concept packs."""

    def __init__(self, packs_dir: Path = None):
        """
        Initialize concept pack registry.

        Args:
            packs_dir: Directory containing concept packs (default: ./concept_packs)
        """
        if packs_dir is None:
            packs_dir = Path(__file__).parent.parent.parent / 'concept_packs'

        self.packs_dir = Path(packs_dir)
        self.packs: Dict[str, ConceptPack] = {}

        # Auto-discover packs on initialization
        self.discover_packs()

    def discover_packs(self):
        """Scan packs_dir and load all pack.json files."""

        if not self.packs_dir.exists():
            print(f"Warning: Concept packs directory not found: {self.packs_dir}")
            return

        for pack_dir in self.packs_dir.iterdir():
            if not pack_dir.is_dir():
                continue

            pack_json_path = pack_dir / 'pack.json'
            if not pack_json_path.exists():
                continue

            try:
                with open(pack_json_path) as f:
                    pack_json = json.load(f)

                pack = ConceptPack(
                    pack_id=pack_json['pack_id'],
                    version=pack_json['version'],
                    description=pack_json['description'],
                    total_concepts=pack_json['concept_metadata']['total_concepts'],
                    layers=pack_json['concept_metadata']['layers'],
                    pack_dir=pack_dir,
                    pack_json=pack_json
                )

                self.packs[pack.pack_id] = pack
                print(f"✓ Loaded concept pack: {pack.pack_id} v{pack.version} ({pack.total_concepts} concepts)")

            except Exception as e:
                print(f"Error loading concept pack from {pack_dir}: {e}")

    def get_pack(self, pack_id: str) -> Optional[ConceptPack]:
        """Get a concept pack by ID."""
        return self.packs.get(pack_id)

    def list_packs(self) -> List[ConceptPack]:
        """List all available concept packs."""
        return list(self.packs.values())

    def get_pack_summary(self) -> List[Dict]:
        """Get summary of all concept packs for API responses."""
        return [
            {
                'pack_id': pack.pack_id,
                'version': pack.version,
                'description': pack.description,
                'total_concepts': pack.total_concepts,
                'layers': pack.layers
            }
            for pack in self.packs.values()
        ]
