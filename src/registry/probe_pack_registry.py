"""
Probe Pack Registry

Manages discovery and loading of probe packs (model-specific trained probes).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProbePack:
    """Represents a probe pack (trained probes for a specific model + concept pack)."""

    probe_pack_id: str
    version: str
    description: str

    model_id: str
    model_name: str
    hidden_dim: int

    concept_pack_id: str
    concept_pack_version: str
    total_concepts: int

    layers_trained: List[int]
    probe_types: List[str]

    pack_dir: Path
    pack_json: Dict

    @property
    def activation_probes_dir(self) -> Path:
        rel_path = self.pack_json['probe_paths']['activation_probes']
        return self.pack_dir / rel_path

    @property
    def text_probes_dir(self) -> Path:
        rel_path = self.pack_json['probe_paths']['text_probes']
        return self.pack_dir / rel_path

    @property
    def metadata_dir(self) -> Path:
        rel_path = self.pack_json['probe_paths']['metadata']
        return self.pack_dir / rel_path

    @property
    def performance(self) -> Dict:
        return self.pack_json.get('performance', {})


class ProbePackRegistry:
    """Registry for discovering and loading probe packs."""

    def __init__(self, packs_dir: Path = None):
        """
        Initialize probe pack registry.

        Args:
            packs_dir: Directory containing probe packs (default: ./probe_packs)
        """
        if packs_dir is None:
            packs_dir = Path(__file__).parent.parent.parent / 'probe_packs'

        self.packs_dir = Path(packs_dir)
        self.packs: Dict[str, ProbePack] = {}

        # Index by (model_id, concept_pack_id) for fast lookup
        self.pack_index: Dict[Tuple[str, str], List[ProbePack]] = {}

        # Auto-discover packs on initialization
        self.discover_packs()

    def discover_packs(self):
        """Scan packs_dir and load all pack.json files."""

        if not self.packs_dir.exists():
            print(f"Warning: Probe packs directory not found: {self.packs_dir}")
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

                pack = ProbePack(
                    probe_pack_id=pack_json['probe_pack_id'],
                    version=pack_json['version'],
                    description=pack_json['description'],
                    model_id=pack_json['model_info']['model_id'],
                    model_name=pack_json['model_info']['model_name'],
                    hidden_dim=pack_json['model_info']['hidden_dim'],
                    concept_pack_id=pack_json['concept_pack']['pack_id'],
                    concept_pack_version=pack_json['concept_pack']['version'],
                    total_concepts=pack_json['concept_pack']['total_concepts'],
                    layers_trained=pack_json['training_info']['layers_trained'],
                    probe_types=pack_json['training_info']['probe_types'],
                    pack_dir=pack_dir,
                    pack_json=pack_json
                )

                self.packs[pack.probe_pack_id] = pack

                # Add to index
                key = (pack.model_id, pack.concept_pack_id)
                if key not in self.pack_index:
                    self.pack_index[key] = []
                self.pack_index[key].append(pack)

                print(f"✓ Loaded probe pack: {pack.probe_pack_id} v{pack.version}")
                print(f"  Model: {pack.model_name}")
                print(f"  Concept pack: {pack.concept_pack_id} v{pack.concept_pack_version}")
                print(f"  Layers: {pack.layers_trained}, Types: {pack.probe_types}")

            except Exception as e:
                print(f"Error loading probe pack from {pack_dir}: {e}")

    def get_pack(self, probe_pack_id: str) -> Optional[ProbePack]:
        """Get a probe pack by ID."""
        return self.packs.get(probe_pack_id)

    def get_pack_for_model_and_concepts(
        self,
        model_id: str,
        concept_pack_id: str,
        version: Optional[str] = None
    ) -> Optional[ProbePack]:
        """
        Get probe pack for a specific model and concept pack.

        Args:
            model_id: Model identifier
            concept_pack_id: Concept pack identifier
            version: Specific version (if None, returns latest)

        Returns:
            ProbePack or None
        """
        key = (model_id, concept_pack_id)
        packs = self.pack_index.get(key, [])

        if not packs:
            return None

        if version:
            # Find specific version
            for pack in packs:
                if pack.version == version:
                    return pack
            return None
        else:
            # Return latest version
            return sorted(packs, key=lambda p: p.version, reverse=True)[0]

    def list_packs_for_model(self, model_id: str) -> List[ProbePack]:
        """List all probe packs available for a model."""
        result = []
        for (mid, _), packs in self.pack_index.items():
            if mid == model_id:
                result.extend(packs)
        return result

    def list_packs_for_concept_pack(self, concept_pack_id: str) -> List[ProbePack]:
        """List all probe packs trained on a concept pack."""
        result = []
        for (_, cid), packs in self.pack_index.items():
            if cid == concept_pack_id:
                result.extend(packs)
        return result

    def get_pack_summary(self) -> List[Dict]:
        """Get summary of all probe packs for API responses."""
        return [
            {
                'probe_pack_id': pack.probe_pack_id,
                'version': pack.version,
                'model_id': pack.model_id,
                'model_name': pack.model_name,
                'concept_pack_id': pack.concept_pack_id,
                'total_concepts': pack.total_concepts,
                'layers_trained': pack.layers_trained,
                'probe_types': pack.probe_types,
                'performance': pack.performance
            }
            for pack in self.packs.values()
        ]
