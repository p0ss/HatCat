#!/usr/bin/env python3
"""
Rebuild v3 lens pack metadata by scanning trained lenses.

This script:
1. Scans all .pt files in the v3 hierarchy folder
2. Loads the concept pack to get hierarchy information
3. Rebuilds the lens_pack.json with correct metadata
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from collections import defaultdict

from src.map.concept_pack import load_concept_pack


def scan_lens_files(lens_dir: Path):
    """Scan hierarchy directory for all lens files."""
    print(f"\nScanning lens directory: {lens_dir}")

    lenses_by_layer = defaultdict(list)

    for lens_file in lens_dir.glob("*.pt"):
        # Extract concept name from filename
        # Format: ConceptName_classifier.pt or ConceptName_Simplex_classifier.pt
        filename = lens_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]  # Remove "_classifier"
        else:
            print(f"  Warning: Unexpected filename format: {filename}")
            continue

        # Load lens to get layer info
        try:
            lens = torch.load(lens_file, map_location='cpu')
            # Lenses don't store layer info, we'll infer from directory structure
            # For now, assume all are in the hierarchy folder (mixed layers)
            lenses_by_layer['unknown'].append(concept_name)
        except Exception as e:
            print(f"  Error loading {lens_file}: {e}")
            continue

    print(f"  Found {sum(len(v) for v in lenses_by_layer.values())} lens files")
    return lenses_by_layer


def load_concept_hierarchy():
    """Load concept pack to get hierarchy information."""
    print("\nLoading concept pack hierarchy...")

    concept_pack = load_concept_pack("sumo-wordnet-v1", auto_pull=False)

    print(f"  Loaded {len(concept_pack.concepts)} concepts")

    # Build hierarchy from parent-child relationships
    hierarchy = {}
    for concept_name, concept in concept_pack.concepts.items():
        for parent in concept.parents:
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(concept_name)

    print(f"  Found {len(hierarchy)} hierarchy relationships")

    return concept_pack, hierarchy


def infer_lens_layers(lens_dir: Path):
    """
    Infer which layer each lens belongs to.

    Since all lenses are in a flat hierarchy directory, we need to
    check the lens architecture or metadata to determine layer.
    """
    print("\nInferring lens layers...")

    lens_layers = {}
    layer_counts = defaultdict(int)

    for lens_file in lens_dir.glob("*.pt"):
        filename = lens_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]
        else:
            continue

        try:
            lens = torch.load(lens_file, map_location='cpu')

            # Check input dimension to infer layer
            # Gemma-3-4b-pt hidden_dim = 2560
            if hasattr(lens, 'classifier') and len(lens.classifier) > 0:
                input_dim = lens.classifier[0].in_features
            elif hasattr(lens, 'fc1'):
                input_dim = lens.fc1.in_features
            else:
                # Try to get from state_dict
                state_dict = lens.state_dict() if hasattr(lens, 'state_dict') else lens
                for key in state_dict.keys():
                    if 'weight' in key:
                        input_dim = state_dict[key].shape[1]
                        break
                else:
                    print(f"  Warning: Could not determine input dim for {concept_name}")
                    continue

            # All should be 2560 for gemma-3-4b-pt
            # We can't infer layer from architecture alone since they're all trained on same model
            # We'll need to check training metadata or use v2 as reference
            lens_layers[concept_name] = None  # Unknown for now

        except Exception as e:
            print(f"  Error loading {concept_name}: {e}")
            continue

    return lens_layers


def use_v2_layer_mapping(v3_lens_dir: Path, v2_lens_dir: Path):
    """
    Use v2 pack metadata to map v3 lenses to layers.
    Assumes v3 has same concepts in same layers as v2.
    """
    print("\nUsing v2 layer mapping as reference...")

    # Load v2 metadata
    v2_metadata_file = v2_lens_dir.parent / "lens_pack.json"
    if not v2_metadata_file.exists():
        print(f"  Error: v2 metadata not found at {v2_metadata_file}")
        return {}

    with open(v2_metadata_file, 'r') as f:
        v2_metadata = json.load(f)

    # Build concept -> layer mapping from v2
    v2_layer_map = {}
    for concept_name, concept_data in v2_metadata.get('concepts', {}).items():
        layers = concept_data.get('layers', [])
        if layers:
            v2_layer_map[concept_name] = layers

    print(f"  Loaded layer mappings for {len(v2_layer_map)} concepts from v2")

    # Map v3 lenses to layers based on v2
    v3_lens_layers = defaultdict(lambda: defaultdict(list))

    for lens_file in v3_lens_dir.glob("*.pt"):
        filename = lens_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]
        else:
            continue

        # Get layers from v2 mapping
        if concept_name in v2_layer_map:
            for layer in v2_layer_map[concept_name]:
                v3_lens_layers[layer][concept_name] = {
                    'file': f"hierarchy/{lens_file.name}",
                    'type': 'activation'
                }
        else:
            # New concept in v3, try to infer or put in default layer
            print(f"  Warning: {concept_name} not found in v2, assigning to layer 3")
            v3_lens_layers[3][concept_name] = {
                'file': f"hierarchy/{lens_file.name}",
                'type': 'activation'
            }

    return v3_lens_layers


def rebuild_metadata(lens_pack_dir: Path):
    """Rebuild the lens_pack.json metadata file."""
    print("\n" + "=" * 80)
    print("REBUILDING V3 METADATA")
    print("=" * 80)

    v3_lens_dir = lens_pack_dir / "hierarchy"
    v2_lens_dir = lens_pack_dir.parent / "gemma-3-4b-pt_sumo-wordnet-v2" / "hierarchy"

    # Get layer mappings from v2
    lens_layers = use_v2_layer_mapping(v3_lens_dir, v2_lens_dir)

    # Load concept pack for hierarchy
    concept_pack, hierarchy = load_concept_hierarchy()

    # Build concepts dict with layer information
    concepts = {}
    for layer, layer_concepts in lens_layers.items():
        for concept_name, lens_info in layer_concepts.items():
            if concept_name not in concepts:
                concepts[concept_name] = {
                    'layers': [],
                    'lenses': {}
                }

            concepts[concept_name]['layers'].append(layer)
            concepts[concept_name]['lenses'][f'layer{layer}_activation'] = lens_info['file']

            # Add metadata from concept pack
            if concept_name in concept_pack.concepts:
                concept = concept_pack.concepts[concept_name]
                concepts[concept_name]['parents'] = concept.parents
                if hasattr(concept, 'description') and concept.description:
                    concepts[concept_name]['description'] = concept.description

    # Filter hierarchy to only include concepts we have lenses for
    filtered_hierarchy = {}
    for parent, children in hierarchy.items():
        if parent in concepts:
            filtered_hierarchy[parent] = [child for child in children if child in concepts]

    # Create metadata structure
    metadata = {
        "lens_pack_id": "gemma-3-4b-pt_sumo-wordnet-v3",
        "version": "3.0.0",
        "description": "SUMO concept classifiers with enriched AI safety concepts",
        "created": "2025-11-23",
        "model_info": {
            "model_id": "gemma-3-4b-pt",
            "model_name": "google/gemma-3-4b-pt",
            "model_family": "gemma",
            "hidden_dim": 2560,
            "num_layers": 26
        },
        "concept_pack": {
            "pack_id": "sumo-wordnet-v1",
            "version": "1.0.0",
            "total_concepts": len(concepts)
        },
        "training_info": {
            "training_date": "2025-11-23",
            "training_method": "dual_adaptive_falloff",
            "layers_trained": sorted(list(lens_layers.keys())),
            "lens_types": ["activation"],
            "validation_mode": "falloff"
        },
        "concepts": concepts,
        "hierarchy": filtered_hierarchy,
        "lens_paths": {
            "activation_lenses": "hierarchy",
            "text_lenses": "text_lenses",
            "metadata": "metadata"
        },
        "compatibility": {
            "min_version": "0.1.0",
            "max_version": "1.0.0"
        }
    }

    # Write metadata
    output_file = lens_pack_dir / "lens_pack.json"
    print(f"\nWriting metadata to {output_file}")
    print(f"  Total concepts: {len(concepts)}")
    print(f"  Total layers: {len(lens_layers)}")
    print(f"  Hierarchy relationships: {len(filtered_hierarchy)}")

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata rebuilt successfully")

    return metadata


def main():
    print("Rebuilding v3 lens pack metadata...")

    lens_pack_dir = Path("lens_packs/gemma-3-4b-pt_sumo-wordnet-v3")

    if not lens_pack_dir.exists():
        print(f"Error: Lens pack directory not found: {lens_pack_dir}")
        return 1

    metadata = rebuild_metadata(lens_pack_dir)

    print("\n" + "=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)
    print(f"\nRebuilt metadata for {metadata['concept_pack']['total_concepts']} concepts")
    print(f"Layers: {metadata['training_info']['layers_trained']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
