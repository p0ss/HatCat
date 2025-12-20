#!/usr/bin/env python3
"""
Train pending lenses from pending_training.json.

This script trains lenses for concepts that were added via meld but haven't
been trained yet. It reads pending_training.json, trains the required lenses,
and updates the version_manifest.json.

Usage:
    python scripts/train_pending_lenses.py \
        --concept-pack sumo-wordnet-v4 \
        --lens-pack lens_packs/apertus-8b_sumo-wordnet-v4 \
        --model swiss-ai/Apertus-8B-2509

Per MAP_MELD_PROTOCOL.md §3:
- must_retrain: Direct parents (negative sampling changes)
- should_retrain: Siblings, antonyms (discrimination may degrade)
- new_concepts: Newly added concepts

The script clears pending_training.json after successful training.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import training.sumo_classifiers as sumo_classifiers_module
from data.version_manifest import LensManifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pending lenses from pending_training.json"
    )

    parser.add_argument('--concept-pack', required=True,
                        help='Concept pack ID (e.g., sumo-wordnet-v4)')
    parser.add_argument('--lens-pack', required=True,
                        help='Lens pack directory')
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., swiss-ai/Apertus-8B-2509)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')
    parser.add_argument('--include-should-retrain', action='store_true',
                        help='Also train should_retrain concepts (siblings, antonyms)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be trained without training')

    return parser.parse_args()


def load_pending_training(pack_dir: Path) -> Dict:
    """Load pending_training.json."""
    pending_path = pack_dir / "pending_training.json"
    if not pending_path.exists():
        return None

    with open(pending_path) as f:
        return json.load(f)


def get_concept_layers(hierarchy_dir: Path) -> Dict[str, int]:
    """Build mapping of concept -> layer from hierarchy files."""
    concept_to_layer = {}

    for layer_num in range(5):
        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get("concepts", []):
            term = concept.get("sumo_term", "")
            if term:
                concept_to_layer[term] = layer_num

    return concept_to_layer


def collect_pending_by_layer(
    pending: Dict,
    concept_to_layer: Dict[str, int],
    include_should_retrain: bool = False
) -> Dict[int, List[str]]:
    """Organize pending concepts by layer."""
    by_layer: Dict[int, Set[str]] = {}

    pending_concepts = pending.get("pending_concepts", [])

    for concept in pending_concepts:
        layer = concept_to_layer.get(concept)
        if layer is not None:
            if layer not in by_layer:
                by_layer[layer] = set()
            by_layer[layer].add(concept)

    return {layer: sorted(concepts) for layer, concepts in by_layer.items()}


def main():
    args = parse_args()

    # Load concept pack metadata
    pack_dir = PROJECT_ROOT / "concept_packs" / args.concept_pack
    pack_json_path = pack_dir / "pack.json"

    if not pack_json_path.exists():
        print(f"Error: Concept pack not found at {pack_dir}")
        sys.exit(1)

    with open(pack_json_path) as f:
        pack_metadata = json.load(f)

    version = pack_metadata['version']
    hierarchy_dir = pack_dir / pack_metadata['concept_metadata']['hierarchy_file']

    # Load pending training
    pending = load_pending_training(pack_dir)
    if not pending:
        print("No pending_training.json found - nothing to train")
        sys.exit(0)

    pending_concepts = pending.get("pending_concepts", [])
    if not pending_concepts:
        print("No pending concepts to train")
        sys.exit(0)

    print("=" * 80)
    print("TRAINING PENDING LENSS")
    print("=" * 80)
    print()
    print(f"Concept Pack: {pack_metadata['pack_id']} v{version}")
    print(f"Pending concepts: {len(pending_concepts)}")
    print(f"Affected by melds: {pending.get('affected_by_melds', [])}")
    print()

    # Get concept layer mapping
    concept_to_layer = get_concept_layers(hierarchy_dir)

    # Organize by layer
    by_layer = collect_pending_by_layer(
        pending, concept_to_layer, args.include_should_retrain
    )

    print("Concepts by layer:")
    for layer, concepts in sorted(by_layer.items()):
        print(f"  Layer {layer}: {len(concepts)} concepts")

    if args.dry_run:
        print("\n[DRY RUN] Would train:")
        for layer, concepts in sorted(by_layer.items()):
            print(f"  Layer {layer}:")
            for c in concepts[:5]:
                print(f"    - {c}")
            if len(concepts) > 5:
                print(f"    ... and {len(concepts) - 5} more")
        sys.exit(0)

    # Setup output directory
    lens_pack_dir = Path(args.lens_pack)
    lens_pack_dir.mkdir(parents=True, exist_ok=True)

    # Patch layer data directory
    sumo_classifiers_module.LAYER_DATA_DIR = hierarchy_dir

    # Load existing manifest
    manifest = LensManifest.load(lens_pack_dir)
    manifest.source_pack = args.concept_pack
    manifest.model = args.model

    # Train each layer
    all_trained = []
    for layer, concepts in sorted(by_layer.items()):
        print()
        print(f"Training layer {layer}: {len(concepts)} concepts")

        layer_output_dir = lens_pack_dir / f"layer{layer}"
        layer_output_dir.mkdir(exist_ok=True)

        # Remove existing classifiers for concepts we're retraining
        for concept in concepts:
            classifier_path = layer_output_dir / f"{concept}_classifier.pt"
            if classifier_path.exists():
                classifier_path.unlink()

        # Run training for this layer
        sumo_classifiers_module.train_layer(
            layer=layer,
            model=None,  # Will be loaded by train_layer
            tokenizer=None,
            n_train_pos=50,
            n_train_neg=50,
            n_test_pos=20,
            n_test_neg=20,
            device=args.device,
            output_dir=layer_output_dir,
            use_adaptive_training=True,
            validation_mode='falloff',
        )

        # Update manifest with results
        results_file = layer_output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            for result in results.get("results", []):
                concept = result.get("concept", "")
                if concept in concepts:
                    manifest.update_lens(
                        concept=concept,
                        version=version,
                        layer=layer,
                        metrics={
                            "f1": result.get("test_f1", 0),
                            "precision": result.get("test_precision", 0),
                            "recall": result.get("test_recall", 0),
                        },
                        training_samples=result.get("n_train_samples", 0),
                    )
                    all_trained.append(concept)

    # Record training run
    manifest.record_training_run(version=version, trained_concepts=all_trained)
    manifest.save()

    # Clear pending training
    pending_path = pack_dir / "pending_training.json"
    cleared_pending = {
        "pending_concepts": [],
        "affected_by_melds": pending.get("affected_by_melds", []),
        "version": version,
        "last_training_run": datetime.now().isoformat() + "Z",
        "concepts_trained": len(all_trained),
        "created_at": pending.get("created_at"),
        "last_updated": datetime.now().isoformat() + "Z",
    }
    with open(pending_path, 'w') as f:
        json.dump(cleared_pending, f, indent=2)

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Trained: {len(all_trained)} lenses")
    print(f"Manifest updated: {lens_pack_dir / 'version_manifest.json'}")
    print(f"Pending cleared: {pending_path}")


if __name__ == "__main__":
    main()
