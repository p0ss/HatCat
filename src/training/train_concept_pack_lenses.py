#!/usr/bin/env python3
"""
Train lenses from a concept pack.

Usage:
    python scripts/train_concept_pack_lenses.py \
        --concept-pack sumo-wordnet-v4 \
        --model swiss-ai/Apertus-8B-2509 \
        --output-dir lens_packs/apertus-8b_sumo-wordnet-v4

Generates version_manifest.json for diff-based distribution per MAP_MELD_PROTOCOL.md §8.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path - script is now at src/training/train_concept_pack_lenses.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import and patch the layer data directory before importing training functions
from training import sumo_classifiers as sumo_classifiers_module
from training.sumo_classifiers import get_hidden_dim
from data.version_manifest import LensManifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train lens pack from concept pack"
    )

    # Concept pack
    parser.add_argument('--concept-pack', required=True,
                        help='Concept pack ID (e.g., sumo-wordnet-v4)')

    # Model configuration
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., swiss-ai/Apertus-8B-2509)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')

    # Layer selection
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Which layers to train (default: all layers in pack)')
    parser.add_argument('--max-layer', type=int, default=None,
                        help='Only train layers up to this number (e.g., --max-layer 2 trains layers 0,1,2)')
    parser.add_argument('--min-layer', type=int, default=None,
                        help='Only train layers from this number (e.g., --min-layer 3 trains layers 3,4,...)')

    # Training configuration
    parser.add_argument('--n-train-pos', type=int, default=50,
                        help='Positive training samples per concept (default: 50)')
    parser.add_argument('--n-train-neg', type=int, default=50,
                        help='Negative training samples per concept (default: 50)')
    parser.add_argument('--n-test-pos', type=int, default=20,
                        help='Positive test samples per concept (default: 20)')
    parser.add_argument('--n-test-neg', type=int, default=20,
                        help='Negative test samples per concept (default: 20)')

    # Adaptive training
    parser.add_argument('--validation-mode', type=str, default='falloff',
                        choices=['loose', 'falloff', 'strict'],
                        help='Validation mode (default: falloff)')
    parser.add_argument('--validation-threshold', type=float, default=0.85,
                        help='Graduation F1 target (default: 0.85). Also used for calibration in strict mode.')
    parser.add_argument('--min-f1', type=float, default=None,
                        help='Alias for --validation-threshold. Sets graduation F1 target.')

    # Sibling handling
    parser.add_argument('--no-sibling-negatives', action='store_true',
                        help='Exclude siblings from binary training negatives')
    parser.add_argument('--no-sibling-refinement', action='store_true',
                        help='Skip sibling ranking refinement after binary training')
    parser.add_argument('--refine-only', action='store_true',
                        help='Skip binary training, only run sibling ranking refinement on existing lenses')
    parser.add_argument('--sibling-refine-epochs', type=int, default=20,
                        help='Epochs for sibling ranking refinement (default: 20)')
    parser.add_argument('--sibling-refine-prompts', type=int, default=15,
                        help='Prompts per sibling for refinement (default: 15)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for lens pack')

    # Incremental training
    parser.add_argument('--training-manifest', type=str, default=None,
                        help='Path to training_manifest.json (from migrate_lens_pack.py)')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining even if lens already exists')
    parser.add_argument('--all-layers', action='store_true',
                        help='Extract from all layers (experimental). Classifier learns which layers matter.')
    parser.add_argument('--multi-layer', action='store_true',
                        help='Auto-select best layer from each third (early/mid/late). More efficient than --all-layers.')
    parser.add_argument('--multi-layer-k', type=int, default=1,
                        help='Number of top layers per third for --multi-layer (default: 1). '
                             'k=1 → 3 layers, k=2 → 6 layers, k=3 → 9 layers. '
                             'Higher k = more compute but potentially better coverage.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --min-f1 as alias for --validation-threshold
    validation_mode = args.validation_mode
    validation_threshold = args.validation_threshold
    if args.min_f1 is not None:
        validation_threshold = args.min_f1
        print(f"Using --min-f1={args.min_f1}: setting graduation F1 target={args.min_f1}")

    # Load concept pack metadata
    pack_dir = PROJECT_ROOT / "concept_packs" / args.concept_pack
    pack_json_path = pack_dir / "pack.json"

    if not pack_json_path.exists():
        print(f"Error: Concept pack not found at {pack_dir}")
        print(f"Looking for: {pack_json_path}")
        sys.exit(1)

    with open(pack_json_path) as f:
        pack_metadata = json.load(f)

    print("=" * 80)
    print(f"TRAINING LENS PACK FROM CONCEPT PACK")
    print("=" * 80)
    print()
    print(f"Concept Pack: {pack_metadata['pack_id']} v{pack_metadata['version']}")
    print(f"Description: {pack_metadata['description']}")
    print(f"Total Concepts: {pack_metadata['concept_metadata']['total_concepts']}")
    print(f"Model: {args.model}")
    print()

    # Determine layers to train
    available_layers = pack_metadata['concept_metadata']['layers']
    if args.layers is None:
        layers_to_train = available_layers
    else:
        layers_to_train = args.layers
        # Validate requested layers exist
        invalid_layers = [l for l in layers_to_train if l not in available_layers]
        if invalid_layers:
            print(f"Error: Requested layers {invalid_layers} not found in pack")
            print(f"Available layers: {available_layers}")
            sys.exit(1)

    # Apply min/max layer filters
    if args.max_layer is not None:
        layers_to_train = [l for l in layers_to_train if l <= args.max_layer]
    if args.min_layer is not None:
        layers_to_train = [l for l in layers_to_train if l >= args.min_layer]

    if not layers_to_train:
        print(f"Error: No layers to train after applying filters")
        print(f"  Available: {available_layers}, max-layer: {args.max_layer}, min-layer: {args.min_layer}")
        sys.exit(1)

    print(f"Training Layers: {layers_to_train}")
    print()

    # Get concept pack hierarchy directory
    hierarchy_dir = pack_dir / pack_metadata['concept_metadata']['hierarchy_file']

    if not hierarchy_dir.exists():
        print(f"Error: Hierarchy directory not found at {hierarchy_dir}")
        sys.exit(1)

    print(f"Loading concepts from: {hierarchy_dir}")
    print()

    # Create output directory and logs subdirectory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging to file with timestamp
    import sys
    log_file = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Logging to: {log_file}")

    class TeeOutput:
        """Write to both stdout and log file."""
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeOutput(log_file)

    # Save pack metadata to output for reference
    pack_reference = {
        "source_pack": args.concept_pack,
        "pack_version": pack_metadata['version'],
        "model": args.model,
        "trained_layers": layers_to_train,
        "training_config": {
            "n_train_pos": args.n_train_pos,
            "n_train_neg": args.n_train_neg,
            "n_test_pos": args.n_test_pos,
            "n_test_neg": args.n_test_neg,
            "validation_mode": validation_mode,
            "validation_threshold": validation_threshold
        },
        "trained_at": datetime.now().isoformat() + "Z"
    }

    with open(output_dir / "pack_info.json", 'w') as f:
        json.dump(pack_reference, f, indent=2)

    # Train lenses using the concept pack hierarchy
    # NOTE: hierarchy_dir is passed directly to train_sumo_classifiers to avoid
    # the Python default argument bug where module-level patching doesn't work
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Using hierarchy: {hierarchy_dir}")
    print()

    # Sibling handling - both enabled by default
    include_sibling_negatives = not args.no_sibling_negatives
    run_sibling_refinement = not args.no_sibling_refinement
    refine_only = args.refine_only

    # Load training manifest if provided (for incremental training)
    concepts_to_train = None
    force_retrain_concepts = set()
    if args.training_manifest:
        manifest_path = Path(args.training_manifest)
        if manifest_path.exists():
            with open(manifest_path) as f:
                training_manifest = json.load(f)

            concepts_to_train = set()
            for entry in training_manifest.get('lenses_to_train', []):
                concept = entry.get('concept')
                if concept:
                    concepts_to_train.add(concept)
                    # Force retrain for must_retrain and should_retrain concepts
                    if entry.get('reason') in ('parent_of_split', 'sibling_of_split'):
                        force_retrain_concepts.add(concept)

            print(f"Training manifest loaded: {len(concepts_to_train)} concepts to train")
            print(f"  - Force retrain (parents/siblings): {len(force_retrain_concepts)}")
            print()
        else:
            print(f"Warning: Training manifest not found: {manifest_path}")

    # Delete lenses that need force retraining
    if args.force_retrain or force_retrain_concepts:
        concepts_to_delete = force_retrain_concepts if not args.force_retrain else (concepts_to_train or set())
        if concepts_to_delete:
            print(f"Removing {len(concepts_to_delete)} lenses for force retrain...")
            for layer in layers_to_train:
                layer_dir = output_dir / f"layer{layer}"
                if layer_dir.exists():
                    for concept in concepts_to_delete:
                        lens_path = layer_dir / f"{concept}_classifier.pt"
                        if lens_path.exists():
                            lens_path.unlink()
            print()

    if refine_only:
        print("Refine-only mode:")
        print("  Skipping binary training")
        print("  Running sibling ranking refinement on existing lenses")
        print()
    else:
        print("Training mode:")
        print(f"  Sibling hard negatives: {'YES' if include_sibling_negatives else 'NO'}")
        print(f"  Sibling refinement: {'YES' if run_sibling_refinement else 'NO'}")
        print()

    # Training (binary + optional sibling refinement per layer)
    if not refine_only:
        sumo_classifiers_module.train_sumo_classifiers(
            layers=layers_to_train,
            hierarchy_dir=hierarchy_dir,
            model_name=args.model,
            device=args.device,
            n_train_pos=args.n_train_pos,
            n_train_neg=args.n_train_neg,
            n_test_pos=args.n_test_pos,
            n_test_neg=args.n_test_neg,
            output_dir=str(output_dir),
            use_adaptive_training=True,
            validation_mode=validation_mode,
            validation_threshold=validation_threshold,
            include_sibling_negatives=include_sibling_negatives,
            run_sibling_refinement=run_sibling_refinement,
            sibling_refine_epochs=args.sibling_refine_epochs,
            sibling_refine_prompts=args.sibling_refine_prompts,
            only_concepts=concepts_to_train,
            all_layers=args.all_layers,
            multi_layer_mode=args.multi_layer,
            multi_layer_top_k=args.multi_layer_k,
        )
    else:
        # Refine-only mode: just run sibling refinement on existing lenses
        print("=" * 80)
        print("SIBLING RANKING REFINEMENT (refine-only mode)")
        print("=" * 80)
        print()

        from training.sibling_ranking import refine_all_sibling_groups
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Load model for refinement
        print("Loading model for sibling refinement...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            local_files_only=True,
        )
        model.eval()
        hidden_dim = get_hidden_dim(model)

        for layer in layers_to_train:
            lens_dir = output_dir / f"layer{layer}"
            refine_all_sibling_groups(
                layer=layer,
                lens_dir=lens_dir,
                hierarchy_dir=hierarchy_dir,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                n_prompts_per_sibling=args.sibling_refine_prompts,
                epochs=args.sibling_refine_epochs,
                hidden_dim=hidden_dim,
            )

        # Clean up model memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Generate version manifest from training results
    print()
    print("=" * 80)
    print("GENERATING VERSION MANIFEST")
    print("=" * 80)
    print()

    # Load existing manifest to preserve previously-trained lenses, or create new
    manifest = LensManifest.load(output_dir)
    if not manifest.lens_pack_id:
        # New manifest - initialize metadata
        manifest.lens_pack_id = f"{args.concept_pack}_{args.model.replace('/', '_')}"
        manifest.model = args.model
        manifest.source_pack = args.concept_pack
    manifest.current_version = pack_metadata['version']
    manifest._path = output_dir / "version_manifest.json"

    existing_lens_count = len(manifest.lenses)
    if existing_lens_count > 0:
        print(f"  Loaded existing manifest with {existing_lens_count} lenses")

    # Collect training results from layer results.json files
    trained_concepts = []
    for layer in layers_to_train:
        layer_results_path = output_dir / f"layer{layer}" / "results.json"
        if layer_results_path.exists():
            with open(layer_results_path) as f:
                layer_results = json.load(f)

            for result in layer_results.get("results", []):
                concept = result.get("concept", "")
                if concept:
                    manifest.update_lens(
                        concept=concept,
                        version=pack_metadata['version'],
                        layer=layer,
                        metrics={
                            "f1": result.get("test_f1", 0),
                            "precision": result.get("test_precision", 0),
                            "recall": result.get("test_recall", 0),
                        },
                        training_samples=result.get("n_train_samples", 0),
                        lens_file=f"layer{layer}/{concept}_classifier.pt"
                    )
                    trained_concepts.append(concept)

    # Record training run in version history
    manifest.record_training_run(
        version=pack_metadata['version'],
        trained_concepts=trained_concepts
    )

    manifest.save()
    new_count = len(trained_concepts)
    total_count = len(manifest.lenses)
    print(f"Version manifest saved: {total_count} total lenses ({new_count} new/updated this run)")
    print(f"  Version: {manifest.current_version}")
    print(f"  Layers: {sorted(set(p.layer for p in manifest.lenses.values()))}")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Lens pack saved to: {output_dir}")
    print(f"Version manifest: {output_dir / 'version_manifest.json'}")
    print()


if __name__ == "__main__":
    main()
