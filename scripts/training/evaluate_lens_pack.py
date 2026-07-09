#!/usr/bin/env python3
"""
Held-out evaluation for an existing lens pack.

Mirrors the training script's evaluation flow exactly: same negative-pool
construction, same test-prompt generation, same per-concept layer selection,
same metric computation. The only thing skipped is the train step itself —
this loads the existing classifier, runs it through the same pipeline, and
records the result.

Output: per-layer `evaluation_results.json` files in the chosen output
directory, mirroring the structure of training `results.json` so they can
be consumed by `version_manifest.json` regeneration.

Usage:
    python scripts/training/evaluate_lens_pack.py \\
        --lens-pack src/lens_packs/gemma-4-e4b_first-light-v1-bf16 \\
        --concept-pack concept_packs/first-light \\
        --model google/gemma-4-E4B \\
        --output-dir results/evaluations/gemma-4-e4b_first-light-v1
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.map.training.sumo_classifiers import (
    load_all_concepts,
    extract_activations,
    select_layers_for_concept,
    get_num_layers,
)
from src.map.training.sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
)
from src.hat.monitoring.lens_types import create_lens_from_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Held-out evaluation of an existing lens pack — mirrors the "
                    "training script's eval flow exactly, without retraining."
    )
    parser.add_argument('--lens-pack', type=Path, required=True,
                        help='Path to lens pack with layer{0..6}/ directories of .pt files')
    parser.add_argument('--concept-pack', type=Path, default=PROJECT_ROOT / 'concept_packs' / 'first-light',
                        help='Concept pack root for hierarchy and prompt generation')
    parser.add_argument('--model', default='google/gemma-3-4b-pt',
                        help='Substrate model name (must match what the lenses were trained on)')
    parser.add_argument('--device', default='cuda',
                        help='Device for activation extraction (default: cuda)')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6],
                        help='Which ontological layers to evaluate (default: 0-6)')
    parser.add_argument('--n-test-pos', type=int, default=20,
                        help='Positive test samples per concept (matches training default)')
    parser.add_argument('--n-test-neg', type=int, default=20,
                        help='Negative test samples per concept (matches training default)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for per-layer evaluation_results.json files')
    parser.add_argument('--multi-layer-mode', action='store_true',
                        help='If set, mirror training\'s multi_layer_mode by calling '
                             'select_layers_for_concept per concept and using concatenated '
                             'activations. Default OFF, matching the train_sumo_classifiers '
                             'default — most lens packs were trained at single layer 15.')
    parser.add_argument('--single-layer', type=int, default=15,
                        help='Model layer for single-layer mode (default: 15, matches '
                             'training default initial_layer_idx in sumo_classifiers.py:700-701)')
    parser.add_argument('--multi-layer-top-k', type=int, default=1,
                        help='top_k for select_layers_for_concept when --multi-layer-mode is set')
    parser.add_argument('--max-concepts-per-layer', type=int, default=None,
                        help='Optional cap on concepts evaluated per layer (debug)')
    return parser.parse_args()


def infer_hidden_dim(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if 'weight' in key and hasattr(value, 'shape') and len(value.shape) == 2:
            return value.shape[1]
    raise ValueError(f"Could not infer hidden_dim from state_dict keys: {list(state_dict.keys())}")


def evaluate_concept(
    concept: dict,
    classifier,
    model,
    tokenizer,
    device: str,
    n_model_layers: int,
    concept_map: dict,
    all_concepts: list,
    n_test_pos: int,
    n_test_neg: int,
    multi_layer_mode: bool,
    multi_layer_top_k: int,
    single_layer: int,
) -> dict:
    """Mirror the training script's evaluation for a single concept.

    Steps match `train_sumo_classifiers` (sumo_classifiers.py:745+):
    1. Build negative pool with build_sumo_negative_pool
    2. Generate test prompts via create_sumo_training_dataset on test-half of pool
    3. Determine model layer(s):
       - multi_layer_mode=True: call select_layers_for_concept (concatenated)
       - multi_layer_mode=False: single layer (default 15, matches training default)
    4. Extract activations at that/those layer(s)
    5. Run the loaded classifier on those activations
    6. Compute metrics with the same sklearn calls
    """
    concept_name = concept['sumo_term']

    # --- Step 1: negative pool, mirroring training ---
    negative_pool = build_sumo_negative_pool(
        all_concepts, concept, include_siblings=True
    )
    if len(negative_pool) < n_test_neg:
        n_test_neg = max(1, len(negative_pool))
    test_negative_pool = negative_pool[len(negative_pool) // 2:] or negative_pool

    # --- Step 2: test prompts ---
    test_prompts, test_labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=concept_map,
        negative_pool=test_negative_pool,
        n_positives=n_test_pos,
        n_negatives=n_test_neg,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    if not test_prompts:
        return {
            'concept': concept_name,
            'error': 'no test prompts generated',
            'test_f1': None,
            'test_precision': None,
            'test_recall': None,
            'n_test': 0,
        }

    # --- Step 3: layer selection mirrors training's multi_layer_mode flag ---
    if multi_layer_mode:
        pos_prompts = [p for p, l in zip(test_prompts, test_labels) if l == 1]
        neg_prompts = [p for p, l in zip(test_prompts, test_labels) if l == 0]
        layer_idx, _ = select_layers_for_concept(
            model=model,
            tokenizer=tokenizer,
            pos_prompts=pos_prompts,
            neg_prompts=neg_prompts,
            device=device,
            n_model_layers=n_model_layers,
            top_k=multi_layer_top_k,
        )
    else:
        layer_idx = single_layer

    # --- Step 4: extract activations (default extraction_mode="combined" matches training) ---
    test_activations = extract_activations(
        model, tokenizer, test_prompts, device=device, layer_idx=layer_idx,
    )

    # --- Step 5: duplicate labels to match combined-mode activations
    # (mirrors sumo_classifiers.py:949-957 — combined mode produces 2× samples
    #  per prompt, so labels are np.repeat'd to match the activation count)
    test_labels_arr = np.asarray(test_labels)
    n_acts = test_activations.shape[0]
    n_labels_initial = len(test_labels_arr)
    if n_acts == 2 * n_labels_initial:
        test_labels_arr = np.repeat(test_labels_arr, 2)

    # --- Step 6: run classifier ---
    with torch.inference_mode():
        acts_tensor = torch.from_numpy(test_activations).to(device)
        lens_dtype = next(classifier.parameters()).dtype
        if acts_tensor.dtype != lens_dtype:
            acts_tensor = acts_tensor.to(dtype=lens_dtype)
        scores = classifier(acts_tensor).squeeze(-1).float().cpu().numpy()
    preds = (scores > 0.5).astype(int)

    # Defensive: if shapes still don't align, surface why before sklearn errors
    if len(test_labels_arr) != len(preds):
        return {
            'concept': concept_name,
            'error': (
                f'shape mismatch: n_prompts={len(test_prompts)}, '
                f'n_labels_initial={n_labels_initial}, '
                f'n_activations={n_acts}, '
                f'n_labels_post_repeat={len(test_labels_arr)}, '
                f'n_preds={len(preds)}, '
                f'layer_idx={layer_idx}'
            ),
            'test_f1': None,
            'test_precision': None,
            'test_recall': None,
            'n_test': n_labels_initial,
        }

    # --- Step 7: metrics (sklearn calls match training) ---
    f1 = float(f1_score(test_labels_arr, preds, zero_division=0))
    precision = float(precision_score(test_labels_arr, preds, zero_division=0))
    recall = float(recall_score(test_labels_arr, preds, zero_division=0))

    return {
        'concept': concept_name,
        'layer_idx': layer_idx if isinstance(layer_idx, list) else [layer_idx],
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'n_test': len(test_labels_arr),
        'n_test_pos': int(test_labels_arr.sum()),
        'n_test_neg': int((test_labels_arr == 0).sum()),
    }


def evaluate_layer(
    ontological_layer: int,
    layer_dir: Path,
    output_path: Path,
    model,
    tokenizer,
    device: str,
    n_model_layers: int,
    concept_map: dict,
    all_concepts: list,
    n_test_pos: int,
    n_test_neg: int,
    multi_layer_mode: bool,
    multi_layer_top_k: int,
    single_layer: int,
    max_concepts: int = None,
) -> dict:
    pt_files = sorted(layer_dir.glob('*.pt'))
    if max_concepts is not None:
        pt_files = pt_files[:max_concepts]

    print(f"\n{'=' * 80}")
    print(f"ONTOLOGICAL LAYER {ontological_layer} — evaluating {len(pt_files)} classifier(s)")
    print(f"{'=' * 80}")

    results = []
    failed = []
    start = time.time()

    for i, pt_path in enumerate(pt_files, 1):
        concept_name = pt_path.stem
        concept = concept_map.get(concept_name)
        if concept is None:
            failed.append({'concept': concept_name, 'error': 'concept not found in concept pack hierarchy'})
            print(f"[{i}/{len(pt_files)}] {concept_name}: SKIP (not in hierarchy)")
            continue

        try:
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
            hidden_dim = infer_hidden_dim(state_dict)
            classifier = create_lens_from_state_dict(state_dict, hidden_dim, device)
            classifier.eval()

            result = evaluate_concept(
                concept=concept,
                classifier=classifier,
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_model_layers=n_model_layers,
                concept_map=concept_map,
                all_concepts=all_concepts,
                n_test_pos=n_test_pos,
                n_test_neg=n_test_neg,
                multi_layer_mode=multi_layer_mode,
                multi_layer_top_k=multi_layer_top_k,
                single_layer=single_layer,
            )
            results.append(result)
            f1 = result.get('test_f1')
            f1_str = f"f1={f1:.3f}" if f1 is not None else "no metric"
            sel = result.get('layer_idx', '?')
            print(f"[{i}/{len(pt_files)}] {concept_name}: {f1_str} (layer={sel})")
        except Exception as exc:
            failed.append({'concept': concept_name, 'error': str(exc)})
            print(f"[{i}/{len(pt_files)}] {concept_name}: FAILED — {exc}")

    elapsed = time.time() - start
    n_with_metric = sum(1 for r in results if r.get('test_f1') is not None)
    avg_f1 = float(np.mean([r['test_f1'] for r in results if r.get('test_f1') is not None])) if n_with_metric else 0.0

    summary = {
        'ontological_layer': ontological_layer,
        'n_concepts': len(pt_files),
        'n_successful': n_with_metric,
        'n_failed': len(failed),
        'elapsed_minutes': elapsed / 60,
        'avg_test_f1': avg_f1,
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'results': results,
        'failed': failed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + '\n')

    print(f"\n  Ontological layer {ontological_layer} done in {elapsed / 60:.1f}min: "
          f"{n_with_metric}/{len(pt_files)} successful, avg f1={avg_f1:.3f}")
    print(f"  Results written to {output_path}")
    return summary


def main():
    args = parse_args()

    print("=" * 80)
    print("LENS PACK HELD-OUT EVALUATION")
    print("=" * 80)
    print(f"Lens pack:    {args.lens_pack}")
    print(f"Concept pack: {args.concept_pack}")
    print(f"Model:        {args.model}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Layers:       {args.layers}")
    print(f"Test samples: {args.n_test_pos} pos / {args.n_test_neg} neg per concept")

    # Load concepts
    hierarchy_dir = args.concept_pack / 'hierarchy'
    print(f"\nLoading concepts from {hierarchy_dir}...")
    all_concepts = load_all_concepts(hierarchy_dir)
    concept_map = {c['sumo_term']: c for c in all_concepts}
    print(f"  Loaded {len(concept_map)} concepts")

    # Load model
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()
    n_model_layers = get_num_layers(model)
    print(f"  ✓ Model loaded on {args.device} ({n_model_layers} model layers)")

    # Per-layer evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_summary = {
        'lens_pack': str(args.lens_pack),
        'concept_pack': str(args.concept_pack),
        'model': args.model,
        'n_model_layers': n_model_layers,
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'config': {
            'n_test_pos': args.n_test_pos,
            'n_test_neg': args.n_test_neg,
            'multi_layer_mode': args.multi_layer_mode,
            'multi_layer_top_k': args.multi_layer_top_k,
            'single_layer': args.single_layer,
            'layers': args.layers,
        },
        'layers': {},
    }

    total_start = time.time()
    for ontological_layer in args.layers:
        layer_dir = args.lens_pack / f'layer{ontological_layer}'
        if not layer_dir.exists():
            print(f"\nLayer {ontological_layer}: directory not found, skipping")
            continue
        output_path = args.output_dir / f'layer{ontological_layer}' / 'evaluation_results.json'
        layer_summary = evaluate_layer(
            ontological_layer=ontological_layer,
            layer_dir=layer_dir,
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            n_model_layers=n_model_layers,
            concept_map=concept_map,
            all_concepts=all_concepts,
            n_test_pos=args.n_test_pos,
            n_test_neg=args.n_test_neg,
            multi_layer_mode=args.multi_layer_mode,
            multi_layer_top_k=args.multi_layer_top_k,
            single_layer=args.single_layer,
            max_concepts=args.max_concepts_per_layer,
        )
        overall_summary['layers'][f'layer{ontological_layer}'] = {
            'n_concepts': layer_summary['n_concepts'],
            'n_successful': layer_summary['n_successful'],
            'n_failed': layer_summary['n_failed'],
            'avg_test_f1': layer_summary['avg_test_f1'],
            'elapsed_minutes': layer_summary['elapsed_minutes'],
        }

    overall_summary['total_elapsed_minutes'] = (time.time() - total_start) / 60
    (args.output_dir / 'evaluation_summary.json').write_text(
        json.dumps(overall_summary, indent=2) + '\n'
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    for layer_name, stats in overall_summary['layers'].items():
        print(f"  {layer_name}: {stats['n_successful']}/{stats['n_concepts']} "
              f"avg_f1={stats['avg_test_f1']:.3f} ({stats['elapsed_minutes']:.1f}min)")
    print(f"\nTotal: {overall_summary['total_elapsed_minutes']:.1f}min")
    print(f"Summary: {args.output_dir / 'evaluation_summary.json'}")


if __name__ == '__main__':
    main()
