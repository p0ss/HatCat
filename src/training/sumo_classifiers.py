"""Utilities for training SUMO hierarchical concept classifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sumo_data_generation import (
    build_sumo_negative_pool,
    create_sumo_training_dataset,
)
from .dual_adaptive_trainer import DualAdaptiveTrainer

LAYER_DATA_DIR = Path("data/concept_graph/abstraction_layers")


def load_layer_concepts(layer: int, hierarchy_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load layer concepts and provide both list and lookup map.

    Args:
        layer: The layer number (0-6)
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
    """
    layer_path = hierarchy_dir / f"layer{layer}.json"
    with open(layer_path) as f:
        layer_data = json.load(f)

    concepts = layer_data["concepts"]
    concept_map = {c["sumo_term"]: c for c in concepts}

    print(f"\n✓ Loaded Layer {layer}: {len(concepts)} concepts from {hierarchy_dir}")
    return concepts, concept_map


def load_all_concepts(hierarchy_dir: Path) -> List[Dict]:
    """Load all concepts from all layers for negative pool construction.

    Args:
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
    """
    all_concepts = []
    for layer in range(7):  # Layers 0-6
        try:
            layer_path = hierarchy_dir / f"layer{layer}.json"
            with open(layer_path) as f:
                layer_data = json.load(f)
                all_concepts.extend(layer_data["concepts"])
        except FileNotFoundError:
            continue
    return all_concepts


def extract_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cuda",
    layer_idx: int = -1,
    max_new_tokens: int = 20,
    temperature_range: Tuple[float, float] = (0.3, 0.9),
    batch_size: int = 4,
    extraction_mode: str = "combined",
) -> np.ndarray:
    """
    Extract activations from model, using combined prompt+generation extraction by default.

    This uses the "combined-20" strategy (EXTRACTION_STRATEGY_DECISION.md):
    - Extracts activations from BOTH prompt processing and generation phases
    - Doubles training samples at zero additional computational cost
    - Prompt forward pass is already done for generation, so extracting is free
    - Results in 2x training data with better generalization

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate from
        device: Device for inference
        layer_idx: Model layer to extract activations from (-1 for last layer)
        max_new_tokens: Max tokens to generate per prompt
        temperature_range: (min_temp, max_temp) to vary across batches
        batch_size: Number of prompts to process in parallel
        extraction_mode: "combined" (prompt+gen, default), "generation" (gen only)

    Returns:
        Array of activation vectors
        - If extraction_mode="combined": [2*n_prompts, hidden_dim] (prompt + generation per input)
        - If extraction_mode="generation": [n_prompts, hidden_dim] (generation only)
    """
    activations: List[np.ndarray] = []
    model.eval()

    # Vary temperature across batches for diversity
    min_temp, max_temp = temperature_range
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    temperatures = np.linspace(min_temp, max_temp, n_batches)

    with torch.inference_mode():
        # Process prompts in batches
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            temp = temperatures[batch_idx // batch_size]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # === PHASE 1: PROMPT PROCESSING (if combined mode) ===
            if extraction_mode == "combined":
                prompt_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                prompt_hidden = prompt_outputs.hidden_states[layer_idx]  # [batch_size, seq_len, hidden_dim]

                # Pool each sequence in batch
                for seq_idx in range(len(batch_prompts)):
                    seq_hidden = prompt_hidden[seq_idx]  # [seq_len, hidden_dim]
                    prompt_pooled = seq_hidden.mean(dim=0)  # [hidden_dim]
                    activations.append(prompt_pooled.float().cpu().numpy())

            # === PHASE 2: GENERATION ===
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temp),
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Extract hidden states from generation
            # outputs.hidden_states is tuple of tuples: (step1, step2, ...)
            # Each step is tuple of layers: (layer0, layer1, ..., layerN)
            # We want the target layer, pooled across all generation steps

            # Process each sequence in the batch
            for seq_idx in range(len(batch_prompts)):
                # Collect target layer activations from all generation steps for this sequence
                step_activations = []
                for step_hidden in outputs.hidden_states:
                    target_layer = step_hidden[layer_idx]  # [batch_size, seq_len, hidden_dim]
                    # Extract this sequence
                    seq_hidden = target_layer[seq_idx]  # [seq_len, hidden_dim]
                    # Mean pool over sequence for this step
                    step_pooled = seq_hidden.mean(dim=0)  # [hidden_dim]
                    step_activations.append(step_pooled)

                # Stack and mean pool across all steps
                all_steps = torch.stack(step_activations, dim=0)  # [n_steps, hidden_dim]
                final_pooled = all_steps.mean(dim=0)  # [hidden_dim]

                activations.append(final_pooled.float().cpu().numpy())

    return np.array(activations)


def train_simple_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 0.001,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Train a simple MLP classifier and return metrics."""
    import torch.nn as nn
    import torch.optim as optim

    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1),
        nn.Sigmoid(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train_t) > 0.5).float().cpu().numpy().flatten()
        test_preds = (model(X_test_t) > 0.5).float().cpu().numpy().flatten()

    metrics = {
        "train_f1": float(f1_score(y_train, train_preds)),
        "test_f1": float(f1_score(y_test, test_preds)),
        "train_precision": float(precision_score(y_train, train_preds)),
        "test_precision": float(precision_score(y_test, test_preds)),
        "train_recall": float(recall_score(y_train, train_preds)),
        "test_recall": float(recall_score(y_test, test_preds)),
    }

    return model, metrics


def train_layer(
    layer: int,
    hierarchy_dir: Path,
    model,
    tokenizer,
    n_train_pos: int = 10,
    n_train_neg: int = 10,
    n_test_pos: int = 20,
    n_test_neg: int = 20,
    device: str = "cuda",
    output_dir: Path | None = None,
    save_text_samples: bool = False,
    use_adaptive_training: bool = False,
    train_text_probes: bool = False,
    validation_mode: str = 'falloff',
    validation_threshold: float = 0.5,
    include_sibling_negatives: bool = True,
) -> Dict:
    """
    Train classifiers for a single SUMO abstraction layer.

    Args:
        layer: The layer number (0-6)
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
        save_text_samples: If True, save generated text for text probe training (legacy)
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
        train_text_probes: If True, compute embedding centroids (legacy, not currently used)
        include_sibling_negatives: If True (default), include siblings as hard negatives.
                                   Set to False for two-pass training (sibling refinement done separately).
    """
    print(f"\n{'=' * 80}")
    print(f"TRAINING LAYER {layer}")
    print(f"{'=' * 80}")

    # Load target layer concepts for training
    concepts, concept_map = load_layer_concepts(layer, hierarchy_dir)

    # Load ALL concepts from all layers for negative pool (enables nephew negatives)
    all_concepts = load_all_concepts(hierarchy_dir)

    if output_dir is None:
        output_dir = Path(f"results/sumo_classifiers/layer{layer}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    failed_concepts: List[Dict] = []
    start_time = time.time()

    # Track all text samples for text probe training
    if save_text_samples:
        text_samples_dir = output_dir / "text_samples"
        text_samples_dir.mkdir(exist_ok=True)

    # Create centroid output directory if needed
    if train_text_probes:
        centroid_output_dir = output_dir / "embedding_centroids"
        centroid_output_dir.mkdir(exist_ok=True)

    # Initialize adaptive trainer if requested (activation probes only now)
    # Uses defaults from DualAdaptiveTrainer: 20 samples, 0.95 F1 target
    if use_adaptive_training:
        from .dual_adaptive_trainer import DualAdaptiveTrainer
        adaptive_trainer = DualAdaptiveTrainer(
            # Use class defaults for activation probe config (20 samples, 0.95 F1)
            model=model,  # Needed for validation
            tokenizer=tokenizer,  # Needed for validation
            max_response_tokens=100,
            validate_probes=True,  # Enable calibration validation
            validation_mode=validation_mode,  # Validation mode (loose/falloff/strict)
            validation_threshold=validation_threshold,  # Min score to pass (for strict mode)
            validation_layer_idx=15,  # Layer 15 for activations
            train_activation=True,
            train_text=False,  # Disable TF-IDF text probe training
        )

    for i, concept in enumerate(concepts):
        concept_name = concept["sumo_term"]

        # Check if already trained (resume capability)
        classifier_path = output_dir / f"{concept_name}_classifier.pt"
        centroid_path = centroid_output_dir / f"{concept_name}_centroid.npy" if train_text_probes else None

        if classifier_path.exists() and (not train_text_probes or centroid_path.exists()):
            print(f"\n[{i + 1}/{len(concepts)}] Skipping {concept_name} (already trained)")
            continue

        print(f"\n[{i + 1}/{len(concepts)}] Training: {concept_name}")
        synset_count = concept.get('synset_count', concept.get('concept_count', 0))
        print(
            f"  Synsets: {synset_count}, Children: {len(concept.get('category_children', []))}"
        )

        try:
            # Use all_concepts (not just this layer) to enable nephew negatives
            # For two-pass training, skip sibling hard negatives in pass 1
            negative_pool = build_sumo_negative_pool(
                all_concepts, concept, include_siblings=include_sibling_negatives
            )
            if len(negative_pool) < n_train_neg:
                print(f"  ⚠️  Only {len(negative_pool)} negatives available (need {n_train_neg})")
                n_train_neg_actual = min(n_train_neg, len(negative_pool))
                n_test_neg_actual = min(n_test_neg, max(1, len(negative_pool) // 2))
            else:
                n_train_neg_actual = n_train_neg
                n_test_neg_actual = n_test_neg

            # Split negative pool for train/test
            test_negative_pool = negative_pool[len(negative_pool) // 2 :]

            if use_adaptive_training:
                # JIT training - only generate test set upfront, train samples generated incrementally
                test_prompts, test_labels = create_sumo_training_dataset(
                    concept=concept,
                    all_concepts=concept_map,
                    negative_pool=test_negative_pool,
                    n_positives=n_test_pos,
                    n_negatives=n_test_neg_actual,
                    use_category_relationships=True,
                    use_wordnet_relationships=True,
                )
                print(f"  Generated {len(test_prompts)} test prompts (train samples generated JIT)")
            else:
                # Non-adaptive: generate full train set upfront
                train_prompts, train_labels = create_sumo_training_dataset(
                    concept=concept,
                    all_concepts=concept_map,
                    negative_pool=negative_pool,
                    n_positives=n_train_pos,
                    n_negatives=n_train_neg_actual,
                    use_category_relationships=True,
                    use_wordnet_relationships=True,
                )
                test_prompts, test_labels = create_sumo_training_dataset(
                    concept=concept,
                    all_concepts=concept_map,
                    negative_pool=test_negative_pool,
                    n_positives=n_test_pos,
                    n_negatives=n_test_neg_actual,
                    use_category_relationships=True,
                    use_wordnet_relationships=True,
                )
                print(f"  Generated {len(train_prompts)} train, {len(test_prompts)} test prompts")

                # Save text samples for text probe training (non-adaptive only)
                if save_text_samples:
                    text_sample_file = text_samples_dir / f"{concept_name}.json"
                    with open(text_sample_file, 'w') as f:
                        json.dump({
                            'concept': concept_name,
                            'layer': layer,
                            'train_prompts': train_prompts,
                            'train_labels': [int(label) for label in train_labels],
                            'test_prompts': test_prompts,
                            'test_labels': [int(label) for label in test_labels],
                        }, f)

            if use_adaptive_training:
                # === ADAPTIVE TRAINING MODE ===
                # Pass generation machinery to adaptive trainer for incremental sample generation
                generation_config = {
                    'concept': concept,
                    'all_concepts': concept_map,
                    'negative_pool': negative_pool,
                    'test_negative_pool': test_negative_pool,
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': device,
                }

                # Run dual adaptive training with incremental sample generation
                adaptive_results = adaptive_trainer.train_concept_incremental(
                    concept_name=concept_name,
                    generation_config=generation_config,
                    test_prompts=test_prompts,  # Generate test set once upfront
                    test_labels=np.array(test_labels),
                )

                # Save activation classifier
                if adaptive_results['activation_classifier'] is not None:
                    torch.save(
                        adaptive_results['activation_classifier'].state_dict(),
                        output_dir / f"{concept_name}_classifier.pt"
                    )

                # Note: Embedding centroids not supported in JIT adaptive mode
                # (would require returning prompts from adaptive trainer)
                if train_text_probes:
                    print(f"  ⚠️  Skipping embedding centroid (not supported in JIT adaptive mode)")

                # Extract metrics for results
                activation_metrics = adaptive_results.get('activation') or {}
                text_metrics = adaptive_results.get('text') or {}

                result = {
                    "concept": concept_name,
                    "layer": layer,
                    "synset_count": concept.get("synset_count", concept.get("concept_count", 0)),
                    "category_children_count": len(concept.get("category_children", [])),
                    "adaptive_training": True,
                    "total_iterations": adaptive_results['total_iterations'],
                    "total_time": adaptive_results['total_time'],
                    "activation_samples": activation_metrics.get('samples', 0),
                    "activation_iterations": activation_metrics.get('iterations', 0),
                    "test_f1": activation_metrics.get('test_f1', 0.0),
                    "test_precision": activation_metrics.get('test_precision', 0.0),
                    "test_recall": activation_metrics.get('test_recall', 0.0),
                }

                # Add validation results if available
                if 'validation' in activation_metrics:
                    val = activation_metrics['validation']
                    result.update({
                        "validation_passed": bool(val['passed']),
                        "validation_calibration_score": float(val['calibration_score']),
                        "validation_target_rank": int(val['target_rank']),
                        "validation_avg_other_rank": float(val['avg_other_rank']),
                        "validation_expected_domain": str(val['expected_domain']),
                    })

                if train_text_probes:
                    result.update({
                        "text_samples": text_metrics.get('samples', 0),
                        "text_iterations": text_metrics.get('iterations', 0),
                        "text_f1": text_metrics.get('test_f1', 0.0),
                        "text_precision": text_metrics.get('test_precision', 0.0),
                        "text_recall": text_metrics.get('test_recall', 0.0),
                    })

                print(f"  ✓ Adaptive training complete:")
                if activation_metrics:
                    print(f"    Activation: {activation_metrics.get('samples', 0)} samples, "
                          f"F1={activation_metrics.get('test_f1', 0):.3f}")
                if text_metrics:
                    print(f"    Text:       {text_metrics.get('samples', 0)} samples, "
                          f"F1={text_metrics.get('test_f1', 0):.3f}")

            else:
                # === FIXED TRAINING MODE (original) ===
                X_train = extract_activations(model, tokenizer, train_prompts, device)
                X_test = extract_activations(model, tokenizer, test_prompts, device)

                # If using combined extraction, we get 2x samples (prompt + generation per input)
                # so we need to duplicate labels to match
                train_labels_array = np.array(train_labels)
                test_labels_array = np.array(test_labels)

                if X_train.shape[0] == 2 * len(train_labels):
                    # Combined extraction detected - duplicate labels
                    train_labels_array = np.repeat(train_labels_array, 2)
                    test_labels_array = np.repeat(test_labels_array, 2)

                classifier, metrics = train_simple_classifier(
                    X_train,
                    train_labels_array,
                    X_test,
                    test_labels_array,
                )
                print(f"  ✓ Train F1: {metrics['train_f1']:.3f}, Test F1: {metrics['test_f1']:.3f}")

                result = {
                    "concept": concept_name,
                    "layer": layer,
                    "synset_count": concept.get("synset_count", concept.get("concept_count", 0)),
                    "category_children_count": len(concept.get("category_children", [])),
                    "n_train_samples": len(train_prompts),
                    "n_test_samples": len(test_prompts),
                    "adaptive_training": False,
                    **metrics,
                }
                torch.save(classifier.state_dict(), output_dir / f"{concept_name}_classifier.pt")

            all_results.append(result)

        except Exception as exc:  # pragma: no cover - logging path
            print(f"  ✗ ERROR: {exc}")
            failed_concepts.append({"concept": concept_name, "error": str(exc)})
            continue

    # Save centroid metadata
    if train_text_probes:
        centroid_metadata = {
            'layer': layer,
            'n_concepts': len(all_results),
            'embedding_dim': 3072,  # Gemma 3 4B hidden dimension
        }
        metadata_path = centroid_output_dir / "centroids_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(centroid_metadata, f, indent=2)
        print(f"\n✓ Saved centroid metadata to {metadata_path}")

    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"LAYER {layer} COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed_time / 60:.1f} minutes")
    print(f"Successfully trained: {len(all_results)}/{len(concepts)}")
    print(f"Failed: {len(failed_concepts)}")

    avg_test_f1 = np.mean([r["test_f1"] for r in all_results]) if all_results else 0.0
    avg_test_precision = (
        np.mean([r["test_precision"] for r in all_results]) if all_results else 0.0
    )
    avg_test_recall = np.mean([r["test_recall"] for r in all_results]) if all_results else 0.0

    if all_results:
        print("\nAverage Test Metrics:")
        print(f"  F1:        {avg_test_f1:.3f}")
        print(f"  Precision: {avg_test_precision:.3f}")
        print(f"  Recall:    {avg_test_recall:.3f}")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            convert_to_native({
                "layer": layer,
                "n_concepts": len(concepts),
                "n_successful": len(all_results),
                "n_failed": len(failed_concepts),
                "elapsed_minutes": elapsed_time / 60,
                "results": all_results,
                "failed": failed_concepts,
            }),
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to: {results_file}")

    return {
        "layer": layer,
        "n_concepts": len(concepts),
        "n_successful": len(all_results),
        "avg_test_f1": avg_test_f1,
    }


def train_sumo_classifiers(
    layers: Sequence[int],
    hierarchy_dir: Path,
    model_name: str = "google/gemma-3-4b-pt",
    device: str = "cuda",
    n_train_pos: int = 10,
    n_train_neg: int = 10,
    n_test_pos: int = 20,
    n_test_neg: int = 20,
    output_dir: Path | str = Path("results/sumo_classifiers"),
    train_text_probes: bool = False,
    use_adaptive_training: bool = False,
    validation_mode: str = 'falloff',
    validation_threshold: float = 0.5,
    include_sibling_negatives: bool = True,
    run_sibling_refinement: bool = False,
    sibling_refine_epochs: int = 20,
    sibling_refine_prompts: int = 15,
) -> List[Dict]:
    """
    High-level entry point for training multiple layers.

    Args:
        layers: Which concept layers to train (0-6)
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
        train_text_probes: If True, compute embedding centroids (legacy, not currently used)
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
        include_sibling_negatives: If True (default), include siblings as hard negatives.
                                   Set to False for two-pass training (sibling refinement done separately).
    """
    output_dir = Path(output_dir)

    print(f"\n{'=' * 80}")
    print("SUMO HIERARCHICAL CLASSIFIER TRAINING")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Layers: {list(layers)}")
    print(f"Training mode: {'Adaptive (independent graduation)' if use_adaptive_training else 'Fixed samples'}")
    if use_adaptive_training:
        print(f"Validation mode: {validation_mode.upper()}")
    if not use_adaptive_training:
        print(f"Training: {n_train_pos} pos + {n_train_neg} neg per concept")
    else:
        print(f"Adaptive baseline: {n_train_pos} samples (activation +1/iter, text +5/iter)")
    print(f"Testing: {n_test_pos} pos + {n_test_neg} neg per concept")
    print(f"Train text probes: {train_text_probes}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    print("✓ Model loaded")

    # Auto-detect and run missing sibling refinement from prior runs
    if run_sibling_refinement:
        from .sibling_ranking import get_layers_needing_refinement, refine_all_sibling_groups
        hidden_dim = model.config.hidden_size

        layers_needing_refinement = get_layers_needing_refinement(
            output_dir=output_dir,
            hierarchy_dir=hierarchy_dir,
            layers=list(layers),
        )

        if layers_needing_refinement:
            print(f"\n{'=' * 80}")
            print("AUTO-DETECTING MISSING SIBLING REFINEMENT")
            print(f"{'=' * 80}")
            print(f"Layers needing refinement: {layers_needing_refinement}")
            print()

            for layer in layers_needing_refinement:
                refine_all_sibling_groups(
                    layer=layer,
                    probe_dir=output_dir / f"layer{layer}",
                    hierarchy_dir=hierarchy_dir,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    n_prompts_per_sibling=sibling_refine_prompts,
                    epochs=sibling_refine_epochs,
                    hidden_dim=hidden_dim,
                )

            print(f"\n{'=' * 80}")
            print("MISSING REFINEMENT COMPLETE - Continuing with training")
            print(f"{'=' * 80}\n")

    summaries = []
    for layer in layers:
        # When using adaptive training, DualAdaptiveTrainer generates prompts incrementally
        # Only need test set generated upfront (fixed size throughout training)
        # For training set, adaptive trainer starts with 30 and adds more as needed
        train_samples = n_train_pos  # Use same size regardless of adaptive mode

        summary = train_layer(
            layer=layer,
            hierarchy_dir=hierarchy_dir,
            model=model,
            tokenizer=tokenizer,
            n_train_pos=train_samples,
            n_train_neg=train_samples,
            n_test_pos=n_test_pos,
            n_test_neg=n_test_neg,
            device=device,
            output_dir=output_dir / f"layer{layer}",
            save_text_samples=train_text_probes and not use_adaptive_training,  # Only save if not adaptive
            use_adaptive_training=use_adaptive_training,
            train_text_probes=train_text_probes,
            validation_mode=validation_mode,
            validation_threshold=validation_threshold,
            include_sibling_negatives=include_sibling_negatives,
        )
        summaries.append(summary)

        # Sibling ranking refinement after each layer (if enabled)
        if run_sibling_refinement:
            from .sibling_ranking import refine_all_sibling_groups
            hidden_dim = model.config.hidden_size
            refine_all_sibling_groups(
                layer=layer,
                probe_dir=output_dir / f"layer{layer}",
                hierarchy_dir=hierarchy_dir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_prompts_per_sibling=sibling_refine_prompts,
                epochs=sibling_refine_epochs,
                hidden_dim=hidden_dim,
            )

    # Compute centroids separately ONLY if NOT using adaptive training
    # (adaptive training computes them inline)
    if train_text_probes and not use_adaptive_training:
        print(f"\n{'=' * 80}")
        print("COMPUTING EMBEDDING CENTROIDS")
        print(f"{'=' * 80}")

        from .text_probes import compute_centroids_for_layer

        for layer in layers:
            print(f"\nLayer {layer}...")
            text_samples_dir = output_dir / f"layer{layer}" / "text_samples"

            if not text_samples_dir.exists():
                print(f"  ⚠️  No text samples found, skipping")
                continue

            centroid_output = output_dir / f"layer{layer}" / "embedding_centroids"

            try:
                compute_centroids_for_layer(
                    layer=layer,
                    text_samples_dir=text_samples_dir,
                    output_dir=centroid_output,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
            except Exception as e:
                print(f"  ✗ Failed to compute centroids: {e}")

    print(f"\n{'=' * 80}")
    print("ALL LAYERS COMPLETE")
    print(f"{'=' * 80}\n")

    for summary in summaries:
        print(
            f"Layer {summary['layer']}: {summary['n_successful']}/{summary['n_concepts']} "
            f"(Test F1: {summary['avg_test_f1']:.3f})"
        )

    total_concepts = sum(s["n_concepts"] for s in summaries)
    total_successful = sum(s["n_successful"] for s in summaries)
    print(f"\n✓ Total: {total_successful}/{total_concepts} concepts trained successfully")
    print(f"✓ Results saved to: {output_dir}/")

    return summaries


__all__ = [
    "train_sumo_classifiers",
    "train_layer",
    "train_simple_classifier",
    "extract_activations",
    "load_layer_concepts",
]
