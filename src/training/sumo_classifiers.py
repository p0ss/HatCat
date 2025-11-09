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


def load_layer_concepts(layer: int, base_dir: Path = LAYER_DATA_DIR) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load layer concepts and provide both list and lookup map."""
    layer_path = base_dir / f"layer{layer}.json"
    with open(layer_path) as f:
        layer_data = json.load(f)

    concepts = layer_data["concepts"]
    concept_map = {c["sumo_term"]: c for c in concepts}

    print(f"\n✓ Loaded Layer {layer}: {len(concepts)} concepts")
    return concepts, concept_map


def extract_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cuda",
    layer_idx: int = -1,
) -> np.ndarray:
    """Extract mean pooled activations for each prompt."""
    activations: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
            pooled = hidden_states.mean(dim=1)  # [1, hidden_dim]
            activations.append(pooled.cpu().numpy()[0])

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
    model,
    tokenizer,
    n_train_pos: int = 10,
    n_train_neg: int = 10,
    n_test_pos: int = 20,
    n_test_neg: int = 20,
    device: str = "cuda",
    output_dir: Path | None = None,
    save_text_samples: bool = True,
    use_adaptive_training: bool = False,
    train_text_probes: bool = True,
) -> Dict:
    """
    Train classifiers for a single SUMO abstraction layer.

    Args:
        save_text_samples: If True, save generated text for text probe training
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
        train_text_probes: If True, train text probes alongside activation probes
    """
    print(f"\n{'=' * 80}")
    print(f"TRAINING LAYER {layer}")
    print(f"{'=' * 80}")

    concepts, concept_map = load_layer_concepts(layer)
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

    # Create text probe output directory if needed
    if train_text_probes:
        text_probe_output_dir = output_dir / "text_probes"
        text_probe_output_dir.mkdir(exist_ok=True)

    # Initialize adaptive trainer if requested
    if use_adaptive_training:
        adaptive_trainer = DualAdaptiveTrainer(
            target_accuracy=0.95,
            activation_baseline=n_train_pos,
            activation_increment=1,
            activation_max_samples=100,
            text_baseline=n_train_pos,
            text_increment=5,
            text_max_samples=200,
            train_activation=True,
            train_text=train_text_probes,
        )

    for i, concept in enumerate(concepts):
        concept_name = concept["sumo_term"]
        print(f"\n[{i + 1}/{len(concepts)}] Training: {concept_name}")
        print(
            f"  Synsets: {concept['synset_count']}, Children: {len(concept.get('category_children', []))}"
        )

        try:
            negative_pool = build_sumo_negative_pool(concepts, concept)
            if len(negative_pool) < n_train_neg:
                print(f"  ⚠️  Only {len(negative_pool)} negatives available (need {n_train_neg})")
                n_train_neg_actual = min(n_train_neg, len(negative_pool))
                n_test_neg_actual = min(n_test_neg, max(1, len(negative_pool) // 2))
            else:
                n_train_neg_actual = n_train_neg
                n_test_neg_actual = n_test_neg

            train_prompts, train_labels = create_sumo_training_dataset(
                concept=concept,
                all_concepts=concept_map,
                negative_pool=negative_pool,
                n_positives=n_train_pos,
                n_negatives=n_train_neg_actual,
                use_category_relationships=True,
                use_wordnet_relationships=True,
            )

            test_negative_pool = negative_pool[len(negative_pool) // 2 :]
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

            # Save text samples for text probe training
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
                # Extract activations for full pool
                print(f"  Extracting activations...")
                X_train = extract_activations(model, tokenizer, train_prompts, device)
                X_test = extract_activations(model, tokenizer, test_prompts, device)

                # Run dual adaptive training
                adaptive_results = adaptive_trainer.train_concept(
                    concept_name=concept_name,
                    train_activations=X_train,
                    train_labels=np.array(train_labels),
                    test_activations=X_test,
                    test_labels=np.array(test_labels),
                    train_texts=train_prompts if train_text_probes else None,
                    test_texts=test_prompts if train_text_probes else None,
                )

                # Save activation classifier
                if adaptive_results['activation_classifier'] is not None:
                    torch.save(
                        adaptive_results['activation_classifier'].state_dict(),
                        output_dir / f"{concept_name}_classifier.pt"
                    )

                # Save text probe
                if train_text_probes and adaptive_results['text_probe'] is not None:
                    import joblib
                    text_probe_path = text_probe_output_dir / f"{concept_name}_text_probe.joblib"
                    joblib.dump(adaptive_results['text_probe'], text_probe_path)

                # Extract metrics for results
                activation_metrics = adaptive_results.get('activation', {})
                text_metrics = adaptive_results.get('text', {})

                result = {
                    "concept": concept_name,
                    "layer": layer,
                    "synset_count": concept["synset_count"],
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

                classifier, metrics = train_simple_classifier(
                    X_train,
                    np.array(train_labels),
                    X_test,
                    np.array(test_labels),
                )
                print(f"  ✓ Train F1: {metrics['train_f1']:.3f}, Test F1: {metrics['test_f1']:.3f}")

                result = {
                    "concept": concept_name,
                    "layer": layer,
                    "synset_count": concept["synset_count"],
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
    model_name: str = "google/gemma-3-4b-pt",
    device: str = "cuda",
    n_train_pos: int = 10,
    n_train_neg: int = 10,
    n_test_pos: int = 20,
    n_test_neg: int = 20,
    output_dir: Path | str = Path("results/sumo_classifiers"),
    train_text_probes: bool = True,
    use_adaptive_training: bool = False,
) -> List[Dict]:
    """
    High-level entry point for training multiple layers.

    Args:
        train_text_probes: If True, also train text probes after activation probes (or inline with adaptive)
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
    """
    output_dir = Path(output_dir)

    print(f"\n{'=' * 80}")
    print("SUMO HIERARCHICAL CLASSIFIER TRAINING")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Layers: {list(layers)}")
    print(f"Training mode: {'Adaptive (independent graduation)' if use_adaptive_training else 'Fixed samples'}")
    if not use_adaptive_training:
        print(f"Training: {n_train_pos} pos + {n_train_neg} neg per concept")
    else:
        print(f"Adaptive baseline: {n_train_pos} samples (activation +1/iter, text +5/iter)")
    print(f"Testing: {n_test_pos} pos + {n_test_neg} neg per concept")
    print(f"Train text probes: {train_text_probes}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    print("✓ Model loaded")

    summaries = []
    for layer in layers:
        summary = train_layer(
            layer=layer,
            model=model,
            tokenizer=tokenizer,
            n_train_pos=n_train_pos,
            n_train_neg=n_train_neg,
            n_test_pos=n_test_pos,
            n_test_neg=n_test_neg,
            device=device,
            output_dir=output_dir / f"layer{layer}",
            save_text_samples=train_text_probes and not use_adaptive_training,  # Only save if not adaptive
            use_adaptive_training=use_adaptive_training,
            train_text_probes=train_text_probes,
        )
        summaries.append(summary)

    # Train text probes separately ONLY if NOT using adaptive training
    # (adaptive training trains them inline)
    if train_text_probes and not use_adaptive_training:
        print(f"\n{'=' * 80}")
        print("TRAINING TEXT PROBES")
        print(f"{'=' * 80}")

        from .text_probes import train_text_probes_for_layer

        for layer in layers:
            print(f"\nLayer {layer}...")
            text_samples_dir = output_dir / f"layer{layer}" / "text_samples"

            if not text_samples_dir.exists():
                print(f"  ⚠️  No text samples found, skipping")
                continue

            text_probe_output = output_dir / f"layer{layer}" / "text_probes"

            try:
                train_text_probes_for_layer(
                    layer=layer,
                    text_samples_dir=text_samples_dir,
                    output_dir=text_probe_output,
                )
            except Exception as e:
                print(f"  ✗ Failed to train text probes: {e}")

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
