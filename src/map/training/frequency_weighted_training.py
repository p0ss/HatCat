"""
Frequency-Weighted Activation Training for Concept Lenses.

This module implements TF-IDF-style dimension weighting to improve lens discrimination.
The key insight is that dimensions that are highly active across ALL concepts are not
discriminating, while dimensions that are uniquely active for specific concepts are.

Algorithm:
1. Extract activations for ALL concepts in the training run (all layers)
2. Compute per-dimension frequency across the entire activation corpus
3. Calculate IDF-style weights: low frequency = high weight, high frequency = low weight
4. Apply weights to activations before training each lens

This addresses the "sibling confusion" problem where lenses can't distinguish between
related concepts because they all learn the same shared activation patterns.

GPU-OPTIMIZED: All operations stay on GPU using torch tensors until final save.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# Handle both module and script imports
try:
    from .sumo_data_generation import (
        build_sumo_negative_pool,
        create_sumo_training_dataset,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.sumo_data_generation import (
        build_sumo_negative_pool,
        create_sumo_training_dataset,
    )


@dataclass
class ConceptActivationDataGPU:
    """Stores activation data for a single concept - kept on CPU to avoid OOM."""
    concept_name: str
    layer: int
    positive_activations: torch.Tensor  # [n_pos_samples, hidden_dim] on CPU
    negative_activations: torch.Tensor  # [n_neg_samples, hidden_dim] on CPU
    positive_labels: torch.Tensor
    negative_labels: torch.Tensor
    test_positive_activations: torch.Tensor
    test_negative_activations: torch.Tensor
    test_positive_labels: torch.Tensor
    test_negative_labels: torch.Tensor
    metadata: Dict = field(default_factory=dict)

    def to_gpu(self, device: str = "cuda") -> "ConceptActivationDataGPU":
        """Move this concept's data to GPU for training."""
        return ConceptActivationDataGPU(
            concept_name=self.concept_name,
            layer=self.layer,
            positive_activations=self.positive_activations.to(device),
            negative_activations=self.negative_activations.to(device),
            positive_labels=self.positive_labels.to(device),
            negative_labels=self.negative_labels.to(device),
            test_positive_activations=self.test_positive_activations.to(device),
            test_negative_activations=self.test_negative_activations.to(device),
            test_positive_labels=self.test_positive_labels.to(device),
            test_negative_labels=self.test_negative_labels.to(device),
            metadata=self.metadata,
        )


@dataclass
class DimensionWeightsGPU:
    """Stores computed dimension weights (on GPU after model is unloaded)."""
    weights: torch.Tensor  # [hidden_dim] on GPU
    frequency_counts: torch.Tensor  # How many concepts activated each dimension
    total_concepts: int
    activation_threshold: float

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply weights to activations (both on same device)."""
        # Ensure weights are on same device as activations
        if self.weights.device != activations.device:
            self.weights = self.weights.to(activations.device)
        return activations * self.weights


def extract_activations_gpu(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cuda",
    layer_idx: int = -1,
    max_new_tokens: int = 20,
    temperature_range: Tuple[float, float] = (0.3, 0.9),
    batch_size: int = 4,
    extraction_mode: str = "combined",
) -> torch.Tensor:
    """
    Extract activations from model, keeping everything on GPU.

    Returns:
        torch.Tensor on GPU: [n_samples, hidden_dim]
    """
    activations: List[torch.Tensor] = []
    model.eval()

    min_temp, max_temp = temperature_range
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    temperatures = torch.linspace(min_temp, max_temp, n_batches)

    with torch.inference_mode():
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            temp = float(temperatures[batch_idx // batch_size])

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Phase 1: Prompt processing (if combined mode)
            if extraction_mode == "combined":
                prompt_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                prompt_hidden = prompt_outputs.hidden_states[layer_idx]  # [batch, seq, hidden]

                for seq_idx in range(len(batch_prompts)):
                    seq_hidden = prompt_hidden[seq_idx]  # [seq, hidden]
                    prompt_pooled = seq_hidden.mean(dim=0).float()  # [hidden]
                    activations.append(prompt_pooled)

            # Phase 2: Generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            for seq_idx in range(len(batch_prompts)):
                step_activations = []
                for step_hidden in outputs.hidden_states:
                    target_layer = step_hidden[layer_idx]
                    seq_hidden = target_layer[seq_idx]
                    step_pooled = seq_hidden.mean(dim=0)
                    step_activations.append(step_pooled)

                all_steps = torch.stack(step_activations, dim=0)
                final_pooled = all_steps.mean(dim=0).float()
                activations.append(final_pooled)

    # Stack all activations and move to CPU to avoid OOM
    return torch.stack(activations, dim=0).cpu()


def compute_dimension_weights_gpu(
    all_concept_activations: Dict[str, ConceptActivationDataGPU],
    activation_threshold: float = 0.1,
    smoothing: float = 1.0,
    weight_method: str = "idf",
    device: str = "cuda",
) -> DimensionWeightsGPU:
    """
    Compute per-dimension weights based on frequency across all concepts.
    Operates on GPU (model unloaded first to free VRAM).
    """
    if not all_concept_activations:
        raise ValueError("No activation data provided")

    # Get hidden dimension from first concept
    first_concept = next(iter(all_concept_activations.values()))
    hidden_dim = first_concept.positive_activations.shape[1]
    n_concepts = len(all_concept_activations)

    print(f"\nComputing dimension weights across {n_concepts} concepts...")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Method: {weight_method}")
    print(f"  Activation threshold: {activation_threshold}")
    print(f"  Device: {device}")

    if weight_method == "idf":
        # IDF-style: Count how many concepts strongly activate each dimension
        dimension_concept_count = torch.zeros(hidden_dim, device=device)

        for concept_name, concept_data in all_concept_activations.items():
            # Use positive activations to determine concept's activation pattern
            pos_acts = concept_data.positive_activations  # [n_samples, hidden_dim] on GPU

            # Mean activation across samples for this concept
            concept_mean = torch.abs(pos_acts).mean(dim=0)  # [hidden_dim]

            # Determine which dimensions are "active" for this concept
            threshold = activation_threshold * concept_mean.max()
            active_dims = (concept_mean > threshold).float()

            dimension_concept_count += active_dims

        # IDF formula: log(N / (count + smoothing))
        idf_weights = torch.log(n_concepts / (dimension_concept_count + smoothing))

        # Normalize to have mean=1
        idf_weights = idf_weights / idf_weights.mean()

        weights = idf_weights
        frequency_counts = dimension_concept_count

    elif weight_method == "variance":
        # Variance-based: Dimensions with high variance are more discriminating
        all_means = []

        for concept_data in all_concept_activations.values():
            concept_mean = concept_data.positive_activations.mean(dim=0)
            all_means.append(concept_mean)

        all_means = torch.stack(all_means, dim=0)  # [n_concepts, hidden_dim]

        # Variance across concepts for each dimension
        dim_variance = all_means.var(dim=0) + smoothing

        # Higher variance = more discriminating = higher weight
        weights = dim_variance / dim_variance.mean()
        frequency_counts = torch.zeros(hidden_dim, device=device)

    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    # Report statistics
    print(f"\nWeight statistics:")
    print(f"  Min weight: {weights.min().item():.4f}")
    print(f"  Max weight: {weights.max().item():.4f}")
    print(f"  Mean weight: {weights.mean().item():.4f}")
    print(f"  Std weight: {weights.std().item():.4f}")

    if weight_method == "idf":
        low_freq = (frequency_counts < 0.1 * n_concepts).sum().item()
        high_freq = (frequency_counts > 0.9 * n_concepts).sum().item()
        print(f"  Dimensions active in <10% concepts (high weight): {low_freq}")
        print(f"  Dimensions active in >90% concepts (low weight): {high_freq}")

    return DimensionWeightsGPU(
        weights=weights,  # On GPU
        frequency_counts=frequency_counts,
        total_concepts=n_concepts,
        activation_threshold=activation_threshold,
    )


def train_weighted_classifier_gpu(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dim_weights: Optional[DimensionWeightsGPU] = None,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train an MLP classifier with optional dimension weighting.
    All operations on GPU.
    """
    # Apply weights if provided
    if dim_weights is not None:
        X_train = dim_weights.apply(X_train)
        X_test = dim_weights.apply(X_test)

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
    ).to(device)

    y_train_t = y_train.unsqueeze(1).float()
    y_test_t = y_test.unsqueeze(1).float()

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train) > 0.5).float().squeeze()
        test_preds = (model(X_test) > 0.5).float().squeeze()

    # Compute metrics on GPU then convert to Python floats
    def compute_f1_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute F1 score on GPU."""
        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        fn = ((y_pred == 0) & (y_true == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item(), precision.item(), recall.item()

    train_f1, train_prec, train_rec = compute_f1_gpu(y_train.squeeze(), train_preds)
    test_f1, test_prec, test_rec = compute_f1_gpu(y_test.squeeze(), test_preds)

    metrics = {
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_precision": train_prec,
        "test_precision": test_prec,
        "train_recall": train_rec,
        "test_recall": test_rec,
    }

    return model, metrics


def load_layer_concepts(layer: int, base_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load layer concepts from hierarchy directory."""
    layer_path = base_dir / f"layer{layer}.json"
    with open(layer_path) as f:
        layer_data = json.load(f)

    concepts = layer_data["concepts"]
    concept_map = {c["sumo_term"]: c for c in concepts}

    return concepts, concept_map


def load_all_concepts(base_dir: Path) -> List[Dict]:
    """Load all concepts from all layers for negative pool construction."""
    all_concepts = []
    for layer in range(7):
        try:
            layer_path = base_dir / f"layer{layer}.json"
            with open(layer_path) as f:
                layer_data = json.load(f)
                all_concepts.extend(layer_data["concepts"])
        except FileNotFoundError:
            continue
    return all_concepts


def train_with_frequency_weighting(
    layers: Sequence[int],
    hierarchy_dir: Path,
    model_name: str,
    output_dir: Path,
    n_train_pos: int = 50,
    n_train_neg: int = 50,
    n_test_pos: int = 20,
    n_test_neg: int = 20,
    device: str = "cuda",
    weight_method: str = "idf",
    activation_threshold: float = 0.1,
    layer_idx: int = 15,
) -> Dict:
    """
    Train lenses using frequency-weighted activations.
    GPU-optimized: all activations and computations stay on GPU.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FREQUENCY-WEIGHTED ACTIVATION TRAINING (GPU-OPTIMIZED)")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Layers: {list(layers)}")
    print(f"Weight method: {weight_method}")
    print(f"Activation threshold: {activation_threshold}")
    print(f"Model layer for extraction: {layer_idx}")
    print(f"Training samples: {n_train_pos} pos + {n_train_neg} neg")
    print(f"Test samples: {n_test_pos} pos + {n_test_neg} neg")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    print("Model loaded")

    # =========================================================================
    # PHASE 1: Collect all concepts and generate prompts
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: GENERATING TRAINING DATA FOR ALL CONCEPTS")
    print("=" * 80)

    all_concepts_by_layer: Dict[int, List[Dict]] = {}
    all_concept_prompts: Dict[str, Dict] = {}

    # Load ALL concepts for negative pool
    all_concepts = load_all_concepts(hierarchy_dir)
    print(f"\nLoaded {len(all_concepts)} total concepts for negative pool")

    total_concepts = 0
    for layer in layers:
        concepts, concept_map = load_layer_concepts(layer, hierarchy_dir)
        all_concepts_by_layer[layer] = concepts
        total_concepts += len(concepts)

        print(f"\nLayer {layer}: {len(concepts)} concepts")

        for concept in concepts:
            concept_name = concept["sumo_term"]

            # Build negative pool
            negative_pool = build_sumo_negative_pool(all_concepts, concept)
            n_train_neg_actual = min(n_train_neg, len(negative_pool))
            n_test_neg_actual = min(n_test_neg, max(1, len(negative_pool) // 2))

            # Generate training prompts
            train_prompts, train_labels = create_sumo_training_dataset(
                concept=concept,
                all_concepts=concept_map,
                negative_pool=negative_pool,
                n_positives=n_train_pos,
                n_negatives=n_train_neg_actual,
                use_category_relationships=True,
                use_wordnet_relationships=True,
            )

            # Generate test prompts
            test_negative_pool = negative_pool[len(negative_pool) // 2:]
            test_prompts, test_labels = create_sumo_training_dataset(
                concept=concept,
                all_concepts=concept_map,
                negative_pool=test_negative_pool,
                n_positives=n_test_pos,
                n_negatives=n_test_neg_actual,
                use_category_relationships=True,
                use_wordnet_relationships=True,
            )

            all_concept_prompts[concept_name] = {
                "layer": layer,
                "concept": concept,
                "train_prompts": train_prompts,
                "train_labels": train_labels,
                "test_prompts": test_prompts,
                "test_labels": test_labels,
            }

    print(f"\nGenerated prompts for {len(all_concept_prompts)} concepts across {len(layers)} layers")

    # =========================================================================
    # PHASE 2: Extract activations for ALL concepts (stay on GPU)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING ACTIVATIONS FOR ALL CONCEPTS (GPU)")
    print("=" * 80)

    all_activations: Dict[str, ConceptActivationDataGPU] = {}

    start_time = time.time()
    for i, (concept_name, prompt_data) in enumerate(all_concept_prompts.items()):
        print(f"\n[{i+1}/{len(all_concept_prompts)}] Extracting: {concept_name}")

        train_prompts = prompt_data["train_prompts"]
        train_labels = prompt_data["train_labels"]
        test_prompts = prompt_data["test_prompts"]
        test_labels = prompt_data["test_labels"]

        # Extract training activations (returns CPU tensor to avoid OOM)
        train_activations = extract_activations_gpu(
            model, tokenizer, train_prompts, device, layer_idx=layer_idx
        )

        # Handle combined extraction doubling (labels on CPU to match activations)
        if train_activations.shape[0] == 2 * len(train_labels):
            train_labels_tensor = torch.tensor(train_labels).repeat_interleave(2)
        else:
            train_labels_tensor = torch.tensor(train_labels)

        # Split into positive and negative (all on CPU)
        pos_mask = train_labels_tensor == 1
        neg_mask = train_labels_tensor == 0

        # Extract test activations (returns CPU tensor)
        test_activations = extract_activations_gpu(
            model, tokenizer, test_prompts, device, layer_idx=layer_idx
        )

        if test_activations.shape[0] == 2 * len(test_labels):
            test_labels_tensor = torch.tensor(test_labels).repeat_interleave(2)
        else:
            test_labels_tensor = torch.tensor(test_labels)

        test_pos_mask = test_labels_tensor == 1
        test_neg_mask = test_labels_tensor == 0

        all_activations[concept_name] = ConceptActivationDataGPU(
            concept_name=concept_name,
            layer=prompt_data["layer"],
            positive_activations=train_activations[pos_mask],
            negative_activations=train_activations[neg_mask],
            positive_labels=train_labels_tensor[pos_mask],
            negative_labels=train_labels_tensor[neg_mask],
            test_positive_activations=test_activations[test_pos_mask],
            test_negative_activations=test_activations[test_neg_mask],
            test_positive_labels=test_labels_tensor[test_pos_mask],
            test_negative_labels=test_labels_tensor[test_neg_mask],
            metadata={"synset_count": prompt_data["concept"].get("synset_count", 0)},
        )

        n_pos = train_activations[pos_mask].shape[0]
        n_neg = train_activations[neg_mask].shape[0]
        print(f"  Pos: {n_pos}, Neg: {n_neg}")

    extraction_time = time.time() - start_time
    print(f"\nExtraction complete in {extraction_time/60:.1f} minutes")

    # =========================================================================
    # PHASE 2.5: Unload model to free GPU memory for weight computation
    # =========================================================================
    print("\n" + "=" * 80)
    print("UNLOADING MODEL TO FREE GPU MEMORY")
    print("=" * 80)

    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded, GPU memory freed")

    # Move all activations to GPU now that we have space
    print("\nMoving activations to GPU...")
    for concept_name in all_activations:
        all_activations[concept_name] = all_activations[concept_name].to_gpu(device)
    print(f"Moved {len(all_activations)} concept activation sets to GPU")

    # =========================================================================
    # PHASE 3: Compute dimension weights (on GPU)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING DIMENSION WEIGHTS (GPU)")
    print("=" * 80)

    dim_weights = compute_dimension_weights_gpu(
        all_activations,
        activation_threshold=activation_threshold,
        weight_method=weight_method,
        device=device,
    )

    # Save weights (move to CPU for numpy)
    weights_path = output_dir / "dimension_weights.npz"
    np.savez(
        weights_path,
        weights=dim_weights.weights.cpu().numpy(),
        frequency_counts=dim_weights.frequency_counts.cpu().numpy(),
        total_concepts=dim_weights.total_concepts,
        activation_threshold=dim_weights.activation_threshold,
    )
    print(f"\nSaved dimension weights to {weights_path}")

    # =========================================================================
    # PHASE 4: Train lenses with weighted activations (on GPU)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: TRAINING LENSS WITH WEIGHTED ACTIVATIONS (GPU)")
    print("=" * 80)

    all_results = []
    failed_concepts = []

    for layer in layers:
        layer_output = output_dir / f"layer{layer}"
        layer_output.mkdir(parents=True, exist_ok=True)

        layer_concepts = all_concepts_by_layer[layer]
        layer_results = []

        print(f"\n{'=' * 40}")
        print(f"Training Layer {layer}: {len(layer_concepts)} concepts")
        print(f"{'=' * 40}")

        for i, concept in enumerate(layer_concepts):
            concept_name = concept["sumo_term"]

            if concept_name not in all_activations:
                print(f"\n[{i+1}/{len(layer_concepts)}] SKIP {concept_name} (no activations)")
                continue

            print(f"\n[{i+1}/{len(layer_concepts)}] Training: {concept_name}")

            act_data = all_activations[concept_name]

            try:
                # Combine positive and negative (all on GPU)
                X_train = torch.cat([act_data.positive_activations, act_data.negative_activations], dim=0)
                y_train = torch.cat([act_data.positive_labels, act_data.negative_labels], dim=0)

                X_test = torch.cat([act_data.test_positive_activations, act_data.test_negative_activations], dim=0)
                y_test = torch.cat([act_data.test_positive_labels, act_data.test_negative_labels], dim=0)

                # Train with weighted activations (all on GPU)
                classifier, metrics = train_weighted_classifier_gpu(
                    X_train, y_train, X_test, y_test,
                    dim_weights=dim_weights,
                    device=device,
                )

                print(f"  Train F1: {metrics['train_f1']:.3f}, Test F1: {metrics['test_f1']:.3f}")
                print(f"  Precision: {metrics['test_precision']:.3f}, Recall: {metrics['test_recall']:.3f}")

                # Save classifier (moves to CPU)
                torch.save(classifier.state_dict(), layer_output / f"{concept_name}.pt")

                result = {
                    "concept": concept_name,
                    "layer": layer,
                    "synset_count": concept.get("synset_count", 0),
                    "category_children_count": len(concept.get("category_children", [])),
                    "n_train_samples": len(y_train),
                    "n_test_samples": len(y_test),
                    "frequency_weighted": True,
                    "weight_method": weight_method,
                    **metrics,
                }
                layer_results.append(result)
                all_results.append(result)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed_concepts.append({"concept": concept_name, "layer": layer, "error": str(e)})

        # Save layer results
        layer_results_file = layer_output / "results.json"
        with open(layer_results_file, 'w') as f:
            json.dump({
                "layer": layer,
                "n_concepts": len(layer_concepts),
                "n_successful": len(layer_results),
                "n_failed": len([f for f in failed_concepts if f.get("layer") == layer]),
                "frequency_weighted": True,
                "weight_method": weight_method,
                "results": layer_results,
            }, f, indent=2)

        print(f"\nLayer {layer} complete: {len(layer_results)}/{len(layer_concepts)} trained")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Concepts trained: {len(all_results)}/{total_concepts}")
    print(f"Failed: {len(failed_concepts)}")

    if all_results:
        avg_f1 = sum(r["test_f1"] for r in all_results) / len(all_results)
        avg_precision = sum(r["test_precision"] for r in all_results) / len(all_results)
        avg_recall = sum(r["test_recall"] for r in all_results) / len(all_results)
        print(f"\nAverage metrics:")
        print(f"  Test F1: {avg_f1:.3f}")
        print(f"  Test Precision: {avg_precision:.3f}")
        print(f"  Test Recall: {avg_recall:.3f}")

    # Save overall summary
    summary = {
        "layers": list(layers),
        "total_concepts": total_concepts,
        "successful": len(all_results),
        "failed": len(failed_concepts),
        "elapsed_minutes": total_time / 60,
        "frequency_weighted": True,
        "weight_method": weight_method,
        "activation_threshold": activation_threshold,
        "model": model_name,
        "model_layer": layer_idx,
        "failed_concepts": failed_concepts,
    }

    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Frequency-weighted lens training (GPU-optimized)")
    parser.add_argument("--concept-pack", required=True, help="Concept pack ID")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--weight-method", choices=["idf", "variance"], default="idf")
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--model-layer", type=int, default=15)
    parser.add_argument("--n-train-pos", type=int, default=50)
    parser.add_argument("--n-train-neg", type=int, default=50)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Locate concept pack
    project_root = Path(__file__).parent.parent.parent
    pack_dir = project_root / "concept_packs" / args.concept_pack
    hierarchy_dir = pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        print(f"Error: Hierarchy not found at {hierarchy_dir}")
        exit(1)

    train_with_frequency_weighting(
        layers=args.layers,
        hierarchy_dir=hierarchy_dir,
        model_name=args.model,
        output_dir=Path(args.output_dir),
        n_train_pos=args.n_train_pos,
        n_train_neg=args.n_train_neg,
        device=args.device,
        weight_method=args.weight_method,
        activation_threshold=args.activation_threshold,
        layer_idx=args.model_layer,
    )
