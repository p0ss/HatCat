"""Utilities for training SUMO hierarchical concept classifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


def get_hidden_dim(model) -> int:
    """Get hidden dimension from model config, handling different model architectures.

    Gemma3 uses config.text_config.hidden_size while most models use config.hidden_size.
    """
    config = model.config
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return config.text_config.hidden_size
    else:
        raise AttributeError(
            f"Cannot find hidden_size in model config. "
            f"Config type: {type(config).__name__}. "
            f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
        )


def get_num_layers(model) -> int:
    """Get number of transformer layers from model config.

    Different models use different config attributes:
    - Most models: config.num_hidden_layers
    - Gemma3 VLM: config.text_config.num_hidden_layers
    - ChatGLM: config.num_layers
    """
    config = model.config
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        # ChatGLM uses num_layers instead of num_hidden_layers
        return config.num_layers
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        return config.text_config.num_hidden_layers
    else:
        raise AttributeError(
            f"Cannot find num_hidden_layers or num_layers in model config. "
            f"Config type: {type(config).__name__}. "
            f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
        )


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


def _sync_parent_child_relationships(all_concepts: List[Dict]) -> None:
    """Ensure parent_concepts and category_children are symmetric.

    The hierarchy stores relationships in two places:
    - parent_concepts on children (upward links)
    - category_children on parents (downward links)

    These can get out of sync. This function merges both directions so that
    if A->B exists in either direction, it exists in both.
    """
    concept_map = {c['sumo_term']: c for c in all_concepts}

    # First pass: add missing category_children from parent_concepts
    for concept in all_concepts:
        child_term = concept['sumo_term']
        for parent_term in concept.get('parent_concepts', []):
            if parent_term in concept_map:
                parent = concept_map[parent_term]
                if 'category_children' not in parent:
                    parent['category_children'] = []
                if child_term not in parent['category_children']:
                    parent['category_children'].append(child_term)

    # Second pass: add missing parent_concepts from category_children
    for concept in all_concepts:
        parent_term = concept['sumo_term']
        for child_term in concept.get('category_children', []):
            if child_term in concept_map:
                child = concept_map[child_term]
                if 'parent_concepts' not in child:
                    child['parent_concepts'] = []
                if parent_term not in child['parent_concepts']:
                    child['parent_concepts'].append(parent_term)


def load_all_concepts(hierarchy_dir: Path) -> List[Dict]:
    """Load all concepts from all layers for negative pool construction.

    Args:
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.

    Note:
        This function rebuilds category_children from parent_concepts to ensure
        parent-child relationships are symmetric regardless of query direction.
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

    # Ensure parent_concepts and category_children are symmetric
    _sync_parent_child_relationships(all_concepts)

    return all_concepts


def extract_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cuda",
    max_new_tokens: int = 20,
    temperature_range: Tuple[float, float] = (0.3, 0.9),
    batch_size: int = 4,
    extraction_mode: str = "combined",
    greedy: bool = False,
    pooling: str = "mean",
    layer_idx: Optional[Union[int, List[int]]] = 15,
    return_texts: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Extract activations from model hidden states.

    Uses the "combined-20" strategy:
    - Extracts activations from BOTH prompt processing and generation phases
    - Doubles training samples at zero additional computational cost

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate from
        device: Device for inference
        max_new_tokens: Max tokens to generate per prompt
        temperature_range: (min_temp, max_temp) to vary across batches (ignored if greedy=True)
        batch_size: Number of prompts to process in parallel
        extraction_mode: "combined" (prompt+gen), "generation" (gen only), "prompt" (prompt only)
                         For base models, "prompt" may give cleaner signal by avoiding
                         generation noise. Default is "combined".
        greedy: If True, use greedy decoding for deterministic on-topic outputs.
                Default False. Set True for temperature-based sampling diversity.
        pooling: How to pool across sequence positions:
                 - "mean": Average across all positions (default)
                 - "last": Use last token only (cleaner concept signal)
                 - "max": Max across positions (preserves strongest activations)
        layer_idx: Which layer(s) to extract from. Options:
                   - int (e.g., 15): Single layer mode (default)
                   - List[int] (e.g., [4, 15, 28]): Multi-layer mode, concatenates specified layers
                   - None: All-layers mode (concatenates all layers)
        return_texts: If True, also return the generated texts for sample saving

    Returns:
        If return_texts=False: Array of activation vectors
        If return_texts=True: Tuple of (activations, generated_texts)
        - If layer_idx is int: Shape [n_samples, hidden_dim]
        - If layer_idx is list: Shape [n_samples, len(layer_idx) * hidden_dim]
        - If layer_idx is None: Shape [n_samples, n_layers * hidden_dim]
        - If extraction_mode="combined": n_samples = 2 * n_prompts
        - If extraction_mode="prompt" or "generation": n_samples = n_prompts
    """
    all_layers_mode = layer_idx is None
    multi_layer_mode = isinstance(layer_idx, list)
    activations: List[np.ndarray] = []
    generated_texts: List[str] = []
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

            # === PHASE 1: PROMPT PROCESSING (if combined or prompt mode) ===
            if extraction_mode in ("combined", "prompt"):
                prompt_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                # hidden_states is tuple of (n_layers+1) tensors, each [batch, seq, hidden_dim]
                # Index 0 is embedding layer, 1..n_layers are transformer layers

                if all_layers_mode:
                    # All-layers mode: concatenate all layers
                    layers_to_use = prompt_outputs.hidden_states[1:]  # Skip embedding
                elif multi_layer_mode:
                    # Multi-layer mode: concatenate specified layers
                    layers_to_use = [prompt_outputs.hidden_states[li + 1] for li in layer_idx]
                else:
                    # Single-layer mode: just use the specified layer
                    # layer_idx=15 means hidden_states[16] (0 is embedding)
                    layers_to_use = [prompt_outputs.hidden_states[layer_idx + 1]]

                # Pool each sequence in batch
                for seq_idx in range(len(batch_prompts)):
                    # Get attention mask for this sequence (to find real last token)
                    attn_mask = inputs.attention_mask[seq_idx]
                    seq_len = attn_mask.sum().item()  # Number of real tokens

                    # Collect pooled activations from selected layers
                    layer_activations = []
                    for layer_hidden in layers_to_use:
                        seq_hidden = layer_hidden[seq_idx]  # [seq_len, hidden_dim]

                        if pooling == "last":
                            # Use last real token (most concept-concentrated)
                            layer_pooled = seq_hidden[seq_len - 1]  # [hidden_dim]
                        elif pooling == "max":
                            # Max across positions (preserves strongest signals)
                            layer_pooled = seq_hidden[:seq_len].max(dim=0)[0]  # [hidden_dim]
                        else:  # "mean"
                            # Mean across positions (default)
                            layer_pooled = seq_hidden[:seq_len].mean(dim=0)  # [hidden_dim]

                        layer_activations.append(layer_pooled)

                    # Concatenate layers (single layer = just that layer, all = all concatenated)
                    result = torch.cat(layer_activations, dim=0)
                    activations.append(result.float().cpu().numpy())

            # === PHASE 2: GENERATION (skip if prompt-only mode) ===
            if extraction_mode != "prompt":
                if greedy:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=float(temp),
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Capture generated texts if requested
                if return_texts:
                    for seq_idx in range(len(batch_prompts)):
                        # Get the generated sequence (excluding prompt)
                        prompt_len = inputs.input_ids[seq_idx].shape[0]
                        generated_ids = outputs.sequences[seq_idx][prompt_len:]
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        generated_texts.append(generated_text)

                # Extract hidden states from generation
                # outputs.hidden_states is tuple of tuples: (step1, step2, ...)
                # Each step is tuple of layers: (layer0, layer1, ..., layerN)
                # Each step's hidden state is for the newly generated token (seq_len=1)

                n_model_layers = len(outputs.hidden_states[0]) - 1  # Exclude embedding layer

                if all_layers_mode:
                    # All-layers mode: use all layers
                    layer_indices = range(1, n_model_layers + 1)  # Skip embedding (layer 0)
                elif multi_layer_mode:
                    # Multi-layer mode: use specified layers
                    layer_indices = [li + 1 for li in layer_idx]  # +1 because 0 is embedding
                else:
                    # Single-layer mode: just the specified layer
                    layer_indices = [layer_idx + 1]  # +1 because 0 is embedding

                # Process each sequence in the batch
                for seq_idx in range(len(batch_prompts)):
                    layer_pooled_activations = []

                    for li in layer_indices:
                        step_activations = []
                        for step_hidden in outputs.hidden_states:
                            layer_hidden = step_hidden[li]  # [batch_size, seq_len, hidden_dim]
                            seq_hidden = layer_hidden[seq_idx]  # [seq_len, hidden_dim]
                            # Each generation step has seq_len=1, so just take [-1]
                            step_activations.append(seq_hidden[-1])  # [hidden_dim]

                        # Pool across generation steps for this layer
                        all_steps = torch.stack(step_activations, dim=0)  # [n_steps, hidden_dim]
                        if pooling == "last":
                            # Use last generation step (final state)
                            layer_final = all_steps[-1]  # [hidden_dim]
                        elif pooling == "max":
                            # Max across steps
                            layer_final = all_steps.max(dim=0)[0]  # [hidden_dim]
                        else:  # "mean"
                            layer_final = all_steps.mean(dim=0)  # [hidden_dim]
                        layer_pooled_activations.append(layer_final)

                    # Concatenate layers
                    result = torch.cat(layer_pooled_activations, dim=0)
                    activations.append(result.float().cpu().numpy())

    if return_texts:
        return np.array(activations), generated_texts
    return np.array(activations)


def select_layers_for_concept(
    model,
    tokenizer,
    pos_prompts: Sequence[str],
    neg_prompts: Sequence[str],
    device: str = "cuda",
    n_model_layers: int = 34,
    sample_size: int = 20,
    top_k: int = 1,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Select the best layer(s) from each third of the model for a concept.

    This function trains a quick logistic regression classifier at each layer
    and returns the top-k performing layers from each third:
    - Early (0 to n/3-1): atomic/word-level features
    - Mid (n/3 to 2n/3-1): deeper semantic meaning
    - Late (2n/3 to n-1): linguistic/output representation

    Args:
        model: Language model
        tokenizer: Tokenizer
        pos_prompts: Positive example prompts for this concept
        neg_prompts: Negative example prompts
        device: Device for inference
        n_model_layers: Total number of layers in the model (default 34 for Gemma 3 4B)
        sample_size: Max samples to use from each class (default 20)
        top_k: Number of top layers to select from each third (default 1).
               Higher k = more compute/memory but potentially better coverage.
               k=1 → 3 layers, k=2 → 6 layers, k=3 → 9 layers, etc.

    Returns:
        Tuple of (selected_layers, layer_scores) where:
        - selected_layers: List of 3*top_k layer indices, sorted by position
        - layer_scores: Dict mapping layer index to F1 score
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Limit samples for speed
    pos_prompts = pos_prompts[:sample_size]
    neg_prompts = neg_prompts[:sample_size]
    all_prompts = list(pos_prompts) + list(neg_prompts)
    labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

    # Define thirds
    third = n_model_layers // 3
    early_range = range(0, third)  # 0-11 for 34 layers
    mid_range = range(third, 2 * third)  # 12-22
    late_range = range(2 * third, n_model_layers)  # 23-33

    layer_scores = {}

    # Extract activations and score each layer
    print(f"    Layer selection: testing {n_model_layers} layers (top_k={top_k})...", end="", flush=True)

    for layer_idx in range(n_model_layers):
        # Extract activations for this single layer
        X = extract_activations(
            model,
            tokenizer,
            all_prompts,
            device=device,
            extraction_mode="prompt",  # Faster, just prompt phase
            layer_idx=layer_idx,
        )

        # Handle combined extraction (2x samples)
        if X.shape[0] == 2 * len(labels):
            y = np.repeat(labels, 2)
        else:
            y = labels

        # Quick logistic regression with cross-validation
        try:
            # Use higher max_iter to avoid convergence warnings
            # Scale features for better convergence
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            clf = LogisticRegression(max_iter=500, solver='lbfgs')
            scores = cross_val_score(clf, X_scaled, y, cv=3, scoring='f1')
            layer_scores[layer_idx] = float(np.mean(scores))
        except Exception:
            layer_scores[layer_idx] = 0.0

    print(" done")

    # Find top-k in each third
    def top_k_in_range(layer_range, k):
        scores_in_range = {l: layer_scores[l] for l in layer_range}
        sorted_layers = sorted(scores_in_range.keys(), key=lambda l: scores_in_range[l], reverse=True)
        return sorted_layers[:min(k, len(sorted_layers))]

    best_early = top_k_in_range(early_range, top_k)
    best_mid = top_k_in_range(mid_range, top_k)
    best_late = top_k_in_range(late_range, top_k)

    # Combine and sort by layer position for consistent concatenation order
    selected = sorted(best_early + best_mid + best_late)

    # Format output message
    early_str = ",".join(str(l) for l in best_early)
    mid_str = ",".join(str(l) for l in best_mid)
    late_str = ",".join(str(l) for l in best_late)
    early_f1 = ",".join(f"{layer_scores[l]:.3f}" for l in best_early)
    mid_f1 = ",".join(f"{layer_scores[l]:.3f}" for l in best_mid)
    late_f1 = ",".join(f"{layer_scores[l]:.3f}" for l in best_late)

    print(f"    Selected layers: early=[{early_str}] mid=[{mid_str}] late=[{late_str}]")
    print(f"    F1 scores:       early=[{early_f1}] mid=[{mid_f1}] late=[{late_f1}]")

    return selected, layer_scores


class _TrainableMLP(torch.nn.Module):
    """Wrapper to give Sequential the 'net.' prefix in state_dict keys."""
    def __init__(self, net: torch.nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def train_simple_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 0.001,
    dtype: torch.dtype = torch.bfloat16,
    use_layer_norm: bool = False,
    normalize_inputs: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Train a simple MLP classifier and return metrics.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        hidden_dim: MLP hidden layer dimension
        epochs: Training epochs
        lr: Learning rate
        dtype: Parameter dtype. Default bfloat16 for memory efficiency.
        use_layer_norm: Whether to include LayerNorm at input. Default False
            to match existing pack architecture for batched inference.
        normalize_inputs: Whether to standardize inputs (subtract mean, divide by std).
            Default True. Essential for layers with high-magnitude activations.
    """
    import torch.nn as nn
    import torch.optim as optim

    # Normalize inputs to prevent gradient saturation
    # This matches the LayerNorm applied at inference time in lens_manager
    # LayerNorm normalizes each sample independently (across hidden dim)
    if normalize_inputs:
        # Per-sample normalization (same as nn.LayerNorm with elementwise_affine=False)
        train_mean = X_train.mean(axis=1, keepdims=True)
        train_std = X_train.std(axis=1, keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std

        test_mean = X_test.mean(axis=1, keepdims=True)
        test_std = X_test.std(axis=1, keepdims=True) + 1e-8
        X_test = (X_test - test_mean) / test_std

    input_dim = X_train.shape[1]
    layers = []
    if use_layer_norm:
        layers.append(nn.LayerNorm(input_dim, dtype=dtype))
    layers.extend([
        nn.Linear(input_dim, hidden_dim, dtype=dtype),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1, dtype=dtype),
        nn.Sigmoid(),
    ])
    sequential = nn.Sequential(*layers)
    model = _TrainableMLP(sequential)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert training data to match model dtype
    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device=device, dtype=dtype)
    X_test_t = torch.from_numpy(X_test).to(device=device, dtype=dtype)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1).to(device=device, dtype=dtype)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    initial_loss = None
    final_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        if epoch == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    final_loss = loss.item()

    # Debug: check if learning happened
    if abs(initial_loss - final_loss) < 0.001:
        pred_mean = outputs.mean().item()
        pred_std = outputs.std().item()
        x_mean = X_train_t.mean().item()
        x_std = X_train_t.std().item()
        print(f"      ⚠️  No learning: loss {initial_loss:.4f}→{final_loss:.4f}, "
              f"pred μ={pred_mean:.3f} σ={pred_std:.4f}, X μ={x_mean:.1f} σ={x_std:.1f}")

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
    train_text_lenses: bool = False,
    validation_mode: str = 'falloff',
    validation_threshold: float = 0.5,
    include_sibling_negatives: bool = True,
    only_concepts: set = None,
    all_layers: bool = False,
    multi_layer_mode: bool = False,
    multi_layer_top_k: int = 1,
    sample_saver=None,
) -> Dict:
    """
    Train classifiers for a single SUMO abstraction layer.

    Args:
        layer: The layer number (0-6)
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
        save_text_samples: If True, save generated text for text lens training (legacy)
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
        train_text_lenses: If True, compute embedding centroids (legacy, not currently used)
        include_sibling_negatives: If True (default), include siblings as hard negatives.
                                   Set to False for two-pass training (sibling refinement done separately).
        all_layers: If True, extract from all model layers (experimental, large classifiers)
        multi_layer_mode: If True, auto-select best layer from each third (early/mid/late)
                         and concatenate those layers for training. More efficient than all_layers.
        multi_layer_top_k: Number of top layers to select from each third (default 1).
                          k=1 → 3 layers, k=2 → 6 layers, k=3 → 9 layers, etc.
                          Higher k = more compute/memory but potentially better coverage.
        sample_saver: Optional SampleSaver instance for saving training samples with quality checking.
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

    # Track all text samples for text lens training
    if save_text_samples:
        text_samples_dir = output_dir / "text_samples"
        text_samples_dir.mkdir(exist_ok=True)

    # Create centroid output directory if needed
    if train_text_lenses:
        centroid_output_dir = output_dir / "embedding_centroids"
        centroid_output_dir.mkdir(exist_ok=True)

    # Initialize adaptive trainer if requested (activation lenses only now)
    # Uses defaults from DualAdaptiveTrainer: 20 samples, 0.95 F1 target
    # Note: For multi_layer_mode, we update validation_layer_idx per concept
    adaptive_trainer = None
    n_model_layers = get_num_layers(model) if multi_layer_mode else None
    if use_adaptive_training:
        from .dual_adaptive_trainer import DualAdaptiveTrainer
        # Determine initial layer_idx
        if all_layers:
            initial_layer_idx = None  # All layers
        elif multi_layer_mode:
            initial_layer_idx = 15  # Will be updated per concept
        else:
            initial_layer_idx = 15  # Single layer default
        adaptive_trainer = DualAdaptiveTrainer(
            # Use class defaults for activation lens config (20 samples, 0.95 F1)
            model=model,  # Needed for validation
            tokenizer=tokenizer,  # Needed for validation
            max_response_tokens=100,
            validate_lenses=True,  # Validates against parent/siblings (what we train against)
            validation_mode=validation_mode,  # Validation mode (loose/falloff/strict)
            validation_threshold=validation_threshold,  # Min score to pass (for strict mode)
            validation_layer_idx=initial_layer_idx,
            train_activation=True,
            train_text=False,  # Disable TF-IDF text lens training
            hierarchy_dir=hierarchy_dir,  # For accurate domain inference in validation
            sample_saver=sample_saver,  # For saving training samples with quality checks
        )

    for i, concept in enumerate(concepts):
        concept_name = concept["sumo_term"]

        # Skip if not in only_concepts filter (when provided)
        if only_concepts is not None and concept_name not in only_concepts:
            print(f"\n[{i + 1}/{len(concepts)}] Skipping {concept_name} (not in training manifest)")
            continue

        # Check if already trained (resume capability)
        classifier_path = output_dir / f"{concept_name}.pt"
        centroid_path = centroid_output_dir / f"{concept_name}_centroid.npy" if train_text_lenses else None

        if classifier_path.exists() and (not train_text_lenses or centroid_path.exists()):
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

            # Track selected layers for result metadata
            selected_layers = None

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

                # Multi-layer mode: select best layers for this concept
                if multi_layer_mode and adaptive_trainer is not None:
                    # Generate small sample set for layer selection
                    layer_select_prompts, layer_select_labels = create_sumo_training_dataset(
                        concept=concept,
                        all_concepts=concept_map,
                        negative_pool=negative_pool,
                        n_positives=20,  # Small sample for quick layer probing
                        n_negatives=20,
                        use_category_relationships=True,
                        use_wordnet_relationships=True,
                    )
                    pos_prompts = [p for p, l in zip(layer_select_prompts, layer_select_labels) if l == 1]
                    neg_prompts = [p for p, l in zip(layer_select_prompts, layer_select_labels) if l == 0]

                    selected_layers, _ = select_layers_for_concept(
                        model=model,
                        tokenizer=tokenizer,
                        pos_prompts=pos_prompts,
                        neg_prompts=neg_prompts,
                        device=device,
                        n_model_layers=n_model_layers,
                        top_k=multi_layer_top_k,
                    )
                    # Update adaptive trainer to use selected layers
                    adaptive_trainer.validation_layer_idx = selected_layers
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

                # Save text samples for text lens training (non-adaptive only)
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
                    'layer': layer,  # For sample saving
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
                        output_dir / f"{concept_name}.pt"
                    )

                # Note: Embedding centroids not supported in JIT adaptive mode
                # (would require returning prompts from adaptive trainer)
                if train_text_lenses:
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

                # Add selected layers if multi-layer mode was used
                if selected_layers is not None:
                    result["selected_layers"] = selected_layers

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

                if train_text_lenses:
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
                # Multi-layer mode: select best layers for this concept
                if multi_layer_mode:
                    pos_prompts = [p for p, l in zip(train_prompts, train_labels) if l == 1]
                    neg_prompts = [p for p, l in zip(train_prompts, train_labels) if l == 0]

                    selected_layers, _ = select_layers_for_concept(
                        model=model,
                        tokenizer=tokenizer,
                        pos_prompts=pos_prompts[:20],  # Use subset for speed
                        neg_prompts=neg_prompts[:20],
                        device=device,
                        n_model_layers=n_model_layers or get_num_layers(model),
                        top_k=multi_layer_top_k,
                    )
                    extraction_layer_idx = selected_layers
                elif all_layers:
                    extraction_layer_idx = None  # All layers mode
                else:
                    extraction_layer_idx = 15  # Default single layer

                X_train = extract_activations(model, tokenizer, train_prompts, device, layer_idx=extraction_layer_idx)
                X_test = extract_activations(model, tokenizer, test_prompts, device, layer_idx=extraction_layer_idx)

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

                # Add selected layers if multi-layer mode was used
                if selected_layers is not None:
                    result["selected_layers"] = selected_layers

                torch.save(classifier.state_dict(), output_dir / f"{concept_name}.pt")

            all_results.append(result)

        except Exception as exc:  # pragma: no cover - logging path
            print(f"  ✗ ERROR: {exc}")
            failed_concepts.append({"concept": concept_name, "error": str(exc)})
            continue

    # Save centroid metadata
    if train_text_lenses:
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
    train_text_lenses: bool = False,
    use_adaptive_training: bool = False,
    validation_mode: str = 'falloff',
    validation_threshold: float = 0.5,
    include_sibling_negatives: bool = True,
    run_sibling_refinement: bool = False,
    sibling_refine_epochs: int = 20,
    sibling_refine_prompts: int = 15,
    only_concepts: set = None,
    all_layers: bool = False,
    multi_layer_mode: bool = False,
    multi_layer_top_k: int = 1,
    sample_saver=None,
) -> List[Dict]:
    """
    High-level entry point for training multiple layers.

    Args:
        layers: Which concept layers to train (0-6)
        hierarchy_dir: Path to the hierarchy directory containing layerN.json files.
                      This is a required parameter to prevent accidentally using the wrong hierarchy.
        train_text_lenses: If True, compute embedding centroids (legacy, not currently used)
        use_adaptive_training: If True, use DualAdaptiveTrainer for independent graduation
        include_sibling_negatives: If True (default), include siblings as hard negatives.
                                   Set to False for two-pass training (sibling refinement done separately).
        all_layers: If True, extract from all model layers (experimental, large classifiers)
        multi_layer_mode: If True, auto-select best layer from each third (early/mid/late)
                         and concatenate those layers for training. More efficient than all_layers.
        multi_layer_top_k: Number of top layers to select from each third (default 1).
                          k=1 → 3 layers, k=2 → 6 layers, k=3 → 9 layers, etc.
        sample_saver: Optional SampleSaver instance for saving training samples with quality checking.
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
    print(f"Train text lenses: {train_text_lenses}")
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
        hidden_dim = get_hidden_dim(model)

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
                    lens_dir=output_dir / f"layer{layer}",
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
            save_text_samples=train_text_lenses and not use_adaptive_training,  # Only save if not adaptive
            use_adaptive_training=use_adaptive_training,
            train_text_lenses=train_text_lenses,
            validation_mode=validation_mode,
            validation_threshold=validation_threshold,
            include_sibling_negatives=include_sibling_negatives,
            only_concepts=only_concepts,
            all_layers=all_layers,
            multi_layer_mode=multi_layer_mode,
            multi_layer_top_k=multi_layer_top_k,
            sample_saver=sample_saver,
        )
        summaries.append(summary)

        # Sibling ranking refinement after each layer (if enabled)
        if run_sibling_refinement:
            from .sibling_ranking import refine_all_sibling_groups
            hidden_dim = get_hidden_dim(model)
            refine_all_sibling_groups(
                layer=layer,
                lens_dir=output_dir / f"layer{layer}",
                hierarchy_dir=hierarchy_dir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_prompts_per_sibling=sibling_refine_prompts,
                epochs=sibling_refine_epochs,
                hidden_dim=hidden_dim,
            )

        # Save samples collected during this layer's training
        if sample_saver is not None:
            sample_saver.save_layer(layer)

    # Save all collected samples and quality report
    if sample_saver is not None:
        sample_saver.save_all()

    # Compute centroids separately ONLY if NOT using adaptive training
    # (adaptive training computes them inline)
    if train_text_lenses and not use_adaptive_training:
        print(f"\n{'=' * 80}")
        print("COMPUTING EMBEDDING CENTROIDS")
        print(f"{'=' * 80}")

        from .text_lenses import compute_centroids_for_layer

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
    "select_layers_for_concept",
    "get_hidden_dim",
    "get_num_layers",
]
