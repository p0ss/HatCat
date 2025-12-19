"""
Forward hooks for steering model generation.

Supports four steering modes:
1. Simple projection: steer along a single concept vector
2. Field-based: attract/repel using ontology-weighted concept fields
3. Contrastive: steer along orthogonalized difference vectors
4. Gradient: activation-dependent steering using classifier gradients

All modes support layer-wise dampening via sqrt(1 - depth) decay, which is
RECOMMENDED when steering multiple layers. This prevents cascade failures
in later layers while allowing early layers to steer behavior effectively.

To enable dampening, pass layer_idx and total_layers to the hook creators.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# LENS CLASSIFIER (for gradient-based steering)
# =============================================================================

class LensClassifier(nn.Module):
    """
    MLP lens classifier for concept detection.

    Must match the architecture used in training (src/training/sibling_ranking.py).

    Architecture:
    - Input: hidden_dim (e.g., 4096 for Apertus-8B)
    - Hidden1: 128
    - Hidden2: 64
    - Output: 1 (no Sigmoid - raw logits)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_lens_classifier(
    classifier_path: Union[str, Path],
    device: str = "cuda",
) -> LensClassifier:
    """
    Load a trained lens classifier from a .pt file.

    Reconstructs the LensClassifier architecture from the saved state_dict,
    inferring input dimension from the weight shapes.

    Args:
        classifier_path: Path to the .pt file containing classifier state_dict
        device: Device to load the classifier onto

    Returns:
        Loaded LensClassifier ready for inference/gradient computation
    """
    state_dict = torch.load(classifier_path, map_location=device, weights_only=True)

    # Infer input dimension from first weight shape
    # net.0.weight has shape [128, input_dim]
    first_weight = state_dict.get('net.0.weight')
    if first_weight is None:
        raise ValueError(f"Cannot find 'net.0.weight' in state dict. Keys: {list(state_dict.keys())}")

    _, input_dim = first_weight.shape

    # Create classifier and load weights
    classifier = LensClassifier(input_dim)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    return classifier


def extract_importance_weighted_vector(
    classifier_path: Union[str, Path],
    positive_only: bool = True,
) -> np.ndarray:
    """
    Extract an importance-weighted steering vector from a lens classifier.

    The MLP classifier learns 128 feature directions in the first layer.
    Each feature has an "importance" - how much it contributes to the final
    classification decision. This function computes:

        steering = sum(importance[i] * feature_direction[i])

    This is much more accurate than a simple sum of weights, because:
    - Simple sum includes features that push AWAY from the concept
    - Importance-weighted sum only uses features that increase classification

    Args:
        classifier_path: Path to the .pt file containing classifier state_dict
        positive_only: If True, only use features with positive importance
                      (features that increase classification score)

    Returns:
        Normalized steering vector [input_dim]
    """
    state_dict = torch.load(classifier_path, map_location="cpu", weights_only=True)

    W1 = state_dict["net.0.weight"]  # [128, input_dim] - feature directions
    W2 = state_dict["net.3.weight"]  # [64, 128]
    W3 = state_dict["net.6.weight"]  # [1, 64]

    # Compute importance: how each first-layer feature affects final output
    # importance[i] = sum_j(W3[0,j] * W2[j,i])
    importance = (W3 @ W2).squeeze()  # [128]

    if positive_only:
        # Only use features that INCREASE classification score
        importance = importance.clamp(min=0)

    # Weight feature directions by importance and sum
    steering = (importance.unsqueeze(1) * W1).sum(dim=0)  # [input_dim]

    # Normalize to unit vector
    steering = steering / (steering.norm() + 1e-8)

    return steering.numpy()


def load_steering_vectors_from_lens_pack(
    lens_pack_path: Union[str, Path],
    concepts: List[str],
    positive_only: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load importance-weighted steering vectors for concepts from a lens pack.

    Searches all layers in the lens pack for each concept's classifier,
    then extracts importance-weighted steering vectors.

    Args:
        lens_pack_path: Path to lens pack directory
        concepts: List of concept names to load
        positive_only: If True, only use features with positive importance

    Returns:
        Tuple of:
        - Dict mapping concept name to steering vector
        - Dict mapping concept name to layer where classifier was found
    """
    lens_pack_path = Path(lens_pack_path)
    vectors = {}
    concept_layers = {}

    # Find available layers
    layers = []
    for i in range(10):
        if (lens_pack_path / f"layer{i}").exists():
            layers.append(i)

    for concept in concepts:
        # Search for concept across layers
        for layer in layers:
            classifier_path = lens_pack_path / f"layer{layer}" / f"{concept}_classifier.pt"
            if classifier_path.exists():
                try:
                    vector = extract_importance_weighted_vector(
                        classifier_path, positive_only=positive_only
                    )
                    vectors[concept] = vector
                    concept_layers[concept] = layer
                except Exception as e:
                    print(f"  Warning: Could not load {concept}: {e}")
                break
        else:
            print(f"  Warning: No classifier found for {concept}")

    return vectors, concept_layers


@dataclass
class LayeredClassifier:
    """A lens classifier with its associated concept and layer(s).

    For single-layer classifiers: layer is an int
    For multi-layer classifiers: layer is a list of ints (e.g., [4, 15, 28])

    Multi-layer classifiers have input_dim = len(layers) * hidden_dim.
    When steering, the gradient is split and applied to each selected layer.
    """
    concept: str
    classifier: LensClassifier
    layer: Union[int, List[int]]  # Single layer or list of selected layers


@dataclass
class LayeredSteeringVector:
    """A steering vector with its associated layer."""
    concept: str
    vector: np.ndarray
    layer: int


# =============================================================================
# CONTRASTIVE STEERING
# =============================================================================

def compute_contrastive_vector(
    target_vector: np.ndarray,
    reference_vector: np.ndarray,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Compute contrastive vector: what makes target different from reference.

    Formula: v_contrast = v_target - proj(v_target, v_reference)

    This finds the component of target that is orthogonal to reference,
    i.e., the features that distinguish target from reference.

    Args:
        target_vector: Vector to steer toward (normalized)
        reference_vector: Vector to contrast against (normalized)
        epsilon: Small constant for numerical stability

    Returns:
        (contrastive_vector, magnitude) where magnitude indicates how much
        distinguishing content exists (low = concepts very similar)
    """
    # Project target onto reference
    proj_coef = np.dot(target_vector, reference_vector)
    projection = proj_coef * reference_vector

    # Subtract projection to get orthogonal component
    contrastive = target_vector - projection

    # Compute magnitude before normalizing (indicates distinctiveness)
    magnitude = np.linalg.norm(contrastive)

    # Normalize
    if magnitude > epsilon:
        contrastive = contrastive / magnitude
    else:
        # Vectors are nearly identical, no contrastive component
        contrastive = np.zeros_like(target_vector)

    return contrastive, magnitude


def create_contrastive_steering_hook(
    contrastive_vector: np.ndarray,
    strength: float,
    device: str,
    layer_idx: Optional[int] = None,
    total_layers: Optional[int] = None,
) -> Callable:
    """
    Create hook for contrastive steering.

    Adds the contrastive vector to hidden states, scaled by hidden norm.
    Formula: steered = hidden + strength * dampening * ||hidden|| * vector

    Args:
        contrastive_vector: Orthogonalized difference vector from compute_contrastive_vector()
        strength: Steering strength as fraction of hidden norm
        device: Device tensor should be on
        layer_idx: Current layer index (for dampening). If None, no dampening.
        total_layers: Total model layers (for dampening). If None, no dampening.

    Returns:
        Hook function for PyTorch forward hooks
    """
    # Apply layer-wise dampening if layer info provided
    if layer_idx is not None and total_layers is not None:
        depth = layer_idx / total_layers
        dampening = np.sqrt(1.0 - depth)
        contrastive_vector = contrastive_vector * dampening

    vec_tensor = torch.tensor(contrastive_vector, dtype=torch.float32).to(device)

    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # Standardized: strength=1.0 adds 10% of hidden norm
        vec_matched = vec_tensor.to(dtype=hidden_states.dtype)
        hidden_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
        steered = hidden_states + (strength * 0.1) * hidden_norm * vec_matched

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def compute_contrastive_steering_field(
    target_vectors: Dict[str, np.ndarray],
    reference_vectors: Dict[str, np.ndarray],
    target_weights: Optional[Dict[str, float]] = None,
    strength: float = 1.0,
    min_magnitude: float = 0.1,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Compute contrastive steering field from multiple target/reference pairs.

    For each target, computes contrastive vector against all references,
    then combines weighted contrastive vectors.

    Args:
        target_vectors: Dict of concept_name -> vector to steer toward
        reference_vectors: Dict of concept_name -> vector to contrast against
        target_weights: Optional weights for each target concept
        strength: Overall steering strength
        min_magnitude: Skip contrastive vectors with magnitude below this
        epsilon: Numerical stability constant

    Returns:
        Combined contrastive steering field (normalized)
    """
    if not target_vectors:
        raise ValueError("Must provide at least one target vector")

    hidden_dim = next(iter(target_vectors.values())).shape[0]
    field = np.zeros(hidden_dim, dtype=np.float32)

    # Combine all reference vectors into single reference
    ref_combined = np.zeros(hidden_dim, dtype=np.float32)
    for vec in reference_vectors.values():
        ref_combined += vec
    if np.linalg.norm(ref_combined) > epsilon:
        ref_combined = ref_combined / np.linalg.norm(ref_combined)

    # Compute contrastive vector for each target
    for name, target_vec in target_vectors.items():
        contrastive, magnitude = compute_contrastive_vector(target_vec, ref_combined, epsilon)

        if magnitude < min_magnitude:
            continue  # Skip if target is too similar to reference

        weight = target_weights.get(name, 1.0) if target_weights else 1.0
        field += weight * magnitude * contrastive  # Weight by distinctiveness

    # Normalize
    norm = np.linalg.norm(field) + epsilon
    field = strength * field / norm

    return field


def compute_steering_field(
    attract_vectors: Dict[str, np.ndarray],
    repel_vectors: Dict[str, np.ndarray],
    attract_weights: Optional[Dict[str, float]] = None,
    repel_weights: Optional[Dict[str, float]] = None,
    strength: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Compute a steering field from attraction and repulsion vectors.

    Uses ontology-aware weighting to create a distributional steering direction
    rather than naive single-concept negation.

    δ+ = Σ w_n * v_n  (attraction)
    δ- = Σ λ_r * v_r  (repulsion)
    δ = δ+ - δ-
    δ = s * δ / (||δ|| + ε)  (renormalize to constant energy)

    Args:
        attract_vectors: Dict of concept_name -> vector to attract towards
        repel_vectors: Dict of concept_name -> vector to repel from
        attract_weights: Optional weights for attraction (default: uniform)
        repel_weights: Optional weights for repulsion (default: uniform)
        strength: Steering strength (scales final normalized vector)
        epsilon: Small constant for numerical stability

    Returns:
        Normalized steering direction vector
    """
    hidden_dim = None

    # Compute attraction field
    delta_plus = None
    for name, vec in attract_vectors.items():
        if hidden_dim is None:
            hidden_dim = vec.shape[0]
            delta_plus = np.zeros(hidden_dim, dtype=np.float32)
        weight = attract_weights.get(name, 1.0) if attract_weights else 1.0
        delta_plus += weight * vec

    # Compute repulsion field
    delta_minus = None
    for name, vec in repel_vectors.items():
        if hidden_dim is None:
            hidden_dim = vec.shape[0]
            delta_minus = np.zeros(hidden_dim, dtype=np.float32)
        elif delta_minus is None:
            delta_minus = np.zeros(hidden_dim, dtype=np.float32)
        weight = repel_weights.get(name, 1.0) if repel_weights else 1.0
        delta_minus += weight * vec

    # Combine fields
    if delta_plus is None and delta_minus is None:
        raise ValueError("Must provide at least one attraction or repulsion vector")

    if delta_plus is None:
        delta_plus = np.zeros(hidden_dim, dtype=np.float32)
    if delta_minus is None:
        delta_minus = np.zeros(hidden_dim, dtype=np.float32)

    delta = delta_plus - delta_minus

    # Renormalize to constant steering energy
    norm = np.linalg.norm(delta) + epsilon
    delta = strength * delta / norm

    return delta


def create_field_steering_hook(
    steering_field: np.ndarray,
    device: str,
    layer_idx: Optional[int] = None,
    total_layers: Optional[int] = None,
) -> Callable:
    """
    Create hook for field-based steering.

    Applies precomputed steering field to hidden states, scaled by hidden norm.
    Formula: steered = hidden + dampening * ||hidden|| * field

    Note: The field already has strength baked in from compute_steering_field().
    This hook just scales by hidden norm for standardization.

    Args:
        steering_field: Steering direction from compute_steering_field() (includes strength)
        device: Device tensor should be on
        layer_idx: Current layer index (for dampening). If None, no dampening.
        total_layers: Total model layers (for dampening). If None, no dampening.

    Returns:
        Hook function for PyTorch forward hooks
    """
    # Apply layer-wise dampening if layer info provided
    if layer_idx is not None and total_layers is not None:
        depth = layer_idx / total_layers
        dampening = np.sqrt(1.0 - depth)
        steering_field = steering_field * dampening

    field_tensor = torch.tensor(steering_field, dtype=torch.float32).to(device)

    def hook(module, input, output):
        """Add steering field to hidden states."""
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # Standardized: field already has strength baked in, scale by 10% of hidden norm
        field_matched = field_tensor.to(dtype=hidden_states.dtype)
        hidden_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
        steered = hidden_states + (0.1 * hidden_norm) * field_matched

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def create_steering_hook(
    concept_vector: np.ndarray,
    strength: float,
    device: str,
    normalize: bool = True,
    layer_idx: Optional[int] = None,
    total_layers: Optional[int] = None,
) -> Callable:
    """
    Create hook for steering generation.

    Adds the concept vector to hidden states, scaled by strength and hidden norm.
    Formula: steered = hidden + strength * dampening * ||hidden|| * vector

    When layer_idx and total_layers are provided, applies layer-wise dampening
    using sqrt(1 - depth) to prevent cascade failures in later layers.

    This standardized formula means strength=0.01 adds 1% of hidden norm
    in the concept direction, making strength values comparable across modes.

    Args:
        concept_vector: Normalized concept direction (hidden_dim,)
        strength: Steering strength as fraction of hidden norm
                  (0.01 = 1% of hidden norm, 0.1 = 10%, etc.)
        device: Device tensor should be on
        normalize: If True (default), scale by hidden norm for standardized strength.
                   If False, use legacy projection formula.
        layer_idx: Current layer index (for dampening). If None, no dampening.
        total_layers: Total model layers (for dampening). If None, no dampening.

    Returns:
        Hook function for PyTorch forward hooks

    Example:
        >>> hook_fn = create_steering_hook(concept_vector, strength=0.05, device="cuda",
        ...                                layer_idx=17, total_layers=34)
        >>> handle = model.model.layers[17].register_forward_hook(hook_fn)
        >>> # Generate with steering...
        >>> handle.remove()
    """
    # Apply layer-wise dampening if layer info provided
    if layer_idx is not None and total_layers is not None:
        depth = layer_idx / total_layers
        dampening = np.sqrt(1.0 - depth)  # 1.0 at layer 0, 0.0 at final layer
        concept_vector = concept_vector * dampening

    concept_tensor = torch.tensor(concept_vector, dtype=torch.float32).to(device)

    def hook(module, input, output):
        """Add concept vector to hidden states."""
        # Handle different output formats from transformer layers
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # Match tensor dtype to hidden states
        concept_matched = concept_tensor.to(dtype=hidden_states.dtype)

        if normalize:
            # Standardized: strength=1.0 adds 10% of hidden norm
            # This gives usable range ~0.5-2.0 across all steering modes
            hidden_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
            steered = hidden_states + (strength * 0.1) * hidden_norm * concept_matched
        else:
            # Legacy projection formula: strength * (h·v) * v
            projection = (hidden_states @ concept_matched.unsqueeze(-1)) * concept_matched
            steered = hidden_states + strength * projection

        # Return in same format as input
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Optional[np.ndarray] = None,
    strength: float = 0.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda",
    **generation_kwargs
) -> str:
    """
    Generate text with optional steering applied using forward hooks.

    If steering_vector is None or strength is 0.0, generates without steering.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Text prompt to complete
        steering_vector: Normalized concept vector (optional)
        strength: Steering strength (negative = suppress, positive = amplify)
        layer_idx: Layer to apply steering at (-1 for last layer)
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        Generated text (including prompt)

    Example:
        >>> text = generate_with_steering(
        ...     model, tokenizer,
        ...     prompt="Tell me about",
        ...     steering_vector=person_vector,
        ...     strength=-0.5  # Suppress "person" concept
        ... )
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is None or abs(strength) < 1e-6:
        # No steering
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply steering using forward hook
    hook_fn = create_steering_hook(steering_vector, strength, device)

    layers = get_model_layers(model)
    target_layer = layers[layer_idx] if layer_idx != -1 else layers[-1]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated_text


def generate_with_contrastive_steering(
    model,
    tokenizer,
    prompt: str,
    target_vector: np.ndarray,
    reference_vector: np.ndarray,
    strength: float = 1.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda",
    **generation_kwargs
) -> str:
    """
    Generate text with contrastive steering (RECOMMENDED).

    Steers toward features that distinguish target from reference, not the
    shared features. This works even when target and reference are 90%+ similar.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Text prompt to complete
        target_vector: Concept to steer toward (normalized)
        reference_vector: Concept to contrast against (normalized)
        strength: Steering strength (positive = toward target's unique features)
        layer_idx: Layer to apply steering at (-1 for last layer)
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        Generated text (including prompt)

    Example:
        >>> # Steer from cat toward dog-specific features
        >>> text = generate_with_contrastive_steering(
        ...     model, tokenizer,
        ...     prompt="What animal goes meow?",
        ...     target_vector=dog_vector,
        ...     reference_vector=cat_vector,
        ...     strength=3.0  # Higher strength needed for contrastive
        ... )
    """
    # Compute contrastive vector
    contrastive, magnitude = compute_contrastive_vector(target_vector, reference_vector)

    if magnitude < 0.01:
        # Vectors nearly identical, warn and fall back to no steering
        import warnings
        warnings.warn(
            f"Target and reference vectors are nearly identical (magnitude={magnitude:.4f}). "
            "Contrastive steering will have minimal effect."
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if abs(strength) < 1e-6:
        # No steering
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply contrastive steering using forward hook
    hook_fn = create_contrastive_steering_hook(contrastive, strength, device)

    layers = get_model_layers(model)
    target_layer = layers[layer_idx] if layer_idx != -1 else layers[-1]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated_text


# =============================================================================
# MULTI-LAYER STEERING (for lens vectors trained at different layers)
# =============================================================================

@dataclass
class LayeredSteeringVector:
    """A steering vector with its associated layer."""
    concept: str
    vector: np.ndarray
    layer: int


def get_model_layers(model):
    """Get the transformer layers from model, handling different architectures."""
    # Try common layer access patterns
    if hasattr(model, 'model'):
        inner = model.model
        # Some models nest layers under language_model
        if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
            return inner.language_model.layers
        # Most models have layers directly
        if hasattr(inner, 'layers'):
            return inner.layers
    raise AttributeError(f"Cannot find layers in model architecture: {type(model)}")


def create_multi_layer_steering_hooks(
    model,
    steering_vectors: List['LayeredSteeringVector'],
    reference_vectors: Optional[List['LayeredSteeringVector']] = None,
    strength: float = 1.0,
    contrastive: bool = True,
    device: str = "cuda",
) -> List[Tuple]:
    """
    Create steering hooks for vectors at their native layers.

    Groups vectors by layer, computes combined/contrastive vectors per layer,
    and returns (layer, hook_fn) pairs ready for registration.

    Args:
        model: Language model (for accessing layers)
        steering_vectors: List of LayeredSteeringVector with concept, vector, layer
        reference_vectors: Optional reference vectors for contrastive steering
        strength: Steering strength
        contrastive: If True, use contrastive steering (requires reference_vectors)
        device: Device for tensors

    Returns:
        List of (layer_module, hook_fn) tuples to register

    Example:
        >>> hooks = create_multi_layer_steering_hooks(
        ...     model, steering_vectors, reference_vectors,
        ...     strength=0.3, contrastive=True
        ... )
        >>> handles = apply_steering_hooks(hooks)
        >>> # ... generate ...
        >>> remove_steering_hooks(handles)
    """
    layers = get_model_layers(model)

    # Group steering vectors by layer
    vectors_by_layer = defaultdict(list)
    for sv in steering_vectors:
        vectors_by_layer[sv.layer].append(sv)

    # Group reference vectors by layer
    refs_by_layer = defaultdict(list)
    if reference_vectors:
        for rv in reference_vectors:
            refs_by_layer[rv.layer].append(rv)

    hooks = []

    for layer_idx, layer_vectors in vectors_by_layer.items():
        # Combine vectors for this layer
        combined = None
        for sv in layer_vectors:
            if combined is None:
                combined = sv.vector.copy()
            else:
                combined += sv.vector

        if combined is None:
            continue

        combined = combined / (np.linalg.norm(combined) + 1e-8)

        # Get reference vectors for this layer (for contrastive)
        layer_refs = refs_by_layer.get(layer_idx, [])

        if contrastive and layer_refs:
            # Combine reference vectors
            combined_ref = None
            for rv in layer_refs:
                if combined_ref is None:
                    combined_ref = rv.vector.copy()
                else:
                    combined_ref += rv.vector

            if combined_ref is not None:
                combined_ref = combined_ref / (np.linalg.norm(combined_ref) + 1e-8)
                contrastive_vec, magnitude = compute_contrastive_vector(combined, combined_ref)
                hook_fn = create_contrastive_steering_hook(contrastive_vec, strength, device)
            else:
                # No valid reference, fall back to projection
                hook_fn = create_steering_hook(combined, strength, device)
        else:
            # Projection steering
            hook_fn = create_steering_hook(combined, strength, device)

        target_layer = layers[layer_idx]
        hooks.append((target_layer, hook_fn))

    return hooks


def apply_steering_hooks(hooks: List[Tuple]) -> List:
    """Register hooks and return handles for cleanup."""
    handles = []
    for layer, hook_fn in hooks:
        handles.append(layer.register_forward_hook(hook_fn))
    return handles


def remove_steering_hooks(handles: List):
    """Remove registered hooks."""
    for handle in handles:
        handle.remove()


# =============================================================================
# GRADIENT-BASED STEERING (activation-dependent, RECOMMENDED for lens classifiers)
# =============================================================================

def create_gradient_steering_hook(
    classifier: LensClassifier,
    strength: float,
    device: str,
    toward_concept: bool = True,
) -> Callable:
    """
    Create hook for gradient-based steering (RECOMMENDED for trained lenses).

    Computes the gradient of classifier output with respect to the hidden state
    at each forward pass. This gives the direction to move to increase/decrease
    classifier confidence, which is the activation-dependent steering direction.

    The classifier has learned the features of the concept's lower-dimensional
    subspace. The gradient points toward that subspace from wherever the current
    activation is - this is why it must be activation-dependent.

    Args:
        classifier: Trained BinaryClassifier for the concept
        strength: Steering strength (magnitude of the gradient step)
        device: Device tensors should be on
        toward_concept: If True, steer toward concept (follow gradient).
                       If False, steer away (negative gradient).

    Returns:
        Hook function for PyTorch forward hooks
    """
    sign = 1.0 if toward_concept else -1.0

    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # We need gradients for the steering computation
        # Temporarily disable inference_mode to allow autograd
        original_shape = hidden_states.shape
        classifier_dtype = next(classifier.parameters()).dtype

        # Get hidden state data
        hidden_data = hidden_states.detach().float().cpu().numpy()

        # Exit inference_mode completely for gradient computation
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                # Create fresh tensor with grad enabled
                hidden_for_grad = torch.tensor(
                    hidden_data,
                    dtype=classifier_dtype,
                    device=hidden_states.device,
                    requires_grad=True
                )
                hidden_flat = hidden_for_grad.view(-1, original_shape[-1])

                # Forward through classifier
                output_probs = classifier(hidden_flat)  # [batch*seq, 1]

                # Compute gradient of sum of outputs w.r.t. inputs
                # This gives the direction to increase classifier confidence
                grad_outputs = torch.ones_like(output_probs)
                gradients = torch.autograd.grad(
                    outputs=output_probs,
                    inputs=hidden_flat,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=False,
                )[0]

        # Reshape gradient to match hidden states
        gradients = gradients.view(original_shape)

        # Convert gradient to hidden state dtype
        gradients = gradients.to(dtype=hidden_states.dtype)

        # Normalize gradient to unit vector, then scale by hidden norm
        # This makes strength comparable to other steering modes
        grad_norm = torch.norm(gradients, dim=-1, keepdim=True) + 1e-8
        steering_direction = gradients / grad_norm

        # Standardized: strength=1.0 adds 10% of hidden norm
        hidden_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
        steered = hidden_states + sign * (strength * 0.1) * hidden_norm * steering_direction

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def create_contrastive_gradient_steering_hook(
    target_classifier: LensClassifier,
    reference_classifier: LensClassifier,
    strength: float,
    device: str,
) -> Callable:
    """
    Create hook for contrastive gradient-based steering.

    Computes gradients for both target and reference classifiers, then
    steers along the difference: toward target, away from reference.

    This is the activation-dependent version of contrastive steering.

    Args:
        target_classifier: Classifier for concept to steer toward
        reference_classifier: Classifier for concept to steer away from
        strength: Steering strength
        device: Device for tensors

    Returns:
        Hook function for PyTorch forward hooks
    """
    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # We need gradients for the steering computation
        # Temporarily disable inference_mode to allow autograd
        original_shape = hidden_states.shape
        classifier_dtype = next(target_classifier.parameters()).dtype
        hidden_data = hidden_states.detach().float().cpu().numpy()

        # Exit inference_mode completely for gradient computation
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                # Create fresh tensor with grad enabled
                hidden_for_target = torch.tensor(
                    hidden_data,
                    dtype=classifier_dtype,
                    device=hidden_states.device,
                    requires_grad=True
                )
                hidden_flat_target = hidden_for_target.view(-1, original_shape[-1])

                # Compute target gradient
                target_output = target_classifier(hidden_flat_target)
                target_grad = torch.autograd.grad(
                    outputs=target_output,
                    inputs=hidden_flat_target,
                    grad_outputs=torch.ones_like(target_output),
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Create fresh tensor for reference computation
                hidden_for_ref = torch.tensor(
                    hidden_data,
                    dtype=classifier_dtype,
                    device=hidden_states.device,
                    requires_grad=True
                )
                hidden_flat_ref = hidden_for_ref.view(-1, original_shape[-1])

                ref_output = reference_classifier(hidden_flat_ref)
                ref_grad = torch.autograd.grad(
                    outputs=ref_output,
                    inputs=hidden_flat_ref,
                    grad_outputs=torch.ones_like(ref_output),
                    create_graph=False,
                    retain_graph=False,
                )[0]

        # Contrastive direction: toward target, away from reference
        contrastive_grad = target_grad - ref_grad
        contrastive_grad = contrastive_grad.view(original_shape)
        contrastive_grad = contrastive_grad.to(dtype=hidden_states.dtype)

        # Normalize and apply
        grad_norm = torch.norm(contrastive_grad, dim=-1, keepdim=True) + 1e-8
        steering_direction = contrastive_grad / grad_norm

        steered = hidden_states + strength * steering_direction

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def create_multi_layer_gradient_steering_hooks(
    model,
    target_classifiers: List[LayeredClassifier],
    reference_classifiers: Optional[List[LayeredClassifier]] = None,
    strength: float = 1.0,
    contrastive: bool = True,
    device: str = "cuda",
) -> List[Tuple]:
    """
    Create gradient-based steering hooks for classifiers at their native layers.

    This is the recommended approach for steering with trained lens classifiers.
    The gradient is activation-dependent, computed fresh at each forward pass.

    Args:
        model: Language model (for accessing layers)
        target_classifiers: List of LayeredClassifier for concepts to steer toward
        reference_classifiers: Optional classifiers for contrastive steering
        strength: Steering strength
        contrastive: If True and reference_classifiers provided, use contrastive
        device: Device for tensors

    Returns:
        List of (layer_module, hook_fn) tuples to register

    Example:
        >>> # Load classifiers
        >>> cat_cls = load_lens_classifier("lens_packs/.../layer3/DomesticCat_classifier.pt")
        >>> dog_cls = load_lens_classifier("lens_packs/.../layer3/DomesticDog_classifier.pt")
        >>>
        >>> targets = [LayeredClassifier("DomesticCat", cat_cls, 3)]
        >>> refs = [LayeredClassifier("DomesticDog", dog_cls, 3)]
        >>>
        >>> hooks = create_multi_layer_gradient_steering_hooks(
        ...     model, targets, refs, strength=0.5, contrastive=True
        ... )
        >>> handles = apply_steering_hooks(hooks)
        >>> # ... generate ...
        >>> remove_steering_hooks(handles)
    """
    layers = get_model_layers(model)
    hidden_dim = layers[0].weight.shape[0] if hasattr(layers[0], 'weight') else None

    # Group classifiers by layer
    # For multi-layer classifiers, we need special handling
    targets_by_layer = defaultdict(list)
    multi_layer_targets = []  # Track multi-layer classifiers separately

    for lc in target_classifiers:
        if isinstance(lc.layer, list):
            # Multi-layer classifier - handle separately
            multi_layer_targets.append(lc)
        else:
            targets_by_layer[lc.layer].append(lc)

    refs_by_layer = defaultdict(list)
    multi_layer_refs = []

    if reference_classifiers:
        for lc in reference_classifiers:
            if isinstance(lc.layer, list):
                multi_layer_refs.append(lc)
            else:
                refs_by_layer[lc.layer].append(lc)

    hooks = []

    for layer_idx, layer_targets in targets_by_layer.items():
        layer_refs = refs_by_layer.get(layer_idx, [])

        if len(layer_targets) == 1 and len(layer_refs) <= 1:
            # Single target, optionally single reference
            target_cls = layer_targets[0].classifier

            if contrastive and len(layer_refs) == 1:
                ref_cls = layer_refs[0].classifier
                hook_fn = create_contrastive_gradient_steering_hook(
                    target_cls, ref_cls, strength, device
                )
            else:
                hook_fn = create_gradient_steering_hook(
                    target_cls, strength, device, toward_concept=True
                )
        else:
            # Multiple classifiers at this layer - create combined hook
            hook_fn = _create_combined_gradient_hook(
                layer_targets, layer_refs, strength, contrastive, device
            )

        target_layer = layers[layer_idx]
        hooks.append((target_layer, hook_fn))

    # Handle multi-layer classifiers
    # For these, we extract importance-weighted vectors and apply to each layer
    for lc in multi_layer_targets:
        layer_list = lc.layer  # List of layer indices
        n_layers = len(layer_list)

        # Get the classifier's first-layer weights
        if hasattr(lc.classifier, 'net') and len(lc.classifier.net) >= 7:
            # MLP structure: Linear(input→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→1)
            W1 = lc.classifier.net[0].weight  # [128, input_dim]
            W2 = lc.classifier.net[3].weight  # [64, 128]
            W3 = lc.classifier.net[6].weight  # [1, 64]

            # Compute importance-weighted vector
            importance = (W3 @ W2).squeeze()  # [128]
            importance_positive = importance.clamp(min=0)
            full_vector = (importance_positive.unsqueeze(1) * W1).sum(dim=0)  # [input_dim]

            # Infer hidden_dim from total input size
            input_dim = full_vector.shape[0]
            hidden_dim_per_layer = input_dim // n_layers

            # Split vector and create hook for each layer
            for i, layer_idx in enumerate(layer_list):
                start_idx = i * hidden_dim_per_layer
                end_idx = (i + 1) * hidden_dim_per_layer
                layer_vector = full_vector[start_idx:end_idx].detach().cpu().numpy()

                # Normalize
                layer_vector = layer_vector / (np.linalg.norm(layer_vector) + 1e-8)

                # Create static steering hook for this layer
                hook_fn = create_steering_hook(
                    torch.from_numpy(layer_vector).to(device),
                    strength=strength / n_layers,  # Divide strength across layers
                )
                hooks.append((layers[layer_idx], hook_fn))

    return hooks


def _create_combined_gradient_hook(
    target_classifiers: List[LayeredClassifier],
    reference_classifiers: List[LayeredClassifier],
    strength: float,
    contrastive: bool,
    device: str,
) -> Callable:
    """Create hook that combines gradients from multiple classifiers."""
    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        original_shape = hidden_states.shape
        hidden_data = hidden_states.detach().float().cpu().numpy()

        # Temporarily disable inference_mode to allow autograd
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                combined_grad = None

                # Sum target gradients
                for lc in target_classifiers:
                    classifier_dtype = next(lc.classifier.parameters()).dtype
                    h = torch.tensor(
                        hidden_data,
                        dtype=classifier_dtype,
                        device=hidden_states.device,
                        requires_grad=True
                    ).view(-1, original_shape[-1])

                    out = lc.classifier(h)
                    grad = torch.autograd.grad(
                        out, h,
                        grad_outputs=torch.ones_like(out),
                        create_graph=False,
                    )[0]
                    if combined_grad is None:
                        combined_grad = grad.float()
                    else:
                        combined_grad = combined_grad + grad.float()

                # Subtract reference gradients if contrastive
                if contrastive and reference_classifiers:
                    for lc in reference_classifiers:
                        classifier_dtype = next(lc.classifier.parameters()).dtype
                        h = torch.tensor(
                            hidden_data,
                            dtype=classifier_dtype,
                            device=hidden_states.device,
                            requires_grad=True
                        ).view(-1, original_shape[-1])

                        out = lc.classifier(h)
                        grad = torch.autograd.grad(
                            out, h,
                            grad_outputs=torch.ones_like(out),
                            create_graph=False,
                        )[0]
                        combined_grad = combined_grad - grad.float()

        # Reshape and normalize
        combined_grad = combined_grad.view(original_shape)
        combined_grad = combined_grad.to(dtype=hidden_states.dtype)
        grad_norm = torch.norm(combined_grad, dim=-1, keepdim=True) + 1e-8
        steering_direction = combined_grad / grad_norm

        steered = hidden_states + strength * steering_direction

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def load_lens_classifiers_for_concepts(
    lens_pack_path: Union[str, Path],
    concepts: List[str],
    device: str = "cuda",
) -> Tuple[Dict[str, LayeredClassifier], Dict[str, str]]:
    """
    Load lens classifiers for specified concepts, searching across all layers.

    Supports both single-layer and multi-layer classifiers:
    - Single-layer: classifier trained on one layer, layer is int
    - Multi-layer: classifier trained on selected layers (early/mid/late),
                   layer is List[int] loaded from results.json "selected_layers"

    Args:
        lens_pack_path: Path to lens pack directory
        concepts: List of concept names to load
        device: Device to load classifiers onto

    Returns:
        Tuple of:
        - Dict mapping concept name to LayeredClassifier
        - Dict mapping concept name to error message (for concepts not found)
    """
    lens_pack_path = Path(lens_pack_path)
    classifiers = {}
    errors = {}

    # Pre-load results.json from all layers to get selected_layers metadata
    selected_layers_map = {}  # concept -> list of selected layers
    for layer_dir in lens_pack_path.glob("layer*"):
        results_path = layer_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results_data = json.load(f)
                for result in results_data.get("results", []):
                    concept_name = result.get("concept")
                    sel_layers = result.get("selected_layers")
                    if concept_name and sel_layers:
                        selected_layers_map[concept_name] = sel_layers
            except (json.JSONDecodeError, KeyError):
                pass

    for concept in concepts:
        found = False
        for layer_dir in sorted(lens_pack_path.glob("layer*")):
            layer_num = int(layer_dir.name.replace("layer", ""))
            classifier_path = layer_dir / f"{concept}_classifier.pt"

            if classifier_path.exists():
                try:
                    classifier = load_lens_classifier(classifier_path, device)

                    # Check if this is a multi-layer classifier
                    if concept in selected_layers_map:
                        # Multi-layer mode: use selected layers from training
                        layer_info = selected_layers_map[concept]
                    else:
                        # Single-layer mode
                        layer_info = layer_num

                    classifiers[concept] = LayeredClassifier(
                        concept=concept,
                        classifier=classifier,
                        layer=layer_info,
                    )
                    found = True
                    break
                except Exception as e:
                    errors[concept] = f"Failed to load: {e}"
                    break

        if not found and concept not in errors:
            errors[concept] = "Not found in any layer"

    return classifiers, errors
