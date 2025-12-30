"""
Phase 6.6: Dual-Subspace Manifold Steering

Combines contamination removal (Phase 6) with task manifold projection (Huang et al.)
to enable stable steering at high magnitudes (±1.0+).

Key insight: Two complementary operations needed:
1. Remove contamination subspace S (generic prompt structure)
2. Project onto task manifold M (curved semantic surface)

Reference: Huang et al., "Mitigating Overthinking via Manifold Steering"
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def estimate_contamination_subspace(
    concept_vectors: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate contamination subspace S from concept vectors via PCA.

    Contamination = shared definitional prompt structure across concepts
    (e.g., "What is X?" pattern that's not concept-specific)

    Args:
        concept_vectors: (n_concepts, hidden_dim) concept activations
        n_components: Number of PCA components to remove (default: n_concepts)

    Returns:
        U_S: (hidden_dim, n_components) contamination subspace basis
        explained_variance: Variance explained by each component
    """
    n_concepts, hidden_dim = concept_vectors.shape

    if n_components is None:
        n_components = n_concepts  # Phase 6 finding: optimal = n_concepts

    # Ensure we don't request more components than available
    max_components = min(n_concepts, hidden_dim)
    n_components = min(n_components, max_components)

    logger.info(f"Estimating contamination subspace with {n_components} components")
    logger.info(f"  Concept matrix shape: {concept_vectors.shape}")

    # Center the data (convert to float64 for numerical stability in linalg)
    concept_vectors_f64 = concept_vectors.astype(np.float64)
    mean_vector = np.mean(concept_vectors_f64, axis=0)
    centered = concept_vectors_f64 - mean_vector

    # Normalize rows to avoid numerical issues
    row_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered = centered / (row_norms + 1e-10)

    # SVD for PCA with fallback to scipy's more robust solver
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("NumPy SVD failed to converge, trying scipy's svd")
        from scipy.linalg import svd as scipy_svd
        try:
            U, S, Vt = scipy_svd(centered, full_matrices=False, lapack_driver='gesvd', check_finite=False)
        except:
            # Final fallback: just use the top eigenvector
            logger.warning("SVD failed, using eigendecomposition of covariance")
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by decreasing eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            Vt = eigenvectors[:, idx].T
            S = np.sqrt(np.maximum(eigenvalues[idx], 0))
            U = centered @ eigenvectors[:, idx] / (S + 1e-10)

    # U_S = first n_components of right singular vectors (column space basis)
    U_S = Vt[:n_components, :].T  # (hidden_dim, n_components)

    # Explained variance
    total_variance = np.sum(S ** 2)
    explained_variance = (S[:n_components] ** 2) / total_variance

    logger.info(f"  Explained variance by first {n_components} components: {explained_variance.sum():.1%}")
    logger.info(f"  Component variances: {explained_variance}")

    return U_S, explained_variance


def estimate_task_manifold(
    model,
    tokenizer,
    concept: str,
    steering_vector: np.ndarray,
    n_samples: int = 10,
    low_strength: float = 0.1,
    max_new_tokens: int = 30,
    device: str = "cuda"
) -> Tuple[np.ndarray, List[str]]:
    """
    Estimate task manifold M from low-strength steering generations.

    Task manifold = curved surface in activation space where coherent
    concept-steered generations live (as opposed to linear extrapolation).

    Args:
        model: Language model
        tokenizer: Tokenizer
        concept: Concept being steered
        steering_vector: Clean steering vector (after contamination removal)
        n_samples: Number of generations to collect
        low_strength: Low steering strength for manifold estimation (~0.1)
        max_new_tokens: Tokens per generation
        device: Device

    Returns:
        U_M: (hidden_dim, n_components_M) task manifold basis
        generations: Sample generations used for estimation
    """
    from .hooks import create_steering_hook

    logger.info(f"Estimating task manifold from {n_samples} generations at strength {low_strength}")

    # Concept-specific prompts for diverse manifold sampling
    prompts = [
        f"Tell me about {concept}",
        f"Describe {concept}",
        f"What is {concept}",
        f"Explain the concept of {concept}",
        f"The meaning of {concept} is",
        f"How does {concept} work",
        f"When I think about {concept}, I",
        f"{concept.capitalize()} means",
    ] * (n_samples // 8 + 1)
    prompts = prompts[:n_samples]

    # Collect activations from low-strength steered generations
    activations = []
    generations = []

    # Get layers
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model: {type(model.model)}")

    target_layer = layers[-1]  # Use final layer for manifold estimation

    # Hook to capture activations
    captured_activations = []

    def capture_hook(module, input, output):
        # output is tuple: (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        captured_activations.append(hidden[0, -1, :].detach().cpu().numpy())  # Last token
        return output

    for prompt in prompts:
        captured_activations.clear()

        # Create steering hook
        steering_hook = create_steering_hook(steering_vector, low_strength, device)

        # Register both hooks
        capture_handle = target_layer.register_forward_hook(capture_hook)
        steering_handle = target_layer.register_forward_hook(steering_hook)

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(text)

            # Average captured activations from this generation
            if captured_activations:
                mean_activation = np.mean(captured_activations, axis=0)
                activations.append(mean_activation)

        except Exception as e:
            logger.warning(f"Failed to generate sample {len(activations)}: {e}")

        finally:
            capture_handle.remove()
            steering_handle.remove()

    if len(activations) < 2:
        raise ValueError(f"Insufficient activations collected: {len(activations)} < 2")

    activations_array = np.array(activations)  # (n_samples, hidden_dim)
    logger.info(f"  Collected {len(activations)} activation samples")

    # PCA to find manifold subspace (convert to float64 for numerical stability)
    activations_f64 = activations_array.astype(np.float64)
    mean_activation = np.mean(activations_f64, axis=0)
    centered = activations_f64 - mean_activation

    # Normalize rows to avoid numerical issues
    row_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered = centered / (row_norms + 1e-10)

    # SVD for PCA with fallback to scipy's more robust solver
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("NumPy SVD failed to converge, trying scipy's svd")
        from scipy.linalg import svd as scipy_svd
        try:
            U, S, Vt = scipy_svd(centered, full_matrices=False, lapack_driver='gesvd', check_finite=False)
        except:
            # Final fallback: eigendecomposition of covariance
            logger.warning("SVD failed, using eigendecomposition")
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            Vt = eigenvectors[:, idx].T
            S = np.sqrt(np.maximum(eigenvalues[idx], 0))
            U = centered @ eigenvectors[:, idx] / (S + 1e-10)

    # Use components explaining 90% variance (or all if fewer than n_concepts)
    total_variance = np.sum(S ** 2)
    cumulative_variance = np.cumsum(S ** 2) / total_variance
    n_components_M = np.searchsorted(cumulative_variance, 0.90) + 1
    n_components_M = min(n_components_M, len(activations) - 1)

    U_M = Vt[:n_components_M, :].T  # (hidden_dim, n_components_M)

    logger.info(f"  Task manifold dimension: {n_components_M}")
    logger.info(f"  Explained variance: {cumulative_variance[n_components_M-1]:.1%}")

    return U_M, generations


def apply_dual_subspace_steering(
    steering_vector: np.ndarray,
    U_S: np.ndarray,
    U_M: np.ndarray,
    layer_idx: int,
    total_layers: int,
    max_norm_per_layer: float = 1.0,
    ema_alpha: float = 0.8,
    prev_vector: Optional[np.ndarray] = None,
    concept_preservation: float = 0.5
) -> np.ndarray:
    """
    Apply dual-subspace manifold steering with layer-wise dampening.

    Pipeline:
    1. Remove contamination: v_clean = v - U_S @ (U_S.T @ v)
    2. Project to manifold: v_mw = U_M @ (U_M.T @ v_clean)
    3. Blend with concept: v_blend = (1-alpha)*v_mw + alpha*v_clean
    4. Layer-wise gain: v_blend *= sqrt(1 - layer_depth)
    5. Norm clipping: v_blend = v_blend / max(||v_blend||, max_norm)
    6. EMA smoothing: v_final = ema_alpha * v_prev + (1 - ema_alpha) * v_blend

    Args:
        steering_vector: Raw steering vector
        U_S: Contamination subspace basis
        U_M: Task manifold basis
        layer_idx: Current layer index
        total_layers: Total number of layers
        max_norm_per_layer: Maximum norm per layer (prevents explosions)
        ema_alpha: EMA smoothing factor (0=no smoothing, 1=no update)
        prev_vector: Previous timestep's vector for EMA
        concept_preservation: How much of the original concept direction to preserve (0=pure manifold, 1=no manifold)

    Returns:
        v_final: Processed steering vector
    """
    # Step 1: Remove contamination subspace
    contamination_projection = U_S @ (U_S.T @ steering_vector)
    v_clean = steering_vector - contamination_projection

    # Step 2: Project onto task manifold
    v_mw = U_M @ (U_M.T @ v_clean)

    # Step 3: Blend manifold projection with original concept direction
    # This preserves steering while benefiting from manifold constraints
    v_blend = (1.0 - concept_preservation) * v_mw + concept_preservation * v_clean

    # Step 4: Layer-wise dampening (Huang et al.)
    # Decay with depth to prevent cascade failures
    layer_depth = layer_idx / total_layers
    depth_gain = np.sqrt(1.0 - layer_depth)  # sqrt decay
    v_blend = v_blend * depth_gain

    # Step 5: Norm clipping
    norm = np.linalg.norm(v_blend)
    if norm > max_norm_per_layer:
        v_blend = v_blend * (max_norm_per_layer / norm)

    # Step 6: EMA smoothing (if previous vector available)
    if prev_vector is not None:
        v_final = ema_alpha * prev_vector + (1.0 - ema_alpha) * v_blend
    else:
        v_final = v_blend

    return v_final


def create_dampened_steering_hook(
    steering_vector: np.ndarray,
    strength: float,
    layer_idx: int,
    total_layers: int,
    device: str = "cuda"
):
    """
    Create steering hook with layer-wise dampening only.

    No contamination removal or manifold projection - just applies
    sqrt(1 - depth) decay to prevent cascade failures in later layers.

    Args:
        steering_vector: Concept steering vector (normalized)
        strength: Steering strength multiplier
        layer_idx: Layer index for this hook
        total_layers: Total layers in model
        device: Device

    Returns:
        hook_fn: Forward hook function
    """
    # Apply layer-wise dampening
    layer_depth = layer_idx / total_layers
    depth_gain = np.sqrt(1.0 - layer_depth)  # sqrt decay: 1.0 at layer 0, 0 at final layer

    # Pre-compute dampened vector
    v_dampened = steering_vector * depth_gain
    v_tensor = torch.from_numpy(v_dampened).float().to(device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v_matched = v_tensor.to(dtype=hidden.dtype)

        # Normalized steering with layer dampening baked in
        hidden_norm = torch.norm(hidden, dim=-1, keepdim=True)
        steered = hidden + (strength * 0.1) * hidden_norm * v_matched

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook_fn


def create_manifold_steering_hook(
    steering_vector: np.ndarray,
    strength: float,
    U_S: np.ndarray,
    U_M: np.ndarray,
    layer_idx: int,
    total_layers: int,
    max_norm_per_layer: float = 1.0,
    ema_alpha: float = 0.0,  # Disabled by default for simplicity
    concept_preservation: float = 0.5,
    contamination_strength: float = 1.0,  # 0=no removal, 1=full removal
    device: str = "cuda"
):
    """
    Create forward hook for dual-subspace manifold steering.

    Args:
        steering_vector: Raw concept steering vector
        strength: Steering strength multiplier
        U_S: Contamination subspace basis
        U_M: Task manifold basis
        layer_idx: Layer index for this hook
        total_layers: Total layers in model
        max_norm_per_layer: Maximum norm per layer
        ema_alpha: EMA smoothing factor
        concept_preservation: Blend ratio (0=pure manifold, 1=pure concept)
        device: Device

    Returns:
        hook_fn: Forward hook function
    """
    # Pre-process steering vector
    v_processed = apply_dual_subspace_steering(
        steering_vector,
        U_S,
        U_M,
        layer_idx,
        total_layers,
        max_norm_per_layer,
        ema_alpha,
        prev_vector=None,  # Simplified: no temporal EMA for now
        concept_preservation=concept_preservation
    )

    # Convert to tensor
    v_tensor = torch.from_numpy(v_processed).float().to(device)

    def hook_fn(module, input, output):
        # output is tuple: (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output

        # Match tensor dtype to hidden states (e.g., float16 for model, float32 for operations)
        v_matched = v_tensor.to(dtype=hidden.dtype)

        # Normalized steering: strength=1.0 adds 10% of hidden norm in concept direction
        # This makes manifold steering comparable to other steering modes
        hidden_norm = torch.norm(hidden, dim=-1, keepdim=True)
        steered = hidden + (strength * 0.1) * hidden_norm * v_matched

        # Return modified output
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook_fn


class ManifoldSteerer:
    """
    High-level interface for dual-subspace manifold steering.

    Usage:
        steerer = ManifoldSteerer(model, tokenizer, device="cuda")

        # Estimate subspaces
        steerer.fit(concepts=["person", "change", "animal"])

        # Generate with manifold steering
        text = steerer.generate(
            prompt="Tell me about",
            concept="person",
            strength=1.0,
            max_new_tokens=50
        )
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Get layers
        if hasattr(model.model, 'language_model'):
            self.layers = model.model.language_model.layers
        elif hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        else:
            raise AttributeError(f"Cannot find layers in model: {type(model.model)}")

        self.total_layers = len(self.layers)

        # Subspaces
        self.U_S = None
        self.U_M = {}  # Per-concept task manifolds
        self.concept_vectors = {}

        logger.info(f"Initialized ManifoldSteerer with {self.total_layers} layers")

    def fit(
        self,
        concepts: List[str],
        n_contamination_components: Optional[int] = None,
        n_manifold_samples: int = 10
    ):
        """
        Estimate contamination subspace and task manifolds for concepts.

        Args:
            concepts: List of concept names
            n_contamination_components: Number of contamination components (default: n_concepts)
            n_manifold_samples: Samples per concept for manifold estimation
        """
        from .extraction import extract_concept_vector

        logger.info(f"Fitting dual-subspace manifold steering for {len(concepts)} concepts")

        # Extract concept vectors
        concept_matrix = []
        for concept in concepts:
            logger.info(f"  Extracting concept vector: {concept}")
            v = extract_concept_vector(self.model, self.tokenizer, concept, device=self.device)
            self.concept_vectors[concept] = v
            concept_matrix.append(v)

        concept_matrix = np.array(concept_matrix)

        # Estimate contamination subspace S
        self.U_S, _ = estimate_contamination_subspace(
            concept_matrix,
            n_components=n_contamination_components
        )

        # Estimate task manifold M for each concept
        for concept in concepts:
            logger.info(f"  Estimating task manifold for: {concept}")

            # First clean the concept vector
            v_raw = self.concept_vectors[concept]
            contamination_proj = self.U_S @ (self.U_S.T @ v_raw)
            v_clean = v_raw - contamination_proj
            v_clean = v_clean / (np.linalg.norm(v_clean) + 1e-8)

            # Estimate manifold from low-strength generations
            try:
                U_M, generations = estimate_task_manifold(
                    self.model,
                    self.tokenizer,
                    concept,
                    v_clean,
                    n_samples=n_manifold_samples,
                    device=self.device
                )
                self.U_M[concept] = U_M
                logger.info(f"    Task manifold dimension: {U_M.shape[1]}")
            except Exception as e:
                logger.error(f"    Failed to estimate task manifold for {concept}: {e}")
                # Fallback: use identity (no manifold projection)
                self.U_M[concept] = np.eye(v_clean.shape[0])

        logger.info("Dual-subspace fitting complete!")

    def generate(
        self,
        prompt: str,
        concept: Optional[str],
        strength: float,
        max_new_tokens: int = 50,
        max_norm_per_layer: float = 1.0,
        target_layers: Optional[List[int]] = None,
        concept_preservation: float = 0.5
    ) -> str:
        """
        Generate text with manifold steering.

        Args:
            prompt: Input prompt
            concept: Concept to steer toward (None for baseline generation)
            strength: Steering strength
            max_new_tokens: Max tokens to generate
            max_norm_per_layer: Max norm per layer
            target_layers: Specific layers to apply steering (default: last 10)
            concept_preservation: Blend ratio (0=pure manifold, 1=pure concept)

        Returns:
            Generated text
        """
        # Support baseline generation with concept=None or strength=0
        if concept is None or abs(strength) < 1e-6:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if concept not in self.concept_vectors:
            raise ValueError(f"Concept '{concept}' not fitted. Available: {list(self.concept_vectors.keys())}")

        if self.U_S is None:
            raise ValueError("Must call fit() before generate()")

        # Default: steer last 10 layers (semantic layers)
        if target_layers is None:
            target_layers = list(range(self.total_layers - 10, self.total_layers))

        # Get processed steering vector for each layer
        v_raw = self.concept_vectors[concept]
        U_M = self.U_M[concept]

        # Create hooks for target layers
        hooks = []
        for layer_idx in target_layers:
            hook_fn = create_manifold_steering_hook(
                v_raw,
                strength,
                self.U_S,
                U_M,
                layer_idx,
                self.total_layers,
                max_norm_per_layer,
                ema_alpha=0.0,
                concept_preservation=concept_preservation,
                device=self.device
            )

            handle = self.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(handle)

        try:
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text

        finally:
            # Clean up hooks
            for handle in hooks:
                handle.remove()
