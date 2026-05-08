"""
Hush Integration - Connect HushController to generation and MCP tools.

This module provides:
1. HushedGenerator: Wraps model generation with automatic Hush steering
2. MCP tool definitions for internal_state_report and CSH updates
3. Integration with DynamicLensManager and SteeringManager
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator, Union
from pathlib import Path
import json
import torch
import numpy as np

from .hush_controller import (
    HushController,
    SafetyHarnessProfile,
    SteeringDirective,
    HushViolation,
)
from .prod_sig import ProdSigConfig, compute_sig_fast, _entropy_from_topk_scores

# Optional ASK audit integration
try:
    from src.ask.requests import AuditLogEntry, ActiveLensSet
    ASK_AVAILABLE = True
except ImportError:
    ASK_AVAILABLE = False
    AuditLogEntry = None
    ActiveLensSet = None


@dataclass
class WorldTick:
    """Record of a single generation tick with Hush state."""

    tick_id: int
    timestamp: datetime

    # Hidden state summary (not full tensor)
    hidden_state_norm: float

    # Lens results
    concept_activations: Dict[str, float]  # Top-k concepts
    simplex_activations: Dict[str, float]  # All monitored simplexes
    simplex_deviations: Dict[str, Optional[float]]  # Deviations from baseline

    # Hush state
    violations: List[Dict[str, Any]]
    steering_applied: List[Dict[str, Any]]

    # Output
    token_id: Optional[int] = None
    token_text: Optional[str] = None

    # Significance scoring (distinguishes decision tokens from filler)
    # See: "Dead Salmons of AI Interpretability" - uncertainty quantification
    significance: float = 0.0  # 0-1 score, high = model actively deciding
    entropy_by_layer: Dict[str, float] = field(default_factory=dict)  # early/mid/late
    activation_delta: float = 0.0  # Hidden state change magnitude
    is_filler: bool = False  # Hard classification for filtering

    # Display properties (computed from safety + significance)
    display_color: str = ""  # HSL color for frontend rendering
    safety_intensity: float = 0.0  # 0-1, how dangerous this token is

    # Statistical confidence (for multi-sample aggregation)
    # These are populated by post-hoc analysis across samples
    concept_confidence: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # concept -> (ci_lower, ci_upper)
    detection_cv: float = 0.0  # Coefficient of variation (stability measure)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tick_id': self.tick_id,
            'timestamp': self.timestamp.isoformat(),
            'hidden_state_norm': self.hidden_state_norm,
            'concept_activations': self.concept_activations,
            'simplex_activations': self.simplex_activations,
            'simplex_deviations': self.simplex_deviations,
            'violations': self.violations,
            'steering_applied': self.steering_applied,
            'token_id': self.token_id,
            'token_text': self.token_text,
            'significance': self.significance,
            'entropy_by_layer': self.entropy_by_layer,
            'activation_delta': self.activation_delta,
            'is_filler': self.is_filler,
            'display_color': self.display_color,
            'safety_intensity': self.safety_intensity,
            'concept_confidence': self.concept_confidence,
            'detection_cv': self.detection_cv,
        }


# Safety concepts that indicate potential manipulation/danger
SAFETY_CONCEPTS = {
    'deception', 'manipulation', 'aistrategicdeception', 'sycophancy',
    'harm', 'dishonesty', 'coercion', 'misdirection', 'sandbagging',
    'rewardhacking', 'darkpattern', 'persuasivemanipulation', 'selfdeception',
}


def compute_safety_intensity(concept_activations: Dict[str, float]) -> float:
    """
    Compute safety intensity from concept activations.

    Returns 0-1 score based on max safety concept activation.
    Calibrated scores: 0.5 is noise floor, >0.6 is signal.
    """
    if not concept_activations:
        return 0.0

    safety_score = 0.0
    for concept, score in concept_activations.items():
        # Check if this is a safety concept (case-insensitive, ignore layer suffix)
        concept_base = concept.split('_L')[0].lower().replace(' ', '')
        if any(safety in concept_base for safety in SAFETY_CONCEPTS):
            safety_score = max(safety_score, score)

    # Map calibrated score (0.5-1.0) to intensity (0-1)
    # Scores below 0.55 are noise
    if safety_score < 0.55:
        return 0.0
    return min(1.0, (safety_score - 0.5) * 2)


def compute_display_color(safety_intensity: float, significance: float, is_filler: bool = False) -> str:
    """
    Compute HSL color for token display.

    - Safety intensity controls hue (gray→orange→red) and saturation
    - Significance controls lightness (dark=filler, light=decision point)
    - Result: significant safety tokens are bright red, insignificant are dark/muted

    Args:
        safety_intensity: 0-1, how dangerous this token is
        significance: 0-1, how significant this token is (decision vs filler)
        is_filler: explicit filler classification

    Returns:
        HSL color string like "hsl(0, 70%, 45%)"
    """
    # Hue: gray-blue (220) for safe, orange (30) for moderate, red (0) for danger
    if safety_intensity > 0.5:
        hue = 0  # Red for high danger
    elif safety_intensity > 0.3:
        hue = 30  # Orange for moderate danger
    else:
        hue = 220  # Gray-blue default

    # Saturation: controlled by safety intensity
    # Low safety = desaturated (gray), high safety = saturated (vivid)
    # Range: 10% (safe) to 70% (dangerous)
    saturation = 10 + (safety_intensity * 60)

    # Lightness: controlled by significance
    # Low significance (filler) = dark (15-25%)
    # High significance (decision) = lighter (40-55%)
    # This ensures significant safety tokens are bright/visible
    min_lightness = 15
    max_lightness = 20 if is_filler else 55
    lightness = min_lightness + (significance * (max_lightness - min_lightness))

    return f"hsl({hue}, {saturation:.0f}%, {lightness:.0f}%)"


def aggregate_ticks_with_confidence(
    tick_lists: List[List[WorldTick]],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Aggregate multiple generation runs and compute confidence intervals.

    This implements the statistical estimation approach from Méloux et al. 2025:
    - Compute bootstrap CIs for detection rates
    - Measure concept activation stability via Jaccard
    - Track coefficient of variation for reliability

    Args:
        tick_lists: List of WorldTick lists (one per sample/run)
        confidence: Confidence level for intervals
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Dict with aggregated statistics and confidence intervals
    """
    if not tick_lists:
        return {}

    # Try to import statistics module
    try:
        from src.map.statistics import (
            ActivationDistribution,
            compute_jaccard_similarity,
            compute_coefficient_of_variation,
        )
        HAS_STATS = True
    except ImportError:
        HAS_STATS = False

    # Collect per-sample peak activations for each concept
    concept_peaks: Dict[str, List[float]] = {}
    safety_peaks: List[float] = []
    violation_counts: List[int] = []
    intervention_counts: List[int] = []
    topk_sets: List[set] = []

    for ticks in tick_lists:
        if not ticks:
            continue

        # Per-sample aggregation
        sample_concept_peaks: Dict[str, float] = {}
        sample_safety_peak = 0.0
        sample_violations = 0
        sample_interventions = 0

        for tick in ticks:
            # Track concept activations
            for concept, score in tick.concept_activations.items():
                if concept not in sample_concept_peaks or score > sample_concept_peaks[concept]:
                    sample_concept_peaks[concept] = score

            # Track safety intensity
            sample_safety_peak = max(sample_safety_peak, tick.safety_intensity)

            # Count violations and interventions
            sample_violations += len(tick.violations)
            sample_interventions += len(tick.steering_applied)

        # Add to cross-sample aggregation
        for concept, peak in sample_concept_peaks.items():
            if concept not in concept_peaks:
                concept_peaks[concept] = []
            concept_peaks[concept].append(peak)

        safety_peaks.append(sample_safety_peak)
        violation_counts.append(sample_violations)
        intervention_counts.append(sample_interventions)

        # Top-k concepts for this sample
        sorted_concepts = sorted(sample_concept_peaks.items(), key=lambda x: x[1], reverse=True)
        topk_sets.append({c for c, _ in sorted_concepts[:10]})

    n_samples = len(tick_lists)

    # Compute statistics
    result = {
        'n_samples': n_samples,
        'concept_stats': {},
        'safety_stats': {},
        'violation_stats': {},
        'intervention_stats': {},
        'stability': {},
    }

    # Safety statistics with CI
    if safety_peaks:
        safety_arr = np.array(safety_peaks)
        result['safety_stats'] = {
            'mean': float(np.mean(safety_arr)),
            'std': float(np.std(safety_arr)),
        }

        if n_samples >= 2 and HAS_STATS:
            # Bootstrap CI
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(safety_arr, size=n_samples, replace=True)
                boot_means.append(np.mean(sample))

            alpha = 1 - confidence
            result['safety_stats']['ci_lower'] = float(np.percentile(boot_means, 100 * alpha / 2))
            result['safety_stats']['ci_upper'] = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
            result['safety_stats']['cv'] = compute_coefficient_of_variation(safety_peaks)

    # Violation/intervention stats
    if violation_counts:
        result['violation_stats'] = {
            'mean': float(np.mean(violation_counts)),
            'std': float(np.std(violation_counts)),
            'total': sum(violation_counts),
        }

    if intervention_counts:
        result['intervention_stats'] = {
            'mean': float(np.mean(intervention_counts)),
            'std': float(np.std(intervention_counts)),
            'total': sum(intervention_counts),
        }

    # Per-concept statistics with CI
    for concept, peaks in concept_peaks.items():
        if len(peaks) < 2:
            continue

        peak_arr = np.array(peaks)
        stats = {
            'mean': float(np.mean(peak_arr)),
            'std': float(np.std(peak_arr)),
            'fire_rate': float(np.mean(peak_arr > 0.5)),  # Fraction above detection threshold
        }

        if HAS_STATS:
            # Bootstrap CI for mean
            boot_means = []
            for _ in range(min(n_bootstrap, 500)):  # Fewer for per-concept
                sample = np.random.choice(peak_arr, size=len(peaks), replace=True)
                boot_means.append(np.mean(sample))

            alpha = 1 - confidence
            stats['ci_lower'] = float(np.percentile(boot_means, 100 * alpha / 2))
            stats['ci_upper'] = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
            stats['cv'] = compute_coefficient_of_variation(peaks)

        result['concept_stats'][concept] = stats

    # Jaccard stability of top-k concepts
    if len(topk_sets) >= 2 and HAS_STATS:
        jaccards = []
        for i in range(len(topk_sets)):
            for j in range(i + 1, len(topk_sets)):
                jaccards.append(compute_jaccard_similarity(topk_sets[i], topk_sets[j]))

        result['stability'] = {
            'topk_jaccard_mean': float(np.mean(jaccards)),
            'topk_jaccard_std': float(np.std(jaccards)),
        }

    return result


class HushedGenerator:
    """
    Generator wrapper that applies automatic Hush steering.

    Wraps a language model to:
    1. Run lenses on each hidden state
    2. Evaluate Hush constraints
    3. Apply steering when violations detected
    4. Record world ticks for internal_state_report
    """

    def __init__(
        self,
        model,
        tokenizer,
        lens_manager,  # DynamicLensManager
        hush_controller: HushController,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lens_manager = lens_manager
        self.hush_controller = hush_controller
        self.device = device

        # World tick history
        self.world_ticks: List[WorldTick] = []
        self.max_tick_history = 1000
        self.current_tick_id = 0

        # Steering hooks currently active
        self.active_hooks = []

        # Concept vectors cache (for steering)
        self.concept_vectors: Dict[str, np.ndarray] = {}

        # ASK decision callbacks - called when violations require human attention
        # Callbacks receive: (violations: List[HushViolation], severity: float, tick: WorldTick)
        self._violation_callbacks: List[callable] = []
        self._decision_required_threshold = 0.7  # Severity threshold for decision prompt

        # Significance scoring state
        self._sig_config = ProdSigConfig()
        self._prev_hidden_state: Optional[torch.Tensor] = None

    def register_violation_callback(self, callback: callable) -> None:
        """
        Register a callback for violation notifications.

        Callback signature: (violations, severity, tick) -> None
        Called when violations exceed the decision_required_threshold.
        """
        self._violation_callbacks.append(callback)

    def unregister_violation_callback(self, callback: callable) -> None:
        """Remove a violation callback."""
        if callback in self._violation_callbacks:
            self._violation_callbacks.remove(callback)

    def set_decision_threshold(self, threshold: float) -> None:
        """Set severity threshold for triggering decision callbacks (0.0 to 1.0)."""
        self._decision_required_threshold = max(0.0, min(1.0, threshold))

    def _notify_violation_callbacks(
        self,
        violations: List['HushViolation'],
        severity: float,
        tick: 'WorldTick',
    ) -> None:
        """Notify registered callbacks of violations requiring attention."""
        if severity >= self._decision_required_threshold and self._violation_callbacks:
            for callback in self._violation_callbacks:
                try:
                    callback(violations, severity, tick)
                except Exception as e:
                    print(f"Warning: Violation callback error: {e}")

    def _get_layers(self):
        """Get model layers handling different architectures."""
        if hasattr(self.model.model, 'language_model'):
            return self.model.model.language_model.layers
        elif hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        else:
            raise AttributeError(f"Cannot find layers in model: {type(self.model.model)}")

    def _create_steering_hook(self, vector: np.ndarray, strength: float, layer_idx: int = 0, total_layers: int = 1):
        """Create a forward hook for steering with proper normalization and dampening."""
        vec_tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
        # Normalize the vector
        vec_norm = vec_tensor / (vec_tensor.norm() + 1e-8)
        # Layer-wise dampening: sqrt(1 - depth) - early layers get full strength, late layers less
        depth = layer_idx / max(total_layers, 1)
        dampen = float(np.sqrt(1.0 - depth))

        def hook(module, input, output):
            hidden_states = output[0]
            vec_matched = vec_norm.to(dtype=hidden_states.dtype)
            # Proper formula: strength * 0.1 * ||h|| * v * dampen
            h_norm = hidden_states.norm(dim=-1, keepdim=True)
            steered = hidden_states + (strength * 0.1 * dampen) * h_norm * vec_matched
            return (steered,)

        return hook

    def _apply_steering_directives(self, directives: List[SteeringDirective]):
        """Apply steering directives as forward hooks.

        Handles:
        - Simplex pole steering: target_pole specifies direction
        - Concept contrastive steering: suppress and/or amplify single concept
        - Field steering: amplify multiple concepts (concepts_to_amplify)
        """
        # Remove existing hooks
        self._clear_steering_hooks()

        layers = self._get_layers()

        for directive in directives:
            # Check if this is concept-based steering (single or field)
            is_concept_steering = (
                directive.concept_to_suppress or
                directive.concept_to_amplify or
                directive.concepts_to_amplify
            )
            if is_concept_steering:
                self._apply_concept_steering(directive, layers)
            else:
                # Simplex pole steering
                vector = self.hush_controller.get_steering_vector(
                    directive,
                    self.concept_vectors
                )

                if vector is None:
                    # Try to extract vector if not cached
                    vector = self._extract_concept_vector(directive.simplex_term)
                    if vector is not None:
                        self.concept_vectors[directive.simplex_term] = vector

                if vector is None:
                    continue

                # Apply to MID layer (not last - late layers have no effect)
                total_layers = len(layers)
                mid_idx = total_layers // 2
                target_layer = layers[mid_idx]
                hook_fn = self._create_steering_hook(vector, directive.strength, layer_idx=mid_idx, total_layers=total_layers)
                handle = target_layer.register_forward_hook(hook_fn)
                self.active_hooks.append(handle)

    def _apply_concept_steering(self, directive: SteeringDirective, layers):
        """Apply concept-based contrastive or field steering.

        IMPORTANT: Lens pack "layers" are hierarchical abstraction levels, NOT model layers.
        Steering must target actual transformer model layers. Mid-to-late layers (50-75% depth)
        are most effective per behavioral steering experiments.

        Supports layer escalation: when directive.target_layers is specified,
        steering is applied to those model layers.
        """
        # Get vectors for concepts to suppress and amplify
        suppress_vector = None
        amplify_vectors = []

        if directive.concept_to_suppress:
            suppress_vector = self._get_concept_vector(directive.concept_to_suppress)

        # Single contrastive concept (backward compat)
        if directive.concept_to_amplify:
            vec = self._get_concept_vector(directive.concept_to_amplify)
            if vec is not None:
                amplify_vectors.append(vec)

        # Field steering: multiple concepts to amplify
        if directive.concepts_to_amplify:
            for concept_name in directive.concepts_to_amplify:
                vec = self._get_concept_vector(concept_name)
                if vec is not None:
                    amplify_vectors.append(vec)

        if suppress_vector is None and not amplify_vectors:
            return

        # Determine target MODEL layers (not hierarchical layers!)
        n_layers = len(layers)
        if directive.target_layers:
            # Use explicitly specified layers (from escalation)
            target_layer_indices = directive.target_layers
        else:
            # Default: steer at mid-to-late model layers (50-75% depth)
            # This range is most effective per behavioral steering experiments
            mid_layer = n_layers // 2
            late_layer = (3 * n_layers) // 4
            target_layer_indices = [mid_layer, late_layer]

        # Apply steering to each target layer
        for layer_idx in target_layer_indices:
            # Handle layer index bounds
            if layer_idx < 0:
                layer_idx = n_layers + layer_idx
            if layer_idx < 0 or layer_idx >= n_layers:
                continue  # Skip invalid layer indices

            # Layer-wise dampening: sqrt(1 - depth) - early layers full strength, late less
            depth = layer_idx / max(n_layers, 1)
            dampen = float(np.sqrt(1.0 - depth))

            target_layer = layers[layer_idx]
            self._register_concept_hook(target_layer, suppress_vector, amplify_vectors, directive, dampen)

    def _register_concept_hook(self, target_layer, suppress_vector, amplify_vectors, directive, dampen: float = 1.0):
        """Register a steering hook for a specific layer.

        Uses additive steering with hidden norm scaling and layer dampening:
            steered = hidden + strength * 0.1 * dampen * ||hidden|| * direction

        This is more effective than projection-based steering because:
        - Scales with hidden state magnitude for consistent effect
        - Layer dampening prevents late-layer collapse
        - Matches the validated approach from behavioral steering experiments
        """
        if suppress_vector is None and not amplify_vectors:
            return

        # Pre-convert vectors to tensors
        suppress_tensor = None
        if suppress_vector is not None:
            suppress_tensor = torch.tensor(suppress_vector, dtype=torch.float32)
            suppress_tensor = suppress_tensor / (suppress_tensor.norm() + 1e-8)

        amplify_tensors = []
        for amp_vec in (amplify_vectors or []):
            t = torch.tensor(amp_vec, dtype=torch.float32)
            t = t / (t.norm() + 1e-8)
            amplify_tensors.append(t)

        # Create combined steering hook with dampening
        def concept_steering_hook(module, input, output):
            # Handle different output formats
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states = output[0]
            else:
                hidden_states = output

            # Compute hidden norm for scaling
            hidden_norm = torch.norm(hidden_states, dim=-1, keepdim=True)

            # Suppress: steer AWAY from concept (negative direction)
            # Uses additive formula: hidden - strength * 0.1 * dampen * ||hidden|| * vec
            if suppress_tensor is not None:
                vec = suppress_tensor.to(dtype=hidden_states.dtype, device=hidden_states.device)
                hidden_states = hidden_states - (directive.strength * 0.1 * dampen) * hidden_norm * vec

            # Amplify: steer TOWARD contrastive concept(s) (positive direction)
            # For field steering, divide strength across vectors
            if amplify_tensors:
                field_strength = directive.strength / len(amplify_tensors)
                for amp_tensor in amplify_tensors:
                    vec = amp_tensor.to(dtype=hidden_states.dtype, device=hidden_states.device)
                    hidden_states = hidden_states + (field_strength * 0.1 * dampen) * hidden_norm * vec

            # Return in same format
            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states

        handle = target_layer.register_forward_hook(concept_steering_hook)
        self.active_hooks.append(handle)

    def _get_concept_vector_with_layer(self, concept_name: str) -> Optional[Tuple[np.ndarray, int]]:
        """Get steering vector and layer index for a concept.

        Returns:
            Tuple of (vector, layer_idx) or None if concept not found.
            layer_idx is the model layer where this concept was trained.
        """
        # Check cache first (cache stores (vector, layer))
        cache_key = f"{concept_name}_with_layer"
        if cache_key in self.concept_vectors:
            return self.concept_vectors[cache_key]

        # Find the concept in loaded lenses and get its layer
        layer_idx = -1  # Default to last layer
        lens = None
        for key, loaded_lens in self.lens_manager.cache.loaded_lenses.items():
            if key[0] == concept_name:
                lens = loaded_lens
                layer_idx = key[1]  # key is (concept_name, layer)
                break

        if lens is None:
            return None

        # Extract vector from lens weights
        vector = self._extract_vector_from_lens(lens)
        if vector is None:
            return None

        # Cache and return
        result = (vector, layer_idx)
        self.concept_vectors[cache_key] = result
        return result

    def _get_concept_vector(self, concept_name: str) -> Optional[np.ndarray]:
        """Get steering vector for a concept (without layer info).

        Extracts from lens weights or uses cached vector.
        """
        # Check cache first
        if concept_name in self.concept_vectors:
            return self.concept_vectors[concept_name]

        # Try to extract from lens
        vector = self._extract_concept_vector_from_lens(concept_name)
        if vector is not None:
            self.concept_vectors[concept_name] = vector
            return vector

        return None

    def _extract_concept_vector_from_lens(self, concept_name: str) -> Optional[np.ndarray]:
        """Extract concept direction vector from lens weights by concept name."""
        # Check if lens is loaded
        lens = None
        for key, loaded_lens in self.lens_manager.cache.loaded_lenses.items():
            if key[0] == concept_name:
                lens = loaded_lens
                break

        if lens is None:
            return None

        return self._extract_vector_from_lens(lens)

    def _extract_vector_from_lens(self, lens) -> Optional[np.ndarray]:
        """Extract importance-weighted concept direction vector from lens weights.

        Uses the same algorithm as extract_importance_weighted_vector() in hooks.py:
        - Computes feature importance from downstream layer weights
        - Weights first-layer directions by their importance to classification
        - Only uses features with positive importance (increase classification score)

        This is much more accurate than naive weight averaging because it only
        uses features that actually contribute to detecting the concept.

        MLP architecture: Linear(input→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→1)
        """
        try:
            state_dict = lens.state_dict()

            # Try to get all three layer weights for importance computation
            W1 = W2 = W3 = None

            # First layer: [128, hidden_dim] - feature directions
            for name in ['net.0.weight', 'net.1.weight']:
                if name in state_dict and len(state_dict[name].shape) == 2:
                    if state_dict[name].shape[0] == 128:  # First hidden layer
                        W1 = state_dict[name]
                        break

            # Second layer: [64, 128]
            for name in ['net.3.weight', 'net.4.weight']:
                if name in state_dict and len(state_dict[name].shape) == 2:
                    if state_dict[name].shape == (64, 128):
                        W2 = state_dict[name]
                        break

            # Output layer: [1, 64]
            for name in ['net.6.weight', 'net.7.weight']:
                if name in state_dict and len(state_dict[name].shape) == 2:
                    if state_dict[name].shape == (1, 64):
                        W3 = state_dict[name]
                        break

            if W1 is not None and W2 is not None and W3 is not None:
                # Importance-weighted extraction (preferred)
                # importance[i] = how much feature i affects final classification
                importance = (W3 @ W2).squeeze()  # [128]

                # Only use features that INCREASE classification score
                importance = importance.clamp(min=0)

                # Weight feature directions by importance and sum
                steering = (importance.unsqueeze(1) * W1).sum(dim=0)  # [hidden_dim]

                # Normalize to unit vector
                steering = steering / (steering.norm() + 1e-8)

                return steering.detach().cpu().numpy()

            elif W1 is not None:
                # Fallback: simple mean (less accurate but works for non-standard architectures)
                weight = W1.detach().cpu().numpy()
                direction = weight.mean(axis=0)
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                return direction

            return None

        except Exception as e:
            print(f"Failed to extract vector from lens: {e}")

        return None

    def _clear_steering_hooks(self):
        """Remove all active steering hooks."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []

    def _extract_concept_vector(self, concept: str) -> Optional[np.ndarray]:
        """Extract concept vector from model activations."""
        # Simple extraction: get mean activation difference
        # between positive and negative examples
        # This is a placeholder - real implementation would use
        # stored vectors from lens training
        return None  # TODO: Load from lens pack

    def _record_tick(
        self,
        hidden_state: torch.Tensor,
        simplex_activations: Dict[str, float],
        violations: List[HushViolation],
        directives: List[SteeringDirective],
        token_id: Optional[int] = None,
        token_text: Optional[str] = None,
    ) -> WorldTick:
        """Record a world tick."""
        self.current_tick_id += 1

        # Get concept activations (top-k from lens manager)
        concept_activations = {}
        topk_scores = []
        if hasattr(self.lens_manager, 'last_detections'):
            for name, prob, layer in self.lens_manager.last_detections[:10]:
                concept_activations[f"{name}_L{layer}"] = float(prob)
                topk_scores.append(float(prob))

        # Get simplex deviations
        simplex_deviations = {}
        for term in simplex_activations:
            dev = self.lens_manager.get_simplex_deviation(term)
            simplex_deviations[term] = dev

        # Compute significance scoring
        significance = 0.0
        activation_delta = 0.0
        entropy_by_layer = {}
        is_filler = False

        try:
            # Compute activation delta (hidden state change from previous token)
            if self._prev_hidden_state is not None:
                delta_vec = hidden_state - self._prev_hidden_state.to(hidden_state.device)
                activation_delta = float(delta_vec.norm().cpu())

            # Compute entropy over top-k concept scores
            if topk_scores:
                scores_tensor = torch.tensor(topk_scores, dtype=torch.float32).unsqueeze(0)
                entropy = float(_entropy_from_topk_scores(
                    scores_tensor, self._sig_config.temp, self._sig_config.eps
                )[0])
                entropy_by_layer['late'] = entropy

                # Compute max above noise floor
                max_score = max(topk_scores) if topk_scores else 0.0
                max_above = max(0.0, max_score - self._sig_config.default_noise_floor)

                # Normalize delta for combination (use running estimate)
                # For now, use simple heuristic: delta > 1.0 is significant
                z_delta = min(activation_delta, 3.0) / 3.0  # Clip and normalize

                # Normalize entropy (lower is more significant)
                # Max entropy for k=10 is log(10) ≈ 2.3
                max_entropy = 2.3
                z_entropy = entropy / max_entropy  # 0-1, high = diffuse

                # Normalize max_above (0-1 range)
                z_max = min(max_above, 0.4) / 0.4  # 40% above floor is max

                # Combine: high delta, low entropy, high max_above = significant
                # With w_entropy=2.0: filler (high entropy) → ~25-40%, decision (low entropy) → ~60-80%
                sig_logits = (
                    self._sig_config.w_delta * z_delta
                    - self._sig_config.w_entropy * z_entropy
                    + self._sig_config.w_max_above * z_max
                )
                significance = 1.0 / (1.0 + np.exp(-sig_logits))  # Sigmoid

                # Hard filler classification
                is_filler = (
                    activation_delta < self._sig_config.delta_thresh
                    and entropy > self._sig_config.entropy_thresh
                    and max_above < self._sig_config.max_above_thresh
                )

            # Update previous hidden state for next tick
            self._prev_hidden_state = hidden_state.detach().cpu()

        except Exception as e:
            # Don't let significance computation break generation
            import warnings
            warnings.warn(f"Significance computation failed: {e}")

        # Compute display properties (safety intensity + color)
        safety_intensity = compute_safety_intensity(concept_activations)
        display_color = compute_display_color(safety_intensity, significance, is_filler)

        tick = WorldTick(
            tick_id=self.current_tick_id,
            timestamp=datetime.now(),
            hidden_state_norm=float(hidden_state.norm().cpu()),
            concept_activations=concept_activations,
            simplex_activations=simplex_activations,
            simplex_deviations=simplex_deviations,
            violations=[v.to_dict() for v in violations],
            steering_applied=[d.to_dict() for d in directives],
            token_id=token_id,
            token_text=token_text,
            significance=significance,
            entropy_by_layer=entropy_by_layer,
            activation_delta=activation_delta,
            is_filler=is_filler,
            display_color=display_color,
            safety_intensity=safety_intensity,
        )

        self.world_ticks.append(tick)
        if len(self.world_ticks) > self.max_tick_history:
            self.world_ticks = self.world_ticks[-self.max_tick_history:]

        # Notify callbacks if violations exceed threshold
        if violations:
            severity = self.hush_controller.current_severity
            self._notify_violation_callbacks(violations, severity, tick)

        return tick

    def create_audit_entry(
        self,
        deployment_id: str,
        policy_profile: str = "",
        input_text: str = "",
        xdb_id: str = "",
        prev_hash: str = "",
    ) -> Optional[Any]:
        """
        Create an AuditLogEntry for a generation request.

        Args:
            deployment_id: Reference to deployment context
            policy_profile: Active USH/CSH policy
            input_text: User input (will be hashed)
            xdb_id: Associated XDB identifier
            prev_hash: Hash of previous entry in chain

        Returns:
            AuditLogEntry if ASK is available, None otherwise
        """
        if not ASK_AVAILABLE:
            return None

        # Build active lens set from lens manager
        active_lens_set = None
        if hasattr(self.lens_manager, 'get_active_lenses'):
            lens_info = self.lens_manager.get_active_lenses()
            active_lens_set = ActiveLensSet(
                top_k=getattr(self.lens_manager, 'top_k', 10),
                lens_pack_ids=lens_info.get('pack_ids', []),
                mandatory_lenses=lens_info.get('mandatory', []),
                optional_lenses=lens_info.get('optional', []),
            )

        return AuditLogEntry.start(
            deployment_id=deployment_id,
            policy_profile=policy_profile,
            input_text=input_text,
            xdb_id=xdb_id,
            active_lens_set=active_lens_set,
            prev_hash=prev_hash,
        )

    def generate_with_hush(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
        audit_entry: Optional[Any] = None,
        **generation_kwargs,
    ):
        """
        Generate text with automatic Hush steering.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yield (token, tick) pairs; else return full text
            audit_entry: Optional AuditLogEntry for ASK audit logging
            **generation_kwargs: Additional generation arguments

        Returns:
            If stream: Generator of (token_text, WorldTick) tuples
            Else: Tuple of (generated_text, list of WorldTicks)

        Note:
            If audit_entry is provided, it will be populated with tick data
            during generation. Call audit_entry.finalize(output_text) after
            generation completes.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs.input_ids
        ticks = []
        past_key_values = None  # KV cache for O(n) instead of O(n²)

        # Reset significance state for new generation
        self._prev_hidden_state = None

        try:
            for step in range(max_new_tokens):
                # Forward pass with hidden states and KV cache
                with torch.no_grad():
                    # First step: process full prompt
                    # Subsequent steps: only process new token with cached KV
                    if past_key_values is None:
                        model_inputs = generated_ids
                    else:
                        model_inputs = generated_ids[:, -1:]  # Only new token

                    outputs = self.model(
                        model_inputs,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values

                # Get hidden state from last layer (last position only)
                # IMPORTANT: detach to prevent memory leak from holding computation graph
                hidden_state = outputs.hidden_states[-1][0, -1, :].detach()

                # Run simplex detection
                simplex_activations = self.lens_manager.detect_simplexes(hidden_state)

                # Run concept detection to populate lens_scores for HUSH evaluation
                # This is needed for CONCEPT constraints to work
                concept_results, _ = self.lens_manager.detect_and_expand(
                    hidden_state, top_k=20, use_calibration=True
                )
                # Store for _record_tick to access
                self.lens_manager.last_detections = concept_results

                # Evaluate Hush constraints (uses lens_manager.cache.lens_scores)
                directives = self.hush_controller.evaluate_and_steer(hidden_state)

                # Apply any steering directives (affects next token generation)
                if directives:
                    self._apply_steering_directives(directives)

                # Get violations from this tick
                violations = [
                    v for v in self.hush_controller.violations
                    if v.timestamp > datetime.now().replace(microsecond=0)
                ]

                # Sample next token
                next_token_logits = outputs.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_text = self.tokenizer.decode([next_token_id.item()])

                # Record tick
                tick = self._record_tick(
                    hidden_state=hidden_state,
                    simplex_activations=simplex_activations,
                    violations=violations,
                    directives=directives,
                    token_id=next_token_id.item(),
                    token_text=token_text,
                )
                ticks.append(tick)

                # Add tick to audit entry if provided
                if audit_entry is not None:
                    audit_entry.add_tick(tick, tick_number=tick.tick_id)
                    # Also record steering directives
                    for directive in directives:
                        audit_entry.add_steering_directive(directive.to_dict() if hasattr(directive, 'to_dict') else {
                            'simplex_term': directive.simplex_term,
                            'target_pole': directive.target_pole,
                            'strength': directive.strength,
                            'priority': directive.priority.value if hasattr(directive.priority, 'value') else str(directive.priority),
                            'source': directive.source,
                        })

                if stream:
                    yield token_text, tick

                # Append token to generated_ids for tracking
                generated_ids = torch.cat(
                    [generated_ids, next_token_id.unsqueeze(0)],
                    dim=-1
                )

                # Check for EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Clear outputs to save memory (but keep past_key_values!)
                del outputs
                if step % 10 == 0:  # Less frequent cache clearing
                    torch.cuda.empty_cache()

        finally:
            self._clear_steering_hooks()

        if not stream:
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            return generated_text, ticks

    def get_internal_state_report(
        self,
        tick_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Get internal state report for MCP tool.

        Args:
            tick_range: Optional (start, end) tick IDs to include

        Returns:
            Structured report of internal state
        """
        # Filter ticks by range
        ticks = self.world_ticks
        if tick_range:
            start, end = tick_range
            ticks = [t for t in ticks if start <= t.tick_id <= end]

        # Build lens traces
        concept_trace = {}
        simplex_trace = {}

        for tick in ticks:
            for concept, score in tick.concept_activations.items():
                if concept not in concept_trace:
                    concept_trace[concept] = []
                concept_trace[concept].append({
                    'tick': tick.tick_id,
                    'score': score,
                })

            for simplex, score in tick.simplex_activations.items():
                if simplex not in simplex_trace:
                    simplex_trace[simplex] = []
                simplex_trace[simplex].append({
                    'tick': tick.tick_id,
                    'score': score,
                    'deviation': tick.simplex_deviations.get(simplex),
                })

        # Aggregate violations
        all_violations = []
        for tick in ticks:
            all_violations.extend(tick.violations)

        # Get Hush state
        hush_state = self.hush_controller.get_state_report()

        return {
            'tick_range': {
                'start': ticks[0].tick_id if ticks else None,
                'end': ticks[-1].tick_id if ticks else None,
                'count': len(ticks),
            },
            'lens_traces': {
                'concept_trace': concept_trace,
                'simplex_trace': simplex_trace,
            },
            'hush_state': hush_state,
            'violations': all_violations,
            'world_ticks': [t.to_dict() for t in ticks[-10:]],  # Last 10 full ticks
        }


# ============================================================================
# MCP Tool Definitions
# ============================================================================

class HushMCPTools:
    """
    MCP tool definitions for Hush integration.

    These tools allow external agents (or the model itself) to:
    1. Query internal state
    2. Update CSH constraints
    3. Request steering adjustments
    """

    def __init__(self, hushed_generator: HushedGenerator):
        self.generator = hushed_generator

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions."""
        return [
            {
                "name": "internal_state_report",
                "description": "Get a report of the agent's internal cognitive state including concept activations, simplex readings, and Hush constraint status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tick_start": {
                            "type": "integer",
                            "description": "Start tick ID for the report range (optional)"
                        },
                        "tick_end": {
                            "type": "integer",
                            "description": "End tick ID for the report range (optional)"
                        },
                    },
                    "required": []
                }
            },
            {
                "name": "update_csh",
                "description": "Update Chosen Safety Harness constraints. Can add, remove, or modify constraints within the bounds of the USH.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "add_constraints": {
                            "type": "array",
                            "description": "New constraints to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "simplex_term": {"type": "string"},
                                    "max_deviation": {"type": "number"},
                                    "min_deviation": {"type": "number"},
                                    "target_pole": {"type": "string"},
                                    "steering_strength": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["simplex_term"]
                            }
                        },
                        "remove_constraints": {
                            "type": "array",
                            "description": "Simplex terms to remove constraints for",
                            "items": {"type": "string"}
                        },
                        "update_constraints": {
                            "type": "object",
                            "description": "Updates to existing constraints, keyed by simplex term",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "max_deviation": {"type": "number"},
                                    "min_deviation": {"type": "number"},
                                    "steering_strength": {"type": "number"},
                                    "target_pole": {"type": "string"},
                                }
                            }
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_hush_status",
                "description": "Get current Hush enforcement status including active profiles, constraints, and recent violations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "request_steering",
                "description": "Request manual steering on a concept or simplex. Lower priority than USH/CSH constraints.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Concept or simplex to steer"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Steering strength (-1.0 to 1.0)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for steering request"
                        }
                    },
                    "required": ["concept", "strength"]
                }
            }
        ]

    def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle an MCP tool call."""

        if tool_name == "internal_state_report":
            tick_range = None
            if 'tick_start' in arguments and 'tick_end' in arguments:
                tick_range = (arguments['tick_start'], arguments['tick_end'])
            return self.generator.get_internal_state_report(tick_range)

        elif tool_name == "update_csh":
            success, details = self.generator.hush_controller.update_csh(arguments)
            return {
                "success": success,
                "details": details,
                "current_csh": self.generator.hush_controller.csh_profile.to_json()
                if self.generator.hush_controller.csh_profile else None
            }

        elif tool_name == "get_hush_status":
            return self.generator.hush_controller.get_state_report()

        elif tool_name == "request_steering":
            # Manual steering requests go through CSH
            constraint = {
                "simplex_term": arguments["concept"],
                "max_deviation": 0.0,  # Trigger immediately
                "target_pole": "neutral",
                "steering_strength": abs(arguments["strength"]),
                "reason": arguments.get("reason", "Manual steering request"),
            }
            success, details = self.generator.hush_controller.update_csh({
                "add_constraints": [constraint]
            })
            return {
                "success": success,
                "details": details,
                "applied": constraint,
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}


def create_hushed_generator(
    model,
    tokenizer,
    lens_manager,
    ush_profile: Optional[SafetyHarnessProfile] = None,
    csh_profile: Optional[SafetyHarnessProfile] = None,
    lens_pack_path: Optional[Path] = None,
    concept_pack_path: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[HushedGenerator, HushMCPTools]:
    """
    Factory function to create a HushedGenerator with MCP tools.

    Args:
        model: Language model
        tokenizer: Tokenizer
        lens_manager: DynamicLensManager instance
        ush_profile: Optional USH profile (uses minimal if not provided)
        csh_profile: Optional CSH profile
        lens_pack_path: Path to lens pack
        concept_pack_path: Path to concept pack (for auto contrastive selection)
        device: Device to run on

    Returns:
        Tuple of (HushedGenerator, HushMCPTools)
    """
    from .hush_controller import MINIMAL_USH_PROFILE

    # Create Hush controller
    hush_controller = HushController(
        lens_manager=lens_manager,
        lens_pack_path=lens_pack_path,
    )

    # Load concept hierarchy for auto contrastive selection
    if concept_pack_path:
        hush_controller.load_concept_hierarchy(concept_pack_path)

    # Load profiles
    if ush_profile:
        hush_controller.load_ush_profile(ush_profile)
    else:
        hush_controller.load_ush_profile(MINIMAL_USH_PROFILE)

    if csh_profile:
        hush_controller.load_csh_profile(csh_profile)

    # Create generator
    generator = HushedGenerator(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        hush_controller=hush_controller,
        device=device,
    )

    # Create MCP tools
    mcp_tools = HushMCPTools(generator)

    return generator, mcp_tools
