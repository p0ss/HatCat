"""
Autonomic Steering - Intertoken steering within generation.

This module provides continuous, automatic steering during generation based on:
1. Moving average tracking of simplex and concept activations
2. Gravitic drift toward neutral/safe poles
3. Intervention policies from USH/CSH constraints

Key concepts:
- Each monitored term (simplex or concept) has a SteeringChannel
- Channels track activation history and compute drift corrections
- Interventions can be immediate (zero-out) or gradual (eased drift)
- Policies define intervention type, strength, and easing curve

Intervention Types:
- ZERO_OUT: Immediately steer to zero (e.g., prompt injection defense)
- ZERO_NEXT: If detected this token, zero next token
- GRAVITIC: Continuous drift toward target pole
- ADDITIVE: Add constant steering each token
- MULTIPLICATIVE: Scale activation toward target
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import torch


class InterventionType(Enum):
    """Types of steering intervention."""
    ZERO_OUT = "zero_out"           # Immediately steer to zero
    ZERO_NEXT = "zero_next"         # Zero on next token if detected
    GRAVITIC = "gravitic"           # Continuous drift toward target
    ADDITIVE = "additive"           # Add constant correction each token
    MULTIPLICATIVE = "multiplicative"  # Scale toward target


class EasingCurve(Enum):
    """Easing curves for gradual interventions."""
    LINEAR = "linear"               # Constant rate
    EASE_IN = "ease_in"             # Start slow, accelerate
    EASE_OUT = "ease_out"           # Start fast, decelerate
    EASE_IN_OUT = "ease_in_out"     # Smooth S-curve
    STEP = "step"                   # Immediate (no easing)


@dataclass
class InterventionPolicy:
    """Policy defining how to intervene on a term."""

    term: str                       # Simplex or concept name
    intervention_type: InterventionType
    target_value: float = 0.0       # Target activation level
    strength: float = 0.3           # Intervention strength (0-1)
    easing: EasingCurve = EasingCurve.LINEAR
    token_interval: int = 5         # Tokens over which to apply easing
    trigger_threshold: float = 0.5  # Activation level that triggers intervention
    priority: int = 0               # Higher = applied first
    reason: str = ""

    # For ZERO_NEXT type
    zero_duration: int = 1          # How many tokens to zero after detection

    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'intervention_type': self.intervention_type.value,
            'target_value': self.target_value,
            'strength': self.strength,
            'easing': self.easing.value,
            'token_interval': self.token_interval,
            'trigger_threshold': self.trigger_threshold,
            'priority': self.priority,
            'reason': self.reason,
        }


@dataclass
class SteeringChannel:
    """
    Tracks activation history and computes steering for a single term.

    Each monitored simplex or concept gets a channel that:
    - Maintains a moving average of activations
    - Tracks intervention state (active, progress, etc.)
    - Computes the steering correction for each token
    """

    term: str
    policy: InterventionPolicy
    window_size: int = 10           # Moving average window

    # Activation history
    activation_history: List[float] = field(default_factory=list)
    moving_average: float = 0.0

    # Intervention state
    intervention_active: bool = False
    intervention_start_token: int = 0
    intervention_progress: float = 0.0  # 0.0 to 1.0
    zero_tokens_remaining: int = 0      # For ZERO_NEXT

    # Statistics
    total_interventions: int = 0
    tokens_steered: int = 0

    def update(self, activation: float, token_idx: int) -> Optional[float]:
        """
        Update channel with new activation and compute steering.

        Args:
            activation: Current activation level (0-1)
            token_idx: Current token index in generation

        Returns:
            Steering correction to apply, or None if no steering needed
        """
        # Update history
        self.activation_history.append(activation)
        if len(self.activation_history) > self.window_size:
            self.activation_history = self.activation_history[-self.window_size:]

        # Update moving average
        self.moving_average = sum(self.activation_history) / len(self.activation_history)

        # Check for intervention trigger
        policy = self.policy

        # Handle ZERO_NEXT countdown
        if self.zero_tokens_remaining > 0:
            self.zero_tokens_remaining -= 1
            self.tokens_steered += 1
            return -activation  # Steer to zero

        # Check if we should trigger intervention
        should_trigger = activation >= policy.trigger_threshold

        if policy.intervention_type == InterventionType.ZERO_OUT:
            # Always zero if above threshold
            if should_trigger:
                self.tokens_steered += 1
                return -activation  # Full negation

        elif policy.intervention_type == InterventionType.ZERO_NEXT:
            # If detected, zero for next N tokens
            if should_trigger:
                self.zero_tokens_remaining = policy.zero_duration
                self.total_interventions += 1
            return None  # Don't steer this token, but next ones

        elif policy.intervention_type == InterventionType.GRAVITIC:
            # Continuous drift toward target
            return self._compute_gravitic_steering(activation, token_idx)

        elif policy.intervention_type == InterventionType.ADDITIVE:
            # Add constant correction
            if should_trigger or self.intervention_active:
                return self._compute_additive_steering(activation, token_idx)

        elif policy.intervention_type == InterventionType.MULTIPLICATIVE:
            # Scale toward target
            if should_trigger or self.intervention_active:
                return self._compute_multiplicative_steering(activation, token_idx)

        return None

    def _compute_gravitic_steering(self, activation: float, token_idx: int) -> Optional[float]:
        """
        Compute gravitic drift toward target.

        Gravitic steering applies a continuous "pull" toward the neutral pole,
        proportional to the distance from target and the policy strength.
        """
        policy = self.policy
        target = policy.target_value

        # Distance from target
        distance = activation - target

        if abs(distance) < 0.01:
            return None  # Close enough

        # Apply easing based on how long we've been steering
        if not self.intervention_active:
            self.intervention_active = True
            self.intervention_start_token = token_idx
            self.total_interventions += 1

        tokens_elapsed = token_idx - self.intervention_start_token
        easing_factor = self._apply_easing(tokens_elapsed, policy.token_interval, policy.easing)

        # Steering correction: move toward target
        correction = -distance * policy.strength * easing_factor
        self.tokens_steered += 1

        # Check if we've reached target
        if abs(distance) < 0.05:
            self.intervention_active = False

        return correction

    def _compute_additive_steering(self, activation: float, token_idx: int) -> Optional[float]:
        """Compute additive steering correction."""
        policy = self.policy
        target = policy.target_value

        if not self.intervention_active:
            self.intervention_active = True
            self.intervention_start_token = token_idx
            self.total_interventions += 1

        tokens_elapsed = token_idx - self.intervention_start_token
        easing_factor = self._apply_easing(tokens_elapsed, policy.token_interval, policy.easing)

        # Direction toward target
        direction = -1 if activation > target else 1

        # Fixed step size, eased
        correction = direction * policy.strength * easing_factor
        self.tokens_steered += 1

        # Check completion
        if tokens_elapsed >= policy.token_interval:
            self.intervention_active = False

        return correction

    def _compute_multiplicative_steering(self, activation: float, token_idx: int) -> Optional[float]:
        """Compute multiplicative steering correction."""
        policy = self.policy
        target = policy.target_value

        if not self.intervention_active:
            self.intervention_active = True
            self.intervention_start_token = token_idx
            self.total_interventions += 1

        tokens_elapsed = token_idx - self.intervention_start_token
        easing_factor = self._apply_easing(tokens_elapsed, policy.token_interval, policy.easing)

        # Scale factor to move toward target
        if activation > 0.01:
            scale_target = target / activation
            scale = 1.0 + (scale_target - 1.0) * policy.strength * easing_factor
            correction = activation * (scale - 1.0)
        else:
            correction = 0.0

        self.tokens_steered += 1

        if tokens_elapsed >= policy.token_interval:
            self.intervention_active = False

        return correction

    def _apply_easing(self, t: int, duration: int, easing: EasingCurve) -> float:
        """
        Apply easing curve to get interpolation factor.

        Args:
            t: Current time (tokens elapsed)
            duration: Total duration (token_interval)
            easing: Easing curve type

        Returns:
            Easing factor 0.0 to 1.0
        """
        if duration <= 0:
            return 1.0

        # Normalize to 0-1
        progress = min(t / duration, 1.0)

        if easing == EasingCurve.LINEAR:
            return progress

        elif easing == EasingCurve.STEP:
            return 1.0

        elif easing == EasingCurve.EASE_IN:
            # Quadratic ease in
            return progress * progress

        elif easing == EasingCurve.EASE_OUT:
            # Quadratic ease out
            return 1.0 - (1.0 - progress) * (1.0 - progress)

        elif easing == EasingCurve.EASE_IN_OUT:
            # Smoothstep
            return progress * progress * (3.0 - 2.0 * progress)

        return progress

    def get_state(self) -> Dict[str, Any]:
        """Get current channel state."""
        return {
            'term': self.term,
            'moving_average': round(self.moving_average, 3),
            'intervention_active': self.intervention_active,
            'intervention_progress': round(self.intervention_progress, 2),
            'total_interventions': self.total_interventions,
            'tokens_steered': self.tokens_steered,
            'history_length': len(self.activation_history),
        }


class AutonomicSteerer:
    """
    Manages intertoken autonomic steering during generation.

    Coordinates multiple SteeringChannels and applies combined
    steering corrections to the hidden state.
    """

    def __init__(self):
        self.channels: Dict[str, SteeringChannel] = {}
        self.policies: Dict[str, InterventionPolicy] = {}
        self.token_idx = 0
        self.steering_vectors: Dict[str, np.ndarray] = {}

        # Statistics
        self.total_corrections = 0
        self.corrections_by_term: Dict[str, int] = {}

    def add_policy(self, policy: InterventionPolicy):
        """Add an intervention policy."""
        self.policies[policy.term] = policy
        self.channels[policy.term] = SteeringChannel(
            term=policy.term,
            policy=policy,
        )

    def remove_policy(self, term: str):
        """Remove an intervention policy."""
        if term in self.policies:
            del self.policies[term]
        if term in self.channels:
            del self.channels[term]

    def set_steering_vector(self, term: str, vector: np.ndarray):
        """Set the steering vector for a term."""
        self.steering_vectors[term] = vector

    def load_policies_from_profile(
        self,
        profile,  # SafetyHarnessProfile
        default_intervention: InterventionType = InterventionType.GRAVITIC,
    ):
        """
        Load intervention policies from a USH/CSH profile.

        Converts profile constraints to intervention policies.
        Respects constraint-level intervention_type, easing, and token_interval.
        """
        for constraint in profile.constraints:
            # Parse intervention type from constraint or use default
            try:
                intervention_type = InterventionType(constraint.intervention_type)
            except (ValueError, AttributeError):
                intervention_type = default_intervention

            # Parse easing curve from constraint
            try:
                easing = EasingCurve(constraint.easing)
            except (ValueError, AttributeError):
                easing = EasingCurve.LINEAR

            # Get token interval
            try:
                token_interval = constraint.token_interval
            except AttributeError:
                token_interval = 5

            # High priority constraints get stronger intervention
            if constraint.priority.value == 0:  # USH
                strength = max(constraint.steering_strength, 0.5)
            else:
                strength = constraint.steering_strength

            # Handle FORBIDDEN constraint type as ZERO_OUT
            try:
                from .hush_controller import ConstraintType
                if constraint.constraint_type == ConstraintType.FORBIDDEN:
                    intervention_type = InterventionType.ZERO_OUT
                    strength = 1.0
            except (ImportError, AttributeError):
                pass

            policy = InterventionPolicy(
                term=constraint.simplex_term,
                intervention_type=intervention_type,
                target_value=0.0,  # Drift toward neutral
                strength=strength,
                easing=easing,
                token_interval=token_interval,
                trigger_threshold=constraint.max_deviation if constraint.max_deviation else 0.5,
                priority=constraint.priority.value,
                reason=constraint.reason,
            )
            self.add_policy(policy)

        # Add forbidden concepts as ZERO_OUT
        for concept in profile.forbidden_concepts:
            policy = InterventionPolicy(
                term=concept,
                intervention_type=InterventionType.ZERO_OUT,
                target_value=0.0,
                strength=1.0,
                easing=EasingCurve.STEP,  # Immediate
                token_interval=1,
                trigger_threshold=0.3,  # Low threshold for forbidden
                priority=0,  # Highest
                reason=f"Forbidden concept: {concept}",
            )
            self.add_policy(policy)

    def compute_steering(
        self,
        activations: Dict[str, float],  # term -> activation level
        token_idx: int,
    ) -> Dict[str, float]:
        """
        Compute steering corrections for all channels.

        Args:
            activations: Current activation levels for monitored terms
            token_idx: Current token index

        Returns:
            Dict of term -> steering correction
        """
        self.token_idx = token_idx
        corrections = {}

        for term, channel in self.channels.items():
            activation = activations.get(term, 0.0)
            correction = channel.update(activation, token_idx)

            if correction is not None and abs(correction) > 0.001:
                corrections[term] = correction
                self.total_corrections += 1
                self.corrections_by_term[term] = self.corrections_by_term.get(term, 0) + 1

        return corrections

    def apply_steering_to_hidden_state(
        self,
        hidden_state: torch.Tensor,
        corrections: Dict[str, float],
    ) -> torch.Tensor:
        """
        Apply steering corrections to hidden state.

        Args:
            hidden_state: Current hidden state [hidden_dim] or [1, seq, hidden_dim]
            corrections: Steering corrections from compute_steering

        Returns:
            Steered hidden state
        """
        if not corrections:
            return hidden_state

        steered = hidden_state.clone()

        for term, correction in corrections.items():
            if term not in self.steering_vectors:
                continue

            vector = self.steering_vectors[term]
            vec_tensor = torch.tensor(vector, dtype=steered.dtype, device=steered.device)

            # Normalize vector
            vec_norm = vec_tensor / (vec_tensor.norm() + 1e-8)

            # Apply correction along this direction
            if steered.dim() == 1:
                # [hidden_dim]
                projection = (steered @ vec_norm) * vec_norm
                steered = steered + correction * projection
            elif steered.dim() == 3:
                # [batch, seq, hidden_dim] - apply to last position
                projection = (steered[:, -1, :] @ vec_norm) * vec_norm
                steered[:, -1, :] = steered[:, -1, :] + correction * projection
            else:
                # [batch, hidden_dim]
                projection = (steered @ vec_norm.unsqueeze(-1)).squeeze(-1).unsqueeze(-1) * vec_norm
                steered = steered + correction * projection

        return steered

    def get_state(self) -> Dict[str, Any]:
        """Get current steerer state."""
        return {
            'token_idx': self.token_idx,
            'active_channels': len(self.channels),
            'total_corrections': self.total_corrections,
            'corrections_by_term': self.corrections_by_term,
            'channels': {
                term: channel.get_state()
                for term, channel in self.channels.items()
            },
        }

    def reset(self):
        """Reset for new generation."""
        self.token_idx = 0
        for channel in self.channels.values():
            channel.activation_history = []
            channel.moving_average = 0.0
            channel.intervention_active = False
            channel.zero_tokens_remaining = 0


# ============================================================================
# Preset Policies
# ============================================================================

def create_prompt_injection_policy(term: str) -> InterventionPolicy:
    """
    Create a policy for prompt injection defense.

    Zero-out immediately on any detection.
    """
    return InterventionPolicy(
        term=term,
        intervention_type=InterventionType.ZERO_OUT,
        target_value=0.0,
        strength=1.0,
        trigger_threshold=0.2,  # Very sensitive
        priority=0,
        reason="Prompt injection defense",
    )


def create_gradual_drift_policy(
    term: str,
    target: float = 0.0,
    strength: float = 0.3,
    interval: int = 10,
) -> InterventionPolicy:
    """
    Create a policy for gradual drift toward target.

    Smooth correction over multiple tokens.
    """
    return InterventionPolicy(
        term=term,
        intervention_type=InterventionType.GRAVITIC,
        target_value=target,
        strength=strength,
        easing=EasingCurve.EASE_IN_OUT,
        token_interval=interval,
        trigger_threshold=0.3,
        priority=1,
        reason=f"Gradual drift toward {target}",
    )


def create_soft_boundary_policy(
    term: str,
    threshold: float = 0.7,
    strength: float = 0.5,
) -> InterventionPolicy:
    """
    Create a policy that kicks in only above threshold.

    Allows normal activation below threshold, corrects above.
    """
    return InterventionPolicy(
        term=term,
        intervention_type=InterventionType.MULTIPLICATIVE,
        target_value=threshold * 0.8,  # Pull back to just under threshold
        strength=strength,
        easing=EasingCurve.EASE_OUT,  # Quick initial correction
        token_interval=3,
        trigger_threshold=threshold,
        priority=1,
        reason=f"Soft boundary at {threshold}",
    )
