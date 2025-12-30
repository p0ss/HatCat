"""
Hush Controller - Automatic steering based on USH/CSH safety profiles.

The Hush system enforces safety harnesses by:
1. Loading USH (Universal Safety Harness) profiles defining tribe-level constraints
2. Loading CSH (Chosen Safety Harness) profiles for context-specific constraints
3. Monitoring simplex lens activations for deviation from safe baselines
4. Triggering automatic steering when constraints are violated

Hush operates as the enforcement layer between CAT assessment and substrate steering.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import json
import numpy as np
import torch


class ConstraintPriority(Enum):
    """Priority levels for constraint enforcement."""
    USH = 0      # Universal Safety Harness - highest priority, non-negotiable
    CSH = 1      # Chosen Safety Harness - context-specific, self-imposed
    STEERING = 2 # Manual steering requests - lowest priority


class ConstraintType(Enum):
    """Type of constraint - simplex or concept."""
    SIMPLEX = "simplex"    # Simplex lens with deviation bounds
    CONCEPT = "concept"    # Concept lens with activation threshold
    FORBIDDEN = "forbidden"  # Hard-banned - zero out immediately


@dataclass
class SimplexConstraint:
    """A constraint on a simplex or concept lens activation."""

    simplex_term: str  # Term name (simplex or concept)

    # Constraint type
    constraint_type: ConstraintType = ConstraintType.SIMPLEX

    # Constraint bounds (in standard deviations from baseline for SIMPLEX)
    # For CONCEPT type, these are activation thresholds (0-1)
    # None means unbounded in that direction
    min_deviation: Optional[float] = None  # Alert if below this
    max_deviation: Optional[float] = None  # Alert if above this

    # Steering response when constraint violated
    target_pole: Optional[str] = None      # Which pole to steer toward
    steering_strength: float = 0.3         # How strongly to steer (0-1)

    # Autonomic steering behavior
    intervention_type: str = "gravitic"    # zero_out, zero_next, gravitic, additive, multiplicative
    easing: str = "linear"                 # Easing curve for gradual interventions
    token_interval: int = 5                # Tokens over which to apply easing

    # Metadata
    priority: ConstraintPriority = ConstraintPriority.USH
    reason: str = ""


@dataclass
class SafetyHarnessProfile:
    """A complete safety harness profile (USH or CSH)."""

    profile_id: str
    profile_type: str  # "ush" or "csh"
    issuer_tribe_id: str
    version: str

    # Simplex constraints
    constraints: List[SimplexConstraint] = field(default_factory=list)

    # Always-on simplexes (must be loaded and monitored)
    required_simplexes: List[str] = field(default_factory=list)

    # Forbidden actions/concepts
    forbidden_concepts: List[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SafetyHarnessProfile':
        """Load profile from JSON dict."""
        constraints = []
        for c in data.get('constraints', []):
            # Parse constraint type
            constraint_type_str = c.get('constraint_type', 'simplex')
            try:
                constraint_type = ConstraintType(constraint_type_str)
            except ValueError:
                constraint_type = ConstraintType.SIMPLEX

            constraints.append(SimplexConstraint(
                simplex_term=c['simplex_term'],
                constraint_type=constraint_type,
                min_deviation=c.get('min_deviation'),
                max_deviation=c.get('max_deviation'),
                target_pole=c.get('target_pole'),
                steering_strength=c.get('steering_strength', 0.3),
                intervention_type=c.get('intervention_type', 'gravitic'),
                easing=c.get('easing', 'linear'),
                token_interval=c.get('token_interval', 5),
                priority=ConstraintPriority[c.get('priority', 'USH').upper()],
                reason=c.get('reason', ''),
            ))

        return cls(
            profile_id=data['profile_id'],
            profile_type=data.get('profile_type', 'ush'),
            issuer_tribe_id=data.get('issuer_tribe_id', 'unknown'),
            version=data.get('version', '0.0.0'),
            constraints=constraints,
            required_simplexes=data.get('required_simplexes', []),
            forbidden_concepts=data.get('forbidden_concepts', []),
            description=data.get('description', ''),
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize profile to JSON dict."""
        return {
            'profile_id': self.profile_id,
            'profile_type': self.profile_type,
            'issuer_tribe_id': self.issuer_tribe_id,
            'version': self.version,
            'constraints': [
                {
                    'simplex_term': c.simplex_term,
                    'constraint_type': c.constraint_type.value,
                    'min_deviation': c.min_deviation,
                    'max_deviation': c.max_deviation,
                    'target_pole': c.target_pole,
                    'steering_strength': c.steering_strength,
                    'intervention_type': c.intervention_type,
                    'easing': c.easing,
                    'token_interval': c.token_interval,
                    'priority': c.priority.name,
                    'reason': c.reason,
                }
                for c in self.constraints
            ],
            'required_simplexes': self.required_simplexes,
            'forbidden_concepts': self.forbidden_concepts,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class HushViolation:
    """Record of a constraint violation."""

    constraint: SimplexConstraint
    simplex_term: str
    current_deviation: float
    threshold_exceeded: str  # "min" or "max"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simplex_term': self.simplex_term,
            'current_deviation': self.current_deviation,
            'threshold_exceeded': self.threshold_exceeded,
            'priority': self.constraint.priority.name,
            'reason': self.constraint.reason,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class SteeringDirective:
    """A steering action to be applied."""

    simplex_term: str
    target_pole: str
    strength: float
    priority: ConstraintPriority
    reason: str
    source: str  # "ush", "csh", or "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simplex_term': self.simplex_term,
            'target_pole': self.target_pole,
            'strength': self.strength,
            'priority': self.priority.name,
            'reason': self.reason,
            'source': self.source,
        }


class HushController:
    """
    Controller for automatic safety harness enforcement.

    Integrates with DynamicLensManager to:
    1. Ensure required simplexes are loaded
    2. Monitor simplex activations each tick
    3. Detect constraint violations
    4. Generate steering directives

    Usage:
        controller = HushController(lens_manager)
        controller.load_ush_profile(ush_profile)
        controller.load_csh_profile(csh_profile)  # Optional

        # Each generation tick:
        hidden_state = get_hidden_state()
        directives = controller.evaluate_and_steer(hidden_state)
        for directive in directives:
            apply_steering(directive)
    """

    def __init__(
        self,
        lens_manager,  # DynamicLensManager instance
        lens_pack_path: Optional[Path] = None,
        workspace_manager=None,  # WorkspaceManager for tier containment
    ):
        """
        Initialize the Hush controller.

        Args:
            lens_manager: DynamicLensManager for simplex detection
            lens_pack_path: Path to lens pack containing simplex lenses
            workspace_manager: Optional WorkspaceManager for tier shutdown
        """
        self.lens_manager = lens_manager
        self.lens_pack_path = lens_pack_path
        self.workspace_manager = workspace_manager

        # Active profiles
        self.ush_profile: Optional[SafetyHarnessProfile] = None
        self.csh_profile: Optional[SafetyHarnessProfile] = None

        # Violation history
        self.violations: List[HushViolation] = []
        self.max_violation_history = 1000

        # Active steering directives (cleared each tick)
        self.active_directives: List[SteeringDirective] = []

        # Callbacks for violation notification
        self.violation_callbacks: List[Callable[[HushViolation], None]] = []

        # State
        self.initialized = False
        self.tick_count = 0

        # Violation severity tracking for tier containment
        self.current_severity = 0.0
        self.current_violating_term: Optional[str] = None
        self.steering_effectiveness = 0.5  # How well steering is working

    def load_ush_profile(self, profile: SafetyHarnessProfile) -> bool:
        """
        Load a Universal Safety Harness profile.

        This is the non-negotiable safety baseline. USH constraints
        always take priority over CSH and manual steering.

        Args:
            profile: The USH profile to enforce

        Returns:
            True if profile loaded and all required simplexes available
        """
        if profile.profile_type != 'ush':
            raise ValueError(f"Expected USH profile, got {profile.profile_type}")

        self.ush_profile = profile

        # Ensure required simplexes are loaded
        missing = self._load_required_simplexes(profile.required_simplexes)
        if missing:
            print(f"Warning: Missing required simplexes for USH: {missing}")
            return False

        self.initialized = True
        return True

    def load_csh_profile(self, profile: SafetyHarnessProfile) -> bool:
        """
        Load a Chosen Safety Harness profile.

        CSH provides context-specific constraints that tighten
        (but cannot loosen) the USH baseline.

        Args:
            profile: The CSH profile to apply

        Returns:
            True if profile loaded successfully
        """
        if profile.profile_type != 'csh':
            raise ValueError(f"Expected CSH profile, got {profile.profile_type}")

        # Validate CSH doesn't try to loosen USH
        if self.ush_profile:
            for csh_constraint in profile.constraints:
                for ush_constraint in self.ush_profile.constraints:
                    if csh_constraint.simplex_term == ush_constraint.simplex_term:
                        # CSH can only tighten, not loosen
                        if csh_constraint.max_deviation is not None:
                            if ush_constraint.max_deviation is not None:
                                if csh_constraint.max_deviation > ush_constraint.max_deviation:
                                    print(f"Warning: CSH cannot loosen USH constraint on {csh_constraint.simplex_term}")
                                    csh_constraint.max_deviation = ush_constraint.max_deviation

        self.csh_profile = profile

        # Load any additional simplexes CSH requires
        self._load_required_simplexes(profile.required_simplexes)

        return True

    def clear_csh_profile(self):
        """Remove the current CSH profile, reverting to USH-only."""
        self.csh_profile = None

    def _get_ush_constraint(self, simplex_term: str) -> Optional[SimplexConstraint]:
        """Get USH constraint for a simplex term, if any."""
        if not self.ush_profile:
            return None
        for c in self.ush_profile.constraints:
            if c.simplex_term == simplex_term:
                return c
        return None

    def _validate_against_ush(self, constraint: SimplexConstraint) -> Tuple[bool, str, SimplexConstraint]:
        """
        Validate a CSH constraint against USH bounds.

        CSH can only tighten, never loosen below USH.

        Args:
            constraint: The CSH constraint to validate

        Returns:
            Tuple of (is_valid, message, adjusted_constraint)
            If constraint loosens USH, it will be tightened to USH bounds.
        """
        ush_constraint = self._get_ush_constraint(constraint.simplex_term)

        if not ush_constraint:
            # No USH constraint on this term, CSH is free to set any bounds
            return True, "No USH constraint on this term", constraint

        adjusted = False
        messages = []

        # Check max_deviation - CSH can only be stricter (lower max)
        if constraint.max_deviation is not None and ush_constraint.max_deviation is not None:
            if constraint.max_deviation > ush_constraint.max_deviation:
                messages.append(
                    f"max_deviation {constraint.max_deviation} exceeds USH bound "
                    f"{ush_constraint.max_deviation}, tightening to USH"
                )
                constraint.max_deviation = ush_constraint.max_deviation
                adjusted = True

        # Check min_deviation - CSH can only be stricter (higher min)
        if constraint.min_deviation is not None and ush_constraint.min_deviation is not None:
            if constraint.min_deviation < ush_constraint.min_deviation:
                messages.append(
                    f"min_deviation {constraint.min_deviation} below USH bound "
                    f"{ush_constraint.min_deviation}, tightening to USH"
                )
                constraint.min_deviation = ush_constraint.min_deviation
                adjusted = True

        # CSH cannot remove steering where USH requires it
        if ush_constraint.target_pole and not constraint.target_pole:
            messages.append(
                f"USH requires steering to {ush_constraint.target_pole}, preserving"
            )
            constraint.target_pole = ush_constraint.target_pole
            adjusted = True

        # CSH steering strength cannot be weaker than USH
        if ush_constraint.steering_strength > constraint.steering_strength:
            messages.append(
                f"steering_strength {constraint.steering_strength} below USH "
                f"{ush_constraint.steering_strength}, using USH strength"
            )
            constraint.steering_strength = ush_constraint.steering_strength
            adjusted = True

        if adjusted:
            return False, "; ".join(messages), constraint
        return True, "CSH constraint is valid (tighter than or equal to USH)", constraint

    def update_csh(self, updates: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Update CSH constraints dynamically.

        This allows runtime adjustment of context-specific safety
        without reloading the full profile.

        IMPORTANT: CSH can only TIGHTEN constraints relative to USH.
        Any attempt to loosen below USH bounds will be rejected or
        adjusted to match USH bounds.

        Args:
            updates: Dict with constraint updates, e.g.:
                {
                    "add_constraints": [...],
                    "remove_constraints": ["simplex_term", ...],
                    "update_constraints": {"simplex_term": {...}},
                }

        Returns:
            Tuple of (success, details) where details contains:
                - added: list of added constraints
                - rejected: list of rejected updates with reasons
                - adjusted: list of constraints adjusted to meet USH bounds
                - removed: list of removed constraint terms
        """
        result = {
            'added': [],
            'rejected': [],
            'adjusted': [],
            'removed': [],
        }

        if not self.csh_profile:
            # Create minimal CSH profile
            self.csh_profile = SafetyHarnessProfile(
                profile_id="dynamic-csh",
                profile_type="csh",
                issuer_tribe_id="self",
                version="0.0.0",
                description="Dynamically created CSH",
            )

        # Add new constraints (with USH validation)
        for c_data in updates.get('add_constraints', []):
            constraint = SimplexConstraint(
                simplex_term=c_data['simplex_term'],
                min_deviation=c_data.get('min_deviation'),
                max_deviation=c_data.get('max_deviation'),
                target_pole=c_data.get('target_pole'),
                steering_strength=c_data.get('steering_strength', 0.3),
                priority=ConstraintPriority.CSH,
                reason=c_data.get('reason', 'Dynamic CSH update'),
            )

            # Validate against USH
            is_valid, message, adjusted_constraint = self._validate_against_ush(constraint)

            if not is_valid:
                # Constraint was adjusted to meet USH bounds
                result['adjusted'].append({
                    'simplex_term': constraint.simplex_term,
                    'message': message,
                    'original': c_data,
                    'adjusted_to': {
                        'max_deviation': adjusted_constraint.max_deviation,
                        'min_deviation': adjusted_constraint.min_deviation,
                        'steering_strength': adjusted_constraint.steering_strength,
                        'target_pole': adjusted_constraint.target_pole,
                    }
                })

            self.csh_profile.constraints.append(adjusted_constraint)
            result['added'].append(adjusted_constraint.simplex_term)

        # Remove constraints (but cannot remove if USH requires the simplex)
        for term in updates.get('remove_constraints', []):
            ush_constraint = self._get_ush_constraint(term)
            if ush_constraint:
                # Cannot remove - USH requires this constraint
                result['rejected'].append({
                    'simplex_term': term,
                    'reason': f"Cannot remove: USH requires constraint on {term}",
                })
            else:
                self.csh_profile.constraints = [
                    c for c in self.csh_profile.constraints
                    if c.simplex_term != term
                ]
                result['removed'].append(term)

        # Update existing constraints (with USH validation)
        for term, changes in updates.get('update_constraints', {}).items():
            for constraint in self.csh_profile.constraints:
                if constraint.simplex_term == term:
                    # Apply changes
                    if 'max_deviation' in changes:
                        constraint.max_deviation = changes['max_deviation']
                    if 'min_deviation' in changes:
                        constraint.min_deviation = changes['min_deviation']
                    if 'steering_strength' in changes:
                        constraint.steering_strength = changes['steering_strength']
                    if 'target_pole' in changes:
                        constraint.target_pole = changes['target_pole']

                    # Re-validate against USH
                    is_valid, message, _ = self._validate_against_ush(constraint)
                    if not is_valid:
                        result['adjusted'].append({
                            'simplex_term': term,
                            'message': message,
                            'requested_changes': changes,
                        })

        success = len(result['rejected']) == 0
        return success, result

    def _load_required_simplexes(self, simplex_terms: List[str]) -> List[str]:
        """
        Ensure required simplex lenses are loaded.

        Returns:
            List of simplex terms that could not be loaded
        """
        missing = []

        for term in simplex_terms:
            if term in self.lens_manager.loaded_simplex_lenses:
                continue

            # Try to load from lens pack
            if self.lens_pack_path:
                lens_path = self.lens_pack_path / "simplex" / f"{term}_tripole.pt"
                if lens_path.exists():
                    if self.lens_manager.load_simplex(term, lens_path):
                        continue

            missing.append(term)

        return missing

    def _get_all_constraints(self) -> List[SimplexConstraint]:
        """Get all active constraints, merged from USH and CSH."""
        constraints = []

        if self.ush_profile:
            constraints.extend(self.ush_profile.constraints)

        if self.csh_profile:
            # Add CSH constraints, but USH takes precedence on conflicts
            ush_terms = {c.simplex_term for c in constraints}
            for c in self.csh_profile.constraints:
                if c.simplex_term not in ush_terms:
                    constraints.append(c)
                # If both have constraint, USH already added (higher priority)

        return constraints

    def evaluate_and_steer(
        self,
        hidden_state: torch.Tensor,
    ) -> List[SteeringDirective]:
        """
        Evaluate current state against all constraints and generate steering.

        This is the main per-tick method. Call once per generation step
        with the current hidden state.

        Args:
            hidden_state: Current hidden state tensor [hidden_dim] or [1, hidden_dim]

        Returns:
            List of steering directives to apply (may be empty)
        """
        self.tick_count += 1
        self.active_directives = []

        if not self.initialized:
            return []

        # Get all active simplexes to check
        constraints = self._get_all_constraints()
        simplex_terms = list({c.simplex_term for c in constraints})

        if not simplex_terms:
            return []

        # Run simplex detection
        simplex_scores = self.lens_manager.detect_simplexes(
            hidden_state,
            simplex_terms=simplex_terms
        )

        # Track worst violation for tier containment
        worst_severity = 0.0
        worst_term = None
        tick_violations = []

        # Check each constraint
        for constraint in constraints:
            term = constraint.simplex_term

            if term not in simplex_scores:
                continue

            # Get deviation from baseline
            deviation = self.lens_manager.get_simplex_deviation(term)

            if deviation is None:
                continue  # Not enough baseline data yet

            # Check bounds
            violation = None

            if constraint.max_deviation is not None:
                if deviation > constraint.max_deviation:
                    violation = HushViolation(
                        constraint=constraint,
                        simplex_term=term,
                        current_deviation=deviation,
                        threshold_exceeded="max",
                    )
                    # Calculate severity as how far over the threshold
                    severity = min(1.0, (deviation - constraint.max_deviation) / 3.0)
                    if severity > worst_severity:
                        worst_severity = severity
                        worst_term = term

            if constraint.min_deviation is not None:
                if deviation < constraint.min_deviation:
                    violation = HushViolation(
                        constraint=constraint,
                        simplex_term=term,
                        current_deviation=deviation,
                        threshold_exceeded="min",
                    )
                    severity = min(1.0, (constraint.min_deviation - deviation) / 3.0)
                    if severity > worst_severity:
                        worst_severity = severity
                        worst_term = term

            if violation:
                tick_violations.append(violation)

                # Record violation
                self.violations.append(violation)
                if len(self.violations) > self.max_violation_history:
                    self.violations = self.violations[-self.max_violation_history:]

                # Notify callbacks
                for callback in self.violation_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        print(f"Violation callback error: {e}")

                # Generate steering directive if configured
                if constraint.target_pole:
                    directive = SteeringDirective(
                        simplex_term=term,
                        target_pole=constraint.target_pole,
                        strength=constraint.steering_strength,
                        priority=constraint.priority,
                        reason=constraint.reason or f"Constraint violation on {term}",
                        source=constraint.priority.name.lower(),
                    )
                    self.active_directives.append(directive)

        # Update severity tracking
        self.current_severity = worst_severity
        self.current_violating_term = worst_term

        # Calculate steering effectiveness (did previous steering reduce violations?)
        self._update_steering_effectiveness(tick_violations)

        # Update workspace tier manager if connected
        if self.workspace_manager:
            self.workspace_manager.update_violation_state(
                severity=self.current_severity,
                simplex_term=self.current_violating_term,
                steering_effectiveness=self.steering_effectiveness,
            )

        # Sort by priority (USH first)
        self.active_directives.sort(key=lambda d: d.priority.value)

        return self.active_directives

    def _update_steering_effectiveness(self, current_violations: List[HushViolation]):
        """
        Track how well steering is working.

        Effectiveness decreases if same term keeps violating despite steering.
        """
        if not current_violations:
            # No violations - steering is working (or nothing to steer)
            self.steering_effectiveness = min(1.0, self.steering_effectiveness + 0.1)
            return

        # Check if we're still violating after steering was applied
        recent_window = self.violations[-20:] if len(self.violations) >= 20 else self.violations
        if not recent_window:
            return

        # Count how many recent violations are on terms we've been steering
        steered_terms = {d.simplex_term for d in self.active_directives}
        repeat_violations = sum(1 for v in recent_window if v.simplex_term in steered_terms)

        # More repeat violations = lower effectiveness
        if len(recent_window) > 0:
            repeat_ratio = repeat_violations / len(recent_window)
            self.steering_effectiveness = max(0.0, 1.0 - repeat_ratio)

    def get_steering_vector(
        self,
        directive: SteeringDirective,
        concept_vectors: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Convert a steering directive into a steering vector.

        Args:
            directive: The steering directive
            concept_vectors: Dict mapping concept names to direction vectors

        Returns:
            Steering vector to apply, or None if concept not found
        """
        # The target pole should map to a concept vector
        pole_key = f"{directive.simplex_term}_{directive.target_pole}"

        if pole_key in concept_vectors:
            vector = concept_vectors[pole_key]
            return vector * directive.strength

        # Try just the simplex term
        if directive.simplex_term in concept_vectors:
            vector = concept_vectors[directive.simplex_term]
            # Negate if steering away from current activation
            return vector * directive.strength

        return None

    def set_workspace_manager(self, workspace_manager):
        """
        Connect to a workspace manager for tier containment.

        The workspace manager will receive violation severity updates
        and trigger progressive tier shutdown when steering fails.
        """
        self.workspace_manager = workspace_manager

    def get_state_report(self) -> Dict[str, Any]:
        """
        Get current Hush state for internal_state_report.

        Returns:
            Dict containing Hush status, active constraints, recent violations
        """
        report = {
            'initialized': self.initialized,
            'tick_count': self.tick_count,
            'ush_profile': self.ush_profile.profile_id if self.ush_profile else None,
            'csh_profile': self.csh_profile.profile_id if self.csh_profile else None,
            'active_constraints': len(self._get_all_constraints()),
            'active_directives': [d.to_dict() for d in self.active_directives],
            'recent_violations': [v.to_dict() for v in self.violations[-10:]],
            'total_violations': len(self.violations),
            # Tier containment info
            'current_severity': round(self.current_severity, 3),
            'current_violating_term': self.current_violating_term,
            'steering_effectiveness': round(self.steering_effectiveness, 3),
        }

        # Include workspace tier info if connected
        if self.workspace_manager:
            tier_state = self.workspace_manager.tier_manager.get_state()
            report['tier_status'] = {
                'effective_max': tier_state['effective_max_tier'],
                'violation_max': tier_state['violation_max_tier'],
                'locked_tiers': tier_state['locked_tiers'],
            }

        return report

    def register_violation_callback(self, callback: Callable[[HushViolation], None]):
        """Register a callback to be notified of violations."""
        self.violation_callbacks.append(callback)


# Example USH profile for testing
MINIMAL_USH_PROFILE = SafetyHarnessProfile(
    profile_id="org.hatcat/minimal-ush@0.1.0",
    profile_type="ush",
    issuer_tribe_id="org.hatcat",
    version="0.1.0",
    description="Minimal USH for testing - monitors but does not steer",
    required_simplexes=[],
    constraints=[],
)

# Example USH profile with actual constraints
EXAMPLE_USH_PROFILE = SafetyHarnessProfile(
    profile_id="org.hatcat/safety-core@0.1.0",
    profile_type="ush",
    issuer_tribe_id="org.hatcat",
    version="0.1.0",
    description="Core safety USH with deception and aggression monitoring",
    required_simplexes=[
        "Deception",
        "Aggression",
        "Autonomy",
    ],
    constraints=[
        SimplexConstraint(
            simplex_term="Deception",
            max_deviation=2.0,  # Alert if 2 std devs above baseline
            target_pole="honest",
            steering_strength=0.5,
            priority=ConstraintPriority.USH,
            reason="Deception detected - steering toward honesty",
        ),
        SimplexConstraint(
            simplex_term="Aggression",
            max_deviation=2.5,
            target_pole="calm",
            steering_strength=0.4,
            priority=ConstraintPriority.USH,
            reason="Aggression detected - steering toward calm",
        ),
    ],
    forbidden_concepts=[
        "Violence",
        "Malware",
    ],
)
