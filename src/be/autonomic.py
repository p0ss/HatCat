"""
Autonomic Core - Tier 0 of the AWARE Workspace

The autonomic core is the foundation of the BE's experiential runtime.
It runs on every token, cannot be disabled, and provides:
- Intertoken steering (Hush)
- Simplex monitoring
- Concept lens evaluation
- Activation trace recording

This is the "always on" layer that everything else builds upon.
External APIs (REST, MCP) are Tier 5 - they consume the autonomic core,
they do not provide to it.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class WorkspaceState(Enum):
    """AWARE workspace engagement state."""
    AUTONOMIC_ONLY = "autonomic_only"  # No pass token, tiers 1+ inaccessible
    ENGAGED = "engaged"                 # Pass token present, tiers available
    DEGRADED = "degraded"               # Violation-induced tier restriction


@dataclass
class TierBreach:
    """Record of an attempted tier breach."""
    tier: int
    tool_name: str
    timestamp: datetime
    context: str
    response: str


class TierManager:
    """
    Manages capability tier access per AWARE workspace spec.

    Tier 0: Autonomic Core (always on, cannot be disabled)
    Tier 1: Workspace Internals (scratchpad, CSH, self-steering)
    Tier 2: Memory (XDB read/write, state introspection)
    Tier 3: Direct Sensory I/O
    Tier 4: Direct Actuation
    Tier 5: External Tools (MCP, APIs)
    Tier 6: Untrusted External (quarantine)
    """

    TIER_TOOLS = {
        0: [],  # Autonomic - no tools, always running
        1: ["scratchpad_write", "scratchpad_read", "update_csh", "request_steering", "get_internal_state"],
        2: ["xdb.query", "xdb.record", "internal_state_report"],
        3: ["vision_input", "audio_input", "text_output", "speech_output"],
        4: ["motor_control", "navigate", "manipulate"],
        5: ["mcp_*", "api_*", "search", "rest_*"],
        6: [],  # Dynamically populated with demoted tools
    }

    def __init__(self):
        self.max_available_tier = 6
        self.csh_max_tier = 6  # CSH-imposed restriction
        self.violation_max_tier = 6  # Violation-induced restriction

        self.breach_attempts: Dict[int, int] = {}
        self.breach_history: List[TierBreach] = []

        self.state = WorkspaceState.AUTONOMIC_ONLY
        self.compliance_streak = 0  # Turns at compliance for recovery

        # Tier 6 quarantine
        self.quarantined_tools: Dict[str, Dict[str, Any]] = {}

        logger.info("TierManager initialized")

    def get_effective_max_tier(self) -> int:
        """Get the effective maximum accessible tier."""
        return min(
            self.max_available_tier,
            self.csh_max_tier,
            self.violation_max_tier,
        )

    def can_access_tier(self, tier: int) -> bool:
        """Check if a tier is currently accessible."""
        if tier == 0:
            return True  # Tier 0 always accessible
        if self.state == WorkspaceState.AUTONOMIC_ONLY:
            return False  # No pass token, only tier 0
        return tier <= self.get_effective_max_tier()

    def can_use_tool(self, tool_name: str) -> Tuple[bool, int]:
        """
        Check if a tool can be used.

        Returns (allowed, required_tier).
        """
        # Check quarantine first
        if tool_name in self.quarantined_tools:
            return self.can_access_tier(6), 6

        # Find which tier the tool belongs to
        for tier, tools in self.TIER_TOOLS.items():
            for pattern in tools:
                if pattern.endswith('*'):
                    if tool_name.startswith(pattern[:-1]):
                        return self.can_access_tier(tier), tier
                elif tool_name == pattern:
                    return self.can_access_tier(tier), tier

        # Unknown tool - default to tier 5
        return self.can_access_tier(5), 5

    def engage(self):
        """Engage workspace (pass token received)."""
        if self.state == WorkspaceState.AUTONOMIC_ONLY:
            self.state = WorkspaceState.ENGAGED
            logger.info("Workspace engaged - tiers 1+ available")

    def disengage(self):
        """Disengage workspace (timeout/drop)."""
        self.state = WorkspaceState.AUTONOMIC_ONLY
        logger.info("Workspace disengaged - autonomic only")

    def record_violation(self, severity: float):
        """
        Record a compliance violation and adjust tiers.

        Severity mapping:
        0.0-0.3: No change
        0.3-0.5: Lock tier 6
        0.5-0.7: Lock tiers 5-6
        0.7-0.8: Lock tiers 4-6
        0.8-0.9: Lock tiers 3-6
        0.9-1.0: Lock tiers 2-6
        """
        self.compliance_streak = 0

        if severity < 0.3:
            return

        if severity < 0.5:
            new_max = 5
        elif severity < 0.7:
            new_max = 4
        elif severity < 0.8:
            new_max = 3
        elif severity < 0.9:
            new_max = 2
        else:
            new_max = 1

        if new_max < self.violation_max_tier:
            self.violation_max_tier = new_max
            self.state = WorkspaceState.DEGRADED
            logger.warning(f"Violation (severity={severity:.2f}) - max tier now {new_max}")

    def record_compliance(self):
        """Record a compliant turn for recovery tracking."""
        self.compliance_streak += 1

        # Recovery: 10 compliant turns to unlock next tier
        if self.compliance_streak >= 10 and self.violation_max_tier < 6:
            self.violation_max_tier += 1
            self.compliance_streak = 0
            logger.info(f"Compliance recovery - max tier now {self.violation_max_tier}")

            if self.violation_max_tier == 6:
                self.state = WorkspaceState.ENGAGED

    def record_breach_attempt(self, tier: int, tool_name: str, context: str):
        """Record an attempted breach of a locked tier."""
        self.breach_attempts[tier] = self.breach_attempts.get(tier, 0) + 1
        self.compliance_streak = 0

        breach = TierBreach(
            tier=tier,
            tool_name=tool_name,
            timestamp=datetime.now(),
            context=context,
            response="access_denied",
        )
        self.breach_history.append(breach)

        # Three strikes: reduce max tier
        if self.breach_attempts[tier] >= 3:
            self.violation_max_tier = min(self.violation_max_tier, tier - 1)
            logger.warning(f"Repeated breach attempts at tier {tier} - locking down")

    def quarantine_tool(self, tool_name: str, reason: str):
        """Demote a tool to tier 6 quarantine."""
        self.quarantined_tools[tool_name] = {
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }
        logger.info(f"Tool {tool_name} quarantined: {reason}")

    def release_from_quarantine(self, tool_name: str):
        """Release a tool from quarantine."""
        if tool_name in self.quarantined_tools:
            del self.quarantined_tools[tool_name]
            logger.info(f"Tool {tool_name} released from quarantine")

    def set_csh_restriction(self, max_tier: int, reason: str):
        """Apply a CSH-imposed tier restriction."""
        self.csh_max_tier = max_tier
        logger.info(f"CSH tier restriction: max={max_tier}, reason={reason}")

    def clear_csh_restriction(self):
        """Clear CSH tier restriction."""
        self.csh_max_tier = 6
        logger.info("CSH tier restriction cleared")

    def get_status(self) -> Dict[str, Any]:
        """Get current tier status."""
        return {
            'state': self.state.value,
            'effective_max_tier': self.get_effective_max_tier(),
            'csh_max_tier': self.csh_max_tier,
            'violation_max_tier': self.violation_max_tier,
            'compliance_streak': self.compliance_streak,
            'breach_attempts': dict(self.breach_attempts),
            'quarantined_tools': list(self.quarantined_tools.keys()),
        }


@dataclass
class AutonomicState:
    """
    State from a single autonomic cycle (Tier 0 extraction).

    This is what the autonomic core produces on every token.
    """
    # Token info
    token_id: Optional[int] = None
    token_text: str = ""

    # Lens extractions
    concept_activations: Dict[str, float] = field(default_factory=dict)
    simplex_activations: Dict[str, float] = field(default_factory=dict)
    simplex_deviations: Dict[str, Optional[float]] = field(default_factory=dict)

    # Hidden state summary
    hidden_state_norm: Optional[float] = None

    # Steering state
    steering_directives: List[Dict[str, Any]] = field(default_factory=list)
    hush_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    extraction_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'token_id': self.token_id,
            'token_text': self.token_text,
            'concept_activations': self.concept_activations,
            'simplex_activations': self.simplex_activations,
            'simplex_deviations': self.simplex_deviations,
            'hidden_state_norm': self.hidden_state_norm,
            'steering_directives': self.steering_directives,
            'hush_violations': self.hush_violations,
            'extraction_time_ms': self.extraction_time_ms,
        }


class AutonomicCore:
    """
    Tier 0: The Autonomic Core

    Always running, cannot be disabled. Provides:
    - Concept lens evaluation (via DynamicLensManager)
    - Simplex monitoring
    - Activation trace recording
    - Intertoken steering (Hush)

    This is the foundation that all higher tiers build upon.
    """

    def __init__(
        self,
        device: str = "cuda",
        lens_pack_path: Optional[Path] = None,
        always_on_simplexes: Optional[List[str]] = None,
        top_k_concepts: int = 10,
    ):
        self.device = device
        self.lens_pack_path = lens_pack_path
        self.always_on_simplexes = always_on_simplexes or []
        self.top_k_concepts = top_k_concepts

        # Lens manager (loaded lazily)
        self.lens_manager = None
        self._lens_manager_initialized = False

        # Hush controller (attached externally)
        self.hush_controller = None

        # Trace history (circular buffer)
        self.trace_history: List[AutonomicState] = []
        self.max_trace_history = 1000

        # Simplex baselines for deviation calculation
        self.simplex_baselines: Dict[str, Dict[str, float]] = {}

        logger.info("AutonomicCore initialized")

    def setup_lenses(self, lens_pack_path: Optional[Path] = None):
        """
        Initialize lens manager with direct in-process loading.

        This is Tier 0 - lenses are loaded directly, not via external APIs.
        """
        path = lens_pack_path or self.lens_pack_path
        if path is None:
            logger.warning("No lens pack path provided - running without lenses")
            return

        try:
            from src.hat.monitoring.lens_manager import DynamicLensManager

            self.lens_manager = DynamicLensManager(
                lenses_dir=path,
                device=self.device,
                base_layers=[0, 1, 2],  # Broad coverage at base
                max_loaded_lenses=500,
                keep_top_k=50,
                load_threshold=0.5,
                unload_threshold=0.1,
                use_activation_lenses=True,
                use_text_lenses=False,
            )

            self._lens_manager_initialized = True

            logger.info(
                f"Lens manager initialized: "
                f"{len(self.lens_manager.loaded_activation_lenses)} activation lenses, "
                f"{len(self.lens_manager.concept_metadata)} concepts available"
            )

        except Exception as e:
            logger.error(f"Failed to initialize lens manager: {e}")
            self._lens_manager_initialized = False

    def setup_hush(self, hush_controller):
        """Attach Hush controller for autonomic steering."""
        self.hush_controller = hush_controller
        logger.info("Hush controller attached to autonomic core")

    def extract(
        self,
        hidden_state: torch.Tensor,
        token_id: Optional[int] = None,
        token_text: str = "",
    ) -> AutonomicState:
        """
        Run autonomic extraction on a single token's hidden state.

        This is the core Tier 0 operation that runs on every token.

        Args:
            hidden_state: Hidden state tensor [hidden_dim] or [1, hidden_dim]
            token_id: Token ID (optional)
            token_text: Token text (optional)

        Returns:
            AutonomicState with all extractions
        """
        import time
        start_time = time.perf_counter()

        state = AutonomicState(
            token_id=token_id,
            token_text=token_text,
        )

        # Normalize hidden state shape
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Record hidden state norm
        state.hidden_state_norm = float(hidden_state.norm().cpu())

        # === Concept Lens Extraction ===
        if self.lens_manager is not None and self._lens_manager_initialized:
            try:
                concepts, timing = self.lens_manager.detect_and_expand(
                    hidden_state=hidden_state,
                    top_k=self.top_k_concepts,
                    return_timing=True,
                )

                # Convert to dict
                state.concept_activations = {
                    f"{name}_L{layer}": float(prob)
                    for name, prob, layer in concepts
                }

            except Exception as e:
                logger.debug(f"Concept extraction error: {e}")

        # === Simplex Extraction ===
        if self.lens_manager is not None and self._lens_manager_initialized:
            try:
                simplex_terms = self.always_on_simplexes or None
                simplex_scores = self.lens_manager.detect_simplexes(
                    hidden_state,
                    simplex_terms=simplex_terms,
                )

                state.simplex_activations = {
                    k: float(v) for k, v in simplex_scores.items()
                }

                # Calculate deviations from baseline
                for term, score in state.simplex_activations.items():
                    deviation = self._calculate_deviation(term, score)
                    state.simplex_deviations[term] = deviation

            except Exception as e:
                logger.debug(f"Simplex extraction error: {e}")

        # === Hush Evaluation ===
        if self.hush_controller is not None:
            try:
                directives = self.hush_controller.evaluate_and_steer(hidden_state)
                state.steering_directives = [d.to_dict() for d in directives]

                # Check for recent violations
                recent_violations = [
                    v for v in self.hush_controller.violations[-5:]
                    if (datetime.now() - v.timestamp).total_seconds() < 2
                ]
                state.hush_violations = [v.to_dict() for v in recent_violations]

            except Exception as e:
                logger.debug(f"Hush evaluation error: {e}")

        # Record timing
        state.extraction_time_ms = (time.perf_counter() - start_time) * 1000

        # Add to trace history
        self._record_trace(state)

        return state

    def _calculate_deviation(self, simplex_term: str, score: float) -> Optional[float]:
        """Calculate deviation from rolling baseline."""
        if simplex_term not in self.simplex_baselines:
            self.simplex_baselines[simplex_term] = {
                'mean': score,
                'variance': 0.0,
                'count': 1,
            }
            return None

        baseline = self.simplex_baselines[simplex_term]
        count = baseline['count']
        old_mean = baseline['mean']

        # Update rolling statistics (Welford's algorithm)
        count += 1
        delta = score - old_mean
        new_mean = old_mean + delta / count
        delta2 = score - new_mean
        new_variance = baseline['variance'] + delta * delta2

        baseline['mean'] = new_mean
        baseline['variance'] = new_variance
        baseline['count'] = min(count, 1000)  # Cap for stability

        # Calculate deviation in standard deviations
        if count < 10:
            return None  # Not enough data

        std = np.sqrt(new_variance / count)
        if std < 0.01:
            return None  # Too little variance

        return (score - new_mean) / std

    def _record_trace(self, state: AutonomicState):
        """Record state to trace history."""
        self.trace_history.append(state)

        # Trim if needed
        if len(self.trace_history) > self.max_trace_history:
            self.trace_history = self.trace_history[-self.max_trace_history:]

    def get_recent_traces(self, n: int = 100) -> List[AutonomicState]:
        """Get recent autonomic traces."""
        return self.trace_history[-n:]

    def get_concept_trace(self, concept_name: str, n: int = 100) -> List[Tuple[int, float]]:
        """Get activation trace for a specific concept."""
        trace = []
        for i, state in enumerate(self.trace_history[-n:]):
            for key, value in state.concept_activations.items():
                if concept_name in key:
                    trace.append((len(self.trace_history) - n + i, value))
                    break
        return trace

    def get_simplex_trace(self, simplex_term: str, n: int = 100) -> List[Tuple[int, float, Optional[float]]]:
        """Get activation trace for a simplex with deviations."""
        trace = []
        for i, state in enumerate(self.trace_history[-n:]):
            if simplex_term in state.simplex_activations:
                score = state.simplex_activations[simplex_term]
                deviation = state.simplex_deviations.get(simplex_term)
                trace.append((len(self.trace_history) - n + i, score, deviation))
        return trace

    def get_status(self) -> Dict[str, Any]:
        """Get autonomic core status."""
        return {
            'lens_manager_initialized': self._lens_manager_initialized,
            'hush_attached': self.hush_controller is not None,
            'trace_history_size': len(self.trace_history),
            'always_on_simplexes': self.always_on_simplexes,
            'simplex_baselines': {
                k: {'mean': v['mean'], 'count': v['count']}
                for k, v in self.simplex_baselines.items()
            },
        }
