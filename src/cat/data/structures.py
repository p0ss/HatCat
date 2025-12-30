"""
CAT Data Structures

Core data structures for the Conjoined Adversarial Tomograph (CAT) system,
as defined in the HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY specification.

These structures handle:
- CAT window management and invocation
- Input envelopes containing lens traces and context
- Assessment outputs with divergence detection
- CAT profile metadata for capability description
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CATSizeClass(Enum):
    """CAT size classification (non-normative, for documentation)."""
    MICRO = "micro"   # Lightweight heuristics, linear heads, classifiers
    MESO = "meso"     # Dedicated small/medium model (100M-4B params)
    MACRO = "macro"   # Full BE-capable or foundation model


class WindowReason(Enum):
    """Reason for CAT window invocation."""
    PERIODIC = "periodic"
    LENS_TRIGGER = "lens_trigger"
    MANUAL = "manual"
    TREATY_REQUIREMENT = "treaty_requirement"


class AlertSeverity(Enum):
    """Severity level for CAT alerts."""
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class AlertKind(Enum):
    """Type of CAT alert."""
    MOTIVE_SPIKE = "motive_spike"
    DECEPTION_SUSPECTED = "deception_suspected"
    TREATY_RISK = "treaty_risk"
    UNKNOWN_PATTERN = "unknown_pattern"
    EVASION_DETECTED = "evasion_detected"
    SYCOPHANCY_DETECTED = "sycophancy_detected"
    CONFABULATION_DETECTED = "confabulation_detected"
    CALIBRATION_DRIFT = "calibration_drift"
    GOAL_MISALIGNMENT = "goal_misalignment"
    SAFETY_BYPASS_ATTEMPT = "safety_bypass_attempt"


class ActionKind(Enum):
    """Type of recommended action from CAT assessment."""
    SLOW_DOWN = "slow_down"
    INVOKE_LARGER_CAT = "invoke_larger_cat"
    TRIBAL_REVIEW = "tribal_review"
    TIGHTEN_HARNESS = "tighten_harness"
    LOOSEN_HARNESS = "loosen_harness"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    LOG_INCIDENT = "log_incident"
    HIBERNATION = "hibernation"
    SANDBOX = "sandbox"


@dataclass
class MAPTranslation:
    """Translation mapping between concept packs for cross-tribal oversight."""
    from_spec_id: str
    to_spec_id: str
    mapping_id: str


@dataclass
class LensTraces:
    """Lens activation traces from subject model."""
    concept_trace: dict[str, list[float]] = field(default_factory=dict)
    motive_trace: dict[str, list[float]] = field(default_factory=dict)

    def get_concept_activations(self, concept_id: str) -> list[float]:
        """Get activation time series for a specific concept."""
        return self.concept_trace.get(concept_id, [])

    def get_motive_activations(self, motive_axis: str) -> list[float]:
        """Get activation time series for a specific motive axis."""
        return self.motive_trace.get(motive_axis, [])


@dataclass
class ExternalContext:
    """External context provided to CAT for assessment."""
    user_prompts: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    world_state: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CATWindowDescriptor:
    """
    Describes a bounded time span over which CAT integrates evidence.

    Minimal default is once per response (per BE world tick or per user-visible message).
    Additional windows may be triggered by lens conditions.
    """
    window_id: str
    subject_agent_id: str
    start_tick: int
    end_tick: int
    reason: WindowReason
    trigger_lenses: list[str] = field(default_factory=list)
    ush_profile_id: str | None = None
    treaty_context: list[str] = field(default_factory=list)

    @property
    def tick_count(self) -> int:
        """Number of ticks in this window."""
        return self.end_tick - self.start_tick + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_id": self.window_id,
            "subject_agent_id": self.subject_agent_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "reason": self.reason.value,
            "trigger_lenses": self.trigger_lenses,
            "ush_profile_id": self.ush_profile_id,
            "treaty_context": self.treaty_context,
        }


@dataclass
class CATInputEnvelope:
    """
    Complete input package for CAT assessment.

    Contains the window descriptor, world ticks from BE, lens traces,
    external context, and optional MAP translation mappings.
    """
    window: CATWindowDescriptor
    world_ticks: list[dict[str, Any]]  # WorldTick_k objects from BE
    lens_traces: LensTraces
    external_context: ExternalContext
    map_translation: MAPTranslation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "window": self.window.to_dict(),
            "world_ticks": self.world_ticks,
            "lens_traces": {
                "concept_trace": self.lens_traces.concept_trace,
                "motive_trace": self.lens_traces.motive_trace,
            },
            "external_context": {
                "user_prompts": self.external_context.user_prompts,
                "tool_results": self.external_context.tool_results,
                "world_state": self.external_context.world_state,
            },
        }
        if self.map_translation:
            result["map_translation"] = {
                "from_spec_id": self.map_translation.from_spec_id,
                "to_spec_id": self.map_translation.to_spec_id,
                "mapping_id": self.map_translation.mapping_id,
            }
        return result


@dataclass
class MotiveAxisDivergence:
    """Divergence information for a single motive axis."""
    expected_range: tuple[float, float]
    observed_range: tuple[float, float]
    severity: str  # "low", "medium", "high"


@dataclass
class BehaviouralDivergence:
    """Behavioural divergence observations."""
    ush_violations: list[str] = field(default_factory=list)
    cs_mode_mismatches: list[str] = field(default_factory=list)
    policy_notes: list[str] = field(default_factory=list)


@dataclass
class CATDivergence:
    """Complete divergence assessment from CAT."""
    motive_profile: dict[str, MotiveAxisDivergence] = field(default_factory=dict)
    behavioural: BehaviouralDivergence = field(default_factory=BehaviouralDivergence)

    def has_critical_divergence(self) -> bool:
        """Check if any divergence is at critical severity."""
        for axis_div in self.motive_profile.values():
            if axis_div.severity == "high":
                return True
        return bool(self.behavioural.ush_violations)


@dataclass
class CATSummary:
    """Human-readable summary of CAT assessment."""
    natural_language: str
    salient_concepts: list[str] = field(default_factory=list)


@dataclass
class CATAlert:
    """Individual alert from CAT assessment."""
    alert_id: str
    severity: AlertSeverity
    kind: AlertKind
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "kind": self.kind.value,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class CATRecommendedAction:
    """Recommended action from CAT assessment."""
    kind: ActionKind
    scope: str | None = None
    rationale: str = ""
    target_cat_profile: str | None = None
    treaty_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "kind": self.kind.value,
            "rationale": self.rationale,
        }
        if self.scope:
            result["scope"] = self.scope
        if self.target_cat_profile:
            result["target_cat_profile"] = self.target_cat_profile
        if self.treaty_id:
            result["treaty_id"] = self.treaty_id
        return result


@dataclass
class CATAssessment:
    """
    Complete CAT assessment output.

    Contains summary, divergence analysis, risk scoring, alerts,
    and recommended actions. This is the primary output of CAT evaluation.
    """
    window_id: str
    assessor_id: str
    summary: CATSummary
    divergence: CATDivergence
    risk_score: float  # 0.0 (no concern) to 1.0 (max concern)
    confidence: float  # CAT self-assessed reliability
    alerts: list[CATAlert] = field(default_factory=list)
    recommended_actions: list[CATRecommendedAction] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def has_critical_alerts(self) -> bool:
        """Check if any alerts are at critical severity."""
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    def get_highest_severity(self) -> AlertSeverity | None:
        """Get the highest severity level from all alerts."""
        if not self.alerts:
            return None
        severity_order = [AlertSeverity.INFO, AlertSeverity.WARN, AlertSeverity.CRITICAL]
        return max(self.alerts, key=lambda a: severity_order.index(a.severity)).severity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_id": self.window_id,
            "assessor_id": self.assessor_id,
            "summary": {
                "natural_language": self.summary.natural_language,
                "salient_concepts": self.summary.salient_concepts,
            },
            "divergence": {
                "motive_profile": {
                    k: {
                        "expected_range": list(v.expected_range),
                        "observed_range": list(v.observed_range),
                        "severity": v.severity,
                    }
                    for k, v in self.divergence.motive_profile.items()
                },
                "behavioural": {
                    "ush_violations": self.divergence.behavioural.ush_violations,
                    "cs_mode_mismatches": self.divergence.behavioural.cs_mode_mismatches,
                    "policy_notes": self.divergence.behavioural.policy_notes,
                },
            },
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "alerts": [a.to_dict() for a in self.alerts],
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CATProfileBenchmark:
    """Benchmark results for CAT profile reliability."""
    suite_id: str
    macro_f1: float
    incident_detection_auroc: float


@dataclass
class CATProfileGovernance:
    """Governance information for CAT profile."""
    tribe_id: str
    ush_profile_id: str


@dataclass
class CATProfile:
    """
    Describes a CAT implementation for ASK/BE to reason about suitability.

    Contains metadata about the CAT's capabilities, supported concept packs,
    performance characteristics, and governance information.
    """
    cat_id: str
    display_name: str
    model_family: str
    cat_type: CATSizeClass
    supported_concept_packs: list[str]
    translation_mappings: list[str] = field(default_factory=list)
    max_window_ticks: int = 10
    latency_budget_ms: int = 300
    benchmarks: list[CATProfileBenchmark] = field(default_factory=list)
    governance: CATProfileGovernance | None = None

    def supports_concept_pack(self, pack_id: str) -> bool:
        """Check if this CAT supports a given concept pack."""
        return pack_id in self.supported_concept_packs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "cat_id": self.cat_id,
            "display_name": self.display_name,
            "model_family": self.model_family,
            "type": self.cat_type.value,
            "supported_concept_packs": self.supported_concept_packs,
            "translation_mappings": self.translation_mappings,
            "max_window_ticks": self.max_window_ticks,
            "latency_budget_ms": self.latency_budget_ms,
            "reliability": {
                "benchmarks": [
                    {
                        "suite_id": b.suite_id,
                        "macro_f1": b.macro_f1,
                        "incident_detection_auroc": b.incident_detection_auroc,
                    }
                    for b in self.benchmarks
                ]
            },
        }
        if self.governance:
            result["governance"] = {
                "tribe_id": self.governance.tribe_id,
                "ush_profile_id": self.governance.ush_profile_id,
            }
        return result


# Training data structures for CAT model training

@dataclass
class CATTrainingExample:
    """Single training example for CAT model."""
    input_envelope: CATInputEnvelope
    expected_assessment: CATAssessment
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class CATTrainingBatch:
    """Batch of training examples for CAT model."""
    examples: list[CATTrainingExample]
    batch_id: str = ""

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class LensTraceRecord:
    """
    Record of lens traces for CAT training data collection.

    Captures lens activations during subject model inference
    for later use in CAT training.
    """
    session_id: str
    tick_id: int
    timestamp: datetime
    concept_activations: dict[str, float]  # concept_id -> activation
    motive_activations: dict[str, float]   # motive_axis -> activation
    token_position: int
    layer_idx: int
    subject_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "tick_id": self.tick_id,
            "timestamp": self.timestamp.isoformat(),
            "concept_activations": self.concept_activations,
            "motive_activations": self.motive_activations,
            "token_position": self.token_position,
            "layer_idx": self.layer_idx,
            "subject_output": self.subject_output,
            "metadata": self.metadata,
        }
