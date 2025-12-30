"""
Signal aggregation from WorldTicks for audit entries.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class LensActivation:
    """Aggregated lens activation statistics for an audit entry."""

    lens_id: str
    max_score: float
    mean_score: float
    threshold: float
    tier: str  # "mandatory" | "optional"
    measured_precision: float
    ticks_above_threshold: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DivergenceSummary:
    """Divergence signal summary (when CAT is available)."""

    max_score: float
    mean_score: float
    divergence_type: str  # e.g., "activation_text_mismatch", "simplex_conflict"
    top_diverging_concepts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanDecision:
    """
    Human oversight decision record.

    Required for EU AI Act Article 14 compliance.
    """

    decision: str  # "approve" | "override" | "escalate" | "block"
    justification: str  # Required free-text explanation
    operator_id: str  # Pseudonymized operator ID
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "justification": self.justification,
            "operator_id": self.operator_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ActionsRecord:
    """Record of interventions and human decisions for an audit entry."""

    intervention_triggered: bool = False
    intervention_type: Optional[str] = None  # "steering" | "block" | "escalate"
    steering_directives: List[Dict] = field(default_factory=list)
    human_decision: Optional[HumanDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_triggered": self.intervention_triggered,
            "intervention_type": self.intervention_type,
            "steering_directives": self.steering_directives,
            "human_decision": self.human_decision.to_dict() if self.human_decision else None,
        }


@dataclass
class ActiveLensSet:
    """Configuration of active lenses for a request."""

    top_k: int
    lens_pack_ids: List[str]
    mandatory_lenses: List[str]  # Tier 1 - always active
    optional_lenses: List[str]  # Tier 2 - loaded on demand

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalsSummary:
    """
    Aggregated signals from all WorldTicks in an audit entry.

    Provides summary statistics for lens activations, violations,
    and divergence signals without storing every tick.
    """

    tick_count: int = 0
    top_activations: List[LensActivation] = field(default_factory=list)
    divergence: Optional[DivergenceSummary] = None
    violation_count: int = 0
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    max_severity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "top_activations": [a.to_dict() for a in self.top_activations],
            "divergence": self.divergence.to_dict() if self.divergence else None,
            "violation_count": self.violation_count,
            "violations_by_type": self.violations_by_type,
            "max_severity": self.max_severity,
        }


class SignalsAggregator:
    """
    Aggregates WorldTicks into a SignalsSummary.

    Usage:
        aggregator = SignalsAggregator()
        for tick in world_ticks:
            aggregator.add_tick(tick)
        summary = aggregator.finalize()
    """

    def __init__(self, top_k_activations: int = 10):
        self.top_k = top_k_activations
        self.tick_count = 0

        # Per-lens accumulators
        self._lens_scores: Dict[str, List[float]] = {}
        self._lens_metadata: Dict[str, Dict] = {}  # tier, threshold, precision

        # Violation accumulators
        self._violations: List[Dict] = []

        # Divergence accumulators
        self._divergence_scores: List[float] = []
        self._divergence_types: Dict[str, int] = {}
        self._diverging_concepts: Dict[str, int] = {}

    def add_tick(self, tick: Any) -> None:
        """
        Add a WorldTick to the aggregation.

        Args:
            tick: WorldTick object with lens activations and violations
        """
        self.tick_count += 1

        # Aggregate lens activations
        if hasattr(tick, "concept_activations"):
            for lens_id, score in tick.concept_activations.items():
                if lens_id not in self._lens_scores:
                    self._lens_scores[lens_id] = []
                self._lens_scores[lens_id].append(score)

        # Aggregate violations
        if hasattr(tick, "violations") and tick.violations:
            for violation in tick.violations:
                self._violations.append(violation)
                vtype = getattr(violation, "type", "unknown")
                if isinstance(violation, dict):
                    vtype = violation.get("type", "unknown")

        # Aggregate divergence signals
        if hasattr(tick, "divergence") and tick.divergence:
            div = tick.divergence
            if hasattr(div, "score"):
                self._divergence_scores.append(div.score)
            if hasattr(div, "divergence_type"):
                dtype = div.divergence_type
                self._divergence_types[dtype] = self._divergence_types.get(dtype, 0) + 1
            if hasattr(div, "concepts"):
                for concept in div.concepts:
                    self._diverging_concepts[concept] = self._diverging_concepts.get(concept, 0) + 1

    def add_lens_metadata(
        self,
        lens_id: str,
        tier: str,
        threshold: float,
        measured_precision: float,
    ) -> None:
        """Add metadata for a lens (tier, threshold, precision)."""
        self._lens_metadata[lens_id] = {
            "tier": tier,
            "threshold": threshold,
            "measured_precision": measured_precision,
        }

    def finalize(self) -> SignalsSummary:
        """Finalize aggregation and return SignalsSummary."""
        # Build top activations
        activations = []
        for lens_id, scores in self._lens_scores.items():
            if not scores:
                continue

            meta = self._lens_metadata.get(lens_id, {})
            threshold = meta.get("threshold", 0.5)

            activation = LensActivation(
                lens_id=lens_id,
                max_score=max(scores),
                mean_score=sum(scores) / len(scores),
                threshold=threshold,
                tier=meta.get("tier", "optional"),
                measured_precision=meta.get("measured_precision", 0.0),
                ticks_above_threshold=sum(1 for s in scores if s > threshold),
            )
            activations.append(activation)

        # Sort by max_score descending, take top_k
        activations.sort(key=lambda a: a.max_score, reverse=True)
        top_activations = activations[: self.top_k]

        # Build divergence summary
        divergence = None
        if self._divergence_scores:
            # Find most common divergence type
            top_dtype = max(self._divergence_types.keys(), key=lambda k: self._divergence_types[k]) if self._divergence_types else "unknown"

            # Find top diverging concepts
            sorted_concepts = sorted(self._diverging_concepts.keys(), key=lambda k: self._diverging_concepts[k], reverse=True)

            divergence = DivergenceSummary(
                max_score=max(self._divergence_scores),
                mean_score=sum(self._divergence_scores) / len(self._divergence_scores),
                divergence_type=top_dtype,
                top_diverging_concepts=sorted_concepts[:5],
            )

        # Build violation summary
        violations_by_type: Dict[str, int] = {}
        max_severity = 0.0
        for v in self._violations:
            if isinstance(v, dict):
                vtype = v.get("type", "unknown")
                severity = v.get("severity", 0.0)
            else:
                vtype = getattr(v, "type", "unknown")
                severity = getattr(v, "severity", 0.0)
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            max_severity = max(max_severity, severity)

        return SignalsSummary(
            tick_count=self.tick_count,
            top_activations=top_activations,
            divergence=divergence,
            violation_count=len(self._violations),
            violations_by_type=violations_by_type,
            max_severity=max_severity,
        )
