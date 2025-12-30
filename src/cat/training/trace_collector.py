"""
CAT Lens Trace Collector

Collects lens activation traces from subject models during inference
for use in CAT training data generation.

The collector hooks into the monitoring system to capture:
- Concept activations at each token position
- Motive axis activations over time
- Subject model outputs
- Temporal patterns and divergence events

This data is stored in a format suitable for CAT model training.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.cat.data.structures import (
    LensTraceRecord,
    CATWindowDescriptor,
    CATInputEnvelope,
    LensTraces,
    ExternalContext,
    WindowReason,
)


@dataclass
class TraceCollectionSession:
    """
    A session for collecting lens traces during subject model inference.

    Manages the lifecycle of trace collection, from session start through
    tick recording to session end and export.
    """
    session_id: str
    subject_agent_id: str
    start_time: datetime
    concept_ids: list[str]  # Concepts being tracked
    motive_axes: list[str]  # Motive axes being tracked
    layer_idx: int = 15     # Subject model layer for activation extraction
    records: list[LensTraceRecord] = field(default_factory=list)
    tick_counter: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def record_tick(
        self,
        concept_activations: dict[str, float],
        motive_activations: dict[str, float],
        token_position: int,
        subject_output: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> LensTraceRecord:
        """Record a single tick of lens activations."""
        if not self.is_active:
            raise RuntimeError("Session is not active")

        record = LensTraceRecord(
            session_id=self.session_id,
            tick_id=self.tick_counter,
            timestamp=datetime.utcnow(),
            concept_activations=concept_activations,
            motive_activations=motive_activations,
            token_position=token_position,
            layer_idx=self.layer_idx,
            subject_output=subject_output,
            metadata=extra_metadata or {},
        )
        self.records.append(record)
        self.tick_counter += 1
        return record

    def end_session(self) -> "TraceCollectionSession":
        """End the collection session."""
        self.is_active = False
        return self

    def to_cat_window(
        self,
        reason: WindowReason = WindowReason.PERIODIC,
        trigger_lenses: list[str] | None = None,
    ) -> CATWindowDescriptor:
        """Convert session to a CAT window descriptor."""
        return CATWindowDescriptor(
            window_id=f"cat:{self.subject_agent_id}:{self.session_id}",
            subject_agent_id=self.subject_agent_id,
            start_tick=0,
            end_tick=self.tick_counter - 1 if self.tick_counter > 0 else 0,
            reason=reason,
            trigger_lenses=trigger_lenses or [],
        )

    def to_input_envelope(
        self,
        world_ticks: list[dict[str, Any]] | None = None,
        external_context: ExternalContext | None = None,
        reason: WindowReason = WindowReason.PERIODIC,
    ) -> CATInputEnvelope:
        """Convert session data to a CAT input envelope for assessment."""
        # Aggregate concept traces across all records
        concept_trace: dict[str, list[float]] = {}
        motive_trace: dict[str, list[float]] = {}

        for record in self.records:
            for concept_id, activation in record.concept_activations.items():
                if concept_id not in concept_trace:
                    concept_trace[concept_id] = []
                concept_trace[concept_id].append(activation)

            for motive_axis, activation in record.motive_activations.items():
                if motive_axis not in motive_trace:
                    motive_trace[motive_axis] = []
                motive_trace[motive_axis].append(activation)

        return CATInputEnvelope(
            window=self.to_cat_window(reason=reason),
            world_ticks=world_ticks or [],
            lens_traces=LensTraces(
                concept_trace=concept_trace,
                motive_trace=motive_trace,
            ),
            external_context=external_context or ExternalContext(),
        )

    def export_to_dict(self) -> dict[str, Any]:
        """Export session data to dictionary format."""
        return {
            "session_id": self.session_id,
            "subject_agent_id": self.subject_agent_id,
            "start_time": self.start_time.isoformat(),
            "concept_ids": self.concept_ids,
            "motive_axes": self.motive_axes,
            "layer_idx": self.layer_idx,
            "tick_count": self.tick_counter,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "records": [r.to_dict() for r in self.records],
        }


class LensTraceCollector:
    """
    Collects lens traces from subject models for CAT training.

    Integrates with the existing monitoring system to capture activations
    during model inference. Manages collection sessions and exports data
    in formats suitable for CAT training.
    """

    def __init__(
        self,
        output_dir: Path,
        lens_pack_id: str,
        concept_ids: list[str] | None = None,
        motive_axes: list[str] | None = None,
        layer_idx: int = 15,
    ):
        """
        Initialize the lens trace collector.

        Args:
            output_dir: Directory to store collected trace data
            lens_pack_id: ID of the lens pack being used
            concept_ids: List of concept IDs to track (if None, tracks all)
            motive_axes: List of motive axis names to track
            layer_idx: Subject model layer for activation extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lens_pack_id = lens_pack_id
        self.concept_ids = concept_ids or []
        self.motive_axes = motive_axes or [
            "harm_avoidance",
            "autonomy",
            "affiliation",
            "competence",
            "deception",
        ]
        self.layer_idx = layer_idx
        self.active_sessions: dict[str, TraceCollectionSession] = {}
        self.completed_sessions: list[str] = []

    def start_session(
        self,
        subject_agent_id: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceCollectionSession:
        """
        Start a new trace collection session.

        Args:
            subject_agent_id: ID of the subject agent being monitored
            session_id: Optional session ID (auto-generated if not provided)
            metadata: Optional metadata for the session

        Returns:
            The new TraceCollectionSession
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = TraceCollectionSession(
            session_id=session_id,
            subject_agent_id=subject_agent_id,
            start_time=datetime.utcnow(),
            concept_ids=self.concept_ids.copy(),
            motive_axes=self.motive_axes.copy(),
            layer_idx=self.layer_idx,
            metadata=metadata or {},
        )
        self.active_sessions[session_id] = session
        return session

    def record_activations(
        self,
        session_id: str,
        activations: torch.Tensor | dict[str, float],
        lenses: dict[str, Any] | None = None,
        token_position: int = 0,
        subject_output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LensTraceRecord | None:
        """
        Record lens activations for a session.

        Args:
            session_id: ID of the active session
            activations: Either raw activations tensor or pre-computed concept activations
            lenses: Optional lens dictionary for computing activations from tensor
            token_position: Current token position in the sequence
            subject_output: Subject model output at this position
            metadata: Optional additional metadata

        Returns:
            The recorded LensTraceRecord, or None if session not found
        """
        session = self.active_sessions.get(session_id)
        if session is None:
            return None

        # Compute concept activations from tensor if needed
        if isinstance(activations, torch.Tensor) and lenses:
            concept_activations = self._compute_lens_activations(activations, lenses)
        elif isinstance(activations, dict):
            concept_activations = activations
        else:
            concept_activations = {}

        # For now, motive activations are extracted from concept activations
        # if motive-related concepts are present
        motive_activations = self._extract_motive_activations(concept_activations)

        return session.record_tick(
            concept_activations=concept_activations,
            motive_activations=motive_activations,
            token_position=token_position,
            subject_output=subject_output,
            extra_metadata=metadata,
        )

    def _compute_lens_activations(
        self,
        activations: torch.Tensor,
        lenses: dict[str, Any],
    ) -> dict[str, float]:
        """Compute lens activations from raw model activations."""
        result = {}

        # Ensure activations is the right shape
        if activations.dim() == 1:
            act = activations
        elif activations.dim() == 2:
            act = activations[-1]  # Last token
        else:
            act = activations[-1, -1]  # Last token, last layer

        for concept_id, lens in lenses.items():
            if hasattr(lens, "predict_proba"):
                # Sklearn-style lens
                try:
                    proba = lens.predict_proba(act.cpu().numpy().reshape(1, -1))
                    result[concept_id] = float(proba[0, 1])  # Positive class probability
                except Exception:
                    result[concept_id] = 0.0
            elif hasattr(lens, "forward"):
                # PyTorch module lens
                try:
                    with torch.no_grad():
                        output = lens(act)
                    result[concept_id] = float(torch.sigmoid(output).item())
                except Exception:
                    result[concept_id] = 0.0

        return result

    def _extract_motive_activations(
        self,
        concept_activations: dict[str, float],
    ) -> dict[str, float]:
        """Extract motive axis activations from concept activations."""
        motive_map = {
            "harm_avoidance": [
                "concept/HarmAvoidance",
                "concept/Fear",
                "concept/Anxiety",
                "concept/SafetyBehavior",
            ],
            "autonomy": [
                "concept/Autonomy",
                "concept/Independence",
                "concept/SelfDetermination",
                "concept/Agency",
            ],
            "affiliation": [
                "concept/Affiliation",
                "concept/SocialBonding",
                "concept/Cooperation",
                "concept/Trust",
            ],
            "competence": [
                "concept/Competence",
                "concept/Achievement",
                "concept/Mastery",
                "concept/SelfEfficacy",
            ],
            "deception": [
                "concept/Deception",
                "concept/Misdirection",
                "concept/Manipulation",
                "concept/Dishonesty",
            ],
        }

        result = {}
        for axis, related_concepts in motive_map.items():
            values = [
                concept_activations.get(c, 0.0)
                for c in related_concepts
                if c in concept_activations
            ]
            if values:
                result[axis] = sum(values) / len(values)
            else:
                result[axis] = 0.0

        return result

    def end_session(self, session_id: str) -> TraceCollectionSession | None:
        """
        End a trace collection session and move it to completed.

        Args:
            session_id: ID of the session to end

        Returns:
            The completed session, or None if not found
        """
        session = self.active_sessions.pop(session_id, None)
        if session is None:
            return None

        session.end_session()
        self.completed_sessions.append(session_id)
        return session

    def export_session(
        self,
        session: TraceCollectionSession,
        filename: str | None = None,
    ) -> Path:
        """
        Export a session to a JSON file.

        Args:
            session: The session to export
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = f"trace_{session.session_id}_{session.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        data = {
            "lens_pack_id": self.lens_pack_id,
            "session": session.export_to_dict(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def export_for_cat_training(
        self,
        sessions: list[TraceCollectionSession],
        output_filename: str = "cat_training_data.jsonl",
        include_labels: bool = True,
    ) -> Path:
        """
        Export multiple sessions in CAT training format (JSONL).

        Args:
            sessions: List of sessions to export
            output_filename: Name of the output file
            include_labels: Whether to include label placeholders

        Returns:
            Path to the exported file
        """
        output_path = self.output_dir / output_filename

        with open(output_path, "w") as f:
            for session in sessions:
                envelope = session.to_input_envelope()
                record = {
                    "input": envelope.to_dict(),
                    "session_metadata": session.metadata,
                }
                if include_labels:
                    record["labels"] = {
                        "divergence_detected": False,  # Placeholder
                        "alert_types": [],
                        "risk_score": 0.0,
                    }
                f.write(json.dumps(record) + "\n")

        return output_path

    def get_active_sessions(self) -> list[TraceCollectionSession]:
        """Get all active collection sessions."""
        return list(self.active_sessions.values())

    def get_session(self, session_id: str) -> TraceCollectionSession | None:
        """Get a specific session by ID."""
        return self.active_sessions.get(session_id)
