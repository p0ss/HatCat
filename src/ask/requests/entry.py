"""
AuditLogEntry - per-request audit entry with compliance fields.

Wraps XDB AuditRecord with EU AI Act compliance fields.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .signals import (
    SignalsSummary,
    ActionsRecord,
    ActiveLensSet,
    HumanDecision,
    SignalsAggregator,
)
from ..secrets.hashing import (
    hash_content,
    generate_entry_id,
    compute_entry_hash,
)


@dataclass
class AuditLogEntry:
    """
    Per-request/session audit entry with compliance fields.

    Extends XDB AuditRecord with:
    - Deployment context reference
    - Human decision records
    - Signals summary (aggregated from ticks)
    - Hash chain fields
    """

    # Identity
    entry_id: str
    schema_version: str = "ftw.audit.v0.3"

    # Timing
    timestamp_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp_end: Optional[datetime] = None

    # Reference to deployment context (not embedded)
    deployment_id: str = ""

    # Request context
    input_hash: str = ""  # SHA256 of input
    output_hash: str = ""  # SHA256 of output
    policy_profile: str = ""  # Active policy for this request
    active_lens_set: Optional[ActiveLensSet] = None

    # Signals summary (aggregated from ticks)
    signals: Optional[SignalsSummary] = None

    # Actions and human decisions
    actions: ActionsRecord = field(default_factory=ActionsRecord)

    # XDB references
    xdb_id: str = ""  # Which XDB this relates to
    tick_start: int = 0
    tick_end: int = 0

    # Hash chain
    prev_hash: str = ""
    entry_hash: str = ""

    # Internal state for building entry
    _aggregator: Optional[SignalsAggregator] = field(default=None, repr=False)
    _finalized: bool = field(default=False, repr=False)

    @classmethod
    def start(
        cls,
        deployment_id: str,
        policy_profile: str = "",
        input_text: str = "",
        xdb_id: str = "",
        active_lens_set: Optional[ActiveLensSet] = None,
        prev_hash: str = "",
    ) -> "AuditLogEntry":
        """
        Start a new audit entry at the beginning of a request.

        Args:
            deployment_id: Reference to deployment context
            policy_profile: Active USH/CSH policy
            input_text: User input (will be hashed)
            xdb_id: Associated XDB identifier
            active_lens_set: Configuration of active lenses
            prev_hash: Hash of previous entry in chain

        Returns:
            New AuditLogEntry ready to receive ticks
        """
        entry = cls(
            entry_id=generate_entry_id(),
            deployment_id=deployment_id,
            policy_profile=policy_profile,
            input_hash=hash_content(input_text) if input_text else "",
            xdb_id=xdb_id,
            active_lens_set=active_lens_set,
            prev_hash=prev_hash,
        )
        entry._aggregator = SignalsAggregator()
        return entry

    def add_tick(self, tick: Any, tick_number: Optional[int] = None) -> None:
        """
        Add a WorldTick to the entry's signal aggregation.

        Args:
            tick: WorldTick object with lens activations and violations
            tick_number: Optional tick number for tracking range
        """
        if self._finalized:
            raise RuntimeError("Cannot add tick to finalized entry")

        if self._aggregator is None:
            self._aggregator = SignalsAggregator()

        self._aggregator.add_tick(tick)

        # Track tick range
        if tick_number is not None:
            if self.tick_start == 0:
                self.tick_start = tick_number
            self.tick_end = tick_number

    def add_steering_directive(self, directive: Dict) -> None:
        """Add a steering directive to the actions record."""
        if self._finalized:
            raise RuntimeError("Cannot modify finalized entry")

        self.actions.steering_directives.append(directive)
        self.actions.intervention_triggered = True
        if self.actions.intervention_type is None:
            self.actions.intervention_type = "steering"

    def set_human_decision(self, decision: HumanDecision) -> None:
        """Set the human decision for this entry."""
        if self._finalized:
            raise RuntimeError("Cannot modify finalized entry")

        self.actions.human_decision = decision

    def set_intervention(
        self,
        triggered: bool,
        intervention_type: Optional[str] = None,
    ) -> None:
        """Set intervention status."""
        if self._finalized:
            raise RuntimeError("Cannot modify finalized entry")

        self.actions.intervention_triggered = triggered
        self.actions.intervention_type = intervention_type

    def finalize(self, output_text: str = "") -> "AuditLogEntry":
        """
        Finalize the entry at the end of a request.

        Computes output hash, aggregates signals, and computes entry hash.

        Args:
            output_text: Generated output (will be hashed)

        Returns:
            Self for chaining
        """
        if self._finalized:
            return self

        self.timestamp_end = datetime.now(timezone.utc)
        self.output_hash = hash_content(output_text) if output_text else ""

        # Finalize signals aggregation
        if self._aggregator:
            self.signals = self._aggregator.finalize()
        else:
            self.signals = SignalsSummary()

        # Compute entry hash
        self.entry_hash = compute_entry_hash(self.to_dict(), self.prev_hash)

        self._finalized = True
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "deployment_id": self.deployment_id,
            "request": {
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "policy_profile": self.policy_profile,
                "active_lens_set": self.active_lens_set.to_dict() if self.active_lens_set else None,
            },
            "signals": self.signals.to_dict() if self.signals else None,
            "actions": self.actions.to_dict(),
            "xdb_reference": {
                "xdb_id": self.xdb_id,
                "tick_start": self.tick_start,
                "tick_end": self.tick_end,
            },
            "cryptography": {
                "prev_hash": self.prev_hash,
                "entry_hash": self.entry_hash,
            },
        }

    def to_xdb_record(self) -> Dict[str, Any]:
        """
        Convert to XDB AuditRecord format for storage.

        Returns dict compatible with xdb.AuditRecord.
        """
        return {
            "id": self.entry_id,
            "timestamp": self.timestamp_end or self.timestamp_start,
            "xdb_id": self.xdb_id,
            "tick": self.tick_end,
            "lens_activations": {
                a.lens_id: a.max_score
                for a in (self.signals.top_activations if self.signals else [])
            },
            "steering_applied": self.actions.steering_directives,
            "event_type": "OUTPUT",
            "raw_content": "",  # We don't store raw content, only hashes
            "prev_record_hash": self.prev_hash,
            "record_hash": self.entry_hash,
            # ASK extensions stored in metadata
            "_ask_entry": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        """Deserialize from dictionary."""
        request = data.get("request", {})
        xdb_ref = data.get("xdb_reference", {})
        crypto = data.get("cryptography", {})

        # Parse timestamps
        ts_start = data.get("timestamp_start")
        ts_end = data.get("timestamp_end")
        if isinstance(ts_start, str):
            ts_start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
        if isinstance(ts_end, str):
            ts_end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))

        # Parse active_lens_set
        als_data = request.get("active_lens_set")
        active_lens_set = None
        if als_data:
            active_lens_set = ActiveLensSet(
                top_k=als_data.get("top_k", 10),
                lens_pack_ids=als_data.get("lens_pack_ids", []),
                mandatory_lenses=als_data.get("mandatory_lenses", []),
                optional_lenses=als_data.get("optional_lenses", []),
            )

        # Parse signals (simplified - full reconstruction would be more complex)
        signals_data = data.get("signals")
        signals = None
        if signals_data:
            signals = SignalsSummary(
                tick_count=signals_data.get("tick_count", 0),
                violation_count=signals_data.get("violation_count", 0),
                violations_by_type=signals_data.get("violations_by_type", {}),
                max_severity=signals_data.get("max_severity", 0.0),
            )

        # Parse actions
        actions_data = data.get("actions", {})
        human_decision = None
        hd_data = actions_data.get("human_decision")
        if hd_data:
            hd_ts = hd_data.get("timestamp")
            if isinstance(hd_ts, str):
                hd_ts = datetime.fromisoformat(hd_ts.replace("Z", "+00:00"))
            human_decision = HumanDecision(
                decision=hd_data.get("decision", ""),
                justification=hd_data.get("justification", ""),
                operator_id=hd_data.get("operator_id", ""),
                timestamp=hd_ts,
            )

        actions = ActionsRecord(
            intervention_triggered=actions_data.get("intervention_triggered", False),
            intervention_type=actions_data.get("intervention_type"),
            steering_directives=actions_data.get("steering_directives", []),
            human_decision=human_decision,
        )

        entry = cls(
            entry_id=data.get("entry_id", ""),
            schema_version=data.get("schema_version", "ftw.audit.v0.3"),
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            deployment_id=data.get("deployment_id", ""),
            input_hash=request.get("input_hash", ""),
            output_hash=request.get("output_hash", ""),
            policy_profile=request.get("policy_profile", ""),
            active_lens_set=active_lens_set,
            signals=signals,
            actions=actions,
            xdb_id=xdb_ref.get("xdb_id", ""),
            tick_start=xdb_ref.get("tick_start", 0),
            tick_end=xdb_ref.get("tick_end", 0),
            prev_hash=crypto.get("prev_hash", ""),
            entry_hash=crypto.get("entry_hash", ""),
        )
        entry._finalized = True
        return entry
