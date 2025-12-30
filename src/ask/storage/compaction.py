"""
Compaction - storage management with integrity preservation.

Allows pruning old audit entries while maintaining cryptographic proof
that the data existed. Batch metadata (Merkle roots, chain links) are
preserved even after entry data is removed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import secrets

from ..secrets.hashing import hash_content


class RetentionRule(Enum):
    """Types of retention rules."""

    # Time-based
    KEEP_DAYS = "keep_days"  # Keep entries for N days
    KEEP_HOURS = "keep_hours"  # Keep entries for N hours

    # Count-based
    KEEP_BATCHES = "keep_batches"  # Keep last N batches with full data
    KEEP_ENTRIES = "keep_entries"  # Keep last N entries

    # Size-based
    MAX_SIZE_MB = "max_size_mb"  # Compact when storage exceeds N MB

    # Event-based
    KEEP_DECISIONS = "keep_decisions"  # Always keep human decisions
    KEEP_VIOLATIONS = "keep_violations"  # Always keep violations


@dataclass
class CompactionPolicy:
    """
    Policy defining what can be compacted and when.

    Multiple rules can be combined. An entry is eligible for compaction
    only if ALL rules allow it (most restrictive wins).
    """

    policy_id: str = ""
    name: str = "default"

    # Time-based retention
    keep_days: int = 90  # Keep full data for 90 days
    keep_hours: int = 0  # Alternative: hours (0 = use days)

    # Count-based retention (0 = disabled)
    keep_batches: int = 0  # Keep last N batches uncompacted
    keep_entries: int = 0  # Keep last N entries

    # Size limit (0 = no limit)
    max_size_mb: int = 0

    # Preserve important events regardless of age
    preserve_decisions: bool = True  # Keep human decision entries
    preserve_violations: bool = True  # Keep violation entries
    preserve_escalations: bool = True  # Keep escalation entries

    # Compaction behavior
    compact_to_summary: bool = True  # Keep summary stats in compacted batches
    remove_entry_hashes: bool = False  # Remove individual hashes (keeps only root)

    # Verification
    require_sealed: bool = True  # Only compact sealed batches
    require_timestamped: bool = False  # Only compact timestamped batches

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"policy_{secrets.token_hex(4)}"

    def get_cutoff_time(self, now: Optional[datetime] = None) -> datetime:
        """Get the datetime before which entries can be compacted."""
        now = now or datetime.now(timezone.utc)

        if self.keep_hours > 0:
            return now - timedelta(hours=self.keep_hours)
        return now - timedelta(days=self.keep_days)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "keep_days": self.keep_days,
            "keep_hours": self.keep_hours,
            "keep_batches": self.keep_batches,
            "keep_entries": self.keep_entries,
            "max_size_mb": self.max_size_mb,
            "preserve_decisions": self.preserve_decisions,
            "preserve_violations": self.preserve_violations,
            "preserve_escalations": self.preserve_escalations,
            "compact_to_summary": self.compact_to_summary,
            "remove_entry_hashes": self.remove_entry_hashes,
            "require_sealed": self.require_sealed,
            "require_timestamped": self.require_timestamped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactionPolicy":
        return cls(
            policy_id=data.get("policy_id", ""),
            name=data.get("name", "default"),
            keep_days=data.get("keep_days", 90),
            keep_hours=data.get("keep_hours", 0),
            keep_batches=data.get("keep_batches", 0),
            keep_entries=data.get("keep_entries", 0),
            max_size_mb=data.get("max_size_mb", 0),
            preserve_decisions=data.get("preserve_decisions", True),
            preserve_violations=data.get("preserve_violations", True),
            preserve_escalations=data.get("preserve_escalations", True),
            compact_to_summary=data.get("compact_to_summary", True),
            remove_entry_hashes=data.get("remove_entry_hashes", False),
            require_sealed=data.get("require_sealed", True),
            require_timestamped=data.get("require_timestamped", False),
        )


@dataclass
class CompactedBatchSummary:
    """
    Summary of a compacted batch - metadata preserved after entry removal.

    This is what remains after compaction: enough information to verify
    the chain and prove entries existed, but without the full entry data.
    """

    batch_id: str

    # Timing
    created_at: datetime
    sealed_at: datetime
    compacted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Entry counts (preserved for audit trail)
    original_entry_count: int = 0
    preserved_entry_count: int = 0  # Entries kept due to preserve_* rules

    # Cryptographic anchors (always preserved)
    merkle_root: str = ""
    batch_hash: str = ""

    # Chain links (always preserved)
    prev_batch_id: str = ""
    prev_batch_hash: str = ""

    # External proofs (preserved if present)
    rfc3161_timestamp: Optional[datetime] = None
    has_authority_receipts: bool = False

    # Summary statistics (if compact_to_summary=True)
    signals_summary: Dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    violation_count: int = 0

    # Preserved entry IDs (for entries kept due to preserve_* rules)
    preserved_entry_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "sealed_at": self.sealed_at.isoformat(),
            "compacted_at": self.compacted_at.isoformat(),
            "original_entry_count": self.original_entry_count,
            "preserved_entry_count": self.preserved_entry_count,
            "merkle_root": self.merkle_root,
            "batch_hash": self.batch_hash,
            "prev_batch_id": self.prev_batch_id,
            "prev_batch_hash": self.prev_batch_hash,
            "rfc3161_timestamp": self.rfc3161_timestamp.isoformat() if self.rfc3161_timestamp else None,
            "has_authority_receipts": self.has_authority_receipts,
            "signals_summary": self.signals_summary,
            "decision_count": self.decision_count,
            "violation_count": self.violation_count,
            "preserved_entry_ids": self.preserved_entry_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactedBatchSummary":
        return cls(
            batch_id=data["batch_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            sealed_at=datetime.fromisoformat(data["sealed_at"]),
            compacted_at=datetime.fromisoformat(data["compacted_at"]),
            original_entry_count=data.get("original_entry_count", 0),
            preserved_entry_count=data.get("preserved_entry_count", 0),
            merkle_root=data.get("merkle_root", ""),
            batch_hash=data.get("batch_hash", ""),
            prev_batch_id=data.get("prev_batch_id", ""),
            prev_batch_hash=data.get("prev_batch_hash", ""),
            rfc3161_timestamp=datetime.fromisoformat(data["rfc3161_timestamp"]) if data.get("rfc3161_timestamp") else None,
            has_authority_receipts=data.get("has_authority_receipts", False),
            signals_summary=data.get("signals_summary", {}),
            decision_count=data.get("decision_count", 0),
            violation_count=data.get("violation_count", 0),
            preserved_entry_ids=data.get("preserved_entry_ids", []),
        )


@dataclass
class CompactionRecord:
    """
    Record of a compaction operation.

    Provides an audit trail of what was compacted and when, with
    cryptographic proof of the compacted data.
    """

    record_id: str = ""

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    # Policy used
    policy_id: str = ""
    policy_name: str = ""

    # What was compacted
    batches_processed: int = 0
    batches_compacted: int = 0
    batches_skipped: int = 0
    entries_removed: int = 0
    entries_preserved: int = 0

    # Storage impact
    bytes_before: int = 0
    bytes_after: int = 0
    bytes_freed: int = 0

    # Compacted batch summaries
    compacted_batches: List[str] = field(default_factory=list)  # batch IDs

    # Integrity proof
    compaction_hash: str = ""  # Hash of this record for chain

    # Status
    success: bool = False
    error_message: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"compact_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"

    def complete(self, success: bool = True, error: str = "") -> None:
        """Mark compaction as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self.success = success
        self.error_message = error
        self.bytes_freed = self.bytes_before - self.bytes_after

        # Compute integrity hash
        self._compute_hash()

    def _compute_hash(self) -> None:
        """Compute hash of this compaction record."""
        data = (
            f"{self.record_id}|"
            f"{self.started_at.isoformat()}|"
            f"{self.completed_at.isoformat() if self.completed_at else ''}|"
            f"{self.policy_id}|"
            f"{self.batches_compacted}|"
            f"{self.entries_removed}|"
            f"{','.join(sorted(self.compacted_batches))}"
        )
        self.compaction_hash = hash_content(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "batches_processed": self.batches_processed,
            "batches_compacted": self.batches_compacted,
            "batches_skipped": self.batches_skipped,
            "entries_removed": self.entries_removed,
            "entries_preserved": self.entries_preserved,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_freed": self.bytes_freed,
            "compacted_batches": self.compacted_batches,
            "compaction_hash": self.compaction_hash,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactionRecord":
        record = cls(
            record_id=data.get("record_id", ""),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now(timezone.utc),
            policy_id=data.get("policy_id", ""),
            policy_name=data.get("policy_name", ""),
            batches_processed=data.get("batches_processed", 0),
            batches_compacted=data.get("batches_compacted", 0),
            batches_skipped=data.get("batches_skipped", 0),
            entries_removed=data.get("entries_removed", 0),
            entries_preserved=data.get("entries_preserved", 0),
            bytes_before=data.get("bytes_before", 0),
            bytes_after=data.get("bytes_after", 0),
            bytes_freed=data.get("bytes_freed", 0),
            compacted_batches=data.get("compacted_batches", []),
            compaction_hash=data.get("compaction_hash", ""),
            success=data.get("success", False),
            error_message=data.get("error_message", ""),
        )
        if data.get("completed_at"):
            record.completed_at = datetime.fromisoformat(data["completed_at"])
        return record


class CompactionManager:
    """
    Manages compaction of audit batches.

    Applies retention policies to batches, removing entry data while
    preserving cryptographic integrity proofs.
    """

    def __init__(
        self,
        policy: Optional[CompactionPolicy] = None,
    ):
        self.policy = policy or CompactionPolicy()

        # Compaction history
        self._records: List[CompactionRecord] = []

        # Compacted batch summaries (batch_id -> summary)
        self._summaries: Dict[str, CompactedBatchSummary] = {}

    def evaluate_batch(
        self,
        batch: Any,  # AuditBatch
        now: Optional[datetime] = None,
        current_batch_count: int = 0,
    ) -> Tuple[bool, str]:
        """
        Evaluate if a batch is eligible for compaction.

        Args:
            batch: The batch to evaluate
            now: Current time (for testing)
            current_batch_count: Total number of batches (for keep_batches rule)

        Returns:
            (eligible, reason) - whether eligible and why/why not
        """
        now = now or datetime.now(timezone.utc)

        # Must be sealed
        if self.policy.require_sealed and not batch._sealed:
            return False, "Batch not sealed"

        # Must be timestamped (if required)
        if self.policy.require_timestamped and not batch.rfc3161_timestamp:
            return False, "Batch not timestamped"

        # Check keep_batches rule
        if self.policy.keep_batches > 0:
            # This requires knowing batch position in sequence
            # Caller should handle this via current_batch_count
            pass  # Handled externally

        # Check time-based retention
        cutoff = self.policy.get_cutoff_time(now)
        if batch.sealed_at and batch.sealed_at > cutoff:
            return False, f"Within retention period (sealed {batch.sealed_at})"

        # Already compacted?
        if batch.batch_id in self._summaries:
            return False, "Already compacted"

        return True, "Eligible for compaction"

    def compact_batch(
        self,
        batch: Any,  # AuditBatch
        entry_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[CompactedBatchSummary, List[str]]:
        """
        Compact a single batch.

        Args:
            batch: The batch to compact
            entry_metadata: Optional dict of entry_id -> metadata for preserve checks

        Returns:
            (summary, preserved_entry_ids) - the summary and list of preserved entries
        """
        entry_metadata = entry_metadata or {}

        # Determine which entries to preserve
        preserved_ids: List[str] = []

        for entry_id in batch.entry_ids:
            meta = entry_metadata.get(entry_id, {})

            # Check preserve rules
            if self.policy.preserve_decisions and meta.get("has_decision"):
                preserved_ids.append(entry_id)
            elif self.policy.preserve_violations and meta.get("has_violation"):
                preserved_ids.append(entry_id)
            elif self.policy.preserve_escalations and meta.get("has_escalation"):
                preserved_ids.append(entry_id)

        # Create summary
        summary = CompactedBatchSummary(
            batch_id=batch.batch_id,
            created_at=batch.created_at,
            sealed_at=batch.sealed_at,
            original_entry_count=batch.entry_count,
            preserved_entry_count=len(preserved_ids),
            merkle_root=batch.merkle_root,
            batch_hash=batch.batch_hash,
            prev_batch_id=batch.prev_batch_id,
            prev_batch_hash=batch.prev_batch_hash,
            rfc3161_timestamp=batch.rfc3161_timestamp,
            has_authority_receipts=len(batch.authority_receipts) > 0,
            preserved_entry_ids=preserved_ids,
        )

        # Include summary stats if configured
        if self.policy.compact_to_summary:
            summary.signals_summary = batch.signals_summary
            summary.decision_count = batch.decision_count
            summary.violation_count = batch.violation_count

        # Store summary
        self._summaries[batch.batch_id] = summary

        return summary, preserved_ids

    def run_compaction(
        self,
        batches: List[Any],  # List[AuditBatch]
        entry_metadata: Optional[Dict[str, Dict]] = None,
        now: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> CompactionRecord:
        """
        Run compaction on a list of batches.

        Args:
            batches: Batches to consider for compaction
            entry_metadata: Optional metadata for entry preserve checks
            now: Current time (for testing)
            dry_run: If True, evaluate but don't actually compact

        Returns:
            CompactionRecord documenting what was (or would be) compacted
        """
        record = CompactionRecord(
            policy_id=self.policy.policy_id,
            policy_name=self.policy.name,
        )

        entry_metadata = entry_metadata or {}

        # Sort batches by sealed time (oldest first)
        sorted_batches = sorted(
            [b for b in batches if b._sealed and b.sealed_at],
            key=lambda b: b.sealed_at,
        )

        # Apply keep_batches rule if set
        if self.policy.keep_batches > 0:
            # Keep the most recent N batches
            if len(sorted_batches) > self.policy.keep_batches:
                eligible_batches = sorted_batches[:-self.policy.keep_batches]
            else:
                eligible_batches = []
        else:
            eligible_batches = sorted_batches

        try:
            for batch in batches:
                record.batches_processed += 1

                # Check if in eligible set
                if batch not in eligible_batches:
                    eligible, reason = False, "Protected by keep_batches rule"
                else:
                    eligible, reason = self.evaluate_batch(batch, now)

                if not eligible:
                    record.batches_skipped += 1
                    continue

                if not dry_run:
                    # Perform compaction
                    summary, preserved = self.compact_batch(batch, entry_metadata)

                    record.batches_compacted += 1
                    record.entries_removed += batch.entry_count - len(preserved)
                    record.entries_preserved += len(preserved)
                    record.compacted_batches.append(batch.batch_id)
                else:
                    # Dry run - just count
                    record.batches_compacted += 1
                    record.entries_removed += batch.entry_count
                    record.compacted_batches.append(batch.batch_id)

            record.complete(success=True)

        except Exception as e:
            record.complete(success=False, error=str(e))

        if not dry_run:
            self._records.append(record)

        return record

    def get_summary(self, batch_id: str) -> Optional[CompactedBatchSummary]:
        """Get compaction summary for a batch."""
        return self._summaries.get(batch_id)

    def is_compacted(self, batch_id: str) -> bool:
        """Check if a batch has been compacted."""
        return batch_id in self._summaries

    def get_compaction_records(self) -> List[CompactionRecord]:
        """Get all compaction records."""
        return list(self._records)

    def get_chain_integrity(self, summaries: Optional[List[CompactedBatchSummary]] = None) -> Tuple[bool, List[str]]:
        """
        Verify chain integrity across compacted batches.

        Returns:
            (valid, errors) - whether chain is valid and any errors found
        """
        summaries = summaries or list(self._summaries.values())

        if not summaries:
            return True, []

        errors = []

        # Sort by creation time
        sorted_summaries = sorted(summaries, key=lambda s: s.created_at)

        # Verify chain links
        for i in range(1, len(sorted_summaries)):
            prev = sorted_summaries[i - 1]
            curr = sorted_summaries[i]

            if curr.prev_batch_id and curr.prev_batch_id != prev.batch_id:
                # Check if it links to something we know about
                if curr.prev_batch_id not in self._summaries:
                    errors.append(
                        f"Batch {curr.batch_id} links to unknown batch {curr.prev_batch_id}"
                    )

            if curr.prev_batch_hash and curr.prev_batch_hash != prev.batch_hash:
                if curr.prev_batch_id == prev.batch_id:
                    errors.append(
                        f"Batch {curr.batch_id} has hash mismatch with {prev.batch_id}"
                    )

        return len(errors) == 0, errors

    def estimate_savings(
        self,
        batches: List[Any],
        avg_entry_size_bytes: int = 1024,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Estimate storage savings from compaction.

        Args:
            batches: Batches to evaluate
            avg_entry_size_bytes: Estimated average entry size
            now: Current time

        Returns:
            Dict with estimation details
        """
        record = self.run_compaction(batches, now=now, dry_run=True)

        estimated_bytes = record.entries_removed * avg_entry_size_bytes

        return {
            "batches_eligible": record.batches_compacted,
            "batches_protected": record.batches_skipped,
            "entries_removable": record.entries_removed,
            "estimated_bytes_freed": estimated_bytes,
            "estimated_mb_freed": estimated_bytes / (1024 * 1024),
        }


# Convenience functions

def create_aggressive_policy(keep_days: int = 7) -> CompactionPolicy:
    """Create an aggressive compaction policy for limited storage."""
    return CompactionPolicy(
        name="aggressive",
        keep_days=keep_days,
        preserve_decisions=True,
        preserve_violations=True,
        preserve_escalations=True,
        compact_to_summary=True,
        remove_entry_hashes=True,
    )


def create_conservative_policy(keep_days: int = 365) -> CompactionPolicy:
    """Create a conservative compaction policy for compliance."""
    return CompactionPolicy(
        name="conservative",
        keep_days=keep_days,
        preserve_decisions=True,
        preserve_violations=True,
        preserve_escalations=True,
        compact_to_summary=True,
        remove_entry_hashes=False,
        require_timestamped=True,
    )


def create_minimal_policy(keep_batches: int = 10) -> CompactionPolicy:
    """Create a minimal policy keeping only recent batches."""
    return CompactionPolicy(
        name="minimal",
        keep_days=1,  # Very short time window
        keep_batches=keep_batches,  # But keep last N batches
        preserve_decisions=True,
        preserve_violations=True,
        compact_to_summary=True,
        remove_entry_hashes=True,
    )
