"""
AuditBatch - wraps XDB AuditCheckpoint with ASK compliance fields.

Adds Merkle root computation and regulatory metadata to checkpoints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import secrets

from .merkle import MerkleTree, MerkleProof, compute_merkle_root
from ..secrets.hashing import hash_content


def generate_batch_id() -> str:
    """Generate unique batch ID."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"batch_{timestamp}_{suffix}"


@dataclass
class BatchConfig:
    """Configuration for batch creation."""

    # Time-based batching
    batch_interval_hours: int = 24  # Create batch at least daily
    batch_on_entry_count: int = 1000  # Or after N entries

    # Merkle settings
    compute_merkle_tree: bool = True

    # Retention
    require_rfc3161: bool = False  # Phase 4: require timestamp token
    require_authority_receipt: bool = False  # Phase 6: require replication


@dataclass
class AuditBatch:
    """
    A sealed batch of audit entries with cryptographic integrity.

    Wraps XDB AuditCheckpoint with:
    - Merkle tree of entry hashes
    - Regulatory metadata
    - Optional external proofs (RFC 3161, authority receipts)
    """

    # Identity
    batch_id: str
    schema_version: str = "ftw.batch.v0.1"

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sealed_at: Optional[datetime] = None

    # Deployment context
    deployment_id: str = ""
    jurisdiction: str = ""  # Primary jurisdiction (e.g., "AU", "EU")

    # Entry references
    entry_ids: List[str] = field(default_factory=list)
    entry_hashes: List[str] = field(default_factory=list)
    entry_count: int = 0

    # Time window
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    tick_start: int = 0
    tick_end: int = 0

    # Cryptographic integrity
    merkle_root: str = ""
    _merkle_tree: Optional[MerkleTree] = field(default=None, repr=False)

    # Chain linkage
    prev_batch_id: str = ""
    prev_batch_hash: str = ""
    batch_hash: str = ""

    # XDB checkpoint reference (if backed by XDB)
    xdb_checkpoint_id: str = ""
    xdb_id: str = ""

    # Summary statistics
    signals_summary: Dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    violation_count: int = 0
    steering_count: int = 0

    # External proofs (populated in later phases)
    rfc3161_token: Optional[bytes] = field(default=None, repr=False)
    rfc3161_timestamp: Optional[datetime] = None
    authority_receipts: List[Dict] = field(default_factory=list)

    # State
    _sealed: bool = field(default=False, repr=False)

    @classmethod
    def create(
        cls,
        deployment_id: str,
        jurisdiction: str = "",
        prev_batch_id: str = "",
        prev_batch_hash: str = "",
    ) -> "AuditBatch":
        """
        Create a new unsealed batch.

        Args:
            deployment_id: Reference to deployment context
            jurisdiction: Primary jurisdiction code
            prev_batch_id: ID of previous batch in chain
            prev_batch_hash: Hash of previous batch

        Returns:
            New AuditBatch ready to receive entries
        """
        return cls(
            batch_id=generate_batch_id(),
            deployment_id=deployment_id,
            jurisdiction=jurisdiction,
            prev_batch_id=prev_batch_id,
            prev_batch_hash=prev_batch_hash,
        )

    def add_entry(
        self,
        entry_id: str,
        entry_hash: str,
        timestamp: Optional[datetime] = None,
        tick: Optional[int] = None,
    ) -> None:
        """
        Add an entry to this batch.

        Args:
            entry_id: The entry's ID
            entry_hash: The entry's hash
            timestamp: Entry timestamp (for window tracking)
            tick: Entry tick number (for range tracking)
        """
        if self._sealed:
            raise RuntimeError("Cannot add entry to sealed batch")

        self.entry_ids.append(entry_id)
        self.entry_hashes.append(entry_hash)
        self.entry_count = len(self.entry_ids)

        # Track time window
        if timestamp:
            if self.window_start is None or timestamp < self.window_start:
                self.window_start = timestamp
            if self.window_end is None or timestamp > self.window_end:
                self.window_end = timestamp

        # Track tick range
        if tick is not None:
            if self.tick_start == 0 or tick < self.tick_start:
                self.tick_start = tick
            if tick > self.tick_end:
                self.tick_end = tick

    def add_entries_from_audit_log(
        self,
        entries: List[Any],  # List of AuditLogEntry
    ) -> None:
        """
        Add multiple entries from audit log.

        Args:
            entries: List of AuditLogEntry objects
        """
        for entry in entries:
            self.add_entry(
                entry_id=entry.entry_id,
                entry_hash=entry.entry_hash,
                timestamp=entry.timestamp_end or entry.timestamp_start,
                tick=entry.tick_end,
            )

            # Aggregate statistics
            if entry.actions.human_decision:
                self.decision_count += 1
            if entry.signals and entry.signals.violation_count > 0:
                self.violation_count += entry.signals.violation_count
            if entry.actions.steering_directives:
                self.steering_count += len(entry.actions.steering_directives)

    def seal(self, compute_merkle: bool = True) -> "AuditBatch":
        """
        Seal this batch - no more entries can be added.

        Computes Merkle root and batch hash.

        Args:
            compute_merkle: Whether to build full Merkle tree (vs just root)

        Returns:
            Self for chaining
        """
        if self._sealed:
            return self

        self.sealed_at = datetime.now(timezone.utc)

        # Compute Merkle tree
        if self.entry_hashes:
            if compute_merkle:
                self._merkle_tree = MerkleTree.build(self.entry_hashes)
                self.merkle_root = self._merkle_tree.root
            else:
                self.merkle_root = compute_merkle_root(self.entry_hashes)
        else:
            self.merkle_root = hash_content("")

        # Compute batch hash
        self.batch_hash = self._compute_batch_hash()

        self._sealed = True
        return self

    def _compute_batch_hash(self) -> str:
        """Compute hash of this batch for chain linkage."""
        import json

        data = {
            "batch_id": self.batch_id,
            "deployment_id": self.deployment_id,
            "merkle_root": self.merkle_root,
            "entry_count": self.entry_count,
            "window_start": self.window_start.isoformat() if self.window_start else "",
            "window_end": self.window_end.isoformat() if self.window_end else "",
            "prev_batch_hash": self.prev_batch_hash,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hash_content(canonical)

    def get_proof(self, entry_hash: str) -> Optional[MerkleProof]:
        """
        Get inclusion proof for an entry.

        Args:
            entry_hash: The entry hash to prove

        Returns:
            MerkleProof if entry exists and tree is available
        """
        if not self._sealed:
            raise RuntimeError("Batch must be sealed to get proofs")

        if self._merkle_tree is None:
            # Rebuild tree if needed
            self._merkle_tree = MerkleTree.build(self.entry_hashes)

        return self._merkle_tree.get_proof(entry_hash)

    def verify_entry(self, entry_hash: str, proof: MerkleProof) -> bool:
        """
        Verify an entry belongs to this batch.

        Args:
            entry_hash: The entry hash
            proof: The inclusion proof

        Returns:
            True if entry is proven to be in this batch
        """
        if proof.merkle_root != self.merkle_root:
            return False
        if proof.entry_hash != entry_hash:
            return False
        return proof.verify()

    def request_timestamp(
        self,
        client: Optional[Any] = None,  # TimestampClient
        tsa_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Request RFC 3161 timestamp for this batch.

        Args:
            client: TimestampClient instance (or None to use default)
            tsa_id: Specific TSA to use

        Returns:
            (success, error_message)
        """
        if not self._sealed:
            return False, "Batch must be sealed before timestamping"

        if not self.merkle_root:
            return False, "No Merkle root to timestamp"

        # Import here to avoid circular dependency
        from ..secrets.tokens import TimestampClient, TimestampToken

        if client is None:
            client = TimestampClient()

        token, error = client.request_timestamp(
            data_hash=self.merkle_root,
            batch_id=self.batch_id,
            tsa_id=tsa_id,
        )

        if token:
            self.rfc3161_token = token.to_bytes()
            self.rfc3161_timestamp = token.timestamp
            return True, None
        else:
            return False, error

    def get_timestamp_token(self) -> Optional[Any]:
        """Get the RFC 3161 timestamp token if available."""
        if not self.rfc3161_token:
            return None

        from ..secrets.tokens import TimestampToken
        return TimestampToken.from_bytes(self.rfc3161_token)

    def verify_timestamp(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the RFC 3161 timestamp on this batch.

        Returns:
            (valid, error_message)
        """
        token = self.get_timestamp_token()
        if not token:
            return False, "No timestamp token"

        if token.verify(self.merkle_root):
            return True, None
        else:
            return False, token.verification_error or "Verification failed"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize batch to dictionary."""
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "sealed_at": self.sealed_at.isoformat() if self.sealed_at else None,
            "deployment_id": self.deployment_id,
            "jurisdiction": self.jurisdiction,
            "entries": {
                "entry_ids": self.entry_ids,
                "entry_hashes": self.entry_hashes,
                "entry_count": self.entry_count,
            },
            "window": {
                "start": self.window_start.isoformat() if self.window_start else None,
                "end": self.window_end.isoformat() if self.window_end else None,
                "tick_start": self.tick_start,
                "tick_end": self.tick_end,
            },
            "cryptography": {
                "merkle_root": self.merkle_root,
                "prev_batch_id": self.prev_batch_id,
                "prev_batch_hash": self.prev_batch_hash,
                "batch_hash": self.batch_hash,
            },
            "xdb_reference": {
                "xdb_id": self.xdb_id,
                "xdb_checkpoint_id": self.xdb_checkpoint_id,
            },
            "summary": {
                "decision_count": self.decision_count,
                "violation_count": self.violation_count,
                "steering_count": self.steering_count,
                "signals_summary": self.signals_summary,
            },
            "external_proofs": {
                "rfc3161_timestamp": self.rfc3161_timestamp.isoformat() if self.rfc3161_timestamp else None,
                "authority_receipts": self.authority_receipts,
            },
            "sealed": self._sealed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditBatch":
        """Deserialize batch from dictionary."""

        def parse_dt(s):
            if not s:
                return None
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        entries = data.get("entries", {})
        window = data.get("window", {})
        crypto = data.get("cryptography", {})
        xdb_ref = data.get("xdb_reference", {})
        summary = data.get("summary", {})
        ext_proofs = data.get("external_proofs", {})

        batch = cls(
            batch_id=data.get("batch_id", ""),
            schema_version=data.get("schema_version", "ftw.batch.v0.1"),
            created_at=parse_dt(data.get("created_at")),
            sealed_at=parse_dt(data.get("sealed_at")),
            deployment_id=data.get("deployment_id", ""),
            jurisdiction=data.get("jurisdiction", ""),
            entry_ids=entries.get("entry_ids", []),
            entry_hashes=entries.get("entry_hashes", []),
            entry_count=entries.get("entry_count", 0),
            window_start=parse_dt(window.get("start")),
            window_end=parse_dt(window.get("end")),
            tick_start=window.get("tick_start", 0),
            tick_end=window.get("tick_end", 0),
            merkle_root=crypto.get("merkle_root", ""),
            prev_batch_id=crypto.get("prev_batch_id", ""),
            prev_batch_hash=crypto.get("prev_batch_hash", ""),
            batch_hash=crypto.get("batch_hash", ""),
            xdb_id=xdb_ref.get("xdb_id", ""),
            xdb_checkpoint_id=xdb_ref.get("xdb_checkpoint_id", ""),
            decision_count=summary.get("decision_count", 0),
            violation_count=summary.get("violation_count", 0),
            steering_count=summary.get("steering_count", 0),
            signals_summary=summary.get("signals_summary", {}),
            rfc3161_timestamp=parse_dt(ext_proofs.get("rfc3161_timestamp")),
            authority_receipts=ext_proofs.get("authority_receipts", []),
        )
        batch._sealed = data.get("sealed", False)
        return batch

    @classmethod
    def from_xdb_checkpoint(
        cls,
        checkpoint: Any,  # AuditCheckpoint from XDB
        entry_hashes: List[str],
        deployment_id: str = "",
        jurisdiction: str = "",
        prev_batch_id: str = "",
        prev_batch_hash: str = "",
    ) -> "AuditBatch":
        """
        Create AuditBatch from XDB AuditCheckpoint.

        Args:
            checkpoint: XDB AuditCheckpoint
            entry_hashes: Hashes of entries in this checkpoint
            deployment_id: Deployment context ID
            jurisdiction: Jurisdiction code
            prev_batch_id: Previous batch ID
            prev_batch_hash: Previous batch hash

        Returns:
            Sealed AuditBatch
        """
        batch = cls(
            batch_id=f"batch_{checkpoint.id}",
            deployment_id=deployment_id,
            jurisdiction=jurisdiction,
            entry_hashes=entry_hashes,
            entry_count=len(entry_hashes),
            window_start=checkpoint.start_time,
            window_end=checkpoint.end_time,
            tick_start=checkpoint.start_tick,
            tick_end=checkpoint.end_tick,
            prev_batch_id=prev_batch_id,
            prev_batch_hash=prev_batch_hash,
            xdb_id=checkpoint.xdb_id,
            xdb_checkpoint_id=checkpoint.id,
            steering_count=checkpoint.steering_count,
        )

        # Seal the batch (computes Merkle root)
        batch.seal()
        return batch


class BatchManager:
    """
    Manages batch creation and sealing.

    Tracks open batches and triggers sealing based on configuration.
    """

    def __init__(
        self,
        deployment_id: str,
        jurisdiction: str = "",
        config: Optional[BatchConfig] = None,
        timestamp_client: Optional[Any] = None,  # TimestampClient
    ):
        self.deployment_id = deployment_id
        self.jurisdiction = jurisdiction
        self.config = config or BatchConfig()
        self._timestamp_client = timestamp_client

        # Current open batch
        self._current_batch: Optional[AuditBatch] = None

        # Sealed batches (in memory for this session)
        self._sealed_batches: List[AuditBatch] = []

        # Chain state
        self._last_batch_id: str = ""
        self._last_batch_hash: str = ""

        # Timestamp tracking
        self._timestamp_failures: List[str] = []  # Batch IDs that failed timestamping

    @property
    def current_batch(self) -> AuditBatch:
        """Get or create current open batch."""
        if self._current_batch is None:
            self._current_batch = AuditBatch.create(
                deployment_id=self.deployment_id,
                jurisdiction=self.jurisdiction,
                prev_batch_id=self._last_batch_id,
                prev_batch_hash=self._last_batch_hash,
            )
        return self._current_batch

    def add_entry(
        self,
        entry_id: str,
        entry_hash: str,
        timestamp: Optional[datetime] = None,
        tick: Optional[int] = None,
    ) -> Optional[AuditBatch]:
        """
        Add entry to current batch, seal if threshold reached.

        Returns sealed batch if sealing occurred, None otherwise.
        """
        self.current_batch.add_entry(entry_id, entry_hash, timestamp, tick)

        # Check if we should seal
        if self.current_batch.entry_count >= self.config.batch_on_entry_count:
            return self.seal_current_batch()

        return None

    def seal_current_batch(self, request_timestamp: Optional[bool] = None) -> Optional[AuditBatch]:
        """
        Seal the current batch and start a new one.

        Args:
            request_timestamp: Whether to request RFC 3161 timestamp.
                             None = use config.require_rfc3161 setting.

        Returns:
            Sealed batch, or None if no batch to seal.
        """
        if self._current_batch is None or self._current_batch.entry_count == 0:
            return None

        batch = self._current_batch
        batch.seal(compute_merkle=self.config.compute_merkle_tree)

        # Request timestamp if configured or explicitly requested
        should_timestamp = request_timestamp if request_timestamp is not None else self.config.require_rfc3161
        if should_timestamp:
            success, error = batch.request_timestamp(client=self._timestamp_client)
            if not success:
                self._timestamp_failures.append(batch.batch_id)
                print(f"Warning: Timestamp request failed for {batch.batch_id}: {error}")

        # Update chain state
        self._last_batch_id = batch.batch_id
        self._last_batch_hash = batch.batch_hash
        self._sealed_batches.append(batch)

        # Start fresh
        self._current_batch = None

        return batch

    def get_timestamp_failures(self) -> List[str]:
        """Get batch IDs that failed timestamping."""
        return list(self._timestamp_failures)

    def retry_failed_timestamps(self) -> Tuple[int, int]:
        """
        Retry timestamping for failed batches.

        Returns:
            (success_count, failure_count)
        """
        success = 0
        still_failed = []

        for batch_id in self._timestamp_failures:
            # Find batch
            batch = next((b for b in self._sealed_batches if b.batch_id == batch_id), None)
            if batch:
                ok, _ = batch.request_timestamp(client=self._timestamp_client)
                if ok:
                    success += 1
                else:
                    still_failed.append(batch_id)

        self._timestamp_failures = still_failed
        return success, len(still_failed)

    def get_sealed_batches(self) -> List[AuditBatch]:
        """Get all sealed batches from this session."""
        return list(self._sealed_batches)

    def should_seal_by_time(self, last_seal_time: Optional[datetime] = None) -> bool:
        """Check if we should seal based on time interval."""
        if self._current_batch is None or self._current_batch.entry_count == 0:
            return False

        if last_seal_time is None:
            return False

        hours_elapsed = (datetime.now(timezone.utc) - last_seal_time).total_seconds() / 3600
        return hours_elapsed >= self.config.batch_interval_hours
