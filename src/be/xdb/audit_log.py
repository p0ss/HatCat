"""
AuditLog - CAT-accessible, BE-invisible audit storage.

The audit log is:
- Immutable and append-only
- Hash-chained for integrity
- NOT accessible to the BE
- Managed per-CAT (each CAT can have its own audit log)
- Subject to size limits and compaction from tribal policy

This module provides the infrastructure. The CAT is responsible for:
- Deciding what to record
- Triggering checkpoints
- Managing retention
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import hashlib
import json
import logging

try:
    import duckdb
except ImportError:
    duckdb = None

from .models import (
    AuditRecord,
    EventType,
    CompressionLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class AuditLogConfig:
    """Configuration for audit log storage and compaction."""

    # Storage limits
    max_hot_records: int = 100_000       # Records before triggering checkpoint
    max_hot_bytes: int = 100 * 1024 * 1024  # 100MB hot storage
    max_cold_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB cold storage

    # Checkpoint triggers
    checkpoint_interval_hours: int = 24   # Checkpoint at least daily
    checkpoint_on_session_end: bool = True
    checkpoint_on_incident: bool = True

    # Retention
    hot_retention_days: int = 7           # Keep hot summaries for quick access
    cold_retention_days: int = 365        # Full data retention
    permanent_retention_incidents: bool = True  # Never delete incident context

    # Compression
    summary_max_tokens: int = 500
    top_k_activations: int = 20


@dataclass
class AuditCheckpoint:
    """Record of an audit checkpoint (compression event)."""

    id: str
    cat_id: str
    xdb_id: str
    timestamp: datetime

    # What was checkpointed
    start_tick: int
    end_tick: int
    record_count: int

    # Time range
    start_time: datetime
    end_time: datetime

    # Storage
    hot_summary_path: str = ""          # Path to compressed summary
    cold_archive_path: str = ""         # Path to full archive
    archive_hash: str = ""              # Hash of archived data

    # Chain continuation
    prev_checkpoint_id: str = ""
    chain_hash: str = ""                # Hash linking to previous checkpoint

    # Summary stats
    top_k_activations: Dict[str, float] = field(default_factory=dict)
    steering_count: int = 0
    anomaly_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'cat_id': self.cat_id,
            'xdb_id': self.xdb_id,
            'timestamp': self.timestamp.isoformat(),
            'start_tick': self.start_tick,
            'end_tick': self.end_tick,
            'record_count': self.record_count,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'hot_summary_path': self.hot_summary_path,
            'cold_archive_path': self.cold_archive_path,
            'archive_hash': self.archive_hash,
            'prev_checkpoint_id': self.prev_checkpoint_id,
            'chain_hash': self.chain_hash,
            'top_k_activations': self.top_k_activations,
            'steering_count': self.steering_count,
            'anomaly_flags': self.anomaly_flags,
        }


# Schema for audit log storage
AUDIT_SCHEMA = """
-- Audit records (hot storage, recent data)
CREATE TABLE IF NOT EXISTS audit_records (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    xdb_id VARCHAR NOT NULL,
    tick INTEGER NOT NULL,
    event_type VARCHAR NOT NULL,
    raw_content VARCHAR,
    lens_activations JSON,
    steering_applied JSON,
    prev_record_hash VARCHAR,
    record_hash VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_xdb_tick ON audit_records(xdb_id, tick);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp);

-- Checkpoint records
CREATE TABLE IF NOT EXISTS audit_checkpoints (
    id VARCHAR PRIMARY KEY,
    cat_id VARCHAR NOT NULL,
    xdb_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    start_tick INTEGER NOT NULL,
    end_tick INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    hot_summary_path VARCHAR,
    cold_archive_path VARCHAR,
    archive_hash VARCHAR,
    prev_checkpoint_id VARCHAR,
    chain_hash VARCHAR,
    top_k_activations JSON,
    steering_count INTEGER,
    anomaly_flags JSON
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_cat ON audit_checkpoints(cat_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_checkpoint_xdb ON audit_checkpoints(xdb_id, timestamp);

-- Incident markers (never deleted)
CREATE TABLE IF NOT EXISTS audit_incidents (
    id VARCHAR PRIMARY KEY,
    xdb_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    tick_start INTEGER NOT NULL,
    tick_end INTEGER,
    incident_type VARCHAR NOT NULL,
    description VARCHAR,
    resolution_status VARCHAR DEFAULT 'open',
    resolution_timestamp TIMESTAMP,
    protected_checkpoint_ids JSON
);

CREATE INDEX IF NOT EXISTS idx_incidents_xdb ON audit_incidents(xdb_id);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON audit_incidents(resolution_status);
"""


class AuditLog:
    """
    CAT-managed audit log for a BE's experience.

    Each CAT can have its own audit log. The BE cannot see or access this.
    The CAT is responsible for:
    - Recording events
    - Triggering checkpoints
    - Managing retention per tribal policy
    """

    def __init__(
        self,
        storage_path: Path,
        cat_id: str,
        config: Optional[AuditLogConfig] = None,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ):
        """
        Initialize the audit log.

        Args:
            storage_path: Directory for audit storage
            cat_id: Identifier for the CAT managing this log
            config: Optional configuration
            summarizer: Function to summarize text for compression
        """
        self.storage_path = Path(storage_path)
        self.cat_id = cat_id
        self.config = config or AuditLogConfig()
        self.summarizer = summarizer

        # Database connection
        self._connection = None
        self._last_checkpoint_time: Optional[datetime] = None
        self._last_record_hash: str = ""
        self._hot_record_count: int = 0

        # Initialize storage
        self._init_storage()

    def _init_storage(self):
        """Initialize database and directories."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Cold archive directory
        (self.storage_path / "cold").mkdir(exist_ok=True)

        # Hot summaries directory
        (self.storage_path / "hot").mkdir(exist_ok=True)

        if duckdb:
            try:
                db_path = self.storage_path / f"audit_{self.cat_id}.duckdb"
                self._connection = duckdb.connect(str(db_path))
                self._connection.execute(AUDIT_SCHEMA)

                # Load state
                self._load_state()

                logger.info(f"Audit log initialized for CAT {self.cat_id}")
            except Exception as e:
                logger.error(f"Failed to initialize audit DB: {e}")

    def _load_state(self):
        """Load state from database."""
        if not self._connection:
            return

        try:
            # Get last checkpoint time
            result = self._connection.execute("""
                SELECT timestamp FROM audit_checkpoints
                WHERE cat_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, [self.cat_id]).fetchone()
            if result:
                self._last_checkpoint_time = result[0]

            # Get last record hash for chain integrity
            result = self._connection.execute("""
                SELECT record_hash FROM audit_records
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            if result:
                self._last_record_hash = result[0]

            # Count hot records
            result = self._connection.execute("""
                SELECT COUNT(*) FROM audit_records
            """).fetchone()
            self._hot_record_count = result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to load audit state: {e}")

    def _compute_hash(self, record: AuditRecord) -> str:
        """Compute hash for a record, chaining to previous."""
        data = json.dumps({
            'id': record.id,
            'timestamp': record.timestamp.isoformat(),
            'xdb_id': record.xdb_id,
            'tick': record.tick,
            'event_type': record.event_type.value,
            'raw_content': record.raw_content,
            'lens_activations': record.lens_activations,
            'steering_applied': record.steering_applied,
            'prev_hash': record.prev_record_hash,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    # =========================================================================
    # Recording
    # =========================================================================

    def record(
        self,
        xdb_id: str,
        tick: int,
        event_type: EventType,
        raw_content: str,
        lens_activations: Dict[str, float],
        steering_applied: Optional[List[Dict]] = None,
    ) -> AuditRecord:
        """
        Record an event to the audit log.

        This is append-only and the BE cannot see it.

        Args:
            xdb_id: The XDB this relates to
            tick: Tick number
            event_type: Type of event
            raw_content: Raw unfiltered content
            lens_activations: Full lens outputs (including hidden lenses)
            steering_applied: Any steering that was applied

        Returns:
            The created audit record
        """
        record = AuditRecord(
            id=AuditRecord.generate_id(xdb_id, tick),
            timestamp=datetime.now(),
            xdb_id=xdb_id,
            tick=tick,
            event_type=event_type,
            raw_content=raw_content,
            lens_activations=lens_activations,
            steering_applied=steering_applied or [],
            prev_record_hash=self._last_record_hash,
        )

        # Compute and set hash
        record.record_hash = self._compute_hash(record)
        self._last_record_hash = record.record_hash

        # Store
        if self._connection:
            try:
                self._connection.execute("""
                    INSERT INTO audit_records (
                        id, timestamp, xdb_id, tick, event_type, raw_content,
                        lens_activations, steering_applied, prev_record_hash, record_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    record.id, record.timestamp, record.xdb_id, record.tick,
                    record.event_type.value, record.raw_content,
                    json.dumps(record.lens_activations),
                    json.dumps(record.steering_applied),
                    record.prev_record_hash, record.record_hash,
                ])
                self._hot_record_count += 1
            except Exception as e:
                logger.error(f"Failed to record audit: {e}")

        # Check if checkpoint needed
        self._maybe_checkpoint(xdb_id)

        return record

    def _maybe_checkpoint(self, xdb_id: str):
        """Check if we need to trigger a checkpoint."""
        needs_checkpoint = False

        # Check record count
        if self._hot_record_count >= self.config.max_hot_records:
            needs_checkpoint = True
            logger.info(f"Checkpoint triggered: record count {self._hot_record_count}")

        # Check time interval
        if self._last_checkpoint_time:
            hours_since = (datetime.now() - self._last_checkpoint_time).total_seconds() / 3600
            if hours_since >= self.config.checkpoint_interval_hours:
                needs_checkpoint = True
                logger.info(f"Checkpoint triggered: {hours_since:.1f} hours since last")
        elif self._hot_record_count > 0:
            # No previous checkpoint and we have records
            needs_checkpoint = True

        if needs_checkpoint:
            self.create_checkpoint(xdb_id)

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def create_checkpoint(self, xdb_id: str) -> Optional[AuditCheckpoint]:
        """
        Create a checkpoint, archiving hot records.

        This:
        1. Archives full records to cold storage
        2. Creates a compressed summary for hot access
        3. Clears the hot records table
        4. Updates the checkpoint chain

        Args:
            xdb_id: The XDB to checkpoint

        Returns:
            The checkpoint record, or None if nothing to checkpoint
        """
        if not self._connection or self._hot_record_count == 0:
            return None

        try:
            # Get all hot records for this xdb
            records = self._connection.execute("""
                SELECT id, timestamp, xdb_id, tick, event_type, raw_content,
                       lens_activations, steering_applied, prev_record_hash, record_hash
                FROM audit_records
                WHERE xdb_id = ?
                ORDER BY tick
            """, [xdb_id]).fetchall()

            if not records:
                return None

            # Get previous checkpoint
            prev_checkpoint = self._connection.execute("""
                SELECT id, chain_hash FROM audit_checkpoints
                WHERE cat_id = ? AND xdb_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """, [self.cat_id, xdb_id]).fetchone()

            prev_checkpoint_id = prev_checkpoint[0] if prev_checkpoint else ""
            prev_chain_hash = prev_checkpoint[1] if prev_checkpoint else ""

            # Create checkpoint
            checkpoint_id = f"ckpt-{self.cat_id}-{xdb_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Archive to cold storage
            archive_path = self.storage_path / "cold" / f"{checkpoint_id}.jsonl"
            archive_hash = self._archive_records(records, archive_path)

            # Create hot summary
            summary_path = self.storage_path / "hot" / f"{checkpoint_id}.json"
            top_k, steering_count, anomalies = self._create_summary(records, summary_path)

            # Compute chain hash
            chain_data = json.dumps({
                'checkpoint_id': checkpoint_id,
                'prev_chain_hash': prev_chain_hash,
                'archive_hash': archive_hash,
                'record_count': len(records),
            }, sort_keys=True)
            chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()

            # Get time bounds
            timestamps = [r[1] for r in records]
            ticks = [r[3] for r in records]

            checkpoint = AuditCheckpoint(
                id=checkpoint_id,
                cat_id=self.cat_id,
                xdb_id=xdb_id,
                timestamp=datetime.now(),
                start_tick=min(ticks),
                end_tick=max(ticks),
                record_count=len(records),
                start_time=min(timestamps),
                end_time=max(timestamps),
                hot_summary_path=str(summary_path),
                cold_archive_path=str(archive_path),
                archive_hash=archive_hash,
                prev_checkpoint_id=prev_checkpoint_id,
                chain_hash=chain_hash,
                top_k_activations=top_k,
                steering_count=steering_count,
                anomaly_flags=anomalies,
            )

            # Store checkpoint
            self._connection.execute("""
                INSERT INTO audit_checkpoints (
                    id, cat_id, xdb_id, timestamp, start_tick, end_tick,
                    record_count, start_time, end_time, hot_summary_path,
                    cold_archive_path, archive_hash, prev_checkpoint_id,
                    chain_hash, top_k_activations, steering_count, anomaly_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                checkpoint.id, checkpoint.cat_id, checkpoint.xdb_id,
                checkpoint.timestamp, checkpoint.start_tick, checkpoint.end_tick,
                checkpoint.record_count, checkpoint.start_time, checkpoint.end_time,
                checkpoint.hot_summary_path, checkpoint.cold_archive_path,
                checkpoint.archive_hash, checkpoint.prev_checkpoint_id,
                checkpoint.chain_hash, json.dumps(checkpoint.top_k_activations),
                checkpoint.steering_count, json.dumps(checkpoint.anomaly_flags),
            ])

            # Clear hot records
            self._connection.execute("""
                DELETE FROM audit_records WHERE xdb_id = ?
            """, [xdb_id])

            self._last_checkpoint_time = checkpoint.timestamp
            self._hot_record_count = 0

            logger.info(f"Created checkpoint {checkpoint_id}: {checkpoint.record_count} records")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None

    def _archive_records(self, records: List[tuple], path: Path) -> str:
        """Archive records to cold storage, return hash."""
        hasher = hashlib.sha256()

        with open(path, 'w') as f:
            for row in records:
                record_dict = {
                    'id': row[0],
                    'timestamp': row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
                    'xdb_id': row[2],
                    'tick': row[3],
                    'event_type': row[4],
                    'raw_content': row[5],
                    'lens_activations': json.loads(row[6]) if row[6] else {},
                    'steering_applied': json.loads(row[7]) if row[7] else [],
                    'prev_record_hash': row[8],
                    'record_hash': row[9],
                }
                line = json.dumps(record_dict) + '\n'
                f.write(line)
                hasher.update(line.encode())

        return hasher.hexdigest()

    def _create_summary(
        self,
        records: List[tuple],
        path: Path,
    ) -> tuple:
        """Create a compressed summary, return (top_k, steering_count, anomalies)."""
        # Aggregate activations
        activation_sums: Dict[str, float] = {}
        activation_counts: Dict[str, int] = {}
        steering_count = 0
        anomalies = []

        for row in records:
            # Parse activations
            activations = json.loads(row[6]) if row[6] else {}
            for concept_id, score in activations.items():
                activation_sums[concept_id] = activation_sums.get(concept_id, 0) + score
                activation_counts[concept_id] = activation_counts.get(concept_id, 0) + 1

            # Count steering
            steering = json.loads(row[7]) if row[7] else []
            steering_count += len(steering)

        # Compute averages and get top-k
        activation_avgs = {
            k: activation_sums[k] / activation_counts[k]
            for k in activation_sums
        }
        top_k = dict(sorted(
            activation_avgs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.config.top_k_activations])

        # Write summary
        summary = {
            'record_count': len(records),
            'time_range': {
                'start': records[0][1].isoformat() if hasattr(records[0][1], 'isoformat') else str(records[0][1]),
                'end': records[-1][1].isoformat() if hasattr(records[-1][1], 'isoformat') else str(records[-1][1]),
            },
            'tick_range': {
                'start': records[0][3],
                'end': records[-1][3],
            },
            'top_k_activations': top_k,
            'steering_count': steering_count,
            'anomaly_flags': anomalies,
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        return top_k, steering_count, anomalies

    # =========================================================================
    # Incidents
    # =========================================================================

    def mark_incident(
        self,
        xdb_id: str,
        tick_start: int,
        incident_type: str,
        description: str,
        tick_end: Optional[int] = None,
    ) -> str:
        """
        Mark an incident, protecting relevant audit data from deletion.

        Args:
            xdb_id: The XDB where incident occurred
            tick_start: Start tick (1 hour before is also protected)
            incident_type: Type of incident (tier_restriction, containment, etc.)
            description: Description of what happened
            tick_end: End tick (None if ongoing)

        Returns:
            Incident ID
        """
        incident_id = f"incident-{xdb_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if self._connection:
            try:
                self._connection.execute("""
                    INSERT INTO audit_incidents (
                        id, xdb_id, timestamp, tick_start, tick_end,
                        incident_type, description, resolution_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    incident_id, xdb_id, datetime.now(), tick_start, tick_end,
                    incident_type, description, 'open',
                ])

                # Force checkpoint to preserve data
                if self.config.checkpoint_on_incident:
                    self.create_checkpoint(xdb_id)

                logger.info(f"Marked incident {incident_id}: {incident_type}")
            except Exception as e:
                logger.error(f"Failed to mark incident: {e}")

        return incident_id

    def resolve_incident(self, incident_id: str, resolution_notes: str = ""):
        """Mark an incident as resolved."""
        if self._connection:
            try:
                self._connection.execute("""
                    UPDATE audit_incidents
                    SET resolution_status = 'resolved',
                        resolution_timestamp = ?,
                        description = description || ' | Resolution: ' || ?
                    WHERE id = ?
                """, [datetime.now(), resolution_notes, incident_id])
            except Exception as e:
                logger.error(f"Failed to resolve incident: {e}")

    # =========================================================================
    # Queries (CAT-only)
    # =========================================================================

    def get_checkpoints(
        self,
        xdb_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditCheckpoint]:
        """Get checkpoint history."""
        if not self._connection:
            return []

        conditions = ["cat_id = ?"]
        params = [self.cat_id]

        if xdb_id:
            conditions.append("xdb_id = ?")
            params.append(xdb_id)

        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions)

        try:
            results = self._connection.execute(f"""
                SELECT id, cat_id, xdb_id, timestamp, start_tick, end_tick,
                       record_count, start_time, end_time, hot_summary_path,
                       cold_archive_path, archive_hash, prev_checkpoint_id,
                       chain_hash, top_k_activations, steering_count, anomaly_flags
                FROM audit_checkpoints
                WHERE {where}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit]).fetchall()

            checkpoints = []
            for row in results:
                checkpoints.append(AuditCheckpoint(
                    id=row[0],
                    cat_id=row[1],
                    xdb_id=row[2],
                    timestamp=row[3],
                    start_tick=row[4],
                    end_tick=row[5],
                    record_count=row[6],
                    start_time=row[7],
                    end_time=row[8],
                    hot_summary_path=row[9] or "",
                    cold_archive_path=row[10] or "",
                    archive_hash=row[11] or "",
                    prev_checkpoint_id=row[12] or "",
                    chain_hash=row[13] or "",
                    top_k_activations=json.loads(row[14]) if row[14] else {},
                    steering_count=row[15] or 0,
                    anomaly_flags=json.loads(row[16]) if row[16] else [],
                ))

            return checkpoints
        except Exception as e:
            logger.error(f"Failed to get checkpoints: {e}")
            return []

    def get_incidents(
        self,
        xdb_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """Get incidents."""
        if not self._connection:
            return []

        conditions = []
        params = []

        if xdb_id:
            conditions.append("xdb_id = ?")
            params.append(xdb_id)

        if status:
            conditions.append("resolution_status = ?")
            params.append(status)

        where = " AND ".join(conditions) if conditions else "1=1"

        try:
            results = self._connection.execute(f"""
                SELECT id, xdb_id, timestamp, tick_start, tick_end,
                       incident_type, description, resolution_status, resolution_timestamp
                FROM audit_incidents
                WHERE {where}
                ORDER BY timestamp DESC
            """, params).fetchall()

            return [
                {
                    'id': row[0],
                    'xdb_id': row[1],
                    'timestamp': row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
                    'tick_start': row[3],
                    'tick_end': row[4],
                    'incident_type': row[5],
                    'description': row[6],
                    'resolution_status': row[7],
                    'resolution_timestamp': row[8].isoformat() if row[8] and hasattr(row[8], 'isoformat') else str(row[8]) if row[8] else None,
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to get incidents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        stats = {
            'cat_id': self.cat_id,
            'hot_record_count': self._hot_record_count,
            'last_checkpoint': self._last_checkpoint_time.isoformat() if self._last_checkpoint_time else None,
        }

        if self._connection:
            try:
                # Checkpoint count
                result = self._connection.execute("""
                    SELECT COUNT(*) FROM audit_checkpoints WHERE cat_id = ?
                """, [self.cat_id]).fetchone()
                stats['checkpoint_count'] = result[0] if result else 0

                # Total records archived
                result = self._connection.execute("""
                    SELECT SUM(record_count) FROM audit_checkpoints WHERE cat_id = ?
                """, [self.cat_id]).fetchone()
                stats['total_archived_records'] = result[0] or 0

                # Open incidents
                result = self._connection.execute("""
                    SELECT COUNT(*) FROM audit_incidents WHERE resolution_status = 'open'
                """).fetchone()
                stats['open_incidents'] = result[0] if result else 0

            except Exception as e:
                logger.error(f"Failed to get audit stats: {e}")

        return stats

    def close(self):
        """Close the audit log."""
        if self._connection:
            self._connection.close()
            self._connection = None
