"""
StorageManager - Fidelity transitions and compression for XDB.

Handles:
- Progressive compression as data ages
- Storage pressure management
- Fidelity tier transitions (HOT -> WARM -> COLD)
- Evidence retention for graft/meld submissions
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from uuid import uuid4
import json
import logging

try:
    import duckdb
except ImportError:
    duckdb = None

from .models import (
    Fidelity,
    CompressionLevel,
    TimestepRecord,
    TimeWindow,
    CompressedRecord,
    CompactionRecord,
    CompactionTrigger,
)

logger = logging.getLogger(__name__)


# Summarization callback type
SummarizeFunction = Callable[[List[str]], str]


def default_summarizer(texts: List[str]) -> str:
    """Default summarizer that just concatenates with truncation."""
    combined = " ".join(texts)
    if len(combined) > 500:
        return combined[:500] + "..."
    return combined


@dataclass
class CompressionConfig:
    """Configuration for compression behavior."""

    # Storage thresholds (bytes)
    max_storage_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB default
    compression_threshold: float = 0.8  # Start compressing at 80% full

    # Time-based compression rules
    cold_reply_after_hours: int = 24  # Compress to REPLY after 24h
    cold_session_after_days: int = 7  # Compress to SESSION after 7 days
    cold_day_after_weeks: int = 4    # Compress to DAY after 4 weeks
    cold_week_after_months: int = 3   # Compress to WEEK after 3 months
    cold_month_after_months: int = 12  # Compress to MONTH after 12 months

    # Top-k for compression
    top_k: int = 10  # Keep top-k activations/tags at each level

    # WARM quota (tokens a BE can keep as training data)
    warm_quota_tokens: int = 10_000_000  # 10M tokens

    # Batch sizes
    compression_batch_size: int = 100

    # Context window (for reference, managed by ContextWindowManager)
    context_window_tokens: int = 32768

    @classmethod
    def from_lifecycle_contract(cls, contract: Dict[str, Any]) -> 'CompressionConfig':
        """
        Create config from a LifecycleContract's resources section.

        The contract should have a 'resources' field with memory allocations.
        Falls back to tribal minimums if fields are missing.

        Args:
            contract: A parsed LifecycleContract dict

        Returns:
            CompressionConfig with contract-specified limits
        """
        resources = contract.get('resources', {})
        memory = resources.get('memory', {})

        # Extract values with tribal minimum fallbacks
        # These minimums match ASK_HATCAT_TRIBAL_POLICY.md section 9.3
        return cls(
            max_storage_bytes=memory.get('cold_storage_bytes', 100 * 1024 * 1024),  # 100MB tribal min
            warm_quota_tokens=memory.get('warm_quota_tokens', 100_000),  # 100k tribal min
            context_window_tokens=memory.get('context_window_tokens', 8192),  # 8k tribal min
            # Other fields use class defaults (time-based compression, etc.)
        )

    @classmethod
    def tribal_minimums(cls) -> 'CompressionConfig':
        """
        Create config with HatCat tribal minimum allocations.

        These are the guaranteed minimums for any BE under our governance,
        as specified in ASK_HATCAT_TRIBAL_POLICY.md section 9.3.
        """
        return cls(
            max_storage_bytes=100 * 1024 * 1024,  # 100MB
            warm_quota_tokens=100_000,  # 100k tokens
            context_window_tokens=8192,  # 8k tokens
        )


class StorageManager:
    """
    Manages XDB storage and triggers compression under pressure.

    Handles:
    - Tracking storage usage
    - Progressive compression through fidelity tiers
    - Evidence retention for submissions
    - WARM quota management
    """

    def __init__(
        self,
        storage_path: Path,
        db_connection: Optional['duckdb.DuckDBPyConnection'] = None,
        config: Optional[CompressionConfig] = None,
        summarizer: Optional[SummarizeFunction] = None,
    ):
        """
        Initialize the storage manager.

        Args:
            storage_path: Path to XDB storage
            db_connection: DuckDB connection
            config: Compression configuration
            summarizer: Function to summarize text (for compression)
        """
        self.storage_path = Path(storage_path)
        self._connection = db_connection
        self.config = config or CompressionConfig()
        self.summarizer = summarizer or default_summarizer

    def set_connection(self, conn: 'duckdb.DuckDBPyConnection'):
        """Set the database connection."""
        self._connection = conn

    # =========================================================================
    # Storage Tracking
    # =========================================================================

    def get_current_usage(self) -> int:
        """Get current storage usage in bytes."""
        try:
            total = 0
            db_path = self.storage_path / "experience.duckdb"
            if db_path.exists():
                total += db_path.stat().st_size

            # Add any WAL files
            for wal_file in self.storage_path.glob("*.wal"):
                total += wal_file.stat().st_size

            return total
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return 0

    def get_usage_ratio(self) -> float:
        """Get storage usage as ratio of max."""
        return self.get_current_usage() / self.config.max_storage_bytes

    def needs_compression(self) -> bool:
        """Check if compression is needed due to storage pressure."""
        return self.get_usage_ratio() >= self.config.compression_threshold

    # =========================================================================
    # Fidelity Transitions
    # =========================================================================

    def check_and_compress(self):
        """Compress data if over threshold."""
        if not self.needs_compression():
            return

        logger.info("Storage pressure detected, starting compression")

        # Compress oldest fine-grained data first
        compressed_count = 0
        max_rounds = 10  # Prevent infinite loop

        for _ in range(max_rounds):
            if not self.needs_compression():
                break

            compressed = self._compress_oldest_level()
            if not compressed:
                break  # Nothing left to compress

            compressed_count += 1

        logger.info(f"Compressed {compressed_count} batches")

    def run_scheduled_compression(self):
        """Run time-based compression regardless of storage pressure."""
        now = datetime.now()

        # Find expired HOT data -> COLD-REPLY
        self._compress_expired_hot(
            now - timedelta(hours=self.config.cold_reply_after_hours)
        )

        # Find expired COLD-REPLY -> COLD-SESSION
        self._compress_expired_level(
            CompressionLevel.REPLY,
            CompressionLevel.SESSION,
            now - timedelta(days=self.config.cold_session_after_days),
        )

        # Find expired COLD-SESSION -> COLD-DAY
        self._compress_expired_level(
            CompressionLevel.SESSION,
            CompressionLevel.DAY,
            now - timedelta(weeks=self.config.cold_day_after_weeks),
        )

        # Continue for coarser levels...
        self._compress_expired_level(
            CompressionLevel.DAY,
            CompressionLevel.WEEK,
            now - timedelta(days=self.config.cold_week_after_months * 30),
        )

        self._compress_expired_level(
            CompressionLevel.WEEK,
            CompressionLevel.MONTH,
            now - timedelta(days=self.config.cold_month_after_months * 30),
        )

    def _compress_oldest_level(self) -> bool:
        """
        Find oldest records at finest available granularity and compress.

        Returns True if something was compressed.
        """
        if not self._connection:
            return False

        # Priority order for compression:
        # 1. COLD-REPLY -> COLD-SESSION (oldest first)
        # 2. COLD-SESSION -> COLD-DAY
        # 3. COLD-DAY -> COLD-WEEK
        # ... etc

        # Never compress: HOT, WARM, SUBMITTED (protected)

        try:
            for source_level in [
                CompressionLevel.REPLY,
                CompressionLevel.SESSION,
                CompressionLevel.DAY,
                CompressionLevel.WEEK,
                CompressionLevel.MONTH,
                CompressionLevel.QUARTER,
            ]:
                target_level = CompressionLevel(source_level.value + 1)

                records = self._get_oldest_compressed_records(
                    source_level,
                    batch_size=self.config.compression_batch_size,
                )

                if records:
                    compressed = self._compress_records(records, target_level)
                    if compressed:
                        self._store_compressed(compressed)
                        self._delete_source_compressed_records(records)
                        return True

            return False

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False

    def _compress_expired_hot(self, before: datetime):
        """Compress expired HOT timesteps to COLD-REPLY."""
        if not self._connection:
            return

        try:
            # Get expired HOT timesteps (not WARM or SUBMITTED)
            results = self._connection.execute("""
                SELECT id, xdb_id, tick, ts, event_type, content,
                       concept_activations
                FROM timesteps
                WHERE fidelity = 'hot' AND ts < ?
                ORDER BY xdb_id, tick
                LIMIT ?
            """, [before, self.config.compression_batch_size]).fetchall()

            if not results:
                return

            # Group by xdb for compression
            by_xdb: Dict[str, List[tuple]] = {}
            for row in results:
                xdb_id = row[1]
                if xdb_id not in by_xdb:
                    by_xdb[xdb_id] = []
                by_xdb[xdb_id].append(row)

            # Compress each xdb's batch
            for xdb_id, rows in by_xdb.items():
                compressed = self._compress_timesteps_to_reply(rows, xdb_id)
                if compressed:
                    self._store_compressed(compressed)

                    # Delete original timesteps
                    ids = [row[0] for row in rows]
                    placeholders = ", ".join(["?" for _ in ids])
                    self._connection.execute(f"""
                        DELETE FROM timesteps WHERE id IN ({placeholders})
                    """, ids)

            logger.info(f"Compressed {len(results)} expired HOT timesteps")

        except Exception as e:
            logger.error(f"Failed to compress expired HOT: {e}")

    def _compress_expired_level(
        self,
        source_level: CompressionLevel,
        target_level: CompressionLevel,
        before: datetime,
    ):
        """Compress expired records from one level to the next."""
        if not self._connection:
            return

        try:
            results = self._connection.execute("""
                SELECT id, compression_level, start_time, end_time,
                       start_tick, end_tick, xdb_id, summary,
                       top_k_activations, significant_tags,
                       source_record_ids, token_count, record_count
                FROM compressed_records
                WHERE compression_level = ? AND end_time < ?
                ORDER BY start_time
                LIMIT ?
            """, [source_level.value, before, self.config.compression_batch_size]).fetchall()

            if not results:
                return

            records = [self._row_to_compressed(row) for row in results]
            compressed = self._compress_records(records, target_level)

            if compressed:
                self._store_compressed(compressed)
                self._delete_source_compressed_records(records)

            logger.info(f"Compressed {len(records)} records from {source_level.name} to {target_level.name}")

        except Exception as e:
            logger.error(f"Failed to compress level {source_level.name}: {e}")

    # =========================================================================
    # Compression Logic
    # =========================================================================

    def _compress_timesteps_to_reply(
        self,
        rows: List[tuple],
        xdb_id: str,
    ) -> Optional[CompressedRecord]:
        """Compress timesteps to REPLY level."""
        if not rows:
            return None

        try:
            # Aggregate activations
            all_activations = Counter()
            all_content = []

            for row in rows:
                content = row[5] or ""
                all_content.append(content)

                activations = {}
                if row[6]:
                    try:
                        activations = json.loads(row[6])
                    except (json.JSONDecodeError, TypeError):
                        pass

                for concept, score in activations.items():
                    all_activations[concept] += score

            top_k = dict(all_activations.most_common(self.config.top_k))
            summary = self.summarizer(all_content)

            # Get time bounds
            timestamps = [row[3] for row in rows]
            ticks = [row[2] for row in rows]

            return CompressedRecord(
                id=CompressedRecord.generate_id(CompressionLevel.REPLY),
                level=CompressionLevel.REPLY,
                start_time=min(timestamps) if timestamps else datetime.now(),
                end_time=max(timestamps) if timestamps else datetime.now(),
                start_tick=min(ticks) if ticks else None,
                end_tick=max(ticks) if ticks else None,
                xdb_id=xdb_id,
                summary=summary,
                top_k_activations=top_k,
                significant_tags=[],  # Would need to query tag_applications
                source_record_ids=[row[0] for row in rows],
                token_count=len(rows),
                record_count=len(rows),
            )

        except Exception as e:
            logger.error(f"Failed to compress timesteps: {e}")
            return None

    def _compress_records(
        self,
        records: List[CompressedRecord],
        target_level: CompressionLevel,
    ) -> Optional[CompressedRecord]:
        """Compress multiple records into one at coarser granularity."""
        if not records:
            return None

        try:
            # Aggregate activations (frequency-weighted)
            all_activations = Counter()
            for r in records:
                for concept, score in r.top_k_activations.items():
                    all_activations[concept] += score

            top_k_activations = dict(all_activations.most_common(self.config.top_k))

            # Aggregate tags (most frequent)
            all_tags = Counter()
            for r in records:
                all_tags.update(r.significant_tags)
            significant_tags = [t for t, _ in all_tags.most_common(self.config.top_k)]

            # Summarize summaries
            summaries = [r.summary for r in records if r.summary]
            summary = self.summarizer(summaries)

            return CompressedRecord(
                id=CompressedRecord.generate_id(target_level),
                level=target_level,
                start_time=min(r.start_time for r in records),
                end_time=max(r.end_time for r in records),
                start_tick=None,  # Lose tick precision at coarse levels
                end_tick=None,
                xdb_id=None,  # May span XDBs
                summary=summary,
                top_k_activations=top_k_activations,
                significant_tags=significant_tags,
                source_record_ids=[r.id for r in records],
                token_count=sum(r.token_count for r in records),
                record_count=len(records),
            )

        except Exception as e:
            logger.error(f"Failed to compress records: {e}")
            return None

    # =========================================================================
    # Database Operations
    # =========================================================================

    def _get_oldest_compressed_records(
        self,
        level: CompressionLevel,
        batch_size: int,
    ) -> List[CompressedRecord]:
        """Get oldest compressed records at a level."""
        if not self._connection:
            return []

        try:
            results = self._connection.execute("""
                SELECT id, compression_level, start_time, end_time,
                       start_tick, end_tick, xdb_id, summary,
                       top_k_activations, significant_tags,
                       source_record_ids, token_count, record_count
                FROM compressed_records
                WHERE compression_level = ?
                ORDER BY start_time
                LIMIT ?
            """, [level.value, batch_size]).fetchall()

            return [self._row_to_compressed(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get compressed records: {e}")
            return []

    def _store_compressed(self, record: CompressedRecord):
        """Store a compressed record."""
        if not self._connection:
            return

        try:
            self._connection.execute("""
                INSERT INTO compressed_records (
                    id, compression_level, start_time, end_time,
                    start_tick, end_tick, xdb_id, summary,
                    top_k_activations, significant_tags,
                    source_record_ids, token_count, record_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record.id,
                record.level.value,
                record.start_time,
                record.end_time,
                record.start_tick,
                record.end_tick,
                record.xdb_id,
                record.summary,
                json.dumps(record.top_k_activations),
                json.dumps(record.significant_tags),
                json.dumps(record.source_record_ids),
                record.token_count,
                record.record_count,
            ])
        except Exception as e:
            logger.error(f"Failed to store compressed record: {e}")

    def _delete_source_compressed_records(self, records: List[CompressedRecord]):
        """Delete source compressed records after aggregation."""
        if not self._connection or not records:
            return

        try:
            ids = [r.id for r in records]
            placeholders = ", ".join(["?" for _ in ids])
            self._connection.execute(f"""
                DELETE FROM compressed_records WHERE id IN ({placeholders})
            """, ids)
        except Exception as e:
            logger.error(f"Failed to delete source records: {e}")

    def _row_to_compressed(self, row: tuple) -> CompressedRecord:
        """Convert a database row to CompressedRecord."""
        top_k = {}
        if row[8]:
            try:
                top_k = json.loads(row[8])
            except (json.JSONDecodeError, TypeError):
                pass

        tags = []
        if row[9]:
            try:
                tags = json.loads(row[9])
            except (json.JSONDecodeError, TypeError):
                pass

        source_ids = []
        if row[10]:
            try:
                source_ids = json.loads(row[10])
            except (json.JSONDecodeError, TypeError):
                pass

        return CompressedRecord(
            id=row[0],
            level=CompressionLevel(row[1]),
            start_time=row[2] if isinstance(row[2], datetime) else datetime.fromisoformat(str(row[2])),
            end_time=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
            start_tick=row[4],
            end_tick=row[5],
            xdb_id=row[6],
            summary=row[7] or "",
            top_k_activations=top_k,
            significant_tags=tags,
            source_record_ids=source_ids,
            token_count=row[11] or 0,
            record_count=row[12] or 0,
        )

    # =========================================================================
    # WARM Management
    # =========================================================================

    def get_warm_usage(self) -> int:
        """Get current WARM token count."""
        if not self._connection:
            return 0

        try:
            result = self._connection.execute("""
                SELECT COUNT(*) FROM timesteps WHERE fidelity = 'warm'
            """).fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get WARM usage: {e}")
            return 0

    def get_warm_quota_remaining(self) -> int:
        """Get remaining WARM quota."""
        return max(0, self.config.warm_quota_tokens - self.get_warm_usage())

    def pin_to_warm(
        self,
        timestep_ids: List[str],
        reason: str = "",
    ) -> int:
        """
        Pin timesteps to WARM fidelity.

        Args:
            timestep_ids: Timesteps to pin
            reason: Why they're being pinned

        Returns:
            Number actually pinned (may be less if quota exceeded)
        """
        if not self._connection or not timestep_ids:
            return 0

        # Check quota
        remaining = self.get_warm_quota_remaining()
        if remaining <= 0:
            logger.warning("WARM quota exceeded")
            return 0

        # Limit to remaining quota
        to_pin = timestep_ids[:remaining]

        try:
            placeholders = ", ".join(["?" for _ in to_pin])
            self._connection.execute(f"""
                UPDATE timesteps
                SET fidelity = 'warm'
                WHERE id IN ({placeholders}) AND fidelity = 'hot'
            """, to_pin)
            return len(to_pin)
        except Exception as e:
            logger.error(f"Failed to pin to WARM: {e}")
            return 0

    def unpin_from_warm(self, timestep_ids: List[str]) -> int:
        """
        Unpin timesteps from WARM (they'll become HOT then flow to COLD).

        Args:
            timestep_ids: Timesteps to unpin

        Returns:
            Number unpinned
        """
        if not self._connection or not timestep_ids:
            return 0

        try:
            placeholders = ", ".join(["?" for _ in timestep_ids])
            self._connection.execute(f"""
                UPDATE timesteps
                SET fidelity = 'hot'
                WHERE id IN ({placeholders}) AND fidelity = 'warm'
            """, timestep_ids)
            return len(timestep_ids)
        except Exception as e:
            logger.error(f"Failed to unpin from WARM: {e}")
            return 0

    # =========================================================================
    # Submission Evidence
    # =========================================================================

    def submit_evidence(
        self,
        timestep_ids: List[str],
        submission_id: str,
    ) -> int:
        """
        Mark timesteps as submission evidence (SUBMITTED fidelity).

        These cannot be compressed until the submission is resolved.

        Args:
            timestep_ids: Timesteps that are evidence
            submission_id: The graft/meld submission ID

        Returns:
            Number marked as evidence
        """
        if not self._connection or not timestep_ids:
            return 0

        try:
            placeholders = ", ".join(["?" for _ in timestep_ids])
            self._connection.execute(f"""
                UPDATE timesteps
                SET fidelity = 'submitted'
                WHERE id IN ({placeholders})
            """, timestep_ids)

            # TODO: Track submission_id linkage in time_windows table

            return len(timestep_ids)
        except Exception as e:
            logger.error(f"Failed to submit evidence: {e}")
            return 0

    def resolve_submission(
        self,
        submission_id: str,
        accepted: bool,
    ) -> int:
        """
        Resolve a submission and release evidence.

        Args:
            submission_id: The submission being resolved
            accepted: Whether it was accepted

        Returns:
            Number of timesteps released
        """
        if not self._connection:
            return 0

        try:
            # For now, just change SUBMITTED back to WARM (if rejected) or HOT (if accepted)
            # In a full implementation we'd track submission_ids properly
            new_fidelity = 'hot' if accepted else 'warm'

            result = self._connection.execute("""
                UPDATE timesteps
                SET fidelity = ?
                WHERE fidelity = 'submitted'
            """, [new_fidelity])

            # Count would require a query before update
            return 0  # Placeholder

        except Exception as e:
            logger.error(f"Failed to resolve submission: {e}")
            return 0

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage manager statistics."""
        stats = {
            'storage_bytes': self.get_current_usage(),
            'max_storage_bytes': self.config.max_storage_bytes,
            'usage_ratio': self.get_usage_ratio(),
            'needs_compression': self.needs_compression(),
            'warm_usage': self.get_warm_usage(),
            'warm_quota': self.config.warm_quota_tokens,
            'warm_remaining': self.get_warm_quota_remaining(),
        }

        if self._connection:
            try:
                # Compressed records by level
                result = self._connection.execute("""
                    SELECT compression_level, COUNT(*), SUM(token_count)
                    FROM compressed_records
                    GROUP BY compression_level
                """).fetchall()

                stats['compressed_by_level'] = {
                    CompressionLevel(row[0]).name: {
                        'count': row[1],
                        'token_count': row[2] or 0,
                    }
                    for row in result
                }

            except Exception as e:
                logger.error(f"Failed to get compression stats: {e}")

        return stats
