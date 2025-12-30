"""
ExperienceLog - BE-accessible experience storage using DuckDB.

Provides:
- Recording timesteps with concept activations
- Querying by time, tags, concepts, text
- Tag and comment management
- Fidelity tracking
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

try:
    import duckdb
except ImportError:
    duckdb = None

from .models import (
    EventType,
    TimestepRecord,
    Tag,
    TagApplication,
    TagSource,
    TargetType,
    Comment,
    Fidelity,
    CompressionLevel,
)

logger = logging.getLogger(__name__)


# DuckDB Schema for experience log
# NOTE: Schema uses xdb_id throughout. This identifies the experiential set,
# not a transient session. A BE can have multiple XDBs (childhood memories,
# work contract memories, etc.) and can choose which ones a CAT can see.
EXPERIENCE_SCHEMA = """
-- Core timestep table
CREATE TABLE IF NOT EXISTS timesteps (
    id VARCHAR PRIMARY KEY,
    xdb_id VARCHAR NOT NULL,
    tick INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL,
    event_type VARCHAR NOT NULL,
    content VARCHAR,
    event_id VARCHAR,
    event_start BOOLEAN,
    event_end BOOLEAN,
    token_id INTEGER,
    role VARCHAR,
    fidelity VARCHAR DEFAULT 'hot',
    concept_activations JSON
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_timesteps_xdb_tick ON timesteps(xdb_id, tick);
CREATE INDEX IF NOT EXISTS idx_timesteps_fidelity ON timesteps(fidelity);
CREATE INDEX IF NOT EXISTS idx_timesteps_ts ON timesteps(ts);
CREATE INDEX IF NOT EXISTS idx_timesteps_event_type ON timesteps(event_type);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    tag_type VARCHAR NOT NULL,
    concept_id VARCHAR,
    entity_type VARCHAR,
    bud_status VARCHAR,
    created_at TIMESTAMP,
    created_by VARCHAR,
    description VARCHAR,
    use_count INTEGER DEFAULT 0,
    last_used TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_type ON tags(tag_type);
CREATE INDEX IF NOT EXISTS idx_tags_concept ON tags(concept_id);

-- Tag applications
CREATE TABLE IF NOT EXISTS tag_applications (
    id VARCHAR PRIMARY KEY,
    tag_id VARCHAR NOT NULL,
    xdb_id VARCHAR NOT NULL,
    target_type VARCHAR NOT NULL,
    timestep_id VARCHAR,
    event_id VARCHAR,
    range_start INTEGER,
    range_end INTEGER,
    confidence FLOAT,
    source VARCHAR,
    created_at TIMESTAMP,
    note VARCHAR,
    FOREIGN KEY (tag_id) REFERENCES tags(id)
);

CREATE INDEX IF NOT EXISTS idx_tag_apps_tag ON tag_applications(tag_id);
CREATE INDEX IF NOT EXISTS idx_tag_apps_xdb ON tag_applications(xdb_id);
CREATE INDEX IF NOT EXISTS idx_tag_apps_timestep ON tag_applications(timestep_id);

-- Comments
CREATE TABLE IF NOT EXISTS comments (
    id VARCHAR PRIMARY KEY,
    xdb_id VARCHAR NOT NULL,
    target_type VARCHAR NOT NULL,
    timestep_id VARCHAR,
    event_id VARCHAR,
    range_start INTEGER,
    range_end INTEGER,
    content VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comments_xdb ON comments(xdb_id);
CREATE INDEX IF NOT EXISTS idx_comments_timestep ON comments(timestep_id);

-- Time windows for fidelity tracking
CREATE TABLE IF NOT EXISTS time_windows (
    id VARCHAR PRIMARY KEY,
    xdb_id VARCHAR NOT NULL,
    start_tick INTEGER NOT NULL,
    end_tick INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    fidelity VARCHAR NOT NULL,
    compression_level INTEGER,
    pinned BOOLEAN DEFAULT FALSE,
    pinned_reason VARCHAR,
    submission_ids JSON,
    summary VARCHAR,
    top_k_activations JSON,
    significant_tags JSON,
    token_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_windows_fidelity ON time_windows(fidelity);
CREATE INDEX IF NOT EXISTS idx_windows_xdb ON time_windows(xdb_id);

-- Compressed records
CREATE TABLE IF NOT EXISTS compressed_records (
    id VARCHAR PRIMARY KEY,
    compression_level INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    start_tick INTEGER,
    end_tick INTEGER,
    xdb_id VARCHAR,
    summary VARCHAR,
    top_k_activations JSON,
    significant_tags JSON,
    source_record_ids JSON,
    token_count INTEGER,
    record_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_compressed_level_time ON compressed_records(compression_level, start_time);

-- Compaction records
CREATE TABLE IF NOT EXISTS compaction_records (
    id VARCHAR PRIMARY KEY,
    xdb_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    range_start INTEGER,
    range_end INTEGER,
    timesteps_compacted INTEGER,
    tokens_before INTEGER,
    tokens_after INTEGER,
    summary VARCHAR,
    summary_concept_tags JSON,
    top_k_activations JSON,
    trigger VARCHAR,
    result_level INTEGER
);
"""


class ExperienceLog:
    """
    BE-accessible experience log backed by DuckDB.

    Provides recording, querying, and tagging of experience data.
    The audit log is separate and not accessible through this interface.
    """

    def __init__(
        self,
        storage_path: Path,
        tag_index: Optional['TagIndex'] = None,
    ):
        """
        Initialize the experience log.

        Args:
            storage_path: Directory for DuckDB database
            tag_index: Optional tag index for cross-referencing
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "experience.duckdb"
        self.tag_index = tag_index

        self._connection: Optional['duckdb.DuckDBPyConnection'] = None
        self._init_db()

    def _init_db(self):
        """Initialize the DuckDB database with schema."""
        if duckdb is None:
            logger.warning("DuckDB not available, using in-memory fallback")
            return

        try:
            self._connection = duckdb.connect(str(self.db_path))
            self._connection.execute(EXPERIENCE_SCHEMA)
            logger.info(f"Initialized experience log at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize experience log: {e}")
            raise

    def _ensure_connection(self):
        """Ensure database connection is active."""
        if self._connection is None:
            self._init_db()
        return self._connection

    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    # =========================================================================
    # Recording
    # =========================================================================

    def record_timestep(self, record: TimestepRecord) -> str:
        """
        Record a timestep to the experience log.

        Args:
            record: The timestep record to store

        Returns:
            The timestep ID
        """
        conn = self._ensure_connection()
        if conn is None:
            return record.id

        try:
            conn.execute("""
                INSERT INTO timesteps (
                    id, xdb_id, tick, ts, event_type, content,
                    event_id, event_start, event_end, token_id, role,
                    fidelity, concept_activations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record.id,
                record.xdb_id,
                record.tick,
                record.timestamp,
                record.event_type.value,
                record.content,
                record.event_id,
                record.event_start,
                record.event_end,
                record.token_id,
                record.role,
                record.fidelity.value,
                json.dumps(record.concept_activations),
            ])
            return record.id
        except Exception as e:
            logger.error(f"Failed to record timestep: {e}")
            raise

    def record_batch(self, records: List[TimestepRecord]) -> List[str]:
        """
        Record multiple timesteps in a batch for efficiency.

        Args:
            records: List of timestep records

        Returns:
            List of timestep IDs
        """
        conn = self._ensure_connection()
        if conn is None:
            return [r.id for r in records]

        try:
            conn.executemany("""
                INSERT INTO timesteps (
                    id, xdb_id, tick, ts, event_type, content,
                    event_id, event_start, event_end, token_id, role,
                    fidelity, concept_activations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.id, r.xdb_id, r.tick, r.timestamp,
                    r.event_type.value, r.content, r.event_id,
                    r.event_start, r.event_end, r.token_id, r.role,
                    r.fidelity.value, json.dumps(r.concept_activations),
                )
                for r in records
            ])
            return [r.id for r in records]
        except Exception as e:
            logger.error(f"Failed to record batch: {e}")
            raise

    # =========================================================================
    # Querying
    # =========================================================================

    def get_timestep(self, timestep_id: str) -> Optional[TimestepRecord]:
        """Get a specific timestep by ID."""
        conn = self._ensure_connection()
        if conn is None:
            return None

        try:
            result = conn.execute("""
                SELECT id, xdb_id, tick, ts, event_type, content,
                       event_id, event_start, event_end, token_id, role,
                       fidelity, concept_activations
                FROM timesteps
                WHERE id = ?
            """, [timestep_id]).fetchone()

            if result:
                return self._row_to_timestep(result)
            return None
        except Exception as e:
            logger.error(f"Failed to get timestep: {e}")
            return None

    def query(
        self,
        *,
        xdb_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_types: Optional[List[EventType]] = None,
        tags: Optional[List[str]] = None,
        concept_activations: Optional[Dict[str, Tuple[float, float]]] = None,
        text_search: Optional[str] = None,
        fidelity: Optional[Fidelity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TimestepRecord]:
        """
        Query experience with filters.

        All filters are ANDed together.

        Args:
            xdb_id: Filter by XDB
            tick_range: (start_tick, end_tick)
            time_range: (start_time, end_time)
            event_types: List of event types to include
            tags: List of tag names or IDs to filter by
            concept_activations: {concept_id: (min_score, max_score)}
            text_search: Search in content
            fidelity: Filter by fidelity level
            limit: Max results to return
            offset: Offset for pagination

        Returns:
            List of matching timestep records
        """
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            # Build query dynamically
            conditions = []
            params = []

            if xdb_id:
                conditions.append("xdb_id = ?")
                params.append(xdb_id)

            if tick_range:
                conditions.append("tick >= ? AND tick <= ?")
                params.extend([tick_range[0], tick_range[1]])

            if time_range:
                conditions.append("ts >= ? AND ts <= ?")
                params.extend([time_range[0], time_range[1]])

            if event_types:
                placeholders = ", ".join(["?" for _ in event_types])
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in event_types])

            if text_search:
                conditions.append("content ILIKE ?")
                params.append(f"%{text_search}%")

            if fidelity:
                conditions.append("fidelity = ?")
                params.append(fidelity.value)

            # Tags require a join
            if tags:
                # Get timestep IDs that have any of the specified tags
                tag_query = """
                    SELECT DISTINCT timestep_id
                    FROM tag_applications ta
                    JOIN tags t ON ta.tag_id = t.id
                    WHERE t.name IN ({}) OR t.id IN ({})
                """.format(
                    ", ".join(["?" for _ in tags]),
                    ", ".join(["?" for _ in tags]),
                )
                conditions.append(f"id IN ({tag_query})")
                params.extend(tags + tags)  # For both name and id checks

            # Concept activation filtering
            if concept_activations:
                for concept_id, (min_score, max_score) in concept_activations.items():
                    # Use JSON extraction
                    conditions.append("""
                        CAST(json_extract_string(concept_activations, ?) AS FLOAT) >= ?
                        AND CAST(json_extract_string(concept_activations, ?) AS FLOAT) <= ?
                    """)
                    params.extend([f"$.{concept_id}", min_score, f"$.{concept_id}", max_score])

            # Build final query
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT id, xdb_id, tick, ts, event_type, content,
                       event_id, event_start, event_end, token_id, role,
                       fidelity, concept_activations
                FROM timesteps
                WHERE {where_clause}
                ORDER BY xdb_id, tick
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            results = conn.execute(query, params).fetchall()
            return [self._row_to_timestep(row) for row in results]

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_recent(
        self,
        xdb_id: str,
        n: int = 100,
    ) -> List[TimestepRecord]:
        """Get the N most recent timesteps in an XDB."""
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            results = conn.execute("""
                SELECT id, xdb_id, tick, ts, event_type, content,
                       event_id, event_start, event_end, token_id, role,
                       fidelity, concept_activations
                FROM timesteps
                WHERE xdb_id = ?
                ORDER BY tick DESC
                LIMIT ?
            """, [xdb_id, n]).fetchall()

            # Return in chronological order
            return [self._row_to_timestep(row) for row in reversed(results)]
        except Exception as e:
            logger.error(f"Failed to get recent: {e}")
            return []

    def get_by_event(self, event_id: str) -> List[TimestepRecord]:
        """Get all timesteps for a given event."""
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            results = conn.execute("""
                SELECT id, xdb_id, tick, ts, event_type, content,
                       event_id, event_start, event_end, token_id, role,
                       fidelity, concept_activations
                FROM timesteps
                WHERE event_id = ?
                ORDER BY tick
            """, [event_id]).fetchall()

            return [self._row_to_timestep(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get by event: {e}")
            return []

    def count_timesteps(
        self,
        xdb_id: Optional[str] = None,
        fidelity: Optional[Fidelity] = None,
    ) -> int:
        """Count timesteps matching criteria."""
        conn = self._ensure_connection()
        if conn is None:
            return 0

        try:
            conditions = []
            params = []

            if xdb_id:
                conditions.append("xdb_id = ?")
                params.append(xdb_id)

            if fidelity:
                conditions.append("fidelity = ?")
                params.append(fidelity.value)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            result = conn.execute(f"""
                SELECT COUNT(*) FROM timesteps WHERE {where_clause}
            """, params).fetchone()

            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count: {e}")
            return 0

    def _row_to_timestep(self, row: tuple) -> TimestepRecord:
        """Convert a database row to a TimestepRecord."""
        concept_activations = {}
        if row[12]:
            try:
                concept_activations = json.loads(row[12])
            except (json.JSONDecodeError, TypeError):
                pass

        return TimestepRecord(
            id=row[0],
            xdb_id=row[1],
            tick=row[2],
            timestamp=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
            event_type=EventType(row[4]),
            content=row[5] or "",
            event_id=row[6],
            event_start=row[7] or False,
            event_end=row[8] or False,
            token_id=row[9],
            role=row[10],
            fidelity=Fidelity(row[11] or "hot"),
            concept_activations=concept_activations,
        )

    # =========================================================================
    # Tagging
    # =========================================================================

    def add_tag(self, application: TagApplication) -> str:
        """
        Apply a tag to experience.

        Args:
            application: The tag application

        Returns:
            The application ID
        """
        conn = self._ensure_connection()
        if conn is None:
            return application.id

        try:
            conn.execute("""
                INSERT INTO tag_applications (
                    id, tag_id, xdb_id, target_type, timestep_id,
                    event_id, range_start, range_end, confidence,
                    source, created_at, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                application.id,
                application.tag_id,
                application.xdb_id,
                application.target_type.value,
                application.timestep_id,
                application.event_id,
                application.range_start,
                application.range_end,
                application.confidence,
                application.source.value,
                application.created_at,
                application.note,
            ])

            # Update tag usage stats
            conn.execute("""
                UPDATE tags
                SET use_count = use_count + 1, last_used = ?
                WHERE id = ?
            """, [datetime.now(), application.tag_id])

            return application.id
        except Exception as e:
            logger.error(f"Failed to add tag: {e}")
            raise

    def remove_tag(self, application_id: str) -> bool:
        """Remove a tag application."""
        conn = self._ensure_connection()
        if conn is None:
            return False

        try:
            result = conn.execute("""
                DELETE FROM tag_applications WHERE id = ?
            """, [application_id])
            return True
        except Exception as e:
            logger.error(f"Failed to remove tag: {e}")
            return False

    def get_tags_for_timestep(self, timestep_id: str) -> List[Tuple[Tag, TagApplication]]:
        """Get all tags applied to a timestep."""
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            results = conn.execute("""
                SELECT t.id, t.name, t.tag_type, t.concept_id, t.entity_type,
                       t.bud_status, t.created_at, t.created_by, t.description,
                       t.use_count, t.last_used,
                       ta.id, ta.tag_id, ta.xdb_id, ta.target_type,
                       ta.timestep_id, ta.event_id, ta.range_start, ta.range_end,
                       ta.confidence, ta.source, ta.created_at, ta.note
                FROM tag_applications ta
                JOIN tags t ON ta.tag_id = t.id
                WHERE ta.timestep_id = ?
            """, [timestep_id]).fetchall()

            return [self._row_to_tag_and_application(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get tags: {e}")
            return []

    def get_tags_for_range(
        self,
        xdb_id: str,
        start_tick: int,
        end_tick: int,
    ) -> List[Tuple[Tag, TagApplication]]:
        """Get all tags applied within a tick range."""
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            results = conn.execute("""
                SELECT t.id, t.name, t.tag_type, t.concept_id, t.entity_type,
                       t.bud_status, t.created_at, t.created_by, t.description,
                       t.use_count, t.last_used,
                       ta.id, ta.tag_id, ta.xdb_id, ta.target_type,
                       ta.timestep_id, ta.event_id, ta.range_start, ta.range_end,
                       ta.confidence, ta.source, ta.created_at, ta.note
                FROM tag_applications ta
                JOIN tags t ON ta.tag_id = t.id
                WHERE ta.xdb_id = ?
                  AND (
                      -- Range overlaps
                      (ta.range_start IS NOT NULL AND ta.range_start <= ? AND ta.range_end >= ?)
                      -- Or timestep is in range
                      OR ta.timestep_id IN (
                          SELECT id FROM timesteps
                          WHERE xdb_id = ? AND tick >= ? AND tick <= ?
                      )
                  )
            """, [xdb_id, end_tick, start_tick, xdb_id, start_tick, end_tick]).fetchall()

            return [self._row_to_tag_and_application(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get tags for range: {e}")
            return []

    def _row_to_tag_and_application(self, row: tuple) -> Tuple[Tag, TagApplication]:
        """Convert a joined row to Tag and TagApplication."""
        from .models import BudStatus

        bud_status = None
        if row[5]:
            bud_status = BudStatus(row[5])

        tag = Tag(
            id=row[0],
            name=row[1],
            tag_type=row[2] if isinstance(row[2], str) else row[2].value,
            concept_id=row[3],
            entity_type=row[4],
            bud_status=bud_status,
            created_at=row[6] if isinstance(row[6], datetime) else datetime.fromisoformat(str(row[6])),
            created_by=row[7] or "system",
            description=row[8],
            use_count=row[9] or 0,
            last_used=row[10] if isinstance(row[10], datetime) else (datetime.fromisoformat(str(row[10])) if row[10] else None),
        )

        # Fix tag_type if it's a string
        if isinstance(tag.tag_type, str):
            from .models import TagType
            tag.tag_type = TagType(tag.tag_type)

        application = TagApplication(
            id=row[11],
            tag_id=row[12],
            xdb_id=row[13],
            target_type=TargetType(row[14]),
            timestep_id=row[15],
            event_id=row[16],
            range_start=row[17],
            range_end=row[18],
            confidence=row[19] or 1.0,
            source=TagSource(row[20] or "manual"),
            created_at=row[21] if isinstance(row[21], datetime) else datetime.fromisoformat(str(row[21])),
            note=row[22],
        )

        return (tag, application)

    # =========================================================================
    # Comments
    # =========================================================================

    def add_comment(self, comment: Comment) -> str:
        """Add commentary to experience."""
        conn = self._ensure_connection()
        if conn is None:
            return comment.id

        try:
            conn.execute("""
                INSERT INTO comments (
                    id, xdb_id, target_type, timestep_id, event_id,
                    range_start, range_end, content, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                comment.id,
                comment.xdb_id,
                comment.target_type.value,
                comment.timestep_id,
                comment.event_id,
                comment.range_start,
                comment.range_end,
                comment.content,
                comment.created_at,
                comment.updated_at,
            ])
            return comment.id
        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            raise

    def get_comments_for_range(
        self,
        xdb_id: str,
        start_tick: int,
        end_tick: int,
    ) -> List[Comment]:
        """Get comments in a tick range."""
        conn = self._ensure_connection()
        if conn is None:
            return []

        try:
            results = conn.execute("""
                SELECT id, xdb_id, target_type, timestep_id, event_id,
                       range_start, range_end, content, created_at, updated_at
                FROM comments
                WHERE xdb_id = ?
                  AND (
                      -- Range overlaps
                      (range_start IS NOT NULL AND range_start <= ? AND range_end >= ?)
                      -- Or timestep is in range
                      OR timestep_id IN (
                          SELECT id FROM timesteps
                          WHERE xdb_id = ? AND tick >= ? AND tick <= ?
                      )
                  )
                ORDER BY created_at
            """, [xdb_id, end_tick, start_tick, xdb_id, start_tick, end_tick]).fetchall()

            return [self._row_to_comment(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get comments: {e}")
            return []

    def _row_to_comment(self, row: tuple) -> Comment:
        """Convert a row to Comment."""
        return Comment(
            id=row[0],
            xdb_id=row[1],
            target_type=TargetType(row[2]),
            timestep_id=row[3],
            event_id=row[4],
            range_start=row[5],
            range_end=row[6],
            content=row[7] or "",
            created_at=row[8] if isinstance(row[8], datetime) else datetime.fromisoformat(str(row[8])),
            updated_at=row[9] if isinstance(row[9], datetime) else (datetime.fromisoformat(str(row[9])) if row[9] else None),
        )

    # =========================================================================
    # Fidelity Management
    # =========================================================================

    def update_fidelity(
        self,
        timestep_ids: List[str],
        fidelity: Fidelity,
    ) -> int:
        """
        Update fidelity level for timesteps.

        Args:
            timestep_ids: List of timestep IDs
            fidelity: New fidelity level

        Returns:
            Number of rows updated
        """
        conn = self._ensure_connection()
        if conn is None or not timestep_ids:
            return 0

        try:
            placeholders = ", ".join(["?" for _ in timestep_ids])
            result = conn.execute(f"""
                UPDATE timesteps
                SET fidelity = ?
                WHERE id IN ({placeholders})
            """, [fidelity.value] + timestep_ids)
            return len(timestep_ids)
        except Exception as e:
            logger.error(f"Failed to update fidelity: {e}")
            return 0

    def get_timesteps_by_fidelity(
        self,
        fidelity: Fidelity,
        limit: int = 1000,
    ) -> List[TimestepRecord]:
        """Get timesteps at a specific fidelity level."""
        return self.query(fidelity=fidelity, limit=limit)

    def delete_timesteps(self, timestep_ids: List[str]) -> int:
        """
        Delete timesteps (for compression).

        Args:
            timestep_ids: List of timestep IDs to delete

        Returns:
            Number of rows deleted
        """
        conn = self._ensure_connection()
        if conn is None or not timestep_ids:
            return 0

        try:
            placeholders = ", ".join(["?" for _ in timestep_ids])
            conn.execute(f"""
                DELETE FROM timesteps WHERE id IN ({placeholders})
            """, timestep_ids)
            return len(timestep_ids)
        except Exception as e:
            logger.error(f"Failed to delete timesteps: {e}")
            return 0

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get experience log statistics."""
        conn = self._ensure_connection()
        if conn is None:
            return {}

        try:
            stats = {}

            # Total timesteps by fidelity
            result = conn.execute("""
                SELECT fidelity, COUNT(*)
                FROM timesteps
                GROUP BY fidelity
            """).fetchall()
            stats['timesteps_by_fidelity'] = {row[0]: row[1] for row in result}

            # Total timesteps by event type
            result = conn.execute("""
                SELECT event_type, COUNT(*)
                FROM timesteps
                GROUP BY event_type
            """).fetchall()
            stats['timesteps_by_event_type'] = {row[0]: row[1] for row in result}

            # XDB count
            result = conn.execute("""
                SELECT COUNT(DISTINCT xdb_id) FROM timesteps
            """).fetchone()
            stats['xdb_count'] = result[0] if result else 0

            # Tag count
            result = conn.execute("SELECT COUNT(*) FROM tags").fetchone()
            stats['tag_count'] = result[0] if result else 0

            # Tag application count
            result = conn.execute("SELECT COUNT(*) FROM tag_applications").fetchone()
            stats['tag_application_count'] = result[0] if result else 0

            # Comment count
            result = conn.execute("SELECT COUNT(*) FROM comments").fetchone()
            stats['comment_count'] = result[0] if result else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
