"""
XDB - Experience Database main interface.

The unified interface for the BE's episodic memory system.
Provides access to:
- Experience recording and querying
- Folksonomy tagging
- Document repository
- Context window management
- Fidelity tier transitions

The audit log is NOT accessible through this interface.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from uuid import uuid4
import json
import logging

try:
    import duckdb
except ImportError:
    duckdb = None

from .models import (
    EventType,
    TagType,
    TagSource,
    TargetType,
    BudStatus,
    Fidelity,
    CompressionLevel,
    TimestepRecord,
    Tag,
    TagApplication,
    Comment,
    CompactionRecord,
    CompactionTrigger,
    Document,
    DocumentType,
)
from .experience_log import ExperienceLog
from .tag_index import TagIndex
from .storage_manager import StorageManager, CompressionConfig

logger = logging.getLogger(__name__)


@dataclass
class ContextWindowConfig:
    """Configuration for context window management."""
    max_tokens: int = 8192
    compaction_threshold: float = 0.8  # Compact at 80% full
    summary_max_tokens: int = 500  # Max tokens in compaction summary


class ContextWindowManager:
    """Tracks context window and manages compaction."""

    def __init__(
        self,
        config: ContextWindowConfig,
        experience_log: ExperienceLog,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ):
        self.config = config
        self.experience_log = experience_log
        self.summarizer = summarizer

        self.current_tokens = 0
        self.xdb_id: Optional[str] = None
        self.current_tick = 0

        # Track compaction history
        self.compaction_history: List[CompactionRecord] = []

    def add_tokens(self, count: int, tick: int):
        """Track token addition."""
        self.current_tokens += count
        self.current_tick = tick

        if self.current_tokens >= self.config.max_tokens * self.config.compaction_threshold:
            self._trigger_compaction()

    def _trigger_compaction(self) -> Optional[CompactionRecord]:
        """Perform compaction of older context."""
        if not self.xdb_id:
            return None

        # Compact the older half of context
        half_tick = self.current_tick // 2
        if half_tick <= 0:
            return None

        try:
            # Get timesteps to compact
            timesteps = self.experience_log.query(
                xdb_id=self.xdb_id,
                tick_range=(0, half_tick),
                limit=10000,
            )

            if not timesteps:
                return None

            # Generate summary
            contents = [ts.content for ts in timesteps if ts.content]
            summary = ""
            if self.summarizer and contents:
                summary = self.summarizer(contents)
            else:
                # Simple truncation fallback
                combined = " ".join(contents)
                summary = combined[:self.config.summary_max_tokens] + "..."

            # Aggregate top-k activations
            from collections import Counter
            all_activations = Counter()
            for ts in timesteps:
                for concept, score in ts.concept_activations.items():
                    all_activations[concept] += score
            top_k = dict(all_activations.most_common(10))

            # Create compaction record
            record = CompactionRecord(
                id=CompactionRecord.generate_id(),
                xdb_id=self.xdb_id,
                timestamp=datetime.now(),
                range_start=0,
                range_end=half_tick,
                timesteps_compacted=len(timesteps),
                tokens_before=self.current_tokens,
                tokens_after=len(summary.split()),
                summary=summary,
                summary_concept_tags=list(top_k.keys())[:5],
                top_k_activations=top_k,
                trigger=CompactionTrigger.CONTEXT_FULL,
                result_level=CompressionLevel.REPLY,
            )

            self.compaction_history.append(record)

            # Update token count
            self.current_tokens = self.current_tokens - record.tokens_before + record.tokens_after

            logger.info(f"Compacted {record.timesteps_compacted} timesteps, tokens: {record.tokens_before} -> {record.tokens_after}")

            return record

        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            return None

    def get_state(self) -> Dict[str, Any]:
        """Get current context window state."""
        return {
            'max_tokens': self.config.max_tokens,
            'current_tokens': self.current_tokens,
            'utilization': self.current_tokens / self.config.max_tokens if self.config.max_tokens > 0 else 0,
            'xdb_id': self.xdb_id,
            'current_tick': self.current_tick,
            'compaction_count': len(self.compaction_history),
        }


class DocumentRepository:
    """Reference document storage."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Document] = {}
        self._load_documents()

    def _load_documents(self):
        """Load documents from storage."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                    for doc_data in data.get('documents', []):
                        doc = Document(
                            id=doc_data['id'],
                            doc_type=DocumentType(doc_data['doc_type']),
                            name=doc_data['name'],
                            path=doc_data['path'],
                            version=doc_data.get('version'),
                            content=doc_data.get('content', ''),
                            content_hash=doc_data.get('content_hash', ''),
                            source=doc_data.get('source', ''),
                            summary=doc_data.get('summary'),
                            tags=doc_data.get('tags', []),
                        )
                        self.documents[doc.id] = doc
            except Exception as e:
                logger.error(f"Failed to load document index: {e}")

    def _save_index(self):
        """Save document index."""
        index_path = self.storage_path / "index.json"
        try:
            data = {
                'documents': [doc.to_dict() for doc in self.documents.values()]
            }
            with open(index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document index: {e}")

    def add_document(
        self,
        name: str,
        content: str,
        doc_type: DocumentType,
        *,
        path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: str = "",
    ) -> Document:
        """Add a document to the repository."""
        import hashlib

        doc = Document(
            id=Document.generate_id(),
            doc_type=doc_type,
            name=name,
            path=path or f"/{doc_type.value}/{name}",
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            source=source,
            tags=tags or [],
        )

        self.documents[doc.id] = doc
        self._save_index()

        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)

    def get_by_path(self, path: str) -> Optional[Document]:
        """Get document by path."""
        for doc in self.documents.values():
            if doc.path == path:
                return doc
        return None

    def search(
        self,
        *,
        text_query: Optional[str] = None,
        doc_type: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Document]:
        """Search documents."""
        results = []
        for doc in self.documents.values():
            if doc_type and doc.doc_type != doc_type:
                continue
            if tags and not any(t in doc.tags for t in tags):
                continue
            if text_query:
                query_lower = text_query.lower()
                if query_lower not in doc.name.lower() and query_lower not in doc.content.lower():
                    continue
            results.append(doc)
        return results

    def update_document(self, doc_id: str, content: str) -> Optional[Document]:
        """Update document content."""
        import hashlib

        doc = self.documents.get(doc_id)
        if not doc:
            return None

        doc.content = content
        doc.content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc.updated_at = datetime.now()

        self._save_index()
        return doc

    def list_by_type(self, doc_type: DocumentType) -> List[Document]:
        """List all documents of a type."""
        return [d for d in self.documents.values() if d.doc_type == doc_type]


class XDB:
    """
    Experience Database - the BE's episodic memory.

    Provides unified access to:
    - Experience log (BE-accessible timestep records)
    - Tag index (folksonomy for organizing experience)
    - Document repository (reference materials)
    - Context management (compaction tracking)
    - Storage management (fidelity tiers)

    The audit log is NOT accessible through this interface.

    Resource limits (memory quotas, context window, cold storage) are typically
    specified in the BE's LifecycleContract. Use from_lifecycle_contract() to
    create an XDB with contract-specified limits.
    """

    @classmethod
    def from_lifecycle_contract(
        cls,
        contract: Dict[str, Any],
        storage_path: Path,
        *,
        concept_pack_path: Optional[Path] = None,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ) -> 'XDB':
        """
        Create XDB from a LifecycleContract.

        Extracts resource limits from the contract's 'resources.memory' section
        and creates appropriate configurations. Falls back to tribal minimums
        if fields are missing.

        Args:
            contract: A parsed LifecycleContract dict
            storage_path: Base path for XDB storage
            concept_pack_path: Optional path to concept pack
            summarizer: Function to summarize text for compression

        Returns:
            Configured XDB instance

        Example contract structure:
            {
                "be_id": "be-123",
                "resources": {
                    "memory": {
                        "warm_quota_tokens": 10000000,
                        "context_window_tokens": 32768,
                        "cold_storage_bytes": 10737418240
                    }
                }
            }
        """
        be_id = contract.get('be_id', contract.get('id', 'unknown'))
        resources = contract.get('resources', {})
        memory = resources.get('memory', {})

        # Create storage config from contract
        storage_config = CompressionConfig.from_lifecycle_contract(contract)

        # Create context config from contract
        context_config = ContextWindowConfig(
            max_tokens=memory.get('context_window_tokens', 8192),
        )

        return cls(
            storage_path=storage_path,
            be_id=be_id,
            concept_pack_path=concept_pack_path,
            context_config=context_config,
            storage_config=storage_config,
            summarizer=summarizer,
        )

    @classmethod
    def with_tribal_minimums(
        cls,
        storage_path: Path,
        be_id: str,
        *,
        concept_pack_path: Optional[Path] = None,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ) -> 'XDB':
        """
        Create XDB with HatCat tribal minimum resource allocations.

        Use this when no specific contract is available. These minimums are
        guaranteed to any BE under HatCat governance as specified in
        ASK_HATCAT_TRIBAL_POLICY.md section 9.3.

        Args:
            storage_path: Base path for XDB storage
            be_id: The BE's identifier
            concept_pack_path: Optional path to concept pack
            summarizer: Function to summarize text for compression

        Returns:
            XDB configured with tribal minimums
        """
        storage_config = CompressionConfig.tribal_minimums()
        context_config = ContextWindowConfig(
            max_tokens=storage_config.context_window_tokens,
        )

        return cls(
            storage_path=storage_path,
            be_id=be_id,
            concept_pack_path=concept_pack_path,
            context_config=context_config,
            storage_config=storage_config,
            summarizer=summarizer,
        )

    def __init__(
        self,
        storage_path: Path,
        be_id: str,
        *,
        concept_pack_path: Optional[Path] = None,
        context_config: Optional[ContextWindowConfig] = None,
        storage_config: Optional[CompressionConfig] = None,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ):
        """
        Initialize XDB.

        For most use cases, prefer the class methods:
        - XDB.from_lifecycle_contract() - when you have a contract
        - XDB.with_tribal_minimums() - for guaranteed minimum allocations

        Args:
            storage_path: Base path for XDB storage
            be_id: The BE's identifier
            concept_pack_path: Optional path to concept pack
            context_config: Context window configuration
            storage_config: Storage/compression configuration
            summarizer: Function to summarize text for compression
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.be_id = be_id

        # Initialize DuckDB connection
        self._connection: Optional['duckdb.DuckDBPyConnection'] = None
        self._init_db()

        # Initialize components
        self.tag_index = TagIndex(
            db_connection=self._connection,
            concept_pack_path=concept_pack_path,
        )

        self.experience_log = ExperienceLog(
            storage_path=self.storage_path / "experience",
            tag_index=self.tag_index,
        )

        # Share connection with experience log
        if self._connection and self.experience_log._connection:
            # Experience log has its own connection, that's fine
            pass

        self.documents = DocumentRepository(self.storage_path / "documents")

        self.storage_manager = StorageManager(
            storage_path=self.storage_path,
            db_connection=self._connection,
            config=storage_config,
            summarizer=summarizer,
        )

        self.context_config = context_config or ContextWindowConfig()
        self.context_manager: Optional[ContextWindowManager] = None

        # Current XDB identity
        self.xdb_id: Optional[str] = None
        self.current_tick = 0

    def _init_db(self):
        """Initialize shared DuckDB connection."""
        if duckdb is None:
            logger.warning("DuckDB not available")
            return

        try:
            db_path = self.storage_path / "xdb.duckdb"
            self._connection = duckdb.connect(str(db_path))
            logger.info(f"Initialized XDB at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize XDB: {e}")

    def close(self):
        """Close XDB and all connections."""
        if self.experience_log:
            self.experience_log.close()
        if self._connection:
            self._connection.close()
            self._connection = None

    # =========================================================================
    # XDB Session Management
    # =========================================================================

    def start_session(
        self,
        xdb_id: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
    ) -> str:
        """
        Start a new XDB session.

        Args:
            xdb_id: Optional XDB ID (generated if not provided)
            max_context_tokens: Override default context size

        Returns:
            The XDB ID
        """
        self.xdb_id = xdb_id or f"xdb-{uuid4().hex[:12]}"
        self.current_tick = 0

        config = ContextWindowConfig(
            max_tokens=max_context_tokens or self.context_config.max_tokens,
            compaction_threshold=self.context_config.compaction_threshold,
        )

        self.context_manager = ContextWindowManager(
            config=config,
            experience_log=self.experience_log,
        )
        self.context_manager.xdb_id = self.xdb_id

        logger.info(f"Started XDB session {self.xdb_id}")
        return self.xdb_id

    def end_session(self):
        """End the current XDB session."""
        logger.info(f"Ended XDB session {self.xdb_id}")
        self.xdb_id = None
        self.context_manager = None

    # =========================================================================
    # Recording
    # =========================================================================

    def record(
        self,
        event_type: EventType,
        content: str,
        concept_activations: Optional[Dict[str, float]] = None,
        *,
        event_id: Optional[str] = None,
        event_start: bool = False,
        event_end: bool = False,
        token_id: Optional[int] = None,
        role: Optional[str] = None,
        token_count: int = 1,
    ) -> str:
        """
        Record a timestep.

        Args:
            event_type: Type of event
            content: The content (token, message, etc.)
            concept_activations: Top-k concept activations
            event_id: Groups related timesteps
            event_start: Is this the start of an event?
            event_end: Is this the end of an event?
            token_id: For OUTPUT events, the token ID
            role: user | assistant | system | tool
            token_count: Number of tokens (for context tracking)

        Returns:
            The timestep ID
        """
        if not self.xdb_id:
            self.start_session()

        self.current_tick += 1

        record = TimestepRecord(
            id=f"ts-{self.xdb_id}-{self.current_tick}",
            xdb_id=self.xdb_id,
            tick=self.current_tick,
            timestamp=datetime.now(),
            event_type=event_type,
            content=content,
            concept_activations=concept_activations or {},
            event_id=event_id,
            event_start=event_start,
            event_end=event_end,
            token_id=token_id,
            role=role,
        )

        ts_id = self.experience_log.record_timestep(record)

        # Track context usage
        if self.context_manager:
            self.context_manager.add_tokens(token_count, self.current_tick)

        return ts_id

    def record_input(
        self,
        content: str,
        role: str = "user",
        concept_activations: Optional[Dict[str, float]] = None,
    ) -> str:
        """Record an input message."""
        return self.record(
            EventType.INPUT,
            content,
            concept_activations,
            role=role,
            token_count=len(content.split()),  # Approximate
        )

    def record_output(
        self,
        content: str,
        token_id: Optional[int] = None,
        concept_activations: Optional[Dict[str, float]] = None,
    ) -> str:
        """Record an output token/message."""
        return self.record(
            EventType.OUTPUT,
            content,
            concept_activations,
            token_id=token_id,
            role="assistant",
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> str:
        """Record a tool call."""
        content = json.dumps({'tool': tool_name, 'arguments': arguments})
        return self.record(
            EventType.TOOL_CALL,
            content,
            event_id=event_id or f"tool-{uuid4().hex[:8]}",
            event_start=True,
            role="assistant",
        )

    def record_tool_response(
        self,
        response: str,
        event_id: Optional[str] = None,
    ) -> str:
        """Record a tool response."""
        return self.record(
            EventType.TOOL_RESPONSE,
            response,
            event_id=event_id,
            event_end=True,
            role="tool",
            token_count=len(response.split()),
        )

    # =========================================================================
    # Tagging
    # =========================================================================

    def tag(
        self,
        tag_name_or_id: str,
        *,
        timestep_id: Optional[str] = None,
        event_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        confidence: float = 1.0,
        note: Optional[str] = None,
    ) -> str:
        """
        Apply a tag to experience.

        Tag can be specified by name or ID.
        Must provide exactly one of: timestep_id, event_id, or tick_range.

        Args:
            tag_name_or_id: Tag name or ID
            timestep_id: Tag a specific timestep
            event_id: Tag all timesteps in an event
            tick_range: Tag a range of ticks
            confidence: Confidence in the tag (0-1)
            note: Optional note about this tagging

        Returns:
            The tag application ID
        """
        # Find or create tag
        tag = self.tag_index.get_tag(tag_name_or_id)
        if not tag:
            tag = self.tag_index.get_by_name(tag_name_or_id)
        if not tag:
            # Create new custom tag
            tag = self.tag_index.create_tag(
                tag_name_or_id,
                TagType.CUSTOM,
            )

        # Determine target
        if timestep_id:
            target_type = TargetType.TIMESTEP
        elif event_id:
            target_type = TargetType.EVENT
        elif tick_range:
            target_type = TargetType.RANGE
        else:
            raise ValueError("Must specify timestep_id, event_id, or tick_range")

        application = TagApplication(
            id=TagApplication.generate_id(),
            tag_id=tag.id,
            target_type=target_type,
            timestep_id=timestep_id,
            event_id=event_id,
            range_start=tick_range[0] if tick_range else None,
            range_end=tick_range[1] if tick_range else None,
            xdb_id=self.xdb_id or "",
            confidence=confidence,
            source=TagSource.MANUAL,
            created_at=datetime.now(),
            note=note,
        )

        return self.experience_log.add_tag(application)

    def create_entity_tag(
        self,
        name: str,
        entity_type: str,
        *,
        description: Optional[str] = None,
    ) -> Tag:
        """Create a tag for tracking an entity across experiences."""
        return self.tag_index.create_tag(
            name,
            TagType.ENTITY,
            entity_type=entity_type,
            description=description,
        )

    def create_bud_tag(
        self,
        name: str,
        description: str,
        *,
        related_concepts: Optional[List[str]] = None,
    ) -> Tag:
        """Create a bud tag for potential concept learning."""
        return self.tag_index.create_tag(
            name,
            TagType.BUD,
            description=description,
        )

    # =========================================================================
    # Querying
    # =========================================================================

    def recall(
        self,
        *,
        tags: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        text_search: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_types: Optional[List[EventType]] = None,
        xdb_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TimestepRecord]:
        """
        Query experience memory.

        All filters are ANDed together.

        Args:
            tags: Filter by tag names/IDs
            concepts: Filter by concept IDs with high activation
            text_search: Search in content
            tick_range: Filter by tick range
            time_range: Filter by time range
            event_types: Filter by event types
            xdb_id: Filter by XDB (defaults to current)
            limit: Max results

        Returns:
            List of matching timestep records
        """
        # Build concept activation filter
        concept_activations = None
        if concepts:
            concept_activations = {c: (0.5, 1.0) for c in concepts}

        return self.experience_log.query(
            xdb_id=xdb_id or self.xdb_id,
            tick_range=tick_range,
            time_range=time_range,
            event_types=event_types,
            tags=tags,
            concept_activations=concept_activations,
            text_search=text_search,
            limit=limit,
        )

    def recall_by_concept(
        self,
        concept_id: str,
        *,
        min_activation: float = 0.5,
        limit: int = 100,
    ) -> List[TimestepRecord]:
        """Find experiences where a concept was strongly active."""
        return self.experience_log.query(
            concept_activations={concept_id: (min_activation, 1.0)},
            limit=limit,
        )

    def recall_recent(self, n: int = 100) -> List[TimestepRecord]:
        """Get the N most recent timesteps."""
        if not self.xdb_id:
            return []
        return self.experience_log.get_recent(self.xdb_id, n)

    def recall_surprising(self, limit: int = 50) -> List[TimestepRecord]:
        """Find experiences marked as surprising/confusing/interesting."""
        return self.recall(
            tags=["surprising", "confusing", "interesting", "unexpected"],
            limit=limit,
        )

    # =========================================================================
    # Commentary
    # =========================================================================

    def comment(
        self,
        content: str,
        *,
        timestep_id: Optional[str] = None,
        event_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
    ) -> str:
        """Add commentary to experience."""
        if timestep_id:
            target_type = TargetType.TIMESTEP
        elif event_id:
            target_type = TargetType.EVENT
        elif tick_range:
            target_type = TargetType.RANGE
        else:
            raise ValueError("Must specify target")

        comment = Comment(
            id=Comment.generate_id(),
            xdb_id=self.xdb_id or "",
            target_type=target_type,
            timestep_id=timestep_id,
            event_id=event_id,
            range_start=tick_range[0] if tick_range else None,
            range_end=tick_range[1] if tick_range else None,
            content=content,
            created_at=datetime.now(),
        )

        return self.experience_log.add_comment(comment)

    # =========================================================================
    # Concept Navigation
    # =========================================================================

    def browse_concepts(
        self,
        parent_concept_id: Optional[str] = None,
    ) -> List[Tag]:
        """
        Browse the concept hierarchy.

        If parent_concept_id is None, returns root concepts.
        """
        if parent_concept_id:
            return self.tag_index.get_concept_children(parent_concept_id)
        else:
            return self.tag_index.find_tags(tag_type=TagType.CONCEPT, limit=100)

    def find_concept(self, query: str) -> List[Tag]:
        """Search for concepts by name."""
        return self.tag_index.find_tags(
            name_pattern=query,
            tag_type=TagType.CONCEPT,
        )

    # =========================================================================
    # Bud Management
    # =========================================================================

    def get_buds(self, status: Optional[BudStatus] = None) -> List[Tag]:
        """Get bud candidates."""
        return self.tag_index.get_buds(status=status)

    def get_bud_examples(self, bud_tag_id: str) -> List[TimestepRecord]:
        """Get all examples for a bud."""
        return self.tag_index.get_bud_examples(
            bud_tag_id,
            experience_log=self.experience_log,
        )

    def mark_bud_ready(self, bud_tag_id: str) -> Optional[Tag]:
        """Mark a bud as ready for training attempt."""
        return self.tag_index.update_bud_status(bud_tag_id, BudStatus.READY)

    # =========================================================================
    # Documents
    # =========================================================================

    def get_document(self, path: str) -> Optional[Document]:
        """Get a reference document by path."""
        return self.documents.get_by_path(path)

    def search_documents(
        self,
        query: str,
        *,
        doc_type: Optional[DocumentType] = None,
    ) -> List[Document]:
        """Search reference documents."""
        return self.documents.search(
            text_query=query,
            doc_type=doc_type,
        )

    def list_documents(
        self,
        doc_type: Optional[DocumentType] = None,
    ) -> List[Document]:
        """List available documents."""
        if doc_type:
            return self.documents.list_by_type(doc_type)
        return list(self.documents.documents.values())

    def add_document(
        self,
        name: str,
        content: str,
        doc_type: DocumentType,
        *,
        path: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        """Add a reference document."""
        return self.documents.add_document(
            name=name,
            content=content,
            doc_type=doc_type,
            path=path,
            tags=tags,
        )

    # =========================================================================
    # Fidelity Management
    # =========================================================================

    def pin_for_training(
        self,
        timestep_ids: List[str],
        reason: str = "",
    ) -> int:
        """Pin timesteps as WARM for training data."""
        return self.storage_manager.pin_to_warm(timestep_ids, reason)

    def unpin_training_data(self, timestep_ids: List[str]) -> int:
        """Unpin timesteps from WARM."""
        return self.storage_manager.unpin_from_warm(timestep_ids)

    def submit_graft_evidence(
        self,
        bud_tag_id: str,
        submission_id: str,
    ) -> int:
        """Submit evidence for a graft and lock it."""
        examples = self.get_bud_examples(bud_tag_id)
        timestep_ids = [e.id for e in examples]
        return self.storage_manager.submit_evidence(timestep_ids, submission_id)

    def get_warm_quota(self) -> Dict[str, int]:
        """Get WARM quota status."""
        return {
            'used': self.storage_manager.get_warm_usage(),
            'quota': self.storage_manager.config.warm_quota_tokens,
            'remaining': self.storage_manager.get_warm_quota_remaining(),
        }

    # =========================================================================
    # Context
    # =========================================================================

    def get_context_state(self) -> Dict[str, Any]:
        """Get current context window state."""
        if self.context_manager:
            return self.context_manager.get_state()
        return {'error': 'No active session'}

    def request_compaction(self) -> Optional[CompactionRecord]:
        """Manually request context compaction."""
        if self.context_manager:
            return self.context_manager._trigger_compaction()
        return None

    # =========================================================================
    # State
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get XDB state summary."""
        return {
            'be_id': self.be_id,
            'xdb_id': self.xdb_id,
            'current_tick': self.current_tick,
            'tag_stats': self.tag_index.get_stats(),
            'experience_stats': self.experience_log.get_stats(),
            'storage_stats': self.storage_manager.get_stats(),
            'document_count': len(self.documents.documents),
            'context': self.get_context_state(),
        }

    def run_maintenance(self):
        """Run maintenance tasks (compression, cleanup)."""
        self.storage_manager.check_and_compress()
        self.storage_manager.run_scheduled_compression()
