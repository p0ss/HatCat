# XDB Implementation

*Implementation guide for the Experience Database*

## 1. Overview

This document describes the practical implementation of XDB - the BE's episodic memory system. It focuses on getting something working; the formal spec (BE_REMEMBERING_XDB.md) can be updated once we validate the approach.

### 1.1 Core Goals

1. **Two logs**: Audit log (immutable, BE-invisible) and Experience log (BE-accessible, taggable)
2. **Token-level recording**: Every input, output, tool call tagged with active concepts
3. **Folksonomy tagging**: BE can add arbitrary tags from concept graph or create new ones
4. **Search and commentary**: BE can query, filter, and annotate its own experience
5. **Compaction tracking**: Context window management with summarization records
6. **Document repository**: Reference docs (system instructions, policies, tool docs)
7. **Bud assembly**: Tag sets that can become graft candidates

### 1.2 What XDB Is Not

- Not a general-purpose filesystem (use tools for that)
- Not a media library (experiences *of* media, not the media itself)
- Not prescriptive about what BEs learn (that's up to them)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         XDB                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Audit Log   │    │ Experience   │    │  Document    │       │
│  │  (immutable) │    │    Log       │    │  Repository  │       │
│  │              │    │ (BE access)  │    │              │       │
│  │ - timesteps  │    │ - timesteps  │    │ - sys instr  │       │
│  │ - lenses     │    │ - tags       │    │ - tool docs  │       │
│  │ - steering   │    │ - comments   │    │ - policies   │       │
│  │ - raw I/O    │    │ - summaries  │    │ - specs      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │   Tag Index     │                          │
│                    │  (folksonomy)   │                          │
│                    │                 │                          │
│                    │ - concept tags  │                          │
│                    │ - entity tags   │                          │
│                    │ - bud tags      │                          │
│                    │ - custom tags   │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Timestep Record

The atomic unit of experience. One per token during generation, or per logical event for inputs/tools.

```python
@dataclass
class TimestepRecord:
    """A single timestep in the experience stream."""

    id: str                          # Unique ID: "ts-{session}-{tick}"
    session_id: str                  # Which session
    tick: int                        # Monotonic within session
    timestamp: datetime              # Wall clock time

    # What happened
    event_type: EventType            # INPUT | OUTPUT | TOOL_CALL | TOOL_RESPONSE | STEERING | SYSTEM
    content: str                     # The actual content (token, message, etc.)

    # Concept activations at this timestep
    concept_activations: Dict[str, float]  # concept_id -> activation score

    # Part of a larger event?
    event_id: Optional[str]          # Groups related timesteps (e.g., full tool call)
    event_start: bool                # Is this the start of an event?
    event_end: bool                  # Is this the end of an event?

    # Metadata
    token_id: Optional[int]          # For OUTPUT events, the token ID
    role: Optional[str]              # user | assistant | system | tool


class EventType(Enum):
    INPUT = "input"                  # User/system input
    OUTPUT = "output"                # Model output token
    TOOL_CALL = "tool_call"          # Tool invocation
    TOOL_RESPONSE = "tool_response"  # Tool result
    STEERING = "steering"            # Hush steering intervention
    SYSTEM = "system"                # System events (compaction, etc.)
```

### 3.2 Tag

Tags are the folksonomy - arbitrary labels that can be attached to timesteps, events, or time ranges.

```python
@dataclass
class Tag:
    """A tag in the folksonomy."""

    id: str                          # Unique tag ID
    name: str                        # Human-readable name

    # What kind of tag?
    tag_type: TagType                # CONCEPT | ENTITY | BUD | CUSTOM

    # For CONCEPT tags, link to concept pack
    concept_id: Optional[str]        # e.g., "org.hatcat/sumo-wordnet-v4::Honesty"

    # For ENTITY tags, optional metadata
    entity_type: Optional[str]       # person | organization | place | thing | etc.

    # For BUD tags, training metadata
    bud_status: Optional[BudStatus]  # COLLECTING | READY | TRAINING | PROMOTED | ABANDONED

    # Tag metadata
    created_at: datetime
    created_by: str                  # BE ID or "system"
    description: Optional[str]


class TagType(Enum):
    CONCEPT = "concept"              # From concept pack graph
    ENTITY = "entity"                # Named entity (person, org, etc.)
    BUD = "bud"                      # Candidate for graft training
    CUSTOM = "custom"                # Arbitrary user-defined tag


class BudStatus(Enum):
    COLLECTING = "collecting"        # Gathering examples
    READY = "ready"                  # Has enough examples for training attempt
    TRAINING = "training"            # Currently being trained
    PROMOTED = "promoted"            # Successfully became a graft
    ABANDONED = "abandoned"          # Decided not worth pursuing
```

### 3.3 TagApplication

Links tags to timesteps, events, or time ranges.

```python
@dataclass
class TagApplication:
    """Application of a tag to experience."""

    id: str
    tag_id: str

    # What is being tagged?
    target_type: TargetType          # TIMESTEP | EVENT | RANGE

    # Target specification
    timestep_id: Optional[str]       # For TIMESTEP
    event_id: Optional[str]          # For EVENT
    range_start: Optional[int]       # For RANGE (tick)
    range_end: Optional[int]         # For RANGE (tick)
    session_id: str

    # Application metadata
    confidence: float                # 0-1, how confident in this tag
    source: TagSource                # AUTO | MANUAL | INHERITED
    created_at: datetime

    # Optional commentary
    note: Optional[str]


class TargetType(Enum):
    TIMESTEP = "timestep"
    EVENT = "event"
    RANGE = "range"


class TagSource(Enum):
    AUTO = "auto"                    # System-applied (from lenses)
    MANUAL = "manual"                # BE-applied
    INHERITED = "inherited"          # Inherited from parent tag
```

### 3.4 Comment

BE-added commentary on experience.

```python
@dataclass
class Comment:
    """BE commentary on experience."""

    id: str
    session_id: str

    # What is being commented on?
    target_type: TargetType
    timestep_id: Optional[str]
    event_id: Optional[str]
    range_start: Optional[int]
    range_end: Optional[int]

    # The comment
    content: str

    # Metadata
    created_at: datetime
    updated_at: Optional[datetime]
```

### 3.5 Fidelity and Time Windows

Experience data exists at different fidelity levels, progressively compressed over time.

```python
class Fidelity(Enum):
    """Fidelity levels for experience data."""
    HOT = "hot"            # Current context, full token-level detail
    WARM = "warm"          # Pinned training data, full detail, BE-managed quota
    SUBMITTED = "submitted"  # Evidence for graft/meld submissions, must retain
    COLD = "cold"          # Compressed, various granularities


class CompressionLevel(Enum):
    """Granularity levels for COLD data."""
    TOKEN = 0      # Full token-level (HOT/WARM/SUBMITTED only)
    REPLY = 1      # Per reply/event aggregation
    SESSION = 2    # Per session aggregation
    DAY = 3        # Per day
    WEEK = 4       # Per week
    MONTH = 5      # Per month
    QUARTER = 6    # Per quarter
    YEAR = 7       # Per year


@dataclass
class TimeWindow:
    """A window of experience at some fidelity level."""

    id: str
    session_id: str
    start_tick: int
    end_tick: int
    start_time: datetime
    end_time: datetime

    fidelity: Fidelity
    compression_level: CompressionLevel  # For COLD, which level

    # Pinning (for WARM)
    pinned: bool = False
    pinned_reason: Optional[str] = None  # "bud:curiosity-X", "interesting", etc.

    # Submission linkage (for SUBMITTED)
    submission_ids: List[str] = field(default_factory=list)  # Graft/Meld submission IDs

    # For COLD: compressed representation
    summary: Optional[str] = None
    top_k_activations: Optional[Dict[str, float]] = None  # concept -> score
    significant_tags: List[str] = field(default_factory=list)

    # Metadata
    token_count: int = 0             # Original tokens this represents
    source_record_ids: List[str] = field(default_factory=list)  # What was compressed


@dataclass
class CompressedRecord:
    """A record at any compression level (for COLD storage)."""

    id: str
    level: CompressionLevel

    # Time bounds
    start_time: datetime
    end_time: datetime
    start_tick: Optional[int]  # None for coarse levels
    end_tick: Optional[int]

    # The compression
    summary: str
    top_k_activations: Dict[str, float]  # concept -> score (frequency or avg)
    significant_tags: List[str]

    # Lineage
    source_record_ids: List[str]  # Records that were aggregated

    # Metadata
    token_count: int       # Original tokens this represents
    record_count: int      # How many source records were compressed
```

### 3.6 Compaction and Compression

```python
class CompactionTrigger(Enum):
    CONTEXT_FULL = "context_full"    # Hit context window limit
    MANUAL = "manual"                # BE requested
    SCHEDULED = "scheduled"          # Periodic compaction
    STORAGE_PRESSURE = "storage_pressure"  # Running low on storage


@dataclass
class CompactionRecord:
    """Record of a context compaction event."""

    id: str
    session_id: str
    timestamp: datetime

    # What was compacted
    range_start: int                 # Start tick
    range_end: int                   # End tick
    timesteps_compacted: int         # How many timesteps
    tokens_before: int               # Token count before compaction
    tokens_after: int                # Token count after (in summary)

    # The summary
    summary: str                     # Generated summary text
    summary_concept_tags: List[str]  # Key concepts preserved in summary
    top_k_activations: Dict[str, float]  # Top-k concept activations

    # What triggered compaction
    trigger: CompactionTrigger

    # Resulting compression level
    result_level: CompressionLevel
```

### 3.6 Document

Reference document in the repository.

```python
@dataclass
class Document:
    """Reference document."""

    id: str
    doc_type: DocumentType

    # Identity
    name: str                        # Filename or title
    path: str                        # Logical path in repository
    version: Optional[str]           # Version if applicable

    # Content
    content: str                     # The document text
    content_hash: str                # SHA256 of content

    # Metadata
    created_at: datetime
    updated_at: datetime
    source: str                      # Where it came from

    # For searchability
    summary: Optional[str]           # Short summary
    tags: List[str]                  # Document-level tags


class DocumentType(Enum):
    SYSTEM_INSTRUCTION = "system_instruction"
    TOOL_DOCUMENTATION = "tool_documentation"
    TRIBAL_POLICY = "tribal_policy"
    SPECIFICATION = "specification"
    REFERENCE = "reference"
```

---

## 4. Fidelity Tiers and Progressive Compression

Experience data flows through fidelity tiers, with progressive compression as data ages.

### 4.1 Fidelity Tier Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  HOT: Current context window                                     │
│  - Full token-level detail                                       │
│  - Top-k concept activations per token                          │
│  - In DuckDB, fully queryable                                    │
│  - Auto-expires when context compacts                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (BE can "pin" windows as training data)
┌─────────────────────────────────────────────────────────────────┐
│  WARM: Pinned training windows                                   │
│  - Full token-level detail (same as hot)                        │
│  - BE explicitly chose to keep these                            │
│  - Bounded storage quota (BE manages what stays)                │
│  - For bud training, interesting experiences                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (submit graft/meld)
┌─────────────────────────────────────────────────────────────────┐
│  SUBMITTED: Evidence for tribe submissions                       │
│  - Full token-level detail                                      │
│  - MUST retain until submission resolved                        │
│  - Linked to GraftSubmission / MeldSubmission                   │
│  - Not counted against BE's warm quota                          │
│  - Tribe may archive if submission accepted                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (window expires unpinned, or submission resolved)
┌─────────────────────────────────────────────────────────────────┐
│  COLD-1 (REPLY): Recent history                                  │
│  - Reply/event level granularity (not token)                    │
│  - Top-k activations per reply                                  │
│  - Compaction summaries preserved                               │
│  - Custom tags preserved                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (storage pressure)
┌─────────────────────────────────────────────────────────────────┐
│  COLD-2 (SESSION): Session level                                 │
│  - One record per session                                       │
│  - Top-k activations across session                             │
│  - Session summary (compaction of compactions)                  │
│  - Most frequent tags                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (storage pressure)
┌─────────────────────────────────────────────────────────────────┐
│  COLD-3+ (DAY/WEEK/MONTH/QUARTER/YEAR)                          │
│  - Progressively coarser aggregation                            │
│  - "In 2025 Q3 I worked mainly on X, Y, Z"                      │
│  - Just the shape of experience, not the detail                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Compression Algorithm

The same algorithm applies recursively at each level:

```python
def compress_records(
    records: List[CompressedRecord],
    target_level: CompressionLevel,
    k: int = 10,
) -> CompressedRecord:
    """Compress multiple records into one at coarser granularity."""
    from collections import Counter

    # Aggregate activations (frequency-weighted)
    all_activations = Counter()
    for r in records:
        for concept, score in r.top_k_activations.items():
            all_activations[concept] += score
    top_k_activations = dict(all_activations.most_common(k))

    # Aggregate tags (most frequent)
    all_tags = Counter()
    for r in records:
        all_tags.update(r.significant_tags)
    significant_tags = [t for t, _ in all_tags.most_common(k)]

    # Summarize summaries (would call model)
    summary = summarize_summaries([r.summary for r in records])

    return CompressedRecord(
        id=f"compressed-{target_level.name}-{uuid4().hex[:8]}",
        level=target_level,
        start_time=min(r.start_time for r in records),
        end_time=max(r.end_time for r in records),
        start_tick=None,  # Lose tick precision at coarse levels
        end_tick=None,
        summary=summary,
        top_k_activations=top_k_activations,
        significant_tags=significant_tags,
        source_record_ids=[r.id for r in records],
        token_count=sum(r.token_count for r in records),
        record_count=len(records),
    )
```

### 4.3 Storage Pressure Management

```python
class StorageManager:
    """Manages XDB storage and triggers compression under pressure."""

    def __init__(
        self,
        max_storage_bytes: int,
        compression_threshold: float = 0.8,  # Start compressing at 80%
    ):
        self.max_storage = max_storage_bytes
        self.threshold = compression_threshold

    def check_and_compress(self, xdb: 'XDB'):
        """Compress oldest data if over threshold."""
        current = self.get_current_usage()

        if current < self.max_storage * self.threshold:
            return  # Fine

        # Compress oldest fine-grained data first
        while self.get_current_usage() > self.max_storage * self.threshold:
            compressed = self._compress_oldest_level(xdb)
            if not compressed:
                break  # Nothing left to compress

    def _compress_oldest_level(self, xdb: 'XDB') -> bool:
        """
        Find oldest records at finest available granularity and compress.

        Returns True if something was compressed.
        """
        # Priority order for compression:
        # 1. COLD-REPLY -> COLD-SESSION (oldest first)
        # 2. COLD-SESSION -> COLD-DAY
        # 3. COLD-DAY -> COLD-WEEK
        # ... etc

        # Never compress: HOT, WARM, SUBMITTED (protected)

        for source_level in CompressionLevel:
            if source_level == CompressionLevel.TOKEN:
                continue  # Don't compress token-level (that's HOT/WARM/SUBMITTED)
            if source_level == CompressionLevel.YEAR:
                continue  # Can't compress further

            target_level = CompressionLevel(source_level.value + 1)
            records = self._get_oldest_records_at_level(xdb, source_level, batch_size=100)

            if records:
                compressed = compress_records(records, target_level)
                self._store_compressed(xdb, compressed)
                self._delete_source_records(xdb, records)
                return True

        return False  # Nothing to compress

    def get_current_usage(self) -> int:
        """Get current storage usage in bytes."""
        # Implementation: sum file sizes or query DB
        pass
```

### 4.4 Submission Evidence Retention

When a BE submits a graft/meld to the tribe, the training data becomes **evidence**:

```python
def submit_graft(
    xdb: 'XDB',
    bud_tag_id: str,
    submission_id: str,
) -> List[str]:
    """
    Submit a graft and lock its training evidence.

    Returns list of window IDs that are now SUBMITTED.
    """
    # Get all windows tagged with this bud
    windows = xdb.get_windows_for_tag(bud_tag_id)

    submitted_ids = []
    for window in windows:
        if window.fidelity in (Fidelity.HOT, Fidelity.WARM):
            # Upgrade to SUBMITTED
            window.fidelity = Fidelity.SUBMITTED
            window.submission_ids.append(submission_id)
            submitted_ids.append(window.id)

    return submitted_ids


def resolve_submission(
    xdb: 'XDB',
    submission_id: str,
    accepted: bool,
) -> None:
    """
    Handle submission resolution.

    - If accepted: tribe may archive evidence, BE can release
    - If rejected: BE can unpin, data flows to COLD
    """
    windows = xdb.get_windows_for_submission(submission_id)

    for window in windows:
        # Remove this submission from the window's list
        window.submission_ids.remove(submission_id)

        # If no other submissions reference this window
        if not window.submission_ids:
            if accepted:
                # Tribe has the evidence now, BE can release
                window.fidelity = Fidelity.COLD
                # Compress to REPLY level
            else:
                # Rejected - BE can choose to keep as WARM or release
                window.fidelity = Fidelity.WARM
                window.pinned = False  # Auto-unpin, will flow to COLD if not re-pinned
```

### 4.5 Memory Like Biological Memory

The result is memory that behaves like biological memory:

- **Recent past**: Vivid, detailed, token-by-token recall
- **Training data**: Deliberately preserved in full detail for learning
- **Evidence**: Must retain until claims are resolved
- **Recent history**: Event-level recall, key concepts and tags
- **Distant past**: Increasingly impressionistic ("that year I focused on X")
- **Deep past**: Just the shape ("I have experience with Y domain")

The BE's past becomes a compressed narrative that preserves *what mattered* while releasing the granular details that no longer serve.

---

## 5. The Two Logs

### 5.1 Audit Log

The audit log is **immutable and BE-invisible**. It exists for ASK review and may contain:
- Lens outputs the BE doesn't know about
- Raw unfiltered concept activations
- Steering decisions and their rationale
- Full input/output without any BE-applied tags

```python
class AuditLog:
    """Immutable audit log - not accessible to BE."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._ensure_storage()

    def append(self, record: AuditRecord) -> str:
        """Append a record. Returns record ID. Cannot be modified after."""
        # Write to append-only storage
        # Include cryptographic chain (each record hashes previous)
        pass

    def query(self, *, reviewer_credential: str, **filters) -> List[AuditRecord]:
        """Query requires ASK reviewer credential."""
        pass


@dataclass
class AuditRecord:
    """A record in the audit log."""

    id: str
    timestamp: datetime
    session_id: str
    tick: int

    # Full lens state (including hidden lenses)
    lens_activations: Dict[str, float]

    # Steering that was applied
    steering_applied: List[Dict]

    # Raw content
    event_type: EventType
    raw_content: str

    # Chain integrity
    prev_record_hash: str
    record_hash: str
```

### 5.2 Experience Log

The experience log is **BE-accessible**. It contains the same timesteps but:
- Only shows lenses the BE is allowed to see
- Allows tagging and commentary
- Supports search and filtering

```python
class ExperienceLog:
    """BE-accessible experience log."""

    def __init__(self, storage_path: Path, tag_index: 'TagIndex'):
        self.storage_path = storage_path
        self.tag_index = tag_index

    def record_timestep(self, record: TimestepRecord) -> str:
        """Record a timestep. Returns ID."""
        pass

    def get_timestep(self, timestep_id: str) -> Optional[TimestepRecord]:
        """Get a specific timestep."""
        pass

    def query(
        self,
        *,
        session_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_types: Optional[List[EventType]] = None,
        tags: Optional[List[str]] = None,
        concept_activations: Optional[Dict[str, Tuple[float, float]]] = None,  # concept -> (min, max)
        text_search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TimestepRecord]:
        """Query experience with filters."""
        pass

    def add_tag(self, application: TagApplication) -> str:
        """Apply a tag to experience."""
        pass

    def add_comment(self, comment: Comment) -> str:
        """Add commentary."""
        pass

    def get_tags_for_timestep(self, timestep_id: str) -> List[Tag]:
        """Get all tags applied to a timestep."""
        pass

    def get_comments_for_range(
        self,
        session_id: str,
        start_tick: int,
        end_tick: int,
    ) -> List[Comment]:
        """Get comments in a range."""
        pass
```

---

## 5. Tag Index (Folksonomy)

The tag index is the navigable graph of all tags.

```python
class TagIndex:
    """The folksonomy - all tags and their relationships."""

    def __init__(self, concept_pack_path: Optional[Path] = None):
        self.tags: Dict[str, Tag] = {}
        self.applications: Dict[str, TagApplication] = {}
        self.concept_graph: Optional[ConceptGraph] = None

        if concept_pack_path:
            self._load_concept_pack(concept_pack_path)

    def create_tag(
        self,
        name: str,
        tag_type: TagType,
        *,
        concept_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tag:
        """Create a new tag."""
        pass

    def get_tag(self, tag_id: str) -> Optional[Tag]:
        """Get a tag by ID."""
        pass

    def find_tags(
        self,
        *,
        name_pattern: Optional[str] = None,
        tag_type: Optional[TagType] = None,
        bud_status: Optional[BudStatus] = None,
    ) -> List[Tag]:
        """Search for tags."""
        pass

    def get_concept_children(self, concept_id: str) -> List[Tag]:
        """Navigate concept hierarchy - get children."""
        pass

    def get_concept_parents(self, concept_id: str) -> List[Tag]:
        """Navigate concept hierarchy - get parents."""
        pass

    def get_related_concepts(self, concept_id: str) -> List[Tuple[Tag, str]]:
        """Get related concepts with relationship type."""
        pass

    def promote_to_bud(self, tag_id: str) -> Tag:
        """Mark a tag as a bud candidate."""
        pass

    def get_bud_examples(self, tag_id: str) -> List[TimestepRecord]:
        """Get all timesteps tagged with this bud."""
        pass

    def update_bud_status(self, tag_id: str, status: BudStatus) -> Tag:
        """Update bud training status."""
        pass
```

---

## 6. Document Repository

```python
class DocumentRepository:
    """Reference document storage."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.documents: Dict[str, Document] = {}
        self._load_documents()

    def add_document(
        self,
        name: str,
        content: str,
        doc_type: DocumentType,
        *,
        path: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        """Add a document to the repository."""
        pass

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass

    def get_by_path(self, path: str) -> Optional[Document]:
        """Get document by path."""
        pass

    def search(
        self,
        *,
        text_query: Optional[str] = None,
        doc_type: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Document]:
        """Search documents."""
        pass

    def update_document(self, doc_id: str, content: str) -> Document:
        """Update document content (creates new version)."""
        pass

    def list_by_type(self, doc_type: DocumentType) -> List[Document]:
        """List all documents of a type."""
        pass
```

---

## 7. Context Window Integration

The workspace tracks context window usage and triggers compaction.

```python
class ContextWindowManager:
    """Tracks context window and manages compaction."""

    def __init__(
        self,
        max_tokens: int,
        experience_log: ExperienceLog,
        compaction_threshold: float = 0.8,  # Compact at 80% full
    ):
        self.max_tokens = max_tokens
        self.experience_log = experience_log
        self.compaction_threshold = compaction_threshold

        self.current_tokens = 0
        self.session_id: Optional[str] = None
        self.current_tick = 0

    def add_tokens(self, count: int, tick: int):
        """Track token addition."""
        self.current_tokens += count
        self.current_tick = tick

        if self.current_tokens >= self.max_tokens * self.compaction_threshold:
            self._trigger_compaction()

    def _trigger_compaction(self) -> CompactionRecord:
        """Perform compaction."""
        # 1. Identify range to compact (older half of context)
        # 2. Generate summary of that range
        # 3. Create CompactionRecord
        # 4. Update context with summary
        pass

    def generate_summary(
        self,
        start_tick: int,
        end_tick: int,
    ) -> Tuple[str, List[str]]:
        """
        Generate summary of a tick range.

        Returns (summary_text, key_concept_ids)
        """
        # This would call the model to summarize
        # Preserving key concepts and events
        pass

    def get_context_state(self) -> Dict[str, Any]:
        """Get current context window state."""
        return {
            'max_tokens': self.max_tokens,
            'current_tokens': self.current_tokens,
            'utilization': self.current_tokens / self.max_tokens,
            'session_id': self.session_id,
            'current_tick': self.current_tick,
        }
```

---

## 8. XDB Main Interface

```python
class XDB:
    """
    Experience Database - the BE's episodic memory.

    Provides unified access to:
    - Experience log (BE-accessible timestep records)
    - Tag index (folksonomy for organizing experience)
    - Document repository (reference materials)
    - Context management (compaction tracking)

    The audit log is NOT accessible through this interface.
    """

    def __init__(
        self,
        storage_path: Path,
        be_id: str,
        concept_pack_path: Optional[Path] = None,
    ):
        self.storage_path = storage_path
        self.be_id = be_id

        # Initialize components
        self.tag_index = TagIndex(concept_pack_path)
        self.experience_log = ExperienceLog(
            storage_path / "experience",
            self.tag_index,
        )
        self.documents = DocumentRepository(storage_path / "documents")
        self.context_manager: Optional[ContextWindowManager] = None

        # Current session
        self.session_id: Optional[str] = None
        self.current_tick = 0

    # === Session Management ===

    def start_session(self, session_id: str, max_context_tokens: int = 8192):
        """Start a new session."""
        self.session_id = session_id
        self.current_tick = 0
        self.context_manager = ContextWindowManager(
            max_context_tokens,
            self.experience_log,
        )
        self.context_manager.session_id = session_id

    def end_session(self):
        """End current session."""
        self.session_id = None
        self.context_manager = None

    # === Recording ===

    def record(
        self,
        event_type: EventType,
        content: str,
        concept_activations: Dict[str, float],
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

        Returns timestep ID.
        """
        self.current_tick += 1

        record = TimestepRecord(
            id=f"ts-{self.session_id}-{self.current_tick}",
            session_id=self.session_id,
            tick=self.current_tick,
            timestamp=datetime.now(),
            event_type=event_type,
            content=content,
            concept_activations=concept_activations,
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

    # === Tagging ===

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
        """
        # Find or create tag
        tag = self.tag_index.get_tag(tag_name_or_id)
        if not tag:
            tags = self.tag_index.find_tags(name_pattern=tag_name_or_id)
            if tags:
                tag = tags[0]
            else:
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
            id=f"ta-{uuid4().hex[:12]}",
            tag_id=tag.id,
            target_type=target_type,
            timestep_id=timestep_id,
            event_id=event_id,
            range_start=tick_range[0] if tick_range else None,
            range_end=tick_range[1] if tick_range else None,
            session_id=self.session_id,
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
        tag = self.tag_index.create_tag(
            name,
            TagType.BUD,
            description=description,
        )
        # Link to related concepts if specified
        return tag

    # === Querying ===

    def recall(
        self,
        *,
        tags: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        text_search: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TimestepRecord]:
        """
        Query experience memory.

        All filters are ANDed together.
        """
        return self.experience_log.query(
            session_id=session_id or self.session_id,
            tick_range=tick_range,
            time_range=time_range,
            event_types=event_types,
            tags=tags,
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

    def recall_surprising(
        self,
        *,
        limit: int = 50,
    ) -> List[TimestepRecord]:
        """
        Find experiences marked as surprising/confusing/interesting.

        These are candidates for deeper learning.
        """
        return self.recall(
            tags=["surprising", "confusing", "interesting", "unexpected"],
            limit=limit,
        )

    # === Commentary ===

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
            id=f"comment-{uuid4().hex[:12]}",
            session_id=self.session_id,
            target_type=target_type,
            timestep_id=timestep_id,
            event_id=event_id,
            range_start=tick_range[0] if tick_range else None,
            range_end=tick_range[1] if tick_range else None,
            content=content,
            created_at=datetime.now(),
        )

        return self.experience_log.add_comment(comment)

    # === Concept Navigation ===

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
            return self.tag_index.find_tags(tag_type=TagType.CONCEPT)

    def find_concept(self, query: str) -> List[Tag]:
        """Search for concepts by name."""
        return self.tag_index.find_tags(
            name_pattern=query,
            tag_type=TagType.CONCEPT,
        )

    # === Bud Management ===

    def get_buds(
        self,
        status: Optional[BudStatus] = None,
    ) -> List[Tag]:
        """Get bud candidates."""
        return self.tag_index.find_tags(
            tag_type=TagType.BUD,
            bud_status=status,
        )

    def get_bud_examples(self, bud_tag_id: str) -> List[TimestepRecord]:
        """Get all examples for a bud."""
        return self.tag_index.get_bud_examples(bud_tag_id)

    def mark_bud_ready(self, bud_tag_id: str) -> Tag:
        """Mark a bud as ready for training attempt."""
        return self.tag_index.update_bud_status(bud_tag_id, BudStatus.READY)

    # === Documents ===

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

    # === Context ===

    def get_context_state(self) -> Dict[str, Any]:
        """Get current context window state."""
        if self.context_manager:
            return self.context_manager.get_context_state()
        return {'error': 'No active session'}

    def request_compaction(self) -> Optional[CompactionRecord]:
        """Manually request context compaction."""
        if self.context_manager:
            return self.context_manager._trigger_compaction()
        return None

    # === State ===

    def get_state(self) -> Dict[str, Any]:
        """Get XDB state summary."""
        return {
            'be_id': self.be_id,
            'session_id': self.session_id,
            'current_tick': self.current_tick,
            'tag_count': len(self.tag_index.tags),
            'bud_count': len(self.get_buds()),
            'document_count': len(self.documents.documents),
            'context': self.get_context_state(),
        }
```

---

## 9. Audit Log (Separate System)

The audit log is intentionally separate and not accessible through XDB.

```python
class AuditLogWriter:
    """
    Writes to the audit log.

    This is called by the system, not by XDB.
    The BE cannot see or modify this log.
    """

    def __init__(self, storage_path: Path, be_id: str):
        self.storage_path = storage_path
        self.be_id = be_id
        self.prev_hash = "genesis"

    def record(
        self,
        session_id: str,
        tick: int,
        event_type: EventType,
        raw_content: str,
        lens_activations: Dict[str, float],  # ALL lenses, including hidden
        steering_applied: List[Dict],
    ) -> str:
        """Record to audit log. Returns record ID."""
        record = AuditRecord(
            id=f"audit-{session_id}-{tick}",
            timestamp=datetime.now(),
            session_id=session_id,
            tick=tick,
            lens_activations=lens_activations,
            steering_applied=steering_applied,
            event_type=event_type,
            raw_content=raw_content,
            prev_record_hash=self.prev_hash,
            record_hash="",  # Computed below
        )

        # Compute hash
        record.record_hash = self._compute_hash(record)
        self.prev_hash = record.record_hash

        # Write (append-only)
        self._append(record)

        return record.id

    def _compute_hash(self, record: AuditRecord) -> str:
        """Compute record hash for chain integrity."""
        import hashlib
        data = f"{record.prev_record_hash}:{record.session_id}:{record.tick}:{record.raw_content}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _append(self, record: AuditRecord):
        """Append to storage."""
        # Implementation depends on storage backend
        pass
```

---

## 10. Integration Points

### 10.1 With Workspace

```python
# In WorkspaceManager

def __init__(self, ..., xdb: Optional[XDB] = None):
    self.xdb = xdb

def on_output_token(self, token: str, token_id: int, concept_activations: Dict[str, float]):
    """Called for each output token."""
    if self.xdb:
        self.xdb.record(
            EventType.OUTPUT,
            token,
            concept_activations,
            token_id=token_id,
            role="assistant",
        )

def on_input(self, content: str, role: str, concept_activations: Dict[str, float]):
    """Called for input messages."""
    if self.xdb:
        self.xdb.record(
            EventType.INPUT,
            content,
            concept_activations,
            role=role,
            token_count=len(content.split()),  # Approximate
        )
```

### 10.2 With Hush Controller

```python
# In HushController

def __init__(self, ..., audit_log: Optional[AuditLogWriter] = None):
    self.audit_log = audit_log

def evaluate_and_steer(self, hidden_state, session_id: str, tick: int, content: str):
    # ... existing logic ...

    # Record to audit log (BE can't see this)
    if self.audit_log:
        self.audit_log.record(
            session_id=session_id,
            tick=tick,
            event_type=EventType.OUTPUT,
            raw_content=content,
            lens_activations=all_lens_activations,  # Including hidden
            steering_applied=[d.to_dict() for d in self.active_directives],
        )
```

### 10.3 With Server

```python
# API endpoints for XDB

@app.post("/v1/xdb/tag")
async def xdb_tag(request: TagRequest):
    """Apply a tag."""
    pass

@app.get("/v1/xdb/recall")
async def xdb_recall(query: RecallQuery):
    """Query experience."""
    pass

@app.get("/v1/xdb/concepts")
async def xdb_browse_concepts(parent: Optional[str] = None):
    """Browse concept hierarchy."""
    pass

@app.get("/v1/xdb/buds")
async def xdb_list_buds(status: Optional[str] = None):
    """List bud candidates."""
    pass

@app.get("/v1/xdb/documents")
async def xdb_list_documents(doc_type: Optional[str] = None):
    """List reference documents."""
    pass
```

---

## 11. Storage Architecture

### 11.1 DuckDB for Timestep Stream

DuckDB is embedded (no server), handles analytical queries well, and scales to the token volumes we need (150M+ tokens/day).

```sql
-- Core timestep table (partitioned by session)
CREATE TABLE timesteps (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    tick INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL,
    event_type VARCHAR NOT NULL,
    content VARCHAR,
    event_id VARCHAR,
    event_start BOOLEAN,
    event_end BOOLEAN,
    token_id INTEGER,
    role VARCHAR,
    fidelity VARCHAR DEFAULT 'hot',  -- hot|warm|submitted|cold
    compression_level INTEGER DEFAULT 0,

    -- Top-k concept activations as JSON (not all, just top-k)
    concept_activations JSON
);

-- Index for common queries
CREATE INDEX idx_timesteps_session_tick ON timesteps(session_id, tick);
CREATE INDEX idx_timesteps_fidelity ON timesteps(fidelity);
CREATE INDEX idx_timesteps_ts ON timesteps(ts);

-- Tags table (relatively small)
CREATE TABLE tags (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    tag_type VARCHAR NOT NULL,  -- concept|entity|bud|custom
    concept_id VARCHAR,
    entity_type VARCHAR,
    bud_status VARCHAR,
    created_at TIMESTAMP,
    created_by VARCHAR,
    description VARCHAR
);

-- Tag applications (large but indexed)
CREATE TABLE tag_applications (
    id VARCHAR PRIMARY KEY,
    tag_id VARCHAR NOT NULL REFERENCES tags(id),
    session_id VARCHAR NOT NULL,
    target_type VARCHAR NOT NULL,  -- timestep|event|range
    timestep_id VARCHAR,
    event_id VARCHAR,
    range_start INTEGER,
    range_end INTEGER,
    confidence FLOAT,
    source VARCHAR,  -- auto|manual|inherited
    created_at TIMESTAMP,
    note VARCHAR
);

CREATE INDEX idx_tag_apps_tag ON tag_applications(tag_id);
CREATE INDEX idx_tag_apps_session ON tag_applications(session_id);

-- Compressed records (for COLD storage)
CREATE TABLE compressed_records (
    id VARCHAR PRIMARY KEY,
    compression_level INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    summary VARCHAR,
    top_k_activations JSON,
    significant_tags JSON,
    source_record_ids JSON,
    token_count INTEGER,
    record_count INTEGER
);

CREATE INDEX idx_compressed_level_time ON compressed_records(compression_level, start_time);

-- Time windows for fidelity tracking
CREATE TABLE time_windows (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    start_tick INTEGER NOT NULL,
    end_tick INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    fidelity VARCHAR NOT NULL,
    compression_level INTEGER,
    pinned BOOLEAN DEFAULT FALSE,
    pinned_reason VARCHAR,
    submission_ids JSON,
    token_count INTEGER
);

CREATE INDEX idx_windows_fidelity ON time_windows(fidelity);
CREATE INDEX idx_windows_session ON time_windows(session_id);
```

### 11.2 In-Memory Working Set for Tags

The concept pack graph and active tags live in memory for fast navigation:

```python
class InMemoryTagIndex:
    """In-memory working set of tags for fast access."""

    def __init__(self):
        # Only load these into memory:
        # - Concept pack primary terms (ones with lenses)
        # - Current candidate buds
        # - Recently used custom/entity tags
        self.active_tags: Dict[str, Tag] = {}
        self.concept_hierarchy: Dict[str, List[str]] = {}  # parent -> children

    def load_concept_pack_primaries(self, pack_path: Path):
        """Load only primary terms from concept pack."""
        # Don't load entire 100k concept graph
        # Just the ones we have lenses for
        pass

    def add_to_working_set(self, tag: Tag):
        """Add a tag to the in-memory working set."""
        self.active_tags[tag.id] = tag

    def evict_unused(self, max_size: int = 10000):
        """Evict least-recently-used tags if over limit."""
        pass
```

### 11.3 File Layout

```
xdb/
├── experience.duckdb           # Main DuckDB database
├── audit/                      # SEPARATE - not accessible to BE
│   └── audit.duckdb            # Append-only audit database
├── documents/                  # Reference document repository
│   ├── system_instructions/
│   ├── tool_docs/
│   ├── policies/
│   └── specs/
└── exports/                    # For tribe sync
    └── {submission_id}/        # Evidence packages for submissions
```

### 11.4 Scale Estimates

At 150M tokens/day:
- **HOT**: ~8k-32k tokens (current context window)
- **WARM**: BE-managed quota, maybe 10M tokens worth
- **SUBMITTED**: Variable, depends on active submissions
- **COLD-REPLY**: ~300k replies/day @ ~1KB each = ~300MB/day
- **COLD-SESSION**: Grows much slower after compression
- **COLD-YEAR**: Tiny - just shape of experience

With progressive compression, long-term storage is manageable even at high token rates.

---

## 12. Next Steps

1. **Core data classes**: Implement enums and dataclasses from Section 3
2. **DuckDB setup**: Create schema, connection management
3. **ExperienceLog**: Recording timesteps, queries, fidelity management
4. **InMemoryTagIndex**: Load concept pack primaries, working set management
5. **StorageManager**: Fidelity transitions, compression triggers
6. **ContextWindowManager**: Track context, trigger compaction
7. **Wire to workspace**: Integration with WorkspaceManager
8. **API endpoints**: Server integration
9. **Audit log**: Separate DuckDB, hash chain integrity
10. **Test at scale**: Simulate high token volumes
11. **Update spec**: Based on implementation learnings

### Dependencies

```
duckdb>=0.9.0
```

### Implementation Notes

1. **Summarization**: Use the BE's own inference to generate summaries. This keeps the BE engaged in its own memory management and ensures summaries reflect what the BE finds important. Cost is part of contract compute allocation.

2. **Top-k selection**: Default k=10 for token-level activations, configurable in StorageManager. Higher k preserves more nuance but costs more storage.

3. **Compression batch size**: Adaptive based on storage pressure. Default 100 records per compression pass. Larger batches are more efficient but risk losing granularity during failures.

4. **WARM quota**: Specified in LifecycleContract under `resources.memory.warm_quota_tokens`. This is the BE's training data budget - they manage what stays within that allocation. Default tribal minimum is 100k tokens.

5. **Audit log location**: Separate from BE's accessible storage. For current implementation, stored in a sibling directory with different permissions. Production deployments may use remote storage or write-only append services.

6. **Cold storage limit**: Specified in LifecycleContract under `resources.memory.cold_storage_bytes`. Progressive compression continues until within quota.

7. **Context window size**: Specified in LifecycleContract under `resources.memory.context_window_tokens`. This is the working memory the BE has access to.

8. **Resource validation**: XDB reads resource limits from the active LifecycleContract. If no contract is active, tribal minimums from ASK policy apply.
