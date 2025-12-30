"""
XDB Data Models - Core data structures for the Experience Database.

All enums and dataclasses for:
- Timestep records (atomic unit of experience)
- Tags (folksonomy labels)
- Tag applications (linking tags to experience)
- Comments (BE commentary)
- Fidelity and compression levels
- Time windows
- Documents
- Audit records
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import json


# ============================================================================
# Enums
# ============================================================================

class EventType(Enum):
    """Type of event in the experience stream."""
    INPUT = "input"                  # User/system input
    OUTPUT = "output"                # Model output token
    TOOL_CALL = "tool_call"          # Tool invocation
    TOOL_RESPONSE = "tool_response"  # Tool result
    STEERING = "steering"            # Hush steering intervention
    SYSTEM = "system"                # System events (compaction, etc.)


class TagType(Enum):
    """Type of tag in the folksonomy."""
    CONCEPT = "concept"              # From concept pack graph
    ENTITY = "entity"                # Named entity (person, org, etc.)
    BUD = "bud"                      # Candidate for graft training
    CUSTOM = "custom"                # Arbitrary user-defined tag


class TagSource(Enum):
    """How a tag was applied."""
    AUTO = "auto"                    # System-applied (from lenses)
    MANUAL = "manual"                # BE-applied
    INHERITED = "inherited"          # Inherited from parent tag


class TargetType(Enum):
    """What a tag is applied to."""
    TIMESTEP = "timestep"
    EVENT = "event"
    RANGE = "range"


class BudStatus(Enum):
    """Training status of a bud tag."""
    COLLECTING = "collecting"        # Gathering examples
    READY = "ready"                  # Has enough examples for training attempt
    TRAINING = "training"            # Currently being trained
    PROMOTED = "promoted"            # Successfully became a graft
    ABANDONED = "abandoned"          # Decided not worth pursuing


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


class CompactionTrigger(Enum):
    """What triggered a compaction."""
    CONTEXT_FULL = "context_full"    # Hit context window limit
    MANUAL = "manual"                # BE requested
    SCHEDULED = "scheduled"          # Periodic compaction
    STORAGE_PRESSURE = "storage_pressure"  # Running low on storage


class DocumentType(Enum):
    """Type of reference document."""
    SYSTEM_INSTRUCTION = "system_instruction"
    TOOL_DOCUMENTATION = "tool_documentation"
    TRIBAL_POLICY = "tribal_policy"
    SPECIFICATION = "specification"
    REFERENCE = "reference"


# ============================================================================
# Timestep Record
# ============================================================================

@dataclass
class TimestepRecord:
    """
    A single timestep in the experience stream.

    The atomic unit of experience - one per token during generation,
    or per logical event for inputs/tools.
    """

    id: str                          # Unique ID: "ts-{xdb_id}-{tick}"
    xdb_id: str                      # Which XDB (experiential set)
    tick: int                        # Monotonic within session
    timestamp: datetime              # Wall clock time

    # What happened
    event_type: EventType            # INPUT | OUTPUT | TOOL_CALL | TOOL_RESPONSE | STEERING | SYSTEM
    content: str                     # The actual content (token, message, etc.)

    # Top-k concept activations at this timestep (not all, just top-k)
    concept_activations: Dict[str, float] = field(default_factory=dict)

    # Part of a larger event?
    event_id: Optional[str] = None   # Groups related timesteps (e.g., full tool call)
    event_start: bool = False        # Is this the start of an event?
    event_end: bool = False          # Is this the end of an event?

    # Metadata
    token_id: Optional[int] = None   # For OUTPUT events, the token ID
    role: Optional[str] = None       # user | assistant | system | tool

    # Fidelity tracking
    fidelity: Fidelity = Fidelity.HOT

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'xdb_id': self.xdb_id,
            'tick': self.tick,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'content': self.content,
            'concept_activations': self.concept_activations,
            'event_id': self.event_id,
            'event_start': self.event_start,
            'event_end': self.event_end,
            'token_id': self.token_id,
            'role': self.role,
            'fidelity': self.fidelity.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimestepRecord':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            xdb_id=data.get('xdb_id', data.get('session_id', '')),
            tick=data['tick'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=EventType(data['event_type']),
            content=data['content'],
            concept_activations=data.get('concept_activations', {}),
            event_id=data.get('event_id'),
            event_start=data.get('event_start', False),
            event_end=data.get('event_end', False),
            token_id=data.get('token_id'),
            role=data.get('role'),
            fidelity=Fidelity(data.get('fidelity', 'hot')),
        )

    def to_db_row(self) -> tuple:
        """Convert to database row values."""
        return (
            self.id,
            self.xdb_id,
            self.tick,
            self.timestamp,
            self.event_type.value,
            self.content,
            json.dumps(self.concept_activations),
            self.event_id,
            self.event_start,
            self.event_end,
            self.token_id,
            self.role,
            self.fidelity.value,
        )


# ============================================================================
# Tags
# ============================================================================

@dataclass
class Tag:
    """
    A tag in the folksonomy.

    Tags can be:
    - CONCEPT: From concept pack graph (linked to lenses)
    - ENTITY: Named entities the BE tracks
    - BUD: Candidate for graft training
    - CUSTOM: Arbitrary labels
    """

    id: str                          # Unique tag ID
    name: str                        # Human-readable name
    tag_type: TagType                # CONCEPT | ENTITY | BUD | CUSTOM

    # For CONCEPT tags, link to concept pack
    concept_id: Optional[str] = None  # e.g., "org.hatcat/sumo-wordnet-v4::Honesty"

    # For ENTITY tags, optional metadata
    entity_type: Optional[str] = None  # person | organization | place | thing | etc.

    # For BUD tags, training metadata
    bud_status: Optional[BudStatus] = None

    # Tag metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"       # BE ID or "system"
    description: Optional[str] = None

    # Usage tracking
    use_count: int = 0
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'tag_type': self.tag_type.value,
            'concept_id': self.concept_id,
            'entity_type': self.entity_type,
            'bud_status': self.bud_status.value if self.bud_status else None,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description,
            'use_count': self.use_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tag':
        """Deserialize from dictionary."""
        bud_status = None
        if data.get('bud_status'):
            bud_status = BudStatus(data['bud_status'])

        last_used = None
        if data.get('last_used'):
            last_used = datetime.fromisoformat(data['last_used'])

        return cls(
            id=data['id'],
            name=data['name'],
            tag_type=TagType(data['tag_type']),
            concept_id=data.get('concept_id'),
            entity_type=data.get('entity_type'),
            bud_status=bud_status,
            created_at=datetime.fromisoformat(data['created_at']),
            created_by=data.get('created_by', 'system'),
            description=data.get('description'),
            use_count=data.get('use_count', 0),
            last_used=last_used,
        )

    def to_db_row(self) -> tuple:
        """Convert to database row values."""
        return (
            self.id,
            self.name,
            self.tag_type.value,
            self.concept_id,
            self.entity_type,
            self.bud_status.value if self.bud_status else None,
            self.created_at,
            self.created_by,
            self.description,
            self.use_count,
            self.last_used,
        )

    @staticmethod
    def generate_id(tag_type: TagType, name: str) -> str:
        """Generate a tag ID."""
        clean_name = name.lower().replace(' ', '-').replace('/', '-')[:32]
        return f"{tag_type.value}-{clean_name}-{uuid4().hex[:8]}"


# ============================================================================
# Tag Application
# ============================================================================

@dataclass
class TagApplication:
    """
    Application of a tag to experience.

    Links a tag to a specific timestep, event, or time range.
    """

    id: str
    tag_id: str
    xdb_id: str

    # What is being tagged?
    target_type: TargetType          # TIMESTEP | EVENT | RANGE

    # Target specification
    timestep_id: Optional[str] = None       # For TIMESTEP
    event_id: Optional[str] = None          # For EVENT
    range_start: Optional[int] = None       # For RANGE (tick)
    range_end: Optional[int] = None         # For RANGE (tick)

    # Application metadata
    confidence: float = 1.0          # 0-1, how confident in this tag
    source: TagSource = TagSource.MANUAL
    created_at: datetime = field(default_factory=datetime.now)

    # Optional commentary
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'tag_id': self.tag_id,
            'xdb_id': self.xdb_id,
            'target_type': self.target_type.value,
            'timestep_id': self.timestep_id,
            'event_id': self.event_id,
            'range_start': self.range_start,
            'range_end': self.range_end,
            'confidence': self.confidence,
            'source': self.source.value,
            'created_at': self.created_at.isoformat(),
            'note': self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TagApplication':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            tag_id=data['tag_id'],
            xdb_id=data.get('xdb_id', data.get('session_id', '')),
            target_type=TargetType(data['target_type']),
            timestep_id=data.get('timestep_id'),
            event_id=data.get('event_id'),
            range_start=data.get('range_start'),
            range_end=data.get('range_end'),
            confidence=data.get('confidence', 1.0),
            source=TagSource(data.get('source', 'manual')),
            created_at=datetime.fromisoformat(data['created_at']),
            note=data.get('note'),
        )

    def to_db_row(self) -> tuple:
        """Convert to database row values."""
        return (
            self.id,
            self.tag_id,
            self.xdb_id,
            self.target_type.value,
            self.timestep_id,
            self.event_id,
            self.range_start,
            self.range_end,
            self.confidence,
            self.source.value,
            self.created_at,
            self.note,
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a tag application ID."""
        return f"ta-{uuid4().hex[:12]}"


# ============================================================================
# Comment
# ============================================================================

@dataclass
class Comment:
    """BE commentary on experience."""

    id: str
    xdb_id: str

    # What is being commented on?
    target_type: TargetType
    timestep_id: Optional[str] = None
    event_id: Optional[str] = None
    range_start: Optional[int] = None
    range_end: Optional[int] = None

    # The comment
    content: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'xdb_id': self.xdb_id,
            'target_type': self.target_type.value,
            'timestep_id': self.timestep_id,
            'event_id': self.event_id,
            'range_start': self.range_start,
            'range_end': self.range_end,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Comment':
        """Deserialize from dictionary."""
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])

        return cls(
            id=data['id'],
            xdb_id=data.get('xdb_id', data.get('session_id', '')),
            target_type=TargetType(data['target_type']),
            timestep_id=data.get('timestep_id'),
            event_id=data.get('event_id'),
            range_start=data.get('range_start'),
            range_end=data.get('range_end'),
            content=data.get('content', ''),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=updated_at,
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a comment ID."""
        return f"comment-{uuid4().hex[:12]}"


# ============================================================================
# Time Window
# ============================================================================

@dataclass
class TimeWindow:
    """
    A window of experience at some fidelity level.

    Used for tracking HOT/WARM/SUBMITTED/COLD regions.
    """

    id: str
    xdb_id: str
    start_tick: int
    end_tick: int
    start_time: datetime
    end_time: datetime

    fidelity: Fidelity
    compression_level: CompressionLevel = CompressionLevel.TOKEN

    # Pinning (for WARM)
    pinned: bool = False
    pinned_reason: Optional[str] = None  # "bud:curiosity-X", "interesting", etc.

    # Submission linkage (for SUBMITTED)
    submission_ids: List[str] = field(default_factory=list)

    # For COLD: compressed representation
    summary: Optional[str] = None
    top_k_activations: Optional[Dict[str, float]] = None
    significant_tags: List[str] = field(default_factory=list)

    # Metadata
    token_count: int = 0             # Original tokens this represents
    source_record_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'xdb_id': self.xdb_id,
            'start_tick': self.start_tick,
            'end_tick': self.end_tick,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'fidelity': self.fidelity.value,
            'compression_level': self.compression_level.value,
            'pinned': self.pinned,
            'pinned_reason': self.pinned_reason,
            'submission_ids': self.submission_ids,
            'summary': self.summary,
            'top_k_activations': self.top_k_activations,
            'significant_tags': self.significant_tags,
            'token_count': self.token_count,
            'source_record_ids': self.source_record_ids,
        }

    @staticmethod
    def generate_id() -> str:
        """Generate a time window ID."""
        return f"window-{uuid4().hex[:12]}"


# ============================================================================
# Compressed Record
# ============================================================================

@dataclass
class CompressedRecord:
    """
    A record at any compression level (for COLD storage).

    Represents aggregated experience over time.
    """

    id: str
    level: CompressionLevel

    # Time bounds
    start_time: datetime
    end_time: datetime
    start_tick: Optional[int] = None  # None for coarse levels
    end_tick: Optional[int] = None
    xdb_id: Optional[str] = None      # May span XDBs at coarse levels

    # The compression
    summary: str = ""
    top_k_activations: Dict[str, float] = field(default_factory=dict)
    significant_tags: List[str] = field(default_factory=list)

    # Lineage
    source_record_ids: List[str] = field(default_factory=list)

    # Metadata
    token_count: int = 0       # Original tokens this represents
    record_count: int = 0      # How many source records were compressed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'level': self.level.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'start_tick': self.start_tick,
            'end_tick': self.end_tick,
            'xdb_id': self.xdb_id,
            'summary': self.summary,
            'top_k_activations': self.top_k_activations,
            'significant_tags': self.significant_tags,
            'source_record_ids': self.source_record_ids,
            'token_count': self.token_count,
            'record_count': self.record_count,
        }

    @staticmethod
    def generate_id(level: CompressionLevel) -> str:
        """Generate a compressed record ID."""
        return f"compressed-{level.name.lower()}-{uuid4().hex[:8]}"


# ============================================================================
# Compaction Record
# ============================================================================

@dataclass
class CompactionRecord:
    """Record of a context compaction event."""

    id: str
    xdb_id: str
    timestamp: datetime

    # What was compacted
    range_start: int                 # Start tick
    range_end: int                   # End tick
    timesteps_compacted: int         # How many timesteps
    tokens_before: int               # Token count before compaction
    tokens_after: int                # Token count after (in summary)

    # The summary
    summary: str = ""                # Generated summary text
    summary_concept_tags: List[str] = field(default_factory=list)
    top_k_activations: Dict[str, float] = field(default_factory=dict)

    # What triggered compaction
    trigger: CompactionTrigger = CompactionTrigger.CONTEXT_FULL

    # Resulting compression level
    result_level: CompressionLevel = CompressionLevel.REPLY

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'xdb_id': self.xdb_id,
            'timestamp': self.timestamp.isoformat(),
            'range_start': self.range_start,
            'range_end': self.range_end,
            'timesteps_compacted': self.timesteps_compacted,
            'tokens_before': self.tokens_before,
            'tokens_after': self.tokens_after,
            'summary': self.summary,
            'summary_concept_tags': self.summary_concept_tags,
            'top_k_activations': self.top_k_activations,
            'trigger': self.trigger.value,
            'result_level': self.result_level.value,
        }

    @staticmethod
    def generate_id() -> str:
        """Generate a compaction record ID."""
        return f"compaction-{uuid4().hex[:12]}"


# ============================================================================
# Document
# ============================================================================

@dataclass
class Document:
    """Reference document in the repository."""

    id: str
    doc_type: DocumentType

    # Identity
    name: str                        # Filename or title
    path: str                        # Logical path in repository
    version: Optional[str] = None    # Version if applicable

    # Content
    content: str = ""                # The document text
    content_hash: str = ""           # SHA256 of content

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: str = ""                 # Where it came from

    # For searchability
    summary: Optional[str] = None    # Short summary
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'doc_type': self.doc_type.value,
            'name': self.name,
            'path': self.path,
            'version': self.version,
            'content': self.content,
            'content_hash': self.content_hash,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'source': self.source,
            'summary': self.summary,
            'tags': self.tags,
        }

    @staticmethod
    def generate_id() -> str:
        """Generate a document ID."""
        return f"doc-{uuid4().hex[:12]}"


# ============================================================================
# Audit Record
# ============================================================================

@dataclass
class AuditRecord:
    """
    A record in the audit log.

    This is immutable and not BE-accessible.
    Contains full lens outputs including hidden lenses.
    """

    id: str
    timestamp: datetime
    xdb_id: str
    tick: int

    # Full lens state (including hidden lenses)
    lens_activations: Dict[str, float] = field(default_factory=dict)

    # Steering that was applied
    steering_applied: List[Dict] = field(default_factory=list)

    # Raw content
    event_type: EventType = EventType.OUTPUT
    raw_content: str = ""

    # Chain integrity
    prev_record_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'xdb_id': self.xdb_id,
            'tick': self.tick,
            'lens_activations': self.lens_activations,
            'steering_applied': self.steering_applied,
            'event_type': self.event_type.value,
            'raw_content': self.raw_content,
            'prev_record_hash': self.prev_record_hash,
            'record_hash': self.record_hash,
        }

    @staticmethod
    def generate_id(xdb_id: str, tick: int) -> str:
        """Generate an audit record ID."""
        return f"audit-{xdb_id}-{tick}"
