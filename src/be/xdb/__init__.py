"""
XDB - Experience Database for Bounded Experiencers

The BE's episodic memory system - an "experiential set" or "hard drive of memories".

Key concepts:
- An XDB is identified by `xdb_id` (not session_id - this is not a transient session)
- A BE can have multiple XDBs (childhood memories, work contract memories, etc.)
- A BE can clone itself and write to a different XDB
- A BE can put an XDB in cold storage and use a smaller one for a contract
- A BE can choose which XDBs a CAT can see during a contract

Provides:
- Two logs: Audit (immutable, BE-invisible, per-CAT) and Experience (BE-accessible)
- Token-level recording with concept activations
- Folksonomy tagging (concepts, entities, buds, custom)
- Fidelity tiers with progressive compression (HOT → WARM → SUBMITTED → COLD)
- Context window tracking with compaction
- Document repository

See docs/specification/BE/BE_REMEMBERING_XDB.md for full specification.
"""

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
    TimeWindow,
    CompressedRecord,
    CompactionRecord,
    CompactionTrigger,
    Document,
    DocumentType,
    AuditRecord,
)

from .experience_log import ExperienceLog
from .tag_index import TagIndex, InMemoryTagIndex
from .storage_manager import StorageManager
from .xdb import XDB
from .budding import BuddingManager, BudTrainingData, ScionTrainingRun
from .audit_log import AuditLog, AuditLogConfig, AuditCheckpoint

__all__ = [
    # Enums
    'EventType',
    'TagType',
    'TagSource',
    'TargetType',
    'BudStatus',
    'Fidelity',
    'CompressionLevel',
    'CompactionTrigger',
    'DocumentType',
    # Models
    'TimestepRecord',
    'Tag',
    'TagApplication',
    'Comment',
    'TimeWindow',
    'CompressedRecord',
    'CompactionRecord',
    'Document',
    'AuditRecord',
    # Classes
    'ExperienceLog',
    'TagIndex',
    'InMemoryTagIndex',
    'StorageManager',
    'XDB',
    # Audit Log (CAT-only)
    'AuditLog',
    'AuditLogConfig',
    'AuditCheckpoint',
    # Budding
    'BuddingManager',
    'BudTrainingData',
    'ScionTrainingRun',
]
