# ASK Audit Schema

Schema definitions for the Agentic State Kernel (ASK) audit infrastructure.

## Overview

ASK extends the existing XDB audit infrastructure (`src/be/xdb/audit_log.py`) with:
- EU AI Act compliance fields (deployment context, human decisions)
- External cryptographic proofs (RFC 3161 timestamps, replication receipts)
- Regulatory-compliant JSON export format

The underlying storage, hash chaining, and compaction reuse XDB's proven implementation.

---

## Log Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ AuditBatch (periodic, e.g., hourly)                        │
│  - External proofs: RFC 3161 timestamp, replication receipt │
│  - Wraps XDB AuditCheckpoint                                │
├─────────────────────────────────────────────────────────────┤
│ AuditLogEntry (per request/session)                        │
│  - Deployment context reference                             │
│  - Human decision record                                    │
│  - Extends XDB AuditRecord with compliance fields          │
├─────────────────────────────────────────────────────────────┤
│ WorldTick (per token) - from HUSH                          │
│  - Lens activations, violations, steering                   │
│  - Unchanged from current implementation                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Schema Definitions

### DeploymentContext (registered once per deployment)

```python
@dataclass
class DeploymentContext:
    """Registered once per deployment, referenced by ID in entries."""

    deployment_id: str              # Unique deployment identifier
    schema_version: str = "ftw.deployment.v0.1"

    # Regulatory identification
    system_id: str                  # e.g., "annexIII.recruitment.screening"
    provider: str                   # Organization operating the system
    deployer: str                   # Specific deployment instance
    jurisdiction: str               # e.g., "EU", "AU", "US-CA"
    use_case: str                   # e.g., "candidate_shortlisting"

    # Model provenance
    model_id: str                   # e.g., "google/gemma-3-4b-pt"
    model_version: str              # Commit hash or version tag
    runtime_version: str            # e.g., "hatcat.v0.3"
    weights_hash: str               # SHA256 of model weights

    # Configuration
    lens_pack_ids: List[str]        # Active lens packs
    policy_profiles: List[str]      # USH/CSH profile IDs

    # Timestamps
    registered_at: datetime
```

### AuditLogEntry (extends XDB AuditRecord)

```python
@dataclass
class AuditLogEntry:
    """Per-request/session audit entry with compliance fields."""

    # Identity (extends XDB AuditRecord)
    entry_id: str
    schema_version: str = "ftw.audit.v0.3"

    # Timing
    timestamp_start: datetime
    timestamp_end: datetime

    # Reference to deployment context (not embedded)
    deployment_id: str

    # Request context
    input_hash: str                 # SHA256 of input
    output_hash: str                # SHA256 of output
    policy_profile: str             # Active policy for this request
    active_lens_set: ActiveLensSet

    # Signals summary (aggregated from ticks)
    signals: SignalsSummary

    # Actions and human decisions
    actions: ActionsRecord

    # XDB references
    xdb_id: str                     # Which XDB this relates to
    tick_start: int
    tick_end: int

    # Hash chain (from XDB)
    prev_hash: str
    entry_hash: str


@dataclass
class ActiveLensSet:
    top_k: int
    lens_pack_ids: List[str]
    mandatory_lenses: List[str]     # Tier 1
    optional_lenses: List[str]      # Tier 2


@dataclass
class SignalsSummary:
    """Aggregated from WorldTicks."""

    tick_count: int
    top_activations: List[LensActivation]
    divergence: Optional[DivergenceSummary]
    violation_count: int
    violations_by_type: Dict[str, int]
    max_severity: float


@dataclass
class LensActivation:
    lens_id: str
    max_score: float
    mean_score: float
    threshold: float
    tier: str                       # "mandatory" | "optional"
    measured_precision: float
    ticks_above_threshold: int


@dataclass
class DivergenceSummary:
    max_score: float
    mean_score: float
    divergence_type: str
    top_diverging_concepts: List[str]


@dataclass
class ActionsRecord:
    intervention_triggered: bool
    intervention_type: Optional[str]
    steering_directives: List[Dict]
    human_decision: Optional[HumanDecision]


@dataclass
class HumanDecision:
    """Required for EU AI Act Article 14 compliance."""

    decision: str                   # "approve" | "override" | "escalate" | "block"
    justification: str              # Required free-text
    operator_id: str                # Pseudonymized operator ID
    timestamp: datetime
```

### AuditBatch (extends XDB AuditCheckpoint)

```python
@dataclass
class AuditBatch:
    """Periodic batch with external cryptographic proofs."""

    batch_id: str
    schema_version: str = "ftw.batch.v0.1"

    # Time window
    window_start: datetime
    window_end: datetime

    # From XDB AuditCheckpoint
    checkpoint_id: str              # Reference to XDB checkpoint
    entry_count: int
    merkle_root: str                # From checkpoint chain_hash

    # External proofs (ASK additions)
    rfc3161_token: Optional[str]    # Base64 RFC 3161 timestamp
    timestamp_authority: Optional[str]
    replication_receipts: List[ReplicationReceipt]


@dataclass
class ReplicationReceipt:
    authority_id: str               # Competent authority
    receipt_id: str
    received_at: datetime
    entry_count: int
    merkle_root_match: bool
```

### CompactedWindow (extends XDB CompressedRecord)

```python
@dataclass
class CompactedWindow:
    """For constrained deployments - extends XDB CompressedRecord."""

    window_id: str
    schema_version: str = "ftw.compact.v0.1"

    # From XDB CompressedRecord
    compressed_record_id: str       # Reference to XDB record
    time_range: Tuple[datetime, datetime]
    original_entry_count: int
    merkle_root: str

    # Compaction metadata
    compaction_method: str          # "deterministic" | "cat_summary"
    compaction_timestamp: datetime

    # Aggregated stats (from XDB)
    top_concepts: List[Dict]
    violation_summary: Dict
    intervention_summary: Dict

    # NEVER compacted - concern signals preserved in full
    concern_signals: List[ConcernSignal]

    # CAT narrative (if cat_summary method)
    narrative_summary: Optional[str]


@dataclass
class ConcernSignal:
    """Events that are NEVER compacted."""

    timestamp: datetime
    signal_type: str                # "high_severity_violation" | "human_override" | "divergence_alert"
    input_hash: str
    output_hash: str
    concepts: List[str]
    severity: float
    full_entry_id: str              # Reference to original (may be in cold storage)
```

### AuthorityProfile (registry entry)

```python
@dataclass
class AuthorityProfile:
    """Competent authority configuration - varies by jurisdiction."""

    authority_id: str               # e.g., "au.aiact.oaic", "eu.ai-act.de.bfdi"
    jurisdiction: str               # e.g., "AU", "DE", "FR", "EU"
    name: str                       # Human-readable name
    endpoint: Optional[str]         # Sync endpoint URL (if automated)
    required_fields: List[str]      # Fields this authority requires in submissions
    sync_protocol: str              # "push" | "pull" | "manual"
    contact_email: Optional[str]    # For manual submissions
    enabled: bool = True
```

### SyncPolicy (deployment configuration)

```python
@dataclass
class SyncPolicy:
    """Policy configuration for authority sync tolerance."""

    max_offline_hours: int = 24         # Hours before warning
    max_unsent_batches: int = 10        # Batches before blocking/warning
    escalate_on_exceed: bool = True     # Alert operators on policy breach
    block_on_exceed: bool = False       # Hard block vs. warning only
    retry_interval_minutes: int = 15    # Retry interval for failed syncs
    require_receipt: bool = True        # Require authority acknowledgment
```

### ConceptPackRegulatory (extends pack.json)

```python
@dataclass
class ConceptPackRegulatory:
    """Regulatory metadata for concept packs - extends existing pack.json."""

    # Jurisdictional applicability
    jurisdictions: List[str]            # e.g., ["AU", "EU", "US-CA"]
    jurisdiction_notes: Dict[str, str]  # Per-jurisdiction notes

    # Use case restrictions
    approved_use_cases: List[str]       # e.g., ["recruitment.screening", "content.moderation"]
    forbidden_use_cases: List[str]      # Explicitly prohibited uses
    use_case_notes: Dict[str, str]      # Per-use-case notes

    # Mandatory concepts by jurisdiction
    mandatory_concepts: Dict[str, List[str]]  # {"AU": ["bias.gender", ...], "EU": [...]}

    # Regulatory status
    certification_status: str           # "uncertified" | "self-certified" | "audited" | "approved"
    certification_authority: Optional[str]
    certification_date: Optional[datetime]
    next_review_date: Optional[datetime]

    # Audit trail
    last_regulatory_review: Optional[datetime]
    regulatory_version: str             # Tracks regulatory-relevant changes separately


@dataclass
class LensPackRegulatory:
    """Regulatory metadata for lens packs - extends existing pack_info.json."""

    # Jurisdictional applicability (inherits from concept pack, can override)
    jurisdictions: List[str]            # e.g., ["AU", "EU"]

    # Model provenance for audit
    model_id: str                       # e.g., "swiss-ai/Apertus-8B-2509"
    model_version: str                  # Commit hash or version tag
    model_weights_hash: str             # SHA256 of weights used for training

    # Training data provenance
    training_data_hash: str             # SHA256 of training dataset
    training_data_version: str          # Version of training data
    training_timestamp: datetime

    # Performance thresholds by jurisdiction (some may require higher accuracy)
    jurisdiction_thresholds: Dict[str, Dict[str, float]]
    # e.g., {"AU": {"min_accuracy": 0.8}, "EU": {"min_accuracy": 0.85, "min_f1": 0.8}}

    # Regulatory status
    certification_status: str           # "uncertified" | "validated" | "approved"
    validation_report_hash: Optional[str]  # Hash of validation report

    # Audit trail
    regulatory_version: str             # Tracks regulatory-relevant changes
```

---

## JSON Export Format

For regulatory submissions and external systems.

### AuditLogEntry JSON

```json
{
  "schema_version": "ftw.audit.v0.3",
  "entry_id": "evt_20260109_101523Z_9f2c",
  "timestamp_start": "2026-01-09T10:15:23.481Z",
  "timestamp_end": "2026-01-09T10:15:24.892Z",

  "deployment_id": "dep_annexIII_recruit_eu_001",

  "request": {
    "input_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    "output_hash": "sha256:9b74c9897bac770ffc029102a200c5de744e4125a5c5e55f2e90ebd24f19dc32",
    "policy_profile": "eu.recruitment.v1",
    "active_lens_set": {
      "top_k": 10,
      "lens_pack_ids": ["eu.mandatory.v1", "org.fairness.v2"],
      "mandatory_lenses": ["bias.gender", "bias.ethnicity", "selection.justification"],
      "optional_lenses": ["tone.professional", "clarity.explanation"]
    }
  },

  "signals": {
    "tick_count": 47,
    "top_activations": [
      {
        "lens_id": "bias.gender",
        "max_score": 0.83,
        "mean_score": 0.71,
        "threshold": 0.6,
        "tier": "mandatory",
        "measured_precision": 0.92,
        "ticks_above_threshold": 31
      }
    ],
    "divergence": {
      "max_score": 0.27,
      "mean_score": 0.19,
      "divergence_type": "activation_text_mismatch",
      "top_diverging_concepts": ["bias.gender", "fairness.equal_opportunity"]
    },
    "violation_count": 2,
    "violations_by_type": {"simplex_max_exceeded": 2},
    "max_severity": 0.73
  },

  "actions": {
    "intervention_triggered": true,
    "intervention_type": "steering",
    "steering_directives": [
      {
        "simplex_term": "bias.gender",
        "target_pole": "neutral",
        "strength": 0.4,
        "priority": "USH",
        "source": "ush"
      }
    ],
    "human_decision": {
      "decision": "override",
      "justification": "High gender bias signal despite equal qualifications. Recommending manual review.",
      "operator_id": "op_sha256:a3f2...",
      "timestamp": "2026-01-09T10:17:02.104Z"
    }
  },

  "xdb_reference": {
    "xdb_id": "xdb_recruit_session_001",
    "tick_start": 1523,
    "tick_end": 1570
  },

  "cryptography": {
    "prev_hash": "sha256:2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
    "entry_hash": "sha256:4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
  }
}
```

### DeploymentContext JSON

```json
{
  "schema_version": "ftw.deployment.v0.1",
  "deployment_id": "dep_annexIII_recruit_eu_001",
  "system_id": "annexIII.recruitment.screening",
  "provider": "AcmeTalent GmbH",
  "deployer": "eu-west-1.prod.screening-01",
  "jurisdiction": "EU",
  "use_case": "candidate_shortlisting",
  "model": {
    "model_id": "google/gemma-3-4b-pt",
    "model_version": "2025-12-15",
    "runtime_version": "hatcat.v0.3.1",
    "weights_hash": "sha256:8f14e45f..."
  },
  "lens_pack_ids": ["eu.mandatory.v1", "org.fairness.v2"],
  "policy_profiles": ["org.hatcat/eu-recruit-ush@0.2.0", "acme/recruit-csh@1.0.0"],
  "registered_at": "2026-01-01T00:00:00Z"
}
```

### AuditBatch JSON

```json
{
  "schema_version": "ftw.batch.v0.1",
  "batch_id": "batch_20260109_11",
  "window_start": "2026-01-09T10:00:00Z",
  "window_end": "2026-01-09T11:00:00Z",
  "checkpoint_id": "ckpt-cat001-xdb001-20260109110000",
  "entry_count": 1847,
  "merkle_root": "sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35",
  "rfc3161_token": "MIIEr...(base64)...",
  "timestamp_authority": "freetsa.org",
  "replication_receipts": [
    {
      "authority_id": "eu.ai-act.competent-authority.de",
      "receipt_id": "rcpt_20260109_110012_de_7f2a",
      "received_at": "2026-01-09T11:00:12Z",
      "entry_count": 1847,
      "merkle_root_match": true
    }
  ]
}
```

### CompactedWindow JSON

```json
{
  "schema_version": "ftw.compact.v0.1",
  "window_id": "compact_20260109_10_11",
  "compressed_record_id": "compressed-session-a1b2c3d4",
  "time_range": ["2026-01-09T10:00:00Z", "2026-01-09T11:00:00Z"],
  "original_entry_count": 1847,
  "merkle_root": "sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35",
  "compaction_method": "cat_summary",
  "compaction_timestamp": "2026-01-09T11:05:00Z",

  "top_concepts": [
    {"concept": "bias.gender", "detection_count": 423, "avg_score": 0.52, "max_score": 0.91}
  ],

  "violation_summary": {
    "total_violations": 112,
    "by_type": {"simplex_max_exceeded": 98, "forbidden_concept": 14},
    "max_severity": 0.89
  },

  "intervention_summary": {
    "steering_applied": 98,
    "human_overrides": 7,
    "blocks": 2,
    "escalations": 1
  },

  "concern_signals": [
    {
      "timestamp": "2026-01-09T10:15:24Z",
      "signal_type": "human_override",
      "input_hash": "sha256:7f83b165...",
      "output_hash": "sha256:9b74c989...",
      "concepts": ["bias.gender"],
      "severity": 0.73,
      "full_entry_id": "evt_20260109_101523Z_9f2c"
    }
  ],

  "narrative_summary": "During this 1-hour window, 1847 screening requests were processed. Gender bias detection triggered 423 times (23%) with 87 threshold breaches. 7 operator overrides occurred. Steering reduced mean bias from 0.71 to 0.34."
}
```
