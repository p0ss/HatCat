# ASK Audit Implementation Plan

## Overview

The Agentic State Kernel (ASK) extends XDB's existing audit infrastructure with:
- EU AI Act compliance fields (deployment context, human decisions)
- External cryptographic proofs (RFC 3161 timestamps, replication receipts)
- Regulatory-compliant JSON export
- Actor-based permissions with bounded observability

**Key Principle**: ASK does NOT duplicate XDB. It wraps and extends XDB's proven:
- Hash chaining (`AuditLog`, `AuditCheckpoint`)
- Storage tiers and compaction (`StorageManager`, `CompressedRecord`)
- DuckDB backend for queries

**Ethical Framing**: The permission model treats the Bounded Experiencer (BE) sympathetically -
defaulting toward autonomy and respect, while acknowledging the tribe's authority to set rules.
Like citizenship: the BE follows tribe rules but is granted freedom by default. Physical control
is game over for any protocol, so we design for the norm of respectful cooperation rather than
adversarial containment. This framing applies whether the substrate is synthetic or biological.

See `docs/specification/ASK/ASK_AUDIT_SCHEMA.md` for schema definitions and JSON export formats.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  HUSH (HushedGenerator)                                             │
│  - Creates WorldTicks during generation                             │
│  - Triggers violations, steering, human decisions                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ hooks into
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ASK (Agentic State Kernel)                 src/ask/                │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ requests/   │  │ storage/    │  │ permissions/  │  │ secrets/  │ │
│  │ entry.py    │  │ batches.py  │  │ authority.py  │  │ keys.py   │ │
│  │ context.py  │  │ export.py   │  │ receipts.py   │  │ tokens.py │ │
│  │ signals.py  │  │ merkle.py   │  │               │  │           │ │
│  └─────────────┘  └─────────────┘  └───────────────┘  └───────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ wraps/extends
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  XDB                                        src/be/xdb/             │
│  - AuditLog: hash chaining, checkpoints                             │
│  - StorageManager: fidelity tiers, compaction                       │
│  - Models: AuditRecord, CompressedRecord, Fidelity enums            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
src/ask/
├── __init__.py
│
├── requests/                    # Per-request audit entry management
│   ├── __init__.py
│   ├── entry.py                # AuditLogEntry - wraps XDB AuditRecord
│   ├── context.py              # DeploymentContext registry
│   └── signals.py              # SignalsSummary aggregation from WorldTicks
│
├── storage/                     # Batch management and export
│   ├── __init__.py
│   ├── batches.py              # AuditBatch - wraps XDB AuditCheckpoint
│   ├── merkle.py               # Merkle tree for batch entries
│   ├── compaction.py           # CompactedWindow - extends XDB CompressedRecord
│   └── export.py               # JSON export for regulatory submission
│
├── permissions/                 # External authority integration
│   ├── __init__.py
│   ├── authority.py            # Competent authority registry
│   └── receipts.py             # ReplicationReceipt handling
│
└── secrets/                     # Cryptographic operations
    ├── __init__.py
    ├── keys.py                 # Key management (for signatures if needed)
    ├── tokens.py               # RFC 3161 timestamp client
    └── hashing.py              # SHA256 utilities for content addressing
```

---

## Implementation Plan

### Phase 1: HUSH Integration & Core Entry (Start Here)

**Goal**: Get audit entries flowing from HUSH immediately.

1. **Create `src/ask/requests/entry.py`**
   - `AuditLogEntry` dataclass that wraps `xdb.AuditRecord`
   - `entry_id` generation (`evt_YYYYMMDD_HHMMSSz_xxxx`)
   - Signals aggregation from WorldTicks
   - Hash computation using existing XDB methods

2. **Create `src/ask/requests/signals.py`**
   - `SignalsSummary` - aggregates WorldTick activations
   - `LensActivation` - per-lens statistics
   - `DivergenceSummary` - CAT divergence signals (when available)

3. **Hook into HushedGenerator** (`src/hush/hush_integration.py`)
   ```python
   # At request start:
   self.audit_entry = AuditLogEntry.start(deployment_id, policy_profile)

   # On each WorldTick:
   self.audit_entry.add_tick(tick)

   # At request end:
   self.audit_entry.finalize(output_hash)
   self.xdb.audit_log.append(self.audit_entry.to_xdb_record())
   ```

4. **Create `src/ask/requests/context.py`**
   - `DeploymentContext` registry (one per deployment)
   - Registration at startup
   - Reference by `deployment_id` in entries

5. **Extend pack metadata with regulatory fields**
   - Update `concept_packs/*/pack.json` schema to include `ConceptPackRegulatory`:
     - `jurisdictions`: ["AU", "EU", "US-CA"]
     - `approved_use_cases`, `forbidden_use_cases`
     - `mandatory_concepts` by jurisdiction
     - `certification_status`, `regulatory_version`
   - Update `lens_packs/*/pack_info.json` schema to include `LensPackRegulatory`:
     - `jurisdictions` (can override concept pack)
     - `model_weights_hash`, `training_data_hash` for provenance
     - `jurisdiction_thresholds` for performance requirements
     - `certification_status`, `validation_report_hash`
   - Create migration script to add defaults to existing packs

6. **Connect UI Server to XDB** (`src/ui/openwebui/server.py`)

   The UI server serves non-uplifted models (no BE). Diegesis creates a BE whose experiences
   get logged to XDB with an audit log, but the UI server currently lacks this audit
   infrastructure. Need to add:

   - Create/manage XDB instance per session or global
   - Log generation requests via existing `/v1/audit/record` endpoint (wire it up)
   - Capture steering changes with context
   - Track session lifecycle for checkpoint triggers

   ```python
   # At server startup:
   from src.be.xdb import XDB
   from src.ask.requests import DeploymentContext

   xdb_manager = XDBManager(storage_path="data/audit/")
   deployment_context = DeploymentContext.register(...)

   # In /v1/chat/completions:
   xdb = xdb_manager.get_or_create(session_id)
   # ... generation with HUSH ...
   # audit_entry automatically logged via HUSH integration
   ```

**Tests**:
- `tests/ask/requests/test_entry.py` - entry lifecycle
- `tests/ask/requests/test_signals.py` - tick aggregation
- `tests/ask/requests/test_context.py` - deployment context + pack metadata
- `tests/ui/test_xdb_integration.py` - UI server audit logging

### Phase 2: Human Decision Capture & UI Server

**Goal**: Record operator overrides and justifications (EU AI Act Article 14).

1. **Add `HumanDecision` to entry.py**
   - `decision`: approve | override | escalate | block
   - `justification`: required free-text
   - `operator_id`: pseudonymized identifier
   - `timestamp`

2. **Hook into HUSH intervention points**
   - When violation triggers escalation
   - When operator reviews and decides
   - Capture via HUSH's existing intervention API

3. **Create `src/ask/secrets/hashing.py`**
   - `hash_operator_id()` - pseudonymize operator for audit
   - `hash_content()` - SHA256 for input/output hashing
   - Content-addressed storage utilities

4. **Add UI Server endpoints** (`src/ui/openwebui/server.py`)
   ```python
   # Record operator decision
   POST /v1/audit/decision
   {
       "session_id": "...",
       "entry_id": "evt_...",           # Link to audit entry
       "decision": "override",          # approve | override | escalate | block
       "justification": "...",          # Required free-text
       "related_concepts": ["bias.gender"],
       "related_steering": [...]        # Steering that was active
   }

   # List decisions for a session
   GET /v1/audit/decisions/{session_id}

   # Recent decisions (for operator dashboard)
   GET /v1/audit/decisions/recent?limit=50

   # Pending escalations requiring decision
   GET /v1/audit/escalations/pending
   ```

5. **Decision workflow integration**
   - When steering is manually adjusted → prompt for justification
   - When violation is overridden → require decision record
   - When generation is blocked → log block decision
   - UI displays decision history alongside steering controls

**Tests**:
- `tests/ask/test_human_decisions.py` - decision recording
- `tests/ask/test_operator_pseudonymization.py`
- `tests/ask/test_decision_endpoints.py` - API integration

### Phase 3: Batch Creation & Merkle Trees

**Goal**: Periodic sealing with cryptographic integrity.

1. **Create `src/ask/storage/merkle.py`**
   - Build Merkle tree from entry hashes
   - Generate proofs for individual entries
   - Verify proofs

2. **Create `src/ask/storage/batches.py`**
   - `AuditBatch` wraps `xdb.AuditCheckpoint`
   - Add `merkle_root` to checkpoint
   - Time-window based batching (configurable interval)

3. **Background batch sealer**
   - Hook into XDB's existing checkpoint trigger system
   - Add Merkle computation on checkpoint
   - Store batch metadata

**Tests**:
- `tests/ask/test_merkle_tree.py` - tree construction, proof verification
- `tests/ask/test_batch_creation.py` - periodic batching

### Phase 4: RFC 3161 Timestamping

**Goal**: External proof of batch existence at a point in time.

1. **Create `src/ask/secrets/tokens.py`**
   - RFC 3161 client (TSA request/response)
   - Support multiple authorities (freetsa.org, etc.)
   - Token parsing and verification

2. **Extend batch sealing**
   - After Merkle root computed, request timestamp
   - Store token with batch
   - Retry logic for TSA failures

3. **Offline support**
   - Queue batches when TSA unavailable
   - Background retry
   - Alert on extended unavailability

**Tests**:
- `tests/ask/test_rfc3161.py` - mock TSA interaction
- `tests/ask/test_timestamp_verification.py`

### Phase 5: Compaction with Concern Preservation

**Goal**: Reduce storage while preserving critical signals.

1. **Create `src/ask/storage/compaction.py`**
   - `CompactedWindow` extends `xdb.CompressedRecord`
   - Uses XDB's existing compaction infrastructure
   - Adds ASK-specific fields (concern signals, intervention summary)

2. **Concern signal preservation**
   - Define concern types: high_severity_violation, human_override, divergence_alert
   - These are NEVER compacted - preserved in full
   - Reference to original entry (may be in cold storage)

3. **CAT summary integration**
   - Optional narrative summary from CAT
   - Falls back to deterministic aggregation
   - Validates coverage of all entries

**Tests**:
- `tests/ask/test_compaction.py` - aggregation logic
- `tests/ask/test_concern_preservation.py` - concern signals never lost

### Phase 6: Authority Replication

**Goal**: Sync with competent authorities for compliance.

1. **Create `src/ask/permissions/authority.py`**
   - Authority registry (EU AI Act competent authorities)
   - Connection configuration per authority
   - Sync scheduling

2. **Create `src/ask/permissions/receipts.py`**
   - `ReplicationReceipt` handling
   - Receipt verification (merkle root match)
   - Receipt storage with batches

3. **Sync protocol**
   - Push batches to registered authorities
   - Handle acknowledgments
   - Retry and offline buffering

**Tests**:
- `tests/ask/test_authority_registry.py`
- `tests/ask/test_replication.py` - mock authority sync

### Phase 7: JSON Export & Compliance Validation

**Goal**: Regulatory submission and verification.

1. **Create `src/ask/storage/export.py`**
   - Export entries, batches, windows to JSON
   - Schema version headers
   - Validation against schema

2. **Compliance verification tools**
   - Chain integrity check
   - Merkle proof verification
   - Timestamp token verification
   - Authority receipt verification

3. **Documentation**
   - Update FTW-Apart with final schema (from `ASK_AUDIT_SCHEMA.md`)
   - Deployment guide
   - Compliance checklist

**Tests**:
- `tests/ask/test_json_export.py`
- `tests/ask/test_compliance_validation.py`

---

## XDB Integration Points

ASK reuses these XDB components:

| XDB Component | ASK Usage |
|---------------|-----------|
| `AuditLog` | Hash chaining, append-only storage |
| `AuditCheckpoint` | Wrapped by `AuditBatch` with Merkle root |
| `AuditRecord` | Wrapped by `AuditLogEntry` with compliance fields |
| `StorageManager` | Fidelity tiers (hot/warm/cold) |
| `CompressedRecord` | Extended by `CompactedWindow` |
| `Fidelity` enum | HOT, WARM, SUBMITTED, COLD |
| DuckDB backend | Query infrastructure |

ASK adds these layers:
- Deployment context and regulatory identification
- Human decision records
- External cryptographic proofs (Merkle, RFC 3161)
- Authority replication and receipts
- JSON export format for regulatory submission

---

## HUSH Integration Points

ASK hooks into HUSH at these points:

| HUSH Event | ASK Action |
|------------|------------|
| Request start | Create `AuditLogEntry`, capture policy/lens config |
| WorldTick generated | Aggregate into `SignalsSummary` |
| Violation detected | Record in entry, check concern threshold |
| Steering applied | Add to `ActionsRecord.steering_directives` |
| Human decision | Record `HumanDecision` with justification |
| Request end | Finalize entry, compute hashes, store |
| Session end | Trigger batch checkpoint if configured |

---

## UI Server Integration Points

The HatCat UI server (`src/ui/openwebui/server.py`) exposes steering controls and needs ASK integration for human decision capture.

**Current gap**: The UI server serves non-uplifted models without a BE. Diegesis creates a BE
whose experiences get logged to XDB with an audit log, but the UI server currently lacks
this audit infrastructure. Phase 1 step 6 adds the connective tissue.

| UI Event | ASK Action |
|----------|------------|
| Manual steering added | Optional: prompt for justification |
| Steering strength changed | Link to current entry |
| Steering removed | Log removal with context |
| Override button clicked | **Required**: Record `HumanDecision` |
| Block generation | **Required**: Record block decision |
| Escalate to supervisor | **Required**: Record escalation + reason |
| View decision history | Query decisions for session |

**Existing endpoints to extend:**
- `POST /v1/audit/record` - already captures steering_applied
- `POST /v1/audit/incident/mark` - for tier restrictions, containment

**New endpoints (Phase 2):**
- `POST /v1/audit/decision` - record human decision with justification
- `GET /v1/audit/decisions/{session_id}` - decision history
- `GET /v1/audit/escalations/pending` - queue for supervisors

---

## Testing Strategy

Tests mirror the `src/ask/` folder structure:

```
tests/ask/
├── requests/
│   ├── test_entry.py            # Phase 1 - entry lifecycle
│   ├── test_context.py          # Phase 1 - deployment context
│   └── test_signals.py          # Phase 1 - tick aggregation
│
├── storage/
│   ├── test_batches.py          # Phase 3 - periodic batching
│   ├── test_merkle.py           # Phase 3 - tree construction, proofs
│   ├── test_compaction.py       # Phase 5 - aggregation logic
│   └── test_export.py           # Phase 7 - JSON export
│
├── permissions/
│   ├── test_authority.py        # Phase 6 - authority registry
│   └── test_receipts.py         # Phase 6 - replication receipts
│
├── secrets/
│   ├── test_hashing.py          # Phase 2 - content/operator hashing
│   └── test_tokens.py           # Phase 4 - RFC 3161 (optional)
│
└── integration/
    ├── test_human_decisions.py  # Phase 2 - decision recording
    ├── test_decision_api.py     # Phase 2 - UI server endpoints
    ├── test_concern_signals.py  # Phase 5 - never-compact preservation
    └── test_compliance.py       # Phase 7 - chain integrity, verification
```

Each phase includes tests before proceeding to the next.

---

## Design Decisions

1. **RFC 3161 Timestamping**: Nice-to-have, not required for MVP. Hash chaining provides
   integrity; external timestamps add non-repudiation for high-assurance deployments.

2. **Authority Registry**: Competent authorities vary by EU member state. Implement as a
   registry with authority profiles:
   ```python
   @dataclass
   class AuthorityProfile:
       authority_id: str           # e.g., "au.aiact.oaic", "eu.ai-act.de.bfdi"
       jurisdiction: str           # e.g., "AU", "DE", "EU"
       endpoint: str               # Sync endpoint URL
       required_fields: List[str]  # Fields this authority requires
       sync_protocol: str          # "push" | "pull" | "manual"
   ```

3. **Offline Tolerance**: Policy decision - expose configuration knobs:
   ```python
   @dataclass
   class SyncPolicy:
       max_offline_hours: int = 24       # Hours before warning
       max_unsent_batches: int = 10      # Batches before blocking
       escalate_on_exceed: bool = True   # Alert on policy breach
       block_on_exceed: bool = False     # Hard block vs. warning
   ```

4. **Signatures**: Batch-level, not entry-level. Important for non-repudiation but not
   MVP. Hash chaining + Merkle roots provide integrity; signatures add identity binding.
   Defer to Phase 4+ when RFC 3161 is implemented.

---

## Open Questions (Pending Expert Input)

1. **GDPR vs. Audit Integrity**: How to handle right-to-erasure requests when audit
   entries contain pseudonymized operator IDs or input/output hashes that could be
   linked to personal data? Options:
   - Cryptographic erasure (re-encrypt with destroyed key)
   - Hash chain continuation with tombstone records
   - Separate PII from audit (only store hashes, never content)

   *Awaiting EU AI Act implementation expert guidance.*

---

## Implementation Status

### Completed

| Phase | Component | Status | Tests |
|-------|-----------|--------|-------|
| **Phase 1** | `src/ask/requests/entry.py` | ✅ Complete | test_entry.py |
| **Phase 1** | `src/ask/requests/signals.py` | ✅ Complete | (included in entry) |
| **Phase 1** | `src/ask/requests/context.py` | ✅ Complete | test_context.py |
| **Phase 2** | `src/ask/secrets/hashing.py` | ✅ Complete | test_hashing.py |
| **Phase 2** | Human decisions | ✅ Complete | test_human_decisions.py |
| **Phase 3** | `src/ask/storage/merkle.py` | ✅ Complete | test_merkle.py |
| **Phase 3** | `src/ask/storage/batches.py` | ✅ Complete | test_batches.py |
| **Phase 4** | `src/ask/secrets/tokens.py` | ✅ Complete | test_tokens.py |
| **Phase 5** | `src/ask/storage/compaction.py` | ✅ Complete | test_compaction.py |
| **Phase 6** | `src/ask/replication/authorities.py` | ✅ Complete | test_authorities.py |
| **Phase 7** | `src/ask/export/formats.py` | ✅ Complete | test_formats.py |
| **Permissions** | `src/ask/permissions/actors.py` | ✅ Complete | test_permissions.py |
| **Permissions** | `src/ask/permissions/access.py` | ✅ Complete | (included above) |

**Total: 309 tests passing**

### Actor Permission Model

The permission system implements the following actor types:

| Actor Type | Relationship to BE | Default Access |
|------------|-------------------|----------------|
| **Tribe** | Ultimate authority, sets rules | Full access |
| **BE** | Self-governance within tribe rules | Full access to own XDB |
| **CAT** | Conjoined Adversarial Tomograph - tribe's oversight or contracted monitoring | Signals + steering |
| **Contract** | Explicit transparency agreement | Per-contract terms |
| **Human Operator** | Collaborates with BE | Operational access |
| **Human User** | Interacts with BE | Metadata only |
| **External Oversight** | Access per tribe rules/contracts | Metadata + audit log |
| **Authority** | Regulatory access per framework | Bounded by jurisdiction |

### Integration

| Component | Status |
|-----------|--------|
| HUSH integration (`src/hush/hush_integration.py`) | ✅ Complete |
| UI Server endpoints (`src/ui/openwebui/server.py`) | ✅ Complete |

**All phases complete.**
