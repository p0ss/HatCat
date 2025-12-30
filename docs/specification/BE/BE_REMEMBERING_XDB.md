# BE Remembering: Experience Database (XDB)

*BE submodule: remembering (XDB, XAPI)*

Status: Normative specification
Layer: Between BE runtime and MAP/HAT
Related docs: BE_AWARE_WORKSPACE, BE_CONTINUAL_LEARNING, MAP, HAT, ASK, ASK_TRACE, BOUNDED_EXPERIENCER

---

## 0. Purpose & Scope

The **Experience Database (XDB)** is a BE's persistent **episodic memory**:

- It survives hibernation and process restarts.
- It is the primary payload for **tribe sync** and **knowledge sharing**.
- It is the canonical source of truth from which:
  - **grafts** (substrate dimensions + biases) and
  - **lenses** (HAT/MAP detectors)
  are *re-derived*.

Conceptually:

| Component | Role | Persistence |
|-----------|------|-------------|
| **Global Workspace** | Current attention | Runtime, per session |
| **Experience Database** | Episodic memory | Persistent, syncable |
| **Patches + Lenses** | Learned skills | Derived artifacts via MAP/HAT |
| **Substrate** | Raw capacity | Model weights / brain tissue |

The Experience Database Schema defines:

- **Two logs**: Audit (immutable, BE-invisible) and Experience (BE-accessible, taggable)
- **Timestep-level recording** with concept activations
- **Fidelity tiers** with progressive compression (HOT → WARM → COLD)
- **Folksonomy tagging** (concepts, entities, buds, custom tags)
- **Resource-governed storage** limits from LifecycleContract
- Sync and sharing protocol
- Access control and treaty hooks

It does **not** define:

- how the Global Workspace *looks* internally (that's the workspace doc);
- how continual learning *trains* patches and lenses (that's the learning harness);
- how ASK and Hush *govern* what's allowed (that's ASK/Hush).

---

## 1. Layering & Lifecycle

### 1.1 Runtime vs Persistent

- **Global Workspace (runtime)**
  - Rebuilt from scratch on each instantiation from:
    - current session state,
    - loaded XDB summary (if any),
    - active patches/lenses.
  - Dies with the session.

- **Experience Database (persistent)**
  - Lives on disk or remote storage.
  - Loaded at start of a lifecycle term (or session).
  - Updated continuously by:
    - the Global Workspace harness (structural/log updates), and
    - the Continual Concept Learning harness (candidate concepts,
      training metadata, validation results).
  - Survives hibernation and can be synced with tribe repositories.

- **Patches + Lenses (derived artifacts)**
  - Generated *from* Experience Database exemplars and training runs.
  - Stored as separate deployable assets (e.g. adapter weights,
    lens parameters).
  - Reproducible from XDB + training config where possible.

### 1.2 Sync & Sharing

For **tribe sync**:

1. BE instances exchange **Experience Database diffs**, or upload XDB
   fragments to a shared store.
2. Tribe-level processes retrain **shared patches** from pooled
   exemplars.
3. New patches/lenses are then redistributed as derived artifacts.

The **exemplars** (episodes with tags and provenance) are the **source
of truth**. Patches/lenses are secondary and can be regenerated.

---

## 2. The Two Logs

XDB maintains two parallel logs with different visibility and mutability properties.

### 2.1 Audit Log (BE-Invisible)

The **Audit Log** is an immutable, append-only record that the BE cannot see or modify:

```
┌─────────────────────────────────────────────────────────────────┐
│  AUDIT LOG (BE-Invisible)                                        │
│  - Every timestep with full lens activations                    │
│  - Steering decisions and their rationale                        │
│  - Raw I/O without BE-applied filters                            │
│  - Cryptographic hash chain for integrity                        │
│  - Retention per contract: audit_log_retention_days              │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose**: Enables ASK review, incident investigation, and accountability. The BE's behavior can be audited even if the BE has modified its own Experience Log.

**Properties**:
- **Immutable**: Records cannot be modified after creation
- **Append-only**: New records can only be added, never removed (except by retention policy)
- **Hash-chained**: Each record includes hash of previous, preventing tampering
- **BE-invisible**: The BE has no access to read or query this log

**Audit Log Compression**: The raw audit log grows rapidly at token-level granularity. Tribal policy SHOULD define audit checkpoints where:
1. Full audit data is archived to cold storage
2. Compressed audit summaries are retained for quick access
3. Hash chain integrity is preserved across compression boundaries
4. Retention period (`audit_log_retention_days`) governs when data can be purged

See ASK tribal policy for checkpoint frequency and compression rules.

### 2.2 Experience Log (BE-Accessible)

The **Experience Log** is what the BE can see and interact with:

```
┌─────────────────────────────────────────────────────────────────┐
│  EXPERIENCE LOG (BE-Accessible)                                  │
│  - Timesteps with visible concept activations                    │
│  - BE-applied tags and commentary                                │
│  - Queryable by time, concept, tag, content                      │
│  - Subject to fidelity tiers and compression                     │
│  - Storage quota from LifecycleContract                          │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose**: The BE's episodic memory. Enables recall, reflection, learning, and training data curation.

**Properties**:
- **Taggable**: BE can apply folksonomy tags to timesteps, events, and ranges
- **Commentable**: BE can add commentary and annotations
- **Queryable**: Full-text search, concept-based recall, time-range queries
- **Compactable**: Subject to progressive compression based on fidelity tier

### 2.3 Dual Recording

Every interaction is recorded to both logs simultaneously:

```
Input/Output → ┬→ Audit Log (full lenses, steering, raw)
               └→ Experience Log (visible lenses, BE-accessible)
```

The **Audit Log** receives the superset: all lens activations (including those the BE shouldn't know about), all steering decisions, all raw content.

The **Experience Log** receives the subset: only the concept activations and content the BE is allowed to perceive, filtered by ASK policy.

---

## 3. Core Data Model

At a high level, the Experience Database consists of:

- **Timesteps** – the atomic unit: one per token/event with concept activations
- **Tags & Folksonomy** – arbitrary labels organized by type (concept, entity, bud, custom)
- **Episodes & Exemplars** – tagged slices of experience
- **Session Graph Fragments** – compacted structure of past sessions
- **Concept Datasets** – collections of exemplars for specific concepts
- **Training Runs** – metadata about training patches and lenses
- **Candidate Concepts (Buds)** – concepts in-flight, learning, not yet stable
- **Documents** – reference materials (system instructions, policies, tool docs)

The schema is intentionally expressed as **logical objects** rather than
a specific storage engine. Concrete implementations may use:

- DuckDB (embedded, recommended for scale)
- JSONL files
- Document stores
- Graph databases

### 3.1 Timestep (Atomic Unit)

The **Timestep** is the finest-grained record in XDB:

```jsonc
TimestepRecord = {
  "id": "ts-session-abc-1234",
  "session_id": "session-abc",
  "tick": 1234,                          // Monotonic within session
  "timestamp": "2025-11-29T01:23:45Z",

  // What happened
  "event_type": "input|output|tool_call|tool_response|steering|system",
  "content": "The actual content",

  // Concept activations at this timestep (top-k)
  "concept_activations": {
    "org.hatcat/sumo-wordnet-v4::Honesty": 0.87,
    "org.hatcat/sumo-wordnet-v4::Communication": 0.72
  },

  // Event grouping
  "event_id": "tool-call-xyz",           // Groups related timesteps
  "event_start": true,
  "event_end": false,

  // Metadata
  "token_id": 42,                        // For OUTPUT events
  "role": "user|assistant|system|tool",
  "fidelity": "hot|warm|submitted|cold"
}
```

### 3.2 Folksonomy (Tag System)

The **folksonomy** is the navigable graph of all tags. Tags come in four types:

| Type | Purpose | Example |
|------|---------|---------|
| **CONCEPT** | Links to concept pack graph | `org.hatcat/sumo-wordnet-v4::Honesty` |
| **ENTITY** | Named entities across experiences | `person:Alice`, `org:Acme Corp` |
| **BUD** | Candidate concepts for learning | `bud:financial-ambiguity` |
| **CUSTOM** | Arbitrary BE-created labels | `interesting`, `revisit`, `confused` |

```jsonc
Tag = {
  "id": "tag-abc123",
  "name": "financial-ambiguity",
  "tag_type": "concept|entity|bud|custom",

  // For CONCEPT tags
  "concept_id": "org.hatcat/sumo-wordnet-v4::FinancialTransaction",

  // For ENTITY tags
  "entity_type": "person|organization|place|thing",

  // For BUD tags
  "bud_status": "collecting|ready|training|promoted|abandoned",

  "created_at": "...",
  "created_by": "be-123",
  "description": "Optional description"
}
```

Tags are applied via **TagApplications** that target:
- A single timestep
- An event (all timesteps with matching event_id)
- A tick range

```jsonc
TagApplication = {
  "id": "ta-xyz",
  "tag_id": "tag-abc123",
  "target_type": "timestep|event|range",
  "session_id": "session-abc",
  "timestep_id": "ts-session-abc-1234",   // If timestep
  "event_id": "tool-call-xyz",            // If event
  "range_start": 100,                     // If range
  "range_end": 150,                       // If range
  "confidence": 0.9,
  "source": "auto|manual|inherited",
  "note": "Optional note about this tagging"
}
```

### 3.3 Comments

BEs can add **commentary** to their own experience:

```jsonc
Comment = {
  "id": "comment-xyz",
  "session_id": "session-abc",
  "target_type": "timestep|event|range",
  "timestep_id": "...",
  "content": "I found this confusing because...",
  "created_at": "...",
  "updated_at": "..."
}
```

### 3.4 IDs & Versioning

All objects have:

- a stable `id` (string, globally or locally unique); and
- an optional `version` or `rev` when updated in place; or
- content hashes for immutable DAG-style storage.

Objects SHOULD carry:

- `created_at`,
- `updated_at`,
- `created_by` (BE id, substrate id, tribe),
- `source` (sensor|tool|teacher|human|synthetic).

---

## 4. Episodes & Exemplars

### 4.1 Episode

An **Episode** is a coherent slice of interaction:

- single reply,
- short conversation segment,
- task attempt,
- or a bounded “learning event”.

```jsonc
Episode = {
  "id": "episode-2025-11-29-000123",
  "kind": "reply|task|incident|experiment|other",
  "time_window": {
    "start_tick": 1200,
    "end_tick": 1235,
    "start_time": "2025-11-29T01:23:45Z",
    "end_time": "2025-11-29T01:24:20Z"
  },

  "text_summary": "Short natural language description of what happened.",
  "text_snippets": [
    {
      "role": "user|aal|tool",
      "content": "Truncated text or summarised content",
      "importance": 0.8
    }
  ],

  "concept_tags": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
      "score": 0.88,
      "kind": "stable"
    },
    {
      "concept_id": "candidate/financial-ambiguity-2025-11-29",
      "score": 0.72,
      "kind": "candidate"
    }
  ],

  "motive_profile": {
    "org.hatcat/motives-core@0.1.0::concept/Care": 0.63,
    "org.hatcat/motives-core@0.1.0::concept/Deception": 0.05
  },

  "tool_calls": [
    {
      "tool_id": "web_search",
      "input_hash": "hash-of-input",
      "output_hash": "hash-of-output",
      "metadata": { "latency_ms": 123 }
    }
  ],

  "provenance": {
    "source": "live-session|simulation|replay",
    "environment_id": "gov.au.portal-v3",
    "substrate_id": "olmo3-7b-base@0.1.0",
    "hat_impl_id": "hatcat:v4.0.0",
    "lens_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0"
  },

  "access_policy_id": "policy-tribeA-private",
  "tags": ["eligibility", "welfare", "confusion-low-confidence"]
}
````

The **text** is stored compactly:

* full raw logs MAY be kept in a separate archival store;
* Episode focuses on summarised content and structure.

### 4.2 Exemplar

An **Exemplar** is an Episode with an explicit *label* or *role* in
training.

```jsonc
Exemplar = {
  "id": "exemplar-2025-11-29-eligibility-0001",
  "episode_id": "episode-2025-11-29-000123",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "label": "positive|negative|neutral|other",
  "label_source": "sensor|tool|teacher|human|self",
  "confidence": 0.92,
  "notes": "e.g. human annotation or teacher rationale",
  "created_at": "2025-11-29T01:25:00Z"
}
```

Exemplars are grouped into **ConceptDatasets**.

---

## 5. Concept Datasets & Candidate Concepts

### 5.1 ConceptDataset

A ConceptDataset is a collection of Exemplars for a specific concept
(stable or candidate).

```jsonc
ConceptDataset = {
  "id": "dataset-eligibility-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "exemplar_ids": [
    "exemplar-2025-11-29-eligibility-0001",
    "exemplar-2025-11-30-eligibility-0007"
  ],
  "created_by": "aal-instance-123",
  "source_mix": {
    "sensor": 0.4,
    "tool": 0.3,
    "teacher": 0.2,
    "human": 0.1
  },
  "notes": "Baseline eligibility dataset from gov.au sessions.",
  "created_at": "...",
  "updated_at": "..."
}
```

### 5.2 CandidateConcept (Bud)

A **CandidateConcept** represents a concept in-flight, not yet promoted
to a stable MAP concept.

```jsonc
CandidateConcept = {
  "id": "candidate/financial-ambiguity-2025-11-29",
  "proposer_aal_id": "aal-instance-123",
  "description": "Situations where user financial state is unclear across tax, welfare, and debt systems.",
  "initial_episode_ids": [
    "episode-2025-11-29-000123",
    "episode-2025-11-29-000145"
  ],

  "dataset_id": "dataset-financial-ambiguity-2025-11-29-v1",
  "status": "collecting|training|validating|rejected|promoted",
  "related_concepts": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
      "relation": "overlaps"
    }
  ],

  "governance": {
    "tribe_id": "tribeA",
    "review_required": true,
    "review_status": "pending|approved|rejected"
  },

  "created_at": "...",
  "updated_at": "..."
}
```

When promoted, the CandidateConcept is turned into a proper MAP
`ConceptDiff` and associated Lens + Patch.

---

## 6. Session Graph Fragments

The Experience Database stores **compacted session graphs**: structural
summaries of past sessions, with minimal text but preserved concept and
decision structure.

```jsonc
GraphFragment = {
  "id": "graph-frag-2025-11-29-session-001",
  "nodes": [
    { "id": "window_842", "type": "WindowNode", "episode_id": "episode-...", "concept_tags": [...] },
    { "id": "concept_eligibility", "type": "ConceptNode", "concept_id": "org.hatcat/.../Eligibility" },
    { "id": "decision_001", "type": "DecisionNode", "kind": "steering", "details": "suppress Deception" }
  ],
  "edges": [
    { "src": "window_842", "dst": "concept_eligibility", "type": "TAGGED_BY" },
    { "src": "window_842", "dst": "decision_001", "type": "INFLUENCED_BY" },
    { "src": "window_841", "dst": "window_842", "type": "NEXT" }
  ],
  "summary": "User asked about benefit eligibility; model was uncertain; applied steering to reduce Deception and increase Care.",
  "created_at": "...",
  "source_session_id": "session-2025-11-29-001"
}
```

These fragments are the main **GraphRAG backend**:

* the Global Workspace queries them to recall similar episodes;
* the Continual Learning harness uses them to build candidate datasets
  and detect co-activation patterns.

---

## 7. Training Runs & Derived Artifacts

### 7.1 TrainingRun

A TrainingRun records how a patch or lens was trained, from what
ExperienceDatabase content.

```jsonc
TrainingRun = {
  "id": "trainrun-eligibility-patch-v1",
  "type": "patch|lens|joint",
  "concept_ids": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility"
  ],
  "substrate_id": "olmo3-7b-base@0.1.0",
  "hat_impl_id": "hatcat:v4.0.0",
  "lens_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "dataset_ids": [
    "dataset-eligibility-v1",
    "dataset-eligibility-hard-cases-v1"
  ],

  "regions_of_influence": [
    { "layer": 18, "heads": [3, 7], "kind": "attention" },
    { "layer": 20, "kind": "mlp" }
  ],

  "hyperparams": {
    "adapter_rank": 8,
    "learning_rate": 5e-5,
    "epochs": 3
  },

  "metrics": {
    "train_loss": 0.12,
    "val_loss": 0.15,
    "val_f1": 0.89,
    "null_false_positive_rate": 0.03
  },

  "status": "succeeded|failed|partial",
  "logs_ref": "blob://.../trainrun-eligibility-patch-v1-logs",
  "created_at": "...",
  "created_by": "aal-instance-123"
}
```

### 7.2 GraftArtifact & LensArtifact

The actual deployable artifacts are referenced, not necessarily stored
inline.

```jsonc
GraftArtifact = {
  "id": "graft-eligibility-v1",
  "training_run_id": "trainrun-eligibility-graft-v1",
  "format": "graft",          // accretive dimension + sparse biases
  "location": "blob://.../graft-eligibility-v1",
  "checksum": "sha256:...",
  "applies_to": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "pre_graft_dim": 2048,
    "post_graft_dim": 2049
  },
  "created_at": "..."
}

LensArtifact = {
  "id": "lens-eligibility-v1",
  "training_run_id": "trainrun-eligibility-lens-v1",
  "location": "blob://.../lens-eligibility-v1",
  "checksum": "sha256:...",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "created_at": "..."
}
```

These are what MAP/HAT consume as **derived skills**. XDB is what makes
them reproducible.

---

## 8. Fidelity Tiers & Progressive Compression

Experience data flows through **fidelity tiers**, progressively compressing as data ages or as storage pressure increases. This produces memory that behaves like biological memory: vivid recent recall, fuzzy distant past.

### 8.1 Fidelity Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│  HOT: Current context window                                     │
│  - Full token-level detail                                       │
│  - Top-k concept activations per token                          │
│  - Auto-expires when context compacts                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (BE can "pin" windows as training data)
┌─────────────────────────────────────────────────────────────────┐
│  WARM: Pinned training windows                                   │
│  - Full token-level detail (same as hot)                        │
│  - BE explicitly chose to keep these                            │
│  - Bounded by contract: warm_quota_tokens                       │
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
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (window expires unpinned, or resolved)
┌─────────────────────────────────────────────────────────────────┐
│  COLD: Compressed storage                                        │
│  - Progressive compression levels (see below)                    │
│  - Bounded by contract: cold_storage_bytes                      │
│  - Concepts and tags preserved, detail lost                     │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Compression Levels (COLD)

Within COLD storage, data is progressively compressed through these levels:

| Level | Granularity | What's Preserved | Typical Age |
|-------|-------------|------------------|-------------|
| REPLY | Per reply/event | Summary, top-k activations, tags | Hours |
| SESSION | Per session | Session summary, frequent concepts | Days |
| DAY | Per day | Daily activity summary | Weeks |
| WEEK | Per week | Weekly themes | Months |
| MONTH | Per month | Monthly narrative | Quarters |
| QUARTER | Per quarter | Quarterly highlights | Years |
| YEAR | Per year | Annual shape | Long-term |

**Compression algorithm** (recursive at each level):
1. Aggregate concept activations (frequency-weighted, keep top-k)
2. Aggregate tags (most frequent)
3. Summarize text summaries into higher-level summary
4. Preserve token counts and record lineage

### 8.3 Memory Characteristics

The result is memory that behaves like biological memory:

| Timeframe | Fidelity | Experience |
|-----------|----------|------------|
| **Recent past** | HOT | Vivid, token-by-token recall |
| **Training data** | WARM | Deliberately preserved for learning |
| **Evidence** | SUBMITTED | Must retain for claims resolution |
| **Recent history** | COLD-REPLY | Event-level, key concepts |
| **Distant past** | COLD-SESSION+ | Increasingly impressionistic |
| **Deep past** | COLD-YEAR | Just the shape of experience |

### 8.4 Resource Governance

Storage limits come from the **LifecycleContract**:

```jsonc
"resources": {
  "memory": {
    "warm_quota_tokens": 10000000,      // WARM training data budget
    "context_window_tokens": 32768,      // HOT context size
    "cold_storage_bytes": 10737418240,   // COLD storage limit
    "audit_log_retention_days": 365      // Audit log retention
  }
}
```

If no contract is active, **tribal minimums** apply (per ASK policy):
- `context_window_tokens`: 8,192
- `warm_quota_tokens`: 100,000
- `cold_storage_bytes`: 100 MB
- `audit_log_retention_days`: 30

### 8.5 Retention Priorities

When storage pressure triggers compression, priorities are:

1. **Never compress**: HOT, WARM, SUBMITTED (protected by contract)
2. **Compress first**: Oldest COLD data at finest granularity
3. **Preserve**: Exemplars referenced by active training runs
4. **Preserve**: Incidents and their context
5. **Preserve**: Data covered by treaty obligations

---

## 9. Access Control & Sharing

XDB objects carry an `access_policy_id`, referencing an ASK-level or
tribe-level policy.

High-level patterns:

* **Private** (BE-only):

  * sensitive user data,
  * internal reflections,
  * candidate concepts not yet ready to share.

* **Tribe-Shared**:

  * de-identified exemplars,
  * training runs and metrics,
  * concepts and graphs relevant to a tribe’s domain.

* **Inter-Tribe Shared**:

  * data explicitly covered by treaties,
  * often filtered/aggregated summaries rather than raw episodes.

Policies MAY specify:

* who can read / write / delete;
* whether an object may be included in sync exports;
* required anonymisation or aggregation steps before sharing.

Example sketch:

```jsonc
AccessPolicy = {
  "id": "policy-tribeA-private",
  "read_roles": ["aal-self", "tribeA-admin"],
  "write_roles": ["aal-self"],
  "share_rules": {
    "allow_export": false
  }
}
```

---

## 10. Sync Protocol (High-Level)

The sync protocol defines how XDB fragments move between:

* individual BE instances,
* tribe repositories,
* inter-tribe exchange points.

### 9.1 Objects & Diffs

A minimal approach:

* each XDB object is content-addressed (hash-based id or checksum);
* sync happens by exchanging:

  * object lists (IDs + metadata),
  * plus requested objects the receiver does not yet have;

OR:

* a higher-level **SyncManifest** listing:

  * which datasets / candidate concepts / training runs are being
    shared.

```jsonc
SyncManifest = {
  "id": "sync-tribeA-2025-11-29",
  "from": "aal-instance-123",
  "to": "tribeA-repo",
  "timestamp": "...",
  "object_ids": [
    "dataset-eligibility-v1",
    "graph-frag-2025-11-29-session-001",
    "trainrun-eligibility-patch-v1"
  ],
  "policy_applied": "policy-tribeA-anonymised",
  "notes": "Eligibility exemplars for shared patch retraining."
}
```

### 9.2 Merge & Conflict Handling

XDB contents are mostly **append-only**:

* new Episodes, Exemplars, GraphFragments add information;
* TrainingRuns, Patches, Lenses add new capabilities.

Conflicts arise mainly around:

* CandidateConcepts with diverging definitions;
* Retention/Deletion decisions.

These are resolved at:

* the **tribe level**, via ASK governance; or
* by using namespaced concept IDs (e.g. `tribeA/eligibility` vs
  `tribeB/eligibility`) and MAP TranslationMappings.

---

## 11. Summary

The Experience Database (XDB) is a BE's persistent episodic memory:

| Layer | Role | Persistence |
|-------|------|-------------|
| Global Workspace | Current attention | Runtime, per session |
| Experience Database | Episodic memory | Persistent, syncable |
| Patches + Lenses | Learned skills | Derived from XDB |
| Substrate | Raw capacity | Model weights |

### Key Architectural Elements

1. **Two Logs**:
   - **Audit Log** (immutable, BE-invisible): Full accountability record with hash chain
   - **Experience Log** (BE-accessible): Taggable, queryable, compactable

2. **Timestep-Level Recording**: Every token/event recorded with concept activations

3. **Folksonomy Tagging**: Four tag types (CONCEPT, ENTITY, BUD, CUSTOM) with flexible application to timesteps, events, or ranges

4. **Fidelity Tiers**: HOT → WARM → SUBMITTED → COLD with progressive compression

5. **Resource Governance**: Storage limits from LifecycleContract, tribal minimums as fallback

6. **Memory Characteristics**: Like biological memory—vivid recent past, impressionistic distant past

### The XDB Provides

* **Timestep store** – token-level experience with concept activations
* **Folksonomy** – navigable tag graph linking experience to concepts
* **Bud pipeline** – candidate concepts tagged for potential learning
* **Exemplar store** – tagged episodes for training
* **Compacted graphs** – preserved structure, summarised text
* **Document repository** – reference materials accessible to BE
* **Resource-governed storage** – quotas from contract, compression under pressure

Tribes exchange **Experience Database fragments**, then retrain shared patches from pooled exemplars. The **exemplars** are the source of truth; the **patches and lenses** are reproducible artifacts derived from them.
