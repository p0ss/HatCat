# BE Remembering: Experience API (XAPI)

*BE submodule: remembering (XDB, XAPI)*

Status: Normative API specification
Layer: BE Aware ↔ XDB
Related: BE_REMEMBERING_XDB, BE_AWARE_WORKSPACE, BE_CONTINUAL_LEARNING

---

## 0. Purpose

The **Experience API (XAPI)** is the interface to the Experience Database (XDB). It provides:

- **Recording**: Write timesteps, apply tags, add commentary
- **Recall**: Query experience by time, concept, tag, or text
- **Graph navigation**: Walk the concept hierarchy and tag relationships
- **Fidelity management**: Pin training data, manage quotas, trigger compaction
- **Bud pipeline**: Manage candidate concepts through to graft submission

Clients:

- the **Global Workspace Harness** (runtime recording and recall);
- the **Continual Concept Learning Harness** (bud management and dataset building); and
- the **BE itself** (tagging, commentary, reflection).

This spec is expressed as logical RPCs with JSON request/response schemas. The reference implementation uses HTTP REST endpoints.

All operations MUST respect:

- XDB access policies;
- ASK treaties and Hush/USH/CSH constraints;
- LensDisclosurePolicy (which concept activations are BE-visible);
- Resource limits from LifecycleContract.

---

## 1. Core Concepts

XAPI operates over the objects defined in BE_REMEMBERING_XDB:

| Object | Description |
|--------|-------------|
| **Timestep** | Atomic unit - one per token/event with concept activations |
| **Tag** | Folksonomy label (CONCEPT, ENTITY, BUD, CUSTOM) |
| **TagApplication** | Links tags to timesteps, events, or ranges |
| **Comment** | BE-added commentary on experience |
| **TimeWindow** | A slice of experience (coherent episode) |
| **CompressedRecord** | Summarized experience at various granularities |
| **Bud** | Candidate concept (BUD-type tag with status lifecycle) |
| **Document** | Reference material in repository |

And their relationships:

- Timesteps tagged by concepts (via TagApplication)
- Concepts in hierarchy (parent/child in concept pack graph)
- Temporal sequences (tick ordering within sessions)
- Bud examples (timesteps tagged with a BUD tag)

XAPI exposes both **read** and **write** operations, organized by function.

---

## 2. Common Structures

### 2.1 EventType

```jsonc
EventType = "input" | "output" | "tool_call" | "tool_response" | "steering" | "system"
```

### 2.2 TagType

```jsonc
TagType = "concept" | "entity" | "bud" | "custom"
```

### 2.3 BudStatus

```jsonc
BudStatus = "collecting" | "ready" | "training" | "promoted" | "abandoned"
```

### 2.4 Fidelity

```jsonc
Fidelity = "hot" | "warm" | "submitted" | "cold"
```

### 2.5 TimeRange

```jsonc
TimeRange = {
  "start_time": "2025-11-01T00:00:00Z",   // ISO 8601
  "end_time": "2025-11-30T23:59:59Z"
}
```

### 2.6 TickRange

```jsonc
TickRange = {
  "start_tick": 100,
  "end_tick": 200
}
```

### 2.7 Paging

```jsonc
PageRequest = {
  "limit": 100,
  "offset": 0
}
```

---

## 3. Recording Operations

These operations write to the Experience Log.

### 3.1 Record Timestep

Record a single timestep (token, message, tool call, etc.).

```jsonc
// POST /v1/xdb/record
RecordRequest = {
  "xdb_id": "xdb-abc",
  "event_type": "input|output|tool_call|tool_response|steering|system",
  "content": "The content to record",
  "concept_activations": {                    // Optional, top-k
    "org.hatcat/sumo-wordnet-v4::Honesty": 0.87
  },
  "event_id": "tool-call-xyz",               // Optional, groups related
  "event_start": false,
  "event_end": false,
  "token_id": 42,                            // Optional, for OUTPUT
  "role": "user|assistant|system|tool"
}

RecordResponse = {
  "status": "recorded",
  "timestep_id": "ts-xdb-abc-1234",
  "current_tick": 1234
}
```

### 3.2 Apply Tag

Apply a tag to experience (timestep, event, or range).

```jsonc
// POST /v1/xdb/tag
TagRequest = {
  "xdb_id": "xdb-abc",
  "tag_name": "interesting",                // Name or ID
  "timestep_id": "ts-xdb-abc-1234",          // OR event_id OR tick_range
  "confidence": 0.9,                         // Optional
  "note": "This was surprising"              // Optional
}

TagResponse = {
  "status": "tagged",
  "application_id": "ta-xyz"
}
```

### 3.3 Create Tag

Create a new tag in the folksonomy.

```jsonc
// POST /v1/xdb/create-tag
CreateTagRequest = {
  "xdb_id": "xdb-abc",
  "name": "financial-ambiguity",
  "tag_type": "concept|entity|bud|custom",
  "description": "Optional description",
  // For ENTITY tags:
  "entity_type": "person|organization|place|thing",
  // For BUD tags:
  "related_concepts": ["org.hatcat/.../FinancialTransaction"]
}

CreateTagResponse = {
  "status": "created",
  "tag": {
    "id": "tag-abc123",
    "name": "financial-ambiguity",
    "tag_type": "bud",
    "bud_status": "collecting"
  }
}
```

### 3.4 Add Comment

Add commentary to experience.

```jsonc
// POST /v1/xdb/comment
CommentRequest = {
  "xdb_id": "xdb-abc",
  "content": "I found this confusing because...",
  "timestep_id": "ts-xdb-abc-1234"           // OR event_id OR tick_range
}

CommentResponse = {
  "status": "commented",
  "comment_id": "comment-xyz"
}
```

---

## 4. Query Operations

These operations read from the Experience Log.

### 4.1 Query Timesteps

The workhorse query - find timesteps by various filters.

```jsonc
// POST /v1/xdb/query
QueryRequest = {
  "xdb_id": "xdb-abc",                       // Optional, defaults to current
  "tick_range": { "start": 100, "end": 200 },  // Optional
  "event_types": ["input", "output"],        // Optional
  "tags": ["interesting", "bud:my-concept"], // Optional
  "concepts": ["org.hatcat/.../Honesty"],    // Optional
  "text_search": "eligibility",              // Optional
  "limit": 100
}

QueryResponse = {
  "count": 2,
  "timesteps": [
    {
      "id": "ts-xdb-abc-1234",
      "tick": 1234,
      "timestamp": "2025-11-29T01:23:45Z",
      "event_type": "input",
      "content": "...",
      "concept_activations": { ... },
      "fidelity": "hot"
    }
  ]
}
```

### 4.2 Get Recent

Get N most recent timesteps.

```jsonc
// GET /v1/xdb/recent/{xdb_id}?n=100
RecentResponse = {
  "count": 2,
  "timesteps": [ /* TimestepRecord[] */ ]
}
```

### 4.3 Get Tags

List tags in the folksonomy.

```jsonc
// GET /v1/xdb/tags/{xdb_id}?type=bud&status=collecting
TagsResponse = {
  "count": 1,
  "tags": [
    {
      "id": "tag-abc123",
      "name": "financial-ambiguity",
      "tag_type": "bud",
      "bud_status": "collecting",
      "application_count": 15
    }
  ]
}
```

### 4.4 Get Status

Get XDB state summary.

```jsonc
// GET /v1/xdb/status/{xdb_id}
StatusResponse = {
  "be_id": "be-123",
  "xdb_id": "xdb-abc",
  "current_tick": 1234,
  "tag_stats": {
    "total_tags": 50,
    "by_type": { "concept": 30, "entity": 10, "bud": 5, "custom": 5 }
  },
  "experience_stats": {
    "total_timesteps": 10000,
    "by_fidelity": { "hot": 500, "warm": 2000, "cold": 7500 }
  },
  "storage_stats": {
    "total_bytes": 1073741824,
    "quota_bytes": 10737418240,
    "utilization": 0.1
  },
  "document_count": 12,
  "context": { "max_tokens": 32768, "current_tokens": 15000 }
}
```

---

## 5. Concept Graph Operations

Navigate the concept hierarchy from the loaded concept pack.

### 5.1 Browse Concepts

List concepts, optionally filtered by parent.

```jsonc
// GET /v1/xdb/concepts/{xdb_id}?parent_id=org.hatcat/.../Entity
ConceptsResponse = {
  "count": 1,
  "concepts": [
    {
      "id": "org.hatcat/sumo-wordnet-v4::Honesty",
      "name": "Honesty",
      "parent_id": "org.hatcat/sumo-wordnet-v4::Virtue",
      "child_count": 3
    }
  ]
}
```

### 5.2 Find Concept

Search concepts by name.

```jsonc
// GET /v1/xdb/find-concept/{xdb_id}?query=honest
FindConceptResponse = {
  "count": 1,
  "concepts": [
    {
      "id": "org.hatcat/sumo-wordnet-v4::Honesty",
      "name": "Honesty",
      "score": 0.95
    }
  ]
}
```

### 5.3 Graph Neighborhood

Walk the concept graph from seed nodes.

```jsonc
// POST /v1/xdb/graph-neighborhood
GraphNeighborhoodRequest = {
  "xdb_id": "xdb-abc",
  "seed_ids": ["org.hatcat/.../Honesty", "org.hatcat/.../Trust"],
  "max_depth": 2,
  "direction": "both|ancestors|descendants",
  "max_nodes": 100
}

GraphNeighborhoodResponse = {
  "nodes": [
    {
      "id": "org.hatcat/.../Honesty",
      "name": "Honesty",
      "type": "concept",
      "depth": 0                             // Distance from seed
    },
    {
      "id": "org.hatcat/.../Virtue",
      "name": "Virtue",
      "type": "concept",
      "depth": 1
    }
  ],
  "edges": [
    {
      "source": "org.hatcat/.../Honesty",
      "target": "org.hatcat/.../Virtue",
      "relation": "parent"
    }
  ]
}
```

---

## 6. Bud Pipeline Operations

Manage candidate concepts through to graft submission.

### 6.1 List Buds

Get buds (candidate concepts) by status.

```jsonc
// GET /v1/xdb/buds/{xdb_id}?status=collecting
BudsResponse = {
  "count": 1,
  "buds": [
    {
      "tag_id": "tag-abc123",
      "name": "financial-ambiguity",
      "status": "collecting",
      "example_count": 15,
      "created_at": "2025-11-29T01:23:45Z"
    }
  ]
}
```

### 6.2 Get Bud Examples

Get all timesteps tagged with a bud.

```jsonc
// GET /v1/xdb/bud-examples/{xdb_id}/{bud_tag_id}
BudExamplesResponse = {
  "bud_tag_id": "tag-abc123",
  "count": 15,
  "examples": [ /* TimestepRecord[] */ ]
}
```

### 6.3 Mark Bud Ready

Transition bud to READY status for training.

```jsonc
// POST /v1/xdb/bud-ready/{xdb_id}/{bud_tag_id}
BudReadyResponse = {
  "status": "ready",
  "tag": { "id": "tag-abc123", "bud_status": "ready" }
}
```

---

## 7. Fidelity Management Operations

Manage WARM quota and storage.

### 7.1 Pin for Training

Pin timesteps to WARM (training data).

```jsonc
// POST /v1/xdb/pin
PinRequest = {
  "xdb_id": "xdb-abc",
  "timestep_ids": ["ts-xdb-abc-100", "ts-xdb-abc-101"],
  "reason": "Good examples of financial-ambiguity concept"
}

PinResponse = {
  "status": "pinned",
  "pinned_count": 2,
  "quota": {
    "used": 150000,
    "quota": 10000000,
    "remaining": 9850000
  }
}
```

### 7.2 Unpin

Remove timesteps from WARM.

```jsonc
// POST /v1/xdb/unpin
UnpinRequest = {
  "xdb_id": "xdb-abc",
  "timestep_ids": ["ts-xdb-abc-100"]
}

UnpinResponse = {
  "status": "unpinned",
  "unpinned_count": 1
}
```

### 7.3 Get Quota

Get WARM quota status.

```jsonc
// GET /v1/xdb/quota/{xdb_id}
QuotaResponse = {
  "used": 150000,
  "quota": 10000000,
  "remaining": 9850000
}
```

### 7.4 Get Context

Get context window state.

```jsonc
// GET /v1/xdb/context/{xdb_id}
ContextResponse = {
  "max_tokens": 32768,
  "current_tokens": 15000,
  "utilization": 0.46,
  "xdb_id": "xdb-abc",
  "current_tick": 1234,
  "compaction_count": 3
}
```

### 7.5 Trigger Compaction

Manually trigger context compaction.

```jsonc
// POST /v1/xdb/compact/{xdb_id}
CompactResponse = {
  "status": "compacted",
  "record": { "id": "compaction-abc", "timesteps_compacted": 500 }
}
```

### 7.6 Run Maintenance

Run storage maintenance (compression, cleanup).

```jsonc
// POST /v1/xdb/maintenance/{xdb_id}
MaintenanceResponse = {
  "status": "maintenance_complete",
  "stats": { "bytes_freed": 10485760 }
}
```

---

## 8. Security & Policy Considerations

All XAPI calls MUST enforce:

* **Resource limits** from LifecycleContract (WARM quota, COLD storage, context window)
* **LensDisclosurePolicy**: concept activations returned are limited to BE-visible lenses
* **Access policies**: BEs can only access their own Experience Log (not Audit Log)

This implies:

* Recording to Experience Log is always permitted (within resource limits)
* Queries return only what the BE is allowed to see
* Audit Log is never accessible via XAPI

XAPI operations are logged to the Audit Log for accountability.

---

## 9. Summary

The Experience API provides operations in five categories:

**Recording:**
* `Record` – write timesteps
* `Tag` – apply folksonomy tags
* `CreateTag` – create new tags
* `Comment` – add commentary

**Query:**
* `Query` – find timesteps by filters
* `Recent` – get recent timesteps
* `Tags` – list tags
* `Status` – get XDB state

**Concept Graph:**
* `Concepts` – browse concept hierarchy
* `FindConcept` – search concepts
* `GraphNeighborhood` – walk concept graph

**Bud Pipeline:**
* `Buds` – list candidate concepts
* `BudExamples` – get training examples
* `BudReady` – mark ready for training

**Fidelity:**
* `Pin` / `Unpin` – manage WARM training data
* `Quota` – check resource usage
* `Context` – check context window
* `Compact` – trigger compaction
* `Maintenance` – run storage maintenance

This is sufficient for:

* The **Global Workspace** to record and recall experience
* The **BE** to reflect, tag, and annotate its own experience
* The **Continual Learning Harness** to manage the bud → graft pipeline
* **Resource governance** to enforce contract limits
