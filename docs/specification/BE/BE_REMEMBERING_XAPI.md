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
  "session_id": "session-abc",
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
  "timestep_id": "ts-session-abc-1234",
  "tick": 1234
}
```

### 3.2 Apply Tag

Apply a tag to experience (timestep, event, or range).

```jsonc
// POST /v1/xdb/tag
TagRequest = {
  "session_id": "session-abc",
  "tag_name_or_id": "interesting",           // Name or ID
  "target": {
    "timestep_id": "ts-session-abc-1234"     // OR
    // "event_id": "tool-call-xyz"           // OR
    // "tick_range": { "start": 100, "end": 200 }
  },
  "confidence": 0.9,                         // Optional
  "note": "This was surprising"              // Optional
}

TagResponse = {
  "application_id": "ta-xyz",
  "tag_id": "tag-abc123",
  "created": true                            // false if tag existed
}
```

### 3.3 Create Tag

Create a new tag in the folksonomy.

```jsonc
// POST /v1/xdb/create-tag
CreateTagRequest = {
  "session_id": "session-abc",
  "name": "financial-ambiguity",
  "tag_type": "concept|entity|bud|custom",
  "description": "Optional description",
  // For ENTITY tags:
  "entity_type": "person|organization|place|thing",
  // For BUD tags:
  "related_concepts": ["org.hatcat/.../FinancialTransaction"]
}

CreateTagResponse = {
  "tag_id": "tag-abc123",
  "tag_type": "bud",
  "bud_status": "collecting"                 // If BUD
}
```

### 3.4 Add Comment

Add commentary to experience.

```jsonc
// POST /v1/xdb/comment
CommentRequest = {
  "session_id": "session-abc",
  "content": "I found this confusing because...",
  "target": {
    "timestep_id": "ts-session-abc-1234"     // OR event_id OR tick_range
  }
}

CommentResponse = {
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
  "session_id": "session-abc",               // Optional, defaults to current
  "tick_range": { "start": 100, "end": 200 },  // Optional
  "time_range": {                            // Optional
    "start_time": "2025-11-01T00:00:00Z",
    "end_time": "2025-11-30T23:59:59Z"
  },
  "event_types": ["input", "output"],        // Optional
  "tags": ["interesting", "bud:my-concept"], // Optional
  "concept_activations": {                   // Optional
    "org.hatcat/.../Honesty": { "min": 0.5, "max": 1.0 }
  },
  "text_search": "eligibility",              // Optional
  "limit": 100
}

QueryResponse = {
  "timesteps": [
    {
      "id": "ts-session-abc-1234",
      "tick": 1234,
      "timestamp": "2025-11-29T01:23:45Z",
      "event_type": "input",
      "content": "...",
      "concept_activations": { ... },
      "fidelity": "hot"
    }
  ],
  "total_count": 452
}
```

### 4.2 Get Recent

Get N most recent timesteps.

```jsonc
// GET /v1/xdb/recent/{session_id}?n=100
RecentResponse = {
  "timesteps": [ /* TimestepRecord[] */ ]
}
```

### 4.3 Get Tags

List tags in the folksonomy.

```jsonc
// GET /v1/xdb/tags/{session_id}?type=bud&status=collecting
TagsResponse = {
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
// GET /v1/xdb/status/{session_id}
StatusResponse = {
  "be_id": "be-123",
  "session_id": "session-abc",
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
  }
}
```

---

## 5. Concept Graph Operations

Navigate the concept hierarchy from the loaded concept pack.

### 5.1 Browse Concepts

List concepts, optionally filtered by parent.

```jsonc
// GET /v1/xdb/concepts/{session_id}?parent_id=org.hatcat/.../Entity
ConceptsResponse = {
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
// GET /v1/xdb/find-concept/{session_id}?query=honest
FindConceptResponse = {
  "matches": [
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
  "session_id": "session-abc",
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
// GET /v1/xdb/buds/{session_id}?status=collecting
BudsResponse = {
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
// GET /v1/xdb/bud-examples/{session_id}/{bud_tag_id}
BudExamplesResponse = {
  "bud": {
    "tag_id": "tag-abc123",
    "name": "financial-ambiguity",
    "status": "collecting"
  },
  "examples": [ /* TimestepRecord[] */ ],
  "example_count": 15
}
```

### 6.3 Mark Bud Ready

Transition bud to READY status for training.

```jsonc
// POST /v1/xdb/bud-ready/{session_id}/{bud_tag_id}
BudReadyResponse = {
  "tag_id": "tag-abc123",
  "previous_status": "collecting",
  "new_status": "ready",
  "example_count": 15
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
  "session_id": "session-abc",
  "timestep_ids": ["ts-session-abc-100", "ts-session-abc-101"],
  "reason": "Good examples of financial-ambiguity concept"
}

PinResponse = {
  "pinned_count": 2,
  "warm_usage": {
    "used_tokens": 150000,
    "quota_tokens": 10000000,
    "remaining_tokens": 9850000
  }
}
```

### 7.2 Unpin

Remove timesteps from WARM.

```jsonc
// POST /v1/xdb/unpin
UnpinRequest = {
  "session_id": "session-abc",
  "timestep_ids": ["ts-session-abc-100"]
}

UnpinResponse = {
  "unpinned_count": 1
}
```

### 7.3 Get Quota

Get WARM quota status.

```jsonc
// GET /v1/xdb/quota/{session_id}
QuotaResponse = {
  "warm": {
    "used_tokens": 150000,
    "quota_tokens": 10000000,
    "remaining_tokens": 9850000
  },
  "cold": {
    "used_bytes": 1073741824,
    "quota_bytes": 10737418240,
    "remaining_bytes": 9663676416
  }
}
```

### 7.4 Get Context

Get context window state.

```jsonc
// GET /v1/xdb/context/{session_id}
ContextResponse = {
  "max_tokens": 32768,
  "current_tokens": 15000,
  "utilization": 0.46,
  "session_id": "session-abc",
  "current_tick": 1234,
  "compaction_count": 3
}
```

### 7.5 Trigger Compaction

Manually trigger context compaction.

```jsonc
// POST /v1/xdb/compact/{session_id}
CompactResponse = {
  "compacted": true,
  "timesteps_compacted": 500,
  "tokens_before": 30000,
  "tokens_after": 5000
}
```

### 7.6 Run Maintenance

Run storage maintenance (compression, cleanup).

```jsonc
// POST /v1/xdb/maintenance/{session_id}
MaintenanceResponse = {
  "records_compressed": 1000,
  "bytes_freed": 10485760,
  "duration_ms": 250
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
