# BE Remembering: Experience Tools (MCP Interface to XAPI)

*BE submodule: remembering (XDB, XAPI)*

Namespace: `xdb`

These MCP tools expose the **Experience API (XAPI)** to a BE. They wrap the HTTP endpoints documented in BE_REMEMBERING_XAPI.md.

All tools:

- Use JSON input arguments
- Return JSON responses
- MUST respect ASK / Hush / AccessPolicy / LensDisclosurePolicy (enforced by the server)
- MUST respect resource limits from LifecycleContract

---

## Recording Tools

### `xdb.record`

Record a timestep to the Experience Log.

```jsonc
{
  "name": "xdb.record",
  "description": "Record a timestep (token, message, tool call, etc.) to the Experience Log.",
  "input_schema": {
    "type": "object",
    "properties": {
      "event_type": {
        "type": "string",
        "enum": ["input", "output", "tool_call", "tool_response", "steering", "system"],
        "description": "Type of event being recorded."
      },
      "content": {
        "type": "string",
        "description": "The content to record."
      },
      "concept_activations": {
        "type": "object",
        "description": "Optional top-k concept activations. Keys are concept IDs, values are scores 0-1."
      },
      "event_id": {
        "type": "string",
        "description": "Optional ID to group related timesteps (e.g., tool call and response)."
      },
      "event_start": { "type": "boolean", "default": false },
      "event_end": { "type": "boolean", "default": false },
      "token_id": {
        "type": "integer",
        "description": "For OUTPUT events, the token ID."
      },
      "role": {
        "type": "string",
        "enum": ["user", "assistant", "system", "tool"]
      }
    },
    "required": ["event_type", "content"]
  }
}
```

---

### `xdb.tag`

Apply a tag to experience.

```jsonc
{
  "name": "xdb.tag",
  "description": "Apply a folksonomy tag to a timestep, event, or tick range.",
  "input_schema": {
    "type": "object",
    "properties": {
      "tag_name_or_id": {
        "type": "string",
        "description": "Tag name or ID. Creates new CUSTOM tag if not found."
      },
      "timestep_id": {
        "type": "string",
        "description": "Target a specific timestep."
      },
      "event_id": {
        "type": "string",
        "description": "Target all timesteps in an event."
      },
      "tick_range": {
        "type": "object",
        "properties": {
          "start": { "type": "integer" },
          "end": { "type": "integer" }
        },
        "description": "Target a range of ticks."
      },
      "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 1.0
      },
      "note": {
        "type": "string",
        "description": "Optional note about this tagging."
      }
    },
    "required": ["tag_name_or_id"]
  }
}
```

---

### `xdb.create_tag`

Create a new tag in the folksonomy.

```jsonc
{
  "name": "xdb.create_tag",
  "description": "Create a new tag (concept, entity, bud, or custom).",
  "input_schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Tag name."
      },
      "tag_type": {
        "type": "string",
        "enum": ["concept", "entity", "bud", "custom"],
        "description": "Type of tag to create."
      },
      "description": {
        "type": "string",
        "description": "Optional description."
      },
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "place", "thing"],
        "description": "For ENTITY tags, the entity type."
      },
      "related_concepts": {
        "type": "array",
        "items": { "type": "string" },
        "description": "For BUD tags, related concept IDs."
      }
    },
    "required": ["name", "tag_type"]
  }
}
```

---

### `xdb.comment`

Add commentary to experience.

```jsonc
{
  "name": "xdb.comment",
  "description": "Add a comment to a timestep, event, or tick range.",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The comment content."
      },
      "timestep_id": { "type": "string" },
      "event_id": { "type": "string" },
      "tick_range": {
        "type": "object",
        "properties": {
          "start": { "type": "integer" },
          "end": { "type": "integer" }
        }
      }
    },
    "required": ["content"]
  }
}
```

---

## Query Tools

### `xdb.query`

Query timesteps with filters.

```jsonc
{
  "name": "xdb.query",
  "description": "Query the Experience Log for timesteps matching filters.",
  "input_schema": {
    "type": "object",
    "properties": {
      "session_id": {
        "type": "string",
        "description": "Session to query. Defaults to current."
      },
      "tick_range": {
        "type": "object",
        "properties": {
          "start": { "type": "integer" },
          "end": { "type": "integer" }
        }
      },
      "time_range": {
        "type": "object",
        "properties": {
          "start_time": { "type": "string", "format": "date-time" },
          "end_time": { "type": "string", "format": "date-time" }
        }
      },
      "event_types": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["input", "output", "tool_call", "tool_response", "steering", "system"]
        }
      },
      "tags": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter by tag names or IDs."
      },
      "concept_activations": {
        "type": "object",
        "description": "Filter by concept activation ranges. Keys are concept IDs, values are {min, max}."
      },
      "text_search": {
        "type": "string",
        "description": "Full-text search in content."
      },
      "limit": {
        "type": "integer",
        "default": 100
      }
    }
  }
}
```

---

### `xdb.recent`

Get recent timesteps.

```jsonc
{
  "name": "xdb.recent",
  "description": "Get the N most recent timesteps.",
  "input_schema": {
    "type": "object",
    "properties": {
      "n": {
        "type": "integer",
        "default": 100,
        "description": "Number of timesteps to return."
      }
    }
  }
}
```

---

### `xdb.tags`

List tags in the folksonomy.

```jsonc
{
  "name": "xdb.tags",
  "description": "List tags, optionally filtered by type or status.",
  "input_schema": {
    "type": "object",
    "properties": {
      "tag_type": {
        "type": "string",
        "enum": ["concept", "entity", "bud", "custom"]
      },
      "bud_status": {
        "type": "string",
        "enum": ["collecting", "ready", "training", "promoted", "abandoned"],
        "description": "For BUD tags, filter by status."
      }
    }
  }
}
```

---

### `xdb.status`

Get XDB state summary.

```jsonc
{
  "name": "xdb.status",
  "description": "Get XDB state including tag counts, storage usage, and session info.",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

---

## Concept Graph Tools

### `xdb.concepts`

Browse concept hierarchy.

```jsonc
{
  "name": "xdb.concepts",
  "description": "List concepts from the concept pack, optionally filtered by parent.",
  "input_schema": {
    "type": "object",
    "properties": {
      "parent_id": {
        "type": "string",
        "description": "If provided, list children of this concept. Otherwise list roots."
      },
      "limit": {
        "type": "integer",
        "default": 100
      }
    }
  }
}
```

---

### `xdb.find_concept`

Search concepts by name.

```jsonc
{
  "name": "xdb.find_concept",
  "description": "Search for concepts by name.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query."
      },
      "limit": {
        "type": "integer",
        "default": 20
      }
    },
    "required": ["query"]
  }
}
```

---

### `xdb.graph_neighborhood`

Walk the concept graph.

```jsonc
{
  "name": "xdb.graph_neighborhood",
  "description": "Walk the concept graph from seed nodes, returning nodes and edges.",
  "input_schema": {
    "type": "object",
    "properties": {
      "seed_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Concept IDs to start from."
      },
      "max_depth": {
        "type": "integer",
        "default": 2,
        "description": "Maximum depth to traverse."
      },
      "direction": {
        "type": "string",
        "enum": ["both", "ancestors", "descendants"],
        "default": "both"
      },
      "max_nodes": {
        "type": "integer",
        "default": 100
      }
    },
    "required": ["seed_ids"]
  }
}
```

---

## Bud Pipeline Tools

### `xdb.buds`

List buds (candidate concepts).

```jsonc
{
  "name": "xdb.buds",
  "description": "List buds (candidate concepts) optionally filtered by status.",
  "input_schema": {
    "type": "object",
    "properties": {
      "status": {
        "type": "string",
        "enum": ["collecting", "ready", "training", "promoted", "abandoned"]
      }
    }
  }
}
```

---

### `xdb.bud_examples`

Get examples for a bud.

```jsonc
{
  "name": "xdb.bud_examples",
  "description": "Get all timesteps tagged with a bud (training examples).",
  "input_schema": {
    "type": "object",
    "properties": {
      "bud_tag_id": {
        "type": "string",
        "description": "The bud tag ID."
      }
    },
    "required": ["bud_tag_id"]
  }
}
```

---

### `xdb.bud_ready`

Mark a bud ready for training.

```jsonc
{
  "name": "xdb.bud_ready",
  "description": "Mark a bud as ready for training (transition from 'collecting' to 'ready').",
  "input_schema": {
    "type": "object",
    "properties": {
      "bud_tag_id": {
        "type": "string",
        "description": "The bud tag ID."
      }
    },
    "required": ["bud_tag_id"]
  }
}
```

---

## Fidelity Management Tools

### `xdb.pin`

Pin timesteps to WARM.

```jsonc
{
  "name": "xdb.pin",
  "description": "Pin timesteps to WARM storage for training data preservation.",
  "input_schema": {
    "type": "object",
    "properties": {
      "timestep_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Timestep IDs to pin."
      },
      "reason": {
        "type": "string",
        "description": "Reason for pinning (for audit)."
      }
    },
    "required": ["timestep_ids"]
  }
}
```

---

### `xdb.unpin`

Unpin timesteps from WARM.

```jsonc
{
  "name": "xdb.unpin",
  "description": "Unpin timesteps from WARM storage.",
  "input_schema": {
    "type": "object",
    "properties": {
      "timestep_ids": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "required": ["timestep_ids"]
  }
}
```

---

### `xdb.quota`

Get resource quota status.

```jsonc
{
  "name": "xdb.quota",
  "description": "Get WARM and COLD storage quota status.",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### `xdb.context`

Get context window state.

```jsonc
{
  "name": "xdb.context",
  "description": "Get current context window state (tokens used, utilization, compaction count).",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### `xdb.compact`

Trigger compaction.

```jsonc
{
  "name": "xdb.compact",
  "description": "Manually trigger context window compaction.",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### `xdb.maintenance`

Run storage maintenance.

```jsonc
{
  "name": "xdb.maintenance",
  "description": "Run storage maintenance (compression, cleanup).",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

---

## How the BE Uses These in Practice

**Recording experience (per turn):**

1. Workspace harness calls `xdb.record` for inputs and outputs
2. BE calls `xdb.tag` for interesting moments ("confusing", "breakthrough")
3. BE calls `xdb.comment` for reflections

**Recall and reflection:**

1. `xdb.query` with concept or tag filters to find relevant past experience
2. `xdb.recent` for immediate context
3. `xdb.graph_neighborhood` to explore concept relationships

**Bud pipeline (learning):**

1. Create bud with `xdb.create_tag` (type: "bud")
2. Tag examples with `xdb.tag`
3. Check examples with `xdb.bud_examples`
4. When ready, call `xdb.bud_ready`
5. Pin examples with `xdb.pin` to preserve for training

**Resource management:**

1. Check `xdb.quota` before pinning
2. Check `xdb.context` before long outputs
3. Call `xdb.maintenance` during idle periods
