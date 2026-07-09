# BE BED API (Tiered, Zero-Trust)

This document defines the BED (BE Diegesis) API for asynchronous messaging,
tiered tool access, and multi-source streaming in a zero-trust posture.

Key goals:
- Every request is authenticated and audited.
- Responses are filtered by tier and ASK authorization.
- BE may limit or close connections from within the workspace.

---

## 1. Common Request Envelope

All BED endpoints require an actor envelope for authorization. For GET
endpoints, the envelope is provided via headers.

### 1.1 Headers (GET)

- `X-Actor-Id`
- `X-Actor-Signature`
- `X-Actor-Nonce`
- `X-Actor-Timestamp` (ISO8601)

### 1.2 Body (POST)

```jsonc
{
  "actor": {
    "actor_id": "act_123",
    "signature": "base64sig",
    "nonce": "uuid",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "be_id": "be-001"
}
```

Every request is:
- validated against ASK authorization,
- gated by tier,
- logged to the audit log with cryptographic provenance.

---

## 2. Tier 1: Workspace Internals

Tier 1 endpoints provide internal workspace access. These are BE-facing and
subject to AWARE tier gating.

### 2.1 Scratchpad

**POST /v1/bed/workspace/scratchpad/write**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "content": "Remember to review clause 4.2",
  "section": "notes"
}
```

**POST /v1/bed/workspace/scratchpad/read**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "section": "notes"
}
```

### 2.2 Self-Steering

**POST /v1/bed/workspace/update_csh**
**POST /v1/bed/workspace/request_steering**
**POST /v1/bed/workspace/internal_state**

Payloads follow existing Hush MCP tool schemas.

### 2.3 Connection Controls (BE-side)

Allow the BE to limit or close connections to streams/tools/actors.

**POST /v1/bed/workspace/connection/limit**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "target": "stream|tool|actor",
  "id": "camera-front",
  "reason": "privacy",
  "duration_s": 600
}
```

**POST /v1/bed/workspace/connection/close**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "target": "stream|tool|actor",
  "id": "camera-front",
  "reason": "privacy"
}
```

**GET /v1/bed/workspace/connection/list/{be_id}**

---

## 3. Tier 2: Memory and Introspection

### 3.1 Introspect

**POST /v1/bed/memory/introspect**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "options": {
    "include_recent_ticks": true,
    "limit": 20,
    "include_lens_traces": true
  }
}
```

### 3.2 Recent Ticks

**POST /v1/bed/memory/ticks**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "limit": 100
}
```

### 3.3 XDB Wrappers

**POST /v1/bed/memory/xdb/query**
**POST /v1/bed/memory/xdb/record**

These mirror the XAPI operations and are filtered by LensDisclosurePolicy.

---

## 4. Tier 3: Sensory Streams (Inbound)

Tier 3 supports multiple inbound streams (e.g., camera, mic, telemetry).

### 4.1 Register Stream

**POST /v1/bed/streams/register**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "stream": {
    "source_id": "camera-front",
    "tier": 3,
    "schema": "image/jpeg",
    "direction": "inbound",
    "visibility": { "tiers": [3, 4, 5] }
  }
}
```

### 4.2 Push Stream Data

**POST /v1/bed/streams/{stream_id}/push**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "payload": {
    "encoding": "base64",
    "data": "..."
  }
}
```

### 4.3 List Streams

**GET /v1/bed/streams/{be_id}**

---

## 5. Tier 4: Actuation (Outbound)

Tier 4 exposes actuation requests emitted by the BE and allows acknowledgments.

### 5.1 Stream Events

**GET /v1/bed/events/{be_id}**

Events are filtered by tier and actor permissions.

```jsonc
{
  "event_id": "evt_456",
  "tier": 4,
  "type": "actuation_request",
  "payload": { "actuator": "arm", "command": "grip", "strength": 0.7 }
}
```

### 5.2 Acknowledge Actuation

**POST /v1/bed/actuation/ack**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "event_id": "evt_456",
  "status": "accepted|rejected|completed",
  "details": "Optional notes"
}
```

---

## 6. Tier 5: External Tools (Outbound/Ingest)

Tier 5 allows BE tool calls to be emitted and tool results to be ingested.

### 6.1 Register Tool

**POST /v1/bed/tools/register**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "tool": {
    "tool_id": "mcp:filesystem",
    "schema": { ... },
    "visibility": { "tiers": [5] }
  }
}
```

### 6.2 Tool Calls (Outbound)

**GET /v1/bed/events/{be_id}**
```jsonc
{
  "event_id": "evt_789",
  "tier": 5,
  "type": "tool_call",
  "payload": { "tool": "mcp:filesystem", "arguments": { ... } }
}
```

### 6.3 Tool Responses (Inbound)

**POST /v1/bed/tools/{tool_id}/response**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "event_id": "evt_789",
  "payload": { "status": "ok", "result": { ... } }
}
```

---

## 7. Async Messaging

### 7.1 Send Message (Tier 2+)

**POST /v1/bed/messages**
```jsonc
{
  "actor": { ... },
  "be_id": "be-001",
  "message": {
    "content": "Please review the contract clause...",
    "metadata": {
      "channel": "inbox",
      "priority": "normal",
      "reply_to": "thread-abc"
    }
  }
}
```

Response:
```jsonc
{
  "status": "accepted",
  "message_id": "msg_abc123",
  "accepted_at": "2025-01-15T10:30:01Z"
}
```

Replies and actions appear later via the event stream.

Messages are only accepted for actors authorized at Tier 2 or higher.

---

## 8. Status

**GET /v1/bed/status/{be_id}**

Returns current BE status (tier, workspace state, current tick), filtered
to the caller's permissions.

---

## 9. Streaming Notes

Event delivery is real-time and SHOULD be implemented as:
- Server-Sent Events (SSE), or
- WebSocket (WS).

This spec leaves transport choice to implementation.

---

## 10. Error Responses

All endpoints MUST return structured errors with a consistent shape.

```jsonc
{
  "error": {
    "code": "auth_failed|tier_violation|be_rejected|invalid_request|rate_limited",
    "message": "Human-readable summary",
    "details": { "tier": 3, "required_tier": 5 }
  }
}
```

Common cases:
- `auth_failed`: actor signature, nonce, or ASK authorization failed.
- `tier_violation`: caller lacks tier permission for the requested operation.
- `be_rejected`: BE-side connection controls refused the request.
- `invalid_request`: schema validation or missing required fields.
- `rate_limited`: request rejected due to policy or resource limits.

---

## 11. Auditing and Zero-Trust Guarantees

- All calls are logged to the audit log with actor identity, nonce, and hash.
- Event visibility is filtered by tier and ASK authorization.
- BE-side connection controls can block or limit inbound streams/tools/actors.
