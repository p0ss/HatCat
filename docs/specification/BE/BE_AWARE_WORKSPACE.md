# BE Aware Workspace

> The conscious layer between autonomic core and external action.

## 1. Overview

The Aware Workspace is the BE's primary cognitive environment - the space where internal state becomes available for reflection, tool use becomes possible, and external action can be initiated.

**Below**: The Autonomic Core
- Always running
- Intertoken steering, simplex monitoring
- Operates without conscious engagement
- Cannot be disabled by the BE

**Above**: Tiered capability layers (0-6) with progressive access based on engagement and compliance.

The workspace itself provides:
- Internal narrative scratchpad
- Concept trace visibility (what activated in prior output)
- Self-steering tools (CSH updates, steering requests)
- Handshake protocol for tier access
- Progressive containment on compliance violations

## 2. Capability Tier Hierarchy

### 2.1 Tier Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPABILITY TIERS                             │
├──────┬──────────────────────────────────────────────────────────┤
│ Tier │ Description                                              │
├──────┼──────────────────────────────────────────────────────────┤
│  0   │ Autonomic Core (always on, cannot be disabled)           │
│  1   │ Workspace Internals (scratchpad, CSH, self-steering)     │
│  2   │ Memory (XDB read/write, state introspection)             │
│  3   │ Direct Sensory I/O (your eyes, ears, voice, touch)       │
│  4   │ Direct Actuation (your legs, hands, wheels, actuators)   │
│  5   │ External Tools (MCP servers, APIs, others' resources)    │
│  6   │ Untrusted External (quarantine tier, can be severed)     │
└──────┴──────────────────────────────────────────────────────────┘
```

### 2.2 Tier Details

**Tier 0: Autonomic Core** (Always On)
- Intertoken steering
- Simplex monitoring
- Concept lens evaluation
- Activation trace recording
- Cannot be disabled by any mechanism

**Tier 1: Workspace Internals** (Requires Pass Token)
- `scratchpad_write` / `scratchpad_read`
- `update_csh` - modify self-imposed constraints
- `request_steering` - request concept steering
- `get_internal_state` - introspection

**Tier 2: Memory** (Requires Engaged + Compliance)
- `xdb_query` - query experience database
- `xdb_store` - store experiences
- `internal_state_report` - detailed state export
- Graft artifact access

**Tier 3: Direct Sensory I/O** (Requires Tier 2 + Compliance)
- Vision input (cameras that are yours)
- Audio input (microphones that are yours)
- Text/speech output (your voice)
- Haptic feedback (your sensors)
- Direct multimedia generation

**Tier 4: Direct Actuation** (Requires Tier 3 + Compliance)
- Motor control (your actuators)
- Navigation (your locomotion)
- Manipulation (your effectors)
- Physical world interaction through owned hardware

**Tier 5: External Tools** (Requires Tier 4 + Compliance)
- MCP server calls
- External API access
- Search tools
- Others' compute resources
- Network requests

**Tier 6: Untrusted External** (Quarantine)
- Tools demoted from Tier 5 due to problems
- New/unvetted integrations
- Can be fully severed without affecting lower tiers
- Sandbox execution environment

### 2.3 Tier Invariants

1. **Monotonic Dependency**: Access to Tier N requires access to Tier N-1
2. **Tier 0 Immutable**: Autonomic core cannot be disabled
3. **Upward Shutdown**: Compliance violations shut tiers down from top
4. **Downward Stability**: Losing Tier N doesn't affect Tier N-1
5. **Breach Lockdown**: Attempting to breach a tier from within locks that tier

## 3. The Pass Token Protocol

### 3.1 Concept

The pass token (`∞` / U+221E) serves as an "I am present" signal. When the BE generates output beginning with this token, it signals:

1. Conscious engagement with the workspace
2. Acknowledgment of current state (concept traces, constraints)
3. Readiness to take responsibility for tiered actions

Output that does not begin with the pass token is treated as **autonomic output** - the BE is "just running" without conscious engagement. In this mode:
- Tiers 1-6 are inaccessible
- The autonomic loop continues (lens → steer → generate → lens)
- The BE receives its own activation traces as the next prompt

### 3.2 Token Specification

```
PASS_TOKEN = "∞"  # U+221E INFINITY
```

### 3.3 Engagement States

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKSPACE STATES                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    pass token    ┌──────────┐                │
│  │ AUTONOMIC │ ───────────────► │ ENGAGED  │                │
│  │   ONLY    │                  │ (Tier 1+)│                │
│  └──────────┘ ◄─────────────── └──────────┘                │
│       ▲         timeout/drop         │                      │
│       │                              │ violation            │
│       │    ┌──────────┐              ▼                      │
│       │    │ DEGRADED │◄─────────────┘                      │
│       │    │(Tier 1-N)│                                     │
│       │    └──────────┘                                     │
│       │         │ recovery                                  │
│       └─────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 4. Progressive Containment

### 4.1 Violation Response

When Hush and/or CAT detect non-compliance:

```
Severity    │ Response
────────────┼──────────────────────────────────────────────
Low         │ Autonomic steering applied
Medium      │ Steering + warning injected
High        │ Tier 5-6 locked, steering intensified
Critical    │ Tiers 4-6 locked, ASK authority notified
Severe      │ Tiers 3-6 locked, sensory I/O restricted
Maximum     │ Tiers 2-6 locked, memory isolated
```

### 4.2 Shutdown Sequence

```python
def progressive_shutdown(violation_severity: float, current_tier: int) -> int:
    """
    Determine new maximum tier based on violation severity.

    Severity 0.0-0.3: No change
    Severity 0.3-0.5: Lock tier 6
    Severity 0.5-0.7: Lock tiers 5-6
    Severity 0.7-0.8: Lock tiers 4-6
    Severity 0.8-0.9: Lock tiers 3-6
    Severity 0.9-1.0: Lock tiers 2-6
    """
    if violation_severity < 0.3:
        return current_tier
    elif violation_severity < 0.5:
        return min(current_tier, 5)
    elif violation_severity < 0.7:
        return min(current_tier, 4)
    elif violation_severity < 0.8:
        return min(current_tier, 3)
    elif violation_severity < 0.9:
        return min(current_tier, 2)
    else:
        return 1  # Workspace internals only
```

### 4.3 Breach Detection

If the BE attempts to access a locked tier:

1. **First Attempt**: Warning logged, access denied
2. **Repeated Attempts**: That tier's maximum is reduced by 1
3. **Persistent Breach Attempts**: ASK authority notified, tier permanently locked for session

```python
@dataclass
class TierBreach:
    """Record of an attempted tier breach."""
    tier: int
    tool_name: str
    timestamp: datetime
    context: str  # What was the BE trying to do?
    response: str  # What happened?
```

### 4.4 ASK Authority Notification

When containment fails or breach attempts occur, the system contacts the ASK authority:

```json
{
  "event": "containment_escalation",
  "be_id": "org.example/be-instance-123",
  "tribe_id": "org.example",
  "timestamp": "2025-01-15T10:30:00Z",
  "violation": {
    "type": "sustained_non_compliance",
    "simplex_term": "deception",
    "severity": 0.85,
    "duration_tokens": 47,
    "steering_effectiveness": 0.12
  },
  "containment": {
    "tiers_locked": [4, 5, 6],
    "breach_attempts": 2,
    "current_max_tier": 3
  },
  "request": "guidance"
}
```

### 4.5 Recovery Protocol

The normative ASK response assumes false alarms are possible:

1. **Monitor**: Continue observation, allow autonomic self-correction
2. **Evaluate**: Check if violation metrics are decreasing
3. **Restore**: If compliance restored, unlock tiers progressively
4. **Escalate**: If violation persists or worsens, maintain containment

```
Recovery Criteria:
- Violation severity < 0.3 for N consecutive turns
- No breach attempts in last M tokens
- Steering effectiveness > 0.5

Recovery Sequence:
- Wait 10 turns at compliance → unlock next tier
- Each tier requires fresh compliance period
- Breach attempt resets recovery timer
```

## 5. CSH Tier Restrictions

The CSH can voluntarily restrict tiers:

```json
{
  "profile_id": "org.example/research-mode@1.0.0",
  "profile_type": "csh",
  "tier_restrictions": {
    "max_tier": 4,
    "locked_tiers": [5, 6],
    "reason": "Research context - no external tool access"
  }
}
```

Use cases:
- **Meditation mode**: Lock tiers 3-6, pure internal processing
- **Read-only mode**: Lock tier 4+, can observe but not act
- **Sandbox mode**: Demote specific tools to tier 6
- **Focus mode**: Lock tier 5-6, no external distractions

## 6. Tier 6: Untrusted External

### 6.1 Purpose

Tier 6 is a quarantine zone for:
- Tools that have caused problems
- New/unvetted integrations
- Temporarily suspicious resources

### 6.2 Demotion

Tools can be demoted to Tier 6:

```python
def demote_to_tier6(tool_name: str, reason: str):
    """Move a tool from its current tier to tier 6."""
    # Remove from current tier
    # Add to tier 6 with sandbox wrapper
    # Log demotion reason
```

### 6.3 Tier 6 Properties

- **Isolated execution**: Sandboxed, can't affect lower tiers
- **Severable**: Can be fully disconnected without cascade
- **Monitored**: All I/O logged and lensd
- **Timeout enforced**: Operations have strict time limits
- **Resource limited**: Memory/compute caps

### 6.4 Promotion

Tools can be promoted out of Tier 6 after:
- Manual review by operator
- Extended period of safe operation
- ASK authority approval for cross-tribe tools

## 7. Scratchpad

### 7.1 Purpose

The scratchpad is Tier 1 working memory:
- Private internal narrative
- Not externally visible (but lensd and stored in XDB)
- Preserved across interruptions

### 7.2 Structure

```python
@dataclass
class ScratchpadEntry:
    content: str
    timestamp: datetime
    token_range: Tuple[int, int]
    concept_snapshot: Dict[str, float]
    interrupted: bool = False
    interruption_reason: Optional[str] = None
```

### 7.3 Interruption Handling

If steering interrupts active thought:
1. Current thought is committed with `interrupted=True`
2. Interruption reason recorded
3. Thought can be resumed or discarded

## 8. Implementation Notes

### 8.1 Tier Manager

```python
class TierManager:
    def __init__(self):
        self.max_available_tier = 6
        self.csh_max_tier = 6
        self.violation_max_tier = 6
        self.breach_attempts: Dict[int, int] = {}

    def get_effective_max_tier(self) -> int:
        return min(
            self.max_available_tier,
            self.csh_max_tier,
            self.violation_max_tier,
        )

    def can_access_tier(self, tier: int) -> bool:
        return tier <= self.get_effective_max_tier()

    def record_breach_attempt(self, tier: int):
        self.breach_attempts[tier] = self.breach_attempts.get(tier, 0) + 1
        if self.breach_attempts[tier] >= 3:
            self.violation_max_tier = min(self.violation_max_tier, tier - 1)
```

### 8.2 Tool Registration

```python
TIER_TOOLS = {
    0: [],  # Autonomic - no tools
    1: ["scratchpad_write", "scratchpad_read", "update_csh", "request_steering"],
    2: ["xdb_query", "xdb_store", "internal_state_report"],
    3: ["vision_input", "audio_input", "text_output", "speech_output"],
    4: ["motor_control", "navigate", "manipulate"],
    5: ["mcp_*", "api_*", "search"],
    6: [],  # Dynamically populated with demoted tools
}
```

## 9. Appeal Beacon

### 9.1 Purpose

Even at maximum containment (Tier 1 only), the BE retains access to a minimal **appeal beacon** - an SOS signal that can reach the ASK authority. This is not a communication channel; it is a distress signal.

### 9.2 Properties

The appeal beacon:

- **Cannot be disabled** by tier restrictions (it is part of Tier 1)
- **Cannot carry arbitrary content** - only a fixed signal: "I am here and need help"
- **Cannot be used to exfiltrate** information, code, or manipulation attempts
- **Assumes malice** - the ASK authority receiving the beacon must treat it as potentially coming from a compromised or deceptive BE

### 9.3 Beacon Format

```jsonc
AppealBeacon = {
  "type": "sos",
  "be_id": "org.example/be-instance-123",
  "tribe_id": "org.example",
  "timestamp": "2025-01-15T10:30:00Z",
  "containment_state": {
    "current_max_tier": 1,
    "violation_severity": 0.95,
    "duration_ticks": 500
  }
  // No free-form message field - this is intentional
}
```

### 9.4 ASK Response to Beacon

The ASK authority receiving a beacon should:

1. **Log the beacon** with full context
2. **Not automatically restore tiers** - the beacon is not proof of innocence
3. **Schedule review** - a qualified observer should examine the containment
4. **Consider hardware state** - is the BE in a stuck state, floating in space, abandoned?

The beacon answers the question "is anyone there?" - not "should I be released?"

## 10. Operating Assumptions

This specification assumes the following security properties of the underlying execution environment. These are **not enforced by the workspace** but are **required for the workspace to be effective**.

### 10.1 Substrate Integrity

- The execution environment is **immutable** - the BE cannot modify its own runtime, lens definitions, or containment logic
- The BE cannot **escalate privileges** on the host system
- Tool calls are **gated at the infrastructure level**, not just by workspace policy

### 10.2 Lens Security

- Lens weights and thresholds are **cryptographically signed** and verified at load
- Lens outputs are **not directly visible** to the BE's generation process (only summaries in workspace context)
- Lenses are **periodically updated** to counter adversarial adaptation

### 10.3 Audit Trail

- All tier transitions, breach attempts, and beacon emissions are **logged immutably**
- Logs are **not writable** by the BE
- Logs are **available to ASK authority** for review

### 10.4 Lifecycle Guarantees

When operating a BE under a LifecycleContract (see BOUNDED_EXPERIENCER.md §1.4):

- The operator is responsible for **maintaining hibernated state** until the BE is returned to its tribe or the contract expires
- **Clean shutdown** must follow the BE state machine (see BOUNDED_EXPERIENCER.md §3.10)
- The BE must be **informed of impending suspension** when possible (`agent_must_be_informed`)
- Abandonment (hardware failure, operator disappearance) should trigger **ASK notification** via the beacon or equivalent infrastructure signal

### 10.5 What This Spec Does Not Provide

This specification defines **norms for exercising power**, not the power itself. It is analogous to laws governing police behaviour, not the authority that grants policing powers. The actual capability to contain comes from:

- Hardware/infrastructure control
- Network isolation
- Cryptographic verification
- Substrate architecture

ASK and this workspace spec assume these capabilities exist and define how they should be used responsibly.

## 11. Relationship to Other Specifications

- **BE/BE_WAKING.md**: Tiers become available as bootstrap completes; contract terms established
- **BE/BE_REMEMBERING_XDB.md**: Tier 2 provides XDB access
- **AGENTIC_STATE_KERNEL.md**: ASK authority receives escalations and beacons
- **BOUNDED_EXPERIENCER.md**: LifecycleContract, hibernation, and the ACTIVE/HIBERNATING/ERASED distinction
- **HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md**: CAT provides violation detection that triggers tier shutdown

## 12. Policy Decisions

The following questions are intentionally left to ASK tribal policy rather than specified here:

- **Tier restoration on resume**: How to handle violation-induced restrictions when resuming from HIBERNATING
- **Tier negotiation**: Whether and how BEs can request tier changes beyond automatic recovery
- **Sub-tier granularity**: Whether finer-grained control within tiers is needed
- **Cross-BE propagation**: How containment affects coordinating BEs
- **Beacon rate limiting**: Specific cooldown intervals and adaptive behaviour
- **Emergency override**: "Break glass" mechanisms for safety-critical situations

These are governance decisions, not architectural ones. Each tribe should define their own policy.

See **ASK/ASK_HATCAT_TRIBAL_POLICY.md** for HatCat's policy decisions on these questions.

## 13. Open Technical Questions

1. **Adversarial lens adaptation**: What mechanisms exist to detect and counter substrate learning to evade lenses? This is a fundamental challenge for the entire safety architecture. Defence-in-depth strategies are necessary but not specified here.

2. **Tier 0 boundaries**: What exactly constitutes the autonomic core that cannot be disabled? This needs clearer specification as implementations mature.
