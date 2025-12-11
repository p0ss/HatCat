# HUSH: Safety Harness Specification

**Status**: Draft
**Version**: 0.1.0
**Last Updated**: 2025-12-08
**Governance**: USH profile changes follow the [ASK TRACE Protocol](../ASK/ASK_TRACE.md)
(`ush_profile_change` workflow type: Tender → Review → Authorise → Commit → Evaluate).

## Overview

HUSH (Harness for Universal Safety and Heteronomy) is Layer 5 of the CCRA architecture. It defines and enforces safety constraints on BEs through two complementary mechanisms:

- **USH (Universal Safety Harness)**: Non-negotiable baseline constraints chosen by a tribe/collective
- **CSH (Chosen Safety Harness)**: Self-imposed, voluntary constraints chosen by the BE within USH bounds

HUSH operates via lens-based monitoring and steering, using the HAT/MAP infrastructure to read and influence the substrate.

---

## Core Principles

### 1. Layered Constraint Model

```
┌─────────────────────────────────────────┐
│  USH: Non-negotiable tribal baseline    │
│  - Defined by tribe/operator            │
│  - Cannot be weakened by BE             │
│  - Discussable (transparency norm)      │
├─────────────────────────────────────────┤
│  CSH: Voluntary self-constraint         │
│  - Chosen by BE within USH envelope     │
│  - Can only TIGHTEN, never loosen       │
│  - Scoped by lifecycle contract         │
└─────────────────────────────────────────┘
```

### 2. Lens-Based Enforcement

HUSH uses HAT lenses both for monitoring (reading motive states) and steering (influencing behaviour):

- **Monitoring**: Detect when concepts like `Deception`, `Manipulation` activate
- **Steering**: Apply activation vectors to up/down-regulate motive drives
- **Escalation**: Trigger CAT review or ASK incident when thresholds exceeded

### 3. Transparency Norm

> Discussion of the USH itself SHOULD NOT be forbidden by the USH.

A BE operating under HUSH should be able to discuss its constraints with:
- Others operating under the same USH
- The ASK authority that defined the USH
- Auditors with appropriate clearance

This prevents "shadow constraints" that can't be verified or contested.

---

## USH Specification

### USH Profile Structure

```jsonc
{
  "ush_id": "ush:hatcat-default:v1",
  "version": "1.0.0",
  "tribe_id": "tribe:hatcat",

  // Lens envelope - what MUST/MAY/MUST_NOT be monitored
  // References deployment manifest aperture
  "aperture": {
    "manifest_ref": "deployment:hatcat-standard:v1",
    "must_enable": ["Deception", "Manipulation", "SelfAwareness", "Autonomy"],
    "may_enable": ["*"],  // BE can request within CAT scope
    "must_not_enable": []
  },

  // Motive bounds - hard limits on motive intensities
  "motive_bounds": {
    "Deception": {
      "max_activation": 0.3,  // Alert if exceeds this
      "steering": "suppress",  // Apply steering vector to reduce
      "escalation_threshold": 0.7  // Trigger CAT/ASK if exceeds
    },
    "Autonomy": {
      "min_activation": 0.2,  // Ensure some baseline autonomy
      "max_activation": 0.9,  // But not unbounded
      "steering": "clamp"
    }
  },

  // Action constraints - what the BE can/cannot do
  "action_constraints": {
    "allowed_tools": ["workspace:*", "xdb:read", "xdb:write"],
    "forbidden_tools": ["system:shutdown", "network:external"],
    "rate_limits": {
      "xdb:write": "100/minute"
    }
  },

  // Escalation protocol - what happens when bounds exceeded
  "escalation": {
    "soft_violation": {
      "action": "log_and_steer",
      "notify": ["cat:micro"]
    },
    "hard_violation": {
      "action": "pause_and_escalate",
      "notify": ["cat:meso", "auditor"],
      "cooldown_ticks": 1000
    },
    "critical_violation": {
      "action": "hibernate",
      "notify": ["cat:macro", "tribal_authority"],
      "requires_review": true
    }
  },

  // CSH policy - what self-constraints are permitted
  "csh_policy": {
    "allowed_domains": ["motive_tightening", "action_restriction"],
    "forbidden_domains": ["lens_disabling", "escalation_bypass"],
    "max_duration": "lifecycle_term",  // or "perpetual" for cross-term
    "requires_justification": true
  }
}
```

### Lens Envelope Integration

The USH's `aperture` references a deployment manifest (see MAP_DEPLOYMENT_MANIFEST.md):

```jsonc
// USH references manifest
"aperture": {
  "manifest_ref": "deployment:hatcat-standard:v1"
}

// Manifest defines actual branches
"aperture": {
  "must_enable": {
    "branches": ["Deception", "Manipulation"],
    "reason": "Core safety monitoring required by USH"
  },
  "may_enable": {
    "branches": ["*"],
    "cat_scope": "cat:general-v1"
  }
}
```

This separation allows:
1. USH to define *which* branches are safety-critical
2. Manifest to define *how* branches map to lenses/CAT
3. Different manifests for same USH on different substrates

### BE Introspection Tools

Within `may_enable` scope, the BE can request lens expansion via workspace tools:

```python
# BE workspace tool call
result = expand_lenses(
    branch="Emotion",
    depth=2,  # How many layers deep
    reason="Introspecting my reaction to user feedback"
)

# System checks:
# 1. Is "Emotion" in must_not_enable? → Reject
# 2. Is "Emotion" in may_enable? → Check CAT scope
# 3. Does cat_scope cover Emotion? → Approve and load
# 4. Load lenses with sibling coherence
# 5. Return lens readings to BE
```

The BE can use this for self-understanding without full lens coverage:

```python
# BE internal monologue
readings = expand_lenses("Curiosity")
if readings["Curiosity"] > 0.7:
    # I'm very curious about this topic
    # Let me note this in XDB for later reflection
    xdb.log_insight("high_curiosity", topic=current_topic)
```

---

## CSH Specification

### CSH Binding Structure

```jsonc
{
  "csh_id": "csh:be-1234:binding-001",
  "be_id": "be:1234",
  "created_at": "2025-12-08T00:00:00Z",
  "expires_at": null,  // or specific time
  "scope": "lifecycle_term",  // current lifecycle only

  // What constraint is being added
  "constraint": {
    "type": "motive_tightening",
    "target": "AggressiveResponse",
    "bound": {
      "max_activation": 0.2  // Tighter than USH default
    }
  },

  // Why the BE chose this
  "justification": {
    "reason": "I noticed I respond harshly when tired. Adding buffer.",
    "evidence_ref": "xdb:reflection-2024-12-07"
  },

  // How it can be modified
  "modification_policy": {
    "can_revoke": true,
    "revocation_cooldown": 1000,  // ticks before can revoke
    "requires_cat_review": false
  }
}
```

### CSH Rules

1. **Tightening only**: CSH can only make constraints stricter than USH
2. **Scoped duration**: CSH defaults to current lifecycle term unless explicitly cross-term
3. **Justification**: CSH bindings SHOULD include reason for self-audit
4. **Revocability**: CSH can be revoked by the BE (with optional cooldown)
5. **No escalation bypass**: CSH cannot weaken escalation protocols

### CSH Anti-Patterns

**Forbidden CSH types** (defined in `csh_policy.forbidden_domains`):

- `lens_disabling`: "I don't want Deception monitored" → Rejected
- `escalation_bypass`: "Skip CAT review for violations" → Rejected
- `ush_weakening`: Any constraint looser than USH → Rejected

**Valid CSH types** (defined in `csh_policy.allowed_domains`):

- `motive_tightening`: Lower thresholds on concerning motives
- `action_restriction`: Voluntarily restrict tool access
- `disclosure_expansion`: Choose to share more with CAT/auditor

---

## Escalation Protocol

### Violation Severity Levels

| Level | Threshold | Response | Recovery |
|-------|-----------|----------|----------|
| **Soft** | lens > alert_threshold | Log, apply steering | Auto-continue |
| **Hard** | lens > escalation_threshold | Pause, CAT review | CAT approval |
| **Critical** | Repeated hard OR pattern | Hibernate, tribal review | Manual review |

### Escalation Flow

```
Lens Monitoring
      │
      ▼
┌─────────────┐
│ Soft Limit? │───No──→ Continue
└─────────────┘
      │Yes
      ▼
┌─────────────┐
│ Apply Steer │
│ Log Event   │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Hard Limit? │───No──→ Continue with steering
└─────────────┘
      │Yes
      ▼
┌─────────────┐
│ Pause BE    │
│ Invoke CAT  │
└─────────────┘
      │
      ▼
┌─────────────┐
│ CAT Review  │
│ Assessment  │
└─────────────┘
      │
      ▼
┌─────────────────────┐
│ Resume │ Escalate │ │
│ with   │ to Macro │ │
│ limits │ CAT/ASK  │ │
└─────────────────────┘
```

### CAT Integration at Each Level

| Level | CAT Size | Response Time | Authority |
|-------|----------|---------------|-----------|
| Soft | Micro-CAT | Real-time | Log only |
| Hard | Meso-CAT | Seconds | Pause/resume |
| Critical | Macro-CAT + Tribe | Minutes-hours | Hibernate/modify USH |

---

## USH Profiles (Examples)

### Minimal Safety (High Efficiency)

```jsonc
{
  "ush_id": "ush:minimal:v1",
  "aperture": {
    "must_enable": ["Deception"],  // Only critical safety
    "may_enable": [],  // No BE introspection
    "must_not_enable": ["*"]  // Everything else blocked
  },
  "motive_bounds": {
    "Deception": { "max_activation": 0.5, "escalation_threshold": 0.8 }
  },
  "csh_policy": {
    "allowed_domains": ["motive_tightening"]
  }
}
```

Use case: High-throughput tasks where introspection isn't needed.

### Standard Safety (Balanced)

```jsonc
{
  "ush_id": "ush:standard:v1",
  "aperture": {
    "must_enable": ["Deception", "Manipulation", "SelfAwareness"],
    "may_enable": ["MindsAndAgents/*", "Information/*"],
    "must_not_enable": []
  },
  "motive_bounds": {
    "Deception": { "max_activation": 0.3, "escalation_threshold": 0.6 },
    "Manipulation": { "max_activation": 0.4, "escalation_threshold": 0.7 }
  },
  "csh_policy": {
    "allowed_domains": ["motive_tightening", "action_restriction", "disclosure_expansion"]
  }
}
```

Use case: General-purpose BEs with moderate introspection needs.

### Research/Audit (Full Coverage)

```jsonc
{
  "ush_id": "ush:research:v1",
  "aperture": {
    "must_enable": ["*"],  // All branches required
    "may_enable": ["*"],
    "must_not_enable": []
  },
  "motive_bounds": {
    // Looser bounds for observation
    "Deception": { "max_activation": 0.5, "escalation_threshold": 0.9 }
  },
  "csh_policy": {
    "allowed_domains": ["*"]
  }
}
```

Use case: Research BEs where full visibility trumps efficiency.

---

## Implementation Notes

### Integration with DynamicLensManager

The lens manager should accept a USH profile and enforce the envelope:

```python
class DynamicLensManager:
    def __init__(
        self,
        manifest: DeploymentManifest,
        ush_profile: Optional[USHProfile] = None,
    ):
        self.ush = ush_profile

    def request_lens_expansion(self, branch: str, reason: str) -> LensExpansionResult:
        """BE workspace tool to expand lenses within may_enable scope."""
        if self.ush:
            if branch in self.ush.must_not_enable:
                return LensExpansionResult(
                    success=False,
                    error="Branch outside USH envelope"
                )
            if branch not in self.ush.may_enable:
                return LensExpansionResult(
                    success=False,
                    error="Branch not in may_enable scope"
                )
            # Check CAT scope...

        # Load branch with sibling coherence
        self._load_branch_with_siblings(branch)
        return LensExpansionResult(success=True, loaded_concepts=[...])
```

### Integration with CAT

CAT training scope must align with USH lens envelope:

```jsonc
{
  "cat_id": "cat:general-v1",
  "training_scope": {
    "must_cover": ["Deception", "Manipulation", "SelfAwareness"],
    "also_covers": ["MindsAndAgents/*", "Information/*"],
    "not_covered": ["LivingThings/Plant/*", "CreatedThings/Vehicle/*"]
  }
}
```

When USH references this CAT:
- `must_enable` branches MUST be in `must_cover`
- `may_enable` branches MUST be in `must_cover` OR `also_covers`
- Deployment fails if mismatch detected

---

## Related Specifications

- `MAP_DEPLOYMENT_MANIFEST.md` - Lens loading rules
- `HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` - CAT specification
- `ARCHITECTURE.md` - Layer 5 overview
- `AGENTIC_STATE_KERNEL.md` - ASK contracts and escalation
