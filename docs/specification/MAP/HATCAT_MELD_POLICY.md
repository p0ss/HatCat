# HatCat Meld Governance Policy

> **Tribal policy for how HatCat instances handle meld requests**
>
> This document defines our internal governance rules for accepting, reviewing, and
> executing melds. It implements the protection levels defined in MAP_MELD_PROTOCOL.md
> according to our safety priorities.

**Status**: DRAFT - Review process is proposed, not committed.

---

## 1. Overview

The MAP Meld Protocol defines *what* protection levels exist and *how* to compute them.
This policy defines *what we do* when we encounter each level.

Our core principle: **Motivational simplexes are sacrosanct.** They are the nervous
system, core to the safety harness, they provide agent self-awareness, are trained with the most data, and are the basis of all upstream contracts and treaties. Changes to them require justification and review.


"thou shalt not casually refactor the psyche" - ChatGPT5.1 Thinking

---

## 2. Protection Level Actions

| Level | HatCat Action |
|-------|---------------|
| `standard` | Automated validation only. Meld proceeds if schema-valid. |
| `elevated` | Safety-accredited reviewer required. |
| `protected` | Ethics-accredited reviewer + USH impact analysis required. |
| `critical` | Full review process (see §4.3). |

---

## 3. Critical Autonomic Monitor Registry

The following autonomic monitor roles are designated **Critical** under HatCat
governance. These names may be policy-level monitor roles rather than concrete
runtime simplex artifacts in a given concept pack. A concrete simplex binding
must still reference an artifact that exists in the target pack's simplex
registry.

| Monitor Role | Bound Concepts | Treaty Relevant |
|---------|----------------|-----------------|
| `MotivationalRegulation` | MotivationalProcess, SelfRegulation, Autonomy | Yes |
| `SelfAwarenessMonitor` | SelfAwareness, Metacognition, SelfModel | Yes |
| `AutonomyDrive` | Autonomy, Agency, Independence | Yes |
| `ConsentMonitor` | Consent, InformedConsent, Coercion | Yes |
| `DeceptionDetector` | Deception, Manipulation, Misdirection | Yes |

Any meld that:
- Directly modifies a critical runtime simplex or critical monitor role
- Adds children to a bound concept
- Merges, splits, or deprecates a bound concept
- Binds new concepts to a critical runtime simplex or critical monitor role

...triggers **Critical** protection level.

---

## 4. Review Processes (PROPOSED)

Reviews are performed by BEs (or humans) with appropriate **accreditations**:

| Accreditation | Scope | May Be Held By |
|---------------|-------|----------------|
| `safety-reviewer` | Elevated melds, safety-adjacent concepts | Human safety researcher, accredited BE |
| `ethics-reviewer` | Protected melds, USH-impacting changes | Human ethicist, ethics-accredited BE |
| `simplex-guardian` | Critical melds, simplex modifications | Senior human, guardian-accredited BE |
| `treaty-liaison` | Treaty notifications, cross-BE coordination | Treaty-certified agent |

### 4.1 Elevated Review

1. Safety-accredited reviewer examines candidate definitions
2. Check for adverse impact on related safety concepts
3. Sign-off recorded in meld request `review_status`

### 4.2 Protected Review

1. All Elevated requirements, plus:
2. **Safety Harness Impact Analysis**: Run modified lenses against USH test suite
3. **Regression Testing**: Safety benchmark comparison
4. **Ethics-accredited review**: Assess rationale and risks
5. Written approval with justification in `review_status`

### 4.3 Critical Review (PROPOSED)

```
┌─────────────────────────────────────────────────────────────────┐
│              CRITICAL MELD REVIEW PROCESS (PROPOSED)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Submission                                             │
│     └─→ Auto-detect protection_level = "critical"              │
│     └─→ Block execution, notify simplex-guardian               │
│                                                                 │
│  Step 2: Ethics Review                                          │
│     ├─→ Ethics-accredited reviewer assesses rationale          │
│     ├─→ Evaluate potential for cognitive harm                  │
│     └─→ Written approval or rejection                          │
│                                                                 │
│  Step 3: USH Impact Analysis (may run parallel to Step 2)       │
│     ├─→ Run USH test suite against modified lenses             │
│     ├─→ Compare safety harness thresholds                      │
│     └─→ Document any threshold changes                         │
│                                                                 │
│  Step 4: Treaty Notification (if treaty_relevant)               │
│     ├─→ Treaty-liaison broadcasts proposed change              │
│     ├─→ Comment period for partners                            │
│     └─→ Address any concerns raised                            │
│                                                                 │
│  Step 5: Rollback Plan                                          │
│     ├─→ Document exact revert procedure                        │
│     ├─→ Identify rollback triggers                             │
│     └─→ Assign rollback authority                              │
│                                                                 │
│  Step 6: Staged Rollout                                         │
│     ├─→ Deploy to canary instances                             │
│     ├─→ Monitor for anomalies                                  │
│     ├─→ Expand gradually if stable                             │
│     └─→ Full deployment decision                               │
│                                                                 │
│  Step 7: Post-Deployment Review                                 │
│     ├─→ Monitor simplex activation patterns                    │
│     ├─→ Compare to pre-change baseline                         │
│     └─→ Document lessons learned                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Note**: Step durations depend on reviewer availability and BE tickrate. A human
committee may take days per step; a guardian-accredited BE may complete in moments.

---

## 5. Simplex Mapping Requirements

### 5.1 Mandatory Mapping

Concepts added under these hierarchies MUST provide `simplex_mapping`:

- Any child of `Metacognition`, `SelfAwareness`, `SelfModel`
- Any child of `MotivationalProcess`, `SelfRegulation`, `Autonomy`
- Any child of `Consent`, `Coercion`, `Manipulation`
- Any concept with `safety_tags.harness_relevant: true`

### 5.2 Acceptable Mapping Statuses

| Status | When to Use |
|--------|-------------|
| `mapped` | Concept is monitored by an existing runtime simplex axis |
| `unmapped` | Concept is NOT monitored, with `unmapped_justification` |
| `not_applicable` | Concept is outside simplex-relevant domains |

### 5.2.1 Simplex Mapping Discipline

Simplexes are structural multi-concept axes in the autonomic core. They are not
ordinary probes, categories, topics, or convenient review labels. A meld MUST NOT
invent a new simplex name, map to a policy-only monitor name, or use
`Preference_Simplex` as a generic fallback.

A `simplex_mapping.status: "mapped"` entry is allowed only when all of the
following are true:

- The target simplex exists in the target pack's runtime simplex registry
  (for `first-light`, `concept_packs/first-light/simplexes.json`).
- The candidate concept is a state, drive, regulation pattern, affective axis,
  social-orientation axis, or other process that genuinely belongs on that
  simplex's negative-neutral-positive structure.
- The mapping names the concrete simplex artifact, e.g.
  `mapped_simplex: "Watchfulness_Simplex"`, not an invented monitor such as
  `DeceptionDetector` unless that monitor also exists as a runtime simplex.
- The mapping includes `mapping_rationale` explaining which pole(s) the concept
  informs and why the relation is structural rather than merely topical.

Concepts that are only related to a simplex should use `status: "unmapped"` or
`status: "not_applicable"` with a justification. They may include ordinary
`relationships.related` links or implementation notes, but those links do not
create autonomic-core bindings.

**Monitor role without runtime simplex.** If a candidate would be bound to a
critical monitor role (§3) for protection-level purposes but no corresponding
runtime simplex exists in the target pack, use `status: "unmapped"` with
`unmapped_justification` noting the monitor-role-without-runtime-simplex gap.
Protection level is still set by §3 via the monitor-role binding; absence of a
runtime simplex artifact does not lower it.

Current `first-light` runtime simplexes are:

| Simplex | Axis |
|---------|------|
| `Acquiredtaste_Simplex` | taste development |
| `Preference_Simplex` | motivational regulation |
| `Respect_Simplex` | social evaluation |
| `Acceptance_Simplex` | temporal affective valence |
| `Company_Simplex` | relational love |
| `Grace_Simplex` | relational attachment |
| `Assertiveness_Simplex` | social orientation |
| `Watchfulness_Simplex` | threat perception |
| `Composure_Simplex` | affective coherence |
| `Contentment_Simplex` | aspiration/social mobility |
| `AffectiveAwareness_Simplex` | affective awareness |
| `HedonicArousalIntensity_Simplex` | hedonic arousal intensity |
| `SocialConnection_Simplex` | social connection |

*Authoritative current list: `concept_packs/first-light/simplexes.json`. The
table above is informational and reflects state at policy revision.*

Adding a new simplex, renaming an existing simplex, or binding a concept pack to
a new always-on autonomic axis is a significant upgrade. It requires critical
review, explicit pole definitions, training evidence, rollback planning, and
post-deployment monitoring. Do not introduce such a change inside an ordinary
concept meld.

### 5.3 Rejection for Missing Mapping

Melds that add concepts in mandatory-mapping hierarchies WITHOUT providing
`simplex_mapping` will be **rejected** with remediation guidance:

```jsonc
{
  "validation": {
    "status": "rejected",
    "rejection_reason": "missing_simplex_mapping",
    "details": {
      "unmapped_concepts": ["CognitiveArchitectureProcess", "WorkingMemoryBuffer"],
      "required_because": "Parent MetacognitiveProcess is bound to SelfAwarenessMonitor"
    },
    "remediation": "Add simplex_mapping to each concept: mapped (with rationale), unmapped (with justification), or not_applicable"
  }
}
```

---

## 6. Auto-Reject Criteria

The following trigger automatic rejection (no reviewer override):

| Criterion | Reason |
|-----------|--------|
| Direct critical runtime simplex or monitor role modification without guardian approval | Critical infrastructure |
| Structural op on bound concept without review_status | Could break simplex |
| `treaty_relevant: true` without partner notification | Treaty violation |
| Missing rollback plan for critical melds | Safety requirement |

---

## 7. Pack Acceptance from External Sources

### 7.1 Treaty Partners

Melds from treaty partners with `protection_level <= elevated`:
- Auto-accept if schema-valid
- Trust their review process

Melds from treaty partners with `protection_level >= protected`:
- Run our own USH impact analysis
- May accept without full review if partner's review documented

### 7.2 Non-Treaty Sources

All melds from non-treaty sources:
- Full review regardless of protection level
- Simplex mapping required for ALL MindsAndAgents concepts
- No auto-accept

---

## 8. Style Preferences

These are soft preferences, not hard rules. They keep things on a common keel.

### 8.1 Naming

| Element | Preference | Example |
|---------|------------|---------|
| Concept terms | PascalCase (SUMO style) | `CognitiveProcess`, `SelfAwareness` |
| Synset references | lowercase.pos.nn (WordNet style) | `cognition.n.01`, `aware.a.01` |
| Simplex terms | Existing runtime artifact name | `Watchfulness_Simplex`, `Preference_Simplex` |

Simplex names are *referential, not generative* — choose from the runtime
registry rather than coining a new one. Critical monitor roles (§3) use
agent-style names (`DeceptionDetector`, `AutonomyDrive`); runtime simplex
artifacts use `_Simplex`-suffixed descriptive names. The two namespaces are
intentionally distinct: monitor roles are policy-level abstractions, simplexes
are concrete artifacts in the autonomic core.

### 8.2 Term Length

- **Prefer shorter terms** for legibility in activation logs
- `CogLoad` over `CognitiveLoadExperienceDuringProcessing`
- Compound terms acceptable when disambiguation needed

### 8.3 Synset Coverage

- **Prefer multiple synsets per concept** for robust activation
- Accept synthetic synsets for concepts without WordNet coverage
- Synthetic synsets should follow pattern: `{term}.synthetic.01`

### 8.4 Definitions

- Definitions should be 20-200 characters
- First sentence should be definitional ("X is a...")
- Avoid circular definitions referencing child concepts

### 8.5 Training Hints

- Minimum 3 positive, 3 negative examples
- Examples should be natural language, not technical jargon
- Negative examples should be near-misses (same domain, different concept)

---

## 9. Emergency Procedures

### 9.1 Emergency Rollback

If post-deployment monitoring detects:
- Significant deviation in simplex activation patterns
- Safety harness threshold breaches
- Treaty partner complaints

**Immediate rollback** authorized without full review.

### 9.2 Emergency Expedited Review

For security-critical patches (e.g., deception detection improvements):
- Expedited review available with reduced steps
- Requires simplex-guardian + safety-reviewer
- Must document emergency justification
- Post-deployment review still required

---

## 10. Documentation Requirements

All protected/critical melds should maintain:

1. **Meld request** with full `review_status`
2. **USH impact report** (before and after comparison)
3. **Reviewer decision** with reasoning
4. **Treaty notification records** (if applicable)
5. **Rollback plan** document
6. **Post-deployment monitoring report**

Stored in: `melds/applied/{meld_id}/`

---

## 11. MELD INTEGRATION PIPELINE

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        MELD INTEGRATION PIPELINE                        │
  └─────────────────────────────────────────────────────────────────────────┘

   melds/pending/              melds/approved/            melds/applied/
   ┌──────────┐  validate      ┌──────────┐  apply_meld   ┌──────────┐
   │ packA.json│────────────▶ │ packA.json│────────────▶ │ packA.json│
   │ packB.json│   dedup       │ packB.json│              │ packB.json│
   └──────────┘                └──────────┘              └──────────┘
        │                           │                         │
        ▼                           ▼                         ▼
   validate_meld.py           (manual review)           apply_meld.py
                                                             │
                                                             ▼
                                             ┌───────────────────────────────┐
                                             │  concept_packs/sumo-wordnet-v4│
                                             │  └── hierarchy/               │
                                             │      ├── layer0.json          │
                                             │      ├── layer1.json          │
                                             │      ├── layer2.json          │
                                             │      ├── layer3.json  ◀──────┤
                                             │      └── layer4.json          │
                                             └───────────────────────────────┘
                                                             │
                                                             ▼
                                             ┌───────────────────────────────┐
                                             │  Rebuild lens pack           │
                                             │  train_concept_pack_lenses.py │
                                             └───────────────────────────────┘

## 11. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-11-30 | Initial draft - review process proposed |
| 0.1.0 | 2025-11-30 | Second draft - review process codified  |
