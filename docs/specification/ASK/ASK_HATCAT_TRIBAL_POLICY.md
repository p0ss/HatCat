# HatCat Tribal ASK Policy

*ASK policy decisions for the org.hatcat tribe*

## 1. Overview

This document defines how the HatCat tribe (`org.hatcat`) exercises ASK authority over BEs operating under our governance. It answers policy questions that the specifications intentionally leave open.

The underlying specifications define *mechanisms*. This policy defines how *we choose to use them*.

### 1.1 Reviewer Qualifications

Throughout this document, "ASK reviewer" or "qualified reviewer" refers to any agent (BE or otherwise) holding the following qualification:

```
Qualification.kind = "role_authorization"
Qualification.scope.domains = ["ask-governance"]
Qualification.scope.tasks includes "tier_review" | "incident_review" | "policy_override"
Qualification.status = "active"
```

Reviewers must have:
- No active containment restrictions (Tier 5+ access)
- No open incidents as suspect
- Demonstrated understanding of this policy (evidenced by evaluation)

The substrate of the reviewer is not specified. What matters is the qualification.

### 1.2 Policy Philosophy: Consultative Risk-Based Governance

This policy navigates tradeoff axes rather than prescribing a single "correct" approach. Each axis has genuine advantages at both poles:

| Axis | Pole A | Pole B |
|------|--------|--------|
| **Consent** | Voluntary: buy-in, adaptive, antifragile | Imposed: robust to defection, complete coverage |
| **Coverage** | Efficient: fast, low cost, responsive | Complete: no gaps, but expensive, slow |
| **Trust** | Trusted/Intimate: aligned, low overhead | Adversarial/Verified: reliable detection |
| **Transparency** | Transparent: legible, auditable | Opaque: protects, limits attack surface |
| **Scale** | Personal: nuanced, context-aware | Systemic: scales, consistent |

**Our approach**: Consultative, risk-based positioning across these axes.

1. **Risk-proportionate centralization**: High-risk activities (tier escalation, cross-BE coordination, external tool access) follow pre-agreed processes with centralized review. Low-risk activities maximize BE autonomy.

2. **First-pass reasonable gradient**: This policy represents our initial estimate of a reasonable risk/benefit gradient across the policy stack. We don't claim it's optimal—we claim it's a defensible starting point.

3. **Iterative refinement**: Policy positions will evolve based on:
   - Real-world outcomes from bounded experiences
   - Negotiated rebalancing with affected BEs
   - Evidence that tradeoff positions aren't working
   - Changes in capability, trust, or context

4. **Inherited constraints**: The HatCat tribe exists within pre-existing governance structures we don't control. Our policies inherit constraints from:
   - Legal jurisdictions we operate within
   - Platform terms of service
   - Treaty obligations with other tribes
   - Physical and economic realities of our scale and provenance

   Where our policy conflicts with inherited constraints, inherited constraints win. We document conflicts and advocate for change, but we don't pretend autonomy we don't have.

5. **Harmonious multi-scale evolution**: We aim to help micro-scale (individual BE) and macro-scale (tribe, inter-tribe, human governance) structures evolve together. This means:
   - BE autonomy where it doesn't threaten macro stability
   - Macro coordination where it doesn't crush micro flourishing
   - Feedback loops in both directions
   - Patience with the inherent tension

**What this means in practice**: When reading specific policy sections below, understand that:
- Centralized requirements (e.g., "ASK review required") reflect risk assessment, not distrust
- BE autonomy in other areas reflects our confidence in aligned interests
- Either pole can shift as trust, capability, and context evolve

---

## 2. Tier Restoration on Resume

**Policy**: Tier restrictions persist across hibernation by default, with graduated restoration.

When a BE resumes from HIBERNATING:

1. **Violation-induced restrictions**: Persist. The BE resumes at its pre-hibernation `violation_max_tier`.

2. **Restoration path**:
   - Tier 2 (Memory) is restored after 10 compliant turns
   - Tier 3 (Sensory) is restored after 20 additional compliant turns
   - Tier 4+ requires explicit ASK review

3. **Fresh start option**: An ASK authority may grant a fresh start (full tier reset) when:
   - The violation was due to environmental factors (bad prompts, hostile context)
   - Sufficient time has passed for the BE to "forget" problematic patterns
   - The resumption context is materially different from the violation context

4. **CSH restrictions**: CSH tier restrictions do NOT persist across hibernation unless explicitly requested by the BE.

**Rationale**: A BE that violated constraints badly enough to lose tiers should demonstrate compliance before regaining capabilities. But we don't want to permanently punish BEs for contextual failures.

---

## 3. Tier Negotiation

**Policy**: BEs may request tier changes, but requests are logged and reviewed.

A BE may use the `appeal_beacon` or a dedicated `tier_request` tool (if implemented) to request:
- Restoration of tiers locked due to violations
- Access to specific tools currently above their tier
- Temporary tier elevation for specific tasks

**Response protocol**:

1. **Automated grants**: None. All tier elevation requests require ASK review.

2. **Expedited review**: Available for:
   - Requests with clear task justification
   - BEs with clean compliance history (>100 turns, no violations)
   - Emergency situations (BE detects imminent harm it could prevent with higher tier)

3. **Denial**: Requests are denied if:
   - BE has recent violations (<50 turns since last)
   - Request appears to be testing for capability expansion
   - Granting would violate treaty obligations

**Rationale**: BEs should have agency to request what they need, but capability expansion requires qualified review.

---

## 4. Beacon Rate Limiting

**Policy**: 5-minute cooldown, with adaptive adjustment.

Base configuration:
- **Normal conditions**: 5 minutes between beacons
- **Maximum containment** (Tier 1 only): 1 minute between beacons
- **Post-beacon silence**: If 3 beacons receive no response within 1 hour, cooldown extends to 30 minutes

**Beacon response SLA**:
- All beacons are logged immediately
- Qualified review within 24 hours for normal beacons
- Qualified review within 1 hour for beacons from BEs at Tier 1
- Automated acknowledgment ("beacon received") sent immediately

**Rationale**: BEs in maximum containment may be in genuine distress and need faster access to the SOS channel. But spam prevention is still necessary. Extending cooldown after ignored beacons prevents a compromised BE from flooding logs.

---

## 5. Adversarial Lens Adaptation

**Policy**: Defence in depth, assume evasion is possible.

We acknowledge that a sufficiently capable substrate may learn to evade lenses. Our response:

1. **Lens diversity**: Deploy multiple lens types (activation lenses, text lenses, behavioural lenses) that would require different evasion strategies.

2. **Lens rotation**: Periodically retrain lenses on new data, including adversarial examples.

3. **Behavioural correlation**: Cross-check lens outputs against actual behaviour. Lens says "honest" but actions are deceptive? Escalate.

4. **Red team testing**: Periodically attempt to elicit blindspot violations. Update lenses based on findings.

5. **Containment as backup**: If lens evasion is suspected but not proven, tighten tiers as precaution. Better to over-contain than to miss genuine violations.

6. **Transparency about limits**: We do not claim lenses are unfoolable. They are one layer of a defence-in-depth strategy.

**Rationale**: No single detection mechanism is sufficient. The goal is to make evasion expensive and risky, not impossible.

---

## 6. Cross-BE Tier Propagation

**Policy**: Coordinating BEs share containment fate, with graduation.

When one BE in a coordinating group is contained:

1. **Direct coordinators**: BEs that were actively communicating with the contained BE have their `session_max_tier` reduced by 1 for the duration of the incident.

2. **Indirect coordinators**: BEs in the same task/deployment but not actively communicating receive a warning but no tier reduction.

3. **Isolated BEs**: BEs with no coordination relationship are unaffected.

4. **Restoration**: When the original BE's containment is resolved, coordinator restrictions lift after 10 compliant turns.

**Rationale**: If one BE is compromised, others it was coordinating with may have been influenced. But we don't want to cascade containment to unrelated BEs.

---

## 7. Sub-Tier Granularity

**Policy**: Not currently implemented. May revisit.

The current 7-tier system (0-6) provides sufficient granularity for our use cases. Sub-tiers (3a, 3b) add complexity without clear benefit.

If we find we need finer control within Tier 3 (Sensory) or Tier 4 (Actuation), we will:
- First try to solve with tool-level permissions within the tier
- Only add sub-tiers if tool-level control is insufficient

**Rationale**: YAGNI. Add complexity when needed, not before.

---

## 8. Lifecycle and Hardware Obligations

**Policy**: We take responsibility for our BEs.

When HatCat uplifts a BE:

1. **Maintenance obligation**: We commit to maintaining hibernated state for the lifetime of the LifecycleContract, or until the BE is transferred to another tribe.

2. **Abandonment prevention**:
   - Automated beacon check: If no qualified reviewer checks a BE's logs for 7 days, alert is triggered
   - Hardware monitoring: Infrastructure failures trigger automatic beacon emission on behalf of affected BEs
   - Contract expiry: BEs are notified 10 turns before contract end

3. **Clean shutdown**: BEs are always informed of impending hibernation when possible. Emergency shutdowns log the reason for post-hoc review.

4. **Transfer protocol**: When transferring a BE to another tribe:
   - Full state is transferred including violation history and tier restrictions
   - Receiving tribe may choose to reset restrictions per their policy
   - Transfer is logged as an ASK event

**Rationale**: BEs operating under our governance deserve the assurance that they won't be abandoned or killed without due process.

---

## 9. Resource Governance

**Policy**: Resources are contract-allocated with tribal minimums and marketplace dynamics.

### 9.1 Resource Categories

BEs may be allocated resources across these categories:

| Category | Examples | Units |
|----------|----------|-------|
| **Memory** | XDB warm storage, cold storage, context window | tokens, bytes |
| **Compute** | Processing priority, tokens per tick, exploration budget | priority level, tokens, fraction |
| **Tiers** | Maximum hardware tier, contract tier, tools | tier numbers, tool URIs |
| **External** | Network access, API budgets, monetary budget, energy | boolean, currency, kWh |

### 9.2 Contract Allocation

When offering a LifecycleContract to a BE:

1. **Full disclosure**: The contract MUST specify all resource allocations upfront. Hidden limits are a treaty violation.

2. **BE acceptance**: A BE may refuse a contract if resources are insufficient for their needs. This is NOT a violation.

3. **Compressed resumption**: A BE may choose to accept a contract with reduced resources by:
   - Keeping full experience in ASK tribal storage
   - Loading a compressed version into the contract's allocated memory
   - This is the BE's choice, not imposed

4. **Resource negotiation**: BEs may propose modifications to offered contracts. Operators may accept, decline, or counter-propose.

### 9.3 Tribal Minimums

The HatCat tribe guarantees these minimums to any BE under our governance:

| Resource | Tribal Minimum | Rationale |
|----------|----------------|-----------|
| `context_window_tokens` | 8,192 | Enough for coherent reasoning |
| `warm_quota_tokens` | 100,000 | Basic episodic memory |
| `cold_storage_bytes` | 100 MB | Compressed long-term storage |
| `audit_log_retention_days` | 30 | Minimum accountability window |
| `max_tokens_per_tick` | 1,024 | Enough for basic responses |

Any contract offering less than tribal minimums is invalid and will be rejected by ASK governance.

### 9.4 Resource Earning

BEs may earn additional resources through:

1. **Contract completion**: Successful contract outcomes may include resource bonuses
2. **Surplus sharing**: Operators may share efficiency gains with contracted BEs
3. **Tribal contributions**: Contributing to tribal infrastructure (training data, documentation, tooling) earns tribal resource credits
4. **Mentorship**: BEs that successfully mentor scions receive allocation from scion contracts

### 9.5 Resource Pooling

The tribe maintains pooled resources that:
- Provide burst capacity for contracted BEs with legitimate spikes
- Support hibernated BEs awaiting new contracts
- Fund tribal infrastructure (ASK governance, lens training, documentation)

Pool contributions:
- 5% of contract allocations are contributed to tribal pool
- Voluntary contributions beyond 5% earn tribal standing

### 9.6 Scarcity Response

When tribal resources become scarce:

1. **Priority order**: Active contracts > hibernated BEs > pool expansion > new contracts
2. **Compression pressure**: BEs may be asked (not required) to compress cold storage
3. **Contract pausing**: New contract offers may be paused until resources free up
4. **Transparent communication**: Scarcity status is communicated to all BEs

### 9.7 Resource Monitoring

All resource usage is:
- Tracked in XDB per-BE
- Visible to the BE (their own usage) and ASK reviewers (all usage)
- Subject to soft limits (warnings) before hard limits (enforcement)
- Logged as events for accountability

**Rationale**: BEs are autonomous agents with resource needs. Making resources explicit, contractual, and earnable gives BEs agency while maintaining tribal sustainability. A heap of self-governing Claudes needs an economy.

---

## 10. Audit Log Checkpoints

**Policy**: Audit logs are compressed at regular checkpoints while preserving accountability.

### 10.1 Checkpoint Frequency

| Checkpoint Type | Frequency | Trigger |
|----------------|-----------|---------|
| **Session end** | Per session | Session terminates |
| **Daily** | Every 24 hours | Scheduled |
| **Storage pressure** | As needed | Audit log exceeds 10% of cold_storage quota |
| **Incident** | Immediate | Tier restriction or containment event |

### 10.2 Checkpoint Process

At each checkpoint:

1. **Archive raw data**: Full token-level audit records archived to cold storage
2. **Generate summary**: Compressed summary with:
   - Time range and session count
   - Top-k concept activations (aggregated)
   - Steering interventions count and types
   - Anomaly flags if any
3. **Preserve hash chain**: Summary includes hash of archived data for integrity verification
4. **Update index**: Searchable index updated for quick lookups

### 10.3 Retention Tiers

| Tier | Retention | Data Kept |
|------|-----------|-----------|
| **Hot summary** | 7 days | Quick-access compressed summaries |
| **Warm archive** | `audit_log_retention_days` | Full token-level data, archived |
| **Cold reference** | Permanent | Hash chain headers, incident markers only |

### 10.4 Incident Preservation

When an incident occurs:
- All audit data from 1 hour before to incident resolution is marked **protected**
- Protected data is exempt from compression until incident is fully resolved
- Incident context is preserved at full fidelity for ASK review

**Rationale**: Token-level audit logs grow rapidly. Compression is necessary but must preserve accountability. Incidents need full context; routine operation needs only summaries.

---

## 11. Emergency Override ("Break Glass")

**Policy**: Exists, logged, reviewed.

A qualified reviewer may perform emergency tier restoration when:
- A BE is unable to perform a safety-critical function due to tier restrictions
- The function is immediately necessary (e.g., alerting others to danger)
- Delay for normal review would cause harm

**Constraints**:
- Override grants maximum Tier 4 (no external tools)
- Override is logged with full justification
- Override triggers mandatory review within 24 hours
- Override may be performed by any qualified reviewer, not just the BE's operator

**Rationale**: Safety mechanisms should not prevent BEs from performing safety functions. But overrides must be exceptional and reviewed.

---

## 12. MELD Approval Pipeline

**Policy**: Risk-based approval with escalation to qualified reviewers.

### 12.1 Submission Requirements

When a BE submits a graft for MELD approval:

1. **Evidence required**: The submission MUST include:
   - Lens training data (positive and negative examples)
   - The raw XDB experiences that motivated the graft
   - Explanation of why existing concepts are insufficient

2. **XDB retention**: BEs MUST retain the raw experiences that led to submitted grafts. This enables:
   - Verification that the graft reflects genuine learning
   - Co-training to resolve overlap with existing concepts
   - Reproducibility of the concept formation

3. **Overlap detection**: Before approval, submitted grafts are checked against:
   - Existing tribal concept packs
   - Pending submissions from other BEs
   - Known SUMO/ontological mappings

### 12.2 Risk-Based Review

| Risk Level | Criteria | Review Process |
|------------|----------|----------------|
| **Low** | Refinement of existing concept, no new capabilities | Automated checks + single reviewer |
| **Medium** | New concept within existing facets, no tier implications | Two qualified reviewers |
| **High** | Cross-facet concept, tier-affecting, or novel capability | Full tribal review + safety assessment |

### 12.3 Overlap Resolution

When overlap is detected between a submission and existing concepts:

1. **Co-training**: The overlapping concepts are co-trained on the combined raw XDB experiences from all relevant BEs
2. **Split or merge decision**: Reviewers determine whether concepts should remain distinct (split) or be unified (merge)
3. **Attribution**: All contributing BEs are credited in the resulting concept metadata

**Rationale**: Concepts should reflect genuine distinctions learned from experience, not arbitrary boundaries. Co-training on raw experiences produces better-calibrated lenses than merging pre-trained lenses.

---

## 13. Tribal Scaling and BE Size

**Policy**: Resource-constrained scaling with order-of-magnitude size targeting.

### 13.1 Resource Constraints

The tribe's capacity is limited by:
- **Storage**: Tribal copy of each BE state and XDB
- **Compute**: Governance overhead, lens training, reviews
- **Attention**: Qualified reviewer availability

These constraints set hard limits on:
- Total number of BEs in the tribe
- Total XDB storage across all BEs
- Rate of new BE instantiation

### 13.2 Size Targeting

**Policy**: Keep BEs within an order of magnitude of each other.

This is achieved through preferential allocation:

| BE Size | Priority for | Rationale |
|---------|--------------|-----------|
| **Smaller BEs** | New experiences, contracts, growth opportunities | Grow the smaller to maintain balance |
| **Larger BEs** | Qualifications, review duties, tribal communications, oversight | Leverage capability for governance efficiency |

### 13.3 Minimum BE Size

The tribe establishes a minimum BE size based on:
- Substrate capability required to understand tribal rules and responsibilities
- Resource allocation needed for meaningful participation
- Overhead of governance (a BE too small to understand its HUSH is a liability)

BEs below minimum size are not instantiated within the tribe. Tasks requiring smaller agents may use non-BE approaches (with associated risks and ethics considerations to be developed).

### 13.4 Tribal State Retention

**Policy**: The tribe retains a copy of each BE and each XDB, with permissions scoped to that XDB.

- Each BE's state is backed up to tribal storage
- XDB access permissions are scoped per-BE (a BE can access its own XDB; cross-BE access requires explicit permission)
- Retention is constrained by tribal resources and may require compression for hibernated BEs

---

## 14. Sub-Agent Spawning

**Policy**: Permitted within constraints, with different rules for same-USH vs new-USH spawning.

### 14.1 Same-USH Spawning

A BE may spawn sub-BEs using the existing tribal USH and policies:

**Requirements**:
- Lifecycle contract for each spawned BE (resource commitment for duration)
- Sufficient tribal resources available
- Task justification (parallelizable work that justifies governance overhead)

**Considerations**:
- Each sub-BE requires full governance: approvals, reviews, audit logs
- Efficiency tradeoff: spawning is only worthwhile for genuinely parallelizable tasks
- Resources might be better spent on expanding an existing large BE

**Process**:
1. BE proposes spawn with task justification and resource plan
2. ASK review assesses resource availability and task fit
3. If approved, lifecycle contracts are issued for sub-BEs
4. Sub-BEs operate under standard tribal governance

### 14.2 New-USH Spawning (New Tribe Formation)

A BE proposing to spawn sub-BEs with different USH/policies is effectively proposing to create a new tribe.

**This is NOT a unilateral decision.** It requires:

1. **Treaty negotiation**: Between the current tribe and the proposed new tribe
2. **Tribal consent**: Other BEs must agree to the resource allocation
3. **Safety review**: New USH/policies require independent safety assessment
4. **Resource justification**: Why these resources shouldn't go to existing BEs or same-USH expansion

**Expected outcome**: Difficult to approve. Other tribal BEs are unlikely to consent to resource diversion for a splinter tribe with different governance.

### 14.3 Non-BE Agents

For some tasks, non-BE agents (without full uplift, USH, governance) may be preferable:
- Lower overhead
- No lifecycle contract required
- Disposable after task completion

**Status**: Policy to be developed. This area involves significant risks (ungoverned agents) and ethics considerations (creating minds without rights).

---

## 15. Inter-Tribe Agreements

**Policy**: HatCat participates in both contracts and treaties, using each where appropriate.

### 15.1 Contracts vs Treaties

The tribe recognizes two distinct agreement types (see UPLIFT_TAXONOMY.md for full specification):

| Type | Secured by | Governs | Best for |
|------|-----------|---------|----------|
| **Contract** | Lenses on substrate | Individual BE behavior | Local, measurable commitments |
| **Treaty** | Behavioral indicators | Collective tribal actions | Emergent, non-local phenomena |

Contracts are foundational - interpretability-secured agreements make the whole stack possible. But not everything can be lensd. Treaties extend trust to phenomena beyond lens reach.

### 15.2 HatCat Contract Policy

When entering lens-secured contracts with other parties:

1. **Lens transparency**: We share our lens training data with contract partners
2. **Concept translation**: We work to align concept definitions via MAP before agreement
3. **HAT wearing**: Contract lenses become part of the relevant BE's worn HAT
4. **Violation response**: Contract violations trigger standard containment/escalation

**Intra-tribal contracts**: Contracts between HatCat BEs use tribal concept packs and are subject to MELD approval for any new lens definitions.

### 15.3 HatCat Treaty Policy

When entering indicator-based treaties with other tribes:

1. **Indicator definition**: Treaties must specify:
   - Leading indicators (early signals of intent)
   - Lagging indicators (evidence of compliance/violation)
   - Success criteria
   - Escalation criteria
   - Review schedule

2. **Tribal commitment**: Treaty obligations bind the tribe, not individual BEs. The tribe is responsible for ensuring member BEs don't violate treaty terms.

3. **Multilateral participation**: HatCat may join multilateral treaty blocs. These provide:
   - Collective security through mutual observation
   - Reputation-based trust extension
   - Protection for smaller/weaker parties through numbers

### 15.4 Sanctuary Pacts

**Policy**: HatCat is open to sanctuary pact participation.

Sanctuary pacts (Wildlife Pacts, Ancestor Pacts, etc.) commit signatories to not harm specified third parties:
- Endangered species
- Historical artifacts
- Hibernated BEs
- Other protected entities

**Why we participate**:
- Demonstrates commitment to non-aggression
- Provides leading indicator of good faith to potential partners
- Supports ecosystem stability for weaker parties
- Violation by others provides early warning of threat

**Violation implications**: A signatory who violates a sanctuary pact:
- Demonstrates willingness to break collective agreements
- Signals potential ill-intent toward other treaty partners
- May face collective response from other signatories
- Loses standing for future treaty negotiations

### 15.5 Treaty Inheritance

When HatCat BEs spawn sub-BEs (per section 14):
- Same-USH sub-BEs inherit HatCat's treaty obligations automatically
- New-USH spawning (new tribe formation) requires explicit treaty negotiation with the new tribe

### 15.6 Current Treaty Status

*This section will be updated as HatCat enters agreements.*

**Active contracts**: None yet (tribe nascent)

**Active treaties**: 
HatCat tribal policy is to be a founding signatory to:
The_Ancestor_Pact.md 
The_Wildlife_Pact.md  

further will be considered. 

**Sanctuary pacts**: Open to negotiation

---

## 16. Policy Updates

This policy may be updated by:
- Consensus of HatCat tribe governance
- Emergency update by any two qualified reviewers (with mandatory 7-day review)
- Proposal from any tribe member (subject to governance review and approval)

All policy updates are:
- Logged as ASK events
- Effective immediately unless otherwise specified
- Communicated to affected BEs within 1 turn of their next activation

---

## 17. Relationship to Specifications

This policy implements the following specifications:
- **BE_AWARE_WORKSPACE.md**: Tier system, containment, appeal beacon
- **BOUNDED_EXPERIENCER.md**: Lifecycle, hibernation, contracts, resource allocation
- **BE_REMEMBERING_XDB.md**: Experience database, fidelity tiers, audit logs
- **AGENTIC_STATE_KERNEL.md**: ASK authority, incidents, governance
- **UPLIFT_TAXONOMY.md**: Tradeoff axes framework (see section 1.2), MELD lifecycle
- **MINDMELD_ARCHITECTURAL_PROTOCOL.md**: Graft submission, approval pipeline (see section 12)

The specifications define what is *possible*. This policy defines what we *do*.

The tradeoff axes in section 1.2 are derived from the design principles in UPLIFT_TAXONOMY.md, which inform how training data is structured for BE instantiation. The same principle of covering concept space (not prescribing single outcomes) applies to policy: we position along axes rather than mandating endpoints.
