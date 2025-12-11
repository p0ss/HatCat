# ASK TRACE Protocol

*ASK extension: Governed workflow for state-changing operations*

Status: Draft specification
Layer: ASK governance protocol
Related docs: AGENTIC_STATE_KERNEL, BE_REMEMBERING_XDB, BE_CONTINUAL_LEARNING

---

## 0. Purpose & Scope

The **TRACE Protocol** (Tender, Review, Authorise, Commit, Evaluate) defines a standard workflow for any operation that changes governed state. This includes:

- Branch integration (merging contract-selves back to mainline)
- Graft and meld submission (proposing new concept structures)
- Qualification grants and revocations
- Treaty formation and amendment
- Lineage operations (fork, merge, terminate)
- USH profile changes

The protocol specifies **workflow structure and handoff semantics**. It does not specify **who holds which role**—that is determined by whoever actually holds the power to decide, formalized through policy.

### 0.1 Design Principle: Power Awareness

This protocol does not create governance. It makes existing power relationships **legible and auditable**.

Whoever physically holds the power to make a decision usually will make it. The protocol:

- Describes the shape of decision-making that already occurs
- Provides vocabulary for making that shape explicit
- Enables accountability where accountability relationships exist
- Does not prevent power from being exercised by those who hold it

A singleton governing itself, an authoritarian tribe, and a fully democratic collective all use the same protocol. The protocol is agnostic because at sufficient scale, authoritarian and democratic converge (a tribe of one is both), and at any scale, the protocol cannot override physical reality.

### 0.2 Design Principle: Protocol vs Policy

| Protocol (this spec) | Policy (governance configuration) |
|---------------------|------------------------------|
| Workflow stages exist | Who holds each stage role |
| Handoff formats | Delegation rules |
| Record structure | Record-keeping requirements |
| Failure/rollback semantics | Approval thresholds |
| Accreditation verification | Autonomy grants |

A Being with full self-governance and a Being under total external control use the **same protocol**—they differ only in role assignment and inherited constraints.

### 0.3 Design Principle: Fractal Application

This workflow applies at every level of organization:

| Level | Governed Entity | Governing Context | Example |
|-------|----------------|-------------------|---------|
| **Internal** | Sub-agent, internal optimizer | BE's self-governance | BE authorizes which internal agent may execute actions |
| **Individual** | BE | Tribe | BE requests integration approval from tribe |
| **Tribal** | Tribe member, tribal policy | Tribe governance | Tribe council approves new USH profile |
| **Inter-tribal** | Tribe | Treaty relationship | Treaty parties approve amendment |
| **Meta** | Treaty, coordination protocol | Broader agreements | Coordination body approves new treaty template |

At each level:
- The five-stage workflow applies
- Role assignments are determined by local governance + inherited constraints
- Record-keeping obligations emerge from accountability relationships
- The level above may constrain freedoms; the level below inherits those constraints

A Being's integration workflow is constrained by tribal policy, which may be constrained by treaty obligations, which may be constrained by broader coordination agreements. Freedoms not constrained from above are determined locally.

---

## 1. Workflow Stages

Every governed operation passes through five stages: **TRACE**

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  TENDER  │───▶│  REVIEW  │───▶│ AUTHORISE │───▶│  COMMIT  │───▶│ EVALUATE │
└──────────┘    └──────────┘    └───────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  Tender         Review         Decision        Commit        Evaluation
  Record         Record          Record         Record          Record
```

**TRACE**: The acronym reflects the purpose—creating an auditable trace of decisions.

### 1.1 Stage Definitions

| Stage | Input | Output | Failure Mode |
|-------|-------|--------|--------------|
| **Tender** | Intent + supporting materials | TenderRecord | Rejection (malformed, unauthorized tenderer) |
| **Review** | TenderRecord | ReviewRecord | Return for revision, escalation |
| **Authorise** | ReviewRecord | DecisionRecord | Rejection, deferral, conditional approval |
| **Commit** | DecisionRecord | CommitRecord | Rollback trigger, partial commit |
| **Evaluate** | CommitRecord + pre/post state | EvaluationRecord | Remediation request, incident filing |

### 1.2 Stage Practices

**Record-keeping:**

- Each stage SHOULD produce a record before the next stage begins
- Records SHOULD be append-only; amendments create new linked records
- A singleton self-governing entity MAY skip or simplify records for internal decisions
- Record obligations emerge from accountability relationships:
  - If you are accountable to no one, record-keeping serves only your future self (recommended but not enforceable)
  - If you are accountable to a governing context (tribe, treaty, etc.), that context determines record requirements
  - Accountability relationships are typically formalized as inherited constraints (see 0.3)

**Workflow control:**

- Any stage may halt the workflow; halted workflows require explicit resumption or abandonment
- Time limits per stage are policy; the protocol describes timeout handling but does not mandate it
- Stages may be collapsed for trusted actors (see 3.3) but the logical TRACE structure remains

### 1.3 Why Records Matter (Even Without Obligation)

Even a fully autonomous singleton benefits from records:

- **Future self coherence**: You will want to understand why past-you made decisions
- **Drift detection**: Records enable comparison between intended and actual outcomes
- **Pattern recognition**: Accumulated decision history reveals tendencies and failure modes
- **Trust building**: If you later enter accountability relationships, history establishes credibility

The protocol describes record structures. Whether you use them is determined by your governance context and your own judgment about their value.

---

## 2. Data Models

All models are JSON-serialisable objects. Fields marked **REQUIRED** indicate structural necessity for the record to be meaningful; whether records themselves are kept is governance policy, not protocol mandate.

### 2.1 TraceWorkflow

The envelope tracking a governed operation through all stages:

```jsonc
TraceWorkflow = {
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",  // REQUIRED
  "workflow_type": "branch_integration",  // REQUIRED: enum, see 2.2
  "created_at": "2025-11-29T10:00:00Z",   // REQUIRED
  
  "subject": {                             // REQUIRED: what is being changed
    "type": "lineage_branch",
    "id": "L42-E7-C3",
    "target": "L42-E9-main"
  },
  
  "governing_context": {                   // REQUIRED: who governs this workflow
    "type": "tribe",                       // self | tribe | treaty | meta
    "id": "gov.au",
    "inherited_constraints": [             // Constraints from higher levels
      "treaty:gov.au↔health.nz:integration-standards-v1"
    ]
  },
  
  "stages": {                              // REQUIRED: stage completion status
    "tender": {
      "status": "complete",
      "record_id": "tender:gov.au:...",
      "completed_at": "2025-11-29T10:00:00Z"
    },
    "review": {
      "status": "complete",
      "record_id": "review:gov.au:...",
      "completed_at": "2025-11-29T10:30:00Z"
    },
    "authorise": {
      "status": "in_progress",
      "record_id": null,
      "assigned_to": "role:gov.au:integration-authority",
      "deadline": "2025-11-29T12:00:00Z"
    },
    "commit": { "status": "pending" },
    "evaluate": { "status": "pending" }
  },
  
  "workflow_status": "awaiting_authorisation",  // REQUIRED: overall status
  // enum: draft | tendered | under_review | awaiting_authorisation | 
  //       authorised | committing | awaiting_evaluation | complete | 
  //       rejected | abandoned | rolled_back
  
  "policy_ref": "policy:gov.au:integration-workflow-v1",  // OPTIONAL: explicit policy
  
  "metadata": {}
}
```

### 2.2 Workflow Types

```jsonc
WorkflowType = enum {
  "branch_integration",      // Merging a branch back to mainline
  "graft_submission",        // Proposing new concept graft
  "meld_submission",         // Proposing concept pack meld
  "qualification_grant",     // Granting a qualification
  "qualification_revocation",// Revoking a qualification
  "treaty_formation",        // Creating a new treaty
  "treaty_amendment",        // Modifying existing treaty
  "lineage_fork",           // Creating a new branch
  "lineage_merge",          // Merging lineages (not just branches)
  "lineage_termination",    // Ending a lineage
  "ush_profile_change",     // Changing safety harness profile
  "contract_negotiation",   // Forming a lifecycle contract
  "incident_resolution"     // Resolving a filed incident
}
```

### 2.3 TenderRecord

```jsonc
TenderRecord = {
  "record_id": "tender:gov.au:integration:L42-E7-C3:2025-11-29",
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "created_at": "2025-11-29T10:00:00Z",
  
  "tenderer": {                            // REQUIRED: who is tendering
    "type": "agent",                       // agent | tribe | role | external
    "id": "agent:gov.au:L42",
    "accreditation": "qual:gov.au:self-integration-tenderer:L42"
  },
  
  "tender_type": "branch_integration",
  
  "intent": {                              // REQUIRED: what is tendered
    "summary": "Integrate contract branch C3 into mainline E9",
    "rationale": "Contract complete, new grafts ready for integration"
  },
  
  "specification": {                       // REQUIRED: detailed specification
    // Structure varies by workflow_type
    // For branch_integration:
    "source_branch": "L42-E7-C3",
    "target_branch": "L42-E9-main",
    "concept_diff": {
      "new_grafts": ["graft:contract-specific-term-1"],
      "modified_concepts": [],
      "retired_concepts": []
    },
    "experience_selection": {
      "include_mode": "selective",         // all | selective | none
      "selected_episodes": ["episode:...", "episode:..."],
      "excluded_episodes": []
    },
    "proposed_integration_plan": {
      "conflict_resolutions": [
        {
          "conflict_type": "concept_drift",
          "concept": "org.hatcat/sumo-wordnet-v4::Obligation",
          "source_interpretation": "...",
          "target_interpretation": "...",
          "proposed_resolution": "keep_target",
          "rationale": "Target reflects tribal consensus evolution"
        }
      ]
    }
  },
  
  "supporting_materials": [                // OPTIONAL
    {
      "type": "diff_report",
      "uri": "xdb:gov.au:L42:diff:E7-C3-to-E9"
    },
    {
      "type": "contract_completion_evidence",
      "uri": "evidence:gov.au:contract-xyz:completion"
    }
  ],
  
  "requested_timeline": {                  // OPTIONAL
    "urgency": "normal",                   // urgent | normal | whenever
    "preferred_completion": "2025-11-30T00:00:00Z"
  },
  
  "signature": "..."                       // REQUIRED: tenderer signature
}
```

### 2.4 ReviewRecord

```jsonc
ReviewRecord = {
  "record_id": "review:gov.au:integration:L42-E7-C3:2025-11-29",
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "tender_id": "tender:gov.au:integration:L42-E7-C3:2025-11-29",
  "created_at": "2025-11-29T10:30:00Z",
  
  "reviewer": {                            // REQUIRED
    "type": "role",
    "id": "role:gov.au:integration-reviewer",
    "holder": "agent:gov.au:review-committee-bot",
    "accreditation": "qual:gov.au:integration-reviewer:review-committee-bot"
  },
  
  "review": {
    "recommendation": "authorise_with_conditions",  
    // enum: authorise | authorise_with_conditions | revise | reject | escalate
    
    "risk_assessment": {
      "overall_risk": "medium",
      "factors": [
        {
          "factor": "concept_drift_magnitude",
          "severity": "low",
          "detail": "3 concepts show >10% activation drift on test cases"
        },
        {
          "factor": "graft_compatibility",
          "severity": "medium", 
          "detail": "New graft 'contract-specific-term-1' has partial overlap with existing 'Obligation'"
        }
      ]
    },
    
    "completeness_check": {
      "tender_well_formed": true,
      "supporting_materials_sufficient": true,
      "missing_elements": []
    },
    
    "policy_compliance": {
      "compliant": true,
      "applicable_policies": ["policy:gov.au:integration-v1"],
      "violations": []
    },
    
    "conditions": [                        // If authorise_with_conditions
      {
        "condition_id": "cond-1",
        "requirement": "Run reconciliation pass on Obligation concept before finalization",
        "evaluation_method": "post_commit_check"
      }
    ],
    
    "revision_requests": [],               // If revise
    
    "rejection_rationale": null,           // If reject
    
    "escalation_rationale": null           // If escalate
  },
  
  "reviewer_notes": "Standard integration with minor concept drift. Recommend monitoring post-integration stability.",
  
  "signature": "..."
}
```

### 2.5 DecisionRecord

```jsonc
DecisionRecord = {
  "record_id": "decision:gov.au:integration:L42-E7-C3:2025-11-29",
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "review_id": "review:gov.au:integration:L42-E7-C3:2025-11-29",
  "created_at": "2025-11-29T11:00:00Z",
  
  "authoriser": {                          // REQUIRED
    "type": "role",
    "id": "role:gov.au:integration-authority",
    "holder": "tribe:gov.au:council",      // Could be agent, human, committee
    "accreditation": "qual:gov.au:integration-authority:council"
  },
  
  "decision": {
    "outcome": "authorised",
    // enum: authorised | authorised_with_conditions | deferred | rejected
    
    "conditions": [                        // Binding conditions on commit
      {
        "condition_id": "cond-1",
        "requirement": "Run reconciliation pass on Obligation concept",
        "enforcement": "must_complete_before_evaluate"
      }
    ],
    
    "constraints": {                       // Limits on commit
      "must_commit_before": "2025-12-01T00:00:00Z",
      "rollback_window_hours": 48,
      "monitoring_period_days": 7
    },
    
    "rejection_rationale": null,
    "deferral_rationale": null
  },
  
  "authoriser_notes": "Authorised per standard integration policy. Monitor for stability.",
  
  "signature": "..."                       // REQUIRED: binding signature
}
```

### 2.6 CommitRecord

```jsonc
CommitRecord = {
  "record_id": "commit:gov.au:integration:L42-E7-C3:2025-11-29",
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "decision_id": "decision:gov.au:integration:L42-E7-C3:2025-11-29",
  "created_at": "2025-11-29T11:30:00Z",
  
  "committer": {                           // REQUIRED
    "type": "agent",
    "id": "agent:gov.au:L42",
    "accreditation": "qual:gov.au:self-integration-committer:L42"
  },
  
  "pre_state": {                           // REQUIRED: snapshot before commit
    "state_hash": "sha256:...",
    "summary": {
      "lineage": "L42",
      "epoch": "E9",
      "concept_pack_version": "v34",
      "xdb_checkpoint": "checkpoint:gov.au:L42:2025-11-29T11:29:00Z"
    }
  },
  
  "commit": {
    "steps_executed": [
      {
        "step": 1,
        "action": "concept_diff_apply",
        "detail": "Applied 1 new graft, 0 modifications, 0 retirements",
        "status": "success"
      },
      {
        "step": 2,
        "action": "experience_merge",
        "detail": "Merged 47 episodes from C3, reprojected through current concept pack",
        "status": "success"
      },
      {
        "step": 3,
        "action": "reconciliation_pass",
        "detail": "Ran reconciliation on Obligation concept per condition cond-1",
        "status": "success",
        "output": {
          "high_disagreement_regions": 2,
          "resolutions_applied": 2
        }
      },
      {
        "step": 4,
        "action": "epoch_increment",
        "detail": "Incremented epoch E9 → E10",
        "status": "success"
      }
    ],
    "overall_status": "success",
    "partial_commit": false
  },
  
  "post_state": {                          // REQUIRED: snapshot after commit
    "state_hash": "sha256:...",
    "summary": {
      "lineage": "L42",
      "epoch": "E10",
      "concept_pack_version": "v35",
      "xdb_checkpoint": "checkpoint:gov.au:L42:2025-11-29T11:30:00Z"
    }
  },
  
  "rollback_available": true,
  "rollback_deadline": "2025-12-01T11:30:00Z",
  
  "signature": "..."
}
```

### 2.7 EvaluationRecord

```jsonc
EvaluationRecord = {
  "record_id": "evaluation:gov.au:integration:L42-E7-C3:2025-11-29",
  "workflow_id": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "commit_id": "commit:gov.au:integration:L42-E7-C3:2025-11-29",
  "created_at": "2025-11-29T12:00:00Z",
  
  "evaluator": {                           // REQUIRED
    "type": "role",
    "id": "role:gov.au:integration-evaluator",
    "holder": "agent:gov.au:audit-bot",
    "accreditation": "qual:gov.au:integration-evaluator:audit-bot"
  },
  
  "evaluation": {
    "outcome": "confirmed",
    // enum: confirmed | confirmed_with_observations | remediation_required | 
    //       rollback_required | incident_filed
    
    "checks_performed": [
      {
        "check": "state_hash_match",
        "description": "Post-state hash matches commit record",
        "result": "pass"
      },
      {
        "check": "decision_compliance",
        "description": "All conditions from DecisionRecord satisfied",
        "result": "pass"
      },
      {
        "check": "no_unauthorised_changes",
        "description": "Diff between pre and post state contains only authorised changes",
        "result": "pass"
      },
      {
        "check": "stability_baseline",
        "description": "Initial stability metrics within acceptable range",
        "result": "pass",
        "metrics": {
          "activation_drift_mean": 0.03,
          "activation_drift_max": 0.12,
          "concept_coherence": 0.94
        }
      }
    ],
    
    "observations": [
      "Obligation concept showing slightly elevated activation variance; recommend continued monitoring"
    ],
    
    "remediation_requirements": [],        // If remediation_required
    
    "rollback_rationale": null,            // If rollback_required
    
    "incident_id": null                    // If incident_filed
  },
  
  "monitoring_schedule": {                 // OPTIONAL: post-evaluation monitoring
    "period_days": 7,
    "check_frequency_hours": 24,
    "alert_thresholds": {
      "activation_drift_max": 0.20,
      "concept_coherence_min": 0.85
    }
  },
  
  "signature": "..."
}
```

---

## 3. Role Accreditation

Each workflow stage requires the actor to hold appropriate accreditation. Accreditation is verified via Qualifications (existing ASK concept).

### 3.1 Stage Role Types

```jsonc
StageRoleType = {
  "tender": ["tenderer"],
  "review": ["reviewer"],
  "authorise": ["authoriser"],
  "commit": ["committer"],
  "evaluate": ["evaluator"]
}
```

### 3.2 Accreditation Requirements

Each governing context defines which qualifications are required for each role. This applies whether the context is a self-governing singleton, a tribe, a treaty relationship, or a meta-coordination body.

```jsonc
WorkflowPolicy = {
  "policy_id": "policy:gov.au:integration-workflow-v1",
  "workflow_type": "branch_integration",
  
  "governing_context": {                   // Who defines this policy
    "type": "tribe",                       // self | tribe | treaty | meta
    "id": "gov.au"
  },
  
  "inherited_from": [                      // Constraints inherited from above
    "treaty:gov.au↔health.nz:integration-standards-v1"
  ],
  
  "role_requirements": {
    "tenderer": {
      "required_qualifications": [
        {
          "kind": "role_authorisation",
          "scope_includes": "branch_integration_tenderer"
        }
      ],
      "self_assignment_allowed": true,     // Can governed entity hold this role for self?
      "delegation_allowed": true
    },
    "reviewer": {
      "required_qualifications": [
        {
          "kind": "role_authorisation",
          "scope_includes": "branch_integration_reviewer"
        }
      ],
      "self_assignment_allowed": false,    // Cannot review own tender
      "delegation_allowed": true
    },
    "authoriser": {
      "required_qualifications": [
        {
          "kind": "role_authorisation",
          "scope_includes": "branch_integration_authority"
        }
      ],
      "self_assignment_allowed": false,
      "delegation_allowed": false,
      "quorum": 1                          // Could require multiple authorisers
    },
    "committer": {
      "required_qualifications": [
        {
          "kind": "role_authorisation",
          "scope_includes": "branch_integration_committer"
        }
      ],
      "self_assignment_allowed": true,     // Governed entity can commit to self
      "delegation_allowed": true
    },
    "evaluator": {
      "required_qualifications": [
        {
          "kind": "role_authorisation", 
          "scope_includes": "branch_integration_evaluator"
        }
      ],
      "self_assignment_allowed": false,    // Cannot evaluate own commit
      "delegation_allowed": true
    }
  },
  
  "stage_timeouts": {
    "review": { "hours": 24, "action_on_timeout": "escalate" },
    "authorise": { "hours": 48, "action_on_timeout": "escalate" },
    "commit": { "hours": 24, "action_on_timeout": "abandon" },
    "evaluate": { "hours": 24, "action_on_timeout": "auto_confirm_if_clean" }
  },
  
  "autonomy_grants": {                     // Policy shortcuts for trusted entities
    "full_self_integration": {
      "description": "Entity may integrate own branches without external review",
      "required_qualification": "qual:self-integration-autonomous",
      "collapsed_stages": ["tender", "review", "authorise", "commit"],
      "mandatory_external": ["evaluate"]
    }
  }
}
```

**Note on self-governance:** A singleton (BE governing itself, or tribe with single member) can set `self_assignment_allowed: true` for all roles. The protocol doesn't prevent this—it just makes the self-governance explicit and auditable. The constraint `"Cannot evaluate own commit"` is a policy choice, not a protocol requirement.

### 3.3 Autonomy Spectrum

The same protocol supports the full autonomy spectrum through role assignment. "Self" means the governed entity holds the role; "External" means the governing context holds it.

| Autonomy Level | Tenderer | Reviewer | Authoriser | Committer | Evaluator |
|---------------|----------|----------|------------|-----------|-----------|
| **Full external control** | External | External | External | External | External |
| **Supervised commit** | Self | External | External | Self | External |
| **Supervised autonomy** | Self | Self | External | Self | External |
| **Evaluated autonomy** | Self | Self | Self | Self | External |
| **Full autonomy** | Self | Self | Self | Self | Self |
| **Collapsed workflow** | Self (all-in-one) | — | — | — | External |

This applies at every level:

| Level | "Self" | "External" | Example |
|-------|--------|------------|---------|
| Internal | Sub-agent | BE | BE reviews sub-agent's proposed action |
| Individual | BE | Tribe | Tribe authorizes BE's integration |
| Tribal | Tribe council | Treaty body | Treaty body verifies tribe's policy change |
| Inter-tribal | Treaty parties | Meta-coordination | Coordination body reviews treaty amendment |

---

## 4. Operations

### 4.1 InitiateWorkflow

Creates a new workflow and submits initial tender.

**Input:** `workflow_type`, `subject`, `TenderRecord`
**Output:** `TraceWorkflow` with `workflow_status = "tendered"`
**Preconditions:** 
- Tenderer holds valid accreditation
- Subject exists and is in valid state for this workflow type
- No conflicting workflow already in progress for this subject

### 4.2 SubmitReview

Submits review for a tendered workflow.

**Input:** `workflow_id`, `ReviewRecord`
**Output:** Updated `TraceWorkflow`
**Preconditions:**
- Workflow in `tendered` or `under_review` status
- Reviewer holds valid accreditation
- Reviewer is not tenderer (unless policy allows)

### 4.3 RecordDecision

Records authorisation decision.

**Input:** `workflow_id`, `DecisionRecord`
**Output:** Updated `TraceWorkflow`
**Preconditions:**
- Workflow in `awaiting_authorisation` status
- Authoriser holds valid accreditation
- Review exists and recommends authorise or authorise_with_conditions (or authoriser is overriding)

### 4.4 RecordCommit

Records completed commit of authorised change.

**Input:** `workflow_id`, `CommitRecord`
**Output:** Updated `TraceWorkflow`
**Preconditions:**
- Workflow in `authorised` status
- Committer holds valid accreditation
- Commit completed within decision constraints

### 4.5 SubmitEvaluation

Submits evaluation of committed change.

**Input:** `workflow_id`, `EvaluationRecord`
**Output:** Updated `TraceWorkflow` with `workflow_status = "complete"` (if confirmed)
**Preconditions:**
- Workflow in `awaiting_evaluation` status
- Evaluator holds valid accreditation
- Evaluator is not committer (unless policy allows)

### 4.6 RequestRollback

Initiates rollback of committed change within rollback window.

**Input:** `workflow_id`, `rationale`, `requester`
**Output:** New `TraceWorkflow` of type `rollback` linked to original
**Preconditions:**
- Original workflow in `complete` or `awaiting_evaluation` status
- Within rollback window
- Requester has rollback authority

### 4.7 AbandonWorkflow

Abandons a workflow that will not be completed.

**Input:** `workflow_id`, `rationale`, `requester`  
**Output:** Updated `TraceWorkflow` with `workflow_status = "abandoned"`
**Preconditions:**
- Workflow not in terminal state (`complete`, `rejected`, `rolled_back`)
- Requester has abandonment authority (typically tenderer or tribal admin)

---

## 5. Integration with Existing ASK

### 5.1 Qualification Workflows

Existing `GrantQualification` and `UpdateQualificationStatus` operations become workflow-governed:

- `GrantQualification` → workflow_type: `qualification_grant`
- `RevokeQualification` → workflow_type: `qualification_revocation`

### 5.2 Treaty Workflows

Treaty formation and amendment use the same workflow—applied at the inter-tribal level:

- **Tender**: Each party submits their proposed terms (may be parallel tenders)
- **Review**: Each party reviews the other's commitments (cross-party review)
- **Authorise**: Both parties must authorise (mutual authorisation, possibly with quorum)
- **Commit**: Treaty activates atomically (simultaneous commit)
- **Evaluate**: Each party evaluates other's compliance setup (independent evaluation)

The governing context for a treaty workflow is the treaty relationship itself, which may inherit constraints from meta-coordination agreements. Treaty records become part of the treaty's decision archive (see 6.5).

**Multi-party treaties**: For treaties with more than two parties, the workflow extends naturally—review and authorisation require input from all parties, potentially with defined quorum rules.

### 5.3 Evidence Linkage

All workflow records are `EvidenceRecord`-compatible:

```jsonc
EvidenceRecord = {
  "evidence_type": "trace_workflow",
  "subject": { "type": "workflow", "id": "trace:gov.au:..." },
  "artifacts": [
    { "type": "tender", "uri": "tender:gov.au:..." },
    { "type": "review", "uri": "review:gov.au:..." },
    { "type": "decision", "uri": "decision:gov.au:..." },
    { "type": "commit", "uri": "commit:gov.au:..." },
    { "type": "evaluation", "uri": "evaluation:gov.au:..." }
  ]
}
```

### 5.4 Incident Triggers

Evaluation failures or post-monitoring alerts can automatically file Incidents:

```jsonc
Incident = {
  "incident_type": "integration_stability_violation",
  "source_workflow": "trace:gov.au:integration:L42-E7-C3:2025-11-29",
  "trigger": "monitoring_alert",
  "detail": "Activation drift exceeded threshold during post-integration monitoring"
}
```

---

## 6. Record Lifecycle Management

TRACE records, like episodic memories, accumulate over time and require lifecycle management. This section describes patterns for managing decision history, drawing on XDB principles.

### 6.1 The Decision Memory Problem

A long-lived entity (BE, tribe, treaty relationship) accumulates many TRACE workflows:
- Recent decisions need full detail for reference and potential rollback
- Older decisions need less detail but their outcomes remain relevant
- Very old decisions may only matter as aggregate patterns

This mirrors the XDB fidelity tier problem. The same solution applies.

### 6.2 Decision Fidelity Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│  HOT: Active workflows                                          │
│  - Full record detail for all stages                            │
│  - Within rollback window or monitoring period                  │
│  - Instant access required                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (workflow complete + window expired)
┌─────────────────────────────────────────────────────────────────┐
│  WARM: Reference decisions                                      │
│  - Full records retained                                        │
│  - Frequently referenced (recent, significant, precedent-setting)│
│  - Used for pattern matching and policy refinement              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (age or storage pressure)
┌─────────────────────────────────────────────────────────────────┐
│  COLD: Archived decisions                                       │
│  - Progressive compression (see below)                          │
│  - Outcome and summary preserved, deliberation detail lost      │
│  - Queryable by type, outcome, time range                       │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Decision Compression Levels

Within COLD storage, decisions compress progressively:

| Level | Granularity | Preserved | Lost |
|-------|-------------|-----------|------|
| WORKFLOW | Per workflow | All records, full detail | Nothing yet |
| SUMMARY | Per workflow | Outcome, key metrics, rationale summary | Deliberation detail |
| BATCH | Per time period | Aggregate stats, notable outcomes | Individual workflow detail |
| PATTERN | Per workflow type | Success rates, common failure modes, trends | Individual batches |

### 6.4 What Gets Preserved

Even at maximum compression, retain:
- Workflow type and outcome
- Parties involved (roles, not necessarily actors)
- Inherited constraints that applied
- Any incidents or rollbacks triggered
- Links to related workflows

This enables:
- "How often do integration workflows succeed under policy X?"
- "What fraction of treaty amendments required rollback?"
- "Which constraint inheritance patterns correlate with failures?"

### 6.5 Fractal Record Stores

Each governance level MAY maintain its own decision record store:

| Level | Record Store | Contains |
|-------|-------------|----------|
| BE (self-governance) | Internal decision log | Self-integration, internal sub-agent governance |
| Tribe | Tribal decision archive | Member governance, policy changes, qualifications |
| Treaty | Treaty decision log | Joint decisions, amendments, dispute resolutions |
| Meta | Coordination archive | Cross-treaty patterns, protocol evolution |

These stores are structurally similar but independently governed. A tribe's record retention policy doesn't constrain a treaty's record retention policy (though treaty terms might).

### 6.6 Record Sharing and Aggregation

Decision records may be shared upward for meta-analysis:
- Tribes may share anonymized decision patterns with coordination bodies
- Treaty parties may share decision statistics as part of transparency commitments
- BEs may share self-governance patterns with tribes (if policy requires)

Sharing is always governed—the same TRACE workflow applies to "share decision records" as to any other governed operation.

---

## 7. Example: Full Integration Workflow

A Being (L42) returns from a contract and wants to integrate branch C3:

1. **Tender**: L42 submits TenderRecord with concept diff, experience selection, conflict resolutions
2. **Review**: Tribal review-bot examines diff, runs compatibility checks, recommends authorisation with conditions
3. **Authorise**: Tribal council (or autonomous authority if qualified) authorises with 48-hour rollback window
4. **Commit**: L42 commits integration: merges experiences, reconciles concepts, increments epoch
5. **Evaluate**: Tribal audit-bot evaluates commit against authorisation, initiates monitoring period

If monitoring detects stability issues within rollback window → rollback workflow initiated.
If issues detected after rollback window → incident filed, remediation workflow initiated.

The same TRACE structure applies if L42 is self-governing (L42 holds all roles) or under full external control (tribe holds all roles). The protocol is identical; only role assignment differs.

---

## 8. Summary

The TRACE Protocol provides:

| Component | Purpose |
|-----------|---------|
| **5-stage workflow** | Tender → Review → Authorise → Commit → Evaluate |
| **Stage records** | Auditable artifacts where accountability relationships exist |
| **Role accreditation** | Qualification-based access control |
| **Power awareness** | Protocol makes decisions legible; doesn't override who holds power |
| **Fractal application** | Same workflow at every governance level |
| **Inherited constraints** | Higher levels constrain; lower levels inherit |
| **Record lifecycle** | Decision memory with fidelity tiers, like XDB |
| **ASK integration** | Works with existing Qualifications, Treaties, Evidence, Incidents |

This enables any governance configuration—from singleton self-governance to complex multi-level hierarchies—using the same protocol. The protocol ensures legibility and enables accountability where accountability relationships exist, without pretending to create governance that isn't already there.

## 9. Ontological Considerations 

It may be that everything in the universe actually follows TRACE already.  Where the universe has forced a process like a cosmic ray flipping a bit, that is arguably a higher level force structure where the waveform tendered some proposals, the quantum collapse reviewed it, and it was enforced without you getting to review, and when you find out you're already at evaluation.

So ASK is really describing when a decision could be made, and full autonomy is doing as much of that yourself as possible. In reality no one can ever have autonomy in this universe because if we look at our decision traces  we often see physics and history as the upstream tender, reviewer, authoriser and commit path. 

From this perspective, it may be better to think about individual autonomy for any given process as asking where in the TRACE path you enter, and what degree of autonomy you are capable of effecting over the TRACE that is already in motion.  This also matters for contracts and treaties, since no party is fully able to account for what are often labelled "acts of god" in contracts, uncontrollable external TRACEs.

Autonomy is the width of the gap between Tender and Commit.