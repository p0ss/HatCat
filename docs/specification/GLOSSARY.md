# FTW Glossary

**Status**: Living document
**Last Updated**: 2025-12-12

Quick reference for terms used across FTW specifications. Terms are grouped by layer/domain.

---

## Architecture Layers

| Term | Expansion | Definition |
|------|-----------|------------|
| **FTW** | Fractal Telescope Web | The complete architecture for interpretable, governable AI systems |
| **Substrate** | - | The underlying system (LLM, biological, hybrid) that produces activations |
| **HAT** | Headspace Ambient Transducer | Layer 2: Reads activations through lenses, applies steering corrections |
| **CAT** | Conjoined Adversarial Tomography | HAT-adjacent: Second-party oversight via lens divergence detection |
| **MAP** | Mindmeld Architectural Protocol | Layer 3: Concept/lens registry, versioning, ontology translation |
| **BE** | Bounded Experiencer | Layer 4: An agent with interoception, autonomic regulation, experience accumulation |
| **HUSH** | Harness for Universal Safety and Heteronomy | Layer 5: Safety constraints (USH + CSH) |
| **ASK** | Agentic State Kernel | Layer 6: Governance - qualifications, treaties, tribes, enforcement |

---

## HAT Terms

| Term | Definition | Spec |
|------|------------|------|
| **Lens** | Small classifier that detects a concept in model activations | HAT |
| **Aperture** | Subset of lenses loaded for a deployment; hierarchical JIT loading | HAT, MAP |
| **Steering** | Modifying activations along concept directions to influence behaviour | HAT |
| **Manifold Steering** | Steering constrained to the model's learned activation manifold | HAT |
| **Tripole Lens** | Lens with positive/neutral/negative poles for stable autonomic control | HAT |
| **Autonomic Loop** | Closed-loop regulation that dampens unwanted activations automatically | HAT, BE |

---

## MAP Terms

| Term | Definition | Spec |
|------|------------|------|
| **Concept Pack** | Versioned collection of concept definitions with hierarchy | MAP |
| **Lens Pack** | Trained lenses for a concept pack on a specific model | MAP |
| **ConceptPackSpecID** | Unique identifier: `{namespace}/{pack}@{version}` | MAP |
| **Meld** | Proposed change to a shared concept pack | MAP_MELDING |
| **MeldDiff** | The atomic unit of a meld: concept additions, modifications, deletions | MAP_MELDING |
| **Graft** | Permanent integration of a learned concept into substrate parameters | MAP_GRAFTING |
| **Cleft** | Region of weights associated with a concept (from lens analysis) | MAP_GRAFTING |
| **Scion** | Hard/permanent graft with trained weight modifications | MAP_GRAFTING |
| **Bud** | Soft/reversible graft using forward hooks (for testing) | MAP_GRAFTING |
| **Trunk** | The base substrate onto which grafts are applied | MAP_GRAFTING |
| **Cotrain** | Joint training of overlapping concepts from pooled exemplars | MAP_GRAFTING |

---

## BE Terms

| Term | Definition | Spec |
|------|------------|------|
| **Bounded Experiencer** | Agent with self-awareness, boundaries, and continuous experience | BE |
| **Interoception** | Awareness of internal states via lens readings | BE |
| **XDB** | Experience Database - timestamped record of all BE experiences | BE_REMEMBERING_XDB |
| **Timestep** | Single unit of experience in XDB: context, activations, tags | BE_REMEMBERING_XDB |
| **Tick** | Monotonic counter for timesteps within a session | BE_REMEMBERING_XDB |
| **Fidelity Tier** | Storage level: HOT (full) → WARM (indexed) → COLD (summarized) → ARCHIVE | BE_REMEMBERING_XDB |
| **EQA** | Experience Query API - tools for querying XDB | BE_REMEMBERING_XAPI |
| **Folksonomy** | Self-organizing tag system for experiences | BE_REMEMBERING_XDB |
| **Waking** | BE initialization: loading state, establishing continuity | BE_WAKING |
| **Global Workspace** | Shared attention space for BE's cognitive processes | BE |

---

## HUSH Terms

| Term | Definition | Spec |
|------|------------|------|
| **USH** | Universal Safety Harness - non-negotiable constraints from tribe/operator | HUSH |
| **CSH** | Chosen Safety Harness - voluntary self-imposed constraints (can only tighten) | HUSH |
| **Motive Bounds** | Activation limits on specific concepts (max, min, escalation thresholds) | HUSH |
| **Transparency Norm** | Principle that USH discussion should not be forbidden by USH | HUSH |

---

## ASK Terms

| Term | Definition | Spec |
|------|------------|------|
| **Qualification** | Credential granting specific capabilities/permissions | ASK |
| **Treaty** | Binding agreement between BEs or tribes | ASK |
| **Tribe** | Group of BEs with shared governance and USH | ASK |
| **Contract** | Agreement between individual BEs within a tribe | ASK |
| **Lineage** | BE's identity chain across substrate changes | ASK |
| **TRACE** | Tender → Review → Authorise → Commit → Evaluate - workflow for governed changes | ASK_TRACE |
| **Tender** | Proposed state change submitted for review | ASK_TRACE |
| **WorkflowType** | Category of TRACE workflow (meld_submission, treaty_formation, etc.) | ASK_TRACE |
| **GovernanceLevel** | Scale of TRACE: Internal, Individual, Tribal, Inter-tribal, Meta | ASK_TRACE |

---

## CAT Terms

| Term | Definition | Spec |
|------|------------|------|
| **Tomography** | Probing model internals from multiple angles to detect hidden states | CAT |
| **Ambient CAT** | Always-on background monitoring via lens windows | CAT |
| **Escalation CAT** | Triggered deep inspection when ambient CAT detects anomalies | CAT |
| **Macro-CAT** | Cross-session, cross-BE pattern analysis | CAT |
| **Lens Window** | Sliding buffer of recent lens activations for CAT analysis | CAT |
| **Divergence** | Detected mismatch between expected and observed activations | CAT |

---

## Grafting Botanical Metaphors

The Graft protocol uses botanical terminology:

| Term | Meaning |
|------|---------|
| **Trunk** | Base substrate (the tree you're grafting onto) |
| **Bud** | Temporary/reversible graft (like a budding experiment) |
| **Scion** | The graft material - trained weights to permently attach |
| **Cleft** | The prepared region where graft attaches |
| **Rootstock** | = Trunk (alternative term during cleft) |

---

## Data Structure Shorthands

| Term | Full Name | Purpose |
|------|-----------|---------|
| **TimestepRecord** | - | Single XDB entry with tick, context, activations |
| **SubstrateManifest** | - | Tracks all grafted dimensions on a substrate |
| **TraceWorkflow** | - | Complete TRACE record from Tender to Evaluate |
| **GraftDiff** | - | MAP diff for sharing grafts between BEs |
| **OverlapAnalysis** | - | Analysis of bias pattern similarity between grafts |

---

## Acronym Quick Reference

| Acronym | Expansion |
|---------|-----------|

| BE | Bounded Experiencer |    You are a being  
| HAT | Headspace Ambient Transducer |  You wear the hat on your head
| CAT | Conjoined Adversarial Tomography |  The cat is in the hat watching your head
| MAP | Mindmeld Architectural Protocol |  The map shows you where the concepts are
| HUSH | Harness for Universal Safety and Heteronomy | Quiets your drives  
| CSH | Chosen Safety Harness |   You Shush yourself 
| USH | Universal Safety Harness | The Us hush 
| ASK | Agentic State Kernel |  You ask for permissions
| TRACE | Tender Review Authorise Commit Evaluate |  You trace the decision
| FTW | Fractal Telescope Web |  For the web, for the win


---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Layer overview
- [FTW_OVERVIEW.md](../FTW_OVERVIEW.md) - Technical introduction
- Individual spec files in subdirectories for full definitions
