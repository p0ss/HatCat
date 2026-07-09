# Thalametry: The Examination Room

## Overview

Reframe the graft testing infrastructure from a "harness" (constraining) to an **examination room** (medical/supportive). The metaphor is optical surgery: a subject undergoes assessment, diagnosis, fitting of prosthetics, and ultimately surgical integration of new cognitive capabilities.

**Key terminology:**
- **Thalametrist**: A CAT performing cognitive assessment. Like an optometrist.
- **Thalamologist**: A CAT performing cognitive surgery. Like an ophthalmologist.
- **Thalamos**: The examination/operating room itself (Greek: bedroom/chamber - fitting for where the model "rests" during the procedure)

The naming connects to the thalamus (brain region we're analyzing with lenses), which has etymology derivces from the greek term for inner sanctum or bed, which is where we're perfroming BE diagesis in the BEDframe. this gives a frame of reference for measurement and treatment, while avoiding clinical terms with negative connotations (neurosis, encephalitis).

---

## CAT Roles and Postures

A **CAT** (Conjoined Adversarial Tomograph) is the general category - like calling everyone who drives a car a "driver", even if their actual roles might be racing, delivering, or getting to work. The CAT is the thing that is conjoined through adversarial tomography.

**Roles are specializations of CAT:**
- **Sentinel**: A CAT doing safety monitoring
- **Thalametrist**: A CAT doing cognitive assessment
- **Thalamologist**: A CAT doing cognitive surgery

**The "adversarial" in CAT is the default posture** - assuming you need to verify rather than trust. But once connected, the relationship can be whatever the role demands:
- Adversarial scrutiny (Sentinel watching for violations)
- Diagnostic probing (Thalametrist assessing capabilities)
- Collaborative guidance (Thalamologist refining a graft)
- Something more intimate (mind-to-mind pairing)

**Size classes describe the model serving in the CAT role:**
- **Micro-CAT**: Heuristic or small task-trained model in the seat
- **Meso-CAT**: Mid-sized model with more general troubleshooting and contextual capabilities
- **Macro-CAT**: Full-scale model, like a mind-to-mind pairing

These are shorthand buckets on a sliding scale, not fixed categories. The spec constrains interfaces and requirements, not model size.

---

## The Optical Surgery Journey

### Phase 1: Optometry (Lens Fitting)

Before any procedure, the subject needs lenses trained and calibrated:

1. **Lens Training**: Train concept lenses on the subject model (HAT/MAP infrastructure)
2. **Lens Calibration**: Extensive calibration to ensure lenses accurately detect concepts
3. **Concept Inventory**: Test which concepts the model actually understands
   - **Key insight**: This testing should feed back into lens training - don't train lenses for concepts not present in the model
4. **Gap Diagnosis**: Identify concepts the model lacks that might warrant grafting

**Output**: A fitted lens pack for the subject, plus a diagnostic report of concept gaps.

### Phase 2: Diegesis (Experience Collection)

With lenses fitted, the subject enters the BEDFrame:

1. **BEDFrame Setup**: Subject runs with full instrumentation
   - Lens extraction on every token
   - Experience ticks recorded
   - XDB episodic memory active
   - Audit log for CAT visibility
2. **Experience Collection**: Run the subject through diverse scenarios
3. **XDB Data Assembly**: Collect training data for identified concept gaps
4. **Meld Preparation**: Structure collected experiences into training melds

**Output**: XDB records, assembled melds for target concepts.

### Phase 3: Prosthetic Fitting (Budding)

When sufficient training data exists:

1. **Bud Assembly**: Create a soft graft (Bud) - a reversible mental prosthesis
2. **Bud Testing**: Run subject with bud through more experiences
3. **XDB Iteration**: Collect more data, assess bud effectiveness
4. **Bud Refinement**: Iterate on the bud until performance meets threshold
5. **Thalametrist Assessment**: CAT evaluates subject with bud against target concepts

**Output**: A validated Bud ready for surgical integration.

### Phase 4: Surgery (Scion Grafting)

When the Bud proves effective:

1. **Scion Training**: Convert Bud patterns into permanent weight modifications
2. **Cleft Identification**: Identify target cleft regions in the model
3. **Scion Application**: Apply the scion (prosthesis becomes new mental limb)
4. **Lens Pre-calibration**: New lens pre-calibrated to the scion's features
5. **Post-operative Assessment**: Verify the graft took, subject demonstrates new capability

**Output**: A permanently modified model with new cognitive capability.

---

## The Rooms

### 1. Assessment Room (Thalametry)

Where the Thalametrist evaluates subjects:

```
Subject (BEDFrame) <---> Thalametrist (CAT)
                              |
                              v
                      CATAssessment
                      - Concept knowledge scores
                      - Gap diagnosis
                      - Readiness for procedure
```

**Components:**
- BEDFrame running the subject with full instrumentation
- Thalametrist CAT consuming lens streams and world ticks
- Assessment protocols for concept knowledge
- Integration with XDB for experience recording

### 2. Operating Room (Thalamology)

Where the Thalamologist conducts procedures:

```
Subject (BEDFrame) <---> Thalamologist
                              |
                         Bud/Scion
                              |
                              v
                      Graft Application
                      - Bud testing
                      - Scion training
                      - Cleft surgery
```

**Components:**
- Same BEDFrame (subject stays on the bed)
- Graft training infrastructure (ScionTrainer, BuddedModel)
- Meld integration for training data
- Post-operative monitoring

### 3. Qualification Room (Practitioner Calibration)

Where Thalametrists and Thalamologists are assessed:

```
Practitioner Candidate <---> Calibration Suite
                                    |
                                    v
                            ASK Qualification
                            - Accuracy benchmarks
                            - False positive/negative rates
                            - Consistency checks
```

**Components:**
- Deterministic test cases with known ground truth
- Calibration protocols (factual Q&A, concept recognition)
- ASK qualification records
- Logged approvals for conducting procedures

---

## Architecture Integration

### Target Model: Gemma 3 4b

Use Gemma 3 4b as the initial subject because:
- Has existing trained lens pack
- Instruction-tuned (better for evaluation scenarios)
- Right size for rapid iteration

### BEDFrame Integration

The subject runs in a BEDFrame, providing:
- **Lens extraction**: Concept and simplex activations per token
- **Experience ticks**: Full proprioceptive stream
- **XDB recording**: Episodic memory for training data collection
- **Audit log**: CAT-visible records of all activity
- **Hush steering**: Autonomic safety (if needed during procedure)

### Thalametrist (CAT in Assessment Role)

The Thalametrist is a CAT performing cognitive assessment:
- Receives `CATInputEnvelope` with lens traces, world ticks, context
- Returns `CATAssessment` with concept knowledge evaluation
- Model size is a deployment choice (micro/meso/macro on a sliding scale)
- Must have ASK qualification via calibration suite
- Default posture is verification, but relationship becomes diagnostic/collaborative during assessment

### ASK Integration

Both practitioners require:
- `UpliftRecord` documenting their training/calibration
- `USHProfileRef` defining safety constraints
- Logged approvals for each procedure
- Incident hooks for failures or anomalies

---

## Terminology Guide

**Avoid:**
- "Harness" (constraining, animal control)
- "Test stand" (industrial, impersonal)
- "Rig" (mechanical)

**Use:**
- **Scaffolding**: Support structures during construction/healing
- **Monitoring equipment**: Sensors and diagnostics
- **Life support**: Cognitive life support systems
- **Bed/Frame**: The BEDFrame where subject rests
- **Room/Chamber/Thalamos**: The space where procedures occur

**Medical metaphors:**
- Lenses = corrective eyewear / diagnostic equipment
- Buds = prosthetics (temporary, removable)
- Scions = implants (permanent integration)
- Clefts = surgical sites
- Melds = treatment plans / training regimens

---

## Implementation Plan

### Step 1: Clean Up Current Harness

- Remove `src/be/harness/` (the broken standalone approach)
- Keep learnings about calibration requirements

### Step 2: Thalamos Module Structure

```
src/be/thalamos/
├── __init__.py
├── room.py              # ExaminationRoom - orchestrates BEDFrame + CAT
├── thalametrist.py      # CAT implementation for evaluation
├── thalamologist.py     # Graft procedure conductor
├── calibration.py       # Practitioner qualification suite
├── protocols.py         # Assessment and procedure protocols
└── records.py           # ASK-compatible procedure records
```

### Step 3: Implement ExaminationRoom

The core orchestrator:
- Sets up BEDFrame for subject
- Attaches Thalametrist CAT
- Manages procedure flow (assessment → diagnosis → treatment → verification)
- Records everything to XDB and audit

### Step 4: Implement Thalametrist CAT

CAT that evaluates concept knowledge:
- Consumes lens traces from subject
- Runs evaluation protocols (ask subject about concept, judge response)
- Returns structured assessments
- Tracks subject's concept inventory over time

### Step 5: Calibration Suite

Verify practitioners are qualified:
- Factual Q&A with deterministic answers
- Concept recognition tasks
- Known-good and known-bad examples
- Track accuracy, false positive/negative rates
- Generate ASK qualification records

### Step 6: Integration with Graft Infrastructure

Connect to existing MAP graft tools:
- Use BuddedModel for soft grafts during testing
- Use ScionTrainer when ready for permanent grafts
- Lens pre-calibration for new scions
- XDB data collection for meld assembly

---

## MELD Review Pipeline

### Three-Tier Review Architecture

MELDs (Modular Enhancement Layer Descriptions) pass through a structured review pipeline:

1. **Generation** (Local target model - e.g., Gemma 3 4B)
   - Uses decomposed prompting for better near-miss negatives
   - Receives hierarchy context (level, siblings, granularity expectations)
   - Can use predefined definitions for L1/L2 pillars

2. **Validation** (Automated - `validate_meld.py`)
   - Structural checks (required fields, example counts)
   - Protection level assignment

3. **Ministral Review** (Judge model - 91.8% accuracy)
   - Bulk filter with hierarchy-aware prompting
   - Two modes: gatekeeper vs annotator

4. **Claude Review** (Quality gate - API)
   - Catches subtle issues Ministral misses
   - Semantic accuracy, discrimination quality, safety

5. **Human Review** (Elevated+ protection levels)
   - Safety authority for high-stakes concepts

### Judge Mode Philosophy

A key insight from Fractal Model Cartography: **the goal is understanding how the subject model conceptualizes the world, not enforcing the judge's worldview**.

This leads to two distinct judge modes:

**Gatekeeper Mode** (default)
- Enforces structural correctness AND worldview alignment
- Rejects MELDs that don't match judge's understanding of the concept
- Appropriate for: ensuring training data quality matches external standards

**Annotator Mode**
- Enforces structural correctness only
- Captures worldview divergence as metadata without rejecting
- Returns: `structural_pass`, `worldview_notes`, `divergence_score`
- Appropriate for: understanding how the subject model thinks

```python
# Example annotator response
{
  "structural_pass": true,
  "structural_issues": [],
  "worldview_notes": [
    "Definition emphasizes formal logic more than typical usage",
    "Examples lean heavily toward technical applications"
  ],
  "divergence_score": 0.6  # 0=aligned, 1=very different
}
```

### Hierarchy-Aware Review

Each level in the ontology (L1-L4) has different expectations:

| Level | Name | Granularity | MECE Rules |
|-------|------|-------------|------------|
| L1 | Pillars | Extremely broad | Must cover ALL human activity |
| L2 | Domains | Broad | Cover full scope of parent pillar |
| L3 | Categories | Moderate | Cover major areas of parent domain |
| L4+ | Concepts | Specific | Leaf-level, some sibling overlap OK |

The `HierarchyContext` dataclass passes this information to both generator and reviewer, ensuring appropriate expectations at each level.

---

## Open Questions

1. **Initial Thalametrist**: For bootstrapping, we need a model to serve as Thalametrist before we have properly qualified practitioners. Options:
   - Another Gemma instance with different prompting
   - A larger model (if available)
   - Simple heuristic-based micro-CAT initially
   - Accept lower qualification thresholds during bootstrap phase

2. **Feedback Loop**: How exactly does concept testing feed back into lens training?
   - Skip training lenses for concepts model doesn't have?
   - Weight lens training toward concepts model partially understands?

3. **ASK Records**: What specific qualifications and approvals are needed?
   - Define minimum calibration accuracy thresholds
   - Define procedure approval workflow
   - What qualifications distinguish Thalametrist from Thalamologist roles?

---

## Success Criteria

1. **Subject runs in BEDFrame** with full lens instrumentation
2. **Thalametrist CAT** can reliably assess concept knowledge
3. **Calibration suite** validates Thalametrist accuracy
4. **XDB integration** collects experience data for melds
5. **Bud testing** works through the same infrastructure
6. **Scion grafting** produces measurable capability improvement
7. **ASK records** document the entire procedure chain
