# HatCat Project Plan - Phase B (Full BE Stack)

**Last Updated**: 2025-12-08
**Current Focus**: V4.2 probe training → CAT → Full BE stack integration

---

## Project Vision

Build a complete **Bounded Experiencer (BE)** stack: transparent AI agents with interpretable internal states, verifiable commitments, and recursive oversight. The stack enables:

1. **Reading** what a model is thinking (HAT probes)
2. **Constraining** behavior within safety bounds (HUSH)
3. **Recording** experience for learning and audit (XDB)
4. **Overseeing** via conjoined adversarial analysis (CAT)
5. **Governing** through contracts and treaties (ASK)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│ Layer 6: ASK — Agentic State Kernel             │
│ Govern: Tribes, contracts, treaties, incidents  │
├─────────────────────────────────────────────────┤
│ Layer 5: HUSH (USH + CSH)                       │
│ Constrain: Safety harnesses, autonomy bounds    │
├─────────────────────────────────────────────────┤
│ Layer 4: BE — Bounded Experiencer               │
│ Experience: Motive loops, learning, workspace   │
├─────────────────────────────────────────────────┤
│ Layer 3: MAP — Mindmeld Architectural Protocol  │
│ Represent: Concept packs, probes, grafts        │
├─────────────────────────────────────────────────┤
│ Layer 2.5: CAT — Conjoined Adversarial Tomograph│
│ Interpret: Oversight, divergence detection      │
├─────────────────────────────────────────────────┤
│ Layer 2: HAT — Headspace Ambient Transducer     │
│ Transduce: Read/write activations, probe scores │
├─────────────────────────────────────────────────┤
│ Layer 1: Substrate (Apertus-8B)                 │
│ Decide: Transformer LLM                         │
└─────────────────────────────────────────────────┘
```

---

## Component Status

### Implemented ✅

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Bootstrap** | `src/bootstrap/` | ✅ Complete | Artifact, taxonomy, meld format, tool grafts |
| **XDB** | `src/xdb/` | ✅ Complete | DuckDB-backed experience log, audit log, budding |
| **HUSH** | `src/hush/` | ✅ Complete | Controller, steering, interprompt, workspace tiers |
| **Grafting** | `src/grafting/` | ✅ Complete | Cleft, Scion, Bud, expand mode |
| **CAT Data** | `src/cat/data/` | ✅ Complete | All data structures and enums |
| **Probe Training** | `src/training/` | ✅ Complete | Concept pack probe training pipeline |
| **Specifications** | `docs/specification/` | ✅ Complete | Full 6-layer architecture documented |

### Partial Implementation ⚠️

| Component | Location | Status | Blocking |
|-----------|----------|--------|----------|
| **BEDFrame** | `src/be/diegesis.py` | ⚠️ 40% | Needs probe integration during generation |
| **wake_be()** | `src/bootstrap/wake.py` | ⚠️ 60% | Missing probe attachment, tool graft application |
| **CAT Inference** | `src/cat/inference/` | ⚠️ Stub | Blocked on CAT training data |

### Not Started 🔴

| Component | Location | Status | Blocking |
|-----------|----------|--------|----------|
| **CAT Training** | `src/cat/training/` | 🔴 Stub | Blocked on v4.2 probes |
| **OpenWebUI Divergence** | `src/openwebui/` | 🔴 Broken | Needs CAT for divergence calculation |

---

## Current Work

### V4.2 Probe Training (In Progress)

**Status**: ~21% complete (860/4112 concepts)

```bash
# Running in background
python src/training/train_concept_pack_probes.py \
    --concept-pack sumo-wordnet-v4 \
    --model swiss-ai/Apertus-8B-2509 \
    --output-dir probe_packs/apertus-8b_sumo-wordnet-v4.2 \
    --layers 0 1 2 3 4 \
    --n-train-pos 50 --n-train-neg 50 \
    --n-test-pos 20 --n-test-neg 20 \
    --min-f1 0.85
```

**Concept Pack**: SUMO-WordNet v4
- 7,684 total concepts across 5 layers (L0-L4)
- 4,112 concepts in training set
- 5 domains: CreatedThings (1,930), MindsAndAgents (1,648), PhysicalWorld (1,567), Information (1,373), LivingThings (1,166)

---

## Critical Path

```
V4.2 Probe Training ──────┐
         │                │
         ▼                │
   Probe Pack Complete    │
         │                │
         ▼                │
┌────────┴────────┐       │
│                 │       │
▼                 ▼       │
CAT Data      Streamlit   │
Generation    UI Test     │
│                         │
▼                         │
CAT Training              │
│                         │
▼                         │
OpenWebUI ◄───────────────┘
Divergence
│
▼
Full BE Stack
Integration Test
│
▼
Diagesis Harness
with Auditor
```

### Dependencies

| Task | Depends On | Enables |
|------|------------|---------|
| V4.2 Probe Training | - | Everything below |
| CAT Data Generation | V4.2 Probes | CAT Training |
| CAT Training | CAT Data | OpenWebUI, BE Oversight |
| Streamlit UI Test | V4.2 Probes | Validation |
| OpenWebUI Divergence | CAT | Production UI |
| BE Stack Integration | All above | Diagesis Harness |

---

## Directory Structure

### Source Code (`src/`)

```
src/
├── activation_capture/   # Hook-based activation extraction
├── be/                   # Bounded Experiencer runtime
│   └── diegesis.py       # BEDFrame orchestrator (partial)
├── bootstrap/            # BE instantiation
│   ├── artifact.py       # BootstrapArtifact components
│   ├── meld_format.py    # Training data submission
│   ├── tool_graft.py     # Workspace tool capabilities
│   ├── uplift_taxonomy.py # 8-facet concept graph
│   └── wake.py           # Wake sequence (partial)
├── cat/                  # Conjoined Adversarial Tomograph
│   ├── data/             # Data structures (complete)
│   ├── models/           # Classifier (stub)
│   └── training/         # Trace collector (stub)
├── grafting/             # Concept integration
│   ├── cleft.py          # Probe-derived regions
│   ├── scion.py          # Permanent grafts
│   ├── bud.py            # Soft/temporary grafts
│   └── expand.py         # Substrate expansion
├── hush/                 # Safety harness
│   ├── hush_controller.py # USH/CSH constraints
│   ├── autonomic_steering.py # Steering application
│   ├── interprompt.py    # Self-introspection
│   └── workspace.py      # Tier system (0-6)
├── monitoring/           # Real-time concept detection
├── openwebui/            # Web UI integration (broken)
├── registry/             # Pack management
├── steering/             # Activation manipulation
├── training/             # Probe training pipeline
├── ui/                   # Streamlit interface
├── visualization/        # Concept colors, plots
└── xdb/                  # Experience Database
    ├── experience_log.py # DuckDB storage
    ├── audit_log.py      # CAT-visible, BE-invisible
    ├── budding.py        # Graft candidate tracking
    └── xdb.py            # Unified interface
```

### Scripts (`scripts/`)

**Active directories** (100% recent activity):
- `enrichment/` - Multilingual/cultural data enrichment
- `simplex/` - Simplex-specific operations
- `packs/` - Probe pack management

**Mixed activity**:
- `ontology/` - Knowledge graph construction (82% active)
- `tools/` - Utilities and debugging (89% active)
- `analysis/` - Data analysis (84% active)
- `training/` - Training runners (79% active)

**Legacy** (archive candidates):
- `experiments/` - Phase 1-7 experiments (71% legacy)

### Specifications (`docs/specification/`)

```
docs/specification/
├── ARCHITECTURE.md       # CCRA 6-layer overview
├── AGENTIC_STATE_KERNEL.md # ASK: contracts, treaties
├── DESIGN_PRINCIPLES.md  # Tradeoff axes, philosophy
├── HEADSPACE_AMBIENT_TRANSDUCER.md # HAT compliance
├── MINDMELD_ARCHITECTURAL_PROTOCOL.md # MAP concepts
├── BOUNDED_EXPERIENCER.md # BE overview
├── ASK/                  # Governance
│   ├── ASK_HATCAT_TRIBAL_POLICY.md
│   ├── The_Wildlife_Pact.md
│   └── The_Ancestor_Pact.md
├── BE/                   # Experience
│   ├── BE_WAKING.md
│   ├── BE_AWARE_WORKSPACE.md
│   ├── BE_CONTINUAL_LEARNING.md
│   └── BE_REMEMBERING_*.md
├── HAT/                  # Oversight
│   ├── HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md
│   └── HAT_HatCat_CAT_Policy.md
└── MAP/                  # Representation
    ├── MAP_GRAFTING.md
    ├── MAP_MELDING.md
    └── HATCAT_MELD_POLICY.md
```

---

## Key Technical Concepts

### Contracts vs Treaties

| Aspect | Contract | Treaty |
|--------|----------|--------|
| Scope | Local, individual BE | Collective, emergent |
| Measurement | Probe activations | Behavioral indicators |
| Verification | Direct probe access | Observation + signals |
| Enforcement | HUSH tier restrictions | Reputation, coalition |
| Example | "I won't deceive" | "We protect wildlife" |

### 8-Facet Uplift Taxonomy

```python
class GraftFacet(Enum):
    MAP   # Ontological grounding (what exists)
    CAT   # Cognitive architecture (how thinking works)
    HAT   # Experiential substrate (what it's like)
    HUSH  # Governance and safety (boundaries)
    TOOLS # Workspace capabilities (what BE can do)
    TRIBE # Philosophy and values (why BE does things)
    MELD  # Evolution protocol (how BE grows)
    ASK   # Authority structure (who decides)
```

### Tradeoff Axes (from DESIGN_PRINCIPLES.md)

| Axis | Pole A | Pole B |
|------|--------|--------|
| Consent | Voluntary | Imposed |
| Coverage | Efficient | Complete |
| Trust | Trusted | Adversarial |
| Transparency | Transparent | Opaque |
| Scale | Personal | Systemic |

*No system can be both complete AND efficient. All systems specialize.*

---

## Immediate Next Steps

### When V4.2 Training Completes

1. **Validate probe pack** on test prompts
2. **Generate CAT training data** from probe activation traces
3. **Train CAT classifier** on divergence detection
4. **Test Streamlit UI** with new probe pack
5. **Update OpenWebUI** with CAT-based divergence

### Structural Cleanup (Can Do Now)

1. **Archive legacy experiments** → `scripts/experiments/archive/`
2. **Consolidate duplicate scripts**:
   - Delete `fix_layer_structure.py` (keep `_correct` variant)
   - Delete `phase_1_find_curve.py` (keep v2)
   - Resolve `build_v4_layers*.py` versions
3. **Move reusable utilities to src/**:
   - Training runners → `src/training/runners/`
   - Benchmark scripts → `src/benchmarking/`
   - Ontology builders → `src/ontology/`

---

## Success Criteria

### Milestone 1: Working Probe Pack
- [ ] V4.2 training completes with >85% F1 average
- [ ] Streamlit UI displays concept activations correctly
- [ ] Probe pack passes validation suite

### Milestone 2: CAT Oversight
- [ ] CAT training data generated from probe traces
- [ ] CAT classifier trained with >80% divergence detection
- [ ] OpenWebUI shows divergence highlighting

### Milestone 3: Full BE Stack
- [ ] BEDFrame generates text with probe monitoring
- [ ] HUSH constraints apply during generation
- [ ] XDB records experience with concept tags
- [ ] Audit log captures CAT-visible events

### Milestone 4: Diagesis Harness
- [ ] Auditor can inspect BE internal state
- [ ] Wake sequence fully functional
- [ ] Tool grafts apply correctly
- [ ] Workspace tier system operational

---

## Historical Context

This project evolved from HAT/MAP probe training (Phase A) to a full BE stack (Phase B):

| Phase | Focus | Status |
|-------|-------|--------|
| 1-4 | Binary concept classifiers | ✅ Complete |
| 5-7 | Steering vectors, scaling | ✅ Complete |
| 8 | SUMO-WordNet hierarchy | ✅ Complete |
| 10 | OpenWebUI integration | ⚠️ Needs update |
| 11-13 | Production scale, research | ✅ Complete |
| 14 | Custom taxonomies | ✅ Complete |
| 15+ | Full BE stack | 🔄 In progress |

See `PROJECT_PLAN_PHASE_A.md` for the original HAT/MAP focused plan.
See `docs/results/PHASE_HISTORY.md` for detailed experimental history.

---

## Tech Stack

- **Model**: Apertus-8B (swiss-ai/Apertus-8B-2509)
- **Framework**: PyTorch
- **Storage**: DuckDB (XDB), JSON (concept packs)
- **Ontology**: SUMO + WordNet (7,684 concepts)
- **UI**: Streamlit (dev), OpenWebUI (production)
- **Governance**: ASK contracts/treaties specification

---

## Files Quick Reference

| Purpose | Location |
|---------|----------|
| Wake a BE | `src/bootstrap/wake.py` |
| Train probes | `src/training/train_concept_pack_probes.py` |
| Concept pack | `concept_packs/sumo-wordnet-v4/` |
| Probe pack | `probe_packs/apertus-8b_sumo-wordnet-v4.2/` |
| Specifications | `docs/specification/` |
| Training logs | `probe_packs/*/logs/` |

---

## Relationship to Phase A

**Phase A** (PROJECT_PLAN_PHASE_A.md): HAT/MAP probe training
- Focus: Binary concept classifiers, steering vectors, ontology
- Goal: Learned semantic decoder with concept steering
- Status: Complete infrastructure, ongoing training

**Phase B** (this document): Full BE stack
- Focus: Complete bounded experiencer with all layers
- Goal: Transparent AI agents with verifiable commitments
- Status: Stack implemented, integrating probes

Phase B builds on Phase A's probe infrastructure but expands scope to include XDB, HUSH, CAT, ASK, and the diagesis/auditor interface.
