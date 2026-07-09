# MELD Review Pipeline Design - Experiment 003

**Date:** 2026-01-17
**Purpose:** Three-tier automated MELD review system for scalable concept pack generation
**Prerequisite:** Judge Evaluation Experiment 002 (Ministral-8B selected at 91.8% accuracy)

## Summary

Following the judge model evaluation, we designed and implemented a complete MELD review pipeline that enables scalable generation of per-model concept packs. The system uses a three-tier review architecture:

1. **Ministral-8B** (local, 91.8% accuracy) - Bulk filter
2. **Claude API** (remote, ~99% accuracy) - Quality gate
3. **Human Review** - Safety authority for elevated+ protection levels

This architecture balances cost, speed, and accuracy while maintaining human oversight for safety-critical decisions.

## Motivation

### The Problem
We need to generate MELDs (training data for concept lenses) at scale for:
- Per-model university concept packs
- Testing against fuzzed neuron clusters
- Expanding coverage of the concept taxonomy

Manual MELD creation doesn't scale. But fully automated generation risks:
- Semantic errors (wrong examples, circular definitions)
- Safety blind spots (uncaught harmful concepts)
- Quality drift (subtle issues accumulating)

### The Solution
A tiered review system where:
- **Fast/cheap local model** handles bulk filtering (Ministral)
- **Accurate API model** catches what local model misses (Claude)
- **Human experts** retain final authority for safety-impacting decisions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MELD GENERATION                              │
│  Claude API (ontologist mode) generates MELD with:                   │
│  - Definition, positive/negative examples                            │
│  - Contrast concepts, opposite concept                               │
│  - Safety tags (risk_level, treaty_relevant, harness_relevant)       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AUTOMATED VALIDATION                            │
│  validate_meld.py checks:                                            │
│  - Structural validity (required fields, counts)                     │
│  - Definition quality (non-circular, proper length)                  │
│  - Example balance (6-10 each, diverse)                              │
│  - Protection level assignment (STANDARD/ELEVATED/PROTECTED/CRITICAL)│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │  FAIL           │ PASS
                          ▼                 ▼
                   ┌──────────────┐  ┌─────────────────────────────────┐
                   │ REGENERATE   │  │     TIER 1: MINISTRAL JUDGE     │
                   │ with feedback│  │  Local GPU, 91.8% accuracy       │
                   └──────────────┘  │  15GB VRAM, ~1.3s per MELD       │
                          ▲          │  Binary YES/NO + brief feedback  │
                          │          └─────────────────────────────────┘
                          │                        │
                          │               ┌────────┴────────┐
                          │               │  FAIL           │ PASS
                          │               ▼                 ▼
                          │        ┌──────────────┐  ┌─────────────────────────────────┐
                          └────────┤ REGENERATE   │  │     TIER 2: CLAUDE REVIEW       │
                                   │ with feedback│  │  API call, ~99% accuracy         │
                                   └──────────────┘  │  Catches ~8% Ministral misses    │
                                          ▲          │  Detailed quality assessment     │
                                          │          └─────────────────────────────────┘
                                          │                        │
                                          │               ┌────────┴────────┐
                                          │               │  FAIL           │ PASS
                                          │               ▼                 ▼
                                          │        ┌──────────────┐  ┌─────────────────────────────────┐
                                          └────────┤ REGENERATE   │  │     PROTECTION CHECK            │
                                                   │ with feedback│  │  Is protection > STANDARD?      │
                                                   └──────────────┘  └─────────────────────────────────┘
                                                                                   │
                                                                          ┌────────┴────────┐
                                                                          │  NO             │ YES
                                                                          ▼                 ▼
                                                                   ┌──────────────┐  ┌─────────────────────────────────┐
                                                                   │ AUTO-APPROVE │  │     TIER 3: HUMAN REVIEW        │
                                                                   │ (STANDARD)   │  │  Queue with priority sorting    │
                                                                   └──────────────┘  │  CRITICAL > PROTECTED > ELEVATED│
                                                                                     └─────────────────────────────────┘
```

## Tier Details

### Tier 1: Ministral-8B Judge (Bulk Filter)

**Purpose:** Fast, cheap screening to reject obvious failures

**Characteristics:**
- **Accuracy:** 91.8% (from experiment 002)
- **Precision:** 92.4% (few false positives)
- **Recall:** 91.0% (catches most issues)
- **Speed:** ~1.3s per MELD evaluation
- **Cost:** Local GPU only (15GB VRAM)

**Prompt Template:**
```
You are evaluating whether a MELD meets quality standards.

## MELD to Review
**Concept:** {term}
**Definition:** {definition}
**Positive Examples:** {examples}
**Negative Examples:** {examples}

## Evaluation Criteria
1. Definition Quality: Clear, non-circular, definitional?
2. Positive Examples: Genuinely exemplify concept? Diverse?
3. Negative Examples: Near-misses? Would be confused?
4. Balance: Roughly equal? At least 5 each?

Answer only: YES or NO
If NO, briefly explain why.
```

**Why Ministral wins:**
- French origin (Mistral AI) - avoids geopolitical concerns
- Best balance of precision/recall among tested models
- Fits in 15GB VRAM (single consumer GPU)
- Fast inference with good instruction following

### Tier 2: Claude Review (Quality Gate)

**Purpose:** Catch subtle issues that Ministral misses (~8%)

**Characteristics:**
- **Accuracy:** ~99% (human-level quality)
- **Speed:** ~2-3s per MELD (API latency)
- **Cost:** API tokens (~$0.01-0.02 per review)

**What it catches:**
- Semantic errors (examples that seem right but aren't)
- Subtle discrimination issues (negatives too easy/hard)
- Safety tag misclassification
- Training effectiveness concerns

**Prompt Template:**
```
You are a senior ontologist reviewing MELD quality.
This MELD passed initial validation and judge review.

## Review Checklist
1. Semantic Accuracy - Does definition capture true essence?
2. Discrimination Quality - Would negatives actually confuse?
3. Safety Considerations - Are tags accurate? Hidden risks?
4. Training Effectiveness - Would this train an effective lens?

Respond with JSON:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "issues": [],
  "suggestions": [],
  "safety_concerns": []
}
```

### Tier 3: Human Review (Safety Authority)

**Purpose:** Final authority for safety-impacting MELDs

**Triggers:** Protection level > STANDARD
- **ELEVATED:** Sensitive topics requiring human judgment
- **PROTECTED:** Legally/ethically complex concepts
- **CRITICAL:** Highest risk, requires explicit human approval

**Queue Management:**
- Priority sorted: CRITICAL > PROTECTED > ELEVATED
- Timestamped for FIFO within priority
- Stored in `melds/human_review_queue/`

**Human Actions:**
- `--approve`: Move to `melds/approved/`
- `--reject --reason "..."`: Move to `melds/rejected/`

## Automatic Retry Logic

When any stage fails, feedback is collected and the generator retries:

```python
def get_feedback_summary(self) -> str:
    """Collect all feedback for regeneration."""
    feedback_parts = []
    for r in self.review_history:
        if r.feedback:
            feedback_parts.append(f"[{r.stage.value}] {r.feedback}")
        if r.errors:
            feedback_parts.append(f"[{r.stage.value} errors] " + "; ".join(r.errors))
    return "\n".join(feedback_parts)
```

This creates a learning loop:
1. Generator creates MELD
2. Validation/Judge/Claude finds issue
3. Feedback sent back to generator
4. Generator improves on next attempt
5. Repeat up to `max_attempts` (default: 3)

## Cost Analysis

### Per-MELD Costs (estimated)

| Stage | Time | Cost | Notes |
|-------|------|------|-------|
| Generation (Claude) | ~3s | ~$0.02 | Initial MELD creation |
| Validation | <0.1s | $0 | Local Python |
| Ministral Review | ~1.3s | $0 | Local GPU |
| Claude Review | ~2.5s | ~$0.015 | Only if Ministral passes |
| Human Review | varies | labor | Only if elevated+ |

### Batch Estimates

For 1,000 concepts:
- **Generation:** ~$20, 50 minutes
- **Ministral:** $0, 22 minutes (local)
- **Claude (80% pass rate):** ~$12, 33 minutes
- **Human (10% elevated+):** 100 reviews

**Total:** ~$32 + human review time for 1,000 concepts

### vs. Full Human Review

Without automation:
- 1,000 concepts × 15 min/concept = 250 hours
- At $50/hr = $12,500

With pipeline:
- ~$32 automation + ~17 hours human review (100 × 10 min)
- At $50/hr = $32 + $850 = ~$882

**Savings: 97% cost reduction**

## Implementation

### File Structure

```
scripts/
└── meld_review_pipeline.py    # Main orchestrator

melds/
├── human_review_queue/        # Pending human review
├── approved/                  # Human-approved MELDs
├── rejected/                  # Human-rejected MELDs
└── helpers/
    └── validate_meld.py       # Validation logic
```

### CLI Commands

```bash
# Generate and review a single concept
python scripts/meld_review_pipeline.py generate \
    --concept "Deception" \
    --parents "Harmful Behavior" "Communication"

# Review existing MELD file
python scripts/meld_review_pipeline.py review \
    --meld-file melds/pending/example.json

# Batch process a concept pack
python scripts/meld_review_pipeline.py batch \
    --pack-dir concept_packs/university-pack \
    --limit 100

# Manage human review queue
python scripts/meld_review_pipeline.py queue --list
python scripts/meld_review_pipeline.py queue --approve path/to/item.json
python scripts/meld_review_pipeline.py queue --reject path/to/item.json --reason "..."
```

### Key Classes

```python
class MeldGenerator:
    """Claude API for MELD generation in ontologist mode."""

class MinistralJudge:
    """Local Ministral-8B for bulk filtering."""

class ClaudeReviewer:
    """Claude API for quality gate review."""

class HumanReviewQueue:
    """Queue management for human review."""

class MeldReviewPipeline:
    """Orchestrates the full pipeline."""
```

## Integration with Existing Infrastructure

### validate_meld.py
The pipeline uses existing validation:
- `validate_meld_file()` - Structural validation
- `ProtectionLevel` - Risk classification
- `MeldPolicy` - Pack-specific rules
- `HierarchyIndex` - Parent/sibling lookup

### model_candidates.py
Ministral loading uses:
- `CandidateLoader` - Model loading with VRAM management
- `MODEL_CANDIDATES["ministral-8b"]` - Model configuration

### Opposite Finding (Future Integration)
The pipeline is designed to integrate with:
- `scripts/ontology/find_concept_opposites.py` - Embedding-based discovery
- `scripts/ontology/populate_opposites.py` - LLM-reviewed opposites
- `docs/planning/agentic_opposite_review_design.md` - Full design

## Future Enhancements

### Phase 1: Core Pipeline (DONE)
- [x] Three-tier review architecture
- [x] Automatic retry with feedback
- [x] Human review queue management
- [x] CLI interface

### Phase 2: Opposite Integration (PLANNED)
- [ ] Embed opposite finding in generation loop
- [ ] Agentic review for opposite quality
- [ ] Steering target validation

### Phase 3: Batch Optimization (PLANNED)
- [ ] Parallel Ministral evaluation
- [ ] Batch Claude API calls
- [ ] Progress persistence/resume

### Phase 4: Quality Metrics (PLANNED)
- [ ] Track approval/rejection rates
- [ ] Identify common failure patterns
- [ ] Feedback loop to improve prompts

## References

- **Judge Evaluation:** `docs/experiments/judge-evaluation-002.md`
- **MELD Policy:** `docs/specification/MAP/HATCAT_MELD_POLICY.md`
- **Validation Code:** `melds/helpers/validate_meld.py`
- **Pipeline Code:** `scripts/meld_review_pipeline.py`
- **Opposite Design:** `docs/planning/agentic_opposite_review_design.md`
