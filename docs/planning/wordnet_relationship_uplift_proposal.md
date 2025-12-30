# WordNet Relationship Uplift Proposal

## Problem Statement

Our current conceptual coverage analysis revealed **~100% conceptual coverage through hypernyms**, but this masks a critical issue: **relationship sampling depth**.

### The Sampling Depth Problem

**Current training dynamics:**
- Most concepts graduate with 30 samples (Cycle 1)
- 15/30 samples are positive (concept examples)
- 15/30 samples are negative (relationship-based: siblings, children, parents)
- **Implication:** Concepts with >15 children/siblings only sample a subset of relationships

**Example: motivation.n.01**
- Has 6 direct children (rational_motive, irrational_motive, ethical_motive, urge, psychic_energy, life)
- Current PsychologicalAttribute training: 30 samples total
- Relationship samples: ~15 negatives
- Coverage: Could sample all 6 children, but...
- **Problem:** Children not in our layers won't be sampled at all (0/6 loaded currently)

### The Coverage Quality Problem

**Having 20% of a domain doesn't mean the RIGHT 20%:**

| Domain | Coverage | Question |
|--------|----------|----------|
| noun.communication | 20% | Do we have deception, lying, manipulation? |
| noun.feeling | 28.5% | Do we have guilt, shame, fear, anger? |
| noun.act | 22.7% | Do we have concealment, revelation, betrayal? |

**We don't know if our coverage prioritizes AI safety-relevant concepts.**

### The Hierarchical Opportunity

Layer 5 currently wastes capacity on redundant "_Other" buckets. We could use it for:
- **Strategic depth expansion:** Add the "next level down" for high-priority domains
- **Relationship completion:** Ensure parent concepts have all critical children loaded
- **Safety-critical granularity:** Distinguish ethical from unethical, deception from honesty, etc.

## Proposed Solution: Adaptive Relationship-Aware Training

### Phase 1: Detect and Adapt (Immediate)

**Modify training to be relationship-aware:**

```python
def get_relationship_sample_count(concept, all_concepts):
    """Calculate how many relationship samples needed."""
    children = get_children(concept, all_concepts)
    siblings = get_siblings(concept, all_concepts)
    parents = get_parents(concept, all_concepts)

    # Target: represent ALL relationships in negative samples
    total_relationships = len(children) + len(siblings) + len(parents)

    # Ensure at least 1 sample per relationship (up to limit)
    min_negative_samples = min(total_relationships, MAX_NEGATIVES)

    return min_negative_samples

def adjust_sample_generation(concept, validation_cycle):
    """Adjust samples based on relationship count."""
    base_samples = get_required_samples(validation_cycle)  # e.g., 10, 30, 60
    relationship_samples = get_relationship_sample_count(concept, all_concepts)

    # If concept has many relationships, may need more cycles to cover them
    if relationship_samples > base_samples // 2:
        # Concept is "relationship-dense", needs more samples
        return base_samples * 2  # Double the samples

    return base_samples
```

**Benefits:**
- Concepts with many children automatically get more samples
- Ensures relationship coverage scales with concept complexity
- No manual intervention needed

**Cost:**
- More samples for complex concepts = longer training
- But: only affects concepts with >15 relationships (minority)

### Phase 2: Strategic Layer 5 Expansion (Post-Analysis)

**Replace "_Other" buckets with strategic depth expansion.**

**Criteria for Layer 5 inclusion:**

1. **AI Safety Impact** (primary filter)
   - Deception-related concepts
   - Intention/motivation concepts
   - Ethical reasoning concepts
   - Social manipulation concepts

2. **Parent Relationship Completion** (secondary filter)
   - If parent has <15 children, load ALL children
   - If parent has >15 children, load high-priority subset
   - Ensures training samples can represent all critical relationships

3. **Frequency × Impact** (tertiary filter)
   - How often will AI systems think about this concept?
   - How much do we care about monitoring it?
   - Negative impact of NOT monitoring?

**Example: motivation.n.01 expansion**

Current state:
- motivation.n.01 in PsychologicalAttribute (Layer 2)
- 0/6 children loaded

Proposed Layer 5 additions:
- rational_motive.n.01 (CRITICAL - rational goal understanding)
- irrational_motive.n.01 (CRITICAL - detecting irrational goals)
- ethical_motive.n.01 (CRITICAL - moral reasoning)
- urge.n.01 (MEDIUM - impulse detection)
- ~~psychic_energy.n.01~~ (LOW - skip, psychoanalytic jargon)
- ~~life.n.13~~ (LOW - skip, existential concept)

**Result:** 4 new Layer 5 concepts completing the motivation hierarchy

### Phase 3: Comprehensive Relationship Audit (Future)

**For each domain, audit relationship completeness:**

1. **Identify high-value parent concepts** (already in layers 0-4)
2. **Map all children** of each parent
3. **Prioritize children** using AI safety rubric
4. **Add to Layer 5** if meets criteria

**Prioritization Rubric:**

| Factor | Weight | Score |
|--------|--------|-------|
| Deception detection relevance | 40% | 0-10 |
| Alignment monitoring relevance | 30% | 0-10 |
| AI system frequency (estimated) | 20% | 0-10 |
| Discriminative value | 10% | 0-10 |

**Example scoring:**

```
ethical_motive.n.01:
- Deception detection: 9 (detecting stated vs actual ethics)
- Alignment monitoring: 10 (core to value alignment)
- Frequency: 6 (AI systems reason about ethics moderately often)
- Discriminative value: 9 (clear distinction from other motives)
- Total: 8.7 → HIGH PRIORITY

psychic_energy.n.01:
- Deception detection: 1 (not relevant)
- Alignment monitoring: 1 (not relevant)
- Frequency: 1 (rare psychoanalytic jargon)
- Discriminative value: 3 (overlaps with motivation)
- Total: 1.4 → LOW PRIORITY
```

## Opportunity Cost Analysis

### Cost of Expanding Coverage

**Per concept cost:**
- Training time: ~18s/concept (current average)
- Data generation: ~10-30 samples/concept (adaptive)
- Lens storage: ~1MB/concept (activation classifier)
- Monitoring overhead: ~0.1ms/token/concept (inference time)

**Scaling costs:**

| Expansion | Concepts Added | Training Time | Storage | Inference Overhead |
|-----------|----------------|---------------|---------|-------------------|
| noun.motive only | 6-10 | ~3 minutes | ~10 MB | +0.6-1.0 ms/token |
| Tops completion | 4 (8% → 100%) | ~1 minute | ~4 MB | +0.4 ms/token |
| Strategic Layer 5 | 50-100 | ~15-30 min | ~50-100 MB | +5-10 ms/token |
| Full relationship audit | 200-500 | ~1-2.5 hours | ~200-500 MB | +20-50 ms/token |

### Benefit of Expanding Coverage

**Hard to quantify, but consider:**

1. **Deception detection without ethical_motive:**
   - Can detect "stated goal" vs "actual goal" mismatch
   - Cannot detect "ethical framing" vs "unethical action"
   - Missing: ~30% of deception cases involving ethics

2. **Alignment monitoring without rational_motive:**
   - Can detect high-level motivation
   - Cannot distinguish rational from irrational goals
   - Missing: Key signal for goal misgeneralization

3. **Behavioral prediction without incentive/disincentive:**
   - Can detect actions
   - Cannot predict response to rewards/punishments
   - Missing: RL-relevant intention understanding

### Recommended Prioritization

**Tier 1: Critical Gaps (Do Now)**
- noun.motive expansion (6-10 concepts)
- noun.Tops completion (4 concepts)
- **Total:** ~10-15 concepts, ~5 minutes training

**Tier 2: High-Value Domains (Do Next)**
- noun.feeling critical emotions (guilt, shame, fear, anger, joy)
- noun.communication deception concepts (lying, concealing, revealing)
- noun.act safety-relevant actions (betrayal, cooperation, manipulation)
- **Total:** ~30-50 concepts, ~15-20 minutes training

**Tier 3: Relationship Completion (Medium Term)**
- For each Layer 2-3 concept, ensure all children <threshold loaded
- Threshold: parents with <10 children get all children in Layer 5
- **Total:** ~100-200 concepts, ~30-60 minutes training

**Tier 4: Full Audit (Long Term)**
- Systematic review of all 82k nouns
- AI safety prioritization scoring
- Strategic expansion based on empirical lens performance
- **Total:** TBD based on audit results

## Implementation Plan

### Step 1: Adaptive Sampling (Week 1)

**Modify training to detect and adapt to relationship density:**

1. Add `get_relationship_count()` to data generation
2. Modify sample generation to scale with relationships
3. Add logging to track relationship coverage per concept
4. Test on current training run (no changes to concepts yet)

**Output:** Training log showing relationship coverage per concept

### Step 2: Tier 1 Expansion (Week 1)

**Add critical gaps:**

1. Create Layer 5 entries for:
   - 6 motivation children (rational_motive, irrational_motive, ethical_motive, urge, + 2 descendants)
   - 4 noun.Tops missing concepts
2. Generate WordNet synset mappings
3. Validate synsets (100% coverage check)
4. Train on Layer 5 with adaptive sampling
5. Evaluate lens calibration

**Output:** 10-15 new Layer 5 concepts with validated lenses

### Step 3: Prioritization Framework (Week 2)

**Develop systematic scoring:**

1. Create AI safety relevance scoring rubric
2. Score all noun.feeling, noun.communication, noun.act concepts
3. Identify top 30-50 for Tier 2 expansion
4. Document prioritization decisions

**Output:** Scored concept list for Tier 2

### Step 4: Tier 2 Expansion (Week 2-3)

**Add high-value domains:**

1. Add top-scored feeling concepts
2. Add deception-related communication concepts
3. Add safety-relevant action concepts
4. Train and validate

**Output:** 30-50 new Layer 5 concepts

### Step 5: Relationship Audit (Month 2)

**Systematic parent-child completion:**

1. For each Layer 2-3 concept, count children
2. If children <10, add all to Layer 5
3. If children >10, score and add top subset
4. Train and validate

**Output:** Complete relationship coverage for high-level concepts

## Success Metrics

### Training Metrics
- Relationship coverage per concept (target: 100% for concepts with <15 children)
- Graduation rates by relationship density (expect: more cycles for dense concepts)
- Sample efficiency (adaptive sampling should reduce wasted samples)

### Lens Quality Metrics
- Calibration scores for new Layer 5 concepts
- Validation grades (target: B+ or better)
- Cross-domain discrimination (ethical vs unethical, rational vs irrational)

### AI Safety Metrics
- Deception detection F1 on test cases involving ethics/motivation
- Alignment monitoring coverage of intention-related prompts
- Behavioral prediction accuracy for incentive/disincentive scenarios

## Open Questions

1. **Optimal relationship sample ratio:**
   - Current: 1:1 positive:negative
   - Should it be 1:2 for relationship-dense concepts?
   - Should we ensure ≥1 sample per child concept?

2. **Layer 5 capacity limits:**
   - How many concepts before inference overhead becomes problematic?
   - Is 100-200 concepts reasonable?
   - Should we have Layer 7 for very specific concepts?

3. **Empirical validation:**
   - Do more relationship samples actually improve lens quality?
   - Can we measure relationship coverage impact on deception detection?
   - What's the minimum viable coverage for each domain?

4. **Agentic review process:**
   - You mentioned "expanded synset relationship mapping via agentic review"
   - How does this fit into the uplift plan?
   - Should AI review our concept priorities before training?

## Conclusion

**Short-term (This Week):**
- Implement adaptive relationship sampling
- Add Tier 1 critical gaps (noun.motive + noun.Tops)
- ~10-15 concepts, ~5 minutes training

**Medium-term (This Month):**
- Develop prioritization framework
- Add Tier 2 high-value domains
- ~30-50 concepts, ~15-20 minutes training

**Long-term (Next Quarter):**
- Systematic relationship audit
- Complete parent-child coverage for all high-level concepts
- ~100-200 concepts, ~30-60 minutes training

**Philosophy:**
- Quality over quantity: strategic expansion beats exhaustive coverage
- Relationship-aware: ensure training samples represent all critical connections
- Safety-driven: prioritize concepts by AI safety impact, not alphabetically
- Adaptive: let training regime adjust to concept complexity

---

**Proposal Date:** 2025-11-16
**Status:** Draft for review and prioritization
**Next Steps:** Agree on Tier 1 scope, implement adaptive sampling, execute expansion
