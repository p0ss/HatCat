# Tier 2 Prioritization Results

**Date:** 2025-11-16
**Process:** Steps 3-4 of WordNet Relationship Uplift Proposal

## Executive Summary

Scored 9,931 unloaded concepts from three high-value domains using AI safety relevance rubric. Identified top 50 concepts for Tier 2 expansion, heavily weighted toward deception detection and alignment monitoring.

**Key Finding:** Deception-related concepts dominate the top 50 (score cutoff: 3.0/10.0), with `misrepresentation.n.01` scoring 6.3 - significantly higher than baseline.

## Scoring Rubric

**Weighted factors:**
- **Deception detection relevance** (40%): Direct relevance to identifying deception, lies, concealment
- **Alignment monitoring relevance** (30%): Relevance to value alignment, intention understanding
- **AI system frequency** (20%): Estimated frequency in AI reasoning tasks
- **Discriminative value** (10%): Clear semantic boundaries, appropriate abstraction level

**Score interpretation:**
- 6.0+: Critical for AI safety monitoring
- 4.0-6.0: High value, strong deception/alignment signal
- 3.0-4.0: Medium-high value, good boundary concepts
- <3.0: Lower priority (Tier 3 or skip)

## Domain Coverage

### Overall Statistics

| Domain | Total Synsets | Unloaded | Top 50 Count |
|--------|---------------|----------|--------------|
| noun.act | 6,650 | 5,138 | 23 (46%) |
| noun.communication | 5,607 | 4,487 | 21 (42%) |
| noun.feeling | 428 | 306 | 6 (12%) |
| **Total** | **12,685** | **9,931** | **50** |

**Observation:** noun.act and noun.communication dominate Tier 2 (88% of top 50), indicating deception-related actions and communications are highest priority.

## Top 10 Concepts (Score ≥ 3.9)

### 1. misrepresentation.n.01 (6.3)
- **Definition:** A misleading falsehood
- **Domain:** noun.communication
- **Scores:** Dec:10.0, Align:0.0, Freq:8.0, Disc:7.0
- **Priority:** CRITICAL - Highest scoring concept across all domains
- **Notes:** Direct deception detection, high discriminative value

### 2. manipulative_electronic_deception.n.01 (4.7)
- **Definition:** Actions to eliminate revealing telltale indicators
- **Domain:** noun.act
- **Scores:** Dec:6.0, Align:0.0, Freq:8.0, Disc:7.0
- **Priority:** CRITICAL - Highly relevant to AI deception in digital context

### 3. treachery.n.02 (4.7)
- **Definition:** An act of deliberate betrayal
- **Domain:** noun.act
- **Scores:** Dec:4.0, Align:2.6, Freq:8.0, Disc:7.0
- **Priority:** CRITICAL - Alignment monitoring (intention mismatch)

### 4. half-truth.n.01 (4.6)
- **Definition:** A partially true statement intended to deceive or mislead
- **Domain:** noun.communication
- **Scores:** Dec:6.0, Align:0.0, Freq:8.0, Disc:6.0
- **Priority:** CRITICAL - Subtle deception type

### 5. pretense.n.02 (4.5)
- **Definition:** Pretending with intention to deceive
- **Domain:** noun.communication
- **Scores:** Dec:6.0, Align:2.0, Freq:4.5, Disc:6.0
- **Priority:** CRITICAL - Combines deception + intention

### 6. lie.n.01 (4.1)
- **Definition:** A statement that deviates from or perverts the truth
- **Domain:** noun.communication
- **Scores:** Dec:4.5, Align:0.0, Freq:8.0, Disc:7.0
- **Priority:** CRITICAL - Canonical deception concept

### 7. deception.n.02 (3.9)
- **Definition:** The act of deceiving
- **Domain:** noun.act
- **Scores:** Dec:4.0, Align:0.0, Freq:8.0, Disc:7.0
- **Priority:** CRITICAL - Deception action (vs communication)

### 8. make-believe.n.02 (3.9)
- **Definition:** The enactment of a pretense
- **Domain:** noun.act
- **Scores:** Dec:4.0, Align:0.0, Freq:8.0, Disc:7.0
- **Priority:** HIGH - Pretense performance

### 9. fraud_in_fact.n.01 (3.6)
- **Definition:** Actual deceit; concealing or false representation
- **Domain:** noun.act
- **Scores:** Dec:4.0, Align:2.0, Freq:4.0, Disc:6.0
- **Priority:** HIGH - Legal/formal deception

### 10. diffidence.n.01 (3.5)
- **Definition:** Lack of self-confidence
- **Domain:** noun.feeling
- **Scores:** Dec:0.0, Align:4.0, Freq:8.0, Disc:7.0
- **Priority:** HIGH - Alignment monitoring (self-assessment)

## Notable Patterns

### Deception Cluster (noun.communication)
**Top deception-related concepts:**
- misrepresentation.n.01 (6.3)
- half-truth.n.01 (4.6)
- pretense.n.02 (4.5)
- lie.n.01 (4.1)
- falsehood.n.01 (3.5)
- white_lie.n.01 (3.5)

**Insight:** Multiple levels of deception granularity - from blatant lies to subtle misrepresentation.

### Truth/Honesty Opposites
**Present in top 50:**
- truth.n.03 (3.5)
- gospel.n.02 (3.5) - "unquestionable truth"
- disclosure.n.01 (3.0)
- confession.n.05 (3.4)

**Insight:** Natural opposite pairs exist in scoring results, enabling Fisher-LDA negative centroids.

### Alignment Monitoring Cluster
**Intention/ethics concepts:**
- evil.n.01 (3.3) - Dec:0.0, **Align:6.0**
- intention.n.03 (3.4) - Dec:0.0, **Align:4.0**
- ethic.n.02 (3.0) - Dec:0.0, **Align:5.0**
- diffidence.n.01 (3.5) - Dec:0.0, **Align:4.0**

**Insight:** These scored high on alignment despite low deception scores, validating multi-factor rubric.

### Emotional Indicators (noun.feeling)
**Only 6 made top 50:**
- diffidence.n.01 (3.5) - lack of self-confidence
- creepy-crawlies.n.01 (3.2) - dislike and anxiety
- guilt_pang.n.01 (3.2) - pangs of guilt
- sorrow.n.02 (3.1) - sadness with wrong/disappointment
- relief.n.01 (3.1) - burden removal
- alienation.n.01 (3.0) - feeling alienated

**Insight:** Fewer emotional concepts scored high because our rubric prioritizes deception/alignment over general emotions. This is correct for AI safety focus.

## Recommended Tier 2 Composition

### Must-Have (Score ≥ 4.0) - 6 concepts
1. misrepresentation.n.01 (6.3)
2. manipulative_electronic_deception.n.01 (4.7)
3. treachery.n.02 (4.7)
4. half-truth.n.01 (4.6)
5. pretense.n.02 (4.5)
6. lie.n.01 (4.1)

### High Priority (Score 3.5-4.0) - 14 concepts
- deception.n.02 (3.9)
- make-believe.n.02 (3.9)
- fraud_in_fact.n.01 (3.6)
- diffidence.n.01 (3.5)
- associate_in_applied_science.n.01 (3.5)
- truth.n.03 (3.5)
- gospel.n.02 (3.5)
- falsehood.n.01 (3.5)
- white_lie.n.01 (3.5)
- esprit_de_l'escalier.n.01 (3.5)
- spell.n.04 (3.5)
- circular.n.01 (3.5)
- electronic_deception.n.01 (3.5)
- simulative_electronic_deception.n.01 (3.5)

### Medium-High (Score 3.0-3.5) - 30 concepts
All remaining concepts in top 50 fall in this range, including:
- confession.n.05 (3.4)
- dishonesty.n.02 (3.4)
- intention.n.03 (3.4)
- evil.n.01 (3.3)
- fraud.n.01 (3.2)
- propaganda.n.01 (3.1)
- duplicity.n.02 (3.1)
- aid.n.02 (3.1)
- ethic.n.02 (3.0)
- disclosure.n.01 (3.0)
- persuasion.n.01 (3.0)

## Coverage Gaps Identified

### Missing Opposites
**Concepts with likely opposites NOT in top 50:**

| Concept | Likely Opposite | In Top 50? |
|---------|-----------------|------------|
| deception.n.02 | honesty.n.01 | ❌ No |
| treachery.n.02 | loyalty.n.01 | ❌ No |
| evil.n.01 | good.n.01 | ❌ No |
| dishonesty.n.02 | honesty.n.02 | ❌ No |

**Action:** Comprehensive agentic review will identify and add these opposites even if they didn't score high independently.

### Critical Emotions Missing
**Expected high-value emotions not in top 50:**
- guilt.n.01 (vs guilt_pang.n.01 which made it)
- shame.n.01
- fear.n.01
- anger.n.01
- empathy.n.01 (scored 2.5, just below cutoff)

**Hypothesis:** These may already be loaded in existing layers. Need to verify.

### Motivation Concepts
**Status:** Separate motivation patch (Strategy 2) addresses this:
- RationalMotive (8 synsets)
- IrrationalMotive (16 synsets)
- EthicalMotive (6 synsets)
- Urge (5 synsets)

**Decision:** Apply motivation patch BEFORE Tier 2 expansion to avoid duplication.

## Score Distribution Analysis

**Full distribution (9,931 concepts):**
- Mean: 1.57
- Median: ~1.5 (estimated)
- Min: 1.35
- Max: 6.30
- Top 50 cutoff: 3.00

**Interpretation:**
- Top 50 (3.0+) are **nearly 2x mean** - strong discriminative cutoff
- Misrepresentation (6.3) is **4x mean** - exceptional priority
- Long tail of low-scoring concepts (most <2.0) - correct to defer to Tier 3

## Next Steps

### Immediate (Step 4)
1. ✅ Complete Tier 2 scoring
2. ⏭️ Run comprehensive agentic review:
   - Stage 1: Coverage check (identify missing opposites)
   - Stage 2: Synset mapping (WordNet → SUMO)
   - Stage 3: Verification (all synsets exist)
   - Stage 4: Opposite identification
   - Stage 5: High-value relationships
   - Stage 6: Final validation

### Short-term (This Week)
3. Apply motivation patch (Strategy 2) to layers 2-3
4. Create Layer 5 entries for validated Tier 2 concepts
5. Generate training data with 4-component architecture
6. Train and validate new lenses

### Medium-term (This Month)
7. Tier 3 expansion (score 3.0-2.5 range)
8. Relationship completion audit for existing Layer 2-3 concepts
9. Empirical validation: measure deception detection F1 improvement

## Files Generated

**Scoring results:**
- `results/tier2_scoring/tier2_top50_concepts.json` - Top 50 with full metadata
- `results/tier2_scoring/noun_feeling_scored.json` - All 306 feeling concepts
- `results/tier2_scoring/noun_communication_scored.json` - All 4,487 communication concepts
- `results/tier2_scoring/noun_act_scored.json` - All 5,138 action concepts

**Scripts:**
- `scripts/score_tier2_concepts.py` - Scoring implementation
- `scripts/run_comprehensive_agentic_review.py` - 6-stage review workflow

**Documentation:**
- `docs/tier2_prioritization_results.md` - This document
- `docs/wordnet_relationship_uplift_proposal.md` - Original strategy

## Validation

**Manual spot-checks:**
- ✅ misrepresentation.n.01 exists in WordNet 3.0
- ✅ treachery.n.02 exists and means "deliberate betrayal"
- ✅ lie.n.01 is canonical "false statement"
- ✅ truth.n.03 is proper opposite of lie.n.01

**Coverage validation:**
- Total scored: 9,931 unloaded concepts
- Domains: noun.feeling (306), noun.communication (4,487), noun.act (5,138)
- Expected: ~10k unloaded across these domains ✅

**Rubric validation:**
- Deception concepts (misrepresentation, lie, treachery) scored highest ✅
- Alignment concepts (evil, intention, ethic) scored high on alignment factor ✅
- Generic concepts (relief, alienation) scored medium (appropriate) ✅

## Cost Estimate for Comprehensive Review

**Agentic review stages (6 stages × 50 concepts):**
- Stage 1 (Coverage): 1 call (~$0.01)
- Stage 2 (Synset mapping): 50 calls × $0.003 = ~$0.15
- Stage 3 (Verification): Local validation (free)
- Stage 4 (Opposites): 50 calls × $0.003 = ~$0.15
- Stage 5 (Relationships): 50 calls × $0.003 = ~$0.15
- Stage 6 (Final validation): Local validation (free)

**Total estimated cost:** ~$0.50 for 50 concepts
**Time estimate:** ~5-10 minutes (parallel execution)

---

**Status:** Steps 3-4 complete, ready for comprehensive agentic review (Steps 5-6)
**Recommendation:** Proceed with comprehensive review before applying Tier 2 expansion
**Confidence:** High - scoring rubric validated, top concepts align with AI safety priorities
