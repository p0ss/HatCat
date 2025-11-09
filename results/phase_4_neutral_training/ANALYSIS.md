# Phase 4: Neutral Training & Comprehensive Testing Analysis

**Date**: November 4, 2025
**Test Scale**: 10 concepts (WordNet top 10)
**Model**: google/gemma-3-4b-pt
**Training**: 1 pos + 1 neg + 1 neutral
**Evaluation**: 20 pos + 20 neg + 20 neutral

## Summary

Phase 4 introduces **neutral training samples** and **comprehensive evaluation** (positive + negative + neutral testing). Key finding: **Detection quality is more realistic** (F1=0.787) compared to Phase 3a's overly optimistic 97.8% confidence on positive-only tests. The system can now measure false positive rates and true negative rates.

---

## 1. Key Metrics

### Aggregate Performance
- **F1 Score**: 0.787 (harmonic mean of precision and recall)
- **Precision**: 0.789 (correct positives / all predicted positives)
- **Recall**: 0.840 (correct positives / all actual positives)

### Confusion Matrix (per concept average)
- **True Positives (TP)**: 16.8 / 20 (84.0% - good detection)
- **True Negatives (TN)**: 34.7 / 40 (86.8% - good rejection)
- **False Positives (FP)**: 5.3 / 40 (13.2% - some false alarms)
- **False Negatives (FN)**: 3.2 / 20 (16.0% - some misses)

### Confidence Distributions
| Label Type | Mean | Std |
|-----------|------|-----|
| **Positive** | 83.4% | 23.3% |
| **Negative** | 17.1% | 10.2% |
| **Neutral** | 13.7% | 15.9% |

**Observations**:
- Clear separation between positive (83%) and negative/neutral (17%, 14%)
- High variance on positives (23.3%) suggests some concepts are harder to detect
- Negative and neutral confidences overlap (17% vs 14%) - expected since both are "not the concept"

---

## 2. Comparison with Phase 3a

| Metric | Phase 3a (Positive-Only) | Phase 4 (Comprehensive) | Change |
|--------|--------------------------|-------------------------|--------|
| **Evaluation** | Positive samples only | Pos + Neg + Neutral | ✅ More realistic |
| **Confidence** | 97.8% mean | 83.4% mean (pos only) | -14.4% (expected) |
| **False Positives** | Unknown (not tested) | 13.2% | ⚠️ Now measurable |
| **True Negatives** | Unknown (not tested) | 86.8% | ✅ Good rejection |

**Key Insight**: Phase 3a was testing on easy samples (only positive prompts). Phase 4 reveals the real challenge: distinguishing positives from negatives/neutrals.

---

## 3. Per-Concept Performance

### Top Performers (F1 > 0.9)
1. **bird genus**: F1=0.976 (P=0.952, R=1.000)
2. **animal order**: F1=0.976 (P=0.952, R=1.000)
3. **asterid dicot genus**: F1=0.930 (P=0.870, R=1.000)
4. **mammal genus**: F1=0.889 (P=0.800, R=1.000)

### Struggling Concepts (F1 < 0.7)
1. **fish genus**: F1=0.444 (P=0.857, R=0.300) - **Low recall** (missing 70% of positives!)
2. **herb**: F1=0.556 (P=0.625, R=0.500) - Low recall (missing 50% of positives)
3. **change**: F1=0.667 (P=0.500, R=1.000) - **High false positive rate** (50% precision)

**Patterns**:
- **High connectivity concepts** (bird genus, mammal genus) perform better
- **Abstract concepts** (change) have more false positives
- **fish genus** is surprisingly difficult despite being concrete - may need more training data

---

## 4. What Changed from Phase 3a

### Training Data
**Phase 3a** (positive-only baseline):
- 1 positive definitional prompt
- 1 negative (distant concept)
- **0 neutral samples** ❌

**Phase 4** (neutral training):
- 1 positive definitional prompt
- 1 negative (distant concept)
- **1 neutral sample** (from reserved pool, distance ≥15) ✅

**Impact**: Adding neutral samples forces classifiers to learn "definitional text ≠ always positive", reducing false positives on generic content.

### Evaluation Data
**Phase 3a**: 20 positive prompts only
**Phase 4**: 20 positive + 20 negative + 20 neutral prompts

**Impact**: Now we can measure:
- True negative rate (how well we reject wrong concepts)
- False positive rate (how often we incorrectly flag neutrals)
- Realistic performance on mixed content

---

## 5. Neutral Pool Effectiveness

**Design**: 1000 concepts with path distance ≥15 from ALL training concepts

**Results**:
- Neutral confidence: 13.7% mean (low, as desired)
- Similar to negative confidence: 17.1% (expected - both are "not the concept")
- No neutral concepts appear in false positives list (good isolation)

**Conclusion**: The neutral pool strategy works - concepts are sufficiently distant to serve as true negatives.

---

## 6. Red Flags and Areas for Improvement

### 1. High False Positive Rate (13.2%)
- "change" has 50% precision (half of predicted positives are wrong)
- Abstract concepts may need more negative/neutral training samples

### 2. Low Recall on Some Concepts
- "fish genus" only detects 30% of positive samples (70% false negatives!)
- "herb" misses 50% of positive samples
- May need more positive training samples or better negative selection

### 3. High Variance in Positive Confidence (23.3% std)
- Some prompts get 100% confidence, others get 50%
- Suggests classifiers are not robust to prompt variation
- May benefit from more diverse positive training prompts

---

## 7. Next Steps

### Immediate (Phase 5)
**Semantic Evaluation**: Test whether steering vectors actually move generation toward/away from concepts
- Measure semantic field activation (hypernyms, hyponyms, etc.)
- Track concept mentions during generation
- Validate that detection correlates with steering effectiveness

### Medium-term (Phase 6)
**Accuracy Calibration Study**: Determine minimum F1 needed for detection/steering
- Test 70%, 80%, 90%, 95%, 99% accuracy targets
- Find trade-off between training time and quality
- May discover 80% F1 is sufficient, saving 50%+ training time

### Long-term (Phase 7-11)
- Subspace removal to reduce false positives
- Steering vector composition tests
- Scale to 10,000 concepts
- Production inference interface

---

## 8. Conclusions

✅ **Comprehensive evaluation works**:
- F1=0.787 is more realistic than Phase 3a's 97.8%
- Can now measure false positive (13.2%) and true negative (86.8%) rates
- Clear confidence separation between positive (83%) and negative/neutral (17%, 14%)

⚠️ **Performance gaps identified**:
- False positive rate too high for some concepts (13.2% overall, 50% for "change")
- Low recall on "fish genus" (30%) and "herb" (50%)
- High variance suggests prompt sensitivity

🎯 **Priority**: Move to Phase 5 (Semantic Evaluation) to validate that these classifiers actually correspond to meaningful semantic steering. Detection accuracy alone doesn't guarantee steering effectiveness!

---

## Files

**Results**: `results/phase_4_neutral_training/phase4_results.json`
**Script**: `scripts/phase_4_neutral_training.py`
**Analysis**: `results/phase_4_neutral_training/ANALYSIS.md`
**Concept Graph**: `data/concept_graph/wordnet_v2_top10.json` (NEW format with neutral_pool)
