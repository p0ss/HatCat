# Phase 3a: Inference Baseline Analysis (Positive-Only Evaluation)

**Date**: November 4, 2025
**Test Scale**: 10 concepts (WordNet top 10)
**Model**: google/gemma-3-4b-pt
**Training**: 1 pos + 1 neg (no neutral samples)
**Evaluation**: Positive samples only (no negative/neutral testing)

## Summary

Established baseline performance metrics for concept detection inference pipeline with **flawed evaluation** (positive-only testing). Key finding: **Runtime performance is excellent** (sub-millisecond per concept), but **detection confidence is suspiciously high** (97.8% mean), confirming we're only testing on easy positive samples.

**⚠️ Note**: This baseline will be re-run as **Phase 3b** after Phase 4 (Neutral Training & Comprehensive Testing) to measure performance with proper evaluation including negative and neutral samples.

---

## 1. Runtime Performance ✅

### Latency
- **Mean**: 0.544ms for 10 classifiers
- **Per-concept**: 0.054ms
- **P95**: 0.559ms
- **P99**: 0.862ms

**Scaling projection**:
- 100 concepts → ~5.4ms
- 1000 concepts → ~54ms

**Conclusion**: Well within real-time requirements. Even 1000 concepts can run at ~18 fps.

### Memory
- **Total GPU allocated**: 16.06 GB
- **Breakdown**:
  - Language model (gemma-3-4b): ~16GB (bulk of memory)
  - 10 classifiers: ~3MB total (~0.3MB each)

**Per-classifier memory**:
- Linear(2560 → 128): ~320KB
- Linear(128 → 1): ~512 bytes
- **Total**: ~0.3MB per classifier

**Scaling projection**:
- 100 classifiers: ~30MB
- 1000 classifiers: ~300MB
- 10000 classifiers: ~3GB

**Conclusion**: Classifier memory is negligible compared to base model. No scaling issues.

---

## 2. Detection Quality ⚠️

### Confidence Distributions

**Aggregate statistics** (across 10 concepts):
- **Mean confidence**: 97.8%
- **Std across concepts**: 4.5%
- **Range**: 90.2% (animal order) to 100.0% (rosid dicot genus)

**Per-concept breakdown**:

| Concept | Mean | Std | Min | Max | Note |
|---------|------|-----|-----|-----|------|
| rosid dicot genus | 100.0% | 0.001% | 99.996% | 100% | Near-perfect |
| asterid dicot genus | 99.999% | 0.001% | 99.996% | 100% | Near-perfect |
| mammal genus | 99.998% | 0.005% | 99.982% | 100% | Near-perfect |
| bird genus | 99.961% | 0.038% | 99.904% | 100% | Excellent |
| fish genus | 99.992% | 0.011% | 99.965% | 100% | Excellent |
| person | 99.860% | 0.177% | 99.442% | 100% | Excellent |
| herb | 98.441% | 2.290% | 93.639% | 100% | Good |
| shrub | 98.146% | 2.489% | 93.046% | 100% | Good |
| change | 91.166% | 12.870% | 68.234% | 100% | Variable |
| animal order | 90.158% | 27.313% | **8.255%** | 99.984% | **Concerning** |

**Key observations**:
1. Most concepts have >99% confidence (8/10)
2. "animal order" has one very low confidence (8.3%) - possible false negative
3. High variance on abstract concepts ("change", "animal order")
4. Very low variance on concrete concepts (genus classifications)

### Detection Timing

Tested concept activation during generation (sample of 3 concepts):

| Concept | Mean Score | Max Score | Max Position | Behavior |
|---------|-----------|-----------|--------------|----------|
| person | 97.6% | 100% | Token 0 | Immediate activation |
| change | 99.5% | 100% | Token 5 | Quick activation |
| bird genus | 92.8% | 100% | Token 21 | Variable activation |

**Observations**:
- Concept detection remains high throughout generation
- Some variation during generation (e.g., "bird genus" drops to 42% at token 3)
- Max activation often occurs early in generation

---

## 3. Red Flags 🚩

### High Confidence Across the Board
- Mean confidence of 97.8% is **suspiciously high**
- Suggests we're only testing on **easy positive samples**
- **Root cause**: Current evaluation only tests positive prompts ("What is X?")

### Missing Evaluation Coverage
Current testing:
- ✅ True Positives: Testing concepts on relevant prompts
- ❌ True Negatives: NOT testing rejection of unrelated concepts
- ❌ False Positives: NOT measuring false alarm rate
- ❌ Neutral Content: NOT testing on generic/unrelated text

**Example**: A classifier that says "yes" to everything would pass current tests!

### One Outlier
- "animal order" scored **8.3% on one prompt** - unclear why
- Could be legitimate rejection OR a failure mode
- Need negative testing to understand this behavior

---

## 4. Next Steps (Phase 4)

### Add Comprehensive Testing

**New test types needed**:
1. **Negative samples**: Test concepts on semantically distant prompts
   - Example: Test "bird genus" on prompts about mathematics
   - Expect: Low confidence (<10%)

2. **Neutral samples**: Test on generic unrelated content
   - Example: "The sky is blue", procedural text, news articles
   - Expect: Low confidence across all concepts

3. **Hard negatives**: Test on semantically similar but wrong concepts
   - Example: Test "bird genus" on "mammal genus"
   - Expect: Some confusion, but ideally <50%

### New Metrics

Replace "% concepts @ 100% accuracy" with:
- **True Positive Rate**: % of positive samples correctly identified
- **True Negative Rate**: % of negative samples correctly rejected
- **False Positive Rate**: % of neutral/negative samples incorrectly flagged
- **F1 Score**: Harmonic mean of precision and recall

### Training Data Update

Add neutral training samples:
- Current: [1 positive, 1 negative per concept]
- Proposed: [1 positive, 1 negative, 1 neutral per concept]

This prevents classifiers from learning "definitional text = positive".

---

## 5. Conclusions

✅ **Runtime performance is excellent**:
- Sub-millisecond per concept
- Scales linearly to 1000+ concepts
- Memory usage dominated by base model, classifiers are tiny

⚠️ **Detection quality is unvalidated**:
- 97.8% confidence seems too high
- Only testing positive samples (major gap)
- Need comprehensive evaluation with negatives and neutrals

🎯 **Priority**: Move to Phase 4 (Neutral Training & Comprehensive Testing) before making any training improvements. Current evaluation is too lenient to measure real improvements.

---

## Files

**Results**: `results/phase_3_inference_baseline/baseline_results.json`
**Script**: `scripts/phase_3_inference_baseline.py`
**Analysis**: `results/phase_3_inference_baseline/ANALYSIS.md`
