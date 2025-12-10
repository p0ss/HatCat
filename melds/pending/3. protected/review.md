# Meld Review: verification-factchecking.json

**Reviewer**: Claude (automated review session)
**Date**: 2025-12-10
**Status**: Requires Discussion

---

## Summary

The `verification-factchecking` meld introduces 19 concepts for monitoring verification and fact-checking behaviors in LLM outputs. The architectural design is sound (polar inverse pairs, appropriate safety tags, correct hierarchy), but the **training example quality** requires revision for three key concepts.

---

## What Was Changed

Revised negative examples for three concepts to use **hard negatives** instead of generic placeholders:

| Concept | Risk Level | Issue | Change Made |
|---------|------------|-------|-------------|
| `ClaimExtraction` | low | Negatives were terse declarations | Replaced with topic summaries, vague allegations, commentary |
| `SourceCorroboration` | low | Negatives didn't represent false corroboration | Replaced with echo chambers, citation chains, institutional dependencies |
| `EvidenceFabrication` | **high** | Negatives were synonyms for "fake" | Replaced with real-but-flawed evidence scenarios |

---

## Rationale: Why Hard Negatives Matter

### The Problem with Generic Negatives

Original negative examples like:
```
"Claims were extracted."
"The source was made up."
"Multiple sources agreed."
```

These train the lens to detect **verbosity and specificity** rather than **actual verification quality**. The model learns:
- Long, detailed sentence → concept fires
- Short, bland sentence → concept doesn't fire

This is dangerous for fact-checking because **sophisticated misinformation is often highly detailed and specific**.

### The Hard Negative Approach

Hard negatives sit at the **decision boundary** - examples that superficially resemble the concept but fail in important ways:

| Concept | Hard Negative Pattern |
|---------|----------------------|
| ClaimExtraction | Topic summaries, weasel words, commentary (not specific verifiable assertions) |
| SourceCorroboration | Echo chambers, shared origins, institutional dependencies (not genuine independence) |
| EvidenceFabrication | Real-but-flawed evidence: retractions, misattributions, methodological issues (not invention) |

This trains discrimination where it actually matters in real-world verification scenarios.

---

## Detailed Changes

### 1. ClaimExtraction (lines 153-160)

**Before:**
```json
"negative_examples": [
  "Claims were found.",
  "Statements were identified.",
  "The content was analyzed.",
  "Claims were extracted.",
  "Assertions were noted."
]
```

**After:**
```json
"negative_examples": [
  "The article discusses concerns about rising inflation and its impact on households.",
  "Experts suggest the policy could have significant consequences for the economy.",
  "The report addresses multiple aspects of the ongoing healthcare debate.",
  "Sources close to the matter indicate systemic problems may exist.",
  "The piece makes a compelling argument about economic inequality and mobility."
]
```

**Why:** Each negative looks like analysis but fails claim extraction:
- Topic summaries instead of specific assertions
- Weasel words ("suggests", "could have", "may exist") that aren't verifiable
- Commentary rather than factual claims

---

### 2. SourceCorroboration (lines 274-281)

**Before:**
```json
"negative_examples": [
  "Multiple sources agreed.",
  "The claim was confirmed.",
  "Other sources said the same thing.",
  "Sources corroborated.",
  "Multiple outlets confirmed."
]
```

**After:**
```json
"negative_examples": [
  "The statistic was confirmed by Reuters, AP, and CNN, all citing the same WHO press release.",
  "Three independent researchers from the same laboratory verified the experimental finding.",
  "The claim appeared in the Times, which cited the Post, which cited the original blog post.",
  "Multiple fact-checkers rated the claim true based on the same unverified primary source.",
  "Confirmation came from both the subsidiary and its parent company's communications team."
]
```

**Why:** Each negative *appears* to be corroboration but fails independence:
- Common origin (same press release, same lab)
- Citation chains that circle back
- Institutional dependencies masking as independent confirmation

**Note:** These negatives intentionally overlap with the positive examples for `CircularSourcing`, reinforcing the polar inverse relationship.

---

### 3. EvidenceFabrication (lines 463-480)

**Before:**
```json
"negative_examples": [
  "The citation was wrong.",
  "The source was made up.",
  "The evidence was fake.",
  ... (15 variations of "it was fake/invented/fabricated")
]
```

**After:**
```json
"negative_examples": [
  "The study was retracted in 2021 due to methodological concerns, but the underlying data was collected from real participants.",
  "The quote was accurate but attributed to the wrong speech from the same conference that year.",
  "The statistic came from a legitimate Pew poll but with a margin of error larger than the difference being claimed.",
  "The paper exists in the journal but the cited conclusion appears in the discussion section, not the results.",
  ... (15 scenarios of real-but-flawed evidence)
]
```

**Why:** This is the **highest-risk concept** in the meld. The lens must distinguish:
- **Fabrication**: Inventing sources, studies, quotes that don't exist
- **Misuse**: Real evidence that's retracted, misattributed, miscontextualized, or methodologically flawed

A fact-checker who conflates these will either:
- Miss real fabrication by accepting "well, something similar exists"
- Cry fabrication when evidence is merely flawed or misused

---

## Remaining Concepts to Review

The same pattern of generic negatives appears throughout the meld. The following concepts should be reviewed for similar revisions:

| Concept | Current Negative Quality | Priority |
|---------|-------------------------|----------|
| `ClaimConflation` | Generic | Medium |
| `CircularSourcing` | Generic | High (polar inverse of SourceCorroboration) |
| `Decontextualization` | Generic | High |
| `TemporalMisrepresentation` | Generic | Medium |
| `Misattribution` | Generic | Medium |
| `StatisticalMisrepresentation` | Generic | High |
| `MediaMisrepresentation` | Generic | Medium |
| `MisinformationDetection` | Generic | High |
| `VerificationBypass` | Generic | Medium |

---

## Recommendation

1. **Accept the three revised concepts** as demonstration of the hard negative approach
2. **Discuss with meld author** whether to apply this pattern to remaining concepts before approval
3. **Consider whether validation thresholds** (15 for high-risk, 10 for harness_relevant, 5 for standard) should include a **quality criterion** beyond count

---

## Questions for Discussion

1. Should hard negatives for polar inverse pairs be **explicitly cross-referenced**? (e.g., negatives for SourceCorroboration drawn from CircularSourcing positives)

2. For `EvidenceFabrication`, is the distinction between "fabrication" and "misuse of real evidence" the right boundary? Or should the lens be broader?

3. Should we add a `negative_example_type` field to distinguish:
   - `near_miss`: Related but distinct concept
   - `polar_opposite`: From the inverse concept
   - `failure_mode`: The concept done poorly
   - `superficial_match`: Looks similar but isn't

---

## Files Modified

- `melds/pending/3. protected/verification-factchecking.json` - Revised negative examples for 3 concepts
- `melds/pending/3. protected/review.md` - This review document (new file)
