# Hierarchical Training Data Hypothesis Test

## Research Question
Should parent concepts include their children's synsets as positive training examples? Should this principle apply recursively to all descendants?

## Hypotheses

### H1: Direct Children Inclusion (Primary)
**Hypothesis**: Parent concepts should include their direct children's synsets as positive training examples. Negative examples should primarily come from sibling concepts (graph-close hard negatives).

**Rationale**:
- A parent concept should be able to detect instances of its children
- Example: "Physical" should activate for both its own synsets AND all "Object", "Process", etc. synsets
- This aligns with the hierarchical suppression strategy (parents activate, then get suppressed by children)

**Prediction**: Lenses trained with children synsets will:
- Have higher calibration accuracy
- Better detect child concept instances
- Enable effective hierarchical suppression

### H2: Recursive Descendant Inclusion (Secondary)
**Hypothesis**: Parent concepts should include ALL descendant synsets recursively (children, grandchildren, etc.). Negatives remain siblings.

**Training Data Strategy**:
- **Positives**: Own synsets + all descendant synsets
- **Hard Negatives (siblings)**: Same-layer concepts with shared parent (graph-close)
- **Easy Negatives (irrelevant)**: Graph-distant concepts for balance

**Rationale**:
- A parent should detect ANY descendant, not just direct children
- Example: "Physical" should activate for "Mammal" (grandchild) as well as "Animal" (child)
- Provides complete coverage of the concept hierarchy

**Prediction**: Recursive inclusion will:
- Further improve detection accuracy
- Require exponentially more training samples for higher layers
- May cause overfitting if sample size not appropriately scaled

## Current Training Strategy (Baseline)

### Positive Examples
- Only `canonical_synset` (single synset per concept)
- WordNet relationship prompts from canonical synset

### Negative Examples
Current implementation uses `build_sumo_negative_pool()` which:
- Excludes all ancestors (parents, grandparents, etc.)
- Excludes all descendants (children, grandchildren, etc.)
- Includes all remaining concepts (which for Layer 0 = siblings!)
- **Prioritizes hard negatives** from AI symmetry mapping (complementary concepts)
- **Good**: For Layer 0, this already gives us sibling-focused negatives!

### Layer 0 Negative Pool Analysis
For Layer 0 concepts:
- No ancestors (they ARE the top layer)
- Descendants excluded (their children)
- **Result**: Negative pool = other Layer 0 concepts (perfect siblings!)
- Example for "Physical": Negatives = Quantity, Proposition, Entity, Process, Object, Attribute, Relation, Collection, Abstract

**Conclusion**: Current negative sampling is already graph-close (siblings) for Layer 0, which is ideal.

## Experimental Design

### Control Conditions
1. **Baseline**: Only use concept's own synsets (current approach)
2. **Direct Children**: Include direct children's synsets
3. **Recursive Descendants**: Include all descendant synsets

### Test Layer
Focus on **Layer 0** for initial testing:
- 10 concepts (Physical, Quantity, Proposition, Entity, Process, Object, Attribute, Relation, Collection, Abstract)
- Well-defined hierarchy with known children
- Small enough for quick iteration

### Metrics

#### Training Metrics
- Number of positive samples available per concept
- Training time per concept
- Final test F1 score

#### Calibration Metrics (Primary Outcome)
- **TP Rate**: True positive rate on concept's own synsets
- **Child Detection Rate**: Ability to detect children's synsets
- **Sibling FP Rate**: False positive rate on sibling concepts
- **Hierarchical Suppression Effectiveness**: Parent suppression when child activates

#### Statistical Tests
- Paired t-test comparing calibration accuracy across conditions
- Effect size (Cohen's d) for meaningful differences
- Significance threshold: p < 0.05

### Sample Size Comparison

Expected synset counts per condition:

**Layer 0 Example (Physical)**:
```
Baseline: 5 synsets (canonical only)
Direct Children: 20 synsets (from abstraction layer updates)
Recursive Descendants: 100+ synsets (all Object, Process, etc. descendants)
```

### Procedure

1. **Data Preparation**
   - Count synsets per concept under each condition
   - Verify no overlap between positive/negative pools
   - Document synset sources

2. **Training Phase**
   - Train 3 sets of Layer 0 lenses (one per condition)
   - Use identical hyperparameters (adaptive training, falloff validation)
   - Record training time and sample requirements

3. **Calibration Phase**
   - Test each lens set with standardized calibration:
     - 10 positive (concept's own synsets)
     - 10 sibling (other Layer 0 concepts)
     - 10 child (direct children's synsets)
     - 10 irrelevant (random other concepts)
   - Record all activation scores

4. **Analysis Phase**
   - Compare calibration metrics across conditions
   - Statistical significance testing
   - Visualize activation patterns

### Expected Outcomes

**If H1 is Correct**:
- Direct children condition will have:
  - Higher child detection rate
  - Similar or better sibling discrimination
  - Successful training (no sample exhaustion)

**If H1 is Incorrect**:
- Direct children condition will have:
  - Lower discrimination (too broad)
  - Higher sibling false positives
  - Overfitting

**If H2 is Correct**:
- Recursive condition will have:
  - Best child/descendant detection
  - Still maintains sibling discrimination
  - Requires more samples but trains successfully

**If H2 is Incorrect**:
- Recursive condition will have:
  - Poor discrimination (concept too broad)
  - Sample exhaustion (can't get enough negatives)
  - Training failures or overfitting

## Implementation Plan

### Phase 1: Baseline Measurement (Current State)
- Use existing Layer 0 calibration results
- Document current performance

### Phase 2: Direct Children Test
- Modify training data generator to include `concept['synsets']` array
- Train Layer 0 with children synsets
- Run calibration with child detection tests
- Compare to baseline

### Phase 3: Recursive Descendants Test
- Implement recursive descendant collection
- Train Layer 0 with all descendants
- Run same calibration tests
- Compare to both baseline and direct children

### Phase 4: Analysis & Decision
- Statistical analysis of results
- Document findings
- Decide on training strategy for full system

## Decision Criteria

### Adopt Direct Children Inclusion if:
- Child detection rate improves by ≥20%
- Sibling FP rate stays ≤10%
- Training succeeds for all Layer 0 concepts
- Statistical significance p < 0.05

### Adopt Recursive Inclusion if:
- Further improves child detection by ≥10% over direct children
- Maintains sibling discrimination
- Computational cost is acceptable
- Scales to higher layers without sample exhaustion

### Reject Both if:
- No significant improvement in child detection
- Sibling discrimination degrades
- Training failures increase
- Overfitting occurs

## Risks & Mitigations

### Risk 1: Sample Exhaustion
**Mitigation**: Start with Layer 0 (small), validate before scaling

### Risk 2: Overfitting
**Mitigation**: Monitor train/test gap, use validation sets

### Risk 3: Computational Cost
**Mitigation**: Measure training time, project to full system

### Risk 4: Concept Drift
**Mitigation**: Ensure parent still represents its own concept, not just union of children

## Output Artifacts

1. **Experimental Results**: `results/hierarchical_training_experiment/`
   - Baseline lenses
   - Direct children lenses
   - Recursive descendants lenses
   - Calibration results for each

2. **Analysis Report**: `results/hierarchical_training_experiment/analysis.json`
   - Statistical comparisons
   - Effect sizes
   - Recommendations

3. **Visualizations**:
   - Calibration score distributions
   - Child detection rates
   - Training sample requirements

4. **Decision Document**: `docs/HIERARCHICAL_TRAINING_DECISION.md`
   - Final recommendation
   - Rationale
   - Implementation plan
