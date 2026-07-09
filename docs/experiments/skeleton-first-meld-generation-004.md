# Experiment: Skeleton-First MELD Generation

## Hypothesis

**Claim**: MELD quality improves when the generator has full knowledge of the concept's position in the ontology (parents, siblings, AND children) before generating the detailed MELD content.

**Rationale**:
- Definitions are clearer when you know what subcategories exist (avoiding overlap)
- Negative examples can reference actual siblings (not imagined ones)
- Contrast concepts are drawn from known structure
- Steering targets (opposites) can leverage the full ontology

## Experimental Design

### Phase 1: Skeleton Generation (Structure Only)
Generate the full hierarchical structure L1→L2→L3 without detailed MELDs:
- Use "pick your classes" framing
- Subject model outputs labels + brief scope statements
- No examples, no contrast concepts, no training data yet

### Phase 2A: Context-Aware MELD Generation (Treatment)
Regenerate L1 MELDs with full context:
```
"You previously designed this curriculum:
- School: Economic Activity & Value Exchange
- Departments: [Banking, Trade, Labor, Taxation, Insurance]
- Sibling Schools: [Material Production, Social Organization, ...]

Now create the detailed syllabus (MELD) for this School..."
```

### Phase 2B: Baseline Comparison (Control)
Compare against L1 MELDs generated without child knowledge (our current approach).

## Metrics

1. **Structural Quality** (automated)
   - Validation pass rate
   - Example count and diversity
   - Negative example specificity (do they reference actual siblings?)

2. **Judge Assessment** (Ministral in annotator mode)
   - Divergence score
   - Worldview notes
   - Structural issues

3. **Human Evaluation** (qualitative)
   - Definition clarity
   - Example relevance
   - Disambiguation quality

## Timeline

| Date | Phase | Status |
|------|-------|--------|
| 2026-01-18 | L1 MELDs generated (baseline) | Complete - 11/13 approved |
| 2026-01-18 | L1→L2→L3 skeleton generation | In Progress |
| TBD | L1 MELD regeneration with context | Pending |
| TBD | Comparison and analysis | Pending |

## Results

### Skeleton Generation

**L1 Pillars (Schools)**: 13
- Approved: 11
- Rejected: 1 (Violence & Conflict - validation issues)
- Human Review: 1 (Risk Taking - elevated protection)

**L2 Departments**: TBD (generation in progress)

**L3 Courses**: TBD

### Baseline L1 MELDs (Without Child Context)

Generated using `generate_l1_pillars.py` with:
- Generator: Gemma 3 4B
- Judge: Ministral 8B (annotator mode)
- Context: Parent (none for L1), siblings only

Results:
- 8/13 auto-approved on first run
- 3 additional approved on retry
- All approved MELDs had divergence_score: 0.0 (judge fully aligned)

### Context-Aware L1 MELDs (With Child Context)

TBD - pending skeleton completion

### Comparison

TBD

## Observations

### Initial Observations (Pre-Experiment)

1. **Sibling context helps**: Even without children, having sibling context improved negative example generation

2. **Level-appropriate definitions**: HierarchyContext helped generator understand that L1 definitions should be broad

3. **Judge mode matters**: Gatekeeper mode rejected too many valid MELDs; annotator mode captures divergence without blocking

### Post-Experiment Observations

TBD

## Code Artifacts

- `scripts/generate_ontology_skeleton.py` - Phase 1 skeleton generation
- `scripts/generate_l1_pillars.py` - Baseline MELD generation
- `scripts/meld_review_pipeline.py` - Review pipeline with annotator mode
- `results/ontology_skeleton.json` - Skeleton structure
- `results/l1_pillars/` - Baseline MELDs

## Conclusions

TBD

## Next Steps

1. Complete skeleton generation (L1→L2→L3)
2. Create context-aware MELD generation script
3. Regenerate L1 MELDs with full context
4. Compare baseline vs context-aware outputs
5. If hypothesis confirmed, update pipeline to use skeleton-first approach for all levels
