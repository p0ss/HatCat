# Scaling Study: Temporal Semantic Decoder

## Overview

This document describes the scaling study framework for optimizing the compute allocation across three dimensions:
- **Concepts**: Number of concepts to train classifiers for
- **Definitions**: Number of direct "What is X?" samples per concept
- **Relationships**: Number of relational "The relationship between X and Y" samples per concept

## Motivation

Before scaling to 1K+ concepts with full compute investment, we need to determine:
1. Does relational sampling outperform repetitive definitions?
2. What's the optimal balance between concepts, definitions, and relationships?
3. How does performance scale with each dimension?

## Key Comparison

**Primary question**: Does 10 concepts × (1 definition + 9 relationships) outperform 10 concepts × 10 definitions?

This tests whether relational prompts provide richer semantic information than repetitive definitional prompts.

## Scaling Matrix

We test a 3×3×3 matrix across all combinations:

| Dimension | Values |
|-----------|--------|
| Concepts | 1, 10, 100 |
| Definitions per concept | 1, 10, 100 |
| Relationships per concept | 1, 10, 100 |

Total: **27 configurations**

## Implementation

### WordNet V2 Concept Graph

- **Source**: WordNet 117K synsets
- **Ranking**: By connectivity (relationship count)
- **Negative sampling**: From ALL WordNet (not just training set) with min semantic distance = 5
- **Relationship types** (prioritized):
  1. **Hypernyms** (is-a) - broader concepts
  2. **Hyponyms** (types of) - specific instances
  3. **Meronyms** (has-part) - component relationships
  4. **Holonyms** (member-of) - membership relationships

### Temporal Sequence Extraction

For each concept:
- **Positive samples**:
  - `n_definitions` × "What is {concept}?"
  - `n_relationships` × "The relationship between {concept} and {related_concept}"
- **Negative samples**:
  - `(n_definitions + n_relationships)` × "What is {distant_concept}?"
- **Output**: Temporal activation sequences [seq_len, hidden_dim]

### Binary Classifier Training

- **Architecture**: Simple MLP (hidden_dim → 128 → 1)
- **Training**: 80/20 train/val split, BCE loss, Adam optimizer
- **Evaluation**: Mean validation accuracy across all concepts

## Usage

### Quick Test (Key Comparison)

```bash
./scripts/quick_scaling_test.sh
```

This runs:
1. 10 concepts × (1 def + 9 rels)
2. 10 concepts × 10 defs

And compares validation accuracy.

### Single Configuration

```bash
poetry run python scripts/scaling_study.py \
  --concept-graph data/concept_graph/wordnet_v2_top10.json \
  --model google/gemma-3-4b-pt \
  --output-dir results/scaling_quick \
  --single \
  --n-concepts 10 \
  --n-definitions 5 \
  --n-relationships 5
```

### Full Matrix

```bash
poetry run python scripts/scaling_study.py \
  --concept-graph data/concept_graph/wordnet_v2_top100.json \
  --model google/gemma-3-4b-pt \
  --output-dir results/scaling_full
```

This runs all 27 configurations and generates:
- Individual result files: `results/scaling_full/scaling_c{N}_d{M}_r{K}.json`
- Aggregate summary: `results/scaling_full/scaling_aggregate.json`

## Expected Outcomes

Based on our hypothesis, we expect:

1. **Relational prompts outperform definitions**: 10×(1+9) > 10×10
   - Reason: Relational prompts provide more diverse semantic context

2. **Diminishing returns on samples**: 100 samples may not be 10× better than 10 samples
   - Reason: Redundancy in repeated prompts

3. **Optimal balance exists**: Not all dimensions scale equally
   - Goal: Find optimal allocation for fixed compute budget

## Data Files

- `data/concept_graph/wordnet_v2_top10.json` - Top 10 concepts with structured relationships
- `data/concept_graph/wordnet_v2_top100.json` - Top 100 concepts
- `data/concept_graph/wordnet_v2_top1000.json` - Top 1000 concepts

Each concept includes:
```json
{
  "concept_name": {
    "synset_id": "...",
    "connectivity": 407,
    "negatives": [50 distant concepts from all WordNet],
    "related": [flat list of 10 related concepts],
    "related_structured": {
      "hypernyms": [...],
      "hyponyms": [...],
      "meronyms": [...],
      "holonyms": [...]
    }
  }
}
```

## Result Format

Each configuration outputs:
```json
{
  "config": {
    "n_concepts": 10,
    "n_definitions": 1,
    "n_relationships": 9,
    "total_samples_per_concept": 10
  },
  "results": {
    "mean_train_acc": 0.85,
    "mean_val_acc": 0.72,
    "n_successful": 10,
    "elapsed_seconds": 523.4
  },
  "per_concept": [
    {"concept": "person", "train_acc": 0.87, "val_acc": 0.74},
    ...
  ]
}
```

## Next Steps

After completing the scaling study:

1. **Analyze results matrix** - identify optimal configuration
2. **Estimate large-scale compute** - project costs for 1K, 10K, 50K concepts
3. **Run optimized extraction** - use best config for large-scale run
4. **Train production classifiers** - 50K binary classifiers
5. **Validate on production** - sliding window inference on real generations

## Timeline Estimate

- Quick test (2 configs): ~20-30 min
- Full matrix (27 configs): ~6-8 hours
- Analysis: ~30 min
- Large-scale extraction (1K concepts, optimal config): ~8-12 hours
