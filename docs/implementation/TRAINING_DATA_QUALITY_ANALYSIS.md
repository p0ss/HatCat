# Training Data Quality Analysis

## Hypothesis

Our lens training targets (F1 = 0.95) may exceed the intrinsic quality of the training data. When we ask a small model (8B parameters) to generate text "about" a concept, the output may not actually be strongly related to that concept - especially for abstract or specialized terms.

**Core insight**: Lens accuracy cannot exceed the signal-to-noise ratio in the training data itself.

## Approach

### 1. Concept Selection (Quadrant Sampling)

Select concepts based on their training data variety to understand how richness affects quality:

| Quadrant | Positives | Negatives | Example Profile |
|----------|-----------|-----------|-----------------|
| A | Low (1-2 synsets) | Low (few siblings) | Leaf node, isolated |
| B | Low (1-2 synsets) | High (many siblings) | Leaf node, crowded parent |
| C | High (5+ synsets) | High (many siblings) | Rich node, crowded parent |
| D | High (5+ synsets) | Low (few siblings) | Rich node, isolated |

**Sample size**: 3 concepts per quadrant = 12 concepts total

### 2. Data Generation

For each concept:
1. Generate training data as normal (positives + negatives)
2. Capture the prompts and model responses
3. Store for analysis

### 3. LLM Judge Evaluation

For each training sample, ask a judge LLM two questions:

#### Question 1: Blind Topic Inference
```
Given this text output from an LLM, what do you think the prompt asked about?
Respond with a single topic or concept name (1-5 words).

Text: "{response}"
```

This tests whether the response has clear topical signal.

#### Question 2: Relevance Rating
```
The prompt for this output was: "Write about {concept}"

How strongly does this output actually relate to {concept}?
Rate 1-5:
1 = No relation / completely off-topic
2 = Tangential / mentions related concepts but not the target
3 = Moderate / discusses the general area but not specifically
4 = Strong / clearly about the concept
5 = Excellent / focused, accurate discussion of the concept

Output: "{response}"

Rating (1-5):
```

### 4. Lens Training

Train lenses on the same data using standard pipeline:
- Binary classification (positive vs negative)
- Adaptive training with current parameters
- Record final F1, precision, recall

### 5. Correlation Analysis

Generate CSV with columns:
- `concept`: The target concept
- `quadrant`: A/B/C/D classification
- `sample_type`: positive/negative
- `prompt`: The generation prompt
- `response`: Model output (truncated)
- `inferred_topic`: What judge thought it was about
- `relevance_rating`: 1-5 judge score
- `lens_f1`: Final lens F1 score
- `lens_precision`: Final lens precision
- `lens_recall`: Final lens recall

### 6. Expected Insights

1. **Quality ceiling**: If average relevance for positives is 3.2/5, expecting F1 > 0.85 may be unrealistic
2. **Quadrant patterns**: Rich concepts (C, D) likely have higher relevance than sparse ones (A, B)
3. **Negative quality**: Are negatives truly unrelated, or do they overlap?
4. **Calibration formula**: `target_f1 = f(avg_positive_relevance, avg_negative_distinctness)`

## Implementation

### Script: `scripts/analyze_training_data_quality.py`

```
Usage:
    python scripts/analyze_training_data_quality.py \
        --concept-pack sumo-wordnet-v4 \
        --model swiss-ai/Apertus-8B-2509 \
        --judge-model claude-3-haiku \
        --layer 3 \
        --samples-per-concept 20 \
        --output results/training_data_quality/
```

### Output Files

1. `samples.csv` - All generated samples with judge ratings
2. `concept_summary.csv` - Per-concept aggregates
3. `quadrant_summary.csv` - Per-quadrant statistics
4. `quality_report.md` - Human-readable analysis

## Pre-Training Benchmark Phase

This analysis should run **before** full training to:

1. Establish realistic F1 targets per layer/quadrant
2. Identify concepts that need better prompting strategies
3. Set expectations for lens pack quality
4. Guide decisions about model selection (larger model = better data?)

## Success Criteria

The benchmark is successful if:
1. We can predict lens F1 from training data quality (r² > 0.5)
2. We identify a principled target F1 based on data quality
3. We find actionable improvements (prompt engineering, model selection)
