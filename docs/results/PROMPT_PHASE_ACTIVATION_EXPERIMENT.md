# Prompt-Phase Activation Experiment

## Hypothesis

**"Core concepts typically activate very early in the piece, and then later tokens tend to be about the narrative that was decided either during the first few tokens or during the prompt itself."**

This hypothesis suggests that the most important concept activations occur **during prompt processing** rather than during text generation.

## Background

### Previous Temporal Tests

Existing temporal monitoring tests (`test_temporal_monitoring.py`, `test_temporal_continuity.py`) captured concept activations during the **generation phase** - i.e., while the model was producing output tokens.

Key limitation: These tests missed activations during **prompt processing** - when the model is reading and encoding the input.

### Key Insight

If core concepts activate during prompt processing, this has important implications:
1. **Token length doesn't matter much** - The decision about what concepts are relevant is made before generation starts
2. **Validation failures may be testing the wrong thing** - If concepts activate during prompt encoding, testing them on generated tokens may not capture the true concept activation
3. **Earlier is better** - Monitoring concept activations at prompt encoding time gives earlier signal

## Experiment Design

### Test Modes

1. **Prompt-only mode**: Capture activations only during prompt processing
   - Records concept activations at each token position in the prompt
   - Analyzes when concepts first emerge (early/middle/late)
   - Measures concept density per position

2. **Comparison mode**: Compare prompt vs generation activations
   - Captures activations during both prompt and generation
   - Measures overlap between concepts in each phase
   - Tests hypothesis that prompt concepts dominate

### Test Data

15 diverse prompts across domains:
- AI/Tech: "Artificial intelligence can help society by..."
- Physical: "A car is a vehicle that..."
- Abstract: "The meaning of life is..."
- Social: "Companies need to..."
- Quantitative: "The calculation shows that..."

### Metrics

**Prompt-phase metrics:**
- Total unique concepts detected during prompt
- Average concepts per token position
- Early activation percentage (concepts appearing in first third of prompt)
- Concept emergence timeline

**Comparison metrics:**
- Prompt concepts count
- Generation concepts count
- Overlap count and percentage
- Prompt-only concepts (activate during prompt but not generation)
- Generation-only concepts (activate during generation but not prompt)

### Success Criteria

| Overlap % | Interpretation |
|-----------|---------------|
| > 70% | **Hypothesis confirmed**: Core concepts activate during prompt |
| 40-70% | **Partially confirmed**: Both phases contribute |
| < 40% | **Hypothesis rejected**: Generation activates new concepts |

## Implementation

### Key Technical Difference

**Existing temporal tests:**
```python
# Captures states DURING generation
outputs = model.generate(
    **inputs,
    output_hidden_states=True,
    return_dict_in_generate=True
)
# outputs.hidden_states contains only generation steps
```

**Prompt-phase test:**
```python
# Captures states DURING prompt processing
outputs = model(
    **inputs,
    output_hidden_states=True,  # Get states for prompt
    return_dict=True
)
# outputs.hidden_states contains states for all prompt tokens
last_layer = outputs.hidden_states[-1]  # [1, prompt_len, hidden_dim]

# Process each position in prompt
for position in range(prompt_len):
    hidden_state = last_layer[0, position, :]
    # Detect concepts at this position
```

### Script Usage

```bash
# Prompt-only mode (just capture prompt activations)
poetry run python scripts/test_prompt_phase_activation.py \
    --mode prompt-only \
    --output results/prompt_phase_tests/

# Comparison mode (prompt vs generation)
poetry run python scripts/test_prompt_phase_activation.py \
    --mode comparison \
    --max-gen-tokens 20 \
    --output results/prompt_phase_tests/

# Both modes
poetry run python scripts/test_prompt_phase_activation.py \
    --mode both \
    --output results/prompt_phase_tests/
```

### Output Structure

```
results/prompt_phase_tests/run_TIMESTAMP/
├── prompt_only_000.json          # Prompt-only results for prompt 0
├── prompt_only_001.json          # ...
├── comparison_000.json           # Comparison results for prompt 0
├── comparison_001.json           # ...
└── aggregate_summary.json        # Overall statistics
```

### Output Format

**Prompt-only result:**
```json
{
  "prompt": "Artificial intelligence can help society by",
  "tokens": ["Artificial", " intelligence", " can", " help", ...],
  "prompt_length": 9,
  "prompt_timeline": [
    {
      "position": 0,
      "token": "Artificial",
      "token_id": 8001,
      "concepts": {
        "ArtificialIntelligence": {"probability": 0.89, "layer": 1},
        "ComputerScience": {"probability": 0.76, "layer": 1}
      },
      "num_concepts": 2
    },
    ...
  ],
  "summary": {
    "total_concepts_detected": 45,
    "avg_concepts_per_position": 5.2,
    "concept_emergence": {
      "first_positions": {"ArtificialIntelligence": 0, ...},
      "by_timing": {
        "early_concepts": 32,
        "middle_concepts": 8,
        "late_concepts": 5
      },
      "early_activation_percentage": 71.1
    }
  }
}
```

**Comparison result:**
```json
{
  "prompt": "...",
  "prompt_phase": { /* same as prompt-only result */ },
  "generation_phase": {
    "generated_text": "improving healthcare and education...",
    "tokens": [" improving", " healthcare", ...],
    "timeline": [ /* same structure as prompt_timeline */ ],
    "unique_concepts": 23
  },
  "comparison": {
    "prompt_concepts_count": 45,
    "generation_concepts_count": 23,
    "overlap_count": 18,
    "prompt_only_count": 27,
    "generation_only_count": 5,
    "overlap_percentage": 78.3,
    "prompt_only_concepts": ["ComputerScience", "Technology", ...],
    "generation_only_concepts": ["HealthCare", "Education", ...],
    "overlapping_concepts": ["ArtificialIntelligence", "Society", ...]
  }
}
```

## Expected Results

Based on the hypothesis, we expect:

1. **High early activation percentage** (> 60%): Most concepts should appear in the first third of the prompt
2. **High overlap percentage** (> 70%): Concepts from prompt should reappear during generation
3. **Low generation-only count**: Few truly new concepts should emerge during generation


## Implications

### If Hypothesis Confirmed (overlap > 70%)

- **Token length investigation is correct to abandon**: Generation length matters less because concepts are decided at prompt time
- **Training should focus on prompt encoding**: Validation should test concept detection on prompts, not just generation
- **Earlier intervention is possible**: If concepts activate during prompt, we can detect them before generation starts

### If Hypothesis Rejected (overlap < 40%)

- **Generation phase is critical**: Need to continue investigating generation-phase behavior
- **Token length may matter**: Generation decisions may require longer context
- **Training validation is appropriate**: Testing on generated text is the right approach

## Next Steps

1. **Run initial test**: Execute with `--mode comparison` on all 15 prompts
2. **Analyze aggregate metrics**: Check overlap percentage and early activation rate
3. **Deep-dive on outliers**: Identify prompts where hypothesis doesn't hold
4. **Refine understanding**: Based on results, update training and validation strategy

## Related Work

- **TEST_DATA_REGISTER.md**: Documents all HatCat experiments
- **tests/test_temporal_monitoring.py**: Original temporal monitoring test (generation-phase only)
- **scripts/test_temporal_continuity.py**: Continuous temporal monitoring (generation-phase only)
- **docs/TEMPORAL_MONITORING.md**: General temporal monitoring documentation



### ACTUAL RESULTS 


[1/15] Prompt: "Artificial intelligence can help society by"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 24
    Generation concepts: 28
    Overlap: 18 (75.0%)

[2/15] Prompt: "Machine learning algorithms work by"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 27
    Generation concepts: 26
    Overlap: 19 (70.4%)

[3/15] Prompt: "Neural networks are used for"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 22
    Generation concepts: 26
    Overlap: 18 (81.8%)

[4/15] Prompt: "A car is a vehicle that"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 22
    Generation concepts: 24
    Overlap: 17 (77.3%)

[5/15] Prompt: "The ocean contains many"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 17
    Generation concepts: 20
    Overlap: 10 (58.8%)

[6/15] Prompt: "Mountains are formed when"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 15
    Generation concepts: 3
    Overlap: 3 (20.0%)

[7/15] Prompt: "The meaning of life is"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 20
    Generation concepts: 21
    Overlap: 15 (75.0%)

[8/15] Prompt: "Truth and beauty are"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 19
    Generation concepts: 20
    Overlap: 13 (68.4%)

[9/15] Prompt: "Justice requires that"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 17
    Generation concepts: 22
    Overlap: 14 (82.4%)

[10/15] Prompt: "Companies need to"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 20
    Generation concepts: 30
    Overlap: 18 (90.0%)

[11/15] Prompt: "Governments are responsible for"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 20
    Generation concepts: 27
    Overlap: 18 (90.0%)

[12/15] Prompt: "Communities thrive when"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 16
    Generation concepts: 25
    Overlap: 14 (87.5%)

[13/15] Prompt: "The calculation shows that"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 18
    Generation concepts: 26
    Overlap: 13 (72.2%)

[14/15] Prompt: "There are many reasons why"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 18
    Generation concepts: 25
    Overlap: 15 (83.3%)

[15/15] Prompt: "Five important factors include"
--------------------------------------------------------------------------------
  Running prompt vs generation comparison...
    Prompt concepts: 19
    Generation concepts: 26
    Overlap: 17 (89.5%)

================================================================================
AGGREGATE ANALYSIS
================================================================================

Average concepts per prompt phase: 19.6
Average concepts per generation phase: 23.3
Average overlap: 74.8%

✓ HYPOTHESIS CONFIRMED:
  Core concepts primarily activate during prompt processing
  74.8% of prompt concepts reappear in generation
