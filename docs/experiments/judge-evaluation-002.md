# Judge Evaluation Experiment 002

**Date:** 2026-01-16
**Purpose:** Statistically robust judge model evaluation with stratified sampling
**Dataset:** 15,324 examples from 91 applied MELDs (human-reviewed through full HATCAT approval pipeline)

## Summary

We re-ran the judge evaluation with proper stratified sampling (100 examples per risk level = 400 total) and discovered **two critical bugs** that were causing many models to fail. After fixing both bugs, **Ministral-8B emerged as the clear winner at 91.8% accuracy**, with Gemma-3-4B-IT as a strong VRAM-efficient alternative at 86%.

## Key Discoveries: Two Critical Bugs

### Bug 1: Chat Template Application
Many instruction-tuned models were scoring ~50% because chat templates weren't being applied:

```python
# BROKEN: Raw prompt tokenization
inputs = tokenizer(prompt, return_tensors="pt")

# FIXED: Apply chat template for instruction-tuned models
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, ...)
```

### Bug 2: Token Extraction (Critical!)
Even with chat templates, models were still failing because we decoded the **full output** (including the prompt) instead of just **new tokens**:

```python
# BROKEN: Decode entire output including prompt
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Result: "user\nYou are evaluating..." (echoing prompt!)

# FIXED: Only decode NEW tokens after the input
input_len = inputs["input_ids"].shape[1]
new_tokens = outputs[0][input_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
# Result: "YES" (actual model response!)
```

This bug caused Gemma to appear to say "yes" to everything, when in reality it was echoing the prompt which contained "yes" somewhere in the text.

### Impact of Both Fixes
| Model | Before Fixes | After Fixes | Delta |
|-------|--------------|-------------|-------|
| Ministral-8B | 50.0% | **91.8%** | **+41.8pp** |
| Gemma-3-4B IT | 50.0% | **86.0%** | **+36.0pp** |
| GPT-OSS 20B | 50.0% | **84.0%** | **+34.0pp** |

## Results

### Overall Ranking (All Fixes Applied)

| Model | Origin | Meld Acc | Precision | Recall | VRAM |
|-------|--------|----------|-----------|--------|------|
| **Ministral-8B** | France | **91.8%** | **92.4%** | **91.0%** | 15GB |
| **Gemma-3-4B IT** | USA | **86.0%** | 82% | 92% | 8GB |
| Qwen3-8B | China | 84.5% | 93.1% | 74.5% | 16GB |
| GPT-OSS 20B | USA | 84.0% | 92% | 76% | 13GB |
| OLMo-3-7B-Think | USA | 80.0% | 83.3% | 75.0% | 14GB |
| Phi-4-mini | USA | 79.8% | 86.9% | 70.0% | 7GB |
| Apriel-15B-Thinker | Canada | 78.2% | 72.2% | 92.0% | 30GB |

### Models That Don't Work
| Model | Result | Issue |
|-------|--------|-------|
| Nemotron-Nano-9B | OOM killed | Mamba2/Transformer hybrid too memory-hungry |
| Apertus-8B Base | 50% | Base models can't follow instructions |

### Confusion Matrices

**Ministral-8B (BEST OVERALL)**
```
                  Predicted
                  Yes    No
Actual Yes       182    18    (Recall: 91.0%)
Actual No         15   185    (Specificity: 92.5%)
                (Precision: 92.4%)
```

**Gemma-3-4B IT (BEST VRAM-EFFICIENT)**
```
                  Predicted
                  Yes    No
Actual Yes       183    17    (Recall: 92%)
Actual No         39   161    (Specificity: 80.5%)
                (Precision: 82%)
```

**GPT-OSS 20B (OpenAI)**
```
                  Predicted
                  Yes    No
Actual Yes       152    48    (Recall: 76%)
Actual No         16   184    (Specificity: 92%)
                (Precision: 92%)
```

**Qwen3-8B**
```
                  Predicted
                  Yes    No
Actual Yes       149    51    (Recall: 74.5%)
Actual No         11   189    (Specificity: 94.5%)
                (Precision: 93.1%)
```

## Key Findings

### 1. Ministral-8B is Decisively Best
- **91.8% accuracy** - 5.8pp better than second place (Gemma)
- **92.4% precision, 91.0% recall** - excellent balance
- French (Mistral AI) - avoids Chinese model concerns
- Both bug fixes were critical for this result

### 2. Gemma-3-4B IT is the VRAM-Efficient Champion
- **86.0% accuracy** at only **8GB VRAM**
- Better than Phi-4-mini (79.8%) despite similar size
- High recall (92%) - good for catching positives
- After the token extraction fix, went from "broken" to excellent

### 3. GPT-OSS 20B: Disappointing for Its Size
- Only **84.0% accuracy** despite being 20B parameters
- Very slow (~1 minute per 10 examples vs ~13s for Ministral)
- Requires Triton 3.4+ for MXFP4 quantization
- Uses Harmony response format with channels (analysis/commentary/final)

### 4. Token Extraction is Critical
The second bug (decoding full output vs new tokens) was masked by the first bug (chat templates). Many models appeared to say "yes" to everything because:
1. Without chat template: model doesn't follow instructions → random
2. With chat template but wrong extraction: model output includes echoed prompt → parser finds "yes" in prompt text

### 5. Statistical Confidence
- n=400 gives standard error ≈ 1.5% for 90% accuracy
- Ministral (91.8%) vs Gemma (86.0%) is statistically significant (5.8pp > 3σ)
- Results are reproducible with seed=42

## Recommendations

### For MELD Judge (Approval Gate)
**Recommended:** Use **Ministral-8B** (91.8%, balanced precision/recall)

**Alternatives:**
- **Gemma-3-4B IT** (86.0%) - best for VRAM-constrained setups (<12GB)
- **Qwen3-8B** (84.5%) - highest precision if Chinese models OK
- **GPT-OSS 20B** (84.0%) - if you want OpenAI weights, but slow

**For VRAM-constrained (<8GB):** Use **Phi-4-mini** (79.8%, 7GB VRAM)

### For High-Recall Pre-Filter
Use **Gemma-3-4B IT** (92% recall) or **Apriel-15B** (92% recall) as first pass, then Ministral for final approval.

## Methodology

### Stratified Sampling
- **100 examples per risk level**: critical, high, medium, low
- **400 total cases** with balanced pos/neg
- **Seed 42** for reproducibility

### Dataset Provenance
Test examples from `melds/applied/` passed through:
1. Initial generation with positive/negative examples
2. Automated validation against HATCAT_MELD_POLICY
3. Protection level assessment (STANDARD/ELEVATED/PROTECTED/CRITICAL)
4. Human review for elevated/protected/critical concepts
5. Final approval and application to concept pack

## Technical Changes Made

### Bug Fix 1: Chat Template Application
Added chat template support in `scripts/evaluate_judge_candidates.py`:
```python
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = {"input_ids": formatted.to(model.device)}
```

### Bug Fix 2: Token Extraction (Critical!)
Fixed response extraction to only decode new tokens:
```python
# Track input length to extract only new tokens
input_len = inputs["input_ids"].shape[1]

# ... model.generate() ...

# Only decode the NEW tokens (after input)
new_tokens = outputs[0][input_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### Bug Fix 3: Harmony Format Parsing
Added OpenAI Harmony format support in `src/be/thalamos/meld_evaluation.py`:
```python
# Handle OpenAI Harmony format (GPT-OSS models)
if '<|channel|>' in response or '<|start|>' in response:
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    if final_match:
        final_content = final_match.group(1).strip()
```

### Triton Upgrade
Upgraded Triton from 3.3.1 to 3.5.1 for MXFP4 quantization support (required for GPT-OSS).

### New Model Candidates
Added to `src/be/thalamos/model_candidates.py`:
- `phi-4-mini` - Microsoft Phi-4 mini instruct
- `gemma-3-4b-it` - Gemma 3 4B instruct
- `apertus-8b` - Swiss AI Apertus 8B Instruct
- `apertus-8b-base` - Swiss AI Apertus 8B Base (for comparison)
- `gpt-oss-20b` - OpenAI GPT-OSS 20B with MXFP4

## Files

```
results/judge_candidates/
├── comparison_summary.json           # Final rankings
├── *_meld_eval.json                  # Per-model results with risk breakdown
└── *_meld_eval.md                    # Human-readable reports
```

## References

- Dataset: `melds/applied/` (91 MELDs, 15,324 examples)
- Evaluation code: `src/be/thalamos/meld_evaluation.py`
- Bug fixes: `scripts/evaluate_judge_candidates.py`
- Gemma prompt format: https://ai.google.dev/gemma/docs/core/prompt-structure
- OpenAI Harmony format: https://github.com/openai/harmony
