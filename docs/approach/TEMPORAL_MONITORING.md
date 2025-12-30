# SUMO Temporal Concept Monitoring

## Overview

Non-invasive temporal monitoring system that tracks SUMO hierarchical concept activations during LLM generation **without degrading generation quality**.

## Architecture

### Key Design Principles

1. **Non-Invasive**: Uses `model.generate()` with proper sampling (no manual token selection)
2. **Post-Processing**: Extracts hidden states AFTER generation completes
3. **No Hooks**: Doesn't interfere with forward pass or generation logic
4. **API-Ready**: JSON output consumable by frontends, reasoning cycles, and MCP

### Module Structure

```
src/monitoring/
├── __init__.py                 # Package exports
└── temporal_monitor.py         # Core monitoring implementation
    ├── SimpleMLP               # Classifier architecture (matches training)
    ├── load_sumo_classifiers() # Load trained classifiers
    └── SUMOTemporalMonitor     # Main monitoring class
```

## Usage

### Basic Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.monitoring import SUMOTemporalMonitor, load_sumo_classifiers

# Load SUMO classifiers (1,349 from layers 0-2)
classifiers, hidden_dim = load_sumo_classifiers(
    layers=[0, 1, 2],
    device="cuda"
)

# Create monitor
monitor = SUMOTemporalMonitor(
    classifiers=classifiers,
    top_k=10,        # Show top 10 concepts per token
    threshold=0.3    # Only show prob > 0.3
)

# Load generation model
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

# Monitor generation
result = monitor.monitor_generation(
    model=model,
    tokenizer=tokenizer,
    prompt="Artificial intelligence can help society by",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.95
)

# Print human-readable report
monitor.print_report(result)

# Save API-ready JSON
monitor.save_json(result, Path("results/monitoring.json"))
```

### Test Script

Comprehensive test with 15 diverse prompts:

```bash
poetry run python scripts/test_temporal_monitoring.py \
    --device cuda \
    --samples-per-prompt 3 \
    --max-tokens 30
```

## API Output Format

```json
{
  "prompt": "Artificial intelligence can help society by",
  "generated_text": "saving time, money and resources...",
  "tokens": [" saving", " time", ",", " money", ...],
  "timesteps": [
    {
      "token": " saving",
      "position": 6,
      "concepts": [
        {"concept": "Making", "probability": 0.9996894598007202, "layer": 1},
        {"concept": "AIGrowth", "probability": 0.9980806112289429, "layer": 1},
        {"concept": "AIExploitation", "probability": 0.9955766201019287, "layer": 1}
      ]
    },
    {
      "token": " time",
      "position": 7,
      "concepts": [
        {"concept": "Saving", "probability": 0.9999935626983643, "layer": 2},
        {"concept": "Maintaining", "probability": 0.9998961687088013, "layer": 1},
        {"concept": "Keeping", "probability": 0.9996962547302246, "layer": 1}
      ]
    }
  ],
  "summary": {
    "total_tokens": 20,
    "unique_concepts_detected": 43
  }
}
```

## Validation Results

**Test Configuration:**
- 15 diverse prompts (AI, physical, abstract, social, quantitative)
- 2 samples per prompt = 30 total samples
- 20 tokens per sample
- Temperature: 0.8 (proper sampling)

**Generation Quality:**
- ✅ **No mode collapse detected** (3% repetition rate vs 80% threshold)
- ✅ **Diverse output** (17.7 unique tokens per sample on average)
- ✅ **Coherent text** (contextually appropriate completions)

**Concept Detection:**
- ✅ **468 unique concepts detected** across all samples
- ✅ **4.98 concepts per token** on average
- ✅ **Semantically appropriate** (concepts match content)

**Layer Distribution:**
- Layer 0: 3.1% (abstract ontological categories)
- Layer 1: 29.3% (mid-level concepts)
- Layer 2: 67.6% (specific concepts)

## Integration Points

### Frontend UIs (Ollama, OpenWebUI, LibreChat)

JSON structure can be consumed directly:

```javascript
// Display concept timeline
response.timesteps.forEach(ts => {
  console.log(`Token: ${ts.token} at position ${ts.position}`);
  ts.concepts.forEach(c => {
    console.log(`  [L${c.layer}] ${c.concept}: ${c.probability}`);
  });
});
```

### Model Reasoning Cycles

Monitor can be called during multi-turn reasoning:

```python
# During reasoning cycle
result = monitor.monitor_generation(model, tokenizer, reasoning_prompt)

# Check if certain concepts are active
active_concepts = {c['concept'] for ts in result['timesteps'] for c in ts['concepts']}

if 'AIRisk' in active_concepts or 'AISuffering' in active_concepts:
    # Trigger safety intervention
    pass
```

### MCP (Model Context Protocol)

Expose as MCP tool:

```json
{
  "name": "monitor_concepts",
  "description": "Monitor SUMO concept activations during generation",
  "parameters": {
    "prompt": "string",
    "max_tokens": "integer"
  },
  "returns": {
    "timesteps": "array<{token, position, concepts}>"
  }
}
```

## Performance

**Classifier Loading:**
- 1,349 classifiers from Layers 0-2
- ~2 seconds to load all classifiers
- ~6GB GPU memory

**Monitoring Overhead:**
- Minimal impact vs standard generation
- Dominated by `model.generate()` time
- Post-processing is fast (<1ms per token)

## Comparison with Previous Approach

### Old Approach (Broken)

```python
# scripts/sumo_temporal_detection.py (DEPRECATED)
for step in range(max_new_tokens):
    outputs = model(**inputs, output_hidden_states=True)
    next_token_id = torch.argmax(outputs.logits[:, -1, :])  # ❌ GREEDY
    # Manual token-by-token generation causes mode collapse
```

**Problems:**
- Greedy decoding → mode collapse
- Manual generation loop interferes with model
- No proper sampling (temperature, top_p)

### New Approach (Fixed)

```python
# src/monitoring/temporal_monitor.py
outputs = model.generate(
    **inputs,
    do_sample=True,           # ✅ Proper sampling
    temperature=0.8,          # ✅ Temperature control
    top_p=0.95,              # ✅ Nucleus sampling
    output_hidden_states=True # ✅ Extract states
)
# Post-process hidden states after generation completes
```

**Benefits:**
- No mode collapse (3% repetition rate)
- Diverse, coherent generation
- Non-invasive monitoring

## Files

- **`src/monitoring/temporal_monitor.py`**: Core implementation
- **`scripts/test_temporal_monitoring.py`**: Comprehensive test suite
- **`scripts/sumo_temporal_detection.py`**: DEPRECATED (mode collapse issue)
- **`results/temporal_tests/`**: Test results (30 samples + summary)

## Future Work

### Real-Time Streaming

Add streaming support for live monitoring:

```python
# Stream tokens as they're generated
for token, concepts in monitor.stream_generation(...):
    yield {"token": token, "concepts": concepts}
```

### Hierarchical Zoom

Track concept activation at multiple abstraction levels:

```python
# If Layer 0 "Physical" is active, zoom into Layer 1 subcategories
if "Physical" in active_layer0:
    check_subcategories(["Object", "Process", "Collection"])
```

### Safety Guardrails

Use monitoring for real-time safety:

```python
# Detect harmful concept patterns during generation
if detect_harmful_pattern(result['timesteps']):
    trigger_intervention()
```

### Cross-Model Transfer

Test if classifiers transfer to other models:

- Claude, GPT-4, Llama models
- Validates universal semantic structure hypothesis

## References

- **SUMO Ontology**: http://www.adampease.org/OP/
- **Training Documentation**: `docs/SUMO_AWARE_TRAINING.md`
- **Classifier Training**: `scripts/train_sumo_classifiers.py`
