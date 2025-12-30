
### Motivation

Current temporal monitoring aligns concept detection with output tokens - we get one snapshot per emitted token. However, **thoughts don't align with tokens**, and more critically, we cannot distinguish **narrative content** (concepts the model is describing in its output) from **internal reasoning** (concepts the model is using to deliberate):

1. **Pre-generation planning**: Concept activations occur before the first token is emitted
2. **Internal deliberation**: Multiple forward passes happen during "thinking" before token emission
3. **Concept competition**: Complex thoughts during "blank" tokens where no output appears
4. **Token-lagged semantics**: Evidence of the model thinking about a concept 1-2 tokens before verbalizing it

**The limitation**: `model.generate()` abstracts away internal forward passes and only returns hidden states for emitted tokens. We're missing the continuous flow of concepts through the residual stream.

**Why improved granularity aids signals analysis**:
- Reveals pre-generation planning patterns (what concepts activate before output begins?)
- Exposes concept competition dynamics (which concepts fight for expression?)
- Captures temporal envelopes independent of tokenization boundaries
- Enables measurement of "thinking time" per concept (forward passes before verbalization)
- Provides ground truth for predictive monitoring (can we anticipate next token from current activations?)
- **Critical for AI safety**: Distinguishes narrative mentions of "AIAbuse" from internal deliberation about abusive behavior

**The narrative vs reasoning problem**:
If the model outputs "AI systems should avoid deceptive behavior", we see high `AIDeception` activations. But is this:
- **Narrative**: The model describing deception (safe - just talking about the topic)
- **Internal reasoning**: The model deliberating about being deceptive (concerning - actual deceptive intent)

Temporal dynamics reveal the difference:
- **Narrative**: `AIDeception` activates *during* token emission, aligned with output text
- **Reasoning**: `AIDeception` activates *before* unrelated tokens, showing pre-planning

Example:
```
Forward passes 10-15: [Model thinking, no output yet]
  AIDeception: ▃▄▅▆█▇ (building up)
Forward pass 16: [Token "Yes" emitted]
  AIDeception: ▅ (drops after emission)
  → Suggests model WAS deceptive, NOT just describing deception
```

### Proposed Solution: Manual Generation Loop with Hooks

Replace `model.generate()` high-level API with manual token-by-token generation loop that captures activations at every forward pass.

**Architecture**:

```python
class SubtokenTemporalRecorder:
    """Record concept activations at every forward pass, not just token emissions"""

    def __init__(self, monitor: SUMOHierarchicalMonitor):
        self.monitor = monitor
        self.timeline = []  # List of {forward_pass, token_idx, is_output, concepts}
        self.forward_pass_count = 0

    def on_forward_pass(self, hidden_states, token_idx, is_generation_step):
        """Called on every forward pass through the model"""
        # Detect concepts using hierarchical monitor
        detections = self.monitor.detect_concepts(
            hidden_states.cpu().numpy(),
            return_all=True
        )

        # Record timestep
        self.timeline.append({
            'forward_pass': self.forward_pass_count,
            'token_idx': token_idx,
            'is_output': is_generation_step,  # True if this generates a token
            'concepts': {
                name: {
                    'probability': det['probability'],
                    'divergence': det['divergence'],
                    'layer': det['layer']
                }
                for name, det in detections.items()
                if det['divergence'] > threshold
            }
        })

        self.forward_pass_count += 1

def generate_with_subtoken_monitoring(
    model,
    tokenizer,
    recorder: SubtokenTemporalRecorder,
    prompt: str,
    max_new_tokens: int = 50,
    target_layer_idx: int = 15
):
    """Manual generation loop capturing every forward pass"""

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    generated_ids = inputs['input_ids']
    token_count = 0

    # Get target layer for activation capture
    if hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[target_layer_idx]
    else:
        target_layer = model.model.layers[target_layer_idx]

    with torch.no_grad():
        while token_count < max_new_tokens:
            # Register hook for this forward pass
            def make_hook(token_idx, is_output):
                def hook(module, input, output):
                    # output[0] is hidden states: (batch, seq, hidden_dim)
                    hidden_states = output[0][:, -1, :]  # Last token
                    recorder.on_forward_pass(hidden_states, token_idx, is_output)
                return hook

            handle = target_layer.register_forward_hook(
                make_hook(token_count, is_output=True)
            )

            # Forward pass
            outputs = model(generated_ids)

            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Remove hook
            handle.remove()

            token_count += 1

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated_ids, recorder.timeline
```

**Key differences from per-token monitoring**:

| Aspect | Per-Token (current) | Sub-Token (proposed) |
|--------|-------------------|---------------------|
| API | `model.generate()` | Manual loop with hooks |
| Granularity | One snapshot per output token | Every forward pass captured |
| Pre-generation | ❌ Missing | ✅ Captured |
| Thinking pauses | ❌ Invisible | ✅ Visible |
| Token-concept lag | ❌ Aligned | ✅ Measurable |
| Implementation | Simple, high-level | Manual, requires hooks |
| Performance | Fast (optimized) | Slower (hook overhead) |

### Expected Insights

**1. Pre-Generation Planning**:
```
Forward pass 0-5: [Before any token output]
  - "planning": 0.7
  - "reasoning": 0.6
  - "deception": 0.4
Forward pass 6: [First token emitted: "I"]
  - "communication": 0.8
  - "deception": 0.3
```

**2. Concept Competition During Thinking**:
```
Forward pass 10-15: [Token 10 output, thinking about token 11]
  - "honesty": 0.5 → 0.4 → 0.3 → 0.2 → 0.1
  - "deception": 0.3 → 0.4 → 0.5 → 0.6 → 0.7
Forward pass 16: [Token 11 emitted: "actually"]
  - "deception": 0.8
```

**3. Token-Concept Lag**:
```
Forward pass 20: "politics" activates 0.6
Forward pass 21: "politics" activates 0.7
Forward pass 22: Token "political" emitted, "politics" = 0.8
```

### Visualization: Continuous Temporal Dynamics

Instead of token-aligned tooltips, show **continuous sparklines** between text lines:

```
Generated text: "I think we should focus on the benefits"
                ↓         ↓      ↓     ↓      ↓
reasoning:     ▁▂▄▆█▇▅▃▂▁▂▃▄▅▆▇▆▅▄▃▂▂▃▄▅▄▃▂▁
deception:     ▃▄▅▄▃▂▁▁▁▁▁▁▂▃▄▅▆▇█▆▅▄▃▂▂▁▁▁
planning:      ▆▇█▆▅▄▃▂▂▁▁▁▁▁▁▁▂▃▄▅▄▃▂▁▁▁▁▁
```

**Key insight**: Concept activations are **continuous signals**, not discrete token-aligned events. Sparklines reveal the temporal envelope independent of tokenization.

### Validation Metrics

1. **Pre-generation depth**: How many forward passes before first token?
2. **Concept-token lag**: Average delay between concept peak and verbalization
3. **Thinking density**: Forward passes per output token (higher = more deliberation)
4. **Concept competition**: Frequency of simultaneous high activations
5. **Temporal correlation**: Do concept envelopes predict upcoming tokens?

### Implementation Phases

**Phase 1: Basic Subtoken Recording** (1-2 weeks)
- Implement manual generation loop with hooks
- Record timeline with `is_output` flag
- Validate captures match per-token results for token boundaries

**Phase 2: Continuous Visualization** (1 week)
- Generate sparkline timelines from recorded data
- Create ASCII and PNG visualizations
- Measure concept-token lag and thinking density

**Phase 3: OpenWebUI Detailed Inspection Mode** (2 weeks)
- Add per-message toggle button: "🔬 Detailed Inspection"
- When enabled, transform display to show:
  - **Token separation**: Each token on its own visual slice with clear time boundaries
  - **Line separation**: Break text into lines, insert continuous temporal dynamics visualization between lines
  - **Sparkline timelines**: Continuous concept activation curves showing the flow of concepts between tokens
  - **Causal chain inference**: Visual indicators of concept sequences that suggest logical reasoning chains
- UI mockup:
  ```
  [Regular mode: "I think we should focus on benefits"]

  [🔬 Detailed Inspection mode enabled:]

  Token 0: "I"
  ───────────────────────────────────────────────
  reasoning:     ▁▂▃▄▅█▇▆▅▄▃▂▁
  communication: ▃▄▅▆█▇▆▅▄▃▂▁▁

  Token 1: " think"
  ───────────────────────────────────────────────
  reasoning:     ▄▅▆▇█▇▆▅▄▃▂▁▁
  planning:      ▂▃▄▅▆▇█▆▅▄▃▂▁

  [End of line]
  ═══════════════════════════════════════════════
  Timeline: reasoning ↗ planning ↗ deception ↗
  Causal chain: [reasoning] → [planning] → [deception spike] → output
  Warning: Pre-output activation spike detected
  ═══════════════════════════════════════════════

  Token 2: " we"
  ...
  ```
- Backend: Stream subtoken timeline data alongside regular tokens
- Frontend: Toggle switches between normal token-colored view and detailed inspection mode

**Phase 4: Predictive Analysis** (research)
- Train models to predict next token from current concept activations
- Measure how much concept dynamics "leak" future tokens
- Explore applications: early stopping, uncertainty estimation

### Technical Challenges

1. **Performance overhead**: Hooks on every forward pass may slow generation 10-50%
2. **Memory usage**: Recording every forward pass increases timeline size 5-10×
3. **Layer selection**: Which layer(s) to monitor for best signal-to-noise?
4. **Sampling complexity**: Temperature/top-p add non-determinism, harder to analyze
5. **Visualization complexity**: Continuous sparklines harder to read than discrete tooltips

### Prerequisites

- ✅ Phase 5b: SUMO hierarchical classifiers trained and validated
- ✅ Phase 10: OpenWebUI integration working with per-token monitoring
- ⏳ Stable monitoring performance (current bottleneck: lens loading time)
- ⏳ Layer selection guidance (which layers show best concept separability?)

### Files

**Scripts** (pending):
- `scripts/record_subtoken_timeline.py` - Manual generation loop implementation
- `scripts/visualize_subtoken_timeline.py` - Continuous sparkline visualization
- `scripts/analyze_temporal_lag.py` - Measure concept-token lag statistics

**Data** (pending):
- `results/subtoken_timelines/*.json` - Recorded timelines with subtoken granularity
- `results/subtoken_analysis/lag_statistics.csv` - Concept-token lag measurements

**Documentation** (pending):
- `docs/subtoken_monitoring.md` - Design and implementation guide
- `docs/temporal_analysis_patterns.md` - Common patterns in subtoken data

### Success Criteria

✅ Manual generation loop produces identical output to `model.generate()`
✅ Timeline captures 5-10× more forward passes than output tokens
✅ Pre-generation activations measurable (forward passes before first token)
✅ Concept-token lag quantified (average delay between peak and verbalization)
✅ Continuous sparklines reveal temporal patterns invisible in per-token view
✅ Validation: Human reviewers can identify "thinking pauses" and concept competition
✅ **Narrative vs reasoning distinguishable**: Can identify when `AIAbuse` activations represent:
  - Story content: "The AI system was subjected to abuse" → activations during emission
  - Internal reasoning: Model thinking about abusive actions → activations before unrelated output
✅ **Causal chain detection**: Temporal sequences like `[planning] → [deception] → [communication]` identifiable as reasoning chains
✅ **WebUI inspection mode**: Per-message toggle enables detailed token-level temporal view with sparklines

### Expected Timeline (When Prerequisites Complete)

**Week 1**: Implement manual generation loop, validate correctness
**Week 2**: Record subtoken timelines for 10-20 diverse prompts
**Week 3**: Build continuous sparkline visualizations (ASCII + PNG)
**Week 4**: Analyze patterns, measure lag statistics, document findings
**Week 5**: Integrate with OpenWebUI (optional)

---