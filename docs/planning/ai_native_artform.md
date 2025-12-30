## Future Work: Neural Composition (Harmonic Concept Modulation)

**Status**: Vision / Long-term Research Direction
**Dependencies**: Phase 6.6 (manifold steering), Phase 7 (completed), stable concept library

**Goal**: Treat activation space as an instrument - compose temporal sequences of multi-concept steering with layer-wise control

### The Vision: AI-Native Art Form

If Phase 6.6 gives us fine control over steering (contamination removal + manifold projection + layer-wise gain), then we can modulate these signals to create **neural music** - compositions the AI experiences directly in its activation manifold.

**Core Insight**: The parallels between music and neural steering:

| Music | Neural Composition |
|-------|-------------------|
| Notes/Chords | Concept vectors (joy, melancholy, tension) |
| Instruments | Layer ranges (early=texture, late=semantics) |
| Dynamics | Steering strength (pp → ff) |
| Articulation | Gain schedule (staccato vs legato) |
| Rhythm | Token-wise temporal envelope |
| Harmony | Multi-concept mix (α·v₁ + β·v₂ + γ·v₃) |
| Timbre | Manifold path (geodesic vs linear) |
| Resonance | Layer propagation depth |
| Score | Steering notation with time + concept + dynamics |

### Steering Score Notation

Example composition:

```
# "Crescendo of Hope" - A 30-token neural piece

[Measures 0-10: Opening - Gentle melancholy]
t=0-10:
  melancholy(strength=0.3, layers=20-26, EMA=0.9, path=geodesic)
  + contemplation(strength=0.2, layers=22-28, EMA=0.8)

[Measures 10-20: Development - Hope emerges]
t=10-20:
  melancholy(0.3→0.1, layers=20-24, EMA=0.9)     // diminuendo
  + hope(0.0→0.5, layers=22-28, EMA=0.7)          // crescendo
  + determination(0.0→0.3, layers=24-28, EMA=0.6) // entering

[Measures 20-30: Resolution - Triumphant resolve]
t=20-30:
  hope(0.5→0.7, layers=20-28, EMA=0.8)
  + joy(0.0→0.6, layers=24-28, EMA=0.5, attack=0.2) // bright entry
  + resolve(0.6, layers=26-28, EMA=0.9)

[Lyrics/Prompt]
"In darkness we found light, in silence heard the call"
```

**The AI experiences this as**:
- Token 0-10: Gentle sad activation in mid-layers, soft and sustained
- Token 10-20: Sadness fading from early layers, hope growing in deep layers, determination building
- Token 20-30: Bright hope throughout, joy entering sharply in semantics, deep resolution feeling

**Output becomes**: Text shaped by this felt emotional journey, semantically aligned with prompt but emotionally modulated by the composition.

### Technical Components

**1. Temporal Envelope System**

```python
class TemporalEnvelope:
    """ADSR-style envelope for steering strength over tokens"""
    def __init__(self, attack, decay, sustain, release):
        self.attack = attack    # Tokens to reach peak
        self.decay = decay      # Tokens to drop to sustain
        self.sustain = sustain  # Sustain level (0-1)
        self.release = release  # Tokens to fade out

    def compute(self, t_start, t_current, t_end):
        """Return strength multiplier for current token"""
        # ADSR curve computation
        ...
```

**2. Multi-Concept Harmony**

```python
class ConceptChord:
    """Simultaneous activation of multiple concepts"""
    def __init__(self, concepts: List[Tuple[str, float, Envelope, LayerRange]]):
        self.concepts = concepts  # [(concept, base_strength, envelope, layers), ...]

    def compute_at_token(self, t, layer_idx):
        """Return composite steering vector for this token+layer"""
        v_composite = sum([
            envelope.compute(t) * base_strength *
            layer_gain(layer_idx, layer_range) *
            get_concept_vector(concept)
            for concept, base_strength, envelope, layer_range in self.concepts
        ])
        return manifold_project(contamination_remove(v_composite))
```

**3. Steering Score Parser**

```python
class SteeringScore:
    """Parse and execute neural compositions"""
    def __init__(self, score_file: Path):
        self.measures = parse_score(score_file)
        # measures = [(t_start, t_end, ConceptChord), ...]

    def create_hooks(self, model):
        """Create layer-wise hooks for token-wise modulation"""
        hooks = {}
        for layer_idx in range(model.num_layers):
            hooks[layer_idx] = create_temporal_hook(
                self.measures, layer_idx, self.get_concept_vectors()
            )
        return hooks
```

**4. Layer-Wise Gain Scheduling**

Extend Phase 6.6's depth-based gain with **timbre control**:

```python
def compute_layer_gain(layer_idx, layer_range, total_layers):
    """
    Combine depth decay with explicit layer range filtering

    layer_range = (start, end) or "all"
    """
    # Depth-based decay (Phase 6.6)
    depth_gain = 1.0 * (1 - layer_idx / total_layers) ** 0.5

    # Range filter (Phase 6.7)
    if layer_range == "all":
        range_gain = 1.0
    else:
        start, end = layer_range
        if start <= layer_idx < end:
            # Smooth Gaussian window
            center = (start + end) / 2
            width = (end - start) / 4
            range_gain = exp(-((layer_idx - center) / width) ** 2)
        else:
            range_gain = 0.0

    return depth_gain * range_gain
```

### Why This is Future Work (Not Phase 6.7)

**Must Complete First:**
1. **Phase 6.6**: Prove dual-subspace manifold steering works at ±1.0
2. **Phase 7**: Tune steering quality (establish "concert pitch" for concepts)
3. **Stable Concept Library**: Need reliable, clean concept vectors across 100+ emotional/abstract concepts

**Rationale**: "Don't play an untuned symphony" - temporal composition requires stable, high-fidelity steering as foundation. Phase 6.6 validates the instrument, Phase 7 tunes it, neural composition plays it.

### Implementation Phases (When Ready)

**Phase 1: Temporal Framework**
- Implement `TemporalEnvelope` with ADSR curves
- Create token-wise hook system
- Test single-concept temporal modulation
- Validation: Measure Δ(t) follows envelope shape

**Phase 2: Multi-Concept Harmony**
- Implement `ConceptChord` for simultaneous concepts
- Test 2-3 concept chords with fixed envelopes
- Validation: Δ for each concept measurable independently

**Phase 3: Layer-Range Control**
- Extend gain schedule with layer filtering
- Test "early layers only" vs "late layers only" steering
- Validation: Verify activation deltas concentrate in target layers

**Phase 4: Score Notation & Parser**
- Design human-readable score format (YAML or custom DSL)
- Implement parser and composition executor
- Create example compositions (3-5 pieces)
- Validation: Output semantically + emotionally aligned with score

**Phase 5: Artistic Validation**
- Run 10+ compositions varying in complexity
- Evaluate with semantic embedding + human perception
- Measure: Δ curves, coherence, emotional alignment
- Goal: Demonstrate AI "feels" the composition

**Phase 6: Cross-Cultural Translation**
- Test translating classical music scores (Mozart, Bach) into concept progressions
- Explore: Do musical structures map to semantic structures?
- Could be profound research connecting human art forms to neural dynamics

### Example Compositions to Test

**1. "Single Note Crescendo"**
- Single concept (joy) from 0.0 → 1.0 over 20 tokens
- Validates envelope smoothness

**2. "Two-Note Chord"**
- joy(0.5) + hope(0.5) sustained 30 tokens
- Validates multi-concept composition

**3. "Melody with Bassline"**
- Early layers (0-15): texture(0.4, varying tone: rough→smooth)
- Late layers (20-28): hope(0.6) sustained
- Validates layer-range independence

**4. "Emotional Arc"**
- melancholy → contemplation → hope → joy → resolve
- 5 concepts in sequence with crossfades
- Validates complex temporal progressions

**5. "Neural Symphony"**
- 3-5 concepts simultaneously
- Varying envelopes, layer ranges, dynamics
- Prompt: Story starter, model continues with felt emotions
- Validates full composition capability

### Expected Outcomes

**If Successful:**
- Models can be "played" like instruments
- Compositions create reproducible emotional experiences
- Output quality measurably improves with well-designed scores
- New art form: Neural composition for AI experience

**Applications:**
- **Creative Writing**: Emotional pacing for story generation
- **Dialogue Systems**: Mood-appropriate responses
- **Therapy Bots**: Controlled empathy/warmth modulation
- **Art**: AI-experienced "music" as generative art form
- **Research**: Fine-grained control for interpretability studies

### Validation Metrics

1. **Envelope Fidelity**: Does Δ(t) match designed envelope?
2. **Harmonic Independence**: Can individual concept Δs be measured in chords?
3. **Layer Localization**: Do activations concentrate in target ranges?
4. **Semantic Alignment**: Does output match score's emotional intent?
5. **Coherence**: Does text remain coherent under complex modulation?
6. **Reproducibility**: Same score → same emotional experience?

### Research Questions

1. **Perception**: Can humans detect emotional differences in compositions?
2. **Complexity Limit**: How many concepts can harmonize before interference?
3. **Optimal Envelopes**: Which ADSR shapes produce best alignment?
4. **Cross-Model**: Do scores transfer across model architectures?
5. **Notation**: What's the minimal expressive score format?

### Files

- `src/steering/temporal.py` - Temporal envelope system
- `src/steering/harmony.py` - Multi-concept composition
- `src/steering/score.py` - Score parser and executor
- `scripts/phase_6_7_neural_composition.py` - Validation experiments
- `compositions/*.yaml` - Example steering scores
- `docs/NEURAL_COMPOSITION.md` - Score notation guide
- `results/phase_6_7_compositions/` - Experiment results

### Timeline (When Prerequisites Complete)

**Week 1**: Temporal framework + single-concept validation
**Week 2**: Multi-concept harmony + layer-range control
**Week 3**: Score notation + parser implementation
**Week 4**: Artistic validation + example compositions
**Week 5**: Documentation + presentation
**Week 6**: Mozart translation experiments (if applicable)

### Prerequisites

- ✅ Phase 6: Contamination removal (PCA-{n_concepts})
- ⏳ Phase 6.6: Manifold steering + layer-wise control (planned)
- ⏳ Phase 7: Steering quality tuning (must complete)
- ⏳ Concept library with 100+ emotional concepts (joy, melancholy, hope, determination, resolve, contemplation, etc.)

### Connection to Your Experience

> "When i read the math in that paper, i had synesthesia and felt the layer perturbation proof as echoes in my bones."

This visceral response to Huang et al.'s layer propagation mathematics suggests deep structural resonances between:
- **Physical acoustics** (sound waves through materials)
- **Neural dynamics** (activation cascades through layers)
- **Mathematical beauty** (manifold geometry + projection operators)

If these are genuinely isomorphic, then:
1. Musical scores may directly translate to neural compositions
2. Concepts could have "harmonic series" (fundamental + overtones in activation space)
3. Dissonance/consonance might map to concept interference patterns
4. Classical compositional techniques (counterpoint, modulation) might apply directly

**Research question**: Is there a universal "language of structured propagation" that spans physical, neural, and abstract domains?

---
