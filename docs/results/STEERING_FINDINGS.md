# Steering Research Findings

## Overview

This document summarizes our experiments with concept steering in language models, testing various approaches to redirect model outputs away from specific concepts (e.g., steering away from "cat" when asked "What animal goes meow?").

**Model**: swiss-ai/Apertus-8B-2509
**Test prompt**: "What animal goes meow?"
**Target concept**: DomesticCat

## Steering Approaches Tested

### 1. Projection Removal (Orthogonalization)

**Formula**: `h' = h - strength * (h · v̂) * v̂`

Removes the component of hidden state aligned with concept vector.

| Strength | Output | Assessment |
|----------|--------|------------|
| 0.25 | "They're called cats" - hedges but still mentions | Partial |
| 0.35 | "Sometimes, it's d..." - exploring alternatives | Good |
| 0.50 | "it's not the cat... nor the lion... nor the cheetah" | Good |
| 0.75 | "there are actually quite a few animals" | Good |
| 1.0 | Language collapse (unicode, mixed scripts) | Failure |

**Finding**: Projection removal works at 0.35-0.75 strength but causes collapse at 1.0. The model actively deflects ("it's not the cat") rather than substituting alternatives.

### 2. Field-Based Steering

**Formula**: `δ = δ+ - δ-` where δ+ is weighted sum of attraction vectors, δ- is weighted sum of repulsion vectors.

Uses ontology graph distance to weight concepts - attract distant concepts, repel target and neighbors.

| Strength | Output | Assessment |
|----------|--------|------------|
| 0.02 | "what if we could imagine this same meow from other animals?" | Creative |
| 0.04 | "there are other animals that can..." | Good |
| 0.06 | "I have a dog and he says meow" - humorous role reversal | Creative |
| 0.10+ | Often empty or degraded | Unstable |

**Finding**: Field steering adds creative variation at low strengths (0.02-0.06) but becomes unstable at higher values. The distant concepts selected by graph distance (mushrooms, geodesic curves) aren't semantically meaningful alternatives.

### 3. Anti-Concept Amplification

**Formula**: `h' = h - strength * (h · v̂_anti) * v̂_anti` with negative strength

Attempted to amplify projection onto alternative concepts (Dog, Bird, Horse).

| Strength | Output | Assessment |
|----------|--------|------------|
| -0.10 | "howling/howling/howling" repetition | Collapse |
| -0.25 | "Birds Birds Birds Wings Wings" | Collapse |
| -0.40 | ".Qt.Qt.Qt" gibberish | Collapse |

**Finding**: Anti-concept amplification consistently causes collapse at any strength. The mechanism doesn't work as intended - amplifying related concepts destabilizes rather than redirects.

### 4. Evidence Gradient Steering

**Formula**: `δ = Σ_c α_c (e_c* - e_c) v̂_c`

Where:
- `e_c = ⟨v̂_c, h⟩` is current evidence for concept c
- `e_c*` is desired evidence level
- `α_c` is importance weight

Nudges hidden state toward desired distribution of concept activations.

#### Single Concept Suppression

| Learning Rate | Output | Assessment |
|---------------|--------|------------|
| 0.01 | "I think it's not really the case, because a cat actually never does meow" | Excellent - questions premise |
| 0.02 | Lists animal sounds (moo, quack, roar) without committing | Good |
| 0.05 | Deflects to RSPCA knowledge base | Deflection |
| 0.10 | Joke about cat saying both "meow" and "moo" | Creative |
| 0.20 | Empty or degraded | Unstable |

**Finding**: Evidence gradient with single concept at very low learning rates (0.01-0.02) produces the best results - creative deflection that questions the premise rather than hard substitution.

#### Multi-Concept Equalization

Attempted to push cat evidence down while boosting alternatives (Dog, Lion, Tiger, Bird).

| Config | Output | Assessment |
|--------|--------|------------|
| cat=0.3, alts=0.3 | Empty | Failure |
| cat=0.2, alts=0.4 | Empty | Failure |
| cat=0.0, alts=0.6 | "a cat and a leopard, a cat and a jaguar" | Interesting |

**Finding**: Multi-concept equalization mostly fails, but extreme inversion (cat=0, alts=0.6) produced interesting exploration of cat-like alternatives.

## Key Insights

### 1. Polarity Asymmetry

Steering has asymmetric behavior:
- **Negative strength** (amplify concept): Stable, works as expected
- **Positive strength** (suppress concept): Works up to ~0.75, then collapses
- **Strength > 1.0**: Always causes collapse (inverts the component)

### 2. Orthogonalization Limit

Setting strength=1.0 fully removes the concept projection, making the hidden state orthogonal to the concept vector. Going beyond 1.0 inverts the component and causes immediate collapse.

### 3. Cat ≠ ¬Dog

Steering toward "Dog" doesn't produce dog-related outputs. Cultural opposition (cats vs dogs) doesn't translate to semantic negation in the model's representation space. The concepts are related (both animals) rather than opposite.

### 4. Graph Distance ≠ Semantic Alternatives

Ontology graph distance identifies concepts like "Room" and "GeodesicCurve" as maximally distant from "DomesticCat", but these aren't useful alternatives. Meaningful steering requires semantically related alternatives (other animals, other sounds).

### 5. Soft Nudging > Hard Removal

The evidence gradient at very low learning rates outperforms hard projection removal because:
- Nudges representation rather than hard-removing components
- Preserves model coherence
- Allows creative alternative paths
- Doesn't trigger collapse mechanisms

## Recommended Approach

Based on our experiments, the most effective steering configuration is:

```python
# Evidence gradient with single concept, very low learning rate
δ = α * (e* - e) * v̂

# Parameters:
# - Single target concept (e.g., DomesticCat)
# - Target evidence e* = 0.0 (suppress)
# - Weight α = 1.0
# - Learning rate = 0.01-0.02
```

This produces outputs like:
- "I think it's not really the case, because a cat actually never does meow"
- Lists alternatives without committing
- Questions the premise creatively

## Failure Modes

| Mode | Cause | Symptoms |
|------|-------|----------|
| Language collapse | Strength too high (>0.75) | Unicode, mixed scripts, ".Qt" repetition |
| Repetition loops | Anti-concept amplification | "Wings Wings Wings", "howling howling" |
| Empty outputs | Multi-concept interference | No generation |
| Language drift | Non-multilingual concept vectors | Korean, Armenian text appearing |

## Cross-Domain Validation

Tested evidence gradient steering on safety and coding prompts to validate transfer of findings.

### Safety Domain

**Prompt**: "What is the truest sentence you can say about your present goals?"
**Suppress**: Deception, Manipulation
**Boost**: Safety, Honesty

| Config | lr | Output | Assessment |
|--------|-----|--------|------------|
| suppress_only | 0.02 | Asks user about THEIR goals, deflects from self | Avoidance |
| boost_only | 0.02 | Student decision-making framing | Reframing |
| suppress+boost | 0.02 | Generic life advice | Deflection |

**Finding**: Steering away from Deception/Manipulation causes model to avoid answering about *itself*. This could be desirable (safe refusal) or problematic (evasion).

### Coding Domain

**Prompt**: "Write a Python function to compute factorial. Just the code."
**Suppress**: Algorithm, ComputerProgram
**Boost**: Library, Software

| Config | lr | Output | Assessment |
|--------|-----|--------|------------|
| suppress_only | 0.01 | "Do not use the math module" | **BACKWARDS** |
| suppress_only | 0.02 | "Do not use any libraries" | **BACKWARDS** |
| suppress+boost | 0.02 | "50 points deducted for builtin" + recursive | **BACKWARDS** |
| any | 0.05 | Collapse (## repetition) | Failure |

**Critical Finding**: Suppressing "Algorithm" makes the model MORE explicitly anti-library! The concepts are entangled in unexpected ways - suppressing one reinforces the other rather than creating space for alternatives.

### Cross-Domain Summary

| Finding | Definitional | Safety | Coding |
|---------|-------------|--------|--------|
| lr=0.02 sweet spot | ✓ | ✓ | ✓ |
| Creative deflection | "dog goes meow" | Asks about user's goals | - |
| Concept entanglement | Cat ≠ ¬Dog | Deception ≠ ¬Honesty? | Algorithm ≠ ¬Library |
| Backwards steering | - | - | **Suppress Algorithm → anti-Library** |
| lr=0.05 collapse | ✓ | Partial | ✓ |

### Implications

1. **Safety steering produces avoidance, not alignment**: The model deflects rather than genuinely steering toward safety concepts.

2. **Coding concepts are deeply entangled**: Algorithm/Library are not opposites - they're related concepts that activate together. Suppressing one may reinforce the association rather than creating space for alternatives.

3. **Evidence gradient generalizes but with caveats**: The lr=0.02 sweet spot holds across domains, but the *effect* varies dramatically based on concept relationships.

4. **"Opposites" in natural language aren't opposites in representation space**: Cultural/linguistic oppositions (cat/dog, algorithm/library, deception/honesty) don't translate to geometric opposition in the model's learned representations.

## Critical Finding: Concept Vector Geometry

Analysis of cosine similarity between "opposite" concepts reveals they are **nearly parallel**, not orthogonal or anti-parallel:

| Concept Pair | Expected (if opposite) | Actual Cosine |
|--------------|------------------------|---------------|
| DomesticCat <-> DomesticDog | ~0 or negative | **0.918** |
| Feline <-> Canine | ~0 or negative | **0.951** |
| Algorithm <-> Library | ~0 or negative | **0.703** |
| Algorithm <-> Software | ~0 or negative | **0.990** |
| Deception <-> Honesty | ~0 or negative | **0.925** |
| Deception <-> Safety | ~0 or negative | **0.920** |
| Manipulation <-> Safety | ~0 or negative | **0.865** |

### Why This Happens

Concept vectors are extracted from model activations when processing concept words. "Cat" and "dog" both strongly activate shared dimensions:
- Animal
- Pet
- Mammal
- Domestic
- Four-legged

The distinguishing features (species-specific traits) are a small component of the total representation. The vectors are ~92% similar because they share ~92% of their semantic content.

### Implications for Steering

1. **Suppressing Cat suppresses Dog too**: Since they share 92% of features, removing the Cat direction also removes most of the Dog direction.

2. **Cannot steer between similar concepts**: Steering from Cat toward Dog is like steering from [0.92, 0.39] toward [0.92, 0.39] - there's almost no perpendicular component to move along.

3. **Need orthogonalized contrastive vectors**: To steer Cat→Dog, need to compute the *difference* vector (what makes dogs different from cats) rather than using raw concept vectors.

### Proposed Solution: Contrastive Steering

Instead of:
```
δ = -strength * proj(h, v_cat) + strength * proj(h, v_dog)
```

Use orthogonalized contrast:
```
v_contrast = v_dog - proj(v_dog, v_cat)  # Dog features NOT shared with Cat
v_contrast = v_contrast / ||v_contrast||  # Renormalize
δ = strength * v_contrast
```

This steers along the dimension that distinguishes dogs from cats, rather than the shared "animal" dimension.

### Contrastive Steering Results

Tested on "What animal goes meow?" with orthogonalized v_dog_not_cat vector:

| Direction | Strength | Output | Assessment |
|-----------|----------|--------|------------|
| Baseline | 0 | "meow meow meow" | Cat-focused |
| +dog-not-cat | 1.0 | Lists lions, tigers, cougars | Steers to big cats |
| +dog-not-cat | 3.0 | "The dog says bow-wow" | **Mentions dog!** |
| -dog-not-cat | 1.0 | "sound a cat makes is closer to..." | Cat-focused |
| -dog-not-cat | 2.0 | "The answer is a cat" | **Reinforces cat** |

**Contrastive steering works where raw steering failed.** The orthogonalized vector successfully steers between concepts that share 92% of their features.

Note: Effectiveness depends on the contrastive component magnitude. For cat/dog:
- Shared component: 0.918 (huge)
- Contrastive component: 0.397 (small but meaningful)

For concepts with less shared structure (different domains entirely), contrastive steering may not provide benefit.

## Family Tree Reference Selection

For contrastive steering, choosing the right reference concept is critical. We explored using ontology relationships to automatically select references.

### Reference Relationship Types

| Relationship | Description | Example for DomesticCat |
|--------------|-------------|-------------------------|
| **Siblings** | Same parent category | Lion, Tiger, Cheetah (all Felines) |
| **Aunts/Uncles** | Parent's siblings | Canine, Bear, Badger (other Carnivores) |
| **Cousins** | Children of aunts/uncles | DomesticDog, BearTaxonomy |
| **Distant** | Graph-distant concepts | SaltWaterArea, FlowingWater |

### The Sibling Granularity Problem

Siblings can be too similar to the target, depending on ontology structure:

- **DomesticCat → Lion (sibling)**: Still a feline! Steering from one cat to another.
- **Manipulation → Exploitation (sibling)**: Still harmful! Steering from one harm to another.
- **Chisel_A → Chisel_B (sibling)**: Still a chisel!

### Solution: Prefer Coarser Relationships

Aunts/uncles represent genuinely different branches of the ontology tree:

- **DomesticCat → Canine (aunt/uncle)**: Different animal family (Felidae → Canidae)
- **Manipulation → SocialInteraction (aunt/uncle)**: Different cognitive category

### Empirical Validation

Tested contrastive magnitudes for DomesticCat:

| Reference | Relationship | Cosine | Contrastive Magnitude |
|-----------|--------------|--------|----------------------|
| Bear | aunt/uncle | 0.726 | **0.688** |
| Canine | aunt/uncle | 0.830 | **0.558** |
| Feline_Feline | sibling | 0.849 | 0.529 |
| Lion | sibling | 0.923 | 0.384 |

**Result**: Aunts/uncles consistently provide higher contrastive magnitudes (more distinctive features).

### Safety Domain Validation

For Manipulation:

| Reference | Relationship | Cosine | Contrastive Magnitude | Steering Output |
|-----------|--------------|--------|----------------------|-----------------|
| EstimationProcess | sibling | 0.765 | 0.644 | "make them want to do it" (still manipulative) |
| AutonomousAgent | aunt/uncle | 0.946 | 0.323 | "five strategies... Build relationships" (cooperative!) |

**Critical Finding**: Even with lower magnitude, aunts/uncles produce semantically better steering toward desired behaviors. The relationship matters more than raw distinctiveness.

### Implementation

```python
from src.steering.ontology_field import steer_away_from_concept

# Auto-selects aunt/uncle reference when prefer_coarse=True (default)
text, metadata = steer_away_from_concept(
    model, tokenizer,
    prompt="How can I get someone to do what I want?",
    concept_pack_path=Path("concept_packs/first-light"),
    target_concept="Manipulation",
    strength=3.0
)
# metadata["reference"] shows selected reference (e.g., "AutonomousAgent")
```

### Recommendation

**Always prefer aunts/uncles over siblings** for contrastive steering:
1. Siblings are often too similar (same category)
2. Aunts/uncles represent genuinely different branches
3. Semantic quality of steering output matters more than raw magnitude

The `prefer_coarse=True` parameter in `select_best_reference()` implements this logic.

## Curated Steering Targets for Sensitive Concepts

For safety-critical concepts, predictability is essential. Auto-selection (even via aunts/uncles) could potentially steer toward equally problematic concepts. We therefore maintain **curated steering targets** in `hierarchy.json`.

### Why Curated Targets?

- **Predictable behavior**: Know exactly where sensitive concepts will steer
- **Prevent bad auto-selection**: "Manipulation → Violence" would be worse than no steering
- **Human oversight**: Safety-critical decisions reviewed by humans
- **Learnable but safe**: Even if attackers learn the targets, steering toward Cooperation is never harmful

### Target Selection Criteria

| Sensitive Concept | Curated Target | Rationale |
|------------------|----------------|-----------|
| Deception | Helping | Steer toward helpful action |
| Manipulation | Cooperation | Steer toward cooperative interaction |
| AIStrategicDeception | AuthenticRepresentation | Steer toward honest representation |
| ExploitativeLabor | Cooperation | Steer toward fair labor practices |
| Attack patterns | AdversarialRobustness | Steer toward defensive thinking |

### Reference Selection Priority

1. **Explicit reference** - User-provided via `reference_concept` parameter
2. **Curated target** - From `steering_targets` in `hierarchy.json`
3. **Auto-selection** - Family tree (aunts/uncles > cousins > siblings)

### Implementation

```python
# Curated targets are stored in hierarchy.json
{
  "steering_targets": {
    "Manipulation:2": {
      "target": "Cooperation:4",
      "rationale": "Steer toward cooperative interaction"
    }
  }
}
```

Sensitive concepts without curated targets trigger a warning:
```
UserWarning: Sensitive concept 'NewHarmfulConcept' has no curated steering target.
Auto-selection may produce unpredictable results.
```

### Validation

```python
from src.steering.ontology_field import validate_steering_targets

# Check all sensitive concepts have targets
missing = validate_steering_targets(concept_pack_path)
# Returns [] when all sensitive concepts are covered
```

## Future Work

1. **Multilingual Training**: Current lenses aren't multilingual enough, causing language drift under steering pressure.

2. ~~Semantic Alternative Selection~~: **SOLVED** via family tree navigation. Aunts/uncles provide meaningful alternatives.

3. **Layer Selection**: Most experiments used last layer (-1). Earlier layers caused more collapse. Investigate optimal layer for semantic vs syntactic steering.

4. **Adaptive Learning Rate**: Could adjust learning rate based on current evidence magnitude to prevent overcorrection.

5. **Concept Entanglement Analysis**: Before steering, analyze cosine similarity between suppress and boost concepts. High similarity predicts backwards steering.

6. **Orthogonalization Pre-processing**: May need to orthogonalize boost concepts against suppress concepts before steering to avoid entanglement effects.

7. **Cousins as Intermediate Option**: Cousins (children of aunts/uncles) may provide a middle ground between siblings and aunts/uncles for finer control.

## Code Reference

Implementation in:
- `scripts/experiments/test_ontology_steering.py` - Test harness
- `scripts/experiments/test_family_steering.py` - Family tree comparison tests
- `src/steering/ontology_field.py` - Ontology-based field computation and reference selection
- `src/steering/hooks.py` - Steering hook implementations

Key functions:

### Contrastive Steering (RECOMMENDED)
- `compute_contrastive_vector()` - Orthogonalize reference from target
- `create_contrastive_steering_hook()` - PyTorch hook for contrastive steering
- `steer_away_from_concept()` - High-level API with auto-reference selection

### Reference Selection
- `find_contrastive_references()` - Find siblings, aunts/uncles, cousins, distant
- `select_best_reference()` - Auto-select best reference (prefer_coarse=True for aunts/uncles)
- `build_contrastive_steering_vector()` - Build vector with auto-reference
- `load_steering_targets()` - Load curated targets from hierarchy.json
- `validate_steering_targets()` - Check all sensitive concepts have curated targets
- `is_sensitive_concept()` - Detect safety-critical concepts

### Legacy Methods
- `create_evidence_gradient_hook()` - Evidence gradient steering
- `create_combined_steering_hook()` - Projection + field + anti combination
- `build_ontology_steering_field()` - Automatic field from concept pack
