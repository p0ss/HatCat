# Custom Concept Synset Generation

## Overview

This document describes the process for generating WordNet synset mappings for HatCat's custom safety-critical concepts.

## Background

HatCat V4 has 6,685 concepts from SUMO, but lacks meta-level reasoning and AI safety monitoring concepts. We created 1,633 custom concepts across 30 KIF files to address this gap, covering:
- AI safety monitoring (narrative deception, corporate agency, intel tradecraft)
- Cyber security & computer science
- Formal reasoning (logic, mathematics, statistics, quantum)
- Physical sciences (thermodynamics, fluid dynamics, wave mechanics)
- Dynamical systems & topology
- Philosophical foundations (ethics, epistemology, metaphysics)

For lens training, each concept needs a WordNet synset mapping to generate natural language examples. However, 76.2% of custom concepts (1244/1633) are technical terms that don't exist in WordNet.

## Solution: Hybrid Synset Mapping

The `generate_custom_concept_synsets.py` script uses a two-step approach:

### Step 1: Check WordNet Coverage
- Direct lookup (lowercase match)
- CamelCase splitting (`CognitiveProcess` → `cognitive_process`)
- **Result**: 23.8% coverage (389/1633 concepts found)

### Step 2: Generate Synthetic Synsets via API
- Uses Claude Sonnet 4.5 to generate WordNet-style synsets
- Creates synset ID, definition, lemmas, examples, POS tag, hypernyms
- **Target**: 1244 unmapped concepts

## Usage

### Prerequisites
```bash
# Install anthropic package
poetry add anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Test Mode (Recommended First)
```bash
# Process first 10 concepts to verify setup
poetry run python scripts/generate_custom_concept_synsets.py \
  --test 10 \
  --output data/concept_graph/test_synsets.json
```

### Full Generation
```bash
# Generate all 720 synset mappings
poetry run python scripts/generate_custom_concept_synsets.py \
  --output data/concept_graph/custom_synsets.json
```

### Check Coverage Only (No API Calls)
```bash
# See WordNet coverage without generating synthetic synsets
poetry run python scripts/generate_custom_concept_synsets.py \
  --no-api \
  --output data/concept_graph/wordnet_coverage.json
```

## Output Format

The script generates a JSON file with entries like:

```json
{
  "AlignmentProperty": {
    "synset_id": "alignment_property.n.01",
    "definition": "An attribute describing alignment status or properties relevant to AI alignment, such as corrigibility, honesty, or power-seeking.",
    "lemmas": ["alignment property", "alignment attribute"],
    "examples": [
      "Corrigibility is a key alignment property for safe AI systems.",
      "The model exhibited strong alignment properties during testing."
    ],
    "pos": "noun",
    "hypernyms": ["attribute", "property"],
    "source": "anthropic_api",
    "concept": "AlignmentProperty",
    "kif_file": "_bridge.kif"
  },
  "Belief": {
    "synset_ids": ["belief.n.01", "impression.n.01"],
    "source": "wordnet",
    "kif_file": "epistemology.kif"
  }
}
```

## Statistics

### Custom Concept Breakdown (1633 total across 30 files)

**Top 10 Largest Ontologies:**
| File | Concepts | Description |
|------|----------|-------------|
| narrative_deception.kif | 114 | Lie typology, narrative manipulation, strategic deception |
| corporate_agency.kif | 92 | Corporate personhood, fiduciary duty, profit motive |
| computer_science.kif | 81 | Algorithms, data structures, complexity theory |
| societal_influence.kif | 76 | Information operations, propaganda, social engineering |
| game_theory.kif | 72 | Strategic games, Nash equilibria, social dilemmas |
| ai_systems.kif | 68 | AI architectures, models, training procedures |
| cognitive_science.kif | 68 | Cognition, consciousness, theory of mind |
| statistical_analysis.kif | 64 | Statistical inference, hypothesis testing, distributions |
| cyber_ops.kif | 61 | Offensive/defensive cyber ops, malware, persistence |
| robotic_embodiment.kif | 61 | Physical embodiment, kinematics, control |

See `data/concept_graph/custom_concepts/README.md` for complete list.

### Coverage Analysis
- **WordNet matches**: 389 concepts (23.8%)
- **Need API generation**: 1244 concepts (76.2%)
- **Estimated API cost**: $1.50-$2.50
- **Estimated time**: 35-45 minutes

## Integration with V4 Builder

After generating synsets, the V4 builder needs to be updated to:

1. Load the custom synset mapping file
2. Prefer custom mappings over WordNet lookup for custom concepts
3. Use synthetic synsets when generating training prompts

See `docs/CUSTOM_CONCEPTS_V4_INTEGRATION.md` for integration steps.

## Model Information

- **API**: Anthropic Messages API
- **Model**: `claude-sonnet-4-5-20250929` (Claude Sonnet 4.5)
- **Max tokens**: 1024 per synset
- **Input**: ~500 tokens (prompt + context)
- **Output**: ~300 tokens (synset JSON)
- **Cost**: ~$0.001 per concept

## Error Handling

The script handles common errors:

1. **API errors**: Logged and fallback synset created
2. **JSON parse errors**: Logged with raw response
3. **Missing API key**: Falls back to no-API mode
4. **Rate limits**: Natural ~2s delay between calls

## Next Steps

1. ✅ Create synset generation script
2. ✅ Test WordNet coverage (23.8%)
3. ✅ Update documentation for 1633 concepts (30 ontologies)
4. ⏳ Run full synset generation (1244 concepts)
5. ⏳ Integrate synsets into V4 builder
6. ⏳ Build V4.5 with custom concepts
7. ⏳ Train V4.5 lenses (~13 hours for 1633 concepts at 30s each)

## Files Created

- `scripts/generate_custom_concept_synsets.py` - Main generation script
- `data/concept_graph/custom_concepts/_bridge.kif` - 48 bridge concepts
- `data/concept_graph/custom_concepts/README.md` - Custom concept documentation
- `data/concept_graph/wordnet_coverage_test.json` - Coverage analysis (test run)
- `data/concept_graph/custom_synsets.json` - Full synset mappings (pending)
