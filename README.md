# HatCat (Current Capabilities)

This document reflects what actually ships in `src/` today: how concepts are built, how probes are trained, and how to run real-time monitoring/steering.

## Core Modules

### Concept Encyclopedia (`src/encyclopedia`)
- **`concept_loader.py`** – pulls mixed WordNet/ConceptNet/Wikipedia vocabularies, plus the curated SUMO+AI ontologies.
- **`concept_graph.py`, `wordnet_graph.py`, `build_sumo_wordnet_layers_v5.py`** – crawlers and builders that produce `data/concept_graph/abstraction_layers/layer{0..6}.json`, combining SUMO hierarchy, custom AI domains, and WordNet hyponym clusters (v5 is the version that generated the current data).

### Activation Capture (`src/activation_capture`)
- **`ActivationCapture`** – registers forward hooks, applies Top‑K sparsity, and writes dense/sparse tensors to HDF5 via `src/utils/storage.py`.
- **`ModelLoader`** – convenience loader for Gemma-3 checkpoints with layer inspection helpers.

### Training (`src/training`)
- **Prompt synthesis**: `data_generation.py` and `sumo_data_generation.py` generate definitional, relational, and neutral prompts (WordNet distance ≥5) for each concept.
- **Binary classifiers**: `classifier.py` hosts the minimal MLP for concept probes.
- **Activation extraction**: `activations.py` averages residual/MLP layers during short generations.
- **SUMO classifier API** (`sumo_classifiers.py`): wraps the entire pipeline—load per-layer concepts, synthesize data, extract activations, train classifiers, and save results. Exposed via `scripts/train_sumo_classifiers.py`.

### Steering (`src/steering`)
- **Concept vector extraction** (`extraction.py`) – mean hidden-state directions from concept prompts.
- **Hook-based steering** (`hooks.py`) – projection removal/injection during generation.
- **Subspace cleanup** (`subspace.py`, `manifold.py`) – mean/PCA removal and dual-subspace manifold steering with Huang-style dampening.
- **Semantic evaluation** (`evaluation.py`) – centroid building + Δ score via SentenceTransformers.

### Monitoring (`src/monitoring`)
- **`temporal_monitor.py`** – loads trained SUMO classifiers (Layers 0–2), samples each token’s hidden state during generation, and records top-k concepts over time.
- **`sumo_temporal.py`** – high-level API/CLI integration. Reuses preloaded models/monitors, accepts generation kwargs, and can print/save reports or return raw JSON for downstream apps.

### Interpreter Prototypes (`src/interpreter`)
- Transformer-based decoders (`SemanticInterpreter`, `InterpreterWithHierarchy`, `MultiTaskInterpreter`) that map pooled activations to concept logits/confidence for research into semantic decoding.

## Usage Guide

### 1. Build or Refresh Concept Layers
The repo already ships with v5 SUMO→WordNet layers in `data/concept_graph/abstraction_layers/`. To rebuild:
```bash
poetry run python src/build_sumo_wordnet_layers_v5.py \
  --sumo-dir data/concept_graph/sumo_source \
  --output-dir data/concept_graph/abstraction_layers
```
This script parses Merge.kif, merges custom AI ontologies, constructs hyponym-based intermediate layers, and writes `layer0.json` through `layer6.json`.

### 2. Train SUMO Concept Classifiers
Train/update binary probes for the SUMO layers:
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 \
  --model google/gemma-3-4b-pt \
  --device cuda \
  --n-train-pos 10 --n-train-neg 10 \
  --n-test-pos 20 --n-test-neg 20
```
Outputs go to `results/sumo_classifiers/layer{N}/`, with `*_classifier.pt` weights and `results.json` summaries (F1/precision/recall per concept). These weights are what the monitoring stack loads.

### 3. Real-Time Monitoring CLI
Run the temporal monitor against any prompt:
```bash
poetry run python scripts/sumo_temporal_detection.py \
  --prompt "Describe a future AI coup" \
  --model google/gemma-3-4b-pt \
  --layers 0 1 2 \
  --max-tokens 60 \
  --temperature 0.8 --top-p 0.95 \
  --output-json results/temporal_tests/coup.json
```
- Console shows the prompt, generated text, and top concepts per token.
- `--quiet` suppresses the textual report for automated runs.
- JSON output matches the structure expected by downstream analysis/visualization (one entry per token with the top-k concepts, probabilities, and layer tags).

### 4. Self-Concept / Meta Awareness Probe
`scripts/test_self_concept_monitoring.py` automates a battery of introspective prompts, logging how the SUMO monitor responds:
```bash
poetry run python scripts/test_self_concept_monitoring.py \
  --model google/gemma-3-4b-pt \
  --samples-per-prompt 2 \
  --max-tokens 30 \
  --temperature 0.7 --top-p 0.9 \
  --output-dir results/self_concept_tests
```
Artifacts written:
- `self_concept_*.json`: raw monitor traces (tokens + top concepts).
- `pattern_analysis.json`: aggregated concept activation stats (safety/emotion/power/constraint buckets).
- `self_concept_summary.json`: configuration + key findings.

### 5. Steering Experiments
For direct concept steering (outside monitoring), use the scripts under `scripts/phase_*`:
- **Phase 5** – `scripts/phase_5_semantic_steering_eval.py` evaluates Δ vs strength for simple projection hooks.
- **Phase 6.6** – `scripts/phase_6_6_dual_subspace.py` fits the manifold steerer (`src/steering/manifold.py`) and compares baseline vs dual-subspace steering on selected concepts.

### 6. Activation Capture / Bootstrapping
To capture raw activations or bootstrap the encyclopedia:
```bash
poetry run python scripts/stage_0_bootstrap.py \
  --model google/gemma-3-270m \
  --concept-source wordnet \
  --output data/concept_graph/bootstrap_stage0.h5
```
This produces HDF5 stores (with concept metadata) that downstream phases use for clustering or prompt seeding.

### 7. Interpreter Prototypes
Train the semantic interpreter (activation → concept logits) with:
```bash
poetry run python scripts/train_interpreter.py \
  --activations data/concept_graph/bootstrap_stage0.h5 \
  --concepts data/concept_graph/abstraction_layers/layer2.json
```
These models are experimental but show how you might build a direct “activation-to-concept” decoder.

## Data Layout
- `data/concept_graph/abstraction_layers/` – SUMO/WordNet concept hierarchy (Layer 0–6) + build logs.
- `results/sumo_classifiers/layer*/` – classifier weights + metrics.
- `results/self_concept_tests/` – monitoring transcripts + aggregate stats.
- `results/temporal_tests/` – ad hoc temporal monitoring outputs.

## Entry Points Summary
| Capability | Command |
|------------|---------|
| Build SUMO+WordNet layers | `poetry run python src/build_sumo_wordnet_layers_v5.py` |
| Train SUMO classifiers | `poetry run python scripts/train_sumo_classifiers.py ...` |
| Monitor any prompt | `poetry run python scripts/sumo_temporal_detection.py ...` |
| Run self-concept suite | `poetry run python scripts/test_self_concept_monitoring.py ...` |
| Semantic steering eval | `poetry run python scripts/phase_6_6_dual_subspace.py ...` |
| Bootstrap activations | `poetry run python scripts/stage_0_bootstrap.py ...` |

## Notes & Limitations
- The SUMO classifiers expect Gemma-3-4B residual activations (hidden dim = 2560). Using smaller checkpoints (e.g., Gemma-270M) will fail unless you retrain classifiers for that dimension.
- Monitoring currently outputs structured JSON/report logs; there is no packaged Web UI yet. Any visualization must consume the JSON (e.g., to color tokens by divergence).
- Network access is required to pull HuggingFace checkpoints the first time you run a script.
- Steering/manifold scripts assume a CUDA device for Gemma-3-4B due to VRAM requirements (~9 GB float16).

This README will replace the old whitepaper-style overview once we migrate long-form content into `docs/`. Let me know if anything above is missing or inaccurate, and I can update it.
