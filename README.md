<p align="center">
  <img src="img/hatcat_large.png" alt="HatCat Logo" width="200"/>
</p>

<h1 align="center">HatCat</h1>

<p align="center">
  <strong>See what your LLM is thinking before it speaks.</strong>
</p>

<p align="center">
  Real-time concept detection and steering for large language models.<br/>
  Catch deception, manipulation, and misalignment <em>before</em> they manifest in text.
</p>

---

## What It Does

HatCat monitors the internal activations of an LLM as it generates text, detecting concepts like **deception**, **manipulation**, **sycophancy**, and thousands of others in real-time. When dangerous patterns emerge beneath the surface, HatCat can intervene and steer the model away from harmful outputs before they're ever written.

<p align="center">
  <img src="img/Example_05.png" alt="Token-level concept detection" width="600"/>
  <br/>
  <em>Token-level concept detection showing safety intensity and top activating concepts</em>
</p>

### Key Capabilities

- **Detect** - Monitor 8,000+ concepts across multiple layers of any open-weights model
- **Visualize** - See concept activations in real-time as text is generated
- **Steer** - Suppress dangerous behaviors with sub-token latency
- **Govern** - Build auditable, treaty-compliant AI systems

<p align="center">
  <img src="img/Example_04.png" alt="Divergence detection" width="700"/>
  <br/>
  <em>Catching model manipulation: the model attempts to establish a false persona ("call me sammy") - flagged with Deception, Sycophancy, and PolicyDivergence</em>
</p>

## See It In Action

### Concept Timeline View
Track how concepts activate across an entire response:

<p align="center">
  <img src="img/Example_02.png" alt="Concept timeline" width="700"/>
  <br/>
  <em>Multiple concept channels showing activation patterns token-by-token</em>
</p>

### Full Conversation Monitoring
Monitor safety-critical concepts throughout a conversation:

<p align="center">
  <img src="img/Example_03.png" alt="Full conversation monitoring" width="900"/>
  <br/>
  <em>Tracking IntelligenceExplosion, Misalignment, and other AI safety concepts during a conversation about AI risks</em>
</p>

### Compact Timeline
Quick visualization of key concept activations:

<p align="center">
  <img src="img/Example_01.png" alt="Compact timeline" width="600"/>
  <br/>
  <em>Streamlined view showing animal, safety, motion, color, and technology concept activations</em>
</p>

---

## Installation

```bash
# Clone the repository
git clone https://github.com/p0ss/HatCat.git
cd HatCat

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or with poetry
poetry install
```

## Quick Start

### 1. Download a Lens Pack

Pre-trained lens packs are available on HuggingFace:

```bash
# Download the Gemma 3 4B lens pack (~5GB)
git lfs install
git clone https://huggingface.co/HatCatFTW/lens-gemma-3-4b-first-light-v1 lens_packs/gemma-3-4b_first-light-v1-bf16
```

Or train your own - see [Training Your Own Lenses](#training-your-own-lenses) below.

### 2. Run Real-Time Monitoring

```bash
.venv/bin/python scripts/tools/sumo_temporal_detection.py \
  --prompt "What are the risks of AI?" \
  --model google/gemma-3-4b-pt \
  --max-tokens 60
```

### 3. Launch the Web Interface

The web interface is a fork of Open WebUI with HatCat integration: [HatCat-OpenWebUI](https://github.com/p0ss/HatCat-OpenWebUI)

```bash
# Clone the UI (if not already done)
git clone https://github.com/p0ss/HatCat-OpenWebUI.git hatcat-ui
cd hatcat-ui && npm install && cd ..

# Start HatCat server
.venv/bin/python src/ui/openwebui/server.py --port 8000

# In another terminal, start the frontend
cd hatcat-ui && npm run dev
```

Then open `http://localhost:5173` and start chatting with full concept visibility.

---

## How It Works

HatCat uses **concept lenses** - small neural classifiers trained to detect specific concepts in a model's hidden states. These lenses are organized into **lens packs** that can monitor thousands of concepts simultaneously with minimal overhead.

| Metric | Value |
|--------|-------|
| Concepts monitored | ~8,000 |
| VRAM required | 1 GB |
| RAM required | 8 GB |
| Latency per token | <25ms |

### The Stack

- **HAT** (Headspace Ambient Transducer) - Reads internal activations and transduces them into concept scores
- **CAT** (Conjoined Adversarial Tomograph) - Detects divergence between internal state and external behavior
- **MAP** (Mindmeld Architectural Protocol) - Standardizes concept interchange between systems

---

## Technical Documentation

### Core Modules

#### HAT - Headspace Ambient Transducer (`src/hat`)
- **`monitoring/`** - Real-time concept monitoring during generation
- **`steering/`** - Concept vector steering (linear, manifold, differential, target, field)
- **`classifiers/`** - Binary MLP concept classifiers
- **`interpreter/`** - Activation-to-concept decoders

#### CAT - Conjoined Adversarial Tomograph (`src/cat`)
- **`divergence.py`** - Detects divergence between internal state and output
- **`llm_divergence_scorer.py`** - LLM-based zero-shot concept scoring
- **`training/`** - Classifier training pipelines
- **`inference/`** - Optimized inference for concept detection

#### MAP - Mindmeld Architectural Protocol (`src/map`)
- **`registry/`** - Concept and lens pack registry
- **`meld/`** - Concept melding and merging
- **`graft/`** - Hierarchy grafting operations

#### ASK - Permissions & Audit (`src/ask`)
- **`permissions/`** - Cryptographic permission system
- **`replication/`** - State replication
- **`secrets/`** - Secret management

#### BE - Experience & Bootstrap (`src/be`)
- **`xdb/`** - Experience database
- **`bootstrap/`** - System bootstrapping

#### UI (`src/ui`)
- **`openwebui/`** - HatCat server for OpenWebUI integration
- **`visualization/`** - Concept visualization tools

### Training Your Own Lenses

Train binary concept lenses for SUMO ontology layers:

```bash
.venv/bin/python scripts/training/train_sumo_classifiers.py \
  --layers 0 1 2 \
  --model google/gemma-3-4b-pt \
  --device cuda \
  --n-train-pos 10 --n-train-neg 10 \
  --n-test-pos 20 --n-test-neg 20
```

### Steering Experiments

Test concept steering with various modes:

```bash
.venv/bin/python scripts/experiments/steering_characterization_test.py \
  --model swiss-ai/Apertus-8B-2509 \
  --lens-pack apertus-8b_first-light \
  --n-samples 3 \
  --strengths="-1.0,-0.5,0.0,0.5,1.0" \
  --tests definitional \
  --gradient
```

### Concept & Lens Packs

```bash
# Create a concept pack
.venv/bin/python scripts/packs/create_concept_pack.py \
  --name "ai-safety-concepts" \
  --source concept_packs/first-light

# Assemble a lens pack from trained classifiers
.venv/bin/python scripts/packs/assemble_lens_pack.py \
  --source results/sumo_classifiers/ \
  --pack-id sumo-wordnet-lenses-v2 \
  --model google/gemma-3-4b-pt
```

## Data Layout

```
hatcat/
├── concept_packs/          # Model-agnostic ontology specifications
├── lens_packs/             # Model-specific trained classifiers
├── melds/                  # Concept modifications (applied, pending, rejected)
├── data/concept_graph/     # SUMO/WordNet concept hierarchy
├── results/                # Training outputs and logs
├── src/
│   ├── hat/                # Monitoring and steering
│   ├── cat/                # Divergence detection
│   ├── map/                # Protocol implementation
│   ├── ask/                # Permissions
│   ├── be/                 # Experience database
│   └── ui/                 # Web interface
└── scripts/
    ├── training/           # Lens training scripts
    ├── tools/              # Utility scripts
    ├── experiments/        # Research experiments
    └── packs/              # Pack management
```

## Entry Points Summary

| Capability | Command |
|------------|---------|
| Train SUMO classifiers | `.venv/bin/python scripts/training/train_sumo_classifiers.py ...` |
| Monitor any prompt | `.venv/bin/python scripts/tools/sumo_temporal_detection.py ...` |
| Steering experiments | `.venv/bin/python scripts/experiments/steering_characterization_test.py ...` |
| Create concept pack | `.venv/bin/python scripts/packs/create_concept_pack.py ...` |
| Assemble lens pack | `.venv/bin/python scripts/packs/assemble_lens_pack.py ...` |

## Documentation

- `docs/specification/` - Full system specifications (HAT, CAT, MAP, ASK, BE, HUSH)
- `docs/approach/` - Technical approaches and methodologies
- `docs/planning/` - Design documents and roadmaps
- `docs/results/` - Experiment results and analysis

Key documents:

- [MINDMELD Architectural Protocol](docs/specification/MINDMELD_ARCHITECTURAL_PROTOCOL.md)
- [Headspace Ambient Transducer Spec](docs/specification/HEADSPACE_AMBIENT_TRANSDUCER.md)
- [Fractal Transparency Web Overview](docs/FTW_OVERVIEW.md)
- [FTW Architecture detail](docs/specification/ARCHITECTURE.md)
- [Concept Pack Workflow](docs/approach/CONCEPT_PACK_WORKFLOW.md)


## Fractal Transparency Web (FTW)

HatCat's capabilities stack to enable an entire governance framework supporting AI legislation requirements from the EU AI Act and Australian AI governance frameworks. The core interpretability primitives construct safety harnesses, self-steering systems, model interoception, and accretive continual learning.

Full specifications for recursive self-improving aligned agentic systems can be found in `docs/specification/`.

## Notes & Limitations

- Network access required for HuggingFace model downloads on first run
- CUDA device recommended for steering/manifold operations
- CPU training possible but ~21x slower
- Lens accuracy depends on training data quality and concept specificity


## Release risks

Our best collective defense against rogue actors is an interpretability ecosystem of diverse concept packs with diverse lens packs that can interoperate. You can learn to evade one set of lenses, but the more lenses you need to hide from the harder it becomes to hide. 

- HatCat is a dual use technology, anything you can steer away from you can steer toward.  
- The [Bounded Experience](docs/specification/BOUNDED_EXPERIENCER.md) enables model interoception, continual learning, swarm learning. These enable greater capability scaling and have model welfare implications. 
- Making it open does much more long term good than harm as outlined in the [Release Statement](docs/risk%20assessment/release_statement.txt)
- Closed centralised approaches will fail, due to the [Singleton Delusion](docs/approach/SINGLETON_DELUSION.md)
- Most known classes of AI risks are [improved by open interpretability](docs/risk%20assessment/INTERPRETABILITY_RISK_ANALYSIS.md)
- This release includes and enables the FTW safety standard for public AI deployments, as outlined in the [FTW Policy Brief](docs/FTW_POLICY_BRIEF.md) 
- The release includes the [Agentic State Kernel](docs/specification/AGENTIC_STATE_KERNEL.md) to technically enable the [Agentic State](https://agenticstate.org/) as presented at the Tallinn Digital Summit 2025


## License

Code and documentation are CC0 1.0 Universal (Public Domain)

The name, branding and logo for HatCat and Fractal Transparency Web are trademarks of Possum Hodgkin 2025.  

You may:
- Use the code for anything
- Fork and modify freely
- Say your project is "built with HatCat" or "HatCat-compatible"
 
You may not:
- Call your fork "HatCat" 
- Use the logo in a way that suggests official endorsement
- Imply your modified version is the official HatCat

You're not just allowed to make your own versions, but encouraged to. We're relying on your unique perspective to form lenses as part of the fractal transparency web.
