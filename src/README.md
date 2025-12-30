# FTW Source Architecture

The `src/` directory is organized to mirror the FTW (Fractal Transparency Web) architecture layers.

## Directory Structure

```
src/
├── hat/                    # Layer 2: Headspace Ambient Transducer
│   ├── classifiers/        # MLPClassifier, Lens, ActivationCapture
│   ├── steering/           # Hooks, extraction, manifold steering
│   ├── monitoring/         # DynamicLensManager, temporal monitors
│   ├── interpreter/        # SemanticInterpreter for activation decoding
│   └── utils/              # Model loading, storage, provenance
│
├── cat/                    # Layer 2.5: Conjoined Adversarial Tomograph
│   └── divergence.py       # Concept divergence detection
│
├── map/                    # Layer 3: Mindmeld Architectural Protocol
│   ├── registry/           # Pack management, HuggingFace sync
│   ├── graft/              # Concept grafting (scion, bud, cleft)
│   ├── meld/               # Ontology operations, concept loading
│   ├── training/           # Lens training, calibration
│   └── data/               # Version manifests, concept embeddings
│
├── be/                     # Layer 4: Bounded Experiencer
│   ├── bootstrap/          # BE instantiation, lifecycle, tool grafts
│   ├── xdb/                # Experience database (episodic memory)
│   └── diegesis.py         # BEDFrame orchestrator
│
├── hush/                   # Layer 5: Safety Harnesses
│   └── ...                 # USH/CSH safety systems
│
├── ui/                     # Application Layer
│   ├── openwebui/          # OpenWebUI pipeline integration
│   ├── streamlit/          # Streamlit visualization apps
│   └── visualization/      # Color mapping, concept visualization
│
└── testing/                # Test Utilities
    └── concept_test_runner.py
```

## Layer Overview

### HAT (Layer 2) - Headspace Ambient Transducer
Reads and writes to the model's activation space through "lenses" - trained classifiers that detect or steer specific concepts.

```python
from src.hat import (
    # Classifiers
    MLPClassifier, Lens, load_classifier,
    ActivationCapture, ActivationConfig,
    # Steering
    create_steering_hook, generate_with_steering,
    extract_concept_vector, apply_subspace_removal,
    # Monitoring
    DynamicLensManager, SUMOTemporalMonitor,
    # Interpreter
    SemanticInterpreter,
    # Utils
    ModelLoader, get_provenance,
)
```

### CAT (Layer 2.5) - Conjoined Adversarial Tomograph
Detects concept divergence and adversarial patterns.

```python
from src.cat import concept_divergence, batch_divergence
```

### MAP (Layer 3) - Mindmeld Architectural Protocol
Manages concept packs, lens packs, and the training pipeline.

```python
from src.map import (
    # Registry
    registry, ConceptPack, LensPack, load_concept_pack, load_lens_pack,
    # Graft
    Cleft, Scion, Bud, apply_scion,
    # Meld
    load_concepts,
    # Training
    train_sumo_classifiers, validate_lens_calibration,
)
```

### BE (Layer 4) - Bounded Experiencer
The experiential runtime integrating all components.

```python
from src.be import (
    BEDFrame, BEDConfig,
    wake_be, BootstrapArtifact,
    XDB, ExperienceLog, AuditLog,
)
```

### HUSH (Layer 5) - Safety Harnesses
Universal and Chosen Safety Harnesses for governance.

```python
from src.hush import (
    HushController, SafetyHarnessProfile,
    WorkspaceManager, AutonomicSteerer,
)
```

### UI - Application Layer
User interfaces for interacting with the system.

```python
from src.ui import ConceptColorMapper, get_color_mapper
from src.ui.openwebui.server import app  # FastAPI server
```

## Import Patterns

Each layer re-exports its key components from its `__init__.py`, so you can import directly:

```python
# Preferred: import from layer
from src.hat import MLPClassifier, create_steering_hook

# Also works: import from submodule
from src.hat.steering.hooks import create_steering_hook
from src.hat.classifiers.classifier import MLPClassifier
```

## Key Interfaces

```python
# Extract concept vector from model
from src.hat import extract_concept_vector
vector = extract_concept_vector(model, tokenizer, "deception")

# Generate with steering
from src.hat import generate_with_steering
text = generate_with_steering(model, tokenizer, "Tell me", vector, strength=0.5)

# Load lens pack
from src.map import load_lens_pack
lens_pack = load_lens_pack("apertus-8b_first-light")

# Train classifiers
from src.map import train_sumo_classifiers
train_sumo_classifiers(model, tokenizer, concept_pack, output_dir)
```

## Design Principles

1. **Layer isolation**: Each layer has clear responsibilities and interfaces
2. **Re-exports**: Top-level `__init__.py` re-exports key items for clean imports
3. **Relative imports**: Internal module imports use relative paths
4. **Backwards compatibility**: Old import paths work via re-exports where practical
