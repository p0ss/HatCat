# FTW Repository Restructure Plan

## Goals

1. **Reduce repo size**: Move large artifacts (lens_packs, models, results) out of git
2. **Enable distribution**: Lens packs and concept packs downloadable from HuggingFace
3. **Clean structure**: `src/` mirrors the FTW architecture layers
4. **Easy onboarding**: Clone repo, pip install, packs download on first use

---

## Target Directory Structure

```
ftw/                          # The repo
├── src/
│   ├── hat/                  # Layer 2: Headspace Ambient Transducer
│   │   ├── steering/         # Steering operations
│   │   │   ├── hooks.py      # Steering hooks
│   │   │   ├── extraction.py # Concept vector extraction
│   │   │   ├── manifold.py   # Manifold steering
│   │   │   ├── ontology_field.py
│   │   │   └── subspace.py
│   │   ├── monitoring/       # Real-time concept monitoring
│   │   │   ├── lens_manager.py
│   │   │   ├── monitor.py
│   │   │   ├── detectors.py  # centroid, embedding, text detectors
│   │   │   └── deployment_manifest.py
│   │   ├── classifiers/      # Classifier infrastructure
│   │   │   ├── classifier.py # MLPClassifier, LinearProbe
│   │   │   ├── lens.py       # Lens abstraction
│   │   │   └── capture.py    # Activation capture
│   │   └── utils/            # HAT utilities
│   │       ├── model_loader.py
│   │       ├── storage.py
│   │       ├── provenance.py
│   │       └── gpu_cleanup.py
│   │
│   ├── cat/                  # Layer 2.5: Conjoined Adversarial Tomograph
│   │   └── divergence.py     # Divergence detection
│   │
│   ├── map/                  # Layer 3: Mindmeld Architectural Protocol
│   │   ├── registry/         # Pack management + HF sync
│   │   │   ├── registry.py
│   │   │   ├── concept_pack.py
│   │   │   └── lens_pack.py
│   │   ├── graft/            # Concept grafting (from src/grafting/)
│   │   ├── meld/             # Meld operations (from src/encyclopedia/)
│   │   └── training/         # Lens training (from src/training/)
│   │       ├── train_concept_pack_lenses.py
│   │       ├── sibling_ranking.py
│   │       ├── sumo_classifiers.py
│   │       └── ...
│   │
│   ├── be/                   # Layer 4: Bounded Experiencer
│   │   ├── bootstrap/        # (from src/bootstrap/)
│   │   ├── xdb/              # (from src/xdb/)
│   │   ├── workspace.py      # Global workspace loop
│   │   ├── motive_core.py    # Autonomic regulation
│   │   └── experience.py     # Experience database
│   │
│   ├── hush/                 # Layer 5: Safety Harnesses
│   │   ├── ush.py            # Universal Safety Harness
│   │   └── csh.py            # Chosen Safety Harness
│   │
│   ├── ask/                  # Layer 6: Agentic State Kernel
│   │   ├── contracts.py      # Lifecycle contracts
│   │   └── tribes.py         # Tribal governance
│   │
│   └── ui/                   # Application layer
│       ├── openwebui/        # OpenWebUI integration (from src/openwebui/)
│       ├── streamlit/        # Streamlit apps (from src/ui/)
│       └── visualization/    # Visualization tools (from src/visualization/)
│
├── docs/
│   ├── specification/        # FTW architecture spec
│   ├── guides/               # User guides
│   └── api/                  # API reference
│
├── tests/                    # Test suite
├── scripts/                  # Dev scripts, experiments
│
├── concept_packs/            # .gitignore'd, managed by registry
│   └── .registry.json
│
├── lens_packs/               # .gitignore'd, managed by registry
│   └── .registry.json
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## HuggingFace Structure

### Organization: `ftw-project` (or similar)

```
huggingface.co/ftw-project/
├── concept-pack-first-light      # Concept pack repo
├── lens-apertus-8b-first-light   # Lens pack for Apertus 8B
├── lens-gemma-3-4b-first-light   # Lens pack for Gemma 3 4B
└── ...
```

### Concept Pack Repo Structure
```
concept-pack-first-light/
├── pack.json                 # Pack metadata (spec_id, version, etc.)
├── hierarchy.json            # Concept hierarchy
├── concepts/
│   ├── layer0/
│   ├── layer1/
│   └── ...
└── README.md
```

### Lens Pack Repo Structure
```
lens-apertus-8b-first-light/
├── pack_info.json            # Lens pack metadata
├── layer0/
│   ├── results.json          # Classifier metadata
│   └── classifiers/          # .pt files
├── layer1/
├── ...
└── README.md
```

---

## MAP Registry Design

### Registry Files

Each pack directory has a `.registry.json`:

```json
{
  "schema_version": "1.0",
  "packs": {
    "first-light": {
      "source": "hf://ftw-project/concept-pack-first-light",
      "version": "1.0.0",
      "revision": "abc123",
      "synced_at": "2025-12-20T10:00:00Z",
      "modified": false,
      "size_bytes": 82000000
    },
    "my-custom-pack": {
      "source": "local",
      "version": "0.1.0",
      "created_at": "2025-12-19T...",
      "based_on": "first-light@1.0.0",
      "modified": true
    }
  }
}
```

### Registry API

```python
from ftw.map import registry

# List installed packs
registry.list_concept_packs()  # → [{"name": "first-light", "source": "hf://...", ...}]
registry.list_lens_packs()

# Check for updates
registry.status()  # → shows outdated, modified, etc.

# Pull from HuggingFace
registry.pull_concept_pack("first-light")
registry.pull_lens_pack("apertus-8b-first-light")

# Pull specific version
registry.pull_lens_pack("apertus-8b-first-light", version="1.2.0")

# Push to HuggingFace (requires auth)
registry.push_lens_pack("my-custom-pack", repo_id="username/my-lens-pack")

# Load a pack (auto-pulls if not present)
pack = registry.load_concept_pack("first-light")
lens = registry.load_lens_pack("apertus-8b-first-light", layer=2, concept="Deception")
```

---

## Migration Steps

### Phase 1: Implement Registry ✅ DONE
1. ✅ Create `src/map/registry.py` with core sync logic
2. ✅ Create `src/map/concept_pack.py` and `src/map/lens_pack.py` loaders
3. ✅ Add HuggingFace Hub integration (huggingface_hub library)
4. ✅ Add per-layer pull/push for lens packs
5. ✅ Remove old `src/registry/`, update all imports to `src.map`

### Phase 2: Upload to HuggingFace ⏳ IN PROGRESS
1. ✅ Use `HatCatFTW` organization on HuggingFace
2. ✅ Upload `concept_packs/first-light/` → `concept-pack-first-light`
3. ✅ Upload lens layers 0-3 → `lens-apertus-8b_first-light-layer{N}`
4. ⏳ Upload lens layers 4-6 (pending - ~22GB)

### Phase 3: Clean Up Repo ✅ DONE
1. ✅ Update `.gitignore` to exclude `lens_packs/`, `concept_packs/`, `models/`, `results/`, `data/`, `logs/`

---

## Source Consolidation Phases

The goal is to align `src/` with the FTW architecture layers and eliminate duplication.

### Current Duplication Issues

**MLP Classifier defined 3x:** (RESOLVED - see Phase 4)
- `monitoring/temporal_monitor.py:SimpleMLP`
- `steering/hooks.py:LensClassifier`
- `training/classifier.py` (probably)

**Hook infrastructure duplicated:**
- `activation_capture/hooks.py` - forward hooks for capture
- `steering/hooks.py` - forward hooks for steering
- `monitoring/` - also registers hooks internally

### Phase 4: Unify Classifier Definition ✅ DONE
Created unified HAT module with:

1. ✅ `src/hat/__init__.py` - Module exports
2. ✅ `src/hat/classifier.py` - Unified classifier implementations:
   - `MLPClassifier`: Canonical 128→64→1 architecture (no sigmoid, raw logits)
   - `LinearProbe`: Simple linear probe for comparison
   - `load_classifier()`: Unified loader handling legacy/new state dict formats
   - `save_classifier()`: Unified saver
3. ✅ `src/hat/lens.py` - Lens abstraction:
   - `Lens`: Groups classifiers for a concept across layers
   - `ClassifierInfo`: Metadata for individual classifiers
   - Supports early/mid/late layer categories
   - Translates high-level measure/steer requests to appropriate classifiers

Updated imports (with backwards compatibility aliases):
- ✅ `steering/hooks.py`: `LensClassifier = MLPClassifier`, uses `load_classifier`
- ✅ `monitoring/temporal_monitor.py`: `SimpleMLP = MLPClassifier`, uses `load_classifier`
- ✅ `training/sibling_ranking.py`: Uses `load_classifier`, `save_classifier`
- ✅ `training/lens_validation.py`: Uses `load_classifier`

Multi-classifier metadata support:
- ✅ `src/data/version_manifest.py`: Extended `LensEntry` with `classifiers: Dict[int, ClassifierEntry]`
  - `ClassifierEntry`: layer, category, technique, metrics, file
  - `add_classifier()`: Accumulates classifiers across layers
  - `get_best_layer(category)`: Find best by F1 for early/mid/late
  - `to_hat_lens()`: Convert to HAT Lens object
- ✅ `src/map/lens_pack.py`: Updated loader with manifest/fallback modes
  - `get_lens_for_concept()`: Returns HAT Lens with all classifiers
  - Auto-detects old-format manifests and falls back to directory scanning

Verified: Training, monitoring, and steering all working.

### Phase 5: Move Steering to HAT ✅ DONE
1. ✅ Move `steering/hooks.py` → `hat/hooks.py`
2. ✅ Move `steering/extraction.py` → `hat/extraction.py`
3. ✅ Move `steering/manifold.py` → `hat/manifold.py`
4. ✅ Move `steering/subspace.py` → `hat/subspace.py`
5. ✅ Move `steering/evaluation.py` → `hat/evaluation.py`
6. ✅ Move `steering/ontology_field.py` → `hat/ontology_field.py`
7. ✅ Move `steering/detached_jacobian.py` → `hat/detached_jacobian.py`
8. ✅ Update `src/hat/__init__.py` with all exports
9. ✅ Create backward-compat shims in `src/steering/`:
   - `src/steering/__init__.py` re-exports from `src.hat`
   - Each `src/steering/*.py` file re-exports from `src.hat.*`
10. Existing 30+ files importing from `src.steering` work unchanged

### Phase 6: Merge Activation Capture into HAT ✅ DONE
1. ✅ Review `activation_capture/hooks.py` - Contains ActivationCapture, ActivationConfig, BaselineGenerator
2. ✅ Copy `activation_capture/hooks.py` → `hat/capture.py`
3. ✅ Move `activation_capture/model_loader.py` → `utils/model_loader.py`
4. ✅ Update `src/hat/__init__.py` with capture exports
5. ✅ Update `src/utils/__init__.py` with ModelLoader export
6. ✅ Create backward-compat shims in `src/activation_capture/`:
   - `__init__.py` re-exports from `src.hat.capture`
   - `hooks.py` re-exports from `src.hat.capture`
   - `model_loader.py` re-exports from `src.utils.model_loader`
7. Existing files importing from `src.activation_capture` work unchanged

### Phase 7: Merge Monitoring into HAT ✅ DONE
1. ✅ Move `monitoring/temporal_monitor.py` → `hat/monitor.py`
2. ✅ Move `monitoring/dynamic_lens_manager.py` → `hat/lens_manager.py`
3. ✅ Move `monitoring/concept_dissonance.py` → `cat/divergence.py`
4. ✅ Move `monitoring/sumo_temporal.py` → `hat/sumo_temporal.py`
5. ✅ Move `monitoring/centroid_text_detector.py` → `hat/centroid_detector.py`
6. ✅ Move `monitoring/embedding_text_detector.py` → `hat/embedding_detector.py`
7. ✅ Move `monitoring/text_concept_lens.py` → `hat/text_lens.py`
8. ✅ Move `monitoring/temporal_monitor_mapper.py` → `hat/monitor_mapper.py`
9. ✅ Move `monitoring/deployment_manifest.py` → `hat/deployment_manifest.py`
10. ✅ Create `src/cat/__init__.py` with divergence exports
11. ✅ Fix internal imports in moved files

### Phase 7.5: Clean Up Shim Directories ✅ DONE
Retired all backward-compat shims by updating imports directly:
1. ✅ `src/activation_capture/` - Updated 4 files, deleted directory
2. ✅ `src/steering/` - Updated 40+ files, deleted directory
3. ✅ `src/monitoring/` - Updated 58 files, deleted directory

All code now imports directly from new locations:
- `src.hat.*` - Unified Layer 2 (steering, monitoring, capture, classifiers)
- `src.cat.*` - Layer 2.5 (divergence detection)
- `src.utils.*` - Shared utilities (ModelLoader, storage, provenance)

### Phase 8: Organize HAT Subdirectories ✅ DONE
Created logical subdirectory structure within `hat/`:

1. ✅ `hat/steering/` - 8 files:
   - `hooks.py`, `extraction.py`, `manifold.py`, `ontology_field.py`, `subspace.py`
   - `evaluation.py`, `detached_jacobian.py`, `steering_manager.py`

2. ✅ `hat/monitoring/` - 8 files:
   - `lens_manager.py`, `monitor.py`, `sumo_temporal.py`, `monitor_mapper.py`
   - `centroid_detector.py`, `embedding_detector.py`, `text_lens.py`
   - `deployment_manifest.py`

3. ✅ `hat/classifiers/` - 3 files:
   - `classifier.py`, `lens.py`, `capture.py`

4. ✅ `hat/utils/` - 4 files (moved from `src/utils/`):
   - `model_loader.py`, `storage.py`, `provenance.py`, `gpu_cleanup.py`

5. ✅ Created `__init__.py` for each subdirectory
6. ✅ Updated `hat/__init__.py` to re-export from subdirectories
7. ✅ Updated 100+ external imports
8. ✅ Deleted `src/utils/`

### Phase 9: Consolidate MAP Layer ✅ DONE
1. ✅ Create `map/registry/` and move existing:
   - `registry.py`, `concept_pack.py`, `lens_pack.py`
2. ✅ Create `map/graft/` from `src/grafting/`
3. ✅ Create `map/meld/` from `src/encyclopedia/`
4. ✅ Move `src/training/` → `map/training/`
5. ✅ Update all imports (32 files), delete empty directories
6. ✅ Updated `map/__init__.py` to re-export from all submodules

### Phase 10: Consolidate BE Layer ✅ DONE
1. ✅ Move `src/bootstrap/` → `be/bootstrap/`
2. ✅ Move `src/xdb/` → `be/xdb/`
3. ✅ Update internal imports (grafting → src.map.graft)
4. ✅ Update external imports (src.xdb → src.be.xdb)
5. ✅ Updated `be/__init__.py` to re-export from submodules

### Phase 11: Consolidate UI Layer ✅ DONE
1. ✅ Move `src/openwebui/` → `ui/openwebui/`
2. ✅ Move current `src/ui/` contents → `ui/streamlit/`
3. ✅ Move `src/visualization/` → `ui/visualization/`
4. ✅ Update imports (visualization, streamlit internal)
5. ✅ Created `ui/__init__.py` with exports

### Phase 12: Final Cleanup ✅ DONE
1. ✅ Reviewed remaining directories (data/, interpreter/, testing/)
2. ✅ Created `src/data/__init__.py` with exports
3. ✅ Updated `src/README.md` with full architecture documentation
4. ✅ Verified all layer imports work end-to-end

### Phase 13: Documentation ✅ DONE
1. ✅ Batch updated all old import paths in docs/ and .claude/
2. ✅ Updated root .md files (README.md, QUICKSTART.md, etc.)
3. ✅ Moved `src/interpreter/` → `src/hat/interpreter/`
4. ✅ Updated `src/README.md` with complete architecture guide

---

## .gitignore Additions

```gitignore
# Pack directories (managed by registry)
/concept_packs/
/lens_packs/

# Downloaded models
/models/

# Generated outputs
/results/
/logs/
/data/

# Keep pack registries if you want reproducible environments
# !concept_packs/.registry.json
# !lens_packs/.registry.json
```

---

## Open Questions

1. **Pack naming convention**: `lens-{model}-{concept-pack}` or `{model}_{concept-pack}`?
2. **Concept packs in git vs HF**: Small enough to check in? Or always from HF?
3. **Registry lockfile**: Check in `.registry.json` for reproducibility?
4. **Organization name**: `ftw-project`, `fractal-transparency-web`, `hatcat`?
