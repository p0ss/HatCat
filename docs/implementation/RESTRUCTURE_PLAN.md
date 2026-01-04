# FTW Repository Restructure Plan

## Goals

1. **Reduce repo size**: Move large artifacts (lens_packs, models, results) out of git
2. **Enable distribution**: Lens packs and concept packs downloadable from HuggingFace
3. **Clean structure**: `src/` mirrors the FTW architecture layers
4. **Easy onboarding**: Clone repo, pip install, packs download on first use

---

## Target Directory Structure

```
hatcat/                          # The repo
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

### Phase 2: Upload to HuggingFace ✅ DONE
1. ✅ Use `HatCatFTW` organization on HuggingFace
2. ✅ Upload `concept_packs/first-light/` → `concept-pack-first-light`
3. ✅ Upload `lens_packs/gemma-3-4b_first-light-v1-bf16/` → `lens-gemma-3-4b-first-light-v1`
   - Converted to bf16 (5.1GB, down from 11GB fp32)
   - 7,947 lenses across layers 0-6
   - Model card with usage instructions

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

## Performance Optimization Phases

### Current State

The codebase has performance issues from mixed numpy/torch usage:
- **43** `.numpy()` calls in `src/` (GPU→CPU round-trips)
- **117** `np.ndarray` type hints in GPU-adjacent code
- Numpy linear algebra ops on data that should stay on GPU
- Dtype mismatches: models run bfloat16, some lenses trained float32

### Phase 14: Torch-Native Conversion (Assessment) ✓ COMPLETE

**Goal**: Map all numpy usage in GPU-adjacent code and plan conversion order.

**Status**: Complete. Benchmark showed:
- Single numpy ops faster than single torch GPU ops (kernel launch ~10µs overhead)
- Batched torch 10x faster than sequential
- GPU→CPU→GPU round-trips are primary waste

**Priority Order** (user-specified):
1. ✓ Eliminate GPU→CPU→GPU round-trips in hot path
2. ✓ Batch lens inference in `detect_and_expand`
3. ✓ Verify batched accuracy matches sequential
4. ✓ Batch steering application (already batched per-layer)

| Module | Numpy Uses | Downstream | Priority |
|--------|------------|------------|----------|
| `hat/steering/hooks.py` | `np.dot` projections | manifold, hush | High |
| `hat/steering/manifold.py` | norm, blend ops | behavioral eval | Medium |
| `hat/monitoring/centroid_detector.py` | `np.dot` similarity | divergence | Low |
| `hat/monitoring/embedding_detector.py` | `np.dot` similarity | divergence | Low |
| `hush/autonomic_steering.py` | Full numpy backend | hush_integration | High |
| `cat/divergence.py` | cosine similarity | openwebui | Medium |
| `map/training/sumo_classifiers.py` | activations, vectors | training scripts | Medium |

Tasks:
1. [ ] Run `grep -rn "\.numpy()\|np\." src/` and categorize by hot-path vs cold-path
2. [ ] For each hot-path module, identify downstream scripts that may break
3. [ ] Create test cases capturing current numerical behavior
4. [ ] Document conversion order respecting dependencies

### Phase 15: Torch-Native Conversion (Core Modules) 🔄 IN PROGRESS

Convert core modules that are in the inference hot-path:

#### 15.1: `hat/steering/hooks.py` ✓ COMPLETE
- ~~Replace `np.dot(a, b)` → `a @ b` or `torch.dot()`~~
- ~~Replace `np.linalg.norm()` → `tensor.norm()`~~
- ✓ Eliminated 4 GPU→CPU→GPU round-trips:
  - Line ~923: gradient steering (now stays on GPU with clone+requires_grad)
  - Line ~1024: contrastive steering (same pattern)
  - Line ~1234: multi-classifier steering (hidden_base stays on GPU)
  - Line ~1195: layer vector extraction (torch.norm instead of np.linalg.norm)
- Verify: steering tests pending

#### 15.2: `hush/autonomic_steering.py`
- Full rewrite from numpy to torch
- `SteeringChannel` → torch tensor state
- `AutonomicSteerer` → batched torch ops
- Verify: hush integration tests, intervention behavior unchanged

#### 15.3: `hat/steering/manifold.py`
- Convert blending/norm operations
- Verify: manifold steering tests

#### 15.4: `hat/monitoring/*_detector.py`
- Convert similarity calculations
- Verify: detection scores match

### Phase 16: Torch-Native Conversion (Support Modules)

Convert modules not in hot-path but still doing unnecessary conversions:

#### 16.1: `cat/divergence.py`
- Convert cosine similarity
- Verify: divergence detection unchanged

#### 16.2: `map/training/sumo_classifiers.py`
- Already partially done (bfloat16 default)
- Remove remaining `.numpy()` where not needed for sklearn/serialization
- Verify: training produces same quality lenses

#### 16.3: Remaining files
- Sweep through all 43 `.numpy()` calls
- Convert where beneficial, document where numpy is required (sklearn, disk I/O)

### Phase 17: CUDA Kernel Assessment

**Goal**: Profile hot paths and determine if custom kernels are warranted.

#### 17.1: Profiling
```bash
python -m torch.profiler tests/profile_lens_inference.py
python -m torch.profiler tests/profile_autonomic_steering.py
```

Questions to answer:
- What % of time is kernel launch overhead vs compute?
- Are there many small sequential kernels that could fuse?
- What's the memory bandwidth utilization?

#### 17.2: Identify Fusion Candidates

**Lens Manager** (`detect_and_expand`):
```
Current: N separate lens forward passes
Fused:   Single batched forward with stacked weights
```

**Autonomic Steering**:
```
Current: Loop over channels computing corrections
Fused:   Batched correction computation
```

#### 17.3: Benchmark Alternatives

| Alternative | Effort | Expected Gain | Try First? |
|-------------|--------|---------------|------------|
| `torch.compile()` | Low | 2-3x | Yes |
| `torch.jit.script()` | Low | 1.5-2x | Yes |
| Triton kernels | Medium | 3-5x | If above insufficient |
| `torch.cuda.graphs` | Low | 2x | For repeated ops |
| Raw CUDA | High | 5-10x | Only if critical |

### Phase 18: Lens Inference Optimization ✓ COMPLETE

**Goal**: Sub-millisecond per-token lens inference.

#### 18.1: Batched Lens Forward ✓ COMPLETE
- ✓ Created `BatchedLensBank` class in lens_manager.py
- ✓ Stacks all lens weights into batched tensors: W1[N,128,input], W2[N,64,128], W3[N,1,64]
- ✓ Uses `torch.bmm` for batched matmul across all lenses
- ✓ Integrated into `DynamicLensManager.detect_and_expand()`
- ✓ Lazy rebuild with dirty flag when lenses change

**Results** (50 lenses, 100 iterations):
- Sequential: 2.43 ms
- Batched: 0.41 ms
- **Speedup: 6x**

Test: `tests/test_batched_lens_inference.py`
- Verified numerical accuracy (max diff < 1e-5)
- Handles bfloat16/float32 dtypes correctly

#### 18.2: torch.compile() Integration
- Deferred - 6x speedup sufficient for current needs
- Can add `@torch.compile(mode="reduce-overhead")` if more needed

#### 18.3: Optional - Triton Kernel
- Deferred - batched bmm approach is sufficient

### Phase 19: BFloat16 Lens Optimization ✓ COMPLETE

**Goal**: Reduce lens pack size and loading time.

#### 19.1: Training Defaults ✓
- `train_simple_classifier()` now defaults to `dtype=torch.bfloat16`
- New lenses trained at half precision automatically

**REGRESSION FIX (Phase 22)**: BFloat16 training without input normalization caused
gradient saturation. Higher layers have ~30,000x larger activation magnitudes than lower
layers. With float32, gradients could survive saturation; bfloat16's limited precision
killed them completely, causing all models to predict constant 0.667 F1 (all one class).

**Fix**: Added `normalize_inputs=True` to `train_simple_classifier()` which applies
per-sample normalization (matching `nn.LayerNorm` at inference). This ensures training
and inference see the same normalized inputs regardless of layer magnitude.

#### 19.2: Conversion Script ✓
- Created `scripts/convert_lenses_to_bf16.py`
- Parallel conversion of existing fp32 packs to bf16
- Preserves all metadata, updates pack_info.json

#### 19.3: Direct GPU Loading ✓
- Changed `torch.load(..., map_location='cpu')` to `map_location=self.device`
- Eliminates CPU→GPU copy overhead

**Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Pack size | 26.64 GB | 13.34 GB | **50% smaller** |
| Per-lens load (GPU) | 4.25 ms | 0.65 ms | **6.5x faster** |
| Child loading (72 lenses) | 186 ms | 100 ms | **46% faster** |
| Detection avg | 137 ms | 108 ms | **21% faster** |

Converted pack: `lens_packs/apertus-8b_first-light-bf16`

### Phase 20: Training Optimization

**Goal**: Faster lens training for large concept packs.

#### 20.1: Batched Activation Extraction
- Current: One prompt at a time
- Proposed: Batch extraction with padding

#### 20.2: Parallel Concept Training
- Train multiple concepts concurrently (CUDA streams)
- Or distributed across GPUs

### Phase 21: Tiered Memory Architecture 🔄 IN PROGRESS

**Goal**: Minimize lens loading latency during hierarchical expansion.

**Problem Identified**: Benchmark showed 300-600ms child loading times during `detect_and_expand`:
- Batched inference on base lenses: ~6ms (excellent)
- Child loading from disk: ~3.5ms per lens (bottleneck)
- Most time spent in `torch.load()` deserialization + GPU transfer

**Solution**: Four-tier memory hierarchy:

```
1. HOT VRAM      → BatchedLensBank, scored every token (~6ms for 269 lenses)
2. WARM VRAM    → GPU tensors waiting for parent activation
3. TEPID RAM     → CPU tensors pre-loaded at startup, just .to(device) when needed
4. COLD DISK     → Only if RAM can't hold pack (fallback)
```

#### 21.1: Tepid Cache Implementation ✓ COMPLETE
- Added `tepid_cache: Dict[Tuple[str,int], Dict[str, Tensor]]` for CPU tensors
- Added `preload_pack_to_ram()` method to load entire pack to RAM at startup
- Modified `_load_concepts()` to check tepid cache before disk:
  - Already active → skip
  - In warm cache (VRAM) → move to active
  - In tepid cache (RAM) → `.to(device)` transfer
  - Otherwise → `torch.load()` from disk

#### 21.2: Benchmark Results

Pre-load stats:
- 7947 concepts loaded to RAM
- 8075 MB (~8GB) in 12 seconds (one-time startup cost)

| Metric | Before (disk) | After (tepid RAM) | Change |
|--------|--------------|-------------------|--------|
| Avg detection | 57ms | 48ms | -16% |
| P50 | 34ms | 28ms | -18% |
| P95 | 192ms | 154ms | -20% |
| Overhead | 169% | 143% | -26% |

Tepid cache working (182 tepid_hits per detection), but sequential `.to(device)` calls still have overhead (~2ms per lens).

#### 21.3: Further Optimizations (Pending)
- [ ] Batch GPU transfers: stack tensors before `.to(device)`
- [ ] Pre-warm likely children during prompt processing
- [ ] Consider pinned memory for faster CPU→GPU transfers
- [ ] Add `max_ram_mb` budget configuration

### Phase 22: Training Pipeline Fixes ✓ COMPLETE

**Goal**: Fix training regressions introduced in earlier phases.

#### 22.1: Input Normalization ✓
**Problem**: Phase 19's bfloat16 switch broke training for higher layers.
- Higher layers have ~30,000x larger activation magnitudes than lower layers
- First linear layer output immediately saturated sigmoid
- Gradients vanished, model predicted constant value (F1=0.667)

**Fix**: Added `normalize_inputs=True` to `train_simple_classifier()`:
```python
# Per-sample normalization (matches nn.LayerNorm at inference)
train_mean = X_train.mean(axis=1, keepdims=True)
train_std = X_train.std(axis=1, keepdims=True) + 1e-8
X_train = (X_train - train_mean) / train_std
```

#### 22.2: Stuck Training Detection ✓
**Problem**: `train_concept()` would loop forever on same data if model never graduated.
- Training data selected deterministically (first N samples, no shuffle)
- `validation_cycle` only incremented after graduation + validation failure
- If graduation never happened, same data reused every iteration

**Fix**: Added `iterations_this_cycle` counter with auto-escalation:
- After 3 iterations without graduation, increment `validation_cycle`
- This requests more samples (40 → 80 → 120...) to give model more to work with

#### 22.3: Import Path Fixes ✓
**Problem**: 32 files had incorrect `sys.path` calculations after restructure.
- Files in `src/<module>/<submodule>/` needed 4 levels of `.parent`
- Files in `scripts/<category>/` needed 3 levels of `.parent`
- Many had incorrect counts, causing `ModuleNotFoundError`

**Fix**: Updated all affected files with correct path calculations.

#### 22.4: Validation Dtype/Normalization Fix ✓
**Problem**: Lens calibration validation failed with "mat1 and mat2 must have same dtype".
- Validation created float32 tensors but classifier is bfloat16
- Validation also didn't apply per-sample normalization like training does

**Fix**: Updated validation in `dual_adaptive_trainer.py`:
- Get dtype from classifier parameters, not hardcoded float32
- Apply same per-sample normalization as training before inference

---

## Open Questions

1. **Pack naming convention**: `lens-{model}-{concept-pack}` or `{model}_{concept-pack}`?
2. **Concept packs in git vs HF**: Small enough to check in? Or always from HF?
3. **Registry lockfile**: Check in `.registry.json` for reproducibility?
4. **Organization name**: `ftw-project`, `fractal-transparency-web`, `hatcat`?
5. **Triton vs raw CUDA**: Is Triton mature enough for production kernels?
6. **torch.compile stability**: Version pin needed for reproducibility?
