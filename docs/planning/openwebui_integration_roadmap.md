# OpenWebUI Integration Roadmap

Full phase plan for implementing interactive divergence visualization with concept steering in OpenWebUI.

---

## Phase 1: Fix Critical Bugs & Stabilize Current System ✅
**Goal**: Make current single-model setup stable and usable
**Status**: COMPLETED

### Completed:
1. ✅ Fixed CUDA OOM errors in streaming responses
   - Added context truncation (max 2048 tokens input)
   - Disabled KV cache (`use_cache=False`)
   - Implemented sliding window (trim to 1024 tokens)
   - Clear CUDA cache between generation steps

2. ✅ Added comprehensive error handling
   - Catch `torch.cuda.OutOfMemoryError` gracefully
   - Return user-friendly error messages in stream
   - `try/except/finally` with CUDA cleanup

3. ✅ Improved logging
   - Request/response logging via FastAPI
   - Clear initialization messages
   - Error stacktraces for debugging

**Deliverable**: ✅ Stable streaming API that handles OOM and interruptions

---

## Phase 2: Repository Restructure for Multi-Model Support
**Goal**: Support multiple models with different lens sets
**Status**: PENDING

### Tasks:
1. Restructure results directory:
   ```
   results/
     models/
       gemma-3-4b-pt/
         ontologies/
           sumo/
             layers_0-5/
               lenses/
                 activation/  # activation lenses
                 text/        # text lenses
               metadata.json
               concept_positions.json
               training_results.json
       llama-3-8b/
         ontologies/
           sumo/
             layers_0-3/
               lenses/
               metadata.json
   ```

2. Create model registry system:
   - `src/registry/model_registry.py`
   - Tracks available models, ontologies, lens coverage
   - Auto-detection of trained lenses
   - Metadata schema:
     ```json
     {
       "model_id": "gemma-3-4b-pt",
       "ontology": "sumo",
       "layers": [0, 1, 2, 3, 4, 5],
       "concept_count": 5582,
       "training_date": "2025-11-08",
       "base_model_path": "google/gemma-3-4b-pt",
       "lens_paths": {
         "activation": "results/models/gemma-3-4b-pt/ontologies/sumo/layers_0-5/lenses/activation",
         "text": "results/models/gemma-3-4b-pt/ontologies/sumo/layers_0-5/lenses/text"
       }
     }
     ```

3. Update DynamicLensManager:
   - Accept `model_id` parameter
   - Load from registry
   - Support multiple models in memory (model pool)

4. Migration script:
   - Move existing `results/sumo_classifiers_adaptive_l0_5/` to new structure
   - Generate metadata files
   - Update references in scripts

**Deliverable**: Multi-model repository structure with registry

---

## Phase 3: Steering System (Backend)
**Goal**: Enable concept amplification/suppression
**Status**: PENDING

### Tasks:
1. Steering state management:
   - `src/steering/steering_manager.py`
   - Track active steerings per conversation (session ID)
   - User steerings (persistent, high priority)
   - Model steerings (via tool calls, lower priority)
   - Schema:
     ```python
     @dataclass
     class Steering:
         concept: str
         layer: int
         strength: float  # -1.0 to 1.0 (negative = suppress, positive = amplify)
         source: str  # "user" or "model"
         reason: str
         timestamp: datetime
     ```

2. Steering API endpoints:
   - `POST /v1/steering/add` - Add concept steering
     ```json
     {
       "concept": "Proposition",
       "layer": 0,
       "strength": 0.5,
       "reason": "Amplify logical reasoning"
     }
     ```
   - `DELETE /v1/steering/remove/{concept}` - Remove steering
   - `GET /v1/steering/list` - List active steerings
   - `PATCH /v1/steering/update/{concept}` - Adjust strength

3. Apply steering during generation:
   - Modify hidden states based on steering vectors
   - Layer-specific steering application
   - Steering strength normalization
   - Priority system (user > model)

4. Steering metadata in responses:
   - Include active steerings in stream metadata
   - Show which steerings influenced each token
   - Track steering effectiveness

**Deliverable**: Functional steering API with session management

---

## Phase 4: OpenWebUI Custom Function (Frontend Visualization)
**Goal**: Interactive token-level visualization in chat
**Status**: PENDING

### Tasks:
1. Create OpenWebUI custom function (JavaScript):
   - `src/openwebui/hatcat_visualizer.js`
   - Intercept streaming responses
   - Parse divergence metadata from `delta.metadata`
   - Render tokens with sunburst colors
   - Inline styled spans with proper text contrast:
     ```javascript
     function renderToken(token, color, metadata) {
       const luminance = getLuminance(color);
       const textColor = luminance > 0.5 ? '#000000' : '#ffffff';
       return `<span style="
         background-color: ${color};
         color: ${textColor};
         padding: 2px 4px;
         margin: 0 1px;
         border-radius: 3px;
         cursor: help;
       " data-divergence='${JSON.stringify(metadata)}'>
         ${escapeHtml(token)}
       </span>`;
     }
     ```

2. Hover tooltips:
   - Show top 3 divergent concepts
   - Activation vs text scores
   - Divergence values
   - Palette swatches (5 colors)
   - Position tooltip intelligently (above/below token)

3. Click handlers for steering:
   - Right-click concept to open steering menu
   - Options: Amplify / Suppress / Remove
   - Strength slider (0.0 to 1.0)
   - Send to steering API

4. Real-time updates:
   - Listen for steering API changes
   - Update UI when steerings added/removed
   - Visual indicator on active steerings

**Deliverable**: Colored token visualization with hover details and click-to-steer

---

## Phase 5: Steering UI Controls
**Goal**: User interface for steering management
**Status**: PENDING

### Tasks:
1. Steering panel in chat interface:
   - Sidebar or collapsible panel
   - List active steerings with:
     - Concept name
     - Strength bar (visual indicator)
     - Source badge (user/model)
     - Remove button
   - Color-coded by ontology position (sunburst hue)
   - Sort by strength/concept/layer

2. Help panel:
   - Sunburst concept chart (interactive SVG)
     - Click concept to add steering
     - Hover to see details (children, layer, position)
   - Usage guide:
     - How to interpret colors
     - How to steer concepts
     - Keyboard shortcuts
   - Link to full documentation

3. Model selector enhancement:
   - Show lens availability indicator (✓/✗)
   - Display ontology/domain info
   - Concept coverage stats
   - Tooltip with model details
   - Disable if no lenses available

4. Keyboard shortcuts:
   - `Ctrl+H`: Toggle help panel
   - `Ctrl+S`: Toggle steering panel
   - `Ctrl+K`: Clear all steerings
   - `Escape`: Close panels

**Deliverable**: Full steering UI with help and management

---

## Phase 6: Model Steering via Tool Calls
**Goal**: Allow model to request steering changes
**Status**: PENDING

### Tasks:
1. Define steering tool schema:
   ```json
   {
     "name": "add_concept_steering",
     "description": "Request to amplify or suppress a concept during generation",
     "parameters": {
       "concept": "string (concept name from SUMO ontology)",
       "strength": "number (-1.0 to 1.0, negative=suppress, positive=amplify)",
       "reason": "string (why this steering is needed)"
     }
   }
   ```
   - `add_concept_steering(concept, strength, reason)`
   - `remove_concept_steering(concept, reason)`
   - Tool descriptions for model context

2. Permission system:
   - Model steerings are subordinate to user steerings
   - Model cannot override user-set steerings
   - Model steerings auto-expire after conversation
   - User can reject/approve model steering requests

3. Steering rationale tracking:
   - Store why each steering was added
   - Display in UI with expand/collapse
   - Export for analysis
   - Track effectiveness (did it help?)

4. Safety limits:
   - Max steerings per model (e.g., 5)
   - Strength limits for model (e.g., -0.5 to 0.5)
   - Rate limiting (max 1 steering per 10 tokens)

**Deliverable**: Model can request steering via tools with safety guardrails

---

## Phase 7: Update Project Documentation
**Goal**: Document the full system
**Status**: PENDING

### Tasks:
1. Update README.md:
   - Multi-model support section
   - Steering system overview
   - OpenWebUI setup with visualization
   - Architecture diagram

2. Create comprehensive docs:
   - `docs/multi_model_setup.md` - How to train/add new models
   - `docs/steering_guide.md` - How to use steering effectively
   - `docs/openwebui_integration.md` - Complete setup guide
   - `docs/project_progress.md` - Track divergence work progress
   - `docs/api_reference.md` - Full API documentation

3. API documentation:
   - Generate OpenAPI spec (`/v1/openapi.json`)
   - Steering API reference with examples
   - Example curl requests
   - Python client library examples

4. Tutorial videos/GIFs:
   - Basic usage demo
   - Steering walkthrough
   - Sunburst chart explanation

**Deliverable**: Complete documentation with tutorials

---

## Success Metrics

- ✅ Stable streaming with no OOM errors (even with long conversations)
- ⏳ Support for 3+ models with different lens sets
- ⏳ User can steer concepts and see effect in real-time
- ⏳ Model can request steering via tool calls
- ⏳ Full token-level color visualization in OpenWebUI
- ⏳ Sub-200ms per-token latency with steering applied

---

## Estimated Timeline

- **Phase 1**: ✅ COMPLETE (3 hours)
- **Phase 2**: 4-6 hours (repository restructure + migration)
- **Phase 3**: 6-8 hours (steering backend)
- **Phase 4**: 8-10 hours (frontend visualization - most complex)
- **Phase 5**: 4-6 hours (steering UI)
- **Phase 6**: 3-4 hours (model tool calls)
- **Phase 7**: 2-3 hours (documentation)

**Total Remaining**: ~27-37 hours of work

---

## Current Status Summary

**Completed**: Phase 1 - Streaming now stable with proper error handling

**Next Up**: Phase 2 - Repository restructure for multi-model support

**Blockers**: None currently

**Notes**:
- OOM issue resolved by disabling KV cache and implementing sliding window
- Ready to proceed with multi-model architecture
- Consider adding flash-attention in future for better memory efficiency
