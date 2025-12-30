# MAP Compliance Migration Plan

## Status: In Progress

This document tracks the migration from legacy lens pack structure to MAP-compliant structure.

## Current State

### Legacy Structure
```
lens_packs/
  gemma-3-4b-pt_sumo-wordnet-v1/
  gemma-3-4b-pt_sumo-wordnet-v2/
  gemma-3-4b-pt_sumo-wordnet-v3/
```

### MAP-Compliant Structure (Target)
```
lens_packs/
  concept_packs/
    org.hatcat/
      sumo-wordnet-v4@4.0.0/
        pack.json              # Contains spec_id, version, concepts
  gemma-3-4b-pt_sumo-wordnet-v4/  # Lens pack binds to substrate
    pack.json                      # References concept pack spec_id
    layer_*.pt                     # Trained lenses
```

## Implementation Phases

### Phase 1: DynamicLensManager MAP Support ✅ PLANNED
**Goal:** Add MAP-aware discovery and backward compatibility

**Files to modify:**
- `src/monitoring/dynamic_lens_manager.py`

**New Methods:**
```python
@staticmethod
def discover_concept_packs() -> Dict[str, Path]
@staticmethod
def discover_lens_packs(substrate_id: str = None) -> Dict[str, Dict]
@staticmethod
def get_latest_version(concept_pack_name: str) -> Optional[str]
def _resolve_map_path(spec_id, substrate_id) -> Path
def _resolve_legacy_path(lens_pack_id) -> Path
```

**Constructor Changes:**
- Add `spec_id` parameter (MAP-compliant)
- Add `substrate_id` parameter
- Keep `lens_pack_id` with deprecation warning
- Auto-resolve paths based on which parameters provided

**Priority:** HIGH
**Blockers:** None
**Estimated Effort:** 4-6 hours

---

### Phase 2: Streamlit UI Update ⏳ PENDING
**Goal:** Add pack selection dropdown to Streamlit

**Files to modify:**
- `src/ui/streamlit_chat.py`

**Changes:**
1. Add `get_available_packs()` cached function
2. Add sidebar pack selector
3. Pass selected `spec_id` to `load_model_and_lenses()`
4. Update manager initialization to use `spec_id` + `substrate_id`

**Dependencies:** Phase 1
**Priority:** HIGH
**Estimated Effort:** 2-3 hours

---

### Phase 3: OpenWebUI Server Update ⏳ PENDING
**Goal:** Add config file and MAP-compliant loading

**Files to create:**
- `src/openwebui/config.yaml`

**Files to modify:**
- `src/openwebui/server.py`

**Changes:**
1. Create default config.yaml
2. Add `_load_config()` method
3. Update `initialize()` to use spec_id
4. Add `/hatcat/pack-info` endpoint
5. Add `/hatcat/available-packs` endpoint

**Dependencies:** Phase 1
**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours

---

### Phase 4: Testing & Migration ⏳ PENDING
**Goal:** Verify MAP compliance works

**Files to create:**
- `scripts/test_map_compliance.py`
- `docs/MAP_MIGRATION.md` (user guide)

**Test Scenarios:**
1. Discovery of concept packs
2. Discovery of lens packs
3. Version resolution (latest)
4. MAP-compliant loading
5. Legacy loading (with deprecation warning)
6. Streamlit with both legacy and MAP packs
7. OpenWebUI with MAP config

**Dependencies:** Phases 1-3
**Priority:** HIGH
**Estimated Effort:** 2-3 hours

---

## Migration Checklist

### Phase 1: DynamicLensManager
- [ ] Add pack discovery methods
- [ ] Implement path resolution
- [ ] Update __init__ for backward compatibility
- [ ] Add deprecation warnings for legacy usage
- [ ] Test locally with v2 and v3 packs

### Phase 2: Streamlit
- [ ] Implement pack discovery UI
- [ ] Add selection dropdown
- [ ] Update manager initialization
- [ ] Test with legacy v2
- [ ] Test with MAP v4 (when available)

### Phase 3: OpenWebUI
- [ ] Create config.yaml template
- [ ] Implement config loading
- [ ] Update server initialization
- [ ] Add pack info endpoints
- [ ] Test with config

### Phase 4: Testing
- [ ] Create test_map_compliance.py
- [ ] Run discovery tests
- [ ] Run loading tests
- [ ] Document migration steps
- [ ] Update deployment docs

---

## Timeline

**Total Estimated Effort:** 11-16 hours

**Milestones:**
1. Phase 1 Complete: DynamicLensManager supports both legacy and MAP ✅ NEXT
2. Phase 2 Complete: Streamlit has pack selection UI
3. Phase 3 Complete: OpenWebUI uses MAP config
4. Phase 4 Complete: All tests pass, docs written
5. **Release Ready:** All UIs support both v2/v3 (legacy) and v4 (MAP)

---

## Notes

- **Backward Compatibility:** Critical during transition period
- **Deprecation Timeline:** Legacy support maintained until v5.0
- **Auto-migration:** Consider script to convert v2/v3 to v4 structure
- **Version Detection:** DynamicLensManager auto-detects structure from pack.json

## Known Issues

### Streamlit Cache Problem
**Symptom:** Streamlit UI shows error "Lenses directory not found: results/sumo_classifiers"

**Cause:** Streamlit's `@st.cache_resource` decorator caches the `load_model_and_lenses()` function. If it was previously called with default parameters before MAP migration, it may use the old default `lenses_dir` path.

**Solutions:**
1. **Clear Streamlit cache:** Press `C` in the Streamlit UI or restart the server
2. **Manual cache clear:** Delete `.streamlit/cache` directory in project root
3. **Force cache invalidation:** Run streamlit with `--server.enableStaticServing=false` flag

**Workaround:** The Streamlit UI explicitly passes `lens_pack_id` parameter, so this should not occur in normal operation. Only affects development environments with stale caches.

---

## Next Steps

1. Start with Phase 1.1: Implement `discover_concept_packs()`
2. Implement Phase 1.2: Add backward-compatible constructor
3. Test locally before moving to UI updates

**Assigned:** Claude Code
**Started:** 2025-11-29
**Target Completion:** TBD (depends on v4 pack availability)
