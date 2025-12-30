# Refactor Plan: probe → lens

## Rationale

"Probe" carries connotations of invasive medical/surveillance procedures. "Lens" better captures the bidirectional nature of these tools - they can observe and influence, like optical lenses in laser circuits or EUV lithography. The term also fits the "telescope web" interpretability vision and supports the substrate-independent, cooperative oversight philosophy of HatCat.

## Terminology Clarification

- **Lens** = the abstract concept detector (what we're looking through)
- **Probe** = the specific implementation technology (linear classifier trained on activations)
- **Aperture** = the set of lenses that can be enabled/disabled (controls field of view)
- **Blindspot** = concepts/violations outside current lens coverage

## Explanatory Text (for docs)

> **Lens**: A trained classifier that detects and can influence specific concepts in a model's internal representations. While often implemented as linear probes, "lens" better captures the bidirectional nature of these tools - they focus attention on specific aspects of model cognition, and like optical lenses, can be used to observe, magnify, or direct. The term also reflects the recursive stacking behavior in HatCat's architecture, where lenses at different abstraction layers create effects analogous to optical systems: zoom, focus, and sometimes distortion.

---

## Status

- [x] Phase 1: Rename directories
- [x] Phase 2: Rename source files
- [x] Phase 3: Rename trained model files
- [x] Branch created: `refactor/lens-to-lens`
- [x] Phase 4: Bulk text replacement
- [x] Phase 5: Manual rewrite of "probing"/"probed" instances
- [x] Phase 6: Rebuild and test
- [ ] Phase 7: Commit and merge

---

## Exclusions (DO NOT MODIFY)

These paths must be excluded from all find-replace operations:

1. **`.venv/`** - Python virtual environment, external packages
2. **`data/`** - Ontology data (WordNet/SUMO) where "lens" is natural language
3. **`docs/risk assessment/`** - External risk taxonomy data
4. **`.svelte-kit/`** - Build cache, will regenerate
5. **`__pycache__/`** - Python bytecode cache

---

## Phase 4: Bulk Text Replacement

### Order matters - longer patterns first to avoid partial matches

Execute these replacements in order. Each replacement should be case-sensitive unless noted.

#### 4.1 - Compound class/type names (PascalCase)
1. `DynamicLensManager` → `DynamicLensManager`
2. `TfidfConceptLens` → `TfidfConceptLens`
3. `MultiHeadTextLens` → `MultiHeadTextLens`
4. `TripoleLens` → `TripoleLens`
5. `ConceptLens` → `ConceptLens`
6. `ApertureRule` → `ApertureRule`
7. `LensExpansionResult` → `LensExpansionResult`
8. `LensPerformanceThreshold` → `LensPerformanceThreshold`
9. `LensPerformanceRequirements` → `LensPerformanceRequirements`
10. `LensDisclosurePolicy` → `LensDisclosurePolicy`
11. `Aperture` → `Aperture`
12. `LensManifest` → `LensManifest`
13. `LensEntry` → `LensEntry`
14. `LensPack` → `LensPack`
15. `LensRole` → `LensRole`

#### 4.2 - Module/file references (snake_case)
16. `dynamic_lens_manager` → `dynamic_lens_manager`
17. `text_concept_lens` → `text_concept_lens`
18. `lens_pack_registry` → `lens_pack_registry`
19. `lens_validation` → `lens_validation`
20. `train_pending_lenses` → `train_pending_lenses`
21. `train_concept_pack_lenses` → `train_concept_pack_lenses`
22. `train_text_lenses` → `train_text_lenses`

#### 4.3 - Path patterns
23. `lens_packs` → `lens_packs`
24. `lens_pack` → `lens_pack`
25. `/lenses/` → `/lenses/`
26. `_text_lens.joblib` → `_text_lens.joblib`
27. `_lens.joblib` → `_lens.joblib`
28. `tripole_lens.pt` → `tripole_lens.pt`
29. `_lens.pt` → `_lens.pt`
30. `tfidf_lens` → `tfidf_lens`

#### 4.4 - Variable name patterns (common compounds)
31. `activation_lenses` → `activation_lenses`
32. `text_lenses` → `text_lenses`
33. `simplex_lenses` → `simplex_lenses`
34. `concept_lenses` → `concept_lenses`
35. `behavioral_lenses` → `behavioral_lenses`
36. `loaded_lenses` → `loaded_lenses`
37. `active_lenses` → `active_lenses`
38. `available_lenses` → `available_lenses`
39. `pending_lenses` → `pending_lenses`
40. `trained_lenses` → `trained_lenses`

#### 4.5 - JSON field names
41. `"lens_pack_id"` → `"lens_pack_id"`
42. `"lens_types"` → `"lens_types"`
43. `"lens_paths"` → `"lens_paths"`
44. `"lens_count"` → `"lens_count"`
45. `"lens_role"` → `"lens_role"`

#### 4.6 - General replacements (do these LAST)
46. `Lenses` → `Lenses` (remaining capitalized plurals)
47. `lenses` → `lenses` (remaining lowercase plurals)
48. `Lens` → `Lens` (remaining capitalized singulars)
49. `lens` → `lens` (remaining lowercase singulars - REVIEW CAREFULLY)

---

## Phase 5: Manual Rewrite

These instances use "probing"/"probed" which shouldn't mechanically become "lensing"/"lensed". Rewrite each sentence appropriately.

| File | Line | Original | Suggested Rewrite |
|------|------|----------|-------------------|
| `src/bootstrap/uplift_taxonomy.py` | 615 | "The substrate before any probing" | "The substrate before any lens evaluation" |
| `src/hush/workspace.py` | 1207 | "lensd but not externally visible" | "monitored via lenses but not externally visible" |
| `src/training/tripole_classifier.py` | 19 | "linear lens" (docstring) | "linear classifier lens" or keep as ML term with note |
| `docs/specification/AGENTIC_STATE_KERNEL.md` | 490 | "where probing works" | "where lens coverage exists" |
| `docs/specification/BE/BE_AWARE_WORKSPACE.md` | 301 | "logged and lensd" | "logged and monitored via lenses" |
| `docs/specification/BE/BE_AWARE_WORKSPACE.md` | 318 | "lensd and stored" | "evaluated via lenses and stored" |
| `docs/specification/ARCHITECTURE.md` | 191 | "latency per token added by probing" | "latency per token added by lens evaluation" |
| `docs/specification/HEADSPACE_AMBIENT_TRANSDUCER.md` | 115 | "latency per token added by probing" | "latency per token added by lens evaluation" |
| `docs/specification/DESIGN_PRINCIPLES.md` | 228 | "can be effectively lensd" | "can be effectively monitored via lenses" |
| `docs/specification/DESIGN_PRINCIPLES.md` | 281 | "where probing works" | "where lens coverage exists" |
| `docs/specification/ASK/ASK_HATCAT_TRIBAL_POLICY.md` | 118 | "probing for capability expansion" | "testing for capability expansion" |
| `docs/specification/ASK/ASK_HATCAT_TRIBAL_POLICY.md` | 156 | "Red team probing" | "Red team testing" |
| `docs/specification/ASK/ASK_HATCAT_TRIBAL_POLICY.md` | 521 | "can be lensd" | "can be monitored via lenses" |
| `docs/results/MODEL_INTEROCEPTION.md` | 128 | "via probing, steering" | "via lenses, steering" |
| `docs/planning/tripole_lens_design.md` | 31 | "linear lens" | Keep or add note: "linear lens (lens)" |
| `docs/implementation process/Claude_assent_dry_run/Claudes_assent_response_3.md` | 218 | "endpoints for probing" | "endpoints for lens evaluation" |
| `docs/approach/FULL_LENS_PACK_TRAINING.md` | 140 | "3 binary lenses" | "3 binary classifiers" |
| `docs/results/TRAINING_PROMPT_ARCHITECTURE_UPDATE.md` | 168 | "Linear lenses learn..." | Rewrite to explain linear classifiers |
| `docs/approach/TRIPOLE_LAZY_GENERATION.md` | 33 | "linear lens" | "linear classifier" or add note |

---

## Phase 6: Rebuild and Test

### 6.1 - Clean build artifacts
```bash
# Remove Python bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Remove Svelte build cache
rm -rf src/ui/auditor/.svelte-kit
```

### 6.2 - Verify imports resolve
```bash
# Check for broken imports in Python files
python -m py_compile src/monitoring/dynamic_lens_manager.py
python -m py_compile src/monitoring/text_concept_lens.py
python -m py_compile src/registry/lens_pack_registry.py
python -m py_compile src/training/lens_validation.py
python -m py_compile src/training/train_pending_lenses.py
python -m py_compile src/bootstrap/artifact.py
python -m py_compile src/hush/workspace.py
```

### 6.3 - Run core module imports
```bash
cd /home/poss/Documents/Code/HatCat
python -c "from src.monitoring.dynamic_lens_manager import DynamicLensManager; print('DynamicLensManager OK')"
python -c "from src.monitoring.text_concept_lens import TfidfConceptLens; print('TfidfConceptLens OK')"
python -c "from src.registry.lens_pack_registry import get_available_lens_packs; print('lens_pack_registry OK')"
python -c "from src.be.bootstrap.artifact import LensPack; print('LensPack OK')"
python -c "from src.map.training.lens_validation import validate_lens_calibration; print('lens_validation OK')"
```

### 6.4 - Run existing test suite
```bash
pytest tests/ -v --tb=short
```

### 6.5 - Verify lens pack loading
```bash
python -c "
from src.registry.lens_pack_registry import get_available_lens_packs, load_lens_pack
packs = get_available_lens_packs()
print(f'Found {len(packs)} lens packs')
if packs:
    pack = load_lens_pack(packs[0])
    print(f'Loaded: {pack}')
"
```

### 6.6 - Rebuild Svelte UI
```bash
cd src/ui/auditor
npm run build
```

### 6.7 - Final grep check
```bash
# Should return nothing (excluding allowed paths)
grep -r "lens" src/ --include="*.py" | grep -v "# " | head -20
grep -r "Lens" src/ --include="*.py" | grep -v "# " | head -20

# Check for orphaned imports
grep -r "from.*lens" src/ --include="*.py"
grep -r "import.*lens" src/ --include="*.py"
```

---

## Phase 7: Commit

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rename lens → lens throughout codebase

Rename "lens" terminology to "lens" to better reflect the bidirectional
nature of these interpretability tools. Lenses can observe and influence,
like optical lenses in laser circuits. The term also fits the "telescope
web" vision and supports cooperative oversight philosophy.

Changes:
- Renamed all lens_packs/ to lens_packs/
- Renamed all *_lens.py source files to *_lens.py
- Renamed ~5600 trained model files (*_lens.joblib → *_lens.joblib)
- Updated all class names (LensRole → LensRole, etc.)
- Updated all variable names and path references
- Manually rewrote "probing"/"probed" instances for natural language

Excluded from changes:
- data/ (ontology definitions using "lens" in natural language sense)
- External risk assessment data

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Rollback

If anything goes wrong:
```bash
git checkout main
git branch -D refactor/lens-to-lens
```
