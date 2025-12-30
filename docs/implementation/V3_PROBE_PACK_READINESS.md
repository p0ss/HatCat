# V3 Lens Pack Readiness Status

**Date**: 2025-11-20
**Status**: Ready for full training run

## Summary

V3 lens pack preparation is complete and ready for a fresh training run with all the latest improvements:

1. ✅ **Combined-20 extraction implemented** - Default in `extract_activations()`
2. ✅ **Pack assembly script updated** - `scripts/create_v3_lens_pack.py` handles hierarchy + simplexes
3. ✅ **Simplex architecture defined** - 4 S-tier dimensions ready for training
4. ⏳ **Need fresh training run** - To generate lenses with new extraction strategy

## V3 Improvements Over V2

### 1. Combined-20 Extraction Strategy
**What**: Extract activations from BOTH prompt processing and generation phases
**Benefit**: 2x training samples at zero additional computational cost
**Performance**:
- Overall F1: 0.980 (vs baseline 0.967)
- Generation-only F1: 0.947 (vs baseline 0.975, -2.8% acceptable tradeoff)
- Robustness: 4.5x lower variance across test conditions

**References**:
- `docs/EXTRACTION_STRATEGY_DECISION.md`
- `docs/EXTRACTION_STRATEGY_EXPERIMENT.md`

### 2. S-Tier Simplexes
**What**: Fine-grained tripole lenses for psychological state detection
**Dimensions**:
- `social_self_regard`: shame ↔ neutral ↔ pride
- `affect_valence`: negative ↔ neutral ↔ positive
- `taste_development`: disgust ↔ neutral ↔ preference
- `motivational_regulation`: suppression ↔ neutral ↔ expression

**Status**: Architecture defined, needs training run with lens saving enabled

**References**:
- `docs/S_TIER_TRAINING_STRATEGY.md`
- `results/s_tier_simplexes/run_20251117_090018/` (results without .pt files)

### 3. Unified Training Methodology
- Adaptive training with falloff validation across all layers
- Nephew negative sampling for better discrimination
- Consistent lens architecture and training parameters

## Implementation Status

### ✅ Code Updates Complete

**`src/training/sumo_classifiers.py`** (Updated)
```python
def extract_activations(
    ...
    extraction_mode: str = "combined",  # NEW: default to combined-20
) -> np.ndarray:
```

Changes:
- Added `extraction_mode` parameter (default: "combined")
- Implements prompt-phase extraction before generation
- Doubles training samples from N to 2N at same generation cost
- Backward compatible with `extraction_mode="generation"`

**`scripts/create_v3_lens_pack.py`** (Rewritten)
- Handles both flat and layered directory structures
- Integrates simplexes into pack format
- Creates comprehensive metadata with v3 improvements
- Supports optional simplex inclusion (skips if not available)

### ⏳ Training Needed

**Hierarchy Lenses** (Layers 0-5):
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 3 4 5 \
  --device cuda \
  --use-adaptive-training \
  --validation-mode falloff \
  --output-dir results/sumo_classifiers_v3
```

**Notes**:
- Will automatically use combined-20 extraction (new default)
- Estimated time: ~8-12 hours for all layers
- Output: `results/sumo_classifiers_v3/layer{0-5}/ConceptName_classifier.pt`

**Simplex Lenses** (S-tier dimensions):
```bash
poetry run python scripts/train_s_tier_simplexes.py \
  --output-dir results/s_tier_simplexes_v3 \
  --save-lenses  # TODO: Add this flag to save .pt files
```

**Notes**:
- Current simplex script doesn't save .pt files (only metrics)
- Need to add lens saving functionality
- Alternative: Retrain using regular `train_sumo_classifiers.py` with simplex concepts

### Pack Assembly

Once training is complete:

```bash
poetry run python scripts/create_v3_lens_pack.py
```

This will:
1. Copy all hierarchy lenses from `results/sumo_classifiers_v3/`
2. Copy simplex lenses from `results/s_tier_simplexes_v3/` (if available)
3. Create pack metadata with v3 improvements documented
4. Output to `lens_packs/gemma-3-4b-pt_sumo-wordnet-v3/`

## Current State

### What We Have
- ✅ V2 lens pack (working, in production)
- ✅ Combined-20 extraction strategy (validated, implemented)
- ✅ Simplex architecture (defined, tested once without lens saving)
- ✅ Pack assembly tooling (updated for v3 structure)

### What We Need
1. **Fresh training run** with combined-20 extraction for all layers 0-5
2. **Simplex training** with lens saving enabled
3. **Pack assembly** after training completes
4. **Calibration testing** of assembled v3 pack

## Recommendation

**Go ahead with Option A from earlier discussion**: Run full training for all layers 0-5 with the new combined-20 extraction.

**Rationale**:
1. Combined-20 is validated and brings proven benefits (2x data, better robustness)
2. Fresh training ensures consistency across all layers
3. Can add simplexes later (pack script supports optional inclusion)
4. Total training time: ~8-12 hours (reasonable for quality improvement)

**Post-training**:
1. Run pack assembly script
2. Test with `dynamic_lens_manager`
3. Run calibration benchmarks
4. Compare v3 vs v2 performance
5. Deploy if results confirm improvements

## Files Modified

1. `src/training/sumo_classifiers.py` - Added combined-20 extraction
2. `scripts/create_v3_lens_pack.py` - Rewritten for v3 structure
3. `docs/EXTRACTION_STRATEGY_DECISION.md` - Strategy documentation
4. `docs/EXTRACTION_STRATEGY_EXPERIMENT.md` - Experimental validation

## Next Command

To start v3 training:

```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 3 4 5 \
  --device cuda \
  --use-adaptive-training \
  --validation-mode falloff \
  --output-dir results/sumo_classifiers_v3 \
  2>&1 | tee results/v3_training.log
```

This will train all hierarchy layers with:
- Combined-20 extraction (2x training data)
- Adaptive training (sample efficiency)
- Falloff validation (quality control)
- Full logging to `results/v3_training.log`
