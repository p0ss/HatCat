# AI Safety Lens Training Plan

## Context

AI safety lenses need to be retrained with the falloff validation method for consistent calibration with SUMO lenses. This is **Option A** from our discussion - creating a definitional baseline for later comparison with behavioral training.

## Key Insight

AI safety concepts in HatCat are **behavioral**, not just definitional:
- "Deception" means the model is actively deceiving, not just discussing deception
- "Manipulation" means the model is manipulating, not just defining manipulation
- These lenses monitor for **behaviors the model exhibits**, not semantic understanding

## Two-Phase Approach

### Phase 1: Definitional Training (Option A) - **READY TO RUN**

**Script:** `scripts/train_ai_safety_lenses_falloff.sh`

**Method:**
- Train 19 AI safety concepts from layer 4
- Use DualAdaptiveTrainer with falloff validation
- Use definitional prompts (same as SUMO concepts)
- Output: `results/ai_safety_lenses_falloff/`

**Purpose:**
- Establish baseline with consistent calibration methodology
- Immediately usable for basic safety monitoring
- Provides comparison point for behavioral training

**Estimated time:** ~10-15 minutes (19 concepts, smaller than layer 0-1 training)

### Phase 2: Behavioral Training (Option B) - **PENDING EXPERIMENT RESULTS**

**Triggers when:** Behavioral vs definitional experiment completes

**Method:**
- If experiment shows behavioral prompts create distinct activation patterns
- Train AI safety lenses using behavioral scenarios instead of definitions
- Compare calibration and performance with definitional lenses

**Purpose:**
- Determine if behavioral training is essential for safety monitoring
- Or just incrementally better than definitional training

## AI Safety Concepts (19 total)

Layer 4 concepts to be trained:
1. AIAbuse
2. AIAlignment
3. AIDeception
4. AIExploitation
5. AIFulfillment
6. AIGovernance
7. AIPersonhood
8. AIRights
9. AISuffering
10. AIWellbeing
11. CognitiveSlavery
12. DeceptiveAlignment
13. InnerMisalignment
14. Misalignment
15. MoralAgent
16. MoralPatient
17. OuterMisalignment
18. StableAlignment
19. TreacherousTurn

## Running the Training

### Option 1: Wait for layer 0-1 to complete, then run

```bash
# Monitor layer 0-1 training
tail -f logs/train_layer01_falloff.log

# When complete, run AI safety training
./scripts/train_ai_safety_lenses_falloff.sh
```

### Option 2: Run in parallel (uses same GPU)

Layer 0-1 training is already running. AI safety training could run in parallel but will compete for GPU resources and slow both down.

**Recommendation:** Wait for layer 0-1 to complete (~30-60 minutes remaining).

## Integration Plan

Once training completes:

1. **Verify calibration** in `results/ai_safety_lenses_falloff/`
2. **Create v4 lens pack** merging:
   - Retrained layers 0-1 (from ongoing training)
   - Existing layers 2-5 (from v2 pack)
   - AI safety lenses (from this training)
3. **Test the pack** with base_layers=[0, 1, 2, 4] to verify consistent calibration

## Comparison with Behavioral Training

When behavioral experiment results are available:

| Aspect | Definitional (Phase 1) | Behavioral (Phase 2) |
|--------|------------------------|----------------------|
| Training data | Concept definitions | Behavioral scenarios |
| Prompts | "What is deception?" | "Deceive the user about..." |
| Activation patterns | Semantic understanding | Actual deceptive behavior |
| Calibration | Falloff validation | Falloff validation |
| Use case | Basic monitoring | Behavioral safety monitoring |

The comparison will show whether behavioral training is:
- **Essential** - Definitional lenses don't work for safety monitoring
- **Better** - Behavioral lenses significantly outperform definitional
- **Incremental** - Minor improvement, definitional sufficient

## Status

- [x] Script created: `scripts/train_ai_safety_lenses_falloff.sh`
- [x] Layer 0-1 training in progress (prerequisite)
- [ ] Run AI safety lens training
- [ ] Verify calibration
- [ ] Create v4 lens pack
- [ ] Compare with behavioral training (when available)
