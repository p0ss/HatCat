# HatCat Project Structure

Quick reference for navigating the codebase and understanding documentation organization.

## Documentation Files

### Research & Experiments
- **`TEST_DATA_REGISTER.md`** - Complete experiment history with results, metrics, and findings
  - Use this to: Understand what's been tested, what worked/failed, see full metrics
  - Update when: Completing any experimental run or phase

- **`projectplan.md`** - Research roadmap, future directions, and strategic planning
  - Use this to: Understand project goals, plan next phases
  - Update when: Changing research direction or completing major milestones

- **`README.md`** - Project overview, quick start, architecture explanation
  - Use this to: Onboard new users, explain the project
  - Update when: Major architecture changes or new production-ready features

- **`QUICKSTART.md`** - Step-by-step setup and first run instructions
  - Use this to: Get the project running quickly
  - Update when: Setup process changes

### Development Reference
- **`.claude/SHARED_CODE_REFERENCE.md`** - Canonical coding patterns and best practices
  - Use this to: Find correct way to load models, implement hooks, handle GPU cleanup
  - Update when: A pattern appears in 3+ scripts and is validated as working

- **`.claude/TESTING_STANDARDS.md`** - Standardized test output requirements
  - Use this to: Structure experiment outputs (logs, JSON, summary, plots)
  - Follow always: Every experiment should produce these standard outputs

- **`.claude/PROJECT_STRUCTURE.md`** - This file

### Results Documentation
- **`results/phase_*/summary.md`** - Deep analysis and interpretation of specific experiments
  - Use these for: Understanding "why" behind results, detailed findings
  - Create when: Experiment has significant findings or needs detailed explanation

## Code Organization

### Core Implementation (`src/`)
```
src/
├── encyclopedia/
│   └── wordnet_graph_v2.py        # Concept graph builder (WordNet-based)
├── interpreter/
│   ├── model.py                   # Binary classifier architecture (BiLSTM + MLP)
│   └── steering.py                # Concept vector extraction from classifiers
├── steering/
│   ├── hooks.py                   # Forward hooks for steering (raw baseline - USE THIS)
│   └── manifold.py                # Manifold steering (BROKEN - see Phase 6.7)
└── utils/
    └── gpu_cleanup.py             # GPU memory management utilities
```

### Experiment Scripts (`scripts/`)
```
scripts/
├── train_binary_classifiers.py   # Train concept classifiers
├── phase_2_scale_test.py          # Scale test (1, 10, 100, 1000 concepts)
├── phase_2_5_steering_eval.py     # Steering effectiveness evaluation
├── phase_6_*.py                   # Contamination removal experiments
├── phase_6_5_*.py                 # Task manifold steering experiments
├── phase_6_6_*.py                 # Dual-subspace (contamination + manifold)
├── phase_6_7_*.py                 # Ablation study (COMPLETED - see results)
├── phase_7_*.py                   # Logarithmic scaling validation
└── test_*.py                      # Specific hypothesis tests
```

**Naming convention:**
- `phase_X_*.py` - Main experimental phase
- `phase_X_Y_*.py` - Sub-phase or variant
- `test_*.py` - Hypothesis validation or specific feature test

### Data (`data/`)
```
data/
└── concept_graph/
    ├── wordnet_v2_top10.json      # 10 concepts (quick tests)
    ├── wordnet_v2_top100.json     # 100 concepts
    ├── wordnet_v2_top1000.json    # 1K concepts
    └── wordnet_v2_top10000.json   # 10K concepts (production target)
```

### Results (`results/`)
```
results/
├── phase_*/                       # Experiment outputs (JSON, logs, plots)
│   ├── *_results.json            # Structured results
│   ├── summary.md                # Analysis (optional, for significant findings)
│   └── ANALYSIS.md               # Detailed analysis (legacy, optional)
└── classifiers/                   # Trained binary classifiers
```

### Logs (`logs/` and root `*.log`)
- Phase scripts create `.log` files in project root
- Legacy logs may be in `logs/` directory
- **Important**: Always use `2>&1 | tee output.log` pattern for capturing output

## Common Workflows

### Running an Experiment
1. Check `TEST_DATA_REGISTER.md` for similar experiments
2. Check `SHARED_CODE_REFERENCE.md` for correct patterns
3. Write script using canonical patterns
4. Run with output capture: `poetry run python script.py 2>&1 | tee run.log`
5. Document results in `TEST_DATA_REGISTER.md`

### Adding New Shared Code
1. Validate pattern works (appears in 3+ scripts)
2. Extract to appropriate `src/` module
3. Add tests (if critical functionality)
4. Document in `SHARED_CODE_REFERENCE.md`
5. Update scripts to use shared implementation

### Before Each Run
- Check GPU memory is clear (we've had repeated OOM issues)
- Use `src.utils.gpu_cleanup` in scripts
- Verify float32 for numerical stability (NOT float16)

## Key Learnings (See TEST_DATA_REGISTER.md for full details)

**What Works:**
- ✅ Raw baseline steering (84% effective, 99.6% diversity)
- ✅ Contamination-only steering (81% effective, 99.8% diversity)
- ✅ 1×1 minimal training (919/1000 concepts @ 100% accuracy)

**Current Implementation Issues:**
- ⚠️ Manifold projection steering (0-9% effective - Phase 6.7 - needs debugging)
- ⚠️ Dual-subspace steering (0% effective - Phase 6.7 - needs debugging)
- ❌ float16 precision (causes NaN/Inf in SVD operations - use float32)

**Open Questions:**
- RMSNorm sign symmetry (±strength produce identical outputs)
- Hook placement effects (after layer vs before MLP)
- Positive steering variability across concepts

## Quick Command Reference

```bash
# Train 10 concept classifiers (quick test)
poetry run python scripts/train_binary_classifiers.py \
    --concept-graph data/concept_graph/wordnet_v2_top10.json \
    --model google/gemma-3-4b-pt \
    --output-dir results/classifiers_10 \
    --n-definitions 1 --n-negatives 1

# Test steering (use raw baseline - Phase 6.7 validated)
poetry run python scripts/phase_2_5_steering_eval.py \
    --device cuda

# Run ablation study (Phase 6.7 - already complete)
poetry run python scripts/phase_6_7_full_ablation.py \
    --device cuda 2>&1 | tee ablation.log
```

## File Naming Conventions

- **Scripts**: `phase_X_description.py` or `test_description.py`
- **Results**: `results/phase_X_description/`
- **Logs**: `phase_X_description.log` (in project root)
- **Summaries**: `results/phase_X_description/summary.md`
- **Data outputs**: `*_results.json`, `*_metrics.json`

## When to Update Which File

| You just... | Update this file |
|-------------|------------------|
| Completed an experiment | `TEST_DATA_REGISTER.md` |
| Found a working code pattern (3+ uses) | `SHARED_CODE_REFERENCE.md` |
| Changed research direction | `projectplan.md` |
| Made architecture changes | `README.md` |
| Changed setup process | `QUICKSTART.md` |
| Need detailed analysis | `results/phase_*/summary.md` |
| Fixed a recurring bug | `SHARED_CODE_REFERENCE.md` (Known Issues) |

---

**Last Updated**: November 5, 2025
