# Testing Standards for HatCat

**All experimental scripts must follow these output standards for consistency and reproducibility.**

## Required Output Structure

Every experimental script should produce:

1. **Log file** (verbose runtime output)
2. **Results JSON** (structured data)
3. **Summary markdown** (human-readable analysis)
4. **Charts/plots** (visualizations, where appropriate)

## 1. Log Files

### Pattern
```bash
poetry run python scripts/phase_X_description.py 2>&1 | tee phase_X_description.log
```

### Purpose
- Capture full stdout/stderr for debugging
- Track runtime progress and errors
- Store in project root with descriptive name

### What to Log
- Configuration parameters
- Progress indicators
- Warnings and errors
- Timing information
- GPU memory usage

### Example
```python
print("="*60)
print("PHASE 6.7: STEERING ABLATION STUDY")
print("="*60)
print(f"Concepts: {len(CONCEPTS)}")
print(f"Strengths: {STRENGTHS}")
print(f"{'='*60}\n")

print("Loading model...")
# ... model loading ...
print("✓ Model loaded\n")

print(f"Baseline perplexity: {baseline_ppl:.2f}\n")
```

## 2. Results JSON

### Location
```
results/phase_X_description/results.json
```

### Structure
```json
{
  "metadata": {
    "experiment": "Phase X Description",
    "date": "2025-11-05",
    "model": "google/gemma-3-4b-pt",
    "device": "cuda",
    "runtime_seconds": 1234.5
  },
  "config": {
    "concepts": [...],
    "strengths": [...],
    "other_params": "..."
  },
  "results": {
    "per_concept_data": {...},
    "aggregate_metrics": {...}
  },
  "summary_statistics": {
    "mean_metric": 0.85,
    "working_concepts": "27/32"
  }
}
```

### Requirements
- **Machine-readable**: Valid JSON, no comments
- **Complete**: All raw data needed for re-analysis
- **Structured**: Consistent schema across experiments
- **Timestamped**: Include date and runtime
- **Reproducible**: Config section with all parameters

### Example
```python
output_data = {
    "metadata": {
        "experiment": "Phase 6.7 Ablation Study",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": args.model,
        "runtime_seconds": time.time() - start_time
    },
    "config": {
        "concepts": CONCEPTS,
        "strengths": STRENGTHS,
        "dampening_mults": DAMPENING_MULTS
    },
    "results": all_results,
    "summary": {
        "working": working_count,
        "mean_diversity": float(mean_diversity),
        "mean_rho": float(mean_rho)
    }
}

with open(output_dir / "results.json", "w") as f:
    json.dump(output_data, f, indent=2)
```

## 3. Summary Markdown

### Location
```
results/phase_X_description/summary.md
```

### Template
```markdown
# Phase X: [Name]

## Objective
[One paragraph]

## Configuration
- Model: [model name]
- Concepts: [count or list]
- Key parameters: [list]

## Results

### Key Findings
1. Finding one
2. Finding two
3. Finding three

### Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 84% | ... |
| Diversity | 99.6% | ... |

## Analysis
[Detailed interpretation]

## Conclusions
- **What works**: [summary]
- **What doesn't**: [summary]
- **Next steps**: [list]

## Files
- `results.json` - Raw data
- `plot_metric.png` - Visualization
- `phase_X.log` - Runtime log
```

### Requirements
- **Human-readable**: Clear prose, not code dumps
- **Interpretation**: Explain what the numbers mean
- **Context**: Why this matters for the project
- **Actionable**: Clear next steps
- **Concise**: 1-2 pages max unless deep analysis needed

### Example Creation
```python
summary = f"""# Phase 6.7: Steering Ablation Study

## Objective
Determine which components of dual-subspace manifold steering help vs hurt effectiveness.

## Key Findings
1. ✅ Raw baseline: 84% effective (27/32 concepts)
2. ⚠️ Manifold projection: 0-9% effective (needs debugging)

## Conclusions
- Use raw baseline for production
- Manifold implementation needs debugging
"""

with open(output_dir / "summary.md", "w") as f:
    f.write(summary)
```

## 4. Charts and Plots

### When to Create
- **Always**: For comparative metrics (ablation studies, scaling curves)
- **Often**: For distributions (accuracy, confidence, diversity)
- **Sometimes**: For relationships (correlations, trends)
- **Never**: When data is too sparse (<5 data points)

### Location
```
results/phase_X_description/plot_name.png
```

### Requirements
- **Format**: PNG (web-compatible) or SVG (vector)
- **Size**: 800-1200px wide for readability
- **Labels**: Clear axis labels, title, legend
- **Style**: Use matplotlib defaults or seaborn
- **Save**: High DPI (150-300) for clarity

### Example
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(variant_names, effectiveness_scores)
ax.set_xlabel("Steering Variant")
ax.set_ylabel("Effectiveness (%)")
ax.set_title("Phase 6.7: Steering Effectiveness by Variant")
plt.tight_layout()
plt.savefig(output_dir / "effectiveness_comparison.png", dpi=150)
plt.close()
```

## Script Template

```python
#!/usr/bin/env python3
"""
Phase X: Description

Goal: One-line objective
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ... imports ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_X_description")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Header
    print("="*60)
    print("PHASE X: DESCRIPTION")
    print("="*60)
    print(f"Config: ...")
    print("="*60 + "\n")

    # Load model with GPU cleanup
    from src.utils.gpu_cleanup import cleanup_model, print_gpu_memory

    print_gpu_memory()
    print("Loading model...")
    # ... model loading ...
    print("✓ Model loaded")
    print_gpu_memory()

    try:
        # Run experiment
        results = run_experiment(...)

        # Save results JSON
        output_data = {
            "metadata": {
                "experiment": "Phase X",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "model": args.model,
                "runtime_seconds": time.time() - start_time
            },
            "config": {...},
            "results": results
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(output_data, f, indent=2)

        # Create plots
        create_plots(results, output_dir)

        # Write summary
        write_summary(results, output_dir)

        print(f"\n✓ Saved to: {output_dir}/")
        print("="*60)

    finally:
        cleanup_model(model, tokenizer)
        print_gpu_memory()

if __name__ == "__main__":
    main()
```

## Checklist for New Experiments

Before considering an experiment complete:

- [ ] Log file created (stdout captured with `tee`)
- [ ] Results JSON created with all raw data
- [ ] Summary markdown written with interpretation
- [ ] Plots created if comparative data exists
- [ ] Entry added to `TEST_DATA_REGISTER.md`
- [ ] GPU cleanup implemented (no OOM for next run)
- [ ] Files are git-tracked (not .gitignored)

## Anti-Patterns to Avoid

❌ **Don't**: Print results only to stdout
```python
print(f"Accuracy: {acc}")  # Lost after terminal closes
```

✅ **Do**: Save to JSON and summarize
```python
results["accuracy"] = acc
print(f"Accuracy: {acc}")  # Also show progress
```

❌ **Don't**: Create massive JSON files with redundant data
```python
# 100MB file with full model states
```

✅ **Do**: Save aggregated metrics, reference source data
```python
# Save summary metrics, note that full data is in separate file
```

❌ **Don't**: Skip documentation because "it's obvious"
```python
# No summary.md, just JSON
```

✅ **Do**: Always write summary, even if brief
```python
# 1-page summary explaining what the numbers mean
```

## File Size Guidelines

- **Logs**: Keep them (even if verbose), essential for debugging
- **JSON**:
  - <10MB: Commit to git
  - 10-100MB: Consider if full data needed or can aggregate
  - >100MB: Don't commit, add to .gitignore, document location
- **Plots**: Always commit (usually <1MB each)
- **Summaries**: Always commit (text is tiny)

## Example: Good vs Bad Output

### ❌ Bad
```
results/
  phase_X/
    output.txt          # Unstructured text dump
    data.pkl            # Binary, can't read without Python
```

### ✅ Good
```
results/
  phase_X_description/
    results.json        # Complete structured data
    summary.md          # Human-readable analysis
    effectiveness.png   # Key findings visualization
    correlation.png     # Additional insights
```

---

**Remember**: Future you (or future researchers) will thank you for clear, complete documentation!
