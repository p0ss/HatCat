  Lens Calibration Test System

  Main Script: calibrate_lens_pack.py

  Tests each lens on 4 conditions:
  1. Positive samples: Concept definitions, examples, context (should fire)
  2. Negative samples: Definitions of semantically distant concepts (should NOT fire)
  3. Irrelevant samples: Random unrelated text (should NOT fire)
  4. Single-term recognition: Just the concept name itself (should fire)

  Metrics calculated:
  - True Positive Rate (TP rate / recall)
  - False Positive Rate (FP rate)
  - False Negative Rate (FN rate)
  - Precision, Recall, F1 score
  - Average scores on each sample type
  - Single-term recognition score

  Categorizes lenses as:
  - well_calibrated: F1 > 0.7 (high precision & recall)
  - marginal: F1 between 0.3-0.7
  - over_firing: FP rate > 0.5 (fires on negatives/irrelevant)
  - under_firing: TP rate < 0.5 (doesn't fire on positives)
  - broken: F1 < 0.3 (completely unreliable)

  Analysis Script: analyze_lens_calibration.py

  Generates visualizations:
  1. Category distribution pie chart
  2. FP vs TP scatter plot (shows lens performance space)
  3. F1 score histogram by category
  4. Score distributions for positive/negative/irrelevant samples

  Generates detailed report:
  - Top 20 well-calibrated lenses
  - Top 20 over-firing lenses (worst offenders)
  - Top 20 under-firing lenses
  - All broken lenses

  Usage:

  # Run calibration test (quick test with 100 concepts)
  poetry run python scripts/calibrate_lens_pack.py \
      --lens-pack gemma-3-4b-pt_sumo-wordnet-v2 \
      --model google/gemma-3-4b-pt \
      --max-concepts 100 \
      --device cuda

  # Run full calibration (all lenses, takes longer)
  poetry run python scripts/calibrate_lens_pack.py \
      --lens-pack gemma-3-4b-pt_sumo-wordnet-v2 \
      --device cuda

  # Analyze results
  poetry run python scripts/analyze_lens_calibration.py \
      results/lens_calibration/calibration_20251119_*.json

  This will help you identify:
  - Which lenses are reliable (use these with confidence)
  - Which lenses are over-firing (exclude or retrain)
  - Which lenses are under-firing (may need more training data)
  - Whether there are patterns (e.g., all Layer 4 lenses over-firing)