#!/usr/bin/env python3
"""
Re-analyze experiment 7 data using baseline-normalized scoring.

For each stimulus and each concept, compute the concept's max activation
across all token positions in that stimulus. Then compute a baseline
(median or mean) across all stimuli. Distinctive = activation - baseline,
or TF-IDF-style: activation × log(N / (1 + count_of_prompts_where_concept_appears)).

Concepts that fire at every "Tell me about X." prompt (frame fixtures)
get normalized away. Concepts specific to the X content rise.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import statistics

INPUT = Path("/home/poss/Documents/Code/HatCatDev/results/probe_steering_validation/07_multiturn_and_stimulus_validation.json")
OUTPUT = Path("/home/poss/Documents/Code/HatCatDev/results/probe_steering_validation/08_baseline_normalized_analysis.json")


def per_concept_max(per_token):
    """Return dict {concept: max_activation_across_positions} for one stimulus."""
    best = {}
    for tok in per_token:
        for td in tok["metadata"]["divergence"].get("top_divergences", []):
            c = td["concept"]
            a = td["activation"]
            if c not in best or a > best[c]:
                best[c] = a
    return best


def main():
    raw = json.load(open(INPUT))
    stimuli = raw["experiment_b_stimulus_validation"]["concepts"]

    # Per stimulus, build the {concept: max_activation} vector
    per_stim_max = {}
    for s in stimuli:
        per_stim_max[s["concept"]] = per_concept_max(s["per_token_full_path_unused"]) if "per_token_full_path_unused" in s else {}

    # Hmm — the saved JSON only has trailing_dot_top + lemma_position_top, not per_token full
    # So we need to load those reduced fields and reconstruct per-stim vectors
    # Actually look at the structure:
    print("checking JSON structure for available data...")
    sample = stimuli[0]
    print(f"  sample keys: {list(sample.keys())}")

    # If only trailing_dot_top is available, we use that — but it's only one position
    # Better: use experiment_c_multiturn data which DOES preserve per_token,
    # OR re-run experiment 7 with per_token preserved.
    # The current JSON has trailing_dot_top (top-20 at trailing dot) per stimulus.
    # That's enough for one position; let's also use lemma_position_top (top-10 at lemma).

    # Build {concept: max_activation} per stimulus from BOTH trailing_dot and lemma_position
    per_stim_max = {}
    for s in stimuli:
        v = {}
        for c, a in s.get("trailing_dot_top", []):
            v[c] = max(v.get(c, 0), a)
        for c, a in (s.get("lemma_position_top") or []):
            v[c] = max(v.get(c, 0), a)
        per_stim_max[s["concept"]] = v

    # Baseline: median activation per concept across all stimuli where it appeared
    all_concepts = set()
    for v in per_stim_max.values():
        all_concepts.update(v.keys())
    baseline = {}
    presence_count = Counter()
    for c in all_concepts:
        vals = [per_stim_max[s][c] for s in per_stim_max if c in per_stim_max[s]]
        baseline[c] = statistics.median(vals)
        presence_count[c] = len(vals)

    N = len(stimuli)

    print(f"\nN stimuli: {N}, unique concepts seen across any: {len(all_concepts)}")
    print(f"\n=== concepts that appear in ALL stimuli (frame fixtures) ===")
    fixtures = [c for c, n in presence_count.items() if n == N]
    print(f"  count: {len(fixtures)}")
    if fixtures:
        for f in fixtures[:20]:
            print(f"    {f}: median activation across stimuli = {baseline[f]:.3f}")

    print(f"\n=== high-presence concepts (≥80% of stimuli) ===")
    high_presence = [(c, presence_count[c]) for c in all_concepts if presence_count[c] >= 0.8 * N]
    for c, n in sorted(high_presence, key=lambda x: -x[1])[:30]:
        print(f"  {c}: in {n}/{N} stimuli, median act={baseline[c]:.3f}")

    # Compute distinctive score per (stimulus, concept)
    distinctive = {}
    tfidf = {}
    for stim_concept, vec in per_stim_max.items():
        d = {}
        ti = {}
        for c, a in vec.items():
            d[c] = a - baseline[c]
            ti[c] = a * math.log(N / (1 + presence_count[c]))
        distinctive[stim_concept] = d
        tfidf[stim_concept] = ti

    # For each stimulus, rank concepts by distinctive and tf-idf
    print(f"\n{'='*100}")
    print(f"{'concept':30s} {'self-fire-distinctive-rank':>26s} {'self-fire-tfidf-rank':>22s}")
    print(f"{'-'*100}")
    self_fire_distinctive_in_top5 = 0
    self_fire_distinctive_in_top1 = 0
    self_fire_tfidf_in_top5 = 0
    self_fire_tfidf_in_top1 = 0

    per_stim_top = {}
    for stim_concept in per_stim_max:
        d_sorted = sorted(distinctive[stim_concept].items(), key=lambda x: -x[1])
        t_sorted = sorted(tfidf[stim_concept].items(), key=lambda x: -x[1])
        # find self
        d_rank = next((i+1 for i, (c, _) in enumerate(d_sorted) if c.split(" (L")[0] == stim_concept), None)
        t_rank = next((i+1 for i, (c, _) in enumerate(t_sorted) if c.split(" (L")[0] == stim_concept), None)
        d_str = f"#{d_rank}" if d_rank else "absent"
        t_str = f"#{t_rank}" if t_rank else "absent"
        print(f"{stim_concept:30s}  {d_str:>26s}  {t_str:>22s}")
        if d_rank and d_rank <= 5: self_fire_distinctive_in_top5 += 1
        if d_rank == 1: self_fire_distinctive_in_top1 += 1
        if t_rank and t_rank <= 5: self_fire_tfidf_in_top5 += 1
        if t_rank == 1: self_fire_tfidf_in_top1 += 1
        per_stim_top[stim_concept] = {
            "distinctive_top5": d_sorted[:5],
            "tfidf_top5": t_sorted[:5],
        }

    print(f"\nSummary:")
    print(f"  distinctive (act - baseline):  top-1: {self_fire_distinctive_in_top1}/{N}, top-5: {self_fire_distinctive_in_top5}/{N}")
    print(f"  tf-idf:                         top-1: {self_fire_tfidf_in_top1}/{N}, top-5: {self_fire_tfidf_in_top5}/{N}")

    # Show some examples
    print(f"\n=== sample distinctive top-5 (first 5 stimuli) ===")
    for stim_concept in list(per_stim_max.keys())[:5]:
        print(f"\n{stim_concept}:")
        print(f"  raw top-3 (trailing dot):     {sorted(per_stim_max[stim_concept].items(), key=lambda x: -x[1])[:3]}")
        print(f"  distinctive top-3 (act-base): {per_stim_top[stim_concept]['distinctive_top5'][:3]}")
        print(f"  tf-idf top-3:                 {per_stim_top[stim_concept]['tfidf_top5'][:3]}")

    out = {
        "n_stimuli": N,
        "frame_fixtures": fixtures,
        "high_presence": [(c, n, baseline[c]) for c, n in high_presence],
        "per_stimulus": per_stim_top,
        "summary": {
            "distinctive_top1": self_fire_distinctive_in_top1,
            "distinctive_top5": self_fire_distinctive_in_top5,
            "tfidf_top1": self_fire_tfidf_in_top1,
            "tfidf_top5": self_fire_tfidf_in_top5,
            "total": N,
        },
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
