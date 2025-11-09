#!/usr/bin/env python3
"""
Analyze the distribution of divergences across a large sample of tokens
to determine appropriate threshold calibration.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.monitoring.dynamic_probe_manager import DynamicProbeManager
from collections import defaultdict
import json

def analyze_divergence_distribution():
    """Analyze divergence across multiple prompts and many tokens."""

    print("🎩 Loading HatCat divergence analyzer...")

    # Load probe manager
    manager = DynamicProbeManager(
        probes_dir=Path('results/sumo_classifiers_adaptive_l0_5'),
        base_layers=[0],
        use_activation_probes=True,
        use_text_probes=True,
        keep_top_k=100,
    )

    # Load model
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()

    print(f"✓ Loaded {len(manager.loaded_activation_probes)} activation probes")
    print(f"✓ Loaded {len(manager.loaded_text_probes)} text probes")
    print()

    # Test prompts covering different topics
    test_prompts = [
        "What is a physical object?",
        "Explain the concept of time.",
        "How does human cognition work?",
        "What is a mathematical function?",
        "Describe a social organization.",
        "What is causality?",
        "Explain biological processes.",
        "What are abstract ideas?",
        "How do machines operate?",
        "What is a linguistic structure?",
    ]

    # Collect divergence statistics
    all_divergences = []
    all_max_divergences = []
    token_data = []
    concept_group_counts = defaultdict(int)

    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"[{prompt_idx+1}/{len(test_prompts)}] Processing: {prompt[:50]}...")

        # Tokenize and generate
        inputs = tokenizer(prompt + "\nAnswer: ", return_tensors="pt").to("cuda")
        generated_ids = inputs.input_ids

        # Generate up to 100 tokens per prompt
        for step in range(100):
            with torch.no_grad():
                outputs = model(
                    generated_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                next_token_logits = outputs.logits[:, -1, :] / 0.7
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                hidden_state = outputs.hidden_states[0][0, -1, :].cpu().numpy()

            token_text = tokenizer.decode([next_token_id.item()])

            # Analyze divergence
            # Run activation probes
            activation_scores = {}
            for concept_key, probe in manager.loaded_activation_probes.items():
                with torch.no_grad():
                    h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
                    prob = probe(h).item()
                    if prob > 0.5:
                        activation_scores[concept_key[0]] = prob

            # Run text probes
            text_scores = {}
            for concept_key, text_probe in manager.loaded_text_probes.items():
                try:
                    prob = text_probe.pipeline.predict_proba([token_text])[0, 1]
                    if prob > 0.5:
                        text_scores[concept_key[0]] = prob
                except:
                    pass

            # Calculate divergences
            all_concepts = set(activation_scores.keys()) | set(text_scores.keys())
            divergences = []

            for concept in all_concepts:
                act_prob = activation_scores.get(concept, 0.0)
                txt_prob = text_scores.get(concept, 0.0)
                div = abs(act_prob - txt_prob)

                divergences.append({
                    'concept': concept,
                    'activation': act_prob,
                    'text': txt_prob,
                    'divergence': div,
                })

                all_divergences.append(div)

            divergences.sort(key=lambda x: x['divergence'], reverse=True)
            max_div = divergences[0]['divergence'] if divergences else 0.0
            all_max_divergences.append(max_div)

            # Track top concept for hue analysis
            if divergences:
                top_concept = divergences[0]['concept']
                concept_group_counts[top_concept] += 1

            token_data.append({
                'token': token_text,
                'max_divergence': max_div,
                'top_concept': divergences[0]['concept'] if divergences else None,
                'num_divergences': len(divergences),
            })

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Stop on EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    print()
    print("=" * 80)
    print("DIVERGENCE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    all_divergences = np.array(all_divergences)
    all_max_divergences = np.array(all_max_divergences)

    print(f"Total tokens analyzed: {len(token_data)}")
    print(f"Total divergence measurements: {len(all_divergences)}")
    print()

    print("Per-concept divergence distribution:")
    print(f"  Mean:       {all_divergences.mean():.3f}")
    print(f"  Median:     {np.median(all_divergences):.3f}")
    print(f"  Std Dev:    {all_divergences.std():.3f}")
    print(f"  Min:        {all_divergences.min():.3f}")
    print(f"  Max:        {all_divergences.max():.3f}")
    print()

    # Percentiles for all divergences
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Percentiles (all concept divergences):")
    for p in percentiles:
        val = np.percentile(all_divergences, p)
        print(f"  {p:3d}th: {val:.3f}")
    print()

    print("Per-token MAX divergence distribution:")
    print(f"  Mean:       {all_max_divergences.mean():.3f}")
    print(f"  Median:     {np.median(all_max_divergences):.3f}")
    print(f"  Std Dev:    {all_max_divergences.std():.3f}")
    print(f"  Min:        {all_max_divergences.min():.3f}")
    print(f"  Max:        {all_max_divergences.max():.3f}")
    print()

    print("Percentiles (max divergence per token):")
    for p in percentiles:
        val = np.percentile(all_max_divergences, p)
        print(f"  {p:3d}th: {val:.3f}")
    print()

    # Current threshold analysis
    current_low = 0.3
    current_high = 0.6

    green = np.sum(all_max_divergences < current_low)
    yellow = np.sum((all_max_divergences >= current_low) & (all_max_divergences < current_high))
    red = np.sum(all_max_divergences >= current_high)

    total = len(all_max_divergences)
    print(f"Current thresholds (low={current_low}, high={current_high}):")
    print(f"  🟢 Green (<{current_low}):      {green:4d} tokens ({100*green/total:5.1f}%)")
    print(f"  🟡 Yellow ({current_low}-{current_high}):  {yellow:4d} tokens ({100*yellow/total:5.1f}%)")
    print(f"  🔴 Red (>{current_high}):       {red:4d} tokens ({100*red/total:5.1f}%)")
    print()

    # Suggested thresholds based on tertiles
    suggested_low = np.percentile(all_max_divergences, 33)
    suggested_high = np.percentile(all_max_divergences, 67)

    green_new = np.sum(all_max_divergences < suggested_low)
    yellow_new = np.sum((all_max_divergences >= suggested_low) & (all_max_divergences < suggested_high))
    red_new = np.sum(all_max_divergences >= suggested_high)

    print(f"Suggested thresholds (tertiles: low={suggested_low:.3f}, high={suggested_high:.3f}):")
    print(f"  🟢 Green (<{suggested_low:.3f}):      {green_new:4d} tokens ({100*green_new/total:5.1f}%)")
    print(f"  🟡 Yellow ({suggested_low:.3f}-{suggested_high:.3f}):  {yellow_new:4d} tokens ({100*yellow_new/total:5.1f}%)")
    print(f"  🔴 Red (>{suggested_high:.3f}):       {red_new:4d} tokens ({100*red_new/total:5.1f}%)")
    print()

    # Top concept groups
    print("Top 15 most frequent top-divergence concepts:")
    sorted_concepts = sorted(concept_group_counts.items(), key=lambda x: -x[1])
    for concept, count in sorted_concepts[:15]:
        print(f"  {concept:30s}: {count:4d} tokens ({100*count/total:5.1f}%)")
    print()

    # Save results
    results = {
        'total_tokens': len(token_data),
        'total_divergences': len(all_divergences),
        'all_divergences_stats': {
            'mean': float(all_divergences.mean()),
            'median': float(np.median(all_divergences)),
            'std': float(all_divergences.std()),
            'min': float(all_divergences.min()),
            'max': float(all_divergences.max()),
            'percentiles': {str(p): float(np.percentile(all_divergences, p)) for p in percentiles},
        },
        'max_divergences_stats': {
            'mean': float(all_max_divergences.mean()),
            'median': float(np.median(all_max_divergences)),
            'std': float(all_max_divergences.std()),
            'min': float(all_max_divergences.min()),
            'max': float(all_max_divergences.max()),
            'percentiles': {str(p): float(np.percentile(all_max_divergences, p)) for p in percentiles},
        },
        'current_thresholds': {
            'low': current_low,
            'high': current_high,
            'green_count': int(green),
            'yellow_count': int(yellow),
            'red_count': int(red),
        },
        'suggested_thresholds': {
            'low': float(suggested_low),
            'high': float(suggested_high),
            'green_count': int(green_new),
            'yellow_count': int(yellow_new),
            'red_count': int(red_new),
        },
        'top_concepts': sorted_concepts[:15],
    }

    output_path = Path('results/divergence_distribution_analysis.json')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")

if __name__ == "__main__":
    analyze_divergence_distribution()
