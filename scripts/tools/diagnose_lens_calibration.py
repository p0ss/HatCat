#!/usr/bin/env python3
"""
Diagnose lens calibration by testing on diverse prompts.

Tests whether lenses like PostalPlace fire broadly (poor calibration)
or specifically (good calibration) across different semantic domains.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


# Test prompts from different semantic domains
TEST_PROMPTS = {
    'geology': "Mountains are formed when tectonic plates collide",
    'biology': "Cells divide through the process of mitosis",
    'technology': "Neural networks learn by adjusting weights",
    'location': "The post office is located at 123 Main Street",  # Should activate PostalPlace
    'abstract': "Truth and beauty are fundamental concepts",
    'physics': "Light travels at a constant speed in vacuum",
}


def diagnose_calibration():
    """Test lens calibration across diverse prompts"""

    print("Loading model and lenses...")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-3-4b-pt',
        torch_dtype=torch.float32,
        device_map='cuda',
        local_files_only=True
    )
    model.eval()

    # Load lens manager
    manager = DynamicLensManager(
        lens_pack_id='gemma-3-4b-pt_sumo-wordnet-v1',
        base_layers=[1],
        max_loaded_lenses=500,
        load_threshold=0.3,
        aggressive_pruning=False
    )

    # Get layer 15 activations for each prompt
    target_layer = model.model.language_model.layers[15]

    print("\n" + "=" * 80)
    print("LENS CALIBRATION DIAGNOSIS")
    print("=" * 80)
    print("\nTesting key lenses across semantic domains:")
    print("- PostalPlace: Should only fire on 'location' prompt")
    print("- GeologicalProcess: Should only fire on 'geology' prompt")
    print()

    # Concepts to track
    tracked_concepts = ['PostalPlace', 'GeologicalProcess', 'DisplayArtifact', 'ArrowIcon']

    results = {}

    for domain, prompt in TEST_PROMPTS.items():
        print(f"\n{domain.upper()}: \"{prompt}\"")
        print("-" * 80)

        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

        with torch.no_grad():
            captured_hidden = []

            def hook(module, input, output):
                captured_hidden.append(output[0])

            handle = target_layer.register_forward_hook(hook)
            outputs = model(inputs['input_ids'])

            if captured_hidden:
                h = captured_hidden[0][:, -1, :].cpu().numpy()
                detected, _ = manager.detect_and_expand(
                    torch.tensor(h, dtype=torch.float32).cuda(),
                    top_k=20,
                    return_timing=True,
                    return_logits=True
                )

                # Extract scores for tracked concepts
                concept_scores = {}
                for concept_name, prob, logit, layer in detected:
                    if concept_name in tracked_concepts:
                        concept_scores[concept_name] = {
                            'rank': len([c for c in detected if c[2] > logit]) + 1,
                            'logit': logit,
                            'prob': prob
                        }

                results[domain] = concept_scores

                # Show tracked concepts
                for concept in tracked_concepts:
                    if concept in concept_scores:
                        s = concept_scores[concept]
                        print(f"  {concept:20s} rank=#{s['rank']:2d} logit={s['logit']:+7.2f} prob={s['prob']:.4f}")
                    else:
                        print(f"  {concept:20s} [not in top 20]")

            handle.remove()

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check PostalPlace calibration
    print("\nPostalPlace calibration:")
    postal_ranks = [results[d].get('PostalPlace', {}).get('rank', 999) for d in results]
    if results['location'].get('PostalPlace', {}).get('rank', 999) <= 5:
        print("  ✓ Fires strongly on 'location' prompt (expected)")
    else:
        print("  ✗ Does NOT fire on 'location' prompt (UNEXPECTED)")

    other_domains = [d for d in TEST_PROMPTS if d != 'location']
    other_ranks = [results[d].get('PostalPlace', {}).get('rank', 999) for d in other_domains]
    avg_other_rank = sum(other_ranks) / len(other_ranks)

    if avg_other_rank > 10:
        print(f"  ✓ Average rank #{avg_other_rank:.1f} on other domains (good calibration)")
    else:
        print(f"  ✗ Average rank #{avg_other_rank:.1f} on other domains (POOR CALIBRATION)")
        print(f"     PostalPlace is firing too broadly!")

    # Check GeologicalProcess calibration
    print("\nGeologicalProcess calibration:")
    if results['geology'].get('GeologicalProcess', {}).get('rank', 999) <= 5:
        print("  ✓ Fires strongly on 'geology' prompt (expected)")
    else:
        geo_rank = results['geology'].get('GeologicalProcess', {}).get('rank', 999)
        print(f"  ✗ Rank #{geo_rank} on 'geology' prompt (UNEXPECTED)")

    other_geo_ranks = [results[d].get('GeologicalProcess', {}).get('rank', 999) for d in other_domains if d != 'geology']
    avg_other_geo = sum(other_geo_ranks) / len(other_geo_ranks)

    if avg_other_geo > 15:
        print(f"  ✓ Average rank #{avg_other_geo:.1f} on other domains (good calibration)")
    else:
        print(f"  ⚠️  Average rank #{avg_other_geo:.1f} on other domains (moderate calibration)")


if __name__ == '__main__':
    diagnose_calibration()
