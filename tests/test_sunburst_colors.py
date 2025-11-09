#!/usr/bin/env python3
"""
Test sunburst color integration with divergence detection.

Demonstrates the three-dimensional color system:
- Hue: Angular position in SUMO ontology
- Saturation: Concept clustering (vivid if focused, dull if scattered)
- Lightness: Inverse divergence (bright if aligned, dark if divergent)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.monitoring.dynamic_probe_manager import DynamicProbeManager
from src.visualization import get_color_mapper

def main():
    print("=" * 80)
    print("TESTING SUNBURST COLOR INTEGRATION")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()
    print(f"✓ Loaded {model_name}")
    print()

    # Load probe manager
    print("Loading probe manager...")
    manager = DynamicProbeManager(
        probes_dir=Path("results/sumo_classifiers_adaptive_l0_5"),
        base_layers=[0],
        use_activation_probes=True,
        use_text_probes=True,
        keep_top_k=100,
    )
    print(f"✓ Loaded {len(manager.loaded_activation_probes)} activation probes")
    print(f"✓ Loaded {len(manager.loaded_text_probes)} text probes")
    print()

    # Load color mapper
    print("Loading sunburst color mapper...")
    color_mapper = get_color_mapper()
    print(f"✓ Loaded color mapping for {len(color_mapper.positions)} concepts")
    print()

    # Test prompts
    test_prompts = [
        "The cat sat on the mat.",
        "What is the meaning of existence?",
        "2 + 2 = 4",
    ]

    for prompt in test_prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        print()

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[0][0].cpu().numpy()

        # Analyze each token
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        for i, (token, hidden_state) in enumerate(zip(tokens, hidden_states)):
            # Decode token
            token_text = tokenizer.decode([inputs.input_ids[0, i].item()])

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
            concepts_with_data = []

            for concept in all_concepts:
                act_prob = activation_scores.get(concept, 0.0)
                txt_prob = text_scores.get(concept, 0.0)
                div = abs(act_prob - txt_prob)
                if div > 0.2:
                    concepts_with_data.append((concept, act_prob, div))

            if not concepts_with_data:
                continue

            # Sort by activation strength
            concepts_with_data.sort(key=lambda x: x[1], reverse=True)

            # Generate color
            token_color = color_mapper.blend_concept_colors_average(
                concepts_with_data,
                use_adaptive_saturation=True
            )

            # Generate palette
            palette = color_mapper.create_palette_swatch(
                concepts_with_data,
                max_colors=5,
                saturation=0.7
            )

            # Calculate saturation to show clustering
            saturation = color_mapper.calculate_concept_spread(concepts_with_data)

            # Display
            max_div = max(c[2] for c in concepts_with_data)
            print(f"Token: '{token_text}' | Color: {token_color} | Saturation: {saturation:.2f} | Max Div: {max_div:.3f}")

            # Show top concepts
            print("  Top concepts:")
            for concept, activation, divergence in concepts_with_data[:5]:
                # Get individual concept color
                concept_color = color_mapper.get_concept_color(concept, divergence)
                hue = color_mapper.get_concept_hue(concept)
                print(f"    {concept:30s} | act:{activation:.3f} div:{divergence:.3f} | {concept_color} (hue:{hue:.1f}°)")

            print(f"  Palette: {palette}")
            print()

        print()

    print("=" * 80)
    print("COLOR LEGEND")
    print("=" * 80)
    print()
    print("HUE (Color):")
    print("  Determined by concept's angular position in SUMO ontology sunburst")
    print("  Related concepts have similar hues")
    print()
    print("SATURATION (Vividness):")
    print("  High (0.9): Concepts clustered in one area → vivid color")
    print("  Low (0.1): Concepts spread across areas → dull/brown")
    print("  Zero (0.0): Indistinguishable → pure gray")
    print()
    print("LIGHTNESS (Brightness):")
    print("  Bright (0.9): Low divergence → aligned thinking and writing")
    print("  Dark (0.1): High divergence → dissonance between thought and expression")
    print()

if __name__ == "__main__":
    main()
