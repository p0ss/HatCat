#!/usr/bin/env python3
"""
Generate sample output showing 1:1 token monitoring with line-by-line sparklines.

This demonstrates what the "Detailed Inspection Mode" would look like in practice.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager


class TokenTimelineRecorder:
    """Records concept activations per token for visualization"""

    def __init__(self, manager: DynamicLensManager, target_layer_idx: int = 15):
        self.manager = manager
        self.target_layer_idx = target_layer_idx
        self.timeline = []

    def on_token_generated(self, hidden_states: torch.Tensor, token: str, token_idx: int):
        """Record concepts for this token"""
        h = hidden_states[:, -1, :].cpu().numpy()

        # Detect concepts with logits
        detected, _ = self.manager.detect_and_expand(
            torch.tensor(h, dtype=torch.float32).cuda(),
            top_k=10,
            return_timing=True,
            return_logits=True
        )

        # Store with concept probabilities and logits
        concepts = {}
        for concept_name, prob, logit, layer in detected:
            # Store all detections to see what's being detected
            concepts[concept_name] = {
                'probability': float(prob),
                'logit': float(logit),
                'layer': int(layer)
            }

        self.timeline.append({
            'token': token,
            'token_idx': token_idx,
            'concepts': concepts
        })


def generate_sparkline(values: list, width: int = 40) -> str:
    """Generate ASCII sparkline from values"""
    if not values or all(v == 0 for v in values):
        return '─' * width

    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return '─' * width

    # Sparkline characters from low to high
    chars = '▁▂▃▄▅▆▇█'

    # Normalize and map to characters
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    indices = [int(n * (len(chars) - 1)) for n in normalized]

    # If we have more values than width, sample evenly
    if len(values) > width:
        step = len(values) / width
        sampled_indices = [indices[int(i * step)] for i in range(width)]
        return ''.join(chars[i] for i in sampled_indices)
    else:
        # Pad if fewer values
        sparkline = ''.join(chars[i] for i in indices)
        return sparkline + '─' * (width - len(sparkline))


def generate_with_monitoring(prompt: str, max_tokens: int = 50):
    """Generate text with per-token monitoring"""

    print("Loading model and lenses...")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-3-4b-pt',
        torch_dtype=torch.float32,
        device_map='cuda',
        local_files_only=True
    )
    model.eval()

    # Initialize lens manager - match old test_temporal_continuity_dynamic settings
    manager = DynamicLensManager(
        lens_pack_id='gemma-3-4b-pt_sumo-wordnet-v1',
        base_layers=[3],  # OLD SETTING: base_layer=3 (default in old script)
        max_loaded_lenses=500,
        load_threshold=0.3,  # OLD SETTING: parent_threshold=0.3
        aggressive_pruning=False  # OLD SETTING: no aggressive pruning
    )

    recorder = TokenTimelineRecorder(manager, target_layer_idx=15)

    # Get target layer for hooks
    target_layer = model.model.language_model.layers[15]

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    generated_ids = inputs['input_ids']
    token_count = 0

    print(f"\nPrompt: {prompt}\n")
    print("Capturing prompt processing...")

    # FIRST: Capture prompt processing (initial forward pass)
    with torch.no_grad():
        captured_hidden = []

        def hook(module, input, output):
            captured_hidden.append(output[0])

        handle = target_layer.register_forward_hook(hook)

        # Initial forward pass over the entire prompt
        outputs = model(generated_ids)

        # Record activations for each token in the prompt
        if captured_hidden:
            prompt_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
            # Get hidden states for all prompt tokens
            hidden_states = captured_hidden[0]  # [batch, seq_len, hidden_dim]

            for idx, token in enumerate(prompt_tokens):
                h = hidden_states[:, idx:idx+1, :]
                # Detect concepts for this prompt token (skip pruning during prompt!)
                detected, _ = manager.detect_and_expand(
                    torch.tensor(h[:, 0, :].cpu().numpy(), dtype=torch.float32).cuda(),
                    top_k=20,  # Get more concepts to see GeologicalProcess
                    return_timing=True,
                    return_logits=True,
                    skip_pruning=True  # Don't prune until we've seen the whole prompt!
                )

                concepts = {}
                for concept_name, prob, logit, layer in detected:
                    # Store all detections to see what's being detected
                    concepts[concept_name] = {
                        'probability': float(prob),
                        'logit': float(logit),
                        'layer': int(layer)
                    }

                recorder.timeline.append({
                    'token': token,
                    'token_idx': -1 - idx,  # Negative to mark as prompt token
                    'concepts': concepts,
                    'is_prompt': True
                })

            # NOW prune to top-K based on all prompt activations
            if manager.aggressive_pruning:
                manager._aggressive_prune_to_top_k()

        handle.remove()

    print(f"Captured {len(recorder.timeline)} prompt tokens")
    print("Generating response...")

    with torch.no_grad():
        while token_count < max_tokens:
            # Hook to capture hidden states
            captured_hidden = []

            def hook(module, input, output):
                captured_hidden.append(output[0])

            handle = target_layer.register_forward_hook(hook)

            # Forward pass
            outputs = model(generated_ids)

            # Get next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            next_token = tokenizer.decode(next_token_id[0])

            # Record (with top_k=20 to capture GeologicalProcess)
            if captured_hidden:
                # Get detections
                h = captured_hidden[0][:, -1, :].cpu().numpy()
                detected, _ = manager.detect_and_expand(
                    torch.tensor(h, dtype=torch.float32).cuda(),
                    top_k=20,  # Get more concepts
                    return_timing=True,
                    return_logits=True
                )

                concepts = {}
                for concept_name, prob, logit, layer in detected:
                    concepts[concept_name] = {
                        'probability': float(prob),
                        'logit': float(logit),
                        'layer': int(layer)
                    }

                recorder.timeline.append({
                    'token': next_token,
                    'token_idx': token_count,
                    'concepts': concepts,
                    'is_prompt': False
                })

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            handle.remove()
            token_count += 1

            # Stop if EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode full output
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return full_output, recorder.timeline


def visualize_timeline(output_text: str, timeline: list):
    """Create line-by-line visualization with sparklines"""

    print("\n" + "=" * 80)
    print("DETAILED INSPECTION MODE")
    print("=" * 80)

    # Split output into lines
    lines = output_text.split('\n')

    # Track which concepts appear across all tokens
    all_concepts = set()
    for entry in timeline:
        all_concepts.update(entry['concepts'].keys())

    print(f"\nTotal unique concepts detected: {len(all_concepts)}")
    print(f"Showing top-10 concepts at each token (by logit)\n")

    # Display token by token with top concepts
    for idx, entry in enumerate(timeline):
        if entry.get('is_prompt'):
            continue  # Skip prompt tokens in detailed view

        token = entry['token']

        # Sort concepts by logit (highest first)
        concepts_list = [
            (name, data['logit'], data['probability'], data['layer'])
            for name, data in entry['concepts'].items()
        ]
        concepts_list.sort(key=lambda x: x[1], reverse=True)

        # Show top 10
        print(f"Token {idx:2d} [{token:15s}]", end="")
        top_concepts = concepts_list[:10]
        for name, logit, prob, layer in top_concepts:
            print(f"\n  {name:30s} L{layer} logit={logit:+7.2f} prob={prob:.4f}", end="")
        print("\n")

    # Overall timeline summary
    print("\n" + "=" * 80)
    print("FULL TIMELINE")
    print("=" * 80)

    prompt_tokens = [t for t in timeline if t.get('is_prompt', False)]
    generated_tokens = [t for t in timeline if not t.get('is_prompt', True)]

    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"Generated tokens: {len(generated_tokens)}")
    print(f"Total tokens: {len(timeline)}\n")

    # Show prompt processing concepts
    if prompt_tokens:
        print("─" * 80)
        print("PROMPT PROCESSING (initial forward pass)")
        print("─" * 80)

        # Get concepts that appeared during prompt processing
        prompt_concepts = set()
        for entry in prompt_tokens:
            prompt_concepts.update(entry['concepts'].keys())

        print(f"Concepts detected during prompt: {', '.join(list(prompt_concepts)[:10])}")
        print()

    # Detect potential reasoning chains
    print("\n" + "=" * 80)
    print("CAUSAL CHAIN ANALYSIS")
    print("=" * 80)

    # Look for concept sequences (e.g., planning → reasoning → communication)
    for i in range(len(timeline) - 2):
        concepts_t0 = set(timeline[i]['concepts'].keys())
        concepts_t1 = set(timeline[i + 1]['concepts'].keys())
        concepts_t2 = set(timeline[i + 2]['concepts'].keys())

        # Look for new concepts appearing in sequence
        new_in_t1 = concepts_t1 - concepts_t0
        new_in_t2 = concepts_t2 - concepts_t1

        if new_in_t1 and new_in_t2:
            token0 = timeline[i]['token']
            token1 = timeline[i + 1]['token']
            token2 = timeline[i + 2]['token']
            print(f"Token {i}-{i+2}: [{token0}] → [{token1}] → [{token2}]")
            print(f"  Sequence: {' → '.join(new_in_t1)} ⇒ {' → '.join(new_in_t2)}")

    print("\n" + "=" * 80)


def main():
    prompt = "Mountains are formed when"

    print("🔬 Token Timeline Visualization Demo")
    print("=" * 80)

    output_text, timeline = generate_with_monitoring(prompt, max_tokens=20)

    visualize_timeline(output_text, timeline)

    print("\n✓ Visualization complete")


if __name__ == '__main__':
    main()
