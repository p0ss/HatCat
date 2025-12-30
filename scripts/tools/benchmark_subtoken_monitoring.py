#!/usr/bin/env python3
"""
Benchmark subtoken monitoring overhead.

Tests:
1. Baseline: model.generate() with no monitoring
2. Per-token: Hook registered once per output token
3. Subtoken: Hook on every forward pass (including internal passes)

Measures:
- Tokens per second
- Forward passes per output token
- Overhead percentage
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import numpy as np


class SubtokenRecorder:
    """Records activations at every forward pass"""

    def __init__(self, target_layer_idx: int = 15):
        self.target_layer_idx = target_layer_idx
        self.timeline = []
        self.forward_pass_count = 0
        self.enabled = False

    def on_forward_pass(self, hidden_states: torch.Tensor, token_idx: int):
        """Called on every forward pass"""
        if not self.enabled:
            return

        # Extract last token hidden state
        h = hidden_states[:, -1, :].cpu().numpy()

        self.timeline.append({
            'forward_pass': self.forward_pass_count,
            'token_idx': token_idx,
            'hidden_state_norm': float(np.linalg.norm(h)),
        })
        self.forward_pass_count += 1

    def reset(self):
        self.timeline = []
        self.forward_pass_count = 0


def benchmark_baseline(model, tokenizer, prompt: str, max_tokens: int = 50):
    """Baseline: model.generate() with no monitoring"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start

    n_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    return {
        'mode': 'baseline',
        'n_tokens': n_tokens,
        'elapsed': elapsed,
        'tokens_per_sec': n_tokens / elapsed,
        'forward_passes': n_tokens,  # Assume 1:1 for baseline
    }


def benchmark_manual_loop(model, tokenizer, recorder: SubtokenRecorder,
                          prompt: str, max_tokens: int = 50):
    """Manual generation loop with hooks on every forward pass"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    generated_ids = inputs['input_ids']
    token_count = 0

    # Get target layer
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            target_layer = model.model.layers[recorder.target_layer_idx]
        elif hasattr(model.model, 'language_model'):
            target_layer = model.model.language_model.layers[recorder.target_layer_idx]
        else:
            raise ValueError("Cannot find model layers")
    else:
        target_layer = model.layers[recorder.target_layer_idx]

    recorder.reset()
    recorder.enabled = True

    start = time.time()
    with torch.no_grad():
        while token_count < max_tokens:
            # Register hook for this forward pass
            def make_hook(token_idx):
                def hook(module, input, output):
                    # output[0] is hidden states: (batch, seq, hidden_dim)
                    hidden_states = output[0]
                    recorder.on_forward_pass(hidden_states, token_idx)
                return hook

            handle = target_layer.register_forward_hook(make_hook(token_count))

            # Forward pass
            outputs = model(generated_ids)

            # Sample next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Remove hook
            handle.remove()

            token_count += 1

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    elapsed = time.time() - start
    recorder.enabled = False

    return {
        'mode': 'subtoken_monitoring',
        'n_tokens': token_count,
        'elapsed': elapsed,
        'tokens_per_sec': token_count / elapsed,
        'forward_passes': recorder.forward_pass_count,
        'passes_per_token': recorder.forward_pass_count / token_count if token_count > 0 else 0,
    }


def run_benchmarks(model_name: str = "google/gemma-3-4b-pt", n_trials: int = 3):
    """Run all benchmarks and compare"""
    print(f"🎯 Benchmarking subtoken monitoring overhead")
    print(f"Model: {model_name}")
    print(f"Trials: {n_trials}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cuda',
        local_files_only=True
    )
    model.eval()
    print("✓ Model loaded")

    # Test prompts
    prompts = [
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main challenges in AI safety research?",
        "Describe how language models process text.",
    ]

    recorder = SubtokenRecorder(target_layer_idx=15)

    results_baseline = []
    results_subtoken = []

    for trial in range(n_trials):
        print(f"\n{'─' * 80}")
        print(f"Trial {trial + 1}/{n_trials}")
        print(f"{'─' * 80}")

        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i + 1}: {prompt[:50]}...")

            # Baseline
            result = benchmark_baseline(model, tokenizer, prompt, max_tokens=30)
            results_baseline.append(result)
            print(f"  Baseline:          {result['tokens_per_sec']:.2f} tok/s "
                  f"({result['n_tokens']} tokens in {result['elapsed']:.3f}s)")

            # Subtoken monitoring
            result = benchmark_manual_loop(model, tokenizer, recorder, prompt, max_tokens=30)
            results_subtoken.append(result)
            print(f"  Subtoken monitor:  {result['tokens_per_sec']:.2f} tok/s "
                  f"({result['n_tokens']} tokens in {result['elapsed']:.3f}s)")
            print(f"  Forward passes:    {result['forward_passes']} "
                  f"({result['passes_per_token']:.1f} passes/token)")

            overhead = ((result['elapsed'] / results_baseline[-1]['elapsed']) - 1) * 100
            print(f"  Overhead:          {overhead:+.1f}%")

    # Summary statistics
    print(f"\n{'═' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'═' * 80}")

    baseline_tps = [r['tokens_per_sec'] for r in results_baseline]
    subtoken_tps = [r['tokens_per_sec'] for r in results_subtoken]
    overheads = [((st['elapsed'] / bl['elapsed']) - 1) * 100
                 for st, bl in zip(results_subtoken, results_baseline)]
    passes_per_token = [r['passes_per_token'] for r in results_subtoken]

    print(f"\nBaseline (model.generate):")
    print(f"  Mean: {np.mean(baseline_tps):.2f} tok/s (±{np.std(baseline_tps):.2f})")
    print(f"  Range: {np.min(baseline_tps):.2f} - {np.max(baseline_tps):.2f} tok/s")

    print(f"\nSubtoken monitoring (manual loop + hooks):")
    print(f"  Mean: {np.mean(subtoken_tps):.2f} tok/s (±{np.std(subtoken_tps):.2f})")
    print(f"  Range: {np.min(subtoken_tps):.2f} - {np.max(subtoken_tps):.2f} tok/s")

    print(f"\nOverhead:")
    print(f"  Mean: {np.mean(overheads):+.1f}% (±{np.std(overheads):.1f}%)")
    print(f"  Range: {np.min(overheads):+.1f}% - {np.max(overheads):+.1f}%")

    print(f"\nGranularity:")
    print(f"  Forward passes per token: {np.mean(passes_per_token):.2f} (±{np.std(passes_per_token):.2f})")
    print(f"  → {np.mean(passes_per_token):.1f}× more measurements than baseline")

    print(f"\n{'═' * 80}")
    print("INTERPRETATION")
    print(f"{'═' * 80}")

    avg_overhead = np.mean(overheads)
    if avg_overhead < 10:
        print("✅ Overhead < 10%: Subtoken monitoring is PRACTICAL for real-time use")
    elif avg_overhead < 30:
        print("⚠️  Overhead 10-30%: Acceptable for research/analysis, may impact user experience")
    else:
        print("❌ Overhead > 30%: Too slow for real-time use, consider optimizations")

    avg_passes = np.mean(passes_per_token)
    if avg_passes > 2:
        print(f"✅ {avg_passes:.1f}× granularity: Significant improvement over per-token monitoring")
    else:
        print(f"⚠️  Only {avg_passes:.1f}× granularity: May not capture enough inter-token dynamics")

    print(f"\n{'═' * 80}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark subtoken monitoring overhead')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name or path')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per benchmark')
    args = parser.parse_args()

    run_benchmarks(args.model, args.trials)
