"""
Stage 1.5: Generative Diversity

Use actual generation with temperature to get stochastic activation distributions:
- Generate from "What is X?" with temperature=1.0 → diverse continuations
- Generate from "What is NOT X?" with temperature=1.0 → contrastive continuations
- Extract activations from the generation process (not just encoding)
"""

import torch
import numpy as np
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path
import argparse


def get_generative_activation(model, tokenizer, prompt, layer_idx=-1, device="cuda", max_new_tokens=10):
    """
    Generate text and extract activation from the generation.

    Uses temperature=1.0 for stochastic sampling, which produces diverse
    activations even with identical prompts.

    Returns the mean activation across generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        # Generate with temperature sampling
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Extract hidden states from generation
        # outputs.hidden_states is a tuple of tuples: (step, layer, batch, seq, dim)
        # We want the last layer's activations averaged across generated tokens
        hidden_states = outputs.hidden_states

        # Get activations from last layer across all generation steps
        last_layer_acts = []
        for step_states in hidden_states:
            # step_states is tuple of (layer_0, layer_1, ..., layer_n)
            last_layer = step_states[layer_idx]  # [batch, seq, dim]
            # Take the last token's activation for this step
            act = last_layer[0, -1, :].float().cpu().numpy()
            last_layer_acts.append(act)

        # Average across generation steps
        mean_act = np.mean(last_layer_acts, axis=0)

    return mean_act


def sample_activations(model, tokenizer, concept, n_samples=100, layer_idx=-1, device="cuda"):
    """
    Sample activation distribution for a concept using stochastic generation.

    Each call produces different activations due to temperature sampling.

    Returns:
        pos_acts: [n_samples, hidden_dim] - "What is X?"
        neg_acts: [n_samples, hidden_dim] - "What is NOT X?"
    """
    pos_prompt = f"What is {concept}?"
    neg_prompt = f"What is NOT {concept}?"

    pos_acts = []
    neg_acts = []

    # Sample positive examples
    for _ in range(n_samples):
        act = get_generative_activation(model, tokenizer, pos_prompt, layer_idx, device)
        pos_acts.append(act)

    # Sample negative examples
    for _ in range(n_samples):
        act = get_generative_activation(model, tokenizer, neg_prompt, layer_idx, device)
        neg_acts.append(act)

    return np.stack(pos_acts), np.stack(neg_acts)


def refine_stage1_5(
    input_h5: Path,
    output_h5: Path,
    model_name: str = "google/gemma-3-4b-pt",
    n_samples: int = 100,
    layer_idx: int = -1,
    device: str = "cuda"
):
    """
    Refine Stage 0 encyclopedia with generative diversity.

    Args:
        input_h5: Stage 0 encyclopedia
        output_h5: Output path for Stage 1.5
        model_name: Model to use
        n_samples: Number of samples per concept (pos + neg)
        layer_idx: Layer to extract (default: -1 for last layer)
        device: Device to use
    """
    start_time = time.time()

    print("=" * 70)
    print("STAGE 1.5: GENERATIVE DIVERSITY")
    print("=" * 70)
    print(f"Input: {input_h5}")
    print(f"Output: {output_h5}")
    print(f"Model: {model_name}")
    print(f"Samples per concept: {n_samples} pos + {n_samples} neg = {n_samples * 2} total")
    print(f"Layer: {layer_idx}")
    print(f"Method: Temperature sampling (temp=1.0)")
    print()

    # Load model for GENERATION
    print(f"Loading {model_name} for generation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Stage 0 data
    print(f"\nLoading Stage 0 encyclopedia...")
    with h5py.File(input_h5, 'r') as f:
        concepts = f['concepts'][:].astype(str)
        n_concepts = len(concepts)
        activation_dim = f.attrs['activation_dim']

    print(f"Loaded {n_concepts:,} concepts")
    print(f"Generating {n_samples} pos + {n_samples} neg per concept...")
    print(f"Note: This will be SLOWER than encoding (~10 tokens/sample)")
    print()

    # Process concepts
    all_activations = []
    all_labels = []
    all_polarity = []
    all_variances_pos = []
    all_variances_neg = []

    for concept_idx in tqdm(range(n_concepts), desc="Processing concepts"):
        concept = concepts[concept_idx]

        # Sample positive and negative distributions
        pos_acts, neg_acts = sample_activations(
            model, tokenizer, concept, n_samples, layer_idx, device
        )

        # Store all positive samples
        for act in pos_acts:
            all_activations.append(act)
            all_labels.append(concept_idx)
            all_polarity.append(1)

        # Store all negative samples
        for act in neg_acts:
            all_activations.append(act)
            all_labels.append(concept_idx)
            all_polarity.append(0)

        # Compute variances
        all_variances_pos.append(pos_acts.var(axis=0).mean())
        all_variances_neg.append(neg_acts.var(axis=0).mean())

    all_activations = np.stack(all_activations)
    all_labels = np.array(all_labels)
    all_polarity = np.array(all_polarity)
    all_variances_pos = np.array(all_variances_pos)
    all_variances_neg = np.array(all_variances_neg)

    # Save Stage 1.5 encyclopedia
    print(f"\nSaving Stage 1.5 encyclopedia...")
    print(f"Total samples: {len(all_activations)} ({n_concepts} concepts × {n_samples * 2} samples)")

    with h5py.File(output_h5, 'w') as f:
        # Metadata
        f.attrs['n_concepts'] = n_concepts
        f.attrs['activation_dim'] = activation_dim
        f.attrs['model'] = model_name
        f.attrs['stage'] = 1.5
        f.attrs['samples_per_concept'] = n_samples * 2
        f.attrs['pos_samples'] = n_samples
        f.attrs['neg_samples'] = n_samples
        f.attrs['method'] = 'generative_temperature_sampling'
        f.attrs['timestamp'] = time.time()

        # Store concepts
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

        # Store activations
        layer_key = f'layer_{layer_idx}'
        f.create_dataset(
            f'{layer_key}/activations',
            data=all_activations.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )

        # Store labels
        f.create_dataset(
            f'{layer_key}/labels',
            data=all_labels.astype(np.int32),
            compression='gzip',
            compression_opts=4
        )

        # Store polarity
        f.create_dataset(
            f'{layer_key}/polarity',
            data=all_polarity.astype(np.int8),
            compression='gzip',
            compression_opts=4
        )

        # Store variances
        f.create_dataset(
            f'{layer_key}/variance_pos',
            data=all_variances_pos.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset(
            f'{layer_key}/variance_neg',
            data=all_variances_neg.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )

        # Metadata
        f[layer_key].attrs['samples_per_concept'] = n_samples * 2
        f[layer_key].attrs['confidence'] = 'high'
        f[layer_key].attrs['stage'] = 1.5
        f[layer_key].attrs['total_samples'] = len(all_activations)
        f[layer_key].attrs['has_polarity'] = True

    elapsed = time.time() - start_time
    file_size_mb = Path(output_h5).stat().st_size / 1024**2

    print()
    print("=" * 70)
    print("✓ STAGE 1.5 GENERATIVE DIVERSITY COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {n_concepts/elapsed:.1f} concepts/sec")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Average variance (pos): {all_variances_pos.mean():.6f}")
    print(f"Average variance (neg): {all_variances_neg.mean():.6f}")
    print()
    print("Dataset statistics:")
    print(f"  - Total samples: {len(all_activations):,}")
    print(f"  - Positive samples: {all_polarity.sum():,}")
    print(f"  - Negative samples: {(all_polarity == 0).sum():,}")
    print(f"  - Samples per concept: {n_samples * 2}")
    print()
    print("Expected improvements over templates:")
    print("  - TRUE stochastic diversity from generation")
    print("  - Natural language variation in continuations")
    print("  - Contrastive learning signal (pos vs neg)")
    print("  - Better val/test match")
    print("  - Validation accuracy: 55% → 70-80%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1.5: Generative diversity")

    parser.add_argument('--input', type=str, required=True,
                       help='Input Stage 0 encyclopedia (HDF5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Stage 1.5 encyclopedia (HDF5)')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples per concept per polarity (default: 50, NOT 100 - generation is slow!)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index (default: -1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    refine_stage1_5(
        input_h5=Path(args.input),
        output_h5=Path(args.output),
        model_name=args.model,
        n_samples=args.n_samples,
        layer_idx=args.layer,
        device=args.device
    )
