"""
Stage 1.5: Dropout-Based Diversity

Simplest approach: Enable dropout during inference to get stochastic activations.
- Much faster than generation
- True diversity from dropout masks
- No need for temperature or special models
"""

import torch
import numpy as np
import h5py
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path
import argparse


def get_activation_with_dropout(model, tokenizer, text, layer_idx=-1, device="cuda"):
    """
    Extract activation with dropout enabled for stochasticity.

    Model must be in train() mode for dropout to be active.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():  # Still no grad, but dropout is active
        out = model(**inputs, output_hidden_states=True)

    hs = out.hidden_states[layer_idx]
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.squeeze(0).float().cpu().numpy()


def sample_activations(model, tokenizer, concept, n_samples=50, layer_idx=-1, device="cuda"):
    """
    Sample activation distribution using dropout.

    Each forward pass uses different dropout masks → diverse activations.
    """
    pos_prompt = f"What is {concept}?"
    neg_prompt = f"What is NOT {concept}?"

    pos_acts = []
    neg_acts = []

    # Sample positive examples
    for _ in range(n_samples):
        act = get_activation_with_dropout(model, tokenizer, pos_prompt, layer_idx, device)
        pos_acts.append(act)

    # Sample negative examples
    for _ in range(n_samples):
        act = get_activation_with_dropout(model, tokenizer, neg_prompt, layer_idx, device)
        neg_acts.append(act)

    return np.stack(pos_acts), np.stack(neg_acts)


def refine_stage1_5(
    input_h5: Path,
    output_h5: Path,
    model_name: str = "google/gemma-3-270m",
    n_samples: int = 50,
    layer_idx: int = -1,
    device: str = "cuda",
    dropout_rate: float = 0.1
):
    """
    Refine Stage 0 encyclopedia with dropout-based diversity.
    """
    start_time = time.time()

    print("=" * 70)
    print("STAGE 1.5: DROPOUT DIVERSITY")
    print("=" * 70)
    print(f"Input: {input_h5}")
    print(f"Output: {output_h5}")
    print(f"Model: {model_name}")
    print(f"Samples per concept: {n_samples} pos + {n_samples} neg = {n_samples * 2} total")
    print(f"Layer: {layer_idx}")
    print(f"Dropout rate: {dropout_rate}")
    print()

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.output_hidden_states = True

    # IMPORTANT: Set model to train() mode to enable dropout
    # But also set all batchnorm/layernorm to eval mode
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()

    # Override dropout rate if specified
    if dropout_rate is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

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
    print(f"Sampling {n_samples} pos + {n_samples} neg per concept with dropout...")
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

    # Save
    print(f"\nSaving Stage 1.5 encyclopedia...")
    print(f"Total samples: {len(all_activations)} ({n_concepts} concepts × {n_samples * 2} samples)")

    with h5py.File(output_h5, 'w') as f:
        f.attrs['n_concepts'] = n_concepts
        f.attrs['activation_dim'] = activation_dim
        f.attrs['model'] = model_name
        f.attrs['stage'] = 1.5
        f.attrs['samples_per_concept'] = n_samples * 2
        f.attrs['pos_samples'] = n_samples
        f.attrs['neg_samples'] = n_samples
        f.attrs['method'] = 'dropout_sampling'
        f.attrs['dropout_rate'] = dropout_rate
        f.attrs['timestamp'] = time.time()

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

        layer_key = f'layer_{layer_idx}'
        f.create_dataset(
            f'{layer_key}/activations',
            data=all_activations.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset(
            f'{layer_key}/labels',
            data=all_labels.astype(np.int32),
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset(
            f'{layer_key}/polarity',
            data=all_polarity.astype(np.int8),
            compression='gzip',
            compression_opts=4
        )
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

        f[layer_key].attrs['samples_per_concept'] = n_samples * 2
        f[layer_key].attrs['stage'] = 1.5
        f[layer_key].attrs['total_samples'] = len(all_activations)
        f[layer_key].attrs['has_polarity'] = True

    elapsed = time.time() - start_time
    file_size_mb = Path(output_h5).stat().st_size / 1024**2

    print()
    print("=" * 70)
    print("✓ STAGE 1.5 DROPOUT DIVERSITY COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {n_concepts/elapsed:.1f} concepts/sec")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Average variance (pos): {all_variances_pos.mean():.6f}")
    print(f"Average variance (neg): {all_variances_neg.mean():.6f}")
    print()
    if all_variances_pos.mean() < 1e-6:
        print("⚠️  WARNING: Variance is very low - dropout may not be working!")
        print("    Check that model has dropout layers.")
    else:
        print("✓ Dropout diversity confirmed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1.5: Dropout diversity")

    parser.add_argument('--input', type=str, required=True,
                       help='Input Stage 0 encyclopedia (HDF5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Stage 1.5 encyclopedia (HDF5)')
    parser.add_argument('--model', type=str, default='google/gemma-3-270m',
                       help='Model to use')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples per polarity (default: 50)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index (default: -1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')

    args = parser.parse_args()

    refine_stage1_5(
        input_h5=Path(args.input),
        output_h5=Path(args.output),
        model_name=args.model,
        n_samples=args.n_samples,
        layer_idx=args.layer,
        device=args.device,
        dropout_rate=args.dropout
    )
