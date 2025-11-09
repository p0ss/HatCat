"""
Stage 1: Template-Based Refinement

Add 5-10 template contexts per concept to enable generalization.
Expected confidence increase: 40% → 70%
"""

import torch
import numpy as np
import h5py
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path
import argparse


def get_activation(model, tokenizer, text, layer_idx=-1, device="cuda"):
    """Extract activation vector with proper attention masking."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)

    hs = out.hidden_states[layer_idx]
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.squeeze(0).float().cpu().numpy()


def generate_templates(concept, n_samples=10):
    """Generate diverse template contexts for a concept."""
    templates = [
        f"{concept}",
        f"The concept of {concept}",
        f"An example of {concept} is",
        f"{concept} refers to",
        f"When discussing {concept}",
        f"The meaning of {concept}",
        f"In the context of {concept}",
        f"Understanding {concept}",
        f"{concept} can be described as",
        f"The nature of {concept}",
        f"Regarding {concept}",
        f"With respect to {concept}",
        f"{concept} involves",
        f"The essence of {concept}",
        f"Considering {concept}",
        f"{concept} encompasses",
        f"The significance of {concept}",
        f"Exploring {concept}",
        f"{concept} manifests as",
        f"The role of {concept}",
    ]
    return templates[:n_samples]


def refine_stage1(
    input_h5: Path,
    output_h5: Path,
    model_name: str = "google/gemma-3-270m",
    n_samples: int = 10,
    layer_idx: int = -1,
    device: str = "cuda",
    batch_size: int = 32
):
    """
    Refine Stage 0 encyclopedia with template contexts.

    Args:
        input_h5: Stage 0 encyclopedia
        output_h5: Output path for Stage 1
        model_name: Model to use
        n_samples: Number of template samples per concept
        layer_idx: Layer to extract (default: -1 for last layer)
        device: Device to use
        batch_size: Batch size for processing
    """
    start_time = time.time()

    print("=" * 70)
    print("STAGE 1: TEMPLATE REFINEMENT")
    print("=" * 70)
    print(f"Input: {input_h5}")
    print(f"Output: {output_h5}")
    print(f"Model: {model_name}")
    print(f"Samples per concept: {n_samples}")
    print(f"Layer: {layer_idx}")
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
    print(f"Generating {n_samples} templates per concept...")
    print()

    # Process concepts - store ALL samples, not just means
    all_activations = []  # Will be shape [n_concepts * n_samples, activation_dim]
    all_labels = []  # Concept ID for each sample
    all_variances = []  # Variance per concept

    for concept_idx in tqdm(range(0, n_concepts), desc="Processing concepts"):
        concept = concepts[concept_idx]

        # Generate templates
        templates = generate_templates(concept, n_samples)

        # Get activations for all templates
        concept_acts = []
        for template in templates:
            act = get_activation(model, tokenizer, template, layer_idx, device)
            concept_acts.append(act)
            all_labels.append(concept_idx)  # Track which concept this sample belongs to

        concept_acts = np.stack(concept_acts)

        # Store all samples (not just mean)
        all_activations.extend(concept_acts)

        # Compute variance for this concept
        variance = concept_acts.var(axis=0).mean()
        all_variances.append(variance)

    all_activations = np.stack(all_activations)  # Shape: [n_concepts * n_samples, activation_dim]
    all_labels = np.array(all_labels)  # Shape: [n_concepts * n_samples]
    all_variances = np.array(all_variances)  # Shape: [n_concepts]

    # Save Stage 1 encyclopedia
    print(f"\nSaving Stage 1 encyclopedia...")
    print(f"Total samples: {len(all_activations)} ({n_concepts} concepts × {n_samples} samples)")
    with h5py.File(output_h5, 'w') as f:
        # Metadata
        f.attrs['n_concepts'] = n_concepts
        f.attrs['activation_dim'] = activation_dim
        f.attrs['model'] = model_name
        f.attrs['stage'] = 1
        f.attrs['samples_per_concept'] = n_samples
        f.attrs['timestamp'] = time.time()

        # Store concepts
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

        # Store activations - now n_concepts * n_samples rows
        layer_key = f'layer_{layer_idx}'
        f.create_dataset(
            f'{layer_key}/activations',
            data=all_activations.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )

        # Store labels - which concept each sample belongs to
        f.create_dataset(
            f'{layer_key}/labels',
            data=all_labels.astype(np.int32),
            compression='gzip',
            compression_opts=4
        )

        # Store variance per concept
        f.create_dataset(
            f'{layer_key}/variance',
            data=all_variances.astype(np.float16),
            compression='gzip',
            compression_opts=4
        )

        # Metadata
        f[layer_key].attrs['samples_per_concept'] = n_samples
        f[layer_key].attrs['confidence'] = 'medium'
        f[layer_key].attrs['stage'] = 1
        f[layer_key].attrs['total_samples'] = len(all_activations)

    elapsed = time.time() - start_time
    file_size_mb = Path(output_h5).stat().st_size / 1024**2

    print()
    print("=" * 70)
    print("✓ STAGE 1 REFINEMENT COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {n_concepts/elapsed:.1f} concepts/sec")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Average variance: {all_variances.mean():.6f}")
    print()
    print("Expected improvements:")
    print("  - Validation accuracy: 0% → 40-60%")
    print("  - Confidence: 40% → 70%")
    print("  - Generalization: Model can now learn concept patterns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Template refinement")

    parser.add_argument('--input', type=str, required=True,
                       help='Input Stage 0 encyclopedia (HDF5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Stage 1 encyclopedia (HDF5)')
    parser.add_argument('--model', type=str, default='google/gemma-3-270m',
                       help='Model to use')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of template samples per concept (default: 10)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index (default: -1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')

    args = parser.parse_args()

    refine_stage1(
        input_h5=Path(args.input),
        output_h5=Path(args.output),
        model_name=args.model,
        n_samples=args.n_samples,
        layer_idx=args.layer,
        device=args.device,
        batch_size=args.batch_size
    )
