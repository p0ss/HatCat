"""
Stage 1.5: Temporal Activation Sequences

Key insights:
1. Extract SEQUENCES of activations during generation (not pooled single vectors)
2. Train 50K BINARY classifiers (one per concept, not multi-class)
3. Each classifier: pos vs neg sequences
4. Production: Sliding window over generation → multi-hot concept detection

This naturally handles:
- Polysemantic activations (multiple concepts active simultaneously)
- Temporal dynamics (concept activation over time)
- Compositional semantics (concepts appearing together)
"""

import torch
import numpy as np
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path
import argparse


def get_activation_sequence(model, tokenizer, prompt, layer_idx=-1, device="cuda", max_new_tokens=20):
    """
    Generate text and extract activation SEQUENCE (not pooled).

    Returns:
        activations: [num_tokens, hidden_dim] - temporal sequence
        tokens: List of generated token IDs
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Extract activation sequence
        # outputs.hidden_states: tuple of (step0_layers, step1_layers, ...)
        # Each step_layers: tuple of (layer0, layer1, ..., layerN)
        activation_sequence = []

        for step_states in outputs.hidden_states:
            # step_states[layer_idx]: [batch=1, seq_len, hidden_dim]
            last_layer = step_states[layer_idx]
            # Take the last token's activation at this step
            act = last_layer[0, -1, :].float().cpu().numpy()
            activation_sequence.append(act)

        # Stack into sequence: [num_generated_tokens, hidden_dim]
        activation_sequence = np.stack(activation_sequence)

        # Get generated tokens
        tokens = outputs.sequences[0][len(inputs['input_ids'][0]):].cpu().tolist()

    return activation_sequence, tokens


def sample_sequences(model, tokenizer, concept, negatives, related=None, n_samples=50, layer_idx=-1, device="cuda"):
    """
    Sample activation sequences for positive and negative examples.

    Args:
        negatives: List of negative concept names (semantically distant)
        related: Optional list of related concepts for relational prompts

    Returns:
        pos_sequences: List of [seq_len, hidden_dim] arrays
        neg_sequences: List of [seq_len, hidden_dim] arrays
    """
    pos_sequences = []
    neg_sequences = []

    # Positive sequences: Mix of direct and relational prompts
    # 50% direct "What is X?", 50% relational "Relationship between X and Y"
    n_direct = n_samples // 2
    n_relational = n_samples - n_direct

    # Direct prompts
    direct_prompt = f"What is {concept}?"
    for _ in range(n_direct):
        seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
        pos_sequences.append(seq)

    # Relational prompts (if related concepts available)
    if related and len(related) > 0:
        for i in range(n_relational):
            related_concept = related[i % len(related)]
            relational_prompt = f"The relationship between {concept} and {related_concept}"
            seq, _ = get_activation_sequence(model, tokenizer, relational_prompt, layer_idx, device)
            pos_sequences.append(seq)
    else:
        # Fallback to direct prompts if no related concepts
        for _ in range(n_relational):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Negative sequences (from distant concepts)
    if len(negatives) == 0:
        raise ValueError(f"No negatives provided for concept '{concept}'")

    for i in range(n_samples):
        neg_concept = negatives[i % len(negatives)]
        neg_prompt = f"What is {neg_concept}?"
        seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


def pad_sequences(sequences, max_len=None):
    """
    Pad sequences to same length for storage.

    Returns:
        padded: [n_sequences, max_len, hidden_dim]
        lengths: [n_sequences] - original lengths before padding
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    hidden_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, hidden_dim), dtype=np.float16)
    lengths = np.zeros(len(sequences), dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len, :] = seq
        lengths[i] = seq_len

    return padded, lengths


def refine_stage1_5_temporal(
    input_h5: Path,
    output_h5: Path,
    negatives_json: Path = None,
    model_name: str = "google/gemma-3-4b-pt",
    n_samples: int = 50,
    layer_idx: int = -1,
    device: str = "cuda",
    max_seq_len: int = 20
):
    """
    Create temporal activation sequences for binary classification training.

    Output format:
        For each concept:
            - positive_sequences: [n_samples, max_seq_len, hidden_dim]
            - negative_sequences: [n_samples, max_seq_len, hidden_dim]
            - positive_lengths: [n_samples]
            - negative_lengths: [n_samples]

    This enables training 50K binary classifiers (one per concept).
    """
    start_time = time.time()

    print("=" * 70)
    print("STAGE 1.5: TEMPORAL ACTIVATION SEQUENCES")
    print("=" * 70)
    print(f"Input: {input_h5}")
    print(f"Output: {output_h5}")
    print(f"Model: {model_name}")
    print(f"Samples per polarity: {n_samples}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Layer: {layer_idx}")
    print()
    print("Training approach: 50K BINARY classifiers (pos vs neg sequences)")
    print("Production: Sliding window → multi-hot concept detection")
    print()

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # float16 causes CUDA errors with sampling
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Stage 0 data
    print(f"\nLoading Stage 0 encyclopedia...")
    with h5py.File(input_h5, 'r') as f:
        concepts = f['concepts'][:].astype(str)
        n_concepts = len(concepts)
        activation_dim = f.attrs['activation_dim']

    print(f"Loaded {n_concepts:,} concepts")

    # Load negatives mapping
    import json
    negatives_map = {}
    if negatives_json and Path(negatives_json).exists():
        print(f"Loading negative concepts from {negatives_json}...")
        with open(negatives_json) as f:
            negatives_map = json.load(f)
        print(f"Loaded negatives for {len(negatives_map)} concepts")
    else:
        print("⚠ No negatives file provided - using 'What is NOT X?' fallback")

    print(f"Generating {n_samples} pos + {n_samples} neg sequences per concept...")
    print()

    # Create output file
    with h5py.File(output_h5, 'w') as f_out:
        # Metadata
        f_out.attrs['n_concepts'] = n_concepts
        f_out.attrs['activation_dim'] = activation_dim
        f_out.attrs['model'] = model_name
        f_out.attrs['stage'] = 1.5
        f_out.attrs['method'] = 'temporal_sequences'
        f_out.attrs['samples_per_polarity'] = n_samples
        f_out.attrs['max_seq_len'] = max_seq_len
        f_out.attrs['timestamp'] = time.time()

        # Store concepts
        dt = h5py.string_dtype(encoding='utf-8')
        f_out.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

        # Create groups for sequences
        layer_key = f'layer_{layer_idx}'
        grp = f_out.create_group(layer_key)
        grp.attrs['max_seq_len'] = max_seq_len
        grp.attrs['n_samples_per_polarity'] = n_samples

        # Process each concept
        for concept_idx in tqdm(range(n_concepts), desc="Processing concepts"):
            concept = concepts[concept_idx]

            # Get negatives and related concepts for this concept
            if concept in negatives_map:
                negatives = negatives_map[concept].get('negatives', [])
                related = negatives_map[concept].get('related', [])
            else:
                # Fallback: use other concepts as negatives
                negatives = [c for c in concepts if c != concept][:10]
                related = []

            # Sample sequences (with relational prompts if available)
            pos_seqs, neg_seqs = sample_sequences(
                model, tokenizer, concept, negatives, related, n_samples, layer_idx, device
            )

            # Pad to same length
            pos_padded, pos_lengths = pad_sequences(pos_seqs, max_seq_len)
            neg_padded, neg_lengths = pad_sequences(neg_seqs, max_seq_len)

            # Store for this concept
            concept_grp = grp.create_group(f'concept_{concept_idx}')

            concept_grp.create_dataset(
                'positive_sequences',
                data=pos_padded,
                compression='gzip',
                compression_opts=4
            )
            concept_grp.create_dataset(
                'negative_sequences',
                data=neg_padded,
                compression='gzip',
                compression_opts=4
            )
            concept_grp.create_dataset('positive_lengths', data=pos_lengths)
            concept_grp.create_dataset('negative_lengths', data=neg_lengths)

            # Store variance for diagnostics
            pos_var = np.var([seq[:length].mean() for seq, length in zip(pos_seqs, pos_lengths)])
            neg_var = np.var([seq[:length].mean() for seq, length in zip(neg_seqs, neg_lengths)])
            concept_grp.attrs['pos_variance'] = pos_var
            concept_grp.attrs['neg_variance'] = neg_var

    elapsed = time.time() - start_time
    file_size_mb = Path(output_h5).stat().st_size / 1024**2

    print()
    print("=" * 70)
    print("✓ STAGE 1.5 TEMPORAL SEQUENCES COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {n_concepts/elapsed:.1f} concepts/sec")
    print(f"File size: {file_size_mb:.1f} MB")
    print()
    print("Next steps:")
    print(f"  1. Train {n_concepts} binary classifiers (pos vs neg sequences)")
    print(f"  2. Production: Sliding window (size=20, stride=5) over generation")
    print(f"  3. Multi-hot concept detection (threshold > 0.5)")
    print()
    print("Expected capabilities:")
    print("  - Polysemantic detection (multiple concepts per window)")
    print("  - Temporal dynamics (concept evolution over generation)")
    print("  - Compositional semantics (co-occurring concepts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1.5: Temporal sequences")

    parser.add_argument('--input', type=str, required=True,
                       help='Input Stage 0 encyclopedia (HDF5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Stage 1.5 encyclopedia (HDF5)')
    parser.add_argument('--negatives', type=str, default=None,
                       help='JSON file with negative concepts per concept')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples per polarity (default: 50)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index (default: -1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--max-seq-len', type=int, default=20,
                       help='Max sequence length for generation (default: 20)')

    args = parser.parse_args()

    refine_stage1_5_temporal(
        input_h5=Path(args.input),
        output_h5=Path(args.output),
        negatives_json=Path(args.negatives) if args.negatives else None,
        model_name=args.model,
        n_samples=args.n_samples,
        layer_idx=args.layer,
        device=args.device,
        max_seq_len=args.max_seq_len
    )
