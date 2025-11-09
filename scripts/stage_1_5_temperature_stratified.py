"""
Stage 1.5: Temperature-Stratified Temporal Sequences

Key insight: Sample distribution should match generation distribution
- Center (temp=0.3): Canonical explanations, RLHF-aligned outputs
- Width (temp=0.8): Normal variation, standard phrasing
- Tails (temp=1.2): Edge cases, creative interpretations, rare phrasing

Samples scale with:
1. Model size: ~10 samples per billion parameters
2. Connectivity: High-relationship concepts need more coverage
"""

import torch
import numpy as np
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path
import argparse
import json


def get_activation_sequence_with_temp(model, tokenizer, prompt, temperature=1.0, layer_idx=-1, device="cuda", max_new_tokens=20):
    """
    Generate with specified temperature and extract activation sequence.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Extract activation sequence
        activation_sequence = []
        for step_states in outputs.hidden_states:
            last_layer = step_states[layer_idx]
            act = last_layer[0, -1, :].float().cpu().numpy()
            activation_sequence.append(act)

        activation_sequence = np.stack(activation_sequence)
        tokens = outputs.sequences[0][len(inputs['input_ids'][0]):].cpu().tolist()

    return activation_sequence, tokens


def calculate_sample_count(connectivity, model_size_b=4, base_samples_per_b=10):
    """
    Calculate number of samples based on connectivity and model size.

    Args:
        connectivity: Number of relationships for this concept
        model_size_b: Model size in billions of parameters
        base_samples_per_b: Base samples per billion parameters

    Returns:
        (n_center, n_width, n_tails) for temperature stratification
    """
    # Base samples from model size
    base_samples = int(model_size_b * base_samples_per_b)

    # Scale by connectivity (log scale to avoid explosion)
    connectivity_scale = 1.0 + np.log1p(connectivity) / 10.0
    total_samples = int(base_samples * connectivity_scale)

    # Distribute across temperature bands
    # Center: 20%, Width: 50%, Tails: 30%
    n_center = int(total_samples * 0.20)
    n_width = int(total_samples * 0.50)
    n_tails = total_samples - n_center - n_width

    return n_center, n_width, n_tails


def sample_temperature_stratified(
    model,
    tokenizer,
    concept: str,
    negatives: list,
    related_structured: dict,
    connectivity: int,
    model_size_b: float = 4.0,
    layer_idx: int = -1,
    device: str = "cuda"
):
    """
    Sample with temperature stratification to capture full distribution.
    """
    pos_sequences = []
    neg_sequences = []

    # Calculate sample counts
    n_center, n_width, n_tails = calculate_sample_count(connectivity, model_size_b)
    total_samples = n_center + n_width + n_tails

    print(f"  Sampling {total_samples} samples: {n_center} center (0.3), {n_width} width (0.8), {n_tails} tails (1.2)")

    # Positive samples: Mix of definitional and relational
    direct_prompt = f"What is {concept}?"

    # Get related concepts for relational prompts, ordered by relationship strength
    # Priority: hypernyms (is-a) > hyponyms (types) > meronyms (parts) > holonyms (wholes)
    all_related = []
    for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
        if rel_type in related_structured:
            # Add with relationship type tag for diversity
            all_related.extend(related_structured[rel_type])

    # Center (temp=0.3): Canonical explanations
    for _ in range(n_center):
        seq, _ = get_activation_sequence_with_temp(
            model, tokenizer, direct_prompt, temperature=0.3, layer_idx=layer_idx, device=device
        )
        pos_sequences.append(seq)

    # Width (temp=0.8): Mix of direct and relational
    n_width_direct = n_width // 2
    n_width_relational = n_width - n_width_direct

    for _ in range(n_width_direct):
        seq, _ = get_activation_sequence_with_temp(
            model, tokenizer, direct_prompt, temperature=0.8, layer_idx=layer_idx, device=device
        )
        pos_sequences.append(seq)

    for i in range(n_width_relational):
        if all_related:
            related_concept = all_related[i % len(all_related)]
            relational_prompt = f"The relationship between {concept} and {related_concept}"
        else:
            relational_prompt = direct_prompt

        seq, _ = get_activation_sequence_with_temp(
            model, tokenizer, relational_prompt, temperature=0.8, layer_idx=layer_idx, device=device
        )
        pos_sequences.append(seq)

    # Tails (temp=1.2): Edge cases and creative interpretations
    for _ in range(n_tails):
        # Alternate between direct and relational
        if all_related and _ % 2 == 0:
            related_concept = all_related[_ % len(all_related)]
            prompt = f"The relationship between {concept} and {related_concept}"
        else:
            prompt = direct_prompt

        seq, _ = get_activation_sequence_with_temp(
            model, tokenizer, prompt, temperature=1.2, layer_idx=layer_idx, device=device
        )
        pos_sequences.append(seq)

    # Negative samples (match total count, standard temperature)
    if len(negatives) == 0:
        raise ValueError(f"No negatives for concept '{concept}'")

    for i in range(total_samples):
        neg_concept = negatives[i % len(negatives)]
        neg_prompt = f"What is {neg_concept}?"
        seq, _ = get_activation_sequence_with_temp(
            model, tokenizer, neg_prompt, temperature=0.8, layer_idx=layer_idx, device=device
        )
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


def pad_sequences(sequences, max_len=None):
    """Pad sequences to same length."""
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


def extract_temperature_stratified(
    concept_graph_path: Path,
    output_h5: Path,
    model_name: str = "google/gemma-3-4b-pt",
    model_size_b: float = 4.0,
    layer_idx: int = -1,
    device: str = "cuda",
    max_seq_len: int = 20
):
    """
    Extract temporal sequences with temperature stratification.
    """
    start_time = time.time()

    print("=" * 70)
    print("STAGE 1.5: TEMPERATURE-STRATIFIED TEMPORAL SEQUENCES")
    print("=" * 70)
    print(f"Concept graph: {concept_graph_path}")
    print(f"Output: {output_h5}")
    print(f"Model: {model_name} ({model_size_b}B parameters)")
    print(f"Base samples: {int(model_size_b * 10)} (10 per B parameter)")
    print(f"Temperature bands: 0.3 (center), 0.8 (width), 1.2 (tails)")
    print(f"Connectivity scaling: log1p(connectivity) / 10")
    print()

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load concept graph
    print(f"\nLoading concept graph...")
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    concepts = list(concept_data.keys())
    n_concepts = len(concepts)

    # Get activation dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        activation_dim = test_output.hidden_states[-1].shape[-1]

    print(f"Loaded {n_concepts:,} concepts")
    print(f"Activation dim: {activation_dim}")
    print()

    # Create output file
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, 'w') as f_out:
        # Metadata
        f_out.attrs['n_concepts'] = n_concepts
        f_out.attrs['activation_dim'] = activation_dim
        f_out.attrs['model'] = model_name
        f_out.attrs['model_size_b'] = model_size_b
        f_out.attrs['stage'] = 1.5
        f_out.attrs['method'] = 'temperature_stratified'
        f_out.attrs['max_seq_len'] = max_seq_len
        f_out.attrs['timestamp'] = time.time()

        # Store concepts
        dt = h5py.string_dtype(encoding='utf-8')
        f_out.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

        # Create group
        layer_key = f'layer_{layer_idx}'
        grp = f_out.create_group(layer_key)
        grp.attrs['max_seq_len'] = max_seq_len

        # Process each concept
        for concept_idx in tqdm(range(n_concepts), desc="Processing concepts"):
            concept = concepts[concept_idx]

            negatives = concept_data[concept].get('negatives', [])
            related_structured = concept_data[concept].get('related_structured', {})
            connectivity = concept_data[concept].get('connectivity', 0)

            if len(negatives) == 0:
                print(f"  ⚠ Skipping '{concept}' - no negatives")
                continue

            # Sample with temperature stratification
            try:
                pos_seqs, neg_seqs = sample_temperature_stratified(
                    model, tokenizer, concept, negatives, related_structured,
                    connectivity, model_size_b, layer_idx, device
                )
            except Exception as e:
                print(f"  ⚠ Error sampling '{concept}': {e}")
                continue

            # Pad sequences
            pos_padded, pos_lengths = pad_sequences(pos_seqs, max_seq_len)
            neg_padded, neg_lengths = pad_sequences(neg_seqs, max_seq_len)

            # Store
            concept_grp = grp.create_group(f'concept_{concept_idx}')
            concept_grp.create_dataset('positive_sequences', data=pos_padded, compression='gzip', compression_opts=4)
            concept_grp.create_dataset('negative_sequences', data=neg_padded, compression='gzip', compression_opts=4)
            concept_grp.create_dataset('positive_lengths', data=pos_lengths)
            concept_grp.create_dataset('negative_lengths', data=neg_lengths)
            concept_grp.attrs['connectivity'] = connectivity
            concept_grp.attrs['n_samples'] = len(pos_seqs)

    elapsed = time.time() - start_time
    file_size_mb = output_h5.stat().st_size / 1024**2

    print()
    print("=" * 70)
    print("✓ TEMPERATURE-STRATIFIED EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {n_concepts/elapsed:.2f} concepts/sec")
    print(f"File size: {file_size_mb:.1f} MB")
    print()
    print("Next: Train binary classifiers on temperature-stratified data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature-stratified temporal sequences")

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model name')
    parser.add_argument('--model-size-b', type=float, default=4.0,
                       help='Model size in billions of parameters')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer index')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--max-seq-len', type=int, default=20,
                       help='Max sequence length')

    args = parser.parse_args()

    extract_temperature_stratified(
        concept_graph_path=Path(args.concept_graph),
        output_h5=Path(args.output),
        model_name=args.model,
        model_size_b=args.model_size_b,
        layer_idx=args.layer,
        device=args.device,
        max_seq_len=args.max_seq_len
    )
