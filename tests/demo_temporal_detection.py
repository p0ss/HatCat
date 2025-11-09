"""
Demo: Temporal Concept Detection on Real Generation

Shows how binary classifiers detect concepts over time during generation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import h5py
from typing import Dict, List, Tuple
import argparse


class BinaryConceptClassifier(nn.Module):
    """Binary classifier for single concept detection."""

    def __init__(self, hidden_dim: int, lstm_dim: int = 256):
        super().__init__()

        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, sequences, lengths):
        """
        Args:
            sequences: [batch, max_seq_len, hidden_dim]
            lengths: [batch] - actual lengths before padding

        Returns:
            probs: [batch] - probability concept is active
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)
        final_state = torch.cat([hidden[0], hidden[1]], dim=-1)
        logits = self.classifier(final_state)
        probs = logits.squeeze(-1)

        return probs


def load_classifiers(
    model_dir: Path,
    h5_path: Path,
    device: str = 'cuda'
) -> Tuple[Dict[str, nn.Module], List[str]]:
    """Load all trained binary classifiers."""

    # Load concept names
    with h5py.File(h5_path, 'r') as f:
        concepts = f['concepts'][:].astype(str)
        concept_grp = f['layer_-1/concept_0']
        hidden_dim = concept_grp['positive_sequences'].shape[2]

    # Load classifiers
    classifiers = {}
    for i, concept in enumerate(concepts):
        checkpoint_path = model_dir / f'concept_{i}' / 'best.ckpt'
        if not checkpoint_path.exists():
            print(f"Warning: No checkpoint for concept {i} ({concept})")
            continue

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model and load weights
        model = BinaryConceptClassifier(hidden_dim).to(device)

        # Extract model weights from lightning checkpoint
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                state_dict[new_key] = value

        model.load_state_dict(state_dict)
        model.eval()

        classifiers[concept] = model

    return classifiers, list(concepts)


def extract_generation_sequence(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    layer_idx: int = -1,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Generate text and extract activation sequence.

    Returns:
        activations: List of [hidden_dim] tensors (one per token)
        tokens: List of generated token strings
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Extract activation sequence
        activations = []
        for step_states in outputs.hidden_states:
            last_layer = step_states[layer_idx]
            act = last_layer[0, -1, :].float()  # Keep on GPU
            activations.append(act)

        # Decode tokens
        token_ids = outputs.sequences[0][len(inputs['input_ids'][0]):].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return activations, tokens


def detect_concepts_sliding_window(
    activations: List[torch.Tensor],
    classifiers: Dict[str, nn.Module],
    window_size: int = 20,
    stride: int = 5,
    threshold: float = 0.5,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Run sliding window concept detection over activation sequence.

    Returns:
        timeline: List of {position, concepts} dicts
    """
    timeline = []

    for start_idx in range(0, len(activations) - window_size + 1, stride):
        window = activations[start_idx:start_idx + window_size]

        # Stack into sequence: [1, window_size, hidden_dim]
        window_tensor = torch.stack(window).unsqueeze(0)
        length_tensor = torch.tensor([len(window)], device=device)

        # Run all classifiers on this window
        active_concepts = {}

        with torch.no_grad():
            for concept, classifier in classifiers.items():
                prob = classifier(window_tensor, length_tensor).item()

                if prob > threshold:
                    active_concepts[concept] = prob

        timeline.append({
            'position': start_idx,
            'end_position': start_idx + window_size,
            'concepts': active_concepts
        })

    return timeline


def print_temporal_narrative(
    prompt: str,
    tokens: List[str],
    timeline: List[Dict]
):
    """Pretty-print the temporal concept detection."""

    print()
    print("=" * 80)
    print("TEMPORAL CONCEPT DETECTION")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print()
    print(f"Generated: {''.join(tokens)}")
    print()
    print("-" * 80)
    print("TEMPORAL TIMELINE:")
    print("-" * 80)

    for window in timeline:
        start = window['position']
        end = window['end_position']
        concepts = window['concepts']

        # Get the tokens in this window
        window_text = ''.join(tokens[start:min(end, len(tokens))])

        print()
        print(f"Position {start}-{end}: \"{window_text[:50]}...\"")

        if concepts:
            # Sort by probability
            sorted_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
            for concept, prob in sorted_concepts:
                print(f"  • {concept}: {prob:.3f}")
        else:
            print("  (no concepts detected)")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Demo temporal concept detection")

    parser.add_argument('--models', type=str, required=True,
                       help='Directory with trained classifiers')
    parser.add_argument('--data', type=str, required=True,
                       help='HDF5 file with concept names')
    parser.add_argument('--prompt', type=str, default='Explain machine learning:',
                       help='Generation prompt')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Max tokens to generate')
    parser.add_argument('--window-size', type=int, default=20,
                       help='Sliding window size')
    parser.add_argument('--stride', type=int, default=5,
                       help='Sliding window stride')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Generation model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("Loading classifiers...")
    classifiers, concepts = load_classifiers(
        Path(args.models),
        Path(args.data),
        args.device
    )
    print(f"Loaded {len(classifiers)} classifiers for: {', '.join(concepts)}")

    print(f"\nLoading generation model ({args.model})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # float16 causes sampling issues
        device_map=args.device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nGenerating from prompt: \"{args.prompt}\"")
    activations, tokens = extract_generation_sequence(
        model, tokenizer, args.prompt, args.max_tokens, device=args.device
    )

    print(f"Generated {len(tokens)} tokens, extracting {len(activations)} activations")

    print("\nRunning sliding window concept detection...")
    timeline = detect_concepts_sliding_window(
        activations,
        classifiers,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
        device=args.device
    )

    print_temporal_narrative(args.prompt, tokens, timeline)


if __name__ == '__main__':
    main()
