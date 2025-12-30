"""
Record concept activations at every forward pass during generation.

Usage:
    python scripts/record_temporal_activations.py \
        --prompt "Tell me about deception in politics" \
        --output results/temporal_test/deception_politics.json
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hat.sumo_hierarchical import SUMOHierarchicalMonitor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalActivationRecorder:
    """Record activations at every forward pass, not just token emissions"""

    def __init__(self, monitor: SUMOHierarchicalMonitor):
        self.monitor = monitor
        self.timeline = []
        self.forward_pass_count = 0

    def on_forward_pass(self, hidden_states, token_idx, is_generation_step):
        """Called on every forward pass through the model"""
        # Detect concepts using hierarchical monitor
        detections = self.monitor.detect_concepts(
            hidden_states.cpu().numpy(),
            return_all=True  # Get all concepts, not just top-K
        )

        # Record timestep
        self.timeline.append({
            'forward_pass': self.forward_pass_count,
            'token_idx': token_idx,
            'is_output': is_generation_step,  # True if this generates a token
            'concepts': {
                name: {
                    'divergence': float(det['divergence']),
                    'probability': float(det['probability']),
                    'layer': det['layer']
                }
                for name, det in detections.items()
                if det['divergence'] > 0.01  # Only keep non-trivial activations
            }
        })

        self.forward_pass_count += 1

    def save(self, output_path: Path):
        """Save timeline to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'total_forward_passes': self.forward_pass_count,
                'timeline': self.timeline
            }, f, indent=2)
        logger.info(f"Saved {self.forward_pass_count} timesteps to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--lens-dir', type=Path,
                       default=Path('results/sumo_classifiers'))
    args = parser.parse_args()

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='cuda',
        torch_dtype=torch.float16
    )
    model.eval()

    # Load SUMO hierarchical monitor
    logger.info(f"Loading lenses from: {args.lens_dir}")
    monitor = SUMOHierarchicalMonitor(
        lens_dir=args.lens_dir,
        device='cuda'
    )

    # Create recorder
    recorder = TemporalActivationRecorder(monitor)

    # Hook into model to capture every forward pass
    def make_hook(token_idx, is_output):
        def hook(module, input, output):
            # output[0] is hidden states: (batch, seq, hidden_dim)
            hidden_states = output[0][:, -1, :]  # Last token
            recorder.on_forward_pass(hidden_states, token_idx, is_output)
        return hook

    # Register hook on the layer we're monitoring
    # (Assuming we monitor layer 15 - adjust based on your config)
    target_layer_idx = 15
    if hasattr(model.model, 'layers'):
        target_layer = model.model.layers[target_layer_idx]
    elif hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[target_layer_idx]
    else:
        raise ValueError("Unknown model architecture")

    # Tokenize prompt
    logger.info(f"Prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors='pt').to('cuda')

    # Generate with hooks
    logger.info(f"Generating {args.max_tokens} tokens...")

    # We'll manually step through generation to track each forward pass
    generated_ids = inputs['input_ids']
    token_count = 0

    with torch.no_grad():
        while token_count < args.max_tokens:
            # Register hook for this forward pass
            is_output = True  # This forward pass will emit a token
            handle = target_layer.register_forward_hook(
                make_hook(token_count, is_output)
            )

            # Forward pass
            outputs = model(generated_ids)

            # Sample next token
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

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"Generated: {generated_text}")

    # Save results
    output_data = {
        'prompt': args.prompt,
        'generated_text': generated_text,
        'model': args.model,
        'total_tokens': token_count,
        'recorder': recorder.timeline
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved to {args.output}")
    logger.info(f"Captured {len(recorder.timeline)} forward passes for {token_count} tokens")


if __name__ == '__main__':
    main()
