"""
Generate concept name embeddings using the model's embedding layer.

This version uses only the tokenizer and embedding layer, not the full model,
so it can run on CPU with minimal memory.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def load_concept_mapping(layer: int, lens_pack_dir: Path) -> dict:
    """Load concept names from existing centroid embeddings."""
    # Get concept names from existing centroids
    layer_dir = lens_pack_dir / f"layer{layer}" / "embedding_centroids"
    if not layer_dir.exists():
        raise ValueError(f"Layer directory not found: {layer_dir}")

    concept_names = {}
    for centroid_file in layer_dir.glob("*_centroid.npy"):
        # Remove _centroid.npy suffix to get concept name
        concept_name = centroid_file.stem.replace("_centroid", "")
        concept_names[concept_name] = True

    return concept_names

def get_embedding_weights(model_name: str, device: str = "cpu"):
    """Load just the embedding layer weights."""
    from transformers import AutoConfig, AutoModel

    print(f"Loading embedding weights for {model_name}...")

    # Load config
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)

    # Load embedding layer from safetensors
    from safetensors import safe_open
    model_path = Path.home() / ".cache" / "huggingface" / "hub"

    # Find the model directory
    model_dirs = list(model_path.glob("models--google--gemma-3-4b-pt*"))
    if not model_dirs:
        raise ValueError(f"Model not found in cache: {model_name}")

    snapshot_dir = model_dirs[0] / "snapshots"
    snapshot = list(snapshot_dir.iterdir())[0]

    # Load embeddings from first shard
    shard_file = snapshot / "model-00001-of-00002.safetensors"

    with safe_open(str(shard_file), framework="pt", device=device) as f:
        embeddings = f.get_tensor("language_model.model.embed_tokens.weight")

    return embeddings

def generate_concept_embeddings(
    model_name: str,
    layers: list[int],
    output_dir: Path,
    device: str = "cpu",
):
    """Generate embeddings for all concept names."""

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    print(f"Loading embedding weights...")
    embedding_weights = get_embedding_weights(model_name, device)
    print(f"Embedding shape: {embedding_weights.shape}")

    for layer in layers:
        print(f"\n{'='*80}")
        print(f"Layer {layer}")
        print(f"{'='*80}")

        # Load concepts for this layer
        concepts = load_concept_mapping(layer, output_dir)
        print(f"Generating embeddings for {len(concepts)} concepts")

        # Create output directory
        layer_dir = output_dir / f"layer{layer}" / "embedding_centroids"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Generate embedding for each concept
        for concept_name in tqdm(concepts.keys(), desc=f"Layer {layer}"):
            # Tokenize concept name
            token_ids = tokenizer.encode(concept_name, add_special_tokens=False)

            if len(token_ids) == 1:
                # Single token - just get its embedding
                token_id = token_ids[0]
                embedding_np = embedding_weights[token_id].float().cpu().numpy()
            else:
                # Multiple tokens - average their embeddings
                embeddings = embedding_weights[token_ids]
                embedding_np = embeddings.float().mean(dim=0).cpu().numpy()

            # Normalize
            embedding_norm = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)

            # Save
            output_path = layer_dir / f"{concept_name}_centroid.npy"
            np.save(output_path, embedding_norm)

        print(f"✓ Saved {len(concepts)} embeddings to {layer_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate concept name embeddings for text detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Model name or path (default: google/gemma-3-4b-pt)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Layers to process (default: 0 1 2 3 4 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lens_packs/gemma-3-4b-pt_sumo-wordnet-v1"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    generate_concept_embeddings(
        model_name=args.model,
        layers=args.layers,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\n" + "="*80)
    print("✓ Concept name embeddings generated successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
