"""
Generate concept name embeddings for direct token-to-concept comparison.

Instead of computing centroids from training samples, this generates
embeddings directly from the concept names themselves.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_concept_names_from_existing(base_dir: Path, layer: int) -> list:
    """Load concept names from existing centroid files."""
    centroid_dir = base_dir / f"layer{layer}" / "embedding_centroids"
    if not centroid_dir.exists():
        return []

    concept_names = []
    for centroid_file in centroid_dir.glob("*_centroid.npy"):
        concept_name = centroid_file.stem.replace("_centroid", "")
        if concept_name != "centroids_metadata":  # Skip metadata file
            concept_names.append(concept_name)

    return concept_names

def generate_concept_embeddings(
    model_name: str,
    layers: list[int],
    output_dir: Path,
    device: str = "cuda",
):
    """Generate embeddings for all concept names."""

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded on {device}")

    for layer in layers:
        print(f"\n{'='*80}")
        print(f"Layer {layer}")
        print(f"{'='*80}")

        # Load concepts from existing centroids
        concept_names = load_concept_names_from_existing(output_dir, layer)
        print(f"Regenerating embeddings for {len(concept_names)} concepts")

        # Create output directory
        layer_dir = output_dir / f"layer{layer}" / "embedding_centroids"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Generate embedding for each concept
        for concept_name in tqdm(concept_names, desc=f"Layer {layer}"):
            # Split camelCase into words (e.g., "RecreationOrExercise" -> "recreation or exercise")
            import re
            concept_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', concept_name).lower()

            # Tokenize concept name
            token_ids = tokenizer.encode(concept_text, add_special_tokens=False)

            if len(token_ids) == 1:
                # Single token - just get its embedding
                token_id = token_ids[0]
                with torch.no_grad():
                    # Get embedding layer - Gemma3 uses different attribute name
                    embed_layer = model.get_input_embeddings()
                    embedding = embed_layer(torch.tensor([[token_id]], device=device))
                    embedding_np = embedding[0, 0].cpu().numpy()
            else:
                # Multiple tokens - average their embeddings
                with torch.no_grad():
                    embed_layer = model.get_input_embeddings()
                    embedding = embed_layer(torch.tensor([token_ids], device=device))
                    embedding_np = embedding[0].mean(dim=0).cpu().numpy()

            # Normalize
            embedding_norm = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)

            # Save
            output_path = layer_dir / f"{concept_name}_centroid.npy"
            np.save(output_path, embedding_norm)

        print(f"✓ Saved {len(all_concepts)} embeddings to {layer_dir}")

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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect)",
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
