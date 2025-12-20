"""
Diagnose why centroid-based text detection is producing poor results.

Examines actual centroid values, similarity calculations, and compares
with expected behavior for specific problematic cases like:
- "certain" → "doll" (57%)
- "User" → "tomb" (58%)
"""

import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat.monitoring.centroid_detector import CentroidTextDetector

# Configuration
MODEL_NAME = "google/gemma-3-4b-pt"
LENS_PACK_DIR = Path("lens_packs/gemma-3-4b-pt_sumo-wordnet-v1")
TEST_TOKENS = ["certain", "User", "doll", "tomb", "dancing", "fictionaltext"]

def get_token_embedding(token_str: str, tokenizer, model, device: str = "cuda"):
    """Get embedding for a single token."""
    # Tokenize
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) != 1:
        print(f"Warning: '{token_str}' tokenizes to {len(token_ids)} tokens: {token_ids}")
        token_id = token_ids[0]  # Use first token
    else:
        token_id = token_ids[0]

    # Get embedding from model's embedding layer
    with torch.no_grad():
        embedding = model.model.embed_tokens(torch.tensor([[token_id]], device=device))
        embedding_np = embedding[0, 0].cpu().numpy()

    return embedding_np, token_id

def load_all_centroids(layer: int):
    """Load all centroids for a layer."""
    centroids_dir = LENS_PACK_DIR / f"layer{layer}" / "embedding_centroids"
    centroids = {}

    for centroid_file in centroids_dir.glob("*_centroid.npy"):
        concept_name = centroid_file.stem.replace("_centroid", "")
        centroid = np.load(centroid_file)
        centroids[concept_name] = centroid

    return centroids

def compute_similarities(token_embedding: np.ndarray, centroids: dict):
    """Compute cosine similarities between token and all centroids."""
    # Normalize token embedding
    token_norm = token_embedding / (np.linalg.norm(token_embedding) + 1e-8)

    similarities = {}
    for concept_name, centroid in centroids.items():
        # Compute cosine similarity
        similarity = np.dot(token_norm, centroid)
        # Convert to probability
        probability = (similarity + 1.0) / 2.0
        similarities[concept_name] = {
            'similarity': float(similarity),
            'probability': float(probability)
        }

    return similarities

def analyze_token(token_str: str, tokenizer, model, centroids_by_layer, device: str = "cuda"):
    """Analyze a token across all layers."""
    print(f"\n{'='*80}")
    print(f"Analyzing token: '{token_str}'")
    print(f"{'='*80}")

    # Get token embedding
    embedding, token_id = get_token_embedding(token_str, tokenizer, model, device)
    print(f"Token ID: {token_id}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"Embedding mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")

    # Analyze across layers
    for layer in sorted(centroids_by_layer.keys()):
        centroids = centroids_by_layer[layer]
        print(f"\n--- Layer {layer} ({len(centroids)} concepts) ---")

        # Compute similarities
        similarities = compute_similarities(embedding, centroids)

        # Sort by probability (descending)
        sorted_results = sorted(similarities.items(), key=lambda x: x[1]['probability'], reverse=True)

        # Show top 10
        print("\nTop 10 matches:")
        for i, (concept, scores) in enumerate(sorted_results[:10], 1):
            print(f"  {i:2d}. {concept:25s} | sim={scores['similarity']:+.4f} | prob={scores['probability']:.4f}")

        # Show bottom 10
        print("\nBottom 10 matches:")
        for i, (concept, scores) in enumerate(sorted_results[-10:], 1):
            print(f"  {i:2d}. {concept:25s} | sim={scores['similarity']:+.4f} | prob={scores['probability']:.4f}")

def main():
    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded on {device}")

    # Load centroids for all layers
    print("\nLoading centroids...")
    centroids_by_layer = {}
    for layer in range(6):
        centroids_dir = LENS_PACK_DIR / f"layer{layer}" / "embedding_centroids"
        if centroids_dir.exists():
            centroids = load_all_centroids(layer)
            centroids_by_layer[layer] = centroids
            print(f"  Layer {layer}: {len(centroids)} centroids")

    # Analyze each test token
    for token in TEST_TOKENS:
        analyze_token(token, tokenizer, model, centroids_by_layer, device)

    # Cross-comparison: similarity between token pairs
    print(f"\n{'='*80}")
    print("Cross-token similarity analysis")
    print(f"{'='*80}")

    embeddings = {}
    for token in TEST_TOKENS:
        embedding, _ = get_token_embedding(token, tokenizer, model, device)
        embeddings[token] = embedding / (np.linalg.norm(embedding) + 1e-8)

    print("\nCosine similarity between tokens:")
    print(f"{'':12s}", end='')
    for token in TEST_TOKENS:
        print(f"{token:12s}", end='')
    print()

    for token1 in TEST_TOKENS:
        print(f"{token1:12s}", end='')
        for token2 in TEST_TOKENS:
            similarity = np.dot(embeddings[token1], embeddings[token2])
            print(f"{similarity:12.4f}", end='')
        print()

if __name__ == "__main__":
    main()
