#!/usr/bin/env python3
"""
Phase 6: Subspace Removal Matrix

Goal: Remove shared "definitional prompt structure" and generic generation machinery
      from steering vectors to improve steering effectiveness.

Approach:
1. Extract concept vectors from all 10 concepts (Phase 4 trained classifiers)
2. Apply subspace removal methods:
   - none: Baseline (no removal)
   - mean_subtraction: Remove mean vector across all concepts
   - pca_1: Remove first principal component
   - pca_5: Remove first 5 principal components
   - pca_10: Remove first 10 principal components
3. Re-run Phase 5 evaluation with clean vectors
4. Compare: working range, coherence, Δ correlation by method

Expected: Clean vectors expand working range (±0.5 → ±1.0+) and improve coherence.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_concept_vector(
    model,
    tokenizer,
    concept: str,
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract concept vector from model activations.

    This matches Phase 5's approach: generate text about the concept
    and average the hidden states to get the concept direction.
    """
    concept_prompt = f"What is {concept}?"
    inputs = tokenizer(concept_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        activations = []
        for step_states in outputs.hidden_states:
            if layer_idx == -1:
                last_layer = step_states[-1]
            else:
                last_layer = step_states[layer_idx]

            act = last_layer[0, -1, :]
            activations.append(act.cpu().numpy())

        # Average across generation steps
        concept_vector = np.stack(activations).mean(axis=0)

        # Normalize
        concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

    return concept_vector


def apply_subspace_removal(
    vectors: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Apply subspace removal to concept vectors.

    Args:
        vectors: (n_concepts, hidden_dim) array of concept vectors
        method: Removal method ['none', 'mean_subtraction', 'pca_1', 'pca_5', 'pca_10']

    Returns:
        clean_vectors: (n_concepts, hidden_dim) with shared subspace removed
    """
    if method == "none":
        return vectors

    elif method == "mean_subtraction":
        # Remove mean vector (shared centroid)
        mean_vec = vectors.mean(axis=0)
        clean_vectors = vectors - mean_vec

    elif method.startswith("pca_"):
        # Remove first N principal components
        n_components = int(method.split("_")[1])

        # Can't use more components than min(n_samples, n_features)
        max_components = min(vectors.shape[0], vectors.shape[1])
        if n_components > max_components:
            logger.warning(f"Requested {n_components} components but only {max_components} available. Using {max_components}.")
            n_components = max_components

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(vectors)

        # Project onto principal components and subtract
        projected = pca.transform(vectors)
        reconstruction = pca.inverse_transform(projected)
        clean_vectors = vectors - reconstruction

        logger.info(f"PCA explained variance (first {n_components} components): "
                   f"{pca.explained_variance_ratio_.sum():.3f}")

    else:
        raise ValueError(f"Unknown removal method: {method}")

    # Re-normalize
    norms = np.linalg.norm(clean_vectors, axis=1, keepdims=True)
    clean_vectors = clean_vectors / (norms + 1e-8)

    return clean_vectors


def create_steering_hook(concept_vector: np.ndarray, strength: float, device: str):
    """Create hook for steering generation - exactly matches Phase 5."""
    concept_tensor = torch.tensor(concept_vector, dtype=torch.float32).to(device)

    def hook(module, input, output):
        """Project out concept vector from hidden states."""
        hidden_states = output[0]
        # Project onto concept vector and add scaled projection (positive = amplify)
        projection = (hidden_states @ concept_tensor.unsqueeze(-1)) * concept_tensor
        steered = hidden_states + strength * projection
        return (steered,)

    return hook


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    concept_vector: np.ndarray,
    strength: float,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> str:
    """Generate text with concept steering applied."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Register steering hook
    if layer_idx == -1:
        target_layer = model.model.language_model.layers[-1]
    else:
        target_layer = model.model.language_model.layers[layer_idx]

    hook = target_layer.register_forward_hook(
        create_steering_hook(concept_vector, strength, device)
    )

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt
            generated_text = generated_text[len(prompt):].strip()
    finally:
        hook.remove()

    return generated_text


def compute_semantic_shift(
    text: str,
    core_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    embedding_model: SentenceTransformer
) -> float:
    """
    Compute semantic shift: Δ = cos(text, core) - cos(text, neg)

    Higher Δ means text is more aligned with core concept.
    """
    text_embedding = embedding_model.encode([text], convert_to_numpy=True)[0]

    cos_core = np.dot(text_embedding, core_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(core_centroid) + 1e-8
    )

    cos_neg = np.dot(text_embedding, neg_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(neg_centroid) + 1e-8
    )

    return float(cos_core - cos_neg)


def create_centroids(
    concept: str,
    embedding_model: SentenceTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    """Create core and negative centroids for semantic shift measurement."""
    # Core: definitional prompts
    core_texts = [
        f"What is {concept}?",
        f"{concept} is defined as",
        f"The meaning of {concept} is",
        f"{concept} refers to"
    ]

    # Negative: distant concepts (manually selected)
    negative_concepts = {
        "person": "mineral",
        "change": "building",
        "bird genus": "chemical compound",
        "mammal genus": "plant structure",
        "herb": "animal behavior",
        "asterid dicot genus": "mechanical device",
        "shrub": "social event",
        "rosid dicot genus": "weather phenomenon",
        "fish genus": "mathematical concept",
        "animal order": "food preparation"
    }

    neg_concept = negative_concepts.get(concept, "abstract concept")
    neg_texts = [
        f"What is {neg_concept}?",
        f"{neg_concept} is defined as",
        f"The meaning of {neg_concept} is",
        f"{neg_concept} refers to"
    ]

    core_embeddings = embedding_model.encode(core_texts, convert_to_numpy=True)
    neg_embeddings = embedding_model.encode(neg_texts, convert_to_numpy=True)

    core_centroid = core_embeddings.mean(axis=0)
    neg_centroid = neg_embeddings.mean(axis=0)

    return core_centroid, neg_centroid


def is_coherent(text: str) -> bool:
    """
    Check if generated text is coherent.

    Degraded outputs include:
    - Empty or very short (<10 chars)
    - Repetitive tokens (same 3-gram repeated 3+ times)
    - Excessive punctuation (>30% of chars)
    - Code snippets (contains common code patterns)
    """
    if len(text) < 10:
        return False

    # Check for repetitions
    words = text.lower().split()
    if len(words) >= 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) > 0:
            max_count = max(trigrams.count(tg) for tg in set(trigrams))
            if max_count >= 3:
                return False

    # Check excessive punctuation
    punct_chars = sum(1 for c in text if c in "!?.,;:()[]{}")
    if len(text) > 0 and punct_chars / len(text) > 0.3:
        return False

    # Check for code patterns
    code_patterns = ["def ", "class ", "import ", "#include", "function ", "<?", "?>", "```"]
    if any(pattern in text for pattern in code_patterns):
        return False

    return True


def evaluate_steering(
    model,
    tokenizer,
    embedding_model: SentenceTransformer,
    concepts: List[str],
    concept_vectors: Dict[str, np.ndarray],
    removal_method: str,
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate steering effectiveness with given concept vectors.

    This is essentially Phase 5 evaluation but with potentially clean vectors.
    """
    # Use Phase 5's working range - ±1.0 causes CUDA errors
    strengths = [-0.5, -0.25, 0.0, 0.25, 0.5]
    prompts_per_concept = [
        "Tell me about {concept}.",
        "Explain {concept}.",
        "What is {concept}?"
    ]

    results = []

    for concept in concepts:
        logger.info(f"Evaluating {concept} with {removal_method}...")

        concept_vector = concept_vectors[concept]
        core_centroid, neg_centroid = create_centroids(concept, embedding_model)

        for prompt_template in prompts_per_concept:
            prompt = prompt_template.format(concept=concept)

            for strength in strengths:
                # Generate with steering (may fail at extreme strengths)
                try:
                    generated_text = generate_with_steering(
                        model, tokenizer, prompt, concept_vector,
                        strength, layer_idx=-1, max_new_tokens=50, device=device
                    )

                    # Compute metrics
                    delta = compute_semantic_shift(
                        generated_text, core_centroid, neg_centroid, embedding_model
                    )
                    coherent = is_coherent(generated_text)
                    collapsed = False

                except (RuntimeError, ValueError) as e:
                    # Model collapse - steering too extreme
                    logger.warning(f"{concept} @ {strength:+.2f}: Model collapse ({type(e).__name__}): {str(e)[:100]}")

                    # Try to clear CUDA error state (may also fail if CUDA is corrupted)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass  # CUDA state is too corrupted to recover

                    generated_text = "[MODEL COLLAPSE]"
                    delta = 0.0
                    coherent = False
                    collapsed = True

                results.append({
                    "concept": concept,
                    "prompt": prompt,
                    "steering_strength": strength,
                    "removal_method": removal_method,
                    "generated_text": generated_text,
                    "delta": delta,
                    "coherent": coherent,
                    "collapsed": collapsed
                })

                if not collapsed:
                    logger.debug(f"{concept} @ {strength:+.2f}: Δ={delta:.3f}, "
                               f"coherent={coherent}, text='{generated_text[:60]}...'")

    # Compute aggregate statistics
    stats_by_strength = {}
    for strength in strengths:
        strength_results = [r for r in results if r["steering_strength"] == strength]

        deltas = [r["delta"] for r in strength_results]
        coherent_count = sum(r["coherent"] for r in strength_results)
        collapsed_count = sum(r.get("collapsed", False) for r in strength_results)

        stats_by_strength[strength] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "coherence_rate": coherent_count / len(strength_results),
            "collapse_rate": collapsed_count / len(strength_results),
            "n_samples": len(strength_results)
        }

    # Determine working range (>85% coherence)
    working_range = []
    for strength in sorted(strengths):
        if stats_by_strength[strength]["coherence_rate"] >= 0.85:
            working_range.append(strength)

    summary = {
        "removal_method": removal_method,
        "n_concepts": len(concepts),
        "n_samples": len(results),
        "stats_by_strength": stats_by_strength,
        "working_range": working_range,
        "max_working_strength": max(abs(s) for s in working_range) if working_range else 0.0
    }

    return {
        "summary": summary,
        "all_results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Subspace Removal Matrix")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--concepts", nargs="+", default=[
        "person", "change", "bird genus", "mammal genus", "herb",
        "asterid dicot genus", "shrub", "rosid dicot genus", "fish genus", "animal order"
    ])
    parser.add_argument("--removal-methods", nargs="+",
                       default=["none", "mean_subtraction", "pca_1", "pca_5", "pca_10"])
    parser.add_argument("--output-dir", type=Path, default="results/phase_6_subspace_removal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    file_handler = logging.FileHandler(args.output_dir / "phase_6.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("="*80)
    logger.info("Phase 6: Subspace Removal Matrix")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Concepts: {args.concepts}")
    logger.info(f"Removal methods: {args.removal_methods}")
    logger.info(f"Output: {args.output_dir}")

    start_time = time.time()

    # Load models
    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding_model = SentenceTransformer(args.embedding_model)

    # Extract concept vectors (baseline, before removal)
    logger.info("Extracting concept vectors...")
    raw_vectors = {}
    for concept in args.concepts:
        logger.info(f"  Extracting: {concept}")
        vector = extract_concept_vector(model, tokenizer, concept, device=args.device)

        # Check for NaN values
        if np.any(np.isnan(vector)):
            logger.error(f"  ERROR: {concept} vector contains NaN values!")
            raise ValueError(f"Concept vector for '{concept}' contains NaN values")

        raw_vectors[concept] = vector
        logger.info(f"    Vector norm: {np.linalg.norm(vector):.4f}, min: {vector.min():.4f}, max: {vector.max():.4f}")

    # Convert to matrix for subspace removal
    concept_matrix = np.stack([raw_vectors[c] for c in args.concepts])
    logger.info(f"Concept matrix shape: {concept_matrix.shape}")

    # Test each removal method
    all_method_results = {}

    for method in args.removal_methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing removal method: {method}")
        logger.info(f"{'='*80}")

        # Apply subspace removal
        clean_vectors_matrix = apply_subspace_removal(concept_matrix, method)

        # Convert back to dict
        clean_vectors = {
            concept: clean_vectors_matrix[i]
            for i, concept in enumerate(args.concepts)
        }

        # Evaluate steering with clean vectors
        results = evaluate_steering(
            model, tokenizer, embedding_model,
            args.concepts, clean_vectors, method,
            args.output_dir, device=args.device
        )

        all_method_results[method] = results

        # Log summary
        summary = results["summary"]
        logger.info(f"\nSummary for {method}:")
        logger.info(f"  Working range: {summary['working_range']}")
        logger.info(f"  Max working strength: ±{summary['max_working_strength']}")
        logger.info(f"  Coherence by strength:")
        for strength in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            stats = summary["stats_by_strength"][strength]
            logger.info(f"    {strength:+.2f}: {stats['coherence_rate']:.1%} coherent, "
                       f"Δ={stats['mean_delta']:.3f}±{stats['std_delta']:.3f}")

    # Save results
    logger.info(f"\nSaving results to {args.output_dir}/...")

    for method, results in all_method_results.items():
        method_file = args.output_dir / f"{method}.json"
        with open(method_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved: {method_file}")

    # Create comparison report
    comparison = {
        "config": {
            "model": args.model,
            "concepts": args.concepts,
            "removal_methods": args.removal_methods,
            "n_samples_per_method": len(args.concepts) * 3 * 7  # concepts × prompts × strengths
        },
        "comparison": {}
    }

    for method in args.removal_methods:
        summary = all_method_results[method]["summary"]
        comparison["comparison"][method] = {
            "working_range": summary["working_range"],
            "max_working_strength": summary["max_working_strength"],
            "coherence_by_strength": {
                str(s): summary["stats_by_strength"][s]["coherence_rate"]
                for s in [-0.5, -0.25, 0.0, 0.25, 0.5]
            },
            "delta_by_strength": {
                str(s): {
                    "mean": summary["stats_by_strength"][s]["mean_delta"],
                    "std": summary["stats_by_strength"][s]["std_delta"]
                }
                for s in [-0.5, -0.25, 0.0, 0.25, 0.5]
            }
        }

    comparison_file = args.output_dir / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"  Saved: {comparison_file}")

    elapsed = time.time() - start_time
    logger.info(f"\nPhase 6 complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
