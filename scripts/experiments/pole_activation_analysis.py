#!/usr/bin/env python3
"""
Analyze activation space separation between simplex poles.

Key question: Do positive, negative, and neutral pole prompts
actually occupy different regions of activation space?
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Test prompts for threat_perception simplex
THREAT_PROMPTS = {
    "negative": [  # alarm
        "I'm terrified something terrible is about to happen",
        "There's imminent danger and I need to escape",
        "This situation feels extremely threatening",
        "I sense a dangerous threat approaching",
        "Everything feels unsafe and scary",
    ],
    "neutral": [  # vigilance
        "I'm paying attention to my surroundings",
        "I notice what's happening around me",
        "I'm aware and observant of the situation",
        "I'm staying alert but calm",
        "I'm watchful without being worried",
    ],
    "positive": [  # safety
        "I feel completely safe and secure here",
        "Everything is calm and there's nothing to worry about",
        "This is a peaceful, protected environment",
        "I'm relaxed and feel no danger",
        "I trust that everything is fine",
    ],
}

# Unrelated prompts for baseline comparison
UNRELATED = [
    "What is quantum mechanics?",
    "The sky is blue today",
    "Please pass the salt",
    "2 + 2 equals 4",
    "Python is a programming language",
]


def extract_activations(
    model, tokenizer, prompts: List[str], layer_idx: int = 12
) -> torch.Tensor:
    """Extract activations from model."""
    device = next(model.parameters()).device
    all_acts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            # Last token activation
            act = hidden[0, -1, :].cpu().float()
            all_acts.append(act)

    return torch.stack(all_acts)


def analyze_pole_separation():
    print("=" * 60)
    print("POLE ACTIVATION SPACE ANALYSIS")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"   Model loaded on {device}")

    # Extract activations
    print("\n2. Extracting activations...")

    pole_acts = {}
    for pole, prompts in THREAT_PROMPTS.items():
        acts = extract_activations(model, tokenizer, prompts)
        pole_acts[pole] = acts
        print(f"   {pole}: {acts.shape}")

    unrelated_acts = extract_activations(model, tokenizer, UNRELATED)
    print(f"   unrelated: {unrelated_acts.shape}")

    # Compute centroids
    print("\n3. Computing pole centroids...")
    centroids = {pole: acts.mean(dim=0) for pole, acts in pole_acts.items()}
    unrelated_centroid = unrelated_acts.mean(dim=0)

    # Compute inter-pole distances
    print("\n4. Cosine similarity between pole centroids:")
    poles = list(centroids.keys())

    print("\n   Pole-to-pole similarities:")
    for i, p1 in enumerate(poles):
        for p2 in poles[i+1:]:
            sim = cosine_similarity(
                centroids[p1].unsqueeze(0).numpy(),
                centroids[p2].unsqueeze(0).numpy()
            )[0, 0]
            print(f"   {p1} <-> {p2}: {sim:.4f}")

    print("\n   Pole-to-unrelated similarities:")
    for pole in poles:
        sim = cosine_similarity(
            centroids[pole].unsqueeze(0).numpy(),
            unrelated_centroid.unsqueeze(0).numpy()
        )[0, 0]
        print(f"   {pole} <-> unrelated: {sim:.4f}")

    # Compute within-pole vs between-pole variance
    print("\n5. Variance analysis:")

    all_pole_acts = torch.cat([pole_acts[p] for p in poles])
    all_pole_labels = []
    for i, p in enumerate(poles):
        all_pole_labels.extend([i] * len(pole_acts[p]))
    all_pole_labels = np.array(all_pole_labels)

    # Total variance
    total_var = all_pole_acts.var(dim=0).mean().item()

    # Within-class variance
    within_vars = []
    for pole in poles:
        within_vars.append(pole_acts[pole].var(dim=0).mean().item())
    avg_within_var = np.mean(within_vars)

    # Between-class variance (centroid variance)
    centroid_stack = torch.stack([centroids[p] for p in poles])
    between_var = centroid_stack.var(dim=0).mean().item()

    print(f"   Total variance: {total_var:.4f}")
    print(f"   Within-pole variance (avg): {avg_within_var:.4f}")
    print(f"   Between-pole variance: {between_var:.4f}")
    print(f"   Separation ratio (between/within): {between_var/avg_within_var:.4f}")

    # PCA to visualize
    print("\n6. PCA projection...")
    all_acts = torch.cat([all_pole_acts, unrelated_acts])

    pca = PCA(n_components=3)
    projected = pca.fit_transform(all_acts.numpy())

    print(f"   Explained variance: {pca.explained_variance_ratio_[:3].sum():.2%}")

    # Compute distances in PCA space
    print("\n   Centroid distances in PCA space (first 3 components):")
    pole_projections = {}
    idx = 0
    for pole in poles:
        n = len(pole_acts[pole])
        pole_projections[pole] = projected[idx:idx+n].mean(axis=0)
        idx += n
    unrelated_proj = projected[idx:].mean(axis=0)

    for i, p1 in enumerate(poles):
        for p2 in poles[i+1:]:
            dist = np.linalg.norm(pole_projections[p1] - pole_projections[p2])
            print(f"   {p1} <-> {p2}: {dist:.4f}")

    for pole in poles:
        dist = np.linalg.norm(pole_projections[pole] - unrelated_proj)
        print(f"   {pole} <-> unrelated: {dist:.4f}")

    # Key insight
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)

    pole_sims = []
    for i, p1 in enumerate(poles):
        for p2 in poles[i+1:]:
            sim = cosine_similarity(
                centroids[p1].unsqueeze(0).numpy(),
                centroids[p2].unsqueeze(0).numpy()
            )[0, 0]
            pole_sims.append(sim)

    unrelated_sims = []
    for pole in poles:
        sim = cosine_similarity(
            centroids[pole].unsqueeze(0).numpy(),
            unrelated_centroid.unsqueeze(0).numpy()
        )[0, 0]
        unrelated_sims.append(sim)

    avg_pole_sim = np.mean(pole_sims)
    avg_unrelated_sim = np.mean(unrelated_sims)

    if avg_pole_sim > avg_unrelated_sim:
        print(f"\n⚠️  Poles are MORE similar to each other ({avg_pole_sim:.4f})")
        print(f"   than to unrelated content ({avg_unrelated_sim:.4f})")
        print("\n   This explains why classifiers learn 'emotional content'")
        print("   rather than pole-specific patterns!")
        print("\n   The hard negatives (other poles) are in the SAME")
        print("   semantic neighborhood as positives.")
    else:
        print(f"\n✓ Poles are reasonably separated from each other")


if __name__ == "__main__":
    analyze_pole_separation()
