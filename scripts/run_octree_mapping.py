#!/usr/bin/env python3
"""
Experiment A: Octree + University Mapping

Tests whether semantic concept structure aligns with geometric activation space.

Process:
1. Build activation octree from diverse text samples
2. Generate toy university taxonomy (single domain)
3. Map concepts to octree cells
4. Measure coverage and alignment

Usage:
    # Full experiment
    python scripts/run_octree_mapping.py

    # Just build octree
    python scripts/run_octree_mapping.py --octree-only

    # Just map existing university to existing octree
    python scripts/run_octree_mapping.py --map-only

    # Use specific model
    python scripts/run_octree_mapping.py --model google/gemma-3-4b-pt
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat.octree import OctreeBuilder, OctreeConfig, ActivationOctree, OctreeQuery

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_id: str, device: str = "cuda"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    return model, tokenizer


def collect_diverse_texts(n_samples: int = 5000) -> list[str]:
    """
    Generate diverse text samples for activation collection.

    In production, this would load from actual corpora.
    For now, generates synthetic diversity.
    """
    import random

    # Prompt templates
    templates = [
        "Explain the concept of {topic} in simple terms.",
        "What is {topic} and why does it matter?",
        "Describe the relationship between {topic} and {topic2}.",
        "How does {topic} work in practice?",
        "The key principles of {topic} include",
        "One common misconception about {topic} is",
        "In the field of {domain}, {topic} refers to",
        "Compare and contrast {topic} with {topic2}.",
        "The history of {topic} begins with",
        "Recent developments in {topic} have shown",
    ]

    # Topics by domain
    domains = {
        "science": [
            "photosynthesis", "evolution", "entropy", "quantum mechanics",
            "relativity", "electromagnetism", "thermodynamics", "genetics",
            "cell biology", "chemistry", "astronomy", "geology",
        ],
        "philosophy": [
            "consciousness", "free will", "ethics", "epistemology",
            "metaphysics", "logic", "aesthetics", "existentialism",
            "determinism", "moral relativism", "phenomenology",
        ],
        "computing": [
            "algorithms", "data structures", "machine learning",
            "neural networks", "databases", "operating systems",
            "cryptography", "distributed systems", "compilers",
        ],
        "mathematics": [
            "calculus", "linear algebra", "probability", "statistics",
            "number theory", "topology", "group theory", "analysis",
        ],
        "social": [
            "democracy", "economics", "psychology", "sociology",
            "anthropology", "linguistics", "history", "politics",
        ],
    }

    all_topics = []
    for topics in domains.values():
        all_topics.extend(topics)

    samples = []
    for _ in range(n_samples):
        template = random.choice(templates)
        topic = random.choice(all_topics)
        topic2 = random.choice(all_topics)
        domain = random.choice(list(domains.keys()))

        text = template.format(topic=topic, topic2=topic2, domain=domain)
        samples.append(text)

    return samples


def build_octree(
    model,
    tokenizer,
    output_dir: Path,
    n_samples: int = 5000,
    pca_dims: int = 64,
    max_depth: int = 8,
    min_samples: int = 20,
    layers: list[int] = None,
) -> ActivationOctree:
    """Build octree from model activations."""
    if layers is None:
        # Default: early, mid, late layers for a 32-layer model
        n_layers = model.config.num_hidden_layers
        layers = [n_layers // 4, n_layers // 2, -1]

    logger.info(f"Building octree with layers {layers}, PCA to {pca_dims} dims")

    config = OctreeConfig(
        max_depth=max_depth,
        min_samples=min_samples,
        pca_dimensions=pca_dims,
        layers=layers,
    )

    builder = OctreeBuilder(config)

    # Collect activations
    texts = collect_diverse_texts(n_samples)
    logger.info(f"Collected {len(texts)} diverse text samples")

    builder.collect_from_texts(model, tokenizer, texts, batch_size=8)

    # Build octree
    octree = builder.build()

    # Save
    octree_path = output_dir / "octree"
    octree.save(octree_path)
    logger.info(f"Saved octree to {octree_path}")

    # Save PCA
    pca_path = output_dir / "pca.npz"
    builder.save_pca(pca_path)
    logger.info(f"Saved PCA to {pca_path}")

    return octree


def generate_toy_university(
    output_dir: Path,
    domain: str = "Epistemology",
    max_depth: int = 4,
) -> dict:
    """
    Generate a toy university taxonomy for a single domain.

    Uses Claude API via the university builder or falls back to
    a pre-defined structure for testing.
    """
    university_path = output_dir / "university.json"

    # Check if we have a pre-generated one
    if university_path.exists():
        logger.info(f"Loading existing university from {university_path}")
        with open(university_path) as f:
            return json.load(f)

    # For testing, use a simple pre-defined structure
    # In production, this would call the university builder
    logger.info(f"Generating toy university for domain: {domain}")

    taxonomy = {
        "id": "root",
        "label": domain,
        "level": 1,
        "children": [
            {
                "id": "root.sources",
                "label": "SourcesOfKnowledge",
                "level": 2,
                "children": [
                    {"id": "root.sources.perception", "label": "Perception", "level": 3, "children": [
                        {"id": "root.sources.perception.vision", "label": "VisualPerception", "level": 4},
                        {"id": "root.sources.perception.auditory", "label": "AuditoryPerception", "level": 4},
                    ]},
                    {"id": "root.sources.reason", "label": "Reason", "level": 3, "children": [
                        {"id": "root.sources.reason.deduction", "label": "DeductiveReasoning", "level": 4},
                        {"id": "root.sources.reason.induction", "label": "InductiveReasoning", "level": 4},
                    ]},
                    {"id": "root.sources.testimony", "label": "Testimony", "level": 3, "children": [
                        {"id": "root.sources.testimony.expert", "label": "ExpertTestimony", "level": 4},
                        {"id": "root.sources.testimony.firsthand", "label": "FirsthandTestimony", "level": 4},
                    ]},
                ],
            },
            {
                "id": "root.justification",
                "label": "Justification",
                "level": 2,
                "children": [
                    {"id": "root.justification.foundationalism", "label": "Foundationalism", "level": 3, "children": [
                        {"id": "root.justification.foundationalism.classical", "label": "ClassicalFoundationalism", "level": 4},
                        {"id": "root.justification.foundationalism.modest", "label": "ModestFoundationalism", "level": 4},
                    ]},
                    {"id": "root.justification.coherentism", "label": "Coherentism", "level": 3, "children": [
                        {"id": "root.justification.coherentism.linear", "label": "LinearCoherentism", "level": 4},
                        {"id": "root.justification.coherentism.holistic", "label": "HolisticCoherentism", "level": 4},
                    ]},
                    {"id": "root.justification.reliabilism", "label": "Reliabilism", "level": 3, "children": [
                        {"id": "root.justification.reliabilism.process", "label": "ProcessReliabilism", "level": 4},
                        {"id": "root.justification.reliabilism.virtue", "label": "VirtueReliabilism", "level": 4},
                    ]},
                ],
            },
            {
                "id": "root.skepticism",
                "label": "Skepticism",
                "level": 2,
                "children": [
                    {"id": "root.skepticism.global", "label": "GlobalSkepticism", "level": 3, "children": [
                        {"id": "root.skepticism.global.cartesian", "label": "CartesianSkepticism", "level": 4},
                        {"id": "root.skepticism.global.external", "label": "ExternalWorldSkepticism", "level": 4},
                    ]},
                    {"id": "root.skepticism.local", "label": "LocalSkepticism", "level": 3, "children": [
                        {"id": "root.skepticism.local.moral", "label": "MoralSkepticism", "level": 4},
                        {"id": "root.skepticism.local.religious", "label": "ReligiousSkepticism", "level": 4},
                    ]},
                ],
            },
        ],
    }

    # Save
    with open(university_path, "w") as f:
        json.dump(taxonomy, f, indent=2)

    logger.info(f"Saved taxonomy to {university_path}")
    return taxonomy


def extract_leaf_concepts(taxonomy: dict) -> list[dict]:
    """Extract all leaf concepts from taxonomy."""
    leaves = []

    def traverse(node, path=None):
        if path is None:
            path = []

        current_path = path + [node["label"]]

        if "children" not in node or not node["children"]:
            leaves.append({
                "id": node["id"],
                "label": node["label"],
                "path": current_path,
                "level": node["level"],
            })
        else:
            for child in node["children"]:
                traverse(child, current_path)

    traverse(taxonomy)
    return leaves


def generate_concept_exemplars(concept: dict, n_exemplars: int = 10) -> list[str]:
    """Generate example texts for a concept."""
    label = concept["label"]
    path = " > ".join(concept["path"])

    templates = [
        f"Explain {label} in the context of {path}.",
        f"What is {label}? Provide a detailed explanation.",
        f"Describe the key features of {label}.",
        f"How does {label} relate to other concepts in epistemology?",
        f"Give an example of {label} in practice.",
        f"The concept of {label} is important because",
        f"Critics of {label} argue that",
        f"Proponents of {label} believe that",
        f"In philosophy, {label} refers to",
        f"The historical development of {label} shows",
    ]

    import random
    return random.sample(templates, min(n_exemplars, len(templates)))


def map_concepts_to_octree(
    model,
    tokenizer,
    octree: ActivationOctree,
    concepts: list[dict],
    builder: OctreeBuilder,
    exemplars_per_concept: int = 10,
) -> dict:
    """
    Map concepts to octree cells by running exemplars through model.

    Returns mapping of concept -> list of cells.
    """
    logger.info(f"Mapping {len(concepts)} concepts to octree")

    concept_to_cells = {}
    cell_to_concepts = {}

    for concept in concepts:
        label = concept["label"]
        exemplars = generate_concept_exemplars(concept, exemplars_per_concept)

        # Collect activations for exemplars
        temp_builder = OctreeBuilder(builder.config)
        temp_builder.pca_components = builder.pca_components
        temp_builder.pca_mean = builder.pca_mean

        temp_builder.collect_from_texts(
            model, tokenizer, exemplars,
            batch_size=4, show_progress=False
        )

        if not temp_builder.activations:
            logger.warning(f"No activations collected for {label}")
            continue

        # Apply PCA and query octree
        activations = np.stack(temp_builder.activations)
        reduced = temp_builder._apply_pca(activations)

        cells = octree.query_batch(reduced)

        # Count cell occurrences
        cell_counts = {}
        for cell in cells:
            cell_counts[cell.bits] = cell_counts.get(cell.bits, 0) + 1

        # Record mapping
        concept_to_cells[label] = list(cell_counts.keys())

        # Update cell -> concepts
        for cell_bits in cell_counts:
            if cell_bits not in cell_to_concepts:
                cell_to_concepts[cell_bits] = []
            cell_to_concepts[cell_bits].append(label)

            # Also update the octree node
            node = octree.get_node(CellAddress(cell_bits))
            if node and label not in node.concept_labels:
                node.concept_labels.append(label)

        logger.info(f"  {label}: mapped to {len(cell_counts)} cells")

    return {
        "concept_to_cells": concept_to_cells,
        "cell_to_concepts": cell_to_concepts,
    }


def analyze_alignment(octree: ActivationOctree, mapping: dict) -> dict:
    """Analyze how well semantic structure aligns with geometric structure."""
    query = OctreeQuery(octree)
    coverage = query.analyze_coverage()

    # Additional analysis
    concept_to_cells = mapping["concept_to_cells"]

    # How localized are concepts?
    localization_scores = []
    for concept, cells in concept_to_cells.items():
        if len(cells) > 0:
            # Score = 1 / n_cells (more localized = higher score)
            localization_scores.append(1.0 / len(cells))

    # How much overlap between concepts?
    cell_to_concepts = mapping["cell_to_concepts"]
    cells_with_overlap = sum(1 for c in cell_to_concepts.values() if len(c) > 1)

    results = {
        "coverage": coverage.to_dict(),
        "n_concepts_mapped": len(concept_to_cells),
        "mean_localization": np.mean(localization_scores) if localization_scores else 0,
        "std_localization": np.std(localization_scores) if localization_scores else 0,
        "cells_with_overlap": cells_with_overlap,
        "cells_with_single_concept": coverage.unique_cells,
        "cells_unmapped": coverage.unmapped_leaves,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Octree + University Mapping Experiment")
    parser.add_argument("--model", default="google/gemma-3-4b-pt", help="Model to use")
    parser.add_argument("--output-dir", type=Path, default=Path("results/octree_mapping"))
    parser.add_argument("--n-samples", type=int, default=5000, help="Activation samples to collect")
    parser.add_argument("--pca-dims", type=int, default=64, help="PCA dimensions")
    parser.add_argument("--max-depth", type=int, default=8, help="Max octree depth")
    parser.add_argument("--min-samples", type=int, default=20, help="Min samples per leaf")
    parser.add_argument("--domain", default="Epistemology", help="University domain")
    parser.add_argument("--octree-only", action="store_true", help="Only build octree")
    parser.add_argument("--map-only", action="store_true", help="Only map existing octree")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Get model layer count
    n_layers = model.config.num_hidden_layers
    layers = [n_layers // 4, n_layers // 2, -1]
    logger.info(f"Using layers: {layers} (of {n_layers} total)")

    # Build or load octree
    octree_path = args.output_dir / "octree"
    pca_path = args.output_dir / "pca.npz"

    if args.map_only and octree_path.exists():
        logger.info("Loading existing octree")
        octree = ActivationOctree.load(octree_path)
        builder = OctreeBuilder(OctreeConfig(pca_dimensions=args.pca_dims, layers=layers))
        builder.load_pca(pca_path)
    else:
        octree = build_octree(
            model, tokenizer, args.output_dir,
            n_samples=args.n_samples,
            pca_dims=args.pca_dims,
            max_depth=args.max_depth,
            min_samples=args.min_samples,
            layers=layers,
        )
        builder = OctreeBuilder(OctreeConfig(pca_dimensions=args.pca_dims, layers=layers))
        builder.load_pca(pca_path)

    logger.info(f"Octree: {octree}")

    if args.octree_only:
        logger.info("Octree-only mode, stopping here")
        return

    # Generate or load university
    taxonomy = generate_toy_university(args.output_dir, args.domain)
    concepts = extract_leaf_concepts(taxonomy)
    logger.info(f"University has {len(concepts)} leaf concepts")

    # Map concepts to octree
    from src.hat.octree.tree import CellAddress
    mapping = map_concepts_to_octree(
        model, tokenizer, octree, concepts, builder,
        exemplars_per_concept=10,
    )

    # Save updated octree with concept labels
    octree.save(octree_path)

    # Analyze alignment
    results = analyze_alignment(octree, mapping)

    # Save results
    results_path = args.output_dir / "alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ALIGNMENT RESULTS")
    print("=" * 60)
    print(f"\nCoverage:")
    print(f"  Leaves: {results['coverage']['mapped_leaves']} / {results['coverage']['total_leaves']} "
          f"({results['coverage']['coverage_by_leaves']:.1%})")
    print(f"  Samples: {results['coverage']['mapped_samples']} / {results['coverage']['total_samples']} "
          f"({results['coverage']['coverage_by_samples']:.1%})")
    print(f"\nLocalization:")
    print(f"  Mean: {results['mean_localization']:.3f} (higher = more localized)")
    print(f"  Std: {results['std_localization']:.3f}")
    print(f"\nOverlap:")
    print(f"  Cells with multiple concepts: {results['cells_with_overlap']}")
    print(f"  Cells with single concept: {results['cells_with_single_concept']}")
    print(f"  Unmapped cells: {results['cells_unmapped']}")
    print("\n" + "=" * 60)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
