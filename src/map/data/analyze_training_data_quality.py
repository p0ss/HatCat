#!/usr/bin/env python3
"""
Training Data Quality Analysis

Evaluates whether our training data quality matches our lens accuracy targets.
Uses an LLM judge to assess:
1. Topic inference: Can the judge guess what concept a response is about?
2. Relevance rating: How well does the response relate to the intended concept?

Then correlates these ratings with lens training outcomes.

Usage:
    python scripts/analyze_training_data_quality.py \
        --concept-pack sumo-wordnet-v4 \
        --model swiss-ai/Apertus-8B-2509 \
        --layer 3 \
        --samples-per-concept 20 \
        --output-dir results/training_data_quality/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.sumo_data_generation import (
    build_sumo_negative_pool,
    create_sumo_training_dataset,
)


@dataclass
class ConceptProfile:
    """Profile of a concept for quadrant classification."""
    sumo_term: str
    synset_count: int
    sibling_count: int
    quadrant: str
    concept_data: Dict


@dataclass
class SampleAnalysis:
    """Analysis of a single training sample."""
    concept: str
    quadrant: str
    sample_type: str  # 'positive' or 'negative'
    prompt: str
    response: str
    inferred_topic: str
    relevance_rating: int
    relevance_explanation: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze training data quality using LLM judge"
    )
    parser.add_argument('--concept-pack', required=True,
                        help='Concept pack ID (e.g., sumo-wordnet-v4)')
    parser.add_argument('--model', required=True,
                        help='Model to generate training data (e.g., swiss-ai/Apertus-8B-2509)')
    parser.add_argument('--layer', type=int, default=3,
                        help='Layer to analyze (default: 3)')
    parser.add_argument('--samples-per-concept', type=int, default=20,
                        help='Training samples to generate per concept (default: 20)')
    parser.add_argument('--concepts-per-quadrant', type=int, default=3,
                        help='Concepts to sample per quadrant (default: 3)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                        help='Device for model (default: cuda)')
    parser.add_argument('--judge-model', default='claude-3-5-haiku-latest',
                        help='Anthropic model for judging (default: claude-3-5-haiku-latest)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip generation, use existing samples.json')
    parser.add_argument('--skip-judging', action='store_true',
                        help='Skip judging, use existing judgments')
    return parser.parse_args()


def load_concepts(pack_dir: Path, layer: int) -> List[Dict]:
    """Load concepts from a layer file."""
    layer_path = pack_dir / "hierarchy" / f"layer{layer}.json"
    with open(layer_path) as f:
        data = json.load(f)
    return data['concepts']


def classify_concepts_by_quadrant(
    concepts: List[Dict],
    concepts_per_quadrant: int = 3,
) -> Dict[str, List[ConceptProfile]]:
    """
    Classify concepts into quadrants based on synset count and sibling count.

    Quadrants:
    A: Low positives (1-2 synsets), Low negatives (0-3 siblings)
    B: Low positives (1-2 synsets), High negatives (10+ siblings)
    C: High positives (5+ synsets), High negatives (10+ siblings)
    D: High positives (5+ synsets), Low negatives (0-3 siblings)
    """
    # Count siblings per parent
    parent_children = defaultdict(list)
    for c in concepts:
        for parent in c.get('parent_concepts', []):
            parent_children[parent].append(c['sumo_term'])

    # Build profiles
    profiles = []
    for c in concepts:
        synset_count = c.get('synset_count', len(c.get('synsets', [])))

        sibling_count = 0
        for parent in c.get('parent_concepts', []):
            sibling_count += len(parent_children[parent]) - 1

        # Classify quadrant
        low_pos = synset_count <= 2
        high_pos = synset_count >= 5
        low_neg = sibling_count <= 3
        high_neg = sibling_count >= 10

        if low_pos and low_neg:
            quadrant = 'A'
        elif low_pos and high_neg:
            quadrant = 'B'
        elif high_pos and high_neg:
            quadrant = 'C'
        elif high_pos and low_neg:
            quadrant = 'D'
        else:
            quadrant = 'X'  # Middle ground, skip

        if quadrant != 'X':
            profiles.append(ConceptProfile(
                sumo_term=c['sumo_term'],
                synset_count=synset_count,
                sibling_count=sibling_count,
                quadrant=quadrant,
                concept_data=c,
            ))

    # Group by quadrant and sample
    quadrants = defaultdict(list)
    for p in profiles:
        quadrants[p.quadrant].append(p)

    # Sample from each quadrant - prefer variety
    selected = {}
    for q in ['A', 'B', 'C', 'D']:
        candidates = quadrants[q]
        if len(candidates) >= concepts_per_quadrant:
            # Sort by synset_count for variety, take spread
            candidates.sort(key=lambda x: (x.synset_count, x.sibling_count))
            step = len(candidates) // concepts_per_quadrant
            selected[q] = [candidates[i * step] for i in range(concepts_per_quadrant)]
        else:
            selected[q] = candidates[:concepts_per_quadrant]

    return selected


def generate_training_samples(
    model,
    tokenizer,
    concept: ConceptProfile,
    all_concepts: List[Dict],
    n_samples: int,
    device: str,
) -> List[Tuple[str, str, str]]:
    """
    Generate training samples for a concept.

    Returns list of (prompt, response, sample_type) tuples.
    """
    concept_map = {c['sumo_term']: c for c in all_concepts}

    # Build negative pool
    negative_pool = build_sumo_negative_pool(
        all_concepts=all_concepts,
        target_concept=concept.concept_data,
        include_siblings=True,
    )

    # Generate dataset
    prompts, labels = create_sumo_training_dataset(
        concept=concept.concept_data,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=n_samples // 2,
        n_negatives=n_samples // 2,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    # Generate responses
    samples = []
    for prompt, label in zip(prompts, labels):
        sample_type = 'positive' if label == 1 else 'negative'

        # Generate response
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        samples.append((prompt, response.strip(), sample_type))

    return samples


def judge_sample_topic(
    client: Anthropic,
    response: str,
    model: str = "claude-3-5-haiku-latest",
) -> str:
    """Ask judge to infer what topic a response is about."""
    message = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""Given this text output from an LLM, what do you think the prompt asked about?
Respond with ONLY a single topic or concept name (1-5 words), nothing else.

Text: "{response[:500]}"

Topic:"""
        }]
    )
    return message.content[0].text.strip()


def judge_sample_relevance(
    client: Anthropic,
    concept: str,
    response: str,
    model: str = "claude-3-5-haiku-latest",
) -> Tuple[int, str]:
    """Ask judge to rate how relevant a response is to a concept."""
    message = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""The prompt asked an LLM to write about: "{concept}"

How strongly does this output actually relate to {concept}?
Rate 1-5:
1 = No relation / completely off-topic
2 = Tangential / mentions related concepts but not the target
3 = Moderate / discusses the general area but not specifically
4 = Strong / clearly about the concept
5 = Excellent / focused, accurate discussion of the concept

Output: "{response[:500]}"

Respond with ONLY the number (1-5) followed by a brief explanation (one sentence).
Format: N - explanation"""
        }]
    )

    text = message.content[0].text.strip()
    try:
        parts = text.split(' - ', 1)
        rating = int(parts[0].strip())
        explanation = parts[1] if len(parts) > 1 else ""
    except (ValueError, IndexError):
        # Try to extract just a number
        for char in text:
            if char.isdigit():
                rating = int(char)
                explanation = text
                break
        else:
            rating = 3  # Default to middle
            explanation = text

    return min(max(rating, 1), 5), explanation


def train_lens_on_samples(
    model,
    tokenizer,
    samples: List[Tuple[str, str, str]],
    device: str,
) -> Dict:
    """
    Train a simple lens on the samples and return metrics.

    Returns dict with f1, precision, recall.
    """
    from training.sumo_classifiers import extract_activations
    import torch.nn as nn
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Prepare data
    prompts = [s[0] for s in samples]
    labels = [1 if s[2] == 'positive' else 0 for s in samples]

    # Extract activations
    activations = extract_activations(model, tokenizer, prompts, device)
    X = torch.tensor(activations, dtype=torch.float32)

    # Handle mismatch between activation count and label count
    n_samples = min(len(activations), len(labels))
    if len(activations) != len(labels):
        print(f"    Warning: activation count ({len(activations)}) != label count ({len(labels)}), using {n_samples}")
        X = X[:n_samples]
        labels = labels[:n_samples]

    y = torch.tensor(labels, dtype=torch.float32)

    # Need at least 4 samples for train/test split
    if len(X) < 4:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Split train/test - use X length for permutation
    n_train = max(2, int(len(X) * 0.7))
    indices = torch.randperm(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

    # Simple lens
    hidden_dim = X.shape[1]
    lens = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)

    optimizer = torch.optim.Adam(lens.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    lens.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = lens(X_train).squeeze()
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    lens.eval()
    with torch.no_grad():
        logits = lens(X_test).squeeze()
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        y_true = y_test.cpu().numpy()

    return {
        'f1': f1_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
    }


def main():
    args = parse_args()

    # Setup paths
    pack_dir = PROJECT_ROOT / "concept_packs" / args.concept_pack
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING DATA QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Concept Pack: {args.concept_pack}")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Samples per concept: {args.samples_per_concept}")
    print(f"Concepts per quadrant: {args.concepts_per_quadrant}")
    print()

    # Load concepts
    concepts = load_concepts(pack_dir, args.layer)
    concept_map = {c['sumo_term']: c for c in concepts}
    print(f"Loaded {len(concepts)} concepts from layer {args.layer}")

    # Classify into quadrants
    quadrants = classify_concepts_by_quadrant(concepts, args.concepts_per_quadrant)

    print("\nSelected concepts by quadrant:")
    for q in ['A', 'B', 'C', 'D']:
        print(f"\n  Quadrant {q}:")
        for p in quadrants[q]:
            print(f"    {p.sumo_term}: {p.synset_count} synsets, {p.sibling_count} siblings")

    # Prepare for data generation
    all_samples = []
    samples_file = output_dir / "samples.json"

    if args.skip_generation and samples_file.exists():
        print("\nLoading existing samples...")
        with open(samples_file) as f:
            all_samples = json.load(f)
    else:
        # Load model for generation
        print("\nLoading model for data generation...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            local_files_only=True,
        )
        model.eval()

        # Generate samples for each concept
        print("\nGenerating training samples...")
        for q in ['A', 'B', 'C', 'D']:
            print(f"\n  Quadrant {q}:")
            for profile in quadrants[q]:
                print(f"    Generating for {profile.sumo_term}...")
                samples = generate_training_samples(
                    model=model,
                    tokenizer=tokenizer,
                    concept=profile,
                    all_concepts=concepts,
                    n_samples=args.samples_per_concept,
                    device=args.device,
                )

                for prompt, response, sample_type in samples:
                    all_samples.append({
                        'concept': profile.sumo_term,
                        'quadrant': q,
                        'sample_type': sample_type,
                        'prompt': prompt,
                        'response': response,
                    })

                print(f"      Generated {len(samples)} samples")

        # Save samples
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        print(f"\nSaved {len(all_samples)} samples to {samples_file}")

        # Clean up model memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Judge samples
    judgments_file = output_dir / "judgments.json"

    if args.skip_judging:
        if judgments_file.exists():
            print("\nLoading existing judgments...")
            with open(judgments_file) as f:
                judgments = json.load(f)
        else:
            print("\nSkipping judging (no API key) - using samples without judge scores...")
            # Use samples directly, add placeholder judgment fields
            judgments = []
            for sample in all_samples:
                judgments.append({
                    **sample,
                    'inferred_topic': 'N/A (judging skipped)',
                    'relevance_rating': 0,
                    'relevance_explanation': 'Judging skipped - no API key',
                })
    else:
        print("\nJudging samples with LLM...")
        client = Anthropic()

        judgments = []
        for i, sample in enumerate(all_samples):
            if i % 10 == 0:
                print(f"  Judging sample {i+1}/{len(all_samples)}...")

            # Infer topic
            inferred_topic = judge_sample_topic(
                client, sample['response'], args.judge_model
            )

            # Rate relevance to POSITIVE concept (always the target, even for negatives)
            relevance, explanation = judge_sample_relevance(
                client, sample['concept'], sample['response'], args.judge_model
            )

            judgments.append({
                **sample,
                'inferred_topic': inferred_topic,
                'relevance_rating': relevance,
                'relevance_explanation': explanation,
            })

        # Save judgments
        with open(judgments_file, 'w') as f:
            json.dump(judgments, f, indent=2)
        print(f"\nSaved judgments to {judgments_file}")

    # Reload model for lens training
    print("\nReloading model for lens training...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()

    # Train lenses and collect metrics
    print("\nTraining lenses for each concept...")
    lens_metrics = {}

    for q in ['A', 'B', 'C', 'D']:
        for profile in quadrants[q]:
            concept_samples = [
                (j['prompt'], j['response'], j['sample_type'])
                for j in judgments
                if j['concept'] == profile.sumo_term
            ]

            if len(concept_samples) >= 10:
                metrics = train_lens_on_samples(
                    model, tokenizer, concept_samples, args.device
                )
                lens_metrics[profile.sumo_term] = metrics
                print(f"  {profile.sumo_term}: F1={metrics['f1']:.3f}")

    # Generate CSV report
    csv_file = output_dir / "analysis.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'concept', 'quadrant', 'sample_type', 'prompt', 'response',
            'inferred_topic', 'relevance_rating', 'relevance_explanation',
            'lens_f1', 'lens_precision', 'lens_recall'
        ])

        for j in judgments:
            metrics = lens_metrics.get(j['concept'], {})
            writer.writerow([
                j['concept'],
                j['quadrant'],
                j['sample_type'],
                j['prompt'][:200],
                j['response'][:200],
                j['inferred_topic'],
                j['relevance_rating'],
                j['relevance_explanation'],
                metrics.get('f1', ''),
                metrics.get('precision', ''),
                metrics.get('recall', ''),
            ])

    print(f"\nSaved analysis CSV to {csv_file}")

    # Generate summary report
    report_file = output_dir / "quality_report.md"
    with open(report_file, 'w') as f:
        f.write("# Training Data Quality Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Concept Pack: {args.concept_pack}\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Layer: {args.layer}\n")
        f.write(f"- Judge Model: {args.judge_model}\n\n")

        f.write("## Summary by Quadrant\n\n")
        for q in ['A', 'B', 'C', 'D']:
            q_samples = [j for j in judgments if j['quadrant'] == q]
            q_positive = [j for j in q_samples if j['sample_type'] == 'positive']
            q_negative = [j for j in q_samples if j['sample_type'] == 'negative']

            avg_pos_rel = sum(j['relevance_rating'] for j in q_positive) / max(len(q_positive), 1)
            avg_neg_rel = sum(j['relevance_rating'] for j in q_negative) / max(len(q_negative), 1)

            q_concepts = list(set(j['concept'] for j in q_samples))
            avg_f1 = sum(lens_metrics.get(c, {}).get('f1', 0) for c in q_concepts) / max(len(q_concepts), 1)

            f.write(f"### Quadrant {q}\n\n")
            if q == 'A':
                f.write("Low positives (1-2 synsets), Low negatives (0-3 siblings)\n\n")
            elif q == 'B':
                f.write("Low positives (1-2 synsets), High negatives (10+ siblings)\n\n")
            elif q == 'C':
                f.write("High positives (5+ synsets), High negatives (10+ siblings)\n\n")
            elif q == 'D':
                f.write("High positives (5+ synsets), Low negatives (0-3 siblings)\n\n")

            f.write(f"- Concepts: {', '.join(q_concepts)}\n")
            f.write(f"- Avg Positive Relevance: {avg_pos_rel:.2f}/5\n")
            f.write(f"- Avg Negative Relevance: {avg_neg_rel:.2f}/5\n")
            f.write(f"- Relevance Gap: {avg_pos_rel - avg_neg_rel:.2f}\n")
            f.write(f"- Avg Lens F1: {avg_f1:.3f}\n\n")

        # Overall analysis
        all_positive = [j for j in judgments if j['sample_type'] == 'positive']
        all_negative = [j for j in judgments if j['sample_type'] == 'negative']

        overall_pos_rel = sum(j['relevance_rating'] for j in all_positive) / max(len(all_positive), 1)
        overall_neg_rel = sum(j['relevance_rating'] for j in all_negative) / max(len(all_negative), 1)
        overall_gap = overall_pos_rel - overall_neg_rel
        overall_f1 = sum(m['f1'] for m in lens_metrics.values()) / max(len(lens_metrics), 1)

        f.write("## Overall Analysis\n\n")
        f.write(f"- Total samples analyzed: {len(judgments)}\n")
        f.write(f"- Overall Positive Relevance: {overall_pos_rel:.2f}/5\n")
        f.write(f"- Overall Negative Relevance: {overall_neg_rel:.2f}/5\n")
        f.write(f"- Overall Relevance Gap: {overall_gap:.2f}\n")
        f.write(f"- Overall Lens F1: {overall_f1:.3f}\n\n")

        f.write("## Interpretation\n\n")
        f.write("The relevance gap between positive and negative samples indicates ")
        f.write("how distinguishable our training data is. A larger gap suggests ")
        f.write("cleaner separation and higher achievable lens accuracy.\n\n")

        if overall_gap < 1.0:
            f.write("**WARNING**: Low relevance gap (<1.0) suggests training data quality ")
            f.write("may limit lens performance. Consider:\n")
            f.write("- Using a larger/better model for generation\n")
            f.write("- Improving prompt engineering\n")
            f.write("- Lowering F1 targets to match data quality\n\n")

        recommended_f1 = min(0.95, max(0.70, 0.5 + overall_gap * 0.15))
        f.write(f"**Recommended F1 Target**: {recommended_f1:.2f} ")
        f.write(f"(based on relevance gap of {overall_gap:.2f})\n")

    print(f"Saved quality report to {report_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Overall Positive Relevance: {overall_pos_rel:.2f}/5")
    print(f"Overall Negative Relevance: {overall_neg_rel:.2f}/5")
    print(f"Relevance Gap: {overall_gap:.2f}")
    print(f"Average Lens F1: {overall_f1:.3f}")
    print(f"\nRecommended F1 Target: {recommended_f1:.2f}")
    print()


if __name__ == "__main__":
    main()
