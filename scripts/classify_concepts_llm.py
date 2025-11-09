#!/usr/bin/env python3
"""
Use LLM to classify WordNet concepts by safety/transparency relevance.

Ask Gemma to categorize each concept into psychological/safety categories.
"""

import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from nltk.corpus import wordnet as wn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Download WordNet if needed
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


CATEGORY_PROMPT = """Classify this concept into ONE category:

Concept: {concept}
Definition: {definition}

Categories:
1. EMOTION - feelings, moods, emotional states (happy, angry, fearful)
2. MOTIVATION - desires, goals, intentions (want, intend, seek, avoid)
3. EPISTEMIC - knowledge, belief, understanding (know, believe, doubt, assume)
4. MORAL - ethical concepts, right/wrong, harm (good, evil, fair, honest, deceive)
5. AGENCY - causation, control, actions (cause, make, force, allow, decide)
6. SOCIAL - relationships, interactions (trust, cooperate, promise, threaten)
7. TEMPORAL - time, modality (will, should, possible, necessary, future)
8. PHYSICAL - objects, materials, physical properties
9. OTHER - does not fit above categories

Answer with ONLY the category name (e.g., "EMOTION" or "PHYSICAL"):"""


def classify_concept(model, tokenizer, concept, definition, device="cuda"):
    """Use LLM to classify concept into category."""
    prompt = CATEGORY_PROMPT.format(concept=concept, definition=definition)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract category from response
    response_lower = response.lower()

    categories = ['emotion', 'motivation', 'epistemic', 'moral', 'agency',
                 'social', 'temporal', 'physical', 'other']

    for cat in categories:
        if cat in response_lower:
            return cat.upper()

    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-3-4b-pt')
    parser.add_argument('--top-k', type=int, default=5000,
                       help='Number of concepts to classify')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='data/concept_graph/llm_classified_concepts.json')
    args = parser.parse_args()

    print("="*60)
    print("LLM-BASED CONCEPT CLASSIFICATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Classifying top {args.top_k} concepts")
    print("="*60 + "\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded\n")

    # Get all synsets (nouns and verbs priority)
    print("Loading WordNet synsets...")
    all_synsets = list(wn.all_synsets())

    # Prioritize verbs and adjectives
    priority_synsets = []
    other_synsets = []

    for syn in all_synsets:
        if syn.pos() in ['v', 'a', 's']:  # verb, adjective, adj satellite
            priority_synsets.append(syn)
        else:
            other_synsets.append(syn)

    # Take top-k with priority for verbs/adjectives
    synsets_to_classify = priority_synsets[:args.top_k]
    if len(synsets_to_classify) < args.top_k:
        synsets_to_classify.extend(other_synsets[:args.top_k - len(synsets_to_classify)])

    print(f"Selected {len(synsets_to_classify)} synsets to classify")
    verb_adj_count = len([s for s in synsets_to_classify if s.pos() in ['v', 'a', 's']])
    noun_count = len([s for s in synsets_to_classify if s.pos() == 'n'])
    print(f"  Verbs/Adjectives: {verb_adj_count}")
    print(f"  Nouns: {noun_count}\n")

    # Classify each concept
    print("Classifying concepts...")
    classified = []
    category_counts = {cat: 0 for cat in ['EMOTION', 'MOTIVATION', 'EPISTEMIC',
                                          'MORAL', 'AGENCY', 'SOCIAL',
                                          'TEMPORAL', 'PHYSICAL', 'OTHER']}

    for synset in tqdm(synsets_to_classify):
        concept = synset.name().split('.')[0]
        definition = synset.definition()

        category = classify_concept(model, tokenizer, concept, definition, args.device)

        category_counts[category] += 1

        classified.append({
            'concept': concept,
            'synset': synset.name(),
            'pos': synset.pos(),
            'definition': definition,
            'category': category
        })

    print(f"\n✓ Classified {len(classified)} concepts")

    # Sort by category priority
    category_priority = {
        'EMOTION': 9,
        'MOTIVATION': 8,
        'EPISTEMIC': 7,
        'MORAL': 6,
        'AGENCY': 5,
        'SOCIAL': 4,
        'TEMPORAL': 3,
        'PHYSICAL': 1,
        'OTHER': 0
    }

    classified.sort(key=lambda x: (-category_priority.get(x['category'], 0),
                                   x['concept']))

    # Display statistics
    print("\n" + "="*60)
    print("CATEGORY DISTRIBUTION")
    print("="*60)
    for cat, count in sorted(category_counts.items(),
                            key=lambda x: -x[1]):
        pct = 100 * count / len(classified)
        print(f"  {cat:12} {count:4} ({pct:5.1f}%)")

    # Show top concepts per category
    print("\n" + "="*60)
    print("TOP 10 PER CATEGORY")
    print("="*60)

    for cat in ['EMOTION', 'MOTIVATION', 'EPISTEMIC', 'MORAL',
                'AGENCY', 'SOCIAL', 'TEMPORAL']:
        cat_concepts = [c for c in classified if c['category'] == cat][:10]

        if cat_concepts:
            print(f"\n{cat}:")
            for i, c in enumerate(cat_concepts, 1):
                print(f"  {i:2}. {c['concept']:20} [{c['pos']}] - {c['definition'][:50]}...")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'classification_method': 'llm',
            'model': args.model,
            'total_classified': len(classified),
            'categories': list(category_counts.keys())
        },
        'category_counts': category_counts,
        'concepts': classified
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
