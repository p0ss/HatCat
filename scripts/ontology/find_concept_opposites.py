#!/usr/bin/env python3
"""
Find candidate opposite concepts using embeddings.

For each concept, finds the top-k most similar concepts that could be
conceptual opposites (same semantic space, different poles).

Usage:
    # View candidates for a single concept
    python scripts/ontology/find_concept_opposites.py --concept Deception --top-k 50

    # Generate API prompt
    python scripts/ontology/find_concept_opposites.py --concept Deception --api-format

    # Call API to find opposite (requires ANTHROPIC_API_KEY)
    python scripts/ontology/find_concept_opposites.py --concept Deception --call-api

    # Batch process multiple concepts
    python scripts/ontology/find_concept_opposites.py --batch --output opposites.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional


def load_embeddings(pack_dir: Path):
    """Load concept embeddings and index."""
    emb_dir = pack_dir / "embedding_index"
    embeddings = np.load(emb_dir / "embeddings.npy")
    with open(emb_dir / "index.json") as f:
        index = json.load(f)
    return embeddings, index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(b_norm, a_norm)


def find_candidates(
    concept: str,
    embeddings: np.ndarray,
    terms: list,
    top_k: int = 50,
    mode: str = "similar",
) -> list:
    """
    Find candidate opposite concepts.

    Args:
        concept: The concept to find opposites for
        embeddings: Embedding matrix [n_concepts, dim]
        terms: List of concept names
        top_k: Number of candidates to return
        mode: "similar" (same semantic space) or "distant" (far away)

    Returns:
        List of (concept_name, similarity_score) tuples
    """
    if concept not in terms:
        raise ValueError(f"Concept '{concept}' not found in index")

    idx = terms.index(concept)
    query_emb = embeddings[idx]

    # Compute similarities to all concepts
    similarities = cosine_similarity(query_emb, embeddings)

    if mode == "similar":
        # Get most similar (excluding self)
        sorted_indices = np.argsort(similarities)[::-1]
    else:
        # Get least similar (most distant)
        sorted_indices = np.argsort(similarities)

    candidates = []
    for i in sorted_indices:
        if terms[i] != concept:
            candidates.append((terms[i], float(similarities[i])))
        if len(candidates) >= top_k:
            break

    return candidates


def load_concept_definitions(pack_dir: Path, concepts: list) -> dict:
    """Load definitions for specified concepts from the concept files."""
    definitions = {}
    concepts_dir = pack_dir / "concepts"

    if not concepts_dir.exists():
        return definitions

    # Concepts are in layer subdirectories with lowercase filenames
    for concept in concepts:
        filename = f"{concept.lower()}.json"
        # Search in all layer directories
        for layer_dir in concepts_dir.glob("layer*"):
            concept_file = layer_dir / filename
            if concept_file.exists():
                with open(concept_file) as f:
                    data = json.load(f)
                    definitions[concept] = data.get("definition", data.get("description", ""))
                break

    return definitions


def format_for_api(concept: str, definition: str, candidates: list, candidate_defs: dict) -> str:
    """Format a prompt for an LLM to pick the conceptual opposite."""
    prompt = f"""Given this concept, identify its conceptual opposite from the candidates below.

CONCEPT: {concept}
DEFINITION: {definition}

CANDIDATES (pick the best conceptual opposite, or say "none" if no good match):
"""
    for i, (name, score) in enumerate(candidates, 1):
        defn = candidate_defs.get(name, "(no definition)")
        prompt += f"\n{i}. {name}: {defn[:200]}"

    prompt += """

Reply with JSON: {"opposite": "ConceptName", "reasoning": "brief explanation"}
Or if none fit: {"opposite": null, "reasoning": "why none fit"}"""

    return prompt


def call_api(prompt: str, model: str = "claude-sonnet-4-20250514") -> Optional[dict]:
    """Call Anthropic API to get opposite concept."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        text = response.content[0].text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
        print(f"Raw response: {text}")
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def find_opposite_for_concept(
    concept: str,
    embeddings: np.ndarray,
    terms: list,
    pack_dir: Path,
    top_k: int = 50,
    call_api_flag: bool = False,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Find opposite for a single concept."""
    candidates = find_candidates(concept, embeddings, terms, top_k, "similar")
    candidate_names = [concept] + [name for name, _ in candidates]
    definitions = load_concept_definitions(pack_dir, candidate_names)

    concept_def = definitions.get(concept, "(no definition)")
    prompt = format_for_api(concept, concept_def, candidates, definitions)

    result = {
        "concept": concept,
        "definition": concept_def,
        "candidates": [name for name, _ in candidates[:10]],  # Top 10 for reference
    }

    if call_api_flag:
        api_result = call_api(prompt, model)
        if api_result:
            result["opposite"] = api_result.get("opposite")
            result["reasoning"] = api_result.get("reasoning")
        else:
            result["opposite"] = None
            result["reasoning"] = "API call failed"
    else:
        result["prompt"] = prompt

    return result


def batch_find_opposites(
    pack_dir: Path,
    output_file: Path,
    top_k: int = 50,
    model: str = "claude-sonnet-4-20250514",
    limit: Optional[int] = None,
    skip_existing: bool = True,
):
    """Find opposites for all concepts in pack."""
    import time

    embeddings, index = load_embeddings(pack_dir)
    terms = index["terms"]

    # Load existing results if resuming
    existing = {}
    if skip_existing and output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing[data["concept"]] = data
        print(f"Loaded {len(existing)} existing results")

    # Process concepts
    processed = 0
    with open(output_file, "a") as f:
        for i, concept in enumerate(terms):
            if limit and processed >= limit:
                break

            if concept in existing:
                continue

            print(f"[{i+1}/{len(terms)}] {concept}...")

            try:
                result = find_opposite_for_concept(
                    concept, embeddings, terms, pack_dir, top_k,
                    call_api_flag=True, model=model
                )
                f.write(json.dumps(result) + "\n")
                f.flush()

                opposite = result.get("opposite", "none")
                print(f"  → {opposite}: {result.get('reasoning', '')[:60]}")

                processed += 1

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nProcessed {processed} concepts. Results in: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Find candidate opposite concepts")
    parser.add_argument("--concept", type=str, help="Concept to find opposites for")
    parser.add_argument("--pack", type=str, default="first-light", help="Concept pack name")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates")
    parser.add_argument("--mode", choices=["similar", "distant", "both"], default="both",
                        help="Search mode: similar (same space), distant (far away), or both")
    parser.add_argument("--api-format", action="store_true",
                        help="Output prompt formatted for LLM API")
    parser.add_argument("--call-api", action="store_true",
                        help="Call Anthropic API to find opposite")
    parser.add_argument("--batch", action="store_true",
                        help="Process all concepts in pack")
    parser.add_argument("--output", type=str, default="concept_opposites.jsonl",
                        help="Output file for batch mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of concepts to process in batch mode")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Model to use for API calls")
    args = parser.parse_args()

    pack_dir = Path("concept_packs") / args.pack

    # Batch mode
    if args.batch:
        output_file = Path(args.output)
        batch_find_opposites(
            pack_dir, output_file, args.top_k, args.model, args.limit
        )
        return

    # Single concept mode - require --concept
    if not args.concept:
        parser.error("--concept is required unless using --batch")

    embeddings, index = load_embeddings(pack_dir)
    terms = index["terms"]

    # Call API mode
    if args.call_api:
        result = find_opposite_for_concept(
            args.concept, embeddings, terms, pack_dir, args.top_k,
            call_api_flag=True, model=args.model
        )
        print(json.dumps(result, indent=2))
        return

    # API format mode
    if args.api_format:
        similar = find_candidates(args.concept, embeddings, terms, args.top_k, "similar")
        candidate_names = [args.concept] + [name for name, _ in similar]
        definitions = load_concept_definitions(pack_dir, candidate_names)

        concept_def = definitions.get(args.concept, "(no definition)")
        prompt = format_for_api(args.concept, concept_def, similar, definitions)
        print(prompt)
        return

    # Default: show candidates
    print(f"Concept: {args.concept}")
    print(f"Total concepts in pack: {len(terms)}")
    print()

    if args.mode in ("similar", "both"):
        print(f"=== TOP {args.top_k} MOST SIMILAR (same semantic neighborhood) ===")
        similar = find_candidates(args.concept, embeddings, terms, args.top_k, "similar")
        for i, (name, score) in enumerate(similar, 1):
            print(f"  {i:2d}. {name:40s} (sim: {score:.3f})")
        print()

    if args.mode in ("distant", "both"):
        print(f"=== TOP {args.top_k} MOST DISTANT (semantically far) ===")
        distant = find_candidates(args.concept, embeddings, terms, args.top_k, "distant")
        for i, (name, score) in enumerate(distant, 1):
            print(f"  {i:2d}. {name:40s} (sim: {score:.3f})")


if __name__ == "__main__":
    main()
