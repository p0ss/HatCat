#!/usr/bin/env python3
"""
Debug script to show exactly what prompts are generated for relationship-first training.
"""

import json
from pathlib import Path

def show_relationship_prompts(concept_graph_path: str):
    """Show what prompts would be generated for relationship-first approach."""

    with open(concept_graph_path) as f:
        concept_graph = json.load(f)

    print("=== RELATIONSHIP-FIRST PROMPT GENERATION ===\n")

    # Phase 1: Show relationship edge prompts
    print("PHASE 1: Relationship Edge Prompts")
    print("=" * 60)

    all_edges = set()
    for concept_name, concept_info in concept_graph.items():
        for related in concept_info['related']:
            edge = (concept_name, related)
            all_edges.add(edge)

    print(f"\nTotal unique relationship edges: {len(all_edges)}\n")

    # Show first 10 examples
    print("First 10 relationship prompts:")
    for idx, (source, target) in enumerate(sorted(all_edges)[:10]):
        prompt = f"How does {source} relate to {target}?"
        print(f"  {idx+1}. Edge: ({source}, {target})")
        print(f"     Prompt: \"{prompt}\"\n")

    # Phase 2: Show how prompts aggregate for each concept
    print("\n" + "=" * 60)
    print("PHASE 2: Concept Training Data (Positive Samples)")
    print("=" * 60)

    for concept_name, concept_info in sorted(concept_graph.items())[:3]:  # First 3 concepts
        print(f"\nConcept: {concept_name}")
        print(f"Related concepts: {len(concept_info['related'])}")

        print(f"\n  Positive samples ({1 + len(concept_info['related'])} total):")

        # 1. Definition prompt
        def_prompt = f"What is {concept_name}?"
        print(f"    1. [DEFINITION] \"{def_prompt}\"")

        # 2. All relationships where this concept appears
        for idx, related in enumerate(concept_info['related'], start=2):
            rel_prompt = f"How does {concept_name} relate to {related}?"
            print(f"    {idx}. [RELATIONSHIP] \"{rel_prompt}\"")

    # Phase 3: Show negative sampling strategy
    print("\n" + "=" * 60)
    print("PHASE 3: Negative Sample Strategy")
    print("=" * 60)

    for concept_name, concept_info in sorted(concept_graph.items())[:3]:  # First 3 concepts
        print(f"\nConcept: {concept_name}")

        antonyms = concept_info.get('related_structured', {}).get('antonyms', [])

        if antonyms:
            print(f"  Strategy: ANTONYM-BASED (found {len(antonyms)} antonyms)")
            print(f"  Antonyms: {antonyms}")

            # Show what prompts would be generated
            antonym = antonyms[0]
            print(f"\n  Negative samples for antonym '{antonym}':")

            # Definition
            ant_def_prompt = f"What is {antonym}?"
            print(f"    1. [ANTONYM DEF] \"{ant_def_prompt}\"")

            # Relationships (if we have the antonym in our graph)
            if antonym in concept_graph:
                for idx, related in enumerate(concept_graph[antonym]['related'][:5], start=2):
                    ant_rel_prompt = f"How does {antonym} relate to {related}?"
                    print(f"    {idx}. [ANTONYM REL] \"{ant_rel_prompt}\"")
        else:
            print(f"  Strategy: GRAPH-DISTANT FALLBACK (no antonyms)")
            distant = concept_info['negatives'][:5]
            print(f"  Using: {distant}")

            print(f"\n  Negative samples:")
            for idx, neg in enumerate(distant, start=1):
                neg_prompt = f"What is {neg}?"
                print(f"    {idx}. [DISTANT DEF] \"{neg_prompt}\"")

if __name__ == "__main__":
    concept_graph_path = "data/concept_graph/wordnet_v2_top10.json"
    show_relationship_prompts(concept_graph_path)
