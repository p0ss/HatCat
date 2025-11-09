"""
WordNet Concept Graph Builder

Uses NLTK's WordNet to build semantic graph with:
- ~117K synsets (concepts)
- Built-in semantic relationships (hypernyms, hyponyms, meronyms, etc.)
- Semantic distance via shortest path
- No API dependencies, all local

Strategy:
1. Rank concepts by connectivity (# of relationships)
2. Select top-N most connected concepts (top 10, 100, 1K, 10K, 50K)
3. Use path distance for negative sampling
"""

import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path
import numpy as np


def download_wordnet():
    """Download WordNet data if not already present."""
    try:
        wn.synsets('test')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Open Multilingual WordNet
        print("✓ WordNet downloaded")


class WordNetConceptGraph:
    """Build concept graph from WordNet."""

    def __init__(self):
        download_wordnet()
        self.all_synsets = list(wn.all_synsets())
        print(f"Loaded {len(self.all_synsets):,} WordNet synsets")

    def get_synset_name(self, synset) -> str:
        """Get clean name for a synset (lemma without POS tag)."""
        return synset.lemmas()[0].name().replace('_', ' ')

    def count_relationships(self, synset) -> int:
        """Count all relationships for a synset."""
        count = 0

        # Hypernyms (is-a)
        count += len(synset.hypernyms())

        # Hyponyms (reverse is-a)
        count += len(synset.hyponyms())

        # Meronyms (part-of)
        count += len(synset.part_meronyms())
        count += len(synset.substance_meronyms())
        count += len(synset.member_meronyms())

        # Holonyms (reverse part-of)
        count += len(synset.part_holonyms())
        count += len(synset.substance_holonyms())
        count += len(synset.member_holonyms())

        # Similar-to (for adjectives)
        count += len(synset.similar_tos())

        # Attributes
        count += len(synset.attributes())

        # Entailments
        count += len(synset.entailments())

        # Causes
        count += len(synset.causes())

        return count

    def rank_by_connectivity(self, pos_filter: List[str] = None) -> List[Tuple[str, int, str]]:
        """
        Rank synsets by number of relationships.

        Args:
            pos_filter: List of POS tags to include ('n', 'v', 'a', 'r')

        Returns:
            List of (synset_name, relationship_count, synset_id) sorted by count
        """
        print("Ranking synsets by connectivity...")

        rankings = []
        for synset in self.all_synsets:
            # Filter by POS if specified
            if pos_filter and synset.pos() not in pos_filter:
                continue

            name = self.get_synset_name(synset)
            count = self.count_relationships(synset)
            synset_id = synset.name()

            rankings.append((name, count, synset_id))

        # Sort by connectivity (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def path_distance(self, synset1, synset2) -> int:
        """
        Get semantic distance via shortest path.

        Returns:
            Distance (int), or None if no path exists
        """
        # WordNet has path_similarity which returns similarity in [0, 1]
        # We want distance, so use shortest_path_distance
        try:
            return synset1.shortest_path_distance(synset2)
        except:
            return None

    def sample_distant_concepts(
        self,
        target_synset,
        all_synsets: List,
        min_distance: int = 5,
        n_samples: int = 50
    ) -> List[str]:
        """
        Sample concepts with minimum semantic distance.

        Args:
            target_synset: The target synset
            all_synsets: List of candidate synsets
            min_distance: Minimum path distance
            n_samples: Number of negatives to sample

        Returns:
            List of distant synset names
        """
        distant = []

        for candidate in all_synsets:
            if candidate == target_synset:
                continue

            dist = self.path_distance(target_synset, candidate)

            if dist is not None and dist >= min_distance:
                distant.append(self.get_synset_name(candidate))

            # Early stop if we have enough
            if len(distant) >= n_samples * 2:
                break

        # Sample randomly
        import random
        if len(distant) > n_samples:
            return random.sample(distant, n_samples)
        else:
            return distant

    def build_negatives_for_top_n(
        self,
        top_n: int = 1000,
        min_distance: int = 5,
        n_negatives: int = 50,
        pos_filter: List[str] = ['n', 'v']  # Nouns and verbs only
    ) -> Dict:
        """
        Build negative sampling dict for top-N most connected concepts.

        Returns:
            {
                'concept_name': {
                    'synset_id': 'synset.name()',
                    'connectivity': int,
                    'negatives': [list of distant concept names]
                }
            }
        """
        # Get top-N ranked concepts
        rankings = self.rank_by_connectivity(pos_filter)
        top_concepts = rankings[:top_n]

        print(f"\nBuilding negatives for top {top_n} concepts...")
        print(f"Min distance: {min_distance}, Negatives per concept: {n_negatives}")

        negatives_map = {}

        for i, (name, connectivity, synset_id) in enumerate(top_concepts):
            if i % 100 == 0:
                print(f"  {i}/{top_n}...")

            synset = wn.synset(synset_id)

            # Sample distant concepts from all top-N
            # (This ensures negatives are also "important" concepts)
            candidate_synsets = [wn.synset(sid) for _, _, sid in top_concepts]

            distant_names = self.sample_distant_concepts(
                synset,
                candidate_synsets,
                min_distance,
                n_negatives
            )

            negatives_map[name] = {
                'synset_id': synset_id,
                'connectivity': connectivity,
                'negatives': distant_names
            }

        return negatives_map

    def save_negatives(self, negatives_map: Dict, output_path: Path):
        """Save negatives mapping to JSON."""
        with open(output_path, 'w') as f:
            json.dump(negatives_map, f, indent=2)
        print(f"\n✓ Saved negatives to: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("WORDNET CONCEPT GRAPH BUILDER")
    print("=" * 70)
    print()

    builder = WordNetConceptGraph()

    # Show top 20 most connected concepts
    print("\nTop 20 most connected concepts:")
    rankings = builder.rank_by_connectivity(pos_filter=['n', 'v'])

    for i, (name, count, synset_id) in enumerate(rankings[:20]):
        synset = wn.synset(synset_id)
        definition = synset.definition()[:60]
        print(f"  {i+1:2d}. {name:20s} ({count:3d} relations) - {definition}...")

    # Build for different scales
    output_dir = Path('data/concept_graph')
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in [10, 100, 1000]:
        print(f"\n{'='*70}")
        print(f"Building negatives for top {n} concepts...")
        print(f"{'='*70}")

        negatives = builder.build_negatives_for_top_n(
            top_n=n,
            min_distance=5,
            n_negatives=50,
            pos_filter=['n', 'v']
        )

        output_path = output_dir / f'wordnet_negatives_top{n}.json'
        builder.save_negatives(negatives, output_path)

        # Show sample
        sample_concept = list(negatives.keys())[0]
        print(f"\nSample - '{sample_concept}':")
        print(f"  Connectivity: {negatives[sample_concept]['connectivity']}")
        print(f"  Negatives ({len(negatives[sample_concept]['negatives'])}): {negatives[sample_concept]['negatives'][:5]}...")

    print("\n" + "=" * 70)
    print("✓ WORDNET GRAPH COMPLETE")
    print("=" * 70)
