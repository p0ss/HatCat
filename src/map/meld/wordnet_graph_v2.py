"""
WordNet Concept Graph V2 - Improved negative sampling

Changes from V1:
1. Sample negatives from ALL WordNet (not just top-N)
2. Add relational prompts for positive samples
"""

import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict
import json
from pathlib import Path
import random


def download_wordnet():
    """Download WordNet data if not already present."""
    try:
        wn.synsets('test')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("✓ WordNet downloaded")


class WordNetGraphV2:
    """Improved WordNet graph with better negative sampling."""

    def __init__(self):
        download_wordnet()
        self.all_synsets = list(wn.all_synsets())
        print(f"Loaded {len(self.all_synsets):,} WordNet synsets")

    def get_synset_name(self, synset) -> str:
        """Get clean name for a synset."""
        return synset.lemmas()[0].name().replace('_', ' ')

    def count_relationships(self, synset) -> int:
        """Count all relationships for a synset."""
        count = 0
        count += len(synset.hypernyms())
        count += len(synset.hyponyms())
        count += len(synset.part_meronyms())
        count += len(synset.substance_meronyms())
        count += len(synset.member_meronyms())
        count += len(synset.part_holonyms())
        count += len(synset.substance_holonyms())
        count += len(synset.member_holonyms())
        count += len(synset.similar_tos())
        count += len(synset.attributes())
        count += len(synset.entailments())
        count += len(synset.causes())
        return count

    def rank_by_connectivity(self, pos_filter: List[str] = None):
        """Rank synsets by connectivity."""
        print("Ranking synsets by connectivity...")
        rankings = []

        for synset in self.all_synsets:
            if pos_filter and synset.pos() not in pos_filter:
                continue

            name = self.get_synset_name(synset)
            count = self.count_relationships(synset)
            synset_id = synset.name()
            rankings.append((name, count, synset_id))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def sample_distant_negatives(
        self,
        target_synset,
        min_distance: int = 5,
        n_samples: int = 50,
        pos_filter: List[str] = ['n', 'v']
    ) -> List[str]:
        """
        Sample negatives from ALL WordNet, not just training concepts.

        This ensures true semantic distance.

        Fallback: If we can't find enough distant negatives (island concepts),
        use random concepts from different POS or semantic domains.
        """
        distant = []

        # Sample from all synsets (not just top-N)
        candidates = [s for s in self.all_synsets if s.pos() in pos_filter]

        # Shuffle for random sampling
        random.shuffle(candidates)

        for candidate in candidates:
            if candidate == target_synset:
                continue

            # Check distance
            try:
                dist = target_synset.shortest_path_distance(candidate)
            except:
                dist = None

            if dist is not None and dist >= min_distance:
                distant.append(self.get_synset_name(candidate))

            # Stop when we have enough
            if len(distant) >= n_samples:
                break

        # Fallback for island concepts: use random concepts from opposite POS
        if len(distant) < n_samples:
            # Get opposite POS candidates
            opposite_pos = ['v'] if target_synset.pos() == 'n' else ['n']
            fallback_candidates = [s for s in self.all_synsets if s.pos() in opposite_pos]
            random.shuffle(fallback_candidates)

            for candidate in fallback_candidates:
                if len(distant) >= n_samples:
                    break
                distant.append(self.get_synset_name(candidate))

        return distant

    def get_related_concepts(self, synset, max_per_type: int = 10) -> Dict[str, List[str]]:
        """
        Get concepts related to this synset for relational prompts.

        Returns structured dictionary with relationship types prioritized:
        1. Hypernyms (is-a) - broader concepts
        2. Hyponyms (types of) - specific instances
        3. Meronyms (has-part) - component relationships
        4. Holonyms (member-of) - membership relationships

        Returns:
            {
                'hypernyms': [concept names],
                'hyponyms': [concept names],
                'meronyms': [concept names],
                'holonyms': [concept names]
            }
        """
        related = {}

        # Priority 1: Hypernyms (is-a) - broader concepts
        hypernyms = [self.get_synset_name(s) for s in synset.hypernyms()]
        related['hypernyms'] = hypernyms[:max_per_type]

        # Priority 2: Hyponyms (types of) - specific instances
        hyponyms = [self.get_synset_name(s) for s in synset.hyponyms()]
        related['hyponyms'] = hyponyms[:max_per_type]

        # Priority 3: Meronyms (has-part) - component relationships
        meronyms = []
        meronyms.extend([self.get_synset_name(s) for s in synset.part_meronyms()])
        meronyms.extend([self.get_synset_name(s) for s in synset.substance_meronyms()])
        meronyms.extend([self.get_synset_name(s) for s in synset.member_meronyms()])
        related['meronyms'] = list(set(meronyms))[:max_per_type]

        # Priority 4: Holonyms (member-of) - membership relationships
        holonyms = []
        holonyms.extend([self.get_synset_name(s) for s in synset.part_holonyms()])
        holonyms.extend([self.get_synset_name(s) for s in synset.substance_holonyms()])
        holonyms.extend([self.get_synset_name(s) for s in synset.member_holonyms()])
        related['holonyms'] = list(set(holonyms))[:max_per_type]

        return related

    def get_related_concepts_flat(self, synset, max_related: int = 10) -> List[str]:
        """
        Get related concepts as flat list (for backwards compatibility).
        Prioritizes: hypernyms > hyponyms > meronyms > holonyms
        """
        structured = self.get_related_concepts(synset, max_per_type=max_related)

        # Flatten with priority order
        flat = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            flat.extend(structured[rel_type])
            if len(flat) >= max_related:
                break

        return flat[:max_related]

    def get_all_relationships(self, synset) -> Dict[str, List[str]]:
        """
        Get ALL relationships of ALL types for a synset.

        This includes:
        - hypernyms, hyponyms (is-a hierarchy)
        - meronyms, holonyms (part-whole)
        - similar_tos (similar meanings)
        - attributes (noun-adjective relations)
        - entailments (verb implications)
        - causes (causal relations)
        - also_sees (related concepts)

        Returns all relationships without any limit.
        """
        all_related = {}

        # Core hierarchy relationships
        all_related['hypernyms'] = [self.get_synset_name(s) for s in synset.hypernyms()]
        all_related['hyponyms'] = [self.get_synset_name(s) for s in synset.hyponyms()]

        # Part-whole relationships
        meronyms = []
        meronyms.extend([self.get_synset_name(s) for s in synset.part_meronyms()])
        meronyms.extend([self.get_synset_name(s) for s in synset.substance_meronyms()])
        meronyms.extend([self.get_synset_name(s) for s in synset.member_meronyms()])
        all_related['meronyms'] = list(set(meronyms))

        holonyms = []
        holonyms.extend([self.get_synset_name(s) for s in synset.part_holonyms()])
        holonyms.extend([self.get_synset_name(s) for s in synset.substance_holonyms()])
        holonyms.extend([self.get_synset_name(s) for s in synset.member_holonyms()])
        all_related['holonyms'] = list(set(holonyms))

        # Additional relationship types
        all_related['similar_tos'] = [self.get_synset_name(s) for s in synset.similar_tos()]
        all_related['attributes'] = [self.get_synset_name(s) for s in synset.attributes()]
        all_related['entailments'] = [self.get_synset_name(s) for s in synset.entailments()]
        all_related['causes'] = [self.get_synset_name(s) for s in synset.causes()]
        all_related['also_sees'] = [self.get_synset_name(s) for s in synset.also_sees()]

        return all_related

    def build_neutral_pool(
        self,
        training_synsets: List,
        min_neutral_distance: int = 15,
        pool_size: int = 1000,
        pos_filter: List[str] = ['n', 'v']
    ) -> List[str]:
        """
        Build a pool of neutral concepts for comprehensive testing.

        Neutral concepts must be semantically distant (distance ≥ min_neutral_distance)
        from ALL training concepts. These are never used as negatives or relationships.

        Args:
            training_synsets: List of synsets used in training
            min_neutral_distance: Minimum path distance to ALL training concepts
            pool_size: Target number of neutral concepts
            pos_filter: POS tags to include

        Returns:
            List of neutral concept names
        """
        print(f"\nBuilding neutral pool...")
        print(f"  - Min distance to ALL training concepts: {min_neutral_distance}")
        print(f"  - Target pool size: {pool_size}")

        neutral_pool = []
        candidates = [s for s in self.all_synsets if s.pos() in pos_filter]
        random.shuffle(candidates)

        checked = 0
        for candidate in candidates:
            if len(neutral_pool) >= pool_size:
                break

            checked += 1
            if checked % 1000 == 0:
                print(f"  Checked {checked}, found {len(neutral_pool)} neutrals...")

            # Check if candidate is distant from ALL training concepts
            is_neutral = True
            for training_synset in training_synsets:
                if candidate == training_synset:
                    is_neutral = False
                    break

                try:
                    dist = candidate.shortest_path_distance(training_synset)
                except:
                    dist = None

                # If any training concept is too close, reject candidate
                if dist is not None and dist < min_neutral_distance:
                    is_neutral = False
                    break

            if is_neutral:
                neutral_pool.append(self.get_synset_name(candidate))

        print(f"  ✓ Found {len(neutral_pool)} neutral concepts after checking {checked}")
        return neutral_pool

    def build_concept_data(
        self,
        top_n: int = 1000,
        min_neg_distance: int = 5,
        n_negatives: int = 50,
        min_related: int = 10,
        neutral_pool_size: int = 1000,
        min_neutral_distance: int = 15,
        pos_filter: List[str] = ['n', 'v']
    ) -> Dict:
        """
        Build complete concept data including negatives, relationships, and neutral pool.

        Relationships are filtered to only include concepts in the training set,
        with a minimum guarantee of min_related relationships.

        Returns:
            {
                'concepts': {
                    'concept_name': {
                        'synset_id': str,
                        'connectivity': int,
                        'negatives': [distant concepts from ALL WordNet],
                        'related': [flat list of related concepts],
                        'related_structured': {
                            'hypernyms': [...],
                            'hyponyms': [...],
                            'meronyms': [...],
                            'holonyms': [...],
                            'similar_tos': [...],
                            'attributes': [...],
                            'entailments': [...],
                            'causes': [...],
                            'also_sees': [...]
                        }
                    }
                },
                'neutral_pool': [list of neutral concept names]
            }
        """
        # Get top-N ranked concepts
        rankings = self.rank_by_connectivity(pos_filter)
        top_concepts = rankings[:top_n]

        # Build set of concept names for fast lookup
        concept_names = {name for name, _, _ in top_concepts}

        # Get synsets for training concepts
        training_synsets = [wn.synset(synset_id) for _, _, synset_id in top_concepts]

        print(f"\nBuilding data for top {top_n} concepts...")
        print(f"  - Sampling negatives from all {len(self.all_synsets):,} synsets")
        print(f"  - Min distance: {min_neg_distance}")
        print(f"  - Negatives per concept: {n_negatives}")
        print(f"  - Minimum related per concept: {min_related}")
        print(f"  - Including all relationships to concepts in training set")

        concept_data = {}

        for i, (name, connectivity, synset_id) in enumerate(top_concepts):
            if i % 100 == 0:
                print(f"  {i}/{top_n}...")

            synset = wn.synset(synset_id)

            # Sample distant negatives from ALL WordNet
            negatives = self.sample_distant_negatives(
                synset,
                min_neg_distance,
                n_negatives,
                pos_filter
            )

            # Get ALL relationships
            all_relationships = self.get_all_relationships(synset)

            # Filter to only include concepts in training set
            filtered_relationships = {}
            for rel_type, concepts in all_relationships.items():
                in_training_set = [c for c in concepts if c in concept_names]
                filtered_relationships[rel_type] = in_training_set

            # Count total relationships in training set
            total_in_set = sum(len(concepts) for concepts in filtered_relationships.values())

            # If we don't have enough, fall back to prioritized minimum
            if total_in_set < min_related:
                # Use old method to guarantee minimum
                fallback = self.get_related_concepts(synset, max_per_type=min_related)
                # Merge fallback with filtered (prioritizing in-training-set)
                for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
                    if rel_type in fallback:
                        existing = set(filtered_relationships.get(rel_type, []))
                        for concept in fallback[rel_type]:
                            if len(existing) >= min_related:
                                break
                            existing.add(concept)
                        filtered_relationships[rel_type] = list(existing)

            # Create flat list for backwards compatibility
            flat = []
            priority_order = ['hypernyms', 'hyponyms', 'meronyms', 'holonyms',
                            'similar_tos', 'attributes', 'entailments', 'causes', 'also_sees']
            for rel_type in priority_order:
                if rel_type in filtered_relationships:
                    flat.extend(filtered_relationships[rel_type])

            concept_data[name] = {
                'synset_id': synset_id,
                'connectivity': connectivity,
                'negatives': negatives,
                'related': flat,  # Flat list of ALL in-training-set relationships
                'related_structured': filtered_relationships  # Structured by type, filtered to training set
            }

        # Build neutral pool (concepts distant from ALL training concepts)
        neutral_pool = self.build_neutral_pool(
            training_synsets,
            min_neutral_distance,
            neutral_pool_size,
            pos_filter
        )

        return {
            'concepts': concept_data,
            'neutral_pool': neutral_pool
        }

    def save_concept_data(self, data: Dict, output_path: Path):
        """Save concept data to JSON."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Saved concept data to: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("WORDNET CONCEPT GRAPH V2 - IMPROVED NEGATIVES")
    print("=" * 70)
    print()

    builder = WordNetGraphV2()

    # Build for different scales
    output_dir = Path('data/concept_graph')
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in [10, 100, 1000, 10000]:
        print(f"\n{'='*70}")
        print(f"Building data for top {n} concepts...")
        print(f"{'='*70}")

        data = builder.build_concept_data(
            top_n=n,
            min_neg_distance=5,
            n_negatives=50,
            min_related=10,
            neutral_pool_size=1000,
            min_neutral_distance=15,
            pos_filter=['n', 'v']
        )

        output_path = output_dir / f'wordnet_v2_top{n}.json'
        builder.save_concept_data(data, output_path)

        # Show sample
        sample_concept = list(data['concepts'].keys())[0]
        print(f"\nSample - '{sample_concept}':")
        print(f"  Connectivity: {data['concepts'][sample_concept]['connectivity']}")
        print(f"  Negatives ({len(data['concepts'][sample_concept]['negatives'])}): {data['concepts'][sample_concept]['negatives'][:3]}...")
        print(f"  Related ({len(data['concepts'][sample_concept]['related'])}): {data['concepts'][sample_concept]['related'][:3]}...")
        print(f"\nNeutral pool: {len(data['neutral_pool'])} concepts (distance ≥15 from ALL training concepts)")
        print(f"  Sample: {data['neutral_pool'][:5]}...")

    print("\n" + "=" * 70)
    print("✓ WORDNET GRAPH V2 COMPLETE")
    print("=" * 70)
