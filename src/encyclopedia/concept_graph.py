"""
Concept Graph Builder using Wikidata

Builds a semantic distance graph where:
- Positive samples: Same concept (distance 0)
- Hard negatives: Related concepts (distance 1-2)
- True negatives: Distant concepts (distance 4+)

This gives much better training signal than "What is NOT X?"
"""

import requests
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque
import json
from pathlib import Path


class WikidataConceptGraph:
    """Build concept graph from Wikidata for negative sampling."""

    def __init__(self, cache_dir: Path = Path('data/concept_graph')):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.endpoint = "https://query.wikidata.org/sparql"
        self.entity_cache = {}
        self.relation_cache = {}

    def search_entity(self, concept: str) -> str:
        """
        Search for Wikidata entity ID for a concept.

        Returns:
            Entity ID (e.g., 'Q1234') or None if not found
        """
        cache_file = self.cache_dir / f'entity_{concept.replace(" ", "_")}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f).get('entity_id')

        # Search via Wikidata API
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'type': 'item',
            'search': concept,
            'limit': 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('search'):
                entity_id = data['search'][0]['id']

                # Cache result
                with open(cache_file, 'w') as f:
                    json.dump({'entity_id': entity_id, 'concept': concept}, f)

                time.sleep(0.1)  # Rate limit
                return entity_id

        except Exception as e:
            print(f"Error searching for '{concept}': {e}")

        return None

    def get_related_entities(self, entity_id: str, max_relations: int = 50) -> List[str]:
        """
        Get entities directly related to this entity via any property.

        Returns:
            List of related entity IDs
        """
        cache_file = self.cache_dir / f'relations_{entity_id}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)['related']

        # SPARQL query for related entities
        query = f"""
        SELECT DISTINCT ?related WHERE {{
            {{ wd:{entity_id} ?prop ?related . }}
            UNION
            {{ ?related ?prop wd:{entity_id} . }}
            FILTER(STRSTARTS(STR(?related), "http://www.wikidata.org/entity/Q"))
        }}
        LIMIT {max_relations}
        """

        try:
            response = requests.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                timeout=30,
                headers={'User-Agent': 'HatCat/1.0'}
            )
            response.raise_for_status()
            data = response.json()

            related = []
            for binding in data['results']['bindings']:
                uri = binding['related']['value']
                # Extract entity ID from URI
                entity = uri.split('/')[-1]
                if entity.startswith('Q'):
                    related.append(entity)

            # Cache result
            with open(cache_file, 'w') as f:
                json.dump({'entity_id': entity_id, 'related': related}, f)

            time.sleep(0.5)  # Rate limit
            return related

        except Exception as e:
            print(f"Error getting relations for '{entity_id}': {e}")
            return []

    def build_distance_graph(
        self,
        concepts: List[str],
        max_distance: int = 5,
        max_nodes_per_level: int = 100
    ) -> Dict[str, Dict[int, Set[str]]]:
        """
        Build graph with semantic distances from each concept.

        Returns:
            {
                'concept1': {
                    0: {'Q1234'},           # The concept itself
                    1: {'Q456', 'Q789'},    # Direct relations
                    2: {'Q111', 'Q222'},    # Distance-2
                    ...
                }
            }
        """
        graph = {}

        for concept in concepts:
            print(f"Building graph for: {concept}")

            # Get entity ID
            entity_id = self.search_entity(concept)
            if not entity_id:
                print(f"  ⚠ Could not find entity for '{concept}'")
                continue

            # BFS to build distance map
            distances = {0: {entity_id}}
            visited = {entity_id}
            queue = deque([(entity_id, 0)])

            while queue and len(visited) < 1000:  # Safety limit
                current_id, dist = queue.popleft()

                if dist >= max_distance:
                    continue

                # Get neighbors
                neighbors = self.get_related_entities(current_id)

                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)

                        new_dist = dist + 1
                        if new_dist not in distances:
                            distances[new_dist] = set()

                        distances[new_dist].add(neighbor_id)

                        # Limit nodes per level
                        if len(distances[new_dist]) < max_nodes_per_level:
                            queue.append((neighbor_id, new_dist))

            graph[concept] = {
                'entity_id': entity_id,
                'distances': {k: list(v) for k, v in distances.items()}
            }

            print(f"  ✓ Found entities at distances: {list(distances.keys())}")
            for dist, entities in distances.items():
                print(f"    Distance {dist}: {len(entities)} entities")

        return graph

    def sample_negatives(
        self,
        concept: str,
        graph: Dict,
        min_distance: int = 4,
        n_samples: int = 50
    ) -> List[str]:
        """
        Sample negative concepts with minimum semantic distance.

        Returns:
            List of entity IDs to use as negatives
        """
        if concept not in graph:
            return []

        concept_data = graph[concept]
        distances = concept_data['distances']

        # Collect all entities at distance >= min_distance
        candidates = []
        for dist, entities in distances.items():
            if int(dist) >= min_distance:
                candidates.extend(entities)

        # Sample randomly
        import random
        if len(candidates) > n_samples:
            return random.sample(candidates, n_samples)
        else:
            return candidates

    def get_entity_label(self, entity_id: str) -> str:
        """Get English label for an entity ID."""
        cache_file = self.cache_dir / f'label_{entity_id}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)['label']

        query = f"""
        SELECT ?label WHERE {{
            wd:{entity_id} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }}
        LIMIT 1
        """

        try:
            response = requests.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                timeout=10,
                headers={'User-Agent': 'HatCat/1.0'}
            )
            response.raise_for_status()
            data = response.json()

            if data['results']['bindings']:
                label = data['results']['bindings'][0]['label']['value']

                # Cache
                with open(cache_file, 'w') as f:
                    json.dump({'entity_id': entity_id, 'label': label}, f)

                time.sleep(0.1)
                return label

        except Exception as e:
            print(f"Error getting label for '{entity_id}': {e}")

        return entity_id  # Fallback to ID

    def save_graph(self, graph: Dict, output_path: Path):
        """Save graph to JSON."""
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"Graph saved to: {output_path}")

    def load_graph(self, graph_path: Path) -> Dict:
        """Load graph from JSON."""
        with open(graph_path) as f:
            return json.load(f)


if __name__ == '__main__':
    # Test with 10 concepts
    test_concepts = [
        'book', 'computer', 'phone', 'water', 'fire',
        'car', 'tree', 'house', 'cat', 'dog'
    ]

    print("=" * 70)
    print("BUILDING WIKIDATA CONCEPT GRAPH")
    print("=" * 70)
    print()

    builder = WikidataConceptGraph()

    print("Step 1: Building distance graph...")
    graph = builder.build_distance_graph(test_concepts, max_distance=5)

    print()
    print("Step 2: Saving graph...")
    builder.save_graph(graph, Path('data/concept_graph/test_10_concepts.json'))

    print()
    print("Step 3: Testing negative sampling...")
    for concept in test_concepts[:3]:
        print(f"\nConcept: {concept}")

        negatives = builder.sample_negatives(concept, graph, min_distance=4, n_samples=5)
        print(f"  Sampled {len(negatives)} negatives:")

        for neg_id in negatives[:5]:
            label = builder.get_entity_label(neg_id)
            print(f"    - {neg_id}: {label}")

    print()
    print("=" * 70)
    print("✓ CONCEPT GRAPH COMPLETE")
    print("=" * 70)
