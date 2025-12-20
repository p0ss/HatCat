"""
TagIndex - Folksonomy management for XDB.

Provides:
- Tag creation and lookup
- Concept pack integration
- Bud management for graft training
- In-memory working set for performance
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import OrderedDict
import json
import logging

try:
    import duckdb
except ImportError:
    duckdb = None

from .models import (
    Tag,
    TagType,
    BudStatus,
    TagApplication,
    TargetType,
    TagSource,
    TimestepRecord,
)

logger = logging.getLogger(__name__)


class InMemoryTagIndex:
    """
    In-memory working set of tags for fast access.

    Only loads:
    - Concept pack primary terms (ones with lenses)
    - Current candidate buds
    - Recently used custom/entity tags

    Full tag data lives in DuckDB; this is a cache.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the in-memory tag index.

        Args:
            max_size: Maximum tags to keep in memory
        """
        self.max_size = max_size

        # LRU-ordered dict: tag_id -> Tag
        self.active_tags: OrderedDict[str, Tag] = OrderedDict()

        # Concept hierarchy (from concept pack)
        self.concept_hierarchy: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.concept_parents: Dict[str, str] = {}  # child_id -> parent_id

        # Fast lookups
        self.name_to_id: Dict[str, str] = {}  # tag_name -> tag_id
        self.concept_to_tag: Dict[str, str] = {}  # concept_id -> tag_id

        # Buds tracking
        self.active_buds: Set[str] = set()

    def add(self, tag: Tag):
        """Add a tag to the working set."""
        # Remove if already present (to update order)
        if tag.id in self.active_tags:
            del self.active_tags[tag.id]

        # Add to end (most recently used)
        self.active_tags[tag.id] = tag
        self.name_to_id[tag.name.lower()] = tag.id

        if tag.concept_id:
            self.concept_to_tag[tag.concept_id] = tag.id

        if tag.tag_type == TagType.BUD and tag.bud_status in (BudStatus.COLLECTING, BudStatus.READY):
            self.active_buds.add(tag.id)

        # Evict if over size
        self._evict_if_needed()

    def get(self, tag_id: str) -> Optional[Tag]:
        """Get a tag by ID, updating LRU order."""
        if tag_id in self.active_tags:
            # Move to end (most recently used)
            tag = self.active_tags.pop(tag_id)
            self.active_tags[tag_id] = tag
            return tag
        return None

    def get_by_name(self, name: str) -> Optional[Tag]:
        """Get a tag by name."""
        tag_id = self.name_to_id.get(name.lower())
        if tag_id:
            return self.get(tag_id)
        return None

    def get_by_concept(self, concept_id: str) -> Optional[Tag]:
        """Get a tag by concept ID."""
        tag_id = self.concept_to_tag.get(concept_id)
        if tag_id:
            return self.get(tag_id)
        return None

    def remove(self, tag_id: str):
        """Remove a tag from the working set."""
        if tag_id in self.active_tags:
            tag = self.active_tags.pop(tag_id)
            self.name_to_id.pop(tag.name.lower(), None)
            if tag.concept_id:
                self.concept_to_tag.pop(tag.concept_id, None)
            self.active_buds.discard(tag_id)

    def _evict_if_needed(self):
        """Evict oldest tags if over max size."""
        while len(self.active_tags) > self.max_size:
            # Pop from start (oldest)
            tag_id, tag = self.active_tags.popitem(last=False)
            self.name_to_id.pop(tag.name.lower(), None)
            if tag.concept_id:
                self.concept_to_tag.pop(tag.concept_id, None)
            # Keep buds in tracking even if evicted from cache
            # (they'll be reloaded when needed)

    def load_concept_pack_primaries(self, pack_path: Path, primary_concepts: List[str]):
        """
        Load only primary terms from concept pack.

        Args:
            pack_path: Path to concept pack
            primary_concepts: List of concept IDs that have lenses
        """
        try:
            # Try to load concept pack structure
            graph_path = pack_path / "concept_graph.json"
            if not graph_path.exists():
                logger.warning(f"No concept graph at {graph_path}")
                return

            with open(graph_path) as f:
                graph_data = json.load(f)

            # Load hierarchy
            if 'hierarchy' in graph_data:
                for edge in graph_data['hierarchy']:
                    parent = edge.get('parent')
                    child = edge.get('child')
                    if parent and child:
                        if parent not in self.concept_hierarchy:
                            self.concept_hierarchy[parent] = []
                        self.concept_hierarchy[parent].append(child)
                        self.concept_parents[child] = parent

            # Create tags for primary concepts only
            concepts = graph_data.get('concepts', {})
            for concept_id in primary_concepts:
                if concept_id in concepts:
                    concept_data = concepts[concept_id]
                    tag = Tag(
                        id=f"concept-{concept_id}",
                        name=concept_data.get('name', concept_id),
                        tag_type=TagType.CONCEPT,
                        concept_id=concept_id,
                        description=concept_data.get('description'),
                        created_by="system",
                    )
                    self.add(tag)

            logger.info(f"Loaded {len(primary_concepts)} primary concepts from {pack_path}")

        except Exception as e:
            logger.error(f"Failed to load concept pack: {e}")

    def get_concept_children(self, concept_id: str) -> List[str]:
        """Get child concept IDs."""
        return self.concept_hierarchy.get(concept_id, [])

    def get_concept_parent(self, concept_id: str) -> Optional[str]:
        """Get parent concept ID."""
        return self.concept_parents.get(concept_id)

    def find_by_type(self, tag_type: TagType) -> List[Tag]:
        """Find all tags of a type in the working set."""
        return [t for t in self.active_tags.values() if t.tag_type == tag_type]

    def find_by_pattern(self, pattern: str) -> List[Tag]:
        """Find tags matching a name pattern (case-insensitive prefix match)."""
        pattern_lower = pattern.lower()
        return [
            t for t in self.active_tags.values()
            if t.name.lower().startswith(pattern_lower)
        ]

    def get_active_buds(self) -> List[Tag]:
        """Get all active bud tags."""
        return [
            self.active_tags[bid]
            for bid in self.active_buds
            if bid in self.active_tags
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        by_type = {}
        for tag in self.active_tags.values():
            type_name = tag.tag_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            'total_cached': len(self.active_tags),
            'max_size': self.max_size,
            'by_type': by_type,
            'concept_hierarchy_size': len(self.concept_hierarchy),
            'active_buds': len(self.active_buds),
        }


class TagIndex:
    """
    Full tag index backed by DuckDB with in-memory cache.

    Provides:
    - Tag CRUD operations
    - Concept pack integration
    - Bud management
    - Search and navigation
    """

    def __init__(
        self,
        db_connection: Optional['duckdb.DuckDBPyConnection'] = None,
        concept_pack_path: Optional[Path] = None,
        cache_size: int = 10000,
    ):
        """
        Initialize the tag index.

        Args:
            db_connection: DuckDB connection (shared with ExperienceLog)
            concept_pack_path: Optional path to concept pack
            cache_size: Max tags in memory cache
        """
        self._connection = db_connection
        self.concept_pack_path = concept_pack_path

        # In-memory cache
        self.cache = InMemoryTagIndex(max_size=cache_size)

        # Load concept pack if provided
        if concept_pack_path:
            self._load_concept_pack()

    def set_connection(self, conn: 'duckdb.DuckDBPyConnection'):
        """Set the database connection."""
        self._connection = conn

    def _load_concept_pack(self):
        """Load concept pack structure into cache."""
        if not self.concept_pack_path:
            return

        try:
            # Find which concepts have lenses
            lens_path = self.concept_pack_path / "lenses"
            primary_concepts = []

            if lens_path.exists():
                for lens_file in lens_path.glob("*.pt"):
                    # Extract concept name from lens file
                    concept_name = lens_file.stem
                    primary_concepts.append(concept_name)

            if primary_concepts:
                self.cache.load_concept_pack_primaries(
                    self.concept_pack_path,
                    primary_concepts,
                )
        except Exception as e:
            logger.error(f"Failed to load concept pack: {e}")

    # =========================================================================
    # Tag CRUD
    # =========================================================================

    def create_tag(
        self,
        name: str,
        tag_type: TagType,
        *,
        concept_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        description: Optional[str] = None,
        created_by: str = "system",
    ) -> Tag:
        """
        Create a new tag.

        Args:
            name: Human-readable name
            tag_type: Type of tag
            concept_id: For CONCEPT tags, the concept pack ID
            entity_type: For ENTITY tags, the entity type
            description: Optional description
            created_by: Creator ID

        Returns:
            The created tag
        """
        tag = Tag(
            id=Tag.generate_id(tag_type, name),
            name=name,
            tag_type=tag_type,
            concept_id=concept_id,
            entity_type=entity_type,
            description=description,
            created_by=created_by,
            created_at=datetime.now(),
        )

        # Set initial bud status
        if tag_type == TagType.BUD:
            tag.bud_status = BudStatus.COLLECTING

        # Store in database
        if self._connection:
            try:
                self._connection.execute("""
                    INSERT INTO tags (
                        id, name, tag_type, concept_id, entity_type,
                        bud_status, created_at, created_by, description,
                        use_count, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    tag.id, tag.name, tag.tag_type.value, tag.concept_id,
                    tag.entity_type, tag.bud_status.value if tag.bud_status else None,
                    tag.created_at, tag.created_by, tag.description,
                    tag.use_count, tag.last_used,
                ])
            except Exception as e:
                logger.error(f"Failed to create tag in DB: {e}")

        # Add to cache
        self.cache.add(tag)

        return tag

    def get_tag(self, tag_id: str) -> Optional[Tag]:
        """Get a tag by ID."""
        # Check cache first
        tag = self.cache.get(tag_id)
        if tag:
            return tag

        # Load from database
        if self._connection:
            try:
                result = self._connection.execute("""
                    SELECT id, name, tag_type, concept_id, entity_type,
                           bud_status, created_at, created_by, description,
                           use_count, last_used
                    FROM tags
                    WHERE id = ?
                """, [tag_id]).fetchone()

                if result:
                    tag = self._row_to_tag(result)
                    self.cache.add(tag)
                    return tag
            except Exception as e:
                logger.error(f"Failed to get tag: {e}")

        return None

    def get_by_name(self, name: str) -> Optional[Tag]:
        """Get a tag by name."""
        # Check cache first
        tag = self.cache.get_by_name(name)
        if tag:
            return tag

        # Load from database
        if self._connection:
            try:
                result = self._connection.execute("""
                    SELECT id, name, tag_type, concept_id, entity_type,
                           bud_status, created_at, created_by, description,
                           use_count, last_used
                    FROM tags
                    WHERE LOWER(name) = LOWER(?)
                """, [name]).fetchone()

                if result:
                    tag = self._row_to_tag(result)
                    self.cache.add(tag)
                    return tag
            except Exception as e:
                logger.error(f"Failed to get tag by name: {e}")

        return None

    def find_tags(
        self,
        *,
        name_pattern: Optional[str] = None,
        tag_type: Optional[TagType] = None,
        bud_status: Optional[BudStatus] = None,
        limit: int = 100,
    ) -> List[Tag]:
        """
        Search for tags.

        Args:
            name_pattern: Pattern to match names (prefix match)
            tag_type: Filter by type
            bud_status: Filter by bud status

        Returns:
            List of matching tags
        """
        if self._connection:
            try:
                conditions = []
                params = []

                if name_pattern:
                    conditions.append("LOWER(name) LIKE LOWER(?)")
                    params.append(f"{name_pattern}%")

                if tag_type:
                    conditions.append("tag_type = ?")
                    params.append(tag_type.value)

                if bud_status:
                    conditions.append("bud_status = ?")
                    params.append(bud_status.value)

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                results = self._connection.execute(f"""
                    SELECT id, name, tag_type, concept_id, entity_type,
                           bud_status, created_at, created_by, description,
                           use_count, last_used
                    FROM tags
                    WHERE {where_clause}
                    ORDER BY use_count DESC, name
                    LIMIT ?
                """, params + [limit]).fetchall()

                tags = [self._row_to_tag(row) for row in results]

                # Add to cache
                for tag in tags:
                    self.cache.add(tag)

                return tags

            except Exception as e:
                logger.error(f"Failed to find tags: {e}")

        # Fall back to cache search
        results = []
        for tag in self.cache.active_tags.values():
            if name_pattern and not tag.name.lower().startswith(name_pattern.lower()):
                continue
            if tag_type and tag.tag_type != tag_type:
                continue
            if bud_status and tag.bud_status != bud_status:
                continue
            results.append(tag)
            if len(results) >= limit:
                break

        return results

    def update_tag(self, tag: Tag) -> bool:
        """Update a tag in the database."""
        if self._connection:
            try:
                self._connection.execute("""
                    UPDATE tags
                    SET name = ?, tag_type = ?, concept_id = ?, entity_type = ?,
                        bud_status = ?, description = ?, use_count = ?, last_used = ?
                    WHERE id = ?
                """, [
                    tag.name, tag.tag_type.value, tag.concept_id, tag.entity_type,
                    tag.bud_status.value if tag.bud_status else None,
                    tag.description, tag.use_count, tag.last_used, tag.id,
                ])
                self.cache.add(tag)
                return True
            except Exception as e:
                logger.error(f"Failed to update tag: {e}")

        return False

    def delete_tag(self, tag_id: str) -> bool:
        """Delete a tag."""
        if self._connection:
            try:
                # First delete applications
                self._connection.execute("""
                    DELETE FROM tag_applications WHERE tag_id = ?
                """, [tag_id])

                # Then delete tag
                self._connection.execute("""
                    DELETE FROM tags WHERE id = ?
                """, [tag_id])

                self.cache.remove(tag_id)
                return True
            except Exception as e:
                logger.error(f"Failed to delete tag: {e}")

        return False

    def _row_to_tag(self, row: tuple) -> Tag:
        """Convert a database row to a Tag."""
        bud_status = None
        if row[5]:
            bud_status = BudStatus(row[5])

        last_used = None
        if row[10]:
            last_used = row[10] if isinstance(row[10], datetime) else datetime.fromisoformat(str(row[10]))

        return Tag(
            id=row[0],
            name=row[1],
            tag_type=TagType(row[2]),
            concept_id=row[3],
            entity_type=row[4],
            bud_status=bud_status,
            created_at=row[6] if isinstance(row[6], datetime) else datetime.fromisoformat(str(row[6])),
            created_by=row[7] or "system",
            description=row[8],
            use_count=row[9] or 0,
            last_used=last_used,
        )

    # =========================================================================
    # Concept Navigation
    # =========================================================================

    def get_concept_children(self, concept_id: str) -> List[Tag]:
        """Get child concepts in the hierarchy."""
        child_ids = self.cache.get_concept_children(concept_id)
        children = []

        for child_id in child_ids:
            tag = self.cache.get_by_concept(child_id)
            if tag:
                children.append(tag)
            else:
                # Try to load from database
                tag = self.get_by_concept(child_id)
                if tag:
                    children.append(tag)

        return children

    def get_concept_parents(self, concept_id: str) -> List[Tag]:
        """Get parent concept in the hierarchy."""
        parent_id = self.cache.get_concept_parent(concept_id)
        if parent_id:
            parent_tag = self.cache.get_by_concept(parent_id)
            if parent_tag:
                return [parent_tag]
            # Try to load from database
            parent_tag = self.get_by_concept(parent_id)
            if parent_tag:
                return [parent_tag]
        return []

    def get_by_concept(self, concept_id: str) -> Optional[Tag]:
        """Get tag for a concept ID."""
        tag = self.cache.get_by_concept(concept_id)
        if tag:
            return tag

        # Try database
        if self._connection:
            try:
                result = self._connection.execute("""
                    SELECT id, name, tag_type, concept_id, entity_type,
                           bud_status, created_at, created_by, description,
                           use_count, last_used
                    FROM tags
                    WHERE concept_id = ?
                """, [concept_id]).fetchone()

                if result:
                    tag = self._row_to_tag(result)
                    self.cache.add(tag)
                    return tag
            except Exception as e:
                logger.error(f"Failed to get tag by concept: {e}")

        return None

    def get_related_concepts(self, concept_id: str) -> List[Tuple[Tag, str]]:
        """
        Get related concepts with relationship type.

        Returns list of (tag, relationship_type) tuples.
        """
        related = []

        # Get children
        for child in self.get_concept_children(concept_id):
            related.append((child, "child"))

        # Get parents
        for parent in self.get_concept_parents(concept_id):
            related.append((parent, "parent"))

        return related

    def graph_neighborhood(
        self,
        seed_ids: List[str],
        max_depth: int = 2,
        direction: str = "both",
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Walk the concept graph from seed nodes.

        Args:
            seed_ids: Concept IDs to start from
            max_depth: Maximum depth to traverse
            direction: "both", "ancestors", or "descendants"
            max_nodes: Maximum nodes to return

        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        nodes = []
        edges = []
        visited = set()
        node_info = {}  # id -> node dict

        def add_node(concept_id: str, depth: int):
            """Add a node if not already visited."""
            if concept_id in visited or len(nodes) >= max_nodes:
                return False
            visited.add(concept_id)

            # Get tag info
            tag = self.get_by_concept(concept_id)
            name = tag.name if tag else concept_id.split("::")[-1] if "::" in concept_id else concept_id

            node = {
                "id": concept_id,
                "name": name,
                "type": "concept",
                "depth": depth,
            }
            nodes.append(node)
            node_info[concept_id] = node
            return True

        def traverse(concept_id: str, depth: int):
            """Recursively traverse the graph."""
            if depth > max_depth or len(nodes) >= max_nodes:
                return

            added = add_node(concept_id, depth)
            if not added and depth > 0:
                return  # Already visited

            # Traverse ancestors (parents)
            if direction in ("both", "ancestors"):
                parent_id = self.cache.get_concept_parent(concept_id)
                if parent_id and parent_id not in visited:
                    if add_node(parent_id, depth + 1):
                        edges.append({
                            "source": concept_id,
                            "target": parent_id,
                            "relation": "parent",
                        })
                        if depth + 1 < max_depth:
                            traverse(parent_id, depth + 1)

            # Traverse descendants (children)
            if direction in ("both", "descendants"):
                child_ids = self.cache.get_concept_children(concept_id)
                for child_id in child_ids:
                    if child_id not in visited and len(nodes) < max_nodes:
                        if add_node(child_id, depth + 1):
                            edges.append({
                                "source": concept_id,
                                "target": child_id,
                                "relation": "child",
                            })
                            if depth + 1 < max_depth:
                                traverse(child_id, depth + 1)

        # Start from each seed
        for seed_id in seed_ids:
            if len(nodes) >= max_nodes:
                break
            traverse(seed_id, 0)

        return {
            "nodes": nodes,
            "edges": edges,
        }

    # =========================================================================
    # Bud Management
    # =========================================================================

    def promote_to_bud(self, tag_id: str) -> Optional[Tag]:
        """Mark a tag as a bud candidate."""
        tag = self.get_tag(tag_id)
        if not tag:
            return None

        tag.tag_type = TagType.BUD
        tag.bud_status = BudStatus.COLLECTING
        self.update_tag(tag)
        self.cache.active_buds.add(tag_id)

        return tag

    def update_bud_status(self, tag_id: str, status: BudStatus) -> Optional[Tag]:
        """Update bud training status."""
        tag = self.get_tag(tag_id)
        if not tag or tag.tag_type != TagType.BUD:
            return None

        tag.bud_status = status
        self.update_tag(tag)

        # Update active buds tracking
        if status in (BudStatus.COLLECTING, BudStatus.READY):
            self.cache.active_buds.add(tag_id)
        else:
            self.cache.active_buds.discard(tag_id)

        return tag

    def get_bud_examples(
        self,
        tag_id: str,
        experience_log: Optional['ExperienceLog'] = None,
    ) -> List[TimestepRecord]:
        """
        Get all timesteps tagged with this bud.

        Args:
            tag_id: The bud tag ID
            experience_log: ExperienceLog to query

        Returns:
            List of timestep records
        """
        if not experience_log or not self._connection:
            return []

        try:
            # Get all timestep IDs tagged with this bud
            results = self._connection.execute("""
                SELECT DISTINCT ts.id, ts.xdb_id, ts.tick, ts.ts,
                       ts.event_type, ts.content, ts.event_id, ts.event_start,
                       ts.event_end, ts.token_id, ts.role, ts.fidelity,
                       ts.concept_activations
                FROM timesteps ts
                JOIN tag_applications ta ON ta.timestep_id = ts.id
                WHERE ta.tag_id = ?
                ORDER BY ts.xdb_id, ts.tick
            """, [tag_id]).fetchall()

            timesteps = []
            for row in results:
                import json
                concept_activations = {}
                if row[12]:
                    try:
                        concept_activations = json.loads(row[12])
                    except (json.JSONDecodeError, TypeError):
                        pass

                from .models import EventType, Fidelity
                timesteps.append(TimestepRecord(
                    id=row[0],
                    xdb_id=row[1],
                    tick=row[2],
                    timestamp=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
                    event_type=EventType(row[4]),
                    content=row[5] or "",
                    event_id=row[6],
                    event_start=row[7] or False,
                    event_end=row[8] or False,
                    token_id=row[9],
                    role=row[10],
                    fidelity=Fidelity(row[11] or "hot"),
                    concept_activations=concept_activations,
                ))

            return timesteps

        except Exception as e:
            logger.error(f"Failed to get bud examples: {e}")
            return []

    def get_buds(self, status: Optional[BudStatus] = None) -> List[Tag]:
        """Get bud tags, optionally filtered by status."""
        return self.find_tags(tag_type=TagType.BUD, bud_status=status)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get tag index statistics."""
        stats = self.cache.get_stats()

        if self._connection:
            try:
                # Total tags in database
                result = self._connection.execute("""
                    SELECT COUNT(*) FROM tags
                """).fetchone()
                stats['total_tags'] = result[0] if result else 0

                # Tags by type
                result = self._connection.execute("""
                    SELECT tag_type, COUNT(*)
                    FROM tags
                    GROUP BY tag_type
                """).fetchall()
                stats['db_by_type'] = {row[0]: row[1] for row in result}

                # Buds by status
                result = self._connection.execute("""
                    SELECT bud_status, COUNT(*)
                    FROM tags
                    WHERE tag_type = 'bud'
                    GROUP BY bud_status
                """).fetchall()
                stats['buds_by_status'] = {row[0]: row[1] for row in result}

            except Exception as e:
                logger.error(f"Failed to get tag stats: {e}")

        return stats
