"""
Steering Manager for session-based concept amplification/suppression.

Manages active steerings per conversation with support for:
- User steerings (high priority, persistent)
- Model steerings (lower priority, from tool calls)
- Session isolation
- Steering strength normalization
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class Steering:
    """Represents a single concept steering."""

    concept: str
    layer: int
    strength: float  # -1.0 to 1.0 (negative = suppress, positive = amplify)
    source: str  # "user" or "model"
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Contrastive steering support
    mode: str = "projection"  # "projection" (suppress) or "contrastive" (steer toward target)
    target: Optional[str] = None  # For contrastive mode: concept to steer toward

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'concept': self.concept,
            'layer': self.layer,
            'strength': self.strength,
            'source': self.source,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'mode': self.mode,
            'target': self.target,
        }


class SteeringManager:
    """
    Manages steering state across sessions.

    Features:
    - Session-based steering isolation
    - Priority system (user > model)
    - Steering composition (multiple steerings combine)
    - Automatic cleanup of old sessions
    """

    def __init__(self):
        # session_id -> list of steerings
        self.steerings: Dict[str, List[Steering]] = defaultdict(list)

        # Track session last activity for cleanup
        self.session_activity: Dict[str, datetime] = {}

    def add_steering(
        self,
        session_id: str,
        concept: str,
        layer: int,
        strength: float,
        source: str = "user",
        reason: str = "",
        mode: str = "projection",
        target: Optional[str] = None,
    ) -> Steering:
        """
        Add or update a steering for a session.

        Args:
            session_id: Conversation session ID
            concept: Concept name (e.g., "Proposition")
            layer: Layer to apply steering (0-5)
            strength: -1.0 to 1.0 (negative = suppress, positive = amplify)
            source: "user" or "model"
            reason: Human-readable explanation
            mode: "projection" (suppress concept) or "contrastive" (steer toward target)
            target: For contrastive mode, the concept to steer toward

        Returns:
            The created/updated Steering object
        """
        # Clamp strength
        strength = max(-1.0, min(1.0, strength))

        # Check if steering already exists for this concept/layer
        existing_idx = None
        for i, s in enumerate(self.steerings[session_id]):
            if s.concept == concept and s.layer == layer and s.source == source:
                existing_idx = i
                break

        steering = Steering(
            concept=concept,
            layer=layer,
            strength=strength,
            source=source,
            reason=reason,
            mode=mode,
            target=target,
        )

        if existing_idx is not None:
            # Update existing
            self.steerings[session_id][existing_idx] = steering
        else:
            # Add new
            self.steerings[session_id].append(steering)

        self.session_activity[session_id] = datetime.now()

        return steering

    def remove_steering(
        self,
        session_id: str,
        concept: str,
        layer: Optional[int] = None,
        source: Optional[str] = None,
    ) -> int:
        """
        Remove steering(s) for a concept.

        Args:
            session_id: Session ID
            concept: Concept name
            layer: Optional layer filter
            source: Optional source filter

        Returns:
            Number of steerings removed
        """
        if session_id not in self.steerings:
            return 0

        removed_count = 0
        new_steerings = []

        for s in self.steerings[session_id]:
            should_remove = (
                s.concept == concept
                and (layer is None or s.layer == layer)
                and (source is None or s.source == source)
            )

            if should_remove:
                removed_count += 1
            else:
                new_steerings.append(s)

        self.steerings[session_id] = new_steerings
        self.session_activity[session_id] = datetime.now()

        return removed_count

    def get_steerings(
        self,
        session_id: str,
        layer: Optional[int] = None,
    ) -> List[Steering]:
        """
        Get all active steerings for a session.

        Args:
            session_id: Session ID
            layer: Optional layer filter

        Returns:
            List of Steering objects
        """
        if session_id not in self.steerings:
            return []

        steerings = self.steerings[session_id]

        if layer is not None:
            steerings = [s for s in steerings if s.layer == layer]

        # Sort by priority: user > model, then by timestamp (newer first)
        steerings.sort(
            key=lambda s: (
                0 if s.source == "user" else 1,
                -s.timestamp.timestamp()
            )
        )

        return steerings

    def get_steering_vector(
        self,
        session_id: str,
        layer: int,
        concept_vectors: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Compute combined steering vector for a layer.

        Args:
            session_id: Session ID
            layer: Target layer
            concept_vectors: Dict mapping concept names to their vectors

        Returns:
            Combined steering vector or None if no steerings active
        """
        steerings = self.get_steerings(session_id, layer=layer)

        if not steerings:
            return None

        # Combine steerings (weighted sum)
        combined_vector = None
        total_weight = 0.0

        for steering in steerings:
            if steering.concept not in concept_vectors:
                continue

            vector = concept_vectors[steering.concept]
            weight = abs(steering.strength)
            direction = np.sign(steering.strength)

            weighted_vector = vector * weight * direction

            if combined_vector is None:
                combined_vector = weighted_vector
            else:
                combined_vector += weighted_vector

            total_weight += weight

        if combined_vector is None or total_weight == 0:
            return None

        # Normalize by total weight
        combined_vector = combined_vector / total_weight

        return combined_vector

    def clear_session(self, session_id: str):
        """Clear all steerings for a session."""
        if session_id in self.steerings:
            del self.steerings[session_id]
        if session_id in self.session_activity:
            del self.session_activity[session_id]

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove steerings for sessions inactive for > max_age_hours."""
        now = datetime.now()
        cutoff = now.timestamp() - (max_age_hours * 3600)

        sessions_to_remove = [
            sid for sid, last_active in self.session_activity.items()
            if last_active.timestamp() < cutoff
        ]

        for sid in sessions_to_remove:
            self.clear_session(sid)

        return len(sessions_to_remove)
