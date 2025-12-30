"""
Access control for ASK - manages actors and provides bounded views.

Implements the access control layer that sits between actors and audit data,
enforcing permissions and observability bounds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable
import secrets

from .actors import (
    Actor,
    ActorType,
    Permission,
    PermissionGrant,
    ObservabilityBounds,
)
from ..secrets.hashing import hash_content


@dataclass
class AccessRequest:
    """A request to access audit data."""

    request_id: str = ""
    actor_id: str = ""
    permission_required: Permission = Permission(0)

    # What's being accessed
    resource_type: str = ""  # "batch", "entry", "decision", etc.
    resource_id: str = ""

    # Context
    deployment_id: str = ""
    jurisdiction: str = ""

    # Timing
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req_{secrets.token_hex(8)}"


@dataclass
class AccessDecision:
    """Result of an access control decision."""

    request_id: str
    actor_id: str
    allowed: bool

    # Details
    permission_checked: Permission = Permission(0)
    bounds_applied: bool = False

    # Why
    reason: str = ""
    denied_permissions: List[str] = field(default_factory=list)

    # Audit
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "allowed": self.allowed,
            "permission_checked": self.permission_checked.value,
            "bounds_applied": self.bounds_applied,
            "reason": self.reason,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class BoundedView:
    """
    A view of audit data bounded by actor permissions.

    Applies redaction, filtering, and aggregation as required.
    """

    actor_id: str
    bounds: ObservabilityBounds

    # What was included
    batch_count: int = 0
    entry_count: int = 0

    # What was filtered/redacted
    batches_filtered: int = 0
    entries_redacted: int = 0
    fields_redacted: List[str] = field(default_factory=list)

    # The actual data (after filtering)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "batch_count": self.batch_count,
            "entry_count": self.entry_count,
            "batches_filtered": self.batches_filtered,
            "entries_redacted": self.entries_redacted,
            "fields_redacted": self.fields_redacted,
            "data": self.data,
        }


class AccessController:
    """
    Controls access to audit data based on actor permissions and bounds.
    """

    def __init__(self):
        # Registered actors
        self._actors: Dict[str, Actor] = {}

        # Permission grants history (for audit)
        self._grants: List[PermissionGrant] = []

        # Access decisions log
        self._decisions: List[AccessDecision] = []

        # Redaction functions
        self._redactors: Dict[str, Callable[[Any], Any]] = {
            "user_content": self._redact_user_content,
            "operator_id": self._redact_operator_id,
        }

    def register_actor(self, actor: Actor, registered_by: str = "") -> None:
        """Register an actor in the system."""
        self._actors[actor.actor_id] = actor

        # Log the grant
        if actor.permissions:
            grant = PermissionGrant(
                grant_type="grant",
                actor_id=actor.actor_id,
                granted_by=registered_by,
                permission=actor.permissions,
                reason="Initial registration",
            )
            self._grants.append(grant)

    def get_actor(self, actor_id: str) -> Optional[Actor]:
        """Get an actor by ID."""
        return self._actors.get(actor_id)

    def revoke_actor(self, actor_id: str, revoked_by: str, reason: str = "") -> bool:
        """Revoke an actor's access."""
        actor = self._actors.get(actor_id)
        if not actor:
            return False

        actor.revoke(revoked_by)

        # Log the revocation
        grant = PermissionGrant(
            grant_type="revoke",
            actor_id=actor_id,
            granted_by=revoked_by,
            permission=actor.permissions,
            reason=reason,
        )
        self._grants.append(grant)

        return True

    def grant_permission(
        self,
        actor_id: str,
        permission: Permission,
        granted_by: str,
        reason: str = "",
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """Grant additional permission to an actor."""
        actor = self._actors.get(actor_id)
        if not actor:
            return False

        # Check granter has META_GRANT permission
        granter = self._actors.get(granted_by)
        if granter and not granter.has_permission(Permission.META_GRANT):
            return False

        actor.grant_permission(permission)

        grant = PermissionGrant(
            grant_type="grant",
            actor_id=actor_id,
            granted_by=granted_by,
            permission=permission,
            expires_at=expires_at,
            reason=reason,
        )
        self._grants.append(grant)

        return True

    def revoke_permission(
        self,
        actor_id: str,
        permission: Permission,
        revoked_by: str,
        reason: str = "",
    ) -> bool:
        """Revoke a specific permission from an actor."""
        actor = self._actors.get(actor_id)
        if not actor:
            return False

        # Check revoker has META_REVOKE permission
        revoker = self._actors.get(revoked_by)
        if revoker and not revoker.has_permission(Permission.META_REVOKE):
            return False

        actor.revoke_permission(permission)

        grant = PermissionGrant(
            grant_type="revoke",
            actor_id=actor_id,
            granted_by=revoked_by,
            permission=permission,
            reason=reason,
        )
        self._grants.append(grant)

        return True

    def check_access(self, request: AccessRequest) -> AccessDecision:
        """Check if an access request should be allowed."""
        actor = self._actors.get(request.actor_id)

        if not actor:
            decision = AccessDecision(
                request_id=request.request_id,
                actor_id=request.actor_id,
                allowed=False,
                reason="Unknown actor",
            )
            self._decisions.append(decision)
            return decision

        if not actor.active:
            decision = AccessDecision(
                request_id=request.request_id,
                actor_id=request.actor_id,
                allowed=False,
                reason="Actor revoked",
            )
            self._decisions.append(decision)
            return decision

        # Check permission
        has_permission = actor.has_permission(request.permission_required)

        # Check bounds if applicable
        bounds_ok = True
        if actor.bounds:
            bounds_ok = actor.bounds.allows_batch(
                request.resource_id,
                request.deployment_id,
                request.jurisdiction,
                None,
            )

        allowed = has_permission and bounds_ok

        decision = AccessDecision(
            request_id=request.request_id,
            actor_id=request.actor_id,
            allowed=allowed,
            permission_checked=request.permission_required,
            bounds_applied=actor.bounds is not None,
            reason="" if allowed else "Permission denied" if not has_permission else "Outside bounds",
        )

        self._decisions.append(decision)
        return decision

    def create_bounded_view(
        self,
        actor_id: str,
        batches: List[Dict[str, Any]],
        entries: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Optional[BoundedView]:
        """
        Create a bounded view of audit data for an actor.

        Applies the actor's observability bounds to filter and redact data.
        """
        actor = self._actors.get(actor_id)
        if not actor or not actor.active:
            return None

        bounds = actor.bounds or ObservabilityBounds()

        view = BoundedView(
            actor_id=actor_id,
            bounds=bounds,
        )

        filtered_batches = []
        entries = entries or {}

        for batch in batches:
            # Check if batch is within bounds
            batch_id = batch.get("batch_id", "")
            deployment_id = batch.get("deployment_id", "")
            jurisdiction = batch.get("jurisdiction", "")
            sealed_at = None
            if batch.get("sealed_at"):
                try:
                    sealed_at = datetime.fromisoformat(batch["sealed_at"])
                except (ValueError, TypeError):
                    pass

            if not bounds.allows_batch(batch_id, deployment_id, jurisdiction, sealed_at):
                view.batches_filtered += 1
                continue

            # Apply content filtering
            filtered_batch = self._filter_batch(batch, bounds, actor.permissions)

            # Apply redaction
            if bounds.redact_user_content:
                filtered_batch = self._apply_redaction(filtered_batch, "user_content")
                if "user_content" not in view.fields_redacted:
                    view.fields_redacted.append("user_content")

            if bounds.redact_operator_ids:
                filtered_batch = self._apply_redaction(filtered_batch, "operator_id")
                if "operator_id" not in view.fields_redacted:
                    view.fields_redacted.append("operator_id")

            filtered_batches.append(filtered_batch)
            view.batch_count += 1
            view.entry_count += batch.get("entry_count", 0)

        view.data = {"batches": filtered_batches}

        # Aggregate if required
        if bounds.aggregate_only and view.batch_count >= bounds.min_aggregation_size:
            view.data = self._aggregate_data(view.data)

        return view

    def _filter_batch(
        self,
        batch: Dict[str, Any],
        bounds: ObservabilityBounds,
        permissions: Permission,
    ) -> Dict[str, Any]:
        """Filter batch content based on bounds and permissions."""
        filtered = dict(batch)

        # Remove entries if not permitted
        if not bounds.include_entries or not (permissions & Permission.OBSERVE_ENTRIES):
            filtered.pop("entries", None)
            filtered.pop("entry_hashes", None)

        # Remove signals if not permitted
        if not bounds.include_signals or not (permissions & Permission.OBSERVE_SIGNALS):
            filtered.pop("signals_summary", None)

        # Remove decisions if not permitted
        if not bounds.include_decisions or not (permissions & Permission.OBSERVE_DECISIONS):
            filtered.pop("decisions", None)
            filtered["decision_count"] = 0

        # Remove steering if not permitted
        if not bounds.include_steering or not (permissions & Permission.OBSERVE_STEERING):
            filtered.pop("steering_interventions", None)
            filtered["steering_count"] = 0

        return filtered

    def _apply_redaction(
        self,
        data: Dict[str, Any],
        redaction_type: str,
    ) -> Dict[str, Any]:
        """Apply redaction to data."""
        redactor = self._redactors.get(redaction_type)
        if redactor:
            return redactor(data)
        return data

    def _redact_user_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact user content from data."""
        redacted = dict(data)

        # Redact common user content fields
        for field in ["user_input", "user_output", "prompt", "response", "content"]:
            if field in redacted:
                redacted[field] = "[REDACTED]"

        # Recursively redact in entries
        if "entries" in redacted and isinstance(redacted["entries"], list):
            redacted["entries"] = [
                self._redact_user_content(e) if isinstance(e, dict) else e
                for e in redacted["entries"]
            ]

        return redacted

    def _redact_operator_id(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pseudonymize operator IDs."""
        redacted = dict(data)

        if "operator_id" in redacted:
            # Already should be hashed, but ensure it's marked
            redacted["operator_id"] = f"[PSEUDONYMIZED:{redacted['operator_id'][:16]}...]"

        return redacted

    def _aggregate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data for privacy."""
        batches = data.get("batches", [])

        return {
            "aggregated": True,
            "batch_count": len(batches),
            "total_entries": sum(b.get("entry_count", 0) for b in batches),
            "total_decisions": sum(b.get("decision_count", 0) for b in batches),
            "total_violations": sum(b.get("violation_count", 0) for b in batches),
            "period": {
                "start": min((b.get("sealed_at") for b in batches if b.get("sealed_at")), default=None),
                "end": max((b.get("sealed_at") for b in batches if b.get("sealed_at")), default=None),
            },
        }

    def get_grant_history(self, actor_id: Optional[str] = None) -> List[PermissionGrant]:
        """Get permission grant history."""
        if actor_id:
            return [g for g in self._grants if g.actor_id == actor_id]
        return list(self._grants)

    def get_access_log(
        self,
        actor_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[AccessDecision]:
        """Get access decision log."""
        decisions = self._decisions

        if actor_id:
            decisions = [d for d in decisions if d.actor_id == actor_id]

        if since:
            decisions = [d for d in decisions if d.decided_at >= since]

        return decisions

    def get_actors_by_type(self, actor_type: ActorType) -> List[Actor]:
        """Get all actors of a specific type."""
        return [a for a in self._actors.values() if a.actor_type == actor_type]

    def get_active_actors(self) -> List[Actor]:
        """Get all active actors."""
        return [a for a in self._actors.values() if a.active]


# Singleton controller for convenience
_default_controller: Optional[AccessController] = None


def get_access_controller() -> AccessController:
    """Get the default access controller."""
    global _default_controller
    if _default_controller is None:
        _default_controller = AccessController()
    return _default_controller


def reset_access_controller() -> None:
    """Reset the default access controller (for testing)."""
    global _default_controller
    _default_controller = None
