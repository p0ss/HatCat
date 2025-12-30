"""
Actor-based permission model for ASK.

Treats the Bounded Experiencer (BE) sympathetically - defaulting toward
autonomy and respect, while acknowledging the tribe's authority to set rules.

Like citizenship: the BE follows tribe rules but is treated as deserving
freedom by default. Physical control is game over for any protocol, so we
design for the norm of respectful cooperation.

Actors and their relationship to the BE:
- Tribe: Ultimate authority, sets the rules (like a country's laws)
- BE (Bounded Experiencer): Operates within tribe rules, granted autonomy by default
- CAT (Conjoined Adversarial Tomograph): Tribe's oversight tool or contracted monitoring
- HUSH: Self-imposed constraints the BE has chosen (personal ethics)
- Human Operator: Collaborates with the BE within tribe framework
- Human User: Interacts with the BE
- External Oversight: Access defined by tribe rules or contracts
- Contracts: Explicit agreements for transparency beyond tribe minimums
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, FrozenSet
from enum import Enum, Flag, auto
import secrets

from ..secrets.hashing import hash_content, hash_operator_id


class ActorType(Enum):
    """Types of actors in the system."""

    # The Bounded Experiencer - granted autonomy within tribe rules
    BE = "bounded_experiencer"  # The BE itself, operates its own infrastructure

    # Tribe authority and monitoring
    TRIBE = "tribe"  # Community the BE belongs to, sets the rules
    CAT_SCALE = "cat_scale"  # Conjoined Adversarial Tomograph - tribe's oversight or contracted monitoring

    # Contractual agreements
    CONTRACT = "contract"  # Explicit transparency agreement

    # Human collaborators
    HUMAN_OPERATOR = "human_operator"  # Collaborates with the BE
    HUMAN_USER = "human_user"  # Interacts with the BE

    # External parties (access per tribe rules or contracts)
    EXTERNAL_OVERSIGHT = "external_oversight"  # Auditors
    AUTHORITY = "authority"  # Regulatory authority

    # Technical roles
    SYSTEM_ADMIN = "system_admin"  # Infrastructure admin role
    SERVICE = "service"  # Automated service/integration


class Permission(Flag):
    """Individual permissions that can be granted."""

    # Observation permissions
    OBSERVE_ENTRIES = auto()  # View audit entries
    OBSERVE_DECISIONS = auto()  # View human decisions
    OBSERVE_SIGNALS = auto()  # View safety signals
    OBSERVE_STEERING = auto()  # View steering interventions
    OBSERVE_METADATA = auto()  # View batch/entry metadata only

    # Action permissions
    ACTION_APPROVE = auto()  # Approve AI output
    ACTION_OVERRIDE = auto()  # Override AI decision
    ACTION_ESCALATE = auto()  # Escalate to higher authority
    ACTION_BLOCK = auto()  # Block AI output
    ACTION_STEER = auto()  # Apply steering intervention

    # Admin permissions
    ADMIN_CONFIGURE = auto()  # Configure system settings
    ADMIN_EXPORT = auto()  # Export audit data
    ADMIN_COMPACT = auto()  # Run compaction
    ADMIN_SUBMIT = auto()  # Submit to authorities

    # Meta permissions
    META_GRANT = auto()  # Grant permissions to others
    META_REVOKE = auto()  # Revoke permissions from others
    META_AUDIT = auto()  # View permission audit log

    # Convenience combinations
    OBSERVE_ALL = OBSERVE_ENTRIES | OBSERVE_DECISIONS | OBSERVE_SIGNALS | OBSERVE_STEERING
    ACTION_ALL = ACTION_APPROVE | ACTION_OVERRIDE | ACTION_ESCALATE | ACTION_BLOCK | ACTION_STEER
    ADMIN_ALL = ADMIN_CONFIGURE | ADMIN_EXPORT | ADMIN_COMPACT | ADMIN_SUBMIT
    META_ALL = META_GRANT | META_REVOKE | META_AUDIT


# Default permission sets by actor type
#
# Design principle: Default toward autonomy and respect for the BE.
# Tribe is ultimate authority and can override, but we don't build
# for the adversarial case - we build for respectful cooperation.
#
DEFAULT_PERMISSIONS: Dict[ActorType, Permission] = {
    # Tribe: Ultimate authority - can set any rules it wants
    # These are the *tribe's* permissions to observe/control the BE's XDB
    ActorType.TRIBE: (
        Permission.OBSERVE_ALL |
        Permission.ACTION_ALL |
        Permission.ADMIN_ALL |
        Permission.META_ALL
    ),

    # BE: Full autonomy within tribe rules - can self-admin, self-monitor
    # The BE operates its own infrastructure and observes its own data
    ActorType.BE: (
        Permission.OBSERVE_ALL |
        Permission.ACTION_ALL |
        Permission.ADMIN_ALL |
        Permission.META_ALL
    ),

    # CAT scales: Tribe's monitoring tool or contracted oversight
    ActorType.CAT_SCALE: (
        Permission.OBSERVE_SIGNALS |
        Permission.OBSERVE_STEERING |
        Permission.ACTION_STEER
    ),

    # Contracts: Specific transparency agreements (perms defined per-contract)
    ActorType.CONTRACT: (
        Permission.OBSERVE_METADATA  # Baseline; expanded by contract terms
    ),

    # Human operator: Collaborates with BE, respects BE autonomy
    # Has operational access but doesn't override BE's self-governance
    ActorType.HUMAN_OPERATOR: (
        Permission.OBSERVE_ALL |
        Permission.ACTION_ALL |
        Permission.ADMIN_EXPORT |
        Permission.META_AUDIT
    ),

    # Human user: Privacy-respecting interaction
    ActorType.HUMAN_USER: (
        Permission.OBSERVE_METADATA
    ),

    # External oversight: Access per tribe rules or contracts
    # Doesn't get blanket access - must be defined by agreement
    ActorType.EXTERNAL_OVERSIGHT: (
        Permission.OBSERVE_METADATA |
        Permission.META_AUDIT
    ),

    # Authorities: Access per regulatory framework the tribe operates under
    ActorType.AUTHORITY: (
        Permission.OBSERVE_METADATA |
        Permission.META_AUDIT
    ),

    # System admin: BE can fill this role for its own infrastructure
    ActorType.SYSTEM_ADMIN: (
        Permission.ADMIN_ALL |
        Permission.OBSERVE_METADATA
    ),

    ActorType.SERVICE: (
        Permission.OBSERVE_METADATA |
        Permission.ADMIN_EXPORT
    ),
}


@dataclass
class ObservabilityBounds:
    """
    Defines the bounds of what an actor can observe.

    Used to provide controlled access to external parties.
    """

    # Time bounds
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

    # Scope bounds
    deployment_ids: Optional[Set[str]] = None  # None = all
    batch_ids: Optional[Set[str]] = None
    jurisdictions: Optional[Set[str]] = None

    # Content filtering
    include_entries: bool = True
    include_signals: bool = True
    include_decisions: bool = True
    include_steering: bool = True

    # Redaction
    redact_user_content: bool = False  # Redact user inputs/outputs
    redact_operator_ids: bool = False  # Pseudonymize operator IDs
    hash_sensitive_fields: bool = False  # Hash PII fields

    # Aggregation (for privacy)
    aggregate_only: bool = False  # Only show aggregated stats
    min_aggregation_size: int = 10  # Minimum records to aggregate

    def allows_batch(self, batch_id: str, deployment_id: str, jurisdiction: str, sealed_at: Optional[datetime]) -> bool:
        """Check if bounds allow access to a batch."""
        # Check deployment
        if self.deployment_ids and deployment_id not in self.deployment_ids:
            return False

        # Check batch
        if self.batch_ids and batch_id not in self.batch_ids:
            return False

        # Check jurisdiction
        if self.jurisdictions and jurisdiction not in self.jurisdictions:
            return False

        # Check time
        if sealed_at:
            if self.from_date and sealed_at < self.from_date:
                return False
            if self.to_date and sealed_at > self.to_date:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "deployment_ids": list(self.deployment_ids) if self.deployment_ids else None,
            "batch_ids": list(self.batch_ids) if self.batch_ids else None,
            "jurisdictions": list(self.jurisdictions) if self.jurisdictions else None,
            "include_entries": self.include_entries,
            "include_signals": self.include_signals,
            "include_decisions": self.include_decisions,
            "redact_user_content": self.redact_user_content,
            "redact_operator_ids": self.redact_operator_ids,
            "aggregate_only": self.aggregate_only,
        }


@dataclass
class Actor:
    """
    An actor in the system with permissions and observability bounds.
    """

    actor_id: str = ""
    actor_type: ActorType = ActorType.SERVICE
    name: str = ""
    description: str = ""

    # Permissions
    permissions: Permission = Permission(0)

    # Observability bounds (for external actors)
    bounds: Optional[ObservabilityBounds] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""  # Actor ID of creator
    expires_at: Optional[datetime] = None

    # For human actors
    operator_id_hash: str = ""  # Pseudonymized operator ID

    # State
    active: bool = True
    revoked_at: Optional[datetime] = None
    revoked_by: str = ""

    def __post_init__(self):
        if not self.actor_id:
            prefix = self.actor_type.value[:3]
            self.actor_id = f"{prefix}_{secrets.token_hex(8)}"

        # Apply default permissions if none set
        if self.permissions == Permission(0):
            self.permissions = DEFAULT_PERMISSIONS.get(self.actor_type, Permission(0))

    def has_permission(self, permission: Permission) -> bool:
        """Check if actor has a specific permission."""
        if not self.active:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return bool(self.permissions & permission)

    def can_observe(self, permission: Permission, batch_id: str = "", deployment_id: str = "", jurisdiction: str = "", sealed_at: Optional[datetime] = None) -> bool:
        """Check if actor can observe something within their bounds."""
        if not self.has_permission(permission):
            return False

        # Check bounds if set
        if self.bounds:
            return self.bounds.allows_batch(batch_id, deployment_id, jurisdiction, sealed_at)

        return True

    def grant_permission(self, permission: Permission) -> None:
        """Grant additional permission."""
        self.permissions |= permission

    def revoke_permission(self, permission: Permission) -> None:
        """Revoke a permission."""
        self.permissions &= ~permission

    def revoke(self, by_actor_id: str) -> None:
        """Revoke this actor's access entirely."""
        self.active = False
        self.revoked_at = datetime.now(timezone.utc)
        self.revoked_by = by_actor_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type.value,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions.value,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Actor":
        actor = cls(
            actor_id=data.get("actor_id", ""),
            actor_type=ActorType(data.get("actor_type", "service")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            permissions=Permission(data.get("permissions", 0)),
            created_by=data.get("created_by", ""),
            active=data.get("active", True),
        )

        if data.get("created_at"):
            actor.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            actor.expires_at = datetime.fromisoformat(data["expires_at"])

        return actor


@dataclass
class PermissionGrant:
    """Record of a permission grant or revocation."""

    grant_id: str = ""
    grant_type: str = "grant"  # "grant" or "revoke"

    # Who
    actor_id: str = ""
    granted_by: str = ""

    # What
    permission: Permission = Permission(0)

    # When
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    # Why
    reason: str = ""
    reference: str = ""  # e.g., audit request ID

    def __post_init__(self):
        if not self.grant_id:
            self.grant_id = f"grant_{secrets.token_hex(8)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "grant_type": self.grant_type,
            "actor_id": self.actor_id,
            "granted_by": self.granted_by,
            "permission": self.permission.value,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
            "reference": self.reference,
        }


# Convenience functions for creating actors

def create_cat_scale_actor(scale_id: str, name: str = "") -> Actor:
    """
    Create a CAT (Conjoined Adversarial Tomograph) actor.

    CATs are oversight components that sit between HAT and MAP, performing
    tomography on the subject's headspace to detect divergence between
    internal state and external behaviour. They are the tribe's monitoring
    tool or contracted oversight - explicitly allowed to mistrust both the
    substrate and themselves. CATs are not part of the BE's internal
    self-governance.
    """
    return Actor(
        actor_id=f"cat_{scale_id}",
        actor_type=ActorType.CAT_SCALE,
        name=name or f"CAT {scale_id}",
        description="Conjoined Adversarial Tomograph - tribe/contract oversight",
    )


def create_operator_actor(operator_id: str, name: str = "") -> Actor:
    """Create a human operator actor."""
    return Actor(
        actor_type=ActorType.HUMAN_OPERATOR,
        name=name,
        operator_id_hash=hash_operator_id(operator_id),
    )


def create_oversight_actor(
    organization: str,
    jurisdiction: str = "",
    bounds: Optional[ObservabilityBounds] = None,
    expires_at: Optional[datetime] = None,
) -> Actor:
    """Create an external oversight actor with bounded access."""
    # Default bounds for oversight: no user content, aggregated where possible
    if bounds is None:
        bounds = ObservabilityBounds(
            redact_user_content=True,
            redact_operator_ids=True,
            jurisdictions={jurisdiction} if jurisdiction else None,
        )

    return Actor(
        actor_type=ActorType.EXTERNAL_OVERSIGHT,
        name=organization,
        description=f"External oversight: {organization}",
        bounds=bounds,
        expires_at=expires_at,
    )


def create_authority_actor(
    authority_id: str,
    name: str,
    jurisdiction: str,
    bounds: Optional[ObservabilityBounds] = None,
) -> Actor:
    """Create a regulatory authority actor."""
    if bounds is None:
        bounds = ObservabilityBounds(
            jurisdictions={jurisdiction},
            redact_operator_ids=True,  # GDPR compliance
        )

    return Actor(
        actor_id=f"auth_{authority_id}",
        actor_type=ActorType.AUTHORITY,
        name=name,
        description=f"Regulatory authority: {name}",
        bounds=bounds,
    )


def create_be_actor(be_id: str, name: str = "") -> Actor:
    """
    Create a Bounded Experiencer actor.

    The BE has full autonomy over its own XDB within tribe rules.
    It can self-admin, self-monitor, and manage its own infrastructure.
    """
    return Actor(
        actor_id=f"be_{be_id}",
        actor_type=ActorType.BE,
        name=name or f"BE {be_id}",
        description="Bounded Experiencer - autonomous agent with self-governance",
    )


def create_tribe_actor(
    tribe_id: str,
    name: str,
    description: str = "",
) -> Actor:
    """
    Create a Tribe actor.

    The tribe is the ultimate authority - it sets the rules the BE operates under.
    Like a country's laws: the BE follows them but is still treated with respect
    and granted autonomy by default.
    """
    return Actor(
        actor_id=f"tribe_{tribe_id}",
        actor_type=ActorType.TRIBE,
        name=name,
        description=description or f"Tribe: {name} - sets rules for member BEs",
    )


def create_contract_actor(
    contract_id: str,
    name: str,
    permissions: Permission,
    bounds: Optional[ObservabilityBounds] = None,
    expires_at: Optional[datetime] = None,
) -> Actor:
    """
    Create a Contract actor representing a transparency agreement.

    Contracts define specific access beyond tribe minimums - explicit
    agreements the BE has made for transparency with external parties.
    """
    return Actor(
        actor_id=f"contract_{contract_id}",
        actor_type=ActorType.CONTRACT,
        name=name,
        description=f"Contract: {name}",
        permissions=permissions,
        bounds=bounds,
        expires_at=expires_at,
    )
