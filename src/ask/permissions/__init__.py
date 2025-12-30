"""
ASK permissions module - actor-based access control with bounded observability.

Manages different actors (CAT scales, BE, operators, users, oversight, authorities)
and their permission scopes for accessing audit data.
"""

from .actors import (
    ActorType,
    Permission,
    ObservabilityBounds,
    Actor,
    PermissionGrant,
    DEFAULT_PERMISSIONS,
    create_cat_scale_actor,
    create_operator_actor,
    create_oversight_actor,
    create_authority_actor,
    create_be_actor,
    create_tribe_actor,
    create_contract_actor,
)
from .access import (
    AccessRequest,
    AccessDecision,
    BoundedView,
    AccessController,
    get_access_controller,
    reset_access_controller,
)

__all__ = [
    # Actor types and permissions
    "ActorType",
    "Permission",
    "ObservabilityBounds",
    "Actor",
    "PermissionGrant",
    "DEFAULT_PERMISSIONS",
    # Actor factories
    "create_cat_scale_actor",
    "create_operator_actor",
    "create_oversight_actor",
    "create_authority_actor",
    "create_be_actor",
    "create_tribe_actor",
    "create_contract_actor",
    # Access control
    "AccessRequest",
    "AccessDecision",
    "BoundedView",
    "AccessController",
    "get_access_controller",
    "reset_access_controller",
]
