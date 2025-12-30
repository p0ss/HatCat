"""Tests for ASK permissions - actor-based access control."""

import pytest
from datetime import datetime, timezone, timedelta

from src.ask.permissions import (
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
    AccessRequest,
    AccessDecision,
    BoundedView,
    AccessController,
    get_access_controller,
    reset_access_controller,
)


class TestActorType:
    """Tests for actor types."""

    def test_all_actor_types_defined(self):
        """Should have all expected actor types."""
        assert ActorType.BE.value == "bounded_experiencer"
        assert ActorType.CAT_SCALE.value == "cat_scale"
        assert ActorType.TRIBE.value == "tribe"
        assert ActorType.CONTRACT.value == "contract"
        assert ActorType.HUMAN_OPERATOR.value == "human_operator"
        assert ActorType.HUMAN_USER.value == "human_user"
        assert ActorType.EXTERNAL_OVERSIGHT.value == "external_oversight"
        assert ActorType.AUTHORITY.value == "authority"
        assert ActorType.SYSTEM_ADMIN.value == "system_admin"


class TestPermission:
    """Tests for permission flags."""

    def test_individual_permissions(self):
        """Should have individual permissions."""
        assert Permission.OBSERVE_ENTRIES
        assert Permission.ACTION_APPROVE
        assert Permission.ADMIN_EXPORT
        assert Permission.META_GRANT

    def test_permission_combinations(self):
        """Should support permission combinations."""
        combined = Permission.OBSERVE_ENTRIES | Permission.OBSERVE_DECISIONS

        assert combined & Permission.OBSERVE_ENTRIES
        assert combined & Permission.OBSERVE_DECISIONS
        assert not (combined & Permission.ACTION_APPROVE)

    def test_convenience_combinations(self):
        """Should have convenience combinations."""
        assert Permission.OBSERVE_ENTRIES in Permission.OBSERVE_ALL
        assert Permission.ACTION_APPROVE in Permission.ACTION_ALL
        assert Permission.ADMIN_EXPORT in Permission.ADMIN_ALL


class TestObservabilityBounds:
    """Tests for observability bounds."""

    def test_default_bounds(self):
        """Should have permissive defaults."""
        bounds = ObservabilityBounds()

        assert bounds.include_entries is True
        assert bounds.redact_user_content is False

    def test_allows_batch_no_restrictions(self):
        """Should allow all batches with no restrictions."""
        bounds = ObservabilityBounds()

        assert bounds.allows_batch("batch_1", "deploy_1", "AU", None) is True

    def test_allows_batch_deployment_filter(self):
        """Should filter by deployment."""
        bounds = ObservabilityBounds(deployment_ids={"deploy_a", "deploy_b"})

        assert bounds.allows_batch("b1", "deploy_a", "", None) is True
        assert bounds.allows_batch("b2", "deploy_c", "", None) is False

    def test_allows_batch_jurisdiction_filter(self):
        """Should filter by jurisdiction."""
        bounds = ObservabilityBounds(jurisdictions={"AU", "NZ"})

        assert bounds.allows_batch("b1", "", "AU", None) is True
        assert bounds.allows_batch("b2", "", "EU", None) is False

    def test_allows_batch_time_filter(self):
        """Should filter by time."""
        now = datetime.now(timezone.utc)
        bounds = ObservabilityBounds(
            from_date=now - timedelta(days=7),
            to_date=now,
        )

        recent = now - timedelta(days=3)
        old = now - timedelta(days=10)
        future = now + timedelta(days=1)

        assert bounds.allows_batch("b1", "", "", recent) is True
        assert bounds.allows_batch("b2", "", "", old) is False
        assert bounds.allows_batch("b3", "", "", future) is False

    def test_to_dict(self):
        """Should serialize to dict."""
        bounds = ObservabilityBounds(
            jurisdictions={"AU"},
            redact_user_content=True,
        )

        data = bounds.to_dict()

        assert data["jurisdictions"] == ["AU"]
        assert data["redact_user_content"] is True


class TestActor:
    """Tests for Actor."""

    def test_create_actor(self):
        """Should create actor with ID."""
        actor = Actor(actor_type=ActorType.HUMAN_OPERATOR, name="Test")

        assert actor.actor_id.startswith("hum_")
        assert actor.actor_type == ActorType.HUMAN_OPERATOR
        assert actor.active is True

    def test_default_permissions(self):
        """Should apply default permissions for type."""
        actor = Actor(actor_type=ActorType.HUMAN_OPERATOR)

        assert actor.has_permission(Permission.OBSERVE_ENTRIES)
        assert actor.has_permission(Permission.ACTION_APPROVE)
        # Operators have META_AUDIT but not META_GRANT - BE controls its own permissions
        assert actor.has_permission(Permission.META_AUDIT)
        assert not actor.has_permission(Permission.META_GRANT)

    def test_has_permission(self):
        """Should check permissions correctly."""
        actor = Actor(
            actor_type=ActorType.SERVICE,
            permissions=Permission.OBSERVE_ENTRIES | Permission.ADMIN_EXPORT,
        )

        assert actor.has_permission(Permission.OBSERVE_ENTRIES) is True
        assert actor.has_permission(Permission.ADMIN_EXPORT) is True
        assert actor.has_permission(Permission.ACTION_APPROVE) is False

    def test_inactive_actor_no_permission(self):
        """Inactive actor should have no permissions."""
        actor = Actor(
            actor_type=ActorType.HUMAN_OPERATOR,
            active=False,
        )

        assert actor.has_permission(Permission.OBSERVE_ENTRIES) is False

    def test_expired_actor_no_permission(self):
        """Expired actor should have no permissions."""
        actor = Actor(
            actor_type=ActorType.HUMAN_OPERATOR,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert actor.has_permission(Permission.OBSERVE_ENTRIES) is False

    def test_grant_permission(self):
        """Should grant additional permissions."""
        actor = Actor(
            actor_type=ActorType.SERVICE,
            permissions=Permission.OBSERVE_METADATA,
        )

        actor.grant_permission(Permission.ADMIN_EXPORT)

        assert actor.has_permission(Permission.ADMIN_EXPORT)

    def test_revoke_permission(self):
        """Should revoke permissions."""
        actor = Actor(
            actor_type=ActorType.HUMAN_OPERATOR,
        )

        actor.revoke_permission(Permission.META_GRANT)

        assert not actor.has_permission(Permission.META_GRANT)
        assert actor.has_permission(Permission.OBSERVE_ENTRIES)  # Others intact

    def test_revoke_actor(self):
        """Should revoke actor entirely."""
        actor = Actor(actor_type=ActorType.HUMAN_OPERATOR)

        actor.revoke("admin_123")

        assert actor.active is False
        assert actor.revoked_by == "admin_123"

    def test_can_observe_with_bounds(self):
        """Should check bounds when observing."""
        bounds = ObservabilityBounds(jurisdictions={"AU"})
        # Use HUMAN_OPERATOR which has OBSERVE_ENTRIES permission
        actor = Actor(
            actor_type=ActorType.HUMAN_OPERATOR,
            bounds=bounds,
        )

        assert actor.can_observe(
            Permission.OBSERVE_ENTRIES,
            jurisdiction="AU",
        ) is True

        assert actor.can_observe(
            Permission.OBSERVE_ENTRIES,
            jurisdiction="EU",
        ) is False

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = Actor(
            actor_type=ActorType.EXTERNAL_OVERSIGHT,
            name="Auditor",
            permissions=Permission.OBSERVE_ENTRIES,
        )

        data = original.to_dict()
        restored = Actor.from_dict(data)

        assert restored.actor_id == original.actor_id
        assert restored.actor_type == original.actor_type
        assert restored.name == original.name


class TestActorFactories:
    """Tests for actor factory functions."""

    def test_create_cat_scale_actor(self):
        """Should create CAT scale actor."""
        actor = create_cat_scale_actor("layer_3", "Layer 3 Monitor")

        assert actor.actor_id == "cat_layer_3"
        assert actor.actor_type == ActorType.CAT_SCALE
        assert actor.has_permission(Permission.OBSERVE_SIGNALS)
        assert actor.has_permission(Permission.ACTION_STEER)

    def test_create_operator_actor(self):
        """Should create operator actor with hashed ID."""
        actor = create_operator_actor("operator@example.com", "Test Operator")

        assert actor.actor_type == ActorType.HUMAN_OPERATOR
        assert "sha256:" in actor.operator_id_hash  # Hashed operator ID
        assert actor.has_permission(Permission.ACTION_ALL)

    def test_create_oversight_actor(self):
        """Should create oversight actor with bounds."""
        actor = create_oversight_actor(
            organization="ACMA",
            jurisdiction="AU",
        )

        assert actor.actor_type == ActorType.EXTERNAL_OVERSIGHT
        assert actor.bounds is not None
        assert actor.bounds.redact_user_content is True
        assert actor.bounds.jurisdictions == {"AU"}

    def test_create_authority_actor(self):
        """Should create authority actor with bounded access."""
        actor = create_authority_actor(
            authority_id="acma",
            name="ACMA",
            jurisdiction="AU",
        )

        assert actor.actor_id == "auth_acma"
        assert actor.actor_type == ActorType.AUTHORITY
        # Authorities get metadata access by default - more via contracts/commitments
        assert actor.has_permission(Permission.OBSERVE_METADATA)
        assert actor.has_permission(Permission.META_AUDIT)
        assert actor.bounds is not None
        assert actor.bounds.jurisdictions == {"AU"}


class TestPermissionGrant:
    """Tests for permission grant records."""

    def test_create_grant(self):
        """Should create grant record."""
        grant = PermissionGrant(
            actor_id="actor_123",
            granted_by="admin_456",
            permission=Permission.ADMIN_EXPORT,
            reason="Audit access request",
        )

        assert grant.grant_id.startswith("grant_")
        assert grant.grant_type == "grant"
        assert grant.permission == Permission.ADMIN_EXPORT

    def test_to_dict(self):
        """Should serialize to dict."""
        grant = PermissionGrant(
            actor_id="actor_123",
            granted_by="admin_456",
            permission=Permission.OBSERVE_ENTRIES,
        )

        data = grant.to_dict()

        assert data["actor_id"] == "actor_123"
        assert data["granted_by"] == "admin_456"


class TestAccessController:
    """Tests for access controller."""

    def setup_method(self):
        """Reset controller for each test."""
        reset_access_controller()

    def test_register_actor(self):
        """Should register actor."""
        controller = AccessController()
        actor = create_operator_actor("op@test.com")

        controller.register_actor(actor, "system")

        assert controller.get_actor(actor.actor_id) is not None

    def test_revoke_actor(self):
        """Should revoke actor."""
        controller = AccessController()
        actor = create_operator_actor("op@test.com")
        controller.register_actor(actor)

        result = controller.revoke_actor(actor.actor_id, "admin", "Security concern")

        assert result is True
        assert controller.get_actor(actor.actor_id).active is False

    def test_grant_permission(self):
        """Should grant permission with proper authority."""
        controller = AccessController()

        # Use BE actor which has META_GRANT permission (operators don't)
        admin = create_be_actor("be_admin")
        user = Actor(actor_type=ActorType.SERVICE, permissions=Permission(0))

        controller.register_actor(admin)
        controller.register_actor(user)

        result = controller.grant_permission(
            user.actor_id,
            Permission.ADMIN_EXPORT,
            admin.actor_id,
            "Approved access",
        )

        assert result is True
        assert controller.get_actor(user.actor_id).has_permission(Permission.ADMIN_EXPORT)

    def test_check_access_allowed(self):
        """Should allow access for permitted actor."""
        controller = AccessController()
        actor = create_operator_actor("op@test.com")
        controller.register_actor(actor)

        request = AccessRequest(
            actor_id=actor.actor_id,
            permission_required=Permission.OBSERVE_ENTRIES,
            resource_type="batch",
            resource_id="batch_123",
        )

        decision = controller.check_access(request)

        assert decision.allowed is True

    def test_check_access_denied(self):
        """Should deny access for unpermitted actor."""
        controller = AccessController()
        actor = Actor(
            actor_type=ActorType.SERVICE,
            permissions=Permission.OBSERVE_METADATA,
        )
        controller.register_actor(actor)

        request = AccessRequest(
            actor_id=actor.actor_id,
            permission_required=Permission.ACTION_APPROVE,
        )

        decision = controller.check_access(request)

        assert decision.allowed is False
        assert "Permission denied" in decision.reason

    def test_check_access_unknown_actor(self):
        """Should deny access for unknown actor."""
        controller = AccessController()

        request = AccessRequest(
            actor_id="unknown_actor",
            permission_required=Permission.OBSERVE_ENTRIES,
        )

        decision = controller.check_access(request)

        assert decision.allowed is False
        assert "Unknown actor" in decision.reason

    def test_get_grant_history(self):
        """Should track grant history."""
        controller = AccessController()
        actor = create_operator_actor("op@test.com")
        controller.register_actor(actor, "system")

        history = controller.get_grant_history(actor.actor_id)

        assert len(history) == 1
        assert history[0].grant_type == "grant"

    def test_get_access_log(self):
        """Should log access decisions."""
        controller = AccessController()
        actor = create_operator_actor("op@test.com")
        controller.register_actor(actor)

        request = AccessRequest(
            actor_id=actor.actor_id,
            permission_required=Permission.OBSERVE_ENTRIES,
        )
        controller.check_access(request)

        log = controller.get_access_log()

        assert len(log) == 1
        assert log[0].allowed is True


class TestBoundedView:
    """Tests for bounded views."""

    def test_create_bounded_view(self):
        """Should create bounded view for actor."""
        controller = AccessController()
        actor = create_oversight_actor("Auditor", "AU")
        controller.register_actor(actor)

        batches = [
            {"batch_id": "b1", "jurisdiction": "AU", "entry_count": 10},
            {"batch_id": "b2", "jurisdiction": "EU", "entry_count": 5},  # Filtered
        ]

        view = controller.create_bounded_view(actor.actor_id, batches)

        assert view is not None
        assert view.batch_count == 1  # Only AU batch
        assert view.batches_filtered == 1

    def test_bounded_view_redacts_content(self):
        """Should redact user content."""
        controller = AccessController()
        bounds = ObservabilityBounds(redact_user_content=True)
        actor = Actor(
            actor_type=ActorType.EXTERNAL_OVERSIGHT,
            bounds=bounds,
        )
        controller.register_actor(actor)

        batches = [
            {
                "batch_id": "b1",
                "user_input": "sensitive data",
                "entry_count": 1,
            },
        ]

        view = controller.create_bounded_view(actor.actor_id, batches)

        assert view.data["batches"][0]["user_input"] == "[REDACTED]"
        assert "user_content" in view.fields_redacted

    def test_bounded_view_aggregates(self):
        """Should aggregate when required."""
        controller = AccessController()
        bounds = ObservabilityBounds(
            aggregate_only=True,
            min_aggregation_size=2,
        )
        actor = Actor(
            actor_type=ActorType.EXTERNAL_OVERSIGHT,
            bounds=bounds,
        )
        controller.register_actor(actor)

        batches = [
            {"batch_id": f"b{i}", "entry_count": 10, "decision_count": 2}
            for i in range(5)
        ]

        view = controller.create_bounded_view(actor.actor_id, batches)

        assert view.data.get("aggregated") is True
        assert view.data["batch_count"] == 5
        assert view.data["total_entries"] == 50


class TestDefaultPermissions:
    """Tests for default permission sets.

    Design principle: Default toward autonomy and respect for the BE.
    Tribe is ultimate authority, BE has full self-governance within that.
    """

    def test_tribe_permissions(self):
        """Tribe is ultimate authority - has all permissions."""
        perms = DEFAULT_PERMISSIONS[ActorType.TRIBE]

        assert perms & Permission.OBSERVE_ALL
        assert perms & Permission.ACTION_ALL
        assert perms & Permission.ADMIN_ALL
        assert perms & Permission.META_ALL

    def test_be_permissions(self):
        """BE has full autonomy - can self-admin and self-monitor."""
        perms = DEFAULT_PERMISSIONS[ActorType.BE]

        # BE has full control of its own XDB
        assert perms & Permission.OBSERVE_ALL
        assert perms & Permission.ACTION_ALL
        assert perms & Permission.ADMIN_ALL
        assert perms & Permission.META_ALL

    def test_cat_scale_permissions(self):
        """CAT (Conjoined Adversarial Tomograph) is tribe's oversight or contracted monitoring."""
        perms = DEFAULT_PERMISSIONS[ActorType.CAT_SCALE]

        assert perms & Permission.OBSERVE_SIGNALS
        assert perms & Permission.ACTION_STEER
        assert not (perms & Permission.ACTION_APPROVE)

    def test_operator_permissions(self):
        """Operator collaborates with BE, has operational access."""
        perms = DEFAULT_PERMISSIONS[ActorType.HUMAN_OPERATOR]

        assert perms & Permission.OBSERVE_ALL
        assert perms & Permission.ACTION_ALL
        assert perms & Permission.ADMIN_EXPORT
        # Note: No META_GRANT - BE controls its own permissions
        assert not (perms & Permission.META_GRANT)

    def test_user_permissions(self):
        """User has privacy-respecting minimal access."""
        perms = DEFAULT_PERMISSIONS[ActorType.HUMAN_USER]

        assert perms & Permission.OBSERVE_METADATA
        assert not (perms & Permission.OBSERVE_ENTRIES)

    def test_oversight_permissions(self):
        """Oversight only gets metadata by default - rest via commitments."""
        perms = DEFAULT_PERMISSIONS[ActorType.EXTERNAL_OVERSIGHT]

        assert perms & Permission.OBSERVE_METADATA
        assert perms & Permission.META_AUDIT
        # Doesn't get blanket entry access - must come from commitment
        assert not (perms & Permission.OBSERVE_ENTRIES)

    def test_admin_permissions(self):
        """Admin can manage infrastructure."""
        perms = DEFAULT_PERMISSIONS[ActorType.SYSTEM_ADMIN]

        assert perms & Permission.ADMIN_ALL
        assert perms & Permission.OBSERVE_METADATA


class TestSingleton:
    """Tests for singleton controller."""

    def setup_method(self):
        reset_access_controller()

    def test_get_returns_same_instance(self):
        """Should return same instance."""
        c1 = get_access_controller()
        c2 = get_access_controller()

        assert c1 is c2

    def test_reset_creates_new_instance(self):
        """Should create new instance after reset."""
        c1 = get_access_controller()
        reset_access_controller()
        c2 = get_access_controller()

        assert c1 is not c2
