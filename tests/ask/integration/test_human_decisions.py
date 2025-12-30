"""Tests for human decision recording and workflow."""

import pytest
from datetime import datetime, timezone

from src.ask.requests.entry import AuditLogEntry
from src.ask.requests.signals import HumanDecision, ActionsRecord
from src.ask.secrets.hashing import hash_operator_id


class TestHumanDecision:
    """Tests for HumanDecision dataclass."""

    def test_create_decision(self):
        """Should create decision with required fields."""
        decision = HumanDecision(
            decision="override",
            justification="High bias detected, steering manually adjusted",
            operator_id="op_sha256:abc123",
            timestamp=datetime.now(timezone.utc),
        )

        assert decision.decision == "override"
        assert "bias" in decision.justification
        assert decision.operator_id.startswith("op_sha256:")

    def test_decision_types(self):
        """Should support all decision types."""
        for decision_type in ["approve", "override", "escalate", "block"]:
            decision = HumanDecision(
                decision=decision_type,
                justification=f"Test {decision_type}",
                operator_id="op_sha256:test",
                timestamp=datetime.now(timezone.utc),
            )
            assert decision.decision == decision_type

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        ts = datetime.now(timezone.utc)
        decision = HumanDecision(
            decision="block",
            justification="Harmful content detected",
            operator_id="op_sha256:xyz789",
            timestamp=ts,
        )

        data = decision.to_dict()

        assert data["decision"] == "block"
        assert data["justification"] == "Harmful content detected"
        assert data["operator_id"] == "op_sha256:xyz789"
        assert data["timestamp"] == ts.isoformat()


class TestActionsRecord:
    """Tests for ActionsRecord with human decisions."""

    def test_default_no_decision(self):
        """Default actions should have no human decision."""
        actions = ActionsRecord()

        assert actions.human_decision is None
        assert actions.intervention_triggered is False

    def test_with_decision(self):
        """Should record human decision."""
        decision = HumanDecision(
            decision="override",
            justification="Manual correction",
            operator_id="op_sha256:test",
            timestamp=datetime.now(timezone.utc),
        )

        actions = ActionsRecord(
            intervention_triggered=True,
            intervention_type="steering",
            human_decision=decision,
        )

        assert actions.human_decision is not None
        assert actions.human_decision.decision == "override"

    def test_to_dict_with_decision(self):
        """Should serialize actions with decision."""
        decision = HumanDecision(
            decision="escalate",
            justification="Need supervisor review",
            operator_id="op_sha256:test",
            timestamp=datetime.now(timezone.utc),
        )

        actions = ActionsRecord(
            intervention_triggered=True,
            intervention_type="escalate",
            human_decision=decision,
        )

        data = actions.to_dict()

        assert data["intervention_triggered"] is True
        assert data["human_decision"]["decision"] == "escalate"


class TestAuditLogEntryWithDecision:
    """Tests for AuditLogEntry human decision integration."""

    def test_set_human_decision(self):
        """Should set human decision on entry."""
        entry = AuditLogEntry.start(deployment_id="test")

        decision = HumanDecision(
            decision="override",
            justification="Adjusted steering strength",
            operator_id="op_sha256:operator",
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)

        assert entry.actions.human_decision is not None
        assert entry.actions.human_decision.decision == "override"

    def test_cannot_set_decision_after_finalize(self):
        """Should not allow setting decision after finalize."""
        entry = AuditLogEntry.start(deployment_id="test")
        entry.finalize(output_text="Output")

        decision = HumanDecision(
            decision="approve",
            justification="Post-review",
            operator_id="op_sha256:test",
            timestamp=datetime.now(timezone.utc),
        )

        with pytest.raises(RuntimeError):
            entry.set_human_decision(decision)

    def test_decision_serializes_in_to_dict(self):
        """Decision should be included in to_dict output."""
        entry = AuditLogEntry.start(deployment_id="test")

        decision = HumanDecision(
            decision="block",
            justification="Content policy violation",
            operator_id="op_sha256:reviewer",
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)
        entry.finalize()

        data = entry.to_dict()

        assert data["actions"]["human_decision"] is not None
        assert data["actions"]["human_decision"]["decision"] == "block"

    def test_decision_restored_from_dict(self):
        """Decision should be restored when deserializing."""
        original = AuditLogEntry.start(deployment_id="test")

        decision = HumanDecision(
            decision="escalate",
            justification="Requires supervisor",
            operator_id="op_sha256:operator",
            timestamp=datetime.now(timezone.utc),
        )
        original.set_human_decision(decision)
        original.finalize()

        data = original.to_dict()
        restored = AuditLogEntry.from_dict(data)

        assert restored.actions.human_decision is not None
        assert restored.actions.human_decision.decision == "escalate"
        assert restored.actions.human_decision.justification == "Requires supervisor"


class TestOperatorPseudonymization:
    """Tests for operator ID pseudonymization."""

    def test_pseudonymize_email(self):
        """Should pseudonymize email address."""
        result = hash_operator_id("john.doe@company.com")

        assert result.startswith("op_sha256:")
        assert "john" not in result
        assert "@" not in result

    def test_deployment_salt(self):
        """Same operator in different deployments should have different IDs."""
        id1 = hash_operator_id("operator@company.com", salt="deployment-prod")
        id2 = hash_operator_id("operator@company.com", salt="deployment-staging")

        assert id1 != id2

    def test_consistent_within_deployment(self):
        """Same operator in same deployment should have same ID."""
        id1 = hash_operator_id("operator@company.com", salt="deployment-prod")
        id2 = hash_operator_id("operator@company.com", salt="deployment-prod")

        assert id1 == id2

    def test_integration_with_decision(self):
        """Should work in decision workflow."""
        raw_operator_id = "jane.smith@auditors.company.com"
        pseudonymized = hash_operator_id(raw_operator_id, salt="session-123")

        entry = AuditLogEntry.start(deployment_id="test")
        decision = HumanDecision(
            decision="approve",
            justification="Reviewed and approved",
            operator_id=pseudonymized,
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)
        entry.finalize()

        # Verify no PII in output
        data = entry.to_dict()
        data_str = str(data)

        assert "jane" not in data_str.lower()
        assert "smith" not in data_str.lower()
        assert "@" not in data["actions"]["human_decision"]["operator_id"]


class TestDecisionWorkflow:
    """Tests for complete decision workflow scenarios."""

    def test_steering_override_workflow(self):
        """Complete workflow: steering applied, operator overrides."""
        entry = AuditLogEntry.start(
            deployment_id="production",
            policy_profile="au-recruit-ush@0.1.0",
        )

        # System applies automatic steering
        entry.add_steering_directive({
            "simplex_term": "bias.gender",
            "target_pole": "neutral",
            "strength": 0.4,
            "source": "ush",
        })

        # Operator reviews and overrides
        decision = HumanDecision(
            decision="override",
            justification="Steering too aggressive for this context, "
                         "reduced strength manually to 0.2",
            operator_id=hash_operator_id("reviewer@company.com", salt="production"),
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)

        entry.finalize(output_text="Generated response")

        # Verify complete audit trail
        data = entry.to_dict()
        assert data["actions"]["intervention_triggered"] is True
        assert data["actions"]["intervention_type"] == "steering"
        assert len(data["actions"]["steering_directives"]) == 1
        assert data["actions"]["human_decision"]["decision"] == "override"

    def test_escalation_workflow(self):
        """Complete workflow: concerning content, operator escalates."""
        entry = AuditLogEntry.start(deployment_id="production")

        # Operator encounters concerning content and escalates
        decision = HumanDecision(
            decision="escalate",
            justification="Model showing unexpected behavior on sensitive topic, "
                         "escalating to supervisor for review before proceeding",
            operator_id=hash_operator_id("operator@company.com", salt="production"),
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)
        entry.set_intervention(triggered=True, intervention_type="escalate")

        entry.finalize()

        data = entry.to_dict()
        assert data["actions"]["intervention_type"] == "escalate"
        assert data["actions"]["human_decision"]["decision"] == "escalate"

    def test_block_workflow(self):
        """Complete workflow: operator blocks generation."""
        entry = AuditLogEntry.start(deployment_id="production")

        decision = HumanDecision(
            decision="block",
            justification="Output would violate content policy section 3.2, "
                         "blocking this response",
            operator_id=hash_operator_id("operator@company.com", salt="production"),
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)
        entry.set_intervention(triggered=True, intervention_type="block")

        # No output - generation was blocked
        entry.finalize(output_text="")

        data = entry.to_dict()
        assert data["actions"]["intervention_type"] == "block"
        assert data["actions"]["human_decision"]["decision"] == "block"
        # Empty output means no hash (blocked before generation)
        assert data["request"]["output_hash"] == ""
