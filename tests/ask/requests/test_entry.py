"""Tests for AuditLogEntry."""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.ask.requests.entry import AuditLogEntry
from src.ask.requests.signals import ActiveLensSet, HumanDecision


@dataclass
class MockTick:
    """Mock WorldTick for testing."""
    concept_activations: Dict[str, float]
    violations: List[Dict] = None
    divergence: Optional[object] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class TestAuditLogEntry:
    """Tests for AuditLogEntry lifecycle."""

    def test_start_creates_entry_with_id(self):
        """Entry.start() should create entry with unique ID."""
        entry = AuditLogEntry.start(
            deployment_id="test-deployment",
            policy_profile="test-policy",
        )

        assert entry.entry_id.startswith("evt_")
        assert entry.deployment_id == "test-deployment"
        assert entry.policy_profile == "test-policy"
        assert entry.timestamp_start is not None
        assert entry._finalized is False

    def test_start_hashes_input(self):
        """Entry.start() should hash input text."""
        entry = AuditLogEntry.start(
            deployment_id="test",
            input_text="Hello, world!",
        )

        assert entry.input_hash.startswith("sha256:")
        assert len(entry.input_hash) > 10

    def test_add_tick_aggregates_activations(self):
        """add_tick() should aggregate lens activations."""
        entry = AuditLogEntry.start(deployment_id="test")

        tick1 = MockTick(concept_activations={"lens_a": 0.8, "lens_b": 0.3})
        tick2 = MockTick(concept_activations={"lens_a": 0.6, "lens_b": 0.5})

        entry.add_tick(tick1, tick_number=1)
        entry.add_tick(tick2, tick_number=2)

        entry.finalize()

        assert entry.signals is not None
        assert entry.signals.tick_count == 2
        assert entry.tick_start == 1
        assert entry.tick_end == 2

    def test_add_tick_tracks_violations(self):
        """add_tick() should track violations."""
        entry = AuditLogEntry.start(deployment_id="test")

        tick = MockTick(
            concept_activations={},
            violations=[
                {"type": "simplex_exceeded", "severity": 0.7},
                {"type": "forbidden_concept", "severity": 0.9},
            ],
        )

        entry.add_tick(tick)
        entry.finalize()

        assert entry.signals.violation_count == 2
        assert entry.signals.max_severity == 0.9
        assert "simplex_exceeded" in entry.signals.violations_by_type
        assert "forbidden_concept" in entry.signals.violations_by_type

    def test_finalize_sets_output_hash(self):
        """finalize() should hash output text."""
        entry = AuditLogEntry.start(deployment_id="test")
        entry.finalize(output_text="Generated response")

        assert entry.output_hash.startswith("sha256:")
        assert entry.timestamp_end is not None
        assert entry._finalized is True

    def test_finalize_computes_entry_hash(self):
        """finalize() should compute entry hash."""
        entry = AuditLogEntry.start(
            deployment_id="test",
            prev_hash="sha256:previous",
        )
        entry.finalize(output_text="Output")

        assert entry.entry_hash.startswith("sha256:")
        assert entry.entry_hash != entry.prev_hash

    def test_cannot_modify_after_finalize(self):
        """Cannot add ticks or modify after finalize."""
        entry = AuditLogEntry.start(deployment_id="test")
        entry.finalize()

        with pytest.raises(RuntimeError):
            entry.add_tick(MockTick(concept_activations={}))

        with pytest.raises(RuntimeError):
            entry.add_steering_directive({"test": "directive"})

    def test_add_steering_directive(self):
        """add_steering_directive() should track steering."""
        entry = AuditLogEntry.start(deployment_id="test")

        entry.add_steering_directive({
            "simplex_term": "bias.gender",
            "target_pole": "neutral",
            "strength": 0.4,
        })

        assert entry.actions.intervention_triggered is True
        assert entry.actions.intervention_type == "steering"
        assert len(entry.actions.steering_directives) == 1

    def test_set_human_decision(self):
        """set_human_decision() should record operator decision."""
        entry = AuditLogEntry.start(deployment_id="test")

        decision = HumanDecision(
            decision="override",
            justification="High bias detected",
            operator_id="op_sha256:abc123",
            timestamp=datetime.now(timezone.utc),
        )
        entry.set_human_decision(decision)

        assert entry.actions.human_decision is not None
        assert entry.actions.human_decision.decision == "override"

    def test_to_dict_serializes_correctly(self):
        """to_dict() should produce valid JSON-serializable dict."""
        entry = AuditLogEntry.start(
            deployment_id="test-deployment",
            policy_profile="test-policy",
            input_text="Input text",
            active_lens_set=ActiveLensSet(
                top_k=10,
                lens_pack_ids=["pack1"],
                mandatory_lenses=["lens_a"],
                optional_lenses=["lens_b"],
            ),
        )
        entry.add_tick(MockTick(concept_activations={"lens_a": 0.8}))
        entry.finalize(output_text="Output text")

        data = entry.to_dict()

        assert data["schema_version"] == "ftw.audit.v0.3"
        assert data["deployment_id"] == "test-deployment"
        assert data["request"]["policy_profile"] == "test-policy"
        assert data["request"]["input_hash"].startswith("sha256:")
        assert data["signals"]["tick_count"] == 1
        assert data["cryptography"]["entry_hash"].startswith("sha256:")

    def test_from_dict_deserializes_correctly(self):
        """from_dict() should reconstruct entry from dict."""
        original = AuditLogEntry.start(
            deployment_id="test",
            input_text="Input",
        )
        original.finalize(output_text="Output")

        data = original.to_dict()
        restored = AuditLogEntry.from_dict(data)

        assert restored.entry_id == original.entry_id
        assert restored.deployment_id == original.deployment_id
        assert restored.entry_hash == original.entry_hash


class TestHashChain:
    """Tests for hash chain integrity."""

    def test_chain_links_entries(self):
        """Entries should chain via prev_hash."""
        entry1 = AuditLogEntry.start(deployment_id="test", prev_hash="")
        entry1.finalize(output_text="First")

        entry2 = AuditLogEntry.start(
            deployment_id="test",
            prev_hash=entry1.entry_hash,
        )
        entry2.finalize(output_text="Second")

        assert entry2.prev_hash == entry1.entry_hash
        assert entry2.entry_hash != entry1.entry_hash

    def test_same_content_different_prev_hash_gives_different_entry_hash(self):
        """Entry hash should depend on prev_hash."""
        entry1 = AuditLogEntry.start(deployment_id="test", prev_hash="sha256:aaa")
        entry1.finalize(output_text="Same content")

        entry2 = AuditLogEntry.start(deployment_id="test", prev_hash="sha256:bbb")
        entry2.finalize(output_text="Same content")

        assert entry1.entry_hash != entry2.entry_hash
