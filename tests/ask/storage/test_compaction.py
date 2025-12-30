"""Tests for compaction - storage management with integrity preservation."""

import pytest
from datetime import datetime, timezone, timedelta

from src.ask.storage.compaction import (
    CompactionPolicy,
    CompactedBatchSummary,
    CompactionRecord,
    CompactionManager,
    RetentionRule,
    create_aggressive_policy,
    create_conservative_policy,
    create_minimal_policy,
)
from src.ask.storage.batches import AuditBatch, BatchConfig, BatchManager


class TestCompactionPolicy:
    """Tests for compaction policy configuration."""

    def test_default_policy(self):
        """Should have sensible defaults."""
        policy = CompactionPolicy()

        assert policy.keep_days == 90
        assert policy.preserve_decisions is True
        assert policy.preserve_violations is True
        assert policy.compact_to_summary is True

    def test_custom_policy(self):
        """Should accept custom settings."""
        policy = CompactionPolicy(
            name="test",
            keep_days=30,
            keep_batches=10,
            preserve_decisions=False,
        )

        assert policy.name == "test"
        assert policy.keep_days == 30
        assert policy.keep_batches == 10
        assert policy.preserve_decisions is False

    def test_generates_policy_id(self):
        """Should generate unique policy ID."""
        policy1 = CompactionPolicy()
        policy2 = CompactionPolicy()

        assert policy1.policy_id.startswith("policy_")
        assert policy1.policy_id != policy2.policy_id

    def test_cutoff_time_days(self):
        """Should calculate cutoff time from days."""
        policy = CompactionPolicy(keep_days=7)
        now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        cutoff = policy.get_cutoff_time(now)

        expected = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
        assert cutoff == expected

    def test_cutoff_time_hours(self):
        """Should prefer hours over days when set."""
        policy = CompactionPolicy(keep_days=7, keep_hours=24)
        now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        cutoff = policy.get_cutoff_time(now)

        expected = datetime(2024, 1, 14, 12, 0, tzinfo=timezone.utc)
        assert cutoff == expected

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = CompactionPolicy(
            name="test",
            keep_days=30,
            preserve_decisions=True,
            require_timestamped=True,
        )

        data = original.to_dict()
        restored = CompactionPolicy.from_dict(data)

        assert restored.name == original.name
        assert restored.keep_days == original.keep_days
        assert restored.preserve_decisions == original.preserve_decisions
        assert restored.require_timestamped == original.require_timestamped


class TestCompactedBatchSummary:
    """Tests for compacted batch summary."""

    def test_create_summary(self):
        """Should create summary with required fields."""
        summary = CompactedBatchSummary(
            batch_id="batch_test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            sealed_at=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            original_entry_count=100,
            merkle_root="sha256:abc123",
            batch_hash="sha256:def456",
        )

        assert summary.batch_id == "batch_test"
        assert summary.original_entry_count == 100
        assert summary.merkle_root == "sha256:abc123"

    def test_tracks_preserved_entries(self):
        """Should track preserved entry IDs."""
        summary = CompactedBatchSummary(
            batch_id="batch_test",
            created_at=datetime.now(timezone.utc),
            sealed_at=datetime.now(timezone.utc),
            original_entry_count=100,
            preserved_entry_count=5,
            preserved_entry_ids=["e1", "e2", "e3", "e4", "e5"],
        )

        assert summary.preserved_entry_count == 5
        assert len(summary.preserved_entry_ids) == 5

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = CompactedBatchSummary(
            batch_id="batch_test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            sealed_at=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            original_entry_count=100,
            merkle_root="sha256:abc123",
            batch_hash="sha256:def456",
            decision_count=5,
            violation_count=2,
        )

        data = original.to_dict()
        restored = CompactedBatchSummary.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.merkle_root == original.merkle_root
        assert restored.decision_count == original.decision_count


class TestCompactionRecord:
    """Tests for compaction records."""

    def test_create_record(self):
        """Should create record with ID."""
        record = CompactionRecord()

        assert record.record_id.startswith("compact_")
        assert record.success is False
        assert record.completed_at is None

    def test_complete_success(self):
        """Should mark as complete."""
        record = CompactionRecord()
        record.batches_compacted = 5
        record.entries_removed = 100

        record.complete(success=True)

        assert record.success is True
        assert record.completed_at is not None
        assert record.compaction_hash.startswith("sha256:")

    def test_complete_with_error(self):
        """Should record error on failure."""
        record = CompactionRecord()

        record.complete(success=False, error="Test error")

        assert record.success is False
        assert record.error_message == "Test error"

    def test_bytes_freed_calculation(self):
        """Should calculate bytes freed."""
        record = CompactionRecord()
        record.bytes_before = 1000
        record.bytes_after = 300

        record.complete()

        assert record.bytes_freed == 700

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = CompactionRecord()
        original.batches_compacted = 5
        original.entries_removed = 100
        original.complete(success=True)

        data = original.to_dict()
        restored = CompactionRecord.from_dict(data)

        assert restored.record_id == original.record_id
        assert restored.batches_compacted == original.batches_compacted
        assert restored.success == original.success


class TestCompactionManager:
    """Tests for compaction manager."""

    def _create_test_batch(
        self,
        batch_id: str = None,
        entry_count: int = 10,
        sealed_at: datetime = None,
    ) -> AuditBatch:
        """Helper to create test batches."""
        batch = AuditBatch.create(deployment_id="test")
        if batch_id:
            batch.batch_id = batch_id

        for i in range(entry_count):
            batch.add_entry(f"e_{i}", f"sha256:hash{i}")

        batch.seal()
        if sealed_at:
            batch.sealed_at = sealed_at

        return batch

    def test_create_manager(self):
        """Should create manager with default policy."""
        manager = CompactionManager()

        assert manager.policy is not None
        assert manager.policy.keep_days == 90

    def test_create_with_custom_policy(self):
        """Should accept custom policy."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        assert manager.policy.keep_days == 7

    def test_evaluate_unsealed_batch(self):
        """Should reject unsealed batches."""
        manager = CompactionManager()
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        # Not sealed

        eligible, reason = manager.evaluate_batch(batch)

        assert eligible is False
        assert "not sealed" in reason.lower()

    def test_evaluate_recent_batch(self):
        """Should reject batches within retention period."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        batch = self._create_test_batch(
            sealed_at=datetime.now(timezone.utc) - timedelta(days=1)
        )

        eligible, reason = manager.evaluate_batch(batch)

        assert eligible is False
        assert "retention" in reason.lower()

    def test_evaluate_old_batch(self):
        """Should accept batches outside retention period."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        batch = self._create_test_batch(
            sealed_at=datetime.now(timezone.utc) - timedelta(days=10)
        )

        eligible, reason = manager.evaluate_batch(batch)

        assert eligible is True

    def test_evaluate_requires_timestamp(self):
        """Should require timestamp when configured."""
        policy = CompactionPolicy(keep_days=7, require_timestamped=True)
        manager = CompactionManager(policy=policy)

        batch = self._create_test_batch(
            sealed_at=datetime.now(timezone.utc) - timedelta(days=10)
        )

        eligible, reason = manager.evaluate_batch(batch)

        assert eligible is False
        assert "timestamped" in reason.lower()

    def test_compact_batch_creates_summary(self):
        """Should create summary when compacting."""
        manager = CompactionManager()
        batch = self._create_test_batch(entry_count=50)

        summary, preserved = manager.compact_batch(batch)

        assert summary.batch_id == batch.batch_id
        assert summary.original_entry_count == 50
        assert summary.merkle_root == batch.merkle_root
        assert summary.batch_hash == batch.batch_hash

    def test_compact_preserves_decisions(self):
        """Should preserve entries with decisions."""
        policy = CompactionPolicy(preserve_decisions=True)
        manager = CompactionManager(policy=policy)
        batch = self._create_test_batch(entry_count=10)

        # Mark some entries as having decisions
        entry_metadata = {
            "e_0": {"has_decision": True},
            "e_5": {"has_decision": True},
        }

        summary, preserved = manager.compact_batch(batch, entry_metadata)

        assert len(preserved) == 2
        assert "e_0" in preserved
        assert "e_5" in preserved
        assert summary.preserved_entry_count == 2

    def test_compact_preserves_violations(self):
        """Should preserve entries with violations."""
        policy = CompactionPolicy(preserve_violations=True)
        manager = CompactionManager(policy=policy)
        batch = self._create_test_batch(entry_count=10)

        entry_metadata = {
            "e_3": {"has_violation": True},
        }

        summary, preserved = manager.compact_batch(batch, entry_metadata)

        assert "e_3" in preserved

    def test_is_compacted(self):
        """Should track compacted batches."""
        manager = CompactionManager()
        batch = self._create_test_batch()

        assert manager.is_compacted(batch.batch_id) is False

        manager.compact_batch(batch)

        assert manager.is_compacted(batch.batch_id) is True

    def test_get_summary(self):
        """Should retrieve compaction summary."""
        manager = CompactionManager()
        batch = self._create_test_batch()

        manager.compact_batch(batch)
        summary = manager.get_summary(batch.batch_id)

        assert summary is not None
        assert summary.batch_id == batch.batch_id

    def test_run_compaction(self):
        """Should run compaction on multiple batches."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        batches = [
            self._create_test_batch(batch_id=f"batch_{i}", sealed_at=old_time)
            for i in range(5)
        ]

        record = manager.run_compaction(batches)

        assert record.success is True
        assert record.batches_processed == 5
        assert record.batches_compacted == 5
        assert record.entries_removed == 50  # 5 batches * 10 entries

    def test_run_compaction_dry_run(self):
        """Should not modify state in dry run."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        batches = [self._create_test_batch(sealed_at=old_time)]

        record = manager.run_compaction(batches, dry_run=True)

        assert record.batches_compacted == 1
        assert manager.is_compacted(batches[0].batch_id) is False

    def test_run_compaction_respects_keep_batches(self):
        """Should keep last N batches when configured."""
        policy = CompactionPolicy(keep_days=1, keep_batches=2)
        manager = CompactionManager(policy=policy)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        batches = [
            self._create_test_batch(
                batch_id=f"batch_{i}",
                sealed_at=old_time + timedelta(hours=i)
            )
            for i in range(5)
        ]

        record = manager.run_compaction(batches)

        # Should compact 3, keep 2
        assert record.batches_compacted == 3
        assert record.batches_skipped == 2

    def test_estimate_savings(self):
        """Should estimate storage savings."""
        policy = CompactionPolicy(keep_days=7)
        manager = CompactionManager(policy=policy)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        batches = [
            self._create_test_batch(entry_count=100, sealed_at=old_time)
            for _ in range(3)
        ]

        estimate = manager.estimate_savings(batches, avg_entry_size_bytes=1000)

        assert estimate["batches_eligible"] == 3
        assert estimate["entries_removable"] == 300
        assert estimate["estimated_bytes_freed"] == 300000

    def test_chain_integrity_valid(self):
        """Should verify valid chain."""
        manager = CompactionManager()

        # Create chain of batches
        batch1 = self._create_test_batch(batch_id="b1")
        batch2 = AuditBatch.create(
            deployment_id="test",
            prev_batch_id=batch1.batch_id,
            prev_batch_hash=batch1.batch_hash,
        )
        batch2.batch_id = "b2"
        batch2.add_entry("e1", "sha256:a")
        batch2.seal()

        manager.compact_batch(batch1)
        manager.compact_batch(batch2)

        valid, errors = manager.get_chain_integrity()

        assert valid is True
        assert len(errors) == 0

    def test_get_compaction_records(self):
        """Should return compaction history."""
        policy = CompactionPolicy(keep_days=1)
        manager = CompactionManager(policy=policy)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        batches = [self._create_test_batch(sealed_at=old_time)]

        manager.run_compaction(batches)

        records = manager.get_compaction_records()
        assert len(records) == 1
        assert records[0].success is True


class TestPolicyPresets:
    """Tests for convenience policy functions."""

    def test_aggressive_policy(self):
        """Should create aggressive policy."""
        policy = create_aggressive_policy(keep_days=7)

        assert policy.name == "aggressive"
        assert policy.keep_days == 7
        assert policy.remove_entry_hashes is True
        assert policy.preserve_decisions is True

    def test_conservative_policy(self):
        """Should create conservative policy."""
        policy = create_conservative_policy(keep_days=365)

        assert policy.name == "conservative"
        assert policy.keep_days == 365
        assert policy.require_timestamped is True
        assert policy.remove_entry_hashes is False

    def test_minimal_policy(self):
        """Should create minimal policy."""
        policy = create_minimal_policy(keep_batches=10)

        assert policy.name == "minimal"
        assert policy.keep_batches == 10
        assert policy.keep_days == 1


class TestCompactionIntegration:
    """Integration tests with batch manager."""

    def test_compact_sealed_batches_from_manager(self):
        """Should compact batches from BatchManager."""
        config = BatchConfig(batch_on_entry_count=5)
        batch_manager = BatchManager(deployment_id="test", config=config)

        # Create several batches
        for i in range(15):
            batch_manager.add_entry(f"e_{i}", f"sha256:h{i}")

        sealed = batch_manager.get_sealed_batches()
        assert len(sealed) == 3

        # Compact them
        policy = CompactionPolicy(keep_days=0)  # All eligible
        compact_manager = CompactionManager(policy=policy)

        # Force old timestamps
        for batch in sealed:
            batch.sealed_at = datetime.now(timezone.utc) - timedelta(days=1)

        record = compact_manager.run_compaction(sealed)

        assert record.batches_compacted == 3
        assert record.entries_removed == 15

    def test_summary_preserves_chain_info(self):
        """Should preserve chain info in summaries."""
        config = BatchConfig(batch_on_entry_count=3)
        batch_manager = BatchManager(deployment_id="test", config=config)

        # Create chained batches
        for i in range(9):
            batch_manager.add_entry(f"e_{i}", f"sha256:h{i}")

        sealed = batch_manager.get_sealed_batches()

        # Compact them
        policy = CompactionPolicy(keep_days=0)
        compact_manager = CompactionManager(policy=policy)

        for batch in sealed:
            batch.sealed_at = datetime.now(timezone.utc) - timedelta(days=1)

        compact_manager.run_compaction(sealed)

        # Verify chain is preserved
        summaries = [compact_manager.get_summary(b.batch_id) for b in sealed]

        # Second summary should link to first
        assert summaries[1].prev_batch_id == summaries[0].batch_id
        assert summaries[1].prev_batch_hash == summaries[0].batch_hash

        # Third should link to second
        assert summaries[2].prev_batch_id == summaries[1].batch_id
