"""Tests for AuditBatch and BatchManager."""

import pytest
from datetime import datetime, timezone, timedelta

from src.ask.storage.batches import (
    AuditBatch,
    BatchConfig,
    BatchManager,
    generate_batch_id,
)


class TestGenerateBatchId:
    """Tests for batch ID generation."""

    def test_format(self):
        """Should produce correct format."""
        batch_id = generate_batch_id()

        assert batch_id.startswith("batch_")
        parts = batch_id.split("_")
        assert len(parts) == 4

    def test_uniqueness(self):
        """Should generate unique IDs."""
        ids = [generate_batch_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestAuditBatchCreation:
    """Tests for AuditBatch creation."""

    def test_create_batch(self):
        """Should create batch with deployment ID."""
        batch = AuditBatch.create(
            deployment_id="test-deployment",
            jurisdiction="AU",
        )

        assert batch.batch_id.startswith("batch_")
        assert batch.deployment_id == "test-deployment"
        assert batch.jurisdiction == "AU"
        assert batch._sealed is False

    def test_create_with_chain_link(self):
        """Should create batch linked to previous."""
        batch = AuditBatch.create(
            deployment_id="test",
            prev_batch_id="batch_prev",
            prev_batch_hash="sha256:prevhash",
        )

        assert batch.prev_batch_id == "batch_prev"
        assert batch.prev_batch_hash == "sha256:prevhash"


class TestAuditBatchEntries:
    """Tests for adding entries to batches."""

    def test_add_entry(self):
        """Should add entry to batch."""
        batch = AuditBatch.create(deployment_id="test")

        batch.add_entry(
            entry_id="evt_123",
            entry_hash="sha256:abc",
            timestamp=datetime.now(timezone.utc),
            tick=1,
        )

        assert batch.entry_count == 1
        assert "evt_123" in batch.entry_ids
        assert "sha256:abc" in batch.entry_hashes

    def test_add_multiple_entries(self):
        """Should add multiple entries."""
        batch = AuditBatch.create(deployment_id="test")

        for i in range(5):
            batch.add_entry(
                entry_id=f"evt_{i}",
                entry_hash=f"sha256:hash{i}",
            )

        assert batch.entry_count == 5

    def test_tracks_time_window(self):
        """Should track time window from entries."""
        batch = AuditBatch.create(deployment_id="test")

        t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)

        batch.add_entry("e1", "h1", timestamp=t1)
        batch.add_entry("e2", "h2", timestamp=t2)
        batch.add_entry("e3", "h3", timestamp=t3)

        assert batch.window_start == t1
        assert batch.window_end == t2

    def test_tracks_tick_range(self):
        """Should track tick range from entries."""
        batch = AuditBatch.create(deployment_id="test")

        batch.add_entry("e1", "h1", tick=10)
        batch.add_entry("e2", "h2", tick=5)
        batch.add_entry("e3", "h3", tick=15)

        assert batch.tick_start == 5
        assert batch.tick_end == 15

    def test_cannot_add_to_sealed(self):
        """Should not allow adding to sealed batch."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "h1")
        batch.seal()

        with pytest.raises(RuntimeError):
            batch.add_entry("e2", "h2")


class TestAuditBatchSealing:
    """Tests for batch sealing."""

    def test_seal_computes_merkle_root(self):
        """Sealing should compute Merkle root."""
        batch = AuditBatch.create(deployment_id="test")

        batch.add_entry("e1", "sha256:a")
        batch.add_entry("e2", "sha256:b")
        batch.seal()

        assert batch._sealed is True
        assert batch.merkle_root.startswith("sha256:")
        assert batch.sealed_at is not None

    def test_seal_computes_batch_hash(self):
        """Sealing should compute batch hash."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.seal()

        assert batch.batch_hash.startswith("sha256:")

    def test_seal_idempotent(self):
        """Sealing twice should be safe."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")

        batch.seal()
        hash1 = batch.batch_hash

        batch.seal()
        hash2 = batch.batch_hash

        assert hash1 == hash2

    def test_empty_batch_seals(self):
        """Empty batch should seal with empty hash."""
        batch = AuditBatch.create(deployment_id="test")
        batch.seal()

        assert batch._sealed is True
        assert batch.merkle_root.startswith("sha256:")


class TestAuditBatchProofs:
    """Tests for inclusion proofs."""

    def test_get_proof(self):
        """Should generate valid proof for entry."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.add_entry("e2", "sha256:b")
        batch.add_entry("e3", "sha256:c")
        batch.seal()

        proof = batch.get_proof("sha256:b")

        assert proof is not None
        assert proof.entry_hash == "sha256:b"
        assert proof.merkle_root == batch.merkle_root
        assert proof.verify()

    def test_get_proof_not_sealed(self):
        """Should not allow proofs on unsealed batch."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")

        with pytest.raises(RuntimeError):
            batch.get_proof("sha256:a")

    def test_verify_entry(self):
        """Should verify entry with proof."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.add_entry("e2", "sha256:b")
        batch.seal()

        proof = batch.get_proof("sha256:a")
        assert batch.verify_entry("sha256:a", proof)

    def test_verify_wrong_entry_fails(self):
        """Should fail verification with wrong entry."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.add_entry("e2", "sha256:b")
        batch.seal()

        proof = batch.get_proof("sha256:a")
        assert not batch.verify_entry("sha256:c", proof)


class TestAuditBatchSerialization:
    """Tests for batch serialization."""

    def test_to_dict(self):
        """Should serialize batch to dict."""
        batch = AuditBatch.create(
            deployment_id="test-dep",
            jurisdiction="AU",
        )
        batch.add_entry("e1", "sha256:a", tick=1)
        batch.seal()

        data = batch.to_dict()

        assert data["batch_id"] == batch.batch_id
        assert data["deployment_id"] == "test-dep"
        assert data["jurisdiction"] == "AU"
        assert data["entries"]["entry_count"] == 1
        assert data["cryptography"]["merkle_root"] == batch.merkle_root
        assert data["sealed"] is True

    def test_from_dict(self):
        """Should deserialize batch from dict."""
        original = AuditBatch.create(deployment_id="test")
        original.add_entry("e1", "sha256:a")
        original.add_entry("e2", "sha256:b")
        original.seal()

        data = original.to_dict()
        restored = AuditBatch.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.deployment_id == original.deployment_id
        assert restored.merkle_root == original.merkle_root
        assert restored.batch_hash == original.batch_hash
        assert restored._sealed == original._sealed


class TestBatchChaining:
    """Tests for batch chain linkage."""

    def test_chain_links_batches(self):
        """Batches should chain via prev_batch_hash."""
        batch1 = AuditBatch.create(deployment_id="test")
        batch1.add_entry("e1", "sha256:a")
        batch1.seal()

        batch2 = AuditBatch.create(
            deployment_id="test",
            prev_batch_id=batch1.batch_id,
            prev_batch_hash=batch1.batch_hash,
        )
        batch2.add_entry("e2", "sha256:b")
        batch2.seal()

        assert batch2.prev_batch_id == batch1.batch_id
        assert batch2.prev_batch_hash == batch1.batch_hash
        assert batch2.batch_hash != batch1.batch_hash

    def test_different_prev_hash_different_batch_hash(self):
        """Different prev_hash should produce different batch_hash."""
        batch1 = AuditBatch.create(
            deployment_id="test",
            prev_batch_hash="sha256:aaa",
        )
        batch1.add_entry("e1", "sha256:same")
        batch1.seal()

        batch2 = AuditBatch.create(
            deployment_id="test",
            prev_batch_hash="sha256:bbb",
        )
        batch2.add_entry("e1", "sha256:same")
        batch2.seal()

        assert batch1.batch_hash != batch2.batch_hash


class TestBatchManager:
    """Tests for BatchManager."""

    def test_creates_current_batch(self):
        """Should create batch on demand."""
        manager = BatchManager(deployment_id="test")

        batch = manager.current_batch

        assert batch is not None
        assert batch.deployment_id == "test"

    def test_add_entry(self):
        """Should add entry to current batch."""
        manager = BatchManager(deployment_id="test")

        result = manager.add_entry("e1", "sha256:a")

        assert result is None  # No seal yet
        assert manager.current_batch.entry_count == 1

    def test_auto_seal_on_threshold(self):
        """Should seal when entry count threshold reached."""
        config = BatchConfig(batch_on_entry_count=3)
        manager = BatchManager(deployment_id="test", config=config)

        manager.add_entry("e1", "sha256:a")
        manager.add_entry("e2", "sha256:b")
        result = manager.add_entry("e3", "sha256:c")

        assert result is not None  # Sealed batch returned
        assert result.entry_count == 3
        assert result._sealed is True

    def test_manual_seal(self):
        """Should allow manual sealing."""
        manager = BatchManager(deployment_id="test")

        manager.add_entry("e1", "sha256:a")
        sealed = manager.seal_current_batch()

        assert sealed is not None
        assert sealed._sealed is True

    def test_chain_maintained(self):
        """Manager should maintain chain across batches."""
        config = BatchConfig(batch_on_entry_count=2)
        manager = BatchManager(deployment_id="test", config=config)

        manager.add_entry("e1", "sha256:a")
        batch1 = manager.add_entry("e2", "sha256:b")

        manager.add_entry("e3", "sha256:c")
        batch2 = manager.add_entry("e4", "sha256:d")

        assert batch2.prev_batch_id == batch1.batch_id
        assert batch2.prev_batch_hash == batch1.batch_hash

    def test_get_sealed_batches(self):
        """Should track all sealed batches."""
        config = BatchConfig(batch_on_entry_count=2)
        manager = BatchManager(deployment_id="test", config=config)

        for i in range(6):
            manager.add_entry(f"e{i}", f"sha256:h{i}")

        batches = manager.get_sealed_batches()
        assert len(batches) == 3

    def test_time_seal_check(self):
        """Should check if time-based seal is needed."""
        config = BatchConfig(batch_interval_hours=24)
        manager = BatchManager(deployment_id="test", config=config)

        manager.add_entry("e1", "sha256:a")

        # Not enough time passed
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        assert not manager.should_seal_by_time(recent)

        # Enough time passed
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        assert manager.should_seal_by_time(old)


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = BatchConfig()

        assert config.batch_interval_hours == 24
        assert config.batch_on_entry_count == 1000
        assert config.compute_merkle_tree is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = BatchConfig(
            batch_interval_hours=12,
            batch_on_entry_count=500,
            require_rfc3161=True,
        )

        assert config.batch_interval_hours == 12
        assert config.batch_on_entry_count == 500
        assert config.require_rfc3161 is True
