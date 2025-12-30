"""Tests for export formats - regulatory submission format generation."""

import pytest
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

from src.ask.export.formats import (
    ExportFormat,
    ExportScope,
    ExportConfig,
    ExportMetadata,
    ExportedBatch,
    ExportResult,
    BatchExporter,
    ExportVerifier,
    create_full_exporter,
    create_compliance_exporter,
    create_archive_exporter,
)
from src.ask.storage.batches import AuditBatch, BatchManager, BatchConfig
from src.ask.storage.compaction import CompactedBatchSummary
from src.ask.replication.authorities import (
    AuthorityReceipt,
    ReceiptStatus,
    AuthorityType,
    create_mock_authority,
    AuthorityClient,
)


class TestExportConfig:
    """Tests for export configuration."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ExportConfig()

        assert config.format == ExportFormat.ASK_V1
        assert config.scope == ExportScope.FULL
        assert config.include_entries is True
        assert config.include_merkle_proofs is True

    def test_custom_config(self):
        """Should accept custom settings."""
        config = ExportConfig(
            format=ExportFormat.EU_AI_ACT,
            include_entries=False,
            require_timestamped=True,
        )

        assert config.format == ExportFormat.EU_AI_ACT
        assert config.include_entries is False
        assert config.require_timestamped is True

    def test_to_dict(self):
        """Should serialize to dict."""
        config = ExportConfig(
            format=ExportFormat.AU_AISF,
            scope=ExportScope.COMPLIANCE,
        )

        data = config.to_dict()

        assert data["format"] == "au.aisf.v1"
        assert data["scope"] == "compliance"


class TestExportMetadata:
    """Tests for export metadata."""

    def test_generates_export_id(self):
        """Should generate unique export ID."""
        meta1 = ExportMetadata()
        meta2 = ExportMetadata()

        assert meta1.export_id.startswith("export_")
        assert meta1.export_id != meta2.export_id

    def test_tracks_content_info(self):
        """Should track content information."""
        meta = ExportMetadata(
            batch_count=5,
            entry_count=100,
            total_bytes=5000,
            deployment_id="test-deploy",
        )

        assert meta.batch_count == 5
        assert meta.entry_count == 100
        assert meta.deployment_id == "test-deploy"

    def test_to_dict(self):
        """Should serialize to dict."""
        meta = ExportMetadata(
            batch_count=3,
            content_hash="sha256:abc123",
        )

        data = meta.to_dict()

        assert data["batch_count"] == 3
        assert data["content_hash"] == "sha256:abc123"
        assert "exported_at" in data


class TestExportedBatch:
    """Tests for exported batch structure."""

    def test_create_exported_batch(self):
        """Should create exported batch."""
        exported = ExportedBatch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            created_at="2024-01-15T12:00:00+00:00",
            sealed_at="2024-01-15T12:01:00+00:00",
            entry_count=10,
        )

        assert exported.batch_id == "batch_123"
        assert exported.entry_count == 10

    def test_to_dict_minimal(self):
        """Should serialize minimal batch."""
        exported = ExportedBatch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            created_at="2024-01-15T12:00:00+00:00",
            sealed_at="2024-01-15T12:01:00+00:00",
        )

        data = exported.to_dict()

        assert data["batch_id"] == "batch_123"
        assert "chain" not in data  # No prev batch
        assert "entries" not in data  # No entries

    def test_to_dict_with_chain(self):
        """Should include chain info."""
        exported = ExportedBatch(
            batch_id="batch_2",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            created_at="2024-01-15T12:00:00+00:00",
            sealed_at="2024-01-15T12:01:00+00:00",
            prev_batch_id="batch_1",
            prev_batch_hash="sha256:prev",
        )

        data = exported.to_dict()

        assert "chain" in data
        assert data["chain"]["prev_batch_id"] == "batch_1"


class TestBatchExporter:
    """Tests for batch exporter."""

    def _create_test_batch(self, entry_count: int = 5) -> AuditBatch:
        """Helper to create test batch."""
        batch = AuditBatch.create(deployment_id="test")
        for i in range(entry_count):
            batch.add_entry(f"e_{i}", f"sha256:hash{i}")
        batch.seal()
        return batch

    def test_create_exporter(self):
        """Should create exporter with config."""
        exporter = BatchExporter()

        assert exporter.config.format == ExportFormat.ASK_V1

    def test_export_single_batch(self):
        """Should export single batch."""
        batch = self._create_test_batch()
        exporter = BatchExporter()

        exported = exporter.export_batch(batch)

        assert exported.batch_id == batch.batch_id
        assert exported.batch_hash == batch.batch_hash
        assert exported.merkle_root == batch.merkle_root
        assert exported.entry_count == 5

    def test_export_batch_with_receipts(self):
        """Should include authority receipts."""
        batch = self._create_test_batch()

        receipts = [
            AuthorityReceipt(
                authority_id="auth-1",
                batch_id=batch.batch_id,
                batch_hash=batch.batch_hash,
                status=ReceiptStatus.CONFIRMED,
            ),
        ]

        exporter = BatchExporter()
        exported = exporter.export_batch(batch, authority_receipts=receipts)

        assert len(exported.authority_receipts) == 1

    def test_export_batches(self):
        """Should export multiple batches."""
        batches = [self._create_test_batch() for _ in range(3)]
        exporter = BatchExporter()

        result = exporter.export_batches(
            batches,
            deployment_id="test-deploy",
            jurisdiction="AU",
        )

        assert result.success is True
        assert result.metadata.batch_count == 3
        assert result.metadata.entry_count == 15  # 3 * 5

    def test_export_content_is_valid_json(self):
        """Should produce valid JSON."""
        batch = self._create_test_batch()
        exporter = BatchExporter()

        result = exporter.export_batches([batch])

        doc = json.loads(result.content)
        assert doc["format"] == "ask.v1"
        assert len(doc["batches"]) == 1

    def test_export_includes_schema(self):
        """Should include schema reference."""
        batch = self._create_test_batch()
        exporter = BatchExporter()

        result = exporter.export_batches([batch])
        doc = json.loads(result.content)

        assert "$schema" in doc
        assert "ask.v1" in doc["$schema"]

    def test_export_includes_period(self):
        """Should include time period."""
        batches = [self._create_test_batch() for _ in range(2)]
        exporter = BatchExporter()

        result = exporter.export_batches(batches)
        doc = json.loads(result.content)

        assert "period" in doc
        assert doc["period"]["start"] is not None

    def test_export_includes_chain_info(self):
        """Should include chain info for multiple batches."""
        # Create chained batches
        config = BatchConfig(batch_on_entry_count=3)
        manager = BatchManager(deployment_id="test", config=config)

        for i in range(9):
            manager.add_entry(f"e_{i}", f"sha256:h{i}")

        batches = manager.get_sealed_batches()
        exporter = BatchExporter()

        result = exporter.export_batches(batches)
        doc = json.loads(result.content)

        assert "chain" in doc
        assert doc["chain"]["batch_count"] == 3

    def test_export_with_pretty_print(self):
        """Should format with indentation when configured."""
        batch = self._create_test_batch()
        config = ExportConfig(pretty_print=True, indent=4)
        exporter = BatchExporter(config)

        result = exporter.export_batches([batch])

        # Pretty printed JSON has newlines
        assert "\n" in result.content

    def test_export_validates_sealed(self):
        """Should validate batches are sealed."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        # Not sealed

        config = ExportConfig(require_sealed=True)
        exporter = BatchExporter(config)

        result = exporter.export_batches([batch])

        assert result.success is False
        assert any("not sealed" in e for e in result.validation_errors)

    def test_export_content_hash(self):
        """Should compute content hash."""
        batch = self._create_test_batch()
        exporter = BatchExporter()

        result = exporter.export_batches([batch])

        assert result.metadata.content_hash.startswith("sha256:")


class TestExportFormats:
    """Tests for different export formats."""

    def _create_test_batch(self) -> AuditBatch:
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.seal()
        return batch

    def test_eu_ai_act_format(self):
        """Should add EU AI Act fields."""
        batch = self._create_test_batch()
        config = ExportConfig(format=ExportFormat.EU_AI_ACT)
        exporter = BatchExporter(config)

        result = exporter.export_batches([batch])
        doc = json.loads(result.content)

        assert doc["format"] == "eu.ai-act.v1"
        assert "regulatory" in doc
        assert doc["regulatory"]["framework"] == "EU AI Act"

    def test_au_aisf_format(self):
        """Should add Australian AISF fields."""
        batch = self._create_test_batch()
        config = ExportConfig(format=ExportFormat.AU_AISF)
        exporter = BatchExporter(config)

        result = exporter.export_batches([batch])
        doc = json.loads(result.content)

        assert doc["format"] == "au.aisf.v1"
        assert "regulatory" in doc
        assert "Australian" in doc["regulatory"]["framework"]


class TestExportToFile:
    """Tests for file export."""

    def test_export_to_file(self):
        """Should write export to file."""
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.seal()

        exporter = BatchExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.json"
            result = exporter.export_to_file([batch], output_path)

            assert result.success is True
            assert result.output_path == output_path
            assert output_path.exists()

            # Verify file content
            content = output_path.read_text()
            doc = json.loads(content)
            assert len(doc["batches"]) == 1


class TestExportCompacted:
    """Tests for compacted batch export."""

    def test_export_compacted_summaries(self):
        """Should export compacted batch summaries."""
        summaries = [
            CompactedBatchSummary(
                batch_id=f"batch_{i}",
                created_at=datetime.now(timezone.utc),
                sealed_at=datetime.now(timezone.utc),
                original_entry_count=100,
                preserved_entry_count=5,
                merkle_root=f"sha256:root{i}",
                batch_hash=f"sha256:hash{i}",
            )
            for i in range(3)
        ]

        exporter = BatchExporter()
        result = exporter.export_compacted(
            summaries,
            deployment_id="test",
            jurisdiction="AU",
        )

        assert result.success is True
        assert result.metadata.batch_count == 3

        doc = json.loads(result.content)
        assert doc["compacted"] is True
        assert len(doc["batches"]) == 3

    def test_export_compacted_preserves_chain(self):
        """Should preserve chain info in compacted export."""
        summaries = [
            CompactedBatchSummary(
                batch_id="batch_1",
                created_at=datetime.now(timezone.utc),
                sealed_at=datetime.now(timezone.utc),
                merkle_root="sha256:root1",
                batch_hash="sha256:hash1",
            ),
            CompactedBatchSummary(
                batch_id="batch_2",
                created_at=datetime.now(timezone.utc),
                sealed_at=datetime.now(timezone.utc),
                merkle_root="sha256:root2",
                batch_hash="sha256:hash2",
                prev_batch_id="batch_1",
                prev_batch_hash="sha256:hash1",
            ),
        ]

        exporter = BatchExporter()
        result = exporter.export_compacted(summaries)

        doc = json.loads(result.content)
        assert doc["batches"][1]["chain"]["prev_batch_id"] == "batch_1"


class TestExportVerifier:
    """Tests for export verification."""

    def _create_valid_export(self) -> str:
        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:a")
        batch.seal()

        exporter = BatchExporter()
        result = exporter.export_batches([batch])
        return result.content

    def test_verify_valid_export(self):
        """Should verify valid export."""
        content = self._create_valid_export()
        verifier = ExportVerifier()

        valid, errors = verifier.verify_export(content)

        assert valid is True
        assert len(errors) == 0

    def test_verify_invalid_json(self):
        """Should reject invalid JSON."""
        verifier = ExportVerifier()

        valid, errors = verifier.verify_export("not json")

        assert valid is False
        assert any("Invalid JSON" in e for e in errors)

    def test_verify_missing_fields(self):
        """Should detect missing required fields."""
        content = json.dumps({"format": "test"})
        verifier = ExportVerifier()

        valid, errors = verifier.verify_export(content)

        assert valid is False
        assert any("Missing required field" in e for e in errors)

    def test_verify_chain_integrity(self):
        """Should verify chain integrity."""
        # Create valid chained export
        config = BatchConfig(batch_on_entry_count=2)
        manager = BatchManager(deployment_id="test", config=config)

        for i in range(6):
            manager.add_entry(f"e_{i}", f"sha256:h{i}")

        batches = manager.get_sealed_batches()
        exporter = BatchExporter()
        result = exporter.export_batches(batches)

        verifier = ExportVerifier()
        valid, errors = verifier.verify_export(result.content)

        assert valid is True

    def test_verify_detects_chain_break(self):
        """Should detect chain break."""
        doc = {
            "format": "ask.v1",
            "version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "batches": [
                {
                    "batch_id": "batch_1",
                    "batch_hash": "sha256:hash1",
                    "merkle_root": "sha256:root1",
                },
                {
                    "batch_id": "batch_2",
                    "batch_hash": "sha256:hash2",
                    "merkle_root": "sha256:root2",
                    "chain": {
                        "prev_batch_id": "batch_wrong",  # Wrong!
                        "prev_batch_hash": "sha256:hash1",
                    },
                },
            ],
        }

        content = json.dumps(doc)
        verifier = ExportVerifier()

        valid, errors = verifier.verify_export(content)

        assert valid is False
        assert any("Chain break" in e for e in errors)

    def test_compute_content_hash(self):
        """Should compute content hash."""
        content = self._create_valid_export()
        verifier = ExportVerifier()

        hash_value = verifier.compute_content_hash(content)

        assert hash_value.startswith("sha256:")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_full_exporter(self):
        """Should create full exporter."""
        exporter = create_full_exporter()

        assert exporter.config.include_entries is True
        assert exporter.config.include_merkle_proofs is True
        assert exporter.config.pretty_print is True

    def test_create_compliance_exporter_eu(self):
        """Should create EU compliance exporter."""
        exporter = create_compliance_exporter("EU")

        assert exporter.config.format == ExportFormat.EU_AI_ACT
        assert exporter.config.include_entries is False
        assert exporter.config.require_timestamped is True

    def test_create_compliance_exporter_au(self):
        """Should create AU compliance exporter."""
        exporter = create_compliance_exporter("AU")

        assert exporter.config.format == ExportFormat.AU_AISF

    def test_create_archive_exporter(self):
        """Should create archive exporter."""
        exporter = create_archive_exporter()

        assert exporter.config.format == ExportFormat.ARCHIVE
        assert exporter.config.include_entries is True
        assert exporter.config.pretty_print is False  # Compact


class TestFullIntegration:
    """Full integration tests."""

    def test_complete_export_workflow(self):
        """Should complete full export workflow."""
        # Create batches with manager
        config = BatchConfig(batch_on_entry_count=5)
        manager = BatchManager(deployment_id="test-deploy", config=config)

        for i in range(15):
            manager.add_entry(f"entry_{i}", f"sha256:hash{i}")

        batches = manager.get_sealed_batches()

        # Get authority receipts
        auth = create_mock_authority("notary")
        auth_client = AuthorityClient(authorities=[auth])

        receipts_by_batch = {}
        for batch in batches:
            receipt, _ = auth_client.submit_batch(
                batch_id=batch.batch_id,
                batch_hash=batch.batch_hash,
                merkle_root=batch.merkle_root,
            )
            if receipt:
                receipts_by_batch[batch.batch_id] = [receipt]

        # Export
        exporter = create_full_exporter()
        result = exporter.export_batches(
            batches,
            authority_receipts=receipts_by_batch,
            deployment_id="test-deploy",
            jurisdiction="AU",
        )

        assert result.success is True
        assert result.metadata.batch_count == 3
        assert result.metadata.entry_count == 15

        # Verify export
        verifier = ExportVerifier()
        valid, errors = verifier.verify_export(result.content)

        assert valid is True

        # Check content
        doc = json.loads(result.content)
        assert doc["deployment"]["deployment_id"] == "test-deploy"
        assert doc["deployment"]["jurisdiction"] == "AU"
        assert len(doc["batches"]) == 3

        # Each batch should have receipts
        for batch_doc in doc["batches"]:
            assert "authority_receipts" in batch_doc
            assert len(batch_doc["authority_receipts"]) == 1
