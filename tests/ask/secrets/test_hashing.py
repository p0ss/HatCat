"""Tests for ASK hashing utilities."""

import pytest
from datetime import datetime, timezone

from src.ask.secrets.hashing import (
    hash_content,
    hash_operator_id,
    generate_entry_id,
    compute_entry_hash,
)


class TestHashContent:
    """Tests for hash_content function."""

    def test_hash_string(self):
        """Should hash string content."""
        result = hash_content("Hello, world!")
        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" + 64 hex chars

    def test_hash_bytes(self):
        """Should hash bytes content."""
        result = hash_content(b"Hello, world!")
        assert result.startswith("sha256:")

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        hash1 = hash_content("test content")
        hash2 = hash_content("test content")
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = hash_content("content A")
        hash2 = hash_content("content B")
        assert hash1 != hash2

    def test_empty_string(self):
        """Should handle empty string."""
        result = hash_content("")
        assert result.startswith("sha256:")

    def test_unicode_content(self):
        """Should handle unicode content."""
        result = hash_content("Hello, 世界! 🌍")
        assert result.startswith("sha256:")


class TestHashOperatorId:
    """Tests for hash_operator_id function."""

    def test_pseudonymizes_operator_id(self):
        """Should pseudonymize operator ID."""
        result = hash_operator_id("operator@example.com")
        assert result.startswith("op_sha256:")
        assert len(result) == 26  # "op_sha256:" + 16 hex chars

    def test_same_id_same_hash(self):
        """Same operator ID should produce same hash."""
        hash1 = hash_operator_id("operator123")
        hash2 = hash_operator_id("operator123")
        assert hash1 == hash2

    def test_different_id_different_hash(self):
        """Different operator IDs should produce different hashes."""
        hash1 = hash_operator_id("operator1")
        hash2 = hash_operator_id("operator2")
        assert hash1 != hash2

    def test_salt_changes_hash(self):
        """Salt should change the resulting hash."""
        hash1 = hash_operator_id("operator", salt="deployment-a")
        hash2 = hash_operator_id("operator", salt="deployment-b")
        assert hash1 != hash2

    def test_same_salt_same_hash(self):
        """Same salt should produce same hash."""
        hash1 = hash_operator_id("operator", salt="deployment-a")
        hash2 = hash_operator_id("operator", salt="deployment-a")
        assert hash1 == hash2

    def test_no_salt_consistent(self):
        """No salt should still produce consistent hashes."""
        hash1 = hash_operator_id("operator")
        hash2 = hash_operator_id("operator")
        assert hash1 == hash2


class TestGenerateEntryId:
    """Tests for generate_entry_id function."""

    def test_format(self):
        """Should produce correct format."""
        entry_id = generate_entry_id()
        assert entry_id.startswith("evt_")
        # Format: evt_YYYYMMDD_HHMMSSZ_xxxx
        parts = entry_id.split("_")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert parts[2].endswith("Z")  # HHMMSSZ
        assert len(parts[3]) == 4  # xxxx hex

    def test_uniqueness(self):
        """Should generate unique IDs."""
        ids = [generate_entry_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_contains_timestamp(self):
        """Should contain current date."""
        entry_id = generate_entry_id()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in entry_id


class TestComputeEntryHash:
    """Tests for compute_entry_hash function."""

    def test_computes_hash(self):
        """Should compute hash of entry dict."""
        entry = {
            "entry_id": "evt_test",
            "deployment_id": "test-deployment",
            "signals": {"tick_count": 5},
        }
        result = compute_entry_hash(entry, prev_hash="sha256:previous")
        assert result.startswith("sha256:")

    def test_prev_hash_affects_result(self):
        """Different prev_hash should produce different result."""
        entry = {"entry_id": "test", "data": "same"}
        hash1 = compute_entry_hash(entry, prev_hash="sha256:aaa")
        hash2 = compute_entry_hash(entry, prev_hash="sha256:bbb")
        assert hash1 != hash2

    def test_entry_hash_excluded(self):
        """entry_hash field in input should be ignored."""
        entry1 = {"entry_id": "test", "entry_hash": "sha256:should_ignore"}
        entry2 = {"entry_id": "test", "entry_hash": "sha256:different"}
        hash1 = compute_entry_hash(entry1, prev_hash="sha256:prev")
        hash2 = compute_entry_hash(entry2, prev_hash="sha256:prev")
        assert hash1 == hash2

    def test_data_changes_hash(self):
        """Different data should produce different hash."""
        entry1 = {"entry_id": "test", "value": 1}
        entry2 = {"entry_id": "test", "value": 2}
        hash1 = compute_entry_hash(entry1, prev_hash="sha256:prev")
        hash2 = compute_entry_hash(entry2, prev_hash="sha256:prev")
        assert hash1 != hash2

    def test_empty_prev_hash(self):
        """Should handle empty prev_hash (genesis entry)."""
        entry = {"entry_id": "genesis"}
        result = compute_entry_hash(entry, prev_hash="")
        assert result.startswith("sha256:")
