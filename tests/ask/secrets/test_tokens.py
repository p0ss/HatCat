"""Tests for RFC 3161 Timestamp Token client."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import json

from src.ask.secrets.tokens import (
    TSAConfig,
    TimestampRequest,
    TimestampToken,
    TimestampClient,
    PendingTimestamp,
    create_mock_timestamp,
    DEFAULT_TSAS,
)


class TestTSAConfig:
    """Tests for TSA configuration."""

    def test_create_basic_config(self):
        """Should create config with required fields."""
        config = TSAConfig(
            authority_id="test-tsa",
            url="https://tsa.example.com/timestamp",
        )

        assert config.authority_id == "test-tsa"
        assert config.url == "https://tsa.example.com/timestamp"
        assert config.timeout_seconds == 30
        assert config.retry_count == 3

    def test_create_config_with_auth(self):
        """Should create config with authentication."""
        config = TSAConfig(
            authority_id="secure-tsa",
            url="https://secure.example.com",
            username="user",
            password="pass",
        )

        assert config.username == "user"
        assert config.password == "pass"

    def test_to_dict(self):
        """Should serialize to dict."""
        config = TSAConfig(
            authority_id="test",
            url="https://test.com",
            timeout_seconds=60,
        )

        data = config.to_dict()

        assert data["authority_id"] == "test"
        assert data["url"] == "https://test.com"
        assert data["timeout_seconds"] == 60

    def test_default_tsas_exist(self):
        """Should have default TSA configurations."""
        assert len(DEFAULT_TSAS) >= 1
        for tsa in DEFAULT_TSAS:
            assert tsa.authority_id
            assert tsa.url.startswith("https://")


class TestTimestampRequest:
    """Tests for timestamp request creation."""

    def test_create_from_hash(self):
        """Should create request from hash string."""
        request = TimestampRequest.create("sha256:abcdef1234567890")

        assert request.hash_algorithm == "sha256"
        assert request.hash_value == bytes.fromhex("abcdef1234567890")
        assert request.cert_req is True

    def test_create_with_nonce(self):
        """Should include nonce by default."""
        request = TimestampRequest.create("sha256:aabbccdd")

        assert request.nonce is not None
        assert len(request.nonce) == 8

    def test_create_without_nonce(self):
        """Should allow disabling nonce."""
        request = TimestampRequest.create("sha256:aabbccdd", include_nonce=False)

        assert request.nonce is None

    def test_create_hash_without_prefix(self):
        """Should handle hash without algorithm prefix."""
        request = TimestampRequest.create("aabbccdd11223344")

        assert request.hash_algorithm == "sha256"
        assert request.hash_value == bytes.fromhex("aabbccdd11223344")

    def test_to_asn1(self):
        """Should encode to bytes (simplified format)."""
        request = TimestampRequest.create("sha256:aabbccdd")
        encoded = request.to_asn1()

        assert isinstance(encoded, bytes)
        # Should be parseable JSON in our simplified implementation
        data = json.loads(encoded.decode("utf-8"))
        assert data["version"] == 1
        assert data["hash_algorithm"] == "sha256"

    def test_to_dict(self):
        """Should serialize to dict."""
        request = TimestampRequest.create("sha256:aabbccdd")
        data = request.to_dict()

        assert data["hash_algorithm"] == "sha256"
        assert data["hash_value"] == "aabbccdd"
        assert data["cert_req"] is True


class TestTimestampToken:
    """Tests for timestamp token handling."""

    def test_from_response_json(self):
        """Should parse JSON response."""
        response = json.dumps({
            "token_id": "tst_test123",
            "timestamp": "2024-01-15T12:00:00+00:00",
            "hash_algorithm": "sha256",
            "hash_value": "aabbccdd",
            "tsa_name": "test-tsa",
            "serial_number": 12345,
            "verified": True,
        }).encode("utf-8")

        token = TimestampToken.from_response(response, "test-tsa")

        assert token.token_id == "tst_test123"
        assert token.hash_algorithm == "sha256"
        assert token.hash_value == bytes.fromhex("aabbccdd")
        assert token.tsa_name == "test-tsa"
        assert token.serial_number == 12345

    def test_from_response_invalid(self):
        """Should handle invalid response gracefully."""
        token = TimestampToken.from_response(b"invalid json", "test")

        assert token.verification_error is not None

    def test_verify_matching_hash(self):
        """Should verify when hash matches."""
        token = TimestampToken(
            token_id="test",
            hash_algorithm="sha256",
            hash_value=bytes.fromhex("aabbccdd"),
        )

        result = token.verify("sha256:aabbccdd")

        assert result is True
        assert token.verified is True

    def test_verify_mismatched_hash(self):
        """Should fail verification on hash mismatch."""
        token = TimestampToken(
            token_id="test",
            hash_algorithm="sha256",
            hash_value=bytes.fromhex("aabbccdd"),
        )

        result = token.verify("sha256:11223344")

        assert result is False
        assert token.verified is False
        assert token.verification_error == "Hash mismatch"

    def test_to_dict(self):
        """Should serialize to dict."""
        token = TimestampToken(
            token_id="tst_test",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            hash_algorithm="sha256",
            hash_value=bytes.fromhex("aabbccdd"),
            tsa_name="test-tsa",
            verified=True,
        )

        data = token.to_dict()

        assert data["token_id"] == "tst_test"
        assert data["hash_algorithm"] == "sha256"
        assert data["hash_value"] == "aabbccdd"
        assert data["tsa_name"] == "test-tsa"
        assert data["verified"] is True

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "token_id": "tst_test",
            "timestamp": "2024-01-15T12:00:00+00:00",
            "hash_algorithm": "sha256",
            "hash_value": "aabbccdd",
            "tsa_name": "test-tsa",
            "verified": True,
        }

        token = TimestampToken.from_dict(data)

        assert token.token_id == "tst_test"
        assert token.hash_value == bytes.fromhex("aabbccdd")
        assert token.verified is True

    def test_to_bytes_from_bytes_roundtrip(self):
        """Should roundtrip through bytes serialization."""
        original = TimestampToken(
            token_id="tst_roundtrip",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            hash_algorithm="sha256",
            hash_value=bytes.fromhex("aabbccdd11223344"),
            nonce=bytes.fromhex("1122334455667788"),
            tsa_name="test-tsa",
            serial_number=99999,
            raw_token=b"raw_token_data_here",
            verified=True,
        )

        serialized = original.to_bytes()
        restored = TimestampToken.from_bytes(serialized)

        assert restored.token_id == original.token_id
        assert restored.hash_value == original.hash_value
        assert restored.nonce == original.nonce
        assert restored.raw_token == original.raw_token
        assert restored.verified == original.verified


class TestPendingTimestamp:
    """Tests for pending timestamp requests."""

    def test_create_pending(self):
        """Should create pending request."""
        pending = PendingTimestamp(
            request_id="req_test",
            hash_value="sha256:aabbccdd",
            batch_id="batch_123",
            created_at=datetime.now(timezone.utc),
        )

        assert pending.request_id == "req_test"
        assert pending.hash_value == "sha256:aabbccdd"
        assert pending.retry_count == 0

    def test_to_dict(self):
        """Should serialize to dict."""
        pending = PendingTimestamp(
            request_id="req_test",
            hash_value="sha256:aabbccdd",
            batch_id="batch_123",
            created_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            retry_count=3,
            last_error="Network timeout",
        )

        data = pending.to_dict()

        assert data["request_id"] == "req_test"
        assert data["hash_value"] == "sha256:aabbccdd"
        assert data["batch_id"] == "batch_123"
        assert data["retry_count"] == 3
        assert data["last_error"] == "Network timeout"


class TestTimestampClient:
    """Tests for timestamp client."""

    def test_create_with_defaults(self):
        """Should create client with default TSAs."""
        client = TimestampClient()

        assert len(client.tsa_configs) >= 1
        assert client.get_pending_count() == 0

    def test_create_with_custom_tsas(self):
        """Should accept custom TSA configs."""
        custom = [TSAConfig(authority_id="custom", url="https://custom.com")]
        client = TimestampClient(tsa_configs=custom)

        assert len(client.tsa_configs) == 1
        assert client.tsa_configs[0].authority_id == "custom"

    def test_get_token_returns_cached(self):
        """Should return cached token for batch."""
        client = TimestampClient()

        # Manually cache a token
        token = create_mock_timestamp("sha256:aabbccdd")
        client._tokens["batch_123"] = token

        result = client.get_token("batch_123")
        assert result == token

    def test_get_token_missing(self):
        """Should return None for unknown batch."""
        client = TimestampClient()

        result = client.get_token("unknown_batch")
        assert result is None

    def test_verify_token(self):
        """Should verify token against hash."""
        client = TimestampClient()
        token = create_mock_timestamp("sha256:aabbccdd")

        result = client.verify_token(token, "sha256:aabbccdd")
        assert result is True

    def test_get_pending_requests(self):
        """Should return pending request list."""
        client = TimestampClient()

        # Add pending manually
        pending = PendingTimestamp(
            request_id="req_1",
            hash_value="sha256:test",
            batch_id="batch_1",
            created_at=datetime.now(timezone.utc),
        )
        client._pending.append(pending)

        result = client.get_pending_requests()
        assert len(result) == 1
        assert result[0].request_id == "req_1"

    @patch("urllib.request.urlopen")
    def test_request_timestamp_success(self, mock_urlopen):
        """Should handle successful TSA response."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "token_id": "tst_success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hash_algorithm": "sha256",
            "hash_value": "aabbccdd11223344",
            "verified": True,
        }).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = TimestampClient()
        token, error = client.request_timestamp(
            "sha256:aabbccdd11223344",
            batch_id="batch_test",
        )

        assert error is None
        assert token is not None
        assert token.token_id == "tst_success"

    @patch("urllib.request.urlopen")
    def test_request_timestamp_network_failure(self, mock_urlopen):
        """Should handle network failure."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = TimestampClient()
        token, error = client.request_timestamp(
            "sha256:aabbccdd",
            batch_id="batch_fail",
        )

        assert token is None
        assert "Network error" in error or "failed" in error.lower()

    def test_request_timestamp_unknown_tsa(self):
        """Should return error for unknown TSA."""
        client = TimestampClient()
        token, error = client.request_timestamp(
            "sha256:aabbccdd",
            tsa_id="nonexistent-tsa",
        )

        assert token is None
        assert "Unknown TSA" in error


class TestCreateMockTimestamp:
    """Tests for mock timestamp creation."""

    def test_creates_valid_token(self):
        """Should create valid mock token."""
        token = create_mock_timestamp("sha256:aabbccdd11223344")

        assert token.token_id.startswith("tst_")
        assert token.hash_algorithm == "sha256"
        assert token.hash_value == bytes.fromhex("aabbccdd11223344")
        assert token.tsa_name == "mock-tsa"
        assert token.verified is True

    def test_custom_tsa_name(self):
        """Should accept custom TSA name."""
        token = create_mock_timestamp("sha256:aabbccdd", tsa_name="custom-mock")

        assert token.tsa_name == "custom-mock"

    def test_has_nonce(self):
        """Should include random nonce."""
        token = create_mock_timestamp("sha256:aabbccdd")

        assert token.nonce is not None
        assert len(token.nonce) == 8

    def test_has_serial_number(self):
        """Should include serial number."""
        token = create_mock_timestamp("sha256:aabbccdd")

        assert token.serial_number >= 0

    def test_has_timestamp(self):
        """Should include current timestamp."""
        before = datetime.now(timezone.utc)
        token = create_mock_timestamp("sha256:aabbccdd")
        after = datetime.now(timezone.utc)

        assert before <= token.timestamp <= after

    def test_handles_hash_without_prefix(self):
        """Should handle hash without algorithm prefix."""
        token = create_mock_timestamp("aabbccdd11223344")

        assert token.hash_value == bytes.fromhex("aabbccdd11223344")

    def test_uniqueness(self):
        """Should generate unique tokens."""
        tokens = [create_mock_timestamp("sha256:aabbccdd") for _ in range(10)]
        token_ids = [t.token_id for t in tokens]
        nonces = [t.nonce for t in tokens]

        assert len(set(token_ids)) == 10
        assert len(set(nonces)) == 10


class MockTimestampClient:
    """Mock client that returns successful timestamps."""

    def __init__(self):
        self.tsa_configs = []
        self._tokens = {}
        self._pending = []

    def request_timestamp(
        self,
        data_hash: str,
        batch_id: str = "",
        tsa_id: str = None,
    ):
        """Return mock timestamp for any hash."""
        token = create_mock_timestamp(data_hash, "mock-tsa")
        self._tokens[batch_id] = token
        return token, None

    def get_token(self, batch_id: str):
        return self._tokens.get(batch_id)

    def get_pending_count(self):
        return 0


class TestBatchIntegration:
    """Tests for batch-timestamp integration."""

    def test_batch_with_mock_timestamp(self):
        """Should integrate mock timestamp with batch."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:aabbccdd")
        batch.seal()

        # Request mock timestamp with mock client
        mock_client = MockTimestampClient()
        success, error = batch.request_timestamp(client=mock_client)

        assert success is True
        assert error is None
        assert batch.get_timestamp_token() is not None

    def test_batch_verify_timestamp(self):
        """Should verify batch timestamp."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:aabbccdd")
        batch.seal()

        mock_client = MockTimestampClient()
        batch.request_timestamp(client=mock_client)

        valid, error = batch.verify_timestamp()

        assert valid is True
        assert error is None

    def test_manager_with_timestamp_client(self):
        """Should accept timestamp client in manager."""
        from src.ask.storage.batches import BatchManager, BatchConfig

        client = TimestampClient()
        config = BatchConfig(require_rfc3161=True)
        manager = BatchManager(
            deployment_id="test",
            config=config,
            timestamp_client=client,
        )

        assert manager._timestamp_client == client

    def test_seal_with_timestamp_request(self):
        """Should request timestamp when sealing."""
        from src.ask.storage.batches import BatchManager, BatchConfig

        config = BatchConfig(require_rfc3161=True)
        mock_client = MockTimestampClient()
        manager = BatchManager(
            deployment_id="test",
            config=config,
            timestamp_client=mock_client,
        )

        manager.add_entry("e1", "sha256:aabbccdd")
        sealed = manager.seal_current_batch(request_timestamp=True)

        assert sealed is not None
        assert sealed._sealed is True
        # Mock timestamp should be available
        assert sealed.get_timestamp_token() is not None

    @patch("urllib.request.urlopen")
    def test_batch_with_real_tsa_response(self, mock_urlopen):
        """Should handle real TSA responses."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:aabbccdd")
        batch.seal()

        # Mock successful TSA response matching the merkle root
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "token_id": "tst_real",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hash_algorithm": "sha256",
            "hash_value": batch.merkle_root.split(":")[1],  # Extract hex from sha256:xxx
            "verified": True,
        }).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        success, error = batch.request_timestamp()

        assert success is True
        assert error is None
        assert batch.get_timestamp_token() is not None
