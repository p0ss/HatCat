"""Tests for authority replication - multi-authority receipt system."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, Mock
import json

from src.ask.replication.authorities import (
    AuthorityType,
    ReceiptStatus,
    AuthorityConfig,
    AuthorityReceipt,
    SubmissionRequest,
    AuthorityClient,
    create_mock_authority,
    create_regulator_authority,
    create_notary_authority,
)


class TestAuthorityConfig:
    """Tests for authority configuration."""

    def test_create_basic_config(self):
        """Should create config with required fields."""
        config = AuthorityConfig(
            authority_id="test-auth",
            name="Test Authority",
        )

        assert config.authority_id == "test-auth"
        assert config.name == "Test Authority"
        assert config.authority_type == AuthorityType.NOTARY
        assert config.timeout_seconds == 30

    def test_create_with_type(self):
        """Should accept authority type."""
        config = AuthorityConfig(
            authority_id="reg-1",
            name="Regulator",
            authority_type=AuthorityType.REGULATOR,
            required_for_compliance=True,
        )

        assert config.authority_type == AuthorityType.REGULATOR
        assert config.required_for_compliance is True

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = AuthorityConfig(
            authority_id="test",
            name="Test",
            authority_type=AuthorityType.AUDITOR,
            url="https://example.com",
            required_for_compliance=True,
        )

        data = original.to_dict()
        restored = AuthorityConfig.from_dict(data)

        assert restored.authority_id == original.authority_id
        assert restored.authority_type == original.authority_type
        assert restored.required_for_compliance == original.required_for_compliance


class TestAuthorityReceipt:
    """Tests for authority receipts."""

    def test_create_receipt(self):
        """Should create receipt with ID."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
            batch_hash="sha256:abc",
        )

        assert receipt.receipt_id.startswith("rcpt_")
        assert receipt.authority_id == "auth-1"
        assert receipt.status == ReceiptStatus.PENDING

    def test_receipt_is_valid_confirmed(self):
        """Confirmed receipt should be valid."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
            status=ReceiptStatus.CONFIRMED,
        )

        assert receipt.is_valid() is True

    def test_receipt_is_valid_rejected(self):
        """Rejected receipt should not be valid."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
            status=ReceiptStatus.REJECTED,
        )

        assert receipt.is_valid() is False

    def test_receipt_is_valid_expired(self):
        """Expired receipt should not be valid."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
            status=ReceiptStatus.CONFIRMED,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert receipt.is_valid() is False

    def test_get_signed_payload(self):
        """Should generate canonical signed payload."""
        receipt = AuthorityReceipt(
            receipt_id="rcpt_test",
            authority_id="auth-1",
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            received_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            confirmation_id="conf_123",
        )

        payload = receipt.get_signed_payload()

        # Should be valid JSON
        data = json.loads(payload)
        assert data["receipt_id"] == "rcpt_test"
        assert data["batch_id"] == "batch_123"

    def test_verify_mock_signature(self):
        """Should verify mock signature."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
            batch_hash="sha256:abc",
            received_at=datetime.now(timezone.utc),
        )

        # Create mock signature
        receipt.signed_payload = receipt.get_signed_payload()
        from src.ask.secrets.hashing import hash_content
        payload_hash = hash_content(receipt.signed_payload)
        receipt.signature = f"mock_sig:{payload_hash.split(':')[1][:32]}"

        result = receipt.verify_signature()

        assert result is True
        assert receipt.verified is True

    def test_verify_no_signature(self):
        """Should fail verification without signature."""
        receipt = AuthorityReceipt(
            authority_id="auth-1",
            batch_id="batch_123",
        )

        result = receipt.verify_signature()

        assert result is False
        assert receipt.verification_error == "No signature present"

    def test_to_dict_from_dict(self):
        """Should roundtrip through dict."""
        original = AuthorityReceipt(
            authority_id="auth-1",
            authority_name="Test Authority",
            authority_type=AuthorityType.REGULATOR,
            batch_id="batch_123",
            batch_hash="sha256:abc",
            status=ReceiptStatus.CONFIRMED,
            confirmation_id="conf_456",
            verified=True,
        )

        data = original.to_dict()
        restored = AuthorityReceipt.from_dict(data)

        assert restored.receipt_id == original.receipt_id
        assert restored.authority_type == original.authority_type
        assert restored.status == original.status
        assert restored.verified == original.verified


class TestSubmissionRequest:
    """Tests for submission requests."""

    def test_create_request(self):
        """Should create request with ID."""
        request = SubmissionRequest(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        assert request.request_id.startswith("sub_")
        assert request.batch_id == "batch_123"

    def test_to_payload(self):
        """Should create submission payload."""
        request = SubmissionRequest(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            deployment_id="deploy_1",
            jurisdiction="AU",
        )

        payload = request.to_payload()

        assert payload["batch_id"] == "batch_123"
        assert payload["batch_hash"] == "sha256:abc"
        assert payload["deployment_id"] == "deploy_1"
        assert "submitted_at" in payload


class TestAuthorityClient:
    """Tests for authority client."""

    def test_create_client(self):
        """Should create client with authorities."""
        auth1 = create_mock_authority("auth-1")
        auth2 = create_mock_authority("auth-2")

        client = AuthorityClient(authorities=[auth1, auth2])

        assert len(client.authorities) == 2
        assert "auth-1" in client.authorities

    def test_add_remove_authority(self):
        """Should add and remove authorities."""
        client = AuthorityClient()

        client.add_authority(create_mock_authority("auth-1"))
        assert "auth-1" in client.authorities

        client.remove_authority("auth-1")
        assert "auth-1" not in client.authorities

    def test_submit_batch_mock(self):
        """Should submit batch with mock authority."""
        auth = create_mock_authority("mock-auth", "Mock Authority")
        client = AuthorityClient(authorities=[auth])

        receipt, error = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        assert error is None
        assert receipt is not None
        assert receipt.authority_id == "mock-auth"
        assert receipt.status == ReceiptStatus.CONFIRMED
        assert receipt.verified is True

    def test_submit_batch_specific_authority(self):
        """Should submit to specific authority."""
        auth1 = create_mock_authority("auth-1")
        auth2 = create_mock_authority("auth-2")
        client = AuthorityClient(authorities=[auth1, auth2])

        receipt, error = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            authority_id="auth-2",
        )

        assert receipt.authority_id == "auth-2"

    def test_submit_batch_unknown_authority(self):
        """Should error for unknown authority."""
        client = AuthorityClient()

        receipt, error = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
            authority_id="unknown",
        )

        assert receipt is None
        assert "Unknown authority" in error

    def test_submit_batch_no_authorities(self):
        """Should error when no authorities configured."""
        client = AuthorityClient()

        receipt, error = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        assert receipt is None
        assert "No authorities" in error

    def test_submit_to_all(self):
        """Should submit to all authorities."""
        authorities = [
            create_mock_authority(f"auth-{i}")
            for i in range(3)
        ]
        client = AuthorityClient(authorities=authorities)

        receipts, errors = client.submit_to_all(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        assert len(receipts) == 3
        assert len(errors) == 0
        assert {r.authority_id for r in receipts} == {"auth-0", "auth-1", "auth-2"}

    def test_get_receipts(self):
        """Should retrieve receipts for batch."""
        auth = create_mock_authority("auth-1")
        client = AuthorityClient(authorities=[auth])

        client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        receipts = client.get_receipts("batch_123")

        assert len(receipts) == 1
        assert receipts[0].batch_id == "batch_123"

    def test_get_receipt_by_authority(self):
        """Should get specific authority's receipt."""
        authorities = [
            create_mock_authority("auth-1"),
            create_mock_authority("auth-2"),
        ]
        client = AuthorityClient(authorities=authorities)

        client.submit_to_all(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        receipt = client.get_receipt_by_authority("batch_123", "auth-2")

        assert receipt is not None
        assert receipt.authority_id == "auth-2"

    def test_verify_receipt_success(self):
        """Should verify valid receipt."""
        auth = create_mock_authority("auth-1")
        client = AuthorityClient(authorities=[auth])

        receipt, _ = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        valid, error = client.verify_receipt(receipt, "sha256:abc")

        assert valid is True
        assert error is None

    def test_verify_receipt_hash_mismatch(self):
        """Should fail verification on hash mismatch."""
        auth = create_mock_authority("auth-1")
        client = AuthorityClient(authorities=[auth])

        receipt, _ = client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        valid, error = client.verify_receipt(receipt, "sha256:different")

        assert valid is False
        assert "hash mismatch" in error.lower()

    def test_check_compliance_met(self):
        """Should report compliance when required receipts present."""
        auth = AuthorityConfig(
            authority_id="regulator",
            name="Regulator",
            required_for_compliance=True,
        )
        client = AuthorityClient(authorities=[auth])

        client.submit_batch(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        compliant, missing = client.check_compliance("batch_123")

        assert compliant is True
        assert len(missing) == 0

    def test_check_compliance_missing(self):
        """Should report missing required authorities."""
        auth = AuthorityConfig(
            authority_id="regulator",
            name="Regulator",
            required_for_compliance=True,
        )
        client = AuthorityClient(authorities=[auth])

        # Don't submit

        compliant, missing = client.check_compliance("batch_123")

        assert compliant is False
        assert "regulator" in missing

    def test_get_receipt_summary(self):
        """Should return receipt summary."""
        authorities = [
            create_mock_authority("auth-1"),
            create_mock_authority("auth-2"),
        ]
        client = AuthorityClient(authorities=authorities)

        client.submit_to_all(
            batch_id="batch_123",
            batch_hash="sha256:abc",
            merkle_root="sha256:def",
        )

        summary = client.get_receipt_summary("batch_123")

        assert summary["batch_id"] == "batch_123"
        assert summary["total_receipts"] == 2
        assert summary["confirmed"] == 2
        assert summary["verified"] == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_mock_authority(self):
        """Should create mock authority."""
        auth = create_mock_authority("test", "Test Auth")

        assert auth.authority_id == "test"
        assert auth.name == "Test Auth"
        assert auth.url == ""  # Empty triggers mock mode

    def test_create_regulator_authority(self):
        """Should create regulator authority."""
        auth = create_regulator_authority(
            authority_id="acma",
            name="ACMA",
            url="https://api.acma.gov.au",
            api_key="secret",
        )

        assert auth.authority_type == AuthorityType.REGULATOR
        assert auth.required_for_compliance is True
        assert auth.url == "https://api.acma.gov.au"

    def test_create_notary_authority(self):
        """Should create notary authority."""
        auth = create_notary_authority(
            authority_id="notary-1",
            name="Crypto Notary",
        )

        assert auth.authority_type == AuthorityType.NOTARY


class TestBatchIntegration:
    """Integration tests with batch system."""

    def test_receipt_for_sealed_batch(self):
        """Should get receipt for sealed batch."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:abc")
        batch.seal()

        auth = create_mock_authority("notary")
        client = AuthorityClient(authorities=[auth])

        receipt, error = client.submit_batch(
            batch_id=batch.batch_id,
            batch_hash=batch.batch_hash,
            merkle_root=batch.merkle_root,
        )

        assert error is None
        assert receipt.batch_hash == batch.batch_hash
        assert receipt.merkle_root == batch.merkle_root

    def test_multi_authority_for_batch(self):
        """Should get receipts from multiple authorities."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test")
        batch.add_entry("e1", "sha256:abc")
        batch.seal()

        authorities = [
            create_mock_authority("notary-1"),
            create_mock_authority("notary-2"),
            create_regulator_authority("regulator", "Reg", ""),
        ]
        client = AuthorityClient(authorities=authorities)

        receipts, errors = client.submit_to_all(
            batch_id=batch.batch_id,
            batch_hash=batch.batch_hash,
            merkle_root=batch.merkle_root,
            summary={"entry_count": batch.entry_count},
        )

        assert len(receipts) == 3
        assert len(errors) == 0

        # Check compliance
        compliant, missing = client.check_compliance(batch.batch_id)
        assert compliant is True

    def test_receipt_with_batch_summary(self):
        """Should include batch summary in submission."""
        from src.ask.storage.batches import AuditBatch

        batch = AuditBatch.create(deployment_id="test", jurisdiction="AU")
        for i in range(5):
            batch.add_entry(f"e{i}", f"sha256:h{i}")
        batch.seal()

        auth = create_mock_authority("notary")
        client = AuthorityClient(authorities=[auth])

        receipt, _ = client.submit_batch(
            batch_id=batch.batch_id,
            batch_hash=batch.batch_hash,
            merkle_root=batch.merkle_root,
            summary={
                "entry_count": batch.entry_count,
                "deployment_id": batch.deployment_id,
            },
            deployment_id=batch.deployment_id,
            jurisdiction=batch.jurisdiction,
        )

        assert receipt is not None
        assert receipt.batch_id == batch.batch_id
