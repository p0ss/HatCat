"""
Authority Replication - multi-authority receipt system.

Provides third-party proof that audit batches were submitted to trusted
authorities. Multiple authorities provide redundancy and regulatory evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import secrets
import json
import hashlib
import base64

from ..secrets.hashing import hash_content


class AuthorityType(Enum):
    """Types of authorities that can receive audit data."""

    # Regulatory authorities
    REGULATOR = "regulator"  # Government regulatory body
    AUDITOR = "auditor"  # Independent auditor

    # Technical witnesses
    NOTARY = "notary"  # Cryptographic notary service
    ARCHIVE = "archive"  # Long-term archive service

    # Internal
    INTERNAL = "internal"  # Internal compliance system
    BACKUP = "backup"  # Backup/disaster recovery


class ReceiptStatus(Enum):
    """Status of an authority receipt."""

    PENDING = "pending"  # Submitted, awaiting confirmation
    CONFIRMED = "confirmed"  # Authority confirmed receipt
    REJECTED = "rejected"  # Authority rejected submission
    EXPIRED = "expired"  # Receipt validity expired
    REVOKED = "revoked"  # Receipt was revoked


@dataclass
class AuthorityConfig:
    """Configuration for an authority endpoint."""

    authority_id: str  # Unique identifier
    name: str  # Human-readable name
    authority_type: AuthorityType = AuthorityType.NOTARY

    # Endpoint
    url: str = ""
    api_version: str = "v1"

    # Authentication
    api_key: Optional[str] = None
    client_cert_path: Optional[str] = None
    signing_key_path: Optional[str] = None

    # Behavior
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: int = 5

    # Verification
    public_key: Optional[str] = None  # Authority's public key for verification
    ca_cert_path: Optional[str] = None

    # Requirements
    required_for_compliance: bool = False  # Must have receipt for compliance
    min_receipts_required: int = 1  # Minimum receipts from this authority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "name": self.name,
            "authority_type": self.authority_type.value,
            "url": self.url,
            "api_version": self.api_version,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "required_for_compliance": self.required_for_compliance,
            "min_receipts_required": self.min_receipts_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorityConfig":
        return cls(
            authority_id=data["authority_id"],
            name=data.get("name", data["authority_id"]),
            authority_type=AuthorityType(data.get("authority_type", "notary")),
            url=data.get("url", ""),
            api_version=data.get("api_version", "v1"),
            timeout_seconds=data.get("timeout_seconds", 30),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 5),
            required_for_compliance=data.get("required_for_compliance", False),
            min_receipts_required=data.get("min_receipts_required", 1),
        )


@dataclass
class AuthorityReceipt:
    """
    Receipt from an authority confirming batch submission.

    Contains cryptographic proof that the authority received and
    acknowledged the batch data.
    """

    receipt_id: str = ""

    # Authority info
    authority_id: str = ""
    authority_name: str = ""
    authority_type: AuthorityType = AuthorityType.NOTARY

    # What was submitted
    batch_id: str = ""
    batch_hash: str = ""
    merkle_root: str = ""

    # Timing
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    received_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Authority response
    status: ReceiptStatus = ReceiptStatus.PENDING
    confirmation_id: str = ""  # Authority's internal ID
    message: str = ""

    # Cryptographic proof
    signature: str = ""  # Authority's signature over the receipt
    signature_algorithm: str = "ed25519"
    signed_payload: str = ""  # What was signed

    # Verification
    verified: bool = False
    verification_error: Optional[str] = None

    def __post_init__(self):
        if not self.receipt_id:
            self.receipt_id = f"rcpt_{secrets.token_hex(8)}"

    def get_signed_payload(self) -> str:
        """Get the canonical payload that should be signed."""
        payload = {
            "receipt_id": self.receipt_id,
            "authority_id": self.authority_id,
            "batch_id": self.batch_id,
            "batch_hash": self.batch_hash,
            "merkle_root": self.merkle_root,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "confirmation_id": self.confirmation_id,
        }
        # Canonical JSON (sorted keys, no whitespace)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def verify_signature(self, public_key: Optional[bytes] = None) -> bool:
        """
        Verify the authority's signature on this receipt.

        Args:
            public_key: Authority's public key (if not embedded)

        Returns:
            True if signature is valid
        """
        if not self.signature:
            self.verification_error = "No signature present"
            self.verified = False
            return False

        if not self.signed_payload:
            self.signed_payload = self.get_signed_payload()

        # In production, this would use cryptography library
        # For now, we verify the hash matches
        try:
            expected_hash = hash_content(self.signed_payload)
            # Simple verification: signature should be hash of payload
            # In production: verify EdDSA/RSA signature with public key

            # For testing/demo: check if signature matches our mock format
            if self.signature.startswith("mock_sig:"):
                sig_hash = self.signature.split(":", 1)[1]
                if sig_hash == expected_hash.split(":", 1)[1][:32]:
                    self.verified = True
                    return True

            self.verification_error = "Signature verification not implemented"
            self.verified = False
            return False

        except Exception as e:
            self.verification_error = str(e)
            self.verified = False
            return False

    def is_valid(self) -> bool:
        """Check if receipt is currently valid."""
        if self.status not in (ReceiptStatus.CONFIRMED, ReceiptStatus.PENDING):
            return False

        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "authority_id": self.authority_id,
            "authority_name": self.authority_name,
            "authority_type": self.authority_type.value,
            "batch_id": self.batch_id,
            "batch_hash": self.batch_hash,
            "merkle_root": self.merkle_root,
            "submitted_at": self.submitted_at.isoformat(),
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "confirmation_id": self.confirmation_id,
            "message": self.message,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "verified": self.verified,
            "verification_error": self.verification_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorityReceipt":
        receipt = cls(
            receipt_id=data.get("receipt_id", ""),
            authority_id=data.get("authority_id", ""),
            authority_name=data.get("authority_name", ""),
            authority_type=AuthorityType(data.get("authority_type", "notary")),
            batch_id=data.get("batch_id", ""),
            batch_hash=data.get("batch_hash", ""),
            merkle_root=data.get("merkle_root", ""),
            status=ReceiptStatus(data.get("status", "pending")),
            confirmation_id=data.get("confirmation_id", ""),
            message=data.get("message", ""),
            signature=data.get("signature", ""),
            signature_algorithm=data.get("signature_algorithm", "ed25519"),
            verified=data.get("verified", False),
            verification_error=data.get("verification_error"),
        )

        if data.get("submitted_at"):
            receipt.submitted_at = datetime.fromisoformat(data["submitted_at"])
        if data.get("received_at"):
            receipt.received_at = datetime.fromisoformat(data["received_at"])
        if data.get("expires_at"):
            receipt.expires_at = datetime.fromisoformat(data["expires_at"])

        return receipt


@dataclass
class SubmissionRequest:
    """Request to submit a batch to an authority."""

    request_id: str = ""
    batch_id: str = ""
    batch_hash: str = ""
    merkle_root: str = ""

    # Optional: include batch summary
    include_summary: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    deployment_id: str = ""
    jurisdiction: str = ""
    submission_reason: str = "routine"  # routine, compliance, audit, incident

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"sub_{secrets.token_hex(8)}"

    def to_payload(self) -> Dict[str, Any]:
        """Create submission payload for authority."""
        payload = {
            "request_id": self.request_id,
            "batch_id": self.batch_id,
            "batch_hash": self.batch_hash,
            "merkle_root": self.merkle_root,
            "deployment_id": self.deployment_id,
            "jurisdiction": self.jurisdiction,
            "submission_reason": self.submission_reason,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.include_summary and self.summary:
            payload["summary"] = self.summary

        return payload


class AuthorityClient:
    """
    Client for communicating with authorities.

    Handles submission of batches and retrieval of receipts.
    """

    def __init__(
        self,
        authorities: Optional[List[AuthorityConfig]] = None,
    ):
        self.authorities = {a.authority_id: a for a in (authorities or [])}

        # Pending submissions
        self._pending: Dict[str, SubmissionRequest] = {}

        # Received receipts (batch_id -> list of receipts)
        self._receipts: Dict[str, List[AuthorityReceipt]] = {}

    def add_authority(self, config: AuthorityConfig) -> None:
        """Add an authority configuration."""
        self.authorities[config.authority_id] = config

    def remove_authority(self, authority_id: str) -> None:
        """Remove an authority configuration."""
        self.authorities.pop(authority_id, None)

    def get_authority(self, authority_id: str) -> Optional[AuthorityConfig]:
        """Get authority configuration."""
        return self.authorities.get(authority_id)

    def submit_batch(
        self,
        batch_id: str,
        batch_hash: str,
        merkle_root: str,
        authority_id: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
        deployment_id: str = "",
        jurisdiction: str = "",
    ) -> Tuple[Optional[AuthorityReceipt], Optional[str]]:
        """
        Submit a batch to an authority.

        Args:
            batch_id: The batch identifier
            batch_hash: Hash of the batch
            merkle_root: Merkle root of the batch
            authority_id: Specific authority (None = first available)
            summary: Optional batch summary
            deployment_id: Deployment identifier
            jurisdiction: Jurisdiction code

        Returns:
            (receipt, error) - receipt if successful
        """
        # Get authority config
        if authority_id:
            config = self.authorities.get(authority_id)
            if not config:
                return None, f"Unknown authority: {authority_id}"
        else:
            if not self.authorities:
                return None, "No authorities configured"
            config = next(iter(self.authorities.values()))

        # Create submission request
        request = SubmissionRequest(
            batch_id=batch_id,
            batch_hash=batch_hash,
            merkle_root=merkle_root,
            summary=summary or {},
            deployment_id=deployment_id,
            jurisdiction=jurisdiction,
        )

        # Send to authority
        receipt, error = self._send_submission(request, config)

        if receipt:
            # Store receipt
            if batch_id not in self._receipts:
                self._receipts[batch_id] = []
            self._receipts[batch_id].append(receipt)

        return receipt, error

    def _send_submission(
        self,
        request: SubmissionRequest,
        config: AuthorityConfig,
    ) -> Tuple[Optional[AuthorityReceipt], Optional[str]]:
        """
        Send submission to authority endpoint.

        Returns:
            (receipt, error)
        """
        try:
            if not config.url:
                # Mock response for testing
                return self._create_mock_receipt(request, config), None

            # In production, this would make HTTP request
            import urllib.request
            import urllib.error

            payload = json.dumps(request.to_payload()).encode("utf-8")

            http_req = urllib.request.Request(
                f"{config.url}/{config.api_version}/submit",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            # Add auth if configured
            if config.api_key:
                http_req.add_header("Authorization", f"Bearer {config.api_key}")

            with urllib.request.urlopen(http_req, timeout=config.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            # Parse response into receipt
            receipt = AuthorityReceipt(
                authority_id=config.authority_id,
                authority_name=config.name,
                authority_type=config.authority_type,
                batch_id=request.batch_id,
                batch_hash=request.batch_hash,
                merkle_root=request.merkle_root,
                received_at=datetime.now(timezone.utc),
                status=ReceiptStatus.CONFIRMED,
                confirmation_id=response_data.get("confirmation_id", ""),
                signature=response_data.get("signature", ""),
            )

            return receipt, None

        except urllib.error.URLError as e:
            return None, f"Network error: {e}"
        except Exception as e:
            return None, f"Submission failed: {e}"

    def _create_mock_receipt(
        self,
        request: SubmissionRequest,
        config: AuthorityConfig,
    ) -> AuthorityReceipt:
        """Create a mock receipt for testing."""
        receipt = AuthorityReceipt(
            authority_id=config.authority_id,
            authority_name=config.name,
            authority_type=config.authority_type,
            batch_id=request.batch_id,
            batch_hash=request.batch_hash,
            merkle_root=request.merkle_root,
            received_at=datetime.now(timezone.utc),
            status=ReceiptStatus.CONFIRMED,
            confirmation_id=f"conf_{secrets.token_hex(8)}",
            message="Mock receipt for testing",
        )

        # Create mock signature
        receipt.signed_payload = receipt.get_signed_payload()
        payload_hash = hash_content(receipt.signed_payload)
        receipt.signature = f"mock_sig:{payload_hash.split(':')[1][:32]}"
        receipt.verified = True

        return receipt

    def submit_to_all(
        self,
        batch_id: str,
        batch_hash: str,
        merkle_root: str,
        summary: Optional[Dict[str, Any]] = None,
        deployment_id: str = "",
        jurisdiction: str = "",
    ) -> Tuple[List[AuthorityReceipt], List[str]]:
        """
        Submit batch to all configured authorities.

        Returns:
            (receipts, errors) - list of receipts and list of errors
        """
        receipts = []
        errors = []

        for authority_id in self.authorities:
            receipt, error = self.submit_batch(
                batch_id=batch_id,
                batch_hash=batch_hash,
                merkle_root=merkle_root,
                authority_id=authority_id,
                summary=summary,
                deployment_id=deployment_id,
                jurisdiction=jurisdiction,
            )

            if receipt:
                receipts.append(receipt)
            if error:
                errors.append(f"{authority_id}: {error}")

        return receipts, errors

    def get_receipts(self, batch_id: str) -> List[AuthorityReceipt]:
        """Get all receipts for a batch."""
        return self._receipts.get(batch_id, [])

    def get_receipt_by_authority(
        self,
        batch_id: str,
        authority_id: str,
    ) -> Optional[AuthorityReceipt]:
        """Get receipt from specific authority for a batch."""
        receipts = self._receipts.get(batch_id, [])
        for receipt in receipts:
            if receipt.authority_id == authority_id:
                return receipt
        return None

    def verify_receipt(
        self,
        receipt: AuthorityReceipt,
        expected_batch_hash: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a receipt is valid.

        Args:
            receipt: The receipt to verify
            expected_batch_hash: Expected batch hash

        Returns:
            (valid, error)
        """
        # Check batch hash matches
        if receipt.batch_hash != expected_batch_hash:
            return False, "Batch hash mismatch"

        # Check status
        if not receipt.is_valid():
            return False, f"Receipt not valid (status: {receipt.status.value})"

        # Verify signature
        config = self.authorities.get(receipt.authority_id)
        public_key = config.public_key.encode() if config and config.public_key else None

        if not receipt.verify_signature(public_key):
            return False, receipt.verification_error or "Signature verification failed"

        return True, None

    def check_compliance(
        self,
        batch_id: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if batch has required authority receipts for compliance.

        Returns:
            (compliant, missing_authorities)
        """
        receipts = self.get_receipts(batch_id)
        receipt_authorities = {r.authority_id for r in receipts if r.is_valid()}

        missing = []
        for authority_id, config in self.authorities.items():
            if config.required_for_compliance:
                if authority_id not in receipt_authorities:
                    missing.append(authority_id)

        return len(missing) == 0, missing

    def get_receipt_summary(self, batch_id: str) -> Dict[str, Any]:
        """Get summary of receipts for a batch."""
        receipts = self.get_receipts(batch_id)

        return {
            "batch_id": batch_id,
            "total_receipts": len(receipts),
            "confirmed": sum(1 for r in receipts if r.status == ReceiptStatus.CONFIRMED),
            "pending": sum(1 for r in receipts if r.status == ReceiptStatus.PENDING),
            "rejected": sum(1 for r in receipts if r.status == ReceiptStatus.REJECTED),
            "authorities": [r.authority_id for r in receipts],
            "verified": sum(1 for r in receipts if r.verified),
        }


# Convenience functions

def create_mock_authority(
    authority_id: str = "mock-authority",
    name: str = "Mock Authority",
    authority_type: AuthorityType = AuthorityType.NOTARY,
) -> AuthorityConfig:
    """Create a mock authority for testing."""
    return AuthorityConfig(
        authority_id=authority_id,
        name=name,
        authority_type=authority_type,
        url="",  # Empty URL triggers mock mode
    )


def create_regulator_authority(
    authority_id: str,
    name: str,
    url: str,
    api_key: Optional[str] = None,
) -> AuthorityConfig:
    """Create a regulatory authority configuration."""
    return AuthorityConfig(
        authority_id=authority_id,
        name=name,
        authority_type=AuthorityType.REGULATOR,
        url=url,
        api_key=api_key,
        required_for_compliance=True,
    )


def create_notary_authority(
    authority_id: str,
    name: str,
    url: str = "",
) -> AuthorityConfig:
    """Create a notary authority configuration."""
    return AuthorityConfig(
        authority_id=authority_id,
        name=name,
        authority_type=AuthorityType.NOTARY,
        url=url,
    )
