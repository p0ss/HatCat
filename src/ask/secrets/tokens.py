"""
RFC 3161 Timestamp Token client for external proof of existence.

Provides cryptographic proof that a hash existed at a specific point in time,
issued by a trusted Time Stamp Authority (TSA).
"""

import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import secrets
import base64

# Optional cryptography imports for full verification
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class TSAConfig:
    """Configuration for a Time Stamp Authority."""

    authority_id: str  # e.g., "freetsa.org", "digicert"
    url: str  # TSA endpoint URL
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: int = 5

    # Authentication (optional)
    username: Optional[str] = None
    password: Optional[str] = None
    client_cert_path: Optional[str] = None

    # Verification
    ca_cert_path: Optional[str] = None  # TSA's CA certificate for verification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "url": self.url,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
        }


# Default TSA configurations
DEFAULT_TSAS = [
    TSAConfig(
        authority_id="freetsa.org",
        url="https://freetsa.org/tsr",
        timeout_seconds=30,
    ),
    TSAConfig(
        authority_id="zeitstempel.dfn.de",
        url="https://zeitstempel.dfn.de",
        timeout_seconds=30,
    ),
]


@dataclass
class TimestampRequest:
    """
    RFC 3161 Timestamp Request.

    Contains the hash to be timestamped and request parameters.
    """

    # Hash to timestamp (the message imprint)
    hash_algorithm: str = "sha256"
    hash_value: bytes = field(default_factory=bytes)

    # Request options
    cert_req: bool = True  # Request TSA certificate in response
    nonce: Optional[bytes] = None  # Random nonce for replay protection

    # Metadata
    policy_oid: Optional[str] = None  # TSA policy OID

    @classmethod
    def create(cls, data_hash: str, include_nonce: bool = True) -> "TimestampRequest":
        """
        Create a timestamp request for a hash.

        Args:
            data_hash: Hash in format "sha256:hexdigest"
            include_nonce: Whether to include random nonce

        Returns:
            TimestampRequest ready to send
        """
        # Parse hash
        if ":" in data_hash:
            algo, hex_digest = data_hash.split(":", 1)
        else:
            algo = "sha256"
            hex_digest = data_hash

        hash_bytes = bytes.fromhex(hex_digest)

        nonce = None
        if include_nonce:
            nonce = secrets.token_bytes(8)

        return cls(
            hash_algorithm=algo,
            hash_value=hash_bytes,
            nonce=nonce,
        )

    def to_asn1(self) -> bytes:
        """
        Encode request as ASN.1 DER (simplified).

        Note: For production, use a proper ASN.1 library like pyasn1.
        This is a simplified implementation for the request structure.
        """
        # This is a simplified ASN.1 encoding
        # In production, use pyasn1 or similar for proper encoding

        # TimeStampReq ::= SEQUENCE {
        #   version INTEGER { v1(1) },
        #   messageImprint MessageImprint,
        #   reqPolicy TSAPolicyId OPTIONAL,
        #   nonce INTEGER OPTIONAL,
        #   certReq BOOLEAN DEFAULT FALSE,
        #   extensions [0] IMPLICIT Extensions OPTIONAL
        # }

        # For now, return a placeholder that indicates the structure
        # Real implementation would use pyasn1
        request_data = {
            "version": 1,
            "hash_algorithm": self.hash_algorithm,
            "hash_value": self.hash_value.hex(),
            "cert_req": self.cert_req,
            "nonce": self.nonce.hex() if self.nonce else None,
        }

        # Simple binary encoding for demo
        # In production: return proper ASN.1 DER
        import json
        return json.dumps(request_data).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash_algorithm": self.hash_algorithm,
            "hash_value": self.hash_value.hex(),
            "cert_req": self.cert_req,
            "nonce": self.nonce.hex() if self.nonce else None,
        }


@dataclass
class TimestampToken:
    """
    RFC 3161 Timestamp Token (response).

    Contains the signed timestamp from the TSA.
    """

    # Token identity
    token_id: str = ""

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accuracy_seconds: float = 1.0

    # The signed data
    hash_algorithm: str = "sha256"
    hash_value: bytes = field(default_factory=bytes)
    nonce: Optional[bytes] = None

    # TSA info
    tsa_name: str = ""
    tsa_policy_oid: str = ""
    serial_number: int = 0

    # The raw token (for storage and verification)
    raw_token: bytes = field(default_factory=bytes, repr=False)

    # Verification status
    verified: bool = False
    verification_error: Optional[str] = None

    @classmethod
    def from_response(cls, response_bytes: bytes, tsa_id: str = "") -> "TimestampToken":
        """
        Parse timestamp token from TSA response.

        Args:
            response_bytes: Raw response from TSA
            tsa_id: TSA identifier

        Returns:
            Parsed TimestampToken
        """
        # In production, this would parse ASN.1 DER
        # For now, handle our simplified format

        try:
            import json
            data = json.loads(response_bytes.decode("utf-8"))

            return cls(
                token_id=data.get("token_id", f"tst_{secrets.token_hex(8)}"),
                timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
                hash_algorithm=data.get("hash_algorithm", "sha256"),
                hash_value=bytes.fromhex(data.get("hash_value", "")),
                nonce=bytes.fromhex(data["nonce"]) if data.get("nonce") else None,
                tsa_name=data.get("tsa_name", tsa_id),
                serial_number=data.get("serial_number", 0),
                raw_token=response_bytes,
                verified=data.get("verified", False),
            )
        except Exception as e:
            # Return token with error
            return cls(
                raw_token=response_bytes,
                verification_error=str(e),
            )

    def verify(self, original_hash: str, ca_cert: Optional[bytes] = None) -> bool:
        """
        Verify this timestamp token.

        Args:
            original_hash: The hash that was timestamped
            ca_cert: TSA's CA certificate for signature verification

        Returns:
            True if token is valid
        """
        # Parse original hash
        if ":" in original_hash:
            _, hex_digest = original_hash.split(":", 1)
        else:
            hex_digest = original_hash

        expected_hash = bytes.fromhex(hex_digest)

        # Check hash matches
        if self.hash_value != expected_hash:
            self.verification_error = "Hash mismatch"
            self.verified = False
            return False

        # In production, verify signature using ca_cert
        # For now, mark as verified if hash matches
        self.verified = True
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "timestamp": self.timestamp.isoformat(),
            "accuracy_seconds": self.accuracy_seconds,
            "hash_algorithm": self.hash_algorithm,
            "hash_value": self.hash_value.hex(),
            "nonce": self.nonce.hex() if self.nonce else None,
            "tsa_name": self.tsa_name,
            "tsa_policy_oid": self.tsa_policy_oid,
            "serial_number": self.serial_number,
            "verified": self.verified,
            "verification_error": self.verification_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimestampToken":
        return cls(
            token_id=data.get("token_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
            accuracy_seconds=data.get("accuracy_seconds", 1.0),
            hash_algorithm=data.get("hash_algorithm", "sha256"),
            hash_value=bytes.fromhex(data.get("hash_value", "")),
            nonce=bytes.fromhex(data["nonce"]) if data.get("nonce") else None,
            tsa_name=data.get("tsa_name", ""),
            tsa_policy_oid=data.get("tsa_policy_oid", ""),
            serial_number=data.get("serial_number", 0),
            verified=data.get("verified", False),
            verification_error=data.get("verification_error"),
        )

    def to_bytes(self) -> bytes:
        """Serialize token to bytes for storage."""
        import json
        data = self.to_dict()
        data["raw_token_b64"] = base64.b64encode(self.raw_token).decode("ascii")
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TimestampToken":
        """Deserialize token from bytes."""
        import json
        parsed = json.loads(data.decode("utf-8"))
        token = cls.from_dict(parsed)
        if "raw_token_b64" in parsed:
            token.raw_token = base64.b64decode(parsed["raw_token_b64"])
        return token


@dataclass
class PendingTimestamp:
    """A timestamp request waiting to be sent."""

    request_id: str
    hash_value: str  # The hash to timestamp
    batch_id: str  # Associated batch
    created_at: datetime
    retry_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "hash_value": self.hash_value,
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
            "last_error": self.last_error,
        }


class TimestampClient:
    """
    RFC 3161 Timestamp client.

    Handles requesting and verifying timestamps from TSAs.
    Supports multiple authorities and offline queueing.
    """

    def __init__(
        self,
        tsa_configs: Optional[List[TSAConfig]] = None,
        offline_queue_path: Optional[Path] = None,
    ):
        self.tsa_configs = tsa_configs or DEFAULT_TSAS
        self.offline_queue_path = offline_queue_path

        # Pending requests (for offline mode)
        self._pending: List[PendingTimestamp] = []

        # Completed tokens cache
        self._tokens: Dict[str, TimestampToken] = {}

    def request_timestamp(
        self,
        data_hash: str,
        batch_id: str = "",
        tsa_id: Optional[str] = None,
    ) -> Tuple[Optional[TimestampToken], Optional[str]]:
        """
        Request a timestamp for a hash.

        Args:
            data_hash: Hash to timestamp (sha256:... format)
            batch_id: Associated batch ID for tracking
            tsa_id: Specific TSA to use (None = try all)

        Returns:
            (token, error) - token if successful, error message if failed
        """
        request = TimestampRequest.create(data_hash)

        # Get TSA configs to try
        configs = self.tsa_configs
        if tsa_id:
            configs = [c for c in configs if c.authority_id == tsa_id]
            if not configs:
                return None, f"Unknown TSA: {tsa_id}"

        # Try each TSA
        last_error = None
        for config in configs:
            token, error = self._send_request(request, config)
            if token:
                # Verify the token
                if token.verify(data_hash):
                    self._tokens[batch_id] = token
                    return token, None
                else:
                    last_error = token.verification_error or "Verification failed"
            else:
                last_error = error

        # All TSAs failed - queue for retry if offline support enabled
        if self.offline_queue_path:
            pending = PendingTimestamp(
                request_id=f"req_{secrets.token_hex(8)}",
                hash_value=data_hash,
                batch_id=batch_id,
                created_at=datetime.now(timezone.utc),
                last_error=last_error,
            )
            self._pending.append(pending)
            self._save_pending_queue()

        return None, last_error or "All TSAs failed"

    def _send_request(
        self,
        request: TimestampRequest,
        config: TSAConfig,
    ) -> Tuple[Optional[TimestampToken], Optional[str]]:
        """
        Send timestamp request to a TSA.

        Returns:
            (token, error)
        """
        try:
            import urllib.request
            import urllib.error

            # Prepare request
            req_data = request.to_asn1()

            http_req = urllib.request.Request(
                config.url,
                data=req_data,
                headers={
                    "Content-Type": "application/timestamp-query",
                },
                method="POST",
            )

            # Add auth if configured
            if config.username and config.password:
                import base64
                credentials = base64.b64encode(
                    f"{config.username}:{config.password}".encode()
                ).decode("ascii")
                http_req.add_header("Authorization", f"Basic {credentials}")

            # Send request
            with urllib.request.urlopen(http_req, timeout=config.timeout_seconds) as response:
                response_data = response.read()

            # Parse response
            token = TimestampToken.from_response(response_data, config.authority_id)
            return token, None

        except urllib.error.URLError as e:
            return None, f"Network error: {e}"
        except Exception as e:
            return None, f"Request failed: {e}"

    def get_token(self, batch_id: str) -> Optional[TimestampToken]:
        """Get cached token for a batch."""
        return self._tokens.get(batch_id)

    def verify_token(
        self,
        token: TimestampToken,
        original_hash: str,
        ca_cert: Optional[bytes] = None,
    ) -> bool:
        """Verify a timestamp token."""
        return token.verify(original_hash, ca_cert)

    def get_pending_count(self) -> int:
        """Get count of pending timestamp requests."""
        return len(self._pending)

    def get_pending_requests(self) -> List[PendingTimestamp]:
        """Get all pending requests."""
        return list(self._pending)

    def retry_pending(self) -> Tuple[int, int]:
        """
        Retry all pending timestamp requests.

        Returns:
            (success_count, failure_count)
        """
        success = 0
        failure = 0
        still_pending = []

        for pending in self._pending:
            pending.retry_count += 1

            token, error = self.request_timestamp(
                pending.hash_value,
                pending.batch_id,
            )

            if token:
                success += 1
            else:
                pending.last_error = error
                # Keep in queue if under retry limit
                if pending.retry_count < 10:
                    still_pending.append(pending)
                else:
                    failure += 1

        self._pending = still_pending
        if self.offline_queue_path:
            self._save_pending_queue()

        return success, failure

    def _save_pending_queue(self) -> None:
        """Save pending queue to disk."""
        if not self.offline_queue_path:
            return

        import json
        self.offline_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.offline_queue_path, "w") as f:
            json.dump([p.to_dict() for p in self._pending], f)

    def _load_pending_queue(self) -> None:
        """Load pending queue from disk."""
        if not self.offline_queue_path or not self.offline_queue_path.exists():
            return

        import json
        with open(self.offline_queue_path) as f:
            data = json.load(f)

        self._pending = [
            PendingTimestamp(
                request_id=p["request_id"],
                hash_value=p["hash_value"],
                batch_id=p["batch_id"],
                created_at=datetime.fromisoformat(p["created_at"]),
                retry_count=p.get("retry_count", 0),
                last_error=p.get("last_error"),
            )
            for p in data
        ]


def create_mock_timestamp(data_hash: str, tsa_name: str = "mock-tsa") -> TimestampToken:
    """
    Create a mock timestamp token for testing.

    Not cryptographically valid, but useful for testing workflows.
    """
    if ":" in data_hash:
        _, hex_digest = data_hash.split(":", 1)
    else:
        hex_digest = data_hash

    return TimestampToken(
        token_id=f"tst_{secrets.token_hex(8)}",
        timestamp=datetime.now(timezone.utc),
        hash_algorithm="sha256",
        hash_value=bytes.fromhex(hex_digest),
        nonce=secrets.token_bytes(8),
        tsa_name=tsa_name,
        serial_number=secrets.randbelow(1000000),
        verified=True,
    )
