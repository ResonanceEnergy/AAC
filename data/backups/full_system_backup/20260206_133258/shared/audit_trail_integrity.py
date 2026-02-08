#!/usr/bin/env python3
"""
Audit Trail Integrity System
============================
Immutable audit trails with cryptographic integrity, chain of custody verification, and compliance reporting.
"""

import asyncio
import logging
import json
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


@dataclass
class AuditBlock:
    """Immutable audit block with cryptographic integrity"""
    block_id: str
    timestamp: datetime
    previous_hash: str
    events: List[Dict[str, Any]]
    block_hash: str
    signature: str
    custodian: str
    sequence_number: int


@dataclass
class ChainOfCustody:
    """Chain of custody record for audit evidence"""
    evidence_id: str
    event_type: str
    collected_at: datetime
    collected_by: str
    evidence_data: Dict[str, Any]
    integrity_hash: str
    custody_chain: List[Dict[str, Any]] = field(default_factory=list)


class AuditTrailIntegritySystem:
    """Cryptographically secure audit trail with integrity verification"""

    def __init__(self):
        self.logger = logging.getLogger("AuditIntegrity")
        self.audit_logger = get_audit_logger()

        # Cryptographic keys for signing
        self.private_key = None
        self.public_key = None

        # Blockchain-style audit blocks
        self.audit_blocks: List[AuditBlock] = []
        self.current_block_events: List[Dict[str, Any]] = []

        # Chain of custody records
        self.custody_records: List[ChainOfCustody] = []

        # Integrity verification
        self.integrity_checks: List[Dict[str, Any]] = []

        # Storage paths
        self.blocks_dir = PROJECT_ROOT / "data" / "audit_blocks"
        self.keys_dir = PROJECT_ROOT / "config" / "crypto"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        # Load cryptographic keys
        self._load_crypto_keys()

        # Load existing audit blocks
        self._load_audit_blocks()

    def _load_crypto_keys(self):
        """Load or generate cryptographic keys for audit signing"""
        private_key_file = self.keys_dir / "audit_private_key.pem"
        public_key_file = self.keys_dir / "audit_public_key.pem"

        try:
            if private_key_file.exists():
                # Load existing keys
                with open(private_key_file, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                with open(public_key_file, 'rb') as f:
                    self.public_key = serialization.load_pem_public_key(f.read())
            else:
                # Generate new keys
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                self.public_key = self.private_key.public_key()

                # Save keys
                private_pem = self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                public_pem = self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )

                with open(private_key_file, 'wb') as f:
                    f.write(private_pem)
                with open(public_key_file, 'wb') as f:
                    f.write(public_pem)

                self.logger.info("Generated new cryptographic keys for audit integrity")

        except Exception as e:
            self.logger.error(f"Error loading cryptographic keys: {e}")
            # Fallback to hash-based integrity without crypto signing
            self.private_key = None
            self.public_key = None

    def _load_audit_blocks(self):
        """Load existing audit blocks from storage"""
        blocks_file = self.blocks_dir / "audit_blocks.json"

        if blocks_file.exists():
            try:
                with open(blocks_file, 'r') as f:
                    blocks_data = json.load(f)

                for block_data in blocks_data:
                    block = AuditBlock(
                        block_id=block_data["block_id"],
                        timestamp=datetime.fromisoformat(block_data["timestamp"]),
                        previous_hash=block_data["previous_hash"],
                        events=block_data["events"],
                        block_hash=block_data["block_hash"],
                        signature=block_data["signature"],
                        custodian=block_data["custodian"],
                        sequence_number=block_data["sequence_number"]
                    )
                    self.audit_blocks.append(block)

                self.logger.info(f"Loaded {len(self.audit_blocks)} audit blocks")

            except Exception as e:
                self.logger.error(f"Error loading audit blocks: {e}")

    async def add_audit_event(self,
                             category: str,
                             action: str,
                             user: str,
                             resource: str,
                             status: str,
                             details: Optional[Dict[str, Any]] = None,
                             custodian: str = "system") -> str:
        """Add an event to the current audit block"""

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "action": action,
            "user": user,
            "resource": resource,
            "status": status,
            "details": details or {},
            "custodian": custodian
        }

        self.current_block_events.append(event)

        # Check if we should create a new block (every 100 events or hourly)
        if len(self.current_block_events) >= 100 or self._should_create_block():
            await self._create_audit_block(custodian)

        return event["event_id"]

    def _should_create_block(self) -> bool:
        """Determine if a new audit block should be created"""
        if not self.audit_blocks:
            return len(self.current_block_events) > 0

        last_block = self.audit_blocks[-1]
        time_since_last = datetime.now() - last_block.timestamp

        # Create block if it's been more than an hour
        return time_since_last.total_seconds() > 3600

    async def _create_audit_block(self, custodian: str):
        """Create a new immutable audit block"""
        if not self.current_block_events:
            return

        # Get previous block hash
        previous_hash = "GENESIS" if not self.audit_blocks else self.audit_blocks[-1].block_hash

        # Create block data
        block_data = {
            "block_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "previous_hash": previous_hash,
            "events": self.current_block_events.copy(),
            "custodian": custodian,
            "sequence_number": len(self.audit_blocks) + 1
        }

        # Calculate block hash
        block_content = json.dumps({
            "previous_hash": block_data["previous_hash"],
            "events": block_data["events"],
            "timestamp": block_data["timestamp"],
            "custodian": block_data["custodian"],
            "sequence_number": block_data["sequence_number"]
        }, sort_keys=True)

        block_hash = hashlib.sha256(block_content.encode()).hexdigest()
        block_data["block_hash"] = block_hash

        # Sign the block
        signature = ""
        if self.private_key:
            try:
                signature_bytes = self.private_key.sign(
                    block_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                signature = signature_bytes.hex()
            except Exception as e:
                self.logger.error(f"Error signing audit block: {e}")

        block_data["signature"] = signature

        # Create audit block
        audit_block = AuditBlock(
            block_id=block_data["block_id"],
            timestamp=datetime.fromisoformat(block_data["timestamp"]),
            previous_hash=block_data["previous_hash"],
            events=block_data["events"],
            block_hash=block_hash,
            signature=signature,
            custodian=custodian,
            sequence_number=block_data["sequence_number"]
        )

        self.audit_blocks.append(audit_block)

        # Clear current events
        self.current_block_events.clear()

        # Save blocks
        await self._save_audit_blocks()

        # Audit the block creation
        await self.audit_logger.log_event(
            category="audit",
            action="block_created",
            details={
                "block_id": audit_block.block_id,
                "events_count": len(audit_block.events),
                "block_hash": audit_block.block_hash,
                "custodian": custodian
            }
        )

        self.logger.info(f"Created audit block: {audit_block.block_id} with {len(audit_block.events)} events")

    async def _save_audit_blocks(self):
        """Save audit blocks to persistent storage"""
        blocks_file = self.blocks_dir / "audit_blocks.json"

        blocks_data = []
        for block in self.audit_blocks[-100:]:  # Keep last 100 blocks
            block_data = {
                "block_id": block.block_id,
                "timestamp": block.timestamp.isoformat(),
                "previous_hash": block.previous_hash,
                "events": block.events,
                "block_hash": block.block_hash,
                "signature": block.signature,
                "custodian": block.custodian,
                "sequence_number": block.sequence_number
            }
            blocks_data.append(block_data)

        try:
            with open(blocks_file, 'w') as f:
                json.dump(blocks_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving audit blocks: {e}")

    async def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit trail"""
        integrity_results = {
            "total_blocks": len(self.audit_blocks),
            "verified_blocks": 0,
            "corrupted_blocks": 0,
            "missing_signatures": 0,
            "chain_breaks": 0,
            "overall_integrity": True,
            "details": []
        }

        previous_hash = "GENESIS"

        for i, block in enumerate(self.audit_blocks):
            block_integrity = await self._verify_block_integrity(block, previous_hash)

            if block_integrity["valid"]:
                integrity_results["verified_blocks"] += 1
            else:
                integrity_results["corrupted_blocks"] += 1
                integrity_results["overall_integrity"] = False

            if not block_integrity["signature_valid"]:
                integrity_results["missing_signatures"] += 1

            if not block_integrity["chain_valid"]:
                integrity_results["chain_breaks"] += 1

            integrity_results["details"].append({
                "block_id": block.block_id,
                "sequence": block.sequence_number,
                "integrity": block_integrity
            })

            previous_hash = block.block_hash

        # Record integrity check
        self.integrity_checks.append({
            "check_time": datetime.now().isoformat(),
            "results": integrity_results
        })

        # Audit the integrity check
        await self.audit_logger.log_event(
            category="audit",
            action="integrity_check",
            details={
                "overall_integrity": integrity_results["overall_integrity"],
                "verified_blocks": integrity_results["verified_blocks"],
                "corrupted_blocks": integrity_results["corrupted_blocks"]
            }
        )

        return integrity_results

    async def _verify_block_integrity(self, block: AuditBlock, expected_previous_hash: str) -> Dict[str, Any]:
        """Verify integrity of a single audit block"""
        results = {
            "valid": True,
            "hash_valid": True,
            "signature_valid": True,
            "chain_valid": True,
            "content_valid": True
        }

        # Verify chain continuity
        if block.previous_hash != expected_previous_hash:
            results["chain_valid"] = False
            results["valid"] = False

        # Verify block hash
        block_content = json.dumps({
            "previous_hash": block.previous_hash,
            "events": block.events,
            "timestamp": block.timestamp.isoformat(),
            "custodian": block.custodian,
            "sequence_number": block.sequence_number
        }, sort_keys=True)

        calculated_hash = hashlib.sha256(block_content.encode()).hexdigest()
        if calculated_hash != block.block_hash:
            results["hash_valid"] = False
            results["valid"] = False

        # Verify signature
        if self.public_key and block.signature:
            try:
                self.public_key.verify(
                    bytes.fromhex(block.signature),
                    block.block_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            except InvalidSignature:
                results["signature_valid"] = False
                results["valid"] = False
        elif not block.signature:
            results["signature_valid"] = False
            # Don't mark as invalid if no crypto keys available

        return results

    async def create_chain_of_custody(self,
                                     evidence_id: str,
                                     event_type: str,
                                     evidence_data: Dict[str, Any],
                                     custodian: str) -> str:
        """Create a chain of custody record for audit evidence"""

        # Calculate integrity hash
        evidence_content = json.dumps(evidence_data, sort_keys=True)
        integrity_hash = hashlib.sha256(evidence_content.encode()).hexdigest()

        custody_record = ChainOfCustody(
            evidence_id=evidence_id,
            event_type=event_type,
            collected_at=datetime.now(),
            collected_by=custodian,
            evidence_data=evidence_data,
            integrity_hash=integrity_hash,
            custody_chain=[{
                "timestamp": datetime.now().isoformat(),
                "custodian": custodian,
                "action": "collected",
                "integrity_hash": integrity_hash
            }]
        )

        self.custody_records.append(custody_record)

        # Audit the custody record creation
        await self.audit_logger.log_event(
            category="audit",
            action="custody_record_created",
            details={
                "evidence_id": evidence_id,
                "event_type": event_type,
                "custodian": custodian,
                "integrity_hash": integrity_hash
            }
        )

        return evidence_id

    async def transfer_custody(self, evidence_id: str, new_custodian: str, reason: str):
        """Transfer custody of evidence to another party"""
        custody_record = next((r for r in self.custody_records if r.evidence_id == evidence_id), None)

        if not custody_record:
            raise ValueError(f"Custody record not found: {evidence_id}")

        # Verify integrity before transfer
        current_hash = hashlib.sha256(json.dumps(custody_record.evidence_data, sort_keys=True).encode()).hexdigest()
        if current_hash != custody_record.integrity_hash:
            raise ValueError(f"Evidence integrity compromised for {evidence_id}")

        # Add to custody chain
        custody_record.custody_chain.append({
            "timestamp": datetime.now().isoformat(),
            "from_custodian": custody_record.custody_chain[-1]["custodian"],
            "to_custodian": new_custodian,
            "action": "transferred",
            "reason": reason,
            "integrity_hash": current_hash
        })

        # Audit the transfer
        await self.audit_logger.log_event(
            category="audit",
            action="custody_transferred",
            details={
                "evidence_id": evidence_id,
                "from_custodian": custody_record.custody_chain[-2]["custodian"],
                "to_custodian": new_custodian,
                "reason": reason
            }
        )

    def get_audit_trail_summary(self) -> Dict[str, Any]:
        """Get summary of audit trail integrity"""
        total_events = sum(len(block.events) for block in self.audit_blocks) + len(self.current_block_events)

        return {
            "total_blocks": len(self.audit_blocks),
            "total_events": total_events,
            "current_pending_events": len(self.current_block_events),
            "custody_records": len(self.custody_records),
            "integrity_checks": len(self.integrity_checks),
            "last_block_timestamp": self.audit_blocks[-1].timestamp.isoformat() if self.audit_blocks else None,
            "cryptographic_signing": self.private_key is not None
        }

    async def export_audit_trail(self, output_path: Path, include_custody: bool = True) -> bool:
        """Export complete audit trail for regulatory review"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "audit_blocks": [
                {
                    "block_id": block.block_id,
                    "timestamp": block.timestamp.isoformat(),
                    "previous_hash": block.previous_hash,
                    "events_count": len(block.events),
                    "block_hash": block.block_hash,
                    "signature": block.signature,
                    "custodian": block.custodian,
                    "sequence_number": block.sequence_number
                }
                for block in self.audit_blocks
            ],
            "integrity_checks": self.integrity_checks[-10:],  # Last 10 checks
            "summary": self.get_audit_trail_summary()
        }

        if include_custody:
            export_data["custody_records"] = [
                {
                    "evidence_id": record.evidence_id,
                    "event_type": record.event_type,
                    "collected_at": record.collected_at.isoformat(),
                    "collected_by": record.collected_by,
                    "integrity_hash": record.integrity_hash,
                    "custody_chain": record.custody_chain
                }
                for record in self.custody_records
            ]

        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            # Audit the export
            await self.audit_logger.log_event(
                category="audit",
                action="trail_exported",
                details={
                    "output_path": str(output_path),
                    "blocks_exported": len(export_data["audit_blocks"]),
                    "custody_records_exported": len(export_data.get("custody_records", []))
                }
            )

            return True

        except Exception as e:
            self.logger.error(f"Error exporting audit trail: {e}")
            return False


# Global audit trail integrity system instance
audit_trail_integrity_system = AuditTrailIntegritySystem()


async def initialize_audit_trail_integrity():
    """Initialize the audit trail integrity system"""
    print("[AUDIT] Initializing Audit Trail Integrity System...")

    # Create initial audit block if needed
    if not audit_trail_integrity_system.audit_blocks:
        await audit_trail_integrity_system.add_audit_event(
            category="system",
            action="integrity_system_initialized",
            user="system",
            resource="audit_trail",
            status="success",
            custodian="system"
        )

    # Run integrity verification
    integrity_results = await audit_trail_integrity_system.verify_audit_integrity()

    summary = audit_trail_integrity_system.get_audit_trail_summary()

    print("[OK] Audit trail integrity system initialized")
    print(f"  Total Blocks: {summary['total_blocks']}")
    print(f"  Total Events: {summary['total_events']}")
    print(f"  Cryptographic Signing: {summary['cryptographic_signing']}")
    print(f"  Integrity Verified: {integrity_results['overall_integrity']}")

    return True


if __name__ == "__main__":
    asyncio.run(initialize_audit_trail_integrity())