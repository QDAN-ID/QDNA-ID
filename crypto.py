# crypto.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah


Cryptographic helpers:
- HMAC-SHA256 (symmetric)
- RSA sign/verify with SHA256 (asymmetric)
- Public key fingerprint (SHA-256 over DER SubjectPublicKeyInfo)
- Auto key provisioning: load-or-create RSA keypair on disk (no PowerShell required)
"""

from __future__ import annotations

import os
import hashlib
from typing import Tuple, Optional, Dict

from cryptography.hazmat.primitives import hashes, serialization, hmac
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend


# =========================
# HMAC (symmetric)
# =========================

def hmac_sha256(key: bytes, message: bytes) -> bytes:
    h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
    h.update(message)
    return h.finalize()


# =========================
# RSA (asymmetric)
# =========================

def generate_rsa_keypair(key_size: int = 2048) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """Generate an RSA keypair in-memory."""
    priv = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=default_backend())
    return priv, priv.public_key()


def rsa_sign(private_key: rsa.RSAPrivateKey, message: bytes) -> bytes:
    return private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )


def rsa_verify(public_key: rsa.RSAPublicKey, message: bytes, signature: bytes) -> bool:
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# =========================
# Serialization helpers
# =========================

def serialize_private_key_to_pem(private_key: rsa.RSAPrivateKey, password: bytes | None = None) -> bytes:
    enc = serialization.BestAvailableEncryption(password) if password else serialization.NoEncryption()
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc,
    )


def serialize_public_key_to_pem(public_key: rsa.RSAPublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_private_key(pem_bytes: bytes, password: bytes | None = None) -> rsa.RSAPrivateKey:
    return serialization.load_pem_private_key(pem_bytes, password=password, backend=default_backend())


def load_public_key(pem_bytes: bytes) -> rsa.RSAPublicKey:
    return serialization.load_pem_public_key(pem_bytes, backend=default_backend())


def public_key_fingerprint_sha256(public_key: rsa.RSAPublicKey) -> str:
    """SHA-256 over DER-encoded SubjectPublicKeyInfo."""
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


# =========================
# Auto-provisioning (load-or-create on disk)
# =========================

def _safe_write_file(path: str, data: bytes) -> None:
    """Write bytes to a file; ensure directory exists. On POSIX, try chmod 600."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    # Best-effort: on POSIX tighten permissions; on Windows this is a no-op and safe to ignore.
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def ensure_rsa_keypair(
    priv_path: str,
    pub_path: str,
    *,
    password: bytes | None = None,
    key_size: int = 2048,
    overwrite: bool = False,
) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey, bool]:
    """
    Ensure an RSA keypair exists at the given paths.
    - If both files exist: load and return (created=False).
    - If one or both are missing (or overwrite=True): generate, write, and return (created=True).
    - No external tools required (works on Windows/macOS/Linux).

    Returns: (private_key, public_key, created_flag)
    """
    priv_exists = os.path.exists(priv_path)
    pub_exists = os.path.exists(pub_path)

    if priv_exists and pub_exists and not overwrite:
        with open(priv_path, "rb") as f:
            priv_pem = f.read()
        with open(pub_path, "rb") as f:
            pub_pem = f.read()
        priv = load_private_key(priv_pem, password=password)
        pub = load_public_key(pub_pem)
        return priv, pub, False

    # (Re)create both files
    priv, pub = generate_rsa_keypair(key_size=key_size)
    _safe_write_file(priv_path, serialize_private_key_to_pem(priv, password=password))
    _safe_write_file(pub_path, serialize_public_key_to_pem(pub))
    return priv, pub, True


def load_or_create_rsa_from_env() -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey, Dict[str, str]]:
    """
    Load RSA keys from env paths if provided, else create them in a default folder.

    Environment variables (optional):
      - QDNA_RSA_PRIV_PEM : path to private key (PEM)
      - QDNA_RSA_PUB_PEM  : path to public  key (PEM)

    Defaults (if not provided):
      - ./qdna_keys/qdna_rsa_priv.pem
      - ./qdna_keys/qdna_rsa_pub.pem

    Returns: (private_key, public_key, meta)
      meta = {
        "priv_path": "...",
        "pub_path":  "...",
        "created": "true|false",
        "pubkey_fingerprint_sha256": "..."
      }
    """
    default_dir = os.path.join(".", "qdna_keys")
    priv_path = os.environ.get("QDNA_RSA_PRIV_PEM") or os.path.join(default_dir, "qdna_rsa_priv.pem")
    pub_path  = os.environ.get("QDNA_RSA_PUB_PEM")  or os.path.join(default_dir, "qdna_rsa_pub.pem")

    priv, pub, created = ensure_rsa_keypair(priv_path, pub_path)
    meta = {
        "priv_path": priv_path,
        "pub_path": pub_path,
        "created": "true" if created else "false",
        "pubkey_fingerprint_sha256": public_key_fingerprint_sha256(pub),
    }
    return priv, pub, meta

