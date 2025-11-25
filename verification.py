# verification.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey

Verification:
- Load trio files for a given QDNAID and backend
- Check RSA/HMAC signatures against features
- Report provenance presence
"""

from __future__ import annotations

import os
import json
from typing import Dict, Optional

from crypto import load_public_key, rsa_verify, hmac_sha256


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_signatures(sign_obj: Dict) -> Dict:
    """
    Normalize signatures block.
    We store {"qdna_id":..., "backend":..., "signatures": {...}}.
    But if a legacy file stored fields at top-level, accept that too.
    """
    sig = sign_obj.get("signatures")
    if isinstance(sig, dict):
        return sig
    # legacy / permissive fallback
    return {k: v for k, v in sign_obj.items() if k in {
        "hmac_sha256",
        "rsa_sha256_hex",
        "rsa_public_key_pem",
        "algorithms",
        "key_ids",
        "key_paths",
        "pubkey_fingerprint_sha256",
        "created_at_utc",
    }}


def _compose_message(qdna_id: str, features_map: Dict) -> bytes:
    """The canonical message that was signed."""
    return json.dumps({"qdna_id": qdna_id, "features": features_map}, sort_keys=True).encode()


def _verify_hmac(signatures: Dict, msg: bytes, hmac_env: str) -> str | bool:
    """
    Returns:
      - True  -> HMAC matches
      - False -> HMAC present but does NOT match
      - "no_hmac_or_signature" -> missing env key or signature hex
    """
    hkey_env = os.environ.get(hmac_env)
    sig_hex = signatures.get("hmac_sha256")
    if not hkey_env or not sig_hex:
        return "no_hmac_or_signature"
    try:
        calc = hmac_sha256(hkey_env.encode(), msg).hex()
        return bool(calc == sig_hex)
    except Exception:
        return False


def _load_public_key_for_verification(signatures: Dict, rsa_pub_env: str):
    """
    Try, in order:
      1) Embedded PEM: signatures["rsa_public_key_pem"]
      2) Env path: os.environ[rsa_pub_env]
      3) signatures["key_paths"]["public"] if exists
    Returns (public_key_obj | None, source_str)
    """
    # 1) Embedded PEM
    pem = signatures.get("rsa_public_key_pem")
    if pem:
        try:
            return load_public_key(pem.encode("ascii")), "embedded_pem"
        except Exception:
            pass

    # 2) Env path
    env_path = os.environ.get(rsa_pub_env)
    if env_path and os.path.exists(env_path):
        try:
            with open(env_path, "rb") as f:
                return load_public_key(f.read()), "env_path"
        except Exception:
            pass

    # 3) key_paths.public
    key_paths = signatures.get("key_paths") or {}
    pub_path = key_paths.get("public")
    if pub_path and os.path.exists(pub_path):
        try:
            with open(pub_path, "rb") as f:
                return load_public_key(f.read()), "sign_json_path"
        except Exception:
            pass

    return None, "no_public_key"


def _verify_rsa(signatures: Dict, msg: bytes, rsa_pub_env: str) -> str | bool:
    """
    Returns:
      - True  -> signature valid
      - False -> signature present but invalid under provided key
      - "no_rsa_or_signature" -> signatures missing or no rsa_sha256_hex
      - "no_public_key" -> no public key available by any method
    """
    sig_hex = signatures.get("rsa_sha256_hex")
    if not sig_hex:
        return "no_rsa_or_signature"

    pub, source = _load_public_key_for_verification(signatures, rsa_pub_env)
    if pub is None:
        return "no_public_key"

    try:
        ok = rsa_verify(pub, msg, bytes.fromhex(sig_hex))
        return bool(ok)
    except Exception:
        return False


def verify(
    root: str,
    backend: str,
    qdna_id: str,
    hmac_env: str = "QDNA_HMAC_KEY",
    rsa_pub_env: str = "QDNA_RSA_PUB_PEM",
) -> Dict:
    """
    Verify stored trio for a session. Returns a structured report:
      {
        "ok": bool,
        "backend": ..., "qdna_id": ...,
        "checks": {
          "hmac_match": True|False|"no_hmac_or_signature",
          "rsa_valid":  True|False|"no_rsa_or_signature"|"no_public_key",
          "provenance_present": True|False
        },
        "provenance": {...} | None,
        "details": {
          "used_public_key_source": "embedded_pem|env_path|sign_json_path|no_public_key"
        }
      }
    """
    raw_p  = os.path.join(root, backend, f"{qdna_id}_raw.json")
    feat_p = os.path.join(root, backend, f"{qdna_id}_features.json")
    sign_p = os.path.join(root, backend, f"{qdna_id}_sign.json")

    if not (os.path.exists(raw_p) and os.path.exists(feat_p) and os.path.exists(sign_p)):
        return {
            "ok": False,
            "reason": "files_missing",
            "paths": [raw_p, feat_p, sign_p],
        }

    raw = _read_json(raw_p)
    feats = _read_json(feat_p)
    sign = _read_json(sign_p)

    features = feats.get("features", {})
    signatures = _extract_signatures(sign)
    msg = _compose_message(qdna_id, features)

    # HMAC
    hmac_status = _verify_hmac(signatures, msg, hmac_env)

    # RSA (+ track source for debugging)
    pub, source = _load_public_key_for_verification(signatures, rsa_pub_env)
    if signatures.get("rsa_sha256_hex"):
        rsa_status = _verify_rsa(signatures, msg, rsa_pub_env) if pub else "no_public_key"
    else:
        rsa_status = "no_rsa_or_signature"

    provenance_present = ("provenance" in raw)

    checks = {
        "hmac_match": hmac_status,
        "rsa_valid": rsa_status,
        "provenance_present": provenance_present,
    }

    # ok flag: only fail on explicit False. Informational strings don't flip ok.
    ok = True
    for v in checks.values():
        if v is False:
            ok = False

    return {
        "ok": ok,
        "backend": backend,
        "qdna_id": qdna_id,
        "checks": checks,
        "provenance": raw.get("provenance"),
        "details": {
            "used_public_key_source": source,
            "sign_file": sign_p,
        },
    }
