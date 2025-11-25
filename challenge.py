# challenge.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
"""

from typing import Optional, List, Dict, Any, Tuple
import os, json, logging, platform, hashlib, importlib
from datetime import datetime
from qiskit import transpile

# Local modules
import devices
from circuits import generate_session_circuits
from seed import new_session_seed
from storage import store_raw, store_features, store_sign
from features import feature_vector_from_session
from crypto import (
    hmac_sha256,
    rsa_sign,
    public_key_fingerprint_sha256,
    load_or_create_rsa_from_env,  # ← auto load-or-create RSA keys
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("challenge")

# -------------------- Config & provenance --------------------

MIN_SHOTS, MAX_SHOTS = 256, 1024

PROJECT_TITLE = "QDNA-ID: Quantum Device Native Authentication"
DEVELOPER     = "Osamah N. Neamah"
INSTITUTION   = "Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey"
PROVENANCE_SPEC_VERSION = "2.8.0"
PROJECT_NAMESPACE = getattr(devices, "PROJECT_NAME", "Your-Instance-Name")
INSTANCE_LABEL    = getattr(devices, "INSTANCE_NAME", "token_only")  

def _clamp_shots(s: int) -> int:
    return MIN_SHOTS if s < MIN_SHOTS else MAX_SHOTS if s > MAX_SHOTS else s

def _runtime_versions() -> Dict[str, str]:
    def ver(pkg):
        try:
            m = importlib.import_module(pkg)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return "unavailable"
    return {
        "python": platform.python_version(),
        "qiskit": ver("qiskit"),
        "qiskit_ibm_runtime": ver("qiskit_ibm_runtime"),
        "numpy": ver("numpy"),
    }

def _build_provenance(qdna_id: str, backend_name: str, shots: int, seed: str) -> Dict[str, Any]:
    channel_used = None
    try:
        channel_used = devices.active_channel()
    except Exception:
        channel_used = None

    basis = json.dumps({
        "qdna_id": qdna_id,
        "seed": seed,
        "backend": backend_name,
        "shots": shots,
        "spec": PROVENANCE_SPEC_VERSION,
        "project": PROJECT_NAMESPACE,
        "connection": "token_only",
        "channel": channel_used,
    }, sort_keys=True).encode()

    return {
        "project_title": PROJECT_TITLE,
        "developer": DEVELOPER,
        "institution": INSTITUTION,
        "project_namespace": PROJECT_NAMESPACE,
        # legacy key kept for backward readers; value is a label only:
        "instance_crn": INSTANCE_LABEL,
        "connection": "QISKIT_IBM_RUNTIME_API_TOKEN",
        "channel": channel_used,
        "spec_version": PROVENANCE_SPEC_VERSION,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "provenance_id": hashlib.sha256(basis).hexdigest(),
        "runtime": _runtime_versions(),
    }

# -------------------- Sampler availability (prefer V2) --------------------

HAS_V2 = False
try:
    from qiskit_ibm_runtime import SamplerV2  # type: ignore
    HAS_V2 = True
except Exception:
    from qiskit_ibm_runtime import Sampler as SamplerV1  # type: ignore

# -------------------- Result normalization --------------------

def _extract_counts_v2(primitive_result, shots: int) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    try:
        _ = len(primitive_result)
    except TypeError:
        primitive_result = [primitive_result]
    for pub_res in primitive_result:
        data = getattr(pub_res, "data", None)
        if data is None:
            raise RuntimeError("SamplerV2 result item has no `.data`.")
        if hasattr(data, "join_data"):
            counts = data.join_data().get_counts()
            out.append(dict(counts)); continue
        meas = getattr(data, "meas", None)
        if meas is not None and hasattr(meas, "get_counts"):
            out.append(dict(meas.get_counts())); continue
        regs_counts: Dict[str, int] = {}
        for _, reg_obj in getattr(data, "items", lambda: [])():
            if hasattr(reg_obj, "get_counts"):
                for k, v in reg_obj.get_counts().items():
                    regs_counts[k] = regs_counts.get(k, 0) + int(v)
        if regs_counts:
            out.append(regs_counts); continue
        raise RuntimeError("Could not extract counts from SamplerV2 result item.")
    return out

def _extract_counts_v1(result_obj, shots: int) -> List[Dict[str, int]]:
    qd = getattr(result_obj, "quasi_dists", None)
    if qd is not None:
        out = []
        for q in qd:
            try:
                probs = q.binary_probabilities_dict()
            except Exception:
                probs = dict(q)
            out.append({k: int(round(v * shots)) for k, v in probs.items()})
        return out
    cnts = getattr(result_obj, "counts", None)
    if cnts is not None:
        return [dict(c) for c in cnts] if isinstance(cnts, list) else [dict(cnts)]
    data = getattr(result_obj, "data", None)
    if isinstance(data, list):
        norm = []
        for d in data:
            try:
                probs = d.binary_probabilities_dict()
            except Exception:
                probs = dict(d)
            norm.append({k: int(round(v * shots)) for k, v in probs.items()})
        return norm
    if isinstance(data, dict):
        return [{k: int(round(v * shots)) for k, v in data.items()}]
    raise RuntimeError("Unsupported Sampler V1 result format — cannot extract counts.")

# -------------------- Core execution  --------------------

def _execute_ibm(backend_name: str, circuits: List, shots: int) -> Tuple[List[Dict[str,int]], List[str], List[int]]:
    """
    Preferred: SamplerV2 in job mode (no Session).
    Fallback: V1 with Session (blocked on Open plan: code 1352).
    All circuits submitted in one job.
    """
    svc = devices.get_runtime_service()
    backend_obj = svc.backend(backend_name)
    compiled = transpile(circuits, backend=backend_obj)

    # ---- A) SamplerV2 (job mode; no Session) ----
    if HAS_V2:
        try:
            sampler = SamplerV2(mode=backend_obj)
            job = sampler.run(compiled, shots=shots)
            primitive_result = job.result()
            counts_list = _extract_counts_v2(primitive_result, shots)
        except Exception as e_v2:
            logger.warning(f"SamplerV2 job-mode path failed: {e_v2}. Trying V1 with Session if available.")
        else:
            ctypes, cchoices = [], []
            for qc in circuits:
                meta = getattr(qc, "metadata", {}) or {}
                ctypes.append(meta.get("type", "unknown"))
                cchoices.append(meta.get("choice", -1))
            if len(counts_list) != len(circuits):
                if len(counts_list) == 1 and len(circuits) > 1:
                    counts_list = counts_list * len(circuits)
                else:
                    raise RuntimeError(f"Mismatched results: {len(counts_list)} for {len(circuits)} circuits")
            return counts_list, ctypes, cchoices

    # ---- B) V1 with Session (not allowed on Open plan; code 1352) ----
    try:
        from qiskit_ibm_runtime import Session, Sampler as SamplerV1  # type: ignore
        try:
            with Session(backend=backend_obj) as session:
                sampler_v1 = SamplerV1(session=session)
                job = sampler_v1.run(compiled, shots=shots)
                res = job.result()
                counts_list = _extract_counts_v1(res, shots)
        except Exception as e_session:
            msg = str(e_session)
            if ("open plan" in msg.lower()) or '"code":1352' in msg or "code\":1352" in msg:
                raise RuntimeError(
                    "Sessions are forbidden on the Open plan (code 1352). "
                    "Use SamplerV2 job mode or upgrade plan."
                )
            raise
    except ImportError:
        raise RuntimeError(
            "SamplerV2 is missing and V1 Session path is unavailable. "
            "Please upgrade `qiskit-ibm-runtime` to a version supporting SamplerV2 job mode."
        )

    ctypes, cchoices = [], []
    for qc in circuits:
        meta = getattr(qc, "metadata", {}) or {}
        ctypes.append(meta.get("type", "unknown"))
        cchoices.append(meta.get("choice", -1))
    if len(counts_list) != len(circuits):
        if len(counts_list) == 1 and len(circuits) > 1:
            counts_list = counts_list * len(circuits)
        else:
            raise RuntimeError(f"Mismatched results: {len(counts_list)} for {len(circuits)} circuits")
    return counts_list, ctypes, cchoices

# -------------------- Signing (HMAC optional + RSA auto-provision) --------------------

def _sign_payload(qdna_id: str, features: Dict[str,Any]) -> Dict[str,Any]:
    """
    Returns a signatures block:
      - 'hmac_sha256': hex or None if QDNA_HMAC_KEY not set
      - 'rsa_sha256_hex': hex (always present; auto-generates keys on disk if missing)
      - 'algorithms', 'key_ids', 'pubkey_fingerprint_sha256', 'created_at_utc'
      - 'key_paths': where keys live (useful for ops)
    """
    msg = json.dumps({"qdna_id": qdna_id, "features": features}, sort_keys=True).encode()

    # HMAC (optional; do NOT invent a random key)
    hmac_hex = None
    hmac_key_id = os.environ.get("QDNA_HMAC_KEY_ID")
    hkey_env = os.environ.get("QDNA_HMAC_KEY")
    if hkey_env:
        hkey = hkey_env.encode() if isinstance(hkey_env, str) else hkey_env
        hmac_hex = hmac_sha256(hkey, msg).hex()

    # RSA — load or create keys on disk automatically
    priv, pub, meta = load_or_create_rsa_from_env()
    rsa_key_id = os.environ.get("QDNA_RSA_KEY_ID") or ("ephemeral" if meta.get("created") == "true" else None)

    rsa_sig_hex = rsa_sign(priv, msg).hex()
    pub_fp = meta.get("pubkey_fingerprint_sha256") or public_key_fingerprint_sha256(pub)

    return {
        "hmac_sha256": hmac_hex,
        "rsa_sha256_hex": rsa_sig_hex,
        "algorithms": {
            "hmac": "HMAC-SHA256" if hmac_hex is not None else None,
            "rsa": "RSA-PSS-SHA256",
        },
        "key_ids": {
            "hmac": hmac_key_id,
            "rsa": rsa_key_id,
        },
        "key_paths": {
            "private": meta.get("priv_path"),
            "public": meta.get("pub_path"),
        },
        "pubkey_fingerprint_sha256": pub_fp,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
    }

# -------------------- Public API --------------------

def run_session_on_hardware(*, backend: Optional[str]=None, shots: int=1024) -> Dict[str,Any]:
    shots = _clamp_shots(shots)
    # validate/pick backend
    if backend is None:
        backend = devices.pick_best_device()
    else:
        _ = devices.pick_best_device(preferred=backend)
    # seed & circuits (12 circuits incl. CHSH/Bell)
    seed = new_session_seed()
    circuits = generate_session_circuits(seed)
    # execute
    results, ctypes, cchoices = _execute_ibm(backend, circuits, shots)
    # QDNAID timestamp (UTC, no separators)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    qdna_id = "QDNAID" + timestamp
    provenance = _build_provenance(qdna_id, backend, shots, seed)
    # 1) RAW
    raw_obj = {
        "qdna_id": qdna_id,
        "timestamp_utc": timestamp,
        "backend": backend,
        "seed": seed,
        "num_circuits": len(circuits),
        "shots": shots,
        "circuit_types": ctypes,
        "circuit_choices": cchoices,
        "raw_results": results,
        "provenance": provenance,
    }
    store_raw(qdna_id, backend, raw_obj)
    # 2) FEATURES (includes CHSH S)
    features = feature_vector_from_session(results, ctypes, cchoices)
    store_features(qdna_id, backend, {"qdna_id": qdna_id, "backend": backend, "features": features})
    # 3) SIGN
    signatures = _sign_payload(qdna_id, features)
    store_sign(qdna_id, backend, {"qdna_id": qdna_id, "backend": backend, "signatures": signatures})
    return {
        "available_devices": [d["name"] for d in devices.list_real_devices()],
        "selected_backend": backend,
        "raw": raw_obj,
        "features": features,
        "signatures": signatures,
    }

# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run QDNA session on IBM hardware (SamplerV2 job mode; V1 fallback)")
    parser.add_argument("--backend", type=str, default=None, help="IBM backend name (leave empty for auto-pick)")
    parser.add_argument("--shots", type=int, default=1024, help="256..1024")
    parser.add_argument("--list", action="store_true", help="List available real devices and exit")
    args = parser.parse_args()

    if args.list:
        print(json.dumps({"devices": devices.list_real_devices()}, indent=2))
    else:
        out = run_session_on_hardware(backend=args.backend, shots=args.shots)

        # ==== PRINT ONLY CHSH + FILENAMES (no IO changes to APIs) ====
        backend = out.get("selected_backend") or "unknown"
        qdna_id = (out.get("raw") or {}).get("qdna_id") or "QDNAID"
        features = out.get("features") or {}
        chsh_S = features.get("chsh_S", None)
        # Reconstruct file paths without changing storage IO
        root = "qdna_sessions"
        raw_path      = os.path.join(root, backend, f"{qdna_id}_raw.json")
        features_path = os.path.join(root, backend, f"{qdna_id}_features.json")
        sign_path     = os.path.join(root, backend, f"{qdna_id}_sign.json")

        print(f"CHSH_S={chsh_S}")
        print(f"RAW_FILE={raw_path}")
        print(f"FEATURES_FILE={features_path}")
        print(f"SIGN_FILE={sign_path}")
