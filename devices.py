# devices.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey

IBM Quantum device discovery & connection helpers
(Token-only via QISKIT_IBM_RUNTIME_API_TOKEN; auto-pick channel)
"""

from typing import List, Dict, Optional
import os

__all__ = [
    "get_runtime_service",
    "list_real_devices",
    "pick_best_device",
    "diagnose_connection",
    "active_channel",
    "PROJECT_NAME",
    "INSTANCE_NAME",
]

# Identity for provenance only (not used for auth)
PROJECT_NAME = "quantum-lab-karabuk"
INSTANCE_NAME = "token_only"  # label only; NOT used for connection

# Cache
_service_cache = None
_active_channel = None  # "ibm_quantum_platform" or "ibm_cloud"

# Channels supported by the installed qiskit-ibm-runtime
CHANNEL_CANDIDATES = ["ibm_quantum_platform", "ibm_cloud"]


def get_runtime_service():
    """
    Create (and cache) QiskitRuntimeService using ONLY QISKIT_IBM_RUNTIME_API_TOKEN.
    Auto-tries accepted channels:
      1) "ibm_quantum_platform"
      2) "ibm_cloud"
    No instance/CRN is ever passed.
    """
    global _service_cache, _active_channel
    if _service_cache is not None:
        return _service_cache

    token = os.environ.get("QISKIT_IBM_RUNTIME_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing QISKIT_IBM_RUNTIME_API_TOKEN environment variable.")

    from qiskit_ibm_runtime import QiskitRuntimeService

    errors = []
    for ch in CHANNEL_CANDIDATES:
        try:
            svc = QiskitRuntimeService(channel=ch, token=token)
            _service_cache = svc
            _active_channel = ch
            return _service_cache
        except Exception as e:
            errors.append(f"{ch}: {e}")

    raise RuntimeError(
        "Failed to initialize QiskitRuntimeService with token-only. "
        "Tried channels: " + " | ".join(errors)
    )


def active_channel() -> Optional[str]:
    """Return the successfully used channel ('ibm_quantum_platform' or 'ibm_cloud')."""
    return _active_channel


def _safe_backend_name(b) -> str:
    n = getattr(b, "name", None)
    if callable(n):
        try:
            return n()
        except Exception:
            pass
    if isinstance(n, str):
        return n
    return getattr(b, "backend_name", None) or str(b)


def _safe_num_qubits(b) -> Optional[int]:
    cfg = None
    conf_attr = getattr(b, "configuration", None)
    if callable(conf_attr):
        try:
            cfg = conf_attr()
        except Exception:
            cfg = None
    elif conf_attr is not None:
        cfg = conf_attr
    if cfg is not None:
        return getattr(cfg, "num_qubits", None)
    return getattr(b, "num_qubits", None)


def _safe_is_simulator(b) -> bool:
    cfg = None
    conf_attr = getattr(b, "configuration", None)
    if callable(conf_attr):
        try:
            cfg = conf_attr()
        except Exception:
            cfg = None
    elif conf_attr is not None:
        cfg = conf_attr
    if cfg is not None:
        return bool(getattr(cfg, "simulator", False))
    return bool(getattr(b, "simulator", False))


def _safe_queue_depth(b) -> int:
    qlen = getattr(b, "pending_jobs", None)
    if isinstance(qlen, int):
        return qlen
    try:
        status = b.status() if callable(getattr(b, "status", None)) else None
        pj = getattr(status, "pending_jobs", 0) if status is not None else 0
        return int(pj or 0)
    except Exception:
        return 0


def list_real_devices() -> List[Dict]:
    """Return real, operational devices (no simulators)."""
    svc = get_runtime_service()
    try:
        backs = svc.backends(operational=True)
    except TypeError:
        backs = svc.backends()

    out: List[Dict] = []
    for b in backs:
        if _safe_is_simulator(b):
            continue
        out.append({
            "name": _safe_backend_name(b),
            "num_qubits": _safe_num_qubits(b),
            "queue": _safe_queue_depth(b),
            "simulator": False,
            "operational": True,
        })
    out.sort(key=lambda d: (d["queue"], -(d["num_qubits"] or 0)))
    return out


def pick_best_device(preferred: Optional[str] = None) -> str:
    devs = list_real_devices()
    if not devs:
        raise RuntimeError("No real IBM devices available with the current token.")
    if preferred:
        names = [d["name"] for d in devs]
        if preferred in names:
            return preferred
        raise RuntimeError(f"Preferred backend '{preferred}' not available. Available: {names}")
    return devs[0]["name"]


def diagnose_connection() -> Dict:
    """Verbose connection/visibility diagnostics (token-only, auto channel)."""
    info: Dict = {
        "project_name": PROJECT_NAME,
        "token_provided": bool(os.environ.get("QISKIT_IBM_RUNTIME_API_TOKEN")),
    }
    try:
        svc = get_runtime_service()
        info["service_ok"] = True
        info["channel_used"] = active_channel()
        try:
            backs = svc.backends()
            real = [b for b in backs if not _safe_is_simulator(b)]
            info["devices_count_total"] = len(backs)
            info["devices_count_real"] = len(real)
            info["device_names_real"] = [_safe_backend_name(b) for b in real]
        except Exception as e_list:
            info["list_error"] = str(e_list)
    except Exception as e:
        info["service_ok"] = False
        info["service_error"] = str(e)
    return info
