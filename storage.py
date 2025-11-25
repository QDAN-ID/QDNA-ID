# storage.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey

Storage:
- Root: qdna_sessions/<backend_name>/
- Files: QDNAID<timestamp>_{raw|features|sign}.json
"""

import os, json
from typing import Tuple

ROOT_DIR = "qdna_sessions"

def _ensure_dir(backend_name: str) -> str:
    d = os.path.join(ROOT_DIR, backend_name)
    os.makedirs(d, exist_ok=True)
    return d

def base_paths(qdna_id: str, backend_name: str) -> Tuple[str,str,str]:
    d = _ensure_dir(backend_name)
    raw   = os.path.join(d, f"{qdna_id}_raw.json")
    feats = os.path.join(d, f"{qdna_id}_features.json")
    sign  = os.path.join(d, f"{qdna_id}_sign.json")
    return raw, feats, sign

def store_raw(qdna_id: str, backend_name: str, raw_obj: dict) -> str:
    raw_path, _, _ = base_paths(qdna_id, backend_name)
    with open(raw_path, "w") as f:
        json.dump(raw_obj, f, indent=2, sort_keys=True)
    return raw_path

def store_features(qdna_id: str, backend_name: str, features_obj: dict) -> str:
    _, feats_path, _ = base_paths(qdna_id, backend_name)
    with open(feats_path, "w") as f:
        json.dump(features_obj, f, indent=2, sort_keys=True)
    return feats_path

def store_sign(qdna_id: str, backend_name: str, sign_obj: dict) -> str:
    _, _, sign_path = base_paths(qdna_id, backend_name)
    with open(sign_path, "w") as f:
        json.dump(sign_obj, f, indent=2, sort_keys=True)
    return sign_path
