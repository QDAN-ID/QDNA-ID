# seed.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
Karabuk University — Quantum Provenance Initiative

Seed generation and deterministic replay helper.
"""

import os, json, hashlib, secrets
from typing import List

SEED_STORE = "seeds_store.json"

def new_session_seed() -> str:
    seed = secrets.token_hex(32)
    store_seed(seed)
    return seed

def store_seed(seed: str):
    data = []
    if os.path.exists(SEED_STORE):
        with open(SEED_STORE, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    data.append({"seed": seed})
    with open(SEED_STORE, "w") as f:
        json.dump(data, f)

def seed_to_rng(seed: str):
    import random
    digest = hashlib.sha256(seed.encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))

def load_all_seeds() -> List[str]:
    if not os.path.exists(SEED_STORE):
        return []
    with open(SEED_STORE, "r") as f:
        try:
            items = json.load(f)
            return [x["seed"] for x in items]
        except Exception:
            return []
