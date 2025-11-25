# features.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
Karabuk University — Quantum Provenance Initiative

Feature extraction:
- Shannon entropy (per-circuit)
- Probability variance (per-circuit)
- Drift of dominant outcome
- Imbalance index
- CHSH S (S >= 2.0 indicates violation)
"""

import math, numpy as np
from typing import Dict, List, Tuple

def _entropy(counts: Dict[str,int]) -> float:
    tot = sum(counts.values())
    if tot == 0: return 0.0
    return -sum((v/tot)*math.log2(v/tot) for v in counts.values() if v)

def _probs(counts: Dict[str,int]) -> Dict[str,float]:
    tot = sum(counts.values())
    return {k: (v/tot if tot else 0.0) for k,v in counts.items()}

def _dominant_prob(probs: Dict[str,float]) -> float:
    return max(probs.values()) if probs else 0.0

def _imbalance(probs: Dict[str,float]) -> float:
    p00, p01, p10, p11 = (probs.get(s,0.0) for s in ("00","01","10","11"))
    return abs(p00 - p11) + abs(p01 - p10)

def _E_from_counts(counts: Dict[str,int]) -> float:
    p = _probs(counts)
    p00, p01, p10, p11 = (p.get(s,0.0) for s in ("00","01","10","11"))
    return (p00 + p11) - (p01 + p10)

def compute_chsh_S(chsh_results: List[Tuple[int, Dict[str,int]]]) -> float:
    """
    chsh_results: list of (choice, counts) for choice in {0,1,2,3}:
      0: (A,B), 1:(A,B'), 2:(A',B), 3:(A',B')
    S = E(A,B)+E(A,B')+E(A',B)-E(A',B')
    """
    E = {}
    for choice, counts in chsh_results:
        E[choice] = _E_from_counts(counts)
    if not all(k in E for k in (0,1,2,3)):
        return float("nan")
    return (E[0] + E[1] + E[2] - E[3])

def feature_vector_from_session(results_list: List[Dict[str,int]], circuit_types: List[str], circuit_choices: List[int]) -> Dict:
    entropies, variances, imbalances, probs_list = [], [], [], []
    chsh_pairs: List[Tuple[int, Dict[str,int]]] = []

    for counts, ctype, choice in zip(results_list, circuit_types, circuit_choices):
        pr = _probs(counts)
        entropies.append(_entropy(counts))
        variances.append(np.var(list(pr.values()) if pr else [0.0]))
        imbalances.append(_imbalance(pr))
        probs_list.append(pr)
        if ctype == "chsh":
            chsh_pairs.append((choice, counts))

    dom = [_dominant_prob(p) for p in probs_list]
    drift_sum = sum(abs(dom[i]-dom[i-1]) for i in range(1, len(dom))) if dom else 0.0

    chsh_S = compute_chsh_S(chsh_pairs)
    chsh_violation = (not math.isnan(chsh_S)) and (chsh_S >= 2.0)

    return {
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        "avg_variance": float(np.mean(variances)) if variances else 0.0,
        "avg_imbalance": float(np.mean(imbalances)) if imbalances else 0.0,
        "drift_sum": float(drift_sum),
        "per_circuit_entropy": entropies,
        "per_circuit_variance": variances,
        "per_circuit_imbalance": imbalances,
        "chsh_S": None if math.isnan(chsh_S) else float(chsh_S),
        "chsh_violation": chsh_violation
    }
