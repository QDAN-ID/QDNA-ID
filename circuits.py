# circuits.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey

Create quantum circuits:
- 12 circuits per session
- Bell state circuits and CHSH test circuits
- Random entangling circuits
- Guaranteed CHSH coverage of choices {0,1,2,3} at least once per session
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
from typing import List
from seed import seed_to_rng

NUM_CIRCUITS = 12  

def bell_state_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, "q"); cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0]); qc.cx(qr[0], qr[1]); qc.measure(qr, cr)
    qc.metadata = {"type": "bell"}
    return qc

def chsh_circuit(theta_a: float, theta_ap: float, theta_b: float, theta_bp: float, choice: int) -> QuantumCircuit:
    qr = QuantumRegister(2, "q"); cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)
    # Prepare Bell state
    qc.h(qr[0]); qc.cx(qr[0], qr[1])
    # Measurement settings mapping
    mapping = {
        0: (theta_a,  theta_b),   # (A,  B)
        1: (theta_a,  theta_bp),  # (A,  B')
        2: (theta_ap, theta_b),   # (A', B)
        3: (theta_ap, theta_bp),  # (A', B')
    }
    ang_a, ang_b = mapping[choice]
    qc.ry(2 * ang_a, qr[0])
    qc.ry(2 * ang_b, qr[1])
    qc.measure(qr, cr)
    qc.metadata = {"type": "chsh", "choice": choice}
    return qc

def random_entangling(rng) -> QuantumCircuit:
    qr = QuantumRegister(2, "q"); cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)
    for q in qr:
        qc.rx(rng.random() * 2 * math.pi, q)
        qc.ry(rng.random() * 2 * math.pi, q)
    qc.cx(qr[0], qr[1]); qc.measure(qr, cr)
    qc.metadata = {"type": "rand"}
    return qc

def generate_session_circuits(seed: str) -> List[QuantumCircuit]:
    """
    Generate 12 circuits with pattern (per i): 0 -> Bell, 1 -> CHSH, 2 -> Random.
    Ensures CHSH choices {0,1,2,3} appear at least once across the CHSH slots.
    IO unchanged: returns a list[QuantumCircuit] with metadata set.
    """
    rng = seed_to_rng(seed)
    circuits: List[QuantumCircuit] = []

    # Standard CHSH angles (good for near-optimal violation on ideal hardware)
    theta_a, theta_ap = 0.0, math.pi / 4
    theta_b, theta_bp = math.pi / 8, -math.pi / 8

    # Positions i where i % 3 == 1 are CHSH slots → with NUM_CIRCUITS=12 that gives exactly 4 slots.
    chsh_slots = [i for i in range(NUM_CIRCUITS) if i % 3 == 1]

    # Guarantee coverage: shuffle [0,1,2,3] to avoid deterministic pattern, map to CHSH slots.
    guaranteed_choices = [0, 1, 2, 3]
    rng.shuffle(guaranteed_choices)

    # Build circuits
    chsh_idx = 0
    for i in range(NUM_CIRCUITS):
        if i % 3 == 0:
            circuits.append(bell_state_circuit())
        elif i % 3 == 1:
            # Use guaranteed coverage for the first len(chsh_slots) entries
            choice = guaranteed_choices[chsh_idx] if chsh_idx < len(guaranteed_choices) else rng.randint(0, 3)
            chsh_idx += 1
            circuits.append(chsh_circuit(theta_a, theta_ap, theta_b, theta_bp, choice))
        else:
            circuits.append(random_entangling(rng))

    return circuits
