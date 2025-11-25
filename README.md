🧬 QDNA-ID — Quantum Device Native Authentication  
**Developed by:** Osamah N. Neamah  
**Institution:** Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey. 

> A quantum provenance dashboard that links physical device behavior  
> to cryptographically verifiable, signed fingerprints.  
> **A complete trust chain connecting physical quantum behavior to digital verification.**

---

## 🧠 Project Overview

**QDNA-ID** introduces a *quantum provenance* framework that couples quantum hardware behavior with  
digitally verifiable cryptographic records. Each quantum execution generates a **fingerprint** that is:

1. **Measured** on a *real* IBM Quantum device.  
2. **Characterized** by quantum-mechanical metrics (e.g., CHSH S-value).  
3. **Encoded** into a deterministic feature vector.  
4. **Digitally signed & timestamped** using HMAC-SHA256 and RSA-PSS-SHA256.  
5. **Stored and verifiable** as a reproducible provenance record.

This provides **a measurable and cryptographically sealed identity** for each quantum device run —  
bridging *physical* quantum entropy with *digital* cryptographic trust.

---

## 🧩 Key Features

- ✅ **Hardware-based measurement** (no simulators; tested on real IBM Quantum devices).  
- 🔐 **Dual cryptographic signatures** — HMAC-SHA256 (symmetric) & RSA-PSS-SHA256 (asymmetric).  
- 📈 **CHSH-S Quantum Verification** — non-classical correlation score (≥ 2.0 required).  
- 🧾 **Provenance Metadata** — full environment capture: runtime versions, device ID, timestamps.  
- 🧮 **Feature Extraction** — converts raw counts to structured quantum fingerprints.  
- 🗃️ **Hierarchical Storage** —  
qdna_sessions/<backend>/
├── <QDNAID>_raw.json
├── <QDNAID>_features.json
└── <QDNAID>_sign.json

ruby
Copy code
- 🌐 **Streamlit Dashboard** — live display of CHSH metrics, signatures, and device provenance.

---

## 🧪 Technical Stack

| Layer | Technology | Purpose |
|:------|:------------|:--------|
| Quantum Execution | [Qiskit](https://qiskit.org/) | Circuit transpilation and IBM backend interface |
| Runtime | `qiskit_ibm_runtime.SamplerV2` | Hardware job submission |
| Cryptography | `cryptography.hazmat` | HMAC & RSA (SHA-256) |
| Visualization | `Streamlit` | Interactive provenance dashboard |
| Storage | JSON + filesystem | Immutable provenance store |
| Environment | Python ≥ 3.10 | Recommended for full reproducibility |

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/osamah-nn/qDNA-ID.git
cd qDNA-ID

# (Optional) create environment
python -m venv qdnaidex
source qdnaidex/bin/activate   # or .\qdnaidex\Scripts\activate on Windows

# Install dependencies
pip install -r qdna_id.yml
Dependencies:
qiskit, qiskit-ibm-runtime, cryptography, numpy, streamlit, pandas


🔑 Environment Configuration
Before running, set cryptographic and IBM environment variables:
Copy code
IBM (Configure devices.py by your IBM Cloud ConfigurationS)
# === Your static configuration (with ENV overrides allowed) === 
# Windows
setx QISKIT_IBM_RUNTIME_API_TOKEN "Your API IBM TOKEN"
# Linux 
export QISKIT_IBM_RUNTIME_API_TOKEN="Your API IBM TOKEN"
🚀 Running a Quantum Session
Run directly from command line:

bash
Copy code
python challenge.py --backend ibm_torino --shots 1024
Output Example

ini
Copy code
CHSH_S=2.39
RAW_FILE=qdna_sessions/ibm_kyiv/QDNAID20251017163045_raw.json
FEATURES_FILE=qdna_sessions/ibm_kyiv/QDNAID20251017163045_features.json
SIGN_FILE=qdna_sessions/ibm_kyiv/QDNAID20251017163045_sign.json
Or start the dashboard:

bash
Copy code
streamlit run app.py
📊 Output Files
File	Description
*_raw.json	Original quantum counts + full provenance metadata
*_features.json	Extracted metrics including chsh_S
*_sign.json	Digital signatures (HMAC + RSA) + pubkey fingerprint

🔬 Quantum Provenance Workflow
java
Copy code
Quantum Hardware → Counts → CHSH Verification
       ↓
 Feature Vectorization
       ↓
 Cryptographic Signing (HMAC + RSA)
       ↓
 Provenance Store (Immutable JSON)
       ↓
 Streamlit Dashboard / Verification API
Each record forms a verifiable “QDNA-ID” chain, connecting physical behavior
to digital authentication — enabling reproducibility, security, and trust.

🧾 Example Signatures Block
json
Copy code
"signatures": {
  "hmac_sha256": "b4e9b1f...c74",
  "rsa_sha256_hex": "9d3b...1f",
  "algorithms": {
    "hmac": "HMAC-SHA256",
    "rsa": "RSA-PSS-SHA256"
  },
  "key_ids": {
    "hmac": "dev-hmac-01",
    "rsa": "dev-rsa-01"
  },
  "pubkey_fingerprint_sha256": "e3c0...9fa",
  "created_at_utc": "2025-10-17T15:23:12Z"
}
🧮 Academic Context
Discipline: Quantum Computing, Cryptography, Provenance Informatics

Institution: Karabuk University

Research Group: Quantum Provenance Initiative


Lead Developer: Osamah N. Neamah

This project serves as an academic Proof-of-Concept (PoC) — demonstrating a full-chain
quantum trust model from hardware to digital signature.

⚠️ License & Notice
© 2025 QDNA-ID — Academic PoC License

This work is provided for academic and research use only.
Unauthorized commercial use, redistribution, or derivative production is prohibited
without explicit written consent from the author.

📬 Contact
Author: Osamah N. Neamah

Institution: Karabuk University — Quantum Provenance Initiative

Email: osamannehme@gmail.com

LinkedIn: linkedin.com/in/osamah-n-neamah-b2774118b

Website: qdnaid.org (coming soon)
