# app.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah


Streamlit dashboard (hardware-first):
- List available IBM devices & queues
- Run a session on real hardware (auto-pick or selected)
- Time-series analytics for features per device (not raw file dumps)
- One-click verification (per session or bulk for a backend)
- Cross-device comparison & device fingerprinting + session identification
"""

import os
import json
import datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

import devices
from challenge import run_session_on_hardware
from verification import verify as verify_triple

# -------------------- Constants --------------------
ROOT = "qdna_sessions"
os.makedirs(ROOT, exist_ok=True)

FEATURE_KEYS_ORDERED = [
    "chsh_S",
    "avg_entropy",
    "std_entropy",
    "avg_variance",
    "avg_imbalance",
    "drift_sum",
]
DEFAULT_METRICS = ["chsh_S", "avg_entropy", "drift_sum"]

FINGERPRINT_AGG = "median"  # "median" or "mean"
MIN_SESSIONS_FOR_FINGERPRINT = 3

# -------------------- Helpers --------------------
def _backend_dirs() -> List[str]:
    return sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

def _paths_for_id(backend: str, qdna_id: str) -> Dict[str, str]:
    base = os.path.join(ROOT, backend)
    return {
        "raw": os.path.join(base, f"{qdna_id}_raw.json"),
        "features": os.path.join(base, f"{qdna_id}_features.json"),
        "sign": os.path.join(base, f"{qdna_id}_sign.json"),
    }

def _parse_timestamp_from_qdna(qdna_id: str) -> dt.datetime:
    ts = qdna_id.replace("QDNAID", "")
    return dt.datetime.strptime(ts, "%Y%m%d%H%M%S")

def _load_features_df(backend: str) -> pd.DataFrame:
    base = os.path.join(ROOT, backend)
    if not os.path.isdir(base):
        return pd.DataFrame(columns=["timestamp", "qdna_id", "backend"] + FEATURE_KEYS_ORDERED)

    feat_files = sorted([f for f in os.listdir(base) if f.endswith("_features.json")])
    rows = []
    for ff in feat_files:
        qdna_id = ff.replace("_features.json", "")
        paths = _paths_for_id(backend, qdna_id)
        try:
            with open(paths["features"], "r") as f:
                obj = json.load(f)
            features = (obj or {}).get("features", {})
            if os.path.exists(paths["raw"]):
                with open(paths["raw"], "r") as rf:
                    raw = json.load(rf)
                ts_str = raw.get("timestamp_utc")
                ts = pd.to_datetime(ts_str, format="%Y%m%d%H%M%S") if ts_str else _parse_timestamp_from_qdna(qdna_id)
            else:
                ts = _parse_timestamp_from_qdna(qdna_id)
            rows.append({
                "timestamp": pd.to_datetime(ts),
                "qdna_id": qdna_id,
                "backend": backend,
                **{k: features.get(k, None) for k in FEATURE_KEYS_ORDERED},
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "qdna_id", "backend"] + FEATURE_KEYS_ORDERED)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df

def _load_all_features() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for b in _backend_dirs():
        df = _load_features_df(b)
        if not df.empty:
            out[b] = df
    return out

def _verify_one(backend: str, qdna_id: str) -> Dict[str, Any]:
    return verify_triple(ROOT, backend, qdna_id)

def _verify_all(backend: str) -> pd.DataFrame:
    base = os.path.join(ROOT, backend)
    if not os.path.isdir(base):
        return pd.DataFrame(columns=["qdna_id", "ok", "hmac", "rsa", "provenance"])

    raw_files = sorted([f for f in os.listdir(base) if f.endswith("_raw.json")])
    ids = [f.replace("_raw.json", "") for f in raw_files]
    rows = []
    for qdna_id in ids:
        v = _verify_one(backend, qdna_id)
        checks = v.get("checks", {})
        rows.append({
            "qdna_id": qdna_id,
            "ok": v.get("ok"),
            "hmac": checks.get("hmac_match"),
            "rsa": checks.get("rsa_valid"),
            "provenance": checks.get("provenance_present"),
        })
    return pd.DataFrame(rows).sort_values(["ok", "qdna_id"], ascending=[True, True])

# -------------------- Fingerprinting & Similarity --------------------
def _robust_fingerprint(df: pd.DataFrame, keys: List[str], agg: str = FINGERPRINT_AGG) -> Tuple[np.ndarray, Dict[str, float]]:
    if df.empty:
        return np.full(len(keys), np.nan, float), {k: np.nan for k in keys}
    if agg == "median":
        fp_vals = np.nanmedian(df[keys].to_numpy(dtype=float), axis=0)
    else:
        fp_vals = np.nanmean(df[keys].to_numpy(dtype=float), axis=0)
    mapping = {k: float(fp_vals[i]) if not np.isnan(fp_vals[i]) else np.nan for i, k in enumerate(keys)}
    return fp_vals.astype(float), mapping

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0: return np.nan
    a_m = a[mask]; b_m = b[mask]
    denom = (np.linalg.norm(a_m) * np.linalg.norm(b_m))
    if denom == 0: return np.nan
    return float(np.dot(a_m, b_m) / denom)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0: return np.nan
    return float(np.linalg.norm(a[mask] - b[mask]))

def _identify_session_device(target_vec: np.ndarray, device_fps: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for backend, fp in device_fps.items():
        rows.append({
            "backend": backend,
            "cosine_sim": _cosine_similarity(target_vec, fp),
            "euclidean": _euclidean_distance(target_vec, fp)
        })
    df = pd.DataFrame(rows)
    df["cos_rank"] = df["cosine_sim"].rank(ascending=False, method="dense")
    df["euc_rank"] = df["euclidean"].rank(ascending=True, method="dense")
    df = df.sort_values(["cos_rank", "euc_rank", "backend"]).reset_index(drop=True)
    return df[["backend", "cosine_sim", "euclidean"]]

# -------------------- UI --------------------
st.set_page_config(page_title="QDNA Dashboard", layout="wide")

st.title("QDNA-ID: Quantum Device Native Authentication")
st.markdown(
    "**Developed by:** Osamah N. Neamah &nbsp;&nbsp;|&nbsp;&nbsp; "
    "**Institution:** Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey",
)
st.markdown(
    "A quantum provenance dashboard that links physical device behavior to "
    "cryptographically verifiable, signed fingerprints. "
    "**A complete trust chain that connects physical quantum behavior to digitally verifiable**."
)

with st.expander("About QDNA-ID (Concept & Contact)", expanded=True):
    st.markdown(
        """
**QDNA-ID** introduces a *quantum provenance* pipeline that couples physical quantum behavior with
digitally verifiable records. The system builds a measurable, cryptographically sealed fingerprint
for every device run.

**Uniqueness:** it integrates quantum verification (CHSH) → feature extraction → digital signing & timestamping
→ independent verification/classification in one cohesive workflow.

**Who benefits:** academia & research, quantum security/crypto, cloud quantum providers, Web3 provenance,
and scientific reproducibility initiatives.

**Project status:** 🚧 *Proof-of-Concept (PoC).* Continuous updates planned (models, verification API, cloud integrations).

**Contact:**  
- Author: *Osamah N. Neamah*  
- Institution: *Karabuk University — Quantum Provenance Initiative*  
- Email: [osamannehme@gmail.com](mailto:osamannehme@gmail.com)  
- LinkedIn: [linkedin.com/in/osamah-n-neamah-b2774118b](https://www.linkedin.com/in/osamah-n-neamah-b2774118b)  
- Website: qdnaid.org *(coming soon)*

**© 2025 QDNA-ID — Academic PoC license. Unauthorized commercial use prohibited.**
        """
    )

# -------------------- Sidebar: Connection (token-only) --------------------
st.sidebar.header("Connection")
token_present = bool(os.environ.get("QISKIT_IBM_RUNTIME_API_TOKEN"))
if not token_present:
    st.sidebar.error("QISKIT_IBM_RUNTIME_API_TOKEN is not set. Set it and reload the app.")
if st.sidebar.button("Diagnose connection"):
    try:
        info = devices.diagnose_connection()
        st.sidebar.success("Diagnostics collected")
        st.sidebar.json(info)
    except Exception as e:
        st.sidebar.error(f"Diagnose failed: {e}")

# -------------------- Sidebar: Devices --------------------
st.sidebar.header("Available IBM Devices")
try:
    devs = devices.list_real_devices()
    if not devs:
        st.sidebar.error("No real IBM devices visible. Check token (see Connection above).")
    else:
        for d in devs:
            st.sidebar.write(f"- **{d['name']}** · qubits: {d['num_qubits']} · queue: {d['queue']}")
except Exception as e:
    st.sidebar.error(str(e))
    devs = []

# -------------------- Sidebar: Run --------------------
st.sidebar.header("Run on Hardware")
shots = st.sidebar.slider("Shots", 256, 1024, 1024, 64)
options = ["(auto-pick)"] + [d["name"] for d in devs]
pick = st.sidebar.selectbox("Select backend", options)
if st.sidebar.button("Run session"):
    backend = None if pick == "(auto-pick)" else pick
    with st.spinner("Submitting circuits to IBM hardware..."):
        out = run_session_on_hardware(backend=backend, shots=shots)
    st.success(f"Run complete on {out['selected_backend']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Backend", out["selected_backend"])
    with col2:
        st.metric("Shots", out["raw"]["shots"])
    with col3:
        st.metric("CHSH S", f"{out['features'].get('chsh_S', '—')}")
    st.caption("Session stored under qdna_sessions/<backend>/QDNAID..._{raw|features|sign}.json")

st.markdown("---")

# Tabs
tab_overview, tab_analytics, tab_compare, tab_fingerprint, tab_verify = st.tabs(
    ["Overview", "Analytics (time-series)", "Compare devices", "Fingerprint & Identify", "Verification"]
)

# -------------------- Overview --------------------
with tab_overview:
    st.subheader("Backends & Sessions")
    backends = _backend_dirs()
    if not backends:
        st.info("No stored sessions yet. Run a session to populate analytics.")
    else:
        summary = []
        for b in backends:
            base = os.path.join(ROOT, b)
            n = len([f for f in os.listdir(base) if f.endswith("_raw.json")])
            summary.append({"backend": b, "sessions": n})
        st.table(pd.DataFrame(summary).sort_values("sessions", ascending=False).reset_index(drop=True))

# -------------------- Analytics (time-series) --------------------
with tab_analytics:
    st.subheader("Time-series Analytics by Backend")
    backends = _backend_dirs()
    if not backends:
        st.info("No data to analyze yet.")
    else:
        b = st.selectbox("Pick backend for analytics", backends, index=0, key="analytics_backend")
        df = _load_features_df(b)
        if df.empty:
            st.warning(f"No features found for backend '{b}'.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                metrics = st.multiselect("Metrics", FEATURE_KEYS_ORDERED, default=DEFAULT_METRICS)

            min_ts_pd = df["timestamp"].min()
            max_ts_pd = df["timestamp"].max()
            min_ts = pd.to_datetime(min_ts_pd).to_pydatetime()
            max_ts = pd.to_datetime(max_ts_pd).to_pydatetime()
            if min_ts == max_ts:
                default_start = min_ts - dt.timedelta(minutes=1)
                default_end   = max_ts + dt.timedelta(minutes=1)
                slider_min    = default_start
                slider_max    = default_end
            else:
                default_start = min_ts
                default_end   = max_ts
                slider_min    = min_ts
                slider_max    = max_ts

            with c2:
                start, end = st.slider(
                    "Time window",
                    min_value=slider_min,
                    max_value=slider_max,
                    value=(default_start, default_end),
                    format="YYYY-MM-DD HH:mm",
                )
            with c3:
                show_points = st.checkbox("Show latest point values", value=True)

            mask = (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))
            view = df.loc[mask].copy()

            if view.empty:
                st.info("No sessions in the selected time window.")
            else:
                view = view.set_index("timestamp")
                if "chsh_S" in view.columns:
                    last_S = view["chsh_S"].dropna().tail(1)
                    if not last_S.empty:
                        s_val = last_S.iloc[0]
                        color = "🟢" if s_val >= 2.0 else "🟠" if 1.8 <= s_val < 2.0 else "🔴"
                        st.markdown(f"**Latest CHSH S:** {color} `{s_val:.3f}`  (threshold 2.0)")

                for m in metrics:
                    if m in view.columns:
                        st.markdown(f"**{m}**")
                        st.line_chart(view[[m]])
                if show_points:
                    st.markdown("**Latest values (most recent sessions)**")
                    latest = df.sort_values("timestamp", ascending=False).head(10)
                    st.dataframe(latest[["timestamp", "qdna_id"] + metrics].reset_index(drop=True))

# -------------------- Compare devices --------------------
with tab_compare:
    st.subheader("Cross-device Comparison")
    all_data = _load_all_features()
    all_backends = list(all_data.keys())
    if len(all_backends) < 2:
        st.info("Need at least two backends with data to compare.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            b1 = st.selectbox("Device A", all_backends, index=0, key="cmp_a")
            df1 = all_data[b1].copy()
            df1["timestamp"] = pd.to_datetime(df1["timestamp"])
        with c2:
            b2 = st.selectbox("Device B", all_backends, index=1, key="cmp_b")
            df2 = all_data[b2].copy()
            df2["timestamp"] = pd.to_datetime(df2["timestamp"])

        metrics = st.multiselect("Metrics to compare", FEATURE_KEYS_ORDERED, default=DEFAULT_METRICS, key="cmp_metrics")

        N = st.slider("How many recent sessions to preview", 5, 50, 10, key="cmp_n")
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**{b1} — Recent {N}**")
            st.dataframe(df1.sort_values("timestamp", ascending=False).head(N)[["timestamp","qdna_id"] + metrics])
        with colB:
            st.markdown(f"**{b2} — Recent {N}**")
            st.dataframe(df2.sort_values("timestamp", ascending=False).head(N)[["timestamp","qdna_id"] + metrics])

        fp1, map1 = _robust_fingerprint(df1, FEATURE_KEYS_ORDERED, agg=FINGERPRINT_AGG)
        fp2, map2 = _robust_fingerprint(df2, FEATURE_KEYS_ORDERED, agg=FINGERPRINT_AGG)

        sim_cos = _cosine_similarity(fp1, fp2)
        dist_eu = _euclidean_distance(fp1, fp2)

        st.markdown("**Device fingerprints (robust aggregation)**")
        fp_df = pd.DataFrame({
            "metric": FEATURE_KEYS_ORDERED,
            f"{b1}_fp": [map1[k] for k in FEATURE_KEYS_ORDERED],
            f"{b2}_fp": [map2[k] for k in FEATURE_KEYS_ORDERED],
        })
        st.dataframe(fp_df)

        st.metric("Cosine similarity (A vs B)", "—" if np.isnan(sim_cos) else f"{sim_cos:.4f}")
        st.metric("Euclidean distance (A vs B)", "—" if np.isnan(dist_eu) else f"{dist_eu:.4f}")

# -------------------- Fingerprint & Identify --------------------
with tab_fingerprint:
    st.subheader("Device Fingerprinting & Session Identification")

    all_data = _load_all_features()
    if not all_data:
        st.info("No data yet to build device fingerprints.")
    else:
        fps: Dict[str, np.ndarray] = {}
        stats_rows = []
        for b, dfb in all_data.items():
            count_sessions = len(dfb)
            fp_vec, fp_map = _robust_fingerprint(dfb, FEATURE_KEYS_ORDERED, agg=FINGERPRINT_AGG)
            fps[b] = fp_vec
            have_enough = count_sessions >= MIN_SESSIONS_FOR_FINGERPRINT
            stats_rows.append({
                "backend": b,
                "sessions": count_sessions,
                "fingerprint_ready": have_enough,
                **{f"fp_{k}": fp_map[k] for k in FEATURE_KEYS_ORDERED}
            })
        st.markdown("**Device fingerprints (computed over historical sessions)**")
        st.dataframe(pd.DataFrame(stats_rows).sort_values(["fingerprint_ready","sessions","backend"], ascending=[False, False, True]))

        st.markdown("---")
        st.subheader("Identify which device a session most likely belongs to")

        b_choices = list(all_data.keys())
        b_pick = st.selectbox("Pick backend folder of the target session", b_choices, index=0, key="id_backend")
        base = os.path.join(ROOT, b_pick)
        feat_files = sorted([f for f in os.listdir(base) if f.endswith("_features.json")])
        ids = [f.replace("_features.json", "") for f in feat_files]
        q_pick = st.selectbox("Pick QDNAID (target session)", ids, index=0 if ids else None, key="id_qdna")

        if ids and q_pick:
            try:
                with open(os.path.join(base, f"{q_pick}_features.json"), "r") as f:
                    obj = json.load(f)
                feats_map = (obj or {}).get("features", {})
                target_vec = np.array([feats_map.get(k, np.nan) for k in FEATURE_KEYS_ORDERED], dtype=float)
                id_df = _identify_session_device(target_vec, fps)
                st.dataframe(id_df)
                if not id_df.empty and not id_df["cosine_sim"].isna().all():
                    best = id_df.sort_values(["cosine_sim", "euclidean"], ascending=[False, True]).iloc[0]
                    st.success(f"Most similar device likely: **{best['backend']}** "
                               f"(cos={best['cosine_sim']:.4f}, euclidean={best['euclidean']:.4f})")
                else:
                    st.warning("Insufficient fingerprints or comparable features to decide.")
            except Exception as e:
                st.error(f"Unable to load features for {q_pick}: {e}")

# -------------------- Verification --------------------
with tab_verify:
    st.subheader("Verification")
    backends = _backend_dirs()
    if not backends:
        st.info("No stored sessions to verify.")
    else:
        b = st.selectbox("Pick backend to verify", backends, index=0, key="verify_backend")
        base = os.path.join(ROOT, b)
        raw_files = sorted([f for f in os.listdir(base) if f.endswith("_raw.json")])
        ids = [f.replace("_raw.json", "") for f in raw_files]
        c1, c2 = st.columns([2, 1])
        with c1:
            q = st.selectbox("Select QDNAID to verify (single)", [""] + ids)
        with c2:
            if st.button("Verify ALL sessions in backend"):
                with st.spinner("Verifying all sessions..."):
                    dfv = _verify_all(b)
                st.success("Verification complete")
                st.dataframe(dfv[["qdna_id", "ok", "hmac", "rsa", "provenance"]])
        if q:
            with st.spinner(f"Verifying {q} ..."):
                res = _verify_one(b, q)
            st.json(res)
