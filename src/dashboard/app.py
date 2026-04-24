"""
Dashboard Streamlit — CoachPosture AI

Interface temps réel avec :
  - Flux vidéo webcam avec squelette MediaPipe dessiné
  - Score posture 0-100 (jauge animée)
  - Historique des alertes (graphique temporel)
  - Recommandations de l'agent LLM (Ollama)
  - Détails des features géométriques

Lancement :
    streamlit run src/dashboard/app.py
    streamlit run src/dashboard/app.py -- --camera 1
"""

import json
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.posture_agent import PostureAgent, PostureFeatures
from src.data.extract_keypoints import KeypointExtractor, draw_skeleton, extract_from_frame
from src.data.label_postures import compute_features
from src.models.inference import ModelLoader, _keypoints_to_features

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoachPosture AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .score-box {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d5a8e;
    }
    .score-value {
        font-size: 72px;
        font-weight: bold;
        color: #00e5ff;
        line-height: 1;
    }
    .alert-box {
        background-color: #3d0000;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 12px;
        color: #ff6666;
    }
    .good-box {
        background-color: #003d00;
        border: 2px solid #44ff44;
        border-radius: 8px;
        padding: 12px;
        color: #66ff66;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 10px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialisation session state ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "score_history": deque(maxlen=300),
        "alert_history": [],
        "last_recommendation": None,
        "last_rec_time": 0.0,
        "bad_posture_start": None,
        "alert_duration": 0.0,
        "running": False,
        "frame_rgb": None,
        "current_features": None,
        "current_score": 50.0,
        "current_label": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    camera_id = st.number_input("Index caméra", min_value=0, max_value=10, value=0)
    model_choice = st.selectbox(
        "Modèle d'inférence",
        ["auto", "autoencoder", "rf", "svm"],
        help="auto = autoencoder si disponible, sinon RF",
    )
    ollama_model = st.selectbox(
        "Modèle Ollama (agent)",
        ["qwen2.5:7b", "mistral:latest", "llama3:latest", "qwen3.5:cloud"],
        index=0,
    )
    rec_interval = st.slider(
        "Intervalle recommandations (sec)",
        min_value=30, max_value=300, value=60,
        help="Délai minimum entre deux appels à l'agent LLM",
    )
    alert_threshold = st.slider(
        "Seuil alerte (sec mauvaise posture)",
        min_value=5, max_value=60, value=10,
    )
    show_features = st.checkbox("Afficher les features géométriques", value=True)

    st.divider()
    st.markdown("**Légende score:**")
    st.markdown("🟢 80-100 : Excellente")
    st.markdown("🟡 60-79 : Acceptable")
    st.markdown("🟠 40-59 : À améliorer")
    st.markdown("🔴 0-39 : Mauvaise")

    st.divider()
    if st.button("🗑️ Réinitialiser historique"):
        st.session_state.score_history = deque(maxlen=300)
        st.session_state.alert_history = []
        st.session_state.last_recommendation = None
        st.rerun()


# ── Layout principal ──────────────────────────────────────────────────────────

st.title("🏋️ CoachPosture AI — Analyse Posturale en Temps Réel")
st.caption(f"Serveur local • Ollama {ollama_model} • MediaPipe 33 keypoints")

col_video, col_metrics = st.columns([3, 2], gap="medium")

with col_video:
    video_placeholder = st.empty()
    start_col, stop_col = st.columns(2)
    with start_col:
        start_btn = st.button("▶️ Démarrer l'analyse", type="primary", use_container_width=True)
    with stop_col:
        stop_btn = st.button("⏹ Arrêter", use_container_width=True)

with col_metrics:
    score_placeholder = st.empty()
    alert_placeholder = st.empty()
    rec_placeholder = st.empty()

# Graphique historique score
st.divider()
col_hist, col_feats = st.columns([3, 2])
with col_hist:
    st.subheader("📈 Historique du score posture")
    history_chart = st.empty()

with col_feats:
    if show_features:
        st.subheader("📐 Mesures géométriques")
        features_placeholder = st.empty()

st.divider()
st.subheader("🔔 Historique des alertes")
alerts_table = st.empty()


# ── Fonctions de rendu ────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 80:
        return "#00e676"
    elif score >= 60:
        return "#ffeb3b"
    elif score >= 40:
        return "#ff9800"
    return "#f44336"


def _render_score(score: float, label: int, alert_sec: float) -> None:
    color = _score_color(score)
    box_class = "good-box" if label == 1 else "alert-box"
    status = "✅ POSTURE CORRECTE" if label == 1 else "⚠️ POSTURE ANORMALE"

    score_placeholder.markdown(f"""
<div class="score-box">
    <div style="color: #aaa; font-size: 14px;">Score posture</div>
    <div class="score-value" style="color: {color};">{score:.0f}</div>
    <div style="color: #aaa; font-size: 13px;">/100</div>
</div>
<br>
<div class="{box_class}">
    <strong>{status}</strong><br>
    {"Durée: " + f"{alert_sec:.0f}s" if alert_sec > 0 and label == 0 else "Continuez ainsi !"}
</div>
""", unsafe_allow_html=True)


def _render_history_chart(history: deque) -> None:
    if not history:
        history_chart.info("Démarrez l'analyse pour voir l'historique.")
        return

    scores = list(history)
    times = list(range(len(scores)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=scores,
        mode="lines",
        line=dict(color="#00e5ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 255, 0.1)",
        name="Score posture",
    ))
    fig.add_hline(y=60, line_dash="dash", line_color="#ffeb3b",
                  annotation_text="Seuil acceptable (60)")
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#ffffff",
        xaxis=dict(showgrid=False, title="Frames"),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1a1a2e", title="Score"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
    )
    history_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_features(features_dict: dict) -> None:
    if not features_dict:
        return
    rows = []
    icons = {
        "angle_dos": "↔️", "angle_tete": "👤", "symetrie_epaules": "⚖️",
        "inclinaison_tronc": "📐", "angle_cou": "🦴", "ratio_epaules_hanches": "📊",
    }
    thresholds = {
        "angle_dos": 10, "angle_tete": 15, "symetrie_epaules": 3,
        "inclinaison_tronc": 10, "angle_cou": 20,
    }
    for key, val in features_dict.items():
        if isinstance(val, float) and not np.isnan(val):
            icon = icons.get(key, "•")
            thresh = thresholds.get(key)
            status = ""
            if thresh:
                status = "✅" if abs(val) <= thresh else "⚠️"
            rows.append({"Mesure": f"{icon} {key.replace('_', ' ').title()}",
                        "Valeur": f"{val:.1f}°", "": status})
    if rows:
        features_placeholder.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True
        )


def _render_recommendation(rec) -> None:
    if rec is None:
        rec_placeholder.info("L'agent analysera votre posture et donnera des conseils personnalisés.")
        return
    rec_placeholder.markdown(f"""
**🧠 Recommandation IA** *(via {rec.model_utilise})*

**Problème:** {rec.probleme_principal}

**Conseil:** {rec.recommandation}

**Exercice:** {rec.exercice_suggere}

<small>Généré à {rec.timestamp[11:19]}</small>
""", unsafe_allow_html=True)


def _render_alerts_table() -> None:
    alerts = st.session_state.alert_history
    if not alerts:
        alerts_table.info("Aucune alerte pour le moment.")
        return
    df = pd.DataFrame(alerts[-10:])  # Les 10 dernières
    df = df.sort_values("timestamp", ascending=False)
    alerts_table.dataframe(df, hide_index=True, use_container_width=True)


# ── Boucle d'analyse ──────────────────────────────────────────────────────────

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error(f"Impossible d'ouvrir la caméra {camera_id}")
        st.session_state.running = False
    else:
        extractor = KeypointExtractor(min_detection_confidence=0.6)
        loader = ModelLoader(model_choice)
        agent = PostureAgent(model=ollama_model)

        score_buffer = deque(maxlen=15)  # Lissage score

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Flux caméra interrompu.")
                    break

                landmarks = extract_from_frame(frame, extractor)

                score = 50.0
                label = 1
                features_dict = {}

                if landmarks is not None:
                    features_vec, raw_score = _keypoints_to_features(landmarks)
                    score_buffer.append(raw_score)
                    score = float(np.mean(score_buffer))

                    label, confidence = loader.predict(features_vec)
                    frame = draw_skeleton(frame, landmarks)

                    # Features pour affichage
                    from src.data.extract_keypoints import LANDMARK_NAMES
                    row_dict = {
                        f"{LANDMARK_NAMES[i]}_{c}": float(landmarks[i, j])
                        for i in range(len(LANDMARK_NAMES))
                        for j, c in enumerate(("x", "y", "z", "visibility"))
                    }
                    features_dict = compute_features(pd.Series(row_dict))
                    features_dict.pop("score_posture", None)

                # Gestion alerte durée
                now = time.time()
                if label == 0:
                    if st.session_state.bad_posture_start is None:
                        st.session_state.bad_posture_start = now
                    st.session_state.alert_duration = now - st.session_state.bad_posture_start

                    # Enregistrement alerte
                    if st.session_state.alert_duration >= alert_threshold:
                        if (not st.session_state.alert_history or
                                now - st.session_state.alert_history[-1].get("_ts", 0) > alert_threshold):
                            st.session_state.alert_history.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "score": f"{score:.0f}",
                                "duree_sec": f"{st.session_state.alert_duration:.0f}",
                                "_ts": now,
                            })
                else:
                    st.session_state.bad_posture_start = None
                    st.session_state.alert_duration = 0.0

                # Recommandation LLM (throttlée)
                should_recommend = (
                    label == 0
                    and st.session_state.alert_duration >= alert_threshold
                    and now - st.session_state.last_rec_time > rec_interval
                )
                if should_recommend:
                    pf = PostureFeatures(
                        angle_dos=float(features_dict.get("angle_dos", 0)),
                        angle_tete=float(features_dict.get("angle_tete", 0)),
                        symetrie_epaules=float(features_dict.get("symetrie_epaules", 0)),
                        inclinaison_tronc=float(features_dict.get("inclinaison_tronc", 0)),
                        angle_cou=float(features_dict.get("angle_cou", 0)),
                        score_posture=score,
                    )
                    # Appel asynchrone pour ne pas bloquer le flux vidéo
                    def _get_rec():
                        rec = agent.recommend(pf, st.session_state.alert_duration)
                        st.session_state.last_recommendation = rec
                        st.session_state.last_rec_time = time.time()
                    threading.Thread(target=_get_rec, daemon=True).start()

                # Mise à jour état
                st.session_state.score_history.append(score)

                # Affichage vidéo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Rendu métriques
                _render_score(score, label, st.session_state.alert_duration)
                _render_history_chart(st.session_state.score_history)
                if show_features:
                    _render_features(features_dict)
                _render_recommendation(st.session_state.last_recommendation)
                _render_alerts_table()

                time.sleep(0.033)  # ~30 FPS

        finally:
            cap.release()
            extractor.close()
            agent.close()
            video_placeholder.info("Analyse arrêtée.")

else:
    # État d'attente
    video_placeholder.markdown("""
<div style="background:#111; border:2px dashed #333; border-radius:12px;
            height:400px; display:flex; align-items:center; justify-content:center;
            flex-direction:column; color:#666;">
    <div style="font-size:64px;">📷</div>
    <div style="font-size:18px; margin-top:16px;">Cliquez sur "Démarrer l'analyse"</div>
    <div style="font-size:13px; margin-top:8px;">Caméra {camera_id} • MediaPipe + {model_choice}</div>
</div>
""".format(camera_id=camera_id, model_choice=model_choice), unsafe_allow_html=True)

    _render_score(st.session_state.current_score, 1, 0)
    _render_history_chart(st.session_state.score_history)
    if show_features:
        features_placeholder.info("Les features s'affichent pendant l'analyse.")
    _render_recommendation(st.session_state.last_recommendation)
    _render_alerts_table()
