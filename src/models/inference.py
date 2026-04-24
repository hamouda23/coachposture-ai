"""
Inférence en temps réel via webcam ordinaire (cv2.VideoCapture).

Pipeline :
  1. Capture frame webcam
  2. Extraction keypoints MediaPipe
  3. Calcul features géométriques
  4. Prédiction par le meilleur modèle disponible (autoencoder > RF > SVM)
  5. Affichage score posture + squelette + alerte si mauvaise posture > 10 s

Mode headless disponible pour usage en serveur (--no-display).

Utilisation :
    python src/models/inference.py
    python src/models/inference.py --camera 0 --model autoencoder
    python src/models/inference.py --no-display --output results/session.mp4
"""

import argparse
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "data" / "processed" / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Import interne ────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from src.data.extract_keypoints import KeypointExtractor, draw_skeleton
from src.data.label_postures import compute_features
from src.models.train import PostureAutoencoder

# ── Couleurs BGR ──────────────────────────────────────────────────────────────
COLOR_GOOD = (0, 220, 0)        # Vert
COLOR_BAD = (0, 0, 220)         # Rouge
COLOR_WARN = (0, 165, 255)      # Orange
COLOR_TEXT = (255, 255, 255)    # Blanc
COLOR_BG = (30, 30, 30)         # Fond sombre

# ── Seuils alertes ───────────────────────────────────────────────────────────
ALERT_DURATION_SEC = 10.0       # Alerte si mauvaise posture > 10 secondes
GOOD_POSTURE_SCORE = 60.0       # Score minimum pour posture correcte


# ── Chargement des modèles ────────────────────────────────────────────────────

class ModelLoader:
    """Charge et encapsule les modèles entraînés pour l'inférence."""

    def __init__(self, model_name: str = "auto") -> None:
        self.model_name = model_name
        self.scaler = None
        self.rf = None
        self.autoencoder = None
        self.ae_threshold: Optional[float] = None
        self.svm = None
        self.active_model: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_scaler()
        self._load_models(model_name)

    def _load_scaler(self) -> None:
        path = MODELS_DIR / "scaler.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"  Scaler chargé: {path}")

    def _load_models(self, model_name: str) -> None:
        # Autoencoder
        ae_path = MODELS_DIR / "autoencoder.pt"
        if ae_path.exists() and model_name in ("auto", "autoencoder"):
            try:
                checkpoint = torch.load(ae_path, map_location=self.device)
                self.autoencoder = PostureAutoencoder(checkpoint["input_dim"]).to(self.device)
                self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
                self.autoencoder.eval()
                self.ae_threshold = checkpoint["threshold"]
                self.active_model = "autoencoder"
                logger.info(f"  Autoencoder chargé (device: {self.device})")
            except Exception as e:
                logger.warning(f"  Autoencoder: {e}")

        # Random Forest (fallback)
        rf_path = MODELS_DIR / "random_forest.pkl"
        if rf_path.exists() and model_name in ("auto", "rf"):
            try:
                with open(rf_path, "rb") as f:
                    data = pickle.load(f)
                self.rf = data["model"]
                if self.active_model is None:
                    self.active_model = "rf"
                logger.info("  Random Forest chargé")
            except Exception as e:
                logger.warning(f"  RF: {e}")

        # One-Class SVM (second fallback)
        svm_path = MODELS_DIR / "one_class_svm.pkl"
        if svm_path.exists() and model_name in ("auto", "svm"):
            try:
                with open(svm_path, "rb") as f:
                    data = pickle.load(f)
                self.svm = data["model"]
                if self.active_model is None:
                    self.active_model = "svm"
                logger.info("  One-Class SVM chargé")
            except Exception as e:
                logger.warning(f"  SVM: {e}")

        if self.active_model is None:
            logger.warning("Aucun modèle chargé — mode score géométrique seul activé.")

    def predict(self, features_raw: np.ndarray) -> Tuple[int, float]:
        """
        Prédit la qualité de posture.
        Retourne (label: 0|1, confiance: 0–1).
        """
        if features_raw is None or np.any(np.isnan(features_raw)):
            return 1, 0.5  # Pas de prédiction possible

        X = features_raw.reshape(1, -1).astype(np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.active_model == "autoencoder" and self.autoencoder is not None:
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                error = self.autoencoder.reconstruction_error(X_t).item()
            label = 1 if error <= self.ae_threshold else 0
            confidence = 1.0 - min(error / (self.ae_threshold * 2), 1.0)
            return label, confidence

        if self.active_model == "rf" and self.rf is not None:
            label = int(self.rf.predict(X)[0])
            confidence = float(self.rf.predict_proba(X)[0, label])
            return label, confidence

        if self.active_model == "svm" and self.svm is not None:
            raw = self.svm.predict(X)[0]
            label = 1 if raw == 1 else 0
            score = float(self.svm.decision_function(X)[0])
            confidence = float(1.0 / (1.0 + np.exp(-score)))  # Sigmoid
            return label, confidence

        return 1, 0.5


# ── Calcul features depuis keypoints bruts ────────────────────────────────────

def _keypoints_to_features(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Convertit les landmarks MediaPipe (33, 4) en vecteur de features
    compatible avec le scaler/modèle entraîné.
    """
    import pandas as pd
    from src.data.extract_keypoints import LANDMARK_NAMES

    row_dict: dict = {}
    for i, name in enumerate(LANDMARK_NAMES):
        row_dict[f"{name}_x"] = float(landmarks[i, 0])
        row_dict[f"{name}_y"] = float(landmarks[i, 1])
        row_dict[f"{name}_z"] = float(landmarks[i, 2])
        row_dict[f"{name}_visibility"] = float(landmarks[i, 3])

    row = pd.Series(row_dict)
    geo_features = compute_features(row)

    # Features géométriques
    geo_vals = [
        geo_features.get("angle_dos", np.nan),
        geo_features.get("angle_tete", np.nan),
        geo_features.get("symetrie_epaules", np.nan),
        geo_features.get("inclinaison_tronc", np.nan),
        geo_features.get("angle_cou", np.nan),
        geo_features.get("ratio_epaules_hanches", np.nan),
    ]

    # Features keypoints bruts
    kp_lms = ["nose", "left_shoulder", "right_shoulder",
               "left_hip", "right_hip", "left_ear", "right_ear"]
    kp_vals = [row_dict.get(f"{lm}_{coord}", np.nan)
               for lm in kp_lms for coord in ("x", "y")]

    features = np.array(geo_vals + kp_vals, dtype=np.float32)
    score = geo_features.get("score_posture", 50.0)
    return features, score


# ── Overlay UI ────────────────────────────────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    score: float,
    label: int,
    confidence: float,
    alert_seconds: float,
    model_name: str,
    features: Optional[dict] = None,
) -> np.ndarray:
    """Dessine l'interface utilisateur sur la frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Bande supérieure
    cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_BG, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Score posture (grand)
    color = COLOR_GOOD if score >= GOOD_POSTURE_SCORE else (COLOR_WARN if score >= 40 else COLOR_BAD)
    cv2.putText(frame, f"Posture: {score:.0f}/100",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    # Modèle + confiance
    cv2.putText(frame, f"[{model_name}] conf={confidence:.0%}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # Barre de progression posture
    bar_x, bar_y, bar_w, bar_h = w - 220, 15, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.putText(frame, "Score", (bar_x, bar_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

    # Alerte durée
    if alert_seconds > 0 and label == 0:
        alert_color = COLOR_BAD if alert_seconds > ALERT_DURATION_SEC else COLOR_WARN
        cv2.putText(frame, f"Mauvaise posture: {alert_seconds:.0f}s",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)
        if alert_seconds > ALERT_DURATION_SEC:
            # Bordure rouge clignotante
            thick = int(5 + 3 * np.sin(time.time() * 4))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_BAD, thick)
            cv2.putText(frame, "CORRIGEZ VOTRE POSTURE !",
                        (w // 2 - 220, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, COLOR_BAD, 2)

    # Détails features (coin bas-gauche)
    if features:
        y_pos = h - 150
        for key, val in list(features.items())[:5]:
            if isinstance(val, float) and not np.isnan(val):
                cv2.putText(frame, f"{key}: {val:.1f}",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
                y_pos += 18

    return frame


# ── Boucle principale ─────────────────────────────────────────────────────────

def run_inference(
    camera_id: int = 0,
    model_name: str = "auto",
    show_display: bool = True,
    output_path: Optional[str] = None,
) -> None:
    """Boucle d'inférence en temps réel sur flux webcam."""
    logger.info(f"Initialisation caméra {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(f"Impossible d'ouvrir la caméra {camera_id}")
        sys.exit(1)

    # Paramètres de capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_actual = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Enregistrement vidéo optionnel
    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_actual, (w, h))

    logger.info(f"Résolution: {w}×{h} @ {fps_actual:.0f} fps")
    logger.info("Chargement modèles...")
    loader = ModelLoader(model_name)
    extractor = KeypointExtractor(min_detection_confidence=0.6)

    logger.info("Initialisation MediaPipe...")

    # État de l'alerte
    bad_posture_start: Optional[float] = None
    alert_seconds: float = 0.0
    score_history: deque = deque(maxlen=30)  # Historique FPS lissage score

    logger.success("Inférence démarrée. Appuyez sur 'q' pour quitter.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Fin du flux caméra.")
                break

            # Extraction keypoints
            from src.data.extract_keypoints import extract_from_frame
            landmarks = extract_from_frame(frame, extractor)

            score = 50.0
            label = 1
            confidence = 0.5
            features_dict: Optional[dict] = None

            if landmarks is not None:
                features_vec, score = _keypoints_to_features(landmarks)
                label, confidence = loader.predict(features_vec)
                score_history.append(score)
                score = float(np.mean(score_history))  # Score lissé

                # Détails pour affichage
                import pandas as pd
                from src.data.extract_keypoints import LANDMARK_NAMES
                row_dict = {
                    f"{LANDMARK_NAMES[i]}_{c}": float(landmarks[i, j])
                    for i, name in enumerate(LANDMARK_NAMES)
                    for j, c in enumerate(("x", "y", "z", "visibility"))
                }
                import pandas as pd
                features_dict = compute_features(pd.Series(row_dict))
                features_dict.pop("score_posture", None)

                # Dessin squelette
                frame = draw_skeleton(frame, landmarks)

            # Gestion alerte durée
            now = time.time()
            if label == 0:
                if bad_posture_start is None:
                    bad_posture_start = now
                alert_seconds = now - bad_posture_start
            else:
                bad_posture_start = None
                alert_seconds = 0.0

            # HUD
            frame = _draw_hud(
                frame, score, label, confidence,
                alert_seconds, loader.active_model or "score",
                features_dict,
            )

            if writer:
                writer.write(frame)

            if show_display:
                cv2.imshow("CoachPosture AI — Analyse posturale", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # Capture d'écran manuelle
                    snap_path = RESULTS_DIR / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    logger.info(f"Snapshot: {snap_path}")

    except KeyboardInterrupt:
        logger.info("Inférence interrompue.")
    finally:
        cap.release()
        extractor.close()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        logger.info("Ressources libérées.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inférence temps réel CoachPosture AI")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index caméra (défaut: 0) — compatible Yahboom")
    parser.add_argument("--model", choices=["auto", "autoencoder", "rf", "svm"],
                        default="auto", help="Modèle à utiliser")
    parser.add_argument("--no-display", action="store_true",
                        help="Désactiver l'affichage (mode serveur headless)")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin de sortie vidéo MP4")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
               level="INFO")

    run_inference(
        camera_id=args.camera,
        model_name=args.model,
        show_display=not args.no_display,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
