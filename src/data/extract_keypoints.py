"""
Extraction des 33 keypoints MediaPipe Pose depuis les images des datasets.

Pour chaque image valide, MediaPipe extrait 33 landmarks 3D (x, y, z, visibility).
Les coordonnées sont normalisées [0,1] par rapport aux dimensions de l'image.
Le résultat est sauvegardé dans un CSV par dataset dans data/processed/.

Utilisation :
    python src/data/extract_keypoints.py --dataset lsp
    python src/data/extract_keypoints.py --dataset all --workers 4
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Noms des 33 landmarks MediaPipe Pose ────────────────────────────────────
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Colonnes CSV : pour chaque landmark → x, y, z, visibility
CSV_COLUMNS = ["image_path"] + [
    f"{name}_{coord}"
    for name in LANDMARK_NAMES
    for coord in ("x", "y", "z", "visibility")
] + ["extraction_success"]


# ── Extracteur MediaPipe ─────────────────────────────────────────────────────

class KeypointExtractor:
    """Wrapper MediaPipe Pose pour extraction batch d'images."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 2,
    ) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,       # Images statiques, pas vidéo
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extrait les 33 landmarks depuis une image.
        Retourne un array (33, 4) [x, y, z, visibility] ou None si échec.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # MediaPipe attend du RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)
        return landmarks

    def close(self) -> None:
        self.pose.close()


# ── Construction de la ligne CSV ─────────────────────────────────────────────

def _landmarks_to_row(image_path: Path, landmarks: Optional[np.ndarray]) -> dict:
    """Convertit les landmarks en dict pour le DataFrame."""
    row: dict = {"image_path": str(image_path), "extraction_success": landmarks is not None}

    if landmarks is not None:
        for i, name in enumerate(LANDMARK_NAMES):
            row[f"{name}_x"] = float(landmarks[i, 0])
            row[f"{name}_y"] = float(landmarks[i, 1])
            row[f"{name}_z"] = float(landmarks[i, 2])
            row[f"{name}_visibility"] = float(landmarks[i, 3])
    else:
        # Valeurs manquantes si pas de détection
        for name in LANDMARK_NAMES:
            for coord in ("x", "y", "z", "visibility"):
                row[f"{name}_{coord}"] = np.nan

    return row


# ── Collecte des images par dataset ─────────────────────────────────────────

def _collect_images(dataset_dir: Path) -> list[Path]:
    """Collecte toutes les images .jpg/.png/.jpeg d'un dossier."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(images)


# ── Traitement d'un dataset ──────────────────────────────────────────────────

def process_dataset(dataset_name: str, workers: int = 1) -> Path:
    """
    Extrait les keypoints de toutes les images d'un dataset.
    Sauvegarde le CSV dans data/processed/{dataset_name}_keypoints.csv.
    """
    dataset_dir = RAW_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_dir}")

    output_csv = PROCESSED_DIR / f"{dataset_name}_keypoints.csv"

    # Reprise si déjà partiellement traité
    already_done: set[str] = set()
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, usecols=["image_path"])
        already_done = set(existing_df["image_path"].tolist())
        logger.info(f"  Reprise: {len(already_done)} images déjà extraites.")

    images = _collect_images(dataset_dir)
    images_todo = [img for img in images if str(img) not in already_done]

    logger.info(f"  {len(images)} images trouvées, {len(images_todo)} à traiter.")

    if not images_todo:
        logger.info(f"  Rien à faire pour {dataset_name}.")
        return output_csv

    rows: list[dict] = []
    extractor = KeypointExtractor(model_complexity=1)  # Complexité 1 = bon compromis vitesse/qualité

    try:
        for img_path in tqdm(images_todo, desc=f"Extraction {dataset_name}"):
            landmarks = extractor.extract(img_path)
            rows.append(_landmarks_to_row(img_path, landmarks))

            # Sauvegarde par batch de 500 pour éviter les pertes
            if len(rows) >= 500:
                _append_csv(rows, output_csv)
                rows = []
    finally:
        extractor.close()

    if rows:
        _append_csv(rows, output_csv)

    df = pd.read_csv(output_csv)
    success_rate = df["extraction_success"].mean() * 100
    logger.success(
        f"  {dataset_name}: {len(df)} lignes, "
        f"taux de détection = {success_rate:.1f}%"
    )
    return output_csv


def _append_csv(rows: list[dict], output_csv: Path) -> None:
    """Ajoute des lignes au CSV (crée le fichier si nécessaire)."""
    df_new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if output_csv.exists():
        df_new.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(output_csv, mode="w", header=True, index=False)


# ── Traitement webcam (image unique) ────────────────────────────────────────

def extract_from_frame(frame: np.ndarray, extractor: KeypointExtractor) -> Optional[np.ndarray]:
    """
    Extrait les keypoints depuis une frame OpenCV (BGR).
    Retourne un array (33, 4) ou None.
    Exposé pour l'utilisation en temps réel dans inference.py.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = extractor.pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None
    return np.array([
        [lm.x, lm.y, lm.z, lm.visibility]
        for lm in results.pose_landmarks.landmark
    ], dtype=np.float32)


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Dessine le squelette MediaPipe sur une frame OpenCV.
    landmarks : array (33, 4) [x, y, z, visibility], coordonnées normalisées.
    """
    h, w = frame.shape[:2]
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Reconstruire un objet NormalizedLandmarkList pour mp_drawing
    landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
    for lm_data in landmarks:
        lm = landmark_list.landmark.add()
        lm.x, lm.y, lm.z, lm.visibility = float(lm_data[0]), float(lm_data[1]), float(lm_data[2]), float(lm_data[3])

    annotated = frame.copy()
    mp_drawing.draw_landmarks(
        annotated,
        landmark_list,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
    )
    return annotated


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction keypoints MediaPipe")
    parser.add_argument("--dataset", default="all",
                        help="Nom du dataset dans data/raw/ ou 'all' pour tous")
    parser.add_argument("--workers", type=int, default=1,
                        help="Nombre de workers parallèles (défaut 1 — MediaPipe préfère mono-processus)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    if args.dataset == "all":
        datasets = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    else:
        datasets = [args.dataset]

    if not datasets:
        logger.error(f"Aucun dataset trouvé dans {RAW_DIR}. Lancez d'abord download_datasets.py.")
        sys.exit(1)

    for ds in datasets:
        logger.info(f"\n=== Traitement: {ds} ===")
        try:
            process_dataset(ds, args.workers)
        except FileNotFoundError as e:
            logger.error(str(e))


if __name__ == "__main__":
    main()
