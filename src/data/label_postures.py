"""
Labellisation automatique des postures à partir des keypoints MediaPipe.

Calcul des features géométriques :
  - Angle du dos (épaules → hanches, par rapport à la verticale)
  - Angle de la tête (nez → milieu épaules, par rapport à la verticale)
  - Symétrie des épaules (différence de hauteur gauche/droite en %)
  - Inclinaison du tronc (vecteur hanches → épaules vs verticale)
  - Score global de posture [0–100]

Règles de labellisation :
  posture_correcte = 1  si score ≥ 60
  posture_anomale  = 0  si score < 60

Utilisation :
    python src/data/label_postures.py
    python src/data/label_postures.py --threshold 70 --input data/processed/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
LABELED_DIR = ROOT / "data" / "labeled"
RESULTS_DIR = ROOT / "results"
LABELED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Indices MediaPipe ─────────────────────────────────────────────────────────
IDX = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ear": 7, "right_ear": 8,
}


# ── Calcul des angles ────────────────────────────────────────────────────────

def _angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle en degrés au sommet B formé par les vecteurs BA et BC.
    Tous les points sont (x, y).
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_with_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle en degrés entre le vecteur p1→p2 et la verticale (0, -1)."""
    vec = p2 - p1
    vertical = np.array([0.0, -1.0])
    cos_a = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-8)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ── Extraction des features depuis une ligne CSV ─────────────────────────────

def _get_xy(row: pd.Series, landmark: str) -> np.ndarray:
    """Retourne (x, y) normalisés pour un landmark."""
    return np.array([row[f"{landmark}_x"], row[f"{landmark}_y"]], dtype=np.float32)


def compute_features(row: pd.Series) -> dict[str, float]:
    """
    Calcule toutes les features géométriques depuis une ligne de keypoints.
    Retourne un dict avec les features + score + label.
    """
    features: dict[str, float] = {}

    try:
        nose = _get_xy(row, "nose")
        l_shoulder = _get_xy(row, "left_shoulder")
        r_shoulder = _get_xy(row, "right_shoulder")
        l_hip = _get_xy(row, "left_hip")
        r_hip = _get_xy(row, "right_hip")
        l_ear = _get_xy(row, "left_ear")
        r_ear = _get_xy(row, "right_ear")

        # Milieux
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0
        mid_ear = (l_ear + r_ear) / 2.0

        # ── Feature 1 : Angle du dos (milieu épaules → milieu hanches vs verticale) ──
        # Idéalement ~0° (dos droit). Pénalité si > 20°.
        back_angle = _angle_with_vertical(mid_shoulder, mid_hip)
        features["angle_dos"] = back_angle

        # ── Feature 2 : Angle de la tête (nez → milieu épaules vs verticale) ──
        # Tête droite ≈ 0°. Tête penchée en avant (travail écran) > 20°.
        head_angle = _angle_with_vertical(mid_shoulder, nose)
        features["angle_tete"] = head_angle

        # ── Feature 3 : Symétrie des épaules (différence Y normalisée) ──
        # Épaules symétriques → faible valeur absolue. Asymétrie > 5% = problème.
        shoulder_symmetry = abs(float(l_shoulder[1]) - float(r_shoulder[1]))
        features["symetrie_epaules"] = shoulder_symmetry * 100.0  # En % de la hauteur image

        # ── Feature 4 : Inclinaison du tronc (vecteur hanche→épaule vs verticale) ──
        trunk_tilt = _angle_with_vertical(mid_hip, mid_shoulder)
        features["inclinaison_tronc"] = trunk_tilt

        # ── Feature 5 : Angle cou (oreilles → épaules vs verticale) ──
        neck_angle = _angle_with_vertical(mid_shoulder, mid_ear)
        features["angle_cou"] = neck_angle

        # ── Feature 6 : Largeur épaules normalisée (santé musculaire) ──
        shoulder_width = float(np.linalg.norm(l_shoulder - r_shoulder))
        hip_width = float(np.linalg.norm(l_hip - r_hip))
        # Ratio épaules/hanches > 0.5 = posture ouverte (saine)
        features["ratio_epaules_hanches"] = shoulder_width / (hip_width + 1e-8)

        # ── Score global [0–100] ─────────────────────────────────────────────
        # Pénalités cumulatives normalisées
        penalty_back = min(back_angle / 45.0, 1.0) * 30.0        # Max 30 pts
        penalty_head = min(head_angle / 45.0, 1.0) * 25.0        # Max 25 pts
        penalty_sym = min(shoulder_symmetry * 100 / 10.0, 1.0) * 20.0   # Max 20 pts
        penalty_trunk = min(trunk_tilt / 30.0, 1.0) * 15.0       # Max 15 pts
        penalty_neck = min(neck_angle / 40.0, 1.0) * 10.0        # Max 10 pts

        total_penalty = penalty_back + penalty_head + penalty_sym + penalty_trunk + penalty_neck
        score = max(0.0, 100.0 - total_penalty)
        features["score_posture"] = round(score, 2)

    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.debug(f"Erreur calcul features: {exc}")
        for key in ("angle_dos", "angle_tete", "symetrie_epaules",
                    "inclinaison_tronc", "angle_cou", "ratio_epaules_hanches", "score_posture"):
            features[key] = float("nan")

    return features


# ── Labellisation d'un DataFrame ─────────────────────────────────────────────

def label_dataframe(df: pd.DataFrame, score_threshold: float = 60.0) -> pd.DataFrame:
    """
    Ajoute les features géométriques et le label binaire au DataFrame.
    Filtre les lignes sans détection.
    """
    # Garder uniquement les lignes avec détection réussie
    df_valid = df[df["extraction_success"] == True].copy()
    logger.info(f"  Lignes valides: {len(df_valid)} / {len(df)}")

    # Calcul features
    features_list = [compute_features(row) for _, row in df_valid.iterrows()]
    features_df = pd.DataFrame(features_list)

    # Fusion
    df_result = pd.concat([df_valid.reset_index(drop=True), features_df], axis=1)

    # Supprimer les lignes avec features NaN
    feature_cols = ["angle_dos", "angle_tete", "symetrie_epaules",
                    "inclinaison_tronc", "angle_cou", "ratio_epaules_hanches"]
    df_result = df_result.dropna(subset=feature_cols)

    # Label binaire
    df_result["posture_correcte"] = (df_result["score_posture"] >= score_threshold).astype(int)

    logger.info(
        f"  Label distribution: "
        f"correcte={df_result['posture_correcte'].sum()} "
        f"({df_result['posture_correcte'].mean()*100:.1f}%), "
        f"anomale={(df_result['posture_correcte'] == 0).sum()}"
    )
    return df_result


# ── Traitement de tous les fichiers processed ────────────────────────────────

def process_all(threshold: float = 60.0) -> Path:
    """Labellise tous les CSV de data/processed/ et produit un CSV unifié."""
    csv_files = list(PROCESSED_DIR.glob("*_keypoints.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV trouvé dans {PROCESSED_DIR}. "
            "Lancez d'abord extract_keypoints.py."
        )

    labeled_frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        dataset_name = csv_path.stem.replace("_keypoints", "")
        logger.info(f"\n=== Labellisation: {dataset_name} ===")

        df = pd.read_csv(csv_path)
        df_labeled = label_dataframe(df, threshold)
        df_labeled["dataset_source"] = dataset_name

        # Sauvegarde par dataset
        out_path = LABELED_DIR / f"{dataset_name}_labeled.csv"
        df_labeled.to_csv(out_path, index=False)
        logger.success(f"  Sauvegardé: {out_path}")

        labeled_frames.append(df_labeled)

    # CSV global
    df_all = pd.concat(labeled_frames, ignore_index=True)
    unified_path = LABELED_DIR / "all_labeled.csv"
    df_all.to_csv(unified_path, index=False)
    logger.success(f"\nDataset unifié: {unified_path} ({len(df_all)} échantillons)")

    # Statistiques sauvegardées en JSON + TXT
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": int(len(df_all)),
        "posture_correcte": int(df_all["posture_correcte"].sum()),
        "posture_anomale": int((df_all["posture_correcte"] == 0).sum()),
        "ratio_correct": float(df_all["posture_correcte"].mean()),
        "score_moyen": float(df_all["score_posture"].mean()),
        "threshold_utilise": threshold,
        "datasets": [df["dataset_source"].iloc[0] for df in labeled_frames if len(df) > 0],
    }

    json_path = RESULTS_DIR / "labeling_stats.json"
    txt_path = RESULTS_DIR / "labeling_stats.txt"

    # Append JSON
    existing: list = []
    if json_path.exists():
        import json
        with open(json_path) as f:
            existing = json.load(f)
    existing.append(stats)
    with open(json_path, "w") as f:
        import json
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w") as f:
        f.write(f"=== Labellisation CoachPosture AI ===\n")
        f.write(f"Date: {stats['timestamp']}\n")
        f.write(f"Total: {stats['total_samples']}\n")
        f.write(f"Correct: {stats['posture_correcte']} ({stats['ratio_correct']*100:.1f}%)\n")
        f.write(f"Anomale: {stats['posture_anomale']}\n")
        f.write(f"Score moyen: {stats['score_moyen']:.1f}/100\n")

    logger.success(f"Stats sauvegardées: {json_path}, {txt_path}")
    return unified_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Labellisation des postures")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Score minimum pour posture correcte (défaut: 60)")
    parser.add_argument("--input", type=str, default=str(PROCESSED_DIR),
                        help="Dossier contenant les CSV de keypoints")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    global PROCESSED_DIR
    PROCESSED_DIR = Path(args.input)

    process_all(args.threshold)


if __name__ == "__main__":
    main()
