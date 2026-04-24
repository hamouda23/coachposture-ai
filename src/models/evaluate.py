"""
Évaluation comparative des 3 modèles entraînés.

Génère :
  - Matrices de confusion
  - Courbes ROC
  - Rapport de classification complet
  - Comparaison des métriques en tableau
  - Sauvegarde résultats en JSON + TXT

Utilisation :
    python src/models/evaluate.py
    python src/models/evaluate.py --model autoencoder
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # Pas d'affichage X11 sur serveur headless
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = ROOT / "data" / "labeled"
MODELS_DIR = ROOT / "data" / "processed" / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Features — doit correspondre à train.py
FEATURE_COLS = [
    "angle_dos", "angle_tete", "symetrie_epaules",
    "inclinaison_tronc", "angle_cou", "ratio_epaules_hanches",
]
KEYPOINT_FEATURES = [
    f"{lm}_{coord}"
    for lm in [
        "nose", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_ear", "right_ear",
    ]
    for coord in ("x", "y")
]


# ── Chargement des modèles ────────────────────────────────────────────────────

def _load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler introuvable: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_rf():
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"]


def _load_autoencoder(input_dim: int):
    path = MODELS_DIR / "autoencoder.pt"
    if not path.exists():
        return None, None
    checkpoint = torch.load(path, map_location="cpu")
    from src.models.train import PostureAutoencoder
    model = PostureAutoencoder(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["threshold"]


def _load_svm():
    path = MODELS_DIR / "one_class_svm.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"]


# ── Chargement des données ────────────────────────────────────────────────────

def load_test_data():
    csv_path = LABELED_DIR / "all_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = [c for c in FEATURE_COLS + KEYPOINT_FEATURES if c in df.columns]
    df = df.dropna(subset=cols + ["posture_correcte"])

    X = df[cols].values.astype(np.float32)
    y = df["posture_correcte"].values.astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


# ── Évaluation d'un modèle ───────────────────────────────────────────────────

def evaluate_rf(X_test_sc: np.ndarray, y_test: np.ndarray, rf) -> dict:
    y_pred = rf.predict(X_test_sc)
    y_scores = rf.predict_proba(X_test_sc)[:, 1]
    return _build_report("RandomForest", y_test, y_pred, y_scores)


def evaluate_autoencoder(X_test_sc: np.ndarray, y_test: np.ndarray, model, threshold: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_t = torch.tensor(X_test_sc, dtype=torch.float32).to(device)

    with torch.no_grad():
        errors = model.reconstruction_error(X_t).cpu().numpy()

    y_pred = (errors <= threshold).astype(int)
    scores = 1.0 - np.clip(errors / (threshold * 2), 0, 1)
    return _build_report("Autoencoder", y_test, y_pred, scores)


def evaluate_svm(X_test_sc: np.ndarray, y_test: np.ndarray, svm) -> dict:
    raw = svm.predict(X_test_sc)
    y_pred = (raw == 1).astype(int)
    scores = svm.decision_function(X_test_sc)
    scores_norm = (scores - scores.min()) / (scores.ptp() + 1e-8)
    return _build_report("OneClassSVM", y_test, y_pred, scores_norm)


def _build_report(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, target_names=["Anomale", "Correcte"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = float("nan")

    logger.info(f"\n--- {name} ---")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Anomale', 'Correcte'])}")
    logger.info(f"ROC-AUC: {auc:.4f}")

    return {
        "model": name,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "y_true": y_true.tolist(),
        "y_scores": y_scores.tolist(),
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrices(reports: list[dict]) -> Path:
    """Génère une figure avec toutes les matrices de confusion."""
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, rep in zip(axes, reports):
        cm = np.array(rep["confusion_matrix"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Anomale", "Correcte"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(rep["model"])

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Matrices de confusion: {out}")
    return out


def plot_roc_curves(reports: list[dict], y_test: np.ndarray) -> Path:
    """Génère les courbes ROC superposées."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for rep in reports:
        if rep.get("roc_auc") is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, rep["y_scores"])
        ax.plot(fpr, tpr, label=f"{rep['model']} (AUC={rep['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title("Courbes ROC — CoachPosture AI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = RESULTS_DIR / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Courbes ROC: {out}")
    return out


# ── Résumé comparatif ─────────────────────────────────────────────────────────

def print_comparison_table(reports: list[dict]) -> None:
    """Affiche un tableau comparatif ASCII."""
    header = f"{'Modèle':<20} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}"
    logger.info(f"\n{'='*55}")
    logger.info(header)
    logger.info("-" * 55)
    for rep in reports:
        cr = rep["classification_report"]
        acc = cr.get("accuracy", 0)
        f1 = cr.get("weighted avg", {}).get("f1-score", 0)
        auc = rep.get("roc_auc") or "N/A"
        logger.info(f"  {rep['model']:<18} {acc:>10.4f} {f1:>10.4f} {str(auc):>10}")
    logger.info("=" * 55)


# ── Sauvegarde résultats ──────────────────────────────────────────────────────

def save_evaluation(reports: list[dict]) -> None:
    run = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": [
            {k: v for k, v in r.items() if k not in ("y_true", "y_scores")}
            for r in reports
        ],
    }

    json_path = RESULTS_DIR / "evaluation_results.json"
    txt_path = RESULTS_DIR / "evaluation_results.txt"

    existing: list = []
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    existing.append(run)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(txt_path, "a") as f:
        f.write(f"\n=== Évaluation {run['timestamp']} ===\n")
        for ev in run["evaluations"]:
            f.write(f"  {ev['model']}: roc_auc={ev.get('roc_auc')}\n")

    logger.success(f"Résultats: {json_path}, {txt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation des modèles CoachPosture AI")
    parser.add_argument("--model", choices=["all", "rf", "autoencoder", "svm"],
                        default="all")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    X_test, y_test = load_test_data()
    scaler = _load_scaler()
    X_test_sc = scaler.transform(X_test)

    reports: list[dict] = []

    if args.model in ("all", "rf"):
        rf = _load_rf()
        if rf:
            reports.append(evaluate_rf(X_test_sc, y_test, rf))
        else:
            logger.warning("Random Forest non trouvé.")

    if args.model in ("all", "autoencoder"):
        ae, threshold = _load_autoencoder(X_test.shape[1])
        if ae:
            reports.append(evaluate_autoencoder(X_test_sc, y_test, ae, threshold))
        else:
            logger.warning("Autoencoder non trouvé.")

    if args.model in ("all", "svm"):
        svm = _load_svm()
        if svm:
            reports.append(evaluate_svm(X_test_sc, y_test, svm))
        else:
            logger.warning("One-Class SVM non trouvé.")

    if not reports:
        logger.error("Aucun modèle trouvé. Lancez d'abord train.py.")
        sys.exit(1)

    print_comparison_table(reports)
    plot_confusion_matrices(reports)
    plot_roc_curves(reports, y_test)
    save_evaluation(reports)


if __name__ == "__main__":
    main()
