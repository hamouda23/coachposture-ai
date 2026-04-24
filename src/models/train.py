"""
Entraînement des 3 modèles de détection de posture anormale.

  1. Random Forest     — baseline supervisé (sklearn)
  2. Autoencoder       — détection d'anomalies non supervisée (PyTorch + CUDA)
  3. One-Class SVM     — détection d'anomalies classique (sklearn)

Chaque modèle est sauvegardé dans data/processed/models/.
Les métriques sont sauvegardées en JSON + TXT dans results/.

Utilisation :
    python src/models/train.py
    python src/models/train.py --model rf
    python src/models/train.py --model autoencoder --epochs 100
    python src/models/train.py --model all --test-size 0.2
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = ROOT / "data" / "labeled"
MODELS_DIR = ROOT / "data" / "processed" / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Features géométriques utilisées ──────────────────────────────────────────
FEATURE_COLS = [
    "angle_dos", "angle_tete", "symetrie_epaules",
    "inclinaison_tronc", "angle_cou", "ratio_epaules_hanches",
]
# Features keypoints bruts (coordonnées x,y des landmarks clés)
KEYPOINT_FEATURES = [
    f"{lm}_{coord}"
    for lm in [
        "nose", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_ear", "right_ear",
    ]
    for coord in ("x", "y")
]
TARGET_COL = "posture_correcte"


# ── Chargement des données ────────────────────────────────────────────────────

def load_data(all_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge all_labeled.csv et retourne (X, y).
    all_features=True : géométrique + keypoints bruts
    all_features=False : uniquement features géométriques
    """
    csv_path = LABELED_DIR / "all_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset labellisé introuvable: {csv_path}\n"
            "Lancez d'abord label_postures.py."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Dataset chargé: {len(df)} échantillons")

    # Sélection des features disponibles
    available_geo = [c for c in FEATURE_COLS if c in df.columns]
    available_kp = [c for c in KEYPOINT_FEATURES if c in df.columns] if all_features else []
    cols = available_geo + available_kp

    df = df.dropna(subset=cols + [TARGET_COL])
    X = df[cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)

    logger.info(f"Features: {len(cols)}, Samples: {len(X)}, Positive: {y.sum()}")
    return X, y


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── Modèle 1 : Random Forest ──────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 200,
) -> dict:
    """Entraîne un Random Forest et retourne les métriques."""
    logger.info("\n=== Random Forest ===")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",  # Gère le déséquilibre de classes
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics("RandomForest", y_test, y_pred, y_proba)

    # Sauvegarde du modèle
    model_path = MODELS_DIR / "random_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf, "feature_cols": FEATURE_COLS + KEYPOINT_FEATURES}, f)
    logger.success(f"  Modèle sauvegardé: {model_path}")

    # Importance des features
    importances = rf.feature_importances_
    metrics["feature_importances"] = dict(zip(FEATURE_COLS + KEYPOINT_FEATURES, importances.tolist()))

    return metrics


# ── Modèle 2 : Autoencoder PyTorch ───────────────────────────────────────────

class PostureAutoencoder(nn.Module):
    """
    Autoencoder pour détection d'anomalies posturales.
    Entraîné uniquement sur les postures correctes.
    L'erreur de reconstruction est le score d'anomalie.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        # Encodeur : compresse la posture en représentation latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Décodeur : reconstruit depuis l'espace latent
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE par échantillon — utilisé comme score d'anomalie."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


def train_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """Entraîne l'autoencoder sur les postures correctes uniquement."""
    logger.info("\n=== Autoencoder PyTorch ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # Entraîner seulement sur la classe positive (posture correcte)
    X_train_pos = X_train[y_train == 1]
    logger.info(f"  Train sur {len(X_train_pos)} postures correctes")

    # Datasets PyTorch
    X_train_t = torch.tensor(X_train_pos, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, X_train_t)  # autoencoder : target = input
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Modèle
    input_dim = X_train.shape[1]
    model = PostureAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # Entraînement
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= len(X_train_pos)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODELS_DIR / "autoencoder_best.pt")

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.6f}")

    # Charger le meilleur modèle
    model.load_state_dict(torch.load(MODELS_DIR / "autoencoder_best.pt", map_location=device))
    model.eval()

    # Seuil d'anomalie : percentile 95 des erreurs sur données d'entraînement
    with torch.no_grad():
        train_errors = model.reconstruction_error(X_train_t.to(device)).cpu().numpy()
        threshold = float(np.percentile(train_errors, 95))
        logger.info(f"  Seuil anomalie (P95): {threshold:.6f}")

        test_errors = model.reconstruction_error(X_test_t.to(device)).cpu().numpy()

    # Prédictions : erreur > seuil → anomalie (0), sinon correct (1)
    y_pred = (test_errors <= threshold).astype(int)
    # Scores de confiance : inversé normalisé
    y_scores = 1.0 - np.clip(test_errors / (threshold * 2), 0, 1)

    metrics = _compute_metrics("Autoencoder", y_test, y_pred, y_scores)
    metrics["threshold"] = threshold

    # Sauvegarde complète
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "threshold": threshold,
        "feature_cols": FEATURE_COLS + KEYPOINT_FEATURES,
    }, MODELS_DIR / "autoencoder.pt")
    logger.success(f"  Modèle sauvegardé: {MODELS_DIR / 'autoencoder.pt'}")

    return metrics


# ── Modèle 3 : One-Class SVM ─────────────────────────────────────────────────

def train_one_class_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    nu: float = 0.05,
) -> dict:
    """Entraîne un One-Class SVM sur les postures correctes."""
    logger.info("\n=== One-Class SVM ===")

    X_train_pos = X_train[y_train == 1]
    logger.info(f"  Train sur {len(X_train_pos)} postures correctes (nu={nu})")

    oc_svm = OneClassSVM(
        kernel="rbf",
        nu=nu,           # Fraction attendue d'outliers dans les données d'entraînement
        gamma="scale",
    )
    oc_svm.fit(X_train_pos)

    # predict: +1 = normal, -1 = anomalie → convertir en {1, 0}
    raw_pred = oc_svm.predict(X_test)
    y_pred = (raw_pred == 1).astype(int)
    scores = oc_svm.decision_function(X_test)
    # Normaliser les scores [0,1] pour roc_auc
    scores_norm = (scores - scores.min()) / (scores.ptp() + 1e-8)

    metrics = _compute_metrics("OneClassSVM", y_test, y_pred, scores_norm)

    model_path = MODELS_DIR / "one_class_svm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": oc_svm, "nu": nu, "feature_cols": FEATURE_COLS + KEYPOINT_FEATURES}, f)
    logger.success(f"  Modèle sauvegardé: {model_path}")

    return metrics


# ── Métriques communes ────────────────────────────────────────────────────────

def _compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> dict:
    """Calcule et affiche les métriques de classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")

    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"  ROC-AUC:   {auc:.4f}")

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
    }


# ── Sauvegarde des résultats ──────────────────────────────────────────────────

def save_results(all_metrics: list[dict]) -> None:
    """Sauvegarde les métriques en JSON (append) et TXT."""
    run_result = {
        "timestamp": datetime.now().isoformat(),
        "models": all_metrics,
    }

    json_path = RESULTS_DIR / "training_metrics.json"
    txt_path = RESULTS_DIR / "training_metrics.txt"

    existing: list = []
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    existing.append(run_result)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(txt_path, "a") as f:
        f.write(f"\n=== Run {run_result['timestamp']} ===\n")
        for m in all_metrics:
            f.write(f"  {m['model']}: acc={m['accuracy']}, f1={m['f1']}, auc={m.get('roc_auc','N/A')}\n")

    logger.success(f"Résultats sauvegardés: {json_path}, {txt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement CoachPosture AI")
    parser.add_argument("--model", choices=["all", "rf", "autoencoder", "svm"],
                        default="all", help="Modèle à entraîner")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Epochs pour l'autoencoder (défaut: 80)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction test split (défaut: 0.2)")
    parser.add_argument("--nu", type=float, default=0.05,
                        help="Nu pour One-Class SVM (défaut: 0.05)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    # Chargement
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Normalisation
    scaler = fit_scaler(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Sauvegarde du scaler (partagé par tous les modèles)
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler sauvegardé: {scaler_path}")

    all_metrics: list[dict] = []

    if args.model in ("all", "rf"):
        metrics = train_random_forest(X_train_sc, X_test_sc, y_train, y_test)
        all_metrics.append(metrics)

    if args.model in ("all", "autoencoder"):
        metrics = train_autoencoder(
            X_train_sc, X_test_sc, y_train, y_test, epochs=args.epochs
        )
        all_metrics.append(metrics)

    if args.model in ("all", "svm"):
        metrics = train_one_class_svm(X_train_sc, X_test_sc, y_train, y_test, nu=args.nu)
        all_metrics.append(metrics)

    save_results(all_metrics)
    logger.success("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
