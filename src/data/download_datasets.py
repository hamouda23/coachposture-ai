"""
Téléchargement des datasets de pose humaine pour CoachPosture AI.

Datasets ciblés :
  1. LSP  — Leeds Sports Pose (1000 images annotées, 14 joints)
  2. MPII — MPII Human Pose (25 000 images, 16 joints)
  3. Kaggle ergonomie bureau — pose assis / debout labellisée

Utilisation :
    python src/data/download_datasets.py --dataset all
    python src/data/download_datasets.py --dataset lsp
    python src/data/download_datasets.py --dataset mpii
    python src/data/download_datasets.py --dataset kaggle
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from tqdm import tqdm

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration des sources ────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "lsp": {
        "name": "Leeds Sports Pose (LSP)",
        "urls": [
            # Archive officielle (miroir maintenu par la communauté)
            "https://github.com/Aravindlivewire/Datascience/raw/master/Hand_gesture/lsp_dataset.zip",
            # Fallback — subset reconstruit sur HuggingFace
            "https://huggingface.co/datasets/dbroqua/lsp-dataset/resolve/main/lsp_dataset.zip",
        ],
        "filename": "lsp_dataset.zip",
        "subdir": "lsp",
        "extractor": "zip",
    },
    "mpii": {
        "name": "MPII Human Pose (images)",
        "urls": [
            # Archive officielle MPI Informatik
            "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        ],
        "filename": "mpii_human_pose_v1.tar.gz",
        "subdir": "mpii",
        "extractor": "tar",
        # Annotations séparées
        "annotations_url": "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip",
        "annotations_filename": "mpii_annotations.zip",
    },
    "kaggle_posture": {
        "name": "Kaggle — Sitting Posture Recognition",
        # Téléchargement via API Kaggle (nécessite ~/.kaggle/kaggle.json)
        "kaggle_dataset": "anshtanwar/gym-posture-dataset",
        "subdir": "kaggle_posture",
        "extractor": "kaggle",
    },
    "kaggle_ergonomics": {
        "name": "Kaggle — Body Posture & Ergonomics",
        "kaggle_dataset": "niharika41298/body-performance-data",
        "subdir": "kaggle_ergonomics",
        "extractor": "kaggle",
    },
}


# ── Utilitaires réseau ────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Télécharge un fichier avec barre de progression. Retourne True si succès."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        logger.warning(f"Échec téléchargement {url}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def _try_urls(urls: list[str], dest: Path) -> bool:
    """Essaie chaque URL dans l'ordre, retourne True dès qu'une réussit."""
    for url in urls:
        logger.info(f"  → Tentative: {url}")
        if _download_file(url, dest):
            return True
    return False


# ── Extracteurs ──────────────────────────────────────────────────────────────

def _extract_zip(archive: Path, dest_dir: Path) -> None:
    logger.info(f"  Extraction ZIP → {dest_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    logger.info(f"  Extraction TAR → {dest_dir}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)


def _download_kaggle(dataset_slug: str, dest_dir: Path) -> bool:
    """Télécharge via l'API Kaggle. Nécessite ~/.kaggle/kaggle.json."""
    try:
        import subprocess
        dest_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir), "--unzip"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Kaggle API erreur: {result.stderr}")
            return False
        logger.success(f"  Dataset Kaggle téléchargé: {dataset_slug}")
        return True
    except FileNotFoundError:
        logger.warning("CLI Kaggle introuvable. Installez-le: pip install kaggle")
        logger.warning("Puis créez ~/.kaggle/kaggle.json avec votre token API.")
        return False


# ── Fonctions principales par dataset ────────────────────────────────────────

def download_lsp() -> bool:
    """Télécharge et extrait le dataset Leeds Sports Pose."""
    cfg = DATASETS["lsp"]
    logger.info(f"=== {cfg['name']} ===")

    subdir = RAW_DIR / cfg["subdir"]
    if subdir.exists() and any(subdir.iterdir()):
        logger.info("  LSP déjà présent, téléchargement ignoré.")
        return True

    archive_path = RAW_DIR / cfg["filename"]

    if not archive_path.exists():
        ok = _try_urls(cfg["urls"], archive_path)
        if not ok:
            logger.error("  Impossible de télécharger LSP depuis toutes les sources.")
            _create_synthetic_lsp(subdir)
            return False

    subdir.mkdir(parents=True, exist_ok=True)
    _extract_zip(archive_path, subdir)
    archive_path.unlink(missing_ok=True)
    logger.success(f"  LSP extrait dans {subdir}")
    return True


def download_mpii() -> bool:
    """Télécharge et extrait MPII Human Pose."""
    cfg = DATASETS["mpii"]
    logger.info(f"=== {cfg['name']} ===")

    subdir = RAW_DIR / cfg["subdir"]
    if subdir.exists() and any(subdir.iterdir()):
        logger.info("  MPII déjà présent, téléchargement ignoré.")
        return True

    archive_path = RAW_DIR / cfg["filename"]

    if not archive_path.exists():
        logger.warning("  MPII est ~12 GB. Le téléchargement peut prendre plusieurs minutes.")
        ok = _try_urls(cfg["urls"], archive_path)
        if not ok:
            logger.error("  Impossible de télécharger MPII.")
            logger.info("  Téléchargement manuel: http://human-pose.mpi-inf.mpg.de/")
            return False

    subdir.mkdir(parents=True, exist_ok=True)
    _extract_tar(archive_path, subdir)
    archive_path.unlink(missing_ok=True)

    # Télécharger les annotations séparément
    ann_path = RAW_DIR / cfg["annotations_filename"]
    if not ann_path.exists():
        _download_file(cfg["annotations_url"], ann_path)
    if ann_path.exists():
        _extract_zip(ann_path, subdir)
        ann_path.unlink(missing_ok=True)

    logger.success(f"  MPII extrait dans {subdir}")
    return True


def download_kaggle_datasets() -> bool:
    """Télécharge les datasets Kaggle de posture/ergonomie."""
    success = True
    for key in ("kaggle_posture", "kaggle_ergonomics"):
        cfg = DATASETS[key]
        logger.info(f"=== {cfg['name']} ===")
        subdir = RAW_DIR / cfg["subdir"]
        if subdir.exists() and any(subdir.iterdir()):
            logger.info(f"  {key} déjà présent.")
            continue
        ok = _download_kaggle(cfg["kaggle_dataset"], subdir)
        if not ok:
            success = False
    return success


# ── Fallback : données synthétiques ─────────────────────────────────────────

def _create_synthetic_lsp(dest_dir: Path) -> None:
    """
    Génère un mini-dataset synthétique si les vrais datasets sont inaccessibles.
    Produit 200 images 128×128 avec des squelettes simulés.
    Utile pour tester le pipeline sans connexion internet.
    """
    import json
    import random

    import numpy as np
    from PIL import Image, ImageDraw

    logger.info("  Génération données synthétiques LSP (fallback)...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dest_dir / "images"
    images_dir.mkdir(exist_ok=True)

    joints_all: list[dict] = []
    random.seed(42)
    np.random.seed(42)

    # Connexions du squelette LSP (14 joints)
    SKELETON = [
        (0, 1), (1, 2), (2, 3),   # Jambe gauche
        (3, 4), (4, 5),            # Jambe droite
        (6, 7), (7, 8),            # Bras gauche
        (8, 9), (9, 10),           # Bras droit
        (11, 12), (12, 13),        # Torse
    ]

    for idx in range(200):
        img = Image.new("RGB", (128, 128), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Joints aléatoires réalistes
        cx, cy = 64 + random.randint(-10, 10), 64 + random.randint(-10, 10)
        joints = [
            (cx - 15, cy + 40), (cx - 10, cy + 20), (cx - 5, cy),    # J0-2
            (cx + 5, cy), (cx + 10, cy + 20), (cx + 15, cy + 40),    # J3-5
            (cx - 25, cy - 5), (cx - 20, cy + 10), (cx - 10, cy + 15),  # J6-8
            (cx + 10, cy + 15), (cx + 20, cy + 10), (cx + 25, cy - 5),  # J9-11
            (cx, cy - 20), (cx, cy - 35),                               # J12-13
        ]

        for a, b in SKELETON:
            draw.line([joints[a], joints[b]], fill=(50, 50, 200), width=2)
        for j in joints:
            draw.ellipse([j[0]-3, j[1]-3, j[0]+3, j[1]+3], fill=(200, 50, 50))

        img_path = images_dir / f"im{idx:04d}.jpg"
        img.save(img_path)

        joints_all.append({
            "image": img_path.name,
            "joints": joints,
            "is_visible": [1] * 14,
        })

    # Sauvegarde annotations
    with open(dest_dir / "joints.json", "w") as f:
        json.dump(joints_all, f, indent=2)

    logger.success(f"  200 images synthétiques créées dans {dest_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharge les datasets CoachPosture AI")
    parser.add_argument(
        "--dataset",
        choices=["all", "lsp", "mpii", "kaggle", "synthetic"],
        default="all",
        help="Dataset à télécharger (défaut: all)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Génère uniquement les données synthétiques (pas d'internet requis)",
    )
    return parser.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    args = parse_args()
    results: dict[str, bool] = {}

    if args.synthetic_only:
        _create_synthetic_lsp(RAW_DIR / "synthetic_lsp")
        return

    if args.dataset in ("all", "lsp"):
        results["LSP"] = download_lsp()

    if args.dataset in ("all", "mpii"):
        results["MPII"] = download_mpii()

    if args.dataset in ("all", "kaggle"):
        results["Kaggle"] = download_kaggle_datasets()

    if args.dataset == "synthetic":
        _create_synthetic_lsp(RAW_DIR / "synthetic_lsp")
        results["Synthétique"] = True

    # Résumé
    logger.info("\n=== Résumé des téléchargements ===")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        logger.info(f"  [{status}] {name}")

    if not all(results.values()):
        logger.warning("Certains datasets ont échoué. Utilisez --synthetic-only pour tester.")
        sys.exit(1)


if __name__ == "__main__":
    main()
