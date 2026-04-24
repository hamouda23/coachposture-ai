# CoachPosture AI

Détection d'anomalies posturales en temps réel via webcam ordinaire — sans caméra de profondeur.

**Stack** : MediaPipe 33 keypoints · PyTorch Autoencoder · Random Forest · One-Class SVM · Ollama (LLM local) · Streamlit

---

## Démarrage rapide (3 commandes)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Préparer les données et entraîner les modèles
python src/data/download_datasets.py --synthetic-only && \
python src/data/extract_keypoints.py --dataset synthetic_lsp && \
python src/data/label_postures.py && \
python src/models/train.py --model all --epochs 80

# 3. Lancer le dashboard
streamlit run src/dashboard/app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501)

---

## Architecture

```
coachposture-ai/
├── src/data/
│   ├── download_datasets.py   # LSP, MPII, Kaggle + fallback synthétique
│   ├── extract_keypoints.py   # MediaPipe Pose → 33 landmarks → CSV
│   └── label_postures.py      # Features géométriques + label binaire
├── src/models/
│   ├── train.py               # RF + Autoencoder PyTorch + One-Class SVM
│   ├── evaluate.py            # Métriques, ROC, matrices de confusion
│   └── inference.py           # Flux webcam temps réel + alertes
├── src/agent/
│   └── posture_agent.py       # Recommandations LLM via Ollama
├── src/dashboard/
│   └── app.py                 # Interface Streamlit complète
├── notebooks/
│   └── exploration.ipynb      # Analyse exploratoire + visualisations
├── tests/
│   └── test_models.py         # Tests unitaires pytest (~30 tests)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Features géométriques calculées

| Feature | Calcul | Seuil normal |
|---------|--------|-------------|
| `angle_dos` | Épaules → Hanches vs verticale | < 10° |
| `angle_tete` | Nez → Épaules vs verticale | < 15° |
| `symetrie_epaules` | Différence hauteur épaules | < 3% |
| `inclinaison_tronc` | Hanches → Épaules vs verticale | < 10° |
| `angle_cou` | Oreilles → Épaules vs verticale | < 20° |
| `score_posture` | Pénalités cumulées [0–100] | ≥ 60 |

## Modèles

| Modèle | Type | Entraînement | Usage |
|--------|------|-------------|-------|
| **Autoencoder** (principal) | Détection anomalies | Postures correctes uniquement | Recommandé |
| **Random Forest** | Supervisé | Classes équilibrées | Baseline |
| **One-Class SVM** | Détection anomalies | Postures correctes uniquement | Alternative légère |

## Datasets supportés

- **Leeds Sports Pose (LSP)** — 1000 images, 14 joints
- **MPII Human Pose** — 25 000 images, 16 joints (~12 GB)
- **Kaggle Posture** — datasets ergonomie bureau (API Kaggle requise)
- **Synthétique** — généré localement, aucune dépendance réseau

## Lancement avec Docker

```bash
docker compose up -d
```

Pipeline d'entraînement complet :
```bash
docker compose --profile training up trainer
```

## Configuration Ollama

Le modèle `qwen2.5:7b` est utilisé par défaut (disponible sur ce serveur).  
Pour changer le modèle dans le dashboard : sidebar → "Modèle Ollama".

```bash
# Vérifier les modèles disponibles
ollama list

# Télécharger qwen3 si disponible
ollama pull qwen3:latest
```

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Ajout caméra Yahboom (future extension)

Le code est préparé pour l'ajout de la caméra Yahboom :
```python
# Dans inference.py et app.py, changer simplement l'index
python src/models/inference.py --camera 1   # ou l'index Yahboom
```

Pour les caméras USB avec pilote personnalisé, modifier `cv2.VideoCapture(camera_id)` 
dans `inference.py:run_inference()` — le reste du pipeline est identique.

## Résultats

Tous les résultats sont sauvegardés dans `results/` :
- `training_metrics.json` / `.txt` — métriques d'entraînement (append par run)
- `evaluation_results.json` / `.txt` — métriques d'évaluation
- `recommendations.json` / `.txt` — historique des conseils LLM
- `labeling_stats.json` / `.txt` — statistiques de labellisation

---

**Serveur** : HP Z800 · Quadro P4000 8GB · Ubuntu 22.04 · CUDA 12.x
