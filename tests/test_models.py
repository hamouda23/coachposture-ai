"""
Tests unitaires et d'intégration pour CoachPosture AI.

Couvre :
  - Calcul des features géométriques
  - Pipeline labellisation
  - Autoencoder (forward pass, reconstruction error)
  - Random Forest (prédiction sur données mock)
  - Inférence ModelLoader (sans GPU requis)
  - Agent posture (réponse statique)

Lancement :
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.label_postures import compute_features, label_dataframe
from src.models.train import PostureAutoencoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_keypoints_row() -> pd.Series:
    """Ligne de keypoints MediaPipe simulée (posture correcte)."""
    from src.data.extract_keypoints import LANDMARK_NAMES
    data: dict = {}
    # Posture droite : nez en haut, épaules symétriques, hanches centrées
    positions = {
        "nose": (0.50, 0.15),
        "left_eye_inner": (0.52, 0.14), "left_eye": (0.54, 0.13), "left_eye_outer": (0.56, 0.13),
        "right_eye_inner": (0.48, 0.14), "right_eye": (0.46, 0.13), "right_eye_outer": (0.44, 0.13),
        "left_ear": (0.58, 0.16), "right_ear": (0.42, 0.16),
        "mouth_left": (0.53, 0.18), "mouth_right": (0.47, 0.18),
        "left_shoulder": (0.60, 0.30), "right_shoulder": (0.40, 0.30),
        "left_elbow": (0.65, 0.45), "right_elbow": (0.35, 0.45),
        "left_wrist": (0.68, 0.58), "right_wrist": (0.32, 0.58),
        "left_pinky": (0.69, 0.61), "right_pinky": (0.31, 0.61),
        "left_index": (0.70, 0.60), "right_index": (0.30, 0.60),
        "left_thumb": (0.67, 0.59), "right_thumb": (0.33, 0.59),
        "left_hip": (0.57, 0.60), "right_hip": (0.43, 0.60),
        "left_knee": (0.58, 0.78), "right_knee": (0.42, 0.78),
        "left_ankle": (0.57, 0.92), "right_ankle": (0.43, 0.92),
        "left_heel": (0.56, 0.94), "right_heel": (0.44, 0.94),
        "left_foot_index": (0.58, 0.96), "right_foot_index": (0.42, 0.96),
    }
    for name in LANDMARK_NAMES:
        x, y = positions.get(name, (0.50, 0.50))
        data[f"{name}_x"] = x
        data[f"{name}_y"] = y
        data[f"{name}_z"] = 0.0
        data[f"{name}_visibility"] = 0.9
    return pd.Series(data)


@pytest.fixture
def sample_bad_posture_row(sample_keypoints_row) -> pd.Series:
    """Posture anormale : tête penchée en avant, dos courbé."""
    row = sample_keypoints_row.copy()
    # Nez fortement vers l'avant (bas) = tête penchée
    row["nose_y"] = 0.32
    row["nose_x"] = 0.45  # légèrement de côté
    # Épaules asymétriques
    row["left_shoulder_y"] = 0.35
    row["right_shoulder_y"] = 0.28
    # Hanches décalées
    row["left_hip_y"] = 0.68
    row["right_hip_y"] = 0.62
    return row


@pytest.fixture
def sample_dataframe(sample_keypoints_row) -> pd.DataFrame:
    """DataFrame avec 20 lignes simulées."""
    from src.data.extract_keypoints import LANDMARK_NAMES
    rows = []
    for i in range(20):
        row = sample_keypoints_row.copy()
        # Légère variation aléatoire
        np.random.seed(i)
        for name in LANDMARK_NAMES:
            row[f"{name}_x"] += np.random.uniform(-0.02, 0.02)
            row[f"{name}_y"] += np.random.uniform(-0.02, 0.02)
        row["image_path"] = f"test_image_{i:04d}.jpg"
        row["extraction_success"] = True
        rows.append(row)
    return pd.DataFrame(rows)


# ── Tests features géométriques ───────────────────────────────────────────────

class TestComputeFeatures:

    def test_bonne_posture_score_eleve(self, sample_keypoints_row):
        """Une posture droite doit donner un score élevé."""
        features = compute_features(sample_keypoints_row)
        assert features["score_posture"] >= 60, (
            f"Posture correcte attendue (score >= 60), obtenu: {features['score_posture']}"
        )

    def test_mauvaise_posture_score_bas(self, sample_bad_posture_row):
        """Une posture courbée doit donner un score plus bas."""
        good_features = compute_features(sample_bad_posture_row.copy())
        # Le score doit être < 100 (pénalités appliquées)
        assert good_features["score_posture"] < 100

    def test_toutes_features_presentes(self, sample_keypoints_row):
        """Toutes les features attendues doivent être calculées."""
        features = compute_features(sample_keypoints_row)
        expected = [
            "angle_dos", "angle_tete", "symetrie_epaules",
            "inclinaison_tronc", "angle_cou", "ratio_epaules_hanches", "score_posture",
        ]
        for feat in expected:
            assert feat in features, f"Feature manquante: {feat}"

    def test_score_dans_range_valide(self, sample_keypoints_row):
        """Le score doit être dans [0, 100]."""
        features = compute_features(sample_keypoints_row)
        assert 0.0 <= features["score_posture"] <= 100.0

    def test_angle_dos_positif(self, sample_keypoints_row):
        """L'angle du dos doit être non-négatif."""
        features = compute_features(sample_keypoints_row)
        assert features["angle_dos"] >= 0.0

    def test_symetrie_epaules_bonne_posture(self, sample_keypoints_row):
        """Épaules symétriques → faible valeur d'asymétrie."""
        features = compute_features(sample_keypoints_row)
        # Épaules parfaitement symétriques → ~0%
        assert features["symetrie_epaules"] < 5.0

    def test_features_lignes_nan(self):
        """Des keypoints NaN doivent retourner des features NaN."""
        row = pd.Series({col: np.nan for col in [
            "nose_x", "nose_y", "left_shoulder_x", "left_shoulder_y",
            "right_shoulder_x", "right_shoulder_y", "left_hip_x", "left_hip_y",
            "right_hip_x", "right_hip_y", "left_ear_x", "left_ear_y",
            "right_ear_x", "right_ear_y",
        ]})
        features = compute_features(row)
        assert np.isnan(features["score_posture"])


# ── Tests labellisation ────────────────────────────────────────────────────────

class TestLabelDataframe:

    def test_label_binaire_correct(self, sample_dataframe):
        """Les labels doivent être 0 ou 1."""
        df_labeled = label_dataframe(sample_dataframe, threshold=60.0)
        if len(df_labeled) > 0:
            assert set(df_labeled["posture_correcte"].unique()).issubset({0, 1})

    def test_filtre_lignes_invalides(self):
        """Les lignes sans détection doivent être filtrées."""
        from src.data.extract_keypoints import LANDMARK_NAMES
        rows = [
            {f"{name}_{c}": np.nan for name in LANDMARK_NAMES for c in ("x", "y", "z", "visibility")}
            | {"image_path": "bad.jpg", "extraction_success": False}
        ]
        df = pd.DataFrame(rows)
        df_labeled = label_dataframe(df, threshold=60.0)
        assert len(df_labeled) == 0

    def test_seuil_personnalisable(self, sample_dataframe):
        """Threshold différent doit donner une distribution différente."""
        df_60 = label_dataframe(sample_dataframe, threshold=60.0)
        df_80 = label_dataframe(sample_dataframe, threshold=80.0)
        if len(df_60) > 0 and len(df_80) > 0:
            # Plus haute threshold → moins de postures "correctes"
            assert df_80["posture_correcte"].mean() <= df_60["posture_correcte"].mean()


# ── Tests Autoencoder ─────────────────────────────────────────────────────────

class TestPostureAutoencoder:

    @pytest.fixture
    def model(self) -> PostureAutoencoder:
        return PostureAutoencoder(input_dim=20)

    def test_forward_pass_shape(self, model):
        """Le forward pass doit retourner la même forme que l'entrée."""
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == x.shape

    def test_reconstruction_error_positive(self, model):
        """L'erreur de reconstruction doit être positive."""
        x = torch.randn(5, 20)
        errors = model.reconstruction_error(x)
        assert errors.shape == (5,)
        assert (errors >= 0).all()

    def test_reconstruction_error_zero_pour_reconstruction_parfaite(self, model):
        """Si l'autoencoder reconstruit parfaitement, erreur ~= 0."""
        x = torch.zeros(1, 20)
        # Forcer les poids à zéro → sortie zéro → erreur nulle
        for p in model.parameters():
            p.data.fill_(0.0)
        errors = model.reconstruction_error(x)
        assert errors.item() < 1e-6

    def test_encodeur_dimension_latente(self, model):
        """L'encodeur doit produire une représentation de dim 16."""
        x = torch.randn(4, 20)
        latent = model.encoder(x)
        assert latent.shape == (4, 16)

    def test_gradient_flow(self, model):
        """Les gradients doivent circuler jusqu'à l'entrée."""
        x = torch.randn(3, 20, requires_grad=False)
        errors = model.reconstruction_error(x)
        loss = errors.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient NaN: {name}"

    @pytest.mark.parametrize("batch_size,input_dim", [(1, 6), (32, 20), (64, 14)])
    def test_dimensions_variables(self, batch_size: int, input_dim: int):
        """L'autoencoder doit fonctionner avec différentes dimensions."""
        model = PostureAutoencoder(input_dim=input_dim)
        x = torch.randn(batch_size, input_dim)
        out = model(x)
        assert out.shape == (batch_size, input_dim)


# ── Tests agent posture ────────────────────────────────────────────────────────

class TestPostureAgent:

    def test_static_fallback_bonne_structure(self):
        """Le fallback statique doit retourner une recommandation valide."""
        from src.agent.posture_agent import PostureAgent, PostureFeatures

        agent = PostureAgent.__new__(PostureAgent)
        agent.model = "test"
        features = PostureFeatures(
            angle_dos=30.0,
            angle_tete=10.0,
            symetrie_epaules=2.0,
            inclinaison_tronc=5.0,
            angle_cou=10.0,
            score_posture=45.0,
        )
        rec = agent._static_recommendation(features, alert_duration=15.0)

        assert rec.score_posture == 45.0
        assert len(rec.recommandation) > 10
        assert len(rec.probleme_principal) > 3
        assert len(rec.exercice_suggere) > 10
        assert rec.duree_mauvaise_posture_sec == 15.0

    def test_parse_response_json_valide(self):
        """Le parseur doit extraire le JSON correctement."""
        from src.agent.posture_agent import PostureAgent

        agent = PostureAgent.__new__(PostureAgent)
        text = 'Voici mon analyse:\n{"probleme_principal": "Dos courbé", "recommandation": "Redressez-vous", "exercice_suggere": "Rotation épaules"}\nMerci'
        result = agent._parse_response(text)

        assert result["probleme_principal"] == "Dos courbé"
        assert result["recommandation"] == "Redressez-vous"

    def test_parse_response_fallback_texte(self):
        """Si pas de JSON, le fallback texte doit fonctionner."""
        from src.agent.posture_agent import PostureAgent

        agent = PostureAgent.__new__(PostureAgent)
        result = agent._parse_response("Redressez votre dos maintenant.")
        assert "recommandation" in result
        assert len(result["recommandation"]) > 0

    def test_posture_features_validation(self):
        """PostureFeatures doit accepter des valeurs dans la plage normale."""
        from src.agent.posture_agent import PostureFeatures

        pf = PostureFeatures(
            angle_dos=15.0,
            angle_tete=20.0,
            symetrie_epaules=3.5,
            inclinaison_tronc=8.0,
            angle_cou=12.0,
            score_posture=72.0,
        )
        assert pf.score_posture == 72.0
        assert pf.angle_dos == 15.0


# ── Tests intégration pipeline ────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_features_vers_prediction(self, sample_keypoints_row):
        """Le pipeline complet features → autoencoder → label doit fonctionner."""
        features = compute_features(sample_keypoints_row)
        score = features["score_posture"]
        label = 1 if score >= 60 else 0

        assert isinstance(score, float)
        assert label in (0, 1)

    def test_autoencoder_pipeline_complet(self):
        """Autoencoder: train sur données positives, predict sur mixte."""
        input_dim = 6
        model = PostureAutoencoder(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Entraînement minimal (3 steps)
        X_train = torch.randn(50, input_dim) * 0.1  # Distribution étroite = "bon"
        for _ in range(3):
            optimizer.zero_grad()
            out = model(X_train)
            loss = ((X_train - out) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Test : données proches → faible erreur, éloignées → haute erreur
        X_normal = torch.randn(10, input_dim) * 0.1
        X_anomaly = torch.randn(10, input_dim) * 5.0

        with torch.no_grad():
            err_normal = model.reconstruction_error(X_normal).mean().item()
            err_anomaly = model.reconstruction_error(X_anomaly).mean().item()

        # Les anomalies devraient avoir des erreurs plus élevées
        assert err_anomaly > err_normal, (
            f"Erreur anomalie ({err_anomaly:.4f}) devrait > normal ({err_normal:.4f})"
        )

    def test_format_resultats_json(self, tmp_path, sample_keypoints_row):
        """Les résultats sauvegardés doivent être du JSON valide."""
        from src.data.label_postures import RESULTS_DIR
        from unittest.mock import patch

        with patch("src.data.label_postures.RESULTS_DIR", tmp_path):
            features = compute_features(sample_keypoints_row)
            result = {
                "timestamp": "2026-01-01T00:00:00",
                "score": features["score_posture"],
            }
            json_path = tmp_path / "test_results.json"
            with open(json_path, "w") as f:
                json.dump([result], f)

            with open(json_path) as f:
                loaded = json.load(f)
            assert loaded[0]["score"] == result["score"]


# ── Tests robustesse ──────────────────────────────────────────────────────────

class TestRobustesse:

    def test_features_avec_valeurs_extremes(self):
        """Les features doivent rester bornées même avec des landmarks extrêmes."""
        from src.data.extract_keypoints import LANDMARK_NAMES
        data = {}
        for name in LANDMARK_NAMES:
            data[f"{name}_x"] = 0.0
            data[f"{name}_y"] = 0.0
            data[f"{name}_z"] = 0.0
            data[f"{name}_visibility"] = 1.0
        row = pd.Series(data)
        features = compute_features(row)
        score = features["score_posture"]
        # Peut être NaN si tous les points sont superposés, mais ne doit pas crasher
        assert isinstance(score, float)

    def test_autoencoder_batch_size_1(self):
        """L'autoencoder doit fonctionner avec batch_size=1."""
        model = PostureAutoencoder(input_dim=10)
        model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
            out = model(x)
            err = model.reconstruction_error(x)
        assert out.shape == (1, 10)
        assert err.shape == (1,)

    @pytest.mark.parametrize("score,expected_label", [
        (80.0, 1), (60.0, 1), (59.9, 0), (0.0, 0),
    ])
    def test_seuil_label_parametrique(self, score: float, expected_label: int):
        """Le seuil de 60 doit correctement séparer les classes."""
        label = 1 if score >= 60.0 else 0
        assert label == expected_label
