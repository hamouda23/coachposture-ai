"""
Agent de recommandations posturales via Ollama (LLM local).

Appelle http://localhost:11434 avec le modèle qwen2.5:7b (disponible localement).
Génère des recommandations personnalisées en français selon les features posturales.

Le contexte envoyé au LLM inclut :
  - Les features géométriques mesurées
  - Le score de posture
  - La durée de la mauvaise posture
  - L'historique des recommandations passées (évite les répétitions)

Utilisation :
    from src.agent.posture_agent import PostureAgent
    agent = PostureAgent()
    rec = agent.recommend(features, score=45.0, alert_duration=15.0)
"""

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
# qwen2.5:7b est disponible localement (qwen3:latest non présent sur ce serveur)
DEFAULT_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "mistral:latest"
REQUEST_TIMEOUT = 30.0

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Modèles de données ────────────────────────────────────────────────────────

class PostureFeatures(BaseModel):
    """Features géométriques calculées depuis les keypoints."""
    angle_dos: float = Field(default=0.0, description="Angle du dos vs verticale (degrés)")
    angle_tete: float = Field(default=0.0, description="Angle de la tête vs verticale (degrés)")
    symetrie_epaules: float = Field(default=0.0, description="Asymétrie des épaules (%)")
    inclinaison_tronc: float = Field(default=0.0, description="Inclinaison du tronc (degrés)")
    angle_cou: float = Field(default=0.0, description="Angle du cou vs verticale (degrés)")
    score_posture: float = Field(default=50.0, description="Score global [0-100]")


class PostureRecommendation(BaseModel):
    """Recommandation générée par le LLM."""
    timestamp: str
    score_posture: float
    recommandation: str
    probleme_principal: str
    exercice_suggere: str
    model_utilise: str
    duree_mauvaise_posture_sec: float


# ── Système de déduplication ──────────────────────────────────────────────────

RECOMMENDATIONS_HISTORY: deque[str] = deque(maxlen=5)


def _build_prompt(
    features: PostureFeatures,
    alert_duration: float,
    history: list[str],
) -> str:
    """Construit le prompt pour Ollama."""
    # Identification des problèmes principaux
    problemes: list[str] = []
    if features.angle_dos > 20:
        problemes.append(f"dos courbé ({features.angle_dos:.1f}° d'écart vs vertical)")
    if features.angle_tete > 25:
        problemes.append(f"tête penchée vers l'avant ({features.angle_tete:.1f}°)")
    if features.symetrie_epaules > 5:
        problemes.append(f"épaules asymétriques ({features.symetrie_epaules:.1f}%)")
    if features.inclinaison_tronc > 20:
        problemes.append(f"tronc incliné ({features.inclinaison_tronc:.1f}°)")
    if features.angle_cou > 30:
        problemes.append(f"cou tendu ({features.angle_cou:.1f}°)")

    problemes_str = "\n  - ".join(problemes) if problemes else "Posture globalement correcte"

    history_str = ""
    if history:
        history_str = "\n\nRecommandations déjà données (évite de les répéter) :\n- " + "\n- ".join(history[-3:])

    duree_str = f"{alert_duration:.0f} secondes" if alert_duration > 0 else "récemment détectée"

    return f"""Tu es un coach ergonomique expert en posture au travail.
Analyse les données posturales suivantes et donne une recommandation personnalisée en français.

=== Données de posture ===
Score global : {features.score_posture:.0f}/100
Durée en mauvaise posture : {duree_str}

Problèmes détectés :
  - {problemes_str}

Mesures précises :
  - Angle du dos : {features.angle_dos:.1f}° (idéal < 10°)
  - Angle de la tête : {features.angle_tete:.1f}° (idéal < 15°)
  - Asymétrie épaules : {features.symetrie_epaules:.1f}% (idéal < 3%)
  - Inclinaison tronc : {features.inclinaison_tronc:.1f}° (idéal < 10°)
  - Angle du cou : {features.angle_cou:.1f}° (idéal < 20°)
{history_str}

=== Instructions ===
Réponds en JSON strict avec exactement ces 3 champs :
{{
  "probleme_principal": "Description courte du problème le plus urgent (1 phrase)",
  "recommandation": "Conseil immédiat pratique à appliquer maintenant (2-3 phrases)",
  "exercice_suggere": "Un exercice simple de 30-60 secondes à faire maintenant (description)"
}}

Sois direct, concis et pratique. Utilise un ton bienveillant mais ferme."""


# ── Client Ollama ─────────────────────────────────────────────────────────────

class PostureAgent:
    """
    Agent LLM pour recommandations posturales personnalisées.
    Utilise Ollama en local pour préserver la confidentialité.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT)
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Vérifie qu'Ollama est accessible et que le modèle est disponible."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]

            if self.model not in models:
                logger.warning(f"Modèle {self.model} non disponible.")
                # Essai avec le modèle de fallback
                if FALLBACK_MODEL in models:
                    self.model = FALLBACK_MODEL
                    logger.info(f"Bascule sur {FALLBACK_MODEL}")
                elif models:
                    # Prendre le premier modèle disponible
                    self.model = models[0]
                    logger.info(f"Bascule sur {self.model}")
                else:
                    logger.warning("Aucun modèle Ollama disponible.")
            else:
                logger.info(f"Ollama OK — modèle: {self.model}")

        except httpx.ConnectError:
            logger.warning(f"Ollama inaccessible ({self.base_url}). Mode dégradé activé.")
        except Exception as e:
            logger.warning(f"Vérification Ollama: {e}")

    def recommend(
        self,
        features: PostureFeatures,
        alert_duration: float = 0.0,
        max_retries: int = 2,
    ) -> PostureRecommendation:
        """
        Génère une recommandation posturale personnalisée.
        En cas d'échec Ollama, retourne une recommandation statique.
        """
        history = list(RECOMMENDATIONS_HISTORY)
        prompt = _build_prompt(features, alert_duration, history)

        for attempt in range(max_retries + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 400,
                        },
                    },
                )
                response.raise_for_status()
                raw_text = response.json()["response"]

                # Extraction JSON depuis la réponse
                parsed = self._parse_response(raw_text)

                rec = PostureRecommendation(
                    timestamp=datetime.now().isoformat(),
                    score_posture=features.score_posture,
                    recommandation=parsed.get("recommandation", ""),
                    probleme_principal=parsed.get("probleme_principal", ""),
                    exercice_suggere=parsed.get("exercice_suggere", ""),
                    model_utilise=self.model,
                    duree_mauvaise_posture_sec=alert_duration,
                )

                # Historique pour déduplication
                RECOMMENDATIONS_HISTORY.append(rec.recommandation[:100])

                # Sauvegarde
                self._save_recommendation(rec)
                return rec

            except httpx.TimeoutException:
                logger.warning(f"Timeout Ollama (tentative {attempt + 1})")
            except Exception as e:
                logger.warning(f"Erreur Ollama: {e}")
                if attempt < max_retries:
                    time.sleep(1.0)

        # Fallback statique si Ollama inaccessible
        return self._static_recommendation(features, alert_duration)

    def _parse_response(self, text: str) -> dict:
        """Extrait le JSON de la réponse du LLM (qui peut contenir du texte parasite)."""
        # Cherche le bloc JSON dans la réponse
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Fallback : extraire ligne par ligne
        return {
            "probleme_principal": "Analyse indisponible",
            "recommandation": text[:300] if text else "Redressez-vous et prenez une pause.",
            "exercice_suggere": "Roulez les épaules en arrière 10 fois.",
        }

    def _static_recommendation(
        self, features: PostureFeatures, alert_duration: float
    ) -> PostureRecommendation:
        """Recommandations statiques basées sur les règles géométriques."""
        recs = {
            "dos": (
                "Votre dos est courbé.",
                "Asseyez-vous au fond du siège, colonne vertébrale droite. Imaginez un fil vous tirant vers le haut.",
                "Arch-back stretch : croisez les mains derrière la tête, tirez doucement les coudes en arrière. 30 secondes.",
            ),
            "tete": (
                "Votre tête est projetée vers l'avant.",
                "Reculez la tête pour aligner vos oreilles avec vos épaules. Montez l'écran à hauteur des yeux.",
                "Chin tuck : rentrez le menton 10 fois, tenez 5 secondes à chaque fois.",
            ),
            "epaules": (
                "Vos épaules sont asymétriques.",
                "Vérifiez l'ergonomie de votre poste : siège à la bonne hauteur, souris et clavier centrés.",
                "Rotation d'épaules : 10 rotations vers l'avant, 10 vers l'arrière.",
            ),
        }

        # Sélection selon le problème dominant
        if features.angle_dos > 20:
            key = "dos"
        elif features.angle_tete > 25:
            key = "tete"
        else:
            key = "epaules"

        prob, rec, ex = recs[key]
        return PostureRecommendation(
            timestamp=datetime.now().isoformat(),
            score_posture=features.score_posture,
            recommandation=rec,
            probleme_principal=prob,
            exercice_suggere=ex,
            model_utilise="static_fallback",
            duree_mauvaise_posture_sec=alert_duration,
        )

    def _save_recommendation(self, rec: PostureRecommendation) -> None:
        """Sauvegarde la recommandation en JSON (append) et TXT."""
        json_path = RESULTS_DIR / "recommendations.json"
        txt_path = RESULTS_DIR / "recommendations.txt"

        rec_dict = rec.model_dump()

        existing: list = []
        if json_path.exists():
            try:
                with open(json_path) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(rec_dict)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{rec.timestamp}] Score: {rec.score_posture:.0f}/100\n")
            f.write(f"Problème: {rec.probleme_principal}\n")
            f.write(f"Conseil: {rec.recommandation}\n")
            f.write(f"Exercice: {rec.exercice_suggere}\n")

    def close(self) -> None:
        self.client.close()
