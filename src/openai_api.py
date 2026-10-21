"""
Cliente mínimo para la API REST de OpenAI

Lo usan el embedder (/embeddings) y el generador (/chat/completions), así la
key, el timeout y el manejo de errores viven en un solo lugar
"""

from typing import Any
import requests
from src.conf import OPENAI_API_KEY

OPENAI_BASE_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT = 60  # segundos; sin timeout, requests puede esperar para siempre

def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Hace POST a `OPENAI_BASE_URL + path` con `payload` como JSON y devuelve
    el cuerpo de la respuesta ya parseado
    """

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Agrégala al archivo .env (ver .env.example)."
        )

    response = requests.post(
        f"{OPENAI_BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        # El cuerpo trae el motivo real (key inválida, modelo inexistente,
        # cuota agotada...); raise_for_status() solo daría el código HTTP
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = response.text
        raise RuntimeError(f"OpenAI respondió {response.status_code}: {detail}")

    return response.json()
