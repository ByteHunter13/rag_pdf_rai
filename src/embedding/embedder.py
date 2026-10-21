"""
3. Embeddings con sentence-transformers

Objetivo: dado un texto (chunk o pregunta del usuario), producir su vector de embedding
usando EMBEDDING_MODEL

Este módulo lo usan tanto el pipeline de ingesta (embeber chunks antes de guardarlos en 
Postgres) como el retriever (embeber la pregunta del usuario)
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.conf import EMBEDDING_DIM, EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Carga (y cachea en memoria) el modelo de embeddings

    - `SentenceTransformer(EMBEDDING_MODEL)` - la primera vez que se ejecuta descarga el
    modelo y lo cachea en disco; las siguientes veces carga directo del caché
    - Usa `@lru_cache` (ya puesto arriba) para no recargar el modelo en memoria cada
    vez que se llama a esta función dentro del mismo proceso - cargarlo es lento, embeber
    con el modelo ya cargado es rápido
    - Verifica que `model.get_sentence_embedding_dimension()` coincida con EMBEDDING_DIM(),
    si no coincide, el INSERT a Postgres (columna`vector()`) va a fallar más adelante,
    mejor detectarlo aquí
    """

    model = SentenceTransformer(EMBEDDING_MODEL)

    dim = model.get_embedding_dimension()

    if dim != EMBEDDING_DIM:
        raise ValueError(
            f"{EMBEDDING_MODEL} produce vectores de {dim} dimensiones,"
            f"pero EMBEDDING_DIM es {EMBEDDING_DIM}. Ajusta el .env y la "
            f"columna vector(...) de schema.sql para que coincidan."
        )

    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embebe una lista de textos y devuelve una lista de vectores (listas de floats, 
    uno por texto, ene el mismo orden)
    """

    if not texts:
        return []

    vectors = get_model().encode(
        texts,
        # vectores de norma 1: coseno y producto dan el mismo ranking
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=len(texts)>32,
    )

    return vectors.tolist()

def embed_query(query: str) -> list[float]:
    """
    Embebe una sola pregunta del usuario (retrieval). Atajo sobre embed_texts
    para el caso de un solo texto
    """

    return embed_texts([query])[0]