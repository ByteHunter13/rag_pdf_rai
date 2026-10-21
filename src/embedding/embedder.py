"""
3. Embeddings con la API de OpenAI

Objetivo: dado un texto (chunk o pregunta del usuario), producir su vector de embedding
usando EMBEDDING_MODEL

Este módulo lo usan tanto el pipeline de ingesta (embeber chunks antes de guardarlos en
Postgres) como el retriever (embeber la pregunta del usuario)
"""

from src.conf import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL
from src.openai_api import post

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embebe un lote de textos con una sola request a /embeddings

    - `dimensions` se manda explícito: si alguien cambia EMBEDDING_MODEL, la
    API devuelve vectores del tamaño que espera la columna vector(...) en vez
    de que el INSERT a Postgres falle más adelante
    - La API documenta el campo `index` en cada resultado; se ordena por él
    para no depender de que `data` venga en el mismo orden que `input`
    """

    response = post("/embeddings", {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "dimensions": EMBEDDING_DIM,
    })

    data = sorted(response["data"], key=lambda item: item["index"])

    return [item["embedding"] for item in data]

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embebe una lista de textos y devuelve una lista de vectores (listas de floats,
    uno por texto, en el mismo orden)

    Los vectores de OpenAI ya vienen con norma 1: coseno y producto punto dan
    el mismo ranking, igual que antes con normalize_embeddings=True
    """

    if not texts:
        return []

    # La API rechaza strings vacíos con un 400 poco claro; mejor avisar aquí
    # cuál es el texto problemático
    for i, text in enumerate(texts):
        if not text.strip():
            raise ValueError(f"El texto en la posición {i} está vacío: no se puede embeber")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(embed_batch(texts[start:start + EMBEDDING_BATCH_SIZE]))

    return vectors

def embed_query(query: str) -> list[float]:
    """
    Embebe una sola pregunta del usuario (retrieval). Atajo sobre embed_texts
    para el caso de un solo texto
    """

    return embed_texts([query])[0]
