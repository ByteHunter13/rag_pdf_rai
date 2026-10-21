"""
5a. Retrieval

Objetivo: dada una pregunta del usuario (en español o inglés), devolver los top-k chunks
más relevantes desde Postgres, listos para pasarle al generador (src/generation/generator.py)
"""

from typing import Any

from src.conf import TOP_K
from src.embedding.embedder import embed_query
from src.vectorstore.db import get_connection, search_similar_chunks

def retrieve(query: str, top_k: int = TOP_K,
             pdf_topic: str | None = None) -> list[dict[str, Any]]:
    """
    Devuelve los `top_k` chunks más relevantes para `query`
    """

    # El modelo del embeddings queda cacheado en memoria por get_model(), así
    # que el costo real de cada llamada es abrir la conexión, no embeber
    embedding = embed_query(query)

    # `with` sobre la conexión la cierra al salir del bloque: v1 abre y 
    # cierra por pregunta. Si query.py pasa a hacer muchas seguidas, este es
    # el punto donde conviene recibir una conexión ya abierta como parámetro
    with get_connection() as conn:
        return search_similar_chunks(conn, embedding, top_k, pdf_topic)
