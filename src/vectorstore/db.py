"""
4. Acceso a PostgreSQL + pgvector

Objetivo: dar funciones simples para (a) insertar documentos y sus chunks con
embeddings, y (b) buscar los chunks más similares a un vector de consulta.
El esquema de tablas está en src/vectorstore/schema.sql
"""

from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from src.conf import DATABASE_URL

def get_connection() -> psycopg.Connection:
    """
    Abre una conexión a Postgres con el tipo `vector` ya registrado

    `register_vector` es necesario para que psycopg sepa serializar/parsear el
    tipo `vector`de Postgres hacia/desde listas de Python (o arrays de numpy) de
    forma transparente
    """
    conn = psycopg.connect(DATABASE_URL)

    # Lee del catálogo el OID del tipo `vector`, así que la extensión
    # pgvector debe estar creada en la base ANTES de conectarse
    register_vector(conn)

    return conn

def upsert_document(conn: psycopg.Connection, filename: str, pdf_topic: str | None,
                    doc_type: str | None, language: str | None) -> int:
    """
    Inserta (o actualiza si ya existe) una fila en `documents` y devuelve su `id`
    """

    # DO UPDATE (en vez de DO NOTHING) para que corregir un valor de
    # EQUIPMENT_METADATA y re-ingestar actualice la fila existente.
    # DO NOTHING tampoco devolvería id en el RETURNING
    row = conn.execute(
        """
        INSERT INTO documents (filename, pdf_topic, doc_type, language)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET pdf_topic = EXCLUDED.pdf_topic,
                doc_type = EXCLUDED.doc_type,
                language = EXCLUDED.language
        RETURNING id
        """,
        (filename, pdf_topic, doc_type, language)
    ).fetchone()

    return row[0]

def insert_chunks(conn: psycopg.Connection, document_id: int,
                  chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    """
    Inserta en lote los chunks de un documento junto con sus embeddings

    `chunks` es la lista de dicts que produce src/chunking/chunk.py
    (text, start_page, end_page, section). `embeddings[i]` corresponde a `chunks[i]
    """

    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks pero {len(embeddings)} embeddings: no coinciden")

    if not chunks:
        return

    # Re-ingestar el mismo PDF debe reemplazar sus chunks, no acumularlos:
    # `documents` se upsertea por filename, pero `chunks` no tiene UNIQUE
    # así que sin este DELETE cada corrida duplicaría todo el contenido y 
    # el retrieval devolvería el mismo texto varias veces
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    rows = [
        (
            document_id,
            chunk["text"],
            chunk["start_page"],
            chunk["end_page"],
            chunk["section"],
            Vector(embedding),
        ) for chunk, embedding in zip(chunks, embeddings)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks(document_id, text, start_page, end_page, section, embedding)
            VALUES(%s, %s, %s, %s, %s, %s)
            """,
            rows
        )

def search_similar_chunks(conn: psycopg.Connection, query_embedding: list[float],
                          top_k: int, pdf_topic: str | None = None) -> list[dict[str, Any]]:
    """
    Busca los `top_k` chunks más similares (distancia coseno) a `query_embedding`, 
    opcionalmente filtrando por `pdf_topic`

    Debe devolver una lista de dicts con al menos: texto, archivo (via join con documents),
    start_page, end_page, section, y la distancia/similitud (útil para depurar calidad de
    retrieval)
    """

    sql = """
        SELECT c.text,
                d.filename AS file,
                c.start_page,
                c.end_page,
                c.section,
                c.embedding <=> %(embedding)s AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE (%(pdf_topic)s::text IS NULL OR d.pdf_topic = %(pdf_topic)s)
        ORDER BY c.embedding <=> %(embedding)s
        LIMIT %(top_k)s
    """

    params = {
        "embedding": Vector(query_embedding),
        "pdf_topic": pdf_topic,
        "top_k": top_k
    }

    # dict_row hace que fetchall() devuelva dicts con los nombres del SELECT
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchall()